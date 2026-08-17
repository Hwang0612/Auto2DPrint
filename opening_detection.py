"""
opening_detection.py
====================
PrintPlan AI — Opening Detection & Z-Aware Toolpath Module
-----------------------------------------------------------
Detects doors and windows from a vector PDF floor plan, then
applies per-layer suppression logic to generate Z-aware G-code
toolpaths for 3D concrete printing.

Supports:
  - OCG (named layer) based detection: "Doors", "Windows", "Wall"
  - Fallback: geometric gap + arc matching
  - Simplified door model: void from Z=0 (no base courses)
  - Window model: void from sill_height to head_height
  - Conventional lintel: PAUSE + resume above lintel
  - Project-wide defaults with per-opening overrides

Dependencies: pdfplumber, dataclasses, math, collections

Author: PrintPlan AI Pipeline
"""

import math
import pdfplumber
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Optional, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class OpeningType(Enum):
    DOOR   = "door"
    WINDOW = "window"

class PrintAction(Enum):
    FULL_WALL    = "full_wall"     # print complete wall contour
    SPLIT_PIERS  = "split_piers"  # suppress opening, print left+right only
    SKIP         = "skip"          # inside lintel thickness — no print
    PAUSE_LINTEL = "pause_lintel"  # emit M0 + crew installs lintel

class LinteldType(Enum):
    TIMBER   = "timber"
    STEEL    = "steel"
    PRECAST  = "precast"
    PRINTED  = "printed"
    NONE     = "none"

# Gap size tolerance in pts for arc-gap matching
ARC_GAP_MATCH_TOLERANCE = 8.0  # pts


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Project-wide 3DCP parameters."""
    layer_height_mm: float = 50.0       # nozzle layer height (mm)
    wall_height_mm: float  = 2700.0     # total wall height (mm)
    lintel_thickness_mm: float = 200.0  # conventional lintel depth (mm)
    lintel_type: LinteldType = LinteldType.TIMBER
    pause_for_lintel: bool = True

    # Project-wide opening defaults
    door_head_height_mm: float  = 2100.0
    window_sill_height_mm: float = 900.0
    window_head_height_mm: float = 2100.0

    # Drawing scale
    drawing_scale: int = 100   # 1:100
    drawing_unit: str  = "mm"


@dataclass
class WallGap:
    """A detected gap in a wall line — candidate opening location."""
    wall_id: str
    direction: str          # "horizontal" or "vertical"
    fixed_coord: float      # Y value for horizontal, X value for vertical (pts)
    gap_start: float        # start of gap along wall (pts)
    gap_end: float          # end of gap along wall (pts)
    gap_mid: float          # midpoint along wall (pts)
    gap_width_pts: float    # gap width in pts
    gap_width_mm: float     # gap width in real mm

    @property
    def cx(self) -> float:
        """Centre X in pts."""
        return self.gap_mid if self.direction == "horizontal" else self.fixed_coord

    @property
    def cy(self) -> float:
        """Centre Y in pts."""
        return self.fixed_coord if self.direction == "horizontal" else self.gap_mid


@dataclass
class DoorArc:
    """A detected door swing arc."""
    index: int
    cx: float       # centre X (pts)
    cy: float       # centre Y (pts)
    radius: float   # approx radius (pts)
    bbox_w: float
    bbox_h: float


@dataclass
class Opening:
    """
    A fully resolved opening (door or window) with both X-Y position
    and Z printing parameters.
    """
    opening_id: str
    opening_type: OpeningType

    # X-Y geometry (pts, from drawing)
    wall_id: str
    wall_direction: str     # "horizontal" or "vertical"
    gap_start_pts: float
    gap_end_pts: float
    gap_mid_pts: float
    fixed_coord_pts: float  # Y for H-wall, X for V-wall
    width_mm: float

    # Z parameters (mm, real-world)
    z_void_bottom_mm: float     # 0 for door, sill_height for window
    z_void_top_mm: float        # head height
    lintel_thickness_mm: float
    lintel_type: LinteldType
    pause_for_lintel: bool

    # World mm coordinates (pipeline coordinate system, origin = outer wall corner)
    world_x_mm: float = 0.0    # centre X of opening in world mm
    world_y_mm: float = 0.0    # centre Y of opening in world mm

    # Override flag
    is_user_overridden: bool = False

    # -----------------------------------------------------------------------
    # Computed Z properties
    # -----------------------------------------------------------------------

    @property
    def z_resume_mm(self) -> float:
        """First printable Z above lintel."""
        if self.lintel_type == LinteldType.NONE:
            return float("inf")
        return self.z_void_top_mm + self.lintel_thickness_mm

    def get_print_action(self, layer_z_mm: float) -> PrintAction:
        """
        Return the print action for a given layer Z height.

        Z-zone model:
          Zone 1 — Below void (window sill zone or nothing for door)
                   → FULL_WALL
          Zone 2 — Inside void (door or window opening)
                   → SPLIT_PIERS
          Zone 3 — Last void layer (lintel seat)
                   → PAUSE_LINTEL (once, at z_void_top)
          Zone 3b — Inside lintel thickness
                   → SKIP
          Zone 4 — Above lintel
                   → FULL_WALL
        """
        z = layer_z_mm

        # Zone 1: below void bottom
        if z < self.z_void_bottom_mm:
            return PrintAction.FULL_WALL

        # Zone 2: inside void
        if self.z_void_bottom_mm <= z < self.z_void_top_mm:
            return PrintAction.SPLIT_PIERS

        # Zone 3: at lintel seat level
        if math.isclose(z, self.z_void_top_mm, abs_tol=1.0):
            if self.pause_for_lintel and self.lintel_type != LinteldType.NONE:
                return PrintAction.PAUSE_LINTEL
            else:
                return PrintAction.SPLIT_PIERS

        # Zone 3b: inside lintel thickness
        if self.z_void_top_mm < z < self.z_resume_mm:
            return PrintAction.SKIP

        # Zone 4: above lintel
        return PrintAction.FULL_WALL

    def summary(self) -> str:
        return (
            f"{self.opening_id} [{self.opening_type.value.upper()}] "
            f"width={self.width_mm:.0f}mm | "
            f"void Z: {self.z_void_bottom_mm:.0f}–{self.z_void_top_mm:.0f}mm | "
            f"resumes at Z={self.z_resume_mm:.0f}mm"
        )


# ---------------------------------------------------------------------------
# PDF Parser
# ---------------------------------------------------------------------------

class FloorPlanParser:
    """
    Parses a vector PDF floor plan to detect wall gaps and door arcs.
    Supports OCG-layer-based detection (preferred) and geometric fallback.
    """

    # Gap width range in real mm considered a door-sized opening
    DOOR_WIDTH_MIN_MM  = 700
    DOOR_WIDTH_MAX_MM  = 1200
    # Dimension tick arcs are tiny — ignore below this bbox size (pts)
    ARC_MIN_BBOX_PTS   = 15
    # Very large arcs are building outlines — ignore above this
    ARC_MAX_BBOX_PTS   = 200
    # Collinearity tolerance for grouping lines
    LINE_GROUP_THRESH  = 1.5   # pts

    def __init__(self, pdf_path: str, config: ProjectConfig):
        self.pdf_path = pdf_path
        self.config   = config
        self.scale_pts_to_mm: Optional[float] = None
        self.page_height: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> Tuple[List[WallGap], List[DoorArc]]:
        """
        Parse the PDF and return:
          - wall_gaps: all detected gaps in wall lines
          - door_arcs: all detected door swing arcs
        """
        with pdfplumber.open(self.pdf_path) as pdf:
            page = pdf.pages[0]
            self.page_height = page.height
            self._calibrate_scale(page)

            lines  = page.lines
            curves = page.curves

            wall_gaps  = self._detect_wall_gaps(lines)
            door_arcs  = self._detect_door_arcs(curves)

        return wall_gaps, door_arcs

    def detect_openings(self, include_unmatched_as_windows: bool = False) -> List[Opening]:
        """
        Full pipeline: parse → match gaps to arcs → classify → resolve Z.
        Returns a list of Opening objects ready for the layer scheduler.

        Args:
            include_unmatched_as_windows: If True, wall gaps not matched to any
                door arc are classified as windows. Default is False — only
                arc-confirmed gaps are returned as doors. Set to True only when
                the drawing explicitly contains window symbols (triple lines).
        """
        wall_gaps, door_arcs = self.parse()
        matched   = self._match_gaps_to_arcs(wall_gaps, door_arcs)
        unmatched = self._find_unmatched_gaps(wall_gaps, matched) if include_unmatched_as_windows else []
        openings  = self._build_openings(matched, unmatched)
        return openings

    # ------------------------------------------------------------------
    # Scale calibration
    # ------------------------------------------------------------------

    def _calibrate_scale(self, page) -> None:
        """
        Mirror the exact coordinate transformation used by Stage M1 (m1_ingest.py):

            k_world = PT_TO_MM * drawing_scale
            x_world = x_pt * k_world
            y_world = (page_h_pt - y_pt) * k_world   ← Y flipped at full page height

        where PT_TO_MM = 25.4 / 72 = 0.352778 mm/pt.

        No origin subtraction — the pipeline origin is always (0, 0) in PDF space.
        The drawing_scale comes from the sidebar (M1Params.drawing_scale),
        which is passed into ProjectConfig when opening detection is run.
        """
        PT_TO_MM = 25.4 / 72.0                                    # 0.352778 mm/pt
        self.scale_pts_to_mm     = PT_TO_MM * self.config.drawing_scale
        self.geo_scale_pts_to_mm = self.scale_pts_to_mm
        self.page_h_pt           = page.height                    # full page height in pts
        # x_origin = 0 (no subtraction, matching pipeline)
        self.x_origin_pts        = 0.0
        self.y_max_pts           = page.height                    # used in _pts_to_world_mm

    def _pts_to_mm(self, pts: float) -> float:
        return pts * self.scale_pts_to_mm

    def _pts_to_world_mm(self, x_pts: float, y_pts: float):
        """
        Convert PDF pts to world mm using the identical transformation as m1_ingest.py:
            x_world = x_pt * k
            y_world = (page_h_pt - y_pt) * k
        No origin subtraction — pipeline origin is (0,0) in PDF space.
        """
        k = self.scale_pts_to_mm
        world_x = x_pts * k
        world_y = (self.page_h_pt - y_pts) * k
        return world_x, world_y

    # ------------------------------------------------------------------
    # Wall gap detection
    # ------------------------------------------------------------------

    def _detect_wall_gaps(self, lines: list) -> List[WallGap]:
        """
        Group collinear lines and find gaps between them.
        Gaps in the door/window size range are candidate openings.
        """
        gaps: List[WallGap] = []
        wall_counter = [0]

        # --- Horizontal walls ---
        h_lines = [l for l in lines if abs(l["y0"] - l["y1"]) < self.LINE_GROUP_THRESH]
        h_groups = defaultdict(list)
        for l in h_lines:
            y_key = round(l["y0"] / self.LINE_GROUP_THRESH) * self.LINE_GROUP_THRESH
            h_groups[y_key].append(l)

        for y, segs in h_groups.items():
            if len(segs) < 2:
                continue
            segs_sorted = sorted(segs, key=lambda l: l["x0"])
            wall_id = f"HW_{wall_counter[0]:03d}"
            wall_counter[0] += 1
            for i in range(len(segs_sorted) - 1):
                end   = segs_sorted[i]["x1"]
                start = segs_sorted[i + 1]["x0"]
                gap_w_pts = start - end
                if gap_w_pts <= 0:
                    continue
                gap_w_mm = self._pts_to_mm(gap_w_pts)
                if self.DOOR_WIDTH_MIN_MM <= gap_w_mm <= self.DOOR_WIDTH_MAX_MM:
                    gaps.append(WallGap(
                        wall_id        = wall_id,
                        direction      = "horizontal",
                        fixed_coord    = y,
                        gap_start      = end,
                        gap_end        = start,
                        gap_mid        = (end + start) / 2,
                        gap_width_pts  = gap_w_pts,
                        gap_width_mm   = gap_w_mm,
                    ))

        # --- Vertical walls ---
        v_lines = [l for l in lines if abs(l["x0"] - l["x1"]) < self.LINE_GROUP_THRESH]
        v_groups = defaultdict(list)
        for l in v_lines:
            x_key = round(l["x0"] / self.LINE_GROUP_THRESH) * self.LINE_GROUP_THRESH
            v_groups[x_key].append(l)

        for x, segs in v_groups.items():
            if len(segs) < 2:
                continue
            segs_sorted = sorted(segs, key=lambda l: l["y0"])
            wall_id = f"VW_{wall_counter[0]:03d}"
            wall_counter[0] += 1
            for i in range(len(segs_sorted) - 1):
                end   = segs_sorted[i]["y1"]
                start = segs_sorted[i + 1]["y0"]
                gap_w_pts = start - end
                if gap_w_pts <= 0:
                    continue
                gap_w_mm = self._pts_to_mm(gap_w_pts)
                if self.DOOR_WIDTH_MIN_MM <= gap_w_mm <= self.DOOR_WIDTH_MAX_MM:
                    gaps.append(WallGap(
                        wall_id        = wall_id,
                        direction      = "vertical",
                        fixed_coord    = x,
                        gap_start      = end,
                        gap_end        = start,
                        gap_mid        = (end + start) / 2,
                        gap_width_pts  = gap_w_pts,
                        gap_width_mm   = gap_w_mm,
                    ))

        return gaps

    # ------------------------------------------------------------------
    # Door arc detection
    # ------------------------------------------------------------------

    def _detect_door_arcs(self, curves: list) -> List[DoorArc]:
        """
        Filter curves to find door swing arcs.
        Dimension ticks (tiny) and building outlines (huge) are excluded.
        """
        arcs: List[DoorArc] = []
        for i, c in enumerate(curves):
            pts = c["pts"]
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            w   = max(xs) - min(xs)
            h   = max(ys) - min(ys)
            size = max(w, h)
            if size < self.ARC_MIN_BBOX_PTS or size > self.ARC_MAX_BBOX_PTS:
                continue
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            radius = size / 2
            arcs.append(DoorArc(
                index  = i,
                cx     = cx,
                cy     = cy,
                radius = radius,
                bbox_w = w,
                bbox_h = h,
            ))
        return arcs

    # ------------------------------------------------------------------
    # Gap ↔ Arc matching
    # ------------------------------------------------------------------

    def _match_gaps_to_arcs(
        self,
        gaps: List[WallGap],
        arcs: List[DoorArc],
    ) -> List[Tuple[WallGap, DoorArc]]:
        """
        Match each wall gap to the nearest door arc whose centre lies
        within the gap bounding box (with tolerance).
        A matched gap+arc pair is classified as a DOOR.
        """
        matched: List[Tuple[WallGap, DoorArc]] = []
        used_arcs: set = set()

        for gap in gaps:
            best_arc  = None
            best_dist = float("inf")
            for arc in arcs:
                if arc.index in used_arcs:
                    continue
                # Arc centre must be near the gap midpoint
                dx = abs(arc.cx - gap.cx)
                dy = abs(arc.cy - gap.cy)
                dist = math.hypot(dx, dy)
                # Tolerance: half the gap width + ARC_GAP_MATCH_TOLERANCE
                tol = gap.gap_width_pts / 2 + ARC_GAP_MATCH_TOLERANCE
                if dist < tol and dist < best_dist:
                    best_dist = dist
                    best_arc  = arc

            if best_arc is not None:
                matched.append((gap, best_arc))
                used_arcs.add(best_arc.index)

        return matched

    def _find_unmatched_gaps(
        self,
        all_gaps: List[WallGap],
        matched: List[Tuple[WallGap, DoorArc]],
    ) -> List[WallGap]:
        """
        Gaps not matched to any arc are classified as WINDOWS
        (when windows are present in the drawing).
        """
        matched_gaps = {id(g) for g, _ in matched}
        return [g for g in all_gaps if id(g) not in matched_gaps]

    # ------------------------------------------------------------------
    # Build Opening objects
    # ------------------------------------------------------------------

    def _build_openings(
        self,
        matched_doors: List[Tuple[WallGap, DoorArc]],
        unmatched_windows: List[WallGap],
    ) -> List[Opening]:
        """
        Construct Opening objects with Z parameters from ProjectConfig defaults.
        """
        cfg      = self.config
        openings: List[Opening] = []
        door_num = 1
        win_num  = 1

        for gap, arc in matched_doors:
            if gap.direction == "horizontal":
                wx, wy = self._pts_to_world_mm(gap.gap_mid, gap.fixed_coord)
            else:
                wx, wy = self._pts_to_world_mm(gap.fixed_coord, gap.gap_mid)
            openings.append(Opening(
                opening_id       = f"D{door_num:02d}",
                opening_type     = OpeningType.DOOR,
                wall_id          = gap.wall_id,
                wall_direction   = gap.direction,
                gap_start_pts    = gap.gap_start,
                gap_end_pts      = gap.gap_end,
                gap_mid_pts      = gap.gap_mid,
                fixed_coord_pts  = gap.fixed_coord,
                width_mm         = gap.gap_width_mm,
                world_x_mm       = wx,
                world_y_mm       = wy,
                # Z — door void always from 0
                z_void_bottom_mm = 0.0,
                z_void_top_mm    = cfg.door_head_height_mm,
                lintel_thickness_mm = cfg.lintel_thickness_mm,
                lintel_type      = cfg.lintel_type,
                pause_for_lintel = cfg.pause_for_lintel,
            ))
            door_num += 1

        for gap in unmatched_windows:
            if gap.direction == "horizontal":
                wx, wy = self._pts_to_world_mm(gap.gap_mid, gap.fixed_coord)
            else:
                wx, wy = self._pts_to_world_mm(gap.fixed_coord, gap.gap_mid)
            openings.append(Opening(
                opening_id       = f"W{win_num:02d}",
                opening_type     = OpeningType.WINDOW,
                wall_id          = gap.wall_id,
                wall_direction   = gap.direction,
                gap_start_pts    = gap.gap_start,
                gap_end_pts      = gap.gap_end,
                gap_mid_pts      = gap.gap_mid,
                fixed_coord_pts  = gap.fixed_coord,
                width_mm         = gap.gap_width_mm,
                world_x_mm       = wx,
                world_y_mm       = wy,
                # Z — window void from sill
                z_void_bottom_mm = cfg.window_sill_height_mm,
                z_void_top_mm    = cfg.window_head_height_mm,
                lintel_thickness_mm = cfg.lintel_thickness_mm,
                lintel_type      = cfg.lintel_type,
                pause_for_lintel = cfg.pause_for_lintel,
            ))
            win_num += 1

        # Sort: doors first, then windows; by wall_id then position
        openings.sort(key=lambda o: (o.opening_type.value, o.wall_id, o.gap_mid_pts))
        return openings


# ---------------------------------------------------------------------------
# Layer Scheduler
# ---------------------------------------------------------------------------

class LayerScheduler:
    """
    Iterates through all print layers and determines per-opening print
    actions at each Z height.
    """

    def __init__(self, openings: List[Opening], config: ProjectConfig):
        self.openings = openings
        self.config   = config

    def get_layer_actions(self, layer_z_mm: float) -> dict:
        """
        For a given layer Z, return a dict mapping opening_id → PrintAction.
        Any opening not in the dict should be treated as FULL_WALL.
        """
        actions = {}
        for op in self.openings:
            action = op.get_print_action(layer_z_mm)
            if action != PrintAction.FULL_WALL:
                actions[op.opening_id] = action
        return actions

    def generate_layer_schedule(self) -> List[dict]:
        """
        Generate the full schedule across all print layers.
        Returns a list of dicts, one per layer:
          {
            'layer_index': int,
            'z_mm': float,
            'actions': {opening_id: PrintAction},
            'has_suppression': bool,
            'has_pause': bool,
          }
        """
        schedule = []
        n_layers = int(self.config.wall_height_mm / self.config.layer_height_mm)

        for i in range(n_layers):
            z = i * self.config.layer_height_mm
            actions = self.get_layer_actions(z)
            schedule.append({
                "layer_index"     : i,
                "z_mm"            : z,
                "actions"         : actions,
                "has_suppression" : any(
                    a == PrintAction.SPLIT_PIERS for a in actions.values()
                ),
                "has_pause"       : any(
                    a == PrintAction.PAUSE_LINTEL for a in actions.values()
                ),
            })

        return schedule


# ---------------------------------------------------------------------------
# G-code Generator
# ---------------------------------------------------------------------------

class GCodeGenerator:
    """
    Generates Z-aware G-code output incorporating opening suppression
    and lintel pause commands.
    """

    def __init__(self, config: ProjectConfig):
        self.config = config

    def generate_layer_header(self, layer_index: int, z_mm: float) -> List[str]:
        return [
            f"",
            f"; ── Layer {layer_index} | Z = {z_mm:.1f} mm ──────────────────────",
        ]

    def generate_full_wall(self, wall_id: str, z_mm: float) -> List[str]:
        return [f"G1 ; {wall_id} full wall @ Z={z_mm:.1f}mm — extrude normally"]

    def generate_split_piers(
        self,
        wall_id: str,
        opening: Opening,
        z_mm: float,
    ) -> List[str]:
        lines = [
            f"G1 ; {wall_id} LEFT pier of {opening.opening_id} @ Z={z_mm:.1f}mm",
            f"G0 ; travel across {opening.opening_id} void (no extrusion)",
            f"G1 ; {wall_id} RIGHT pier of {opening.opening_id} @ Z={z_mm:.1f}mm",
        ]
        return lines

    def generate_lintel_pause(self, opening: Opening, z_mm: float) -> List[str]:
        return [
            f"",
            f"; ═══════════════════════════════════════════════════════════",
            f"; PAUSE — Install {opening.lintel_type.value} lintel for {opening.opening_id}",
            f"; Opening width: {opening.width_mm:.0f}mm",
            f"; Lintel seat level: Z = {z_mm:.1f}mm",
            f"; Resume printing at Z = {opening.z_resume_mm:.1f}mm",
            f"; ═══════════════════════════════════════════════════════════",
            f"M0 ; pause for lintel installation",
            f"",
        ]

    def generate_skip_comment(self, opening: Opening, z_mm: float) -> List[str]:
        return [
            f"; SKIP Z={z_mm:.1f}mm — inside lintel of {opening.opening_id} "
            f"(lintel occupies {opening.z_void_top_mm:.0f}–{opening.z_resume_mm:.0f}mm)"
        ]

    def generate_full_schedule_gcode(
        self,
        schedule: List[dict],
        openings: List[Opening],
    ) -> str:
        """
        Generate complete G-code string for the full wall height,
        incorporating all opening actions.
        """
        opening_map = {op.opening_id: op for op in openings}
        lines: List[str] = [
            "; PrintPlan AI — Z-Aware Toolpath with Opening Detection",
            f"; Layer height: {self.config.layer_height_mm}mm",
            f"; Wall height:  {self.config.wall_height_mm}mm",
            f"; Total layers: {len(schedule)}",
            f"; Openings:     {len(openings)}",
            "",
            "G21 ; units = mm",
            "G90 ; absolute positioning",
            "",
        ]

        lintel_paused: set = set()

        for layer in schedule:
            z   = layer["z_mm"]
            idx = layer["layer_index"]
            actions = layer["actions"]

            lines += self.generate_layer_header(idx, z)

            if not actions:
                # No openings affected — print all walls normally
                lines.append("G1 ; all walls — full contour @ Z={:.1f}mm".format(z))
                continue

            # Process each opening action
            for oid, action in actions.items():
                op = opening_map[oid]

                if action == PrintAction.SPLIT_PIERS:
                    lines += self.generate_split_piers(op.wall_id, op, z)

                elif action == PrintAction.PAUSE_LINTEL:
                    if oid not in lintel_paused:
                        lines += self.generate_lintel_pause(op, z)
                        lintel_paused.add(oid)

                elif action == PrintAction.SKIP:
                    lines += self.generate_skip_comment(op, z)

            # Walls not affected by any opening
            lines.append(f"G1 ; remaining walls — full contour @ Z={z:.1f}mm")

        lines += ["", "; ── Print complete ──────────────────────────────────────", "M2"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streamlit Integration Helper
# ---------------------------------------------------------------------------

def run_opening_detection_ui(pdf_path: str) -> Tuple[List[Opening], ProjectConfig]:
    """
    Streamlit-compatible function.
    Returns detected openings and config for further UI override.

    Usage in app.py:
        from opening_detection import run_opening_detection_ui
        openings, config = run_opening_detection_ui(uploaded_pdf_path)
    """
    import streamlit as st

    st.subheader("🔧 Project Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        layer_h  = st.number_input("Layer height (mm)", value=50, min_value=10, max_value=200, step=5)
        wall_h   = st.number_input("Wall height (mm)",  value=2700, min_value=1000, max_value=6000, step=100)
    with col2:
        door_hh  = st.number_input("Door head height (mm)",    value=2100, min_value=1800, max_value=3000, step=50)
        win_sill = st.number_input("Window sill height (mm)",  value=900,  min_value=0,    max_value=2000, step=50)
        win_hh   = st.number_input("Window head height (mm)",  value=2100, min_value=500,  max_value=3000, step=50)
    with col3:
        lintel_t = st.number_input("Lintel thickness (mm)", value=200, min_value=50, max_value=500, step=25)
        lintel_tp = st.selectbox("Lintel type", ["timber", "steel", "precast", "none"])
        pause    = st.checkbox("Pause G-code for lintel", value=True)

    config = ProjectConfig(
        layer_height_mm      = float(layer_h),
        wall_height_mm       = float(wall_h),
        lintel_thickness_mm  = float(lintel_t),
        lintel_type          = LinteldType(lintel_tp),
        pause_for_lintel     = pause,
        door_head_height_mm  = float(door_hh),
        window_sill_height_mm= float(win_sill),
        window_head_height_mm= float(win_hh),
    )

    st.subheader("📐 Opening Detection")

    with st.spinner("Scanning floor plan for doors and windows..."):
        parser   = FloorPlanParser(pdf_path, config)
        openings = parser.detect_openings()

    if not openings:
        st.warning("No openings detected. Check that your PDF has vector Wall and Door layers.")
        return [], config

    st.success(f"Detected {len(openings)} opening(s)")

    # Show detected openings table with per-row override capability
    st.subheader("📋 Detected Openings — Review & Override")
    st.caption("Z parameters are pre-filled from project defaults. Edit any row to override.")

    override_data = []
    for op in openings:
        with st.expander(f"{op.opening_id} — {op.opening_type.value.title()} | "
                         f"Width: {op.width_mm:.0f}mm | Wall: {op.wall_id}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                if op.opening_type == OpeningType.DOOR:
                    st.text_input("Sill height (mm)", value="0", disabled=True,
                                  key=f"sill_{op.opening_id}")
                    void_bot = 0.0
                else:
                    void_bot = st.number_input(
                        "Sill height (mm)", value=int(op.z_void_bottom_mm),
                        min_value=0, max_value=2000, step=50,
                        key=f"sill_{op.opening_id}"
                    )
            with c2:
                void_top = st.number_input(
                    "Head height (mm)", value=int(op.z_void_top_mm),
                    min_value=500, max_value=3000, step=50,
                    key=f"head_{op.opening_id}"
                )
            with c3:
                lin_t = st.number_input(
                    "Lintel thick. (mm)", value=int(op.lintel_thickness_mm),
                    min_value=0, max_value=500, step=25,
                    key=f"lint_{op.opening_id}"
                )

            # Apply overrides
            op.z_void_bottom_mm   = float(void_bot)
            op.z_void_top_mm      = float(void_top)
            op.lintel_thickness_mm = float(lin_t)
            op.is_user_overridden = True
            override_data.append(op)

    return override_data if override_data else openings, config


def generate_gcode_from_openings(
    openings: List[Opening],
    config: ProjectConfig,
) -> str:
    """
    Generate the full Z-aware G-code given resolved openings and config.
    Call this after run_opening_detection_ui() in your Streamlit app.
    """
    scheduler = LayerScheduler(openings, config)
    schedule  = scheduler.generate_layer_schedule()
    generator = GCodeGenerator(config)
    return generator.generate_full_schedule_gcode(schedule, openings)


# ---------------------------------------------------------------------------
# Standalone test / demo
# ---------------------------------------------------------------------------

def demo(pdf_path: str) -> None:
    """
    Run the full pipeline on a PDF and print results to console.
    """
    print("=" * 60)
    print("PrintPlan AI — Opening Detection Demo")
    print("=" * 60)

    config = ProjectConfig(
        layer_height_mm       = 50.0,
        wall_height_mm        = 2700.0,
        lintel_thickness_mm   = 200.0,
        lintel_type           = LinteldType.TIMBER,
        pause_for_lintel      = True,
        door_head_height_mm   = 2100.0,
        window_sill_height_mm = 900.0,
        window_head_height_mm = 2100.0,
        drawing_scale         = 100,
    )

    parser   = FloorPlanParser(pdf_path, config)
    openings = parser.detect_openings()

    print(f"\nDetected {len(openings)} opening(s):\n")
    for op in openings:
        print(f"  {op.summary()}")

    print("\n--- Layer Schedule (showing suppressed layers only) ---")
    scheduler = LayerScheduler(openings, config)
    schedule  = scheduler.generate_layer_schedule()
    for layer in schedule:
        if layer["actions"]:
            z = layer["z_mm"]
            for oid, action in layer["actions"].items():
                print(f"  Z={z:6.1f}mm  {oid}: {action.value}")

    print("\n--- G-code Preview (first 60 lines) ---")
    generator = GCodeGenerator(config)
    gcode = generator.generate_full_schedule_gcode(schedule, openings)
    for line in gcode.split("\n")[:60]:
        print(line)

    print("\n[Demo complete]")


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "print.pdf"
    demo(pdf)
