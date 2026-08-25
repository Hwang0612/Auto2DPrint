"""
opening_detection.py
====================
PrintPlan AI -- Opening Detection & Z-Aware Toolpath Module
-----------------------------------------------------------
v3.0 -- Reads 'Doors' and 'Windows' OCG layers directly using PyMuPDF.
        Falls back to pdfplumber gap detection for PDFs without named layers.

All Opening objects store world-mm coordinates using the SAME transform as
m1_ingest.py:
    x_world = x_pt * k
    y_world = (page_h_pt - y_pt) * k
    k = (25.4 / 72) * drawing_scale

This ensures the visualizer_3d.py XY-suppression check works correctly.

Author: PrintPlan AI Pipeline
"""

import math
import statistics
import pdfplumber
import fitz                   # PyMuPDF  (pip install pymupdf)
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
    FULL_WALL    = "full_wall"
    SPLIT_PIERS  = "split_piers"
    SKIP         = "skip"
    PAUSE_LINTEL = "pause_lintel"

class LinteldType(Enum):
    TIMBER   = "timber"
    STEEL    = "steel"
    PRECAST  = "precast"
    PRINTED  = "printed"
    NONE     = "none"

# Gap size tolerance for arc-gap matching (pdfplumber fallback only)
ARC_GAP_MATCH_TOLERANCE = 8.0   # pts
# Two door arcs within this distance (mm) = same physical opening
DOOR_MERGE_DIST_MM = 2000.0


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Project-wide 3DCP parameters."""
    layer_height_mm: float = 50.0
    wall_height_mm: float  = 2700.0
    wall_thickness_mm: float = 300.0
    lintel_thickness_mm: float = 200.0
    lintel_type: LinteldType = LinteldType.TIMBER
    pause_for_lintel: bool = True

    door_head_height_mm: float   = 2100.0
    window_sill_height_mm: float = 900.0
    window_head_height_mm: float = 2100.0

    drawing_scale: int = 100
    drawing_unit: str  = "mm"

    @property
    def k(self) -> float:
        """PDF pts -> world mm, matching m1_ingest.py exactly."""
        return (25.4 / 72) * self.drawing_scale


@dataclass
class WallGap:
    """A detected gap in a wall line -- used by the pdfplumber fallback."""
    wall_id: str
    direction: str
    fixed_coord: float    # Y for H-wall, X for V-wall  (pts)
    gap_start: float
    gap_end: float
    gap_mid: float
    gap_width_pts: float
    gap_width_mm: float

    @property
    def cx(self) -> float:
        return self.gap_mid if self.direction == "horizontal" else self.fixed_coord

    @property
    def cy(self) -> float:
        return self.fixed_coord if self.direction == "horizontal" else self.gap_mid


@dataclass
class DoorArc:
    """A detected door swing arc (pdfplumber fallback)."""
    index: int
    cx: float
    cy: float
    radius: float
    bbox_w: float
    bbox_h: float


@dataclass
class Opening:
    """
    A fully resolved opening (door or window).

    XY position is stored in world-mm using the m1_ingest transform so the
    3D visualizer can do a direct mm-vs-mm comparison.

    Legacy pt-space attributes (gap_*_pts, fixed_coord_pts) are preserved
    for backward compatibility with app.py display code.
    """
    opening_id: str
    opening_type: OpeningType

    # -- Display / legacy attributes (used in app.py expander labels) --
    wall_id: str
    wall_direction: str       # "horizontal" or "vertical"
    width_mm: float

    # -- XY in world-mm (m1_ingest coordinate system) --
    cx_world_mm: float        # opening centre X in world mm
    cy_world_mm: float        # opening centre Y in world mm

    # -- Legacy pt-space (kept for backward compat; not used in suppression) --
    gap_start_pts: float = 0.0
    gap_end_pts: float   = 0.0
    gap_mid_pts: float   = 0.0
    fixed_coord_pts: float = 0.0
    gap_width_pts: float   = 0.0   # <- needed by old visualizer code

    # -- Wall geometry --
    wall_thickness_mm: float = 300.0
    # True wall centreline in the perpendicular direction (world mm).
    # For horizontal wall: this is the Y midpoint between the two wall faces.
    # For vertical wall:   this is the X midpoint between the two wall faces.
    # Computed from jamb line midpoints in opening_detection; used by visualizer.
    wall_centre_perp_mm: float = 0.0

    # -- Z parameters --
    z_void_bottom_mm: float = 0.0
    z_void_top_mm: float    = 2100.0
    lintel_thickness_mm: float = 200.0
    lintel_type: LinteldType = LinteldType.TIMBER
    pause_for_lintel: bool   = True

    is_user_overridden: bool = False

    @property
    def z_resume_mm(self) -> float:
        if self.lintel_type == LinteldType.NONE:
            return float("inf")
        return self.z_void_top_mm + self.lintel_thickness_mm

    def get_print_action(self, layer_z_mm: float) -> PrintAction:
        z = layer_z_mm
        if z < self.z_void_bottom_mm:
            return PrintAction.FULL_WALL
        if self.z_void_bottom_mm <= z < self.z_void_top_mm:
            return PrintAction.SPLIT_PIERS
        if math.isclose(z, self.z_void_top_mm, abs_tol=1.0):
            if self.pause_for_lintel and self.lintel_type != LinteldType.NONE:
                return PrintAction.PAUSE_LINTEL
            return PrintAction.SPLIT_PIERS
        if self.z_void_top_mm < z < self.z_resume_mm:
            return PrintAction.SKIP
        return PrintAction.FULL_WALL

    def summary(self) -> str:
        return (
            f"{self.opening_id} [{self.opening_type.value.upper()}] "
            f"width={self.width_mm:.0f}mm | "
            f"centre=({self.cx_world_mm:.0f},{self.cy_world_mm:.0f})mm | "
            f"void Z: {self.z_void_bottom_mm:.0f}-{self.z_void_top_mm:.0f}mm | "
            f"resumes at Z={self.z_resume_mm:.0f}mm"
        )


# ---------------------------------------------------------------------------
# PDF Parser  (strategy A: PyMuPDF OCG layers; strategy B: pdfplumber gaps)
# ---------------------------------------------------------------------------

class FloorPlanParser:
    """
    Parses a vector PDF floor plan to detect door and window openings.

    Strategy A (preferred): reads 'Doors' and 'Windows' OCG layers via PyMuPDF.
    Strategy B (fallback):  detects wall gaps + door arcs via pdfplumber.
    """

    DOOR_WIDTH_MIN_MM  = 700
    DOOR_WIDTH_MAX_MM  = 1500
    ARC_MIN_BBOX_PTS   = 15
    ARC_MAX_BBOX_PTS   = 200
    LINE_GROUP_THRESH  = 1.5   # pts

    def __init__(self, pdf_path: str, config: ProjectConfig):
        self.pdf_path = pdf_path
        self.config   = config

    # -- Public API ----------------------------------------------------------

    def detect_openings(self, include_unmatched_as_windows: bool = False) -> List[Opening]:
        """
        Main entry point.  Tries OCG-layer strategy first, falls back to
        gap detection if the PDF has no 'Doors'/'Windows' layers.
        """
        doc   = fitz.open(self.pdf_path)
        page  = doc[0]
        paths = page.get_drawings()
        layer_names = {p.get("layer") for p in paths if p.get("layer")}

        if "Doors" in layer_names or "Windows" in layer_names:
            return self._detect_from_ocg_layers(doc, page, paths, layer_names)

        # Fallback to pdfplumber gap detection
        return self._detect_from_gaps(include_unmatched_as_windows)

    # -- Strategy A: OCG layer reading ---------------------------------------

    def _detect_from_ocg_layers(self, doc, page, paths, available_layers) -> List[Opening]:
        page_h = page.rect.height
        k      = self.config.k          # pts -> world mm
        cfg    = self.config

        def to_world(x_pt, y_pt):
            """Identical to m1_ingest.py transform."""
            return x_pt * k, (page_h - y_pt) * k

        openings: List[Opening] = []

        # -- DOORS ------------------------------------------------------------
        if "Doors" in available_layers:
            door_paths = [p for p in paths if p.get("layer") == "Doors"]

            # Collect arc groups: each path object with Bezier curves = one door swing
            arc_groups = []
            for p in door_paths:
                curves = [it for it in p.get("items", []) if it[0] == "c"]
                if curves:
                    arc_groups.append(curves)

            # Collect jamb lines (~wall thickness: 150-600 mm)
            # Also collect short jamb-offset lines (20-149 mm) which connect
            # the wall opening edge to the door hinge (length = jamb inset).
            jamb_lines = []
            jamb_offset_lines = []
            for p in door_paths:
                for it in p.get("items", []):
                    if it[0] == "l":
                        p1, p2 = it[1], it[2]
                        wx1, wy1 = to_world(p1.x, p1.y)
                        wx2, wy2 = to_world(p2.x, p2.y)
                        ln = math.hypot(wx2 - wx1, wy2 - wy1)
                        entry = {
                            "p1": (wx1, wy1), "p2": (wx2, wy2), "len": ln,
                            "raw_p1": (p1.x, p1.y), "raw_p2": (p2.x, p2.y),
                        }
                        if 150 <= ln <= 600:
                            jamb_lines.append(entry)
                        elif 20 <= ln <= 149:
                            jamb_offset_lines.append(entry)

            # Build raw door candidates from arc groups
            raw_doors = []
            for arc_curves in arc_groups:
                # Arc start = latch side of opening; arc end = door tip (open position)
                hinge_pt = arc_curves[0][1]
                end_pt   = arc_curves[-1][4]
                hx, hy   = to_world(hinge_pt.x, hinge_pt.y)
                ex, ey   = to_world(end_pt.x, end_pt.y)

                # -- Opening centre & wall direction --------------------------
                # The physical hinge (arc pivot) is at the right-angle corner of
                # the arc -- either (ex, hy) or (hx, ey).
                # Choose the candidate closest to any jamb line endpoint.
                c1 = (ex, hy)
                c2 = (hx, ey)
                if jamb_lines:
                    d1 = min(
                        min(math.hypot(j["p1"][0]-c1[0], j["p1"][1]-c1[1]),
                            math.hypot(j["p2"][0]-c1[0], j["p2"][1]-c1[1]))
                        for j in jamb_lines
                    )
                    d2 = min(
                        min(math.hypot(j["p1"][0]-c2[0], j["p1"][1]-c2[1]),
                            math.hypot(j["p2"][0]-c2[0], j["p2"][1]-c2[1]))
                        for j in jamb_lines
                    )
                    arc_center = c1 if d1 < d2 else c2
                else:
                    arc_center = c1  # default

                # -- Door width from arc radius (fallback) --------------------
                leaf_w = math.hypot(hx - arc_center[0], hy - arc_center[1])

                # Sanity filter: skip arcs outside plausible door-width range
                if not (600 <= leaf_w <= 1400):
                    continue

                # -- Exact opening width from stop-indicator lines -----------
                # Stop-indicator lines (20-149mm) in the Doors layer sit at
                # each jamb face.  Their outer endpoints span the exact rough
                # opening width as drawn.  Search within 2x leaf radius.
                STOP_RADIUS = leaf_w * 2.0
                nearby_stops = [
                    jl for jl in jamb_offset_lines
                    if min(
                        math.hypot(jl["p1"][0]-arc_center[0], jl["p1"][1]-arc_center[1]),
                        math.hypot(jl["p2"][0]-arc_center[0], jl["p2"][1]-arc_center[1])
                    ) < STOP_RADIUS
                ]

                if nearby_stops:
                    stop_pts = [pt for jl in nearby_stops for pt in [jl["p1"], jl["p2"]]]
                    sx = [p[0] for p in stop_pts]
                    sy = [p[1] for p in stop_pts]
                    span_x = max(sx) - min(sx)
                    span_y = max(sy) - min(sy)
                    if span_x >= span_y:
                        door_w = span_x
                        wall_dir_label = "horizontal"
                        cx = (min(sx) + max(sx)) / 2
                        cy = arc_center[1]
                    else:
                        door_w = span_y
                        wall_dir_label = "vertical"
                        cx = arc_center[0]
                        cy = (min(sy) + max(sy)) / 2
                else:
                    # No stop lines found -- fall back to leaf width and arc geometry
                    stop_pts = []
                    door_w = leaf_w
                    cx = arc_center[0]
                    cy = arc_center[1]
                    wall_dx = abs(hx - arc_center[0])
                    wall_dy = abs(hy - arc_center[1])
                    wall_dir_label = "vertical" if wall_dx > wall_dy else "horizontal"

                raw_doors.append({
                    "cx": cx, "cy": cy, "width": door_w, "ori": wall_dir_label,
                    "_hinge_x": cx, "_hinge_y": cy,
                    "_stop_pts": stop_pts,
                })

            # Merge duplicate arcs (two arc symbols for same physical opening)
            merged = _merge_nearby(raw_doors, DOOR_MERGE_DIST_MM, jamb_offset_lines)

            door_count = 0
            for d in merged:
                door_count += 1
                ori = d["ori"]
                cx_d, cy_d = d["cx"], d["cy"]

                # --- Compute true wall centreline from nearby jamb lines -------
                # Jamb lines (150-600 mm) span the full wall thickness.
                # Their midpoint is the wall centreline in the perp direction.
                JAMB_SEARCH = d["width"] * 0.75 + 50.0
                nearby_jambs = [
                    jl for jl in jamb_lines
                    if min(
                        math.hypot(jl["p1"][0] - cx_d, jl["p1"][1] - cy_d),
                        math.hypot(jl["p2"][0] - cx_d, jl["p2"][1] - cy_d),
                    ) < JAMB_SEARCH
                ]
                if nearby_jambs:
                    if ori == "horizontal":
                        # Jamb lines run along Y; midpoint Y = wall centre Y
                        mid_perps = [(jl["p1"][1] + jl["p2"][1]) / 2 for jl in nearby_jambs]
                    else:
                        # Jamb lines run along X; midpoint X = wall centre X
                        mid_perps = [(jl["p1"][0] + jl["p2"][0]) / 2 for jl in nearby_jambs]
                    wall_centre_perp = sum(mid_perps) / len(mid_perps)
                else:
                    # Fallback: offset hinge by half wall thickness
                    half_t = cfg.wall_thickness_mm / 2.0
                    wall_centre_perp = (cy_d + half_t) if ori == "horizontal" else (cx_d + half_t)

                openings.append(Opening(
                    opening_id       = f"D{door_count:02d}",
                    opening_type     = OpeningType.DOOR,
                    wall_id          = f"OCG_DOOR_{door_count:02d}",
                    wall_direction   = ori,
                    width_mm         = d["width"],
                    cx_world_mm      = cx_d,
                    cy_world_mm      = cy_d,
                    # Populate legacy pt fields from world coords (approx, for display)
                    gap_mid_pts      = cx_d / self.config.k,
                    fixed_coord_pts  = cy_d / self.config.k,
                    gap_width_pts    = d["width"] / self.config.k,
                    gap_start_pts    = (cx_d - d["width"]/2) / self.config.k,
                    gap_end_pts      = (cx_d + d["width"]/2) / self.config.k,
                    wall_thickness_mm    = cfg.wall_thickness_mm,
                    wall_centre_perp_mm  = wall_centre_perp,
                    z_void_bottom_mm = 0.0,
                    z_void_top_mm    = cfg.door_head_height_mm,
                    lintel_thickness_mm = cfg.lintel_thickness_mm,
                    lintel_type      = cfg.lintel_type,
                    pause_for_lintel = cfg.pause_for_lintel,
                ))

        # -- WINDOWS ---------------------------------------------------------
        if "Windows" in available_layers:
            win_paths = [p for p in paths if p.get("layer") == "Windows"]

            win_segs = []
            for p in win_paths:
                for it in p.get("items", []):
                    if it[0] == "l":
                        p1, p2 = it[1], it[2]
                        wx1, wy1 = to_world(p1.x, p1.y)
                        wx2, wy2 = to_world(p2.x, p2.y)
                        ln = math.hypot(wx2 - wx1, wy2 - wy1)
                        win_segs.append({
                            "p1": (wx1, wy1), "p2": (wx2, wy2), "len": ln
                        })

            # Cluster into individual window groups
            all_pts = [(s["p1"][0], s["p1"][1]) for s in win_segs] + \
                      [(s["p2"][0], s["p2"][1]) for s in win_segs]
            clusters = _cluster_points(all_pts, gap_mm=2000.0)

            win_count = 0
            for cluster_pts in clusters:
                xs = [p[0] for p in cluster_pts]
                ys = [p[1] for p in cluster_pts]
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2

                # Width = longest segment in this cluster
                def near(seg):
                    return any(
                        math.hypot(seg["p1"][0]-cp[0], seg["p1"][1]-cp[1]) < 2000
                        for cp in cluster_pts
                    )
                cluster_segs = [s for s in win_segs if near(s)]
                if cluster_segs:
                    # Wall orientation from dominant segment direction
                    total_dx = sum(abs(s["p2"][0] - s["p1"][0]) for s in cluster_segs)
                    total_dy = sum(abs(s["p2"][1] - s["p1"][1]) for s in cluster_segs)
                    wall_ori = "vertical" if total_dy > total_dx else "horizontal"
                    # Width = median of WALL-ALIGNED line lengths.
                    # A window CAD symbol has: 2 wall-face lines (correct width),
                    # 1 outer sill line (slightly longer), glass pane lines (slightly
                    # shorter), and perpendicular cross-lines (wrong direction).
                    # Filter to segments aligned with the wall, then take median to
                    # pick the most common (wall-face) length, excluding the sill.
                    if wall_ori == "vertical":
                        aligned = [s for s in cluster_segs
                                   if abs(s["p2"][1]-s["p1"][1]) > abs(s["p2"][0]-s["p1"][0])]
                    else:
                        aligned = [s for s in cluster_segs
                                   if abs(s["p2"][0]-s["p1"][0]) > abs(s["p2"][1]-s["p1"][1])]
                    use_segs = aligned if aligned else cluster_segs
                    lens = sorted(s["len"] for s in use_segs)
                    width = statistics.median(lens)
                else:
                    span_x = max(xs) - min(xs); span_y = max(ys) - min(ys)
                    width   = max(span_x, span_y)
                    wall_ori = "vertical" if span_y > span_x else "horizontal"

                # Window symbol bbox is centred on the wall centreline.
                # Perp direction: Y for horizontal wall, X for vertical wall.
                win_centre_perp = cy if wall_ori == "horizontal" else cx

                win_count += 1
                openings.append(Opening(
                    opening_id       = f"W{win_count:02d}",
                    opening_type     = OpeningType.WINDOW,
                    wall_id          = f"OCG_WIN_{win_count:02d}",
                    wall_direction   = wall_ori,
                    width_mm         = width,
                    cx_world_mm      = cx,
                    cy_world_mm      = cy,
                    gap_mid_pts      = cx / self.config.k,
                    fixed_coord_pts  = cy / self.config.k,
                    gap_width_pts    = width / self.config.k,
                    gap_start_pts    = (cx - width/2) / self.config.k,
                    gap_end_pts      = (cx + width/2) / self.config.k,
                    wall_thickness_mm    = cfg.wall_thickness_mm,
                    wall_centre_perp_mm  = win_centre_perp,
                    z_void_bottom_mm = cfg.window_sill_height_mm,
                    z_void_top_mm    = cfg.window_head_height_mm,
                    lintel_thickness_mm = cfg.lintel_thickness_mm,
                    lintel_type      = cfg.lintel_type,
                    pause_for_lintel = cfg.pause_for_lintel,
                ))

        return openings

    # -- Strategy B: pdfplumber gap detection (unchanged fallback) -----------

    def _detect_from_gaps(self, include_unmatched_as_windows: bool) -> List[Opening]:
        with pdfplumber.open(self.pdf_path) as pdf:
            page = pdf.pages[0]
            self.page_height = page.height
            self._calibrate_scale(page)
            wall_gaps  = self._detect_wall_gaps(page.lines)
            door_arcs  = self._detect_door_arcs(page.curves)

        matched   = self._match_gaps_to_arcs(wall_gaps, door_arcs)
        unmatched = self._find_unmatched_gaps(wall_gaps, matched) if include_unmatched_as_windows else []
        return self._build_openings(matched, unmatched)

    def _calibrate_scale(self, page) -> None:
        scale = self.config.drawing_scale
        self.scale_pts_to_mm = scale * 0.3528

    def _pts_to_mm(self, pts: float) -> float:
        return pts * self.scale_pts_to_mm

    def _detect_wall_gaps(self, lines: list) -> List[WallGap]:
        gaps: List[WallGap] = []
        wall_counter = [0]

        h_lines  = [l for l in lines if abs(l["y0"] - l["y1"]) < self.LINE_GROUP_THRESH]
        h_groups = defaultdict(list)
        for l in h_lines:
            y_key = round(l["y0"] / self.LINE_GROUP_THRESH) * self.LINE_GROUP_THRESH
            h_groups[y_key].append(l)
        for y, segs in h_groups.items():
            if len(segs) < 2: continue
            segs_sorted = sorted(segs, key=lambda l: l["x0"])
            wall_id = f"HW_{wall_counter[0]:03d}"; wall_counter[0] += 1
            for i in range(len(segs_sorted) - 1):
                end = segs_sorted[i]["x1"]; start = segs_sorted[i+1]["x0"]
                gw_pts = start - end
                if gw_pts <= 0: continue
                gw_mm = self._pts_to_mm(gw_pts)
                if self.DOOR_WIDTH_MIN_MM <= gw_mm <= self.DOOR_WIDTH_MAX_MM:
                    gaps.append(WallGap(wall_id, "horizontal", y, end, start,
                                        (end+start)/2, gw_pts, gw_mm))

        v_lines  = [l for l in lines if abs(l["x0"] - l["x1"]) < self.LINE_GROUP_THRESH]
        v_groups = defaultdict(list)
        for l in v_lines:
            x_key = round(l["x0"] / self.LINE_GROUP_THRESH) * self.LINE_GROUP_THRESH
            v_groups[x_key].append(l)
        for x, segs in v_groups.items():
            if len(segs) < 2: continue
            segs_sorted = sorted(segs, key=lambda l: l["y0"])
            wall_id = f"VW_{wall_counter[0]:03d}"; wall_counter[0] += 1
            for i in range(len(segs_sorted) - 1):
                end = segs_sorted[i]["y1"]; start = segs_sorted[i+1]["y0"]
                gw_pts = start - end
                if gw_pts <= 0: continue
                gw_mm = self._pts_to_mm(gw_pts)
                if self.DOOR_WIDTH_MIN_MM <= gw_mm <= self.DOOR_WIDTH_MAX_MM:
                    gaps.append(WallGap(wall_id, "vertical", x, end, start,
                                        (end+start)/2, gw_pts, gw_mm))
        return gaps

    def _detect_door_arcs(self, curves: list) -> List[DoorArc]:
        arcs = []
        for i, c in enumerate(curves):
            pts = c["pts"]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            w = max(xs)-min(xs); h = max(ys)-min(ys)
            size = max(w, h)
            if size < self.ARC_MIN_BBOX_PTS or size > self.ARC_MAX_BBOX_PTS: continue
            arcs.append(DoorArc(i, (min(xs)+max(xs))/2, (min(ys)+max(ys))/2,
                                size/2, w, h))
        return arcs

    def _match_gaps_to_arcs(self, gaps, arcs):
        matched = []; used_arcs = set()
        for gap in gaps:
            best_arc = None; best_dist = float("inf")
            for arc in arcs:
                if arc.index in used_arcs: continue
                dist = math.hypot(arc.cx - gap.cx, arc.cy - gap.cy)
                tol  = gap.gap_width_pts / 2 + ARC_GAP_MATCH_TOLERANCE
                if dist < tol and dist < best_dist:
                    best_dist = dist; best_arc = arc
            if best_arc:
                matched.append((gap, best_arc)); used_arcs.add(best_arc.index)
        return matched

    def _find_unmatched_gaps(self, all_gaps, matched):
        matched_gaps = {id(g) for g, _ in matched}
        return [g for g in all_gaps if id(g) not in matched_gaps]

    def _build_openings(self, matched_doors, unmatched_windows) -> List[Opening]:
        cfg = self.config
        k   = cfg.k
        ph  = None  # page_height not available here; use 0 (Y coord is approximate)
        openings = []; door_num = 1; win_num = 1

        for gap, arc in matched_doors:
            # Approximate world mm from pts (no Y-flip available in gap fallback)
            cx_mm = gap.cx * k; cy_mm = gap.cy * k
            # wall_centre_perp: fixed_coord is the wall line position in pts
            fixed_mm = gap.fixed_coord * k
            door_centre_perp = fixed_mm  # best approximation without Y-flip
            openings.append(Opening(
                opening_id       = f"D{door_num:02d}",
                opening_type     = OpeningType.DOOR,
                wall_id          = gap.wall_id,
                wall_direction   = gap.direction,
                width_mm         = gap.gap_width_mm,
                cx_world_mm      = cx_mm,
                cy_world_mm      = cy_mm,
                gap_start_pts    = gap.gap_start,
                gap_end_pts      = gap.gap_end,
                gap_mid_pts      = gap.gap_mid,
                fixed_coord_pts  = gap.fixed_coord,
                gap_width_pts    = gap.gap_width_pts,
                wall_thickness_mm    = cfg.wall_thickness_mm,
                wall_centre_perp_mm  = door_centre_perp,
                z_void_bottom_mm = 0.0,
                z_void_top_mm    = cfg.door_head_height_mm,
                lintel_thickness_mm = cfg.lintel_thickness_mm,
                lintel_type      = cfg.lintel_type,
                pause_for_lintel = cfg.pause_for_lintel,
            ))
            door_num += 1

        for gap in unmatched_windows:
            cx_mm = gap.cx * k; cy_mm = gap.cy * k
            fixed_mm = gap.fixed_coord * k
            win_centre_perp = fixed_mm
            openings.append(Opening(
                opening_id       = f"W{win_num:02d}",
                opening_type     = OpeningType.WINDOW,
                wall_id          = gap.wall_id,
                wall_direction   = gap.direction,
                width_mm         = gap.gap_width_mm,
                cx_world_mm      = cx_mm,
                cy_world_mm      = cy_mm,
                gap_start_pts    = gap.gap_start,
                gap_end_pts      = gap.gap_end,
                gap_mid_pts      = gap.gap_mid,
                fixed_coord_pts  = gap.fixed_coord,
                gap_width_pts    = gap.gap_width_pts,
                wall_thickness_mm    = cfg.wall_thickness_mm,
                wall_centre_perp_mm  = win_centre_perp,
                z_void_bottom_mm = cfg.window_sill_height_mm,
                z_void_top_mm    = cfg.window_head_height_mm,
                lintel_thickness_mm = cfg.lintel_thickness_mm,
                lintel_type      = cfg.lintel_type,
                pause_for_lintel = cfg.pause_for_lintel,
            ))
            win_num += 1

        openings.sort(key=lambda o: (o.opening_type.value, o.wall_id, o.gap_mid_pts))
        return openings


# ---------------------------------------------------------------------------
# Clustering / merging helpers
# ---------------------------------------------------------------------------

def _merge_nearby(items: list, dist: float, jamb_offsets: list = None) -> list:
    """Merge nearby door candidates into single openings.

    For double doors, the two hinges straddle the wall gap.
    Opening width = hinge_span + left_jamb_offset + right_jamb_offset,
    where the jamb offset lines (short lines in the Doors layer connecting
    the wall opening edge to the hinge) supply the exact inset on each side.
    """
    used = [False] * len(items); merged = []
    jamb_offsets = jamb_offsets or []

    for i, a in enumerate(items):
        if used[i]: continue
        group = [a]; used[i] = True
        for j, b in enumerate(items):
            if used[j]: continue
            if math.hypot(b["cx"]-a["cx"], b["cy"]-a["cy"]) < dist:
                group.append(b); used[j] = True

        # Pool stop-indicator endpoints from all members.
        # Single door: its own 4 stop-line pts span the rough opening.
        # Double door: left + right leaf pts combined span the full opening.
        all_stop_pts = [pt for g in group for pt in g.get("_stop_pts", [])]

        if all_stop_pts:
            sx = [p[0] for p in all_stop_pts]
            sy = [p[1] for p in all_stop_pts]
            span_x = max(sx) - min(sx)
            span_y = max(sy) - min(sy)
            total_width = max(span_x, span_y)
            merged_ori = "horizontal" if span_x >= span_y else "vertical"
            if merged_ori == "horizontal":
                merged_cx = (min(sx) + max(sx)) / 2
                merged_cy = sum(g["cy"] for g in group) / len(group)
            else:
                merged_cx = sum(g["cx"] for g in group) / len(group)
                merged_cy = (min(sy) + max(sy)) / 2
        else:
            # Fallback: use pre-computed per-leaf widths and hinge span
            merged_ori = group[0]["ori"]
            if len(group) == 1:
                total_width = group[0]["width"]
            else:
                leaf_w = max(g["width"] for g in group)
                if merged_ori == "horizontal":
                    span = max(g["_hinge_x"] for g in group) - min(g["_hinge_x"] for g in group)
                else:
                    span = max(g["_hinge_y"] for g in group) - min(g["_hinge_y"] for g in group)
                total_width = span + leaf_w * 0.18
            merged_cx = sum(g["cx"] for g in group) / len(group)
            merged_cy = sum(g["cy"] for g in group) / len(group)

        merged.append({
            "cx":       merged_cx,
            "cy":       merged_cy,
            "width":    total_width,
            "ori":      merged_ori,
            "_hinge_x": sum(g["_hinge_x"] for g in group) / len(group),
            "_hinge_y": sum(g["_hinge_y"] for g in group) / len(group),
        })
    return merged
def _cluster_points(points: list, gap_mm: float) -> list:
    if not points: return []
    clusters = []; used = [False]*len(points)
    for i, pt in enumerate(points):
        if used[i]: continue
        cluster = [pt]; used[i] = True
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(points):
                if used[j]: continue
                if any(math.hypot(other[0]-c[0], other[1]-c[1]) < gap_mm for c in cluster):
                    cluster.append(other); used[j] = True; changed = True
        clusters.append(cluster)
    return clusters


# ---------------------------------------------------------------------------
# Layer Scheduler  (unchanged)
# ---------------------------------------------------------------------------

class LayerScheduler:
    def __init__(self, openings: List[Opening], config: ProjectConfig):
        self.openings = openings
        self.config   = config

    def get_layer_actions(self, layer_z_mm: float) -> dict:
        actions = {}
        for op in self.openings:
            action = op.get_print_action(layer_z_mm)
            if action != PrintAction.FULL_WALL:
                actions[op.opening_id] = action
        return actions

    def generate_layer_schedule(self) -> List[dict]:
        schedule = []
        n_layers = int(self.config.wall_height_mm / self.config.layer_height_mm)
        for i in range(n_layers):
            z = i * self.config.layer_height_mm
            actions = self.get_layer_actions(z)
            schedule.append({
                "layer_index"     : i,
                "z_mm"            : z,
                "actions"         : actions,
                "has_suppression" : any(a == PrintAction.SPLIT_PIERS for a in actions.values()),
                "has_pause"       : any(a == PrintAction.PAUSE_LINTEL for a in actions.values()),
            })
        return schedule


# ---------------------------------------------------------------------------
# G-code Generator  (unchanged)
# ---------------------------------------------------------------------------

class GCodeGenerator:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def generate_layer_header(self, layer_index, z_mm):
        return ["", f"; -- Layer {layer_index} | Z = {z_mm:.1f} mm ----------------------"]

    def generate_full_wall(self, wall_id, z_mm):
        return [f"G1 ; {wall_id} full wall @ Z={z_mm:.1f}mm"]

    def generate_split_piers(self, wall_id, opening, z_mm):
        return [
            f"G1 ; {wall_id} LEFT pier of {opening.opening_id} @ Z={z_mm:.1f}mm",
            f"G0 ; travel across {opening.opening_id} void (no extrusion)",
            f"G1 ; {wall_id} RIGHT pier of {opening.opening_id} @ Z={z_mm:.1f}mm",
        ]

    def generate_lintel_pause(self, opening, z_mm):
        return [
            "",
            "; ===========================================================",
            f"; PAUSE -- Install {opening.lintel_type.value} lintel for {opening.opening_id}",
            f"; Opening width: {opening.width_mm:.0f}mm",
            f"; Lintel seat level: Z = {z_mm:.1f}mm",
            f"; Resume printing at Z = {opening.z_resume_mm:.1f}mm",
            "; ===========================================================",
            "M0 ; pause for lintel installation",
            "",
        ]

    def generate_skip_comment(self, opening, z_mm):
        return [
            f"; SKIP Z={z_mm:.1f}mm -- inside lintel of {opening.opening_id} "
            f"(lintel occupies {opening.z_void_top_mm:.0f}-{opening.z_resume_mm:.0f}mm)"
        ]

    def generate_full_schedule_gcode(self, schedule, openings) -> str:
        opening_map = {op.opening_id: op for op in openings}
        lines = [
            "; PrintPlan AI -- Z-Aware Toolpath with Opening Detection",
            f"; Layer height: {self.config.layer_height_mm}mm",
            f"; Wall height:  {self.config.wall_height_mm}mm",
            f"; Total layers: {len(schedule)}",
            f"; Openings:     {len(openings)}",
            "", "G21 ; units = mm", "G90 ; absolute positioning", "",
        ]
        lintel_paused = set()
        for layer in schedule:
            z = layer["z_mm"]; idx = layer["layer_index"]
            actions = layer["actions"]
            lines += self.generate_layer_header(idx, z)
            if not actions:
                lines.append(f"G1 ; all walls -- full contour @ Z={z:.1f}mm")
                continue
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
            lines.append(f"G1 ; remaining walls -- full contour @ Z={z:.1f}mm")
        lines += ["", "; -- Print complete --------------------------------------", "M2"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streamlit helpers  (unchanged API so app.py works without changes)
# ---------------------------------------------------------------------------

def run_opening_detection_ui(pdf_path: str):
    import streamlit as st
    st.subheader("? Project Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        layer_h = st.number_input("Layer height (mm)", value=50, min_value=10, max_value=200, step=5)
        wall_h  = st.number_input("Wall height (mm)",  value=2700, min_value=1000, max_value=6000, step=100)
    with col2:
        door_hh  = st.number_input("Door head height (mm)",   value=2100, min_value=1800, max_value=3000, step=50)
        win_sill = st.number_input("Window sill height (mm)", value=900,  min_value=0, max_value=2000, step=50)
        win_hh   = st.number_input("Window head height (mm)", value=2100, min_value=500, max_value=3000, step=50)
    with col3:
        lintel_t  = st.number_input("Lintel thickness (mm)", value=200, min_value=50, max_value=500, step=25)
        lintel_tp = st.selectbox("Lintel type", ["timber","steel","precast","none"])
        pause     = st.checkbox("Pause G-code for lintel", value=True)

    config = ProjectConfig(
        layer_height_mm       = float(layer_h),
        wall_height_mm        = float(wall_h),
        lintel_thickness_mm   = float(lintel_t),
        lintel_type           = LinteldType(lintel_tp),
        pause_for_lintel      = pause,
        door_head_height_mm   = float(door_hh),
        window_sill_height_mm = float(win_sill),
        window_head_height_mm = float(win_hh),
    )

    st.subheader("? Opening Detection")
    with st.spinner("Scanning floor plan for doors and windows..."):
        parser   = FloorPlanParser(pdf_path, config)
        openings = parser.detect_openings()

    if not openings:
        st.warning("No openings detected.")
        return [], config

    st.success(f"Detected {len(openings)} opening(s)")
    st.subheader("? Detected Openings -- Review & Override")

    for op in openings:
        with st.expander(f"{op.opening_id} -- {op.opening_type.value.title()} | Width: {op.width_mm:.0f}mm | Wall: {op.wall_id}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                void_bot = 0.0 if op.opening_type == OpeningType.DOOR else st.number_input(
                    "Sill height (mm)", value=int(op.z_void_bottom_mm), min_value=0, max_value=2000, step=50, key=f"sill_{op.opening_id}")
            with c2:
                void_top = st.number_input("Head height (mm)", value=int(op.z_void_top_mm), min_value=500, max_value=3000, step=50, key=f"head_{op.opening_id}")
            with c3:
                lin_t = st.number_input("Lintel thick. (mm)", value=int(op.lintel_thickness_mm), min_value=0, max_value=500, step=25, key=f"lint_{op.opening_id}")
            op.z_void_bottom_mm    = float(void_bot)
            op.z_void_top_mm       = float(void_top)
            op.lintel_thickness_mm = float(lin_t)
            op.is_user_overridden  = True

    return openings, config


def generate_gcode_from_openings(openings, config) -> str:
    scheduler = LayerScheduler(openings, config)
    schedule  = scheduler.generate_layer_schedule()
    generator = GCodeGenerator(config)
    return generator.generate_full_schedule_gcode(schedule, openings)


# ---------------------------------------------------------------------------
# Real XY G-code generation from Stage3Output + openings
# ---------------------------------------------------------------------------

def _point_in_opening_xy_od(px: float, py: float, opening: "Opening") -> bool:
    """Check if world-mm point is within opening XY footprint (with 50mm tolerance)."""
    half_w = opening.width_mm / 2.0 + 50.0
    perp   = 300.0 / 2.0 + 50.0
    if opening.wall_direction == "horizontal":
        in_x = abs(px - opening.cx_world_mm) <= half_w
        in_y = abs(py - opening.cy_world_mm) <= perp
    else:
        in_x = abs(px - opening.cx_world_mm) <= perp
        in_y = abs(py - opening.cy_world_mm) <= half_w
    return in_x and in_y


def _clip_trace_for_openings(
    pts: list,
    openings: list,
    z_mm: float,
) -> list:
    """
    Split a polyline at opening boundaries for SPLIT_PIERS layers.
    Returns a list of sub-polylines; segments entirely inside an opening void
    are dropped; segments crossing a boundary are split at the boundary midpoint.

    Returns list of lists-of-points (each sub-list is a continuous run).
    """
    if not pts or len(pts) < 2:
        return [pts] if pts else []

    # Classify each point as inside (True) or outside (False) any active opening void
    def in_any_void(p):
        for op in openings:
            action = op.get_print_action(z_mm)
            if action == PrintAction.SPLIT_PIERS:
                if _point_in_opening_xy_od(p[0], p[1], op):
                    return True
        return False

    runs = []
    current_run = []
    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        in0 = in_any_void(p0)
        in1 = in_any_void(p1)

        if not in0 and not in1:
            # Both outside — keep segment
            if not current_run:
                current_run.append(p0)
            current_run.append(p1)
        elif in0 and in1:
            # Both inside void — skip; end current run
            if current_run:
                runs.append(current_run)
                current_run = []
        elif not in0 and in1:
            # Crossing into void — add p0, split at midpoint, end run
            mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            if not current_run:
                current_run.append(p0)
            current_run.append(mid)
            runs.append(current_run)
            current_run = []
        else:
            # Crossing out of void — start new run from midpoint
            mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            current_run = [mid, p1]

    if current_run and len(current_run) >= 2:
        runs.append(current_run)

    return [r for r in runs if len(r) >= 2]


def synthesise_gcode_with_openings(
    stage3,
    m4_params,
    openings: list,
    *,
    include_travel: bool = True,
) -> str:
    """
    Generate real XY G-code from Stage3Output with opening awareness.

    For each layer:
      - z < void_bottom           : print FULL trace (no clipping)
      - void_bottom <= z < void_top: SPLIT_PIERS — clip traces at opening boundary
      - z == void_top              : M0 pause for lintel (once per opening)
      - void_top < z < z_resume   : SKIP — suppress segments over opening XY
      - z >= z_resume             : FULL_WALL — print everything (continuous over lintel)

    Output is real G-code: G0 X Y Z F for travel, G1 X Y Z E F for print.
    """
    ps  = m4_params.print_speed_mm_s
    ts  = m4_params.travel_speed_mm_s
    bw  = m4_params.bead_width_mm
    lh  = getattr(m4_params, "layer_height_mm", 20.0)   # may come from M3Params
    pv  = m4_params.pump_volume_per_unit_e
    zl  = m4_params.z_lift_mm

    lines = [
        "; PrintPlan AI -- Opening-Aware G-code (Real XY)",
        f"; Bead width: {bw}mm | Layer height: {lh}mm",
        f"; Print speed: {ps}mm/s | Travel speed: {ts}mm/s",
        f"; Openings: {len(openings)}",
        "",
        "G21 ; mm units",
        "G90 ; absolute positioning",
        "G92 E0 ; reset extruder",
        "",
    ]

    e_pos      = 0.0
    lintel_done = set()   # opening IDs where M0 has been emitted

    for layer in stage3.layers:
        z_mm  = layer.z_mm
        idx   = layer.index
        lines.append(f"; --- Layer {idx}  Z={z_mm:.1f}mm ---")

        # Determine per-opening actions at this Z
        actions = {}
        for op in openings:
            actions[op.opening_id] = op.get_print_action(z_mm)

        # Emit M0 pause for any opening that needs lintel at this Z (once only)
        for op in openings:
            if actions[op.opening_id] == PrintAction.PAUSE_LINTEL:
                if op.opening_id not in lintel_done:
                    lines += [
                        f"; ===== PAUSE: install {op.lintel_type.value} lintel for {op.opening_id} =====",
                        f"; Opening width={op.width_mm:.0f}mm  Z_seat={z_mm:.1f}mm  Z_resume={op.z_resume_mm:.1f}mm",
                        "M0 ; pause -- install lintel",
                        "",
                    ]
                    lintel_done.add(op.opening_id)

        prev_end = None   # last printed XYZ position this layer

        for tr in layer.traces:
            pts = tr.points
            if len(pts) < 2:
                continue
            is_print = tr.kind == "print"

            if is_print:
                # Determine if this trace needs splitting / skipping
                # For each opening that is SPLIT_PIERS or SKIP, filter accordingly
                needs_split = any(
                    actions[op.opening_id] == PrintAction.SPLIT_PIERS
                    for op in openings
                )
                needs_skip = any(
                    actions[op.opening_id] == PrintAction.SKIP and
                    _point_in_opening_xy_od(
                        sum(p[0] for p in pts) / len(pts),
                        sum(p[1] for p in pts) / len(pts),
                        op,
                    )
                    for op in openings
                )

                if needs_skip:
                    lines.append(f"; SKIP trace (inside lintel zone of opening)")
                    continue

                if needs_split:
                    sub_runs = _clip_trace_for_openings(pts, openings, z_mm)
                else:
                    sub_runs = [pts]

                for run in sub_runs:
                    if len(run) < 2:
                        continue
                    # Travel to start of run
                    sx, sy = run[0]
                    if include_travel:
                        lines.append(
                            f"G0 X{sx:.3f} Y{sy:.3f} Z{(z_mm + zl):.3f} F{ts * 60:.0f}"
                        )
                        lines.append(
                            f"G0 X{sx:.3f} Y{sy:.3f} Z{z_mm:.3f} F{ts * 60:.0f}"
                        )
                    # Print moves
                    for p in run[1:]:
                        dx = p[0] - sx; dy = p[1] - sy
                        seg_len = math.hypot(dx, dy)
                        e_pos += seg_len * pv
                        lines.append(
                            f"G1 X{p[0]:.3f} Y{p[1]:.3f} Z{z_mm:.3f}"
                            f" E{e_pos:.4f} F{ps * 60:.0f}"
                        )
                        sx, sy = p

                    prev_end = (run[-1][0], run[-1][1])

            else:
                # Travel move
                if include_travel and len(pts) >= 2:
                    ex, ey = pts[-1]
                    lines.append(
                        f"G0 X{ex:.3f} Y{ey:.3f} Z{z_mm:.3f} F{ts * 60:.0f}"
                    )

        lines.append("")

    lines += [
        "; === Print complete ===",
        "M2",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "2308_Floorplan_drawing.pdf"
    cfg = ProjectConfig(drawing_scale=100)
    ops = FloorPlanParser(pdf, cfg).detect_openings()
    print(f"\nDetected {len(ops)} opening(s):\n")
    for o in ops:
        print(f"  {o.summary()}")
