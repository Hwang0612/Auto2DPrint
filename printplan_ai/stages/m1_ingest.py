"""Stage M1 — Vector-native PDF ingestion.

Reads a CAD-exported vector PDF and extracts wall geometry from the
``2D Walls`` OCG layer (or any layer whose name matches the configured
regex). Outputs world-coordinate line segments in a Pydantic contract
that Stage M2 consumes.
"""

from __future__ import annotations
import re
from pathlib import Path

import fitz  # PyMuPDF

from printplan_ai.config import M1Params, PT_TO_MM
from printplan_ai.models import (
    CoordinateFrame,
    Stage1Output,
    WallSegment,
)


def _detect_scale_geometric(page) -> tuple[float, float, int] | None:
    """Detect drawing scale by cross-referencing dimension text with
    measured PDF line lengths.

    Algorithm
    ---------
    1. Extract all numeric text blocks and their positions (likely
       dimension annotations: "300", "3600", "10000", etc.).
    2. Extract all thin line segments from dimension layers (or by
       stroke width if no layers).
    3. For each text block, find the nearest line segment within a
       proximity threshold.
    4. Compute the scale ratio: declared_value / measured_length.
    5. Apply median-voting across all matched pairs to reject outliers
       (mismatched text-to-line associations).
    6. Compute confidence as the fraction of inliers (within ±10% of
       the median) among all matches.

    Returns
    -------
    (scale, confidence, n_samples) or None if insufficient matches.
    """
    import math
    from collections import Counter

    # ── Step 1: Extract numeric text with positions ───────────────
    text_dict = page.get_text("dict")
    dim_texts: list[dict] = []
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = span["text"].strip()
                if re.match(r"^\d+\.?\d*$", txt):
                    val = float(txt)
                    if val < 10:                        # skip scale numbers like "1", "50"
                        continue
                    bbox = span["bbox"]
                    dim_texts.append({
                        "value": val,
                        "centre": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                    })

    if len(dim_texts) < 3:
        return None

    # ── Step 2: Extract dimension line candidates ─────────────────
    all_drawings = page.get_drawings(extended=True)
    dim_lines: list[dict] = []
    for d in all_drawings:
        if d.get("type") != "s":
            continue
        layer = d.get("layer") or ""
        w_mm = (d.get("width") or 0) * PT_TO_MM
        # Dimension lines: on a "dim" layer, or thin lines (< 0.15 mm)
        is_dim = ("dim" in layer.lower() or
                  (not layer and w_mm < 0.15) or
                  w_mm < 0.15)
        if not is_dim:
            continue
        for item in d["items"]:
            if item[0] != "l":
                continue
            p1 = (item[1].x, item[1].y)
            p2 = (item[2].x, item[2].y)
            length_pt = math.dist(p1, p2)
            if length_pt < 3:                           # skip tiny tick marks
                continue
            dim_lines.append({
                "p1": p1, "p2": p2,
                "length_pt": length_pt,
                "midpoint": ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
            })

    if len(dim_lines) < 3:
        return None

    # ── Step 3: Match text to nearest line ────────────────────────
    PROXIMITY_PT = 50                                   # max text-to-line distance (pt)
    raw_scales: list[float] = []
    for dt in dim_texts:
        best_dist = float("inf")
        best_line = None
        for dl in dim_lines:
            d = math.dist(dt["centre"], dl["midpoint"])
            if d < best_dist:
                best_dist = d
                best_line = dl
        if best_line is None or best_dist > PROXIMITY_PT:
            continue
        measured_mm = best_line["length_pt"] * PT_TO_MM
        if measured_mm < 0.1:
            continue
        scale = dt["value"] / measured_mm
        raw_scales.append(scale)

    if len(raw_scales) < 3:
        return None

    # ── Step 4: Median-voting with outlier rejection ──────────────
    raw_scales.sort()
    median = raw_scales[len(raw_scales) // 2]

    # Round median to nearest standard scale
    STANDARD_SCALES = [1, 2, 5, 10, 20, 25, 50, 75, 100, 150, 200, 250, 500]
    scale = min(STANDARD_SCALES, key=lambda s: abs(s - median))

    # ── Step 5: Confidence = fraction of inliers ──────────────────
    tolerance = 0.10                                    # ±10% of final scale
    inliers = sum(1 for s in raw_scales if abs(s - scale) / scale < tolerance)
    confidence = inliers / len(raw_scales)
    n_samples = len(raw_scales)

    # Require at least 50% inliers to trust the result
    if confidence < 0.50:
        return None

    return (float(scale), round(confidence, 3), n_samples)


def parse_pdf(
    pdf_path: str | Path,
    params: M1Params | None = None,
) -> Stage1Output:
    """Extract wall segments from a vector PDF floor plan.

    Parameters
    ----------
    pdf_path : path-like
        Path to a CAD-exported vector PDF whose wall layer is preserved
        as an Optional Content Group (OCG).
    params : M1Params, optional
        Override the default parsing parameters.

    Returns
    -------
    Stage1Output
        Coordinate frame, wall segments in world millimetres, and meta.
    """
    params = params or M1Params()
    pdf_path = Path(pdf_path)

    doc = fitz.open(pdf_path)
    page = doc[0]                                       # single-page floor plan
    page_h_pt = page.rect.height

    # ── Auto-detect drawing scale ─────────────────────────────────
    # Three-tier strategy applied in priority order:
    #   Tier 1: Geometric calibration — match dimension text values to
    #           measured PDF line lengths, compute scale from the ratio,
    #           use median-voting to reject outliers.
    #   Tier 2: Text extraction — search for "Scale: 1/N" in PDF text.
    #   Tier 3: User-provided or default fallback.
    drawing_scale = params.drawing_scale
    scale_method = "user_override"
    scale_confidence = 1.0
    scale_n_samples = 0

    if drawing_scale is None:
        # ── Tier 1: Geometric calibration ─────────────────────────
        detected = _detect_scale_geometric(page)
        if detected is not None:
            drawing_scale, scale_confidence, scale_n_samples = detected
            scale_method = "geometric_calibration"
        else:
            # ── Tier 2: Text search ───────────────────────────────
            text = page.get_text()
            scale_match = re.search(r"[Ss]cale\s*[:=]?\s*1\s*[:/]\s*(\d+)", text)
            if not scale_match:
                scale_match = re.search(r"1\s*[:/]\s*(\d+)", text)
            if scale_match:
                drawing_scale = float(scale_match.group(1))
                scale_method = "text_extraction"
            else:
                # ── Tier 3: Fallback ──────────────────────────────
                drawing_scale = 50.0
                scale_method = "default_fallback"

    # Coordinate transform: PDF points → world millimetres
    k_world = PT_TO_MM * drawing_scale

    def to_world(x_pt: float, y_pt: float) -> tuple[float, float]:
        x_world = x_pt * k_world
        y_world = (page_h_pt - y_pt) * k_world if params.y_axis_up else y_pt * k_world
        return (x_world, y_world)

    wall_regex = re.compile(params.wall_layer_pattern, re.IGNORECASE)

    # Determine classification strategy:
    # 1. If OCG layer names exist → filter by layer name regex
    # 2. If no layers → fall back to stroke width (thickest = walls)
    all_drawings = page.get_drawings(extended=True)
    has_layers = any(d.get("layer") for d in all_drawings if d.get("type") == "s")

    if not has_layers:
        # No layer names — classify by stroke width.
        # Find the thickest stroke width group (= walls in CAD convention)
        from collections import Counter
        width_counter: Counter = Counter()
        for d in all_drawings:
            if d.get("type") != "s":
                continue
            w_mm = (d.get("width") or 0) * PT_TO_MM
            width_counter[round(w_mm, 2)] += 1
        if width_counter:
            wall_width = max(width_counter.keys())
            wall_width_tol = 0.05  # mm tolerance
        else:
            wall_width = 0.5
            wall_width_tol = 0.05

    # Extract path segments
    walls: list[WallSegment] = []
    for drawing in all_drawings:
        if drawing.get("type") != "s":                  # stroked path only
            continue

        if has_layers:
            # Strategy 1: match layer name
            layer_name = drawing.get("layer") or ""
            if not wall_regex.search(layer_name):
                continue
        else:
            # Strategy 2: match stroke width (thickest group = walls)
            w_mm = (drawing.get("width") or 0) * PT_TO_MM
            if abs(w_mm - wall_width) > wall_width_tol:
                continue

        for item in drawing["items"]:
            if item[0] != "l":                          # skip curves/rects
                continue
            p1 = to_world(item[1].x, item[1].y)
            p2 = to_world(item[2].x, item[2].y)
            walls.append(WallSegment(p1=p1, p2=p2))

    page_w_pt = page.rect.width
    doc.close()

    frame = CoordinateFrame(
        k_world_mm_per_pt=k_world,
        page_bbox_world_mm=(
            0.0, 0.0,
            page_w_pt * k_world,
            page_h_pt * k_world,
        ),
        drawing_scale=f"1:{int(drawing_scale)}",
    )

    # Summary statistics
    def length(seg: WallSegment) -> float:
        return ((seg.p1[0] - seg.p2[0]) ** 2 + (seg.p1[1] - seg.p2[1]) ** 2) ** 0.5

    horiz = sum(1 for s in walls if abs(s.p1[1] - s.p2[1]) < 1)
    vert = sum(1 for s in walls if abs(s.p1[0] - s.p2[0]) < 1)
    oblique = len(walls) - horiz - vert
    total_length_m = sum(length(s) for s in walls) / 1000

    return Stage1Output(
        coordinate_frame=frame,
        walls=walls,
        meta={
            "source_pdf": pdf_path.name,
            "drawing_scale": f"1:{int(drawing_scale)}",
            "scale_method": scale_method,
            "scale_confidence": scale_confidence if scale_method == "geometric_calibration" else None,
            "scale_n_samples": scale_n_samples if scale_method == "geometric_calibration" else None,
            "detection_method": "layer_name" if has_layers else "stroke_width",
            "wall_filter": params.wall_layer_pattern if has_layers else f"{wall_width:.2f} mm",
            "segment_count": len(walls),
            "horizontal_count": horiz,
            "vertical_count": vert,
            "oblique_count": oblique,
            "total_length_m": round(total_length_m, 2),
            "orthogonal": oblique == 0,
        },
    )
