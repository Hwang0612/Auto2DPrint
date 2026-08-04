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

    # Coordinate transform: PDF points → world millimetres
    k_world = PT_TO_MM * params.drawing_scale

    def to_world(x_pt: float, y_pt: float) -> tuple[float, float]:
        x_world = x_pt * k_world
        y_world = (page_h_pt - y_pt) * k_world if params.y_axis_up else y_pt * k_world
        return (x_world, y_world)

    wall_regex = re.compile(params.wall_layer_pattern, re.IGNORECASE)

    # Extract path segments from the wall layer
    walls: list[WallSegment] = []
    for drawing in page.get_drawings(extended=True):
        if drawing.get("type") != "s":                  # stroked path only
            continue
        layer_name = drawing.get("layer") or ""
        if not wall_regex.search(layer_name):
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
        drawing_scale=f"1:{int(params.drawing_scale)}",
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
            "source_layer_pattern": params.wall_layer_pattern,
            "segment_count": len(walls),
            "horizontal_count": horiz,
            "vertical_count": vert,
            "oblique_count": oblique,
            "total_length_m": round(total_length_m, 2),
            "orthogonal": oblique == 0,
        },
    )
