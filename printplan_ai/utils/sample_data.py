"""PrintPlan AI — Sample Geometry & Demo Floor Plan Generator.

Generates realistic synthetic 3D concrete printing floor plan geometries (walls, rooms)
as Stage1Output structures so users can test the pipeline immediately without uploading a PDF.
"""

from __future__ import annotations
from printplan_ai.models import CoordinateFrame, Stage1Output, WallSegment


def get_demo_stage1(preset_name: str = "Residential Pavilion (10x8m)") -> Stage1Output:
    """Generate sample Stage1Output wall segments in world millimetres."""
    
    walls: list[WallSegment] = []
    
    if preset_name == "Tiny Prototype Cell (4x4m)":
        # Simple 4m x 4m single room box
        w_mm, h_mm = 4000.0, 4000.0
        # Perimeter
        walls.append(WallSegment(p1=(0.0, 0.0), p2=(w_mm, 0.0)))
        walls.append(WallSegment(p1=(w_mm, 0.0), p2=(w_mm, h_mm)))
        walls.append(WallSegment(p1=(w_mm, h_mm), p2=(0.0, h_mm)))
        walls.append(WallSegment(p1=(0.0, h_mm), p2=(0.0, 0.0)))
        
        # Interior partition wall
        walls.append(WallSegment(p1=(2000.0, 0.0), p2=(2000.0, 2500.0)))
        
    elif preset_name == "High-Speed Continuous Loop":
        # Smooth continuous multi-ring offset layout
        # Outer ring
        walls.append(WallSegment(p1=(0.0, 0.0), p2=(8000.0, 0.0)))
        walls.append(WallSegment(p1=(8000.0, 0.0), p2=(8000.0, 6000.0)))
        walls.append(WallSegment(p1=(8000.0, 6000.0), p2=(0.0, 6000.0)))
        walls.append(WallSegment(p1=(0.0, 6000.0), p2=(0.0, 0.0)))
        # Inner loop
        walls.append(WallSegment(p1=(1000.0, 1000.0), p2=(7000.0, 1000.0)))
        walls.append(WallSegment(p1=(7000.0, 1000.0), p2=(7000.0, 5000.0)))
        walls.append(WallSegment(p1=(7000.0, 5000.0), p2=(1000.0, 5000.0)))
        walls.append(WallSegment(p1=(1000.0, 5000.0), p2=(1000.0, 1000.0)))

    else:
        # Default: "Residential Pavilion (10x8m)"
        # 10m x 8m multi-room floor plan
        w_mm, h_mm = 10000.0, 8000.0
        
        # Outer perimeter envelope
        walls.append(WallSegment(p1=(0.0, 0.0), p2=(w_mm, 0.0)))
        walls.append(WallSegment(p1=(w_mm, 0.0), p2=(w_mm, h_mm)))
        walls.append(WallSegment(p1=(w_mm, h_mm), p2=(0.0, h_mm)))
        walls.append(WallSegment(p1=(0.0, h_mm), p2=(0.0, 0.0)))
        
        # Room 1: Living / Master dividing wall at X = 6000
        walls.append(WallSegment(p1=(6000.0, 0.0), p2=(6000.0, 8000.0)))
        
        # Room 2: Bedroom / Bath horizontal divider at Y = 4000 (X from 0 to 6000)
        walls.append(WallSegment(p1=(0.0, 4000.0), p2=(4000.0, 4000.0)))
        # Door opening gap between X=4000 and X=4800, then wall continues
        walls.append(WallSegment(p1=(4800.0, 4000.0), p2=(6000.0, 4000.0)))
        
        # Utility enclosure inside Master bedroom
        walls.append(WallSegment(p1=(6000.0, 5000.0), p2=(8500.0, 5000.0)))
        walls.append(WallSegment(p1=(8500.0, 5000.0), p2=(8500.0, 8000.0)))

    # Calculate summary metadata
    def length(s: WallSegment) -> float:
        return ((s.p1[0] - s.p2[0]) ** 2 + (s.p1[1] - s.p2[1]) ** 2) ** 0.5

    horiz = sum(1 for s in walls if abs(s.p1[1] - s.p2[1]) < 1)
    vert = sum(1 for s in walls if abs(s.p1[0] - s.p2[0]) < 1)
    oblique = len(walls) - horiz - vert
    total_length_m = sum(length(s) for s in walls) / 1000.0

    # Determine BBox
    xs = [s.p1[0] for s in walls] + [s.p2[0] for s in walls]
    ys = [s.p1[1] for s in walls] + [s.p2[1] for s in walls]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    frame = CoordinateFrame(
        unit_world="mm",
        y_axis="up",
        k_world_mm_per_pt=0.3527777777777778 * 50,
        page_bbox_world_mm=(min_x, min_y, max_x + 500, max_y + 500),
        drawing_scale="1:50",
    )

    return Stage1Output(
        coordinate_frame=frame,
        walls=walls,
        meta={
            "source_pdf": f"Builtin_Demo_{preset_name.replace(' ', '_')}.pdf",
            "source_layer_pattern": "2D Walls (Built-in Demo)",
            "segment_count": len(walls),
            "horizontal_count": horiz,
            "vertical_count": vert,
            "oblique_count": oblique,
            "total_length_m": round(total_length_m, 2),
            "orthogonal": oblique == 0,
            "is_demo": True,
        },
    )
