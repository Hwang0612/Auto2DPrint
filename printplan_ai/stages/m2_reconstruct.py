"""Stage M2 — Zone detection.

Chains raw wall line segments from Stage 1 into connected zones by
snapping endpoints. Each zone is a group of wall segments whose
endpoints connect — the nozzle can traverse them without lifting.

Doors and windows break the connectivity, creating separate zones
or adding odd-degree nodes within a zone (forcing trail splits in
Stage 3).

Output per zone:
    - segment indices (into the Stage 1 walls list)
    - whether the zone is a perfect closed loop (all nodes even-degree)
    - number of odd-degree nodes (determines minimum trail count)
"""

from __future__ import annotations
import math
from collections import defaultdict

from printplan_ai.config import M2Params
from printplan_ai.models import Stage1Output, Stage2Output, Zone


def _snap(pt: tuple[float, float], tol: float) -> tuple[float, float]:
    return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)


def reconstruct(stage1: Stage1Output, params: M2Params | None = None) -> Stage2Output:
    """Detect connected zones in the wall-line graph."""
    p = params or M2Params()
    walls = [(tuple(s.p1), tuple(s.p2)) for s in stage1.walls]

    # Build adjacency on snapped endpoints
    adj: dict[tuple, list[tuple]] = defaultdict(list)
    for i, (p1, p2) in enumerate(walls):
        n1 = _snap(p1, p.snap_tol_mm)
        n2 = _snap(p2, p.snap_tol_mm)
        adj[n1].append((n2, i))
        adj[n2].append((n1, i))

    # Connected components via BFS
    visited: set[tuple] = set()
    zones: list[Zone] = []
    zone_id = 0
    for start in adj:
        if start in visited:
            continue
        comp_nodes: set[tuple] = set()
        queue = [start]
        while queue:
            n = queue.pop()
            if n in comp_nodes:
                continue
            comp_nodes.add(n)
            for nbr, _ in adj[n]:
                if nbr not in comp_nodes:
                    queue.append(nbr)
        visited |= comp_nodes

        seg_indices = sorted({si for n in comp_nodes for _, si in adj[n]})
        total_length = sum(math.dist(*walls[i]) for i in seg_indices)
        n_odd = sum(1 for n in comp_nodes if len(adj[n]) % 2 == 1)
        is_closed = n_odd == 0
        n_trails = 1 if is_closed else max(1, n_odd // 2)

        zones.append(Zone(
            zone_id=zone_id,
            segment_indices=seg_indices,
            n_segments=len(seg_indices),
            total_length_mm=total_length,
            is_closed_loop=is_closed,
            n_odd_nodes=n_odd,
            n_trails=n_trails,
        ))
        zone_id += 1

    # Sort: closed loops first, then by length descending
    zones.sort(key=lambda z: (not z.is_closed_loop, -z.total_length_mm))
    for i, z in enumerate(zones):
        z.zone_id = i

    return Stage2Output(
        coordinate_frame=stage1.coordinate_frame,
        walls=stage1.walls,
        zones=zones,
        meta={
            "n_zones": len(zones),
            "n_closed_loops": sum(1 for z in zones if z.is_closed_loop),
            "n_open_zones": sum(1 for z in zones if not z.is_closed_loop),
            "total_segments": sum(z.n_segments for z in zones),
            "total_length_m": round(sum(z.total_length_mm for z in zones) / 1000, 2),
            "total_trails": sum(z.n_trails for z in zones),
        },
    )
