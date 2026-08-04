"""Stage M3 — Toolpath generation (continuous-path 3DCP).

Operates directly on the raw wall-line graph from Stage 2. Each wall
line segment IS a nozzle pass. The algorithm finds connected zones,
then decomposes each zone into the minimum number of continuous trails:

    - Closed-loop zones (0 odd nodes) → 1 Eulerian circuit
    - Open zones (2 odd nodes)        → 1 Eulerian path
    - Broken zones (>2 odd nodes)     → N/2 edge-disjoint trails
      where N = number of odd-degree nodes

Every edge is visited exactly once — zero over-print.
"""

from __future__ import annotations
import math
from collections import defaultdict

import networkx as nx

from printplan_ai.config import M2Params, M3Params
from printplan_ai.models import Layer, Stage2Output, Stage3Output, Trace, Zone


# ── Graph building ────────────────────────────────────────────────
def _snap(pt: tuple[float, float], tol: float) -> tuple[float, float]:
    return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)


def _build_zone_graph(
    walls: list, zone: Zone, snap_tol: float,
) -> nx.MultiGraph:
    G = nx.MultiGraph()
    for seg_i in zone.segment_indices:
        p1, p2 = tuple(walls[seg_i].p1), tuple(walls[seg_i].p2)
        n1, n2 = _snap(p1, snap_tol), _snap(p2, snap_tol)
        G.add_edge(n1, n2, key=seg_i, p1=p1, p2=p2,
                   length=math.dist(p1, p2))
    return G


# ── Trail decomposition ──────────────────────────────────────────
def _extract_one_trail(G: nx.MultiGraph, start) -> list[tuple]:
    """Extract one maximal trail from ``start``, removing edges from G
    as they are consumed. Uses Fleury's algorithm: at each step, avoid
    taking a bridge if an alternative exists."""
    trail: list[tuple] = []
    current = start

    while G.degree(current) > 0:
        edges = list(G.edges(current, keys=True, data=True))
        if not edges:
            break

        chosen_idx = 0
        if len(edges) > 1:
            # Check which edges are bridges
            non_bridge_indices = []
            for idx, (u, v, k, data) in enumerate(edges):
                G.remove_edge(u, v, key=k)
                neighbor = v if u == current else u
                is_bridge = (G.degree(neighbor) == 0 or
                             not nx.has_path(G, current, neighbor))
                G.add_edge(u, v, key=k, **data)
                if not is_bridge:
                    non_bridge_indices.append(idx)

            if non_bridge_indices:
                # Among non-bridges, prefer highest-degree neighbor
                chosen_idx = max(non_bridge_indices,
                                 key=lambda i: G.degree(
                                     edges[i][1] if edges[i][0] == current
                                     else edges[i][0]))
            else:
                # All bridges — pick any (prefer highest-degree neighbor)
                chosen_idx = max(range(len(edges)),
                                 key=lambda i: G.degree(
                                     edges[i][1] if edges[i][0] == current
                                     else edges[i][0]))

        u, v, k, data = edges[chosen_idx]
        trail.append((u, v, k))
        G.remove_edge(u, v, key=k)
        current = v if u == current else u

    return trail


def _decompose_zone(
    G: nx.MultiGraph, zone: Zone, layer_index: int, params: M3Params,
) -> list[list[tuple]]:
    """Decompose a zone into edge-disjoint trails."""
    odd = sorted([n for n in G.nodes if G.degree(n) % 2 == 1])

    if len(odd) == 0:
        # Eulerian circuit — rotate start per layer
        nodes = sorted(G.nodes)
        src = nodes[(layer_index % params.seam_stagger_layers) % len(nodes)]
        return [list(nx.eulerian_circuit(G, source=src, keys=True))]

    if len(odd) == 2:
        src = odd[layer_index % 2]
        return [list(nx.eulerian_path(G, source=src, keys=True))]

    # General: extract trails starting from odd-degree nodes
    start_order = odd[layer_index % len(odd):] + odd[:layer_index % len(odd)]
    remaining = G.copy()
    trails: list[list[tuple]] = []

    for start in start_order:
        if remaining.degree(start) == 0:
            continue
        trail = _extract_one_trail(remaining, start)
        if trail:
            trails.append(trail)

    # Safety: consume any leftover edges (even-degree components)
    while remaining.number_of_edges() > 0:
        for n in remaining.nodes:
            if remaining.degree(n) > 0:
                trail = _extract_one_trail(remaining, n)
                if trail:
                    trails.append(trail)
                break
    return trails


# ── Trail to polyline ─────────────────────────────────────────────
def _trail_to_polyline(
    edges: list[tuple], G: nx.MultiGraph,
) -> tuple[list[tuple[float, float]], float]:
    if not edges:
        return [], 0.0
    pts: list[tuple[float, float]] = []
    length = 0.0

    for i, (u, v, k) in enumerate(edges):
        e = G[u][v][k]
        p1, p2, L = e["p1"], e["p2"], e["length"]
        if i == 0:
            if math.dist(p1, u) <= math.dist(p2, u):
                pts.extend([p1, p2])
            else:
                pts.extend([p2, p1])
        else:
            last = pts[-1]
            if math.dist(last, p1) <= math.dist(last, p2):
                if math.dist(last, p1) > 1:
                    pts.append(p1)
                pts.append(p2)
            else:
                if math.dist(last, p2) > 1:
                    pts.append(p2)
                pts.append(p1)
        length += L
    return pts, length


# ── Block ordering with start-point optimisation ──────────────────
def _rotate_closed_trace(trace: Trace, target: tuple[float, float]) -> Trace:
    """Rotate a closed-loop trace so it starts (and ends) at the point
    nearest to ``target``. This minimises the travel FROM a previous
    block's endpoint TO this circuit's start."""
    pts = trace.points
    if len(pts) < 3:
        return trace
    # Find the vertex nearest to target
    best_i = min(range(len(pts)), key=lambda i: math.dist(pts[i], target))
    rotated = pts[best_i:] + pts[1:best_i + 1]   # skip duplicate close point
    return Trace(
        kind=trace.kind, zone_id=trace.zone_id, trail_id=trace.trail_id,
        points=rotated, length_mm=trace.length_mm, is_closed=trace.is_closed,
    )


def _reverse_trace(trace: Trace) -> Trace:
    """Reverse an open trace so it runs end-to-start. For an Euler path
    with 2 odd-degree endpoints this effectively picks the other endpoint
    as the starting node."""
    return Trace(
        kind=trace.kind, zone_id=trace.zone_id, trail_id=trace.trail_id,
        points=list(reversed(trace.points)),
        length_mm=trace.length_mm, is_closed=trace.is_closed,
    )


def _optimise_trace_order(traces: list[Trace]) -> list[Trace]:
    """Order traces and adjust their start/end points to minimise total
    travel distance between consecutive blocks.

    For each candidate next-trace:
      - If closed (circuit): rotate start to nearest point to previous end
      - If open (path): try both directions, pick the shorter travel
    Select the trace+orientation that gives the shortest travel at each step.
    """
    if len(traces) <= 1:
        return traces

    # Start with the longest trace (heuristic: most constrained)
    remaining = list(traces)
    first_idx = max(range(len(remaining)), key=lambda i: remaining[i].length_mm)
    ordered = [remaining.pop(first_idx)]

    while remaining:
        prev_end = ordered[-1].points[-1]
        best_dist = float("inf")
        best_idx = 0
        best_trace = remaining[0]

        for i, tr in enumerate(remaining):
            if tr.is_closed:
                # Rotate to nearest point — travel = distance to nearest vertex
                d = min(math.dist(prev_end, p) for p in tr.points)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
                    best_trace = _rotate_closed_trace(tr, prev_end)
            else:
                # Try both orientations
                d_fwd = math.dist(prev_end, tr.points[0])
                d_rev = math.dist(prev_end, tr.points[-1])
                if d_fwd <= d_rev:
                    if d_fwd < best_dist:
                        best_dist = d_fwd
                        best_idx = i
                        best_trace = tr
                else:
                    if d_rev < best_dist:
                        best_dist = d_rev
                        best_idx = i
                        best_trace = _reverse_trace(tr)

        remaining.pop(best_idx)
        ordered.append(best_trace)

    return ordered


def _assemble_layer(
    index: int, z: float,
    zones: list[Zone], graphs: list[nx.MultiGraph],
    params: M3Params,
) -> Layer:
    print_traces: list[Trace] = []
    trail_id = 0
    for zone, G in zip(zones, graphs):
        trails = _decompose_zone(G, zone, index, params)
        for trail_edges in trails:
            pts, length = _trail_to_polyline(trail_edges, G)
            if len(pts) < 2:
                continue
            is_closed = (zone.is_closed_loop and len(trails) == 1)
            print_traces.append(Trace(
                kind="print", zone_id=zone.zone_id, trail_id=trail_id,
                points=pts, length_mm=length, is_closed=is_closed,
            ))
            trail_id += 1

    # Travel optimisation: order blocks and rotate closed circuits so
    # each block starts at the point nearest to the previous block's end.
    # This is the ONLY place start points are determined — no separate
    # seam stagger step that could undo the travel minimisation.
    print_traces = _optimise_trace_order(print_traces)

    # Interleave travels
    all_traces: list[Trace] = []
    for i, tr in enumerate(print_traces):
        if i > 0:
            prev_end = all_traces[-1].points[-1]
            all_traces.append(Trace(
                kind="travel", zone_id=None, trail_id=None,
                points=[prev_end, tr.points[0]],
                length_mm=math.dist(prev_end, tr.points[0]),
            ))
        all_traces.append(tr)

    total_print = sum(t.length_mm for t in all_traces if t.kind == "print")
    total_travel = sum(t.length_mm for t in all_traces if t.kind == "travel")
    n_prints = sum(1 for t in all_traces if t.kind == "print")

    return Layer(
        index=index, z_mm=z, traces=all_traces,
        n_print_traces=n_prints,
        total_print_length_mm=total_print,
        total_travel_length_mm=total_travel,
    )


# ── Public API ────────────────────────────────────────────────────
def generate_toolpath(
    stage2: Stage2Output,
    params: M3Params | None = None,
    snap_tol: float = 15.0,
    n_layers: int = 10,
) -> Stage3Output:
    """Generate continuous-path toolpath from zone-detected wall geometry."""
    p = params or M3Params()

    graphs = [_build_zone_graph(stage2.walls, z, snap_tol) for z in stage2.zones]

    layers = []
    for n in range(n_layers):
        z = (n + 1) * p.layer_height_mm
        layers.append(_assemble_layer(n, z, stage2.zones, graphs, p))

    total_print = sum(l.total_print_length_mm for l in layers) / 1000
    total_travel = sum(l.total_travel_length_mm for l in layers) / 1000

    return Stage3Output(
        coordinate_frame=stage2.coordinate_frame,
        layers=layers,
        meta={
            "n_layers": n_layers,
            "n_zones": len(stage2.zones),
            "trails_per_layer": sum(z.n_trails for z in stage2.zones),
            "total_print_length_m": round(total_print, 2),
            "total_travel_length_m": round(total_travel, 2),
            "layer_height_mm": p.layer_height_mm,
        },
    )
