"""Shared geometric helpers."""

from __future__ import annotations
from collections import defaultdict


def group_collinear_1d(
    segments: list[tuple[float, float, float]],
    snap_tol: float,
) -> list[dict]:
    """Group axis-aligned segments sharing the fixed coordinate.

    Each input is (fixed_coord, run_start, run_end); output is a list of
    ``{"coord": float, "runs": [(a, b), ...]}`` where overlapping runs are
    merged.
    """
    bins: dict[float, list[tuple[float, float, float]]] = defaultdict(list)
    for coord, a, b in segments:
        key = round(coord / snap_tol) * snap_tol
        bins[key].append((coord, a, b))

    lines: list[dict] = []
    for _, items in bins.items():
        coord = sum(item[0] for item in items) / len(items)
        raw_runs = sorted((a, b) for _, a, b in items)
        merged: list[tuple[float, float]] = []
        for a, b in raw_runs:
            if merged and a <= merged[-1][1] + snap_tol:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        lines.append({"coord": coord, "runs": merged})
    return lines


def union_1d(runs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the union of a set of 1D intervals (list of (start, end))."""
    if not runs:
        return []
    ordered = sorted(runs)
    out: list[tuple[float, float]] = []
    for a, b in ordered:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out
