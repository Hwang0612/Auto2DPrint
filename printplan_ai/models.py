"""Pydantic data models — the JSON contracts between stages and API responses."""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


# ── Coordinate frame ──────────────────────────────────────────────
class CoordinateFrame(BaseModel):
    unit_world: Literal["mm"] = "mm"
    y_axis: Literal["up", "down"] = "up"
    k_world_mm_per_pt: float
    page_bbox_world_mm: tuple[float, float, float, float]
    drawing_scale: str = "1:50"


Point = tuple[float, float]


class LineSegment(BaseModel):
    p1: Point
    p2: Point


# ── Stage 1 output ────────────────────────────────────────────────
class WallSegment(LineSegment):
    """A raw wall line segment extracted from the PDF."""


class Stage1Output(BaseModel):
    coordinate_frame: CoordinateFrame
    walls: list[WallSegment]
    meta: dict = Field(default_factory=dict)


# ── Stage 2 output ────────────────────────────────────────────────
class Zone(BaseModel):
    """A connected group of wall segments that can be printed together."""
    zone_id: int
    segment_indices: list[int]
    n_segments: int
    total_length_mm: float
    is_closed_loop: bool
    n_odd_nodes: int
    n_trails: int = Field(description="1 if closed loop, else n_odd_nodes // 2")


class Stage2Output(BaseModel):
    coordinate_frame: CoordinateFrame
    walls: list[WallSegment]
    zones: list[Zone]
    meta: dict = Field(default_factory=dict)


# ── Stage 3 output ────────────────────────────────────────────────
TraceKind = Literal["print", "travel"]


class Trace(BaseModel):
    kind: TraceKind
    zone_id: int | None = None
    trail_id: int | None = None
    points: list[Point]
    length_mm: float
    is_closed: bool = False


class Layer(BaseModel):
    index: int
    z_mm: float
    traces: list[Trace]
    n_print_traces: int
    total_print_length_mm: float
    total_travel_length_mm: float


class Stage3Output(BaseModel):
    coordinate_frame: CoordinateFrame
    layers: list[Layer]
    meta: dict = Field(default_factory=dict)
