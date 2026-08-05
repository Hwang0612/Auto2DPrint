"""Default parameter set for the pipeline."""

from dataclasses import dataclass, field

PT_TO_MM: float = 25.4 / 72


@dataclass
class M1Params:
    drawing_scale: float | None = None
    """Drawing scale factor. None = auto-detect from PDF text (e.g. 'Scale: 1/50').
    Set explicitly to override (e.g. 50.0 for 1:50)."""
    wall_layer_pattern: str = r"(?!.*(?:centr|dim|anno|hatch)).*wall"
    y_axis_up: bool = True


@dataclass
class M2Params:
    snap_tol_mm: float = 15.0
    """Endpoint snap tolerance for chaining wall segments into zones."""


@dataclass
class M3Params:
    layer_height_mm: float = 20.0
    seam_stagger_layers: int = 4
    """Layer period over which the trail start node rotates."""


@dataclass
class M4Params:
    bead_width_mm: float = 40.0
    """Bead width applied globally to all print traces."""
    pump_volume_per_unit_e: float = 1.0
    print_speed_mm_s: float = 60.0
    travel_speed_mm_s: float = 120.0
    z_lift_mm: float = 10.0


@dataclass
class M5Params:
    thixotropy_A_pa_per_s: float = 5.0
    density_kg_m3: float = 2200.0
    volumetric_tolerance_pct: float = 3.0


@dataclass
class PipelineConfig:
    m1: M1Params = field(default_factory=M1Params)
    m2: M2Params = field(default_factory=M2Params)
    m3: M3Params = field(default_factory=M3Params)
    m4: M4Params = field(default_factory=M4Params)
    m5: M5Params = field(default_factory=M5Params)
