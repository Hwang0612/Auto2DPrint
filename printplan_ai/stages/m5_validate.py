"""Stage M5 — Digital-twin validation. Placeholder.

Consumes G-code from Stage M4 and runs kinematic replay, green-strength
stability check, and volumetric consistency check against a concrete
material model.
"""

from printplan_ai.config import M5Params


def validate(gcode: str, params: M5Params | None = None):
    """Not yet implemented."""
    raise NotImplementedError("M5 digital-twin validation is scheduled for v1.0.")
