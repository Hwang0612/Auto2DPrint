"""Stage M4 — G-code synthesis.

Converts the layered toolpath from Stage M3 into machine-executable
G-code. Each print trace becomes a sequence of G1 moves with volumetric
extrusion; each travel trace becomes a G0 repositioning with Z-lift.

Supported machine dialects (v1.0):
    - gantry_marlin   (Marlin / Klipper firmware on gantry 3DCP systems)

The bead width is applied here — not in Stage 3 — because it is a
machine/material parameter, not a geometric one.

G-code structure per layer
--------------------------
    G92 E0                         ; reset extruder at layer start
    ; --- print trace ---
    G1 X... Y... Z... F... E...    ; extruding moves
    ; --- travel ---
    M400                           ; wait for moves to finish
    G0 Z+lift                      ; lift nozzle
    G0 X... Y... F_travel          ; reposition
    G0 Z_layer                     ; lower nozzle
    G92 E0                         ; reset E before next trace
    ; --- next print trace ---
    ...
"""

from __future__ import annotations
import math
from io import StringIO

from printplan_ai.config import M4Params
from printplan_ai.models import Stage3Output


def _header(p: M4Params) -> str:
    return (
        "; PrintPlan AI — G-code output\n"
        f"; Bead width:    {p.bead_width_mm:.1f} mm\n"
        f"; Print speed:   {p.print_speed_mm_s:.0f} mm/s\n"
        f"; Travel speed:  {p.travel_speed_mm_s:.0f} mm/s\n"
        f"; Z-lift:        {p.z_lift_mm:.1f} mm\n"
        f"; Pump k:        {p.pump_volume_per_unit_e:.4f} mm3/unit-E\n"
        ";\n"
        "; Machine dialect: gantry_marlin\n"
        ";\n"
        "G21            ; mm mode\n"
        "G90            ; absolute positioning\n"
        "M82            ; absolute extrusion\n"
        "G28            ; home all axes\n"
        "G92 E0         ; zero extruder\n"
        "\n"
    )


def _footer() -> str:
    return (
        "\n"
        "; --- end of print ---\n"
        "M400           ; wait for moves to finish\n"
        "G0 Z50         ; raise nozzle\n"
        "M84            ; disable motors\n"
    )


def synthesise_gcode(
    stage3: Stage3Output,
    params: M4Params | None = None,
) -> str:
    """Convert a Stage-3 toolpath into G-code.

    Parameters
    ----------
    stage3 : Stage3Output
        Layered toolpath from Stage M3.
    params : M4Params, optional
        Machine and material parameters.

    Returns
    -------
    str
        Complete G-code program as a string.
    """
    p = params or M4Params()
    buf = StringIO()
    buf.write(_header(p))

    f_print = p.print_speed_mm_s * 60       # mm/min
    f_travel = p.travel_speed_mm_s * 60     # mm/min
    k_pump = p.pump_volume_per_unit_e       # mm3 per unit E

    total_e = 0.0
    total_print_len = 0.0
    total_travel_len = 0.0
    total_volume = 0.0
    n_lines = 0

    for layer in stage3.layers:
        z = layer.z_mm
        buf.write(f"\n; ===== Layer {layer.index}  z={z:.2f} mm =====\n")
        buf.write(f"G92 E0         ; reset E for layer {layer.index}\n")
        layer_e = 0.0

        for trace in layer.traces:
            if trace.kind == "print":
                buf.write(f"\n; --- print trace {trace.trail_id} "
                          f"(zone {trace.zone_id}, "
                          f"{'closed' if trace.is_closed else 'open'}, "
                          f"{trace.length_mm/1000:.1f} m) ---\n")

                pts = trace.points
                # Move to start without extruding (first trace of layer
                # or after a travel — nozzle is already positioned, but
                # emit the first point as a non-extruding move to be safe)
                buf.write(f"G0 X{pts[0][0]:.3f} Y{pts[0][1]:.3f} Z{z:.2f} "
                          f"F{f_travel:.0f}\n")
                n_lines += 1

                for i in range(1, len(pts)):
                    dx = pts[i][0] - pts[i-1][0]
                    dy = pts[i][1] - pts[i-1][1]
                    seg_len = math.hypot(dx, dy)
                    if seg_len < 0.01:
                        continue

                    # Volumetric extrusion: V = L × w × h
                    vol = seg_len * p.bead_width_mm * layer.z_mm / layer.index if layer.index > 0 else seg_len * p.bead_width_mm * (stage3.layers[0].z_mm if stage3.layers else 20)
                    # Simpler: just use layer_height from z spacing
                    layer_h = stage3.layers[0].z_mm if len(stage3.layers) < 2 else stage3.layers[1].z_mm - stage3.layers[0].z_mm
                    vol = seg_len * p.bead_width_mm * layer_h
                    layer_e += vol / k_pump
                    total_volume += vol

                    buf.write(f"G1 X{pts[i][0]:.3f} Y{pts[i][1]:.3f} "
                              f"F{f_print:.0f} E{layer_e:.4f}\n")
                    n_lines += 1
                    total_print_len += seg_len

                # If closed loop, connect back to start
                if trace.is_closed and len(pts) > 2:
                    dx = pts[0][0] - pts[-1][0]
                    dy = pts[0][1] - pts[-1][1]
                    seg_len = math.hypot(dx, dy)
                    if seg_len > 0.01:
                        layer_h = stage3.layers[0].z_mm if len(stage3.layers) < 2 else stage3.layers[1].z_mm - stage3.layers[0].z_mm
                        vol = seg_len * p.bead_width_mm * layer_h
                        layer_e += vol / k_pump
                        total_volume += vol
                        buf.write(f"G1 X{pts[0][0]:.3f} Y{pts[0][1]:.3f} "
                                  f"F{f_print:.0f} E{layer_e:.4f}\n")
                        n_lines += 1
                        total_print_len += seg_len

            elif trace.kind == "travel":
                buf.write(f"\n; --- travel ({trace.length_mm:.0f} mm) ---\n")
                buf.write("M400           ; finish pending moves\n")
                buf.write(f"G0 Z{z + p.z_lift_mm:.2f}  ; lift\n")
                buf.write(f"G0 X{trace.points[1][0]:.3f} "
                          f"Y{trace.points[1][1]:.3f} "
                          f"F{f_travel:.0f}\n")
                buf.write(f"G0 Z{z:.2f}     ; lower\n")
                buf.write(f"G92 E0         ; reset E after travel\n")
                layer_e = 0.0
                n_lines += 5
                total_travel_len += trace.length_mm

        total_e += layer_e

    buf.write(_footer())

    # Append summary as comments
    layer_h = stage3.layers[0].z_mm if len(stage3.layers) < 2 else stage3.layers[1].z_mm - stage3.layers[0].z_mm
    buf.write(f"\n; --- summary ---\n")
    buf.write(f"; Layers:         {len(stage3.layers)}\n")
    buf.write(f"; Layer height:   {layer_h:.1f} mm\n")
    buf.write(f"; Print length:   {total_print_len/1000:.2f} m\n")
    buf.write(f"; Travel length:  {total_travel_len/1000:.2f} m\n")
    buf.write(f"; Concrete vol:   {total_volume/1e9:.4f} m3\n")
    buf.write(f"; G-code lines:   {n_lines}\n")

    return buf.getvalue()
