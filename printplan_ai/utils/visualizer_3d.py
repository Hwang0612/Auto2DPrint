"""PrintPlan AI — WebGL Interactive 3D Visualizer & Material Estimator.

Generates 3D spatial interactive visualizations of 3D concrete printing toolpath stacks,
layer-by-layer slice views, 3D nozzle animation frames, and material/time estimations.
"""

from __future__ import annotations
import math
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from printplan_ai.models import Stage3Output, Layer, Trace


def build_3d_toolpath_fig(
    stage3: Stage3Output,
    color_mode: Literal["Layer", "Trail ID", "Print vs Travel"] = "Layer",
    render_mode: Literal["Extruded Concrete Beads", "Wireframe Toolpath"] = "Extruded Concrete Beads",
    show_travel: bool = True,
    layer_range: tuple[int, int] | None = None,
    bead_width_mm: float = 40.0,
) -> go.Figure:
    """Build an interactive WebGL 3D model of the continuous 3D concrete printed toolpath stack."""
    
    fig = go.Figure()
    
    layers = stage3.layers
    if not layers:
        return fig

    start_layer, end_layer = 0, len(layers) - 1
    if layer_range is not None:
        start_layer = max(0, min(layer_range[0], len(layers) - 1))
        end_layer = max(start_layer, min(layer_range[1], len(layers) - 1))

    selected_layers = layers[start_layer : end_layer + 1]
    
    # Palette definitions
    trail_colors = px.colors.qualitative.Bold
    legend_added = set()

    # Determine dynamic line width based on render mode
    line_width = 12 if render_mode == "Extruded Concrete Beads" else 4

    for layer in selected_layers:
        z = layer.z_mm
        
        for tr in layer.traces:
            pts = tr.points
            if len(pts) < 2:
                continue
            
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [z] * len(pts)
            
            if tr.kind == "print":
                if color_mode == "Layer":
                    norm_z = (layer.index) / max(1, len(layers) - 1)
                    c_idx = int(norm_z * 255)
                    color = px.colors.sequential.Viridis[c_idx % len(px.colors.sequential.Viridis)]
                    group_name = f"Layer {layer.index} (Z={z:.0f}mm)"
                elif color_mode == "Trail ID":
                    trail_id = tr.trail_id if tr.trail_id is not None else 0
                    color = trail_colors[trail_id % len(trail_colors)]
                    group_name = f"Trail T{trail_id}"
                else: # Print vs Travel
                    color = "#38bdf8" # Cyan
                    group_name = "Extruded Concrete Wall"
                
                show_in_legend = group_name not in legend_added
                if show_in_legend:
                    legend_added.add(group_name)

                # 3D Extruded Bead Trace
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        name=group_name,
                        line=dict(color=color, width=line_width),
                        hoverinfo="text",
                        hovertext=[
                            f"<b>Layer {layer.index}</b> (Z={z:.1f} mm)<br>"
                            f"Trail: T{tr.trail_id} | Kind: PRINT<br>"
                            f"Pos: ({x:.1f}, {y:.1f}, {z:.1f}) mm<br>"
                            f"Trace Length: {tr.length_mm/1000:.2f} m"
                            for x, y in zip(xs, ys)
                        ],
                        showlegend=show_in_legend,
                    )
                )

                # Start node marker (Nozzle entry point)
                fig.add_trace(
                    go.Scatter3d(
                        x=[xs[0]],
                        y=[ys[0]],
                        z=[zs[0]],
                        mode="markers",
                        marker=dict(size=4, color="#f59e0b", symbol="circle"),
                        hoverinfo="text",
                        hovertext=f"Start Trail T{tr.trail_id} (L{layer.index})",
                        showlegend=False,
                    )
                )
                
            elif tr.kind == "travel" and show_travel:
                group_name = "Nozzle Travel Path"
                show_in_legend = group_name not in legend_added
                if show_in_legend:
                    legend_added.add(group_name)

                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        name=group_name,
                        line=dict(color="#94a3b8", width=2, dash="dash"),
                        hoverinfo="text",
                        hovertext=[
                            f"<b>Nozzle Travel Move</b> (Layer {layer.index})<br>"
                            f"Pos: ({x:.1f}, {y:.1f}, {z:.1f}) mm"
                            for x, y in zip(xs, ys)
                        ],
                        showlegend=show_in_legend,
                    )
                )

    # Compute bounding box & manual aspect ratio to prevent Z compression distortion
    bbox = stage3.coordinate_frame.page_bbox_world_mm
    x0, y0, x1, y1 = bbox
    range_x = max(100.0, x1 - x0)
    range_y = max(100.0, y1 - y0)
    max_z = max(100.0, layers[-1].z_mm if layers else 100.0)

    # Proportional aspect ratio
    aspect_x = 1.0
    aspect_y = range_y / range_x
    aspect_z = max(0.3, min(0.8, (max_z / range_x) * 3.0))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.95)",
        plot_bgcolor="rgba(15, 23, 42, 0.95)",
        margin=dict(l=10, r=10, b=10, t=30),
        scene=dict(
            xaxis=dict(title="X (mm)", gridcolor="rgba(148, 163, 184, 0.2)", backgroundcolor="rgba(11, 15, 25, 0.95)", tickfont=dict(color="#94a3b8")),
            yaxis=dict(title="Y (mm)", gridcolor="rgba(148, 163, 184, 0.2)", backgroundcolor="rgba(11, 15, 25, 0.95)", tickfont=dict(color="#94a3b8")),
            zaxis=dict(title="Z (mm)", gridcolor="rgba(148, 163, 184, 0.2)", backgroundcolor="rgba(11, 15, 25, 0.95)", tickfont=dict(color="#94a3b8")),
            aspectmode="manual",
            aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z),
            camera=dict(
                eye=dict(x=1.4, y=-1.4, z=1.0)
            )
        ),
        legend=dict(
            font=dict(color="#f8fafc", size=11),
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(15, 23, 42, 0.9)",
            bordercolor="rgba(56, 189, 248, 0.4)",
            borderwidth=1,
        ),
        height=620,
    )
    
    return fig


def build_layer_2d_fig(layer: Layer, layer_idx: int, bbox: tuple[float, float, float, float]) -> go.Figure:
    """Build high-resolution 2D interactive floor plan slice view of a single layer."""
    fig = go.Figure()
    
    colors = ["#38bdf8", "#f43f5e", "#10b981", "#f59e0b", "#a855f7", "#ec4899"]
    
    for tr in layer.traces:
        pts = tr.points
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        if tr.kind == "print":
            c = colors[(tr.trail_id or 0) % len(colors)]
            label = f"Trail T{tr.trail_id} ({'Closed' if tr.is_closed else 'Open'})"
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=c, width=4),
                    marker=dict(size=4),
                    hoverinfo="text",
                    hovertext=[f"X: {x:.1f} mm, Y: {y:.1f} mm" for x, y in zip(xs, ys)]
                )
            )
            # Annotate start point
            fig.add_trace(
                go.Scatter(
                    x=[xs[0]], y=[ys[0]],
                    mode="markers",
                    marker=dict(size=10, color=c, symbol="star"),
                    name=f"Start T{tr.trail_id}",
                    showlegend=False
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    name="Travel Move",
                    line=dict(color="#94a3b8", width=1.5, dash="dot"),
                    showlegend=False
                )
            )
            
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.95)",
        plot_bgcolor="rgba(11, 15, 25, 0.95)",
        title=dict(
            text=f"Layer {layer_idx} (Z = {layer.z_mm:.1f} mm) — {layer.n_print_traces} Nozzle Trails",
            font=dict(color="#f8fafc", size=14)
        ),
        xaxis=dict(
            title=dict(text="X (mm)", font=dict(color="#f8fafc")),
            scaleanchor="y", scaleratio=1,
            gridcolor="rgba(148, 163, 184, 0.2)",
            tickfont=dict(color="#f8fafc", size=11),
        ),
        yaxis=dict(
            title=dict(text="Y (mm)", font=dict(color="#f8fafc")),
            gridcolor="rgba(148, 163, 184, 0.2)",
            tickfont=dict(color="#f8fafc", size=11),
        ),
        legend=dict(
            font=dict(color="#f8fafc", size=12),
            bgcolor="rgba(15, 23, 42, 0.9)",
            bordercolor="rgba(56, 189, 248, 0.4)",
            borderwidth=1
        ),
        margin=dict(l=50, r=40, b=50, t=60),
        height=580,
    )
    return fig


def build_3d_nozzle_anim(stage3: Stage3Output, layer_idx: int = 0, bbox: tuple[float, float, float, float] = (0, 0, 10000, 10000)) -> go.Figure:
    """Build step-by-step 3D animated playback of the nozzle trajectory for a single layer."""
    if not stage3.layers or layer_idx >= len(stage3.layers):
        return go.Figure()
        
    layer = stage3.layers[layer_idx]
    z = layer.z_mm
    
    # Flatten all points in sequential order
    path_x, path_y, path_z, hover_labels = [], [], [], []
    for tr in layer.traces:
        for p in tr.points:
            path_x.append(p[0])
            path_y.append(p[1])
            path_z.append(z)
            hover_labels.append(f"{tr.kind.upper()} | X:{p[0]:.1f}, Y:{p[1]:.1f}, Z:{z:.1f}")

    if not path_x:
        return go.Figure()

    # Dynamic aspect ratio calculation
    x0, y0, x1, y1 = bbox
    range_x = max(100.0, x1 - x0)
    range_y = max(100.0, y1 - y0)
    aspect_y = range_y / range_x

    # Create base frame
    fig = go.Figure(
        data=[
            # Extruded path so far
            go.Scatter3d(
                x=[path_x[0]], y=[path_y[0]], z=[path_z[0]],
                mode="lines", line=dict(color="#38bdf8", width=10), name="Printed Concrete Bead"
            ),
            # Current Nozzle location
            go.Scatter3d(
                x=[path_x[0]], y=[path_y[0]], z=[path_z[0] + 50],
                mode="markers", marker=dict(color="#f59e0b", size=12, symbol="diamond"), name="3D Printer Nozzle Head"
            )
        ],
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.95)",
            plot_bgcolor="rgba(11, 15, 25, 0.95)",
            scene=dict(
                xaxis=dict(title="X (mm)", gridcolor="rgba(148, 163, 184, 0.2)", tickfont=dict(color="#94a3b8")),
                yaxis=dict(title="Y (mm)", gridcolor="rgba(148, 163, 184, 0.2)", tickfont=dict(color="#94a3b8")),
                zaxis=dict(title="Z (mm)", range=[z - 20, z + 200], gridcolor="rgba(148, 163, 184, 0.2)", tickfont=dict(color="#94a3b8")),
                aspectmode="manual",
                aspectratio=dict(x=1.0, y=aspect_y, z=0.4),
                camera=dict(
                    eye=dict(x=1.3, y=-1.3, z=0.8)
                )
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="▶ Play Trajectory", method="animate", args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ],
                x=0.02, y=1.05,
                bgcolor="rgba(15, 23, 42, 0.9)",
                font=dict(color="#38bdf8", size=12)
            )],
            legend=dict(
                font=dict(color="#f8fafc", size=11),
                bgcolor="rgba(15, 23, 42, 0.9)",
                bordercolor="rgba(56, 189, 248, 0.4)",
                borderwidth=1,
                x=0.75, y=0.95
            ),
            margin=dict(l=10, r=10, b=10, t=40),
            height=580
        )
    )

    # Subsample steps for smooth animation frames
    step_size = max(1, len(path_x) // 40)
    indices = list(range(0, len(path_x), step_size))
    if indices[-1] != len(path_x) - 1:
        indices.append(len(path_x) - 1)

    frames = []
    for idx in indices:
        frame_data = [
            go.Scatter3d(
                x=path_x[: idx + 1],
                y=path_y[: idx + 1],
                z=path_z[: idx + 1],
                mode="lines", line=dict(color="#38bdf8", width=10)
            ),
            go.Scatter3d(
                x=[path_x[idx]],
                y=[path_y[idx]],
                z=[path_z[idx] + 50],
                mode="markers", marker=dict(color="#f59e0b", size=12, symbol="diamond")
            )
        ]
        frames.append(go.Frame(data=frame_data, name=f"frame_{idx}"))

    fig.frames = frames
    return fig


def calc_material_metrics(
    stage3: Stage3Output,
    bead_width_mm: float,
    layer_height_mm: float,
    print_speed_mm_s: float,
    density_kg_m3: float = 2300.0,
) -> dict:
    """Calculate concrete volume, mass, print time, and efficiency metrics."""
    meta = stage3.meta
    total_print_m = meta.get("total_print_length_m", 0.0)
    total_travel_m = meta.get("total_travel_length_m", 0.0)
    
    # Volume in cubic meters: L(m) * W(m) * H(m)
    bead_w_m = bead_width_mm / 1000.0
    layer_h_m = layer_height_mm / 1000.0
    
    total_volume_m3 = total_print_m * bead_w_m * layer_h_m
    total_mass_kg = total_volume_m3 * density_kg_m3
    total_mass_tonnes = total_mass_kg / 1000.0
    
    # Time estimation
    print_time_sec = (total_print_m * 1000.0) / max(1.0, print_speed_mm_s)
    # Assume nozzle travel speed is 1.5x print speed
    travel_speed_mm_s = print_speed_mm_s * 1.5
    travel_time_sec = (total_travel_m * 1000.0) / travel_speed_mm_s
    total_time_sec = print_time_sec + travel_time_sec
    
    minutes = int(total_time_sec // 60)
    seconds = int(total_time_sec % 60)
    time_str = f"{minutes}m {seconds}s" if minutes < 60 else f"{minutes // 60}h {minutes % 60}m"
    
    # Efficiency ratio
    total_m = total_print_m + total_travel_m
    efficiency_pct = (total_print_m / total_m * 100.0) if total_m > 0 else 100.0

    return {
        "total_print_m": total_print_m,
        "total_travel_m": total_travel_m,
        "volume_m3": round(total_volume_m3, 3),
        "mass_tonnes": round(total_mass_tonnes, 3),
        "mass_kg": round(total_mass_kg, 1),
        "print_time_sec": round(print_time_sec, 1),
        "time_str": time_str,
        "efficiency_pct": round(efficiency_pct, 1),
    }
