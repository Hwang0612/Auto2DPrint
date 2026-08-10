"""PrintPlan AI — WebGL Interactive 3D Visualizer & Material Estimator.

Generates 3D spatial interactive visualizations of 3D concrete printing toolpath stacks,
layer-by-layer slice views, 3D nozzle animation frames, and material/time estimations.

v2.0 — Replaced Plotly Scatter3d with Three.js TubeGeometry via Streamlit HTML component
for realistic extruded bead rendering with PBR lighting and free OrbitControls.
"""

from __future__ import annotations
import json
import math
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

from printplan_ai.models import Stage3Output, Layer, Trace


# ─────────────────────────────────────────────────────────────────────────────
# THREE.JS WEBGL VIEWER  (replaces build_3d_toolpath_fig Plotly output)
# ─────────────────────────────────────────────────────────────────────────────

def build_3d_toolpath_webgl(
    stage3: Stage3Output,
    color_mode: Literal["Layer", "Trail ID", "Print vs Travel"] = "Layer",
    render_mode: Literal["Extruded Concrete Beads", "Wireframe Toolpath"] = "Extruded Concrete Beads",
    show_travel: bool = True,
    layer_range: tuple[int, int] | None = None,
    bead_width_mm: float = 40.0,
    layer_height_mm: float = 20.0,
    height_px: int = 680,
) -> None:
    """
    Render an interactive Three.js WebGL 3D model of the continuous 3DCP toolpath stack
    via a Streamlit HTML component.

    Replaces the Plotly Scatter3d approach with:
    - TubeGeometry for volumetric bead cross-sections
    - MeshStandardMaterial with PBR roughness/metalness
    - Directional + ambient + rim lighting with shadows
    - OrbitControls for free camera rotation/zoom/pan
    - Elliptical bead cross-section matching real 3DCP geometry

    Args:
        stage3:          Stage3Output from the pipeline.
        color_mode:      How to colour beads — by Layer, Trail ID, or Print vs Travel.
        render_mode:     Extruded Concrete Beads (volumetric) or Wireframe Toolpath (lines).
        show_travel:     Whether to draw nozzle travel (non-print) moves.
        layer_range:     (start, end) layer index range, inclusive.
        bead_width_mm:   Bead cross-section width in mm (horizontal).
        layer_height_mm: Bead cross-section height in mm (vertical = layer height).
        height_px:       Pixel height of the rendered component.
    """
    layers = stage3.layers
    if not layers:
        components.html("<p style='color:#f87171'>No layers to render.</p>", height=80)
        return

    # ── Layer range ──────────────────────────────────────────────────────────
    start_layer = 0
    end_layer = len(layers) - 1
    if layer_range is not None:
        start_layer = max(0, min(layer_range[0], end_layer))
        end_layer   = max(start_layer, min(layer_range[1], end_layer))

    selected_layers = layers[start_layer : end_layer + 1]

    # ── Colour palette ───────────────────────────────────────────────────────
    # Viridis-like hex palette for layer colouring (20 stops)
    VIRIDIS_HEX = [
        "#440154","#482878","#3e4989","#31688e","#26828e",
        "#1f9e89","#35b779","#6ece58","#b5de2b","#fde725",
        "#f0f921","#f89540","#e16462","#b12a90","#6a00a8",
        "#0d0887","#41049d","#6a00a8","#b12a90","#e16462",
    ]
    TRAIL_PALETTE = [
        "#f43f5e","#38bdf8","#a3e635","#f59e0b","#a855f7",
        "#ec4899","#14b8a6","#fb923c","#818cf8","#34d399",
    ]
    n_layers = len(layers)

    # ── Serialise toolpath to JSON for JS ────────────────────────────────────
    js_segments: list[dict] = []

    for layer in selected_layers:
        z = layer.z_mm
        layer_norm = layer.index / max(1, n_layers - 1)
        vi = int(layer_norm * (len(VIRIDIS_HEX) - 1))
        layer_color = VIRIDIS_HEX[vi]

        for tr in layer.traces:
            pts = tr.points
            if len(pts) < 2:
                continue

            is_print = tr.kind == "print"
            if not is_print and not show_travel:
                continue

            # Colour logic
            if is_print:
                if color_mode == "Layer":
                    color = layer_color
                elif color_mode == "Trail ID":
                    tid = tr.trail_id if tr.trail_id is not None else 0
                    color = TRAIL_PALETTE[tid % len(TRAIL_PALETTE)]
                else:  # Print vs Travel
                    color = "#38bdf8"
            else:
                color = "#475569"  # slate travel moves

            js_segments.append({
                "pts":     [[p[0], p[1]] for p in pts],
                "z":       z,
                "color":   color,
                "is_print": is_print,
                "layer":   layer.index,
                "trail":   tr.trail_id if tr.trail_id is not None else 0,
            })

    # ── Bounding box for camera framing ─────────────────────────────────────
    bbox = stage3.coordinate_frame.page_bbox_world_mm
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    max_z = layers[-1].z_mm if layers else 400.0
    span  = max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0)

    segments_json   = json.dumps(js_segments)
    wireframe_mode  = "true" if render_mode == "Wireframe Toolpath" else "false"

    # ── Three.js HTML component ──────────────────────────────────────────────
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0b0f19; overflow:hidden; font-family:system-ui,sans-serif; }}
  canvas {{ display:block; width:100%!important; height:100%!important; }}
  #info {{
    position:absolute; top:10px; right:12px;
    color:#94a3b8; font-size:11px; text-align:right;
    pointer-events:none; line-height:1.6;
  }}
  #info b {{ color:#38bdf8; }}
  #loading {{
    position:absolute; inset:0; display:flex; align-items:center;
    justify-content:center; color:#38bdf8; font-size:14px;
    background:#0b0f19; z-index:10;
  }}
</style>
</head>
<body>
<div id="loading">⚙ Building 3D concrete model…</div>
<canvas id="c"></canvas>
<div id="info">
  <b>PrintPlan AI — 3D Toolpath</b><br>
  Drag to orbit · Scroll to zoom · Right-drag to pan<br>
  Layers {start_layer}–{end_layer} &nbsp;|&nbsp; Z = {max_z:.0f} mm
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

// ── Data from Python ────────────────────────────────────────────────────────
const SEGMENTS    = {segments_json};
const BEAD_W      = {bead_width_mm};
const BEAD_H      = {layer_height_mm};
const WIREFRAME   = {wireframe_mode};
const CENTER_X    = {cx};
const CENTER_Y    = {cy};
const MAX_Z       = {max_z};
const SPAN        = {span};
const SCALE       = 1.0 / SPAN;   // normalise to ~1 unit

// ── Renderer ────────────────────────────────────────────────────────────────
const canvas   = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{ canvas, antialias:true }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled   = true;
renderer.shadowMap.type      = THREE.PCFSoftShadowMap;
renderer.toneMapping         = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.3;
renderer.outputColorSpace    = THREE.SRGBColorSpace;

// ── Scene ───────────────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f19);
scene.fog        = new THREE.FogExp2(0x0b0f19, 0.18);

// ── Camera ──────────────────────────────────────────────────────────────────
const camera = new THREE.PerspectiveCamera(42, window.innerWidth/window.innerHeight, 0.001, 200);
camera.position.set(1.5, 1.2, 1.5);

// ── Lighting ────────────────────────────────────────────────────────────────
// Ambient fill
const ambient = new THREE.AmbientLight(0xffffff, 0.35);
scene.add(ambient);

// Key sun light
const sun = new THREE.DirectionalLight(0xfff4e0, 2.2);
sun.position.set(2, 3, 2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 0.01;
sun.shadow.camera.far  = 20;
sun.shadow.camera.left = sun.shadow.camera.bottom = -2;
sun.shadow.camera.right= sun.shadow.camera.top    =  2;
sun.shadow.bias = -0.0005;
scene.add(sun);

// Cool sky bounce (rim)
const rim = new THREE.DirectionalLight(0x8ab4f8, 0.6);
rim.position.set(-2, 1, -1);
scene.add(rim);

// Warm fill under
const fill = new THREE.HemisphereLight(0x334155, 0x0b0f19, 0.4);
scene.add(fill);

// ── Ground plane ────────────────────────────────────────────────────────────
const groundGeo = new THREE.PlaneGeometry(6, 6);
const groundMat = new THREE.MeshStandardMaterial({{
  color: 0x0f172a, roughness:1.0, metalness:0.0
}});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x  = -Math.PI/2;
ground.position.y  = -0.002;
ground.receiveShadow = true;
scene.add(ground);

// Grid
const grid = new THREE.GridHelper(3, 30, 0x1e293b, 0x1e293b);
grid.position.y = -0.001;
scene.add(grid);

// ── Bead cross-section shape (ellipse) ──────────────────────────────────────
function makeBeadShape() {{
  const shape = new THREE.Shape();
  const hw = (BEAD_W * SCALE) / 2;
  const hh = (BEAD_H * SCALE) / 2;
  const segs = 8;
  for (let i = 0; i <= segs; i++) {{
    const a = (i / segs) * Math.PI * 2;
    const x = Math.cos(a) * hw;
    const y = Math.sin(a) * hh;
    i === 0 ? shape.moveTo(x,y) : shape.lineTo(x,y);
  }}
  return shape;
}}
const beadShape = makeBeadShape();

// ── Material cache ───────────────────────────────────────────────────────────
const matCache = {{}};
function getMat(hexColor, isTravel) {{
  const key = hexColor + (isTravel?'t':'p');
  if (!matCache[key]) {{
    matCache[key] = new THREE.MeshStandardMaterial({{
      color:     new THREE.Color(hexColor),
      roughness: isTravel ? 0.9 : 0.72,
      metalness: 0.0,
      wireframe: WIREFRAME && !isTravel,
      transparent: isTravel,
      opacity:   isTravel ? 0.35 : 1.0,
    }});
  }}
  return matCache[key];
}}

// ── Build geometry asynchronously in chunks (prevents UI freeze) ─────────────
async function buildScene() {{
  const CHUNK = 30;   // segments per frame
  const startMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.01, 8, 8),
    new THREE.MeshStandardMaterial({{ color: 0xf59e0b, emissive: 0xf59e0b, emissiveIntensity:0.5 }})
  );

  for (let si = 0; si < SEGMENTS.length; si++) {{
    const seg     = SEGMENTS[si];
    const pts2d   = seg.pts;
    const z       = seg.z * SCALE;
    const isTravel = !seg.is_print;

    if (pts2d.length < 2) continue;

    // Convert to 3D points (Y → Z axis, Z height on Y axis)
    const path3d = pts2d.map(([x,y]) => new THREE.Vector3(
      (x - CENTER_X) * SCALE,
       z,
      (y - CENTER_Y) * SCALE
    ));

    if (isTravel) {{
      // Travel moves → simple dashed line (cheaper)
      const geo = new THREE.BufferGeometry().setFromPoints(path3d);
      const mat = new THREE.LineBasicMaterial({{
        color: 0x475569, opacity: 0.3, transparent: true
      }});
      scene.add(new THREE.Line(geo, mat));
    }} else {{
      // Print moves → ExtrudeGeometry with elliptical cross-section
      for (let pi = 0; pi < path3d.length - 1; pi++) {{
        const p0 = path3d[pi];
        const p1 = path3d[pi + 1];
        const dist = p0.distanceTo(p1);
        if (dist < 0.0005) continue;

        const extSettings = {{
          steps:       1,
          extrudePath: new THREE.LineCurve3(p0, p1),
          bevelEnabled: false,
        }};

        try {{
          const geo  = new THREE.ExtrudeGeometry(beadShape, extSettings);
          const mesh = new THREE.Mesh(geo, getMat(seg.color, false));
          mesh.castShadow    = true;
          mesh.receiveShadow = true;
          scene.add(mesh);
        }} catch(e) {{
          // Degenerate segment — skip
        }}
      }}

      // Nozzle start marker
      const m = startMarker.clone();
      m.position.copy(path3d[0]);
      scene.add(m);
    }}

    // Yield every CHUNK segments so browser stays responsive
    if (si % CHUNK === 0) {{
      await new Promise(r => setTimeout(r, 0));
    }}
  }}

  // Hide loading overlay once done
  document.getElementById('loading').style.display = 'none';
}}

// ── OrbitControls ────────────────────────────────────────────────────────────
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping  = true;
controls.dampingFactor  = 0.07;
controls.target.set(0, (MAX_Z * SCALE) / 2, 0);
controls.minDistance    = 0.05;
controls.maxDistance    = 8;
controls.update();

// ── Render loop ──────────────────────────────────────────────────────────────
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

// ── Resize handler ───────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

// ── Kick off async build ─────────────────────────────────────────────────────
buildScene();
</script>
</body>
</html>
"""
    components.html(html, height=height_px, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY PLOTLY FALLBACK  (kept for Wireframe / quick debug use)
# ─────────────────────────────────────────────────────────────────────────────

def build_3d_toolpath_fig(
    stage3: Stage3Output,
    color_mode: Literal["Layer", "Trail ID", "Print vs Travel"] = "Layer",
    render_mode: Literal["Extruded Concrete Beads", "Wireframe Toolpath"] = "Extruded Concrete Beads",
    show_travel: bool = True,
    layer_range: tuple[int, int] | None = None,
    bead_width_mm: float = 40.0,
) -> go.Figure:
    """
    Legacy Plotly 3D figure (Scatter3d lines).
    Use build_3d_toolpath_webgl() for realistic rendering.
    Retained for fallback / Wireframe Toolpath mode compatibility.
    """
    fig = go.Figure()
    layers = stage3.layers
    if not layers:
        return fig

    start_layer, end_layer = 0, len(layers) - 1
    if layer_range is not None:
        start_layer = max(0, min(layer_range[0], len(layers) - 1))
        end_layer   = max(start_layer, min(layer_range[1], len(layers) - 1))

    selected_layers = layers[start_layer : end_layer + 1]
    trail_colors    = px.colors.qualitative.Bold
    legend_added    = set()
    line_width      = 12 if render_mode == "Extruded Concrete Beads" else 4

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
                    norm_z   = layer.index / max(1, len(layers) - 1)
                    c_idx    = int(norm_z * 255)
                    color    = px.colors.sequential.Viridis[c_idx % len(px.colors.sequential.Viridis)]
                    group_name = f"Layer {layer.index} (Z={z:.0f}mm)"
                elif color_mode == "Trail ID":
                    trail_id   = tr.trail_id if tr.trail_id is not None else 0
                    color      = trail_colors[trail_id % len(trail_colors)]
                    group_name = f"Trail T{trail_id}"
                else:
                    color      = "#38bdf8"
                    group_name = "Extruded Concrete Wall"

                show_in_legend = group_name not in legend_added
                if show_in_legend:
                    legend_added.add(group_name)

                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
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
                ))
                fig.add_trace(go.Scatter3d(
                    x=[xs[0]], y=[ys[0]], z=[zs[0]], mode="markers",
                    marker=dict(size=4, color="#f59e0b", symbol="circle"),
                    hoverinfo="text",
                    hovertext=f"Start Trail T{tr.trail_id} (L{layer.index})",
                    showlegend=False,
                ))
            elif tr.kind == "travel" and show_travel:
                group_name     = "Nozzle Travel Path"
                show_in_legend = group_name not in legend_added
                if show_in_legend:
                    legend_added.add(group_name)
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    name=group_name,
                    line=dict(color="#94a3b8", width=2, dash="dash"),
                    hoverinfo="text",
                    hovertext=[
                        f"<b>Nozzle Travel Move</b> (Layer {layer.index})<br>"
                        f"Pos: ({x:.1f}, {y:.1f}, {z:.1f}) mm"
                        for x, y in zip(xs, ys)
                    ],
                    showlegend=show_in_legend,
                ))

    bbox    = stage3.coordinate_frame.page_bbox_world_mm
    x0, y0, x1, y1 = bbox
    range_x = max(100.0, x1 - x0)
    range_y = max(100.0, y1 - y0)
    max_z   = max(100.0, layers[-1].z_mm if layers else 100.0)
    aspect_x = 1.0
    aspect_y = range_y / range_x
    aspect_z = max(0.3, min(0.8, (max_z / range_x) * 3.0))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.95)",
        plot_bgcolor="rgba(15, 23, 42, 0.95)",
        margin=dict(l=10, r=10, b=10, t=30),
        scene=dict(
            xaxis=dict(title="X (mm)", gridcolor="rgba(148,163,184,0.2)", backgroundcolor="rgba(11,15,25,0.95)", tickfont=dict(color="#94a3b8")),
            yaxis=dict(title="Y (mm)", gridcolor="rgba(148,163,184,0.2)", backgroundcolor="rgba(11,15,25,0.95)", tickfont=dict(color="#94a3b8")),
            zaxis=dict(title="Z (mm)", gridcolor="rgba(148,163,184,0.2)", backgroundcolor="rgba(11,15,25,0.95)", tickfont=dict(color="#94a3b8")),
            aspectmode="manual",
            aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z),
            camera=dict(eye=dict(x=1.4, y=-1.4, z=1.0)),
        ),
        legend=dict(
            font=dict(color="#f8fafc", size=11),
            yanchor="top", y=0.98, xanchor="left", x=0.02,
            bgcolor="rgba(15,23,42,0.9)",
            bordercolor="rgba(56,189,248,0.4)", borderwidth=1,
        ),
        height=620,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2D LAYER SLICE  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def build_layer_2d_fig(layer: Layer, layer_idx: int, bbox: tuple[float, float, float, float]) -> go.Figure:
    """Build high-resolution 2D interactive floor plan slice view of a single layer."""
    fig = go.Figure()
    colors = ["#38bdf8","#f43f5e","#10b981","#f59e0b","#a855f7","#ec4899"]
    for tr in layer.traces:
        pts = tr.points
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if tr.kind == "print":
            c     = colors[(tr.trail_id or 0) % len(colors)]
            label = f"Trail T{tr.trail_id} ({'Closed' if tr.is_closed else 'Open'})"
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=label,
                line=dict(color=c, width=4), marker=dict(size=4),
                hoverinfo="text",
                hovertext=[f"X: {x:.1f} mm, Y: {y:.1f} mm" for x,y in zip(xs,ys)]
            ))
            fig.add_trace(go.Scatter(
                x=[xs[0]], y=[ys[0]], mode="markers",
                marker=dict(size=10, color=c, symbol="star"),
                name=f"Start T{tr.trail_id}", showlegend=False
            ))
        else:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name="Travel Move",
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
                showlegend=False
            ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15,23,42,0.95)",
        plot_bgcolor="rgba(11,15,25,0.95)",
        title=dict(
            text=f"Layer {layer_idx} (Z = {layer.z_mm:.1f} mm) — {layer.n_print_traces} Nozzle Trails",
            font=dict(color="#f8fafc", size=14)
        ),
        xaxis=dict(title=dict(text="X (mm)", font=dict(color="#f8fafc")), scaleanchor="y", scaleratio=1, gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#f8fafc", size=11)),
        yaxis=dict(title=dict(text="Y (mm)", font=dict(color="#f8fafc")), gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#f8fafc", size=11)),
        legend=dict(font=dict(color="#f8fafc", size=12), bgcolor="rgba(15,23,42,0.9)", bordercolor="rgba(56,189,248,0.4)", borderwidth=1),
        margin=dict(l=50, r=40, b=50, t=60),
        height=580,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# NOZZLE ANIMATION  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def build_3d_nozzle_anim(stage3: Stage3Output, layer_idx: int = 0, bbox: tuple[float, float, float, float] = (0,0,10000,10000)) -> go.Figure:
    """Build step-by-step 3D animated playback of the nozzle trajectory for a single layer."""
    if not stage3.layers or layer_idx >= len(stage3.layers):
        return go.Figure()
    layer = stage3.layers[layer_idx]
    z = layer.z_mm
    path_x, path_y, path_z, hover_labels = [], [], [], []
    for tr in layer.traces:
        for p in tr.points:
            path_x.append(p[0])
            path_y.append(p[1])
            path_z.append(z)
            hover_labels.append(f"{tr.kind.upper()} | X:{p[0]:.1f}, Y:{p[1]:.1f}, Z:{z:.1f}")
    if not path_x:
        return go.Figure()
    x0, y0, x1, y1 = bbox
    range_x = max(100.0, x1 - x0)
    range_y = max(100.0, y1 - y0)
    aspect_y = range_y / range_x
    fig = go.Figure(
        data=[
            go.Scatter3d(x=[path_x[0]], y=[path_y[0]], z=[path_z[0]], mode="lines", line=dict(color="#38bdf8", width=10), name="Printed Concrete Bead"),
            go.Scatter3d(x=[path_x[0]], y=[path_y[0]], z=[path_z[0]+50], mode="markers", marker=dict(color="#f59e0b", size=12, symbol="diamond"), name="3D Printer Nozzle Head")
        ],
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15,23,42,0.95)",
            plot_bgcolor="rgba(11,15,25,0.95)",
            scene=dict(
                xaxis=dict(title="X (mm)", gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#94a3b8")),
                yaxis=dict(title="Y (mm)", gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#94a3b8")),
                zaxis=dict(title="Z (mm)", range=[z-20, z+200], gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#94a3b8")),
                aspectmode="manual",
                aspectratio=dict(x=1.0, y=aspect_y, z=0.4),
                camera=dict(eye=dict(x=1.3, y=-1.3, z=0.8))
            ),
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[
                    dict(label="▶ Play Trajectory", method="animate", args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ],
                x=0.02, y=1.05,
                bgcolor="rgba(15,23,42,0.9)",
                font=dict(color="#38bdf8", size=12)
            )],
            legend=dict(font=dict(color="#f8fafc", size=11), bgcolor="rgba(15,23,42,0.9)", bordercolor="rgba(56,189,248,0.4)", borderwidth=1, x=0.75, y=0.95),
            margin=dict(l=10, r=10, b=10, t=40),
            height=580
        )
    )
    step_size = max(1, len(path_x) // 40)
    indices   = list(range(0, len(path_x), step_size))
    if indices[-1] != len(path_x) - 1:
        indices.append(len(path_x) - 1)
    frames = []
    for idx in indices:
        frames.append(go.Frame(data=[
            go.Scatter3d(x=path_x[:idx+1], y=path_y[:idx+1], z=path_z[:idx+1], mode="lines", line=dict(color="#38bdf8", width=10)),
            go.Scatter3d(x=[path_x[idx]], y=[path_y[idx]], z=[path_z[idx]+50], mode="markers", marker=dict(color="#f59e0b", size=12, symbol="diamond"))
        ], name=f"frame_{idx}"))
    fig.frames = frames
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MATERIAL METRICS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def calc_material_metrics(
    stage3: Stage3Output,
    bead_width_mm: float,
    layer_height_mm: float,
    print_speed_mm_s: float,
    density_kg_m3: float = 2300.0,
) -> dict:
    """Calculate concrete volume, mass, print time, and efficiency metrics."""
    meta            = stage3.meta
    total_print_m   = meta.get("total_print_length_m", 0.0)
    total_travel_m  = meta.get("total_travel_length_m", 0.0)
    bead_w_m        = bead_width_mm  / 1000.0
    layer_h_m       = layer_height_mm / 1000.0
    total_volume_m3 = total_print_m * bead_w_m * layer_h_m
    total_mass_kg   = total_volume_m3 * density_kg_m3
    total_mass_tonnes = total_mass_kg / 1000.0
    print_time_sec  = (total_print_m * 1000.0) / max(1.0, print_speed_mm_s)
    travel_speed_mm_s = print_speed_mm_s * 1.5
    travel_time_sec = (total_travel_m * 1000.0) / travel_speed_mm_s
    total_time_sec  = print_time_sec + travel_time_sec
    minutes  = int(total_time_sec // 60)
    seconds  = int(total_time_sec  % 60)
    time_str = f"{minutes}m {seconds}s" if minutes < 60 else f"{minutes//60}h {minutes%60}m"
    total_m  = total_print_m + total_travel_m
    efficiency_pct = (total_print_m / total_m * 100.0) if total_m > 0 else 100.0
    return {
        "total_print_m":   total_print_m,
        "total_travel_m":  total_travel_m,
        "volume_m3":       round(total_volume_m3, 3),
        "mass_tonnes":     round(total_mass_tonnes, 3),
        "mass_kg":         round(total_mass_kg, 1),
        "print_time_sec":  round(print_time_sec, 1),
        "time_str":        time_str,
        "efficiency_pct":  round(efficiency_pct, 1),
    }
