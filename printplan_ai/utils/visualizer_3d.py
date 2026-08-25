"""PrintPlan AI — WebGL Interactive 3D Visualizer & Material Estimator."""

from __future__ import annotations
import json
import math
from typing import Literal, List, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

from printplan_ai.models import Stage3Output, Layer, Trace


# ---------------------------------------------------------------------------
# Opening-aware segment filtering
# ---------------------------------------------------------------------------

def _point_in_opening_xy(px: float, py: float, opening) -> bool:
    """
    Check whether a world-mm point (px, py) lies within the XY footprint
    of an opening.  Uses cx_world_mm / cy_world_mm stored by opening_detection
    v3.0 — these are in the same coordinate system as the toolpath segments.
    """
    half_w  = opening.width_mm / 2.0 + 50.0   # tolerance 50 mm
    perp    = 300.0 / 2.0 + 50.0              # assume 300 mm wall thickness + tolerance

    if opening.wall_direction == "horizontal":
        # Opening spans along X; wall runs along X, normal is Y
        in_x = abs(px - opening.cx_world_mm) <= half_w
        in_y = abs(py - opening.cy_world_mm) <= perp
    else:
        # Opening spans along Y; wall runs along Y, normal is X
        in_x = abs(px - opening.cx_world_mm) <= perp
        in_y = abs(py - opening.cy_world_mm) <= half_w

    return in_x and in_y


def _segment_suppressed_at_z(
    pts: list,
    z_mm: float,
    openings: list,
    scale_pts_to_mm: float = 1.0,   # kept for API compat, no longer used
) -> bool:
    """
    Return True if the midpoint of a segment falls within any opening's
    XY footprint AND z_mm is inside its void Z range.
    """
    if not openings:
        return False
    n = len(pts)
    mid_x = sum(p[0] for p in pts) / n
    mid_y = sum(p[1] for p in pts) / n
    for op in openings:
        z_suppress_top = op.z_void_top_mm + op.lintel_thickness_mm
        if op.z_void_bottom_mm <= z_mm < z_suppress_top:
            if _point_in_opening_xy(mid_x, mid_y, op):
                return True
    return False


def filter_segments_for_openings(
    js_segments: list,
    openings: list,
    scale_pts_to_mm: float = 1.0,   # kept for API compat
) -> list:
    """
    Remove print segments whose midpoint lies inside an opening void zone.
    Travel moves are never filtered.
    """
    if not openings:
        return js_segments
    filtered = []
    for seg in js_segments:
        if not seg["is_print"]:
            filtered.append(seg)
            continue
        if _segment_suppressed_at_z(seg["pts"], seg["z"], openings):
            continue
        filtered.append(seg)
    return filtered


# ---------------------------------------------------------------------------
# 3D WebGL renderer
# ---------------------------------------------------------------------------

def build_3d_toolpath_webgl(
    stage3,
    color_mode: str = "Layer",
    render_mode: str = "Extruded Concrete Beads",
    show_travel: bool = True,
    layer_range=None,
    bead_width_mm: float = 40.0,
    layer_height_mm: float = 20.0,
    height_px: int = 680,
    openings: Optional[list] = None,
    drawing_scale: int = 100,
) -> None:
    layers = stage3.layers
    if not layers:
        components.html("<p style='color:#f87171'>No layers to render.</p>", height=80)
        return

    start_layer = 0
    end_layer = len(layers) - 1
    if layer_range is not None:
        start_layer = max(0, min(layer_range[0], end_layer))
        end_layer   = max(start_layer, min(layer_range[1], end_layer))

    selected_layers = layers[start_layer : end_layer + 1]

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
    js_segments = []

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
            if is_print:
                if color_mode == "Layer":
                    color = layer_color
                elif color_mode == "Trail ID":
                    tid = tr.trail_id if tr.trail_id is not None else 0
                    color = TRAIL_PALETTE[tid % len(TRAIL_PALETTE)]
                else:
                    color = "#38bdf8"
            else:
                color = "#475569"
            js_segments.append({
                "pts":      [[p[0], p[1]] for p in pts],
                "z":        z,
                "color":    color,
                "is_print": is_print,
                "layer":    layer.index,
                "trail":    tr.trail_id if tr.trail_id is not None else 0,
            })

    bbox  = stage3.coordinate_frame.page_bbox_world_mm
    cx    = (bbox[0] + bbox[2]) / 2.0
    cy    = (bbox[1] + bbox[3]) / 2.0
    max_z = layers[-1].z_mm if layers else 400.0
    span  = max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0)
    # Apply opening void suppression if openings were detected
    if openings:
        js_segments = filter_segments_for_openings(js_segments, openings)

    segments_json  = json.dumps(js_segments)
    wireframe_mode = "true" if render_mode == "Wireframe Toolpath" else "false"

    # --- Lintel blocks & concrete caps above lintel ---
    LINTEL_COLORS = {
        "timber": "#92400e", "steel": "#475569",
        "precast": "#6b7280", "printed": "#1d4ed8",
    }

    def _wall_centreline_for_opening(op, all_layers):
        """
        Find the true wall centreline at the opening.

        Strategy: collect midpoints of all print segments whose along-wall
        coordinate falls inside the opening span. The perpendicular coords
        will cluster around two values (the two parallel PDF wall lines).
        Split into two clusters at the largest gap, average each cluster,
        then take the midpoint of the two cluster centres — this equals the
        midpoint of the two PDF wall-line centre-to-centre positions exactly.

        For a horizontal wall: perp axis = Y  →  centre_Y = (c1_Y + c2_Y) / 2
        For a vertical wall:   perp axis = X  →  centre_X = (c1_X + c2_X) / 2
        """
        # Search window: only segments whose along-wall coord is within the opening span
        half_w = op.width_mm / 2.0 + 50.0
        perp_coords = []

        for layer in all_layers:
            if layer.z_mm >= op.z_void_bottom_mm:
                break                          # only full-wall layers (below void)
            for tr in layer.traces:
                if tr.kind != "print":
                    continue
                for i in range(len(tr.points) - 1):
                    mx = (tr.points[i][0] + tr.points[i+1][0]) / 2
                    my = (tr.points[i][1] + tr.points[i+1][1]) / 2
                    if op.wall_direction == "horizontal":
                        if abs(mx - op.cx_world_mm) <= half_w:
                            perp_coords.append(my)
                    else:
                        if abs(my - op.cy_world_mm) <= half_w:
                            perp_coords.append(mx)

        if not perp_coords:
            return op.cx_world_mm, op.cy_world_mm   # fallback: use arc origin

        # Cluster into two wall lines by finding the largest gap between
        # sorted unique perpendicular values. Each cluster = one PDF wall line.
        sorted_p = sorted(set(round(c, 0) for c in perp_coords))
        if len(sorted_p) == 1:
            centre_perp = sorted_p[0]
        else:
            # Find the index of the largest gap
            gaps = [sorted_p[i+1] - sorted_p[i] for i in range(len(sorted_p) - 1)]
            split_idx = gaps.index(max(gaps))
            wall1 = sorted_p[:split_idx + 1]
            wall2 = sorted_p[split_idx + 1:]
            # Centre of each cluster = mean of its values
            c1 = sum(wall1) / len(wall1)
            c2 = sum(wall2) / len(wall2)
            # Midpoint between the two PDF wall-line centres
            centre_perp = (c1 + c2) / 2.0

        if op.wall_direction == "horizontal":
            return op.cx_world_mm, centre_perp
        else:
            return centre_perp, op.cy_world_mm

    lintel_blocks = []
    cap_blocks = []
    if openings:
        for op in openings:
            lt = op.lintel_type.value if hasattr(op.lintel_type, "value") else str(op.lintel_type)
            z_resume = op.z_void_top_mm + op.lintel_thickness_mm
            wall_t = getattr(op, "wall_thickness_mm", None) or bead_width_mm
            # Find true wall centreline from toolpath geometry
            true_cx, true_cy = _wall_centreline_for_opening(op, layers)
            if lt != "none":
                lintel_blocks.append({
                    "cx": true_cx, "cy": true_cy,
                    "w": op.width_mm, "t": wall_t,
                    "z0": op.z_void_top_mm, "z1": z_resume,
                    "dir": op.wall_direction,
                    "col": LINTEL_COLORS.get(lt, "#92400e"),
                })
            if z_resume < max_z:
                cap_blocks.append({
                    "cx": true_cx, "cy": true_cy,
                    "w": op.width_mm, "t": wall_t,
                    "z0": z_resume, "z1": max_z,
                    "dir": op.wall_direction,
                })
    lintels_json = json.dumps(lintel_blocks)
    caps_json    = json.dumps(cap_blocks)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#0b0f19;overflow:hidden;}}
canvas{{display:block;width:100%!important;height:100%!important;}}
#info{{position:absolute;top:10px;right:12px;color:#94a3b8;font-size:11px;text-align:right;pointer-events:none;line-height:1.6;}}
#info b{{color:#38bdf8;}}
#loading{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#38bdf8;font-size:14px;background:#0b0f19;z-index:10;}}
</style></head><body>
<div id="loading">Building 3D concrete model...</div>
<canvas id="c"></canvas>
<div id="info"><b>PrintPlan AI 3D</b><br>Drag=orbit Scroll=zoom RightDrag=pan<br>Layers {start_layer}-{end_layer} | Z={max_z:.0f}mm</div>
<script type="importmap">{{"imports":{{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}}}</script>
<script type="module">
import * as THREE from 'three';
import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';
const SEGS={segments_json};
const LINTELS={lintels_json};
const CAPS={caps_json};
const BW={bead_width_mm},BH={layer_height_mm},WF={wireframe_mode};
const CX={cx},CY={cy},MZ={max_z},SP={span},SC=1.0/SP;
const renderer=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true}});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(innerWidth,innerHeight);
renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.3;
renderer.outputColorSpace=THREE.SRGBColorSpace;
const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0b0f19);
scene.fog=new THREE.FogExp2(0x0b0f19,0.18);
const camera=new THREE.PerspectiveCamera(42,innerWidth/innerHeight,0.001,200);
camera.position.set(1.5,1.2,1.5);
scene.add(new THREE.AmbientLight(0xffffff,0.35));
const sun=new THREE.DirectionalLight(0xfff4e0,2.2);
sun.position.set(2,3,2);sun.castShadow=true;
sun.shadow.mapSize.set(2048,2048);
sun.shadow.camera.near=0.01;sun.shadow.camera.far=20;
sun.shadow.camera.left=sun.shadow.camera.bottom=-2;
sun.shadow.camera.right=sun.shadow.camera.top=2;
sun.shadow.bias=-0.0005;scene.add(sun);
const rim=new THREE.DirectionalLight(0x8ab4f8,0.6);rim.position.set(-2,1,-1);scene.add(rim);
scene.add(new THREE.HemisphereLight(0x334155,0x0b0f19,0.4));
const gnd=new THREE.Mesh(new THREE.PlaneGeometry(6,6),new THREE.MeshStandardMaterial({{color:0x0f172a,roughness:1}}));
gnd.rotation.x=-Math.PI/2;gnd.position.y=-0.002;gnd.receiveShadow=true;scene.add(gnd);
scene.add(new THREE.GridHelper(3,30,0x1e293b,0x1e293b));
function makeShape(){{
  const s=new THREE.Shape(),hw=(BW*SC)/2,hh=(BH*SC)/2;
  for(let i=0;i<=8;i++){{const a=(i/8)*Math.PI*2;i===0?s.moveTo(Math.cos(a)*hw,Math.sin(a)*hh):s.lineTo(Math.cos(a)*hw,Math.sin(a)*hh);}}
  return s;
}}
const bead=makeShape(),mats={{}};
function mat(c,t){{
  const k=c+(t?'t':'p');
  if(!mats[k])mats[k]=new THREE.MeshStandardMaterial({{color:new THREE.Color(c),roughness:t?0.9:0.72,metalness:0,wireframe:WF&&!t,transparent:t,opacity:t?0.35:1}});
  return mats[k];
}}

async function build(){{
  for(let si=0;si<SEGS.length;si++){{
    const seg=SEGS[si];if(seg.pts.length<2)continue;
    const z=seg.z*SC,travel=!seg.is_print;
    const p3=seg.pts.map(([x,y])=>new THREE.Vector3((x-CX)*SC,z,-(y-CY)*SC));  // negate Y→Z so 3D north matches PDF orientation
    if(travel){{
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p3),new THREE.LineBasicMaterial({{color:0x475569,opacity:0.3,transparent:true}})));
    }}else{{
      for(let pi=0;pi<p3.length-1;pi++){{
        const d=p3[pi].distanceTo(p3[pi+1]);if(d<0.0005)continue;
        try{{
          const geo=new THREE.ExtrudeGeometry(bead,{{steps:1,extrudePath:new THREE.LineCurve3(p3[pi],p3[pi+1]),bevelEnabled:false}});
          const m=new THREE.Mesh(geo,mat(seg.color,false));m.castShadow=true;m.receiveShadow=true;scene.add(m);
        }}catch(e){{}}
      }}
     
    }}
    if(si%30===0)await new Promise(r=>setTimeout(r,0));
  }}
  // --- Lintel blocks (timber/steel/precast) ---
  function addBox(cx,cy,z0,z1,wMM,tMM,dir,color,opacity){{
    const hMM=z1-z0;
    let geoW,geoD;
    if(dir==='horizontal'){{geoW=wMM*SC;geoD=tMM*SC;}}
    else{{geoW=tMM*SC;geoD=wMM*SC;}}
    const geo=new THREE.BoxGeometry(geoW,(hMM*SC),geoD);
    const mat=new THREE.MeshStandardMaterial({{color:new THREE.Color(color),roughness:0.7,metalness:0,transparent:opacity<1,opacity:opacity}});
    const mesh=new THREE.Mesh(geo,mat);
    // cx/cy is the wall centreline — place box directly on it (no offset needed)
    let px=(cx-CX)*SC, pz=-(cy-CY)*SC;
    mesh.position.set(px,((z0+hMM/2)*SC),pz);
    mesh.castShadow=true;mesh.receiveShadow=true;scene.add(mesh);
    const wf=new THREE.LineSegments(new THREE.EdgesGeometry(geo),new THREE.LineBasicMaterial({{color:0x000000,opacity:0.4,transparent:true}}));
    wf.position.copy(mesh.position);scene.add(wf);
  }}
  for(const lb of LINTELS){{
    addBox(lb.cx,lb.cy,lb.z0,lb.z1,lb.w,lb.t,lb.dir,lb.col,1.0);
  }}
  // Concrete cap above lintel (semi-transparent to show it connects to wall)
  for(const cb of CAPS){{
    addBox(cb.cx,cb.cy,cb.z0,cb.z1,cb.w,cb.t,cb.dir,'#94a3b8',0.25);
  }}
  document.getElementById('loading').style.display='none';
}}
const ctrl=new OrbitControls(camera,renderer.domElement);
ctrl.enableDamping=true;ctrl.dampingFactor=0.07;
ctrl.target.set(0,(MZ*SC)/2,0);ctrl.minDistance=0.05;ctrl.maxDistance=8;ctrl.update();
(function loop(){{requestAnimationFrame(loop);ctrl.update();renderer.render(scene,camera);}})();
window.addEventListener('resize',()=>{{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);}});
build();
</script></body></html>"""
    components.html(html, height=height_px, scrolling=False)


def build_3d_toolpath_fig(
    stage3: Stage3Output,
    color_mode: Literal["Layer", "Trail ID", "Print vs Travel"] = "Layer",
    render_mode: Literal["Extruded Concrete Beads", "Wireframe Toolpath"] = "Extruded Concrete Beads",
    show_travel: bool = True,
    layer_range=None,
    bead_width_mm: float = 40.0,
) -> go.Figure:
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
                    norm_z = layer.index / max(1, len(layers) - 1)
                    c_idx  = int(norm_z * 255)
                    color  = px.colors.sequential.Viridis[c_idx % len(px.colors.sequential.Viridis)]
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


def build_layer_2d_fig(layer: Layer, layer_idx: int, bbox: tuple) -> go.Figure:
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
                hovertext=[f"X: {x:.1f} mm, Y: {y:.1f} mm" for x, y in zip(xs, ys)]
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
        title=dict(text=f"Layer {layer_idx} (Z = {layer.z_mm:.1f} mm) — {layer.n_print_traces} Nozzle Trails", font=dict(color="#f8fafc", size=14)),
        xaxis=dict(title=dict(text="X (mm)", font=dict(color="#f8fafc")), scaleanchor="y", scaleratio=1, gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#f8fafc", size=11)),
        yaxis=dict(title=dict(text="Y (mm)", font=dict(color="#f8fafc")), gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#f8fafc", size=11)),
        legend=dict(font=dict(color="#f8fafc", size=12), bgcolor="rgba(15,23,42,0.9)", bordercolor="rgba(56,189,248,0.4)", borderwidth=1),
        margin=dict(l=50, r=40, b=50, t=60),
        height=580,
    )
    return fig


def build_3d_nozzle_anim(stage3: Stage3Output, layer_idx: int = 0, bbox=(0,0,10000,10000)) -> go.Figure:
    if not stage3.layers or layer_idx >= len(stage3.layers):
        return go.Figure()
    layer = stage3.layers[layer_idx]
    z = layer.z_mm
    path_x, path_y, path_z = [], [], []
    for tr in layer.traces:
        for p in tr.points:
            path_x.append(p[0]); path_y.append(p[1]); path_z.append(z)
    if not path_x:
        return go.Figure()
    x0, y0, x1, y1 = bbox
    aspect_y = max(100.0, y1-y0) / max(100.0, x1-x0)
    fig = go.Figure(
        data=[
            go.Scatter3d(x=[path_x[0]], y=[path_y[0]], z=[path_z[0]], mode="lines", line=dict(color="#38bdf8", width=10), name="Printed Concrete Bead"),
            go.Scatter3d(x=[path_x[0]], y=[path_y[0]], z=[path_z[0]+50], mode="markers", marker=dict(color="#f59e0b", size=12, symbol="diamond"), name="3D Printer Nozzle Head")
        ],
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15,23,42,0.95)", plot_bgcolor="rgba(11,15,25,0.95)",
            scene=dict(
                xaxis=dict(title="X (mm)", gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#94a3b8")),
                yaxis=dict(title="Y (mm)", gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#94a3b8")),
                zaxis=dict(title="Z (mm)", range=[z-20, z+200], gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#94a3b8")),
                aspectmode="manual", aspectratio=dict(x=1.0, y=aspect_y, z=0.4),
                camera=dict(eye=dict(x=1.3, y=-1.3, z=0.8))
            ),
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True)]),
                    dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ],
                x=0.02, y=1.05, bgcolor="rgba(15,23,42,0.9)", font=dict(color="#38bdf8", size=12)
            )],
            legend=dict(font=dict(color="#f8fafc", size=11), bgcolor="rgba(15,23,42,0.9)", bordercolor="rgba(56,189,248,0.4)", borderwidth=1, x=0.75, y=0.95),
            margin=dict(l=10, r=10, b=10, t=40), height=580
        )
    )
    step_size = max(1, len(path_x) // 40)
    indices   = list(range(0, len(path_x), step_size))
    if indices[-1] != len(path_x) - 1:
        indices.append(len(path_x) - 1)
    fig.frames = [go.Frame(data=[
        go.Scatter3d(x=path_x[:i+1], y=path_y[:i+1], z=path_z[:i+1], mode="lines", line=dict(color="#38bdf8", width=10)),
        go.Scatter3d(x=[path_x[i]], y=[path_y[i]], z=[path_z[i]+50], mode="markers", marker=dict(color="#f59e0b", size=12, symbol="diamond"))
    ], name=f"f{i}") for i in indices]
    return fig


def calc_material_metrics(
    stage3: Stage3Output,
    bead_width_mm: float,
    layer_height_mm: float,
    print_speed_mm_s: float,
    density_kg_m3: float = 2300.0,
) -> dict:
    meta              = stage3.meta
    total_print_m     = meta.get("total_print_length_m", 0.0)
    total_travel_m    = meta.get("total_travel_length_m", 0.0)
    total_volume_m3   = total_print_m * (bead_width_mm/1000.0) * (layer_height_mm/1000.0)
    total_mass_kg     = total_volume_m3 * density_kg_m3
    total_mass_tonnes = total_mass_kg / 1000.0
    print_time_sec    = (total_print_m * 1000.0) / max(1.0, print_speed_mm_s)
    travel_time_sec   = (total_travel_m * 1000.0) / (print_speed_mm_s * 1.5)
    total_time_sec    = print_time_sec + travel_time_sec
    minutes = int(total_time_sec // 60)
    seconds = int(total_time_sec  % 60)
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
