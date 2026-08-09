"""PrintPlan AI — Streamlit Interface.

Upload a vector PDF floor plan (or load built-in demo geometry) to generate
continuous 3D concrete printing toolpaths with interactive WebGL 3D visualization,
material estimation, and smart menu navigation.
"""

import tempfile, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import streamlit as st

from printplan_ai.config import M1Params, M2Params, M3Params, M4Params
from printplan_ai.stages import parse_pdf, reconstruct, generate_toolpath, synthesise_gcode
from printplan_ai.utils.theme import inject_custom_css, render_header
from printplan_ai.utils.sample_data import get_demo_stage1
from printplan_ai.utils.visualizer_3d import (
    build_3d_toolpath_fig,
    build_layer_2d_fig,
    build_3d_nozzle_anim,
    calc_material_metrics,
)


# ── Password Protection ───────────────────────────────────────────
def check_password() -> bool:
    """Simple password gate."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("<h1 style='text-align: center; color: #38bdf8;'>🏗️ PrintPlan AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Enter password to access 3D Concrete Printing G-Code Platform</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("Access Password", type="password")
        if st.button("Unlock Platform", type="primary", use_container_width=True):
            correct = st.secrets.get("password", "printplan2025") if hasattr(st, "secrets") else "printplan2025"
            try:
                correct = st.secrets["password"]
            except Exception:
                correct = "printplan2025"
                
            if password == correct:
                st.session_state.authenticated = True
                st.rerun()
                return True
            else:
                st.error("Incorrect password")
    return False


# ── Page Configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="Automated Toolpath Platform  of Large Scale 3D Concrete Printing",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not check_password():
    st.stop()

# Inject theme CSS
inject_custom_css()

# ── Sidebar: Presets & Parametric Smart Menu ─────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Print Settings & Presets")
    
    preset = st.selectbox(
        "Quick Configuration Preset",
        options=[
            "Custom Parameters",
            "Residential House (10x8m)",
            "Tiny House Prototype (4x4m)",
            "High-Speed Contours",
        ],
        index=1,
    )

    # Preset values
    default_scale = 50
    default_layers = 20
    default_layer_h = 20.0
    default_bead_w = 40.0
    default_speed = 60.0

    if preset == "Residential House (10x8m)":
        default_scale, default_layers, default_layer_h, default_bead_w, default_speed = 50, 20, 20.0, 40.0, 60.0
    elif preset == "Tiny House Prototype (4x4m)":
        default_scale, default_layers, default_layer_h, default_bead_w, default_speed = 20, 15, 15.0, 30.0, 80.0
    elif preset == "High-Speed Contours":
        default_scale, default_layers, default_layer_h, default_bead_w, default_speed = 100, 40, 25.0, 50.0, 100.0

    with st.expander("🛠️ Parametric Parameters", expanded=True):
        scale = st.number_input("Drawing scale (1:N)", value=default_scale, min_value=1, step=1)
        n_layers = st.number_input("Number of layers", value=default_layers, min_value=1, step=1)
        layer_h = st.number_input("Layer height (mm)", value=default_layer_h, min_value=1.0, step=1.0)
        bead_w = st.number_input("Bead width (mm)", value=default_bead_w, min_value=5.0, step=5.0)
        speed = st.number_input("Print speed (mm/s)", value=default_speed, min_value=10.0, step=10.0)
        wall_regex = st.text_input("Wall layer CAD pattern", value="2D Walls", help="Regex matching vector PDF OCG layer name")

    st.divider()
    
    st.markdown("### 📥 Input Source")
    input_mode = st.radio(
        "Choose Geometry Source",
        options=["⚡ Built-in Demo Floor Plan", "📂 Upload Vector PDF"],
        index=0,
    )

    pdf_file = None
    demo_preset = "Residential Pavilion (10x8m)"

    if input_mode == "📂 Upload Vector PDF":
        pdf_file = st.file_uploader(
            "Drop vector PDF floor plan here",
            type=["pdf"],
            help="CAD PDF with wall layer matching pattern above"
        )
    else:
        demo_preset = st.selectbox(
            "Select Demo Geometry",
            options=["Residential Pavilion (10x8m)", "Tiny Prototype Cell (4x4m)", "High-Speed Continuous Loop"],
            index=0
        )

    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("▶ Run Full Pipeline", type="primary", use_container_width=True)

    st.divider()
    st.caption("PrintPlan AI")
    st.caption("Hoang Khieu · RMIT University")
   


# Initialize pipeline results in Session State
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False


# Execute pipeline on user request or first load if using demo
if run_clicked or (not st.session_state.pipeline_run and input_mode == "⚡ Built-in Demo Floor Plan"):
    with st.spinner("Processing continuous toolpath pipeline..."):
        if input_mode == "📂 Upload Vector PDF":
            if pdf_file is not None:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_file.read())
                    pdf_path = tmp.name
                
                # M1
                stage1 = parse_pdf(pdf_path, M1Params(drawing_scale=scale, wall_layer_pattern=wall_regex))
                Path(pdf_path).unlink(missing_ok=True)
            else:
                st.warning("Please upload a vector PDF file first!")
                st.stop()
        else:
            # Stage 1 demo
            stage1 = get_demo_stage1(preset_name=demo_preset)

        # M2
        stage2 = reconstruct(stage1, M2Params())

        # M3
        stage3 = generate_toolpath(stage2, M3Params(layer_height_mm=layer_h), n_layers=int(n_layers))

        # M4
        gcode = synthesise_gcode(stage3, M4Params(bead_width_mm=bead_w, print_speed_mm_s=speed))

        # Calculate analytics metrics
        metrics = calc_material_metrics(stage3, bead_width_mm=bead_w, layer_height_mm=layer_h, print_speed_mm_s=speed)

        # Save to state
        st.session_state.stage1 = stage1
        st.session_state.stage2 = stage2
        st.session_state.stage3 = stage3
        st.session_state.gcode = gcode
        st.session_state.metrics = metrics
        st.session_state.pipeline_run = True

# Top Header Bar
status_text = "Toolpath Ready" if st.session_state.pipeline_run else "Waiting for Execution"
status_type = "success" if st.session_state.pipeline_run else "info"
render_header(status_text=status_text, status_type=status_type)

if not st.session_state.pipeline_run:
    st.info("👆 Please select an input source or upload a vector PDF floor plan in the sidebar and click **▶ Run Full Pipeline** to begin.")
    st.stop()

# Retrieve active state
stage1 = st.session_state.stage1
stage2 = st.session_state.stage2
stage3 = st.session_state.stage3
gcode = st.session_state.gcode
metrics = st.session_state.metrics


# ── Smart Menu Tab Navigation ────────────────────────────────────
tab_pipeline, tab_3d, tab_analytics, tab_gcode, tab_docs = st.tabs([
    "🚀 Pipeline & Overview",
    "🧊 3D Visualizer & Simulation",
    "📊 Analytics & Concrete Estimator",
    "📜 G-code Inspector & Download",
    "ℹ️ Technical Specs & Architecture",
])


# ── Tab 1: Pipeline & Overview ────────────────────────────────────
with tab_pipeline:
    st.markdown("### ⚡ Pipeline Execution Summary")
    
    # 4 Pipeline Stage Cards
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("M1 Ingestion", f"{stage1.meta.get('segment_count', 0)} segs", help="Extracted CAD segments")
    with col_m2:
        closed_cnt = sum(1 for z in stage2.zones if z.is_closed_loop)
        st.metric("M2 Reconstruction", f"{len(stage2.zones)} zones", f"{closed_cnt} closed loops")
    with col_m3:
        st.metric("M3 Toolpath", f"{len(stage3.layers)} layers", f"{metrics['total_print_m']:.0f} m print")
    with col_m4:
        n_cmds = sum(1 for l in gcode.splitlines() if l.strip() and not l.startswith(";"))
        st.metric("M4 G-code", f"{n_cmds} lines", f"{len(gcode)/1024:.1f} KB")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_preview, col_log = st.columns([3, 2])
    with col_preview:
        st.markdown("#### 📐 2D Floor Plan Toolpath Preview (Layer 0)")
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f172a")
        ax.set_facecolor("#0b1120")
        
        bbox = stage3.coordinate_frame.page_bbox_world_mm
        x0, y0, x1, y1 = bbox
        ax.set_xlim(x0 - 200, x1 + 200)
        ax.set_ylim(y0 - 200, y1 + 200)
        ax.set_aspect("equal")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

        colors_list = ["#38bdf8", "#f43f5e", "#10b981", "#f59e0b", "#a855f7"]
        layer0 = stage3.layers[0]

        for tr in layer0.traces:
            pts = tr.points
            segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
            if tr.kind == "print":
                c = colors_list[(tr.trail_id or 0) % len(colors_list)]
                ax.add_collection(LineCollection(segs, colors=[c], linewidths=2.5, zorder=3))
                ax.plot(pts[0][0], pts[0][1], "o", color=c, markersize=8, markeredgecolor="white", zorder=6)
            else:
                ax.add_collection(LineCollection(segs, colors=["#64748b"], linewidths=1.0, linestyles=":", zorder=2))

        ax.set_title(f"Layer 0 Continuous Nozzle Strokes: {layer0.n_print_traces} Trails", color="#f8fafc", fontweight="bold", fontsize=11)
        ax.grid(True, color="#334155", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_log:
        st.markdown("#### 📝 Execution Log")
        m1_meta = stage1.meta
        st.code(
            f"M1 · Ingest:   {m1_meta.get('segment_count', 0)} segments, {m1_meta.get('total_length_m', 0):.1f} m\n"
            f"M2 · Zones:    {len(stage2.zones)} zones ({closed_cnt} closed), {stage2.meta.get('total_trails', 0)} trails/layer\n"
            f"M3 · Toolpath: {stage3.meta.get('n_layers', 0)} layers, print={stage3.meta.get('total_print_length_m', 0):.1f} m, travel={stage3.meta.get('total_travel_length_m', 0):.1f} m\n"
            f"M4 · G-code:   {n_cmds} commands, {len(gcode)/1024:.1f} KB",
            language=None
        )
        
        st.markdown("#### ⬇️ Quick Download")
        st.download_button(
            label="Download G-code File (.gcode)",
            data=gcode,
            file_name="printplan_output.gcode",
            mime="text/plain",
            type="primary",
            use_container_width=True,
        )


# ── Tab 2: 3D Visualizer & Simulation ─────────────────────────────
with tab_3d:
    st.markdown("### 🧊 WebGL Interactive 3D Toolpath & Simulation")
    
    # 3D Visualizer Controls Toolbar
    c_color, c_range, c_travel = st.columns([2, 3, 2])
    with c_color:
        color_mode = st.selectbox("3D Color Scheme", options=["Layer", "Trail ID", "Print vs Travel"], index=0)
    with c_range:
        max_l = len(stage3.layers) - 1
        layer_range = st.slider("Inspect Layer Height Range", min_value=0, max_value=max_l, value=(0, max_l))
    with c_travel:
        show_travel = st.checkbox("Show Nozzle Travel Moves (Dashed)", value=True)

    # Build and display Plotly 3D Figure
    fig_3d = build_3d_toolpath_fig(
        stage3,
        color_mode=color_mode,
        show_travel=show_travel,
        layer_range=layer_range,
        bead_width_mm=bead_w
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.divider()

    col_slice, col_sim = st.columns([1, 1])
    with col_slice:
        st.markdown("#### 🔍 Single Layer 2D Slice Inspection")
        selected_layer_idx = st.number_input("Layer Index", min_value=0, max_value=max_l, value=0, step=1)
        layer_obj = stage3.layers[selected_layer_idx]
        fig_2d = build_layer_2d_fig(layer_obj, selected_layer_idx, stage3.coordinate_frame.page_bbox_world_mm)
        st.plotly_chart(fig_2d, use_container_width=True)

    with col_sim:
        st.markdown("#### 🎬 3D Nozzle Path Extrusion Playback")
        st.caption("Click ▶ Play to simulate the printer nozzle moving along the G-code toolpath live in 3D.")
        fig_anim = build_3d_nozzle_anim(stage3, layer_idx=selected_layer_idx)
        st.plotly_chart(fig_anim, use_container_width=True)


# ── Tab 3: Analytics & Material Estimator ─────────────────────────
with tab_analytics:
    st.markdown("### 📊 Concrete Material & Production Analytics")

    # Main Metrics Grid
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Concrete Volume", f"{metrics['volume_m3']} m³", help="Total wet concrete volume required")
    c2.metric("Concrete Mass", f"{metrics['mass_tonnes']} tonnes", f"{metrics['mass_kg']} kg")
    c3.metric("Print Time", metrics["time_str"], f"{metrics['print_time_sec']}s active")
    c4.metric("Extrusion Efficiency", f"{metrics['efficiency_pct']}%", f"{metrics['total_print_m']:.0f}m print / {metrics['total_travel_m']:.0f}m travel")

    st.markdown("<br>", unsafe_allow_html=True)

    col_table1, col_table2 = st.columns(2)
    with col_table1:
        st.markdown("#### 🧱 Reconstructed Zone Breakdown")
        zone_data = []
        for z in stage2.zones:
            zone_data.append({
                "Zone ID": z.zone_id,
                "Segments": z.n_segments,
                "Length (m)": round(z.total_length_mm / 1000.0, 2),
                "Loop Type": "Closed Loop" if z.is_closed_loop else "Open Trail",
                "Trails/Layer": z.n_trails,
            })
        st.dataframe(zone_data, use_container_width=True)

    with col_table2:
        st.markdown("#### 🥞 Layer Stack Breakdown")
        layer_data = []
        for l in stage3.layers:
            layer_data.append({
                "Layer #": l.index,
                "Z Height (mm)": round(l.z_mm, 1),
                "Print Traces": l.n_print_traces,
                "Print Length (m)": round(l.total_print_length_mm / 1000.0, 2),
                "Travel Length (m)": round(l.total_travel_length_mm / 1000.0, 2),
            })
        st.dataframe(layer_data, use_container_width=True)


# ── Tab 4: G-code Inspector ───────────────────────────────────────
with tab_gcode:
    st.markdown("### 📜 Continuous G-code Inspector & Syntax Viewer")
    
    col_search, col_dl = st.columns([3, 1])
    with col_search:
        search_query = st.text_input("🔍 Filter G-code commands", placeholder="e.g. G1, E, Layer 0, Z20.0")
    with col_dl:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇ Download G-code",
            data=gcode,
            file_name="printplan_output.gcode",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )

    gcode_lines = gcode.splitlines()
    if search_query.strip():
        filtered_lines = [l for l in gcode_lines if search_query.lower() in l.lower()]
        st.caption(f"Showing {len(filtered_lines)} of {len(gcode_lines)} lines matching '{search_query}'")
        st.code("\n".join(filtered_lines[:500]), language="gcode")
    else:
        st.caption(f"Showing first 500 lines of {len(gcode_lines)} total G-code lines")
        st.code("\n".join(gcode_lines[:500]), language="gcode")


# ── Tab 5: Technical Specs ────────────────────────────────────────
with tab_docs:
    st.markdown("### ℹ️ PrintPlan AI System Architecture & Technical Specifications")
    
    st.markdown("""
    #### Architectural Overview
    PrintPlan AI converts 2D architectural CAD floor plan drawings (vector PDFs) into optimized, collision-free, continuous toolpaths for large-scale **3D Concrete Printing (3DCP)**.

    ```
    ┌─────────────────────────┐
    │ 1. Vector PDF Ingest    │  M1: Parses CAD OCG layers into world mm coordinates
    └────────────┬────────────┘
                 ▼
    ┌─────────────────────────┐
    │ 2. Graph Reconstruction │  M2: Builds topological graph & Eulerian path matching
    └────────────┬────────────┘
                 ▼
    ┌─────────────────────────┐
    │ 3. 3D Toolpath Stack    │  M3: Extrudes continuous trails into N Z-height layers
    └────────────┬────────────┘
                 ▼
    ┌─────────────────────────┐
    │ 4. G-code Synthesis     │  M4: Generates continuous extrusion & nozzle velocity commands
    └─────────────────────────┘
    ```

    #### Key Algorithmic Innovations
    1. **Eulerian Graph Matching**: Closed loops are traversed continuously in 1 stroke without dry-travel retractions.
    2. **Continuous Extrusion Control**: Material feed rates ($E$) are proportional to dynamic Euclidean travel distances.
    3. **Spatial Layer Stacking**: Multi-layer parametric height propagation with customized bead width and print speed profiles.

    ---
    **Developer Note**: *Developed by Hoang Khieu (RMIT University) & CONTECH GLOBAL PTE. LTD.*
    """)
