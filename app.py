"""AUTO2DPRINT — Streamlit Interface.

Upload a vector PDF floor plan (or load built-in demo geometry) to generate
continuous 3D concrete printing toolpaths with interactive WebGL 3D visualization,
material estimation, and smart menu navigation.

v2.1 — Added Opening Detection (doors & windows) with Z-aware G-code suppression.
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
    build_3d_toolpath_webgl,
    build_3d_toolpath_fig,
    build_layer_2d_fig,
    build_3d_nozzle_anim,
    calc_material_metrics,
)

# ── Column Detection Module ───────────────────────────────────────
from column_detection import detect_columns, render_columns_tab

# ── Opening Detection Module ──────────────────────────────────────
from opening_detection import (
    ProjectConfig,
    FloorPlanParser,
    LayerScheduler,
    GCodeGenerator,
    LinteldType,
    OpeningType,
    PrintAction,
)


# ── Password Protection ───────────────────────────────────────────
def check_password() -> bool:
    """Simple password gate."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("<h1 style='text-align: center; color: #38bdf8;'>🏗️ AUTO2DPRINT </h1>", unsafe_allow_html=True)
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
    page_title="Automated Toolpath Platform of Large scale 3D Concrete Printing",
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
        scale      = st.number_input("Drawing scale (1:N)",  value=default_scale,   min_value=1,    step=1)
        n_layers   = st.number_input("Number of layers",      value=default_layers,  min_value=1,    step=1)
        layer_h    = st.number_input("Layer height (mm)",     value=default_layer_h, min_value=1.0,  step=1.0)
        bead_w     = st.number_input("Bead width (mm)",       value=default_bead_w,  min_value=5.0,  step=5.0)
        speed      = st.number_input("Print speed (mm/s)",    value=default_speed,   min_value=10.0, step=10.0)
        wall_regex = st.text_input(
            "Wall layer CAD pattern", value=".*wall.*",
            help="Regex pattern matching CAD wall layer name (e.g. 'Wall', '2D Walls', 'A-WALL')"
        )

    st.divider()

    st.markdown("### 📥 Input Source")
    input_mode = st.radio(
        "Choose Geometry Source",
        options=["⚡ Built-in Demo Floor Plan", "📂 Upload Vector PDF"],
        index=0,
    )

    pdf_file    = None
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
    st.caption("Auto2DPrint")
    st.caption("Hoang Khieu · RMIT University")


# ── Session State Initialisation ─────────────────────────────────
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False
if "openings_detected" not in st.session_state:
    st.session_state.openings_detected = False
if "pdf_path_for_openings" not in st.session_state:
    st.session_state.pdf_path_for_openings = None


# ── Pipeline Execution ───────────────────────────────────────────
if run_clicked or (not st.session_state.pipeline_run and input_mode == "⚡ Built-in Demo Floor Plan"):
    with st.spinner("Processing continuous toolpath pipeline..."):
        if input_mode == "📂 Upload Vector PDF":
            if pdf_file is not None:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_file.read())
                    pdf_path = tmp.name
                stage1 = parse_pdf(pdf_path, M1Params(drawing_scale=scale, wall_layer_pattern=wall_regex))
                # Store path for opening detection (don't delete yet)
                st.session_state.pdf_path_for_openings = pdf_path
                st.session_state.openings_detected = False  # reset on new upload
            else:
                st.warning("Please upload a vector PDF file first!")
                st.stop()
        else:
            stage1 = get_demo_stage1(preset_name=demo_preset)
            st.session_state.pdf_path_for_openings = None
            st.session_state.openings_detected = False

        stage2  = reconstruct(stage1, M2Params())
        stage3  = generate_toolpath(stage2, M3Params(layer_height_mm=layer_h), n_layers=int(n_layers))
        gcode   = synthesise_gcode(stage3, M4Params(bead_width_mm=bead_w, print_speed_mm_s=speed))
        metrics = calc_material_metrics(stage3, bead_width_mm=bead_w, layer_height_mm=layer_h, print_speed_mm_s=speed)

        st.session_state.stage1       = stage1
        st.session_state.stage2       = stage2
        st.session_state.stage3       = stage3
        st.session_state.gcode        = gcode
        st.session_state.metrics      = metrics
        st.session_state.pipeline_run = True


# ── Header ───────────────────────────────────────────────────────
status_text = "Toolpath Ready" if st.session_state.pipeline_run else "Waiting for Execution"
status_type = "success"        if st.session_state.pipeline_run else "info"
render_header(status_text=status_text, status_type=status_type)

if not st.session_state.pipeline_run:
    st.info("👆 Please select an input source or upload a vector PDF floor plan in the sidebar and click **▶ Run Full Pipeline** to begin.")
    st.stop()

# ── Retrieve active state ────────────────────────────────────────
stage1  = st.session_state.stage1
stage2  = st.session_state.stage2
stage3  = st.session_state.stage3
gcode   = st.session_state.gcode
metrics = st.session_state.metrics


# ── Tab Navigation ───────────────────────────────────────────────
tab_pipeline, tab_openings, tab_columns, tab_3d, tab_analytics, tab_gcode, tab_docs = st.tabs([
    "🚀 Pipeline & Overview",
    "🚪 Opening Detection",
    "🏛️ Column Details",              # ← NEW TAB
    "🧊 3D Visualizer & Simulation",
    "📊 Analytics & Concrete Estimator",
    "📜 G-code Inspector & Download",
    "ℹ️ Technical Specs & Architecture",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Pipeline & Overview
# ═══════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.markdown("### ⚡ Pipeline Execution Summary")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("M1 Ingestion",    f"{stage1.meta.get('segment_count', 0)} segs",  help="Extracted CAD segments")
    with col_m2:
        closed_cnt = sum(1 for z in stage2.zones if z.is_closed_loop)
        st.metric("M2 Reconstruction", f"{len(stage2.zones)} zones", f"{closed_cnt} closed loops")
    with col_m3:
        st.metric("M3 Toolpath",     f"{len(stage3.layers)} layers", f"{metrics['total_print_m']:.0f} m print")
    with col_m4:
        n_cmds = sum(1 for l in gcode.splitlines() if l.strip() and not l.startswith(";"))
        st.metric("M4 G-code",       f"{n_cmds} lines",              f"{len(gcode)/1024:.1f} KB")

    if "fallback_note" in stage1.meta:
        st.caption(
            f"ℹ️ **M1 Layer Ingestion Rule**: {stage1.meta['fallback_note']} "
            f"(Matched layers: `{', '.join(stage1.meta.get('matched_layers', []))}`)"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_preview, col_log = st.columns([3, 2])
    with col_preview:
        st.markdown("#### 📐 2D Floor Plan Toolpath Preview (Layer 0)")
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f172a")
        ax.set_facecolor("#0b1120")

        bbox     = stage3.coordinate_frame.page_bbox_world_mm
        x0, y0, x1, y1 = bbox
        ax.set_xlim(x0 - 200, x1 + 200)
        ax.set_ylim(y0 - 200, y1 + 200)
        ax.set_aspect("equal")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

        colors_list = ["#38bdf8","#f43f5e","#10b981","#f59e0b","#a855f7"]
        layer0      = stage3.layers[0]

        for tr in layer0.traces:
            pts  = tr.points
            segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
            if tr.kind == "print":
                c = colors_list[(tr.trail_id or 0) % len(colors_list)]
                ax.add_collection(LineCollection(segs, colors=[c], linewidths=2.5, zorder=3))
                ax.plot(pts[0][0], pts[0][1], "o", color=c, markersize=8, markeredgecolor="white", zorder=6)
            else:
                ax.add_collection(LineCollection(segs, colors=["#64748b"], linewidths=1.0, linestyles=":", zorder=2))

        ax.set_title(
            f"Layer 0 Continuous Nozzle Strokes: {layer0.n_print_traces} Trails",
            color="#f8fafc", fontweight="bold", fontsize=11
        )
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
            f"M3 · Toolpath: {stage3.meta.get('n_layers', 0)} layers, "
            f"print={stage3.meta.get('total_print_length_m', 0):.1f} m, "
            f"travel={stage3.meta.get('total_travel_length_m', 0):.1f} m\n"
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


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Opening Detection (NEW)
# ═══════════════════════════════════════════════════════════════════
with tab_openings:
    st.markdown("### 🚪 Opening Detection — Doors & Windows")
    st.caption(
        "Detects door and window openings from the uploaded floor plan PDF. "
        "Generates a Z-aware G-code with per-layer wall suppression at opening locations."
    )

    # ── Availability check ───────────────────────────────────────
    pdf_available = st.session_state.pdf_path_for_openings is not None

    if not pdf_available:
        st.info(
            "ℹ️ Opening detection requires an uploaded vector PDF floor plan. "
            "Demo geometry does not contain door/window symbols.\n\n"
            "Please upload a PDF in the sidebar and run the pipeline first."
        )
        st.stop()

    # ── Project Z Parameters ─────────────────────────────────────
    st.markdown("#### ⚙️ Project Parameters")
    st.caption("These values apply to all openings unless overridden per-opening below.")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        wall_height_mm    = st.number_input("Wall height (mm)",        value=2700, min_value=1000, max_value=6000, step=100)
        op_layer_h        = st.number_input("Layer height (mm)",       value=int(layer_h), min_value=10, max_value=200, step=5)
    with col_p2:
        door_head_mm      = st.number_input("Door head height (mm)",   value=2100, min_value=1500, max_value=3000, step=50)
        win_sill_mm       = st.number_input("Window sill height (mm)", value=900,  min_value=0,    max_value=2000, step=50)
        win_head_mm       = st.number_input("Window head height (mm)", value=2100, min_value=500,  max_value=3000, step=50)
    with col_p3:
        lintel_thick_mm   = st.number_input("Lintel thickness (mm)",   value=200,  min_value=50,   max_value=500,  step=25)
        lintel_type_sel   = st.selectbox("Lintel type", ["timber", "steel", "precast", "none"])
        pause_lintel      = st.checkbox("Pause G-code at lintel level", value=True)

    op_config = ProjectConfig(
        layer_height_mm        = float(op_layer_h),
        wall_height_mm         = float(wall_height_mm),
        lintel_thickness_mm    = float(lintel_thick_mm),
        lintel_type            = LinteldType(lintel_type_sel),
        pause_for_lintel       = pause_lintel,
        door_head_height_mm    = float(door_head_mm),
        window_sill_height_mm  = float(win_sill_mm),
        window_head_height_mm  = float(win_head_mm),
        drawing_scale          = int(scale),
    )

    # ── Run Detection ────────────────────────────────────────────
    st.markdown("#### 🔍 Detection Options")
    include_windows = st.checkbox(
        "Include unmatched gaps as windows",
        value=False,
        help=(
            "OFF (default): Only gaps confirmed by a door swing arc are detected — use this "
            "when the drawing has doors only (no window symbols).\n\n"
            "ON: Remaining unmatched gaps are also classified as windows — use this "
            "when the drawing contains explicit window triple-line symbols."
        )
    )

    detect_clicked = st.button("🔍 Detect Openings from PDF", type="primary")

    if detect_clicked:
        with st.spinner("Scanning floor plan for doors and windows..."):
            try:
                parser   = FloorPlanParser(st.session_state.pdf_path_for_openings, op_config)
                openings = parser.detect_openings(include_unmatched_as_windows=include_windows)
                st.session_state.openings          = openings
                st.session_state.op_config         = op_config
                st.session_state.openings_detected = True
            except Exception as e:
                st.error(f"Detection failed: {e}")
                st.session_state.openings_detected = False

    # ── Show Results ─────────────────────────────────────────────
    if st.session_state.openings_detected:
        openings  = st.session_state.openings
        op_config = st.session_state.op_config

        if not openings:
            st.warning(
                "No door or window-sized openings detected. "
                "Check that your PDF has a named 'Doors' or 'Wall' layer with standard symbols."
            )
        else:
            # ── Summary metrics ──────────────────────────────────
            n_doors   = sum(1 for o in openings if o.opening_type == OpeningType.DOOR)
            n_windows = sum(1 for o in openings if o.opening_type == OpeningType.WINDOW)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total Openings", len(openings))
            mc2.metric("Doors Detected", n_doors)
            mc3.metric("Windows Detected", n_windows)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Per-opening override table ───────────────────────
            st.markdown("#### 📋 Detected Openings — Review & Override")
            st.caption(
                "Z parameters are pre-filled from project defaults above. "
                "Expand any row to override values for that specific opening."
            )

            for op in openings:
                icon = "🚪" if op.opening_type == OpeningType.DOOR else "🪟"
                with st.expander(
                    f"{icon} {op.opening_id} — {op.opening_type.value.title()} | "
                    f"Width: {op.width_mm:.0f} mm | Wall: {op.wall_id} ({op.wall_direction})"
                ):
                    oc1, oc2, oc3 = st.columns(3)
                    with oc1:
                        if op.opening_type == OpeningType.DOOR:
                            st.text_input(
                                "Sill height (mm)", value="0 (door — fixed)",
                                disabled=True, key=f"sill_{op.opening_id}"
                            )
                        else:
                            new_sill = st.number_input(
                                "Sill height (mm)", value=int(op.z_void_bottom_mm),
                                min_value=0, max_value=2000, step=50,
                                key=f"sill_{op.opening_id}"
                            )
                            op.z_void_bottom_mm = float(new_sill)
                    with oc2:
                        new_head = st.number_input(
                            "Head height (mm)", value=int(op.z_void_top_mm),
                            min_value=500, max_value=3000, step=50,
                            key=f"head_{op.opening_id}"
                        )
                        op.z_void_top_mm = float(new_head)
                    with oc3:
                        new_lint = st.number_input(
                            "Lintel thickness (mm)", value=int(op.lintel_thickness_mm),
                            min_value=0, max_value=500, step=25,
                            key=f"lint_{op.opening_id}"
                        )
                        op.lintel_thickness_mm = float(new_lint)

                    # Show Z-zone summary for this opening
                    st.info(
                        f"**Z-Zone Summary** — "
                        f"Void: {op.z_void_bottom_mm:.0f} → {op.z_void_top_mm:.0f} mm | "
                        f"Lintel: {op.z_void_top_mm:.0f} → {op.z_resume_mm:.0f} mm | "
                        f"Full wall resumes at: {op.z_resume_mm:.0f} mm"
                    )

            # ── Layer Schedule Preview ───────────────────────────
            st.markdown("#### 🥞 Layer Suppression Schedule")
            st.caption("Shows which layers are affected by openings and what print action applies.")

            scheduler = LayerScheduler(openings, op_config)
            schedule  = scheduler.generate_layer_schedule()

            suppressed_layers = [l for l in schedule if l["actions"]]
            if suppressed_layers:
                preview_data = []
                for layer in suppressed_layers:
                    for oid, action in layer["actions"].items():
                        preview_data.append({
                            "Layer #"   : layer["layer_index"],
                            "Z (mm)"    : layer["z_mm"],
                            "Opening"   : oid,
                            "Action"    : action.value,
                        })
                st.dataframe(preview_data, use_container_width=True, height=300)
            else:
                st.info("No layers suppressed — check opening Z parameters.")

            # ── Z-Aware G-code Generation ────────────────────────
            st.markdown("#### 📜 Z-Aware G-code with Opening Suppression")

            gen_gcode_clicked = st.button("⚙️ Generate Z-Aware G-code", type="secondary")

            if gen_gcode_clicked or "opening_gcode" in st.session_state:
                if gen_gcode_clicked:
                    generator = GCodeGenerator(op_config)
                    opening_gcode = generator.generate_full_schedule_gcode(schedule, openings)
                    st.session_state.opening_gcode = opening_gcode

                opening_gcode = st.session_state.opening_gcode

                col_dl, col_info = st.columns([2, 3])
                with col_dl:
                    st.download_button(
                        label="⬇ Download Z-Aware G-code",
                        data=opening_gcode,
                        file_name="printplan_openings.gcode",
                        mime="text/plain",
                        type="primary",
                        use_container_width=True,
                    )
                with col_info:
                    n_lines       = len(opening_gcode.splitlines())
                    n_pauses      = opening_gcode.count("M0")
                    n_suppressed  = sum(1 for l in schedule if l["has_suppression"])
                    st.metric("G-code Lines",       n_lines)
                    st.metric("Lintel Pause Events", n_pauses)
                    st.metric("Suppressed Layers",   n_suppressed)

                search_op = st.text_input(
                    "🔍 Filter G-code", placeholder="e.g. D01, PAUSE, pier, Z=900",
                    key="opening_gcode_search"
                )
                gcode_lines_op = opening_gcode.splitlines()
                if search_op.strip():
                    filtered = [l for l in gcode_lines_op if search_op.lower() in l.lower()]
                    st.caption(f"Showing {len(filtered)} of {n_lines} lines matching '{search_op}'")
                    st.code("\n".join(filtered[:300]), language="gcode")
                else:
                    st.caption(f"Showing first 300 of {n_lines} lines")
                    st.code("\n".join(gcode_lines_op[:300]), language="gcode")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — Column Details
# ═══════════════════════════════════════════════════════════════════
with tab_columns:
    render_columns_tab(
        pdf_path=st.session_state.pdf_path_for_openings,
        drawing_scale=int(scale),
    )


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — 3D Visualizer & Simulation
# ═══════════════════════════════════════════════════════════════════
with tab_3d:
    st.markdown("### 🧊 WebGL Interactive 3D Toolpath & Structure Model")

    # ── Controls toolbar ─────────────────────────────────────────
    c_mode, c_color, c_range, c_travel = st.columns([2, 2, 3, 2])
    with c_mode:
        render_mode = st.selectbox(
            "3D Render Mode",
            options=["Extruded Concrete Beads", "Wireframe Toolpath"],
            index=0,
        )
    with c_color:
        color_mode = st.selectbox(
            "3D Color Scheme",
            options=["Layer", "Trail ID", "Print vs Travel"],
            index=0,
        )
    with c_range:
        max_l       = len(stage3.layers) - 1
        layer_range = st.slider(
            "Inspect Layer Height Range",
            min_value=0, max_value=max_l, value=(0, max_l),
        )
    with c_travel:
        st.markdown("<br>", unsafe_allow_html=True)
        show_travel = st.checkbox("Show Nozzle Travel Moves", value=True)

    # ── 3D Render decision ───────────────────────────────────────
    # Pass detected openings into visualizer if available
    active_openings = st.session_state.get("openings", []) if st.session_state.get("openings_detected") else []
    if active_openings:
        st.info(
            f"\U0001f6aa Opening suppression active \u2014 {len(active_openings)} opening(s) applied to 3D geometry. "
            "Door/window voids are removed from the rendered walls."
        )

    if render_mode == "Extruded Concrete Beads":
        build_3d_toolpath_webgl(
            stage3,
            color_mode=color_mode,
            render_mode=render_mode,
            show_travel=show_travel,
            layer_range=layer_range,
            bead_width_mm=bead_w,
            layer_height_mm=layer_h,
            height_px=700,
            openings=active_openings,
            drawing_scale=int(scale),
        )
        st.caption(
            "\U0001f5b1\ufe0f **Drag** to orbit \xb7 **Scroll** to zoom \xb7 **Right-drag** to pan &nbsp;|&nbsp; "
            "Rendered with Three.js WebGL \u2014 elliptical bead cross-sections, PBR lighting & shadow maps."
        )
    else:
        fig_3d = build_3d_toolpath_fig(
            stage3,
            color_mode=color_mode,
            render_mode=render_mode,
            show_travel=show_travel,
            layer_range=layer_range,
            bead_width_mm=bead_w,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    st.divider()

    # ── Sub-tabs ─────────────────────────────────────────────────
    subtab_2d, subtab_anim = st.tabs([
        "📐 High-Res 2D Single Layer Inspection",
        "🎬 3D Robotic Nozzle Trajectory Playback",
    ])

    with subtab_2d:
        col_ctrl, col_fig = st.columns([1, 3])
        with col_ctrl:
            selected_layer_idx = st.number_input(
                "Select Layer Index", min_value=0, max_value=max_l, value=0, step=1
            )
            layer_obj = stage3.layers[selected_layer_idx]
            st.metric("Layer Z Height",  f"{layer_obj.z_mm:.1f} mm")
            st.metric("Print Length",    f"{layer_obj.total_print_length_mm / 1000.0:.2f} m")
            st.metric("Print Strokes",   f"{layer_obj.n_print_traces}")
        with col_fig:
            fig_2d = build_layer_2d_fig(
                layer_obj, selected_layer_idx,
                stage3.coordinate_frame.page_bbox_world_mm
            )
            st.plotly_chart(fig_2d, use_container_width=True)

    with subtab_anim:
        col_ctrl_anim, col_fig_anim = st.columns([1, 3])
        with col_ctrl_anim:
            selected_anim_layer = st.number_input(
                "Simulation Layer Index",
                min_value=0, max_value=max_l, value=0, step=1, key="anim_layer"
            )
            st.caption(
                "Click ▶ Play inside the 3D canvas toolbar to animate the "
                "robotic nozzle extruding concrete live in 3D space."
            )
        with col_fig_anim:
            fig_anim = build_3d_nozzle_anim(
                stage3,
                layer_idx=selected_anim_layer,
                bbox=stage3.coordinate_frame.page_bbox_world_mm,
            )
            st.plotly_chart(fig_anim, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — Analytics & Material Estimator
# ═══════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("### 📊 Concrete Material & Production Analytics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Concrete Volume",     f"{metrics['volume_m3']} m³",        help="Total wet concrete volume required")
    c2.metric("Concrete Mass",       f"{metrics['mass_tonnes']} tonnes",   f"{metrics['mass_kg']} kg")
    c3.metric("Print Time",          metrics["time_str"],                  f"{metrics['print_time_sec']}s active")
    c4.metric("Extrusion Efficiency",f"{metrics['efficiency_pct']}%",
              f"{metrics['total_print_m']:.0f}m print / {metrics['total_travel_m']:.0f}m travel")

    st.markdown("<br>", unsafe_allow_html=True)

    col_table1, col_table2 = st.columns(2)
    with col_table1:
        st.markdown("#### 🧱 Reconstructed Zone Breakdown")
        zone_data = [
            {
                "Zone ID":      z.zone_id,
                "Segments":     z.n_segments,
                "Length (m)":   round(z.total_length_mm / 1000.0, 2),
                "Loop Type":    "Closed Loop" if z.is_closed_loop else "Open Trail",
                "Trails/Layer": z.n_trails,
            }
            for z in stage2.zones
        ]
        st.dataframe(zone_data, use_container_width=True)

    with col_table2:
        st.markdown("#### 🥞 Layer Stack Breakdown")
        layer_data = [
            {
                "Layer #":         l.index,
                "Z Height (mm)":   round(l.z_mm, 1),
                "Print Traces":    l.n_print_traces,
                "Print Length (m)":  round(l.total_print_length_mm  / 1000.0, 2),
                "Travel Length (m)": round(l.total_travel_length_mm / 1000.0, 2),
            }
            for l in stage3.layers
        ]
        st.dataframe(layer_data, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 5 — G-code Inspector
# ═══════════════════════════════════════════════════════════════════
with tab_gcode:
    st.markdown("### 📜 Continuous G-code Inspector & Syntax Viewer")

    col_search, col_dl = st.columns([3, 1])
    with col_search:
        search_query = st.text_input(
            "🔍 Filter G-code commands",
            placeholder="e.g. G1, E, Layer 0, Z20.0"
        )
    with col_dl:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇ Download G-code",
            data=gcode,
            file_name="printplan_output.gcode",
            mime="text/plain",
            type="primary",
            use_container_width=True,
        )

    gcode_lines = gcode.splitlines()
    if search_query.strip():
        filtered_lines = [l for l in gcode_lines if search_query.lower() in l.lower()]
        st.caption(f"Showing {len(filtered_lines)} of {len(gcode_lines)} lines matching '{search_query}'")
        st.code("\n".join(filtered_lines[:500]), language="gcode")
    else:
        st.caption(f"Showing first 500 lines of {len(gcode_lines)} total G-code lines")
        st.code("\n".join(gcode_lines[:500]), language="gcode")


# ═══════════════════════════════════════════════════════════════════
# TAB 6 — Technical Specs & Architecture
# ═══════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("### ℹ️ PrintPlan AI System Architecture & Technical Specifications")

    st.markdown("""
    #### Architectural Overview
    PrintPlan AI converts 2D architectural CAD floor plan drawings (vector PDFs) into optimized,
    collision-free, continuous toolpaths for large-scale **3D Concrete Printing (3DCP)**.

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
    └────────────┬────────────┘
                 ▼
    ┌─────────────────────────┐
    │ 5. Opening Detection    │  Detects doors/windows → Z-aware suppression & lintel pauses
    └─────────────────────────┘
    ```

    #### Key Algorithmic Innovations
    1. **Eulerian Graph Matching**: Closed loops are traversed continuously in 1 stroke without dry-travel retractions.
    2. **Continuous Extrusion Control**: Material feed rates ($E$) are proportional to dynamic Euclidean travel distances.
    3. **Spatial Layer Stacking**: Multi-layer parametric height propagation with customised bead width and print speed profiles.
    4. **Z-Aware Opening Detection**: Gap + arc matching identifies doors/windows and suppresses wall contours per-layer across 4 Z-zones:
       - Zone 1 (below void): full wall printed
       - Zone 2 (void): left & right piers only
       - Zone 3 (lintel seat): M0 pause for lintel installation
       - Zone 4 (above lintel): full wall resumes

    #### Opening Detection — Input Requirements
    | Document | Provides |
    |---|---|
    | Floor plan PDF (vector) | X-Y positions, widths, door/window type |
    | Project parameters (UI) | Head height, sill height, lintel thickness |

    #### 3D Visualiser Engine (v2.0)
    The **Extruded Concrete Beads** render mode uses a Three.js WebGL engine with:
    - **ExtrudeGeometry** with elliptical cross-section matching real 3DCP bead geometry
    - **MeshStandardMaterial** with PBR roughness/metalness model
    - **PCFSoftShadowMap** for realistic shadow casting between layers
    - **ACES Filmic** tone mapping for photorealistic output
    - **OrbitControls** with damping for smooth camera interaction

    ---
    **Developer Note**: *Developed by Hoang Khieu — RMIT University*
    """)
