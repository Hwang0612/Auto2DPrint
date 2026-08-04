"""PrintPlan AI — Streamlit interface.

Upload a vector PDF floor plan → get continuous G-code toolpath
for large-scale 3D concrete printing.
"""

import streamlit as st
import tempfile, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from printplan_ai.config import M1Params, M2Params, M3Params, M4Params
from printplan_ai.stages import parse_pdf, reconstruct, generate_toolpath, synthesise_gcode


# ── Password protection ──
def check_password():
    """Simple password gate."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🏗️ PrintPlan AI")
    st.caption("Please enter the access password to continue.")
    password = st.text_input("Password", type="password")
    if st.button("Login", type="primary"):
        # Password is stored in Streamlit secrets or defaults to this
        correct = st.secrets.get("password", "printplan2025") if hasattr(st, "secrets") else "printplan2025"
        try:
            correct = st.secrets["password"]
        except Exception:
            correct = "printplan2025"
        if password == correct:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


st.set_page_config(
    page_title="PrintPlan AI",
    page_icon="🏗️",
    layout="wide",
)

if not check_password():
    st.stop()

# ── Header ──
st.title("🏗️ PrintPlan AI")
st.caption("PDF floor plan → continuous G-code toolpath for large-scale 3D concrete printing")

# ── Sidebar: parameters ──
with st.sidebar:
    st.header("Parameters")
    scale = st.number_input("Drawing scale (1:N)", value=50, min_value=1, step=1)
    n_layers = st.number_input("Number of layers", value=20, min_value=1, step=1)
    layer_h = st.number_input("Layer height (mm)", value=20.0, min_value=1.0, step=1.0)
    bead_w = st.number_input("Bead width (mm)", value=40.0, min_value=5.0, step=5.0)
    speed = st.number_input("Print speed (mm/s)", value=60.0, min_value=10.0, step=10.0)

    st.divider()
    st.caption("PrintPlan AI v1.1")
    st.caption("Hoang Khieu · RMIT University")
    st.caption("CONTECH GLOBAL PTE. LTD.")

# ── Main area ──
col_upload, col_result = st.columns([1, 2])

with col_upload:
    st.subheader("Upload")
    pdf_file = st.file_uploader(
        "Drop your vector PDF floor plan here",
        type=["pdf"],
        help="CAD-exported PDF with wall layer preserved as OCG"
    )

    run_clicked = st.button("▶ Run Pipeline", type="primary", use_container_width=True,
                            disabled=(pdf_file is None))

with col_result:
    result_container = st.container()

# ── Pipeline ──
if run_clicked and pdf_file is not None:
    # Save uploaded PDF to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name

    with result_container:
        progress = st.progress(0, text="Running M1 — parsing PDF...")

        # M1
        stage1 = parse_pdf(pdf_path, M1Params(drawing_scale=scale))
        m = stage1.meta
        progress.progress(25, text="Running M2 — detecting zones...")

        # M2
        stage2 = reconstruct(stage1, M2Params())
        closed = sum(1 for z in stage2.zones if z.is_closed_loop)
        progress.progress(50, text="Running M3 — generating toolpath...")

        # M3
        stage3 = generate_toolpath(stage2, M3Params(layer_height_mm=layer_h), n_layers=int(n_layers))
        progress.progress(75, text="Running M4 — synthesising G-code...")

        # M4
        gcode = synthesise_gcode(stage3, M4Params(bead_width_mm=bead_w, print_speed_mm_s=speed))
        n_cmds = sum(1 for l in gcode.splitlines() if l.strip() and not l.startswith(";"))
        progress.progress(100, text="Complete!")

        # ── Results ──
        st.subheader("Pipeline Results")

        # Stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Segments", m["segment_count"])
        c2.metric("Zones", f"{len(stage2.zones)} ({closed} closed)")
        c3.metric("Trails/layer", stage2.meta["total_trails"])
        c4.metric("G-code lines", n_cmds)

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Layers", stage3.meta["n_layers"])
        c6.metric("Print length", f"{stage3.meta['total_print_length_m']:.0f} m")
        c7.metric("Travel length", f"{stage3.meta['total_travel_length_m']:.0f} m")
        c8.metric("G-code size", f"{len(gcode)/1024:.0f} KB")

        # Toolpath plot
        st.subheader("Toolpath Preview (Layer 0)")
        fig, ax = plt.subplots(figsize=(12, 8))
        bbox = stage3.coordinate_frame.page_bbox_world_mm
        x0, y0, x1, y1 = bbox
        ax.set_xlim(x0 - 200, x1 + 200)
        ax.set_ylim(y0 - 200, y1 + 200)
        ax.set_aspect("equal")

        colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
        layer = stage3.layers[0]

        for tr in layer.traces:
            pts = tr.points
            segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
            if tr.kind == "print":
                color = colors[tr.trail_id % len(colors)]
                ax.add_collection(LineCollection(segs, colors=[color], linewidths=3.0, zorder=3))
                ax.plot(pts[0][0], pts[0][1], "o", color=color, markersize=10,
                        markeredgecolor="white", markeredgewidth=2, zorder=6)
                if not tr.is_closed:
                    ax.plot(pts[-1][0], pts[-1][1], "s", color=color, markersize=9,
                            markeredgecolor="white", markeredgewidth=2, zorder=6)
                label = "Circuit" if tr.is_closed else "Trail"
                mid = pts[len(pts) // 2]
                ax.annotate(f"{label} T{tr.trail_id}\n{tr.length_mm / 1000:.1f} m",
                           mid, fontsize=9, fontweight="bold", color=color, ha="center",
                           bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2),
                           zorder=7)
            else:
                ax.add_collection(LineCollection(segs, colors=["#d1d5db"],
                                                 linewidths=1.0, linestyles=":", zorder=2))

        ax.set_title(f"{layer.n_print_traces} continuous nozzle strokes · "
                     f"print={layer.total_print_length_mm / 1000:.1f} m · "
                     f"travel={layer.total_travel_length_mm / 1000:.1f} m",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Log
        st.subheader("Pipeline Log")
        st.code(
            f"M1 · Ingest:   {m['segment_count']} segments, {m['total_length_m']:.1f} m\n"
            f"M2 · Zones:    {len(stage2.zones)} zones ({closed} closed), "
            f"{stage2.meta['total_trails']} trails/layer\n"
            f"M3 · Toolpath: {stage3.meta['n_layers']} layers, "
            f"print={stage3.meta['total_print_length_m']:.1f} m, "
            f"travel={stage3.meta['total_travel_length_m']:.1f} m\n"
            f"M4 · G-code:   {n_cmds} commands, {len(gcode)/1024:.1f} KB",
            language=None
        )

        # Download
        st.download_button(
            label="⬇ Download G-code",
            data=gcode,
            file_name="printplan_output.gcode",
            mime="text/plain",
            type="primary",
            use_container_width=True,
        )

    # Clean up
    Path(pdf_path).unlink(missing_ok=True)
