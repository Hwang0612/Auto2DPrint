"""
column_detection.py
===================
Auto2DPrint — Column Detection Module
--------------------------------------
Reads the 'Columns' OCG layer from a vector PDF floor plan and extracts
column positions, dimensions, and type labels in world-mm coordinates.

Each column in the drawing consists of:
  - 4 lines forming a square outline (the column box)
  - 1 filled black circle dot (centroid cross-check marker)

Column type (e.g. "C1 (250×250)") is read from the PDF text layer.
All columns sharing the same section dimensions are assigned the same type.

Author: Auto2DPrint Pipeline
"""

from __future__ import annotations
import math, re
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import fitz   # PyMuPDF


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ColumnType:
    """A column section type parsed from drawing annotations."""
    type_id: str          # e.g. "C1"
    width_mm: float       # section width  (mm)
    depth_mm: float       # section depth  (mm)

    @property
    def section_label(self) -> str:
        return f"{self.width_mm:.0f}×{self.depth_mm:.0f} mm"

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.depth_mm


@dataclass
class Column:
    """A single detected column instance in the floor plan."""
    col_id: str           # e.g. "C01"
    cx: float             # centre X in world mm
    cy: float             # centre Y in world mm
    width: float          # measured width  (mm) from outline box
    height: float         # measured depth  (mm) from outline box
    x0: float             # lower-left X  (world mm)
    y0: float             # lower-left Y  (world mm)
    x1: float             # upper-right X (world mm)
    y1: float             # upper-right Y (world mm)
    col_type: str = "C1"  # type label from drawing

    @property
    def section_label(self) -> str:
        return f"{self.width:.0f}×{self.height:.0f} mm"

    @property
    def area_mm2(self) -> float:
        return self.width * self.height

    @property
    def area_m2(self) -> float:
        return self.area_mm2 / 1_000_000

    def summary(self) -> str:
        return (
            f"{self.col_id} [{self.col_type}]  "
            f"centre=({self.cx:.0f},{self.cy:.0f})mm  "
            f"section={self.section_label}  area={self.area_mm2:.0f}mm²"
        )


# ---------------------------------------------------------------------------
# Type parser — reads "C1 (250x250)" annotations from PDF text
# ---------------------------------------------------------------------------

# Matches patterns like:  C1 (250x250)  C2(300x300)  C1(250X250)
_TYPE_RE = re.compile(r'(C\d+)\s*\(?\s*(\d+)\s*[xX×]\s*(\d+)\s*\)?')


def parse_column_types(pdf_path: str) -> Dict[str, ColumnType]:
    """
    Scan all text on the page for column type definitions.
    Returns dict keyed by type_id, e.g. {"C1": ColumnType("C1", 250, 250)}.
    """
    doc  = fitz.open(pdf_path)
    page = doc[0]
    text = page.get_text("text")
    types: Dict[str, ColumnType] = {}
    for m in _TYPE_RE.finditer(text):
        tid = m.group(1)            # e.g. "C1"
        w   = float(m.group(2))
        d   = float(m.group(3))
        if tid not in types:
            types[tid] = ColumnType(type_id=tid, width_mm=w, depth_mm=d)
    return types


# ---------------------------------------------------------------------------
# Column detector
# ---------------------------------------------------------------------------

CLUSTER_DIST_MM = 350.0   # lines within this distance belong to one column


def detect_columns(pdf_path: str, drawing_scale: int = 100) -> List[Column]:
    """
    Parse the 'Columns' OCG layer and return a sorted list of Column objects.

    Args:
        pdf_path:       path to the vector PDF floor plan
        drawing_scale:  1:N scale factor (default 100 for 1:100 drawing)

    Returns:
        List of Column objects sorted top-to-bottom, left-to-right.
    """
    k      = (25.4 / 72) * drawing_scale
    doc    = fitz.open(pdf_path)
    page   = doc[0]
    page_h = page.rect.height
    paths  = page.get_drawings()

    def to_world(x_pt: float, y_pt: float):
        return x_pt * k, (page_h - y_pt) * k

    col_paths = [p for p in paths if p.get("layer") == "Columns"]
    if not col_paths:
        return []

    # ── Parse column type definitions from text ──────────────────────────
    col_types = parse_column_types(pdf_path)
    # Determine which type label applies to this drawing
    # (typically one type per floor plan; use first found)
    default_type = list(col_types.keys())[0] if col_types else "C1"

    # ── Separate filled dots from outline line paths ──────────────────────
    filled_dots   = [p for p in col_paths if p.get("fill") == (0.0, 0.0, 0.0)]
    outline_lines = [p for p in col_paths if p.get("fill") is None
                     and any(it[0] == "l" for it in p.get("items", []))]

    # ── Collect midpoints of each outline line in world mm ────────────────
    line_data: List[tuple] = []   # (mid_x, mid_y, path_obj)
    for p in outline_lines:
        for it in p.get("items", []):
            if it[0] == "l":
                wx1, wy1 = to_world(it[1].x, it[1].y)
                wx2, wy2 = to_world(it[2].x, it[2].y)
                line_data.append(((wx1+wx2)/2, (wy1+wy2)/2, p))

    # ── Cluster lines by proximity → one cluster = one column ────────────
    used     = [False] * len(line_data)
    clusters: List[List[int]] = []
    for i, (mx, my, _) in enumerate(line_data):
        if used[i]: continue
        group = [i]; used[i] = True
        for j, (nx, ny, _) in enumerate(line_data):
            if used[j]: continue
            if math.hypot(mx-nx, my-ny) < CLUSTER_DIST_MM:
                group.append(j); used[j] = True
        clusters.append(group)

    # ── Build Column objects from each cluster ────────────────────────────
    columns: List[Column] = []
    for ci, idx_group in enumerate(clusters):
        all_pts: List[tuple] = []
        for idx in idx_group:
            p = line_data[idx][2]
            for it in p.get("items", []):
                if it[0] == "l":
                    all_pts.append(to_world(it[1].x, it[1].y))
                    all_pts.append(to_world(it[2].x, it[2].y))

        xs = [pt[0] for pt in all_pts]
        ys = [pt[1] for pt in all_pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)

        # Determine type: match measured dimensions to parsed types
        w = x1 - x0; h = y1 - y0
        matched_type = default_type
        for tid, ct in col_types.items():
            if (abs(w - ct.width_mm) < 20 and abs(h - ct.depth_mm) < 20):
                matched_type = tid
                break

        columns.append(Column(
            col_id   = f"C{ci+1:02d}",
            cx=(x0+x1)/2, cy=(y0+y1)/2,
            width=w, height=h,
            x0=x0, y0=y0, x1=x1, y1=y1,
            col_type = matched_type,
        ))

    # ── Sort: top → bottom, left → right; then re-number ─────────────────
    columns.sort(key=lambda c: (-round(c.cy, -2), c.cx))
    for i, col in enumerate(columns):
        col.col_id = f"C{i+1:02d}"

    # ── Cross-check against filled dot centres ────────────────────────────
    dot_centres: List[tuple] = []
    for p in filled_dots:
        r = p.get("rect")
        if r:
            wx0, wy0 = to_world(r.x0, r.y0)
            wx1, wy1 = to_world(r.x1, r.y1)
            dot_centres.append(((wx0+wx1)/2, (wy0+wy1)/2))

    for col in columns:
        matched = any(
            math.hypot(col.cx-dx, col.cy-dy) < 150
            for dx, dy in dot_centres
        )
        if not matched:
            col.col_type += " (?)"   # flag for user review

    return columns


# ---------------------------------------------------------------------------
# Streamlit tab renderer
# ---------------------------------------------------------------------------

def render_columns_tab(
    pdf_path: Optional[str],
    drawing_scale: int = 100,
) -> None:
    """
    Render the '🏛️ Column Details' Streamlit tab.
    Call this inside  `with tab_columns:`  in app.py.
    """
    import streamlit as st
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    st.markdown("### 🏛️ Column Detection — Position & Section Schedule")
    st.caption(
        "Reads the **Columns** OCG layer directly from the PDF. "
        "Column type labels (e.g. C1 250×250 mm) are parsed from drawing annotations."
    )

    if pdf_path is None:
        st.info(
            "ℹ️ Column detection requires an uploaded vector PDF. "
            "Please upload a PDF and run the pipeline first."
        )
        return

    with st.spinner("Scanning Columns layer..."):
        columns   = detect_columns(pdf_path, drawing_scale=drawing_scale)
        col_types = parse_column_types(pdf_path)

    if not columns:
        st.warning("No columns detected. Check that your PDF has a 'Columns' OCG layer.")
        return

    # ── Column type legend ────────────────────────────────────────────────
    if col_types:
        st.markdown("#### 🏷️ Column Type Legend")
        type_cols = st.columns(min(len(col_types), 4))
        for i, (tid, ct) in enumerate(col_types.items()):
            type_cols[i % 4].info(
                f"**{ct.type_id}**  \n"
                f"Section: {ct.section_label}  \n"
                f"Area: {ct.area_mm2:.0f} mm²"
            )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary metrics ───────────────────────────────────────────────────
    n_cols   = len(columns)
    n_types  = len(col_types)
    total_a  = sum(c.area_m2 for c in columns)
    # Group by type for count
    type_counts = {}
    for c in columns:
        type_counts[c.col_type] = type_counts.get(c.col_type, 0) + 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Columns",    n_cols)
    m2.metric("Column Types",     n_types)
    m3.metric("Most Common Type", max(type_counts, key=type_counts.get))
    m4.metric("Total XS Area",    f"{total_a:.4f} m²")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Layout: plan + schedule ───────────────────────────────────────────
    col_plot, col_table = st.columns([3, 2])

    with col_plot:
        st.markdown("#### 📐 Column Layout Plan")
        fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0f172a")
        ax.set_facecolor("#0b1120")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.set_aspect("equal")
        ax.grid(True, color="#334155", alpha=0.3, zorder=0)

        all_x = [c.cx for c in columns]; all_y = [c.cy for c in columns]
        pad = 2500
        ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
        ax.set_ylim(min(all_y)-pad, max(all_y)+pad)

        # Type → colour mapping
        TYPE_PALETTE = ["#f59e0b", "#38bdf8", "#10b981", "#f43f5e", "#a855f7"]
        type_list    = list(col_types.keys()) or ["C1"]
        type_colour  = {t: TYPE_PALETTE[i % len(TYPE_PALETTE)]
                        for i, t in enumerate(type_list)}

        for col in columns:
            colour = type_colour.get(col.col_type.replace(" (?)", ""), "#f59e0b")
            # Column box (scaled up ×8 for visibility at building scale)
            vis_w = max(col.width * 4, 500); vis_h = max(col.height * 4, 500)
            rect  = patches.Rectangle(
                (col.cx - vis_w/2, col.cy - vis_h/2), vis_w, vis_h,
                linewidth=1.5, edgecolor=colour,
                facecolor=colour + "33", zorder=3
            )
            ax.add_patch(rect)
            # Centre dot
            ax.plot(col.cx, col.cy, "o", color=colour,
                    markersize=5, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=5)
            # Label: col_id + type
            ax.annotate(
                f"{col.col_id}\n{col.col_type}",
                (col.cx, col.cy),
                color="white", fontsize=6.5, ha="center", va="center",
                fontweight="bold", zorder=6,
                bbox=dict(facecolor="#0b1120", alpha=0.75,
                          edgecolor="none", pad=1.5)
            )

        ax.set_xlabel("X (mm)", color="#94a3b8")
        ax.set_ylabel("Y (mm)", color="#94a3b8")
        ax.set_title(
            f"{n_cols} Columns — {', '.join(f'{v}×{k}' for k,v in type_counts.items())}",
            color="#f8fafc", fontweight="bold", fontsize=11
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_table:
        st.markdown("#### 📋 Column Schedule")
        table_data = [
            {
                "ID":            c.col_id,
                "Type":          c.col_type,
                "Section":       c.section_label,
                "Centre X (mm)": f"{c.cx:.0f}",
                "Centre Y (mm)": f"{c.cy:.0f}",
                "Width (mm)":    f"{c.width:.0f}",
                "Depth (mm)":    f"{c.height:.0f}",
                "Area (mm²)":    f"{c.area_mm2:.0f}",
            }
            for c in columns
        ]
        st.dataframe(table_data, use_container_width=True, height=440)

        import io, csv
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(table_data[0].keys()))
        writer.writeheader(); writer.writerows(table_data)
        st.download_button(
            "⬇ Download Column Schedule (CSV)",
            data=buf.getvalue(),
            file_name="column_schedule.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Per-column expanders ──────────────────────────────────────────────
    st.markdown("#### 🔍 Individual Column Details")
    for col in columns:
        with st.expander(
            f"🏛 {col.col_id} [{col.col_type}] — {col.section_label}  |  "
            f"Centre ({col.cx:.0f}, {col.cy:.0f}) mm"
        ):
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Type",       col.col_type)
            dc1.metric("Section",    col.section_label)
            dc2.metric("Centre X",   f"{col.cx:.0f} mm")
            dc2.metric("Centre Y",   f"{col.cy:.0f} mm")
            dc3.metric("XS Area",    f"{col.area_mm2:.0f} mm²")
            dc3.metric("XS Area",    f"{col.area_m2:.6f} m²")
            st.caption(
                f"Bounding box: ({col.x0:.0f}, {col.y0:.0f}) → "
                f"({col.x1:.0f}, {col.y1:.0f}) mm"
            )


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    pdf   = sys.argv[1] if len(sys.argv) > 1 else "2308_Floorplan_drawing.pdf"
    scale = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    types = parse_column_types(pdf)
    print(f"\nColumn types found in drawing: {list(types.values())}")

    cols = detect_columns(pdf, drawing_scale=scale)
    print(f"\nDetected {len(cols)} column(s):\n")
    for c in cols:
        print(f"  {c.summary()}")
