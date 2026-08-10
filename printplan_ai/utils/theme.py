"""PrintPlan AI — Custom CSS Theme & UI Helpers.

Provides futuristic industrial aesthetic styling, glassmorphism cards,
status pills, custom metric cards, and responsive layout elements.
"""

import streamlit as st


def inject_custom_css():
    """Inject custom CSS rules to create a premium, high-tech industrial aesthetic."""
    css = """
    <style>
    /* Dark Theme Core Adjustments */
    :root {
        --bg-color: #0b0f19;
        --card-bg: rgba(22, 30, 46, 0.75);
        --card-border: rgba(56, 189, 248, 0.2);
        --accent-cyan: #38bdf8;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-amber: #f59e0b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }

    /* Container Spacing & Header Styling */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

          /* Glassmorphism Header Bar - Light Ice-Blue Gradient Tint */
    .header-banner {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.08);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .header-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0284c7, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        color: #475569;
        font-size: 0.95rem;
        margin-top: 4px;
        margin-bottom: 0;
    }

    /* Status Badges */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }

    .status-pill.success {
        background: rgba(16, 185, 129, 0.15);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .status-pill.info {
        background: rgba(56, 189, 248, 0.15);
        color: #0284c7;
        border: 1px solid rgba(2, 132, 199, 0.4);
    }

    .status-pill.warning {
        background: rgba(245, 158, 11, 0.15);
        color: #d97706;
        border: 1px solid rgba(217, 119, 6, 0.4);
    }

    /* Card Panels - Distinct Light Grey Background */
    .custom-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    .custom-card-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Styled Metric Cards - Pure White Card with Darker Slate Border */
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0284c7 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    /* Navigation Radio Style */
    div[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }

    /* Streamlit Tabs Custom Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.8);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0 16px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }

    /* Buttons Styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
        border: none;
        border-radius: 8px;
        font-weight: 700;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
        transition: all 0.2s ease-in-out;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5);
    }

    /* Code Block styling */
    .stCodeBlock {
        border-radius: 8px !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header(status_text: str = "Ready", status_type: str = "success"):
    """Render the glassmorphic top header bar."""
    status_class = "success" if status_type == "success" else ("warning" if status_type == "warning" else "info")
    dot = "🟢" if status_class == "success" else ("🟡" if status_class == "warning" else "🔵")
    
    html = f"""
    <div class="header-banner">
        <div>
            <h1 class="header-title">🏗️ AUTO2DPRINT </h1>
            <p class="header-subtitle">Vector PDF Floor Plan &rarr; Continuous 3D Concrete Printing G-Code Pipeline</p>
        </div>
        <div>
            <span class="status-pill {status_class}">{dot} {status_text}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
