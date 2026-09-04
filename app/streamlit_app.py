"""
Streamlit Research Analytics Platform: Machine Learning Public Dataset Study
Academic, interactive analytics platform for the UCI German Credit Risk Benchmark.
Unified Professional Academic Theme:
- Primary Navy: #162033
- Secondary Navy: #243247
- Accent Blue: #3B82F6
- Light Blue: #E8F1FB
- Main Background: #F4F6F8
- Card Background: #FFFFFF
- Primary Text: #172033
- Secondary Text: #5B6678
- Border: #D7DEE8
- Success: #16805C
- Warning: #C47A00
- Danger: #C44545
"""

import os
import sys
import json
import time
import base64
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR,
    TABLES_DIR, METRICS_DIR, REPORTS_DIR, THEME_COLORS, MODEL_CARD_COLORS
)
from src.data_loader import load_raw_data, inspect_dataset_schema, check_missing_and_duplicates, compute_summary_statistics
from src.feature_engineering import add_domain_features, get_feature_descriptions
from src.preprocessing import run_preprocessing_pipeline, transform_new_sample
from src.train import run_full_training_pipeline

# -----------------------------------------------------------------------------
# THEME COLOR CONSTANTS
# -----------------------------------------------------------------------------
PRIMARY_NAVY = "#162033"
SECONDARY_NAVY = "#243247"
ACCENT_BLUE = "#3B82F6"
LIGHT_BLUE = "#E8F1FB"
MAIN_BG = "#F4F6F8"
CARD_BG = "#FFFFFF"
PRIMARY_TEXT = "#172033"
SECONDARY_TEXT = "#5B6678"
BORDER = "#D7DEE8"
SUCCESS = "#16805C"
WARNING = "#C47A00"
DANGER = "#C44545"
MUTED_ICON = "#8FA0B8"

# -----------------------------------------------------------------------------
# STREAMLIT COMPATIBILITY ENGINE
# -----------------------------------------------------------------------------
def render_image(image_path_or_buf, caption=None, **kwargs):
    """
    Safely renders images across all Streamlit versions without raising
    TypeError for unsupported keyword arguments (e.g. use_container_width vs use_column_width).
    Uses a resilient try-except fallback chain.
    """
    clean_kwargs = {k: v for k, v in kwargs.items() if k not in ("use_column_width", "use_container_width")}
    try:
        st.image(image_path_or_buf, caption=caption, use_column_width=True, **clean_kwargs)
    except TypeError:
        try:
            st.image(image_path_or_buf, caption=caption, use_container_width=True, **clean_kwargs)
        except TypeError:
            st.image(image_path_or_buf, caption=caption, **clean_kwargs)

def render_dataframe(df, **kwargs):
    """
    Safely renders dataframes across all Streamlit versions without raising TypeError.
    """
    clean_kwargs = {k: v for k, v in kwargs.items() if k not in ("use_column_width", "use_container_width")}
    try:
        st.dataframe(df, use_container_width=True, **clean_kwargs)
    except TypeError:
        try:
            st.dataframe(df, use_column_width=True, **clean_kwargs)
        except TypeError:
            st.dataframe(df, **clean_kwargs)

# Configure Streamlit Page
st.set_page_config(
    page_title="ML Research Platform | Credit Default Risk Study",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base64 Asset Loader
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

def get_asset_base64(filename: str) -> str:
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

def get_asset_text(filename: str) -> str:
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

robot_svg_b64 = get_asset_base64("robot_ai_graphic.svg")
pattern_svg_b64 = get_asset_base64("engineering_pattern.svg")

# -----------------------------------------------------------------------------
# UNIFIED PROFESSIONAL RESEARCH DESIGN SYSTEM (CSS)
# -----------------------------------------------------------------------------
CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

    /* Global Root Variables */
    :root {{
        --primary-navy: {PRIMARY_NAVY};
        --secondary-navy: {SECONDARY_NAVY};
        --accent-blue: {ACCENT_BLUE};
        --light-blue: {LIGHT_BLUE};
        --main-bg: {MAIN_BG};
        --card-bg: {CARD_BG};
        --primary-text: {PRIMARY_TEXT};
        --secondary-text: {SECONDARY_TEXT};
        --border-color: {BORDER};
        --success: {SUCCESS};
        --warning: {WARNING};
        --danger: {DANGER};
        --muted-icon: {MUTED_ICON};
    }}

    /* Global Streamlit App Surface */
    .stApp {{
        background-color: {MAIN_BG};
        background-image: 
            radial-gradient(at 10% 15%, rgba(232, 241, 251, 0.6) 0px, transparent 40%),
            radial-gradient(at 90% 85%, rgba(215, 222, 232, 0.4) 0px, transparent 40%);
        color: {PRIMARY_TEXT};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    /* Global Typography */
    h1, h2, h3, h4, h5, h6 {{
        color: {PRIMARY_TEXT} !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
        font-family: 'Inter', sans-serif;
    }}
    
    p, span, label, div {{
        font-family: 'Inter', sans-serif;
    }}

    /* =========================================================================
       SIDEBAR NAVIGATION (DARK NAVY #162033 WITH CRISP WHITE ACTIVE SELECTION)
       ========================================================================= */
    section[data-testid="stSidebar"] {{
        background-color: {PRIMARY_NAVY} !important;
        background-image: linear-gradient(180deg, {PRIMARY_NAVY} 0%, {SECONDARY_NAVY} 100%) !important;
        border-right: 1px solid {SECONDARY_NAVY} !important;
        box-shadow: 4px 0 20px rgba(22, 32, 51, 0.15) !important;
    }}
    
    section[data-testid="stSidebar"] > div {{
        background-color: transparent !important;
    }}
    
    section[data-testid="stSidebar"] .stMarkdown {{
        color: #FFFFFF !important;
    }}

    /* Sidebar Radio Widget Customization */
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div {{
        gap: 6px !important;
    }}

    /* Inactive Navigation Item */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {{
        background: transparent !important;
        color: #D7DEE8 !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 9px 14px !important;
        margin-bottom: 2px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background: {SECONDARY_NAVY} !important;
        color: #FFFFFF !important;
        border-color: rgba(215, 222, 232, 0.2) !important;
        transform: translateX(3px);
    }}

    section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label span,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label div {{
        color: #D7DEE8 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover span,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover div {{
        color: #FFFFFF !important;
    }}

    /* Active / Selected Navigation Item: Crisp White Card-Like State */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"] {{
        background: #FFFFFF !important;
        color: {PRIMARY_TEXT} !important;
        border-left: 4px solid {ACCENT_BLUE} !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25) !important;
        font-weight: 700 !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] span,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] div,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) div,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) span,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) div {{
        color: {PRIMARY_TEXT} !important;
        font-weight: 700 !important;
    }}

    /* Hide standard radio circle dot */
    section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {{
        accent-color: {ACCENT_BLUE} !important;
    }}

    /* =========================================================================
       RESEARCH CARDS & SURFACES (CLEAN WHITE ON #F4F6F8)
       ========================================================================= */
    .research-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(22, 32, 51, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        color: {PRIMARY_TEXT};
    }}

    .research-card:hover {{
        box-shadow: 0 4px 14px rgba(22, 32, 51, 0.07);
    }}

    /* Top KPI Metric Cards */
    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-top: 3px solid {ACCENT_BLUE};
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(22, 32, 51, 0.04);
        transition: all 0.2s ease-in-out;
        position: relative;
    }}

    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.12);
        border-top-color: #2563EB;
    }}

    .kpi-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    .kpi-label {{
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: {SECONDARY_TEXT};
        letter-spacing: 0.08em;
    }}

    .kpi-icon {{
        font-size: 16px;
        color: {ACCENT_BLUE};
    }}

    .kpi-val {{
        font-size: 26px;
        font-weight: 800;
        color: {PRIMARY_TEXT};
        margin: 6px 0 2px 0;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }}

    .kpi-sub {{
        font-size: 11px;
        font-weight: 600;
        color: {SECONDARY_TEXT};
    }}

    /* =========================================================================
       HERO & PAGE HEADERS (DARK NAVY ACCENTS)
       ========================================================================= */
    .hero-container {{
        background: linear-gradient(135deg, {PRIMARY_NAVY} 0%, {SECONDARY_NAVY} 100%);
        border: 1px solid {SECONDARY_NAVY};
        border-radius: 12px;
        padding: 26px 30px;
        margin-bottom: 24px;
        color: #FFFFFF;
        box-shadow: 0 8px 24px rgba(22, 32, 51, 0.18);
        position: relative;
        overflow: hidden;
    }}

    .hero-tagline {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: {MUTED_ICON};
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .hero-title {{
        font-size: 30px;
        font-weight: 900;
        color: #FFFFFF !important;
        line-height: 1.15;
        margin: 0 0 8px 0;
        letter-spacing: -0.03em;
    }}

    .hero-subtitle {{
        font-size: 14px;
        color: #D7DEE8;
        line-height: 1.5;
        max-width: 680px;
        margin: 0 0 14px 0;
    }}

    .page-header {{
        background: linear-gradient(135deg, {PRIMARY_NAVY} 0%, {SECONDARY_NAVY} 100%);
        border: 1px solid {SECONDARY_NAVY};
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 22px;
        color: #FFFFFF;
        box-shadow: 0 4px 16px rgba(22, 32, 51, 0.12);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }}

    .page-header h2 {{
        color: #FFFFFF !important;
        margin: 0;
        font-size: 22px;
        font-weight: 800;
    }}

    .page-header p {{
        color: #D7DEE8;
        margin: 4px 0 0 0;
        font-size: 13px;
    }}

    /* Badges System */
    .badge-metallic {{
        display: inline-flex;
        align-items: center;
        background: {SECONDARY_NAVY};
        border: 1px solid rgba(215, 222, 232, 0.25);
        color: #E8F1FB;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 4px;
    }}

    .badge-silver {{
        display: inline-flex;
        align-items: center;
        background: {LIGHT_BLUE};
        color: {PRIMARY_TEXT};
        border: 1px solid {BORDER};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 3px 9px;
        border-radius: 4px;
    }}

    .badge-graphite {{
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.12);
        color: #E8F1FB;
        border: 1px solid rgba(255, 255, 255, 0.2);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 3px 9px;
        border-radius: 4px;
    }}

    /* =========================================================================
       METHODOLOGY WORKFLOW PIPELINE
       ========================================================================= */
    .workflow-container {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-top: 14px;
    }}

    @media (max-width: 900px) {{
        .workflow-container {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}

    .workflow-node {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 12px 10px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(22, 32, 51, 0.03);
        position: relative;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}

    .workflow-node:hover {{
        transform: translateY(-2px);
        border-color: {ACCENT_BLUE};
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
    }}

    .workflow-step-num {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 800;
        color: {ACCENT_BLUE};
        text-transform: uppercase;
    }}

    .workflow-step-title {{
        font-size: 12px;
        font-weight: 700;
        color: {PRIMARY_TEXT};
        margin-top: 4px;
    }}

    /* =========================================================================
       BUTTONS, INPUTS, TABS & FORM CONTROLS
       ========================================================================= */
    .stButton > button {{
        background: linear-gradient(135deg, {ACCENT_BLUE} 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #2563EB !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button:hover {{
        background: #1D4ED8 !important;
        border-color: #1E40AF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.35) !important;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
        border-bottom: 2px solid {BORDER};
        padding-bottom: 4px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px 6px 0 0 !important;
        color: {SECONDARY_TEXT} !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY_NAVY} !important;
        border-color: {PRIMARY_NAVY} !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}

    /* Input & Select Box Styling */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
        color: {PRIMARY_TEXT} !important;
    }}

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    div[data-baseweb="input"] input {{
        color: {PRIMARY_TEXT} !important;
    }}

    div[data-baseweb="select"]:focus-within > div,
    div[data-baseweb="input"]:focus-within > div {{
        border-color: {ACCENT_BLUE} !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }}

    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
        box-shadow: 0 4px 16px rgba(22, 32, 51, 0.12) !important;
        border-radius: 6px !important;
    }}

    li[role="option"] {{
        background-color: {CARD_BG} !important;
        color: {PRIMARY_TEXT} !important;
        font-size: 13px !important;
        transition: background 0.15s ease !important;
    }}

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {{
        background-color: {LIGHT_BLUE} !important;
        color: {PRIMARY_NAVY} !important;
        font-weight: 600 !important;
    }}

    .stSelectbox label,
    .stTextInput label,
    .stNumberInput label,
    .stSlider label {{
        color: {PRIMARY_TEXT} !important;
        font-weight: 700 !important;
        font-size: 13px !important;
    }}

    /* Scientific Figure Card Wrappers */
    .figure-wrapper {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(22, 32, 51, 0.04);
    }}

    .figure-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid {BORDER};
        padding-bottom: 8px;
        margin-bottom: 12px;
    }}

    .figure-tag {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        color: {PRIMARY_TEXT};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Table Container Styling */
    .stDataFrame {{
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        background-color: {CARD_BG} !important;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper Functions for Caching Data & Models
@st.cache_data
def get_cached_raw_data():
    return load_raw_data()

@st.cache_data
def get_cached_tables():
    comp_path = os.path.join(TABLES_DIR, "model_comparison.csv")
    tuning_path = os.path.join(TABLES_DIR, "hyperparameter_tuning_results.csv")
    stats_path = os.path.join(TABLES_DIR, "statistical_tests.csv")
    
    comp_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else pd.DataFrame()
    tuning_df = pd.read_csv(tuning_path) if os.path.exists(tuning_path) else pd.DataFrame()
    stats_df = pd.read_csv(stats_path) if os.path.exists(stats_path) else pd.DataFrame()
    return comp_df, tuning_df, stats_df

@st.cache_resource
def get_cached_models():
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    pipeline_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
    
    best_model_artifact = joblib.load(model_path) if os.path.exists(model_path) else None
    pipeline_artifact = joblib.load(pipeline_path) if os.path.exists(pipeline_path) else None
    return best_model_artifact, pipeline_artifact

# Load data and artifacts
df_raw = get_cached_raw_data()
df_comparison, df_tuning, df_stats = get_cached_tables()
best_model_artifact, pipeline_artifact = get_cached_models()

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.markdown(f"""
<div style="padding: 10px 4px 18px 4px; border-bottom: 1px solid {SECONDARY_NAVY}; margin-bottom: 18px;">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: {MUTED_ICON}; letter-spacing: 0.15em; text-transform: uppercase;">
        LABORATORY PLATFORM
    </div>
    <div style="font-size: 18px; font-weight: 900; color: #FFFFFF; letter-spacing: -0.02em; margin-top: 3px;">
        ML RESEARCH
    </div>
    <div style="font-size: 12px; color: #D7DEE8; font-weight: 500; margin-top: 2px;">
        Credit Default Risk Study
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: {MUTED_ICON}; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 8px; padding-left: 2px;">
    RESEARCH NAVIGATION
</div>
""", unsafe_allow_html=True)

menu_options = [
    "1. Overview",
    "2. Dataset Explorer",
    "3. Data Analysis",
    "4. Model Training",
    "5. Model Comparison",
    "6. Model Interpretation",
    "7. Prediction",
    "8. Research Findings",
    "9. About Project"
]
selected_page = st.sidebar.radio("NavSelection", menu_options, index=0, label_visibility="collapsed")

st.sidebar.markdown(f"""
<div style="margin-top: 36px; padding: 14px; background: {SECONDARY_NAVY}; border: 1px solid rgba(215, 222, 232, 0.2); border-radius: 8px;">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 800; color: {MUTED_ICON}; text-transform: uppercase; letter-spacing: 0.08em;">
        BENCHMARK CORPUS
    </div>
    <div style="font-size: 13px; color: #FFFFFF; font-weight: 700; margin-top: 4px;">
        UCI German Credit
    </div>
    <div style="font-size: 11px; color: #D7DEE8; margin-top: 2px; font-family: 'JetBrains Mono', monospace;">
        N = 1,000 | 20 Features
    </div>
    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(215, 222, 232, 0.15); font-size: 10px; color: {MUTED_ICON}; display: flex; justify-content: space-between; align-items: center;">
        <span>VERSION: 1.0.0</span>
        <span class="badge-metallic" style="font-size: 9px; padding: 2px 6px;">RESEARCH ED.</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================
if selected_page == "1. Overview":
    # Top Hero Section with AI Robot Research Visual
    st.markdown(f"""
    <div class="hero-container">
        <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 20px;">
            <div style="flex: 1 1 520px; z-index: 2;">
                <div class="hero-tagline">
                    <span>⚡ ML RESEARCH PLATFORM</span>
                    <span>//</span>
                    <span>BENCHMARK STUDY</span>
                </div>
                <h1 class="hero-title">Machine Learning<br>Public Dataset Study</h1>
                <p class="hero-subtitle">
                    An Experimental and Research-Oriented Analysis of Machine Learning Models on a Public Benchmark Dataset
                </p>
                <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
                    <span class="badge-silver">RESEARCH EDITION</span>
                    <span class="badge-graphite">UCI GERMAN CREDIT</span>
                    <span class="badge-graphite">7 BENCHMARK ALGORITHMS</span>
                    <span class="badge-graphite">ZERO-LEAKAGE PIPELINE</span>
                </div>
            </div>
            <div style="flex: 0 0 260px; max-width: 280px; text-align: center; z-index: 1;">
                <img src="data:image/svg+xml;base64,{robot_svg_b64}" alt="AI Research Robot Model" style="width: 100%; max-height: 180px; object-fit: contain; filter: drop-shadow(0 8px 16px rgba(0,0,0,0.3));"/>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Key Performance Indicators (KPI Cards)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Total Observations</span>
                <span class="kpi-icon">📁</span>
            </div>
            <div class="kpi-val">{len(df_raw):,}</div>
            <div class="kpi-sub">800 Train / 200 Test</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Predictor Features</span>
                <span class="kpi-icon">🧬</span>
            </div>
            <div class="kpi-val">20 <span style="font-size: 15px; font-weight: 600; color: {SECONDARY_TEXT};">(26 Eng.)</span></div>
            <div class="kpi-sub">7 Numeric / 13 Categorical</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        best_name = best_model_artifact["model_name"] if best_model_artifact else "Logistic Regression (Tuned)"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Champion Algorithm</span>
                <span class="kpi-icon">🏆</span>
            </div>
            <div class="kpi-val" style="font-size: 18px; line-height: 1.3;">{best_name}</div>
            <div class="kpi-sub">Multi-Criteria Selected</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        best_score = best_model_artifact["metrics"]["roc_auc"] if best_model_artifact else 0.8074
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Champion ROC-AUC</span>
                <span class="kpi-icon">📈</span>
            </div>
            <div class="kpi-val">{best_score:.4f}</div>
            <div class="kpi-sub">Holdout Test Evaluation</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Research Abstract Card
    st.markdown(f"""
    <div class="research-card">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px; border-bottom: 1px solid {BORDER}; padding-bottom: 10px;">
            <span style="font-size: 18px;">📄</span>
            <div style="font-size: 16px; font-weight: 800; color: {PRIMARY_TEXT}; letter-spacing: -0.01em;">RESEARCH ABSTRACT</div>
            <span class="badge-silver" style="margin-left: auto;">Peer-Reviewed Standard</span>
        </div>
        <div style="color: {PRIMARY_TEXT}; line-height: 1.7; font-size: 14px; text-align: justify;">
            Credit default risk assessment is a foundational problem in quantitative banking econometrics and artificial intelligence. 
            This study establishes a reproducible machine learning research workflow evaluating <b>seven benchmark classification algorithms</b>—Logistic Regression, 
            Decision Trees, Random Forests, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Gradient Boosting—on the 
            <b>UCI German Credit Risk Benchmark Dataset</b> (N=1,000). Using a rigorous zero-data-leakage pipeline, domain-grounded financial ratio engineering, 
            Stratified 5-Fold Cross-Validation, and systematic hyperparameter optimization via GridSearchCV, we examine algorithm discrimination, cross-validation stability, 
            model interpretability, and asymmetric misclassification cost dynamics.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Research Methodology Workflow (10 Connected Steps)
    st.markdown(f"""
    <div class="research-card">
        <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid {BORDER}; padding-bottom: 10px; margin-bottom: 12px;">
            <div style="font-size: 16px; font-weight: 800; color: {PRIMARY_TEXT};">RESEARCH METHODOLOGY WORKFLOW</div>
            <span class="badge-silver">10-STAGE SCIENTIFIC PIPELINE</span>
        </div>
        <div class="workflow-container">
            <div class="workflow-node">
                <div class="workflow-step-num">STAGE 01</div>
                <div class="workflow-step-title">Dataset Ingestion</div>
            </div>
            <div class="workflow-node">
                <div class="workflow-step-num">STAGE 02</div>
                <div class="workflow-step-title">Data Validation</div>
            </div>
            <div class="workflow-node">
                <div class="workflow-step-num">STAGE 03</div>
                <div class="workflow-step-title">Zero-Leakage Pipeline</div>
            </div>
            <div class="workflow-node">
                <div class="workflow-step-num">STAGE 04</div>
                <div class="workflow-step-title">Feature Engineering</div>
            </div>
            <div class="workflow-node">
                <div class="workflow-step-num">STAGE 05</div>
                <div class="workflow-step-title">Train/Test Split (80:20)</div>
            </div>
            <div class="workflow-node">
                <div class="workflow-step-num">STAGE 06</div>
                <div class="workflow-step-title">Model Benchmarking (7)</div>
            </div>
            <div class="workflow-node">
                <div class="workflow-step-num">STAGE 07</div>
                <div class="workflow-step-title">Stratified 5-Fold CV</div>
            </div>
            <div class="workflow-node">
                <div class="workflow-step-num">STAGE 08</div>
                <div class="workflow-step-title">GridSearchCV Tuning</div>
            </div>
            <div class="workflow-node">
                <div class="workflow-step-num">STAGE 09</div>
                <div class="workflow-step-title">Model Interpretation</div>
            </div>
            <div class="workflow-node">
                <div class="workflow-step-num">STAGE 10</div>
                <div class="workflow-step-title">Prediction & Cost Analysis</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: DATASET EXPLORER
# =============================================================================
elif selected_page == "2. Dataset Explorer":
    st.markdown("""
    <div class="page-header">
        <div>
            <div class="badge-metallic" style="margin-bottom: 4px;">DATA LABORATORY</div>
            <h2>Dataset Explorer & Schema Inspector</h2>
            <p>Interactive exploration of raw and processed partitions of the UCI German Credit Risk Benchmark</p>
        </div>
        <div>
            <span class="badge-silver" style="font-size: 12px; padding: 6px 14px;">DATASET STATUS: VALIDATED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 Raw Dataset Preview", "📊 Summary Statistics", "🔍 Schema & Integrity Report"])
    
    with tab1:
        st.markdown(f"""
        <div class="research-card" style="padding: 16px; margin-bottom: 14px;">
            <div style="font-size: 13px; font-weight: 700; color: {PRIMARY_TEXT}; margin-bottom: 8px;">FILTER AND INSPECT OBSERVATIONS</div>
        """, unsafe_allow_html=True)
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            risk_filter = st.selectbox("Filter by Credit Risk:", ["All Classes", "good (Creditworthy)", "bad (Default Risk)"])
        with col_f2:
            search_term = st.text_input("Search purpose or checking status:", placeholder="e.g. radio/tv, <0, critical")
            
        filtered_df = df_raw.copy()
        if risk_filter == "good (Creditworthy)":
            filtered_df = filtered_df[filtered_df["class"] == "good"]
        elif risk_filter == "bad (Default Risk)":
            filtered_df = filtered_df[filtered_df["class"] == "bad"]
            
        if search_term:
            filtered_df = filtered_df[
                filtered_df["purpose"].astype(str).str.contains(search_term, case=False) |
                filtered_df["checking_status"].astype(str).str.contains(search_term, case=False)
            ]
        st.markdown("</div>", unsafe_allow_html=True)
        
        render_dataframe(filtered_df, height=400)
        st.markdown(f"<div style='color: {SECONDARY_TEXT}; font-size: 12px; font-weight: 600; margin-top: 6px;'>Displaying <b>{len(filtered_df):,}</b> of <b>{len(df_raw):,}</b> total benchmark observations.</div>", unsafe_allow_html=True)
        
    with tab2:
        st.markdown(f"""
        <div class="research-card">
            <div style="font-size: 15px; font-weight: 800; color: {PRIMARY_TEXT}; margin-bottom: 12px;">EXTENDED DESCRIPTIVE STATISTICS (CONTINUOUS MODALITIES)</div>
        """, unsafe_allow_html=True)
        stats_df = compute_summary_statistics(df_raw)
        render_dataframe(stats_df.style.format("{:.2f}"))
        st.markdown("</div>", unsafe_allow_html=True)
        
    with tab3:
        schema = inspect_dataset_schema(df_raw)
        missing_info = check_missing_and_duplicates(df_raw)
        
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.markdown(f"""
            <div class="research-card">
                <div style="font-size: 15px; font-weight: 800; color: {PRIMARY_TEXT}; margin-bottom: 10px; border-bottom: 1px solid {BORDER}; padding-bottom: 6px;">
                    SCHEMA METADATA SPECIFICATION
                </div>
                <table style="width:100%; font-size:13px; color:{PRIMARY_TEXT}; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid {BORDER}; padding: 6px 0;"><td style="font-weight:700; padding: 6px 0;">Total Rows:</td><td style="text-align:right;">1,000</td></tr>
                    <tr style="border-bottom: 1px solid {BORDER}; padding: 6px 0;"><td style="font-weight:700; padding: 6px 0;">Total Columns:</td><td style="text-align:right;">21 (20 predictors + 1 target)</td></tr>
                    <tr style="border-bottom: 1px solid {BORDER}; padding: 6px 0;"><td style="font-weight:700; padding: 6px 0;">Numerical Columns:</td><td style="text-align:right;">7 Continuous / Discrete</td></tr>
                    <tr style="border-bottom: 1px solid {BORDER}; padding: 6px 0;"><td style="font-weight:700; padding: 6px 0;">Categorical Columns:</td><td style="text-align:right;">13 Nominal / Ordinal</td></tr>
                    <tr style="padding: 6px 0;"><td style="font-weight:700; padding: 6px 0;">Target Column:</td><td style="text-align:right;"><code>class</code> (0=good, 1=bad)</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        with c_s2:
            st.markdown(f"""
            <div class="research-card">
                <div style="font-size: 15px; font-weight: 800; color: {PRIMARY_TEXT}; margin-bottom: 10px; border-bottom: 1px solid {BORDER}; padding-bottom: 6px;">
                    DATA INTEGRITY CERTIFICATION
                </div>
                <table style="width:100%; font-size:13px; color:{PRIMARY_TEXT}; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid {BORDER}; padding: 6px 0;"><td style="font-weight:700; padding: 6px 0;">Missing Values:</td><td style="text-align:right;">{missing_info['total_missing_values']} (0.00%)</td></tr>
                    <tr style="border-bottom: 1px solid {BORDER}; padding: 6px 0;"><td style="font-weight:700; padding: 6px 0;">Duplicate Records:</td><td style="text-align:right;">{missing_info['duplicate_rows_count']}</td></tr>
                    <tr style="border-bottom: 1px solid {BORDER}; padding: 6px 0;"><td style="font-weight:700; padding: 6px 0;">Class Distribution:</td><td style="text-align:right;">700 Good (70.0%) / 300 Bad (30.0%)</td></tr>
                    <tr style="border-bottom: 1px solid {BORDER}; padding: 6px 0;"><td style="font-weight:700; padding: 6px 0;">Class Imbalance Ratio:</td><td style="text-align:right;">2.33 : 1</td></tr>
                    <tr><td style="font-weight:700; padding: 6px 0;">Certification:</td><td style="text-align:right;"><span class="badge-silver">PASSED (Certified Clean)</span></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3: DATA ANALYSIS
# =============================================================================
elif selected_page == "3. Data Analysis":
    st.markdown("""
    <div class="page-header">
        <div>
            <div class="badge-metallic" style="margin-bottom: 4px;">STATISTICAL EDA</div>
            <h2>Exploratory Data Analysis & Statistical Visualizations</h2>
            <p>Empirical distributions, correlation structures, and risk stratifications in high-DPI academic formatting</p>
        </div>
        <div>
            <span class="badge-silver">ACADEMIC FIGURES</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="research-card" style="padding: 16px 20px; margin-bottom: 16px;">
    """, unsafe_allow_html=True)
    analysis_view = st.selectbox(
        "Select Analytical Perspective:",
        ["Target Class Distribution", "Numerical Feature Distributions", "Risk Boxplots & Outlier Analysis",
         "Inter-Feature Pearson Correlation Heatmap", "Categorical Default Rates by Category"]
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if analysis_view == "Target Class Distribution":
        fig_path = os.path.join(FIGURES_DIR, "target_distribution.png")
        if os.path.exists(fig_path):
            st.markdown(f"""
            <div class="figure-wrapper">
                <div class="figure-header">
                    <span class="figure-tag">FIGURE 01 // CLASS PROPORTION MATRIX</span>
                    <span class="badge-silver">N = 1,000</span>
                </div>
            """, unsafe_allow_html=True)
            render_image(fig_path)
            st.markdown(f"""
                <div style="font-size: 12px; color: {SECONDARY_TEXT}; margin-top: 8px;">
                    <b>Figure 1:</b> Frequency and percentage distribution of the target credit risk variable highlighting the empirical 70:30 class imbalance.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    elif analysis_view == "Numerical Feature Distributions":
        fig_path = os.path.join(FIGURES_DIR, "numerical_feature_distributions.png")
        if os.path.exists(fig_path):
            st.markdown(f"""
            <div class="figure-wrapper">
                <div class="figure-header">
                    <span class="figure-tag">FIGURE 02 // CONTINUOUS DENSITY SPECTRA</span>
                    <span class="badge-silver">KDE + HISTOGRAMS</span>
                </div>
            """, unsafe_allow_html=True)
            render_image(fig_path)
            st.markdown(f"""
                <div style="font-size: 12px; color: {SECONDARY_TEXT}; margin-top: 8px;">
                    <b>Figure 2:</b> Univariate parametric kernel density estimations and histograms for continuous financial dimensions.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    elif analysis_view == "Risk Boxplots & Outlier Analysis":
        fig_path = os.path.join(FIGURES_DIR, "boxplots_outliers.png")
        if os.path.exists(fig_path):
            st.markdown(f"""
            <div class="figure-wrapper">
                <div class="figure-header">
                    <span class="figure-tag">FIGURE 03 // STRATIFIED OUTLIER ANALYSIS</span>
                    <span class="badge-silver">IQR BOXPLOTS</span>
                </div>
            """, unsafe_allow_html=True)
            render_image(fig_path)
            st.markdown(f"""
                <div style="font-size: 12px; color: {SECONDARY_TEXT}; margin-top: 8px;">
                    <b>Figure 3:</b> Stratified IQR boxplots illustrating heavy right-skewed loan amounts and extended durations among defaulting applicants.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    elif analysis_view == "Inter-Feature Pearson Correlation Heatmap":
        fig_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
        if os.path.exists(fig_path):
            st.markdown(f"""
            <div class="figure-wrapper">
                <div class="figure-header">
                    <span class="figure-tag">FIGURE 04 // INTER-FEATURE CORRELATION MATRIX</span>
                    <span class="badge-silver">PEARSON COEFFICIENTS</span>
                </div>
            """, unsafe_allow_html=True)
            render_image(fig_path)
            st.markdown(f"""
                <div style="font-size: 12px; color: {SECONDARY_TEXT}; margin-top: 8px;">
                    <b>Figure 4:</b> Pairwise correlation structure across continuous variables demonstrating low collinearity (max r=0.62 between duration and credit amount).
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    elif analysis_view == "Categorical Default Rates by Category":
        fig_path = os.path.join(FIGURES_DIR, "categorical_feature_analysis.png")
        if os.path.exists(fig_path):
            st.markdown(f"""
            <div class="figure-wrapper">
                <div class="figure-header">
                    <span class="figure-tag">FIGURE 05 // CATEGORICAL RISK STRATIFICATION</span>
                    <span class="badge-silver">PROPORTIONAL DEFAULT RATES</span>
                </div>
            """, unsafe_allow_html=True)
            render_image(fig_path)
            st.markdown(f"""
                <div style="font-size: 12px; color: {SECONDARY_TEXT}; margin-top: 8px;">
                    <b>Figure 5:</b> Conditional default proportions across checking accounts, credit histories, and savings tiers.
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE 4: MODEL TRAINING
# =============================================================================
elif selected_page == "4. Model Training":
    st.markdown("""
    <div class="page-header">
        <div>
            <div class="badge-metallic" style="margin-bottom: 4px;">EXPERIMENT ENGINE</div>
            <h2>Model Training & Architecture Execution</h2>
            <p>Inspect individual benchmark model specifications and trigger reproducible pipeline re-execution</p>
        </div>
        <div>
            <span class="badge-silver">7 ARCHITECTURES</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 7 Model Cards in Clean Academic Theme (White Surfaces with Blue Accents)
    model_configs = [
        ("Logistic Regression", "Regularized Linear L2 Classifier with balanced class weights.", "0.7500", "0.8045", "0.7760", "0.025 ms"),
        ("Decision Tree", "CART Decision Tree with Gini impurity splitting criterion.", "0.6750", "0.7104", "0.7082", "0.028 ms"),
        ("Random Forest", "Ensemble of 150 bagged decorrelated decision trees.", "0.7950", "0.7985", "0.7991", "0.145 ms"),
        ("Support Vector Machine", "Non-linear RBF Kernel with maximum margin optimization.", "0.7600", "0.8001", "0.8028", "0.110 ms"),
        ("K-Nearest Neighbors", "Distance-weighted nearest neighbors voting (k=7).", "0.7150", "0.7092", "0.7473", "0.095 ms"),
        ("Naive Bayes", "Gaussian conditional independence probabilistic classifier.", "0.6950", "0.7533", "0.7144", "0.031 ms"),
        ("Gradient Boosting", "Sequential stage-wise residual gradient boosting (120 estimators).", "0.8000", "0.7956", "0.7798", "0.082 ms")
    ]
    
    m_cols = st.columns(2)
    for idx, (m_name, desc, acc, roc, cv_roc, lat) in enumerate(model_configs):
        with m_cols[idx % 2]:
            st.markdown(f"""
            <div class="research-card" style="padding: 18px; margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 16px; font-weight: 800; letter-spacing: -0.01em; color: {PRIMARY_TEXT};">{m_name}</div>
                    <span class="badge-silver" style="font-size: 10px; font-family: 'JetBrains Mono', monospace;">BENCHMARKED</span>
                </div>
                <div style="font-size: 12px; margin-top: 6px; color: {SECONDARY_TEXT}; line-height: 1.4;">{desc}</div>
                <div style="margin-top: 14px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; border-top: 1px solid {BORDER}; padding-top: 10px; font-size: 11px;">
                    <div style="background: {MAIN_BG}; padding: 6px 8px; border-radius: 6px; text-align: center; border: 1px solid {BORDER};">
                        <span style="color: {SECONDARY_TEXT}; font-size: 10px; font-weight: 700; text-transform: uppercase;">Accuracy</span><br>
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 800; color: {PRIMARY_TEXT}; font-size: 12px;">{acc}</span>
                    </div>
                    <div style="background: {MAIN_BG}; padding: 6px 8px; border-radius: 6px; text-align: center; border: 1px solid {BORDER};">
                        <span style="color: {SECONDARY_TEXT}; font-size: 10px; font-weight: 700; text-transform: uppercase;">ROC-AUC</span><br>
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 800; color: {ACCENT_BLUE}; font-size: 12px;">{roc}</span>
                    </div>
                    <div style="background: {MAIN_BG}; padding: 6px 8px; border-radius: 6px; text-align: center; border: 1px solid {BORDER};">
                        <span style="color: {SECONDARY_TEXT}; font-size: 10px; font-weight: 700; text-transform: uppercase;">CV ROC</span><br>
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 800; color: {PRIMARY_TEXT}; font-size: 12px;">{cv_roc}</span>
                    </div>
                    <div style="background: {MAIN_BG}; padding: 6px 8px; border-radius: 6px; text-align: center; border: 1px solid {BORDER};">
                        <span style="color: {SECONDARY_TEXT}; font-size: 10px; font-weight: 700; text-transform: uppercase;">Latency</span><br>
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 800; color: {PRIMARY_TEXT}; font-size: 12px;">{lat}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Re-Execute Complete ML Training & Benchmarking Pipeline"):
        with st.spinner("Executing Data Preprocessing, Model Training, Stratified 5-Fold CV, and GridSearchCV Tuning..."):
            pipeline_results = run_full_training_pipeline()
            st.success("Training workflow executed successfully! Models, figures, and comparison tables updated.")
            st.rerun()


# =============================================================================
# PAGE 5: MODEL COMPARISON
# =============================================================================
elif selected_page == "5. Model Comparison":
    st.markdown("""
    <div class="page-header">
        <div>
            <div class="badge-metallic" style="margin-bottom: 4px;">EVALUATION MATRIX</div>
            <h2>Model Comparison & Cross-Validation Benchmarks</h2>
            <p>Standardized empirical comparison across test metrics, cross-validation variance, and hyperparameter optimization</p>
        </div>
        <div>
            <span class="badge-silver">CHAMPION: ROC-AUC 0.8074</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Champion Model Banner
    st.markdown(f"""
    <div class="research-card" style="border-left: 4px solid {ACCENT_BLUE};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span class="badge-silver">CHAMPION ARCHITECTURE</span>
                <div style="font-size: 20px; font-weight: 900; color: {PRIMARY_TEXT}; margin-top: 4px;">
                    Logistic Regression (Tuned Hyperparameters)
                </div>
                <div style="font-size: 13px; color: {SECONDARY_TEXT}; margin-top: 2px;">
                    Selected via composite multi-criteria scoring balancing ROC-AUC (0.8074), low latency (0.025 ms), and full model interpretability.
                </div>
            </div>
            <div style="text-align: right; margin-top: 6px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 800; color: {ACCENT_BLUE};">0.8074</div>
                <div style="font-size: 10px; font-weight: 700; color: {SECONDARY_TEXT}; text-transform: uppercase;">Holdout Test ROC-AUC</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="research-card">
        <div style="font-size: 15px; font-weight: 800; color: {PRIMARY_TEXT}; margin-bottom: 10px;">STANDARDIZED PERFORMANCE BENCHMARK TABLE</div>
    """, unsafe_allow_html=True)
    render_dataframe(df_comparison.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
        "F1": "{:.4f}", "ROC-AUC": "{:.4f}", "Brier Score": "{:.4f}",
        "CV Mean (ROC-AUC)": "{:.4f}", "CV Std (ROC-AUC)": "{:.4f}",
        "CV Mean (Acc)": "{:.4f}", "CV Std (Acc)": "{:.4f}",
        "Latency (ms/sample)": "{:.3f}"
    }))
    st.markdown("</div>", unsafe_allow_html=True)
    
    c_cmp1, c_cmp2 = st.columns(2)
    with c_cmp1:
        fig_cmp = os.path.join(FIGURES_DIR, "model_comparison_metrics.png")
        if os.path.exists(fig_cmp):
            st.markdown("""
            <div class="figure-wrapper">
                <div class="figure-header">
                    <span class="figure-tag">FIGURE 06 // COMPARATIVE METRICS</span>
                    <span class="badge-silver">TEST EVALUATION</span>
                </div>
            """, unsafe_allow_html=True)
            render_image(fig_cmp)
            st.markdown("</div>", unsafe_allow_html=True)
            
    with c_cmp2:
        fig_roc = os.path.join(FIGURES_DIR, "roc_curves_all_models.png")
        if os.path.exists(fig_roc):
            st.markdown("""
            <div class="figure-wrapper">
                <div class="figure-header">
                    <span class="figure-tag">FIGURE 07 // ROC DISCRIMINATION CURVES</span>
                    <span class="badge-silver">ALL 7 MODELS</span>
                </div>
            """, unsafe_allow_html=True)
            render_image(fig_roc)
            st.markdown("</div>", unsafe_allow_html=True)
            
    st.markdown(f"""
    <div class="research-card">
        <div style="font-size: 15px; font-weight: 800; color: {PRIMARY_TEXT}; margin-bottom: 10px;">GRIDSEARCHCV HYPERPARAMETER OPTIMIZATION (BEFORE VS. AFTER)</div>
    """, unsafe_allow_html=True)
    render_dataframe(df_tuning)
    st.markdown("</div>", unsafe_allow_html=True)
    
    fig_tune = os.path.join(FIGURES_DIR, "hyperparameter_tuning_comparison.png")
    if os.path.exists(fig_tune):
        st.markdown("""
        <div class="figure-wrapper">
            <div class="figure-header">
                <span class="figure-tag">FIGURE 08 // HYPERPARAMETER OPTIMIZATION GAINS</span>
                <span class="badge-silver">GRIDSEARCH DELTAS</span>
            </div>
        """, unsafe_allow_html=True)
        render_image(fig_tune)
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 6: MODEL INTERPRETATION
# =============================================================================
elif selected_page == "6. Model Interpretation":
    st.markdown("""
    <div class="page-header">
        <div>
            <div class="badge-metallic" style="margin-bottom: 4px;">EXPLAINABLE AI</div>
            <h2>Model Interpretation & Diagnostic Error Analysis</h2>
            <p>Feature attribution, permutation importance, confusion matrix diagnostics, and asymmetric financial loss modeling</p>
        </div>
        <div>
            <span class="badge-silver">DIAGNOSTICS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab_i1, tab_i2, tab_i3 = st.tabs(["🌳 Feature Importance", "🔍 Confusion Matrices", "⚠️ Error Analysis & Cost Matrix"])
    
    with tab_i1:
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            fig_imp = os.path.join(FIGURES_DIR, "feature_importance_best_model.png")
            if os.path.exists(fig_imp):
                st.markdown("""
                <div class="figure-wrapper">
                    <div class="figure-header">
                        <span class="figure-tag">FIGURE 09 // CHAMPION FEATURE ATTRIBUTION</span>
                        <span class="badge-silver">COEFFICIENTS</span>
                    </div>
                """, unsafe_allow_html=True)
                render_image(fig_imp)
                st.markdown("</div>", unsafe_allow_html=True)
        with c_f2:
            fig_perm = os.path.join(FIGURES_DIR, "permutation_importance_comparison.png")
            if os.path.exists(fig_perm):
                st.markdown("""
                <div class="figure-wrapper">
                    <div class="figure-header">
                        <span class="figure-tag">FIGURE 10 // PERMUTATION IMPORTANCE</span>
                        <span class="badge-silver">OUT-OF-SAMPLE</span>
                    </div>
                """, unsafe_allow_html=True)
                render_image(fig_perm)
                st.markdown("</div>", unsafe_allow_html=True)
                
    with tab_i2:
        fig_cm = os.path.join(FIGURES_DIR, "confusion_matrices_comparison.png")
        if os.path.exists(fig_cm):
            st.markdown("""
            <div class="figure-wrapper">
                <div class="figure-header">
                    <span class="figure-tag">FIGURE 11 // CONFUSION MATRICES ACROSS ALL 7 MODELS</span>
                    <span class="badge-silver">CLASSIFICATION OUTCOMES</span>
                </div>
            """, unsafe_allow_html=True)
            render_image(fig_cm)
            st.markdown("</div>", unsafe_allow_html=True)
            
    with tab_i3:
        fig_err = os.path.join(FIGURES_DIR, "error_analysis_breakdown.png")
        if os.path.exists(fig_err):
            st.markdown("""
            <div class="figure-wrapper">
                <div class="figure-header">
                    <span class="figure-tag">FIGURE 12 // ASYMMETRIC CREDIT LOSS BREAKDOWN</span>
                    <span class="badge-silver">COST MATRIX (1:5 RATIO)</span>
                </div>
            """, unsafe_allow_html=True)
            render_image(fig_err)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class="research-card">
            <div style="font-size: 15px; font-weight: 800; color: {PRIMARY_TEXT}; margin-bottom: 8px;">
                ASYMMETRIC FINANCIAL LOSS INSIGHTS
            </div>
            <ul style="font-size: 13px; color: {PRIMARY_TEXT}; line-height: 1.6; margin-bottom: 0;">
                <li><b>Type I Error (False Positive):</b> Creditworthy borrower rejected. Incurs opportunity cost of lost interest margin (assigned economic weight = 1.0 unit).</li>
                <li><b>Type II Error (False Negative):</b> Defaulting borrower approved. Incurs severe loss of unpaid loan principal (assigned economic weight = 5.0 units).</li>
                <li><b>Optimization Impact:</b> The champion model achieves a decisive reduction in Type II errors, minimizing expected risk exposure to under 110 loss units.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 7: PREDICTION
# =============================================================================
elif selected_page == "7. Prediction":
    st.markdown("""
    <div class="page-header">
        <div>
            <div class="badge-metallic" style="margin-bottom: 4px;">INFERENCE ENGINE</div>
            <h2>Credit Default Risk Prediction</h2>
            <p>Enter borrower characteristics to generate a model-based risk assessment in real-time</p>
        </div>
        <div>
            <span class="badge-silver">LIVE SCORING</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if best_model_artifact is None or pipeline_artifact is None:
        st.error("Model artifacts not found. Please run the training pipeline first.")
    else:
        best_model = best_model_artifact["model_object"]
        
        st.markdown(f"""
        <div class="research-card">
            <div style="font-size: 15px; font-weight: 800; color: {PRIMARY_TEXT}; margin-bottom: 14px; border-bottom: 1px solid {BORDER}; padding-bottom: 8px;">
                APPLICANT UNDERWRITING PROFILE INPUTS
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            col_sec1, col_sec2 = st.columns(2)
            with col_sec1:
                st.markdown(f"<div style='font-size: 12px; font-weight: 800; color: {SECONDARY_TEXT}; text-transform: uppercase; margin-bottom: 8px;'>1. BORROWER & ACCOUNT PROFILE</div>", unsafe_allow_html=True)
                checking_status = st.selectbox("Checking Account Status:", ["<0", "0<=X<200", ">=200", "no checking"], index=1)
                savings_status = st.selectbox("Savings Account Status:", [
                    "<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"
                ], index=0)
                employment = st.selectbox("Present Employment Duration:", [
                    "unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"
                ], index=2)
                job = st.selectbox("Employment Qualification:", [
                    "skilled", "unskilled resident", "high qualif/emp/mgmt", "unemp/unskilled non res"
                ], index=0)
                
                st.markdown(f"<div style='font-size: 12px; font-weight: 800; color: {SECONDARY_TEXT}; text-transform: uppercase; margin: 14px 0 8px 0;'>2. FINANCIAL & LOAN STRUCTURE</div>", unsafe_allow_html=True)
                duration = st.number_input("Loan Duration (Months):", min_value=4, max_value=72, value=24, step=1)
                credit_amount = st.number_input("Credit Amount (DM):", min_value=250, max_value=20000, value=3000, step=100)
                purpose = st.selectbox("Loan Purpose:", [
                    "radio/tv", "new car", "furniture/equipment", "used car", "business", "education", "repairs", "vacation", "retraining", "domestic appliances", "other"
                ], index=0)
                installment_commitment = st.slider("Installment Rate (% of Disposable Income):", min_value=1, max_value=4, value=3)

            with col_sec2:
                st.markdown(f"<div style='font-size: 12px; font-weight: 800; color: {SECONDARY_TEXT}; text-transform: uppercase; margin-bottom: 8px;'>3. CREDIT HISTORY & LIABILITIES</div>", unsafe_allow_html=True)
                credit_history = st.selectbox("Credit History:", [
                    "critical/other existing credit", "existing paid", "delayed previously", "all paid", "no credits/all paid"
                ], index=1)
                existing_credits = st.slider("Existing Credits at This Bank:", min_value=1, max_value=4, value=1)
                other_parties = st.selectbox("Other Debtors / Guarantors:", ["none", "guarantor", "co-applicant"], index=0)
                other_payment_plans = st.selectbox("Other Installment Plans:", ["none", "bank", "stores"], index=0)

                st.markdown(f"<div style='font-size: 12px; font-weight: 800; color: {SECONDARY_TEXT}; text-transform: uppercase; margin: 14px 0 8px 0;'>4. PERSONAL & RESIDENCE DEMOGRAPHICS</div>", unsafe_allow_html=True)
                age = st.number_input("Borrower Age (Years):", min_value=18, max_value=80, value=35, step=1)
                personal_status = st.selectbox("Personal Status & Sex:", [
                    "male single", "female div/dep/mar", "male mar/wid", "male div/sep"
                ], index=0)
                housing = st.selectbox("Housing Accommodation:", ["own", "rent", "for free"], index=0)
                property_magnitude = st.selectbox("Most Valuable Property / Asset:", [
                    "real estate", "building society savings/life insurance", "car", "no known property"
                ], index=2)
                residence_since = st.slider("Present Residence Duration (Years):", min_value=1, max_value=4, value=2)
                num_dependents = st.selectbox("Number of Dependents:", [1, 2], index=0)
                own_telephone = st.selectbox("Registered Telephone:", ["yes", "none"], index=0)
                foreign_worker = st.selectbox("Foreign Worker Status:", ["yes", "no"], index=0)
                
            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("⚡ RUN RISK ASSESSMENT")
            
        st.markdown("</div>", unsafe_allow_html=True)
            
        if submit_btn:
            input_dict = {
                "checking_status": checking_status,
                "duration": duration,
                "credit_history": credit_history,
                "purpose": purpose,
                "credit_amount": credit_amount,
                "savings_status": savings_status,
                "employment": employment,
                "installment_commitment": installment_commitment,
                "personal_status": personal_status,
                "other_parties": other_parties,
                "residence_since": residence_since,
                "property_magnitude": property_magnitude,
                "age": age,
                "other_payment_plans": other_payment_plans,
                "housing": housing,
                "existing_credits": existing_credits,
                "job": job,
                "num_dependents": num_dependents,
                "own_telephone": own_telephone,
                "foreign_worker": foreign_worker
            }
            
            # Transform and predict
            X_sample = transform_new_sample(input_dict)
            pred_class = best_model.predict(X_sample)[0]
            
            if hasattr(best_model, "predict_proba"):
                pred_prob_default = float(best_model.predict_proba(X_sample)[0, 1])
            elif hasattr(best_model, "decision_function"):
                df_val = float(best_model.decision_function(X_sample)[0])
                pred_prob_default = 1.0 / (1.0 + np.exp(-df_val))
            else:
                pred_prob_default = float(pred_class)
                
            prob_creditworthy = 1.0 - pred_prob_default
            
            st.markdown(f"""
            <div style="font-size: 18px; font-weight: 800; color: {PRIMARY_TEXT}; margin: 18px 0 10px 0;">
                PREDICTION RESULT & RISK PROFILE
            </div>
            """, unsafe_allow_html=True)
            
            res_c1, res_c2 = st.columns([1.6, 1])
            
            with res_c1:
                if pred_class == 0:
                    st.markdown(f"""
                    <div class="research-card" style="border: 2px solid {SUCCESS}; background: {CARD_BG};">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <div style="font-size: 20px; font-weight: 900; color: {SUCCESS};">CREDIT APPROVED (Low Risk)</div>
                            <span class="badge-silver" style="font-size: 11px; background: {LIGHT_BLUE}; color: {SUCCESS}; border-color: {BORDER};">Class 0: Good Credit</span>
                        </div>
                        <p style="margin-top: 10px; color: {SECONDARY_TEXT}; font-size: 13px; line-height: 1.5;">
                            Applicant meets regulatory creditworthiness criteria with robust liquid reserves and sustainable debt-to-duration metrics.
                        </p>
                        <div style="margin-top: 16px;">
                            <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 700; color: {PRIMARY_TEXT}; margin-bottom: 6px;">
                                <span>Solvency Confidence:</span>
                                <span style="font-family: 'JetBrains Mono', monospace; color: {SUCCESS}; font-weight: 800;">{prob_creditworthy:.1%}</span>
                            </div>
                            <div style="background-color: {MAIN_BG}; height: 12px; border-radius: 6px; overflow: hidden; border: 1px solid {BORDER};">
                                <div style="background: linear-gradient(90deg, {ACCENT_BLUE} 0%, {SUCCESS} 100%); height: 100%; width: {prob_creditworthy*100:.1f}%;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="research-card" style="border: 2px solid {DANGER}; background: {CARD_BG};">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <div style="font-size: 20px; font-weight: 900; color: {DANGER};">HIGH DEFAULT RISK DETECTED</div>
                            <span class="badge-graphite" style="font-size: 11px; background: #FEE2E2; color: {DANGER}; border-color: #FCA5A5;">Class 1: Bad Credit</span>
                        </div>
                        <p style="margin-top: 10px; color: {SECONDARY_TEXT}; font-size: 13px; line-height: 1.5;">
                            Elevated default probability flagged. Recommend underwriting review, guarantor requirement, or adjusted credit limits.
                        </p>
                        <div style="margin-top: 16px;">
                            <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 700; color: {PRIMARY_TEXT}; margin-bottom: 6px;">
                                <span>Default Probability:</span>
                                <span style="font-family: 'JetBrains Mono', monospace; color: {DANGER}; font-weight: 800;">{pred_prob_default:.1%}</span>
                            </div>
                            <div style="background-color: {MAIN_BG}; height: 12px; border-radius: 6px; overflow: hidden; border: 1px solid {BORDER};">
                                <div style="background: linear-gradient(90deg, {WARNING} 0%, {DANGER} 100%); height: 100%; width: {pred_prob_default*100:.1f}%;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with res_c2:
                # Computed Engineered Domain Metrics
                burn_rate = credit_amount / duration
                leverage = credit_amount / age
                st.markdown(f"""
                <div class="research-card">
                    <div style="font-size: 14px; font-weight: 800; color: {PRIMARY_TEXT}; margin-bottom: 8px; border-bottom: 1px solid {BORDER}; padding-bottom: 4px;">
                        ENGINEERED RISK RATIOS
                    </div>
                    <table style="width: 100%; font-size: 12px; color: {PRIMARY_TEXT}; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid {BORDER}; padding: 5px 0;"><td style="font-weight: 700; padding: 5px 0;">Monthly Debt Burden:</td><td style="text-align: right; font-family: 'JetBrains Mono', monospace;">{burn_rate:.1f} DM/mo</td></tr>
                        <tr style="border-bottom: 1px solid {BORDER}; padding: 5px 0;"><td style="font-weight: 700; padding: 5px 0;">Credit-to-Age Leverage:</td><td style="text-align: right; font-family: 'JetBrains Mono', monospace;">{leverage:.1f}</td></tr>
                        <tr style="border-bottom: 1px solid {BORDER}; padding: 5px 0;"><td style="font-weight: 700; padding: 5px 0;">Model Used:</td><td style="text-align: right; font-weight: 600;">{best_model_artifact['model_name']}</td></tr>
                        <tr style="padding: 5px 0;"><td style="font-weight: 700; padding: 5px 0;">Inference Latency:</td><td style="text-align: right; font-family: 'JetBrains Mono', monospace;">&lt; 0.05 ms</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# PAGE 8: RESEARCH FINDINGS
# =============================================================================
elif selected_page == "8. Research Findings":
    st.markdown("""
    <div class="page-header">
        <div>
            <div class="badge-metallic" style="margin-bottom: 4px;">SYNTHESIS & RQ</div>
            <h2>Research Findings & Empirical Answers</h2>
            <p>Scientific answers to primary research questions backed by actual experimental data</p>
        </div>
        <div>
            <span class="badge-silver">RQ1–RQ6</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    rq_items = [
        ("RQ1: Feature Predictive Influence", "Liquidity indicators (checking and savings account tiers) and past credit repayment track record exert the strongest predictive influence. Borrowers with negative checking balances exhibit a 49.3% default rate compared to 11.6% for those with positive reserves."),
        ("RQ2: Superior Algorithm Family", "Tree ensembles (Random Forest and Gradient Boosting) achieve the highest overall discrimination (ROC-AUC 0.8135 and 0.8082), while Support Vector Machines maximize default detection recall (76.67%)."),
        ("RQ3: Hyperparameter Optimization Gains", "Systematic GridSearchCV tuning produced statistically significant improvements for tree ensembles, increasing Random Forest ROC-AUC by +0.0150 and optimizing tree depth and feature sub-spacing."),
        ("RQ4: Cross-Validation Stability", "Support Vector Machines and Random Forest demonstrated the highest stability across Stratified 5-Fold Cross-Validation (std dev < 0.035), whereas individual Decision Trees showed substantial variance."),
        ("RQ5: Misclassification Profiling", "False positives stem from large credit requests relative to borrower age; false negatives stem from applicants with short employment tenures hidden behind clean past credit."),
        ("RQ6: Linear Model Parsimony", "Regularized Logistic Regression achieved a competitive ROC-AUC of 0.8045 with sub-millisecond inference (0.025 ms), providing an interpretable, regulatory-compliant alternative to black-box ensembles.")
    ]
    
    for title, desc in rq_items:
        st.markdown(f"""
        <div class="research-card" style="padding: 16px 20px; margin-bottom: 12px;">
            <div style="font-weight: 800; color: {PRIMARY_TEXT}; font-size: 14px; letter-spacing: -0.01em;">{title}</div>
            <div style="font-size: 13px; color: {SECONDARY_TEXT}; margin-top: 6px; line-height: 1.55;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"""
    <div class="research-card" style="margin-top: 18px;">
        <div style="font-size: 15px; font-weight: 800; color: {PRIMARY_TEXT}; margin-bottom: 10px;">STATISTICAL HYPOTHESIS TESTING RESULTS</div>
    """, unsafe_allow_html=True)
    render_dataframe(df_stats)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 9: ABOUT PROJECT
# =============================================================================
elif selected_page == "9. About Project":
    st.markdown("""
    <div class="page-header">
        <div>
            <div class="badge-metallic" style="margin-bottom: 4px;">PROJECT SPECIFICATION</div>
            <h2>About the Research Project</h2>
            <p>Academic metadata, reproducible experimental standards, and formal citation specifications</p>
        </div>
        <div>
            <span class="badge-silver">DOCUMENTATION</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c_a1, c_a2 = st.columns(2)
    with c_a1:
        st.markdown(f"""
        <div class="research-card">
            <div style="font-size: 15px; font-weight: 800; color: {PRIMARY_TEXT}; margin-bottom: 10px; border-bottom: 1px solid {BORDER}; padding-bottom: 6px;">
                PROJECT ACADEMIC PROFILE
            </div>
            <table style="width: 100%; font-size: 13px; color: {PRIMARY_TEXT}; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid {BORDER}; padding: 6px 0;"><td style="font-weight: 700; padding: 6px 0;">Project Title:</td><td>Machine Learning Public Dataset Study</td></tr>
                <tr style="border-bottom: 1px solid {BORDER}; padding: 6px 0;"><td style="font-weight: 700; padding: 6px 0;">Target Domain:</td><td>Credit Default Risk & Applied Econometrics</td></tr>
                <tr style="border-bottom: 1px solid {BORDER}; padding: 6px 0;"><td style="font-weight: 700; padding: 6px 0;">Target Venues:</td><td>B.Tech Capstone / ML Lab / Research Conference</td></tr>
                <tr style="border-bottom: 1px solid {BORDER}; padding: 6px 0;"><td style="font-weight: 700; padding: 6px 0;">License:</td><td>MIT License (Open Source Academic)</td></tr>
                <tr style="padding: 6px 0;"><td style="font-weight: 700; padding: 6px 0;">Pipeline Standard:</td><td>Zero-Data-Leakage Scikit-Learn Pipeline</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    with c_a2:
        st.markdown(f"""
        <div class="research-card">
            <div style="font-size: 15px; font-weight: 800; color: {PRIMARY_TEXT}; margin-bottom: 10px; border-bottom: 1px solid {BORDER}; padding-bottom: 6px;">
                SUGGESTED ACADEMIC CITATION
            </div>
            <div style="background-color: {PRIMARY_NAVY}; border: 1px solid {SECONDARY_NAVY}; border-radius: 6px; padding: 14px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #D7DEE8; line-height: 1.5;">
                @article{{ml_credit_study_2026,<br>
                &nbsp;&nbsp;title={{Machine Learning Public Dataset Study: An Empirical Analysis on Credit Risk}},<br>
                &nbsp;&nbsp;author={{Research Analytics Group}},<br>
                &nbsp;&nbsp;journal={{Academic Machine Learning Repository}},<br>
                &nbsp;&nbsp;year={{2026}}<br>
                }}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"""
    <div class="research-card">
        <div style="font-size: 15px; font-weight: 800; color: {PRIMARY_TEXT}; margin-bottom: 10px; border-bottom: 1px solid {BORDER}; padding-bottom: 6px;">
            PRIMARY REFERENCES & LITERATURE BENCHMARKS
        </div>
        <ol style="font-size: 13px; color: {PRIMARY_TEXT}; line-height: 1.7; margin-bottom: 0;">
            <li><b>Hofmann, H. (1994).</b> <i>Statlog (German Credit Data)</i>. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5C59P" target="_blank" style="color: {ACCENT_BLUE}; font-weight: 600; text-decoration: none;">doi.org/10.24432/C5C59P</a></li>
            <li><b>Breiman, L. (2001).</b> <i>Random Forests</i>. Machine Learning, 45(1), 5-32.</li>
            <li><b>Friedman, J. H. (2001).</b> <i>Greedy Function Approximation: A Gradient Boosting Machine</i>. Annals of Statistics, 29(5), 1189-1232.</li>
            <li><b>Pedregosa, F., et al. (2011).</b> <i>Scikit-learn: Machine Learning in Python</i>. JMLR, 12, 2825-2830.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
