"""
AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement
AIML Hackathon Project | Problem Statement 26100
Organization: Chennai Petroleum Corporation Limited (CPCL)
Theme: Smart Automation

Main Streamlit Application
File: app/app.py
"""

import sys
from pathlib import Path

# Add project root to sys.path to ensure robust module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Import reusable inference and verification services from Step 5
try:
    from src.predict_compliance import (
        load_model,
        assess_bid,
        run_basic_compliance_checks,
        MODEL_FEATURE_COLUMNS
    )
except ImportError as e:
    st.error(f"Failed to import compliance prediction engine: {e}")
    st.stop()

# Import reusable document verification services from Step 7
try:
    from src.document_verification import (
        SUPPORTED_DOCUMENT_TYPES,
        EXPECTED_KEYWORDS,
        STATUS_VERIFIED,
        STATUS_WARNING,
        STATUS_EXTRACTION_FAILED,
        STATUS_UNSUPPORTED,
        extract_text_from_file,
        identify_document_type,
        verify_document_content,
        calculate_document_completeness,
        calculate_document_consistency,
        generate_document_findings
    )
except ImportError as e:
    st.error(f"Failed to import document verification engine: {e}")
    st.stop()

# Import reusable compliance explainability services from Step 8
try:
    from src.explainability import (
        explain_prediction,
        load_feature_importance_table,
        EXPLAINABILITY_DISCLAIMER
    )
except ImportError as e:
    st.error(f"Failed to import compliance explainability engine: {e}")
    st.stop()

# Import reusable review workflow services from Step 9
try:
    from src.review_workflow import (
        init_db,
        create_review_record,
        get_review_record,
        update_review_status,
        add_review_remark,
        get_review_history,
        calculate_review_statistics,
        validate_review_action,
        list_all_reviews,
        VALID_REVIEW_STATUSES,
        VALID_OFFICER_ACTIONS,
        ACTION_STATUS_TARGETS,
        WORKFLOW_DISCLAIMER
    )
except ImportError as e:
    st.error(f"Failed to import review workflow engine: {e}")
    st.stop()



# -------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="GeM Bid Compliance Verification",
    layout="wide"
)

# Professional Government / Enterprise Styling
st.markdown(
    """
    <style>
    /* =========================================================
       1. GLOBAL ENTERPRISE THEME & BASE STYLES
       ========================================================= */
    .stApp {
        background-color: #f8fafc !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        color: #0f172a !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    div.block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6,
    .stSubheader,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4 {
        color: #0f2a4a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }
    div[data-testid="stMarkdownContainer"] h2,
    .stSubheader {
        font-size: 1.4rem !important;
        margin-top: 1.1rem !important;
        margin-bottom: 0.2rem !important;
    }
    div[data-testid="stMarkdownContainer"] h3 {
        font-size: 1.2rem !important;
        margin-top: 0.9rem !important;
        margin-bottom: 0.2rem !important;
    }
    div[data-testid="stMarkdownContainer"] h4 {
        font-size: 1.02rem !important;
        margin-top: 0.75rem !important;
        margin-bottom: 0.2rem !important;
        color: #1e3a5f !important;
    }
    div[data-testid="stCaptionContainer"], .stCaption {
        color: #475569 !important;
        font-size: 0.85rem !important;
    }

    /* =========================================================
       2. GLOBAL HEADER BANNER & DISCLAIMER
       ========================================================= */
    .enterprise-header-banner {
        margin-top: 6px !important;
        background: linear-gradient(135deg, #091e3a 0%, #0f2a4a 55%, #1e3a8a 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(10, 25, 47, 0.12);
        color: #ffffff;
    }
    .header-main-title {
        font-size: 1.95rem;
        font-weight: 700;
        color: #ffffff !important;
        line-height: 1.25;
        margin-bottom: 6px;
        letter-spacing: -0.015em;
    }
    .header-sub-title {
        font-size: 1.02rem;
        color: #bfdbfe !important;
        font-weight: 500;
        margin: 0;
        line-height: 1.4;
    }

    /* Prototype Demo Disclaimer Box */
    .enterprise-disclaimer-box {
        background-color: #f1f5f9;
        border-left: 4px solid #0284c7;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.86rem;
        color: #334155;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        transition: transform 160ms ease, box-shadow 160ms ease;
    }
    .enterprise-disclaimer-box:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .disclaimer-badge {
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #bae6fd;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        white-space: nowrap;
    }

    /* Backward compatibility for previous classes */
    .main-title { font-size: 1.95rem; font-weight: 700; color: #0f2a4a; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #1e3a8a; font-weight: 500; margin-bottom: 0.4rem; }
    .hackathon-badge {
        display: inline-block;
        background-color: #f1f5f9;
        color: #0f2a4a;
        border: 1px solid #cbd5e1;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .disclaimer-banner {
        background-color: #f1f5f9;
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.86rem;
        color: #334155;
        margin-bottom: 1.2rem;
    }

    /* =========================================================
       3. METRIC & KPI CARDS (HIGH CONTRAST & READABLE)
       ========================================================= */
    .kpi-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05) !important;
        text-align: left !important;
        min-height: 104px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease !important;
        box-sizing: border-box !important;
    }
    .kpi-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08) !important;
        border-color: #cbd5e1 !important;
    }
    .kpi-title {
        font-size: 0.78rem !important;
        color: #475569 !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        margin-bottom: 4px !important;
        letter-spacing: 0.03em !important;
    }
    .kpi-value {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        line-height: 1.2 !important;
        letter-spacing: -0.01em !important;
    }
    .kpi-sub {
        font-size: 0.78rem !important;
        color: #64748b !important;
        font-weight: 500 !important;
        margin-top: 4px !important;
    }

    /* Native Streamlit Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05) !important;
        min-height: 104px !important;
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08) !important;
        border-color: #cbd5e1 !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] p {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
        margin-bottom: 4px !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        line-height: 1.2 !important;
    }
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] > div {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }

    /* Semantic Risk Accents (Subtle borders & badges, NOT screaming solid fills) */
    .risk-low {
        color: #16a34a !important;
        font-weight: 700 !important;
    }
    .risk-med {
        color: #d97706 !important;
        font-weight: 700 !important;
    }
    .risk-high {
        color: #dc2626 !important;
        font-weight: 700 !important;
    }
    .risk-card-low,
    .kpi-card.risk-card-low {
        border-left: 4px solid #16a34a !important;
    }
    .risk-card-med,
    .kpi-card.risk-card-med {
        border-left: 4px solid #d97706 !important;
    }
    .risk-card-high,
    .kpi-card.risk-card-high {
        border-left: 4px solid #dc2626 !important;
    }

    /* =========================================================
       4. STANDARDIZED REUSABLE CARDS & CONTAINERS
       ========================================================= */
    .section-card,
    .info-card,
    .risk-card,
    .explanation-card,
    .workflow-card,
    .officer-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05) !important;
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease !important;
        will-change: transform, box-shadow;
    }
    .section-card:hover,
    .info-card:hover,
    .risk-card:hover,
    .explanation-card:hover,
    .workflow-card:hover,
    .officer-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08) !important;
        border-color: #cbd5e1 !important;
    }

    .indicator-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        transition: transform 160ms ease, box-shadow 160ms ease !important;
    }
    .indicator-card:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 3px 8px rgba(15, 23, 42, 0.06) !important;
    }

    .officer-alert {
        background-color: #fef2f2 !important;
        border: 1.5px solid #fecaca !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        margin-top: 15px !important;
        color: #991b1b !important;
        font-weight: 600 !important;
        transition: transform 180ms ease, box-shadow 180ms ease !important;
    }
    .officer-alert:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(153, 27, 27, 0.12) !important;
    }

    /* Badges & Pills */
    .badge-passed {
        background-color: #ecfdf5 !important;
        color: #065f46 !important;
        border: 1px solid #a7f3d0 !important;
        padding: 3px 10px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        display: inline-block !important;
        transition: transform 160ms ease !important;
    }
    .badge-passed:hover {
        transform: translateY(-1px) !important;
    }
    .badge-warning {
        background-color: #fffbeb !important;
        color: #92400e !important;
        border: 1px solid #fde68a !important;
        padding: 3px 10px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        display: inline-block !important;
        transition: transform 160ms ease !important;
    }
    .badge-warning:hover {
        transform: translateY(-1px) !important;
    }
    .badge-failed {
        background-color: #fef2f2 !important;
        color: #991b1b !important;
        border: 1px solid #fecaca !important;
        padding: 3px 10px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        display: inline-block !important;
        transition: transform 160ms ease !important;
    }
    .badge-failed:hover {
        transform: translateY(-1px) !important;
    }
    .keyword-pill {
        display: inline-block !important;
        background-color: #f1f5f9 !important;
        color: #334155 !important;
        padding: 2px 8px !important;
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        margin: 2px 4px 2px 0 !important;
        border: 1px solid #e2e8f0 !important;
        transition: background-color 160ms ease, border-color 160ms ease, transform 160ms ease !important;
    }
    .keyword-pill:hover {
        background-color: #e2e8f0 !important;
        border-color: #cbd5e1 !important;
        transform: translateY(-1px) !important;
    }
    .doc-preview-box {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-size: 0.82rem !important;
        color: #334155 !important;
        max-height: 250px !important;
        overflow-y: auto !important;
        white-space: pre-wrap !important;
        transition: border-color 180ms ease, box-shadow 180ms ease !important;
    }
    .doc-preview-box:hover {
        border-color: #94a3b8 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
    }

    /* =========================================================
       5. PROFESSIONAL BUTTONS
       ========================================================= */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 8px 18px !important;
        min-height: 42px !important;
        transition: transform 160ms ease, box-shadow 160ms ease, background-color 160ms ease, border-color 160ms ease !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.1) !important;
    }
    div.stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08) !important;
    }

    /* Primary Button */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"] {
        background-color: #0f2a4a !important;
        color: #ffffff !important;
        border: 1px solid #0f2a4a !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #1e3a8a !important;
        border-color: #1e3a8a !important;
        box-shadow: 0 4px 12px rgba(15, 42, 74, 0.25) !important;
        color: #ffffff !important;
    }

    /* Secondary Button */
    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="baseButton-secondary"] {
        background-color: #ffffff !important;
        color: #0f2a4a !important;
        border: 1px solid #cbd5e1 !important;
    }
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background-color: #f8fafc !important;
        border-color: #0f2a4a !important;
        color: #0f2a4a !important;
    }

    /* =========================================================
       6. SIDEBAR STYLING & NAVIGATION
       ========================================================= */
    section[data-testid="stSidebar"] {
        width: 270px !important;
        min-width: 270px !important;
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    @media (min-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 270px !important;
            min-width: 270px !important;
        }
        section[data-testid="stSidebar"] > div:first-child {
            width: 270px !important;
            min-width: 270px !important;
        }
    }
    section[data-testid="stSidebar"] div.block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
    }
    .sidebar-brand-container {
        padding: 4px 4px 6px 4px;
    }
    .sidebar-brand-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: #0f2a4a;
        letter-spacing: -0.015em;
        line-height: 1.25;
    }
    .sidebar-brand-sub {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 3px;
        letter-spacing: 0.01em;
    }
    .sidebar-divider {
        height: 1px;
        background-color: #e2e8f0;
        margin: 12px 4px 10px 4px;
    }
    .sidebar-nav-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0 4px;
        margin-bottom: 6px;
    }

    /* Sidebar Radio Group Navigation */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 3px !important;
    }
    /* Hide the circular radio input dot for clean enterprise tab look */
    section[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stWidgetSelectionLabel"] > div:first-child,
    section[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        transition: all 160ms ease !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        margin-bottom: 2px !important;
        border-left: 3px solid transparent !important;
        background-color: transparent !important;
        color: #334155 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        min-height: 38px !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        margin: 0 !important;
        font-size: 0.88rem !important;
        font-weight: inherit !important;
        color: inherit !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #f1f5f9 !important;
        border-left: 3px solid #94a3b8 !important;
        color: #0f2a4a !important;
        padding-left: 14px !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #edf4fc !important;
        border-left: 3px solid #0f2a4a !important;
        color: #0f2a4a !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):hover {
        background-color: #e5effb !important;
        border-left: 3px solid #0a192f !important;
    }

    /* =========================================================
       7. DATA TABLES & DATAFRAMES
       ========================================================= */
    table {
        width: 100% !important;
        border-collapse: collapse !important;
        font-size: 0.88rem !important;
        color: #1e293b !important;
    }
    table thead th {
        background-color: #f1f5f9 !important;
        color: #0f2a4a !important;
        font-weight: 600 !important;
        text-align: left !important;
        padding: 10px 14px !important;
        border-bottom: 2px solid #cbd5e1 !important;
        letter-spacing: 0.02em !important;
    }
    table tbody td {
        padding: 10px 14px !important;
        border-bottom: 1px solid #e2e8f0 !important;
        color: #334155 !important;
    }
    table tbody tr:nth-child(even) {
        background-color: #f8fafc !important;
    }
    table tbody tr:hover {
        background-color: #f1f5f9 !important;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
        transition: box-shadow 180ms ease, border-color 180ms ease !important;
    }
    div[data-testid="stDataFrame"]:hover {
        border-color: #cbd5e1 !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        background-color: #ffffff !important;
        transition: border-color 180ms ease, box-shadow 180ms ease !important;
    }
    div[data-testid="stExpander"]:hover {
        border-color: #cbd5e1 !important;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05) !important;
    }
    div[data-testid="stExpander"] summary {
        border-radius: 8px !important;
        transition: background-color 180ms ease !important;
    }
    div[data-testid="stExpander"] summary:hover {
        background-color: #f8fafc !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------------------
# 2. CACHED RESOURCE LOADERS
# -------------------------------------------------------------

# -------------------------------------------------------------
# 2. CACHED RESOURCE LOADERS & PERFORMANCE ACCELERATORS
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Caches the loaded pipeline to prevent repeated disk I/O."""
    model_path = PROJECT_ROOT / "models" / "compliance_model.pkl"
    if not model_path.exists():
        st.error(
            "Trained compliance model artifact not found (models/compliance_model.pkl). "
            "Please ensure the model file is available in the models/ directory."
        )
        return None
    try:
        return load_model(str(model_path))
    except Exception as e:
        st.error(f"Error loading trained compliance model: {e}")
        return None


@st.cache_data(show_spinner=False)
def get_cached_dataset():
    """Loads the compliance dataset in read-only mode."""
    data_path = PROJECT_ROOT / "data" / "bid_compliance_dataset.csv"
    if not data_path.exists():
        st.error(
            "Demo procurement dataset not found (data/bid_compliance_dataset.csv). "
            "Please ensure the dataset file is available in the data/ directory."
        )
        return None
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        return None


@st.cache_data(show_spinner=False)
def get_model_comparison():
    """Loads precomputed model comparison metrics."""
    comp_path = PROJECT_ROOT / "data" / "processed" / "model_comparison.csv"
    if comp_path.exists():
        return pd.read_csv(comp_path)
    return None


@st.cache_data(show_spinner=False)
def get_feature_importance():
    """Loads precomputed feature importance rankings."""
    fi_path = PROJECT_ROOT / "data" / "processed" / "feature_importance.csv"
    if fi_path.exists():
        return pd.read_csv(fi_path)
    return None


@st.cache_data(show_spinner=False)
def get_dashboard_kpis():
    """Precomputes summary metrics for Dashboard to eliminate redundant DataFrame scans."""
    df = get_cached_dataset()
    if df is None:
        return None
    total_bids = len(df)
    compliant_bids = int((df["actual_compliance"] == 1).sum())
    non_compliant_bids = int((df["actual_compliance"] == 0).sum())
    compliance_rate = (compliant_bids / total_bids) * 100
    low_risk = int((df["risk_level"] == "Low").sum())
    med_risk = int((df["risk_level"] == "Medium").sum())
    high_risk = int((df["risk_level"] == "High").sum())
    avg_score = float(df["compliance_score"].mean())
    avg_delay = float(df["verification_delay_days"].mean())
    return {
        "total_bids": total_bids,
        "compliant_bids": compliant_bids,
        "non_compliant_bids": non_compliant_bids,
        "compliance_rate": compliance_rate,
        "low_risk": low_risk,
        "med_risk": med_risk,
        "high_risk": high_risk,
        "avg_score": avg_score,
        "avg_delay": avg_delay
    }


@st.cache_resource(show_spinner=False)
def get_dashboard_static_charts():
    """Caches all 6 Dashboard Plotly charts to make navigation instantaneous."""
    df = get_cached_dataset()
    if df is None:
        return None, None, None, None, None, None

    kpis = get_dashboard_kpis()
    compliant_bids = kpis["compliant_bids"]
    non_compliant_bids = kpis["non_compliant_bids"]
    low_risk = kpis["low_risk"]
    med_risk = kpis["med_risk"]
    high_risk = kpis["high_risk"]

    target_df = pd.DataFrame({
        "Category": ["Compliant", "Non-Compliant / Review"],
        "Count": [compliant_bids, non_compliant_bids]
    })
    fig_target = px.pie(
        target_df,
        values="Count",
        names="Category",
        title="Compliance Classification Distribution",
        color="Category",
        color_discrete_map={"Compliant": "#2a9d8f", "Non-Compliant / Review": "#e76f51"},
        hole=0.4
    )
    fig_target.update_traces(
        hovertemplate="<b>%{label}</b><br>Submissions: %{value:,}<br>Share: %{percent}<extra></extra>"
    )
    fig_target.update_layout(margin=dict(t=40, b=20, l=20, r=20))
    apply_chart_hover_theme(fig_target)

    elig_counts = df["tender_eligibility"].value_counts().reset_index()
    elig_counts.columns = ["Eligibility Status", "Count"]
    fig_elig = px.bar(
        elig_counts,
        x="Eligibility Status",
        y="Count",
        title="Tender Eligibility Criteria Status",
        color="Eligibility Status",
        color_discrete_map={"Eligible": "#2a9d8f", "Requires Review": "#f4a261", "Not Eligible": "#e76f51"},
        text="Count"
    )
    fig_elig.update_traces(
        hovertemplate="<b>%{x}</b><br>Submissions: %{y:,}<extra></extra>"
    )
    fig_elig.update_layout(margin=dict(t=40, b=20, l=20, r=20), yaxis_title="Submissions")
    apply_chart_hover_theme(fig_elig)

    risk_df = pd.DataFrame({
        "Risk Level": ["Low", "Medium", "High"],
        "Count": [low_risk, med_risk, high_risk]
    })
    fig_risk = px.bar(
        risk_df,
        x="Risk Level",
        y="Count",
        title="Prototype AI Screening Risk Breakdown",
        color="Risk Level",
        color_discrete_map={"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e76f51"},
        text="Count"
    )
    fig_risk.update_traces(
        hovertemplate="<b>%{x} Risk</b><br>Submissions: %{y:,}<extra></extra>"
    )
    fig_risk.update_layout(margin=dict(t=40, b=20, l=20, r=20), yaxis_title="Submissions")
    apply_chart_hover_theme(fig_risk)

    fig_score = px.histogram(
        df,
        x="compliance_score",
        nbins=25,
        title="Compliance Score Spread (0 - 100)",
        color="risk_level",
        color_discrete_map={"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e76f51"},
        opacity=0.8
    )
    fig_score.update_traces(
        hovertemplate="Score: %{x}<br>Bids Count: %{y:,}<extra></extra>"
    )
    fig_score.update_layout(margin=dict(t=40, b=20, l=20, r=20), xaxis_title="Compliance Score", yaxis_title="Bids Count")
    apply_chart_hover_theme(fig_score)

    fig_delay = px.histogram(
        df,
        x="verification_delay_days",
        nbins=20,
        title="Verification Delay Turnaround (Days)",
        color_discrete_sequence=["#3a86ff"],
        opacity=0.8
    )
    fig_delay.update_traces(
        hovertemplate="Turnaround: %{x} Days<br>Bids Count: %{y:,}<extra></extra>"
    )
    fig_delay.update_layout(margin=dict(t=40, b=20, l=20, r=20), xaxis_title="Turnaround Days", yaxis_title="Bids Count")
    apply_chart_hover_theme(fig_delay)

    fig_cons = px.histogram(
        df,
        x="documentation_consistency",
        nbins=20,
        title="Documentation Cross-Record Consistency (%)",
        color_discrete_sequence=["#2a9d8f"],
        opacity=0.8
    )
    fig_cons.update_traces(
        hovertemplate="Consistency: %{x}%<br>Bids Count: %{y:,}<extra></extra>"
    )
    fig_cons.update_layout(margin=dict(t=40, b=20, l=20, r=20), xaxis_title="Consistency Score (%)", yaxis_title="Bids Count")
    apply_chart_hover_theme(fig_cons)

    return fig_target, fig_elig, fig_risk, fig_score, fig_delay, fig_cons


@st.cache_data(show_spinner=False)
def get_cached_assessment(payload_tuple: tuple) -> dict:
    """Caches ML inference and rule verification by input payload signature."""
    model = get_cached_model()
    payload = dict(payload_tuple)
    return assess_bid(payload, model=model)


@st.cache_data(show_spinner=False)
def get_cached_explanation(payload_tuple: tuple, doc_tuple: tuple = None) -> dict:
    """Caches explainability generation to avoid re-running attribution on reruns."""
    model = get_cached_model()
    fi_table = get_feature_importance()
    payload = dict(payload_tuple)
    doc_payload = dict(doc_tuple) if doc_tuple else None
    return explain_prediction(
        payload,
        model=model,
        feature_importance_df=fi_table,
        doc_payload=doc_payload
    )


@st.cache_resource(show_spinner=False)
def get_cached_signals_chart(signals_tuple: tuple):
    """Caches the horizontal feature signal bar chart."""
    signals = [dict(s) for s in signals_tuple]
    signals_df = pd.DataFrame(signals)
    fig_signals = px.bar(
        signals_df.iloc[::-1],
        x="importance",
        y="feature_label",
        orientation="h",
        color="signal_type",
        color_discrete_map={"Positive Signal": "#2a9d8f", "Risk Signal": "#e76f51"},
        title="Top Contributing Feature Signals (Directional Impact)",
        labels={"importance": "Model Reliance (Feature Importance)", "feature_label": "Feature / Condition", "signal_type": "Signal Direction"}
    )
    fig_signals.update_traces(
        hovertemplate="<b>%{y}</b><br>Reliance: %{x:.4f}<extra></extra>"
    )
    fig_signals.update_layout(margin=dict(t=40, b=20, l=220, r=20), height=380)
    apply_chart_hover_theme(fig_signals)
    return fig_signals


@st.cache_data(show_spinner=False)
def get_cached_extracted_text(file_bytes: bytes, file_name: str) -> dict:
    """Caches text extraction output so identical files are not re-parsed."""
    return extract_text_from_file(file_bytes, file_name)


@st.cache_data(show_spinner=False)
def get_cached_doc_completeness(submitted_types: tuple, required_types: tuple) -> dict:
    """Caches document completeness calculation."""
    return calculate_document_completeness(list(submitted_types), list(required_types))


@st.cache_data(show_spinner=False)
def get_cached_doc_consistency(doc_fingerprints: tuple) -> dict:
    """Caches cross-document entity consistency check."""
    docs = [{"document_type": dt, "file_name": fn, "text": txt} for dt, fn, txt in doc_fingerprints]
    return calculate_document_consistency(docs)


@st.cache_data(show_spinner=False)
def get_base_risk_kpis():
    """Caches base risk metrics for Risk Analysis page."""
    df = get_cached_dataset()
    if df is None:
        return 0, 0.0, 0, 0.0, 0, 0.0
    low_c = int((df["risk_level"] == "Low").sum())
    low_rate = float(df[df["risk_level"] == "Low"]["actual_compliance"].mean() * 100) if low_c > 0 else 0.0
    med_c = int((df["risk_level"] == "Medium").sum())
    med_rate = float(df[df["risk_level"] == "Medium"]["actual_compliance"].mean() * 100) if med_c > 0 else 0.0
    high_c = int((df["risk_level"] == "High").sum())
    high_rate = float(df[df["risk_level"] == "High"]["actual_compliance"].mean() * 100) if high_c > 0 else 0.0
    return low_c, low_rate, med_c, med_rate, high_c, high_rate


@st.cache_resource(show_spinner=False)
def get_base_risk_analysis_charts():
    """Caches the 4 unfiltered risk analysis charts."""
    df = get_cached_dataset()
    if df is None:
        return None, None, None, None

    risk_dist = df["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
    risk_dist.columns = ["Risk Level", "Bids Count"]
    fig1 = px.bar(
        risk_dist,
        x="Risk Level",
        y="Bids Count",
        color="Risk Level",
        color_discrete_map={"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e76f51"},
        title="Prototype AI Screening Risk Distribution",
        text="Bids Count"
    )
    fig1.update_traces(textposition='outside', hovertemplate="<b>%{x} Risk</b><br>Bids: %{y:,}<extra></extra>")
    fig1.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=320)
    apply_chart_hover_theme(fig1)

    elig_risk = df.groupby(["risk_level", "tender_eligibility"], observed=False).size().reset_index(name="Count")
    fig2 = px.bar(
        elig_risk,
        x="risk_level",
        y="Count",
        color="tender_eligibility",
        barmode="group",
        title="Tender Eligibility Breakdown by Risk Level",
        color_discrete_map={"Eligible": "#2a9d8f", "Requires Review": "#e9c46a", "Not Eligible": "#e76f51"}
    )
    fig2.update_traces(hovertemplate="<b>%{x} Risk (%{data.name})</b><br>Count: %{y:,}<extra></extra>")
    fig2.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=320, xaxis_title="Risk Level", yaxis_title="Bids Count")
    apply_chart_hover_theme(fig2)

    risk_score = df.groupby("risk_level", observed=False)["compliance_score"].mean().reindex(["Low", "Medium", "High"]).reset_index()
    risk_score.columns = ["Risk Level", "Avg Compliance Score"]
    fig3 = px.bar(
        risk_score,
        x="Risk Level",
        y="Avg Compliance Score",
        color="Risk Level",
        color_discrete_map={"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e76f51"},
        title="Average Compliance Score (0–100) by Risk Level",
        text="Avg Compliance Score"
    )
    fig3.update_traces(texttemplate='%{text:.1f}', textposition='outside', hovertemplate="<b>%{x} Risk</b><br>Avg Score: %{y:.1f} / 100<extra></extra>")
    fig3.update_layout(yaxis=dict(range=[0, 110]), margin=dict(t=40, b=20, l=20, r=20), height=320)
    apply_chart_hover_theme(fig3)

    risk_delay = df.groupby("risk_level", observed=False)["verification_delay_days"].mean().reindex(["Low", "Medium", "High"]).reset_index()
    risk_delay.columns = ["Risk Level", "Avg Delay Days"]
    fig4 = px.bar(
        risk_delay,
        x="Risk Level",
        y="Avg Delay Days",
        color="Risk Level",
        color_discrete_map={"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e76f51"},
        title="Average Verification Delay Turnaround (Days) by Risk Level",
        text="Avg Delay Days"
    )
    fig4.update_traces(texttemplate='%{text:.1f}d', textposition='outside', hovertemplate="<b>%{x} Risk</b><br>Avg Turnaround: %{y:.1f} Days<extra></extra>")
    fig4.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=320)
    apply_chart_hover_theme(fig4)

    return fig1, fig2, fig3, fig4


@st.cache_resource(show_spinner=False)
def get_feature_importance_chart():
    """Caches the static Top 15 Feature Importances chart."""
    fi_data = get_feature_importance()
    if fi_data is None:
        return None
    top15 = fi_data.head(15).iloc[::-1]
    fig_fi = px.bar(
        top15,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Tealgrn",
        title="Top 15 Feature Importances (Random Forest Gini Impurity)"
    )
    fig_fi.update_traces(
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    )
    fig_fi.update_layout(margin=dict(t=40, b=20, l=150, r=20), height=380)
    apply_chart_hover_theme(fig_fi)
    return fig_fi

def apply_chart_hover_theme(fig):
    """
    Applies clean enterprise styling to Plotly figures without modifying underlying data:
    - Pure white card background
    - Crisp subtle gridlines (#f1f5f9)
    - Dark readable labels (#1e293b / #475569)
    - High-contrast dark navy tooltip hoverlabels (#0f172a / #ffffff)
    """
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color="#1e293b",
            size=12
        ),
        title_font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color="#0f2a4a",
            size=14
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            font_size=12,
            font_family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            font_color="#ffffff",
            bordercolor="#334155"
        ),
        margin=dict(l=40, r=30, t=50, b=40)
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#f1f5f9",
        tickfont=dict(color="#475569", size=11),
        title_font=dict(color="#0f2a4a", size=12)
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#f1f5f9",
        tickfont=dict(color="#475569", size=11),
        title_font=dict(color="#0f2a4a", size=12)
    )
    return fig


# -------------------------------------------------------------
# 3. HEADER COMPONENT
# -------------------------------------------------------------
def render_header():
    st.markdown(
        """
        <div class="enterprise-header-banner">
            <div class="header-main-title">
                AI-Powered Integrated Bid Compliance<br>Verification Platform
            </div>
            <div class="header-sub-title">
                AI-Assisted Procurement Compliance Screening &amp; Officer Review
            </div>
        </div>
        <div class="enterprise-disclaimer-box">
            <span class="disclaimer-badge">DEMO NOTICE</span>
            <span><strong>Prototype / Demo System</strong> — AI outputs support procurement review and do not replace authorized officer decisions.</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -------------------------------------------------------------
render_header()

NAV_PAGES = [
    "Dashboard",
    "Bid Verification",
    "Document Verification",
    "Risk Analysis",
    "Model Insights",
    "Officer Review",
    "About System"
]

# Handle programmatic redirection to Officer Review or other views
if "nav_target" in st.session_state:
    target_page = st.session_state.pop("nav_target")
    if target_page in NAV_PAGES:
        st.session_state["nav_selection"] = target_page

if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "Dashboard"

st.sidebar.markdown(
    """
    <div class="sidebar-brand-container">
        <div class="sidebar-brand-title">Bid Compliance AI</div>
        <div class="sidebar-brand-sub">Procurement Intelligence Platform</div>
    </div>
    <div class="sidebar-divider"></div>
    <div class="sidebar-nav-label">NAVIGATION</div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Select System View:",
    NAV_PAGES,
    key="nav_selection",
    label_visibility="collapsed"
)




def render_dashboard_page():
    st.subheader("Procurement Compliance Intelligence Dashboard")
    st.write("AI-assisted screening of bidder compliance, documentation, risk indicators, and officer review status.")

    df = get_cached_dataset()

    if df is not None:
        kpis = get_dashboard_kpis()
        if kpis is None:
            return
        total_bids = kpis["total_bids"]
        compliant_bids = kpis["compliant_bids"]
        non_compliant_bids = kpis["non_compliant_bids"]
        compliance_rate = kpis["compliance_rate"]

        low_risk = kpis["low_risk"]
        med_risk = kpis["med_risk"]
        high_risk = kpis["high_risk"]

        avg_score = kpis["avg_score"]
        avg_delay = kpis["avg_delay"]

        # Retrieve Review Workflow statistics from SQLite (kept dynamic)
        rev_stats = calculate_review_statistics()

        # KPI Row 1: Primary Counts & Risk Tiers
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Bids</div><div class="kpi-value">{total_bids:,}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Compliant</div><div class="kpi-value risk-low">{compliant_bids:,}</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Non-Compliant / Review</div><div class="kpi-value risk-high">{non_compliant_bids:,}</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="kpi-card risk-card-low"><div class="kpi-title">Low Risk</div><div class="kpi-value risk-low">{low_risk:,}</div></div>', unsafe_allow_html=True)
        with k5:
            st.markdown(f'<div class="kpi-card risk-card-med"><div class="kpi-title">Medium Risk</div><div class="kpi-value risk-med">{med_risk:,}</div></div>', unsafe_allow_html=True)
        with k6:
            st.markdown(f'<div class="kpi-card risk-card-high"><div class="kpi-title">High Risk</div><div class="kpi-value risk-high">{high_risk:,}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        # KPI Row 2: Performance & Officer Queue Metrics
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("Overall Compliance Rate", f"{compliance_rate:.1f}%", help="Percentage of bids identified as compliant in the synthetic dataset.")
        with m2:
            st.metric("Average Compliance Score", f"{avg_score:.1f} / 100", help="Mean composite compliance score across all dataset bids.")
        with m3:
            st.metric("Average Verification Delay", f"{avg_delay:.1f} Days", help="Average days taken for preliminary verification checks.")
        with m4:
            st.metric("Total Officer Reviews", f"{rev_stats['total_reviews']}", help="Total bids currently tracked in the officer review queue.")
        with m5:
            st.metric("Under Active Review", f"{rev_stats['under_review']}", help="Bids actively undergoing officer examination.")
        with m6:
            st.metric("Escalated", f"{rev_stats['escalated']}", help="Bids flagged for higher committee or legal scrutiny.")

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

        # AI Screening Flow Visual Section
        st.markdown("#### 🔄 AI-Assisted Screening Pipeline Flow")
        st.caption("End-to-end intelligent compliance screening and human-in-the-loop governance architecture.")
        st.markdown(
            """
            <div class="section-card" style="background-color: #f8fafc; border: 1px solid #cbd5e0; border-radius: 8px; padding: 12px 16px; margin: 10px 0 20px 0; text-align: center; font-size: 0.9rem; font-weight: 500; color: #2d3748;">
                Tender &nbsp;→&nbsp; 
                Bid Submission &nbsp;→&nbsp; 
                Document Verification &nbsp;→&nbsp; 
                AI Prediction &nbsp;→&nbsp; 
                Rule Checks &nbsp;→&nbsp; 
                Risk Assessment &nbsp;→&nbsp; 
                Officer Review &nbsp;→&nbsp; 
                Audit Trail
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        fig_target, fig_elig, fig_risk, fig_score, fig_delay, fig_cons = get_dashboard_static_charts()

        # Section 1: Compliance Overview
        st.markdown("### 1. Compliance Overview")
        c1, c2 = st.columns(2)
        with c1:
            if fig_target:
                st.plotly_chart(fig_target, use_container_width=True)
        with c2:
            if fig_elig:
                st.plotly_chart(fig_elig, use_container_width=True)

        # Section 2: Risk Distribution
        st.markdown("### 2. Risk Distribution")
        c3, c4 = st.columns(2)
        with c3:
            if fig_risk:
                st.plotly_chart(fig_risk, use_container_width=True)
        with c4:
            if fig_score:
                st.plotly_chart(fig_score, use_container_width=True)

        # Section 3: Verification Performance
        st.markdown("### 3. Verification Performance")
        c5, c6 = st.columns(2)
        with c5:
            if fig_delay:
                st.plotly_chart(fig_delay, use_container_width=True)
        with c6:
            if fig_cons:
                st.plotly_chart(fig_cons, use_container_width=True)

        # Section 4: Officer Review Workflow
        st.markdown("### 4. Officer Review Workflow")
        st.caption("Active human-in-the-loop review queue and audit log status in local SQLite storage.")

        rw1, rw2, rw3 = st.columns([1.5, 1.5, 3])
        with rw1:
            st.metric("Review Completion Rate", f"{rev_stats['completion_rate']:.1f}%")
        with rw2:
            st.metric("Total Reviews Logged", f"{rev_stats['total_reviews']}")
        with rw3:
            st.info(
                f"Queue Status: **{rev_stats['not_reviewed']}** Not Reviewed | "
                f"**{rev_stats['under_review']}** Under Active Review | "
                f"**{rev_stats['reviewed']}** Reviewed | "
                f"**{rev_stats['escalated']}** Escalated.\n\n"
                "👉 Use the **Officer Review** page in the sidebar to review bids and record actions."
            )


def render_bid_verification_page():
    st.subheader("Bid Compliance Verification")
    st.write("Evaluate structured bidder information using the trained compliance model and prototype rule checks.")

    df_demo = get_cached_dataset()
    model = get_cached_model()

    # Convenience Sample Selector for Demonstration
    st.markdown("#### ⚡ QUICK DEMO PROFILES")
    preset_cols = st.columns([3, 1, 1, 1, 1])
    sample_choice = preset_cols[0].selectbox(
        "Select Archetypal Test Case:",
        [
            "Compliant Bidder (Bharat Data Systems)",
            "Incomplete Documentation (Metro Heavy Eng)",
            "High-Risk Bidder (XYZ Procurement)",
            "Blacklisted Vendor (Pacific BuildTech)",
            "Debarred Vendor (Pacific Electro-Mech)",
            "Ineligible Bidder (Deccan Civil Works)",
            "Custom Input (Manual Form)"
        ]
    )

    # Defaults dictionary for form population
    state_district_map = {
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem"],
        "Kerala": ["Thiruvananthapuram", "Ernakulam", "Kozhikode", "Thrissur", "Kollam"],
        "Karnataka": ["Bengaluru Urban", "Mysuru", "Hubballi-Dharwad", "Mangaluru", "Belagavi"],
        "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Kurnool"],
        "Telangana": ["Hyderabad", "Warangal", "Medchal-Malkajgiri", "Rangareddy", "Nizamabad"],
        "Maharashtra": ["Mumbai City", "Pune", "Nagpur", "Thane", "Nashik"],
        "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
        "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer"],
        "Delhi": ["New Delhi", "North Delhi", "South Delhi", "West Delhi", "East Delhi"],
        "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Noida", "Prayagraj"],
        "West Bengal": ["Kolkata", "Howrah", "North 24 Parganas", "Darjeeling", "Siliguri"],
        "Odisha": ["Khordha (Bhubaneswar)", "Cuttack", "Sundargarh", "Ganjam", "Sambalpur"]
    }
    states_list = list(state_district_map.keys())
    project_types_list = [
        "Infrastructure", "Construction", "Electrical", "Mechanical", "IT Services",
        "Facility Management", "Transportation", "Consultancy", "Equipment Supply", "Maintenance"
    ]

    # Pre-fill values based on sample choice
    if "compliant bidder" in sample_choice.lower() and "incomplete" not in sample_choice.lower() and df_demo is not None:
        init_row = df_demo[df_demo["bid_id"] == "BID0001"].iloc[0].to_dict()
    elif ("incomplete" in sample_choice.lower() or "documentation gaps" in sample_choice.lower()) and df_demo is not None:
        init_row = df_demo[df_demo["bid_id"] == "BID0008"].iloc[0].to_dict()
    elif "high-risk" in sample_choice.lower() and df_demo is not None:
        init_row = df_demo[df_demo["bid_id"] == "BID0007"].iloc[0].to_dict()
    elif "blacklisted" in sample_choice.lower() and df_demo is not None:
        init_row = df_demo[df_demo["bid_id"] == "BID0073"].iloc[0].to_dict()
    elif "debarred" in sample_choice.lower() and df_demo is not None:
        init_row = df_demo[df_demo["bid_id"] == "BID0033"].iloc[0].to_dict()
    elif "ineligible" in sample_choice.lower() and df_demo is not None:
        init_row = df_demo[df_demo["bid_id"] == "BID0021"].iloc[0].to_dict()
    else:
        init_row = {
            "project_type": "IT Services",
            "state": "Maharashtra",
            "district": "Mumbai City",
            "bid_value_lakhs": 45.0,
            "gst_registration": "Verified",
            "gst_return_filing": "Filed",
            "pan_verification": "Verified",
            "udyam_registration": "Verified",
            "income_tax_compliance": "Compliant",
            "startup_india_status": "Not Applicable",
            "nsic_status": "Not Applicable",
            "epfo_compliance": "Compliant",
            "esic_compliance": "Compliant",
            "oem_authorization": "Valid",
            "make_in_india_compliance": "Compliant",
            "local_content_percentage": 75.0,
            "digilocker_verification": "Verified",
            "blacklist_status": "Clear",
            "debarment_status": "Not Debarred",
            "tender_eligibility": "Eligible",
            "required_documents": 10,
            "submitted_documents": 10,
            "document_completeness": 100.0,
            "documentation_consistency": 85.0,
            "previous_compliance_issues": 0,
            "portal_verification_issues": 0,
            "verification_delay_days": 3,
            "compliance_score": 88.0
        }

    # Pre-fill from Document Verification module if available
    doc_payload = st.session_state.get("doc_verified_payload")
    if doc_payload and "custom" in sample_choice.lower():
        init_row["required_documents"] = int(doc_payload.get("required_documents", init_row.get("required_documents", 10)))
        init_row["submitted_documents"] = int(doc_payload.get("submitted_documents", init_row.get("submitted_documents", 10)))
        init_row["document_completeness"] = float(doc_payload.get("document_completeness", init_row.get("document_completeness", 100.0)))
        init_row["documentation_consistency"] = float(doc_payload.get("documentation_consistency", init_row.get("documentation_consistency", 85.0)))

    if doc_payload:
        st.markdown(
            f"""
            <div style="background-color: #ebf8ff; border-left: 4px solid #3182ce; padding: 12px 16px; border-radius: 4px; margin-top: 10px; margin-bottom: 15px;">
                <strong>📋 Document Analysis Active:</strong> Pre-filled from Document Verification module:<br>
                • <strong>Required Documents:</strong> {doc_payload.get('required_documents')}<br>
                • <strong>Submitted Documents:</strong> {doc_payload.get('submitted_documents')}<br>
                • <strong>Document Completeness:</strong> {doc_payload.get('document_completeness')}%<br>
                • <strong>Documentation Consistency:</strong> {doc_payload.get('documentation_consistency')}% ({doc_payload.get('consistency_status', 'N/A')})<br>
                <em>Statutory registrations, blacklisting, debarment, and tender eligibility remain unaltered as per procurement protocol.</em>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Form with 7 grouped expandable sections
    with st.form("bid_verification_form"):
        with st.expander("1. Bid & Tender Information", expanded=True):
            f1, f2, f3, f4 = st.columns(4)
            p_type = f1.selectbox("Project Type", project_types_list, index=project_types_list.index(init_row.get("project_type", "IT Services")))
            st_state = f2.selectbox("State", states_list, index=states_list.index(init_row.get("state", "Maharashtra")))
            avail_districts = state_district_map.get(st_state, ["Central"])
            dist_idx = avail_districts.index(init_row["district"]) if init_row.get("district") in avail_districts else 0
            st_district = f3.selectbox("District", avail_districts, index=dist_idx)
            b_val = f4.number_input("Bid Value (₹ Lakhs)", min_value=1.0, max_value=5000.0, value=float(init_row.get("bid_value_lakhs", 50.0)), step=5.0)

        with st.expander("2. Statutory Registrations", expanded=True):
            t1, t2, t3, t4 = st.columns(4)
            gst_reg = t1.selectbox("GST Registration", ["Verified", "Not Verified"], index=0 if init_row.get("gst_registration") == "Verified" else 1)
            gst_filing = t2.selectbox("GST Return Filing", ["Filed", "Pending", "Not Filed"], index=["Filed", "Pending", "Not Filed"].index(init_row.get("gst_return_filing", "Filed")))
            pan_ver = t3.selectbox("PAN Verification", ["Verified", "Not Verified"], index=0 if init_row.get("pan_verification") == "Verified" else 1)
            it_comp = t4.selectbox("Income Tax Compliance", ["Compliant", "Pending", "Non-Compliant"], index=["Compliant", "Pending", "Non-Compliant"].index(init_row.get("income_tax_compliance", "Compliant")))

        with st.expander("3. MSME & Policy Compliance", expanded=True):
            m1, m2, m3, m4 = st.columns(4)
            udyam = m1.selectbox("Udyam Registration (MSME)", ["Verified", "Not Verified", "Not Applicable"], index=["Verified", "Not Verified", "Not Applicable"].index(init_row.get("udyam_registration", "Verified")))
            startup = m2.selectbox("Startup India Status", ["Verified", "Not Verified", "Not Applicable"], index=["Verified", "Not Verified", "Not Applicable"].index(init_row.get("startup_india_status", "Not Applicable")))
            nsic = m3.selectbox("NSIC Status", ["Verified", "Not Verified", "Not Applicable"], index=["Verified", "Not Verified", "Not Applicable"].index(init_row.get("nsic_status", "Not Applicable")))
            digi = m4.selectbox("DigiLocker Verification", ["Verified", "Not Verified", "Not Applicable"], index=["Verified", "Not Verified", "Not Applicable"].index(init_row.get("digilocker_verification", "Verified")))

        with st.expander("4. Labour & Social Security", expanded=True):
            l1, l2 = st.columns(2)
            epfo = l1.selectbox("EPFO Compliance", ["Compliant", "Non-Compliant", "Not Applicable"], index=["Compliant", "Non-Compliant", "Not Applicable"].index(init_row.get("epfo_compliance", "Compliant")))
            esic = l2.selectbox("ESIC Compliance", ["Compliant", "Non-Compliant", "Not Applicable"], index=["Compliant", "Non-Compliant", "Not Applicable"].index(init_row.get("esic_compliance", "Compliant")))

        with st.expander("5. Technical / OEM Compliance", expanded=True):
            tc1, tc2, tc3 = st.columns(3)
            oem = tc1.selectbox("OEM Authorization", ["Valid", "Invalid", "Not Applicable"], index=["Valid", "Invalid", "Not Applicable"].index(init_row.get("oem_authorization", "Valid")))
            mii = tc2.selectbox("Make in India Clause", ["Compliant", "Non-Compliant", "Not Applicable"], index=["Compliant", "Non-Compliant", "Not Applicable"].index(init_row.get("make_in_india_compliance", "Compliant")))
            loc_content = tc3.slider("Local Content (%)", 0.0, 100.0, float(init_row.get("local_content_percentage", 65.0)), step=1.0)

        with st.expander("6. Documentation & Verification", expanded=True):
            d1, d2, d3, d4 = st.columns(4)
            req_d = d1.number_input("Required Documents", min_value=1, max_value=30, value=int(init_row.get("required_documents", 10)))
            sub_d = d2.number_input("Submitted Documents", min_value=0, max_value=30, value=int(min(init_row.get("submitted_documents", 10), req_d)))
            calc_completeness = round((sub_d / req_d) * 100.0, 1) if req_d > 0 else 0.0
            doc_comp_val = float(init_row.get("document_completeness", calc_completeness))
            doc_completeness = d3.slider("Doc Completeness (%)", 0.0, 100.0, doc_comp_val, help="Autocalculated as (Submitted / Required * 100)")
            doc_cons_val = float(init_row.get("documentation_consistency", 80.0))
            doc_cons = d4.slider("Documentation Consistency (%)", 0.0, 100.0, doc_cons_val)

            d5, d6, d7, d8 = st.columns(4)
            prev_iss = d5.number_input("Past Compliance Issues", min_value=0, max_value=10, value=int(init_row.get("previous_compliance_issues", 0)))
            port_iss = d6.number_input("Portal Verification Issues", min_value=0, max_value=10, value=int(init_row.get("portal_verification_issues", 0)))
            delay_days = d7.number_input("Verification Delay (Days)", min_value=0, max_value=60, value=int(init_row.get("verification_delay_days", 4)))
            comp_score = d8.slider("Overall Compliance Score", 0.0, 100.0, float(init_row.get("compliance_score", 85.0)))

        with st.expander("7. Tender Eligibility", expanded=True):
            e1, e2, e3 = st.columns(3)
            t_elig = e1.selectbox("Tender Eligibility", ["Eligible", "Requires Review", "Not Eligible"], index=["Eligible", "Requires Review", "Not Eligible"].index(init_row.get("tender_eligibility", "Eligible")))
            b_list = e2.selectbox("Blacklist Status", ["Clear", "Blacklisted"], index=0 if init_row.get("blacklist_status") == "Clear" else 1, help="Statutory central database check")
            deb_stat = e3.selectbox("Debarment Status", ["Not Debarred", "Debarred"], index=0 if init_row.get("debarment_status") == "Not Debarred" else 1, help="Departmental debarment status")

        submit_btn = st.form_submit_button("🔍 VERIFY BID COMPLIANCE", use_container_width=True)

    # ---------------------------------------------------------
    # INFERENCE & EVALUATION TRIGGER
    # ---------------------------------------------------------
    if submit_btn:
        # Build payload
        bid_payload = {
            "bid_id": init_row.get("bid_id", "DEMO-BID-INPUT"),
            "company_name": init_row.get("company_name", "Enterprise Bidder Submission"),
            "tender_id": init_row.get("tender_id", "TND-EVAL-001"),
            "project_type": p_type,
            "state": st_state,
            "district": st_district,
            "bid_value_lakhs": b_val,
            "gst_registration": gst_reg,
            "gst_return_filing": gst_filing,
            "pan_verification": pan_ver,
            "udyam_registration": udyam,
            "income_tax_compliance": it_comp,
            "startup_india_status": startup,
            "nsic_status": nsic,
            "epfo_compliance": epfo,
            "esic_compliance": esic,
            "oem_authorization": oem,
            "make_in_india_compliance": mii,
            "local_content_percentage": loc_content,
            "digilocker_verification": digi,
            "blacklist_status": b_list,
            "debarment_status": deb_stat,
            "tender_eligibility": t_elig,
            "required_documents": req_d,
            "submitted_documents": sub_d,
            "document_completeness": doc_completeness,
            "documentation_consistency": doc_cons,
            "previous_compliance_issues": prev_iss,
            "portal_verification_issues": port_iss,
            "verification_delay_days": delay_days,
            "compliance_score": comp_score
        }

        with st.spinner("Executing ML inference and rule verification pipeline..."):
            try:
                payload_tuple = tuple(sorted(bid_payload.items()))
                assessment = get_cached_assessment(payload_tuple)
            except Exception as ex:
                st.error(f"Inference error during assessment: {ex}")
                st.stop()

        # Display Prominent Results
        st.markdown("---")
        st.markdown("### 📋 Verification Outcome & Risk Assessment")

        pred_outcome = assessment["prediction"]["predicted_compliance"]
        comp_prob = assessment["prediction"]["compliance_percentage"]
        non_comp_prob = assessment["prediction"]["non_compliance_percentage"]
        risk_lvl = assessment["risk_assessment"]["risk_level"]
        risk_reasons = assessment["risk_assessment"]["reasons"]
        officer_status = assessment["procurement_review"]["status_label"]
        officer_msg = assessment["procurement_review"]["message"]

        # Result Banner Columns
        r1, r2, r3 = st.columns([2, 2, 2])

        with r1:
            if pred_outcome == 1:
                st.success("#### PREDICTED: COMPLIANT\nStatutory requirements largely satisfied.")
            else:
                st.error("#### PREDICTED: NON-COMPLIANT / REQUIRES REVIEW\nSignificant compliance or eligibility issues flagged.")

        with r2:
            st.metric(
                "Compliance Confidence",
                comp_prob,
                help="Model-estimated probability that the bid matches the compliance class learned from the synthetic training dataset."
            )
            st.caption(f"Non-Compliance Estimate: **{non_comp_prob}** (Model-generated confidence)")

        with r3:
            if risk_lvl == "Low":
                st.markdown('<div class="kpi-card risk-card-low" title="Prototype screening risk category based on model probability and rule-based verification findings." style="border-left: 6px solid #2f855a;"><div class="kpi-title">Risk Assessment</div><div class="kpi-value risk-low">LOW RISK</div></div>', unsafe_allow_html=True)
            elif risk_lvl == "Medium":
                st.markdown('<div class="kpi-card risk-card-med" title="Prototype screening risk category based on model probability and rule-based verification findings." style="border-left: 6px solid #d69e2e;"><div class="kpi-title">Risk Assessment</div><div class="kpi-value risk-med">MEDIUM RISK</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="kpi-card risk-card-high" title="Prototype screening risk category based on model probability and rule-based verification findings." style="border-left: 6px solid #c53030;"><div class="kpi-title">Risk Assessment</div><div class="kpi-value risk-high">HIGH RISK</div></div>', unsafe_allow_html=True)

        # Risk Rationale
        st.markdown("#### 📌 Risk Rationale")
        for r_reason in risk_reasons:
            st.markdown(f"- {r_reason}")

        # Rule Checks Breakdown (3 Column Panels)
        st.markdown("#### 📑 Statutory & Procedural Checks Breakdown")
        ch1, ch2, ch3 = st.columns(3)

        passed_checks = assessment["verification"]["passed_checks"]
        warning_checks = assessment["verification"]["warning_checks"]
        failed_checks = assessment["verification"]["failed_checks"]

        with ch1:
            st.markdown(f"**Passed Checks ({len(passed_checks)})**")
            for p_item in passed_checks:
                st.markdown(f"✅ <span style='font-size:0.9rem;'>{p_item}</span>", unsafe_allow_html=True)

        with ch2:
            st.markdown(f"**Warning Checks ({len(warning_checks)})**")
            if warning_checks:
                for w_item in warning_checks:
                    st.markdown(f"⚠️ <span style='font-size:0.9rem; color:#b7791f;'>{w_item}</span>", unsafe_allow_html=True)
            else:
                st.caption("No warnings generated.")

        with ch3:
            st.markdown(f"**Failed Checks ({len(failed_checks)})**")
            if failed_checks:
                for f_item in failed_checks:
                    st.markdown(f"❌ <span style='font-size:0.9rem; color:#c53030; font-weight:600;'>{f_item}</span>", unsafe_allow_html=True)
            else:
                st.caption("No statutory defaults detected.")

        # Procurement Officer Directive Banner
        if "REQUIRES" in officer_status or risk_lvl == "High" or pred_outcome == 0:
            st.markdown(
                f'<div class="officer-alert">'
                f'⚠️ <strong>ACTION REQUIRED: {officer_status}</strong><br>'
                f'AI provides decision support only. Final qualification or disqualification must be determined by the authorized Procurement Officer.<br>'
                f'<em>Directive:</em> {officer_msg}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.info(f"ℹ️ **Status: {officer_status}** — {officer_msg}")

        # -------------------------------------------------------------
        # STEP 8: AI COMPLIANCE EXPLANATION & INTELLIGENCE LAYER
        # -------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🧠 AI Compliance Explanation & Intelligence")
        st.caption("Transparent decomposition of model feature signals, positive compliance factors, and rule-based verification findings.")

        # Compute explainability data
        doc_payload = st.session_state.get("doc_verified_payload")
        fi_table = get_feature_importance()
        payload_tuple = tuple(sorted(bid_payload.items()))
        doc_tuple = tuple(sorted(doc_payload.items())) if doc_payload else None
        explanation = get_cached_explanation(payload_tuple, doc_tuple)

        # 1. Prediction & Review Priority Row
        ex_k1, ex_k2, ex_k3, ex_k4, ex_k5 = st.columns(5)
        with ex_k1:
            st.metric("Predicted Outcome", explanation["prediction_label"], help="Binary classification outcome predicted by Random Forest pipeline.")
        with ex_k2:
            st.metric("Compliance Confidence", explanation["compliance_percentage"], help="Model-estimated probability that the bid matches the compliance class learned from the synthetic training dataset.")
        with ex_k3:
            st.metric("Non-Compliance Estimate", explanation["non_compliance_percentage"], help="Complementary probability estimate for non-compliance.")
        with ex_k4:
            st.metric("Risk Level", explanation["risk_level"], help="Prototype screening risk category based on model probability and rule-based verification findings.")
        with ex_k5:
            st.metric("Review Priority", explanation["review_priority"], help="Prototype priority indicator used to help procurement officers focus their review.")

        # Statutory Exclusion or Priority Banner
        st.markdown(
            f"""
            <div class="section-card" style="background-color: #f7fafc; border-left: 5px solid {explanation['review_badge_color']}; padding: 12px 16px; border-radius: 6px; margin: 12px 0;">
                <strong>Review Status:</strong> <span style="color:{explanation['review_badge_color']}; font-weight:700;">{explanation['review_badge']}</span><br>
                <strong>Priority Rationale:</strong> {explanation['priority_rationale']}<br>
                <strong>Operational Urgency:</strong> {explanation['urgency']}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Critical Rule Safety Banner: High ML probability vs. Statutory Exclusion
        has_exclusion = explanation.get("has_exclusion") or assessment.get("procurement_review", {}).get("statutory_exclusions_flagged", False)
        comp_prob = assessment.get("prediction", {}).get("compliance_probability", 0.0)
        exclusion_flags = assessment.get("procurement_review", {}).get("exclusion_flags", [])
        if has_exclusion and comp_prob >= 0.70:
            flag_str = ", ".join(exclusion_flags) if exclusion_flags else "Debarment / Blacklisting / Eligibility Exclusion"
            st.markdown(
                f"""
                <div class="section-card officer-alert" style="background-color: #fff5f5; border: 1.5px solid #feb2b2; border-left: 5px solid #c53030; border-radius: 6px; padding: 12px 16px; margin: 12px 0;">
                    <strong style="color: #9b2c2c;">⚠️ Critical Rule-Based Exclusion Indicator Detected</strong><br>
                    <span style="font-size: 0.9rem; color: #2d3748; line-height: 1.5;">
                        The ML model indicates high compliance probability ({comp_prob*100:.1f}%) based on general profile characteristics, but critical rule-based exclusion indicators (<strong>{flag_str}</strong>) are active and require officer verification.
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # 2. Why this Assessment? Two Columns: Positive Factors vs Risk Factors
        st.markdown("#### 🔍 Why this Assessment?")
        st.caption("Model features and statutory attributes contributing to this decision-support output.")

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("##### 🟢 Positive Compliance Indicators")
            if explanation["positive_factors"]:
                for pos in explanation["positive_factors"][:7]:
                    st.markdown(
                        f"""
                        <div class="indicator-card" style="background-color: #f0fff4; border: 1px solid #c6f6d5; border-radius: 6px; padding: 10px; margin-bottom: 8px;">
                            <strong style="color: #22543d;">✓ {pos['factor']}</strong><br>
                            <span style="font-size: 0.85rem; color: #2d3748;">{pos['description']}</span><br>
                            <span style="font-size: 0.75rem; color: #718096;">Feature weight: {pos['importance']:.4f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No significant positive compliance indicators detected for this record.")

        with exp_col2:
            st.markdown("##### 🔴 Risk Indicators")
            if explanation["risk_factors"]:
                for rsk in explanation["risk_factors"][:7]:
                    border_c = "#feb2b2" if not rsk.get("is_exclusion") else "#e53e3e"
                    bg_c = "#fff5f5" if not rsk.get("is_exclusion") else "#fed7d7"
                    ex_tag = " <span style='background-color:#c53030; color:#fff; font-size:0.7rem; padding:1px 6px; border-radius:3px; margin-left:6px;'>STATUTORY EXCLUSION</span>" if rsk.get("is_exclusion") else ""
                    st.markdown(
                        f"""
                        <div class="indicator-card" style="background-color: {bg_c}; border: 1px solid {border_c}; border-radius: 6px; padding: 10px; margin-bottom: 8px;">
                            <strong style="color: #9b2c2c;">⚠️ {rsk['factor']}{ex_tag}</strong><br>
                            <span style="font-size: 0.85rem; color: #2d3748;">{rsk['description']}</span><br>
                            <span style="font-size: 0.75rem; color: #718096;">Feature weight: {rsk['importance']:.4f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.success("No elevated risk indicators detected in this submission.")

        # Document Verification Integration context if available
        if explanation["doc_verification_context"]["available"]:
            st.markdown(
                f"""
                <div class="section-card" style="background-color: #ebf8ff; border-left: 4px solid #3182ce; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9rem;">
                    <strong>📄 Document Verification Context:</strong> {explanation['doc_verification_context']['explanation']}
                </div>
                """,
                unsafe_allow_html=True
            )

        # 3. Horizontal Bar Chart of Top Model Feature Signals
        st.markdown("#### 📊 Top Model Feature Signals")
        st.caption("Relative weight of contributing signals based on Random Forest feature importances in the trained synthetic model.")

        if explanation["model_signals"]:
            signals_tuple = tuple(tuple(sorted(s.items())) for s in explanation["model_signals"])
            fig_signals = get_cached_signals_chart(signals_tuple)
            st.plotly_chart(fig_signals, use_container_width=True)
            st.caption("ℹ️ *Feature importance indicates model reliance on features in the trained synthetic-data model; it does not establish causation.*")

        # 4. Procurement Officer Review Summary & Checklist
        st.markdown("#### 👤 Procurement Officer Review Summary")
        st.markdown(
            f"""
            <div class="section-card officer-card" style="background-color: #f7fafc; border: 1.5px solid #cbd5e0; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                <h5 style="margin-top:0; color:#2d3748;">Plain-Language Executive Summary</h5>
                <p style="font-size: 0.95rem; line-height: 1.5; color: #2d3748;">
                    {explanation['officer_summary']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("##### 📌 Recommended Verification Checklist for Procurement Officer")
        for rec in explanation["recommendations"]:
            st.markdown(f"- [ ] {rec}")

        # Mandatory Explainability Disclaimer
        st.markdown(
            """
            <div class="disclaimer-banner" style="margin-top: 15px;">
                <strong>Explainability Notice:</strong> Explainability shows model signals and rule-based findings for decision support. Feature importance does not imply causation. Prototype rules and predictions require human verification.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Step 9: Bid Verification to Officer Review Integration
        st.markdown("---")
        btn_c1, btn_c2 = st.columns([2, 3])
        with btn_c1:
            if st.button("📋 OPEN OFFICER REVIEW", type="primary", use_container_width=True, help="Record in review workflow and open Procurement Officer Review view"):
                doc_p = st.session_state.get("doc_verified_payload", {})
                create_review_record(
                    bid_id=assessment["bid_id"],
                    tender_id=assessment["tender_id"],
                    company_name=assessment["company_name"],
                    predicted_compliance=assessment["prediction"]["predicted_compliance"],
                    compliance_probability=assessment["prediction"]["compliance_probability"],
                    non_compliance_probability=assessment["prediction"]["non_compliance_probability"],
                    risk_level=assessment["risk_assessment"]["risk_level"],
                    review_priority=(
                        explanation["review_priority"] 
                        if isinstance(explanation.get("review_priority"), str) 
                        else explanation.get("review_priority", {}).get("priority_label", "STANDARD REVIEW")
                    ),
                    document_completeness=doc_p.get("document_completeness") if doc_p else None,
                    documentation_consistency=doc_p.get("documentation_consistency") if doc_p else None
                )
                st.session_state["review_selected_bid_id"] = assessment["bid_id"]
                st.session_state["review_current_bid_data"] = bid_payload
                st.session_state["review_current_assessment"] = assessment
                st.session_state["review_current_explanation"] = explanation
                st.session_state["nav_target"] = "Officer Review"
                st.rerun()
        with btn_c2:
            st.info("💡 Clicking **OPEN OFFICER REVIEW** saves this bid to the local review queue and opens the Procurement Officer Review interface with pre-loaded AI assessment findings.")




def render_document_verification_page():
    st.subheader("AI-Assisted Document Verification")
    st.write("Extracts text from supported documents and performs prototype content, completeness, and consistency checks.")

    # Supported Formats Info Badge
    st.markdown(
        """
        <div style="background-color: #f7fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
            <div><strong>SUPPORTED:</strong> <code>PDF</code>, <code>TXT</code></div>
            <div style="font-size: 0.85rem; color: #718096;">pypdf Text Extraction &amp; Heuristic Content Verification</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Prototype Disclaimer
    st.markdown(
        """
        <div class="disclaimer-banner" style="border-left-color: #dd6b20; background-color: #fffaf0; margin-bottom: 16px;">
            <strong>⚠️ Prototype Verification Disclaimer:</strong> Document checks use text extraction and heuristic keyword/entity matching. They are not official statutory validation.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state for uploaded documents
    if "uploaded_documents" not in st.session_state:
        st.session_state["uploaded_documents"] = []

    # Helper function for standardized status label mapping
    def get_standard_doc_status(status_raw):
        if status_raw == STATUS_VERIFIED:
            return "VERIFIED CONTENT"
        elif status_raw == STATUS_WARNING:
            return "CONTENT WARNING"
        elif status_raw == STATUS_EXTRACTION_FAILED:
            return "TEXT EXTRACTION FAILED"
        elif status_raw == STATUS_UNSUPPORTED:
            return "UNSUPPORTED FORMAT"
        return str(status_raw)

    # -------------------------------------------------------------
    # SECTION 1: UPLOAD / DEMO DOCUMENTS
    # -------------------------------------------------------------
    st.markdown("### 1. Upload / Demo Documents")
    st.caption("Load archetypal demonstration documents or upload custom PDF and TXT files for tender scrutiny.")

    # Demonstration Packs
    st.markdown("##### ⚡ Quick Demonstration Packs")
    dp_col1, dp_col2, dp_col3, dp_col4 = st.columns([2, 2, 2, 1.5])

    DEMO_TEXTS = {
        "GST Registration Certificate": (
            "GOVERNMENT OF INDIA - GOODS AND SERVICES TAX\n"
            "REGISTRATION CERTIFICATE\n"
            "Registration Number (GSTIN): 27ABCDE1234F1Z5\n"
            "Legal Name: Bharat Data Systems\n"
            "Trade Name: Bharat Data Systems Enterprises\n"
            "Constitutional Status: Private Enterprise\n"
            "Address: Unit 402, Technology Park, Mumbai, Maharashtra - 400076\n"
            "Date of Registration: 01/07/2017\n"
            "This is a computer generated Goods and Services Tax certificate issued under GST statutory provisions."
        ),
        "PAN Card / PAN Document": (
            "INCOME TAX DEPARTMENT - GOVERNMENT OF INDIA\n"
            "PERMANENT ACCOUNT NUMBER CARD\n"
            "Permanent Account Number: ABCDE1234F\n"
            "Name: Bharat Data Systems\n"
            "Father's / Incorporator Name: S. R. Ramanathan\n"
            "Date of Incorporation: 14/05/2015\n"
            "Issued under authority of Income Tax Rules. Valid PAN card proof."
        ),
        "Udyam Registration Certificate": (
            "MINISTRY OF MICRO, SMALL AND MEDIUM ENTERPRISES\n"
            "UDYAM REGISTRATION CERTIFICATE\n"
            "Udyam Registration Number: UDYAM-MH-01-0012345\n"
            "Name of Enterprise: Bharat Data Systems\n"
            "Type of Enterprise: Micro / Small Enterprise\n"
            "Major Activity: Services - IT Services & Automation\n"
            "Statutory MSME registration verified."
        ),
        "Income Tax Compliance Document": (
            "INCOME TAX DEPARTMENT - ANNUAL TAX RETURN COMPLIANCE\n"
            "Assessment Year: 2024-2025\n"
            "Name: Bharat Data Systems\n"
            "PAN: ABCDE1234F\n"
            "Statutory Filing Status: Income Tax Return Filed & Assessed\n"
            "Tax Liability: Cleared in full\n"
            "Compliance Standing: Regular Tax Payer under Income Tax Act."
        ),
        "OEM Authorization Letter": (
            "ORIGINAL EQUIPMENT MANUFACTURER (OEM) AUTHORIZATION\n"
            "Letter Reference: OEM/AUTH/2024/089\n"
            "To: The Procurement Officer, GeM Procurement Platform\n"
            "Subject: OEM Manufacturer Authorization for Tender\n"
            "We, the Original Equipment Manufacturer, hereby authorize:\n"
            "Partner: Bharat Data Systems\n"
            "Address: Unit 402, Technology Park, Mumbai, Maharashtra\n"
            "To submit bids and provide comprehensive OEM warranty and product support.\n"
            "This Authorization is valid and active."
        )
    }

    if dp_col1.button("📁 Load Compliant Demo Pack", help="Loads 5 verified statutory documents for Bharat Data Systems"):
        st.session_state["uploaded_documents"] = []
        for dtype, dtext in DEMO_TEXTS.items():
            fname = dtype.lower().replace(" ", "_").replace("/", "_") + ".txt"
            ext_res = {"status": "SUCCESS", "text": dtext, "preview": dtext[:3000], "is_empty": False}
            findings = generate_document_findings(fname, dtype, ext_res)
            findings["text"] = dtext
            st.session_state["uploaded_documents"].append(findings)
        st.success("Loaded 5 verified statutory documents for Bharat Data Systems.")
        st.rerun()

    if dp_col2.button("⚠️ Load Pack with Gaps & Warnings", help="Loads 4 documents missing OEM and with name variation"):
        st.session_state["uploaded_documents"] = []
        gst_txt = DEMO_TEXTS["GST Registration Certificate"]
        gst_findings = generate_document_findings("gst_cert.txt", "GST Registration Certificate", {"status": "SUCCESS", "text": gst_txt, "preview": gst_txt[:3000], "is_empty": False})
        gst_findings["text"] = gst_txt
        st.session_state["uploaded_documents"].append(gst_findings)

        pan_var_txt = DEMO_TEXTS["PAN Card / PAN Document"].replace("Name: Bharat Data Systems", "Name: Bharat Data Systems Pvt Ltd")
        pan_findings = generate_document_findings("pan_card.txt", "PAN Card / PAN Document", {"status": "SUCCESS", "text": pan_var_txt, "preview": pan_var_txt[:3000], "is_empty": False})
        pan_findings["text"] = pan_var_txt
        st.session_state["uploaded_documents"].append(pan_findings)

        tax_incomplete = (
            "TAX DEPARTMENT - ANNUAL RETURN\n"
            "Assessment Year: 2024-2025\n"
            "Name: Bharat Data Systems\n"
            "Tax clearance pending review."
        )
        tax_findings = generate_document_findings("tax_filing.txt", "Income Tax Compliance Document", {"status": "SUCCESS", "text": tax_incomplete, "preview": tax_incomplete[:3000], "is_empty": False})
        tax_findings["text"] = tax_incomplete
        st.session_state["uploaded_documents"].append(tax_findings)

        other_txt = "General procurement delivery specification sheet. Standard terms apply."
        other_findings = generate_document_findings("specs.txt", "Other Procurement Document", {"status": "SUCCESS", "text": other_txt, "preview": other_txt[:3000], "is_empty": False})
        other_findings["text"] = other_txt
        st.session_state["uploaded_documents"].append(other_findings)

        st.warning("Loaded demo pack with documentation gaps (Missing OEM, Name variation, Partial keywords).")
        st.rerun()

    if dp_col3.button("❌ Load Inconsistent Entities Demo", help="Loads documents with conflicting company names and PAN"):
        st.session_state["uploaded_documents"] = []
        gst_txt = DEMO_TEXTS["GST Registration Certificate"]
        gst_findings = generate_document_findings("gst_cert.txt", "GST Registration Certificate", {"status": "SUCCESS", "text": gst_txt, "preview": gst_txt[:3000], "is_empty": False})
        gst_findings["text"] = gst_txt
        st.session_state["uploaded_documents"].append(gst_findings)

        oem_conflict = (
            "ORIGINAL EQUIPMENT MANUFACTURER AUTHORIZATION\n"
            "We, the Original Equipment Manufacturer, hereby grant OEM Authorization to:\n"
            "M/s Zenith Global Logistics Corp\n"
            "For tender supply and technical commissioning."
        )
        oem_findings = generate_document_findings("oem_authorization.txt", "OEM Authorization Letter", {"status": "SUCCESS", "text": oem_conflict, "preview": oem_conflict[:3000], "is_empty": False})
        oem_findings["text"] = oem_conflict
        st.session_state["uploaded_documents"].append(oem_findings)

        pan_conflict = (
            "PERMANENT ACCOUNT NUMBER CARD\n"
            "Name: Bharat Data Systems\n"
            "PAN: ZZZZZ9999Z\n"
            "Permanent Account Number verification copy."
        )
        pan_findings = generate_document_findings("pan_conflict.txt", "PAN Card / PAN Document", {"status": "SUCCESS", "text": pan_conflict, "preview": pan_conflict[:3000], "is_empty": False})
        pan_findings["text"] = pan_conflict
        st.session_state["uploaded_documents"].append(pan_findings)

        st.error("Loaded demo pack with severe cross-document entity inconsistencies.")
        st.rerun()

    if dp_col4.button("🗑️ Reset Documents", help="Clears all currently uploaded or loaded documents"):
        st.session_state["uploaded_documents"] = []
        if "doc_verified_payload" in st.session_state:
            del st.session_state["doc_verified_payload"]
        st.info("Cleared all document verification records.")
        st.rerun()

    # Mandatory Required Document Types Configuration
    default_required = [
        "GST Registration Certificate",
        "PAN Card / PAN Document",
        "Udyam Registration Certificate",
        "Income Tax Compliance Document",
        "OEM Authorization Letter"
    ]
    with st.expander("⚙️ Configure Mandatory Tender Documents", expanded=False):
        required_docs_selected = st.multiselect(
            "Select Mandatory Required Document Types for Completeness Evaluation:",
            SUPPORTED_DOCUMENT_TYPES,
            default=default_required
        )
    if not ('required_docs_selected' in locals()):
        required_docs_selected = default_required

    # Upload tabs
    upload_tabs = st.tabs(["📄 Single Document Upload", "📁 Batch Multiple Documents"])

    with upload_tabs[0]:
        u_col1, u_col2 = st.columns([1.5, 2])
        with u_col1:
            selected_doc_type = st.selectbox(
                "Document Type:",
                SUPPORTED_DOCUMENT_TYPES,
                key="single_doc_type_select"
            )
        with u_col2:
            single_file = st.file_uploader(
                f"Choose PDF or TXT file for '{selected_doc_type}':",
                type=["pdf", "txt"],
                key="single_doc_file_uploader",
                help="Accepts .pdf (pypdf text extraction) and .txt files"
            )

        if st.button("➕ Process & Add Document", key="btn_add_single_doc"):
            if single_file is not None:
                with st.spinner("Extracting text and verifying content..."):
                    file_bytes = single_file.getvalue()
                    ext_res = get_cached_extracted_text(file_bytes, single_file.name)
                    findings = generate_document_findings(single_file.name, selected_doc_type, ext_res)
                    findings["text"] = ext_res.get("text", "")

                    existing_idx = next(
                        (i for i, d in enumerate(st.session_state["uploaded_documents"]) if d["document_type"] == selected_doc_type),
                        None
                    )
                    if existing_idx is not None:
                        st.session_state["uploaded_documents"][existing_idx] = findings
                        st.info(f"Updated existing entry for '{selected_doc_type}'.")
                    else:
                        st.session_state["uploaded_documents"].append(findings)
                        st.success(f"Added '{selected_doc_type}' ({single_file.name}) successfully.")
                    st.rerun()
            else:
                st.warning("Please choose a PDF or TXT file before clicking Process & Add.")

    with upload_tabs[1]:
        batch_files = st.file_uploader(
            "Upload multiple procurement documents simultaneously (PDF or TXT):",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="batch_files_uploader"
        )
        if st.button("⚡ Process All Uploaded Files", key="btn_process_batch"):
            if batch_files:
                added_count = 0
                with st.spinner(f"Processing {len(batch_files)} documents..."):
                    for b_file in batch_files:
                        file_bytes = b_file.getvalue()
                        ext_res = get_cached_extracted_text(file_bytes, b_file.name)
                        extracted_text = ext_res.get("text", "")
                        inferred = identify_document_type(extracted_text)
                        inferred_type = inferred["identified_type"]

                        findings = generate_document_findings(b_file.name, inferred_type, ext_res)
                        findings["text"] = extracted_text

                        existing_idx = next(
                            (i for i, d in enumerate(st.session_state["uploaded_documents"]) if d["document_type"] == inferred_type),
                            None
                        )
                        if existing_idx is not None:
                            st.session_state["uploaded_documents"][existing_idx] = findings
                        else:
                            st.session_state["uploaded_documents"].append(findings)
                        added_count += 1
                st.success(f"Processed and indexed {added_count} documents.")
                st.rerun()
            else:
                st.warning("Please select one or more files to batch process.")

    uploaded_docs = st.session_state.get("uploaded_documents", [])

    if uploaded_docs:
        submitted_types = tuple(d["document_type"] for d in uploaded_docs)
        required_types = tuple(required_docs_selected)
        completeness_res = get_cached_doc_completeness(submitted_types, required_types)
        doc_fingerprints = tuple((d["document_type"], d["file_name"], d.get("text", "")) for d in uploaded_docs)
        consistency_res = get_cached_doc_consistency(doc_fingerprints)

        total_docs_count = len(uploaded_docs)
        verified_count = sum(1 for d in uploaded_docs if d["status"] == STATUS_VERIFIED)
        warning_count = sum(1 for d in uploaded_docs if d["status"] == STATUS_WARNING)
        failed_count = sum(1 for d in uploaded_docs if d["status"] in [STATUS_EXTRACTION_FAILED, STATUS_UNSUPPORTED])
        comp_pct = completeness_res["completeness_percentage"]
        cons_status_label = "CONSISTENT" if consistency_res["status"] == "CONSISTENT" else "REVIEW REQUIRED"

        st.markdown("---")

        # -------------------------------------------------------------
        # SECTION 2: EXTRACTION RESULTS
        # -------------------------------------------------------------
        st.markdown("### 2. Extraction Results")
        st.caption("Overview of raw text extraction status, file formats, and extracted character volume.")

        ext_rows = []
        for d in uploaded_docs:
            lbl_status = get_standard_doc_status(d["status"])
            ext_rows.append({
                "Document Type": d["document_type"],
                "File Name": d["file_name"],
                "Characters Extracted": len(d.get("text", "")),
                "Extraction Status": lbl_status,
                "Findings Summary": d.get("findings", "Extracted")
            })
        st.dataframe(pd.DataFrame(ext_rows), use_container_width=True, hide_index=True)

        st.markdown("---")

        # -------------------------------------------------------------
        # SECTION 3: DOCUMENT TYPE FINDINGS
        # -------------------------------------------------------------
        st.markdown("### 3. Document Type Findings")
        st.caption("Detailed keyword audit, missing expected terms, content warnings, and text previews.")

        for idx, doc in enumerate(uploaded_docs):
            doc_std_status = get_standard_doc_status(doc["status"])
            if doc_std_status == "VERIFIED CONTENT":
                badge_html = f"<span class='badge-passed'>✓ {doc_std_status}</span>"
            elif doc_std_status == "CONTENT WARNING":
                badge_html = f"<span class='badge-warning'>⚠ {doc_std_status}</span>"
            else:
                badge_html = f"<span class='badge-failed'>✗ {doc_std_status}</span>"

            expander_title = f"{doc['document_type']} — {doc['file_name']} [{doc_std_status}]"

            with st.expander(expander_title, expanded=(idx == 0)):
                d_info1, d_info2 = st.columns([1.5, 2])
                with d_info1:
                    st.markdown(f"**Document Type:** `{doc['document_type']}`")
                    st.markdown(f"**File Name:** `{doc['file_name']}`")
                    st.markdown(f"**Status:** {badge_html}", unsafe_allow_html=True)
                    st.markdown(f"**Findings:** {doc.get('findings', 'N/A')}")

                with d_info2:
                    st.markdown("**Detected Keywords:**")
                    if doc.get("detected_keywords"):
                        chips_html = "".join([f"<span class='keyword-pill'>✓ {kw}</span>" for kw in doc["detected_keywords"]])
                        st.markdown(chips_html, unsafe_allow_html=True)
                    else:
                        st.caption("None detected.")

                    st.markdown("**Missing Expected Keywords:**")
                    if doc.get("missing_keywords"):
                        missing_html = "".join([f"<span class='keyword-pill' style='background-color:#fee2e2; border-color:#fca5a5; color:#991b1b;'>✗ {kw}</span>" for kw in doc["missing_keywords"]])
                        st.markdown(missing_html, unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='font-size:0.85rem; color:#2f855a;'>None (All expected terms detected)</span>", unsafe_allow_html=True)

                    if doc.get("warnings"):
                        st.markdown("**Warnings:**")
                        for w_msg in doc["warnings"]:
                            st.markdown(f"• <span style='font-size:0.85rem; color:#b7791f;'>{w_msg}</span>", unsafe_allow_html=True)

                st.caption("Prototype content verification — not official statutory validation.")

                st.markdown("**Extracted Text Preview (First 3000 Characters):**")
                preview_content = doc.get("extracted_text_preview", "")
                if not preview_content.strip():
                    if doc_std_status == "TEXT EXTRACTION FAILED":
                        st.error("Text extraction could not be completed for this document.")
                    else:
                        st.warning("Warning: No readable text was detected.")
                else:
                    st.markdown(f"<div class='doc-preview-box'>{preview_content}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # -------------------------------------------------------------
        # SECTION 4: COMPLETENESS
        # -------------------------------------------------------------
        st.markdown("### 4. Completeness")
        st.caption("Verification of mandatory tender document submissions against required checklist.")

        c_m1, c_m2, c_m3 = st.columns([1.5, 2, 2])
        with c_m1:
            st.metric(
                "Document Completeness",
                f"{comp_pct:.0f}%",
                help="Formula: (Submitted Required Documents / Total Required Documents) * 100"
            )
        with c_m2:
            st.markdown(f"**Submitted Required ({len(completeness_res['submitted_required_docs'])}):**")
            if completeness_res["submitted_required_docs"]:
                for s_item in completeness_res["submitted_required_docs"]:
                    st.markdown(f"✅ <span style='font-size:0.85rem;'>{s_item}</span>", unsafe_allow_html=True)
            else:
                st.caption("No mandatory documents submitted yet.")
        with c_m3:
            st.markdown(f"**Missing Required ({len(completeness_res['missing_required_docs'])}):**")
            if completeness_res["missing_required_docs"]:
                for m_item in completeness_res["missing_required_docs"]:
                    st.markdown(f"⚠️ <span style='font-size:0.85rem; color:#b7791f;'>{m_item}</span>", unsafe_allow_html=True)
            else:
                st.success("All mandatory documents submitted!")

        st.markdown("---")

        # -------------------------------------------------------------
        # SECTION 5: CROSS-DOCUMENT CONSISTENCY
        # -------------------------------------------------------------
        st.markdown("### 5. Cross-Document Consistency")
        st.caption("Cross-reference company names and statutory numbers extracted across distinct documents.")

        cons_col1, cons_col2 = st.columns([1.5, 3])
        with cons_col1:
            st.metric(
                "Consistency Score",
                f"{consistency_res['score']:.1f}%",
                help="Heuristic comparison across extracted bidder names and statutory identifiers"
            )
            cons_badge_col = "#2f855a" if cons_status_label == "CONSISTENT" else "#c53030"
            st.markdown(f"**Evaluation Status:** <span style='color:{cons_badge_col}; font-weight:700;'>{cons_status_label}</span>", unsafe_allow_html=True)
            st.caption(consistency_res["details"])

        with cons_col2:
            if consistency_res["warnings"]:
                st.markdown("**Consistency Alerts:**")
                for c_warn in consistency_res["warnings"]:
                    st.markdown(f"⚠️ <span style='font-size:0.85rem; color:#b7791f;'>{c_warn}</span>", unsafe_allow_html=True)
            else:
                st.success("No cross-document name or identifier discrepancies detected.")

            if consistency_res["extracted_entities"]:
                with st.expander("View Extracted Bidder Names Across Documents", expanded=False):
                    for d_source, ent_name in consistency_res["extracted_entities"].items():
                        st.markdown(f"• **{d_source}:** `{ent_name}`")

        st.caption("ℹ️ Consistency checks are heuristic and require human verification.")

        st.markdown("---")

        # -------------------------------------------------------------
        # SECTION 6: VERIFICATION SUMMARY
        # -------------------------------------------------------------
        st.markdown("### 6. Verification Summary")
        st.caption("High-level key performance metrics for the current verification batch.")

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Documents</div><div class="kpi-value">{total_docs_count}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Verified Content</div><div class="kpi-value risk-low">{verified_count}</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Content Warnings</div><div class="kpi-value risk-med">{warning_count}</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Failed / Unsupported</div><div class="kpi-value risk-high">{failed_count}</div></div>', unsafe_allow_html=True)
        with k5:
            comp_class = "risk-low" if comp_pct >= 90.0 else ("risk-med" if comp_pct >= 75.0 else "risk-high")
            st.markdown(f'<div class="kpi-card" title="Percentage of required document types detected in the submitted verification set."><div class="kpi-title">Doc Completeness</div><div class="kpi-value {comp_class}">{comp_pct:.0f}%</div></div>', unsafe_allow_html=True)
        with k6:
            cons_class = "risk-low" if cons_status_label == "CONSISTENT" else "risk-high"
            st.markdown(f'<div class="kpi-card" title="Heuristic comparison of extracted bidder/entity information across submitted documents."><div class="kpi-title">Doc Consistency</div><div class="kpi-value {cons_class}" style="font-size:1.05rem; padding-top:6px;">{cons_status_label}</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # -------------------------------------------------------------
        # SECTION 7: SEND RESULTS TO BID VERIFICATION
        # -------------------------------------------------------------
        st.markdown("### 7. Send Results to Bid Verification")
        st.write("Forward the calculated document completeness and consistency metrics directly into the intelligent bid verification assessment pipeline.")

        if st.button("🚀 USE DOCUMENT RESULTS FOR BID VERIFICATION", type="primary", use_container_width=True):
            st.session_state["doc_verified_payload"] = {
                "required_documents": completeness_res["required_count"],
                "submitted_documents": completeness_res["submitted_count"],
                "document_completeness": completeness_res["completeness_percentage"],
                "documentation_consistency": consistency_res["score"],
                "consistency_status": cons_status_label,
                "total_uploaded": len(uploaded_docs),
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.success(
                "Document analysis has been prepared for AI-assisted bid verification. "
                "Please review all extracted results before proceeding."
            )
            st.info(
                f"Prepared values: Required Documents = {completeness_res['required_count']}, "
                f"Submitted Documents = {completeness_res['submitted_count']}, "
                f"Document Completeness = {completeness_res['completeness_percentage']}%, "
                f"Documentation Consistency = {consistency_res['score']}% ({cons_status_label}).\n\n"
                "👉 Navigate to **Bid Verification** in the sidebar to run ML inference and risk assessment."
            )

    else:
        st.info("ℹ️ No documents uploaded yet. Use the upload interface above or click one of the Quick Demonstration Packs to test the prototype verification engine.")

    # Human-in-the-loop statutory notice
    st.markdown("---")
    st.markdown(
        """
        <div style="background-color:#f7fafc; border:1px solid #e2e8f0; border-radius:6px; padding:12px; font-size:0.85rem; color:#4a5568; text-align:center;">
            <strong>Human-in-the-Loop Mandate:</strong> AI provides decision support only. Final document acceptance, qualification, or disqualification must be determined by the authorized Procurement Officer.
        </div>
        """,
        unsafe_allow_html=True
    )




def render_risk_analysis_page():
    st.subheader("Risk Analysis & Screening")
    st.write("Explore AI-assisted compliance risk patterns across the synthetic procurement dataset.")

    df = get_cached_dataset()

    if df is not None:
        # Filter Controls
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        sel_state = f_col1.multiselect("Filter State:", sorted(df["state"].unique()), default=[])
        sel_ptype = f_col2.multiselect("Filter Project Type:", sorted(df["project_type"].unique()), default=[])
        sel_risk = f_col3.multiselect("Filter Risk Level:", ["Low", "Medium", "High"], default=[])
        sel_elig = f_col4.multiselect("Filter Tender Eligibility:", ["Eligible", "Requires Review", "Not Eligible"], default=[])

        filtered_df = df.copy()
        if sel_state:
            filtered_df = filtered_df[filtered_df["state"].isin(sel_state)]
        if sel_ptype:
            filtered_df = filtered_df[filtered_df["project_type"].isin(sel_ptype)]
        if sel_risk:
            filtered_df = filtered_df[filtered_df["risk_level"].isin(sel_risk)]
        if sel_elig:
            filtered_df = filtered_df[filtered_df["tender_eligibility"].isin(sel_elig)]

        # Active Filters Callout Box
        active_summary = []
        if sel_state:
            active_summary.append(f"State: {', '.join(sel_state)}")
        if sel_ptype:
            active_summary.append(f"Project Type: {', '.join(sel_ptype)}")
        if sel_risk:
            active_summary.append(f"Risk Level: {', '.join(sel_risk)}")
        if sel_elig:
            active_summary.append(f"Eligibility: {', '.join(sel_elig)}")
        filter_desc = " | ".join(active_summary) if active_summary else "All Records (No active filter applied)"

        st.markdown(
            f"""
            <div class="section-card" style="background-color: #f7fafc; border: 1px solid #cbd5e0; border-radius: 6px; padding: 10px 14px; margin-bottom: 16px; font-size: 0.9rem;">
                <strong>Active Filters:</strong> {filter_desc} &nbsp;—&nbsp; <strong>{len(filtered_df):,}</strong> matching records of {len(df):,} total
            </div>
            """,
            unsafe_allow_html=True
        )

        # Check if filters are default (unfiltered) for cached performance
        has_active_filters = bool(sel_state or sel_ptype or sel_risk or sel_elig)

        if not has_active_filters:
            low_c, low_rate, med_c, med_rate, high_c, high_rate = get_base_risk_kpis()
            fig1, fig2, fig3, fig4 = get_base_risk_analysis_charts()
        else:
            low_c = (filtered_df["risk_level"] == "Low").sum()
            low_rate = (filtered_df[filtered_df["risk_level"] == "Low"]["actual_compliance"].mean() * 100) if low_c > 0 else 0
            med_c = (filtered_df["risk_level"] == "Medium").sum()
            med_rate = (filtered_df[filtered_df["risk_level"] == "Medium"]["actual_compliance"].mean() * 100) if med_c > 0 else 0
            high_c = (filtered_df["risk_level"] == "High").sum()
            high_rate = (filtered_df[filtered_df["risk_level"] == "High"]["actual_compliance"].mean() * 100) if high_c > 0 else 0

        # Risk Comparison Cards (Prototype AI Screening Risk)
        ra1, ra2, ra3 = st.columns(3)
        with ra1:
            st.markdown(f'<div class="kpi-card risk-card-low"><div class="kpi-title">Prototype Low Risk</div><div class="kpi-value risk-low">{low_c}</div><p style="font-size:0.85rem; color:#718096; margin-top:4px;">Compliance Rate: {low_rate:.1f}%</p></div>', unsafe_allow_html=True)
        with ra2:
            st.markdown(f'<div class="kpi-card risk-card-med"><div class="kpi-title">Prototype Medium Risk</div><div class="kpi-value risk-med">{med_c}</div><p style="font-size:0.85rem; color:#718096; margin-top:4px;">Compliance Rate: {med_rate:.1f}%</p></div>', unsafe_allow_html=True)
        with ra3:
            st.markdown(f'<div class="kpi-card risk-card-high"><div class="kpi-title">Prototype High Risk</div><div class="kpi-value risk-high">{high_c}</div><p style="font-size:0.85rem; color:#718096; margin-top:4px;">Compliance Rate: {high_rate:.1f}%</p></div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

        # 4 Clearly Numbered Risk Charts in 2x2 Layout
        r_c1, r_c2 = st.columns(2)

        with r_c1:
            # 1. Risk Distribution
            st.markdown("##### 1. Risk Distribution")
            if not has_active_filters and fig1 is not None:
                st.plotly_chart(fig1, use_container_width=True)
            else:
                risk_dist = filtered_df["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
                risk_dist.columns = ["Risk Level", "Bids Count"]
                dyn_fig1 = px.bar(
                    risk_dist,
                    x="Risk Level",
                    y="Bids Count",
                    color="Risk Level",
                    color_discrete_map={"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e76f51"},
                    title="Prototype AI Screening Risk Distribution",
                    text="Bids Count"
                )
                dyn_fig1.update_traces(textposition='outside', hovertemplate="<b>%{x} Risk</b><br>Bids: %{y:,}<extra></extra>")
                dyn_fig1.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=320)
                apply_chart_hover_theme(dyn_fig1)
                st.plotly_chart(dyn_fig1, use_container_width=True)

        with r_c2:
            # 2. Risk by Tender Eligibility
            st.markdown("##### 2. Risk by Tender Eligibility")
            if not has_active_filters and fig2 is not None:
                st.plotly_chart(fig2, use_container_width=True)
            else:
                elig_risk = filtered_df.groupby(["risk_level", "tender_eligibility"], observed=False).size().reset_index(name="Count")
                dyn_fig2 = px.bar(
                    elig_risk,
                    x="risk_level",
                    y="Count",
                    color="tender_eligibility",
                    barmode="group",
                    title="Tender Eligibility Breakdown by Risk Level",
                    color_discrete_map={"Eligible": "#2a9d8f", "Requires Review": "#e9c46a", "Not Eligible": "#e76f51"}
                )
                dyn_fig2.update_traces(hovertemplate="<b>%{x} Risk (%{data.name})</b><br>Count: %{y:,}<extra></extra>")
                dyn_fig2.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=320, xaxis_title="Risk Level", yaxis_title="Bids Count")
                apply_chart_hover_theme(dyn_fig2)
                st.plotly_chart(dyn_fig2, use_container_width=True)

        r_c3, r_c4 = st.columns(2)

        with r_c3:
            # 3. Compliance Score by Risk
            st.markdown("##### 3. Compliance Score by Risk")
            if not has_active_filters and fig3 is not None:
                st.plotly_chart(fig3, use_container_width=True)
            else:
                risk_score = filtered_df.groupby("risk_level", observed=False)["compliance_score"].mean().reindex(["Low", "Medium", "High"]).reset_index()
                risk_score.columns = ["Risk Level", "Avg Compliance Score"]
                dyn_fig3 = px.bar(
                    risk_score,
                    x="Risk Level",
                    y="Avg Compliance Score",
                    color="Risk Level",
                    color_discrete_map={"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e76f51"},
                    title="Average Compliance Score (0–100) by Risk Level",
                    text="Avg Compliance Score"
                )
                dyn_fig3.update_traces(texttemplate='%{text:.1f}', textposition='outside', hovertemplate="<b>%{x} Risk</b><br>Avg Score: %{y:.1f} / 100<extra></extra>")
                dyn_fig3.update_layout(yaxis=dict(range=[0, 110]), margin=dict(t=40, b=20, l=20, r=20), height=320)
                apply_chart_hover_theme(dyn_fig3)
                st.plotly_chart(dyn_fig3, use_container_width=True)

        with r_c4:
            # 4. Verification Delay by Risk
            st.markdown("##### 4. Verification Delay by Risk")
            if not has_active_filters and fig4 is not None:
                st.plotly_chart(fig4, use_container_width=True)
            else:
                risk_delay = filtered_df.groupby("risk_level", observed=False)["verification_delay_days"].mean().reindex(["Low", "Medium", "High"]).reset_index()
                risk_delay.columns = ["Risk Level", "Avg Delay Days"]
                dyn_fig4 = px.bar(
                    risk_delay,
                    x="Risk Level",
                    y="Avg Delay Days",
                    color="Risk Level",
                    color_discrete_map={"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e76f51"},
                    title="Average Verification Delay Turnaround (Days) by Risk Level",
                    text="Avg Delay Days"
                )
                dyn_fig4.update_traces(texttemplate='%{text:.1f}d', textposition='outside', hovertemplate="<b>%{x} Risk</b><br>Avg Turnaround: %{y:.1f} Days<extra></extra>")
                dyn_fig4.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=320)
                apply_chart_hover_theme(dyn_fig4)
                st.plotly_chart(dyn_fig4, use_container_width=True)

        # Filtered Table View with Prominent High-Risk Records
        st.markdown("---")
        st.markdown("### 📋 Filtered Bids Record Table")
        st.caption("Inspect individual bid submissions. High Risk records are clearly highlighted for prioritized scrutiny.")

        disp_cols = [
            "bid_id", "company_name", "project_type", "state",
            "compliance_score", "risk_level", "actual_compliance",
            "gst_registration", "pan_verification", "tender_eligibility"
        ]
        table_df = filtered_df[disp_cols].copy()
        table_df.columns = [
            "Bid ID", "Company Name", "Project Type", "State",
            "Compliance Score", "Prototype AI Screening Risk", "Actual Compliance",
            "GST Valid", "PAN Valid", "Tender Eligibility"
        ]
        st.dataframe(table_df.head(100), use_container_width=True, hide_index=True)
        st.caption("ℹ️ *Prototype AI Screening Risk categories are heuristic decision-support indicators and do not represent official GeM risk categories.*")




def render_model_insights_page():
    st.subheader("Model Performance & Explainability")
    st.write("Evaluation results shown below are based on the synthetic/demo dataset.")

    # Candidate Algorithm Benchmark Comparison
    st.markdown("### 1. Candidate Algorithm Benchmark Comparison")
    st.caption("Performance evaluated on synthetic/demo test partition (200 records, 80/20 Stratified Split).")

    comp_data = get_model_comparison()
    if comp_data is not None:
        st.dataframe(comp_data.style.highlight_max(subset=["Accuracy", "F1 Score"], color="#d4edda"), use_container_width=True)
    else:
        # Fallback table with exact preserved observed metrics
        benchmark_table = pd.DataFrame([
            {"Model": "Logistic Regression", "Accuracy": 0.9050, "Precision": 0.8950, "Recall": 0.9300, "F1 Score": 0.9122},
            {"Model": "Decision Tree", "Accuracy": 0.8700, "Precision": 0.8650, "Recall": 0.9000, "F1 Score": 0.8821},
            {"Model": "Random Forest", "Accuracy": 0.9100, "Precision": 0.9052, "Recall": 0.9375, "F1 Score": 0.9211}
        ])
        st.dataframe(benchmark_table.style.highlight_max(subset=["Accuracy", "F1 Score"], color="#d4edda"), use_container_width=True)

    # Selected Prototype Model Card
    st.markdown("### 2. Selected Prototype Model")
    st.caption("Random Forest performance on 200 synthetic test samples (80/20 Stratified Partition).")
    sm1, sm2, sm3, sm4 = st.columns(4)
    with sm1:
        st.metric("Test Accuracy", "91.00%", help="Accuracy measured on the held-out synthetic test dataset.")
    with sm2:
        st.metric("Test F1-Score", "0.9211", help="Harmonic mean of precision and recall on the synthetic test set.")
    with sm3:
        st.metric("Precision (Compliant)", "90.52%", help="Proportion of predicted compliant bids that are truly compliant.")
    with sm4:
        st.metric("Recall (Compliant)", "93.75%", help="Proportion of actual compliant bids correctly identified.")

    st.markdown(
        """
        <div class="section-card" style="background-color: #f0fff4; border: 1.5px solid #9ae6b4; border-radius: 8px; padding: 14px 16px; margin: 12px 0 16px 0;">
            <p style="font-size: 0.90rem; color: #234e52; line-height: 1.5; margin: 0;">
                <strong>Model Rationale:</strong> The Random Forest classifier was chosen as the primary prototype model because it produced the highest observed accuracy (91.00%) and F1-score (0.9211) among evaluated candidates on the synthetic benchmark dataset, while offering native feature importance transparency.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Feature Importance Plot
    st.markdown("### 3. Feature Importance Analysis")
    st.caption("Statistical feature reliance within the trained Random Forest model (Gini importance). Feature importance does not imply causation.")
    fig_fi = get_feature_importance_chart()
    if fig_fi is not None:
        st.plotly_chart(fig_fi, use_container_width=True)

    # Confusion Matrix Image
    st.markdown("### 4. Test Set Confusion Matrix")
    st.caption("Evaluation performed on synthetic/demo test partition (200 test samples).")
    cm_path = PROJECT_ROOT / "data" / "processed" / "confusion_matrix_random_forest.png"
    if cm_path.exists():
        st.image(str(cm_path), width=500)
    else:
        st.info("Confusion matrix graphic not found at data/processed/confusion_matrix_random_forest.png.")

    # Section: How the AI Assessment Works
    st.markdown("---")
    st.markdown("### 5. AI Assessment Workflow Architecture")
    st.caption("End-to-end explainable intelligence architecture translating structured tender submissions into actionable procurement decisions.")

    st.markdown(
        """
        ```
        Structured Bid Information
                ↓
        Preprocessing (ColumnTransformer & Scaler)
                ↓
        Random Forest Prediction (Ensemble Trees)
                ↓
        Compliance Probability (Calibrated Class Probabilities)
                ↓
        Rule-Based Verification (21 Statutory & Technical Checks)
                ↓
        Risk Assessment (Low / Medium / High Tiers)
                ↓
        Explainability Layer (Positive & Risk Factor Attribution)
                ↓
        Procurement Officer Review (Human-in-the-Loop Final Decision)
        ```
        """
    )
    st.markdown(
        """
        - **1. Structured Bid Information:** Extracts numerical indices, statutory categorical claims, and tender specifications.
        - **2. Preprocessing:** Normalizes continuous variables via `StandardScaler` and expands categories via `OneHotEncoder`.
        - **3. Random Forest Prediction:** Generates ensemble classification (0 = Non-Compliant / Review, 1 = Compliant).
        - **4. Compliance Probability:** Calculates calibrated probability confidence scores for both compliant and non-compliant outcomes.
        - **5. Rule-Based Verification:** Evaluates 21 statutory mandates into Passed, Warning, and Failed categories.
        - **6. Risk Assessment:** Synthesizes model confidence and rule flags into Low, Medium, or High Risk tiers.
        - **7. Explainability Layer:** Decomposes decision into genuine positive factors vs. risk indicators without claiming causation.
        - **8. Procurement Officer Review:** Strictly reserves definitive qualification and disqualification authority for the human officer.
        """
    )

    # Important Prototype Limitation Callout
    st.markdown(
        """
        <div class="disclaimer-banner" style="border-left-color: #e53e3e; background-color: #fff5f5; color: #9b2c2c; margin-top: 20px;">
            <strong>⚠️ Important Prototype Limitation:</strong><br>
            The dataset used for development is synthetic/demo data. Model performance on this dataset does not establish performance on real procurement data. Field deployment requires verification against real-world tender submissions, official portal integration, and governance certification.
        </div>
        """,
        unsafe_allow_html=True
    )



def render_officer_review_page():
    st.subheader("Procurement Officer Review")
    st.write("Human-in-the-loop review workspace for examining AI screening results and recording review actions.")

    # Mandatory Decision-Support Disclaimer Banner
    st.markdown(
        """
        <div class="disclaimer-banner" style="border-left-color: #3182ce; background-color: #ebf8ff; margin-bottom: 16px;">
            <strong>⚠️ Human-in-the-Loop Governance Notice:</strong><br>
            AI provides decision support only. This prototype does not make final procurement qualification or disqualification decisions. Final qualification or disqualification must be determined by the authorized Procurement Officer.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Review Statistics KPIs Row
    rev_stats = calculate_review_statistics()
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)
    with kpi_c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Reviews</div><div class="kpi-value">{rev_stats["total_reviews"]}</div></div>', unsafe_allow_html=True)
    with kpi_c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Not Reviewed</div><div class="kpi-value">{rev_stats["not_reviewed"]}</div></div>', unsafe_allow_html=True)
    with kpi_c3:
        st.markdown(f'<div class="kpi-card risk-card-med"><div class="kpi-title">Under Review</div><div class="kpi-value risk-med">{rev_stats["under_review"]}</div></div>', unsafe_allow_html=True)
    with kpi_c4:
        st.markdown(f'<div class="kpi-card risk-card-low"><div class="kpi-title">Reviewed</div><div class="kpi-value risk-low">{rev_stats["reviewed"]}</div></div>', unsafe_allow_html=True)
    with kpi_c5:
        st.markdown(f'<div class="kpi-card risk-card-high"><div class="kpi-title">Escalated</div><div class="kpi-value risk-high">{rev_stats["escalated"]}</div></div>', unsafe_allow_html=True)
    with kpi_c6:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Completion Rate</div><div class="kpi-value">{rev_stats["completion_rate"]:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Select Bid for Review
    st.markdown("### 📂 Select Bid for Scrutiny")
    
    saved_reviews = list_all_reviews()
    df_data = get_cached_dataset()
    dataset_bids = []
    if df_data is not None and not df_data.empty:
        dataset_bids = df_data[["bid_id", "company_name", "risk_level"]].head(30).to_dict("records")

    options_map = {}
    
    # Check session state bid first
    session_bid_id = st.session_state.get("review_selected_bid_id")
    if session_bid_id:
        lbl_current = f"📌 Currently Selected: {session_bid_id}"
        options_map[lbl_current] = session_bid_id

    # Saved reviews in SQLite
    for r in saved_reviews:
        lbl = f"🗃️ [Saved Review] {r['bid_id']} — {r.get('company_name', 'Unknown')} ({r.get('review_status', 'NOT REVIEWED')})"
        options_map[lbl] = r["bid_id"]

    # Sample dataset bids
    for d in dataset_bids:
        b_id = d["bid_id"]
        if b_id not in [options_map[k] for k in options_map]:
            lbl = f"📄 [Dataset Sample] {b_id} — {d.get('company_name', 'Unknown')} ({d.get('risk_level', 'Medium')} Risk)"
            options_map[lbl] = b_id

    if not options_map:
        st.info("No bid records available in the review queue. Run a bid evaluation in Bid Verification or load sample dataset records.")
    else:
        labels = list(options_map.keys())
        default_idx = 0
        if session_bid_id:
            for i, l in enumerate(labels):
                if options_map[l] == session_bid_id:
                    default_idx = i
                    break

        selected_label = st.selectbox("Choose Bid Record for Scrutiny:", labels, index=default_idx)
        selected_bid_id = options_map[selected_label]
        st.session_state["review_selected_bid_id"] = selected_bid_id

        # Retrieve review record from SQLite
        current_record = get_review_record(selected_bid_id)

        # Retrieve bid raw data row if from dataset or session state
        bid_row_dict = None
        if "review_current_bid_data" in st.session_state and st.session_state.get("review_selected_bid_id") == selected_bid_id:
            bid_row_dict = st.session_state["review_current_bid_data"]
        elif df_data is not None:
            match = df_data[df_data["bid_id"] == selected_bid_id]
            if not match.empty:
                bid_row_dict = match.iloc[0].to_dict()

        # If record does not exist in SQLite yet, auto-create it now from AI assessment
        if current_record is None:
            if bid_row_dict is not None:
                payload_tuple = tuple(sorted(bid_row_dict.items()))
                bid_assessment = get_cached_assessment(payload_tuple)
                doc_p = st.session_state.get("doc_verified_payload")
                doc_tuple = tuple(sorted(doc_p.items())) if doc_p else None
                bid_expl = get_cached_explanation(payload_tuple, doc_tuple)
                
                doc_p = st.session_state.get("doc_verified_payload", {})
                current_record = create_review_record(
                    bid_id=selected_bid_id,
                    tender_id=bid_assessment["tender_id"],
                    company_name=bid_assessment["company_name"],
                    predicted_compliance=bid_assessment["prediction"]["predicted_compliance"],
                    compliance_probability=bid_assessment["prediction"]["compliance_probability"],
                    non_compliance_probability=bid_assessment["prediction"]["non_compliance_probability"],
                    risk_level=bid_assessment["risk_assessment"]["risk_level"],
                    review_priority=(
                        bid_expl["review_priority"]
                        if isinstance(bid_expl.get("review_priority"), str)
                        else bid_expl.get("review_priority", {}).get("priority_label", "STANDARD REVIEW")
                    ),
                    document_completeness=doc_p.get("document_completeness") if doc_p else None,
                    documentation_consistency=doc_p.get("documentation_consistency") if doc_p else None
                )
            else:
                current_record = create_review_record(
                    bid_id=selected_bid_id,
                    tender_id="TENDER_UNKNOWN",
                    company_name="VENDOR_UNKNOWN"
                )

        # -------------------------------------------------------------
        # TOP SUMMARY METRICS & WORKFLOW DIAGRAM
        # -------------------------------------------------------------
        st.markdown(
            f"""
            <div class="officer-card" style="background-color: #f7fafc; border: 1.5px solid #cbd5e0; border-radius: 8px; padding: 14px; margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px; font-size: 0.95rem;">
                    <div><strong>Bid ID:</strong> <code>{current_record['bid_id']}</code></div>
                    <div><strong>Tender ID:</strong> <code>{current_record.get('tender_id', 'N/A')}</code></div>
                    <div><strong>Company:</strong> <strong>{current_record.get('company_name', 'N/A')}</strong></div>
                    <div><strong>Prediction:</strong> <span class="{'risk-low' if current_record.get('predicted_compliance') == 'Compliant' else 'risk-high'}">{current_record.get('predicted_compliance', 'N/A')}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 4 Core Top Metrics Row
        t_m1, t_m2, t_m3, t_m4 = st.columns(4)
        with t_m1:
            st.metric("Review Status", current_record.get("review_status", "NOT REVIEWED"), help="Current lifecycle state in procurement scrutiny workflow.")
        with t_m2:
            st.metric("Review Priority", current_record.get("review_priority", "STANDARD REVIEW"), help="Prototype priority indicator used to help procurement officers focus their review.")
        with t_m3:
            st.metric("Risk Level", current_record.get("risk_level", "Medium"), help="Prototype screening risk category based on model probability and rule-based verification findings.")
        with t_m4:
            st.metric("Compliance Probability", f"{current_record.get('compliance_probability', 0.0)*100:.1f}%", help="Model-estimated probability that the bid matches the compliance class learned from the synthetic training dataset.")

        # Visual Workflow Progression Diagram
        st.markdown(
            """
            <div class="officer-card" style="background-color: #edf2f7; border: 1px solid #cbd5e0; border-radius: 6px; padding: 10px 14px; margin: 12px 0; text-align: center; font-size: 0.88rem; font-family: monospace;">
                <strong>WORKFLOW TRANSITIONS:</strong> NOT REVIEWED &nbsp;→&nbsp; UNDER REVIEW &nbsp;→&nbsp; REVIEWED &nbsp;&nbsp;|&nbsp;&nbsp; UNDER REVIEW &nbsp;→&nbsp; ESCALATED
            </div>
            """,
            unsafe_allow_html=True
        )

        # Retrieve explanation object if raw input features are available
        expl = None
        if bid_row_dict is not None:
            payload_tuple = tuple(sorted(bid_row_dict.items()))
            doc_p = st.session_state.get("doc_verified_payload")
            doc_tuple = tuple(sorted(doc_p.items())) if doc_p else None
            expl = get_cached_explanation(payload_tuple, doc_tuple)

        # Critical Rule Safety Banner: High ML probability vs. Statutory Exclusion
        rec_comp_prob = current_record.get("compliance_probability", 0.0)
        has_rec_exclusion = (expl and expl.get("has_exclusion")) or current_record.get("review_priority") == "HIGH PRIORITY REVIEW"
        if has_rec_exclusion and rec_comp_prob >= 0.70:
            ex_flags = expl.get("exclusion_flags", []) if expl else ["Statutory Exclusion on record"]
            flag_str = ", ".join(ex_flags) if ex_flags else "Debarment / Blacklisting / Eligibility Exclusion"
            st.markdown(
                f"""
                <div class="officer-card officer-alert" style="background-color: #fff5f5; border: 1.5px solid #feb2b2; border-left: 5px solid #c53030; border-radius: 6px; padding: 12px 16px; margin: 12px 0;">
                    <strong style="color: #9b2c2c;">⚠️ Critical Rule-Based Exclusion Indicator Detected</strong><br>
                    <span style="font-size: 0.9rem; color: #2d3748; line-height: 1.5;">
                        The ML model indicates high statistical compliance probability ({rec_comp_prob*100:.1f}%), but critical rule-based statutory exclusion indicators (<strong>{flag_str}</strong>) require mandatory officer scrutiny. A high probability score does not override statutory procurement disqualifications.
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------------------
        # SECTION 1: AI ASSESSMENT
        # -------------------------------------------------------------
        st.markdown("### 1. AI Assessment")
        st.caption("Calibrated compliance probabilities, model classification, and plain-language executive rationale.")

        ai_c1, ai_c2, ai_c3 = st.columns(3)
        with ai_c1:
            st.markdown(f"**Predicted Class:** `{current_record.get('predicted_compliance', 'N/A')}`")
        with ai_c2:
            st.markdown(f"**Compliance Confidence:** `{current_record.get('compliance_probability', 0.0)*100:.1f}%`")
        with ai_c3:
            st.markdown(f"**Non-Compliance Probability:** `{current_record.get('non_compliance_probability', 0.0)*100:.1f}%`")

        if expl is not None:
            st.markdown(
                f"""
                <div class="officer-card" style="background-color: #f7fafc; border-left: 4px solid #4a5568; padding: 12px 16px; border-radius: 4px; margin-top: 10px; margin-bottom: 12px;">
                    <strong>Executive AI Assessment Summary:</strong> {expl.get('officer_summary', 'AI assessment completed.')}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Stored AI prediction values loaded from review audit database.")

        st.markdown("---")

        # -------------------------------------------------------------
        # SECTION 2: RISK & RULE FINDINGS
        # -------------------------------------------------------------
        st.markdown("### 2. Risk & Rule Findings")
        st.caption("Decomposition into positive compliance indicators, flagged risk factors, and statutory rule checks.")

        if expl is not None:
            c_pos, c_risk = st.columns(2)
            with c_pos:
                st.markdown("##### ✅ Positive Compliance Indicators")
                pos_factors = expl.get("positive_factors", [])
                if pos_factors:
                    for f in pos_factors:
                        st.markdown(f"- **{f['factor']}**: {f['description']}")
                else:
                    st.info("No standout positive compliance indicators detected.")
            
            with c_risk:
                st.markdown("##### ⚠️ Top Risk Factors")
                risk_factors = expl.get("risk_factors", [])
                if risk_factors:
                    for f in risk_factors:
                        st.markdown(f"- **{f['factor']}**: {f['description']}")
                else:
                    st.success("No critical risk factors flagged.")

            c_pass, c_fail = st.columns(2)
            with c_pass:
                st.markdown("##### 📋 Passed Rule Checks")
                passed = expl.get("rule_findings", {}).get("passed", [])
                for p in passed[:5]:
                    st.markdown(f"✓ {p}")
            with c_fail:
                st.markdown("##### ❌ Failed & Warning Checks")
                failed = expl.get("rule_findings", {}).get("failed", [])
                warnings = expl.get("rule_findings", {}).get("warnings", [])
                if not failed and not warnings:
                    st.success("All baseline statutory rule checks passed.")
                else:
                    for fl in failed:
                        st.markdown(f"✗ <span style='color:#e53e3e;'>{fl}</span>", unsafe_allow_html=True)
                    for w in warnings:
                        st.markdown(f"⚠ <span style='color:#dd6b20;'>{w}</span>", unsafe_allow_html=True)
        else:
            st.info("Full feature attribution requires raw input data. Basic risk tier recorded as: **" + str(current_record.get('risk_level', 'Medium')) + "**.")

        st.markdown("---")

        # -------------------------------------------------------------
        # SECTION 3: DOCUMENT VERIFICATION CONTEXT
        # -------------------------------------------------------------
        st.markdown("### 3. Document Verification Context")
        st.caption("Completeness percentage and cross-document entity consistency verified for this tender submission.")

        doc_comp_val = current_record.get("document_completeness")
        doc_cons_val = current_record.get("documentation_consistency")

        d_vc1, d_vc2 = st.columns(2)
        with d_vc1:
            st.metric("Document Completeness", f"{doc_comp_val:.0f}%" if doc_comp_val is not None else "N/A", help="Percentage of required document types detected in the submitted verification set.")
        with d_vc2:
            st.metric("Documentation Consistency", f"{doc_cons_val:.1f}%" if isinstance(doc_cons_val, (int, float)) else str(doc_cons_val or "N/A"), help="Heuristic comparison of extracted bidder/entity information across submitted documents.")

        if doc_comp_val is not None or doc_cons_val is not None:
            st.markdown(
                f"""
                <div class="officer-card" style="background-color: #ebf8ff; border-left: 4px solid #3182ce; padding: 10px 14px; border-radius: 4px; margin-top: 8px; font-size: 0.9rem;">
                    <strong>Document Verification Record:</strong> 
                    Completeness: <strong>{doc_comp_val if doc_comp_val is not None else 'N/A'}%</strong> | 
                    Consistency: <strong>{doc_cons_val if doc_cons_val is not None else 'N/A'}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.caption("No integrated document verification payload attached. Use the Document Verification page to attach extracted findings.")

        st.markdown("---")

        # -------------------------------------------------------------
        # SECTIONS 4 & 5: OFFICER ACTION & OFFICER REMARKS
        # -------------------------------------------------------------
        st.markdown("### 4. Officer Action & 5. Officer Remarks")
        st.caption("Record human-in-the-loop review actions and rationale notes. All transitions are validated against workflow governance rules.")

        cur_status = current_record.get("review_status", "NOT REVIEWED")
        status_colors = {
            "NOT REVIEWED": ("#4a5568", "#edf2f7"),
            "UNDER REVIEW": ("#b7791f", "#fefcbf"),
            "REVIEWED": ("#22543d", "#c6f6d5"),
            "ESCALATED": ("#742a2a", "#fed7d7")
        }
        text_col, bg_col = status_colors.get(cur_status, ("#4a5568", "#edf2f7"))

        st.markdown(
            f"""
            <div class="officer-card officer-status-badge" style="background-color: {bg_col}; color: {text_col}; padding: 10px 16px; border-radius: 6px; font-weight: 600; display: inline-block; margin-bottom: 14px; border: 1px solid {text_col}; font-size: 0.95rem;">
                Current Review Status: <strong>{cur_status}</strong> &nbsp;|&nbsp; Last Recorded Action: <strong>{current_record.get('officer_action', 'PENDING REVIEW')}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form(key=f"review_action_form_{selected_bid_id}"):
            fcol1, fcol2 = st.columns(2)
            
            with fcol1:
                # Pre-select appropriate action based on current status
                default_act_idx = 0
                if cur_status == "UNDER REVIEW":
                    default_act_idx = 3  # REVIEW COMPLETED
                elif cur_status == "REVIEWED":
                    default_act_idx = 3  # REVIEW COMPLETED
                elif cur_status == "ESCALATED":
                    default_act_idx = 4  # ESCALATED FOR FURTHER REVIEW

                chosen_action = st.selectbox(
                    "Officer Action:",
                    VALID_OFFICER_ACTIONS,
                    index=default_act_idx,
                    help="Record workflow action taken by the Procurement Officer."
                )

            with fcol2:
                default_target = ACTION_STATUS_TARGETS.get(chosen_action, cur_status)
                target_idx = VALID_REVIEW_STATUSES.index(default_target) if default_target in VALID_REVIEW_STATUSES else 0
                target_status_choice = st.selectbox(
                    "Resulting Review Status:",
                    VALID_REVIEW_STATUSES,
                    index=target_idx,
                    help="Target review status after applying this workflow action."
                )

            officer_remarks_input = st.text_area(
                "Officer Remarks / Scrutiny Notes:",
                value=current_record.get("officer_remarks", ""),
                height=110,
                placeholder="Enter procurement scrutiny notes (e.g., 'GST document requires additional verification', 'Turnover certificate cross-verified on MCA-21', etc.)..."
            )

            st.caption("ℹ️ *Every review action is immutably logged with timestamp and previous status in the local SQLite audit trail.*")

            submit_action = st.form_submit_button("💾 SAVE REVIEW ACTION", type="primary")

            if submit_action:
                ok, msg, updated = update_review_status(
                    bid_id=selected_bid_id,
                    action=chosen_action,
                    remarks=officer_remarks_input,
                    target_status=target_status_choice
                )
                if ok:
                    st.success(f"✓ {msg}")
                    st.rerun()
                else:
                    st.error(f"❌ Review action cannot be applied from the current status. {msg}")

        st.markdown("---")

        # -------------------------------------------------------------
        # SECTION 6: AUDIT HISTORY
        # -------------------------------------------------------------
        st.markdown("### 6. Audit History")
        st.caption("Chronological, tamper-evident log of all review actions, status transitions, and officer remarks for this bid.")
        
        audit_history = get_review_history(selected_bid_id)
        if not audit_history:
            st.info("No audit entries recorded for this bid.")
        else:
            audit_display = []
            for entry in audit_history:
                audit_display.append({
                    "Timestamp": entry["timestamp"],
                    "Action": entry["action"],
                    "Previous Status": entry["previous_status"],
                    "New Status": entry["new_status"],
                    "Remarks": entry["remarks"] or "—"
                })
            st.dataframe(pd.DataFrame(audit_display), use_container_width=True, hide_index=True)




def render_about_system_page():
    st.subheader("About the System")
    st.write("AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement.")

    st.markdown(
        """
        <div class="disclaimer-banner" style="border-left-color: #2b6cb0; background-color: #ebf8ff; color: #2c5282; margin-bottom: 20px;">
            <strong>System Context:</strong> Integrated Procurement Verification Platform | Organization: Chennai Petroleum Corporation Limited (CPCL)
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Problem Overview
    st.markdown("### 1. Problem Overview")
    st.write(
        "In public procurement through Government e-Marketplace (GeM), procurement officers must examine high volumes of bidder submissions, "
        "scrutinize statutory compliance documents, verify eligibility criteria, and detect inconsistencies across disparate evidence sources. "
        "Manual verification is time-intensive, cognitively demanding, and subject to oversight under tight bidding schedules."
    )

    # 2. Proposed AI-Assisted Solution
    st.markdown("### 2. Proposed AI-Assisted Solution")
    st.write(
        "The **AI-Powered Integrated Bid Compliance Verification Platform** is an assistive decision-support platform designed to streamline tender scrutiny without replacing human judgment. The system assists procurement officers by:"
    )
    st.markdown(
        """
        - **Organizing bid information** into a structured, unified evaluation schema.
        - **Checking prototype compliance rules** across statutory (GST, PAN, MSME, EPFO, ESIC) and commercial criteria.
        - **Verifying document content** through automated text extraction, keyword auditing, and cross-record consistency checks.
        - **Predicting compliance probability** using an ensemble Random Forest classifier calibrated on procurement patterns.
        - **Identifying risk indicators** and attributing positive vs. elevated risk drivers with granular feature importances.
        - **Generating plain-language explanations** and tailored verification checklists for reviewing officers.
        - **Maintaining officer review history** with governed workflow progression (`NOT REVIEWED → UNDER REVIEW → REVIEWED / ESCALATED`).
        - **Maintaining an immutable audit trail** in local SQLite storage recording timestamps, actions, previous statuses, and officer notes.
        """
    )

    # 3. System Architecture
    st.markdown("### 3. System Architecture")
    st.markdown(
        """
        ```
        ┌────────────────────────────────────────────────────────────────────────┐
        │                          Streamlit Web Portal                          │
        │  [Dashboard] [Bid Verification] [Doc Verification] [Risk Analysis]     │
        │             [Model Insights] [Officer Review] [About System]           │
        └───────────────────────────────────┬────────────────────────────────────┘
                                            │
        ┌───────────────────────────────────▼────────────────────────────────────┐
        │                        Processing & AI Pipeline                        │
        │  ┌──────────────────────┐  ┌─────────────────────┐  ┌────────────────┐ │
        │  │ Document Extractor   │  │ Feature Pipeline    │  │ Rule Engine    │ │
        │  │ (pypdf Text Parsing) │  │ (ColumnTransformer) │  │ (21 Checks)    │ │
        │  └──────────┬───────────┘  └──────────┬──────────┘  └───────┬────────┘ │
        │             │                         │                     │          │
        │  ┌──────────▼───────────┐  ┌──────────▼──────────┐  ┌───────▼────────┐ │
        │  │ Consistency &        │  │ Random Forest Model │  │ Risk Scoring   │ │
        │  │ Completeness Engine  │  │ (Calibrated Probs)  │  │ (Multi-tier)   │ │
        │  └──────────────────────┘  └──────────┬──────────┘  └────────────────┘ │
        │                                       │                                │
        │                            ┌──────────▼──────────┐                     │
        │                            │ Explainability Layer│                     │
        │                            │ (Gini Attribution)  │                     │
        │                            └──────────┬──────────┘                     │
        └───────────────────────────────────────┼────────────────────────────────┘
                                                │
        ┌───────────────────────────────────────▼────────────────────────────────┐
        │                      Governance & Persistence Layer                    │
        │      SQLite Audit Database (data/review_audit.db)                      │
        │      • bid_reviews (status, priority, probabilities, remarks)          │
        │      • review_audit_log (immutable timestamped transitions)            │
        └────────────────────────────────────────────────────────────────────────┘
        ```
        """
    )

    # 4. Main Features
    st.markdown("### 4. Main Features")
    st.markdown(
        """
        - **Executive Intelligence Dashboard:** Real-time visibility into overall bid compliance rates, risk tier distributions, and review workflow completion.
        - **Bid Verification Interface:** Interactive screening of tender submissions with pre-loaded demo archetypes and custom data entry.
        - **AI Document Verification:** Ingestion of PDF and TXT files, automated text extraction, keyword auditing, document completeness scoring, and cross-document entity consistency checks.
        - **Multi-Dimensional Risk Analysis:** Drill-down risk screening by state, project type, risk tier, and tender eligibility with interactive visualizations.
        - **Explainable Model Insights:** Transparent model comparison benchmarks, feature importance attribution, test confusion matrix, and process flowcharts.
        - **Human-in-the-Loop Officer Review:** Dedicated scrutiny console with validated status transitions and tamper-evident chronological audit logging.
        """
    )

    # 5. AI/ML Pipeline
    st.markdown("### 5. AI/ML Pipeline")
    st.write(
        "The machine learning subsystem utilizes a calibrated Random Forest classifier trained on 1,000 synthetic procurement records. Continuous variables are scaled via `StandardScaler`, and categorical features are encoded via `OneHotEncoder`. "
        "The model achieves 91.00% test accuracy and 0.9211 F1 score on the synthetic holdout partition, outputting probabilistic confidence scores rather than hard decisions."
    )

    # 6. Human-in-the-Loop Design
    st.markdown("### 6. Human-in-the-Loop Design")
    st.write(
        "A foundational principle of this platform is **assistive intelligence**. The platform strictly adheres to the following governance constraints:"
    )
    st.markdown(
        """
        - **No Automated Qualification or Disqualification:** The system does not approve or reject bids autonomously.
        - **Non-Prescriptive Findings:** AI outputs provide evidence-backed indicators (`Requires officer review`, `Potential compliance concern detected`) rather than legal mandates.
        - **Controlled State Transitions:** Bids move through explicitly validated stages (`NOT REVIEWED → UNDER REVIEW → REVIEWED / ESCALATED`).
        - **Audit Immutability:** Every review action and remark is logged with UTC timestamps and previous states for accountability and compliance auditing.
        """
    )

    # 7. Prototype Limitations
    st.markdown("### 7. Prototype Limitations")
    st.markdown(
        """
        1. **Synthetic / Demo Dataset:** The dataset used for development is synthetic/demo data. Model performance on this dataset does not establish performance on real procurement data.
        2. **Simulated Statutory Portals:** Verification against GSTN, MCA-21, DigiLocker, and EPFO is simulated via heuristic matching and structured schemas rather than live API integrations.
        3. **Text Extraction Boundary:** PDF extraction uses `pypdf` text parsing; scanned, image-only PDFs without OCR layers require supplementary optical character recognition.
        4. **Production Readiness:** Full enterprise adoption requires security audits, official government API credentials, and institutional domain validation.
        """
    )

    # 8. Technology Stack & Component Distinction
    st.markdown("### 8. Technology Stack & Engine Distinction")
    st.markdown(
        """
        | Layer / Purpose | Technology | Primary Function |
        | :--- | :--- | :--- |
        | **Language Runtime** | Python 3.10+ | Core programming and service orchestration |
        | **Data Wrangling** | Pandas, NumPy | Structured tabular data processing and matrix operations |
        | **Machine Learning** | Scikit-learn, Joblib | Model training, preprocessing pipelines, inference, and serialization |
        | **Web Application** | Streamlit | Responsive, interactive procurement officer dashboard |
        | **Data Visualization** | Plotly | Dynamic charting, KPI visual containers, and distribution graphs |
        | **Document Extraction**| PyPDF | Native text parsing from uploaded PDF bid documents |
        | **Audit Trail Storage**| SQLite 3 | Embedded, tamper-evident transactional audit history logging |
        """
    )

    st.markdown("##### Operational Distinction Across System Engines")
    st.markdown(
        """
        - **AI Prediction:** Statistical inference estimating calibrated compliance probability based on historical training patterns.
        - **Rule-Based Checks:** Deterministic statutory validation rules evaluating explicit legal thresholds (e.g., blacklisting flags, debarment records).
        - **Document Heuristics:** Keyword presence detection, completeness ratios, and cross-record string similarity matching.
        - **Human Review:** Definitive scrutiny, professional discretion, and authoritative procurement qualification/disqualification decisions.
        """
    )

    st.markdown("---")
    st.caption("AI-Powered Integrated Bid Compliance Verification Platform | Organization: Chennai Petroleum Corporation Limited (CPCL)")


# -------------------------------------------------------------
# CONDITIONAL PAGE RENDERING DISPATCHER
# -------------------------------------------------------------
if page == "Dashboard":
    render_dashboard_page()
elif page == "Bid Verification":
    render_bid_verification_page()
elif page == "Document Verification":
    render_document_verification_page()
elif page == "Risk Analysis":
    render_risk_analysis_page()
elif page == "Model Insights":
    render_model_insights_page()
elif page == "Officer Review":
    render_officer_review_page()
elif page == "About System":
    render_about_system_page()

