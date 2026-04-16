import sys
from pathlib import Path
import streamlit as st

# ─── CORE PATH RESOLUTION ──────────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT  = DASHBOARD_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── SINGLE SOURCE OF TRUTH IMPORTS ─────────────────────────────────────────
try:
    from utils.config import (
        SAVED_MODELS_DIR, LOGS_DIR, ARTIFACTS_DIR,
        CHECKPOINT_AUTOENCODER, CHECKPOINT_LSTM, 
        CHECKPOINT_HYBRID, CHECKPOINT_RF, CHECKPOINT_CNN,
        DASHBOARD_TITLE, DASHBOARD_ICON
    )
except ImportError as e:
    st.error(f"❌ Structural Synchronization Failure: {e}")
    st.stop()

# ─── RESOURCE VALIDATION ─────────────────────────────────────────────────────

def check_shap_installed():
    """Detects SHAP availability."""
    try:
        import shap
        return True
    except ImportError:
        return False

def validate_resource(path):
    """Generic file/directory existence check."""
    return Path(path).exists()

def get_available_models():
    """Returns labels for models found in SAVED_MODELS_DIR."""
    models = []
    if validate_resource(CHECKPOINT_AUTOENCODER): models.append("Autoencoder")
    if validate_resource(CHECKPOINT_LSTM):        models.append("BiLSTM")
    if validate_resource(CHECKPOINT_CNN):         models.append("CNN")
    if validate_resource(CHECKPOINT_HYBRID):      models.append("Hybrid")
    if validate_resource(CHECKPOINT_RF):          models.append("Random Forest")
    return models

def safe_page_header(title_suffix):
    """Standardized dashboard header."""
    st.set_page_config(
        page_title=f"{DASHBOARD_TITLE} | {title_suffix}", 
        page_icon=DASHBOARD_ICON, 
        layout="wide"
    )
    st.title(f"{DASHBOARD_ICON} {title_suffix}")
    st.markdown("---")

def apply_global_styles():
    """Consistent light theme styling with high readability."""
    st.markdown("""
    <style>
        .main { background-color: #ffffff; }
        [data-testid="stMetricValue"] { color: #1a1a1a !important; font-weight: 700; font-size: 2.2rem; }
        [data-testid="stMetricLabel"] { color: #555555 !important; font-size: 1.1rem; }
        .stMetric {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e0e0e0;
        }
        h1, h2, h3 { color: #1a1a1a !important; }
        .stMarkdown { color: #1a1a1a; }
    </style>
    """, unsafe_allow_html=True)
