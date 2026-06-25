# ================================================================
# wheatspec_app.py  —  WheatSpec v3 (Ultra-Contrast Edition)
# Manuli Perera | UWA Dissertation 2026 | Open-source MIT
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import (
    train_test_split, LeaveOneOut, cross_val_predict
)
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="WheatSpec | FTIR Chemometrics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# CSS  — High-Contrast White-on-Dark Typography + Times New Roman
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

/* Force Times New Roman globally across all structural layouts */
html, body, [class*="css"], div, p, span, label, button, .stTabs, h1, h2, h3, h4, h5, h6, select, input { 
    font-family: 'Times New Roman', Times, Georgia, serif !important; 
}

/* Hardcode absolute dark viewports to override light-mode browser settings */
.main { background: #0F172A !important; }
div[data-testid="stAppViewContainer"] { background-color: #0F172A !important; }
div[data-testid="stHeader"] { background-color: #0F172A !important; }

/* Global text elements forced to bright chalk-white for optimal contrast */
.main .stMarkdown p, .main span, .main label, h2, h3, h4, h5, h6 {
    color: #F1F5F9 !important;
    font-size: 11.5pt;
}

/* ── Premium Midnight Hero Title Block ── */
.hero {
  background: #1E293B;
  border-radius: 12px; 
  padding: 2.2rem 2rem; 
  margin-bottom: 1.5rem;
  color: #F8FAFC; 
  border: 1px solid #334155;
  border-bottom: 3.5px solid #22C55E;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 6px 0; color: #FFFFFF !important; }
.hero p { font-size: 1.1rem; opacity: 0.95; margin: 0; line-height: 1.5; color: #E2E8F0 !important; }
.hero-chips { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }
.chip { background: #334155; border: 1px solid #475569;
        border-radius: 4px; padding: 3px 12px; font-size: 0.85rem; color: #FFFFFF !important; }

/* ── High-Contrast Dashboard Metric Cards ── */
.mcard { background: #1E293B; border: 1px solid #475569; border-radius: 8px;
         padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
         border-left: 5px solid #22C55E; box-shadow: 0 4px 12px rgba(0,0,0,0.25); }
.mcard.blue { border-left-color: #3B82F6; }
.mcard.amber { border-left-color: #F59E0B; }
.mcard.purple { border-left-color: #8B5CF6; }
.mlabel { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em;
          color: #CBD5E1 !important; font-weight: 600; margin-bottom: 6px; }
.mvalue { font-size: 2rem; font-weight: 700; color: #FFFFFF !important;
          font-family: 'JetBrains Mono', monospace !important; line-height: 1.1; }
.msub { font-size: 0.95rem; color: #E2E8F0 !important; margin-top: 5px; }
.mdiff { font-size: 0.95rem; margin-top: 4px; font-weight: 700; }

/* ── Status and Evaluation Badges ── */
.badge { display: inline-block; padding: 3px 9px; border-radius: 4px;
         font-size: 0.8rem; font-weight: 700; text-transform: uppercase; margin-left: 5px; }
.bexc  { background: #14532D; color: #4ADE80; border: 1px solid #22C55E; }
.bqc   { background: #1E3A8A; color: #93C5FD; border: 1px solid #3B82F6; }
.bscr  { background: #713F12; color: #FDE047; border: 1px solid #EAB308; }
.bpoor { background: #7F1D1D; color: #FCA5A5; border: 1px solid #EF4444; }

/* ── Context Notification Boxes ── */
.kbox { background: #14532D; border: 1px solid #22C55E; border-radius: 6px;
        padding: 0.9rem 1.2rem; font-size: 1rem; color: #FFFFFF !important; margin: 0.8rem 0; }
.wbox { background: #78350F; border: 1px solid #EAB308; border-radius: 6px;
        padding: 0.9rem 1.2rem; font-size: 1rem; color: #FFFFFF !important; margin: 0.8rem 0; }
.ibox { background: #1E3A8A; border: 1px solid #3B82F6; border-radius: 6px;
        padding: 0.9rem 1.2rem; font-size: 1rem; color: #FFFFFF !important; margin: 0.8rem 0; }

/* ── Publication Section Labels ── */
.slabel { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.08em;
          color: #94A3B8; font-weight: 700; margin: 1.6rem 0 0.6rem; }

/* ── Sidebar Panels & Input Fields Overrides ── */
div[data-testid="stSidebarContent"] { background: #1E293B !important; border-right: 2px solid #334155; }
div[data-testid="stSidebarContent"] .stMarkdown p, div[data-testid="stSidebarContent"] label, div[data-testid="stSidebarContent"] caption { 
    color: #FFFFFF !important; 
    font-size: 11pt;
}
div[data-testid="stSidebarContent"] .stCaption { color: #CBD5E1 !important; font-size: 9.5pt; }

/* Tab Bar Adjustments */
.stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #0F172A; padding: 4px; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { background-color: #1E293B; border: 1px solid #334155; border-radius: 6px; padding: 0.6rem 1.5rem; color: #CBD5E1 !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #22C55E !important; color: #0F172A !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# HELPERS
# ================================================================

@st.cache_data
def load_file(f):
    return pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f)

def get_wave_cols(df):
    c = []
    for col in df.columns:
        try: float(str(col).strip()); c.append(col)
        except ValueError: pass
    return sorted(c, key=lambda x: float(str(x).strip()))

def filter_region(wave_cols, lo, hi):
    return [c for c in wave_cols if lo <= float(c) <= hi]

def preprocess(X, snv, sg_deriv, sg_window, sg_poly):
    X = X.copy().astype(float)
    if snv:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True); sd[sd==0]=1.0
        X = (X-mu)/sd
    wl = min(sg_window, X.shape[1])
    if wl % 2 == 0: wl -= 1
    if wl > sg_poly and sg_deriv >= 0:
        X = savgol_filter(X, window_length=wl, polyorder=sg_poly,
                          deriv=sg_deriv, axis=1)
    return np.nan_to_num(X)

def rpd_v(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(np.std(y_true)/rmse) if rmse>1e-9 else np.inf

def rpd_badge(rpd):
    if rpd>=5.0: return "🚀 Excellent","bexc"
    if rpd>=2.9: return "⭐ QC Suitable","bqc"
    if rpd>=1.9: return "✅ Screening","bscr"
    return "❌ Poor","bpoor"

def mcard(label, value, sub, badge_t=None, badge_c=None, diff=None, color=""):
    badge = f'<span class="badge {badge_c}">{badge_t}</span>' if badge_t else ""
    diff_html = ""
    if diff is not None:
        c = "#4ADE80" if diff>=0 else "#F87171"
        diff_html = f'<div class="mdiff" style="color:{c}">{"↑" if diff>=0 else "↓"} {diff:+.3f} vs benchmark</div>'
    return f"""<div class="mcard {color}">
        <div class="mlabel">{label}</div>
        <div class="mvalue">{value}</div>
        <div class="msub">{sub}&nbsp;{badge}</div>
        {diff_html}</div>"""

def optimise_plsr(X_tr, y_tr, max_comp):
    mc = min(max_comp, X_tr.shape[1], X_tr.shape[0]-1)
    if mc < 1: return 1, [0.0]
    scores = []
    for n in range(1, mc+1):
        yc = cross_
