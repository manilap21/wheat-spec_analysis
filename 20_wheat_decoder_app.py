# ================================================================
# wheatspec_app.py  —  WheatSpec v3 (Max-Contrast Sidebar Edition)
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
# CSS  — Times New Roman + Max Contrast Text & Sidebar Controls
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

/* Main layout text elements forced to bright white */
.main .stMarkdown p, .main span, .main label, h2, h3, h4, h5, h6 {
    color: #F1F5F9 !important;
    font-size: 11.5pt;
}

/* ── MAXIMUM SIDEBAR INPUT CONTROLS VISIBILITY ── */
div[data-testid="stSidebarContent"] { 
    background: #1E293B !important; 
    border-right: 2px solid #334155; 
}

/* Force ALL sidebar text, labels, headers, and checkboxes to absolute glowing white */
div[data-testid="stSidebarContent"] label, 
div[data-testid="stSidebarContent"] p, 
div[data-testid="stSidebarContent"] span,
div[data-testid="stSidebarContent"] h2,
div[data-testid="stSidebarContent"] h3,
div[data-testid="stSidebarContent"] div {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Specifically brighten slider numbers and small text area captions */
div[data-testid="stSidebarContent"] small,
div[data-testid="stSidebarContent"] [data-testid="stWidgetLabel"] p,
div[data-testid="stSidebarContent"] .stCaption {
    color: #CBD5E1 !important;
    font-weight: 400 !important;
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
        yc = cross_val_predict(PLSRegression(n_components=n), X_tr, y_tr, cv=LeaveOneOut())
        scores.append(np.sqrt(mean_squared_error(y_tr, yc)))
    best = next(i+1 for i,v in enumerate(scores) if v <= min(scores)*1.01)
    return best, scores

def build_ml(use_bayes, use_rf, use_xgb, rf_n, xgb_n, xgb_lr,
             xgb_depth, n_pca, X_tr):
    pipes = {}
    n_safe = min(n_pca, X_tr.shape[1], X_tr.shape[0]-1)
    if use_bayes:
        pipes["BayesianRidge"] = Pipeline([("sc",StandardScaler()),("m",BayesianRidge())])
    if use_rf:
        pipes["RandomForest"]  = Pipeline([
            ("sc",StandardScaler()),
            ("m",RandomForestRegressor(n_estimators=rf_n, max_features="sqrt",
                                       random_state=42, n_jobs=-1))])
    if use_xgb and n_safe>=1:
        pipes["XGBoost"] = Pipeline([
            ("pca",PCA(n_components=n_safe, random_state=42)),
            ("m",XGBRegressor(n_estimators=xgb_n, learning_rate=xgb_lr,
                               max_depth=xgb_depth, subsample=0.8,
                               random_state=42, verbosity=0))])
    return pipes

# ================================================================
# DATA AUTO-INGESTION HANDLER
# ================================================================
df = wave_cols = non_wave = trait_cols = all_cols = None
if uploaded:
    df         = load_file(uploaded)
    wave_cols  = get_wave_cols(df)
    non_wave   = [c for c in df.columns if c not in wave_cols]
    trait_cols = [c for c in non_wave if c.strip().lower() not in EXCL]
    all_cols   = list(df.columns)

# ================================================================
# INTERACTIVE WORKSPACE CHANNELS
# ================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Spectral preprocessing",
    "📊 Rheology prediction",
    "🔬 Variety classification",
    "🏆 Model tournament",
])

# ──────────────────────────────────────────────────────────────
# TAB 1 — PREPROCESSING MONITOR
# ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Real-time spectral preprocessing monitor")

    if df is None:
        st.markdown('<div class="ibox">📁 Upload a spectral matrix in the sidebar to begin.</div>', unsafe_allow_html=True)
    elif len(wave_cols) < 5:
        st.error("Fewer than 5 numeric wavenumber columns detected — check file format.")
    else:
        st.success(f"**{df.shape[0]} samples** · **{len(wave_cols)} spectral points** · **{len(trait_cols)} trait columns** detected")

        id_col = st.selectbox("Sample label column (hover labels in plot)", ["(none)"] + non_wave)
        if id_col == "(none)": id_col = None

        X_raw  = df[wave_cols].values.astype(float)
        X_proc = preprocess(X_raw, apply_snv, sg_deriv, sg_window, sg_poly)
        wn     = [float(c) for c in wave_cols]
        labels = df[id_col].astype(str).values if id_col else [f"Sample {i}" for i in range(len(df))]
        n_show = min(n_plot, len(df))

        ca, cb = st.columns(2)
        for col, data, title in [(ca, X_raw, "Raw spectra"), (cb, X_proc, "After preprocessing")]:
            with col:
                st.markdown(f'<p class="slabel">{title}</p>', unsafe_allow_html=True)
                fig = go.Figure()
                for i in range(n_show):
                    fig.add_trace(go.Scatter(x=wn, y=data[i], mode="lines", name=labels[i], line=dict(width=0.9), opacity=0.75))
                fig.update_layout(xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed", gridcolor="#334155"),
                                   yaxis_title="Intensity", height=330, template="plotly_dark", 
                                   paper_bgcolor='#1E293B', plot_bgcolor='#1E293B', showlegend=False, margin=dict(l=40,r=10,t=20,b=40))
                st.plotly_chart(fig, use_container_width=True)

        if REGIONS:
            st.markdown('<p class="slabel">Kelly spectral regions overlay</p>', unsafe_allow_html=True)
            palette = px.colors.qualitative.Pastel
            fig_r = go.Figure()
            for i in range(n_show):
                fig_r.add_trace(go.Scatter(x=wn, y=X_proc[i], mode="lines", line=dict(width=0.7, color="#94A3B8"), opacity=0.35, showlegend=False))
            for idx, (rn, (lo, hi)) in enumerate(REGIONS.items()):
                fig_r.add_vrect(x0=lo, x1=hi, fillcolor=palette[idx % len(palette)], opacity=0.18, layer="below", line_width=0,
                                 annotation_text=rn.split(" ")[0], annotation_position="top left", annotation_font_color="#FFFFFF", annotation_font_size=10)
            fig_r.update_layout(xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed", gridcolor="#334155"),
                                yaxis_title="Processed intensity", height=370, template="plotly_dark", paper_bgcolor='#1E293B', plot_bgcolor='#1E293B', margin=dict(l=40,r=10,t=40,b=40))
            st.plotly_chart(fig_r, use_container_width=True)

        if id_col:
            st.markdown('<p class="slabel">Mean spectrum per class</p>', unsafe_allow_html=True)
            df_proc = pd.DataFrame(X_proc, columns=wave_cols)
            df_proc["_label"] = labels
            fig_m = go.Figure()
            for grp, sub in df_proc.groupby("_label"):
                mean_spec = sub.drop("_label", axis=1).mean()
                fig_m.add_trace(go.Scatter(x=wn, y=mean_spec.values, mode="lines", name=str(grp), line=dict(width=2.0)))
            fig_m.update_layout(xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed", gridcolor="#334155"),
                                  yaxis_title="Mean intensity", height=320, template="plotly_dark", paper_bgcolor='#1E293B', plot_bgcolor='#1E293B', margin=dict(l=40,r=10,t=20,b=40))
            st.plotly_chart(fig_m, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 2 — MECHANICAL PHENOTYPES PREDICTOR
# ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Dough rheology prediction — all models, any trait, any region")

    if df is None:
        st.markdown('<div class="ibox">📁 Upload data in the sidebar to begin.</div>', unsafe_allow_html=True)
    elif not trait_cols:
        st.markdown('<div class="wbox">No trait columns detected. Adjust exclusion layers.</div>', unsafe_allow_html=True)
    elif not REGIONS:
        st.markdown('<div class="wbox">No spectral regions defined.</div>', unsafe_allow_html=True)
    elif not any([use_plsr, use_bayes, use_rf, use_xgb]):
        st.markdown('<div class="wbox">Select at least one model in the sidebar.</div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1: sel_trait  = st.selectbox("Trait", trait_cols)
        with c2: sel_region = st.selectbox("Spectral region", list(REGIONS.keys()))

        bench = BENCH.get(sel_trait)
        if bench:
            st.markdown(f'<div class="kbox">📖 <strong>Benchmark R² for {sel_trait}:</strong> {bench} — from your sidebar benchmark settings</div>', unsafe_allow_html=True)

        if st.button("▶ Run prediction", type="primary"):
            df_t = df.dropna(subset=[sel_trait])
            lo, hi = REGIONS[sel_region]
            rc = filter_region(wave_cols, lo, hi)

            if len(rc) < 3:
                st.error(f"Only {len(rc)} spectral points in {sel_region}. Choose a wider region.")
            else:
                X_raw = df_t[rc].values.astype(float)
                X     = preprocess(X_raw, apply_snv, sg_deriv, sg_window, sg_poly)
                y     = df_t[sel_trait].values.astype(float)
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

                results = []
                rmsecv_curve = None

                if use_plsr:
                    with st.spinner("Optimising PLSR via LOO cross-validation…"):
                        best_n, scores = optimise_plsr(X_tr, y_tr, plsr_max)
                        rmsecv_curve = (best_n, scores)
                        pls = PLSRegression(n_components=best_n)
                        pls.fit(X_tr, y_tr)
                        yp = pls.predict(X_te).ravel()
                        results.append({"Model": f"PLSR ({best_n} comp)", "color":"", "R²": round(r2_score(y_te, yp), 4), "RPD": round(rpd_v(y_te, yp), 3), "y_pred": yp})

                ml = build_ml(use_bayes, use_rf, use_xgb, rf_n, xgb_n, xgb_lr, xgb_depth, n_pca, X_tr)
                colors = {"BayesianRidge":"blue", "RandomForest":"", "XGBoost":"purple"}
                for mname, pipe in ml.items():
                    with st.spinner(f"Fitting {mname}…"):
                        try:
                            pipe.fit(X_tr, y_tr)
                            yp = pipe.predict(X_te).ravel()
                            results.append({"Model": mname, "color": colors.get(mname,""), "R²": round(r2_score(y_te, yp), 4), "RPD": round(rpd_v(y_te, yp), 3), "y_pred": yp})
                        except Exception as e:
                            st.warning(f"{mname}: {e}")

                st.markdown('<p class="slabel">Results Leaderboard</p>', unsafe_allow_html=True)
                cols = st.columns(len(results))
                for col, res in zip(cols, results):
                    gr, bc = rpd_badge(res["RPD"])
                    diff = round(res["R²"]-bench, 4) if bench else None
                    with col: st.markdown(mcard(res["Model"], res["R²"], f"RPD {res['RPD']}", gr, bc, diff, res["color"]), unsafe_allow_html=True)

                st.markdown('<p class="slabel">Predicted vs actual parity coordinates</p>', unsafe_allow_html=True)
                nc  = len(results)
                fig = make_subplots(rows=1, cols=nc, subplot_titles=[r["Model"] for r in results])
                lims= [y_te.min()-y_te.std()*0.1, y_te.max()+y_te.std()*0.1]
                pal = ["#22C55E","#3B82F6","#F59E0B","#8B5CF6"]
                for i, res in enumerate(results, 1):
                    fig.add_trace(go.Scatter(x=y_te, y=res["y_pred"], mode="markers", marker=dict(color=pal[(i-1)%len(pal)], size=7, opacity=0.85, line=dict(color="#1E293B",width=0.5)), showlegend=False), row=1, col=i)
                    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(color="#475569",dash="dash",width=1.2), showlegend=False), row=1, col=i)
                fig.update_layout(height=310, template="plotly_dark", paper_bgcolor='#1E293B', plot_bgcolor='#1E293B', margin=dict(l=30,r=10,t=40,b=30))
                st.plotly_chart(fig, use_container_width=True)

                if rmsecv_curve:
                    bn, sc = rmsecv_curve
                    st.markdown('<p class="slabel">PLSR LOO RMSECV Latent Space Optimisation</p>', unsafe_allow_html=True)
                    fc = px.line(x=list(range(1, len(sc)+1)), y=sc, markers=True, labels={"x":"Components","y":"RMSECV"}, color_discrete_sequence=["#22C55E"])
                    fc.add_vline(x=bn, line_dash="dash", line_color="#F59E0B", annotation_text=f"Optimal n={bn}", annotation_position="top right")
                    fc.update_layout(height=250, template="plotly_dark", paper_bgcolor='#1E293B', plot_bgcolor='#1E293B', margin=dict(l=40,r=10,t=20,b=40))
                    st.plotly_chart(fc, use_container_width=True)

                exp = pd.DataFrame([{k:v for k,v in r.items() if k not in ("y_pred","color")} for r in results])
                exp["Benchmark_R2"] = bench
                st.download_button("⬇ Download results CSV", exp.to_csv(index=False), f"{sel_trait}_{sel_region}.csv", "text/csv")

# ──────────────────────────────────────────────────────────────
# TAB 3 — VARIETY VERIFICATION (LDA)
# ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Variety classification — LDA across spectral regions")

    if df is None:
        st.markdown('<div class="ibox">📁 Upload data in the sidebar to begin.</div>', unsafe_allow_html=True)
    elif not non_wave:
        st.warning("No non-spectral columns located.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            var_col   = st.selectbox("Variety / cultivar column", non_wave)
            lda_bench = st.number_input("Classification accuracy benchmark (%)", 0.0, 100.0, 80.0, 1.0) / 100
            upper_case = st.checkbox("Auto-uppercase variety labels", value=True)
        with c2:
            prep_opts = { "Raw": (False, 0), "SG only": (False, sg_deriv), "SNV only": (True, 0), "SNV + SG": (True, sg_deriv) }
            sel_preps = st.multiselect("Preprocessing combos to test", list(prep_opts.keys()), default=["Raw","SNV + SG"])
            show_3d = st.checkbox("Show 3D LDA scatter plane", value=True)
            filter_classes = st.text_input("Filter specific varieties (comma-separated, blank = all)", value="")

        if st.button("▶ Run LDA classification", type="primary"):
            df_v = df.dropna(subset=[var_col]).copy()
            if upper_case: df_v[var_col] = df_v[var_col].astype(str).str.upper().str.strip()
            if filter_classes.strip():
                keep = [v.strip().upper() for v in filter_classes.split(",")]
                df_v = df_v[df_v[var_col].isin(keep)]

            le    = LabelEncoder()
            y_v   = le.fit_transform(df_v[var_col])
            names = list(le.classes_)
            nc    = len(names)

            if nc < 2:
                st.error("Need ≥ 2 valid variety classes.")
            else:
                st.markdown(f'<div class="ibox">Active Classes: <b>{", ".join(names)}</b> · Rows: {len(df_v)}</div>', unsafe_allow_html=True)
                records = []
                total = len(REGIONS) * len(sel_preps)
                prog  = st.progress(0); step = 0

                for rname, (lo, hi) in REGIONS.items():
                    rc = filter_region(wave_cols, lo, hi)
                    if len(rc) < 3: step+=len(sel_preps); continue
                    X_r = df_v[rc].values.astype(float)

                    for pname in sel_preps:
                        snv_, sg_ = prep_opts[pname]
                        Xp = preprocess(X_r, snv_, sg_, sg_window, sg_poly)
                        Xs = StandardScaler().fit_transform(Xp)
                        try:
                            yp  = cross_val_predict(LinearDiscriminantAnalysis(), Xs, y_v, cv=LeaveOneOut())
                            acc = accuracy_score(y_v, yp)
                            per = {names[i]: round(accuracy_score(y_v[y_v==i], yp[y_v==i]),3) for i in range(nc) if (y_v==i).sum()>0}
                            records.append({ "Region":rname, "Preprocessing":pname, "Accuracy":round(acc,4), **{f"Acc_{v}":a for v,a in per.items()} })
                        except Exception: pass
                        step+=1; prog.progress(step/total)

                prog.empty()
                if not records: st.error("LDA failed on all tracking configurations.")
                else:
                    res = pd.DataFrame(records).sort_values("Accuracy", ascending=False)
                    best = res.iloc[0]
                    diff = best["Accuracy"] - lda_bench
                    c = "#4ADE80" if diff>=0 else "#F87171"

                    st.markdown(f"""<div class="mcard" style="max-width:380px"><div class="mlabel">Best configuration</div><div class="mvalue">{best['Accuracy']:.1%}</div><div class="msub">{best['Region']} · {best['Preprocessing']}</div><div class="mdiff" style="color:{c}">{"↑" if diff>=0 else "↓"} {diff:+.1%} vs benchmark ({lda_bench:.0%})</div></div>""", unsafe_allow_html=True)

                    pivot = res.pivot_table(index="Region", columns="Preprocessing", values="Accuracy", aggfunc="max")
                    fig_hm = px.imshow(pivot, text_auto=".0%", color_continuous_scale="RdYlGn", zmin=max(0.2, lda_bench-0.3), zmax=1.0)
                    fig_hm.update_layout(height=max(280,len(REGIONS)*48), template="plotly_dark", paper_bgcolor='#1E293B', plot_bgcolor='#1E293B', margin=dict(l=10,r=10,t=20,b=10))
                    st.plotly_chart(fig_hm, use_container_width=True)

                    if show_3d:
                        best_rname = best["Region"]
                        lo2, hi2   = REGIONS[best_rname]
                        rc2 = filter_region(wave_cols, lo2, hi2)
                        snv_, sg_  = prep_opts[best["Preprocessing"]]
                        Xp2 = preprocess(df_v[rc2].values.astype(float), snv_, sg_, sg_window, sg_poly)
                        Xs2 = StandardScaler().fit_transform(Xp2)
                        n_lda_comp = min(3, nc-1, Xs2.shape[1])
                        n_pca_comp = min(min(12, Xs2.shape[1], Xs2.shape[0]-1))
                        X_pca2 = PCA(n_components=n_pca_comp, random_state=42).fit_transform(Xs2)
                        X_lda2 = LinearDiscriminantAnalysis(n_components=n_lda_comp).fit_transform(X_pca2, y_v)

                        df3d = pd.DataFrame(X_lda2, columns=[f"LD{i+1}" for i in range(n_lda_comp)])
                        df3d["Variety"] = names[y_v] if len(y_v)==len(df3d) else df_v[var_col].values
                        colour_map = {k:v for k,v in {v: CULT_COLOURS.get(v, None) for v in names}.items() if v}

                        st.markdown(f'<p class="slabel">3D LDA variety discriminant plane space — best region: {best_rname}</p>', unsafe_allow_html=True)
                        if n_lda_comp >= 3: fig3d = px.scatter_3d(df3d, x="LD1", y="LD2", z="LD3", color="Variety", color_discrete_map=colour_map, opacity=0.9)
                        else: fig3d = px.scatter(df3d, x="LD1", y="LD2" if n_lda_comp>=2 else "LD1", color="Variety", color_discrete_map=colour_map, opacity=0.9)
                        fig3d.update_traces(marker=dict(size=6, line=dict(width=0.5, color='#1E293B')))
                        fig3d.update_layout(height=520, template="plotly_dark", paper_bgcolor='#1E293B', plot_bgcolor='#1E293B', margin=dict(l=0,r=0,t=30,b=0))
                        st.plotly_chart(fig3d, use_container_width=True)

                    st.markdown('<p class="slabel">Exhaustive Evaluation Logs</p>', unsafe_allow_html=True)
                    st.dataframe(res.reset_index(drop=True), use_container_width=True)
                    st.download_button("⬇ Download LDA results", res.to_csv(index=False), "lda_results.csv", "text/csv")

# ──────────────────────────────────────────────────────────────
# TAB 4 — AUTOML TOURNAMENT ENGINE
# ──────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Full model tournament — all regions × all models")
    st.markdown("""<div class="wbox">⏱ Parallel calibration loop running all selected models across every defined spectral partition window grid.</div>""", unsafe_allow_html=True)

    if df is None: st.markdown('<div class="ibox">📁 Upload data in the sidebar to begin.</div>', unsafe_allow_html=True)
    elif not trait_cols: st.warning("No parameters located in trait array columns.")
    elif not REGIONS: st.warning("No regional spectra slices defined.")
    elif not any([use_plsr,use_bayes,use_rf,use_xgb]): st.warning("Activate at least one validation core checkbox.")
    else:
        sel_targets = st.multiselect("Traits to include in tournament", trait_cols, default=trait_cols[:min(3,len(trait_cols))])

        if st.button("▶ Launch tournament", type="primary") and sel_targets:
            all_rec = []
            total   = len(sel_targets)*len(REGIONS)
            prog    = st.progress(0); status = st.empty(); step = 0

            for target in sel_targets:
                df_t  = df.dropna(subset=[target])
                y_all = df_t[target].values.astype(float)
                bench = BENCH.get(target)

                for rname, (lo, hi) in REGIONS.items():
                    rc = filter_region(wave_cols, lo, hi)
                    status.text(f"Running: {target} × {rname}…")
                    if len(rc) < 3: step+=1; prog.progress(step/total); continue

                    X_raw = df_t[rc].values.astype(float)
                    X     = preprocess(X_raw, apply_snv, sg_deriv, sg_window, sg_poly)
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y_all, test_size=test_size, random_state=42)

                    def _rec(mn, yp):
                        r2  = round(r2_score(y_te,yp),4)
                        rpd = round(rpd_v(y_te,yp),3)
                        gr, _ = rpd_badge(rpd)
                        return {"Target":target,"Region":rname,"Model":mn,"R²":r2,"RPD":rpd,"Grade":gr, "Benchmark":bench, "vs_Benchmark":round(r2-bench,4) if bench else None}

                    if use_plsr:
                        try:
                            bn,_ = optimise_plsr(X_tr, y_tr, plsr_max)
                            pls  = PLSRegression(n_components=bn)
                            pls.fit(X_tr, y_tr)
                            all_rec.append(_rec(f"PLSR({bn})", pls.predict(X_te).ravel()))
                        except Exception: pass

                    ml = build_ml(use_bayes,use_rf,use_xgb,rf_n,xgb_n,xgb_lr,xgb_depth,n_pca,X_tr)
                    for mn, pipe in ml.items():
                        try:
                            pipe.fit(X_tr, y_tr)
                            all_rec.append(_rec(mn, pipe.predict(X_te).ravel()))
                        except Exception: pass

                    step+=1; prog.progress(step/total)

            prog.empty(); status.empty()

            if not all_rec: st.error("Validation loop yielded zero output arrays.")
            else:
                res_df = pd.DataFrame(all_rec)

                st.markdown('<p class="slabel">Tournament Winner Standings</p>', unsafe_allow_html=True)
                best_df = (res_df.sort_values("R²", ascending=False).groupby("Target").first().reset_index())
                cols = st.columns(min(len(best_df),5))
                for col, row in zip(cols, best_df.itertuples()):
                    gr, bc = rpd_badge(row.RPD)
                    with col: st.markdown(mcard(row.Target, row._5, f"{row.Model} · {str(row.Region)[:14]}", gr, bc, row.vs_Benchmark), unsafe_allow_html=True)

                st.markdown('<p class="slabel">R² Performance Scaling by Region and Model Architecture</p>', unsafe_allow_html=True)
                for target in sel_targets:
                    sub = res_df[res_df["Target"]==target]
                    fig = px.bar(sub, x="Region", y="R²", color="Model", barmode="group", title=target, color_discrete_sequence=px.colors.qualitative.Set2)
                    b = BENCH.get(target)
                    if b: fig.add_hline(y=b, line_dash="dash", line_color="#F59E0B", annotation_text=f"Benchmark {b}", annotation_position="top right")
                    fig.update_layout(height=330, template="plotly_dark", paper_bgcolor='#1E293B', plot_bgcolor='#1E293B', margin=dict(l=40,r=10,t=40,b=90), xaxis_tickangle=-30)
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown('<p class="slabel">Global Leaderboard Floorgrid Matrix</p>', unsafe_allow_html=True)
                def _highlight(row):
                    if row["R²"] == res_df["R²"].max(): return ["background:#14532D;font-weight:700;color:#22C55E"]*len(row)
                    return [""]*len(row)

                st.dataframe(res_df.drop(columns=["Benchmark"], errors="ignore").style.apply(_highlight, axis=1), use_container_width=True)
                st.download_button("⬇ Download full results CSV", res_df.to_csv(index=False), "tournament_results.csv", "text/csv")

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown("""
<p style="font-size:0.82rem;color:#64748B;text-align:center;padding:0.5rem 0">
  WheatSpec v3 &nbsp;·&nbsp; Manuli Perera &nbsp;·&nbsp; UWA Dissertation 2026 &nbsp;·&nbsp; Open-source MIT licence &nbsp;·&nbsp; Built on <em>Kelly et al. (2023) Food Chemistry</em>
</p>""", unsafe_allow_html=True)
