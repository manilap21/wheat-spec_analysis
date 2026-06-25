# =============================================================
# wheat_decoder_app.py  —  WheatSpec v2 (fully dynamic)
# Manuli Perera | UWA Dissertation 2026 | Open-source MIT
#
# Run:     streamlit run wheat_decoder_app.py
# Install: pip install streamlit plotly scikit-learn xgboost scipy openpyxl
#
# FULLY DYNAMIC — nothing is hardcoded:
#   ✓ Spectral regions — user can add / edit / delete in sidebar
#   ✓ Benchmark values — user can enter their own per-trait
#   ✓ Target traits    — auto-detected from uploaded CSV columns
#   ✓ Variety column   — user selects from any column
#   ✓ ID / label column— user selects from any column
#   ✓ Model selection  — user toggles which models to include
#   ✓ Preprocessing    — all parameters are sliders/checkboxes
#   ✓ Test split size  — slider
#   ✓ PLSR max comps   — slider
#   ✓ RF n_estimators  — slider
#   ✓ XGBoost params   — sliders
#   ✓ n_samples plotted— slider (prevents browser freeze)
#   ✓ Export all results as CSV at any tab
# =============================================================

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

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="WheatSpec | FTIR Chemometrics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================
# CSS
# =============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.main{background:#FAFAF8}
.hero{background:linear-gradient(135deg,#1B4332 0%,#2D6A4F 60%,#40916C 100%);border-radius:12px;padding:2.5rem 2rem;margin-bottom:1.5rem;color:white}
.hero h1{font-size:2rem;font-weight:600;margin:0 0 0.4rem 0;letter-spacing:-0.5px}
.hero p{font-size:0.92rem;opacity:0.85;margin:0}
.mcard{background:white;border:1px solid #E8E8E4;border-radius:10px;padding:1rem 1.25rem;margin-bottom:0.75rem}
.mlabel{font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#888;font-weight:500}
.mvalue{font-size:1.6rem;font-weight:600;color:#1B4332;font-family:'JetBrains Mono',monospace}
.msub{font-size:0.78rem;color:#666;margin-top:2px}
.slabel{font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;color:#888;font-weight:600;margin:1.5rem 0 0.6rem}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;white-space:nowrap}
.bexc{background:#D1FAE5;color:#065F46}
.bqc{background:#DBEAFE;color:#1E40AF}
.bscr{background:#FEF3C7;color:#92400E}
.bpoor{background:#FEE2E2;color:#991B1B}
.kbox{background:#F0FDF4;border:1px solid #86EFAC;border-radius:8px;padding:0.75rem 1rem;font-size:0.85rem;color:#166534;margin:0.75rem 0}
.wbox{background:#FFFBEB;border:1px solid #FCD34D;border-radius:8px;padding:0.75rem 1rem;font-size:0.85rem;color:#92400E;margin:0.75rem 0}
div[data-testid="stSidebarContent"]{background:#F7F7F5}
</style>
""", unsafe_allow_html=True)

# =============================================================
# HELPERS
# =============================================================

@st.cache_data
def load_file(uploaded):
    if uploaded.name.endswith(".xlsx"):
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded)

def get_wave_cols(df):
    cols = []
    for c in df.columns:
        try:
            float(str(c).strip())
            cols.append(c)
        except ValueError:
            pass
    return sorted(cols, key=lambda x: float(str(x).strip()))

def filter_region(wave_cols, lo, hi):
    return [c for c in wave_cols if lo <= float(c) <= hi]

def apply_preprocessing(X, snv, sg_deriv, sg_window, sg_poly):
    X = X.copy().astype(float)
    if snv:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd
    if sg_deriv >= 0:
        wl = sg_window
        # ensure window is valid
        if wl > X.shape[1]:
            wl = X.shape[1] - (0 if X.shape[1] % 2 == 1 else 1)
        if wl >= sg_poly + 1:
            X = savgol_filter(X, window_length=wl,
                              polyorder=sg_poly, deriv=sg_deriv, axis=1)
    if np.isnan(X).any():
        X = np.nan_to_num(X)
    return X

def rpd_val(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(np.std(y_true) / rmse) if rmse > 1e-9 else np.inf

def rpd_badge(rpd):
    if rpd >= 5.0: return "🚀 Excellent",    "bexc"
    if rpd >= 2.9: return "⭐ QC Suitable",  "bqc"
    if rpd >= 1.9: return "✅ Screening",    "bscr"
    return               "❌ Poor",          "bpoor"

def optimise_plsr(X_tr, y_tr, max_comp):
    loo = LeaveOneOut()
    scores = []
    mc = min(max_comp, X_tr.shape[1], X_tr.shape[0] - 1)
    if mc < 1:
        return 1, [0.0]
    for n in range(1, mc + 1):
        yc = cross_val_predict(PLSRegression(n_components=n), X_tr, y_tr, cv=loo)
        scores.append(np.sqrt(mean_squared_error(y_tr, yc)))
    best = next(i+1 for i,v in enumerate(scores) if v <= min(scores)*1.01)
    return best, scores

def metric_card(label, value, sub, badge_text=None, badge_cls=None, diff=None):
    diff_html = ""
    if diff is not None:
        col = "#065F46" if diff >= 0 else "#991B1B"
        sym = "↑" if diff >= 0 else "↓"
        diff_html = f'<div class="msub" style="color:{col}">{sym} {diff:+.3f} vs benchmark</div>'
    badge_html = ""
    if badge_text:
        badge_html = f'<span class="badge {badge_cls}">{badge_text}</span>'
    return f"""
    <div class="mcard">
        <div class="mlabel">{label}</div>
        <div class="mvalue">{value}</div>
        <div class="msub">{sub} &nbsp;{badge_html}</div>
        {diff_html}
    </div>"""

# =============================================================
# HERO
# =============================================================
st.markdown("""
<div class="hero">
  <h1>🌾 WheatSpec</h1>
  <p>Open-source FTIR chemometrics platform — upload any spectral matrix,
  configure regions and benchmarks, run PLSR · Bayesian · Random Forest · XGBoost
  and compare results. Built on Kelly et al. (2023) methodology.</p>
</div>
""", unsafe_allow_html=True)

# =============================================================
# SIDEBAR — ALL CONFIGURATION LIVES HERE
# =============================================================
with st.sidebar:
    st.markdown("## 📁 Data")
    uploaded = st.file_uploader("FTIR spectral matrix (.csv or .xlsx)",
                                 type=["csv","xlsx"])

    st.markdown("---")
    st.markdown("## ⚙️ Preprocessing")
    apply_snv  = st.checkbox("Standard Normal Variate (SNV)", value=True)
    sg_deriv   = st.selectbox("Savitzky-Golay derivative", [0,1,2], index=1,
                               help="0=smoothing only, 1=baseline removal, 2=band resolution")
    sg_window  = st.slider("SG window size (must be odd)", 5, 31, 11, 2)
    sg_poly    = st.slider("SG polynomial order", 1, 4, 2,
                           help="Must be < window size")
    n_plot     = st.slider("Max spectra to plot (performance)", 5, 100, 40, 5)

    st.markdown("---")
    st.markdown("## 🔬 Models to run")
    use_plsr   = st.checkbox("PLSR",          value=True)
    use_bayes  = st.checkbox("BayesianRidge", value=True)
    use_rf     = st.checkbox("RandomForest",  value=True)
    use_xgb    = st.checkbox("XGBoost",       value=True)

    st.markdown("---")
    st.markdown("## 🎛 Model hyperparameters")
    plsr_max_comp = st.slider("PLSR max components to test", 2, 20, 10)
    test_size     = st.slider("Test set fraction", 0.10, 0.40, 0.20, 0.05)
    rf_n_trees    = st.slider("RF n_estimators",  50, 500, 150, 50)
    xgb_n_trees   = st.slider("XGBoost n_estimators", 50, 500, 150, 50)
    xgb_lr        = st.select_slider("XGBoost learning rate",
                                      [0.005,0.01,0.02,0.05,0.1,0.2], value=0.05)
    xgb_depth     = st.slider("XGBoost max_depth", 2, 8, 3)
    n_pca_xgb     = st.slider("XGBoost PCA components", 5, 50, 15)

    st.markdown("---")
    st.markdown("## 📐 Spectral regions")
    st.caption("Edit, add or remove regions below. Format: name | low | high (cm⁻¹)")

    default_regions = [
        "A1 Moisture | 2990 | 3680",
        "A2 Fat C-H | 2825 | 2990",
        "A3 Fat C=O | 1710 | 1775",
        "A4 Amide I+II | 1480 | 1710",
        "A5 Amide III | 1180 | 1480",
        "A6 Starch | 810 | 1180",
        "Full Spectrum | 650 | 4000",
    ]
    region_text = st.text_area(
        "Regions (one per line: name | low | high)",
        value="\n".join(default_regions),
        height=220,
        help="Add or remove any region. Separate name and wavenumbers with |"
    )

    # Parse regions dynamically
    REGIONS = {}
    for line in region_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            try:
                REGIONS[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                st.warning(f"Skipping invalid region line: {line}")

    st.markdown("---")
    st.markdown("## 📊 Benchmarks")
    st.caption("Enter published R² benchmarks per trait (leave blank to skip comparison)")
    benchmark_text = st.text_area(
        "Benchmarks (one per line: Trait | R²)",
        value="Protein | 0.963\nExtensibility | 0.927\nAbsorption | 0.700\nRmax | 0.482\nDDT | 0.500",
        height=140,
    )
    BENCHMARKS = {}
    for line in benchmark_text.strip().split("\n"):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 2:
            try:
                BENCHMARKS[parts[0]] = float(parts[1])
            except ValueError:
                pass

    st.markdown("---")
    st.markdown("## 🏷 Identity columns")
    st.caption("These column names will be excluded from trait detection")
    id_col_names = st.text_input(
        "Non-trait columns (comma-separated)",
        value="Variety,variety,Cultivar,cultivar,Sample,sample,ID,id,Name,name,Seed"
    )
    ID_COLS = {c.strip().lower() for c in id_col_names.split(",")}

# =============================================================
# LOAD DATA — shared across all tabs
# =============================================================
df           = None
wave_cols    = []
non_wave     = []
trait_cols   = []
available_id = []

if uploaded:
    df        = load_file(uploaded)
    wave_cols = get_wave_cols(df)
    non_wave  = [c for c in df.columns if c not in wave_cols]
    trait_cols= [c for c in non_wave if c.strip().lower() not in ID_COLS]
    available_id = non_wave  # any column can be an ID

# =============================================================
# TABS
# =============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Spectral preprocessing",
    "📊 Rheology prediction",
    "🔬 Variety classification",
    "🏆 Model tournament",
])

# ──────────────────────────────────────────────────────────────
# TAB 1 — PREPROCESSING VISUALISER
# ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Real-time spectral preprocessing monitor")

    if df is None:
        st.info("Upload a spectral matrix in the sidebar to begin.")
    elif len(wave_cols) < 5:
        st.error("Fewer than 5 numeric wavenumber columns found. Check file format.")
    else:
        st.success(f"Loaded **{df.shape[0]} samples** · **{len(wave_cols)} spectral points** · "
                   f"**{len(trait_cols)} trait columns** detected")

        # Label column selection — dynamic
        id_col = None
        if available_id:
            id_col = st.selectbox("Sample label column (for plot hover)", ["(none)"] + available_id)
            if id_col == "(none)":
                id_col = None

        X_raw  = df[wave_cols].values.astype(float)
        X_proc = apply_preprocessing(X_raw, apply_snv, sg_deriv, sg_window, sg_poly)
        wn     = [float(c) for c in wave_cols]
        labels = (df[id_col].astype(str).values if id_col
                  else [f"Sample {i}" for i in range(len(df))])

        n_show = min(n_plot, len(df))

        col_a, col_b = st.columns(2)
        for ax, data, title in [(col_a, X_raw, "Raw spectra"),
                                  (col_b, X_proc, "After preprocessing")]:
            with ax:
                st.markdown(f'<p class="slabel">{title}</p>', unsafe_allow_html=True)
                fig = go.Figure()
                for i in range(n_show):
                    fig.add_trace(go.Scatter(x=wn, y=data[i], mode="lines",
                                             name=labels[i],
                                             line=dict(width=0.8), opacity=0.7))
                fig.update_layout(
                    xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed"),
                    yaxis_title="Intensity", height=340,
                    template="plotly_white", showlegend=False,
                    margin=dict(l=40, r=10, t=20, b=40))
                st.plotly_chart(fig, use_container_width=True)

        # Region overlay — uses dynamic REGIONS dict
        if REGIONS:
            st.markdown('<p class="slabel">Spectral regions overlay</p>', unsafe_allow_html=True)
            palette = px.colors.qualitative.Pastel
            fig_reg = go.Figure()
            for i in range(n_show):
                fig_reg.add_trace(go.Scatter(x=wn, y=X_proc[i], mode="lines",
                                              line=dict(width=0.7, color="#374151"),
                                              opacity=0.4, showlegend=False))
            for idx, (rname, (lo, hi)) in enumerate(REGIONS.items()):
                fig_reg.add_vrect(x0=lo, x1=hi,
                                   fillcolor=palette[idx % len(palette)],
                                   opacity=0.35, layer="below", line_width=0,
                                   annotation_text=rname.split(" ")[0],
                                   annotation_position="top left",
                                   annotation_font_size=9)
            fig_reg.update_layout(
                xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed"),
                yaxis_title="Processed intensity", height=380,
                template="plotly_white",
                margin=dict(l=40, r=10, t=40, b=40),
                title="Preprocessed spectra with user-defined region boundaries")
            st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.warning("No valid regions defined in the sidebar. Add at least one region.")

# ──────────────────────────────────────────────────────────────
# TAB 2 — RHEOLOGY PREDICTION
# ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Dough rheology prediction")

    if df is None:
        st.info("Upload a spectral matrix in the sidebar to begin.")
    elif not trait_cols:
        st.warning("No trait columns detected. Check your 'Non-trait columns' setting in the sidebar.")
    elif not REGIONS:
        st.warning("No spectral regions defined in the sidebar.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            sel_target = st.selectbox("Trait to predict", trait_cols)
        with c2:
            sel_region = st.selectbox("Spectral region", list(REGIONS.keys()))

        if not any([use_plsr, use_bayes, use_rf, use_xgb]):
            st.warning("Select at least one model in the sidebar.")
        else:
            run_btn = st.button("▶ Run prediction", type="primary")
            if run_btn:
                df_t = df.dropna(subset=[sel_target])
                lo, hi = REGIONS[sel_region]
                rc = filter_region(wave_cols, lo, hi)

                if len(rc) < 3:
                    st.error(f"Only {len(rc)} spectral points in {sel_region}. Choose wider region.")
                else:
                    X_raw = df_t[rc].values.astype(float)
                    X     = apply_preprocessing(X_raw, apply_snv, sg_deriv, sg_window, sg_poly)
                    y     = df_t[sel_target].values.astype(float)

                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=test_size, random_state=42)

                    results  = []
                    rmsecv_data = None

                    if use_plsr:
                        with st.spinner("Optimising PLSR via LOO cross-validation..."):
                            best_n, scores = optimise_plsr(X_tr, y_tr, plsr_max_comp)
                            rmsecv_data = (best_n, scores)
                            pls = PLSRegression(n_components=best_n)
                            pls.fit(X_tr, y_tr)
                            yp = pls.predict(X_te).ravel()
                            results.append({"Model": f"PLSR ({best_n} comp)",
                                            "R²": round(r2_score(y_te, yp), 4),
                                            "RPD": round(rpd_val(y_te, yp), 3),
                                            "y_pred": yp})

                    n_pca_safe = min(n_pca_xgb, X_tr.shape[1], X_tr.shape[0]-1)
                    ml_pipes = {}
                    if use_bayes:
                        ml_pipes["BayesianRidge"] = Pipeline([
                            ("sc", StandardScaler()), ("m", BayesianRidge())])
                    if use_rf:
                        ml_pipes["RandomForest"]  = Pipeline([
                            ("sc", StandardScaler()),
                            ("m", RandomForestRegressor(n_estimators=rf_n_trees,
                                                         max_features="sqrt",
                                                         random_state=42, n_jobs=-1))])
                    if use_xgb:
                        ml_pipes["XGBoost"] = Pipeline([
                            ("pca", PCA(n_components=n_pca_safe, random_state=42)),
                            ("m", XGBRegressor(n_estimators=xgb_n_trees,
                                                learning_rate=xgb_lr,
                                                max_depth=xgb_depth,
                                                subsample=0.8, random_state=42,
                                                verbosity=0))])

                    for mname, pipe in ml_pipes.items():
                        with st.spinner(f"Fitting {mname}..."):
                            try:
                                pipe.fit(X_tr, y_tr)
                                yp = pipe.predict(X_te).ravel()
                                results.append({"Model": mname,
                                                "R²": round(r2_score(y_te, yp), 4),
                                                "RPD": round(rpd_val(y_te, yp), 3),
                                                "y_pred": yp})
                            except Exception as e:
                                st.warning(f"{mname} failed: {e}")

                    # Benchmark box
                    bench = BENCHMARKS.get(sel_target)
                    if bench:
                        st.markdown(f"""<div class="kbox">
                            📖 <strong>Benchmark R² for {sel_target}:</strong> {bench}
                            (from your sidebar benchmark settings)
                        </div>""", unsafe_allow_html=True)

                    # Metric cards
                    st.markdown('<p class="slabel">Results</p>', unsafe_allow_html=True)
                    cols = st.columns(len(results))
                    for col, res in zip(cols, results):
                        gr, bc = rpd_badge(res["RPD"])
                        diff = round(res["R²"] - bench, 4) if bench else None
                        with col:
                            st.markdown(metric_card(
                                res["Model"], res["R²"],
                                f"RPD {res['RPD']}", gr, bc, diff
                            ), unsafe_allow_html=True)

                    # Predicted vs actual
                    st.markdown('<p class="slabel">Predicted vs actual</p>',
                                unsafe_allow_html=True)
                    nc = len(results)
                    fig_pva = make_subplots(rows=1, cols=nc,
                                            subplot_titles=[r["Model"] for r in results])
                    lims = [y_te.min() - y_te.std()*0.1, y_te.max() + y_te.std()*0.1]
                    for i, res in enumerate(results, 1):
                        fig_pva.add_trace(
                            go.Scatter(x=y_te, y=res["y_pred"], mode="markers",
                                       marker=dict(color="#40916C", size=6, opacity=0.7),
                                       showlegend=False), row=1, col=i)
                        fig_pva.add_trace(
                            go.Scatter(x=lims, y=lims, mode="lines",
                                       line=dict(color="#999", dash="dash", width=1),
                                       showlegend=False), row=1, col=i)
                    fig_pva.update_layout(height=320, template="plotly_white",
                                          margin=dict(l=30,r=10,t=40,b=30))
                    st.plotly_chart(fig_pva, use_container_width=True)

                    # PLSR component curve
                    if rmsecv_data:
                        best_n, scores = rmsecv_data
                        st.markdown('<p class="slabel">PLSR LOO RMSECV vs n_components</p>',
                                    unsafe_allow_html=True)
                        fig_c = px.line(x=list(range(1, len(scores)+1)), y=scores,
                                         markers=True,
                                         labels={"x":"n_components","y":"RMSECV"},
                                         color_discrete_sequence=["#2D6A4F"])
                        fig_c.add_vline(x=best_n, line_dash="dash",
                                         line_color="#F59E0B",
                                         annotation_text=f"Optimal: {best_n}")
                        fig_c.update_layout(height=260, template="plotly_white",
                                             margin=dict(l=40,r=10,t=20,b=40))
                        st.plotly_chart(fig_c, use_container_width=True)

                    # Export
                    export = pd.DataFrame([{k: v for k,v in r.items()
                                             if k != "y_pred"} for r in results])
                    st.download_button("⬇ Download results",
                                       export.to_csv(index=False),
                                       f"{sel_target}_{sel_region}_results.csv",
                                       "text/csv")

# ──────────────────────────────────────────────────────────────
# TAB 3 — VARIETY CLASSIFICATION (LDA)
# ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Variety classification — LDA")

    if df is None:
        st.info("Upload a spectral matrix in the sidebar to begin.")
    elif not available_id:
        st.warning("No non-spectral columns found for variety labels.")
    else:
        variety_col = st.selectbox("Variety / cultivar column", available_id)
        lda_bench   = st.number_input("Classification accuracy benchmark (%)",
                                       0.0, 100.0, 80.0, 1.0,
                                       help="e.g. 80 = Kelly's LOO accuracy benchmark") / 100

        prep_options = {
            "Raw"           : (False, 0),
            "SG only"       : (False, sg_deriv),
            "SNV only"      : (True,  0),
            "SNV + SG"      : (True,  sg_deriv),
        }
        sel_preps = st.multiselect("Preprocessing options to test",
                                    list(prep_options.keys()),
                                    default=["Raw", "SNV + SG"])

        run_lda = st.button("▶ Run LDA classification", type="primary")

        if run_lda:
            df_v = df.dropna(subset=[variety_col]).copy()
            le   = LabelEncoder()
            y_v  = le.fit_transform(df_v[variety_col])
            names= list(le.classes_)
            n_classes = len(names)

            if n_classes < 2:
                st.error("Need at least 2 variety classes to run LDA.")
            else:
                st.info(f"Classes detected: {', '.join(names)}")
                lda_records = []
                total = len(REGIONS) * len(sel_preps)
                prog  = st.progress(0)
                step  = 0

                for rname, (lo, hi) in REGIONS.items():
                    rc = filter_region(wave_cols, lo, hi)
                    if len(rc) < 3:
                        step += len(sel_preps); continue

                    X_r = df_v[rc].values.astype(float)
                    for pname in sel_preps:
                        snv_, sg_ = prep_options[pname]
                        Xp = apply_preprocessing(X_r, snv_, sg_, sg_window, sg_poly)
                        sc = StandardScaler()
                        Xs = sc.fit_transform(Xp)
                        try:
                            yp  = cross_val_predict(
                                LinearDiscriminantAnalysis(), Xs, y_v,
                                cv=LeaveOneOut())
                            acc = accuracy_score(y_v, yp)
                            per = {names[i]: accuracy_score(y_v[y_v==i], yp[y_v==i])
                                   for i in range(n_classes) if (y_v==i).sum() > 0}
                            lda_records.append({
                                "Region": rname, "Preprocessing": pname,
                                "Accuracy": round(acc, 4),
                                **{f"Acc_{v}": round(a, 3) for v,a in per.items()}
                            })
                        except Exception as e:
                            st.warning(f"{rname} {pname}: {e}")
                        step += 1
                        prog.progress(step / total)

                prog.empty()
                if not lda_records:
                    st.error("LDA failed on all configurations.")
                else:
                    res_df = pd.DataFrame(lda_records).sort_values(
                        "Accuracy", ascending=False)
                    best = res_df.iloc[0]
                    diff = best["Accuracy"] - lda_bench
                    col = "#065F46" if diff >= 0 else "#991B1B"
                    sym = "↑" if diff >= 0 else "↓"

                    st.markdown(f"""<div class="mcard" style="max-width:340px">
                        <div class="mlabel">Best configuration</div>
                        <div class="mvalue">{best['Accuracy']:.1%}</div>
                        <div class="msub">{best['Region']} · {best['Preprocessing']}</div>
                        <div class="msub" style="color:{col}">{sym} {diff:+.1%} vs benchmark ({lda_bench:.0%})</div>
                    </div>""", unsafe_allow_html=True)

                    # Heatmap
                    st.markdown('<p class="slabel">Accuracy — region × preprocessing</p>',
                                unsafe_allow_html=True)
                    pivot = res_df.pivot_table(index="Region", columns="Preprocessing",
                                               values="Accuracy", aggfunc="max")
                    fig_hm = px.imshow(pivot, text_auto=".0%",
                                        color_continuous_scale="RdYlGn",
                                        zmin=max(0, lda_bench-0.3), zmax=1.0)
                    fig_hm.add_hline(y=-0.5,  # visual separator at benchmark
                                      annotation_text=f"Benchmark: {lda_bench:.0%}",
                                      line_color="#F59E0B", line_dash="dash")
                    fig_hm.update_layout(height=max(300, len(REGIONS)*50),
                                          template="plotly_white",
                                          margin=dict(l=10,r=10,t=20,b=10))
                    st.plotly_chart(fig_hm, use_container_width=True)

                    st.markdown('<p class="slabel">Full results</p>',
                                unsafe_allow_html=True)
                    st.dataframe(res_df.reset_index(drop=True), use_container_width=True)
                    st.download_button("⬇ Download LDA results",
                                       res_df.to_csv(index=False),
                                       "lda_results.csv", "text/csv")

# ──────────────────────────────────────────────────────────────
# TAB 4 — FULL TOURNAMENT
# ──────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Full model tournament — all regions × all models")
    st.markdown("""<div class="wbox">
        ⏱ Runs all selected models across all defined spectral regions
        for each selected trait. Allow 2–8 minutes depending on dataset size.
        Adjust n_estimators sliders in the sidebar to speed things up.
    </div>""", unsafe_allow_html=True)

    if df is None:
        st.info("Upload a spectral matrix in the sidebar to begin.")
    elif not trait_cols:
        st.warning("No trait columns detected.")
    elif not REGIONS:
        st.warning("No spectral regions defined in the sidebar.")
    elif not any([use_plsr, use_bayes, use_rf, use_xgb]):
        st.warning("Select at least one model in the sidebar.")
    else:
        sel_targets = st.multiselect(
            "Traits to include",
            trait_cols,
            default=trait_cols[:min(3, len(trait_cols))]
        )

        run_tourn = st.button("▶ Run full tournament", type="primary")

        if run_tourn and sel_targets:
            all_records = []
            total = len(sel_targets) * len(REGIONS)
            prog  = st.progress(0)
            status= st.empty()
            step  = 0

            for target in sel_targets:
                df_t  = df.dropna(subset=[target])
                y_all = df_t[target].values.astype(float)
                bench = BENCHMARKS.get(target)

                for rname, (lo, hi) in REGIONS.items():
                    rc = filter_region(wave_cols, lo, hi)
                    status.text(f"Running: {target} × {rname}…")

                    if len(rc) < 3:
                        step += 1; prog.progress(step/total); continue

                    X_raw = df_t[rc].values.astype(float)
                    X     = apply_preprocessing(X_raw, apply_snv, sg_deriv,
                                                 sg_window, sg_poly)
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y_all, test_size=test_size, random_state=42)

                    def _record(mname, yp):
                        r2  = round(r2_score(y_te, yp), 4)
                        rpd = round(rpd_val(y_te, yp), 3)
                        gr, _ = rpd_badge(rpd)
                        return {
                            "Target": target, "Region": rname, "Model": mname,
                            "R²": r2, "RPD": rpd, "Grade": gr,
                            "vs_Benchmark": round(r2-bench,4) if bench else None,
                            "Benchmark": bench,
                        }

                    if use_plsr:
                        try:
                            best_n, _ = optimise_plsr(X_tr, y_tr, plsr_max_comp)
                            pls = PLSRegression(n_components=best_n)
                            pls.fit(X_tr, y_tr)
                            all_records.append(_record(f"PLSR({best_n})",
                                                       pls.predict(X_te).ravel()))
                        except Exception as e:
                            status.text(f"PLSR {rname}: {e}")

                    n_pca_safe = min(n_pca_xgb, X_tr.shape[1], X_tr.shape[0]-1)
                    ml = {}
                    if use_bayes:
                        ml["BayesianRidge"] = Pipeline([
                            ("sc", StandardScaler()), ("m", BayesianRidge())])
                    if use_rf:
                        ml["RandomForest"]  = Pipeline([
                            ("sc", StandardScaler()),
                            ("m", RandomForestRegressor(n_estimators=rf_n_trees,
                                                         max_features="sqrt",
                                                         random_state=42, n_jobs=-1))])
                    if use_xgb and n_pca_safe >= 1:
                        ml["XGBoost"] = Pipeline([
                            ("pca", PCA(n_components=n_pca_safe, random_state=42)),
                            ("m", XGBRegressor(n_estimators=xgb_n_trees,
                                                learning_rate=xgb_lr,
                                                max_depth=xgb_depth,
                                                subsample=0.8, random_state=42,
                                                verbosity=0))])

                    for mname, pipe in ml.items():
                        try:
                            pipe.fit(X_tr, y_tr)
                            all_records.append(_record(mname, pipe.predict(X_te).ravel()))
                        except Exception as e:
                            status.text(f"{mname} {rname}: {e}")

                    step += 1
                    prog.progress(step / total)

            prog.empty(); status.empty()

            if not all_records:
                st.error("No results generated. Check data and settings.")
            else:
                results_df = pd.DataFrame(all_records)

                # Best per target metric cards
                st.markdown('<p class="slabel">Best model per trait</p>',
                            unsafe_allow_html=True)
                best_df = (results_df.sort_values("R²", ascending=False)
                           .groupby("Target").first().reset_index())
                cols = st.columns(min(len(best_df), 5))
                for col, row in zip(cols, best_df.itertuples()):
                    gr, bc = rpd_badge(row.RPD)
                    bench  = row.Benchmark
                    diff   = row.vs_Benchmark
                    with col:
                        st.markdown(metric_card(
                            row.Target, row._5,  # R²
                            f"{row.Model} · {row.Region.split('(')[0].strip()[:12]}",
                            gr, bc,
                            diff
                        ), unsafe_allow_html=True)

                # R² bar charts per target
                st.markdown('<p class="slabel">R² by region and model</p>',
                            unsafe_allow_html=True)
                for target in sel_targets:
                    sub = results_df[results_df["Target"] == target]
                    fig = px.bar(sub, x="Region", y="R²", color="Model",
                                  barmode="group", title=target,
                                  color_discrete_sequence=px.colors.qualitative.Set2)
                    bench = BENCHMARKS.get(target)
                    if bench:
                        fig.add_hline(y=bench, line_dash="dash",
                                       line_color="#F59E0B",
                                       annotation_text=f"Benchmark {bench}",
                                       annotation_position="top right")
                    fig.update_layout(height=340, template="plotly_white",
                                       margin=dict(l=40,r=10,t=40,b=90),
                                       xaxis_tickangle=-30)
                    st.plotly_chart(fig, use_container_width=True)

                # Full table + download
                st.markdown('<p class="slabel">Full results table</p>',
                            unsafe_allow_html=True)
                st.dataframe(results_df, use_container_width=True)
                st.download_button(
                    "⬇ Download full results CSV",
                    results_df.to_csv(index=False),
                    "tournament_results.csv", "text/csv"
                )

# =============================================================
# FOOTER
# =============================================================
st.markdown("---")
st.markdown("""
<p style="font-size:0.78rem;color:#888;text-align:center">
WheatSpec v2 · Manuli Perera · UWA Dissertation 2026 · Open-source MIT ·
Built on Kelly et al. (2023) <em>Food Chemistry</em> methodology ·
<a href="https://github.com" style="color:#40916C">GitHub</a>
</p>""", unsafe_allow_html=True)
