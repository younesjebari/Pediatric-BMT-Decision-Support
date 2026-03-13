import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PediaBMT · Decision Support",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f7f8fc;
}

input, textarea, select,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
.stSelectbox div[data-baseweb="select"] *,
div[data-baseweb="input"] input,
div[data-baseweb="select"] div,
div[role="listbox"] li,
div[role="option"] {
    color: #111111 !important;
    background-color: #ffffff !important;
}

label, .stSelectbox label, .stNumberInput label,
[data-testid="stWidgetLabel"] p {
    color: #1a1a1a !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

[data-baseweb="menu"] li,
[data-baseweb="menu"] div,
[role="option"] {
    color: #111111 !important;
    background-color: #ffffff !important;
}

.header-banner {
    background: linear-gradient(135deg, #0d2b55 0%, #1a4a8a 60%, #1e6eb5 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.header-banner h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin: 0;
    letter-spacing: -0.5px;
    color: white !important;
}
.header-banner p {
    margin: 0.3rem 0 0;
    font-size: 0.95rem;
    opacity: 0.85;
    font-weight: 300;
    color: white !important;
}
.header-icon { font-size: 2.8rem; }

.card {
    background: white;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    box-shadow: 0 2px 12px rgba(13,43,85,0.07);
    margin-bottom: 1.2rem;
    border: 1px solid #e8ecf4;
}
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #0d2b55;
    margin-bottom: 1.2rem;
    border-bottom: 2px solid #e8ecf4;
    padding-bottom: 0.6rem;
}

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #0d2b55;
    margin: 1rem 0 0.6rem;
    border-left: 3px solid #1e6eb5;
    padding-left: 0.7rem;
}

.result-success {
    background: linear-gradient(135deg, #e6f9f0, #c8f0dd);
    border: 2px solid #2ecc71;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    color: #1a6b40;
}
.result-risk {
    background: linear-gradient(135deg, #fff0ee, #ffdad6);
    border: 2px solid #e74c3c;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    color: #8b1a1a;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    margin-bottom: 0.3rem;
}
.result-subtitle { font-size: 0.95rem; opacity: 0.8; }

.prob-container {
    background: rgba(255,255,255,0.6);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-top: 0.8rem;
}
.prob-bar-bg {
    background: #dce3f0;
    border-radius: 20px;
    height: 14px;
    overflow: hidden;
    margin-top: 0.4rem;
}

.shap-item {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #f0f3fa;
}
.shap-label { flex: 1; font-size: 0.88rem; color: #333; font-weight: 500; }
.shap-bar-bg {
    width: 140px;
    background: #eef1f8;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}
.shap-val { font-size: 0.8rem; min-width: 52px; text-align: right; }

.metric-chip {
    background: #eef2fb;
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    display: inline-block;
    margin: 0.2rem;
    font-size: 0.82rem;
    color: #0d2b55;
    font-weight: 500;
    border: 1px solid #d0d9ef;
}

.disclaimer {
    background: #fffbea;
    border-left: 4px solid #f39c12;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    font-size: 0.82rem;
    color: #7d5a00;
    margin-top: 1rem;
}

.stButton > button {
    background: linear-gradient(135deg, #1a4a8a, #1e6eb5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0d2b55, #1a4a8a) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(13,43,85,0.25) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTES & MAPPINGS
# ─────────────────────────────────────────────
# ⚠️ Noms exacts du dataset bone-marrow.arff
FEATURES = [
    'Recipientage',
    'Rbodymass',
    'CD34kgx10d6',
    'CD3dkgx10d8',
    'Disease',
    'Relapse',
    'Gendermatch',
    'HLAmatch',       # ← HLAmatch (majuscules HLA)
]

FEATURE_LABELS = {
    'Recipientage': 'Âge du receveur (ans)',
    'Rbodymass':    'Masse corporelle (kg)',
    'CD34kgx10d6':  'Dose CD34+ (×10⁶/kg)',
    'CD3dkgx10d8':  'Dose CD3+ (×10⁸/kg)',
    'Disease':      'Type de maladie',
    'Relapse':      'Antécédent de rechute',
    'Gendermatch':  'Compatibilité de genre',
    'HLAmatch':     'Compatibilité HLA',
}

DISEASE_MAP = {
    'ALL (Leucémie aiguë lymphoblastique)': 0,
    'AML (Leucémie aiguë myéloïde)':        1,
    'CML (Leucémie myéloïde chronique)':    2,
    'Autre':                                 3,
}
RELAPSE_MAP     = {'Non': 0, 'Oui': 1}
GENDERMATCH_MAP = {'Matched (compatible)': 0, 'Mismatched (incompatible)': 1}
HLAMATCH_MAP    = {
    'Matched (0 mismatch)':      0,
    '1 antigène incompatible':   1,
    '2 antigènes incompatibles': 2,
    '3 antigènes incompatibles': 3,
}

# ─────────────────────────────────────────────
#  CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    possible_paths = [
        'models/final_model.joblib',
        '../models/final_model.joblib',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'final_model.joblib'),
        os.path.join(os.getcwd(), 'models', 'final_model.joblib'),
    ]
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path  = path
            scaler_path = path.replace('final_model.joblib', 'scaler.joblib')
            break
    if model_path is None:
        return None, None
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler


# ─────────────────────────────────────────────
#  FONCTIONS
# ─────────────────────────────────────────────
def predict(model, scaler, input_df):
    X = input_df.copy()
    X_scaled = scaler.transform(X) if scaler else X.values
    pred  = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    return int(pred), float(proba[1])


def compute_shap_values(model, scaler, input_df):
    try:
        import shap
        X = input_df.copy()
        X_scaled = (
            pd.DataFrame(scaler.transform(X), columns=X.columns)
            if scaler else X
        )
        model_name  = type(model).__name__
        tree_models = ('RandomForestClassifier', 'XGBClassifier',
                       'LGBMClassifier', 'GradientBoostingClassifier')
        if model_name in tree_models:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        else:
            bg          = pd.DataFrame(np.zeros((1, len(FEATURES))), columns=FEATURES)
            explainer   = shap.KernelExplainer(model.predict_proba, bg)
            shap_values = explainer.shap_values(X_scaled, nsamples=100)
            sv          = shap_values[1][0]
        return dict(zip(FEATURES, sv))
    except Exception:
        return None


def shap_bar_html(feature, value, max_abs):
    label = FEATURE_LABELS.get(feature, feature)
    pct   = min(abs(value) / max_abs * 100, 100) if max_abs else 0
    color = "#1a6eb5" if value >= 0 else "#e74c3c"
    sign  = "+" if value >= 0 else "−"
    return (
        f'<div class="shap-item">'
        f'<span class="shap-label">{label}</span>'
        f'<div class="shap-bar-bg"><div style="width:{pct:.0f}%;background:{color};'
        f'height:10px;border-radius:8px;"></div></div>'
        f'<span class="shap-val" style="color:{color}">{sign}{abs(value):.4f}</span>'
        f'</div>'
    )


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <div class="header-icon">🩸</div>
    <div>
        <h1>PediaBMT · Decision Support</h1>
        <p>Système d'aide à la décision pour la prédiction du succès des greffes de moelle osseuse pédiatriques</p>
    </div>
</div>
""", unsafe_allow_html=True)

model, scaler = load_artifacts()
if model is None:
    st.error("⚠️ Modèle introuvable. Lancez d'abord : `python src/train_model.py`")
    st.stop()

model_name = type(model).__name__
st.markdown(
    f'<div style="text-align:right;margin-bottom:1.5rem;">'
    f'<span class="metric-chip">🤖 {model_name}</span>'
    f'<span class="metric-chip">✅ Modèle chargé</span>'
    f'</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
#  FORMULAIRE CENTRÉ
# ─────────────────────────────────────────────
_, col_form, _ = st.columns([1, 2, 1])

with col_form:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋 Données du patient</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Variables numériques</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        recipientage = st.number_input(
            "Âge du receveur (ans)", min_value=0.0, max_value=25.0, value=8.0, step=0.5)
        cd34 = st.number_input(
            "Dose CD34+ (×10⁶/kg)", min_value=0.0, max_value=50.0, value=5.5, step=0.1)
    with col_b:
        rbodymass = st.number_input(
            "Masse corporelle (kg)", min_value=5.0, max_value=150.0, value=28.0, step=0.5)
        cd3 = st.number_input(
            "Dose CD3+ (×10⁸/kg)", min_value=0.0, max_value=100.0, value=3.2, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Variables catégorielles</div>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2)
    with col_c:
        disease_str     = st.selectbox("Type de maladie",        list(DISEASE_MAP.keys()))
        gendermatch_str = st.selectbox("Compatibilité de genre", list(GENDERMATCH_MAP.keys()))
    with col_d:
        relapse_str  = st.selectbox("Antécédent de rechute", list(RELAPSE_MAP.keys()))
        hlamatch_str = st.selectbox("Compatibilité HLA",     list(HLAMATCH_MAP.keys()))

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔬 Analyser la prédiction")

    st.markdown(
        '<div class="disclaimer">⚠️ <strong>Avertissement clinique :</strong> '
        'Cet outil est un support décisionnel basé sur des données statistiques. '
        'Il ne remplace pas le jugement médical d\'un spécialiste.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  RÉSULTATS
# ─────────────────────────────────────────────
if predict_btn:
    input_df = pd.DataFrame([[
        recipientage,
        rbodymass,
        cd34,
        cd3,
        DISEASE_MAP[disease_str],
        RELAPSE_MAP[relapse_str],
        GENDERMATCH_MAP[gendermatch_str],
        HLAMATCH_MAP[hlamatch_str],
    ]], columns=FEATURES)

    with st.spinner("Analyse en cours..."):
        pred, prob_survival = predict(model, scaler, input_df)

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_res, _ = st.columns([1, 2, 1])

    with col_res:
        if pred == 1:
            st.markdown(f"""
            <div class="result-success">
                <div class="result-title">✅ Survie probable</div>
                <div class="result-subtitle">Le modèle prédit un succès de la greffe</div>
                <div class="prob-container">
                    <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                        <span>Probabilité de survie</span>
                        <strong>{prob_survival*100:.1f}%</strong>
                    </div>
                    <div class="prob-bar-bg">
                        <div style="width:{prob_survival*100:.1f}%;height:14px;border-radius:20px;
                             background:linear-gradient(90deg,#27ae60,#2ecc71);"></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-risk">
                <div class="result-title">⚠️ Risque élevé</div>
                <div class="result-subtitle">Le modèle prédit un risque d'échec de la greffe</div>
                <div class="prob-container">
                    <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                        <span>Probabilité de survie</span>
                        <strong>{prob_survival*100:.1f}%</strong>
                    </div>
                    <div class="prob-bar-bg">
                        <div style="width:{prob_survival*100:.1f}%;height:14px;border-radius:20px;
                             background:linear-gradient(90deg,#c0392b,#e74c3c);"></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🔍 Explication SHAP — Impact des variables</div>',
                    unsafe_allow_html=True)

        shap_vals = compute_shap_values(model, scaler, input_df)

        if shap_vals:
            max_abs     = max(abs(v) for v in shap_vals.values()) or 1e-9
            sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)

            st.markdown(
                "".join(shap_bar_html(f, v, max_abs) for f, v in sorted_shap),
                unsafe_allow_html=True
            )
            st.markdown("""
            <div style="margin-top:0.8rem;font-size:0.78rem;color:#666;">
                <span style="color:#1a6eb5;">■</span> Valeur positive = favorise la survie &nbsp;
                <span style="color:#e74c3c;">■</span> Valeur négative = défavorise la survie
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.8))
            labels = [FEATURE_LABELS.get(f, f) for f, _ in sorted_shap]
            values = [v for _, v in sorted_shap]
            colors = ["#1a6eb5" if v >= 0 else "#e74c3c" for v in values]
            ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.55, edgecolor='none')
            ax.axvline(0, color='#aaa', linewidth=0.8, linestyle='--')
            ax.set_xlabel("Valeur SHAP", fontsize=9, color='#555')
            ax.set_title("Contribution de chaque variable à la prédiction",
                         fontsize=10, color='#0d2b55', pad=8, fontweight='600')
            ax.tick_params(labelsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_facecolor('#f7f8fc')
            fig.patch.set_facecolor('#ffffff')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("📊 SHAP non disponible. Installez-le via `pip install shap`.")

        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.78rem;color:#aaa;padding:0.5rem;">
    PediaBMT Decision Support · Centrale Casablanca · Coding Week 2026 ·
    Modèle ML entraîné sur UCI Bone Marrow Transplant Dataset
</div>""", unsafe_allow_html=True)