import streamlit as st
import pandas as pd
import numpy as np
import os

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pediatric BMT Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0a0f1e;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1424;
    border-right: 1px solid #1e3a5f;
}

/* Title */
.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #38bdf8;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.sub-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 300;
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 4px;
    margin-bottom: 2rem;
}

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #38bdf8;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 24px;
}

/* Cards */
.result-card-success {
    background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.result-card-failure {
    background: linear-gradient(135deg, #1c0505 0%, #3b0606 100%);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 8px;
}
.result-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.5rem;
    font-weight: 600;
}
.result-success { color: #10b981; }
.result-failure { color: #ef4444; }

.prob-bar-container {
    background: #1e293b;
    border-radius: 999px;
    height: 8px;
    margin: 12px 0;
    overflow: hidden;
}
.prob-bar-fill-success {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #10b981, #34d399);
}
.prob-bar-fill-failure {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #ef4444, #f87171);
}

/* Input styling */
div[data-baseweb="input"] input,
div[data-baseweb="select"] {
    background-color: #0d1424 !important;
    border-color: #1e3a5f !important;
    color: #e2e8f0 !important;
}

/* Metric */
.metric-box {
    background: #0d1424;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-label {
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #38bdf8;
    font-family: 'IBM Plex Mono', monospace;
}

.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8);
    color: #0a0f1e;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 1px;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    width: 100%;
    font-size: 0.9rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #38bdf8, #7dd3fc);
    transform: translateY(-1px);
}

hr { border-color: #1e3a5f; }

.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<div class="main-title">🩺 Pediatric BMT Decision Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Système d\'aide à la décision pour la greffe de moelle osseuse pédiatrique</div>', unsafe_allow_html=True)

st.markdown("---")

# ─── Sidebar – Patient Info ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">Patient</div>', unsafe_allow_html=True)

    recipient_age = st.number_input("Âge du receveur (ans)", min_value=0.0, max_value=18.0, value=8.0, step=0.5)
    recipient_gender = st.selectbox("Sexe du receveur", ["Masculin", "Féminin"])
    recipient_body_mass = st.number_input("Masse corporelle (kg)", min_value=1.0, max_value=100.0, value=25.0, step=0.5)

    st.markdown('<div class="section-header">Donneur</div>', unsafe_allow_html=True)

    donor_age = st.number_input("Âge du donneur (ans)", min_value=0.0, max_value=80.0, value=35.0, step=0.5)
    donor_age_35 = st.selectbox("Donneur > 35 ans", ["Non", "Oui"])
    donor_gender = st.selectbox("Sexe du donneur", ["Masculin", "Féminin"])

    st.markdown('<div class="section-header">Compatibilité</div>', unsafe_allow_html=True)

    gender_match = st.selectbox("Correspondance de genre", ["Compatible", "Incompatible"])
    donor_abo = st.selectbox("Groupe sanguin donneur (ABO)", ["-1", "0", "1", "2"])
    recipient_abo = st.selectbox("Groupe sanguin receveur (ABO)", ["-1", "0", "1", "2"])
    abo_match = st.selectbox("Compatibilité ABO", ["Compatible", "Incompatible"])

# ─── Main – Clinical Data ──────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-header">Données Cliniques</div>', unsafe_allow_html=True)
    disease = st.selectbox("Maladie", ["ALL", "AML", "chronic", "nonmalignant", "lymphoma"])
    disease_group = st.selectbox("Groupe de maladie", ["malignant", "nonmalignant"])
    stem_cell_source = st.selectbox("Source des cellules souches", ["BM+PBSC", "PBSC", "BM"])
    hla_match = st.selectbox("Correspondance HLA", ["10/10", "9/10", "8/10", "7/10"])

with col2:
    st.markdown('<div class="section-header">Traitement</div>', unsafe_allow_html=True)
    cd34_day = st.number_input("CD34+ (×10⁶/kg) - Jour J", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
    cd3_day = st.number_input("CD3+ (×10⁵/kg) - Jour J", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
    risk_group = st.selectbox("Groupe de risque", ["low risk", "high risk"])
    iiiv = st.selectbox("IIIV", ["Non", "Oui"])

with col3:
    st.markdown('<div class="section-header">Complications</div>', unsafe_allow_html=True)
    cmv_status = st.selectbox("Statut CMV receveur", ["Négatif", "Positif"])
    cmv_donor = st.selectbox("Statut CMV donneur", ["Négatif", "Positif"])
    gvhd_prophylaxis = st.selectbox("Prophylaxie GvHD", ["CsA+MMF", "CsA+MTX", "FK506+MTX", "FK506+MMF"])
    antifungal = st.selectbox("Prophylaxie antifongique", ["Fluconazole", "Itraconazole", "Voriconazole"])

st.markdown("---")

# ─── Prediction Button ─────────────────────────────────────────────────────────
col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
with col_btn2:
    predict_btn = st.button("⚡ ANALYSER")

# ─── Result ───────────────────────────────────────────────────────────────────
if predict_btn:
    st.markdown("---")
    st.markdown('<div class="section-header">Résultat de la Prédiction</div>', unsafe_allow_html=True)

    # ── Simulated prediction (replace with real model later) ──
    # Risk score heuristic for demo
    risk_score = 0.0
    if donor_age_35 == "Oui": risk_score += 0.15
    if gender_match == "Incompatible": risk_score += 0.1
    if abo_match == "Incompatible": risk_score += 0.1
    if risk_group == "high risk": risk_score += 0.2
    if disease_group == "malignant": risk_score += 0.1
    if cmv_status == "Positif" and cmv_donor == "Négatif": risk_score += 0.1
    if hla_match in ["7/10", "8/10"]: risk_score += 0.15
    risk_score = min(risk_score, 0.95)
    success_prob = round(1 - risk_score, 2)
    failure_prob = round(risk_score, 2)
    prediction = "Succès" if success_prob >= 0.5 else "Échec"

    col_res1, col_res2, col_res3 = st.columns([1, 2, 1])

    with col_res2:
        if prediction == "Succès":
            st.markdown(f"""
            <div class="result-card-success">
                <div class="result-label">Prédiction</div>
                <div class="result-value result-success">✓ SUCCÈS</div>
                <div style="margin-top: 16px; color: #94a3b8; font-size: 0.85rem;">Probabilité de survie</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-success" style="width: {success_prob*100}%"></div>
                </div>
                <div style="font-family: 'IBM Plex Mono'; font-size: 1.4rem; color: #10b981;">{success_prob*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-failure">
                <div class="result-label">Prédiction</div>
                <div class="result-value result-failure">✗ ÉCHEC</div>
                <div style="margin-top: 16px; color: #94a3b8; font-size: 0.85rem;">Probabilité d'échec</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-failure" style="width: {failure_prob*100}%"></div>
                </div>
                <div style="font-family: 'IBM Plex Mono'; font-size: 1.4rem; color: #ef4444;">{failure_prob*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key metrics ──
    st.markdown('<div class="section-header">Facteurs de Risque Identifiés</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Compatibilité HLA</div>
            <div class="metric-value">{hla_match}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Groupe de risque</div>
            <div class="metric-value">{'⚠ HIGH' if risk_group == 'high risk' else '✓ LOW'}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">CMV Statut</div>
            <div class="metric-value">{'⚠ +' if cmv_status == 'Positif' else '✓ -'}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Score de risque</div>
            <div class="metric-value">{risk_score*100:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("ℹ️ Ce résultat est une aide à la décision. Le diagnostic final reste sous la responsabilité du médecin.", icon="ℹ️")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color: #334155; font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace;">
    Pediatric BMT Decision Support System · Projet IA Médicale · 2026
</div>
""", unsafe_allow_html=True)