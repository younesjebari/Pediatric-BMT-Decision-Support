import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

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

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

/* Supprimer la barre blanche en haut */
header[data-testid="stHeader"] { background: #0a0f1e !important; height: 0px !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding-top: 3rem !important; }

.stApp { background: #0a0f1e; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #0d1424; border-right: 1px solid #1e3a5f; }

.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem; font-weight: 600;
    color: #38bdf8; letter-spacing: -0.5px; margin-bottom: 0;
}
.sub-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 300; color: #64748b;
    font-size: 0.95rem; margin-top: 4px; margin-bottom: 2rem;
}
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    color: #38bdf8; border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px; margin-bottom: 16px; margin-top: 24px;
}
.result-card-success {
    background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border: 1px solid #10b981; border-radius: 12px; padding: 2rem; text-align: center;
}
.result-card-failure {
    background: linear-gradient(135deg, #1c0505 0%, #3b0606 100%);
    border: 1px solid #ef4444; border-radius: 12px; padding: 2rem; text-align: center;
}
.result-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
    letter-spacing: 3px; text-transform: uppercase; opacity: 0.7; margin-bottom: 8px;
}
.result-value { font-family: 'IBM Plex Mono', monospace; font-size: 2.5rem; font-weight: 600; }
.result-success { color: #10b981; }
.result-failure { color: #ef4444; }
.prob-bar-container { background: #1e293b; border-radius: 999px; height: 8px; margin: 12px 0; overflow: hidden; }
.prob-bar-fill-success { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #10b981, #34d399); }
.prob-bar-fill-failure { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #ef4444, #f87171); }
.metric-box { background: #0d1424; border: 1px solid #1e3a5f; border-radius: 8px; padding: 16px; text-align: center; }
.metric-label { font-size: 0.7rem; letter-spacing: 1.5px; text-transform: uppercase; color: #64748b; font-family: 'IBM Plex Mono', monospace; }
.metric-value { font-size: 1.6rem; font-weight: 600; color: #38bdf8; font-family: 'IBM Plex Mono', monospace; }
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8);
    color: #0a0f1e; font-family: 'IBM Plex Mono', monospace;
    font-weight: 600; letter-spacing: 1px; border: none;
    border-radius: 8px; padding: 0.6rem 2rem; width: 100%; font-size: 0.9rem;
}
hr { border-color: #1e3a5f; }
.stSelectbox label, .stNumberInput label, .stSlider label { color: #94a3b8 !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ─── Chargement du modèle ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    possible_paths = [
        'models/final_model.joblib',
        '../models/final_model.joblib',
        os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.joblib'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'final_model.joblib'),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

model = load_model()

# ─── Mapping ABO ───────────────────────────────────────────────────────────────
ABO_LABELS = {"0": "O", "1": "A", "-1": "B", "2": "AB"}

# ─── Fonction de construction du vecteur de features ──────────────────────────
def build_feature_vector(inputs: dict) -> pd.DataFrame:
    row = {
        'Recipientgender':  1 if inputs['recipient_gender'] == 'Masculin' else 0,
        'Recipientage':     inputs['recipient_age'],
        'Recipientage10':   1 if inputs['recipient_age'] >= 10 else 0,
        'Recipientageint':  0 if inputs['recipient_age'] <= 5 else (1 if inputs['recipient_age'] <= 10 else 2),
        'Rbodymass':        inputs['recipient_body_mass'],
        'RecipientCMV':     1 if inputs['cmv_status'] == 'Positif' else 0,
        'RecipientABO':     int(inputs['recipient_abo']),
        'RecipientRh':      1 if inputs['recipient_rh'] == 'Positif (+)' else 0,
        'Stemcellsource':   1 if inputs['stem_cell_source'] in ['PBSC', 'BM+PBSC'] else 0,
        'Donorage':         inputs['donor_age'],
        'Donorage35':       1 if inputs['donor_age'] >= 35 else 0,
        'DonorCMV':         1 if inputs['cmv_donor'] == 'Positif' else 0,
        'DonorABO':         int(inputs['donor_abo']),
        'Gendermatch':      1 if inputs['gender_match'] == 'Femme → Homme' else 0,
        'ABOmatch':         0 if inputs['abo_match'] == 'Compatible' else 1,
        'CMVstatus':        inputs['cmv_combined'],
        'HLAmatch':         inputs['hla_match_num'],
        'HLAmismatch':      0 if inputs['hla_match_num'] == 0 else 1,
        'Antigen':          inputs['antigen'],
        'Alel':             inputs['allele'],
        'HLAgrI':           inputs['hla_gri'],
        'Disease':          inputs['disease_num'],
        'Diseasegroup':     1 if inputs['disease_group'] == 'malignant' else 0,
        'Riskgroup':        1 if inputs['risk_group'] == 'Haut risque' else 0,
        'Txpostrelapse':    1 if inputs['txpostrelapse'] == 'Oui' else 0,
        'Relapse':          1 if inputs['relapse'] == 'Oui' else 0,
        'IIIV':             1 if inputs['iiiv'] == 'Oui' else 0,
        'aGvHDIIIIV':       1 if inputs['agvhd'] == 'Non' else 0,
        'extcGvHD':         1 if inputs['extcgvhd'] == 'Non' else 0,
        'CD34kgx10d6':      inputs['cd34_day'],
        'CD3dkgx10d8':      inputs['cd3_day'],
        'CD3dCD34':         inputs['cd3_day'] / inputs['cd34_day'] if inputs['cd34_day'] > 0 else 0,
        'ANCrecovery':      15.0,
        'PLTrecovery':      22.0,
        'time_to_aGvHD_III_IV': 1000000.0,
    }
    return pd.DataFrame([row])


# ─── Header ────────────────────────────────────────────────────────────────────
_, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<div class="main-title">🩺 Pediatric BMT Decision Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Système d\'aide à la décision pour la greffe de moelle osseuse pédiatrique · Random Forest + SHAP</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Modèle introuvable. Lancez d'abord : `python src/train_model.py`")

st.markdown("---")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">Patient (Receveur)</div>', unsafe_allow_html=True)
    recipient_age       = st.number_input("Âge du receveur (ans)", 0.0, 20.0, 8.0, 0.5)
    recipient_gender    = st.selectbox("Sexe du receveur", ["Masculin", "Féminin"])
    recipient_body_mass = st.number_input("Masse corporelle (kg)", 1.0, 150.0, 25.0, 0.5)
    recipient_rh        = st.selectbox("Facteur Rh receveur", ["Positif (+)", "Négatif (-)"])
    cmv_status          = st.selectbox("Statut CMV receveur", ["Négatif", "Positif"])
    recipient_abo       = st.selectbox(
        "Groupe sanguin receveur",
        options=["0", "1", "-1", "2"],
        format_func=lambda x: ABO_LABELS[x]
    )

    st.markdown('<div class="section-header">Donneur</div>', unsafe_allow_html=True)
    donor_age    = st.number_input("Âge du donneur (ans)", 0.0, 80.0, 35.0, 0.5)
    donor_gender = st.selectbox("Sexe du donneur", ["Masculin", "Féminin"])
    cmv_donor    = st.selectbox("Statut CMV donneur", ["Négatif", "Positif"])
    donor_abo    = st.selectbox(
        "Groupe sanguin donneur",
        options=["0", "1", "-1", "2"],
        format_func=lambda x: ABO_LABELS[x]
    )

    st.markdown('<div class="section-header">Compatibilité</div>', unsafe_allow_html=True)
    gender_match = st.selectbox("Type correspondance genre", ["Autre", "Femme → Homme"])
    abo_match    = st.selectbox("Compatibilité ABO", ["Compatible", "Incompatible"])

    _cmv_map     = {"Négatif": 0, "Positif": 1}
    cmv_combined = _cmv_map[cmv_donor] * 2 + _cmv_map[cmv_status]

    # Info groupes sanguins
    st.markdown(f"""
    <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:10px;margin-top:8px;font-size:0.78rem;color:#64748b;">
    🩸 Receveur : <b style="color:#38bdf8">{ABO_LABELS[recipient_abo]}</b> &nbsp;|&nbsp;
    Donneur : <b style="color:#38bdf8">{ABO_LABELS[donor_abo]}</b>
    </div>
    """, unsafe_allow_html=True)

# ─── Main ──────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-header">Données Cliniques</div>', unsafe_allow_html=True)
    disease_label = st.selectbox("Maladie", ["ALL", "AML", "chronic", "nonmalignant", "lymphoma"])
    disease_num   = {"ALL": 0, "AML": 1, "chronic": 2, "nonmalignant": 3, "lymphoma": 4}[disease_label]
    disease_group = st.selectbox("Groupe de maladie", ["malignant", "nonmalignant"])
    risk_group    = st.selectbox("Groupe de risque", ["Bas risque", "Haut risque"])
    txpostrelapse = st.selectbox("2ème greffe post-rechute ?", ["Non", "Oui"])
    relapse       = st.selectbox("Rechute de la maladie ?", ["Non", "Oui"])

with col2:
    st.markdown('<div class="section-header">Greffe & Traitement</div>', unsafe_allow_html=True)
    stem_cell_source = st.selectbox("Source cellules souches", ["BM+PBSC", "PBSC", "BM"])
    cd34_day         = st.number_input("CD34+ (×10⁶/kg)", 0.0, 60.0, 5.0, 0.1)
    cd3_day          = st.number_input("CD3+ (×10⁸/kg)", 0.0, 20.0, 1.0, 0.1)

    st.markdown('<div class="section-header">Compatibilité HLA</div>', unsafe_allow_html=True)
    hla_label     = st.selectbox("Correspondance HLA", ["10/10", "9/10", "8/10", "7/10"])
    hla_match_num = {"10/10": 0, "9/10": 1, "8/10": 2, "7/10": 3}[hla_label]
    antigen       = st.selectbox("Différences antigènes", [-1, 0, 1, 2],
                                 format_func=lambda x: {-1:"Aucune", 0:"1 diff", 1:"2 diff", 2:"3 diff"}[x])
    allele        = st.selectbox("Différences allèles", [-1, 0, 1, 2, 3],
                                 format_func=lambda x: {-1:"Aucune", 0:"1 diff", 1:"2 diff", 2:"3 diff", 3:"4 diff"}[x])
    hla_gri       = st.selectbox("HLA groupe (HLAgrI)", [0, 1, 2, 3, 4, 5, 7],
                                 format_func=lambda x: f"Groupe {x}")

with col3:
    st.markdown('<div class="section-header">Complications GvHD</div>', unsafe_allow_html=True)
    iiiv     = st.selectbox("GvHD aigu stade II-III-IV ?", ["Non", "Oui"])
    agvhd    = st.selectbox("GvHD aigu stade III-IV ?", ["Non", "Oui"])
    extcgvhd = st.selectbox("GvHD chronique étendue ?", ["Non", "Oui"])

st.markdown("---")

# ─── Bouton ────────────────────────────────────────────────────────────────────
_, col_btn, _ = st.columns([2, 1, 2])
with col_btn:
    predict_btn = st.button("⚡ ANALYSER")

# ─── Résultat ──────────────────────────────────────────────────────────────────
if predict_btn:
    if model is None:
        st.error("Modèle non chargé. Impossible de prédire.")
    else:
        inputs = dict(
            recipient_age=recipient_age, recipient_gender=recipient_gender,
            recipient_body_mass=recipient_body_mass, recipient_rh=recipient_rh,
            cmv_status=cmv_status, recipient_abo=recipient_abo,
            donor_age=donor_age, cmv_donor=cmv_donor, donor_abo=donor_abo,
            gender_match=gender_match, abo_match=abo_match, cmv_combined=cmv_combined,
            stem_cell_source=stem_cell_source, cd34_day=cd34_day, cd3_day=cd3_day,
            hla_match_num=hla_match_num, antigen=antigen, allele=allele, hla_gri=hla_gri,
            disease_num=disease_num, disease_group=disease_group,
            risk_group=risk_group, txpostrelapse=txpostrelapse, relapse=relapse,
            iiiv=iiiv, agvhd=agvhd, extcgvhd=extcgvhd,
        )

        X_input = build_feature_vector(inputs)
        try:
            X_input = X_input[model.feature_names_in_]
        except AttributeError:
            pass

        prediction   = model.predict(X_input)[0]
        proba        = model.predict_proba(X_input)[0]
        success_prob = round(float(proba[1]), 3)
        failure_prob = round(float(proba[0]), 3)

        st.markdown("---")
        st.markdown('<div class="section-header">Résultat de la Prédiction (Modèle ML · Random Forest)</div>', unsafe_allow_html=True)

        _, col_res, _ = st.columns([1, 2, 1])
        with col_res:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card-success">
                    <div class="result-label">Prédiction du modèle</div>
                    <div class="result-value result-success">✓ SURVIE</div>
                    <div style="margin-top:16px;color:#94a3b8;font-size:0.85rem;">Probabilité de survie estimée</div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fill-success" style="width:{success_prob*100:.0f}%"></div>
                    </div>
                    <div style="font-family:'IBM Plex Mono';font-size:1.4rem;color:#10b981;">{success_prob*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-failure">
                    <div class="result-label">Prédiction du modèle</div>
                    <div class="result-value result-failure">✗ DÉCÈS</div>
                    <div style="margin-top:16px;color:#94a3b8;font-size:0.85rem;">Probabilité de décès estimée</div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fill-failure" style="width:{failure_prob*100:.0f}%"></div>
                    </div>
                    <div style="font-family:'IBM Plex Mono';font-size:1.4rem;color:#ef4444;">{failure_prob*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Métriques clés ──
        st.markdown('<div class="section-header">Facteurs Clés</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">HLA Match</div><div class="metric-value">{hla_label}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Risque</div><div class="metric-value">{"⚠ HAUT" if risk_group=="Haut risque" else "✓ BAS"}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-box"><div class="metric-label">CMV Receveur</div><div class="metric-value">{"⚠ +" if cmv_status=="Positif" else "✓ -"}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Prob. Survie</div><div class="metric-value">{success_prob*100:.0f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SHAP ──
        st.markdown('<div class="section-header">Explication SHAP (Importances des Variables)</div>', unsafe_allow_html=True)
        try:
            import shap
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)

            # Correction : gérer toutes les formes possibles de shap_values
            if isinstance(shap_values, list):
                sv = np.array(shap_values[1]).flatten()
            elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
                sv = shap_values[0, :, 1]
            elif hasattr(shap_values, 'ndim') and shap_values.ndim == 2:
                sv = shap_values[0]
            else:
                sv = np.array(shap_values).flatten()

            feature_names = list(X_input.columns)
            shap_df = pd.DataFrame({
                'Feature':    feature_names,
                'SHAP Value': sv,
                'Abs':        np.abs(sv)
            }).sort_values('Abs', ascending=False).head(12)

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#0d1424')
            ax.set_facecolor('#0d1424')
            colors = ['#10b981' if v >= 0 else '#ef4444' for v in shap_df['SHAP Value']]
            ax.barh(shap_df['Feature'][::-1], shap_df['SHAP Value'][::-1], color=colors[::-1])
            ax.axvline(0, color='#64748b', linewidth=0.8)
            ax.set_xlabel('Valeur SHAP (impact sur la prédiction)', color='#94a3b8')
            ax.tick_params(colors='#94a3b8')
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e3a5f')
            st.pyplot(fig)
            plt.close()
            st.caption("🟢 Vert = favorise la survie · 🔴 Rouge = défavorise la survie")

        except Exception as e:
            st.warning(f"SHAP non disponible : {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("ℹ️ Ce résultat est une aide à la décision. Le diagnostic final reste sous la responsabilité du médecin.", icon="ℹ️")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.75rem;font-family:'IBM Plex Mono',monospace;">
    Pediatric BMT Decision Support System · Projet IA Médicale · École Centrale Casablanca · 2026
</div>
""", unsafe_allow_html=True)