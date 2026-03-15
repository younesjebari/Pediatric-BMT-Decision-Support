import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import hashlib
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
import io
from datetime import date, datetime, timedelta
import calendar
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PediaBMT · Decision Support",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
#  LOGOS SVG (interface web)
# ─────────────────────────────────────────────
def make_logo(size):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
<defs>
  <style>
    @keyframes shL{{0%,42%{{opacity:1}}50%,92%{{opacity:0}}100%{{opacity:1}}}}
    @keyframes sbL{{0%,42%{{opacity:0}}50%,92%{{opacity:1}}100%{{opacity:0}}}}
    @keyframes btL{{0%,100%{{transform:scale(1)}}20%{{transform:scale(1.13)}}40%{{transform:scale(1.06)}}}}
    .hgL{{transform-origin:100px 100px;animation:shL 3.2s ease-in-out infinite,btL 3.2s ease-in-out infinite}}
    .bgL{{opacity:0;animation:sbL 3.2s ease-in-out infinite}}
  </style>
  <linearGradient id="cL{size}" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#0d2b55"/><stop offset="100%" stop-color="#1a6eb5"/>
  </linearGradient>
  <linearGradient id="crL{size}" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#ffffff"/><stop offset="100%" stop-color="#c8e8ff" stop-opacity=".9"/>
  </linearGradient>
</defs>
<circle cx="100" cy="100" r="78" fill="url(#cL{size})"/>
<circle cx="100" cy="100" r="78" fill="none" stroke="#3a7fd4" stroke-width="2"/>
<circle cx="100" cy="100" r="72" fill="none" stroke="#ffffff" stroke-width="0.5" stroke-opacity="0.15"/>
<rect x="84" y="58" width="32" height="84" rx="6" fill="url(#crL{size})"/>
<rect x="58" y="84" width="84" height="32" rx="6" fill="url(#crL{size})"/>
<g class="hgL">
  <path d="M100,118 C100,118 82,107 80,95 C78,83 85,76 93,77 C96.5,77.5 99,81 100,84 C101,81 103.5,77.5 107,77 C115,76 122,83 120,95 C118,107 100,118 100,118Z" fill="#e74c3c" stroke="#c0392b" stroke-width="1.2"/>
  <ellipse cx="91" cy="84" rx="4.5" ry="3" fill="white" fill-opacity=".28" transform="rotate(-25,91,84)"/>
</g>
<g class="bgL" transform="rotate(-38,100,100)">
  <rect x="93" y="68" width="14" height="64" rx="5" fill="#d4c9b0" stroke="#b8ad96" stroke-width="1"/>
  <circle cx="96" cy="70" r="7" fill="#d4c9b0" stroke="#b8ad96" stroke-width="1"/>
  <circle cx="104" cy="70" r="7" fill="#d4c9b0" stroke="#b8ad96" stroke-width="1"/>
  <circle cx="96" cy="130" r="7" fill="#d4c9b0" stroke="#b8ad96" stroke-width="1"/>
  <circle cx="104" cy="130" r="7" fill="#d4c9b0" stroke="#b8ad96" stroke-width="1"/>
  <rect x="96" y="80" width="8" height="40" rx="3" fill="#0d2b55" fill-opacity=".08"/>
</g>
</svg>"""

LOGO_SMALL = make_logo(68)

LOGO_FULL = """<svg width="100%" viewBox="0 0 520 160" xmlns="http://www.w3.org/2000/svg">
<defs>
  <style>
    @keyframes shF{{0%,42%{{opacity:1}}50%,92%{{opacity:0}}100%{{opacity:1}}}}
    @keyframes sbF{{0%,42%{{opacity:0}}50%,92%{{opacity:1}}100%{{opacity:0}}}}
    @keyframes btF{{0%,100%{{transform:scale(1)}}20%{{transform:scale(1.13)}}40%{{transform:scale(1.06)}}}}
    .hgF{{transform-origin:80px 80px;animation:shF 3.2s ease-in-out infinite,btF 3.2s ease-in-out infinite}}
    .bgF{{opacity:0;animation:sbF 3.2s ease-in-out infinite}}
  </style>
  <linearGradient id="cF" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#0d2b55"/><stop offset="100%" stop-color="#1a6eb5"/>
  </linearGradient>
  <linearGradient id="crF" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#ffffff"/><stop offset="100%" stop-color="#c8e8ff" stop-opacity=".9"/>
  </linearGradient>
</defs>
<circle cx="80" cy="80" r="62" fill="url(#cF)"/>
<circle cx="80" cy="80" r="62" fill="none" stroke="#3a7fd4" stroke-width="1.8"/>
<circle cx="80" cy="80" r="57" fill="none" stroke="#ffffff" stroke-width="0.4" stroke-opacity="0.15"/>
<rect x="67" y="44" width="26" height="72" rx="5" fill="url(#crF)"/>
<rect x="44" y="67" width="72" height="26" rx="5" fill="url(#crF)"/>
<g class="hgF">
  <path d="M80,98 C80,98 64,88 62,77 C60,66 67,60 74,61 C77,61.5 79,65 80,67.5 C81,65 83,61.5 86,61 C93,60 100,66 98,77 C96,88 80,98 80,98Z" fill="#e74c3c" stroke="#c0392b" stroke-width="1"/>
  <ellipse cx="72" cy="67" rx="3.5" ry="2.2" fill="white" fill-opacity=".28" transform="rotate(-25,72,67)"/>
</g>
<g class="bgF" transform="rotate(-38,80,80)">
  <rect x="74" y="53" width="12" height="54" rx="4" fill="#d4c9b0" stroke="#b8ad96" stroke-width="0.9"/>
  <circle cx="77" cy="55" r="6" fill="#d4c9b0" stroke="#b8ad96" stroke-width="0.9"/>
  <circle cx="83" cy="55" r="6" fill="#d4c9b0" stroke="#b8ad96" stroke-width="0.9"/>
  <circle cx="77" cy="107" r="6" fill="#d4c9b0" stroke="#b8ad96" stroke-width="0.9"/>
  <circle cx="83" cy="107" r="6" fill="#d4c9b0" stroke="#b8ad96" stroke-width="0.9"/>
</g>
<text x="158" y="58" font-family="Georgia,'Times New Roman',serif" font-size="40" font-weight="700" fill="#0d2b55" letter-spacing="-0.5">Pedia<tspan fill="#1a6eb5">BMT</tspan></text>
<rect x="158" y="72" width="340" height="2" rx="1" fill="#1a6eb5"/>
<text x="160" y="96" font-family="Arial,sans-serif" font-size="13" fill="#2a5a9a" letter-spacing="2.5">DECISION SUPPORT</text>
<text x="160" y="118" font-family="Arial,sans-serif" font-size="11" fill="#6a8aaa" letter-spacing=".4">Pediatric Bone Marrow Transplant · Explainable AI</text>
<circle cx="160" cy="133" r="2" fill="#1a6eb5" fill-opacity=".5"/>
<circle cx="168" cy="133" r="2" fill="#1a6eb5" fill-opacity=".3"/>
<circle cx="176" cy="133" r="2" fill="#1a6eb5" fill-opacity=".15"/>
</svg>"""

# ─────────────────────────────────────────────
#  USERS
# ─────────────────────────────────────────────
USERS_FILE = "users.json"
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE,"r") as f: return json.load(f)
    return {}
def save_users(u):
    with open(USERS_FILE,"w") as f: json.dump(u,f,indent=2)
def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for k,v in [('logged_in',False),('current_user',None),('page','main'),
             ('last_result',None),('appointments',[]),
             ('cal_month',date.today().replace(day=1)),
             ('show_all_appts',False),('show_add_form',False)]:
    if k not in st.session_state: st.session_state[k]=v

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:#f7f8fc;}
input,textarea,select,[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,
[data-testid="stPasswordInput"] input,.stSelectbox div[data-baseweb="select"] *,
div[data-baseweb="input"] input,div[data-baseweb="select"] div{
    color:#111!important;background-color:#fff!important;font-size:.97rem!important;}
label,[data-testid="stWidgetLabel"] p{color:#1a1a1a!important;font-weight:600!important;font-size:.95rem!important;}
[data-baseweb="menu"] li,[data-baseweb="menu"] div,[role="option"]{color:#111!important;background-color:#fff!important;}
.header-banner{background:linear-gradient(135deg,#0d2b55 0%,#1a4a8a 60%,#1e6eb5 100%);border-radius:16px;padding:1.2rem 2rem;margin-bottom:1.5rem;display:flex;align-items:center;justify-content:space-between;}
.header-left{display:flex;align-items:center;gap:1rem;}
.header-banner h1{font-family:'DM Serif Display',serif;font-size:1.75rem;margin:0;color:white!important;}
.header-banner p{margin:.15rem 0 0;font-size:.86rem;opacity:.85;color:white!important;}
.header-user{text-align:right;font-size:.8rem;color:rgba(255,255,255,.82);}
.header-user strong{display:block;font-size:.96rem;color:white;}
.card{background:white;border-radius:14px;padding:1.5rem 1.7rem;box-shadow:0 2px 14px rgba(13,43,85,.07);margin-bottom:1.2rem;border:1px solid #e8ecf4;}
.card-title{font-family:'DM Serif Display',serif;font-size:1.1rem;color:#0d2b55;margin-bottom:1rem;border-bottom:2px solid #e8ecf4;padding-bottom:.5rem;}
.section-title{font-family:'DM Serif Display',serif;font-size:.94rem;color:#0d2b55;margin:.85rem 0 .5rem;border-left:3px solid #1e6eb5;padding-left:.6rem;}
.res-header{background:linear-gradient(135deg,#0d2b55,#1e6eb5);border-radius:16px;padding:2rem 2.5rem;margin-bottom:2rem;color:white;}
.res-header h2{font-family:'DM Serif Display',serif;font-size:1.6rem;margin:0 0 .5rem;color:white!important;}
.res-meta{font-size:.82rem;opacity:.82;display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:.8rem;}
.res-meta span{background:rgba(255,255,255,.12);border-radius:6px;padding:.25rem .7rem;}
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:.9rem;margin-bottom:1.1rem;}
.info-box{background:#f5f8ff;border-radius:10px;padding:.9rem 1.1rem;border-left:4px solid #1e6eb5;}
.info-box-label{font-size:.72rem;color:#5a7aaa;font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:.25rem;}
.info-box-value{font-size:.96rem;color:#1a1a1a;font-weight:600;}
.verdict-success{background:linear-gradient(135deg,#e6f9f0,#c8f0dd);border:2px solid #2ecc71;border-radius:16px;padding:2rem;text-align:center;color:#1a6b40;margin-bottom:1.5rem;}
.verdict-risk{background:linear-gradient(135deg,#fff0ee,#ffdad6);border:2px solid #e74c3c;border-radius:16px;padding:2rem;text-align:center;color:#8b1a1a;margin-bottom:1.5rem;}
.verdict-score{font-family:'DM Serif Display',serif;font-size:3.5rem;font-weight:700;line-height:1;}
.verdict-label{font-size:1.1rem;margin-top:.5rem;font-weight:600;}
.verdict-text{font-size:.86rem;opacity:.8;margin-top:.4rem;}
.prob-bar-bg{background:#dce3f0;border-radius:20px;height:14px;overflow:hidden;margin-top:.8rem;}
.disclaimer-box{background:#fffbea;border:1px solid #f0c040;border-radius:10px;padding:1.1rem 1.3rem;margin-top:1.4rem;}
.disclaimer-title{font-weight:700;color:#7d5a00;font-size:.86rem;margin-bottom:.35rem;}
.disclaimer-text{font-size:.78rem;color:#7d5a00;line-height:1.6;}
.signature-line{border-top:1px dashed #c0a060;margin-top:1.3rem;padding-top:.7rem;font-size:.78rem;color:#aaa;text-align:center;}
.shap-item{display:flex;align-items:center;gap:.7rem;padding:.42rem 0;border-bottom:1px solid #f0f3fa;}
.shap-label{flex:1;font-size:.83rem;color:#333;font-weight:500;}
.shap-bar-bg-s{width:120px;background:#eef1f8;border-radius:8px;height:9px;overflow:hidden;}
.shap-val{font-size:.77rem;min-width:50px;text-align:right;}
.mini-cal{width:100%;border-collapse:collapse;}
.mini-cal th{text-align:center;color:#1a4a8a;font-weight:700;padding:3px 1px;font-size:.7rem;}
.mini-cal td{text-align:center;padding:2px 1px;width:14.2%;}
.cal-cell{display:inline-block;width:26px;height:26px;line-height:26px;border-radius:6px;text-align:center;font-size:.77rem;color:#333;}
.cal-today{background:#1e6eb5!important;color:white!important;font-weight:700;}
.cal-has-appt{border:2px solid #2ecc71;color:#1a6b40;font-weight:700;}
.appt-widget{background:white;border-radius:12px;border:1px solid #e8ecf4;box-shadow:0 2px 10px rgba(13,43,85,.07);overflow:hidden;}
.appt-widget-header{background:linear-gradient(135deg,#0d2b55,#1a4a8a);padding:.55rem .9rem;display:flex;justify-content:space-between;align-items:center;}
.appt-widget-title{color:white;font-weight:700;font-size:.85rem;}
.appt-mini-item{padding:.46rem .85rem;border-bottom:1px solid #f0f3fa;display:flex;align-items:center;gap:.55rem;}
.appt-mini-date{background:#eef2fb;border-radius:6px;padding:2px 5px;font-size:.68rem;font-weight:700;color:#1a4a8a;min-width:40px;text-align:center;}
.appt-mini-name{font-weight:600;font-size:.79rem;color:#1a1a1a;}
.appt-mini-motif{color:#888;font-size:.71rem;}
.appt-mini-empty{padding:.85rem;text-align:center;color:#bbb;font-size:.8rem;}
.appt-full-item{background:#f5f8ff;border-left:4px solid #1e6eb5;border-radius:8px;padding:.6rem .85rem;margin-bottom:.42rem;}
.appt-full-date{font-weight:700;color:#1e6eb5;font-size:.82rem;}
.appt-full-name{font-weight:600;font-size:.85rem;color:#1a1a1a;}
.appt-full-motif{color:#666;font-size:.75rem;}
.metric-chip{background:#eef2fb;border-radius:8px;padding:.32rem .75rem;display:inline-block;margin:.18rem;font-size:.8rem;color:#0d2b55;font-weight:500;border:1px solid #d0d9ef;}
.disclaimer{background:#fffbea;border-left:4px solid #f39c12;border-radius:6px;padding:.7rem .95rem;font-size:.78rem;color:#7d5a00;margin-top:.85rem;}
.stButton>button{background:linear-gradient(135deg,#1a4a8a,#1e6eb5)!important;color:white!important;border:none!important;border-radius:10px!important;padding:.58rem 1.3rem!important;font-size:.93rem!important;font-weight:600!important;width:100%!important;}
.stButton>button:hover{background:linear-gradient(135deg,#0d2b55,#1a4a8a)!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTES ML
# ─────────────────────────────────────────────
# ── Ordre EXACT de IMPORTANT_FEATURES dans data_processing.py / train_model.py ──
FEATURES = [
    'CD3dkgx10d8',   # index 0
    'Rbodymass',     # index 1
    'Recipientage',  # index 2
    'CD34kgx10d6',   # index 3
    'Disease',       # index 4
    'Relapse',       # index 5
    'Gendermatch',   # index 6
    'HLAmatch',      # index 7
]

# Labels ASCII pour matplotlib/PDF (sans accents)
FEATURE_LABELS_PDF = {
    'CD3dkgx10d8' : 'Dose CD3+',
    'Rbodymass'   : 'Masse corporelle (kg)',
    'Recipientage': 'Age du receveur (ans)',
    'CD34kgx10d6' : 'Dose CD34+',
    'Disease'     : 'Type de maladie',
    'Relapse'     : 'Antecedent de rechute',
    'Gendermatch' : 'Compatibilite de genre',
    'HLAmatch'    : 'Compatibilite HLA',
}

# Labels avec accents pour l'interface web
FEATURE_LABELS_UI = {
    'CD3dkgx10d8' : 'Dose CD3+',
    'Rbodymass'   : 'Masse corporelle (kg)',
    'Recipientage': 'Âge du receveur (ans)',
    'CD34kgx10d6' : 'Dose CD34+',
    'Disease'     : 'Type de maladie',
    'Relapse'     : 'Antécédent de rechute',
    'Gendermatch' : 'Compatibilité de genre',
    'HLAmatch'    : 'Compatibilité HLA',
}

# ── Mappings catégoriels alignés avec encode_categories() de data_processing.py ──
# encode_categories fait : df[col].astype('category').cat.codes
# Les codes sont assignés par ordre ALPHABÉTIQUE des valeurs dans le dataset.
#
# Disease : valeurs ARFF = 'ALL','AML','chronic','nonmalignant','lymphoma'
#   → codes alphabétiques : ALL=0, AML=1, chronic=2, lymphoma=3, nonmalignant=4
# Relapse : '0','1' → 0=non, 1=oui (déjà numérique dans l'ARFF)
# Gendermatch : '0','1' → 0, 1
# HLAmatch : '0','1','2','3' → 0,1,2,3

DISEASE_MAP = {
    'ALL (Leucémie aiguë lymphoblastique)' : 0,
    'AML (Leucémie aiguë myéloïde)'        : 1,
    'CML / Chronique'                       : 2,
    'Lymphome'                              : 3,
    'Non-malin'                             : 4,
}
RELAPSE_MAP     = {'Non': 0, 'Oui': 1}
GENDERMATCH_MAP = {'Matched (compatible)': 0, 'Mismatched (incompatible)': 1}
HLAMATCH_MAP    = {
    'Matched (0 mismatch)'      : 0,
    '1 antigène incompatible'   : 1,
    '2 antigènes incompatibles' : 2,
    '3 antigènes incompatibles' : 3,
}
HEURES=[f"{h:02d}:{m:02d}" for h in range(8,19) for m in (0,30)]
MOTIFS=["Consultation initiale","Suivi post-greffe","Résultats d'analyses","Bilan pré-greffe","Urgence médicale","Autre"]
JOURS_FR=["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
MOIS_FR=["","Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]

# ─────────────────────────────────────────────
#  MODÈLE
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """
    Charge model, scaler et features_list sauvegardés par train_model.py.
    Cherche dans plusieurs emplacements possibles.
    """
    search_roots = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
    ]
    for root in search_roots:
        model_path = os.path.join(root, 'models', 'final_model.joblib')
        if os.path.exists(model_path):
            scaler_path   = os.path.join(root, 'models', 'scaler.joblib')
            features_path = os.path.join(root, 'models', 'features_list.joblib')
            model   = joblib.load(model_path)
            scaler  = joblib.load(scaler_path)  if os.path.exists(scaler_path)   else None
            # features_list.joblib contient IMPORTANT_FEATURES dans le bon ordre
            feat_list = joblib.load(features_path) if os.path.exists(features_path) else FEATURES
            return model, scaler, feat_list
    return None, None, FEATURES

def predict_fn(model, scaler, input_df):
    """Prédit avec le scaler et le modèle chargés depuis train_model.py."""
    Xs = scaler.transform(input_df) if scaler else input_df.values
    return int(model.predict(Xs)[0]), float(model.predict_proba(Xs)[0][1])

def compute_shap(model, scaler, input_df):
    """Calcule les valeurs SHAP. input_df doit avoir les colonnes dans l'ordre feat_list."""
    try:
        import shap
    except ImportError:
        st.warning("📊 SHAP non disponible. `pip install shap`")
        return None

    try:
        Xs   = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns) if scaler else input_df.copy()
        cols = list(input_df.columns)
        mname = type(model).__name__

        if mname in ('RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier'):
            exp  = shap.TreeExplainer(model)
            sv   = exp.shap_values(Xs)
            # RandomForest → liste [class0, class1] ; XGB/LGBM → tableau direct
            if isinstance(sv, list) and len(sv) == 2:
                vals = sv[1][0]          # probabilité de survie (classe 1)
            elif isinstance(sv, np.ndarray) and sv.ndim == 2:
                vals = sv[0]             # XGB/LGBM : tableau (n_samples, n_features)
            else:
                vals = sv[0] if isinstance(sv, list) else sv
            return dict(zip(cols, vals))

        else:  # SVM / autres
            bg  = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
            exp = shap.KernelExplainer(model.predict_proba, bg)
            sv  = exp.shap_values(Xs, nsamples=100)
            vals = sv[1][0] if isinstance(sv, list) else sv[0]
            return dict(zip(cols, vals))

    except Exception as e:
        st.warning(f"⚠️ Erreur SHAP : {e}")
        return None

# ─────────────────────────────────────────────
#  GRAPHIQUES SHAP
# ─────────────────────────────────────────────
def shap_bar_html(feature,value,max_abs):
    """Barres SHAP HTML pour l'interface web."""
    label=FEATURE_LABELS_UI.get(feature,feature)
    pct=min(abs(value)/max_abs*100,100) if max_abs else 0
    color="#1a6eb5" if value>=0 else "#e74c3c"; sign="+" if value>=0 else "−"
    return (f'<div class="shap-item"><span class="shap-label">{label}</span>'
            f'<div class="shap-bar-bg-s"><div style="width:{pct:.0f}%;background:{color};height:9px;border-radius:8px;"></div></div>'
            f'<span class="shap-val" style="color:{color}">{sign}{abs(value):.4f}</span></div>')

def make_shap_bars_figure(shap_vals):
    """
    Diagramme 1 : barres horizontales avec fond gris + barre colorée + valeur à droite.
    Style identique à l'image fournie.
    """
    ss   = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    lbls = [FEATURE_LABELS_PDF.get(f,f) for f,_ in ss]
    vals = [v for _,v in ss]
    clrs = ["#1a6eb5" if v>=0 else "#e74c3c" for v in vals]
    n    = len(ss)
    mx   = max(abs(v) for v in vals) if vals else 1

    fig, ax = plt.subplots(figsize=(7, n*0.62+0.4))
    for i,(lbl,val,clr) in enumerate(zip(lbls[::-1], vals[::-1], clrs[::-1])):
        ax.barh(i, mx*1.1, left=-mx*0.05, color='#eef1f8', height=0.5,
                edgecolor='none', zorder=1)
        ax.barh(i, val, color=clr, height=0.5, edgecolor='none', zorder=2)
        sign = "+" if val >= 0 else ""
        ax.text(mx*1.13, i, f"{sign}{val:.4f}",
                va='center', ha='left', fontsize=8.5, color=clr, fontweight='bold')

    ax.set_yticks(range(n))
    ax.set_yticklabels(lbls[::-1], fontsize=8.5)
    ax.axvline(0, color='#888', linewidth=0.8, linestyle='--', zorder=3)
    ax.set_xlim(-mx*0.15, mx*1.6)
    ax.set_xlabel("Valeur SHAP", fontsize=8, color='#555')
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=7.5)
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')
    plt.tight_layout(pad=0.5)
    return fig

def make_shap_classic_figure(shap_vals):
    """
    Diagramme 2 : graphique SHAP classique (barres horizontales sur fond gris clair).
    Style identique à la 3ème image fournie.
    """
    ss   = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    lbls = [FEATURE_LABELS_PDF.get(f,f) for f,_ in ss]
    vals = [v for _,v in ss]
    clrs = ["#1a6eb5" if v>=0 else "#e74c3c" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(lbls[::-1], vals[::-1], color=clrs[::-1], height=0.55, edgecolor='none')
    ax.axvline(0, color='#888', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Valeur SHAP", fontsize=9, color='#555')
    ax.set_title("Contribution de chaque variable a la prediction",
                 fontsize=10, color='#0d2b55', pad=8)
    ax.tick_params(labelsize=8)
    ax.spines[['top','right']].set_visible(False)
    ax.set_facecolor('#f7f8fc')
    fig.patch.set_facecolor('#ffffff')
    plt.tight_layout()
    return fig

def make_shap_figure_web(shap_vals):
    """Graphique SHAP pour l'interface web (avec accents)."""
    ss   = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    lbls = [FEATURE_LABELS_UI.get(f,f) for f,_ in ss]
    vals = [v for _,v in ss]
    clrs = ["#1a6eb5" if v>=0 else "#e74c3c" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(lbls[::-1], vals[::-1], color=clrs[::-1], height=0.55, edgecolor='none')
    ax.axvline(0, color='#aaa', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Valeur SHAP", fontsize=9, color='#555')
    ax.set_title("Contribution de chaque variable à la prédiction",
                 fontsize=10, color='#0d2b55', pad=8)
    ax.tick_params(labelsize=8)
    ax.spines[['top','right']].set_visible(False)
    ax.set_facecolor('#f7f8fc'); fig.patch.set_facecolor('#ffffff')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
#  EN-TÊTE PDF (matplotlib, propre)
# ─────────────────────────────────────────────
def make_logo_png_buf():
    """
    Logo PediaBMT identique au SVG de la page login.
    """
    fig = plt.figure(figsize=(1.6, 1.6))
    ax  = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    # ── Dégradé diagonal bleu marine → bleu (couches concentriques) ──
    n = 100
    for i in range(n, 0, -1):
        t   = 1 - (i / n)          # 0 = bord (foncé) → 1 = centre (clair)
        t   = t * 0.7              # réduire l'amplitude
        r_c = (13  + t*(26  - 13 )) / 255
        g_c = (43  + t*(78  - 43 )) / 255
        b_c = (85  + t*(181 - 85 )) / 255
        ax.add_patch(plt.Circle((0.5, 0.5), i/n * 0.485,
                                 facecolor=(r_c, g_c, b_c),
                                 edgecolor='none', zorder=1))

    # Anneau contour bleu clair (identique au SVG stroke="#3a7fd4")
    ax.add_patch(plt.Circle((0.5, 0.5), 0.485,
                             facecolor='none',
                             edgecolor='#3a7fd4',
                             linewidth=3.0, zorder=2))

    # Anneau intérieur blanc semi-transparent
    ax.add_patch(plt.Circle((0.5, 0.5), 0.445,
                             facecolor='none',
                             edgecolor='white',
                             linewidth=0.8, alpha=0.15, zorder=2))

    # ── Croix médicale blanche (proportions SVG : rx=6, w=32, h=84) ──
    # Barre verticale  (x=84..116, y=58..142 sur 200px → 0.42..0.58, 0.29..0.71)
    ax.add_patch(patches.FancyBboxPatch(
        (0.422, 0.185), 0.156, 0.630,
        boxstyle="round,pad=0.025",
        facecolor='white', edgecolor='none', zorder=3))
    # Barre horizontale (x=58..142, y=84..116 → 0.29..0.71, 0.42..0.58)
    ax.add_patch(patches.FancyBboxPatch(
        (0.185, 0.422), 0.630, 0.156,
        boxstyle="round,pad=0.025",
        facecolor='white', edgecolor='none', zorder=3))

    # Cœur rouge : 2 demi-cercles + triangle (identique au SVG)
    hx, hy, hs = 0.500, 0.565, 0.115
    ax.add_patch(plt.Circle((hx - hs*0.54, hy + hs*0.20), hs*0.58,
                             facecolor='#e74c3c', edgecolor='none', zorder=4))
    ax.add_patch(plt.Circle((hx + hs*0.54, hy + hs*0.20), hs*0.58,
                             facecolor='#e74c3c', edgecolor='none', zorder=4))
    ax.add_patch(plt.Polygon(
        [[hx - hs*1.10, hy + hs*0.20],
         [hx + hs*1.10, hy + hs*0.20],
         [hx,           hy - hs*0.88]],
        facecolor='#e74c3c', edgecolor='none', zorder=4))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=260,
                bbox_inches='tight', transparent=True)
    buf.seek(0); plt.close(fig)
    return buf

# ─────────────────────────────────────────────
#  GÉNÉRATION PDF
# ─────────────────────────────────────────────
def generate_pdf(r):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image, HRFlowable, PageBreak)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.pdfgen import canvas as rl_canvas

    W_PAGE, H_PAGE = A4  # largeur, hauteur en points

    # ── Logo PNG généré une seule fois ──
    logo_buf = make_logo_png_buf()

    # ── Callback dessiné sur chaque page ──
    def draw_page_header(canv, doc):
        """
        Dessine sur chaque page :
        - Bande bleue marine en haut
        - Logo PNG à gauche
        - Texte titre + sous-titre
        - Nom du médecin à droite
        - Numéro de page en bas
        """
        from reportlab.lib.utils import ImageReader
        canv.saveState()

        # Bande bleue (pleine largeur, 2.2 cm de haut)
        bar_h = 2.2 * cm
        bar_y = H_PAGE - bar_h

        # Dégradé simulé avec rectangles
        steps = 60
        for i in range(steps):
            t  = i / steps
            r0 = 13/255 + t*(26/255 - 13/255)
            g0 = 43/255 + t*(78/255 - 43/255)
            b0 = 85/255 + t*(181/255 - 85/255)
            canv.setFillColorRGB(r0, g0, b0)
            x_step = (W_PAGE / steps)
            canv.rect(i*x_step, bar_y, x_step+1, bar_h, fill=1, stroke=0)

        # Logo PNG (carré 1.8cm × 1.8cm, centré verticalement dans la bande)
        logo_size = 1.75 * cm
        logo_x    = 0.25 * cm
        logo_y    = bar_y + (bar_h - logo_size) / 2
        logo_buf.seek(0)
        canv.drawImage(ImageReader(logo_buf),
                       logo_x, logo_y, logo_size, logo_size,
                       preserveAspectRatio=True, mask='auto')

        # Texte titre
        canv.setFillColorRGB(1, 1, 1)
        canv.setFont("Helvetica-Bold", 11)
        canv.drawString(2.3*cm, bar_y + bar_h*0.63,
                        "PediaBMT  |  Decision Support")

        canv.setFont("Helvetica", 6.5)
        canv.setFillColorRGB(1, 1, 1, 0.85)
        canv.drawString(2.3*cm, bar_y + bar_h*0.25,
                        "Systeme d'aide a la decision — Greffes de moelle osseuse pediatriques")

        # Médecin à droite
        canv.setFillColorRGB(1, 1, 1)
        canv.setFont("Helvetica-Bold", 9.5)
        canv.drawRightString(W_PAGE - 0.8*cm,
                             bar_y + bar_h*0.63,
                             f"Dr. {r['medecin']}")
        spec = r.get('specialite','')
        if spec:
            canv.setFont("Helvetica", 7)
            canv.setFillColorRGB(1, 1, 1, 0.85)
            canv.drawRightString(W_PAGE - 0.8*cm,
                                 bar_y + bar_h*0.25, spec)

        # Numéro de page (bas de page)
        canv.setFillColorRGB(0.55, 0.55, 0.55)
        canv.setFont("Helvetica", 8)
        canv.drawCentredString(W_PAGE/2, 1.0*cm,
                               f"PediaBMT · Centrale Casablanca · Page {doc.page}")

        canv.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=3.2*cm,    # espace sous la bande
        bottomMargin=1.8*cm,
        onFirstPage=draw_page_header,
        onLaterPages=draw_page_header,
    )
    story = []

    NAVY  = colors.HexColor('#0d2b55'); BLUE  = colors.HexColor('#1a6eb5')
    LGRAY = colors.HexColor('#f5f8ff'); AMBER = colors.HexColor('#f39c12')
    DGRAY = colors.HexColor('#555555')

    def S(n,**kw): return ParagraphStyle(n,**kw)
    s_title = S('t1',fontName='Helvetica-Bold',fontSize=18,textColor=NAVY,spaceAfter=3,leading=22)
    s_sub   = S('t2',fontName='Helvetica',fontSize=10,textColor=BLUE,spaceAfter=2)
    s_sec   = S('ts',fontName='Helvetica-Bold',fontSize=12,textColor=NAVY,spaceBefore=12,spaceAfter=5)
    s_body  = S('tb',fontName='Helvetica',fontSize=10,textColor=colors.HexColor('#333333'),leading=14)
    s_disc  = S('td',fontName='Helvetica-Oblique',fontSize=8.5,
                textColor=colors.HexColor('#7d5a00'),leading=12)
    s_small = S('ts2',fontName='Helvetica',fontSize=8,textColor=DGRAY,leading=11)

    # Titre du rapport (sans barre d'image)
    story.append(Paragraph("Rapport de Simulation Clinique", s_title))
    story.append(Paragraph(
        "Pediatric Bone Marrow Transplant Decision Support  ·  Centrale Casablanca", s_sub))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=8))

    # Métadonnées
    meta = [["Date & Heure :", r['datetime'],  "Reference :", r['ref']],
            ["Medecin :",      f"Dr. {r['medecin']}", "Modele IA :", r['model_name']]]
    mt = Table(meta, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm])
    mt.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
        ('FONTNAME',(2,0),(2,-1),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('TEXTCOLOR',(0,0),(-1,-1),DGRAY),
        ('TEXTCOLOR',(0,0),(0,-1),NAVY), ('TEXTCOLOR',(2,0),(2,-1),NAVY),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[LGRAY, colors.white]),
        ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#d0d9ef')),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LEFTPADDING',(0,0),(-1,-1),8),
    ]))
    story.append(mt); story.append(Spacer(1, 14))

    # ══ 2. PROFIL PATIENT (1 donnée par ligne) ══
    story.append(Paragraph("1. Profil Clinique du Patient", s_sec))
    profile_rows = [
        ["Age du receveur",       f"{r['recipientage']} ans"],
        ["Masse corporelle",       f"{r['rbodymass']} kg"],
        ["Type de maladie",        r['disease_str']],
        ["Antecedent de rechute",  r['relapse_str']],
        ["Compatibilite de genre", r['gendermatch_str']],
        ["Compatibilite HLA",      r['hlamatch_str']],
    ]
    pdt = Table(profile_rows, colWidths=[7*cm, 11*cm])
    pdt.setStyle(TableStyle([
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
        ('FONTNAME',(1,0),(1,-1),'Helvetica'),
        ('FONTSIZE',(0,0),(-1,-1),9.5),
        ('TEXTCOLOR',(0,0),(0,-1),NAVY),
        ('TEXTCOLOR',(1,0),(1,-1),colors.HexColor('#333333')),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.white, LGRAY]),
        ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#d0d9ef')),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LEFTPADDING',(0,0),(-1,-1),10),
    ]))
    story.append(pdt); story.append(Spacer(1, 14))

    # ══ 3. PROTOCOLE (puissances correctes avec <super>) ══
    story.append(Paragraph("2. Protocole de Traitement Retenu", s_sec))
    pr_data = [
        [Paragraph("<b>Parametre</b>", s_body),
         Paragraph("<b>Valeur retenue</b>", s_body),
         Paragraph("<b>Description</b>", s_body)],
        [Paragraph("Dose CD34+ (x10<super>6</super>/kg)", s_body),
         Paragraph(f"<b>{r['cd34']} x10<super>6</super>/kg</b>", s_body),
         Paragraph("Cellules souches hematopoietiques", s_body)],
        [Paragraph("Dose CD3+ (x10<super>8</super>/kg)", s_body),
         Paragraph(f"<b>{r['cd3']} x10<super>8</super>/kg</b>", s_body),
         Paragraph("Lymphocytes T du greffon", s_body)],
    ]
    prt = Table(pr_data, colWidths=[5.5*cm, 4*cm, 8.5*cm])
    prt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),9.5),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, LGRAY]),
        ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#d0d9ef')),
        ('ALIGN',(1,1),(1,-1),'CENTER'),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LEFTPADDING',(0,0),(-1,-1),10),
    ]))
    story.append(prt); story.append(Spacer(1, 14))

    # ══ 4. PRÉDICTION ══
    prob = r['prob']; pred = r['pred']
    story.append(Paragraph("3. Prediction de l'Intelligence Artificielle", s_sec))
    verdict_txt = ("Probabilite favorable de survie a 1 an avec ce protocole therapeutique."
                   if pred==1 else
                   "Risque eleve d'echec de la greffe. Une revision du protocole est recommandee.")
    sc = "#27ae60" if pred==1 else "#e74c3c"
    lt = "Pronostic FAVORABLE" if pred==1 else "Pronostic DEFAVORABLE"

    pred_data = [[
        Paragraph(f'Score de survie<br/>'
                  f'<font color="{sc}" size="28"><b>{prob*100:.1f}%</b></font>',
                  S('sc', fontName='Helvetica', fontSize=10, textColor=NAVY,
                    alignment=TA_CENTER, leading=34)),
        Paragraph(f'<b>{lt}</b><br/><br/>{verdict_txt}',
                  S('co', fontName='Helvetica', fontSize=10,
                    textColor=colors.HexColor('#333333'), leading=15)),
    ]]
    pt = Table(pred_data, colWidths=[5*cm, 13*cm])
    pt.setStyle(TableStyle([
        ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#d0d9ef')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(0,0),(0,-1),'CENTER'),
        ('TOPPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),12),
        ('LEFTPADDING',(0,0),(-1,-1),10),
        ('BACKGROUND',(0,0),(0,-1),
         colors.HexColor('#e6f9f0') if pred==1 else colors.HexColor('#fff0ee')),
        ('BACKGROUND',(1,0),(1,-1),LGRAY),
    ]))
    story.append(pt); story.append(Spacer(1, 14))

    # ══ 5. SHAP : les DEUX diagrammes (nouvelle page) ══
    story.append(PageBreak())
    story.append(Paragraph("4. Explicabilite de la Prediction (SHAP)", s_sec))
    story.append(Paragraph(
        "Les barres bleues favorisent la survie, les barres rouges la defavorisent.", s_body))
    story.append(Spacer(1, 8))

    if r.get('shap_vals'):
        # ── Diagramme 1 : barres avec fond gris + valeurs ──
        fig1 = make_shap_bars_figure(r['shap_vals'])
        b1   = io.BytesIO()
        fig1.savefig(b1, format='png', dpi=150, bbox_inches='tight')
        b1.seek(0); plt.close(fig1)
        story.append(Image(b1, width=16*cm, height=8*cm))
        story.append(Spacer(1, 12))

        # ── Diagramme 2 : graphique classique ──
        fig2 = make_shap_classic_figure(r['shap_vals'])
        b2   = io.BytesIO()
        fig2.savefig(b2, format='png', dpi=150, bbox_inches='tight')
        b2.seek(0); plt.close(fig2)
        story.append(Image(b2, width=16*cm, height=8.5*cm))
    else:
        story.append(Paragraph("Graphique SHAP non disponible.", s_small))

    story.append(Spacer(1, 18))

    # ══ 6. DISCLAIMER ══
    story.append(HRFlowable(width="100%", thickness=1, color=AMBER, spaceAfter=8))
    story.append(Paragraph("Avertissement Legal et Clinique",
                            S('dh', fontName='Helvetica-Bold', fontSize=10,
                              textColor=colors.HexColor('#7d5a00'), spaceAfter=4)))
    story.append(Paragraph(
        "Ce rapport a ete genere automatiquement par le systeme PediaBMT base sur des algorithmes "
        "d'apprentissage automatique. Il ne remplace en aucun cas le jugement clinique d'un professionnel "
        "de sante habilite. Toute decision therapeutique doit etre validee par un medecin specialiste.",
        s_disc))
    story.append(Spacer(1, 18))

    # ══ 7. SIGNATURE ══
    sig = Table([["Fait le :", r['datetime'], "Signature du medecin :", ""]],
                colWidths=[3*cm, 5*cm, 5*cm, 5*cm])
    sig.setStyle(TableStyle([
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
        ('FONTNAME',(2,0),(2,-1),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),9),('TEXTCOLOR',(0,0),(-1,-1),DGRAY),
        ('LINEBELOW',(3,0),(3,-1),1,DGRAY),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),12),
        ('LEFTPADDING',(0,0),(-1,-1),6),
    ]))
    story.append(sig)
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceBefore=10))

    doc.build(story); buf.seek(0)
    return buf

# ─────────────────────────────────────────────
#  CALENDRIER
# ─────────────────────────────────────────────
def render_calendar():
    today=date.today(); cur=st.session_state.cal_month; y,m=cur.year,cur.month
    appt_dates={a['date'] for a in st.session_state.appointments}
    c1,c2,c3=st.columns([1,3,1])
    with c1:
        if st.button("◀",key="prev_m"):
            first=st.session_state.cal_month.replace(day=1)
            st.session_state.cal_month=(first-timedelta(days=1)).replace(day=1); st.rerun()
    with c2:
        st.markdown(f'<div style="text-align:center;font-weight:700;font-size:.83rem;color:#0d2b55;">{MOIS_FR[m]} {y}</div>',unsafe_allow_html=True)
    with c3:
        if st.button("▶",key="next_m"):
            last=date(y,m,calendar.monthrange(y,m)[1])
            st.session_state.cal_month=(last+timedelta(days=1)); st.rerun()
    first_wd=date(y,m,1).weekday(); days_total=calendar.monthrange(y,m)[1]
    cells=[None]*first_wd+list(range(1,days_total+1))
    while len(cells)%7!=0: cells.append(None)
    html='<table class="mini-cal"><thead><tr>'
    for j in JOURS_FR: html+=f'<th>{j}</th>'
    html+='</tr></thead><tbody>'
    for i in range(0,len(cells),7):
        html+='<tr>'
        for day in cells[i:i+7]:
            if day is None: html+='<td></td>'
            else:
                d=date(y,m,day); css='cal-cell'
                if d==today: css+=' cal-today'
                if d in appt_dates: css+=' cal-has-appt'
                html+=f'<td><span class="{css}">{day}</span></td>'
        html+='</tr>'
    html+='</tbody></table>'
    st.markdown(html,unsafe_allow_html=True)
    st.markdown('<div style="font-size:.68rem;color:#888;margin-top:3px;">'
                '<span style="color:#2ecc71;font-weight:700;">■</span> RDV &nbsp;'
                '<span style="background:#1e6eb5;color:white;border-radius:3px;padding:1px 4px;font-size:.62rem;">■</span> Aujourd\'hui'
                '</div>',unsafe_allow_html=True)

# ═══════════════════════════════════════
#  PAGE AUTH
# ═══════════════════════════════════════
def show_auth_page():
    _,col,_=st.columns([1,1.1,1])
    with col:
        st.markdown(f'<div style="display:flex;flex-direction:column;align-items:center;padding:2rem 0 1rem;">{LOGO_FULL}</div>',unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        tab1,tab2=st.tabs(["🔑 Connexion","📝 Créer un compte"])
        with tab1:
            st.markdown("<br>",unsafe_allow_html=True)
            username=st.text_input("Identifiant",placeholder="Ex: dr.zerhouni",key="login_user")
            password=st.text_input("Mot de passe",type="password",placeholder="••••••••",key="login_pass")
            st.markdown("<br>",unsafe_allow_html=True)
            if st.button("Se connecter →",key="btn_login"):
                users=load_users()
                if not username.strip(): st.error("⚠️ Veuillez saisir votre identifiant.")
                elif username not in users: st.error("❌ Identifiant introuvable.")
                elif users[username]['password']!=hash_password(password): st.error("❌ Mot de passe incorrect.")
                else:
                    st.session_state.logged_in=True
                    st.session_state.current_user={
                        'username':username,'nom':users[username]['nom'],
                        'prenom':users[username]['prenom'],
                        'specialite':users[username].get('specialite',''),
                    }
                    st.session_state.page='main'; st.rerun()
        with tab2:
            st.markdown("<br>",unsafe_allow_html=True)
            cp,cn=st.columns(2)
            with cp: prenom_new=st.text_input("Prénom",placeholder="Mohamed",key="reg_prenom")
            with cn: nom_new=st.text_input("Nom",placeholder="Zerhouni",key="reg_nom")
            spec_new=st.selectbox("Spécialité",["Hématologie pédiatrique","Oncologie pédiatrique","Médecine interne","Transplantation","Autre"],key="reg_spec")
            user_new=st.text_input("Identifiant",placeholder="Ex: dr.zerhouni",key="reg_user")
            pass_new=st.text_input("Mot de passe",type="password",placeholder="Min. 6 caractères",key="reg_pass")
            conf_new=st.text_input("Confirmer",type="password",placeholder="••••••••",key="reg_conf")
            st.markdown("<br>",unsafe_allow_html=True)
            if st.button("Créer mon compte →",key="btn_register"):
                users=load_users()
                if not all([prenom_new.strip(),nom_new.strip(),user_new.strip(),pass_new]):
                    st.error("⚠️ Veuillez remplir tous les champs.")
                elif user_new in users: st.error("❌ Identifiant déjà utilisé.")
                elif len(pass_new)<6: st.error("⚠️ Mot de passe trop court.")
                elif pass_new!=conf_new: st.error("❌ Mots de passe différents.")
                else:
                    users[user_new]={'nom':nom_new.strip(),'prenom':prenom_new.strip(),
                                     'specialite':spec_new,'password':hash_password(pass_new)}
                    save_users(users); st.success(f"✅ Compte créé ! Connectez-vous avec : {user_new}")

# ═══════════════════════════════════════
#  PAGE RÉSULTATS
# ═══════════════════════════════════════
def show_results_page():
    r=st.session_state.last_result
    user=st.session_state.current_user

    st.markdown(f"""
    <div class="header-banner">
        <div class="header-left">{LOGO_SMALL}
            <div><h1>PediaBMT · Rapport de Simulation</h1>
            <p>Résultats détaillés et explicabilité IA</p></div>
        </div>
        <div class="header-user"><strong>Dr. {user['prenom']} {user['nom']}</strong>{user['specialite']}</div>
    </div>""",unsafe_allow_html=True)

    cb,_,cpdf=st.columns([1,3,1])
    with cb:
        if st.button("← Retour à l'analyse"):
            st.session_state.page='main'; st.rerun()
    with cpdf:
        pdf_buf=generate_pdf(r)
        st.download_button(
            label="⬇️ Télécharger le PDF",
            data=pdf_buf,
            file_name=f"PediaBMT_{r['ref']}_{date.today().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown(f"""
    <div class="res-header">
        <h2>Rapport Clinique de Simulation IA</h2>
        <div style="font-size:.88rem;opacity:.9;margin-top:.3rem;">Système d'aide à la décision — Greffe pédiatrique de moelle osseuse</div>
        <div class="res-meta">
            <span>📅 {r['datetime']}</span>
            <span>👨‍⚕️ Dr. {r['medecin']}</span>
            <span>🗂 {r['ref']}</span>
            <span>🤖 {r['model_name']}</span>
        </div>
    </div>""",unsafe_allow_html=True)

    col_left,col_right=st.columns([3,2],gap="large")

    with col_left:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="card-title">👤 Profil Clinique du Patient</div>',unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-grid">
            <div class="info-box"><div class="info-box-label">Âge du receveur</div><div class="info-box-value">{r['recipientage']} ans</div></div>
            <div class="info-box"><div class="info-box-label">Masse corporelle</div><div class="info-box-value">{r['rbodymass']} kg</div></div>
            <div class="info-box"><div class="info-box-label">Type de maladie</div><div class="info-box-value">{r['disease_str']}</div></div>
            <div class="info-box"><div class="info-box-label">Antécédent de rechute</div><div class="info-box-value">{r['relapse_str']}</div></div>
            <div class="info-box"><div class="info-box-label">Compatibilité de genre</div><div class="info-box-value">{r['gendermatch_str']}</div></div>
            <div class="info-box"><div class="info-box-label">Compatibilité HLA</div><div class="info-box-value">{r['hlamatch_str']}</div></div>
        </div>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="card-title">💉 Protocole de Traitement Retenu</div>',unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-grid">
            <div class="info-box" style="border-left-color:#1e6eb5;">
                <div class="info-box-label">Dose CD34+ retenue</div>
                <div class="info-box-value" style="font-size:1.3rem;color:#1e6eb5;">{r['cd34']} ×10⁶/kg</div>
            </div>
            <div class="info-box" style="border-left-color:#1e6eb5;">
                <div class="info-box-label">Dose CD3+ retenue</div>
                <div class="info-box-value" style="font-size:1.3rem;color:#1e6eb5;">{r['cd3']} ×10⁸/kg</div>
            </div>
        </div>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

        # SHAP interface web
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Explicabilité SHAP</div>',unsafe_allow_html=True)
        if r.get('shap_vals'):
            mx=max(abs(v) for v in r['shap_vals'].values()) or 1e-9
            ss=sorted(r['shap_vals'].items(),key=lambda x:abs(x[1]),reverse=True)
            st.markdown("".join(shap_bar_html(f,v,mx) for f,v in ss),unsafe_allow_html=True)
            st.markdown('<div style="margin-top:.6rem;font-size:.74rem;color:#666;">'
                        '<span style="color:#1a6eb5;">■</span> Favorise la survie &nbsp;'
                        '<span style="color:#e74c3c;">■</span> Défavorise la survie</div>',unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            fig=make_shap_figure_web(r['shap_vals'])
            st.pyplot(fig,use_container_width=True); plt.close(fig)
        else:
            st.info("📊 SHAP non disponible. `pip install shap`")
        st.markdown('</div>',unsafe_allow_html=True)

    with col_right:
        prob=r['prob']; pred=r['pred']
        if pred==1:
            st.markdown(f"""
            <div class="verdict-success">
                <div class="verdict-score">{prob*100:.1f}%</div>
                <div class="verdict-label">✅ Survie Probable</div>
                <div class="verdict-text">Le modèle estime une probabilité favorable de survie à 1 an avec ce protocole thérapeutique.</div>
                <div class="prob-bar-bg">
                    <div style="width:{prob*100:.1f}%;height:14px;border-radius:20px;background:linear-gradient(90deg,#27ae60,#2ecc71);"></div>
                </div>
            </div>""",unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-risk">
                <div class="verdict-score">{prob*100:.1f}%</div>
                <div class="verdict-label">⚠️ Risque Élevé</div>
                <div class="verdict-text">Le modèle identifie un risque élevé d'échec. Une révision du protocole est recommandée.</div>
                <div class="prob-bar-bg">
                    <div style="width:{prob*100:.1f}%;height:14px;border-radius:20px;background:linear-gradient(90deg,#c0392b,#e74c3c);"></div>
                </div>
            </div>""",unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-box">
            <div class="disclaimer-title">⚠️ Avertissement Clinique</div>
            <div class="disclaimer-text">Ce rapport est un support décisionnel basé sur l'apprentissage automatique.
            Il ne remplace en aucun cas le jugement clinique d'un professionnel de santé habilité.</div>
            <div class="signature-line">Signature du médecin : ________________________________</div>
        </div>""",unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="text-align:center;font-size:.71rem;color:#aaa;">PediaBMT Decision Support · Centrale Casablanca · Coding Week 2026</div>',unsafe_allow_html=True)

# ═══════════════════════════════════════
#  PAGE PRINCIPALE
# ═══════════════════════════════════════
def show_main_app():
    user  = st.session_state.current_user
    model, scaler, feat_list = load_artifacts()

    st.markdown(f"""
    <div class="header-banner">
        <div class="header-left">{LOGO_SMALL}
            <div><h1>PediaBMT · Decision Support</h1>
            <p>Système d'aide à la décision — Greffes de moelle osseuse pédiatriques</p></div>
        </div>
        <div class="header-user"><strong>Dr. {user['prenom']} {user['nom']}</strong>{user['specialite']}</div>
    </div>""", unsafe_allow_html=True)

    _, col_logout = st.columns([5, 1])
    with col_logout:
        if st.button("🚪 Déconnexion"):
            st.session_state.logged_in  = False
            st.session_state.current_user = None
            st.session_state.page = 'main'; st.rerun()

    if model is None:
        st.error("⚠️ Modèle introuvable. Lancez : `python src/train_model.py`"); st.stop()

    st.markdown(f'<div style="text-align:right;margin-bottom:.9rem;">'
                f'<span class="metric-chip">🤖 {type(model).__name__}</span>'
                f'<span class="metric-chip">✅ Modèle chargé</span></div>', unsafe_allow_html=True)

    col_main, col_right = st.columns([3, 2], gap="large")

    with col_main:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 Données du patient</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Variables numériques</div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            recipientage = st.number_input("Âge du receveur (ans)",  0.0, 25.0,  8.0, .5)
            cd34         = st.number_input("Dose CD34+ (×10⁶/kg)",   0.0, 50.0,  5.5, .1)
        with cb:
            rbodymass    = st.number_input("Masse corporelle (kg)",   5.0, 150.0, 28.0, .5)
            cd3          = st.number_input("Dose CD3+ (×10⁸/kg)",    0.0, 100.0,  3.2, .1)
        st.markdown('<div class="section-title">Variables catégorielles</div>', unsafe_allow_html=True)
        cc, cd = st.columns(2)
        with cc:
            disease_str     = st.selectbox("Type de maladie",        list(DISEASE_MAP.keys()))
            gendermatch_str = st.selectbox("Compatibilité de genre",  list(GENDERMATCH_MAP.keys()))
        with cd:
            relapse_str  = st.selectbox("Antécédent de rechute", list(RELAPSE_MAP.keys()))
            hlamatch_str = st.selectbox("Compatibilité HLA",      list(HLAMATCH_MAP.keys()))
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔬 Analyser la prédiction")
        st.markdown('<div class="disclaimer">⚠️ <strong>Avertissement clinique :</strong> '
                    'Cet outil est un support décisionnel. Il ne remplace pas le jugement médical.</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_btn:
            # Valeurs brutes saisies
            raw_values = {
                'CD3dkgx10d8' : cd3,
                'Rbodymass'   : rbodymass,
                'Recipientage': recipientage,
                'CD34kgx10d6' : cd34,
                'Disease'     : DISEASE_MAP[disease_str],
                'Relapse'     : RELAPSE_MAP[relapse_str],
                'Gendermatch' : GENDERMATCH_MAP[gendermatch_str],
                'HLAmatch'    : HLAMATCH_MAP[hlamatch_str],
            }
            # DataFrame dans l'ordre exact de feat_list (= IMPORTANT_FEATURES du train)
            input_df = pd.DataFrame([[raw_values[f] for f in feat_list]], columns=feat_list)

            with st.spinner("Analyse en cours..."):
                pred, prob  = predict_fn(model, scaler, input_df)
                shap_vals   = compute_shap(model, scaler, input_df)

            st.session_state.last_result = {
                'pred': pred, 'prob': prob, 'shap_vals': shap_vals,
                'ref'     : f"PAT-{random.randint(1000,9999)}",
                'datetime': datetime.now().strftime("%d %B %Y — %Hh%M"),
                'medecin' : f"{user['prenom']} {user['nom']}",
                'specialite': user.get('specialite',''),
                'model_name': type(model).__name__,
                'recipientage': recipientage, 'rbodymass': rbodymass,
                'cd34': cd34, 'cd3': cd3,
                'disease_str'    : disease_str,    'relapse_str': relapse_str,
                'gendermatch_str': gendermatch_str, 'hlamatch_str': hlamatch_str,
            }
            st.session_state.page = 'results'; st.rerun()

    with col_right:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="card-title">📅 Calendrier</div>',unsafe_allow_html=True)
        render_calendar()
        st.markdown('</div>',unsafe_allow_html=True)

        today=date.today()
        rdv_futurs=sorted([a for a in st.session_state.appointments if a['date']>=today],
                          key=lambda x:(x['date'],x['heure']))
        hw='<div class="appt-widget">'
        hw+=(f'<div class="appt-widget-header"><span class="appt-widget-title">📋 Prochains rendez-vous</span>'
             f'<span style="color:#b8c8e0;font-size:.72rem;">{len(rdv_futurs)} à venir</span></div>')
        if rdv_futurs:
            for rdv in rdv_futurs[:3]:
                hw+=(f'<div class="appt-mini-item"><div class="appt-mini-date">{rdv["date"].strftime("%d/%m")}<br>'
                     f'<span style="font-size:.65rem;font-weight:400;">{rdv["heure"]}</span></div>'
                     f'<div><div class="appt-mini-name">{rdv["nom"]}</div>'
                     f'<div class="appt-mini-motif">{rdv["motif"]}</div></div></div>')
        else:
            hw+='<div class="appt-mini-empty">📭 Aucun rendez-vous à venir</div>'
        hw+='</div>'
        st.markdown(hw,unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)

        b1,b2=st.columns(2)
        with b1:
            if st.button("📂 Voir tous"):
                st.session_state.show_all_appts=not st.session_state.show_all_appts
                st.session_state.show_add_form=False; st.rerun()
        with b2:
            if st.button("➕ Ajouter"):
                st.session_state.show_add_form=not st.session_state.show_add_form
                st.session_state.show_all_appts=False; st.rerun()

        if st.session_state.show_all_appts:
            st.markdown('<div class="card" style="margin-top:.6rem;">',unsafe_allow_html=True)
            st.markdown('<div class="card-title">🗓️ Tous les rendez-vous</div>',unsafe_allow_html=True)
            all_s=sorted(st.session_state.appointments,key=lambda x:(x['date'],x['heure']))
            if all_s:
                for i,rdv in enumerate(all_s):
                    ci,cd2=st.columns([5,1])
                    with ci:
                        st.markdown(f'<div class="appt-full-item"><div class="appt-full-date">{rdv["date"].strftime("%d/%m/%Y")} à {rdv["heure"]}</div>'
                                    f'<div class="appt-full-name">{rdv["nom"]}</div><div class="appt-full-motif">{rdv["motif"]}</div></div>',unsafe_allow_html=True)
                    with cd2:
                        st.markdown("<br>",unsafe_allow_html=True)
                        if st.button("🗑️",key=f"del_{i}"):
                            st.session_state.appointments.pop(i); st.rerun()
            else:
                st.markdown('<div style="text-align:center;color:#aaa;padding:.8rem;">📭 Aucun rendez-vous</div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)

        if st.session_state.show_add_form:
            st.markdown('<div class="card" style="margin-top:.6rem;">',unsafe_allow_html=True)
            st.markdown('<div class="card-title">➕ Nouveau rendez-vous</div>',unsafe_allow_html=True)
            nom_p=st.text_input("Nom du patient",placeholder="Ex: Ahmed Benali",key="nom_in")
            date_rdv=st.date_input("Date",value=today,min_value=today,key="date_in")
            heure_r=st.selectbox("Heure",HEURES,index=4,key="heure_in")
            motif_r=st.selectbox("Motif",MOTIFS,key="motif_in")
            if st.button("✅ Confirmer"):
                if nom_p.strip():
                    st.session_state.appointments.append({'date':date_rdv,'heure':heure_r,'nom':nom_p.strip(),'motif':motif_r})
                    st.success(f"✅ RDV : {nom_p} le {date_rdv.strftime('%d/%m/%Y')} à {heure_r}")
                    st.session_state.show_add_form=False; st.rerun()
                else:
                    st.warning("⚠️ Nom du patient requis.")
            st.markdown('</div>',unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="text-align:center;font-size:.71rem;color:#aaa;">PediaBMT Decision Support · Centrale Casablanca · Coding Week 2026</div>',unsafe_allow_html=True)

# ═══════════════════════════════════════
#  ROUTING
# ═══════════════════════════════════════
if not st.session_state.logged_in:
    show_auth_page()
elif st.session_state.page == 'results' and st.session_state.last_result is not None:
    show_results_page()
else:
    st.session_state.page = 'main'
    show_main_app()