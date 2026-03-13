import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import hashlib
import matplotlib.pyplot as plt
import warnings
from datetime import date, timedelta
import calendar
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
#  USERS FILE
# ─────────────────────────────────────────────
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, val in [
    ('logged_in', False), ('current_user', None), ('auth_mode', 'login'),
    ('appointments', []), ('cal_month', date.today().replace(day=1)),
    ('show_all_appts', False), ('show_add_form', False),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
#  CSS GLOBAL
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f7f8fc;
}

/* ── Inputs black text ── */
input, textarea, select,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stPasswordInput"] input,
.stSelectbox div[data-baseweb="select"] *,
div[data-baseweb="input"] input,
div[data-baseweb="select"] div {
    color: #111111 !important;
    background-color: #ffffff !important;
    font-size: 0.97rem !important;
}
label, [data-testid="stWidgetLabel"] p {
    color: #1a1a1a !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* ── Auth page ── */
.auth-bg {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}
.auth-card {
    background: white;
    border-radius: 20px;
    padding: 2.8rem 3rem;
    box-shadow: 0 8px 40px rgba(13,43,85,0.13);
    border: 1px solid #e0e8f5;
    max-width: 440px;
    width: 100%;
}
.auth-logo {
    text-align: center;
    margin-bottom: 1.8rem;
}
.auth-logo-icon {
    font-size: 3.5rem;
    display: block;
    margin-bottom: 0.5rem;
}
.auth-logo-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #0d2b55;
    margin: 0;
}
.auth-logo-sub {
    font-size: 0.85rem;
    color: #888;
    margin-top: 0.2rem;
}
.auth-tab-active {
    background: #0d2b55;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    border: none;
    cursor: pointer;
}
.auth-tab-inactive {
    background: transparent;
    color: #888;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    font-size: 0.95rem;
    border: none;
    cursor: pointer;
}
.auth-divider {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 1rem 0;
    color: #ccc;
    font-size: 0.8rem;
}
.auth-divider::before, .auth-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e8ecf4;
}
.auth-welcome {
    background: linear-gradient(135deg, #e8f0fb, #dce8fa);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 1.2rem;
    font-size: 0.88rem;
    color: #1a4a8a;
    border-left: 4px solid #1e6eb5;
}

/* ── Main app ── */
.header-banner {
    background: linear-gradient(135deg, #0d2b55 0%, #1a4a8a 60%, #1e6eb5 100%);
    border-radius: 16px;
    padding: 1.6rem 2.2rem;
    margin-bottom: 1.8rem;
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-left { display: flex; align-items: center; gap: 1.2rem; }
.header-banner h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    margin: 0;
    color: white !important;
}
.header-banner p { margin: 0.2rem 0 0; font-size: 0.9rem; opacity: 0.85; color: white !important; }
.header-icon { font-size: 2.6rem; }
.header-user {
    text-align: right;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.85);
}
.header-user strong { display: block; font-size: 1rem; color: white; }

.card {
    background: white;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 2px 14px rgba(13,43,85,0.07);
    margin-bottom: 1.2rem;
    border: 1px solid #e8ecf4;
}
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #0d2b55;
    margin-bottom: 1.1rem;
    border-bottom: 2px solid #e8ecf4;
    padding-bottom: 0.6rem;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 0.98rem;
    color: #0d2b55;
    margin: 1rem 0 0.6rem;
    border-left: 3px solid #1e6eb5;
    padding-left: 0.6rem;
}

.result-success {
    background: linear-gradient(135deg, #e6f9f0, #c8f0dd);
    border: 2px solid #2ecc71; border-radius: 14px;
    padding: 1.5rem; text-align: center; color: #1a6b40;
}
.result-risk {
    background: linear-gradient(135deg, #fff0ee, #ffdad6);
    border: 2px solid #e74c3c; border-radius: 14px;
    padding: 1.5rem; text-align: center; color: #8b1a1a;
}
.result-title { font-family: 'DM Serif Display', serif; font-size: 1.6rem; margin-bottom: 0.3rem; }
.result-subtitle { font-size: 0.92rem; opacity: 0.8; }
.prob-container { background: rgba(255,255,255,0.6); border-radius: 10px; padding: 0.7rem 1rem; margin-top: 0.7rem; }
.prob-bar-bg { background: #dce3f0; border-radius: 20px; height: 13px; overflow: hidden; margin-top: 0.3rem; }

.shap-item { display: flex; align-items: center; gap: 0.7rem; padding: 0.45rem 0; border-bottom: 1px solid #f0f3fa; }
.shap-label { flex: 1; font-size: 0.85rem; color: #333; font-weight: 500; }
.shap-bar-bg { width: 120px; background: #eef1f8; border-radius: 8px; height: 9px; overflow: hidden; }
.shap-val { font-size: 0.78rem; min-width: 50px; text-align: right; }

.mini-cal { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.mini-cal th { text-align: center; color: #1a4a8a; font-weight: 700; padding: 3px 1px; font-size: 0.72rem; }
.mini-cal td { text-align: center; padding: 2px 1px; width: 14.2%; }
.cal-cell { display: inline-block; width: 26px; height: 26px; line-height: 26px;
    border-radius: 6px; text-align: center; font-size: 0.78rem; color: #333; }
.cal-today { background: #1e6eb5 !important; color: white !important; font-weight: 700; }
.cal-has-appt { border: 2px solid #2ecc71; color: #1a6b40; font-weight: 700; }

.appt-widget { background: white; border-radius: 12px; border: 1px solid #e8ecf4; box-shadow: 0 2px 10px rgba(13,43,85,0.07); overflow: hidden; }
.appt-widget-header { background: linear-gradient(135deg, #0d2b55, #1a4a8a); padding: 0.6rem 1rem; display: flex; justify-content: space-between; align-items: center; }
.appt-widget-title { color: white; font-weight: 700; font-size: 0.88rem; }
.appt-mini-item { padding: 0.5rem 0.9rem; border-bottom: 1px solid #f0f3fa; display: flex; align-items: center; gap: 0.6rem; }
.appt-mini-date { background: #eef2fb; border-radius: 6px; padding: 2px 6px; font-size: 0.72rem; font-weight: 700; color: #1a4a8a; min-width: 44px; text-align: center; }
.appt-mini-name { font-weight: 600; font-size: 0.82rem; color: #1a1a1a; }
.appt-mini-motif { color: #888; font-size: 0.74rem; }
.appt-mini-empty { padding: 0.9rem; text-align: center; color: #bbb; font-size: 0.82rem; }
.appt-full-item { background: #f5f8ff; border-left: 4px solid #1e6eb5; border-radius: 8px; padding: 0.65rem 0.9rem; margin-bottom: 0.45rem; }
.appt-full-date { font-weight: 700; color: #1e6eb5; font-size: 0.85rem; }
.appt-full-name { font-weight: 600; font-size: 0.88rem; color: #1a1a1a; }
.appt-full-motif { color: #666; font-size: 0.78rem; }

.metric-chip { background: #eef2fb; border-radius: 8px; padding: 0.35rem 0.8rem; display: inline-block; margin: 0.2rem; font-size: 0.82rem; color: #0d2b55; font-weight: 500; border: 1px solid #d0d9ef; }
.disclaimer { background: #fffbea; border-left: 4px solid #f39c12; border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.8rem; color: #7d5a00; margin-top: 0.9rem; }

.stButton > button {
    background: linear-gradient(135deg, #1a4a8a, #1e6eb5) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 0.6rem 1.4rem !important; font-size: 0.95rem !important;
    font-weight: 600 !important; width: 100% !important;
}
.stButton > button:hover { background: linear-gradient(135deg, #0d2b55, #1a4a8a) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTES ML
# ─────────────────────────────────────────────
FEATURES = ['Recipientage','Rbodymass','CD34kgx10d6','CD3dkgx10d8',
            'Disease','Relapse','Gendermatch','HLAmatch']
FEATURE_LABELS = {
    'Recipientage':'Âge du receveur (ans)','Rbodymass':'Masse corporelle (kg)',
    'CD34kgx10d6':'Dose CD34+ (×10⁶/kg)','CD3dkgx10d8':'Dose CD3+ (×10⁸/kg)',
    'Disease':'Type de maladie','Relapse':'Antécédent de rechute',
    'Gendermatch':'Compatibilité de genre','HLAmatch':'Compatibilité HLA',
}
DISEASE_MAP     = {'ALL (Leucémie aiguë lymphoblastique)':0,'AML (Leucémie aiguë myéloïde)':1,'CML (Leucémie myéloïde chronique)':2,'Autre':3}
RELAPSE_MAP     = {'Non':0,'Oui':1}
GENDERMATCH_MAP = {'Matched (compatible)':0,'Mismatched (incompatible)':1}
HLAMATCH_MAP    = {'Matched (0 mismatch)':0,'1 antigène incompatible':1,'2 antigènes incompatibles':2,'3 antigènes incompatibles':3}
HEURES = [f"{h:02d}:{m:02d}" for h in range(8,19) for m in (0,30)]
MOTIFS = ["Consultation initiale","Suivi post-greffe","Résultats d'analyses","Bilan pré-greffe","Urgence médicale","Autre"]
JOURS_FR = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
MOIS_FR  = ["","Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]

# ─────────────────────────────────────────────
#  MODÈLE
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    for path in ['models/final_model.joblib','../models/final_model.joblib',
                 os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','models','final_model.joblib'),
                 os.path.join(os.getcwd(),'models','final_model.joblib')]:
        if os.path.exists(path):
            sp = path.replace('final_model.joblib','scaler.joblib')
            return joblib.load(path), (joblib.load(sp) if os.path.exists(sp) else None)
    return None, None

def predict(model, scaler, input_df):
    Xs = scaler.transform(input_df) if scaler else input_df.values
    return int(model.predict(Xs)[0]), float(model.predict_proba(Xs)[0][1])

def compute_shap(model, scaler, input_df):
    try:
        import shap
        Xs = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns) if scaler else input_df
        if type(model).__name__ in ('RandomForestClassifier','XGBClassifier','LGBMClassifier'):
            exp = shap.TreeExplainer(model); sv = exp.shap_values(Xs)
            return dict(zip(FEATURES, sv[1][0] if isinstance(sv,list) else sv[0]))
        else:
            bg = pd.DataFrame(np.zeros((1,len(FEATURES))),columns=FEATURES)
            exp = shap.KernelExplainer(model.predict_proba, bg)
            return dict(zip(FEATURES, exp.shap_values(Xs,nsamples=100)[1][0]))
    except: return None

def shap_bar(feature, value, max_abs):
    label = FEATURE_LABELS.get(feature, feature)
    pct   = min(abs(value)/max_abs*100,100) if max_abs else 0
    color = "#1a6eb5" if value>=0 else "#e74c3c"
    sign  = "+" if value>=0 else "−"
    return (f'<div class="shap-item"><span class="shap-label">{label}</span>'
            f'<div class="shap-bar-bg"><div style="width:{pct:.0f}%;background:{color};'
            f'height:9px;border-radius:8px;"></div></div>'
            f'<span class="shap-val" style="color:{color}">{sign}{abs(value):.4f}</span></div>')

# ─────────────────────────────────────────────
#  CALENDRIER COMPACT
# ─────────────────────────────────────────────
def render_compact_calendar():
    today = date.today()
    cur   = st.session_state.cal_month
    y, m  = cur.year, cur.month
    appt_dates = {a['date'] for a in st.session_state.appointments}

    c1, c2, c3 = st.columns([1,3,1])
    with c1:
        if st.button("◀", key="prev_m"):
            first = st.session_state.cal_month.replace(day=1)
            st.session_state.cal_month = (first - timedelta(days=1)).replace(day=1)
            st.rerun()
    with c2:
        st.markdown(f'<div style="text-align:center;font-weight:700;font-size:0.85rem;color:#0d2b55;">{MOIS_FR[m]} {y}</div>', unsafe_allow_html=True)
    with c3:
        if st.button("▶", key="next_m"):
            last = date(y, m, calendar.monthrange(y,m)[1])
            st.session_state.cal_month = (last + timedelta(days=1))
            st.rerun()

    first_wd   = date(y, m, 1).weekday()
    days_total = calendar.monthrange(y, m)[1]
    cells      = [None]*first_wd + list(range(1, days_total+1))
    while len(cells) % 7 != 0: cells.append(None)

    html = '<table class="mini-cal"><thead><tr>'
    for j in JOURS_FR: html += f'<th>{j}</th>'
    html += '</tr></thead><tbody>'
    for i in range(0, len(cells), 7):
        html += '<tr>'
        for day in cells[i:i+7]:
            if day is None:
                html += '<td></td>'
            else:
                d = date(y, m, day)
                css = 'cal-cell'
                if d == today:      css += ' cal-today'
                if d in appt_dates: css += ' cal-has-appt'
                html += f'<td><span class="{css}">{day}</span></td>'
        html += '</tr>'
    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#888;margin-top:3px;">'
                '<span style="color:#2ecc71;font-weight:700;">■</span> RDV &nbsp;'
                '<span style="background:#1e6eb5;color:white;border-radius:3px;padding:1px 4px;font-size:0.65rem;">■</span> Aujourd\'hui'
                '</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE LOGIN / SIGNUP
# ═══════════════════════════════════════════════════════
def show_auth_page():
    # Centrer avec colonnes
    _, col_auth, _ = st.columns([1, 1.2, 1])

    with col_auth:
        # Logo
        st.markdown("""
        <div class="auth-logo">
            <span class="auth-logo-icon">🩸</span>
            <div class="auth-logo-title">PediaBMT</div>
            <div class="auth-logo-sub">Decision Support · Centrale Casablanca</div>
        </div>
        """, unsafe_allow_html=True)

        # Tabs Login / Inscription
        tab1, tab2 = st.tabs(["🔑 Connexion", "📝 Créer un compte"])

        # ── TAB CONNEXION ──
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Identifiant", placeholder="Ex: dr.zerhouni", key="login_user")
            password = st.text_input("Mot de passe", type="password", placeholder="••••••••", key="login_pass")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Se connecter →", key="btn_login"):
                users = load_users()
                if username.strip() == "":
                    st.error("⚠️ Veuillez saisir votre identifiant.")
                elif username not in users:
                    st.error("❌ Identifiant introuvable. Créez un compte d'abord.")
                elif users[username]['password'] != hash_password(password):
                    st.error("❌ Mot de passe incorrect.")
                else:
                    st.session_state.logged_in    = True
                    st.session_state.current_user = {
                        'username': username,
                        'nom':      users[username]['nom'],
                        'prenom':   users[username]['prenom'],
                        'specialite': users[username].get('specialite',''),
                    }
                    st.rerun()

        # ── TAB INSCRIPTION ──
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            col_p, col_n = st.columns(2)
            with col_p:
                prenom_new = st.text_input("Prénom", placeholder="Mohamed", key="reg_prenom")
            with col_n:
                nom_new    = st.text_input("Nom",    placeholder="Zerhouni",  key="reg_nom")

            specialite_new = st.selectbox("Spécialité", [
                "Hématologie pédiatrique", "Oncologie pédiatrique",
                "Médecine interne", "Transplantation", "Autre"
            ], key="reg_spec")
            username_new   = st.text_input("Identifiant", placeholder="Ex: dr.zerhouni", key="reg_user")
            password_new   = st.text_input("Mot de passe", type="password",
                                           placeholder="Min. 6 caractères", key="reg_pass")
            password_conf  = st.text_input("Confirmer le mot de passe", type="password",
                                           placeholder="••••••••", key="reg_conf")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Créer mon compte →", key="btn_register"):
                users = load_users()
                if not all([prenom_new.strip(), nom_new.strip(), username_new.strip(), password_new]):
                    st.error("⚠️ Veuillez remplir tous les champs.")
                elif username_new in users:
                    st.error("❌ Cet identifiant est déjà utilisé.")
                elif len(password_new) < 6:
                    st.error("⚠️ Le mot de passe doit contenir au moins 6 caractères.")
                elif password_new != password_conf:
                    st.error("❌ Les mots de passe ne correspondent pas.")
                else:
                    users[username_new] = {
                        'nom':        nom_new.strip(),
                        'prenom':     prenom_new.strip(),
                        'specialite': specialite_new,
                        'password':   hash_password(password_new),
                    }
                    save_users(users)
                    st.success(f"✅ Compte créé avec succès ! Connectez-vous avec l'identifiant : {username_new}")


# ═══════════════════════════════════════════════════════
#  PAGE PRINCIPALE
# ═══════════════════════════════════════════════════════
def show_main_app():
    user   = st.session_state.current_user
    model, scaler = load_artifacts()

    # ── Header avec nom du médecin ──
    st.markdown(f"""
    <div class="header-banner">
        <div class="header-left">
            <div class="header-icon">🩸</div>
            <div>
                <h1>PediaBMT · Decision Support</h1>
                <p>Système d'aide à la décision — Greffes de moelle osseuse pédiatriques</p>
            </div>
        </div>
        <div class="header-user">
            <strong>Dr. {user['prenom']} {user['nom']}</strong>
            {user['specialite']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bouton déconnexion
    _, col_logout = st.columns([5, 1])
    with col_logout:
        if st.button("🚪 Déconnexion"):
            st.session_state.logged_in    = False
            st.session_state.current_user = None
            st.rerun()

    if model is None:
        st.error("⚠️ Modèle introuvable. Lancez d'abord : `python src/train_model.py`")
        st.stop()

    st.markdown(
        f'<div style="text-align:right;margin-bottom:1rem;">'
        f'<span class="metric-chip">🤖 {type(model).__name__}</span>'
        f'<span class="metric-chip">✅ Modèle chargé</span></div>',
        unsafe_allow_html=True)

    # ── Layout ──
    col_main, col_right = st.columns([3, 2], gap="large")

    # ══ GAUCHE ══
    with col_main:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 Données du patient</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Variables numériques</div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            recipientage = st.number_input("Âge du receveur (ans)", 0.0, 25.0, 8.0, 0.5)
            cd34         = st.number_input("Dose CD34+ (×10⁶/kg)",  0.0, 50.0, 5.5, 0.1)
        with cb:
            rbodymass    = st.number_input("Masse corporelle (kg)",  5.0,150.0,28.0, 0.5)
            cd3          = st.number_input("Dose CD3+ (×10⁸/kg)",   0.0,100.0, 3.2, 0.1)

        st.markdown('<div class="section-title">Variables catégorielles</div>', unsafe_allow_html=True)
        cc, cd = st.columns(2)
        with cc:
            disease_str     = st.selectbox("Type de maladie",        list(DISEASE_MAP.keys()))
            gendermatch_str = st.selectbox("Compatibilité de genre", list(GENDERMATCH_MAP.keys()))
        with cd:
            relapse_str     = st.selectbox("Antécédent de rechute",  list(RELAPSE_MAP.keys()))
            hlamatch_str    = st.selectbox("Compatibilité HLA",      list(HLAMATCH_MAP.keys()))

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔬 Analyser la prédiction")
        st.markdown('<div class="disclaimer">⚠️ <strong>Avertissement clinique :</strong> '
                    'Cet outil est un support décisionnel. Il ne remplace pas le jugement médical.</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_btn:
            input_df = pd.DataFrame([[
                recipientage, rbodymass, cd34, cd3,
                DISEASE_MAP[disease_str], RELAPSE_MAP[relapse_str],
                GENDERMATCH_MAP[gendermatch_str], HLAMATCH_MAP[hlamatch_str],
            ]], columns=FEATURES)
            with st.spinner("Analyse en cours..."):
                pred, prob = predict(model, scaler, input_df)

            if pred == 1:
                st.markdown(f"""<div class="result-success">
                    <div class="result-title">✅ Survie probable</div>
                    <div class="result-subtitle">Le modèle prédit un succès de la greffe</div>
                    <div class="prob-container">
                        <div style="display:flex;justify-content:space-between;font-size:0.88rem;">
                            <span>Probabilité de survie</span><strong>{prob*100:.1f}%</strong></div>
                        <div class="prob-bar-bg"><div style="width:{prob*100:.1f}%;height:13px;
                            border-radius:20px;background:linear-gradient(90deg,#27ae60,#2ecc71);"></div></div>
                    </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-risk">
                    <div class="result-title">⚠️ Risque élevé</div>
                    <div class="result-subtitle">Le modèle prédit un risque d'échec de la greffe</div>
                    <div class="prob-container">
                        <div style="display:flex;justify-content:space-between;font-size:0.88rem;">
                            <span>Probabilité de survie</span><strong>{prob*100:.1f}%</strong></div>
                        <div class="prob-bar-bg"><div style="width:{prob*100:.1f}%;height:13px;
                            border-radius:20px;background:linear-gradient(90deg,#c0392b,#e74c3c);"></div></div>
                    </div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card"><div class="card-title">🔍 Explication SHAP</div>', unsafe_allow_html=True)
            sv = compute_shap(model, scaler, input_df)
            if sv:
                mx = max(abs(v) for v in sv.values()) or 1e-9
                ss = sorted(sv.items(), key=lambda x: abs(x[1]), reverse=True)
                st.markdown("".join(shap_bar(f,v,mx) for f,v in ss), unsafe_allow_html=True)
                st.markdown('<div style="margin-top:0.5rem;font-size:0.74rem;color:#666;">'
                            '<span style="color:#1a6eb5;">■</span> Favorise &nbsp;'
                            '<span style="color:#e74c3c;">■</span> Défavorise la survie</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5.5,3.2))
                lbls = [FEATURE_LABELS.get(f,f) for f,_ in ss]
                vals = [v for _,v in ss]
                clrs = ["#1a6eb5" if v>=0 else "#e74c3c" for v in vals]
                ax.barh(lbls[::-1], vals[::-1], color=clrs[::-1], height=0.5, edgecolor='none')
                ax.axvline(0, color='#aaa', linewidth=0.8, linestyle='--')
                ax.set_xlabel("Valeur SHAP", fontsize=8, color='#555')
                ax.set_title("Contribution de chaque variable", fontsize=9, color='#0d2b55', pad=5)
                ax.tick_params(labelsize=7)
                ax.spines[['top','right']].set_visible(False)
                ax.set_facecolor('#f7f8fc'); fig.patch.set_facecolor('#fff')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("📊 SHAP non disponible. `pip install shap`")
            st.markdown('</div>', unsafe_allow_html=True)

    # ══ DROITE ══
    with col_right:
        # Calendrier
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📅 Calendrier</div>', unsafe_allow_html=True)
        render_compact_calendar()
        st.markdown('</div>', unsafe_allow_html=True)

        # Widget RDV
        today = date.today()
        rdv_futurs = sorted(
            [a for a in st.session_state.appointments if a['date'] >= today],
            key=lambda x: (x['date'], x['heure'])
        )
        html_w  = '<div class="appt-widget">'
        html_w += (f'<div class="appt-widget-header">'
                   f'<span class="appt-widget-title">📋 Prochains rendez-vous</span>'
                   f'<span style="color:#b8c8e0;font-size:0.75rem;">{len(rdv_futurs)} à venir</span></div>')
        if rdv_futurs:
            for rdv in rdv_futurs[:3]:
                html_w += (f'<div class="appt-mini-item">'
                           f'<div class="appt-mini-date">{rdv["date"].strftime("%d/%m")}<br>'
                           f'<span style="font-size:0.68rem;font-weight:400;">{rdv["heure"]}</span></div>'
                           f'<div><div class="appt-mini-name">{rdv["nom"]}</div>'
                           f'<div class="appt-mini-motif">{rdv["motif"]}</div></div></div>')
        else:
            html_w += '<div class="appt-mini-empty">📭 Aucun rendez-vous à venir</div>'
        html_w += '</div>'
        st.markdown(html_w, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            if st.button("📂 Voir tous"):
                st.session_state.show_all_appts = not st.session_state.show_all_appts
                st.session_state.show_add_form  = False
                st.rerun()
        with b2:
            if st.button("➕ Ajouter"):
                st.session_state.show_add_form  = not st.session_state.show_add_form
                st.session_state.show_all_appts = False
                st.rerun()

        if st.session_state.show_all_appts:
            st.markdown('<div class="card" style="margin-top:0.7rem;">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🗓️ Tous les rendez-vous</div>', unsafe_allow_html=True)
            all_s = sorted(st.session_state.appointments, key=lambda x:(x['date'],x['heure']))
            if all_s:
                for i, rdv in enumerate(all_s):
                    ci, cd2 = st.columns([5,1])
                    with ci:
                        st.markdown(f'<div class="appt-full-item">'
                                    f'<div class="appt-full-date">{rdv["date"].strftime("%d/%m/%Y")} à {rdv["heure"]}</div>'
                                    f'<div class="appt-full-name">{rdv["nom"]}</div>'
                                    f'<div class="appt-full-motif">{rdv["motif"]}</div></div>',
                                    unsafe_allow_html=True)
                    with cd2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🗑️", key=f"del_{i}"):
                            st.session_state.appointments.pop(i); st.rerun()
            else:
                st.markdown('<div style="text-align:center;color:#aaa;padding:0.8rem;font-size:0.85rem;">📭 Aucun rendez-vous</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.show_add_form:
            st.markdown('<div class="card" style="margin-top:0.7rem;">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">➕ Nouveau rendez-vous</div>', unsafe_allow_html=True)
            nom_p    = st.text_input("Nom du patient", placeholder="Ex: Ahmed Benali", key="nom_in")
            date_rdv = st.date_input("Date", value=today, min_value=today, key="date_in")
            heure_r  = st.selectbox("Heure", HEURES, index=4, key="heure_in")
            motif_r  = st.selectbox("Motif", MOTIFS, key="motif_in")
            if st.button("✅ Confirmer"):
                if nom_p.strip():
                    st.session_state.appointments.append({'date':date_rdv,'heure':heure_r,'nom':nom_p.strip(),'motif':motif_r})
                    st.success(f"✅ RDV ajouté : {nom_p} le {date_rdv.strftime('%d/%m/%Y')} à {heure_r}")
                    st.session_state.show_add_form = False; st.rerun()
                else:
                    st.warning("⚠️ Veuillez saisir le nom du patient.")
            st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown('<div style="text-align:center;font-size:0.74rem;color:#aaa;">'
                'PediaBMT Decision Support · Centrale Casablanca · Coding Week 2026</div>',
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  ROUTING
# ═══════════════════════════════════════════════════════
if st.session_state.logged_in:
    show_main_app()
else:
    show_auth_page()