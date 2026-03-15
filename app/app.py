import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import re

# 1. Configuration & Design
st.set_page_config(page_title="PediaBMT • Expert Portal", page_icon="🩸", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .main-header {
        background: linear-gradient(90deg, #004e92 0%, #000428 100%);
        color: white; padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 25px;
    }
    .status-box { padding: 10px; border-radius: 10px; margin-top: 5px; font-size: 0.85rem; }
    </style>
    """, unsafe_allow_html=True)

# 2. Logique de Validation & Base de Données
if 'user_db' not in st.session_state:
    st.session_state['user_db'] = {"aroua.elachhab@centrale-casablanca.ma": "AdminBMT2026!"}
if 'pending_accounts' not in st.session_state:
    st.session_state['pending_accounts'] = {}

def validate_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def validate_password(p):
    checks = {
        "8+ carac": len(p) >= 8,
        "Majuscule": any(c.isupper() for c in p),
        "Chiffre": any(c.isdigit() for c in p),
        "Spécial": bool(re.search(r"[!@#$%^&*]", p))
    }
    return checks

# 3. Système d'Authentification
if 'auth' not in st.session_state: st.session_state['auth'] = False

if not st.session_state['auth']:
    st.markdown('<div class="main-header"><h1>💉 PediaBMT Secure Gateway</h1></div>', unsafe_allow_html=True)
    tab_log, tab_reg = st.tabs(["🔑 Connexion", "📝 Créer un compte"])

    with tab_log:
        u_email = st.text_input("Email professionnel", key="log_mail")
        u_pwd = st.text_input("Mot de passe", type="password", key="log_pwd")
        if st.button("Accéder au Dashboard"):
            if u_email in st.session_state['user_db'] and st.session_state['user_db'][u_email] == u_pwd:
                st.session_state['auth'] = True
                st.session_state['user'] = u_email
                st.rerun()
            else: st.error("Accès refusé. Identifiants incorrects ou compte non validé.")

    with tab_reg:
        reg_email = st.text_input("Email souhaité")
        if reg_email and not validate_email(reg_email):
            st.error("❌ Format d'email invalide (ex: nom@domaine.com)")
        
        reg_pwd = st.text_input("Mot de passe (Strong)", type="password")
        if reg_pwd:
            checks = validate_password(reg_pwd)
            for label, status in checks.items():
                color = "green" if status else "red"
                st.markdown(f"<span style='color:{color}'>{'✅' if status else '❌'} {label}</span>", unsafe_allow_html=True)
        
        if st.button("Envoyer la demande à l'admin"):
            if validate_email(reg_email) and all(validate_password(reg_pwd).values()):
                st.session_state['pending_accounts'][reg_email] = reg_pwd
                st.success("Demande envoyée. L'administrateur (Aroua Elachhab) doit valider votre accès.")
            else: st.warning("Veuillez corriger les erreurs avant d'envoyer.")
    st.stop()

# 4. Interface Admin & Dashboard
st.markdown(f'<div class="main-header"><h1>🔬 Dashboard d\'Aide à la Décision</h1></div>', unsafe_allow_html=True)

# Section Admin visible uniquement par le mail principal
if st.session_state['user'] == "aroua.elachhab@centrale-casablanca.ma":
    with st.expander("🛠️ Panneau d'Administration (Gestion des comptes)"):
        if st.session_state['pending_accounts']:
            for acc, p in list(st.session_state['pending_accounts'].items()):
                col_a, col_b = st.columns([3, 1])
                col_a.write(f"Demande de : **{acc}**")
                if col_b.button(f"Valider {acc}"):
                    st.session_state['user_db'][acc] = p
                    del st.session_state['pending_accounts'][acc]
                    st.rerun()
        else: st.write("Aucune demande en attente.")

# 5. Diagnostic & SHAP
model = joblib.load('models/final_model.joblib')
col_in, col_out = st.columns([1, 1.5], gap="large")

with col_in:
    st.subheader("📋 Saisie Patient")
    age_r = st.number_input("Âge Receveur", 0.0, 20.0, 8.0)
    cd34 = st.number_input("Dose CD34+", 0.0, 50.0, 5.5)
    disease = st.selectbox("Maladie", [0,1,2,3], format_func=lambda x: ["ALL", "AML", "Non-Mal", "Autre"][x])
    relapse = st.radio("Rechute", [0, 1], format_func=lambda x: "Non" if x==0 else "Oui")

with col_out:
    st.subheader("📊 Analyse IA Explicable")
    input_df = pd.DataFrame([[age_r, 30.0, cd34, 3.2, disease, relapse, 25, 0, 25, 0, 0]], 
                            columns=['Recipientage', 'Rbodymass', 'CD34kgx10d6', 'CD3dkgx10d8', 'Disease', 
                                    'Relapse', 'PLTrecovery', 'extcGvHD', 'Donorage', 'HLAmatch', 'Riskgroup'])

    if st.button("🚀 ANALYSER"):
        prob = model.predict_proba(input_df)[0][1]
        st.metric("Probabilité de Survie", f"{(1-prob)*100:.1f}%")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], matplotlib=True, show=False)
        st.pyplot(plt.gcf())