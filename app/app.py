import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# ==============================================================================
# 1. CONFIGURATION & DESIGN
# ==============================================================================
st.set_page_config(
    page_title="PediaBMT • Expert Portal", 
    page_icon="🩸", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour un design moderne et vibrant
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        background: linear-gradient(90deg, #004e92 0%, #000428 100%);
        color: white; 
        padding: 2rem; 
        border-radius: 15px; 
        text-align: center; 
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #004e92;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 2px solid #004e92;
    }
    .stButton>button:hover {
        background-color: #ffffff;
        color: #004e92;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    .prediction-metric {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
    }
    .shap-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #555;
        font-size: 0.9rem;
    }
    .sidebar-content {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. FONCTIONS UTILITAIRES & INITIALISATION
# ==============================================================================
def validate_email(email):
    """Valide le format d'un email."""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def validate_password(p):
    """Valide la robustesse d'un mot de passe et retourne un dictionnaire de checks."""
    return {
        "8+ caractères": len(p) >= 8,
        "Majuscule": any(c.isupper() for c in p),
        "Chiffre": any(c.isdigit() for c in p),
        "Caractère spécial": bool(re.search(r"[!@#$%^&*(),.?\":{}|<>]", p))
    }

# Initialisation de la "base de données" en session
if 'user_db' not in st.session_state:
    st.session_state['user_db'] = {"aroua.elachhab@centrale-casablanca.ma": "AdminBMT2026!"}
if 'pending_accounts' not in st.session_state:
    st.session_state['pending_accounts'] = {}
if 'auth' not in st.session_state:
    st.session_state['auth'] = False
# --- NOUVEAUX ÉTATS POUR LA GESTION DES PATIENTS ---
if 'patient_records' not in st.session_state:
    st.session_state['patient_records'] = {}
if 'patient_info_submitted' not in st.session_state:
    st.session_state['patient_info_submitted'] = False
if 'current_patient_id' not in st.session_state:
    st.session_state['current_patient_id'] = None
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False


# ==============================================================================
# 3. SYSTÈME D'AUTHENTIFICATION
# ==============================================================================
if not st.session_state['auth']:
    st.markdown('<div class="main-header"><h1>💉 PediaBMT Secure Gateway</h1></div>', unsafe_allow_html=True)
    tab_log, tab_reg = st.tabs(["🔑 Connexion", "📝 Créer un compte"])

    with tab_log:
        with st.form("login_form"):
            u_email = st.text_input("Email professionnel", key="log_mail")
            u_pwd = st.text_input("Mot de passe", type="password", key="log_pwd")
            submitted_login = st.form_submit_button("Accéder au Dashboard")
            if submitted_login:
                if u_email in st.session_state['user_db'] and st.session_state['user_db'][u_email] == u_pwd:
                    st.session_state['auth'] = True
                    st.session_state['user'] = u_email
                    st.rerun()
                else:
                    st.error("Accès refusé. Identifiants incorrects ou compte non validé.")

    with tab_reg:
        with st.form("register_form"):
            reg_email = st.text_input("Email souhaité")
            reg_pwd = st.text_input("Mot de passe (robuste)", type="password", help="Doit contenir 8+ caractères, une majuscule, un chiffre et un symbole.")
            
            if reg_email and not validate_email(reg_email):
                st.error("❌ Format d'email invalide (ex: nom@domaine.com)")
            
            if reg_pwd:
                checks = validate_password(reg_pwd)
                cols = st.columns(2)
                for i, (label, status) in enumerate(checks.items()):
                    with cols[i % 2]:
                        color = "green" if status else "red"
                        st.markdown(f"<span style='color:{color}'>{'✅' if status else '❌'} {label}</span>", unsafe_allow_html=True)
            
            submitted_reg = st.form_submit_button("Envoyer la demande à l'admin")
            if submitted_reg:
                if validate_email(reg_email) and all(validate_password(reg_pwd).values()):
                    st.session_state['pending_accounts'][reg_email] = reg_pwd
                    st.success("✅ Demande envoyée. L'administrateur (Aroua Elachhab) doit valider votre accès.")
                else:
                    st.warning("⚠️ Veuillez corriger les erreurs avant d'envoyer.")
    st.stop()


# ==============================================================================
# 4. INTERFACE ADMIN & DASHBOARD
# ==============================================================================
# --- Message de bienvenue conditionnel ---
if st.session_state['user'] == "aroua.elachhab@centrale-casablanca.ma":
    welcome_message = "Bienvenue, Admin"
else:
    welcome_message = f"Bienvenue, {st.session_state['user']}"

st.markdown(f'<div class="main-header"><h1>🔬 Dashboard d\'Aide à la Décision</h1><p>{welcome_message}</p></div>', unsafe_allow_html=True)


# --- Barre Latérale pour Déconnexion et Admin ---
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    if st.button("🚪 Se Déconnecter"):
        st.session_state['auth'] = False
        st.session_state['patient_info_submitted'] = False
        st.session_state['analysis_done'] = False
        st.rerun()
    
    st.markdown("---")
    # Panneau d'administration
    if st.session_state['user'] == "aroua.elachhab@centrale-casablanca.ma":
        st.subheader("🛠️ Panneau d'Administration")
        
        # --- Gestion des comptes ---
        if st.session_state['pending_accounts']:
            st.write("Comptes en attente de validation :")
            for acc, p in list(st.session_state['pending_accounts'].items()):
                if st.button(f"✅ Valider {acc}", key=f"val_{acc}"):
                    st.session_state['user_db'][acc] = p
                    del st.session_state['pending_accounts'][acc]
                    st.success(f"Le compte {acc} a été validé.")
                    st.rerun()
        else:
            st.info("Aucune demande en attente.")
        
        # --- Consultation des dossiers patients ---
        st.markdown("---")
        st.subheader("📁 Dossiers Patients Sauvegardés")
        if st.session_state['patient_records']:
            selected_patient_id = st.selectbox(
                "Sélectionner un dossier à consulter :", 
                options=list(st.session_state['patient_records'].keys()),
                format_func=lambda pid: f"{st.session_state['patient_records'][pid]['prenom']} {st.session_state['patient_records'][pid]['nom']} (ID: {pid})"
            )
            if selected_patient_id:
                patient_record = st.session_state['patient_records'][selected_patient_id]
                st.json(patient_record) # Affiche le dossier complet de manière lisible
        else:
            st.info("Aucun dossier patient enregistré.")
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# 5. ÉTAPE D'IDENTIFICATION DU PATIENT
# ==============================================================================
if not st.session_state['patient_info_submitted']:
    st.subheader("👤 Identification du Patient")
    with st.form("patient_id_form"):
        st.markdown("**Veuillez entrer les informations du receveur. Tous les champs marqués d'un astérisque (*) sont obligatoires.**")
        
        patient_id = st.text_input("Identifiant Unique du Patient (Carte ID) *", key="pid_input")
        patient_last_name = st.text_input("Nom du Patient *", key="lname_input")
        patient_first_name = st.text_input("Prénom du Patient *", key="fname_input")
        patient_phone = st.text_input("Numéro de Téléphone (Optionnel)", key="phone_input")

        submitted_patient = st.form_submit_button("Valider et Commencer l'Analyse")
        
        if submitted_patient:
            if not patient_id or not patient_last_name or not patient_first_name:
                st.error("Veuillez remplir tous les champs obligatoires (ID, Nom, Prénom).")
            else:
                # Stocker les infos du patient
                patient_data = {
                    'id': patient_id,
                    'nom': patient_last_name,
                    'prenom': patient_first_name,
                    'telephone': patient_phone,
                    'diagnostic_results': None
                }
                st.session_state['patient_records'][patient_id] = patient_data
                st.session_state['current_patient_id'] = patient_id
                st.session_state['patient_info_submitted'] = True
                st.success(f"Patient {patient_first_name} {patient_last_name} identifié avec succès.")
                st.rerun()
    st.stop() # On arrête ici tant que le patient n'est pas identifié


# ==============================================================================
# 6. DIAGNOSTIC & ANALYSE SHAP
# ==============================================================================
# Afficher les infos du patient actuel
current_patient = st.session_state['patient_records'][st.session_state['current_patient_id']]
st.info(f"Analyse en cours pour : **{current_patient['prenom']} {current_patient['nom']}** (ID: {current_patient['id']})")

if st.button("➕ Changer de Patient"):
    st.session_state['patient_info_submitted'] = False
    st.session_state['analysis_done'] = False
    st.rerun()

MODEL_PATH = '../models/final_model.joblib'

if not os.path.exists(MODEL_PATH):
    st.error(f"🚨 Le modèle est introuvable à l'emplacement : `{MODEL_PATH}`.")
    st.stop()

model = joblib.load(MODEL_PATH)

col_in, col_out = st.columns([1, 1.5], gap="large")

with col_in:
    st.subheader("📋 Saisie des Données Cliniques")
    with st.form("prediction_form"):
        # --- Caractéristiques du Receveur ---
        st.markdown("**Caractéristiques du Receveur**")
        age_r = st.number_input("Âge Receveur (années)", min_value=0.0, max_value=30.0, value=8.0, step=0.1)
        body_mass = st.number_input("Masse Corporelle (kg)", min_value=5.0, max_value=100.0, value=30.0, step=0.1)
        disease = st.selectbox("Type de Maladie", options=[0, 1, 2, 3], index=0, format_func=lambda x: ["ALL", "AML", "Non-Maligne", "Autre"][x])
        relapse = st.radio("Rechute avant greffe ?", options=[0, 1], index=0, format_func=lambda x: "Non" if x==0 else "Oui", horizontal=True)
        risk_group = st.radio("Groupe de Risque", options=[0, 1], index=0, format_func=lambda x: "Standard" if x==0 else "Élevé", horizontal=True)

        st.markdown("---")
        # --- Caractéristiques de la Greffe ---
        st.markdown("**Caractéristiques de la Greffe**")
        cd34 = st.number_input("Dose CD34+ (x10⁶/kg)", min_value=0.0, max_value=50.0, value=5.5, step=0.1)
        cd3 = st.number_input("Dose CD3+ (x10⁸/kg)", min_value=0.0, max_value=20.0, value=3.2, step=0.1)
        plt_recovery = st.number_input("Récupération Plaquettaire (jours)", min_value=5, max_value=200, value=25, step=1)
        gvhd = st.radio("GVHD Chronique ?", options=[0, 1], index=0, format_func=lambda x: "Non" if x==0 else "Oui", horizontal=True)

        st.markdown("---")
        # --- Caractéristiques du Donneur ---
        st.markdown("**Caractéristiques du Donneur**")
        donor_age = st.number_input("Âge Donneur (années)", min_value=18.0, max_value=60.0, value=25.0, step=0.1)
        hla_match = st.selectbox("Compatibilité HLA", options=[3, 4, 5, 6], index=0, format_func=lambda x: f"{x}/6")

        submitted = st.form_submit_button("🚀 LANCER L'ANALYSE")

        if submitted:
            st.session_state['analysis_done'] = True
            
            correct_feature_order = [
                'CD3dkgx10d8', 'CD34kgx10d6', 'Rbodymass', 'Recipientage', 
                'PLTrecovery', 'Disease', 'Relapse', 'extcGvHD', 
                'Donorage', 'HLAmatch', 'Riskgroup'
            ]

            input_data = {
                'Recipientage': age_r, 'Rbodymass': body_mass, 'CD34kgx10d6': cd34, 
                'CD3dkgx10d8': cd3, 'Disease': disease, 'Relapse': relapse, 
                'PLTrecovery': plt_recovery, 'extcGvHD': gvhd, 'Donorage': donor_age, 
                'HLAmatch': hla_match, 'Riskgroup': risk_group
            }
            input_df = pd.DataFrame([input_data])
            input_df = input_df[correct_feature_order]
            st.session_state['input_df'] = input_df
            
            proba_relapse = model.predict_proba(st.session_state['input_df'])[0][1]
            st.session_state['proba_survival'] = (1 - proba_relapse) * 100
            
            # --- NOUVEAU : Sauvegarder les résultats dans le dossier du patient ---
            diagnostic_result = {
                'survival_probability': st.session_state['proba_survival'],
                'input_data': input_df.to_dict('records')[0]
            }
            st.session_state['patient_records'][st.session_state['current_patient_id']]['diagnostic_results'] = diagnostic_result
            
            # Analyse SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(st.session_state['input_df'])
            
            if isinstance(shap_values, list):
                shap_values_class_1 = shap_values[1]
                expected_value_class_1 = explainer.expected_value[1]
            else:
                shap_values_class_1 = shap_values
                expected_value_class_1 = explainer.expected_value

            plt.figure()
            shap.force_plot(expected_value_class_1, shap_values_class_1[0], st.session_state['input_df'].iloc[0], matplotlib=True, show=False)
            st.session_state['shap_fig'] = plt.gcf()
            plt.close()

with col_out:
    st.subheader("📊 Résultats de l'Analyse par IA")
    
    if st.session_state['analysis_done']:
        st.markdown('<div class="prediction-metric">', unsafe_allow_html=True)
        st.metric(
            label="Probabilité de Survie à 2 ans", 
            value=f"{st.session_state['proba_survival']:.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🧠 Explicabilité de la Prédiction (SHAP)")
        st.info("Ce graphique montre comment chaque caractéristique a influencé le modèle. Les caractéristiques rouges poussent vers une rechute, les bleues vers la survie.")
        
        st.markdown('<div class="shap-container">', unsafe_allow_html=True)
        st.pyplot(st.session_state['shap_fig'])
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Veuillez remplir les informations cliniques et cliquer sur 'LANCER L'ANALYSE' pour voir les résultats.")

# Pied de page
st.markdown('<div class="footer">PediaBMT Expert Portal © 2024 | Powered by AI & SHAP</div>', unsafe_allow_html=True)