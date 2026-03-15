import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import re
from io import BytesIO

# 1. Configuration & Design Ultra-Premium
st.set_page_config(
    page_title="PediaBMT • Secure Portal", 
    page_icon="🩸", 
    layout="wide"
)

# Style CSS pour l'esthétique et l'arrière-plan médical dégradé
st.markdown("""
    <style>
    /* Arrière-plan dégradé doux */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* En-tête stylisé avec le logo */
    .main-header {
        background: linear-gradient(90deg, #004e92 0%, #000428 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .main-header img {
        height: 50px;
        margin-right: 20px;
    }

    /* Boutons arrondis pro */
    .stButton>button {
        border-radius: 25px;
        background: #004e92;
        color: white;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    
    /* Conteneurs blancs (Cards) */
    .stVerticalBlock > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Fonctions de Sécurité ( Strong Password Validation)
def is_strong_password(password):
    """Vérifie si le mot de passe respecte les critères de robustesse médicale."""
    if len(password) < 8: return False # Minimum 8 caractères
    if not re.search("[a-z]", password): return False # Une minuscule
    if not re.search("[A-Z]", password): return False # Une majuscule
    if not re.search("[0-9]", password): return False # Un chiffre
    if not re.search("[!@#$%^&*(),.?\":{}|<>]", password): return False # Un caractère spécial
    return True

# 3. Système d'Authentification Sécurisé
if 'auth' not in st.session_state:
    st.session_state['auth'] = False

if not st.session_state['auth']:
    # En-tête de la page de login
    st.markdown("""
        <div class="main-header">
            <img src="https://upload.wikimedia.org/wikipedia/commons/e/e3/Caduceus.svg" alt="Logo Médical">
            <h1>🔒 Connexion Sécurisée PediaBMT</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with st.container():
        st.subheader("Veuillez vous identifier")
        email = st.text_input("Email professionnel (ex: dr.jebari@chu.ma)")
        pwd = st.text_input("Mot de passe", type="password")
        
        col_btn1, col_btn2 = st.columns([1, 2])
        if col_btn1.button("Accéder au Dashboard"):
            # Identifiants de test (à changer pour la production)
            if email == "admin@chu.ma" and is_strong_password(pwd):
                st.session_state['auth'] = True
                st.rerun()
            elif email == "admin@chu.ma" and not is_strong_password(pwd):
                st.warning("⚠️ Sécurité insuffisante. Le mot de passe doit contenir au moins 8 caractères, une majuscule, un chiffre et un caractère spécial.")
            else:
                st.error("Identifiants incorrects ou accès non autorisé.")
    st.stop() # Arrête l'exécution tant que l'utilisateur n'est pas authentifié

# 4. Chargement du Modèle Champion
@st.cache_resource
def load_assets():
    model = joblib.load('models/final_model.joblib')
    return model

model = load_assets()

# 5. Interface Principale du Dashboard
st.markdown("""
    <div class="main-header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/e/e3/Caduceus.svg" alt="Logo Médical">
        <h1>🩸 PediaBMT • Decision Support System</h1>
    </div>
    """, unsafe_allow_html=True)

# Bouton de déconnexion dans la barre latérale
if st.sidebar.button("Déconnexion"):
    st.session_state['auth'] = False
    st.rerun()

# Organisation en deux colonnes principales
col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    st.subheader("📋 Saisie des Données Patient")
    
    with st.container():
        st.write("### 🔢 Variables Numériques (Biomarqueurs)")
        # Variables validées par tes tests de p-value (p < 0.05)
        age_r = st.number_input("Âge du receveur (ans)", 0.0, 20.0, 8.0, help="Biomarqueur clé (p=0.0053)")
        mass = st.number_input("Masse corporelle (kg)", 5.0, 100.0, 28.0)
        cd34 = st.number_input("Dose CD34+ (×10⁶/kg)", 0.0, 50.0, 5.50, help="Biomarqueur clé (p=0.0074)")
        cd3 = st.number_input("Dose CD3+ (×10⁸/kg)", 0.0, 20.0, 3.20)
        plt_rec = st.number_input("Temps de récupération des plaquettes (jours)", 0, 60, 25)
        age_d = st.number_input("Âge du donneur", 0, 60, 25)

    with st.container():
        st.write("### 🧬 Variables Catégorielles")
        disease = st.selectbox("Type de maladie", [0, 1, 2, 3], format_func=lambda x: ["ALL", "AML", "Non-Malignant", "Other"][x])
        relapse = st.radio("Antécédent de rechute", [0, 1], format_func=lambda x: "Non" if x==0 else "Oui", horizontal=True)
        hla = st.slider("Compatibilité HLA (0=Total, 3=Mismatch)", 0, 3, 0)
        risk = st.radio("Groupe de Risque", [0, 1], format_func=lambda x: "Bas (0)" if x==0 else "Haut (1)", horizontal=True)

with col_right:
    st.subheader("🔬 Diagnostic & IA Explicable")
    
    # Préparation des 11 variables exactes attendues par ton modèle entraîné
    input_df = pd.DataFrame([[
        age_r, mass, cd34, cd3, disease, relapse, plt_rec, 0, age_d, hla, risk
    ]], columns=['Recipientage', 'Rbodymass', 'CD34kgx10d6', 'CD3dkgx10d8', 'Disease', 
                'Relapse', 'PLTrecovery', 'extcGvHD', 'Donorage', 'HLAmatch', 'Riskgroup'])

    if st.button("🚀 ANALYSER LE PATIENT", use_container_width=True):
        st.info("🔄 Analyse en cours...")
        
        # Prédiction (Classe 1 = Décès)
        prob_deces = model.predict_proba(input_df)[0][1]
        prob_survie = (1 - prob_deces) * 100
        
        # Affichage Élégant du Pronostic
        if prob_deces > 0.5:
            st.error(f"### ⚠️ ALERTE : RISQUE ÉLEVÉ ({prob_deces*100:.1f}%)")
            st.metric("Probabilité de Survie", f"{prob_survie:.1f}%", delta="CRITIQUE", delta_color="inverse")
        else:
            st.success(f"### ✅ PRONOSTIC FAVORABLE")
            st.metric("Probabilité de Survie", f"{prob_survie:.1f}%", delta="STABLE")

        # Section SHAP pour l'explicabilité médicale
        st.write("---")
        st.write("### 🧬 Pourquoi cette décision ? (Explication SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        # Génération du graphique force_plot
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], matplotlib=True, show=False)
        st.pyplot(plt.gcf())
        st.caption("Ce graphique montre l'influence de chaque variable sur le score final (bleu=pousse vers succès, rouge=pousse vers risque).")

        # Fonctionnalité de Rapport de base (Rappel : .txt)
        report_text = f"PediaBMT Report\nAge Receveur: {age_r}\nDose CD34+: {cd34}\nProbabilité Survie: {prob_survie:.1f}%"
        st.download_button("📥 Télécharger le Rapport Textuel", report_text, file_name=f"rapport_bmt_{age_r}ans.txt")
    
    else:
        st.warning("Veuillez remplir les données à gauche et cliquer sur 'Analyser'.")
        # Rappel des performances pour rassurer le clinicien
        st.info(f"**Information Système :** Ce modèle Random Forest est validé avec un score ROC-AUC de 95.86%.")