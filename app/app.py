import streamlit as st
import joblib
import pandas as pd
import os

# --- INITIALISATION DE LA PAGE ET DE LA MÉMOIRE ---
st.set_page_config(page_title="Prédiction Greffe", layout="centered")

if 'connecte' not in st.session_state:
    st.session_state['connecte'] = False

# ==========================================
# 🚪 PARTIE 1 : L'INTERFACE DE LOGIN
# ==========================================
if not st.session_state['connecte']:
    st.title("🔒 Accès Sécurisé - Espace Médical")
    st.markdown("Veuillez vous identifier pour accéder à l'outil de prédiction.")
    
    identifiant = st.text_input("Identifiant Médecin")
    mot_de_passe = st.text_input("Mot de passe", type="password")
    
    if st.button("Se connecter"):
        if identifiant == "medecin" and mot_de_passe == "centrale":
            st.session_state['connecte'] = True
            st.rerun()
        else:
            st.error("❌ Identifiant ou mot de passe incorrect.")

# ==========================================
# 🩺 PARTIE 2 : L'INTERFACE PRINCIPALE 
# ==========================================
else:
    if st.button("Déconnexion"):
        st.session_state['connecte'] = False
        st.rerun()

    st.title("🩺 Prédiction Greffe de Moelle Osseuse")
    st.subheader(" Données du Patient")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Receveur")
        recipientage = st.number_input("Âge du receveur (ans)", min_value=0.0, max_value=25.0, value=8.0, step=0.5)
        rbodymass = st.number_input("Masse corporelle (kg)", min_value=5.0, max_value=100.0, value=25.0, step=0.5)
        disease = st.selectbox("Type de maladie", options=['ALL', 'AML', 'chronic', 'nonmalignant', 'lymphoma'])
        relapse = st.selectbox("Rechute ?", options=['No', 'Yes'])

    with col2:
        st.markdown("###  Traitement")
        cd34 = st.number_input("CD34+ dose (×10⁶/kg)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)
        cd3 = st.number_input("CD3+ dose (×10⁸/kg)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

    # Bouton prédiction
    st.markdown("---")
    if st.button(" ANALYSER", use_container_width=True):
        
        disease_map = {'ALL': 0, 'AML': 1, 'chronic': 2, 'lymphoma': 3, 'nonmalignant': 4}
        relapse_val = 1 if relapse == 'Yes' else 0

        # Données brutes saisies par le médecin
        input_data = pd.DataFrame([{
            'Recipientage': recipientage,
            'Rbodymass': rbodymass,
            'Disease': disease_map[disease],
            'Relapse': relapse_val,
            'CD34kgx10d6': cd34,
            'CD3dkgx10d8': cd3
        }])

        try:
            # --- ⚠️ LES 3 NOUVELLES LIGNES SONT ICI ⚠️ ---
            # 1. On charge le modèle ET le scaler
            model = joblib.load('models/final_model.joblib')
            scaler = joblib.load('models/scaler.joblib')
            
            # 2. On transforme les données brutes avec le scaler
            input_scaled = scaler.transform(input_data)

            # 3. L'IA fait sa prédiction sur les données transformées
            prediction = model.predict(input_scaled)
            proba = model.predict_proba(input_scaled)[0][1] 

            st.markdown("### 📊 Résultat de l'analyse")
            if prediction[0] == 1:
                st.success(f"✅ **Succès de la greffe estimé.** (Niveau de confiance : {proba*100:.1f}%)")
            else:
                st.error(f"⚠️ **Risque d'échec élevé.** (Probabilité de succès : {proba*100:.1f}%)")
                
        except FileNotFoundError:
            st.error("❌ Modèle ou Scaler introuvable. Assurez-vous d'avoir exécuté `python src/train_model.py`.")