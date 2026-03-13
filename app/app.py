import streamlit as st

st.title(" Prédiction Greffe de Moelle Osseuse")

st.subheader(" Données du Patient")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Receveur")
    recipientage = st.number_input(
        "Âge du receveur (ans)",
        min_value=0.0, max_value=25.0,
        value=8.0, step=0.5
    )
    rbodymass = st.number_input(
        "Masse corporelle (kg)",
        min_value=5.0, max_value=100.0,
        value=25.0, step=0.5
    )
    disease = st.selectbox(
        "Type de maladie",
        options=['ALL', 'AML', 'chronic', 
                 'nonmalignant', 'lymphoma']
    )
    relapse = st.selectbox(
        "Rechute ?",
        options=['No', 'Yes']
    )
    ext_cgvhd = st.selectbox(
        "GvHD chronique étendue ?",
        options=['No', 'Yes']
    )

with col2:
    st.markdown("###  Traitement")
    cd34 = st.number_input(
        "CD34+ dose (×10⁶/kg)",
        min_value=0.0, max_value=30.0,
        value=5.0, step=0.1
    )
    cd3 = st.number_input(
        "CD3+ dose (×10⁸/kg)",
        min_value=0.0, max_value=10.0,
        value=2.0, step=0.1
    )
    donorage = st.number_input(
        "Âge du donneur (ans)",
        min_value=18.0, max_value=80.0,
        value=35.0, step=1.0
    )
    hla_match = st.selectbox(
        "Compatibilité HLA",
        options=['10/10', '9/10', '8/10', '7/10']
    )
    risk_group = st.selectbox(
        "Groupe de risque",
        options=['low risk', 'high risk']
    )

st.markdown("###  Récupération")
plt_recovery = st.selectbox(
    "Récupération plaquettes ?",
    options=['Yes', 'No']
)

# Bouton prédiction
if st.button(" ANALYSER", use_container_width=True):
    
    # Créer le dictionnaire des features
    input_data = {
        'CD3dkgx10d8':  cd3,
        'CD34kgx10d6':  cd34,
        'Rbodymass':    rbodymass,
        'Recipientage': recipientage,
        'PLTrecovery':  1 if plt_recovery == 'Yes' else 0,
        'Disease':      disease,
        'Relapse':      1 if relapse == 'Yes' else 0,
        'extcGvHD':     1 if ext_cgvhd == 'Yes' else 0,
        'Donorage':     donorage,
        'HLAmatch':     hla_match,
        'Riskgroup':    risk_group,
    }
    
    st.success("Données enregistrées ")
    st.json(input_data)