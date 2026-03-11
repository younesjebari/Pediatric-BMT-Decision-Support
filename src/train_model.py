import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # <--- 1. Import bien placé ici
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys

# On ajoute le dossier parent pour pouvoir importer data_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import handle_missing_values

def run_training_pipeline():
    print("🚀 Démarrage de la pipeline d'entraînement...")

    # 1. Chargement des données brutes
    data_path = 'data/bone-marrow.arff'
    raw_data, _ = arff.loadarff(data_path)
    df = pd.DataFrame(raw_data)
    
    # Décodage des bytes (propre aux fichiers .arff)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')

    # 2. Nettoyage via ton script data_processing.py
    df_clean = handle_missing_values(df)

    # 3. Encodage (Texte -> Chiffres)
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype('category').cat.codes

    # 4. Préparation X et y
    # On vérifie que survival_time est bien exclu pour éviter la triche (Leakage)
    X = df_clean.drop(['survival_status', 'survival_time'], axis=1, errors='ignore')
    y = df_clean['survival_status']

    # 5. Séparation des données (80% train / 20% test)
    # C'est ce qui permet d'avoir un score réel et non 100%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 6. Équilibrage SMOTE (uniquement sur le train)
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # 7. Entraînement du Champion (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)

    # 8. Sauvegarde du modèle ET des données de test
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, 'models/final_model.joblib')
    joblib.dump((X_test, y_test), 'models/test_data.joblib') # <--- 2. Très important pour l'évaluation !
    
    print(f"✅ Succès : Modèle et {len(X_test)} cas de test sauvegardés.")

if __name__ == "__main__":
    run_training_pipeline()