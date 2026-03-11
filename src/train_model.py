import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
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
    X = df_clean.drop(['survival_status', 'survival_time'], axis=1)
    y = df_clean['survival_status']

    # 5. Équilibrage SMOTE
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X, y)

    # 6. Entraînement du Champion (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)

    # 7. Création du dossier models s'il n'existe pas et sauvegarde
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, 'models/final_model.joblib')
    print("✅ Succès : Le modèle est sauvegardé dans 'models/final_model.joblib'")

if __name__ == "__main__":
    run_training_pipeline()