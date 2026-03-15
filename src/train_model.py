import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os

# On importe la logique de nettoyage centralisée
from data_processing import clean_data, encode_categories, IMPORTANT_FEATURES

def train():
    print("🚀 Démarrage de l'entraînement (Pipeline Machine Learning)...")

    # 1. Chargement des données
    try:
        raw_data, _ = arff.loadarff('../data/bone-marrow.arff')
        df = pd.DataFrame(raw_data)
    except Exception as e:
        print(f"❌ Erreur : Fichier de données introuvable. {e}")
        return

    # 2. Prétraitement (Nettoyage et Encodage)
    df = clean_data(df)
    df = encode_categories(df)
    
    # X = Nos variables validées par l'EDA, y = survie/décès
    X = df[IMPORTANT_FEATURES]
    y = pd.to_numeric(df['survival_status']).astype(int)

    print(f"📊 Analyse de {len(df)} patients sur {len(IMPORTANT_FEATURES)} variables.")

    # 3. Séparation Train (80%) / Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Standardisation (Nécessaire pour SVM et la stabilité des modèles)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Équilibrage SMOTE
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

    # 6. Comparaison des Modèles
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "SVM": SVC(probability=True, random_state=42)
    }

    best_model = None
    best_score = 0
    best_name = ""

    print("\n📈 Évaluation des performances :")
    for name, m in models.items():
        m.fit(X_res, y_res)
        y_proba = m.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"-> {name:15s} | ROC-AUC : {auc:.4f}")
        
        if auc > best_score:
            best_score = auc
            best_model = m
            best_name = name

    # 7. Sauvegarde du trio de production
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_model, '../models/final_model.joblib')
    joblib.dump(scaler, '../models/scaler.joblib')
    joblib.dump(IMPORTANT_FEATURES, '../models/features_list.joblib')
    
    print(f"\n🏆 Meilleur modèle sauvegardé : {best_name}")

if __name__ == "__main__":
    train()