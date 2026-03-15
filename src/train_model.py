import pandas as pd
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
import sys

# 1. GESTION DYNAMIQUE DES CHEMINS (La solution à ton erreur)
# On récupère le chemin du dossier où se trouve ce fichier (src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# On remonte d'un niveau pour avoir la racine du projet
ROOT_DIR = os.path.dirname(CURRENT_DIR)
# On définit les chemins vers data et models de façon absolue
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'bone-marrow.arff')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Ajout du dossier src au système pour pouvoir importer data_processing
sys.path.append(CURRENT_DIR)
try:
    from data_processing import clean_data, encode_categories, IMPORTANT_FEATURES
except ImportError:
    print("❌ Erreur : Impossible de trouver data_processing.py dans le dossier src/")
    sys.exit(1)

def train():
    print(f"📂 Recherche du fichier de données dans : {DATA_PATH}")

    # 2. VÉRIFICATION DE L'EXISTENCE DU FICHIER
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERREUR : Le fichier est introuvable !")
        print(f"Vérifiez que vous avez bien un dossier nommé 'data' à côté du dossier 'src'.")
        print(f"Le fichier doit s'appeler exactement : bone-marrow.arff")
        return

    # 3. CHARGEMENT
    try:
        raw_data, _ = arff.loadarff(DATA_PATH)
        df = pd.DataFrame(raw_data)
        print("✅ Données chargées avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier ARFF : {e}")
        return

    # 4. PRÉPARATION (Utilise ta logique validée par l'EDA)
    df = clean_data(df)
    df = encode_categories(df)
    
    X = df[IMPORTANT_FEATURES]
    y = pd.to_numeric(df['survival_status']).astype(int)

    # 5. SPLIT TRAIN/TEST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. STANDARDISATION (Indispensable pour le Scaler de l'App)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. ÉQUILIBRAGE SMOTE
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

    # 8. COMPARAISON DES MODÈLES
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
        "SVM": SVC(probability=True, random_state=42)
    }

    best_model = None
    best_score = 0
    best_name = ""

    print("\n📊 Évaluation des modèles :")
    for name, m in models.items():
        m.fit(X_res, y_res)
        y_proba = m.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"   - {name:15s} | ROC-AUC : {auc:.4f}")
        
        if auc > best_score:
            best_score = auc
            best_model = m
            best_name = name

    # 9. SAUVEGARDE AUTOMATIQUE
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"📁 Dossier créé : {MODELS_DIR}")

    joblib.dump(best_model, os.path.join(MODELS_DIR, 'final_model.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    joblib.dump(IMPORTANT_FEATURES, os.path.join(MODELS_DIR, 'features_list.joblib'))
    
    print(f"\n🏆 Gagnant : {best_name} (Sauvegardé dans /models/)")

if __name__ == "__main__":
    train()