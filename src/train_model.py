import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import joblib
import os
import sys

# 1. Configuration du chemin pour importer 'src' correctement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importation de tes outils mis à jour
from src.data_processing import preprocess_data, handle_outliers, IMPORTANT_FEATURES

def train():
    print("\n" + "="*70)
    print("🚀 DÉMARRAGE DU BENCHMARK : 4 MODÈLES PRÉDICTIFS (BMT)")
    print("="*70)

    # 2. Chargement et Décodage
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'bone-marrow.arff')

    try:
        raw_data, _ = arff.loadarff(DATA_PATH)
        df = pd.DataFrame(raw_data)
        for col in df.select_dtypes([object]):
            df[col] = df[col].str.decode('utf-8')
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier '{DATA_PATH}' est introuvable.")
        return

    # 3. Pipeline de Traitement unifié
    # preprocess_data gère déjà l'exclusion de 'survival_time' (leakage)
    df_cleaned = preprocess_data(df)
    df_cleaned = handle_outliers(df_cleaned)
    
    # 4. Séparation X et y en utilisant la liste stricte des features
    # Cela garantit la compatibilité avec l'app Flask et les tests
    X = df_cleaned[IMPORTANT_FEATURES]
    y = pd.to_numeric(df['survival_status'], errors='coerce').fillna(0).astype(int)

    # 5. Split et Équilibrage (SMOTE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE aide à compenser le déséquilibre des classes (décès vs succès)
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # 6. Définition des modèles
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "SVM": SVC(random_state=42, probability=True)
    }

    best_model = None
    best_recall = 0

    print(f"{'Modèle':<15} | {'Accuracy':<10} | {'ROC-AUC':<10} | {'Rappel (Recall)':<10}")
    print("-" * 70)

    for name, m in models.items():
        m.fit(X_res, y_res)
        
        y_pred = m.predict(X_test)
        y_prob = m.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred) * 100
        roc = roc_auc_score(y_test, y_prob) * 100
        rec = recall_score(y_test, y_pred) * 100
        
        print(f"{name:<15} | {acc:>8.1f}% | {roc:>8.1f}% | {rec:>13.1f}%")
        
        # Priorité au Rappel (Recall) pour la sécurité médicale (détection des décès)
        if rec > best_recall:
            best_recall = rec
            best_model = m

    # 7. Sauvegarde du champion
    SAVE_PATH = os.path.join(BASE_DIR, '..', 'models', 'final_model.joblib')
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    joblib.dump(best_model, SAVE_PATH)
    
    print("="*70)
    print(f"🏆 GAGNANT : {best_model.__class__.__name__}")
    print(f"Sauvegardé sous : {SAVE_PATH}")
    print("="*70 + "\n")

if __name__ == "__main__":
    train()