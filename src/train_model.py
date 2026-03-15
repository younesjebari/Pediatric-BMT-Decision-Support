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

# Importation de tes outils de traitement
from data_processing import select_important_features, handle_missing_values, handle_outliers, optimize_memory

def train():
    print("\n" + "="*70)
    print("🚀 DÉMARRAGE DU BENCHMARK : 4 MODÈLES PRÉDICTIFS (BMT)")
    print("="*70)

    # 1. Chargement et Décodage
    try:
        raw_data, _ = arff.loadarff('data/bone-marrow.arff')
        df = pd.DataFrame(raw_data)
        for col in df.select_dtypes([object]):
            df[col] = df[col].str.decode('utf-8')
    except FileNotFoundError:
        print("❌ Erreur : Le fichier 'data/bone-marrow.arff' est introuvable.")
        return

    # 2. Pipeline de Traitement (Utilisation des fonctions de data_processing.py)
    # On utilise ta logique centralisée pour garantir la propreté des données
    df = select_important_features(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    
    # Encodage spécifique pour les modèles (catégories -> codes numériques)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    # 3. Séparation X et y
    X = df.drop(columns=['survival_status'])
    y = pd.to_numeric(df['survival_status']).astype(int)

    # 4. Split et Équilibrage (SMOTE)
    # On stratifie pour garder la même proportion de survie dans les deux sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE aide à compenser le manque de données sur les cas de décès
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # 5. Définition des 4 modèles obligatoires
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
        
        # Sélection basée sur le Rappel pour la sécurité médicale
        if rec > best_recall:
            best_recall = rec
            best_model = m

    # 6. Sauvegarde du champion
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/final_model.joblib')
    
    print("="*70)
    print(f"🏆 GAGNANT : {best_model.__class__.__name__}")
    print(f"Ce modèle est sauvegardé pour l'application Streamlit.")
    print("="*70 + "\n")

if __name__ == "__main__":
    train()