import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
import os

def train():
    print("🚀 Démarrage de l'entraînement et de la comparaison des modèles...")

    # 1. Chargement et Nettoyage
    raw_data, _ = arff.loadarff('data/bone-marrow.arff')
    df = pd.DataFrame(raw_data)

    # Décodage des chaînes de caractères (b'string' -> 'string')
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')

    # --- LA GRANDE CORRECTION EST ICI ---
    # On liste nos 6 variables pré-opératoires validées par les statistiques
    features = ['Recipientage', 'Rbodymass', 'Disease', 'Relapse', 'CD34kgx10d6', 'CD3dkgx10d8']

    # On nettoie les valeurs manquantes (l'IA n'aime pas les cases vides)
    for col in ['Recipientage', 'Rbodymass', 'CD34kgx10d6', 'CD3dkgx10d8']:
        df[col] = df[col].fillna(df[col].median())

    # Encodage des variables catégorielles (Disease, Relapse) en nombres (0, 1, 2...)
    for col in ['Disease', 'Relapse']:
        df[col] = df[col].astype('category').cat.codes

    # 2. Séparation de l'énoncé (X) et du corrigé (y)
    X = df[features]  # X ne contient QUE les 6 colonnes d'indices
    
    # y contient UNIQUEMENT la cible (on s'assure que ce sont bien des entiers 0 ou 1)
    y = pd.to_numeric(df['survival_status'], errors='coerce').fillna(0).astype(int)

    # 3. Séparation Entraînement / Test (80% pour apprendre, 20% pour vérifier)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Équilibrage des classes (SMOTE) sur les données d'entraînement
    # k_neighbors=1 est utilisé car nous avons un petit jeu de données
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # 5. Comparaison de 3 modèles
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1)
    }

    best_model = None
    best_score = 0

    print("\n📊 Résultats des modèles :")
    for name, m in models.items():
        # L'algorithme lit l'énoncé (X_res) et regarde le corrigé (y_res) pour apprendre
        m.fit(X_res, y_res)
        
        # On le teste sur des données qu'il n'a jamais vues (X_test)
        score = m.score(X_test, y_test)
        print(f"Modèle {name} - Accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = m

    # 6. Sauvegarde du meilleur modèle
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/final_model.joblib')
    print(f"\n🏆 Meilleur modèle sauvegardé : {best_model.__class__.__name__} (Score: {best_score:.4f})")

if __name__ == "__main__":
    train()

