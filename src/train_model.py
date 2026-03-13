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
    print(" Démarrage de la comparaison des modèles...")
    
    # 1. Chargement et Nettoyage
    raw_data, _ = arff.loadarff('data/bone-marrow.arff')
    df = pd.DataFrame(raw_data)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')
    
    # Encodage rapide des variables catégorielles
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    X = df.drop(['survival_status', 'survival_time'], axis=1)
    y = df['survival_status']

    # 2. Séparation et Équilibrage (SMOTE) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # 3. Comparaison de 3 modèles 
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42)
    }

    best_model = None
    best_score = 0

    for name, m in models.items():
        m.fit(X_res, y_res)
        score = m.score(X_test, y_test)
        print(f"Modèle {name} - Accuracy: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = m

    # 4. Sauvegarde du meilleur modèle 
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/final_model.joblib')
    print(f" Meilleur modèle sauvegardé : {best_model.__class__.__name__}")

if __name__ == "__main__":
    train()