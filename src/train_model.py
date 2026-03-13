import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC  # <-- IMPORT DU SVM
from sklearn.preprocessing import StandardScaler # <-- IMPORT DU SCALER
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
import os

def train():
    print("🚀 Démarrage de l'entraînement et de la comparaison des modèles...")

    # 1. Chargement et Nettoyage
    raw_data, _ = arff.loadarff('data/bone-marrow.arff')
    df = pd.DataFrame(raw_data)

    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')

    features = ['Recipientage', 'Rbodymass', 'Disease', 'Relapse', 'CD34kgx10d6', 'CD3dkgx10d8']

    for col in ['Recipientage', 'Rbodymass', 'CD34kgx10d6', 'CD3dkgx10d8']:
        df[col] = df[col].fillna(df[col].median())

    for col in ['Disease', 'Relapse']:
        df[col] = df[col].astype('category').cat.codes

    X = df[features] 
    y = pd.to_numeric(df['survival_status'], errors='coerce').fillna(0).astype(int)

    # 2. Séparation Entraînement / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. STANDARDISATION (Crucial pour le SVM !)
    # On met toutes les données à la même échelle pour ne pas fausser l'algorithme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Équilibrage des classes (SMOTE) sur les données standardisées
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

    # 5. Comparaison de 4 modèles
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "SVM": SVC(probability=True, random_state=42) # <-- AJOUT DU SVM ICI
    }

    best_model = None
    best_score = 0

    print("\n📊 Résultats des modèles :")
    for name, m in models.items():
        m.fit(X_res, y_res)
        
        # On teste avec X_test_scaled !
        score = m.score(X_test_scaled, y_test)
        print(f"Modèle {name} - Accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = m

    # 6. Sauvegarde du meilleur modèle ET du scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/final_model.joblib')
    
    # On doit obligatoirement sauvegarder le scaler pour le site web
    joblib.dump(scaler, 'models/scaler.joblib') 
    
    print(f"\n🏆 Meilleur modèle sauvegardé : {best_model.__class__.__name__} (Score: {best_score:.4f})")
    print("💾 Scaler sauvegardé dans 'models/scaler.joblib'")

if __name__ == "__main__":
    train()