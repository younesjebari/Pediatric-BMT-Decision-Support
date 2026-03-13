import joblib
import pandas as pd
from scipy.io import arff
from sklearn.metrics import classification_report, roc_auc_score

def evaluate():
    # 1. Charger le modèle et les données
    model = joblib.load('models/final_model.joblib')
    raw_data, _ = arff.loadarff('data/bone-marrow.arff')
    df = pd.DataFrame(raw_data)
    
    # 2. Prétraitement (identique à l'entraînement)
    for col in df.select_dtypes([object]): 
        df[col] = df[col].str.decode('utf-8')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    
    X = df.drop(['survival_status', 'survival_time'], axis=1)
    y = df['survival_status']

    # 3. Prédictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # 4. Affichage des métriques [cite: 18]
    print("\n📊 RAPPORT DE PERFORMANCE MÉDICALE")
    print("-" * 30)
    print(classification_report(y, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y, y_proba):.4f}")

if __name__ == "__main__":
    evaluate()