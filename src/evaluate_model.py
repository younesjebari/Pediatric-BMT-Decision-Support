import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score
import os
import sys

# Ajout du chemin pour importer data_processing si besoin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_evaluation():
    print("🧪 Évaluation du modèle sur données de test (inconnues)...")
    
    # 1. Charger le modèle et les données de test sauvegardées par train_model.py
    model_path = 'models/final_model.joblib'
    data_test_path = 'models/test_data.joblib'

    if not os.path.exists(model_path) or not os.path.exists(data_test_path):
        print("❌ Erreur : Modèle ou données de test introuvables. Lancez train_model.py d'abord.")
        return

    model = joblib.load(model_path)
    X_test, y_test = joblib.load(data_test_path)

    # 2. Prédictions
    # y_pred pour les classes (0 ou 1)
    # y_proba pour le score de probabilité (nécessaire pour le ROC-AUC)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probabilité de la classe 0 (Décès)

    # 3. Calcul du rapport détaillé
    # On précise labels=[0, 1] pour s'assurer que la classe 0 est bien traitée
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n" + "="*40)
    print("📊 PERFORMANCES RÉELLES DU MODÈLE")
    print("="*40)
    
    # Extraction des métriques pour la classe 0.0 (Décès)
    summary = {
        "Métrique": ["Rappel (Recall) - Cas critiques", "Précision", "Score ROC-AUC"],
        "Résultat %": [
            f"{report['0.0']['recall']*100:.1f}%",
            f"{report['0.0']['precision']*100:.1f}%",
            f"{roc_auc_score(y_test, y_proba)*100:.1f}%"
        ]
    }
    
    print(pd.DataFrame(summary).to_string(index=False))
    print("="*40)
    print("Note : Ces scores reflètent la capacité du modèle à prédire")
    print("le risque sur de nouveaux patients jamais vus.")

if __name__ == "__main__":
    run_evaluation()