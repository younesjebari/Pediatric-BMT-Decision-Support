import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from scipy.io import arff
import os
import sys

# Importation du nettoyage pour avoir les mêmes données qu'à l'entraînement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import handle_missing_values

def run_evaluation():
    print("🧪 Chargement du modèle sauvegardé...")
    
    # 1. Vérifier si le modèle existe
    if not os.path.exists('models/final_model.joblib'):
        print("❌ Erreur : Aucun modèle trouvé. Lancez d'abord train_model.py")
        return

    model = joblib.load('models/final_model.joblib')

    # 2. Chargement et préparation des données
    raw_data, _ = arff.loadarff('data/bone-marrow.arff')
    df = pd.DataFrame(raw_data)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')
    
    df_clean = handle_missing_values(df)
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype('category').cat.codes

    X = df_clean.drop(['survival_status', 'survival_time'], axis=1)
    y = df_clean['survival_status']

    # 3. Calcul des scores
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    report = classification_report(y, y_pred, output_dict=True)

    # 4. Affichage du tableau de pourcentage
    print("\n" + "="*30)
    print("📊 RAPPORT DE PERFORMANCE")
    print("="*30)
    
    # On affiche les résultats pour la classe '0.0' (Décès)
    summary = {
        "Métrique": ["Rappel (Décès)", "Précision (Décès)", "Score ROC-AUC"],
        "Résultat %": [
            f"{report['0.0']['recall']*100:.1f}%",
            f"{report['0.0']['precision']*100:.1f}%",
            f"{roc_auc_score(y, y_proba)*100:.1f}%"
        ]
    }
    print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__":
    run_evaluation()