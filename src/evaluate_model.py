import joblib
import pandas as pd
from scipy.io import arff
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate():
    # 1. Charger le modèle et la liste des variables
    try:
        model = joblib.load('../models/final_model.joblib')
        features = joblib.load('../models/features_list.joblib')
        print("💾 Modèle chargé avec succès.")
    except:
        print("❌ Erreur : Le modèle n'existe pas. Lancez train_model.py d'abord.")
        return

    # 2. Charger et préparer les données de test
    raw_data, _ = arff.loadarff('../data/bone-marrow.arff')
    df = pd.DataFrame(raw_data)
    
    # Décodage et encodage rapide pour l'évaluation
    for col in df.select_dtypes([object]): 
        df[col] = df[col].str.decode('utf-8')
    
    # On remplit les vides juste pour l'évaluation globale (ou dropna)
    df = df.dropna(subset=features + ['survival_status'])
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    
    X = df[features]
    y = pd.to_numeric(df['survival_status']).astype(int)

    # 3. Prédictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # 4. Rapport
    print("\n📊 BILAN DE PERFORMANCE FINALE")
    print("-" * 40)
    print(classification_report(y, y_pred))
    print(f"Score ROC-AUC : {roc_auc_score(y, y_proba):.4f}")

    # 5. Matrice de Confusion
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de Confusion (Survie)")
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')
    plt.show()

if _name_ == "_main_":
    evaluate()