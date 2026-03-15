import joblib
import pandas as pd
from scipy.io import arff
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# On importe les outils de traitement pour garantir la même logique que l'entraînement
from data_processing import clean_data, encode_categories

def evaluate():
    print("📋 Lancement de l'évaluation finale du modèle...")

    # 1. Chargement des ressources sauvegardées par train_model.py
    try:
        # On remonte d'un dossier car on est dans src/
        model = joblib.load('../models/final_model.joblib')
        scaler = joblib.load('../models/scaler.joblib')
        features = joblib.load('../models/features_list.joblib')
        print("💾 Modèle, Scaler et liste des variables chargés avec succès.")
    except Exception as e:
        print(f"❌ Erreur : Fichiers manquants dans /models/. Lancez d'abord train_model.py.")
        print(f"Détail : {e}")
        return

    # 2. Chargement des données brutes
    try:
        raw_data, _ = arff.loadarff('../data/bone-marrow.arff')
        df = pd.DataFrame(raw_data)
    except Exception as e:
        print(f"❌ Erreur : Impossible de lire le fichier de données. {e}")
        return

    # 3. Préparation des données (Identique à l'entraînement)
    df = clean_data(df)
    df = encode_categories(df)
    
    # On isole les variables d'entrée et la cible
    X = df[features]
    y = pd.to_numeric(df['survival_status']).astype(int)

    # 4. ÉTAPE CRUCIALE : Application du Scaler
    # On utilise transform() et non fit_transform() car on applique la règle apprise à l'entraînement
    X_scaled = scaler.transform(X)

    # 5. Prédictions
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # 6. Affichage du Rapport de Classification
    print("\n" + "="*40)
    print(" 🏥 BILAN DE PERFORMANCE CLINIQUE")
    print("="*40)
    # On affiche les métriques de précision, rappel et F1-score
    print(classification_report(y, y_pred, target_names=['Survie (0)', 'Décès (1)']))
    
    # Le score ROC-AUC mesure la capacité du modèle à séparer les classes
    auc_score = roc_auc_score(y, y_proba)
    print(f"Capacité de distinction (ROC-AUC) : {auc_score:.4f}")

    # 7. Visualisation de la Matrice de Confusion
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Prédit Survie', 'Prédit Décès'],
                yticklabels=['Réalité Survie', 'Réalité Décès'])
    plt.title("Matrice de Confusion : Aide à la décision BMT")
    plt.ylabel('Observation Clinique')
    plt.xlabel('Prédiction de l\'IA')
    
    print("\n📈 Génération du graphique de la matrice de confusion...")
    plt.show()

if __name__ == "__main__":
    evaluate()