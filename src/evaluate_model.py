import joblib
import pandas as pd
from scipy.io import arff
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
 # Assure-toi que seaborn est bien installé
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# --- GESTION AUTOMATIQUE DES CHEMINS (La solution à ton erreur) ---
# 1. On définit le chemin du dossier 'src' où se trouve ce fichier
current_folder = os.path.dirname(os.path.abspath(__file__))

# 2. On définit la racine du projet (un niveau au-dessus de src)
root_project = os.path.dirname(current_folder)

# 3. On construit les chemins ABSOLUS vers data et models
DATA_PATH = os.path.join(root_project, 'data', 'bone-marrow.arff')
MODELS_DIR = os.path.join(root_project, 'models')

# Ajout du dossier src au chemin système pour l'import de data_processing
sys.path.append(current_folder)

try:
    from data_processing import clean_data, encode_categories
except ImportError:
    print("❌ Erreur : Impossible d'importer data_processing.py")

def evaluate():
    print("📋 Lancement de l'évaluation finale du modèle...")

    # 1. Chargement des ressources avec CHEMINS ABSOLUS
    try:
        model_path = os.path.join(MODELS_DIR, 'final_model.joblib')
        scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
        features_path = os.path.join(MODELS_DIR, 'features_list.joblib')

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        print("💾 Modèle, Scaler et liste des variables chargés avec succès.")
    except Exception as e:
        print(f"❌ Erreur : Fichiers manquants dans {MODELS_DIR}")
        print(f"Détail : {e}")
        return

    # 2. Chargement des données brutes avec CHEMIN ABSOLU
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")
            
        raw_data, _ = arff.loadarff(DATA_PATH)
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
    X_scaled = scaler.transform(X)

    # 5. Prédictions
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # 6. Affichage du Rapport de Classification
    print("\n" + "="*40)
    print(" 🏥 BILAN DE PERFORMANCE CLINIQUE")
    print("="*40)
    print(classification_report(y, y_pred, target_names=['Survie (0)', 'Décès (1)']))
    
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
    
    print("\n📈 Génération du graphique...")
    plt.show()

if __name__ == "__main__":
    evaluate()