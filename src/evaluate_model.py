import matplotlib
matplotlib.use('Agg')  # Indispensable pour éviter l'erreur Tcl/Tkinter sur Windows
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
from scipy.io import arff
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import shap

# Importation de la liste officielle des variables
from data_processing import IMPORTANT_FEATURES

def evaluate():
    print("🔬 Évaluation finale : Validation des performances (0=Succès, 1=Décès)")
    
    # 1. Chargement du modèle
    try:
        model = joblib.load('models/final_model.joblib')
    except Exception as e:
        print(f"❌ Erreur de chargement : {e}")
        return

    # 2. Chargement et décodage des données
    raw_data, _ = arff.loadarff('data/bone-marrow.arff')
    df = pd.DataFrame(raw_data)
    for col in df.select_dtypes([object]): 
        df[col] = df[col].str.decode('utf-8')

    # 3. Prétraitement
    features = [f for f in IMPORTANT_FEATURES if f in df.columns]
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].median())

    X = df[features]
    y = pd.to_numeric(df['survival_status'], errors='coerce').fillna(0).astype(int)

    # 4. Prédictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # 5. Rapport de performance (Format Pourcentage)
    report = classification_report(y, y_pred, target_names=['Succès (0)', 'Décès (1)'], output_dict=True)
    
    print("\n📊 RAPPORT DE PERFORMANCE MÉDICALE")
    print("-" * 60)
    print(f"{'Classe':<15} | {'Précision':<10} | {'Rappel':<10} | {'F1-Score':<10}")
    print("-" * 60)
    
    for label in ['Succès (0)', 'Décès (1)']:
        prec = report[label]['precision'] * 100
        rec = report[label]['recall'] * 100
        f1 = report[label]['f1-score'] * 100
        print(f"{label:<15} | {prec:>8.1f}% | {rec:>8.1f}% | {f1:>8.1f}%")
    
    print("-" * 60)
    print(f"✅ Score ROC-AUC : {roc_auc_score(y, y_proba) * 100:.2f}%")
    print(f"📈 Exactitude (Accuracy) : {report['accuracy'] * 100:.1f}%")

    # 6. Sauvegarde de la Matrice de Confusion
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Prédit Succès', 'Prédit Décès'], 
                yticklabels=['Réalité Succès', 'Réalité Décès'])
    plt.title('Matrice de Confusion (Validation Clinique)')
    plt.savefig('notebooks/confusion_matrix_finale.png')
    plt.close()

    # 7. Sauvegarde du graphique SHAP
    print("\n🧬 Analyse SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('notebooks/analyse_shap_finale.png')
    plt.close()
    
    print("\n✅ Évaluation terminée. Les graphiques sont dans le dossier /notebooks.")

if __name__ == "__main__":
    evaluate()