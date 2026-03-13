import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os

# ─────────────────────────────────────────────
#  FEATURES — noms exacts tels qu'ils sont dans le dataset
# ─────────────────────────────────────────────
FEATURES = [
    'Recipientage',   # Âge du receveur
    'Rbodymass',      # Masse corporelle
    'CD34kgx10d6',    # Dose CD34+
    'CD3dkgx10d8',    # Dose CD3+
    'Disease',        # Type de maladie        (catégorielle)
    'Relapse',        # Antécédent de rechute  (catégorielle)
    'Gendermatch',    # Compatibilité de genre (catégorielle)
    'HLAmatch',       # Compatibilité HLA      (catégorielle) ← nom exact du dataset
]

CATEGORICAL_FEATURES = ['Disease', 'Relapse', 'Gendermatch', 'HLAmatch']
NUMERICAL_FEATURES   = ['Recipientage', 'Rbodymass', 'CD34kgx10d6', 'CD3dkgx10d8']


def train():
    print("🚀 Démarrage de l'entraînement et de la comparaison des modèles...")

    # ── 1. Chargement des données ──────────────────────────────────────────
    raw_data, _ = arff.loadarff('data/bone-marrow.arff')
    df = pd.DataFrame(raw_data)

    # Décodage bytes → string
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')

    # ── 2. Sélection et nettoyage des colonnes utiles ──────────────────────
    df = df[FEATURES + ['survival_status']].copy()

    # Remplissage des valeurs manquantes
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode()[0])

    # ── 3. Encodage des variables catégorielles ────────────────────────────
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype('category').cat.codes

    # ── 4. Cible ───────────────────────────────────────────────────────────
    X = df[FEATURES]
    y = pd.to_numeric(df['survival_status'], errors='coerce').fillna(0).astype(int)

    # ── 5. Séparation train / test ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 6. Standardisation ────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── 7. SMOTE — équilibrage des classes ────────────────────────────────
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"   Après SMOTE — classe 0: {(y_res==0).sum()}, classe 1: {(y_res==1).sum()}")

    # ── 8. Comparaison de 4 modèles ───────────────────────────────────────
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost":      XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
        "LightGBM":     LGBMClassifier(random_state=42, verbose=-1),
        "SVM":          SVC(probability=True, random_state=42),
    }

    best_model      = None
    best_score      = 0
    best_model_name = ""

    print("\n📊 Résultats des modèles :")
    print(f"{'Modèle':<18} {'Accuracy':>10} {'ROC-AUC':>10}")
    print("-" * 42)

    for name, m in models.items():
        m.fit(X_res, y_res)
        acc     = m.score(X_test_scaled, y_test)
        y_proba = m.predict_proba(X_test_scaled)[:, 1]
        auc     = roc_auc_score(y_test, y_proba)
        print(f"{name:<18} {acc:>10.4f} {auc:>10.4f}")

        if auc > best_score:
            best_score      = auc
            best_model      = m
            best_model_name = name

    # ── 9. Sauvegarde ─────────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)

    joblib.dump(best_model, 'models/final_model.joblib')
    joblib.dump(scaler,     'models/scaler.joblib')
    joblib.dump(FEATURES,   'models/features_list.joblib')

    print(f"\n🏆 Meilleur modèle : {best_model_name} (ROC-AUC: {best_score:.4f})")
    print("💾 Fichiers sauvegardés dans models/")
    print("   • final_model.joblib")
    print("   • scaler.joblib")
    print("   • features_list.joblib")


if __name__ == "__main__":
    train()