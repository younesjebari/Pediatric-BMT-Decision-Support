import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.io import arff
from scipy.stats import mannwhitneyu

# Chargement automatique des données pour sélection des features
raw_data, _ = arff.loadarff('../data/bone-marrow.arff')
df_auto = pd.DataFrame(raw_data)
for col in df_auto.select_dtypes([object]): 
    df_auto[col] = df_auto[col].str.decode('utf-8')

# Encodage des catégorielles pour les tests
for col in df_auto.select_dtypes(include=['object']).columns:
    df_auto[col] = df_auto[col].astype('category').cat.codes

# Imputation des valeurs manquantes
for col in df_auto.select_dtypes(include=['number']).columns:
    df_auto[col] = df_auto[col].fillna(df_auto[col].median())

# Cible
y_auto = pd.to_numeric(df_auto['survival_status'], errors='coerce').fillna(0).astype(int)

# Sélection automatique des features (Mann-Whitney U, p < 0.05)
IMPORTANT_FEATURES = []
for col in df_auto.columns:
    if col == 'survival_status':
        continue  # On saute la cible
    group0 = df_auto[y_auto == 0][col]
    group1 = df_auto[y_auto == 1][col]
    if len(group0) > 0 and len(group1) > 0:
        stat, p = mannwhitneyu(group0, group1, alternative='two-sided')
        if p < 0.05:  # Seuil de significativité
            IMPORTANT_FEATURES.append(col)

print(f"🔍 Sélection automatique de {len(IMPORTANT_FEATURES)} features importantes : {IMPORTANT_FEATURES}")

def select_features_from_eda(df, target='survival_status',
                             weak_threshold=0.05,
                             redundancy_threshold=0.85,
                             leakage_cols=None):
    """Sélectionne les features selon la logique de l'EDA (corrélations)."""
    if leakage_cols is None:
        leakage_cols = ['survival_time']

    df = df.copy()

    # On s'assure que la cible est numérique pour le calcul des corrélations
    if target in df.columns:
        df[target] = pd.to_numeric(df[target], errors='coerce')

    num_df = df.select_dtypes(include=['number'])
    corr_matrix = num_df.corr()

    # Variables à faible signal (corrélation abs < seuil)
    if target in corr_matrix.columns:
        weak = corr_matrix[target].abs() < weak_threshold
        weak_features = set(corr_matrix.index[weak]) - {target}
    else:
        weak_features = set()

    # Variables redondantes (corrélation > seuil entre elles)
    redundant = set()
    corr_abs = corr_matrix.abs()
    cols = corr_abs.columns
    for i, col in enumerate(cols):
        for j in range(i + 1, len(cols)):
            other = cols[j]
            if corr_abs.loc[col, other] > redundancy_threshold:
                redundant.add(other)

    # Construction de la liste finale
    to_drop = weak_features | redundant | set(leakage_cols)
    keep = [c for c in df.columns if c not in to_drop]

    # On force la présence de la cible à la fin si elle existe
    if target in df.columns and target not in keep:
        keep.append(target)

    print(f"✅ Sélection automatique : {len(keep)} variables retenues (exclues={len(to_drop)})")
    return df[keep]

def select_important_features(df):
    """Alias pour compatibilité (ancienne API)."""
    return select_features_from_eda(df)

def handle_missing_values(df):
    """Impute les valeurs manquantes (médiane num, mode cat)."""
    df = df.copy()

    # Numériques : médiane
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy="median")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # Catégorielles : mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].astype(str).replace('None', np.nan).replace('nan', np.nan)
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    return df

def handle_outliers(df, factor=3.0):
    """Cap les valeurs extrêmes pour éviter de fausser l'apprentissage."""
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns

    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

def optimize_memory(df):
    """Réduit l'empreinte RAM (float64 -> float32, int64 -> int32)."""
    df = df.copy()
    before = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")

    after = df.memory_usage(deep=True).sum() / 1024**2
    print(f"📉 Mémoire réduite de {before:.2f}MB à {after:.2f}MB ({(1-after/before)*100:.1f}%)")
    return df