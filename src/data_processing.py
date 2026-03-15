import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Variables basées sur tes tests statistiques (p-value < 0.05)
# Note : On a exclu 'survival_time' (leakage) et on garde 'Rbodymass' 
# ici car elle est statistiquement significative, même si corrélée à l'âge.
IMPORTANT_FEATURES = [
    'CD3dkgx10d8',    # Dose CD3+ (p=0.0016)
    'CD34kgx10d6',    # Dose CD34+ (p=0.0070)
    'Rbodymass',      # Masse corporelle (p=0.0033)
    'Recipientage',   # Age receveur (p=0.0050)
    'PLTrecovery',    # Récupération plaquettes (p=0.0071)
    'Disease',        # Type maladie (p=0.0185)
    'Relapse',        # Rechute (p=0.0001)
    'extcGvHD',       # GvHD chronique (p=0.0000)
    'Donorage',       # Age donneur
    'HLAmatch',       # Compatibilité HLA
    'Riskgroup'       # Groupe de risque
]

def select_important_features(df):
    """Garde seulement les variables validées par l'EDA et la cible."""
    target = 'survival_status'
    # On garde les features importantes + la cible si elle est présente
    cols_to_keep = [c for c in IMPORTANT_FEATURES if c in df.columns]
    if target in df.columns:
        cols_to_keep.append(target)
    
    print(f"✅ Variables sélectionnées : {len(cols_to_keep)}")
    return df[cols_to_keep]

def handle_missing_values(df):
    """Impute les valeurs manquantes (Médiane pour num, Mode pour cat)."""
    df = df.copy()
    
    # 1. Numériques : Médiane (plus robuste aux outliers que la moyenne)
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy="median")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # 2. Catégorielles : Valeur la plus fréquente
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        # On sécurise le type objet pour éviter les erreurs de type mixte
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
        # On utilise un facteur 3 (outliers extrêmes) pour ne pas perdre 
        # la réalité médicale des patients "atypiques"
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

def optimize_memory(df):
    """Réduit l'empreinte RAM (float64 -> float32)."""
    df = df.copy()
    before = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
        
    after = df.memory_usage(deep=True).sum() / 1024**2
    print(f"📉 Mémoire réduite de {before:.2f}MB à {after:.2f}MB ({(1-after/before)*100:.1f}%)")
    return df