

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def optimize_memory(df):
    """Réduit l'usage mémoire en downcasting les types numériques."""
    df = df.copy()
    
    before = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    
    after = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Mémoire avant : {before:.4f} MB")
    print(f"Mémoire après : {after:.4f} MB")
    print(f"Réduction     : {(1 - after/before)*100:.1f}%")
    
    return df


def handle_missing_values(df):
    """
    Colonnes numériques → médiane
    Colonnes catégorielles → valeur la plus fréquente
    """
    df = df.copy()
    
    # 1. Traiter les numériques
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy="median")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # 2. Traiter les catégorielles (C'est ici que ça plantait)
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        # On remplace d'abord les None par des strings vides ou une valeur explicite
        # pour éviter l'ambiguïté que Pandas n'aime pas
        df[cat_cols] = df[cat_cols].fillna("missing_value")
        
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    return df

def handle_outliers(df, factor=3.0):
    """
    Cap les outliers avec la méthode IQR.
    Conservateur (factor=3.0) pour préserver
    les valeurs médicales extrêmes mais réelles.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(
            lower=q1 - factor * iqr,
            upper=q3 + factor * iqr
        )
    return df
# Variables importantes sélectionnées par tests statistiques
IMPORTANT_FEATURES = [
    # Numériques significatives
    'CD3dkgx10d8',    # Dose CD3+ (p=0.0016)
    'CD34kgx10d6',    # Dose CD34+ (p=0.0070)
    'Rbodymass',      # Masse corporelle (p=0.0033)
    'Recipientage',   # Age receveur (p=0.0050)
    'PLTrecovery',    # Récupération plaquettes (p=0.0067)
    # Catégorielles significatives
    'Disease',        # Type maladie (p=0.0185)
    'Relapse',        # Rechute (p=0.0001)
    'extcGvHD',       # GvHD chronique (p=0.0000)
    # Cliniquement importantes
    'Donorage',       # Age donneur
    'HLAmatch',       # Compatibilité HLA
    'Riskgroup',      # Groupe de risque
]

def select_important_features(df):
    """Garde seulement les variables importantes."""
    cols = [c for c in IMPORTANT_FEATURES if c in df.columns]
    return df[cols]