

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
    Colonnes numériques   → médiane
    Colonnes catégorielles → valeur la plus fréquente
    """
    df = df.copy()
    
    # Remplace les None par NaN d'abord
    df = df.fillna(value=pd.NA)
    
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy="median")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    return df