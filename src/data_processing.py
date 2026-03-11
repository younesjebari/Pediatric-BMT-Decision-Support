

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
    Colonnes numériques    → médiane
    Colonnes catégorielles → valeur la plus fréquente
    """
    df = df.copy()
    
    # Remplace None par np.nan
    df = df.fillna(np.nan)
    
    # Convertit les colonnes object en string pour sklearn
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).replace("nan", np.nan)

    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy="median")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    if len(cat_cols) > 0:
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