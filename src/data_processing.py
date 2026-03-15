import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# ==============================================================================
# 1. DÉFINITION DES FEATURES (Basé sur le modèle final)
# ==============================================================================
# Liste requise pour les tests et la cohérence du modèle
IMPORTANT_FEATURES = [
    "CD3dkgx10d8", "CD34kgx10d6", "Rbodymass", "Recipientage", 
    "PLTrecovery", "Disease", "Relapse", "extcGvHD", 
    "Donorage", "HLAmatch", "Riskgroup"
]

# Colonnes à exclure pour éviter le Data Leakage identifié dans l'EDA
LEAKAGE_COLUMNS = ["survival_time"]

def preprocess_data(df):
    """
    Nettoyage global : suppression du leakage, encodage et imputer.
    """
    df = df.copy()
    
    # 1. Suppression du Data Leakage (survival_time)
    for col in LEAKAGE_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # 2. Encodage des variables catégorielles (object -> numeric)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
        
    # 3. Imputation des valeurs manquantes (Médiane pour le numérique)
    df = handle_missing_values(df)
    
    return df

def handle_missing_values(df):
    """Impute les valeurs manquantes selon la stratégie de l'EDA."""
    df = df.copy()
    
    # Numériques : Médiane (Moins sensible aux outliers)
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy="median")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # Catégorielles : Mode (Valeur la plus fréquente)
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    return df

def handle_outliers(df, factor=1.5):
    """
    Cap les valeurs extrêmes. 
    Note : Le facteur est passé de 3.0 à 1.5 pour mieux gérer les extrêmes 
    détectés dans l'EDA.
    """
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
    """Réduit l'usage RAM d'environ 50%."""
    df = df.copy()
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df