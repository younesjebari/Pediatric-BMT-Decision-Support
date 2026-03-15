import pandas as pd
import numpy as np

# Liste des variables validées par ton EDA (p < 0.05) ✅
# Nous excluons 'survival_time' (leakage) et 'PLTrecovery' (donnée post-op)
IMPORTANT_FEATURES = [
    'CD3dkgx10d8',    # Corrélation significative (-0.23)
    'Rbodymass',      # Corrélation significative (0.21)
    'Recipientage',   # Corrélation significative (0.20)
    'CD34kgx10d6',    # Corrélation significative (-0.19)
    'Disease',        # Variable clinique majeure
    'Relapse',        # Facteur de risque connu
    'Gendermatch',    # Variable de compatibilité
    'HLAmatch'        # Variable de compatibilité
]

def clean_data(df):
    """Nettoyage et décodage des données ARFF."""
    df = df.copy()
    # Décodage des bytes en chaînes de caractères
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')
    
    # Suppression des lignes avec des valeurs manquantes sur les variables clés
    # Cela garantit la qualité médicale de l'apprentissage
    return df.dropna(subset=IMPORTANT_FEATURES + ['survival_status'])

def encode_categories(df):
    """Conversion des catégories en codes numériques."""
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    return df

# --- COMPATIBILITÉ AVEC LES TESTS GITHUB ACTIONS ---

def optimize_memory(df):
    """Réduction de la précision des types pour économiser la mémoire."""
    df = df.copy()
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    return df

def handle_missing_values(df):
    """Remplissage par défaut pour éviter les erreurs dans les tests unitaires."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("inconnu")
        else:
            df[col] = df[col].fillna(df[col].median())
    return df