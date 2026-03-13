import pandas as pd
import numpy as np

# Liste officielle des variables validées par tes tests statistiques ✅
# Ce sont les variables qui ont un lien réel avec la survie des patients
IMPORTANT_FEATURES = [
    'Recipientage',   # Âge du receveur
    'Rbodymass',      # Masse corporelle
    'CD34kgx10d6',    # Dose CD34+
    'CD3dkgx10d8',    # Dose CD3+
    'Disease',        # Type de maladie
    'Relapse',        # Antécédent de rechute
    'Gendermatch',    # Compatibilité de genre
    'Hlamatch'        # Compatibilité HLA
]

def clean_data(df):
    """
    Décode les données et supprime les lignes avec des valeurs manquantes.
    C'est la méthode la plus fiable pour éviter les biais.
    """
    df = df.copy()
    # Décodage des chaînes de caractères bytes
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')
    
    # Suppression radicale des lignes incomplètes (dropna)
    return df.dropna()

def encode_categories(df):
    """Transforme le texte en codes numériques pour que l'IA comprenne."""
    df = df.copy()
    # On transforme les colonnes 'object' en types 'category' puis en codes
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    return df

# --- FONCTIONS DE COMPATIBILITÉ POUR RÉPARER LES TESTS GITHUB (LA CROIX ROUGE) ---

def optimize_memory(df):
    """Réduit l'usage mémoire (demandé par test_data_processing.py)."""
    df = df.copy()
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    return df

def handle_missing_values(df):
    """Remplit les cases vides (demandé par test_plus_de_nan)."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("valeur_manquante")
        else:
            df[col] = df[col].fillna(df[col].median())
    return df