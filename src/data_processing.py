import pandas as pd
import numpy as np

# Liste finale des variables validées par tes tests statistiques (✅)
# On n'utilise que celles-ci pour le site web et l'entraînement
IMPORTANT_FEATURES = [
    'Recipientage',   # Age receveur
    'Rbodymass',      # Masse corporelle
    'CD34kgx10d6',    # Dose CD34+
    'CD3dkgx10d8',    # Dose CD3+
    'Disease',        # Type de maladie (Catégorielle)
    'Relapse',        # Rechute (Catégorielle)
    'Gendermatch',    # Match de genre (Catégorielle)
    'Hlamatch'        # Match HLA (Catégorielle)
]

def clean_data(df):
    """Nettoyage complet selon tes décisions."""
    df = df.copy()
    
    # 1. Décodage (bytes -> string)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')
    
    # 2. Suppression radicale des lignes avec des cases vides (ta demande)
    # On le fait avant tout calcul pour avoir des données 100% réelles
    df = df.dropna()
    
    return df

def encode_categories(df, target_col='survival_status'):
    """Transforme le texte en nombres pour l'IA."""
    df = df.copy()
    
    # On s'assure que la cible est bien numérique (0 ou 1)
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').astype(int)
    
    # Encodage des catégories (0, 1, 2...)
    # Note: Pour le site web, il faudra garder le dictionnaire des codes
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
        
    return df