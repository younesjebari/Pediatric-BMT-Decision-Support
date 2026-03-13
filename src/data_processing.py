import pandas as pd
import numpy as np
from scipy.io import arff
import os

# --- 1. CONFIGURATION ---
# Liste des variables importantes basées sur l'analyse médicale
IMPORTANT_FEATURES = [
    'CD3dkgx10d8',   # Dose CD3+
    'CD34kgx10d6',   # Dose CD34+
    'Rbodymass',     # Masse corporelle
    'Recipientage',  # Age receveur
    'PLTrecovery',   # Récupération plaquettes
    'Disease',       # Type maladie
    'Relapse',       # Rechute
    'extcGvHD',      # GvHD chronique
    'Donorage',      # Age donneur
    'HLAmatch',      # Compatibilité HLA
    'Riskgroup',     # Groupe de risque
    'survival_status' # CIBLE (A ajuster selon le nom exact dans le .arff)
]

# --- 2. FONCTIONS DE TRAITEMENT ---

def load_data(filepath):
    """Charge le fichier ARFF et gère le formatage initial."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} est introuvable.")
        
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    
    # Décodage des bytes en strings (nécessaire pour le format ARFF)
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode('utf-8')
    
    # Remplacement des '?' (format UCI) par de vrais NaN pour Pandas
    df = df.replace('?', np.nan)
    
    # Conversion forcée en numérique pour les colonnes qui devraient l'être
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
        
    return df

def remove_missing_values(df):
    """Supprime toutes les lignes contenant au moins une valeur manquante."""
    initial_shape = df.shape[0]
    df_clean = df.dropna().copy()
    
    print(f"📊 Nettoyage : {initial_shape - df_clean.shape[0]} lignes supprimées.")
    print(f"✅ Lignes restantes : {df_clean.shape[0]}")
    return df_clean

def optimize_memory(df):
    """Réduit l'usage mémoire en convertissant les types (float64 -> float32)."""
    df = df.copy()
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    return df

def handle_outliers(df, factor=3.0):
    """Limite les valeurs extrêmes via la méthode IQR."""
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)
    return df

def select_important_features(df):
    """Garde uniquement les colonnes définies dans IMPORTANT_FEATURES."""
    # On ne garde que les colonnes qui existent réellement dans le DataFrame
    existing_cols = [c for c in IMPORTANT_FEATURES if c in df.columns]
    return df[existing_cols]

# --- 3. EXECUTION PRINCIPALE ---

if __name__ == "__main__":
    # Chemins
    input_file = "data/bone-marrow.arff"
    output_file = "data/bone_marrow_clean.csv"
    
    print("🚀 Démarrage du Data Processing...")
    
    # Pipeline de traitement
    df = load_data(input_file)
    df = remove_missing_values(df)
    df = handle_outliers(df)
    df = optimize_memory(df)
    df = select_important_features(df)
    
    # Sauvegarde finale
    df.to_csv(output_file, index=False)
    
    print(f"✨ Terminé ! Dataset sauvegardé dans : {output_file}")
    print(f"📏 Dimensions finales : {df.shape}")