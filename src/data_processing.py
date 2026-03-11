import pandas as pd
import numpy as np

def optimize_memory(df):
    """
    Optimise l'usage de la mémoire en ajustant les types de données.
    Passage de float64 à float32 et int64 à int32.
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            df[col] = df[col].astype('int32')
            
    return df
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

def handle_missing_values(df):
    """
    - Colonnes numériques  → imputation par la médiane
    - Colonnes catégorielles → imputation par la valeur la plus fréquente
    """
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy="median")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    return df