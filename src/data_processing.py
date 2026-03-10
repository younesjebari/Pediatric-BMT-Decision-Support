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