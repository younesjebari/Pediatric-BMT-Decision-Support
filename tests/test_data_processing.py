import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
import pandas as pd
import numpy as np
# Mise à jour de l'import : on utilise remove_missing_values
from data_processing import optimize_memory, remove_missing_values 

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age":    [22.0, 35.0, 28.0],
        "score":  [10, 20, 30],
        "groupe": ["A", "B", "A"],
    })

@pytest.fixture
def df_with_missing():
    # Ici, la ligne du milieu (index 1) a des None
    # La ligne index 2 a un None dans score
    return pd.DataFrame({
        "age":    [22.0, None, 28.0],
        "score":  [10, 20, None],
        "groupe": ["A", "B", "A"],
    })

def test_float64_en_float32(sample_df):
    result = optimize_memory(sample_df)
    assert result["age"].dtype == "float32"

def test_int64_en_int32(sample_df):
    result = optimize_memory(sample_df)
    assert result["score"].dtype == "int32"

def test_plus_de_nan(df_with_missing):
    # Teste si la suppression radicale ne laisse aucun NaN
    result = remove_missing_values(df_with_missing)
    assert result.isnull().sum().sum() == 0

def test_shape_diminue(df_with_missing):
    """
    Nouveau test : On vérifie que les lignes incomplètes sont bien SUPPRIMÉES.
    Dans df_with_missing, seule la 1ère ligne est complète.
    """
    result = remove_missing_values(df_with_missing)
    # On s'attend à ce qu'il ne reste qu'une seule ligne
    assert len(result) < len(df_with_missing)
    assert len(result) == 1