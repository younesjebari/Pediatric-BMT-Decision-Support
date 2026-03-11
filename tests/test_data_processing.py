import sys
import os
import pytest
import pandas as pd
import numpy as np

# Ajout du chemin absolu vers le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_processing import optimize_memory, handle_missing_values

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age":   [22.0, 35.0, 28.0],
        "score":  [10, 20, 30],
        "groupe": ["A", "B", "A"],
    })

@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        "age":    [22.0, None, 28.0],
        "score":  [10, 20, None],
        "groupe": ["A", None, "A"],
    })

def test_float64_en_float32(sample_df):
    result = optimize_memory(sample_df)
    assert result["age"].dtype == np.float32

def test_int64_en_int32(sample_df):
    result = optimize_memory(sample_df)
    assert result["score"].dtype == np.int32

def test_memoire_reduite(sample_df):
    # On compare la consommation mémoire réelle
    avant = sample_df.memory_usage(deep=True).sum()
    apres = optimize_memory(sample_df).memory_usage(deep=True).sum()
    assert apres < avant  #

def test_plus_de_nan(df_with_missing):
    # Vérifie que handle_missing_values remplit bien tous les trous
    result = handle_missing_values(df_with_missing)
    assert result.isnull().sum().sum() == 0

def test_shape_inchange(df_with_missing):
    # Vérifie que l'imputation ne supprime pas de données par erreur
    result = handle_missing_values(df_with_missing)
    assert result.shape == df_with_missing.shape