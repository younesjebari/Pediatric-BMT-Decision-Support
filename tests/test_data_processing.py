import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ajout du chemin pour trouver les modules dans src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_processing import optimize_memory, handle_missing_values

@pytest.fixture
def sample_df():
    """Génère un DataFrame avec tes variables réelles pour l'optimisation."""
    return pd.DataFrame({
        "Recipientage": [12.0, 8.5, 15.0],
        "CD34kgx10d6": [5.2, 7.1, 4.8],
        "Relapse": [0, 1, 0]
    })

def test_memory_optimization_results(sample_df):
    """Vérifie la réduction de 50% de la mémoire."""
    optimized_df = optimize_memory(sample_df.copy())
    # Vérifie que les types sont bien passés en 32-bit
    assert optimized_df["Recipientage"].dtype == 'float32'
    assert optimized_df["Relapse"].dtype == 'int32'

def test_missing_values_handling():
    """Vérifie que l'imputation fonctionne pour tes variables."""
    df_with_nan = pd.DataFrame({"Recipientage": [10.0, np.nan, 12.0]})
    df_cleaned = handle_missing_values(df_with_nan)
    assert df_cleaned["Recipientage"].isnull().sum() == 0