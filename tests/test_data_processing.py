import sys
import os
<<<<<<< Updated upstream
=======
# Ajout du chemin pour trouver les modules dans src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

>>>>>>> Stashed changes
import pytest
import pandas as pd
import numpy as np

# Ajout du chemin absolu vers le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_processing import optimize_memory, handle_missing_values

@pytest.fixture
def sample_df():
    """Génère un DataFrame propre pour tester l'optimisation mémoire."""
    return pd.DataFrame({
<<<<<<< Updated upstream
        "age":   [22.0, 35.0, 28.0],
        "score":  [10, 20, 30],
        "groupe": ["A", "B", "A"],
=======
        "Recipientage": [12.0, 8.5, 15.0],
        "CD34kgx10d6":  [5.2, 7.1, 4.8],
        "Relapse":      [0, 1, 0]
>>>>>>> Stashed changes
    })

@pytest.fixture
def df_with_missing():
    """Génère un DataFrame avec des NaN pour tester le nettoyage."""
    return pd.DataFrame({
        "Recipientage": [10.0, np.nan, 20.0],
        "Disease":      ["ALL", "AML", None]
    })

<<<<<<< Updated upstream
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
=======
def test_optimize_memory_efficiency(sample_df):
    """Vérifie que le downcasting réduit bien la taille mémoire."""
    before = sample_df.memory_usage(deep=True).sum()
    after_df = optimize_memory(sample_df)
    after = after_df.memory_usage(deep=True).sum()
    assert after <= before
    assert after_df["Recipientage"].dtype == "float32"

def test_handle_missing_imputation(df_with_missing):
    """Vérifie que les NaN sont remplacés par la médiane pour les numériques."""
>>>>>>> Stashed changes
    result = handle_missing_values(df_with_missing)
    # Médiane de [10, 20] = 15
    assert result["Recipientage"].iloc[1] == 15.0
    assert result.isnull().sum().sum() == 0

<<<<<<< Updated upstream
def test_shape_inchange(df_with_missing):
    # Vérifie que l'imputation ne supprime pas de données par erreur
    result = handle_missing_values(df_with_missing)
    assert result.shape == df_with_missing.shape
=======
def test_categorical_encoding():
    """Vérifie que les colonnes 'object' deviennent numériques (codes)."""
    df = pd.DataFrame({"Disease": ["ALL", "AML", "ALL"]})
    result = handle_missing_values(df)
    assert pd.api.types.is_integer_dtype(result["Disease"])
>>>>>>> Stashed changes
