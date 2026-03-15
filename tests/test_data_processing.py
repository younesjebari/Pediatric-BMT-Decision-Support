import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ajout du chemin absolu vers le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_processing import optimize_memory, handle_missing_values

@pytest.fixture
def sample_df():
    """Génère un DataFrame avec tes vraies variables pour tester l'optimisation."""
    return pd.DataFrame({
        "Recipientage": [12.0, 8.5, 15.0],
        "CD34kgx10d6": [5.2, 7.1, 4.8],
        "Relapse": [0, 1, 0]
    })