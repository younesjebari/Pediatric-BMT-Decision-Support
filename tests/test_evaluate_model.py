import pytest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# 1. Correction du chemin pour trouver 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importation corrigée
from src.evaluate_model import evaluate
from src.data_processing import IMPORTANT_FEATURES

@pytest.fixture
def mock_model():
    """Mock du modèle Random Forest."""
    model = MagicMock()
    # Simule 3 prédictions pour correspondre aux mock_data
    model.predict.return_value = np.array([0, 1, 0])
    model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    model.feature_importances_ = [0.1] * len(IMPORTANT_FEATURES) # Pour SHAP
    return model

@pytest.fixture
def mock_data():
    """Génère un DataFrame avec TOUTES les features requises."""
    data = {feat: [1.0, 2.0, 3.0] for feat in IMPORTANT_FEATURES}
    data['survival_status'] = [0, 1, 0]
    data['survival_time'] = [100, 200, 300] # Variable de leakage
    return pd.DataFrame(data)

# On patche 'src.evaluate_model' car c'est là que les imports sont consommés
@patch('src.evaluate_model.joblib.load')
@patch('src.evaluate_model.arff.loadarff')
@patch('src.evaluate_model.shap.TreeExplainer')
@patch('src.evaluate_model.plt.savefig')
def test_evaluate_function(mock_savefig, mock_shap, mock_arff, mock_joblib, mock_model, mock_data):
    """Test que evaluate() s'exécute sans erreur avec le nouveau pipeline."""
    
    # Configuration des Mocks
    mock_joblib.return_value = mock_model
    # Simule le retour de arff.loadarff (structured array + metadata)
    mock_arff.return_value = (mock_data.to_records(index=False), None) 
    
    # Exécution de la fonction
    try:
        evaluate()
    except Exception as e:
        pytest.fail(f"evaluate() a levé une exception inattendue : {e}")

    # Vérifications critiques
    mock_joblib.assert_called()  # Vérifie que le modèle a été cherché
    mock_arff.assert_called()    # Vérifie que les données ont été cherchées
    
    # Vérifie que le pipeline a bien filtré pour n'utiliser que les IMPORTANT_FEATURES
    # (Le modèle doit être appelé avec le bon nombre de colonnes)
    args, _ = mock_model.predict.call_args
    input_df = args[0]
    assert input_df.shape[1] == len(IMPORTANT_FEATURES), "Le modèle n'a pas reçu le bon nombre de features"
    assert "survival_time" not in input_df.columns, "Le Data Leakage n'a pas été supprimé"

    # Vérifie la génération des visuels
    assert mock_savefig.called, "Les graphiques de performance n'ont pas été sauvegardés"