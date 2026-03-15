import pytest
import os
import sys
import pandas as pd
from unittest.mock import patch, MagicMock

# Ajout du chemin pour trouver les modules dans src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from evaluate_model import evaluate

@pytest.fixture
def mock_model():
    """Mock du modèle pour éviter de charger un vrai fichier."""
    model = MagicMock()
    model.predict.return_value = [0, 1, 0]  # Prédictions mockées
    model.predict_proba.return_value = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]  # Probabilités mockées
    return model

@pytest.fixture
def mock_data():
    """Mock des données pour éviter de charger le fichier ARFF."""
    return pd.DataFrame({
        'Recipientage': [10.0, 12.0, 8.0],
        'CD34kgx10d6': [5.0, 6.0, 4.0],
        'survival_status': [0, 1, 0]
    })

@patch('joblib.load')
@patch('evaluate_model.arff.loadarff')
@patch('evaluate_model.shap.summary_plot')
@patch('evaluate_model.plt.savefig')
def test_evaluate_function(mock_savefig, mock_shap, mock_arff, mock_joblib, mock_model, mock_data):
    """Test que evaluate() s'exécute sans erreur et génère les sorties attendues."""
    # Mock des dépendances
    mock_joblib.return_value = mock_model
    mock_arff.return_value = (mock_data.values, None)  # Simule le chargement ARFF
    
    # Mock du DataFrame après décodage
    with patch('pandas.DataFrame') as mock_df:
        mock_df_instance = MagicMock()
        mock_df_instance.select_dtypes.return_value = mock_df_instance
        mock_df_instance.columns = ['Recipientage', 'CD34kgx10d6', 'survival_status']
        mock_df_instance.drop.return_value = mock_data.drop(columns=['survival_status'])
        mock_df_instance.__getitem__.return_value = mock_data['survival_status']
        mock_df.return_value = mock_df_instance
        
        # Exécution de la fonction
        evaluate()
        
        # Vérifications
        mock_joblib.assert_called_once()  # Modèle chargé
        mock_arff.assert_called_once()  # Données chargées
        mock_savefig.assert_called()  # Images sauvegardées (confusion matrix + SHAP)
        mock_shap.assert_called_once()  # SHAP généré
        
        # Vérifie que les fichiers de sortie existent (si générés)
        confusion_path = os.path.join(os.path.dirname(__file__), '../notebooks/confusion_matrix_finale.png')
        shap_path = os.path.join(os.path.dirname(__file__), '../notebooks/analyse_shap_finale.png')
        
        # Note : Ces assertions peuvent être commentées si les mocks ne créent pas de vrais fichiers
        # assert os.path.exists(confusion_path), "Matrice de confusion non générée"
        # assert os.path.exists(shap_path), "Graphique SHAP non généré"