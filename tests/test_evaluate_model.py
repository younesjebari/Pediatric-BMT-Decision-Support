import pytest
import pandas as pd
import numpy as np
import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_processing import IMPORTANT_FEATURES


@pytest.fixture
def model():
    """Charge le modele entraine."""
    return joblib.load('models/final_model.joblib')


@pytest.fixture
def evaluation_data():
    """Charge et prepare les donnees comme dans evaluate_model.py."""
    from scipy.io import arff

    raw_data, _ = arff.loadarff('data/bone-marrow.arff')
    df = pd.DataFrame(raw_data)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')

    features = [f for f in IMPORTANT_FEATURES if f in df.columns]
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].median())

    X = df[features]
    y = pd.to_numeric(df['survival_status'], errors='coerce').fillna(0).astype(int)
    return X, y


def test_important_features_list():
    """Verifie que IMPORTANT_FEATURES contient 11 variables."""
    assert len(IMPORTANT_FEATURES) == 11


def test_important_features_present_in_data():
    """Verifie que toutes les features existent dans le dataset."""
    from scipy.io import arff

    raw_data, _ = arff.loadarff('data/bone-marrow.arff')
    df = pd.DataFrame(raw_data)
    for feat in IMPORTANT_FEATURES:
        assert feat in df.columns, f"La feature '{feat}' est absente du dataset"


def test_model_accuracy_above_minimum(model, evaluation_data):
    """Verifie que l'accuracy du modele depasse 60%."""
    from sklearn.metrics import accuracy_score

    X, y = evaluation_data
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.60, f"Accuracy trop faible : {acc:.1%}"


def test_model_roc_auc_above_minimum(model, evaluation_data):
    """Verifie que le ROC-AUC du modele depasse 60%."""
    from sklearn.metrics import roc_auc_score

    X, y = evaluation_data
    y_proba = model.predict_proba(X)[:, 1]
    roc = roc_auc_score(y, y_proba)
    assert roc > 0.60, f"ROC-AUC trop faible : {roc:.1%}"


def test_classification_report_has_two_classes(model, evaluation_data):
    """Verifie que le rapport contient les deux classes (0 et 1)."""
    from sklearn.metrics import classification_report

    X, y = evaluation_data
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    assert '0' in report or 'Succès (0)' in report or 0 in report
    assert '1' in report or 'Décès (1)' in report or 1 in report


def test_confusion_matrix_shape(model, evaluation_data):
    """Verifie que la matrice de confusion est 2x2."""
    from sklearn.metrics import confusion_matrix

    X, y = evaluation_data
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    assert cm.shape == (2, 2)


def test_shap_explainer_works(model, evaluation_data):
    """Verifie que SHAP TreeExplainer fonctionne sur le modele."""
    import shap

    X, _ = evaluation_data
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:5])

    # Doit retourner une valeur SHAP par feature par echantillon
    assert shap_values.shape == (5, X.shape[1])


def test_shap_values_not_all_zero(model, evaluation_data):
    """Verifie que les valeurs SHAP ne sont pas toutes nulles."""
    import shap

    X, _ = evaluation_data
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:10])

    assert np.abs(shap_values).sum() > 0, "Les valeurs SHAP sont toutes a zero"
