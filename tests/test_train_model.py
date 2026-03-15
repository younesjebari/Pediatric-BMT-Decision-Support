import pytest
import pandas as pd
import numpy as np
import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_processing import select_important_features, handle_missing_values, handle_outliers


@pytest.fixture
def prepared_data():
    """Charge et prepare les donnees comme dans train_model.py."""
    from scipy.io import arff

    raw_data, _ = arff.loadarff('data/bone-marrow.arff')
    df = pd.DataFrame(raw_data)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')

    df = select_important_features(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    X = df.drop(columns=['survival_status'])
    y = pd.to_numeric(df['survival_status']).astype(int)
    return X, y


def test_data_pipeline_produces_11_features(prepared_data):
    """Verifie que le pipeline produit exactement 11 features."""
    X, _ = prepared_data
    assert X.shape[1] == 11


def test_data_pipeline_no_missing_values(prepared_data):
    """Verifie qu'il n'y a plus de valeurs manquantes apres le pipeline."""
    X, y = prepared_data
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0


def test_target_is_binary(prepared_data):
    """Verifie que la cible est binaire (0 ou 1)."""
    _, y = prepared_data
    assert set(y.unique()).issubset({0, 1})


def test_smote_balances_classes(prepared_data):
    """Verifie que SMOTE equilibre les classes."""
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split

    X, y = prepared_data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42, k_neighbors=1)
    _, y_res = smote.fit_resample(X_train, y_train)

    counts = pd.Series(y_res).value_counts()
    assert counts[0] == counts[1], "SMOTE doit equilibrer les deux classes"


def test_model_file_exists():
    """Verifie que le modele sauvegarde existe."""
    assert os.path.isfile('models/final_model.joblib'), "Le fichier final_model.joblib doit exister"


def test_model_can_predict(prepared_data):
    """Verifie que le modele charge peut faire des predictions."""
    X, _ = prepared_data
    model = joblib.load('models/final_model.joblib')

    y_pred = model.predict(X[:5])
    assert len(y_pred) == 5
    assert all(p in [0, 1] for p in y_pred)


def test_model_predict_proba_shape(prepared_data):
    """Verifie que predict_proba renvoie 2 colonnes (survie, deces)."""
    X, _ = prepared_data
    model = joblib.load('models/final_model.joblib')

    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    # Les probas doivent sommer a ~1
    for row in proba:
        assert abs(row.sum() - 1.0) < 1e-5
