import pandas as pd
import pytest
import numpy as np

from ml_pipeline.data_validation import validate_data


def test_rejects_empty_target():
    X = pd.DataFrame({"feature": [1.0, 2.0]})
    y = pd.Series(dtype=float)

    with pytest.raises(ValueError):
        validate_data(X, y)
def test_valid_data():
    X = pd.DataFrame({"feature": [1.0, 2.0]})
    y = pd.Series([0, 1])

    validate_data(X, y)

def test_rejects_wrong_X_type():
    X = [[1.0, 2.0], [3.0, 4.0]]  # Not a DataFrame
    y = pd.Series([0, 1])

    with pytest.raises(TypeError):
        validate_data(X, y)  # type: ignore[arg-type]

def test_rejects_wrong_y_type():
    X = pd.DataFrame({"feature": [1.0, 2.0]})
    y = [0, 1]  # Not a Series

    with pytest.raises(TypeError):
        validate_data(X, y) # type: ignore[arg-type]

def test_rejects_empty_features():
    X = pd.DataFrame()
    y = pd.Series([0, 1])

    with pytest.raises(ValueError):
        validate_data(X, y)

def test_rejects_mismatched_rows():
    X = pd.DataFrame({"feature": [1.0, 2.0]})
    y = pd.Series([0])  # Mismatched row count

    with pytest.raises(ValueError):
        validate_data(X, y)

def test_rejects_zero_feature_columns():
    X = pd.DataFrame(columns=[])
    y = pd.Series([0, 1])

    with pytest.raises(ValueError):
        validate_data(X, y)

def test_rejects_infinite_features():
    X = pd.DataFrame({"feature": [1.0, np.inf]})
    y = pd.Series([0, 1])

    with pytest.raises(ValueError):
        validate_data(X, y)

def test_rejects_nan_target():
    X = pd.DataFrame({"feature": [1.0, 2.0]})
    y = pd.Series([0, np.nan])

    with pytest.raises(ValueError):
        validate_data(X, y)  # type: ignore[arg-type]