import numpy as np
import pandas as pd


def validate_data(X: pd.DataFrame, y: pd.Series) -> None:
    # TODO: validate input types
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")

    # TODO: reject empty inputs
    if X.empty:
        raise ValueError("X cannot be empty")
    if X.shape[1] == 0:
        raise ValueError("X must have at least one feature")

    if y.empty:
        raise ValueError("y cannot be empty")

    # TODO: validate matching row counts
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    # TODO: reject non-finite values
    if not np.isfinite(X).all().all():
        raise ValueError("X contains non-finite values")

    if not np.isfinite(y).all():
        raise ValueError("y contains non-finite values")
    