"""Feature engineering functions."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from capstone.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


# Column indices used in the custom transformer below
COL_TOTAL_ROOMS = 3
COL_TOTAL_BEDROOMS = 4
COL_POPULATION = 5
COL_HOUSEHOLDS = 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """Add engineered ratio features to the housing dataset.

    New features added:
    - ``rooms_per_household``
    - ``bedrooms_per_room`` (optional)
    - ``population_per_household``

    Parameters
    ----------
    add_bedrooms_per_room:
        Whether to add the ``bedrooms_per_room`` feature.
    """

    def __init__(self, add_bedrooms_per_room: bool = True) -> None:
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X: np.ndarray, y=None):  # noqa: ARG002
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        rooms_per_household = X[:, COL_TOTAL_ROOMS] / X[:, COL_HOUSEHOLDS]
        population_per_household = X[:, COL_POPULATION] / X[:, COL_HOUSEHOLDS]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, COL_TOTAL_BEDROOMS] / X[:, COL_TOTAL_ROOMS]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        return np.c_[X, rooms_per_household, population_per_household]


def build_preprocessing_pipeline(
    numerical_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    add_bedrooms_per_room: bool = True,
) -> ColumnTransformer:
    """Build a full preprocessing pipeline for the housing dataset.

    Parameters
    ----------
    numerical_columns:
        List of numerical feature column names.
    categorical_columns:
        List of categorical feature column names.
    add_bedrooms_per_room:
        Passed through to :class:`CombinedAttributesAdder`.

    Returns
    -------
    ColumnTransformer
        A fitted-ready sklearn ``ColumnTransformer``.
    """
    if numerical_columns is None:
        numerical_columns = NUMERICAL_COLUMNS
    if categorical_columns is None:
        categorical_columns = CATEGORICAL_COLUMNS

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder(add_bedrooms_per_room=add_bedrooms_per_room)),
            ("std_scaler", StandardScaler()),
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, numerical_columns),
            ("cat", OneHotEncoder(), categorical_columns),
        ]
    )
    return full_pipeline
