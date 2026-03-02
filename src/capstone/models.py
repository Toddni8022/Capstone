"""Model training and evaluation utilities."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from capstone.config import CV_FOLDS, MODELS_DIR, RANDOM_STATE


def build_linear_regression_pipeline(preprocessor) -> Pipeline:
    """Return a Pipeline with the given preprocessor and a Linear Regression model.

    Parameters
    ----------
    preprocessor:
        A fitted or unfitted sklearn transformer/pipeline step.

    Returns
    -------
    Pipeline
    """
    return Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])


def build_random_forest_pipeline(
    preprocessor,
    n_estimators: int = 100,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Return a Pipeline with the given preprocessor and a Random Forest model.

    Parameters
    ----------
    preprocessor:
        A fitted or unfitted sklearn transformer/pipeline step.
    n_estimators:
        Number of trees in the forest.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
    """
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators, random_state=random_state
                ),
            ),
        ]
    )


def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
) -> dict[str, float]:
    """Evaluate a pipeline using cross-validation RMSE.

    Parameters
    ----------
    pipeline:
        An unfitted sklearn ``Pipeline``.
    X:
        Feature DataFrame.
    y:
        Target Series.
    cv:
        Number of cross-validation folds.

    Returns
    -------
    dict[str, float]
        Dictionary with ``mean_rmse`` and ``std_rmse``.
    """
    scores = cross_val_score(
        pipeline, X, y, scoring="neg_mean_squared_error", cv=cv
    )
    rmse_scores = np.sqrt(-scores)
    return {
        "mean_rmse": float(rmse_scores.mean()),
        "std_rmse": float(rmse_scores.std()),
    }


def compute_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Model predictions.

    Returns
    -------
    float
        RMSE value.
    """
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def save_model(pipeline: Pipeline, filename: str, models_dir: Path = MODELS_DIR) -> Path:
    """Persist a trained pipeline to disk using joblib.

    Parameters
    ----------
    pipeline:
        A fitted sklearn ``Pipeline``.
    filename:
        Output filename (e.g. ``"random_forest.pkl"``).
    models_dir:
        Directory where the model file is saved.

    Returns
    -------
    Path
        Full path of the saved model file.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / filename
    joblib.dump(pipeline, model_path)
    return model_path


def load_model(filename: str, models_dir: Path = MODELS_DIR) -> Pipeline:
    """Load a previously saved pipeline from disk.

    Parameters
    ----------
    filename:
        Name of the model file (e.g. ``"random_forest.pkl"``).
    models_dir:
        Directory containing the model file.

    Returns
    -------
    Pipeline
        The loaded sklearn pipeline.
    """
    model_path = models_dir / filename
    return joblib.load(model_path)
