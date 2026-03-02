"""Tests for capstone.models — model training and evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from capstone.data import get_features_and_target
from capstone.features import build_preprocessing_pipeline
from capstone.models import (
    build_linear_regression_pipeline,
    build_random_forest_pipeline,
    compute_rmse,
    evaluate_model,
)


@pytest.fixture
def preprocessor():
    return build_preprocessing_pipeline()


def test_build_linear_regression_pipeline_type(preprocessor):
    pipeline = build_linear_regression_pipeline(preprocessor)
    assert isinstance(pipeline, Pipeline)


def test_build_random_forest_pipeline_type(preprocessor):
    pipeline = build_random_forest_pipeline(preprocessor)
    assert isinstance(pipeline, Pipeline)


def test_pipeline_fit_predict(sample_housing_data, preprocessor):
    """Pipeline should fit and produce predictions of the right length."""
    X, y = get_features_and_target(sample_housing_data)
    pipeline = build_linear_regression_pipeline(preprocessor)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)


def test_compute_rmse_perfect_predictions():
    """RMSE must be 0 when predictions equal ground truth."""
    y = pd.Series([1.0, 2.0, 3.0])
    assert compute_rmse(y, np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_compute_rmse_known_value():
    """RMSE of [[0,0], [0,3], [4,0]] errors = sqrt((0+9+16)/3) ≈ 2.646."""
    y_true = pd.Series([0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 3.0, 4.0])
    expected = np.sqrt((0 + 9 + 16) / 3)
    assert compute_rmse(y_true, y_pred) == pytest.approx(expected)


def test_evaluate_model_returns_dict(sample_housing_data, preprocessor):
    """evaluate_model should return mean_rmse and std_rmse keys."""
    X, y = get_features_and_target(sample_housing_data)
    pipeline = build_linear_regression_pipeline(preprocessor)
    result = evaluate_model(pipeline, X, y, cv=3)
    assert "mean_rmse" in result
    assert "std_rmse" in result
    assert result["mean_rmse"] >= 0
    assert result["std_rmse"] >= 0
