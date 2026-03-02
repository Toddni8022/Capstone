"""Tests for capstone.features — feature engineering functions."""

import numpy as np
import pytest

from capstone.features import CombinedAttributesAdder, build_preprocessing_pipeline


def test_combined_attributes_adder_output_shape():
    """Transformer should add 3 columns when add_bedrooms_per_room=True."""
    X = np.ones((10, 9))
    transformer = CombinedAttributesAdder(add_bedrooms_per_room=True)
    X_out = transformer.fit_transform(X)
    assert X_out.shape == (10, 12)


def test_combined_attributes_adder_without_bedrooms():
    """Transformer should add 2 columns when add_bedrooms_per_room=False."""
    X = np.ones((10, 9))
    transformer = CombinedAttributesAdder(add_bedrooms_per_room=False)
    X_out = transformer.fit_transform(X)
    assert X_out.shape == (10, 11)


def test_combined_attributes_adder_rooms_per_household():
    """rooms_per_household = total_rooms / households."""
    X = np.zeros((1, 9))
    X[0, 3] = 100  # total_rooms
    X[0, 4] = 20   # total_bedrooms
    X[0, 5] = 50   # population
    X[0, 6] = 10   # households
    transformer = CombinedAttributesAdder(add_bedrooms_per_room=False)
    X_out = transformer.fit_transform(X)
    rooms_per_household = X_out[0, 9]  # 9th column (0-indexed)
    assert rooms_per_household == pytest.approx(10.0)


def test_build_preprocessing_pipeline_returns_column_transformer(sample_housing_data):
    """build_preprocessing_pipeline should return a ColumnTransformer."""
    from sklearn.compose import ColumnTransformer

    pipeline = build_preprocessing_pipeline()
    assert isinstance(pipeline, ColumnTransformer)


def test_build_preprocessing_pipeline_fit_transform(sample_housing_data):
    """Pipeline should transform housing data to a 2-D numpy array."""
    from capstone.data import get_features_and_target

    X, _ = get_features_and_target(sample_housing_data)
    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X)
    assert X_transformed.ndim == 2
    assert X_transformed.shape[0] == len(sample_housing_data)
