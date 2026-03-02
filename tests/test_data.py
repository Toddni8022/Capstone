"""Tests for capstone.data — data loading and preprocessing utilities."""

import pandas as pd

from capstone.data import create_train_test_split, get_features_and_target


def test_create_train_test_split_sizes(sample_housing_data):
    """Train and test sets should sum to the original number of rows."""
    train, test = create_train_test_split(sample_housing_data, test_size=0.2)
    assert len(train) + len(test) == len(sample_housing_data)


def test_create_train_test_split_no_overlap(sample_housing_data):
    """Train and test sets must not share any row indices."""
    train, test = create_train_test_split(sample_housing_data, test_size=0.2)
    assert set(train.index).isdisjoint(set(test.index))


def test_create_train_test_split_drops_income_cat(sample_housing_data):
    """The helper column income_cat must not leak into the output sets."""
    train, test = create_train_test_split(sample_housing_data)
    assert "income_cat" not in train.columns
    assert "income_cat" not in test.columns


def test_create_train_test_split_reproducible(sample_housing_data):
    """Same random_state must produce identical splits."""
    train1, test1 = create_train_test_split(sample_housing_data, random_state=0)
    train2, test2 = create_train_test_split(sample_housing_data, random_state=0)
    pd.testing.assert_frame_equal(train1.reset_index(drop=True), train2.reset_index(drop=True))
    pd.testing.assert_frame_equal(test1.reset_index(drop=True), test2.reset_index(drop=True))


def test_get_features_and_target_shapes(sample_housing_data):
    """X and y should have correct shapes after splitting target from features."""
    X, y = get_features_and_target(sample_housing_data)
    assert "median_house_value" not in X.columns
    assert len(y) == len(sample_housing_data)
    assert y.name == "median_house_value"
