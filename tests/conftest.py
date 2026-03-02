"""Shared pytest fixtures for the Capstone test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_housing_data() -> pd.DataFrame:
    """Return a small synthetic housing DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "longitude": rng.uniform(-124, -114, n),
            "latitude": rng.uniform(32, 42, n),
            "housing_median_age": rng.integers(1, 52, n).astype(float),
            "total_rooms": rng.integers(100, 5000, n).astype(float),
            "total_bedrooms": rng.integers(20, 1000, n).astype(float),
            "population": rng.integers(50, 3000, n).astype(float),
            "households": rng.integers(20, 900, n).astype(float),
            "median_income": rng.uniform(0.5, 15.0, n),
            "median_house_value": rng.uniform(20000, 500000, n),
            "ocean_proximity": rng.choice(
                ["NEAR BAY", "INLAND", "<1H OCEAN", "NEAR OCEAN", "ISLAND"], n
            ),
        }
    )
