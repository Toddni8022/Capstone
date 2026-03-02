"""Data loading and preprocessing utilities."""

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from capstone.config import (
    DATASET_URL,
    HOUSING_PATH,
    HOUSING_TGZ,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def fetch_housing_data(url: str = DATASET_URL, download_path: Path = HOUSING_TGZ) -> None:
    """Download and extract the housing dataset.

    Parameters
    ----------
    url:
        URL of the compressed dataset archive.
    download_path:
        Local path where the archive is saved.
    """
    download_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, download_path)
    with tarfile.open(download_path) as tgz:
        tgz.extractall(path=download_path.parent)


def load_housing_data(housing_path: Path = HOUSING_PATH) -> pd.DataFrame:
    """Load the housing CSV from disk.

    Parameters
    ----------
    housing_path:
        Directory that contains ``housing.csv``.

    Returns
    -------
    pd.DataFrame
        Raw housing data.
    """
    csv_path = housing_path / "housing.csv"
    return pd.read_csv(csv_path)


def create_train_test_split(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified train/test split on median income category.

    Parameters
    ----------
    data:
        Full housing DataFrame.
    test_size:
        Fraction of data reserved for the test set.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_set, test_set)``
    """
    data = data.copy()
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(data, data["income_cat"]):
        train_set = data.iloc[train_index].drop("income_cat", axis=1)
        test_set = data.iloc[test_index].drop("income_cat", axis=1)

    return train_set, test_set


def get_features_and_target(
    data: pd.DataFrame,
    target: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature columns from the target column.

    Parameters
    ----------
    data:
        DataFrame containing both features and target.
    target:
        Name of the target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)``
    """
    X = data.drop(target, axis=1)
    y = data[target].copy()
    return X, y
