"""Configuration and constants for the Capstone ML project."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model artifacts directory
MODELS_DIR = PROJECT_ROOT / "models"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dataset settings
DATASET_URL = (
    "https://raw.githubusercontent.com/ageron/data/main/housing.tgz"
)
HOUSING_PATH = RAW_DATA_DIR / "housing"
HOUSING_TGZ = RAW_DATA_DIR / "housing.tgz"

# Random seed for reproducibility
RANDOM_STATE = 42

# Train / test split ratio
TEST_SIZE = 0.2

# Cross-validation folds
CV_FOLDS = 5

# Target column
TARGET_COLUMN = "median_house_value"

# Categorical column(s)
CATEGORICAL_COLUMNS = ["ocean_proximity"]

# Numerical columns (derived at runtime, listed for reference)
NUMERICAL_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
