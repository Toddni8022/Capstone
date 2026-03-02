# Capstone — End-to-End Machine Learning Project

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/Toddni8022/Capstone/actions/workflows/ci.yml/badge.svg)

An academic capstone project demonstrating a complete end-to-end machine learning pipeline applied to the **California Housing dataset**. Covers data ingestion, exploratory analysis, feature engineering, model training, hyperparameter tuning, evaluation, and deployment-ready persistence.

---

## Problem Statement

Predict the **median house value** for California census block groups using demographic and geographic features from the 1990 census. This is a supervised regression problem.

---

## Dataset

- **Name:** California Housing Dataset (StatLib / Aurélien Géron)
- **Rows:** ~20,640 census block groups
- **Features:** 9 (longitude, latitude, housing age, rooms, bedrooms, population, households, income, ocean proximity)
- **Target:** `median_house_value` (USD)
- See [`data/README.md`](data/README.md) for full details and how to obtain the data.

---

## Methodology

1. **Data Acquisition** — programmatic download and extraction
2. **EDA** — distributions, geographical maps, correlation analysis
3. **Preprocessing** — stratified split, median imputation, standard scaling, one-hot encoding
4. **Feature Engineering** — ratio features (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`)
5. **Model Selection** — Linear Regression, Decision Tree, Random Forest (CV RMSE comparison)
6. **Hyperparameter Tuning** — Grid Search + Randomized Search with cross-validation
7. **Evaluation** — RMSE on held-out test set, feature importance analysis
8. **Persistence** — trained pipeline saved with `joblib`

See [`docs/methodology.md`](docs/methodology.md) for full details.

---

## Key Results

| Model | Cross-Val RMSE |
|-------|---------------|
| Linear Regression | ~$68,000 |
| Decision Tree | ~$71,000 |
| **Random Forest (tuned)** | **~$47,000** ✓ |

Top features: `median_income`, `bedrooms_per_room` (engineered), `longitude`, `latitude`.

See [`docs/results.md`](docs/results.md) for full analysis.

---

## Project Structure

```
Capstone/
├── .github/workflows/ci.yml   # CI/CD pipeline (lint + test + notebook)
├── data/
│   ├── raw/                   # Original downloaded data (gitignored)
│   ├── processed/             # Cleaned data (gitignored)
│   └── README.md
├── docs/
│   ├── methodology.md         # ML methodology
│   ├── results.md             # Key findings and metrics
│   └── setup.md               # Environment setup guide
├── models/                    # Saved model artifacts (gitignored)
├── notebooks/
│   ├── 02_end_to_end_machine_learning_project.ipynb
│   └── README.md
├── reports/
│   ├── Capstone_Presentation.pdf
│   ├── figures/               # Generated plots (gitignored)
│   └── README.md
├── src/capstone/              # Reusable Python package
│   ├── __init__.py
│   ├── config.py              # Constants and paths
│   ├── data.py                # Data loading and splitting
│   ├── features.py            # Feature engineering pipeline
│   ├── models.py              # Model building and evaluation
│   └── visualize.py           # Plotting utilities
├── tests/                     # Unit tests
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── .editorconfig
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── README.md
├── environment.yml            # Conda environment
├── pyproject.toml
├── requirements.txt
└── setup.py
```

---

## Prerequisites

- Python 3.8+
- pip or conda
- Git

---

## Installation

### pip

```bash
git clone https://github.com/Toddni8022/Capstone.git
cd Capstone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### conda

```bash
git clone https://github.com/Toddni8022/Capstone.git
cd Capstone
conda env create -f environment.yml
conda activate capstone-ml
pip install -e .
```

Or simply:

```bash
make setup
```

---

## How to Run the Notebook

```bash
# Download the dataset
make data

# Launch Jupyter
jupyter notebook notebooks/02_end_to_end_machine_learning_project.ipynb
```

---

## Reproduce Results

```bash
make setup   # install dependencies
make data    # download dataset
# Open and run the notebook top-to-bottom
jupyter notebook notebooks/02_end_to_end_machine_learning_project.ipynb
```

---

## Running Tests

```bash
make test
# or
pytest tests/ -v
```

---

## Linting

```bash
make lint
# or
flake8 src/ tests/
```

---

## Presentation

The project slide deck is available at [`reports/Capstone_Presentation.pdf`](reports/Capstone_Presentation.pdf).

---

## Future Work

- Add XGBoost / LightGBM models for comparison
- Use SHAP values for model explainability
- Build a simple FastAPI inference endpoint
- Add data versioning with DVC
- Extend to more recent California housing data

---

## Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License — see the [`LICENSE`](LICENSE) file for details.
