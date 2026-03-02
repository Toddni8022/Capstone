# Notebooks

This directory contains all Jupyter notebooks for the Capstone ML project.

## Overview

| Notebook | Description |
|----------|-------------|
| `02_end_to_end_machine_learning_project.ipynb` | Full end-to-end ML pipeline: data ingestion, EDA, feature engineering, model training, evaluation, and persistence. Based on Chapter 2 of *Hands-On Machine Learning* (Aurélien Géron). |

## Running the Notebooks

1. Install dependencies (see [setup instructions](../docs/setup.md)):
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `02_end_to_end_machine_learning_project.ipynb` in your browser.

## Notebook Descriptions

### `02_end_to_end_machine_learning_project.ipynb`

Walks through the complete ML workflow applied to the **California Housing dataset**:

- **Data Acquisition** — downloads the housing dataset via URL
- **Exploratory Data Analysis** — distributions, geographical plots, correlation analysis
- **Data Preprocessing** — stratified splitting, imputation, scaling, one-hot encoding
- **Feature Engineering** — derived ratio features (rooms/household, population/household, bedrooms/room)
- **Model Training** — Linear Regression, Decision Trees, Random Forest
- **Hyperparameter Tuning** — Grid Search and Randomized Search with cross-validation
- **Model Evaluation** — RMSE on test set, feature importance analysis
- **Model Persistence** — saving and loading the final model with `joblib`
- **Exercises** — SVR, SelectFromModel, KNN custom transformer
