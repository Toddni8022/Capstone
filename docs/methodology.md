# ML Methodology

## Overview

This project applies supervised regression to the **California Housing dataset** with the goal of predicting median house values for census block groups.

---

## 1. Data Acquisition

The dataset is downloaded programmatically from the StatLib repository (via Aurélien Géron's Hands-On ML data mirror). It contains ~20,640 rows and 10 columns derived from the 1990 California census.

---

## 2. Exploratory Data Analysis (EDA)

- Distribution plots for all numerical features
- Geographical scatter map (longitude vs latitude coloured by `median_house_value`)
- Pearson correlation matrix to identify linear relationships with the target
- Key insight: `median_income` has the strongest positive correlation with `median_house_value`

---

## 3. Data Preprocessing

### Train / Test Split
A **stratified shuffle split** (20% test) on income categories ensures representative income distribution in both sets.

### Imputation
Missing values in `total_bedrooms` (~207 rows) are filled with the **median** using `sklearn.impute.SimpleImputer`.

### Scaling
All numerical features are standardised with **`StandardScaler`** (zero mean, unit variance).

### Encoding
The single categorical feature (`ocean_proximity`) is encoded with **`OneHotEncoder`**.

---

## 4. Feature Engineering

Three ratio features are added by `CombinedAttributesAdder`:

| Feature | Formula |
|---------|---------|
| `rooms_per_household` | `total_rooms / households` |
| `population_per_household` | `population / households` |
| `bedrooms_per_room` | `total_bedrooms / total_rooms` |

These ratio features significantly improve model performance compared to the raw counts.

---

## 5. Model Selection

Three model families were compared using 5-fold cross-validation RMSE:

| Model | Cross-Val RMSE |
|-------|---------------|
| Linear Regression | ~$68,000 |
| Decision Tree | ~$71,000 (overfit) |
| **Random Forest** | **~$50,000** ✓ |

The **Random Forest Regressor** was selected as the best-performing model.

---

## 6. Hyperparameter Tuning

Grid Search and Randomized Search with cross-validation were used to tune:

- `n_estimators` — number of trees
- `max_features` — number of features considered at each split
- `bootstrap` — whether to use bootstrap samples

The best configuration was identified and used for the final model.

---

## 7. Evaluation

The final model is evaluated on the held-out test set:

- **Metric:** Root Mean Squared Error (RMSE)
- **Test RMSE:** ~$47,000–$50,000

Feature importance analysis confirms that `median_income` is the most predictive feature, followed by the engineered `bedrooms_per_room` ratio.

---

## 8. Model Persistence

The final pipeline (preprocessor + model) is saved to disk using **`joblib`** for production deployment.
