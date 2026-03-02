# Data

This directory stores dataset files used in the Capstone ML project.

## Directory Structure

```
data/
├── raw/          # Original, immutable data as downloaded
└── processed/    # Cleaned and transformed data ready for modelling
```

## Dataset

**California Housing Dataset**

- **Source:** StatLib repository via the book *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (Aurélien Géron)
- **URL:** `https://raw.githubusercontent.com/ageron/data/main/housing.tgz`
- **Format:** CSV (`housing.csv`)
- **Rows:** ~20,640 California census block groups (1990 census)
- **Target:** `median_house_value` — median house value for households in a block group (USD)

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `longitude` | float | Block group longitude |
| `latitude` | float | Block group latitude |
| `housing_median_age` | float | Median age of houses in the block group |
| `total_rooms` | float | Total number of rooms in the block group |
| `total_bedrooms` | float | Total number of bedrooms (has missing values) |
| `population` | float | Block group population |
| `households` | float | Number of households |
| `median_income` | float | Median household income (tens of thousands of USD) |
| `ocean_proximity` | str | Categorical proximity to the ocean |
| `median_house_value` | float | **Target** — median house value (USD) |

## How to Obtain the Data

Run the following in a Python environment with the project installed:

```python
from capstone.data import fetch_housing_data
fetch_housing_data()
```

Or use the Makefile:

```bash
make data
```

This will download and extract `housing.csv` into `data/raw/housing/`.

## Notes

- Raw data files are excluded from version control via `.gitignore`.
- Only `.gitkeep` placeholder files are committed to maintain directory structure.
