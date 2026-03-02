# Models

This directory stores trained model artifacts produced by the Capstone ML pipeline.

## Directory Structure

```
models/
└── *.pkl    # Serialised sklearn Pipeline objects saved with joblib
```

## Saved Models

Model files are generated when you run the notebook or training scripts. They are **not** committed to version control.

### Naming Convention

| File | Description |
|------|-------------|
| `random_forest.pkl` | Final Random Forest pipeline (preprocessor + model) |
| `linear_regression.pkl` | Baseline Linear Regression pipeline |

## Loading a Model

```python
from capstone.models import load_model

pipeline = load_model("random_forest.pkl")
predictions = pipeline.predict(X_test)
```

## Notes

- Model files (`.pkl`, `.joblib`) are excluded from version control via `.gitignore`.
- Only the `.gitkeep` placeholder is committed to maintain the directory structure.
- Retrain models by running `make train` or executing the main notebook.
