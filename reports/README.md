# Reports

This directory contains reports, presentations, and generated figures for the Capstone ML project.

## Directory Structure

```
reports/
├── Capstone_Presentation.pdf   # Project slide deck
└── figures/                    # Auto-generated plots and visualizations
```

## Presentation

**`Capstone_Presentation.pdf`** — Slide deck summarising the project, methodology, key findings, and results.

## Figures

Generated figures are saved to `reports/figures/` by calling `visualize` utilities with `save=True`, or via:

```bash
make visualize   # (if defined in the Makefile)
```

| Figure | Description |
|--------|-------------|
| `housing_map.png` | Geographical scatter of California housing prices |
| `correlation_matrix.png` | Feature correlation heatmap |
| `feature_importances.png` | Top feature importances from the Random Forest |
| `predictions_vs_actual.png` | Scatter of model predictions vs ground truth |

## Notes

- Generated figures (`.png`, `.jpg`, `.svg`) are excluded from version control via `.gitignore`.
- Only `.gitkeep` placeholders are committed to maintain directory structure.
