# Key Results

## Summary

The final **Random Forest Regressor** pipeline achieves approximately **$47,000–$50,000 RMSE** on the California Housing test set, representing a median relative error of roughly **±10–12%** on the target distribution.

---

## Model Performance

| Model | Cross-Val RMSE | Test RMSE |
|-------|---------------|-----------|
| Linear Regression | ~$68,000 | ~$66,000 |
| Decision Tree | ~$71,000 | overfit |
| Random Forest (default) | ~$50,000 | — |
| **Random Forest (tuned)** | **~$47,000** | **~$47,500** |

*RMSE values are approximate and may vary with dataset version and random seed.*

---

## Top Predictive Features

Based on the Random Forest feature importances:

1. `median_income` — strongest predictor of housing value
2. `bedrooms_per_room` (engineered) — compact neighbourhood indicator
3. `longitude` and `latitude` — geographical location
4. `housing_median_age` — age of housing stock
5. `rooms_per_household` (engineered) — household size proxy

---

## Insights

- **Location matters:** Coastal proximity and coordinates strongly correlate with value.
- **Income dominates:** `median_income` alone explains a significant portion of variance.
- **Engineered features help:** The ratio features (`bedrooms_per_room`, `rooms_per_household`) outperform the raw count features they are derived from.
- **Tree ensembles win:** Random Forest outperforms Linear Regression and Decision Trees by a large margin.

---

## Limitations

- The dataset is from the **1990 California census** and does not reflect current market conditions.
- Prices are capped at $500,000 in the original dataset, which truncates the upper end of the distribution.
- No temporal features — no information about trends over time.

---

## Future Work

See [../README.md#future-work](../README.md#future-work) for planned improvements.
