# Model Iterations Documentation
## Task: Used Car Price Prediction (Regression)

Dataset: AutoScout24 Germany `autoscout24_germany.csv` (46 405 rows). Rows after cleaning: **43725**.
Target: `price` (EUR). Reported metrics are on the original EUR scale.

| Iteration | Objective | Best model | CV R² (mean ± std) | CV RMSE (EUR, mean) | CV MAE (EUR, mean) | Fit diagnosis |
|---|---|---|---|---|---|---|
| 1 | Establish baseline with raw structured features only | RandomForestRegressor | 0.9246 ± 0.0025 | 3656 | 2000 | Good Fit |
| 2 | Improve generalization with engineered numeric features | RandomForestRegressor | 0.9237 ± 0.0039 | 3677 | 2007 | Good Fit |
| 3 | Stabilize cross-validation variance and reduce error on expensive cars | GradientBoostingRegressor_Tuned | 0.9274 ± 0.0036 | 3587 | 1950 | Good Fit |

## Detailed iteration breakdown

### Iteration 1: Establish baseline with raw structured features only
- Changes vs. previous iteration: Used `year`, `mileage_km`, `hp` and one-hot brand/fuel/gear/offerType/body_type. No engineered features, no target transform.
- Features used (8): `year`, `mileage_km`, `hp`, `brand`, `fuel`, `gear`, `offerType`, `body_type`
- Per-model 5-fold CV results:

| Model | R² (mean) | R² (std) | RMSE (EUR, mean) | MAE (EUR, mean) |
|---|---:|---:|---:|---:|
| Ridge | 0.8539 | 0.0060 | 5090 | 3186 |
| RandomForestRegressor ⭐ | 0.9246 | 0.0025 | 3656 | 2000 |

### Iteration 2: Improve generalization with engineered numeric features
- Changes vs. previous iteration: Added `car_age = CURRENT_YEAR - year`, `km_per_year`, `hp_per_year`. Mapped each brand to a default body type and use that as an additional categorical feature.
- Features used (10): `car_age`, `mileage_km`, `km_per_year`, `hp`, `hp_per_year`, `brand`, `fuel`, `gear`, `offerType`, `body_type`
- Per-model 5-fold CV results:

| Model | R² (mean) | R² (std) | RMSE (EUR, mean) | MAE (EUR, mean) |
|---|---:|---:|---:|---:|
| Ridge | 0.8964 | 0.0032 | 4286 | 2522 |
| RandomForestRegressor ⭐ | 0.9237 | 0.0039 | 3677 | 2007 |

### Iteration 3: Stabilize cross-validation variance and reduce error on expensive cars
- Changes vs. previous iteration: Trained on log1p(price) so percentage errors are penalised evenly across the long tail. Tuned RandomForest (n=400, max_depth=24) and tuned GradientBoosting (lr=0.05, n=500, max_depth=5).
- Features used (10): `car_age`, `mileage_km`, `km_per_year`, `hp`, `hp_per_year`, `brand`, `fuel`, `gear`, `offerType`, `body_type`
- Per-model 5-fold CV results:

| Model | R² (mean) | R² (std) | RMSE (EUR, mean) | MAE (EUR, mean) |
|---|---:|---:|---:|---:|
| RandomForestRegressor_Tuned | 0.9230 | 0.0044 | 3696 | 1977 |
| GradientBoostingRegressor_Tuned ⭐ | 0.9274 | 0.0036 | 3587 | 1950 |

## Final model
- Selected: **GradientBoostingRegressor_Tuned** (R² = 0.9274)
- Trained on `log1p(price)`; the app inverts with `expm1`.
- Saved to `artifacts/final_model.joblib`. Metadata in `artifacts/metadata.json`.