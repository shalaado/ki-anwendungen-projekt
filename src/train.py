"""Train and compare price models for the AutoPrice Pro project.

Runs three documented iterations on the AutoScout24 Germany used-car dataset:

    1. Baseline: raw structured columns, no engineered features, no target
       transform.
    2. Feature engineering: adds ``car_age``, ``km_per_year``, ``hp_per_year``
       and a brand→body-type lookup.
    3. Hyperparameter tuning: same features as iteration 2 but trained on
       ``log1p(price)`` to dampen the heavy right tail; tuned RandomForest
       and GradientBoosting.

After training the best model is persisted to ``artifacts/final_model.joblib``
together with the metadata that the Gradio app needs at inference time.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_processing import (
    ALL_FEATURES,
    BODY_TYPE_SEATS_DEFAULT,
    BRAND_BODY_DEFAULTS,
    CATEGORICAL_FEATURES,
    CURRENT_YEAR,
    NUMERIC_FEATURES,
    TARGET,
    add_engineered_features,
)


def _find_project_root() -> Path:
    """Locate the directory that contains ``app.py`` and ``src/``.

    Identical logic to before — the env var lets ``app.py`` pin the root
    down explicitly so Hugging Face Spaces' symlinked work-dir does not
    collapse ``Path(__file__).resolve().parents[1]`` to ``/``.
    """
    env_root = os.environ.get("AUTOPRICE_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root)
        if (candidate / "app.py").is_file():
            return candidate

    try:
        here = Path(__file__).resolve()
        for parent in [here.parent, *here.parents]:
            if (parent / "app.py").is_file() and (parent / "src").is_dir():
                return parent
    except (NameError, OSError):
        pass

    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "app.py").is_file() and (parent / "src").is_dir():
            return parent

    for hardcoded in (Path("/app"), Path("/home/user/app")):
        if (hardcoded / "app.py").is_file():
            return hardcoded

    return Path.cwd()


PROJECT_ROOT = _find_project_root()
DATA_PATH = PROJECT_ROOT / "data" / "autoscout24_germany.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/leander-ms/autoscout_Analysis/"
    "main/autoscout24-germany-dataset.csv"
)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42


def ensure_dataset(path: Path = DATA_PATH, url: str = DATA_URL) -> Path:
    """Download the AutoScout24 CSV on the fly if it is missing.

    This is what makes the Hugging Face Space self-bootstrapping: even if
    only the Python files were uploaded, the training run can still fetch
    the data.
    """
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Dataset not found locally — downloading from {url}")
    urllib.request.urlretrieve(url, path)
    print(f"Saved {path} ({path.stat().st_size / 1024:.0f} KB)")
    return path


@dataclass
class IterationResult:
    iteration: int
    objective: str
    changes: str
    features_used: list[str]
    models: dict[str, dict[str, float]]
    best_model: str
    fit_diagnosis: str

    def best_metrics(self) -> dict[str, float]:
        return self.models[self.best_model]


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop rows where target is missing or extreme (top/bottom 0.5%)
    df = df.dropna(subset=[TARGET]).copy()
    low, high = df[TARGET].quantile([0.005, 0.995])
    df = df.loc[(df[TARGET] >= low) & (df[TARGET] <= high)].copy()

    # Filter to plausible numeric ranges
    df = df.loc[(df["mileage"] >= 0) & (df["mileage"] <= 500_000)]
    df = df.loc[(df["year"] >= 1980) & (df["year"] <= CURRENT_YEAR)]
    df = df.loc[df["hp"].notna() & (df["hp"] >= 30) & (df["hp"] <= 1500)]

    # Drop categorical garbage (e.g. trailer/truck rows in the make column)
    df = df.loc[df["make"].notna()]
    junk_makes = {"Trailer-Anh�nger", "Caravans-Wohnm", "Trucks-Lkw"}
    df = df.loc[~df["make"].isin(junk_makes)]

    # Drop fully duplicated rows
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def make_baseline_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Iteration 1: raw structured columns only — no engineered features."""
    work = df.copy()
    work["brand"] = work["make"].astype(str).str.strip()
    work["body_type"] = work["brand"].map(BRAND_BODY_DEFAULTS).fillna("Sedan")
    work["mileage_km"] = work["mileage"]

    numeric = ["year", "mileage_km", "hp"]
    categorical = ["brand", "fuel", "gear", "offerType", "body_type"]
    features = numeric + categorical
    return work, features, numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )


def cv_score_model(
    df: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    model,
    use_log_target: bool,
) -> dict[str, float]:
    """5-fold CV. Returns mean/std for R2, RMSE, MAE (always on EUR scale)."""
    X = df[features].reset_index(drop=True)
    y_eur = df[TARGET].astype(float).reset_index(drop=True)
    y_fit = np.log1p(y_eur) if use_log_target else y_eur

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric, categorical)),
            ("regressor", model),
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    r2_scores, rmse_scores, mae_scores = [], [], []

    for train_idx, val_idx in cv.split(X):
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X.iloc[train_idx], y_fit.iloc[train_idx])
        preds = fold_pipe.predict(X.iloc[val_idx])
        if use_log_target:
            preds = np.expm1(preds)
        actual = y_eur.iloc[val_idx]
        residuals = preds - actual
        rmse_scores.append(float(np.sqrt(np.mean(residuals ** 2))))
        mae_scores.append(float(np.mean(np.abs(residuals))))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        r2_scores.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)

    return {
        "cv_r2_mean": float(np.mean(r2_scores)),
        "cv_r2_std": float(np.std(r2_scores)),
        "cv_rmse_mean": float(np.mean(rmse_scores)),
        "cv_rmse_std": float(np.std(rmse_scores)),
        "cv_mae_mean": float(np.mean(mae_scores)),
        "cv_mae_std": float(np.std(mae_scores)),
    }


def diagnose_fit(r2_mean: float) -> str:
    if r2_mean >= 0.80:
        return "Good Fit"
    if r2_mean >= 0.60:
        return "Acceptable (mild overfit possible)"
    return "Underfitting"


def main() -> None:
    ensure_dataset()
    print(f"Loading {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    print(f"Raw rows: {len(raw)}")

    cleaned = load_and_clean(DATA_PATH)
    print(f"Rows after cleaning: {len(cleaned)}")

    iteration_results: list[IterationResult] = []

    # ---------- Iteration 1: baseline ----------
    print("\nIteration 1: baseline (raw structured features, linear target)")
    base_df, base_features, base_num, base_cat = make_baseline_features(cleaned)

    it1_models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    it1_scores: dict[str, dict[str, float]] = {}
    for name, model in it1_models.items():
        it1_scores[name] = cv_score_model(
            base_df, base_features, base_num, base_cat, model, use_log_target=False
        )
        print(f"  {name}: R2={it1_scores[name]['cv_r2_mean']:.4f}, "
              f"RMSE={it1_scores[name]['cv_rmse_mean']:.0f} EUR")

    best_it1 = max(it1_scores, key=lambda n: it1_scores[n]["cv_r2_mean"])
    iteration_results.append(
        IterationResult(
            iteration=1,
            objective="Establish baseline with raw structured features only",
            changes=(
                "Used `year`, `mileage_km`, `hp` and one-hot brand/fuel/gear/"
                "offerType/body_type. No engineered features, no target transform."
            ),
            features_used=base_features,
            models=it1_scores,
            best_model=best_it1,
            fit_diagnosis=diagnose_fit(it1_scores[best_it1]["cv_r2_mean"]),
        )
    )

    # ---------- Iteration 2: feature engineering ----------
    print("\nIteration 2: feature engineering (car_age, km_per_year, hp_per_year)")
    eng_df = add_engineered_features(cleaned, current_year=CURRENT_YEAR)
    it2_models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    it2_scores: dict[str, dict[str, float]] = {}
    for name, model in it2_models.items():
        it2_scores[name] = cv_score_model(
            eng_df, ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, model,
            use_log_target=False,
        )
        print(f"  {name}: R2={it2_scores[name]['cv_r2_mean']:.4f}, "
              f"RMSE={it2_scores[name]['cv_rmse_mean']:.0f} EUR")

    best_it2 = max(it2_scores, key=lambda n: it2_scores[n]["cv_r2_mean"])
    iteration_results.append(
        IterationResult(
            iteration=2,
            objective="Improve generalization with engineered numeric features",
            changes=(
                "Added `car_age = CURRENT_YEAR - year`, `km_per_year`, "
                "`hp_per_year`. Mapped each brand to a default body type and "
                "use that as an additional categorical feature."
            ),
            features_used=list(ALL_FEATURES),
            models=it2_scores,
            best_model=best_it2,
            fit_diagnosis=diagnose_fit(it2_scores[best_it2]["cv_r2_mean"]),
        )
    )

    # ---------- Iteration 3: log target + tuning ----------
    print("\nIteration 3: log-target + tuned models (GradientBoosting and RandomForest)")
    it3_models = {
        "RandomForestRegressor_Tuned": RandomForestRegressor(
            n_estimators=400,
            max_depth=24,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor_Tuned": GradientBoostingRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            random_state=RANDOM_STATE,
        ),
    }
    it3_scores: dict[str, dict[str, float]] = {}
    for name, model in it3_models.items():
        it3_scores[name] = cv_score_model(
            eng_df, ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, model,
            use_log_target=True,
        )
        print(f"  {name}: R2={it3_scores[name]['cv_r2_mean']:.4f}, "
              f"RMSE={it3_scores[name]['cv_rmse_mean']:.0f} EUR")

    best_it3 = max(it3_scores, key=lambda n: it3_scores[n]["cv_r2_mean"])
    iteration_results.append(
        IterationResult(
            iteration=3,
            objective="Stabilize cross-validation variance and reduce error on expensive cars",
            changes=(
                "Trained on log1p(price) so percentage errors are penalised "
                "evenly across the long tail. Tuned RandomForest (n=400, max_depth=24) "
                "and tuned GradientBoosting (lr=0.05, n=500, max_depth=5)."
            ),
            features_used=list(ALL_FEATURES),
            models=it3_scores,
            best_model=best_it3,
            fit_diagnosis=diagnose_fit(it3_scores[best_it3]["cv_r2_mean"]),
        )
    )

    # ---------- Pick final model and persist ----------
    final_iteration = iteration_results[-1]
    final_model = it3_models[final_iteration.best_model]
    print(f"\nFinal model selected: {final_iteration.best_model}")
    print(f"  cv R2 = {final_iteration.best_metrics()['cv_r2_mean']:.4f}")
    print(f"  cv RMSE = {final_iteration.best_metrics()['cv_rmse_mean']:.0f} EUR")

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
            ("regressor", final_model),
        ]
    )

    X_final = eng_df[ALL_FEATURES]
    y_final_log = np.log1p(eng_df[TARGET].astype(float))
    final_pipeline.fit(X_final, y_final_log)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "final_model.joblib"
    joblib.dump(final_pipeline, model_path)

    # Per-brand medians so the app can fill plausible numeric values when
    # the user (or LLM) does not specify them.
    brand_defaults = (
        eng_df.groupby("brand")[NUMERIC_FEATURES]
        .median(numeric_only=True)
        .reset_index()
        .sort_values("brand")
    )
    brand_defaults.to_csv(ARTIFACTS_DIR / "brand_defaults.csv", index=False)

    body_defaults = pd.DataFrame(
        [
            {"body_type": bt, "default_seats": int(seats)}
            for bt, seats in BODY_TYPE_SEATS_DEFAULT.items()
        ]
    )
    body_defaults.to_csv(ARTIFACTS_DIR / "body_type_defaults.csv", index=False)

    metadata = {
        "target": TARGET,
        "target_unit": "EUR",
        "currency_note": "Euros (EUR). At inference the app shows EUR and an indicative CHF figure.",
        "use_log_target": True,
        "current_year": CURRENT_YEAR,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": ALL_FEATURES,
        "model_name": final_iteration.best_model,
        "dataset_used": DATA_PATH.name,
        "rows_after_cleaning": int(len(cleaned)),
        "iteration_summary_file": "artifacts/model_iterations.md",
        "valid_brands": sorted(eng_df["brand"].unique().tolist()),
        "fuel_categories": sorted(eng_df["fuel"].unique().tolist()),
        "gear_categories": sorted(eng_df["gear"].dropna().unique().tolist()),
        "offerType_categories": sorted(eng_df["offerType"].unique().tolist()),
        "body_type_categories": sorted(BODY_TYPE_SEATS_DEFAULT.keys()),
        "final_metrics": final_iteration.best_metrics(),
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------- Iteration markdown ----------
    lines = [
        "# Model Iterations Documentation",
        "## Task: Used Car Price Prediction (Regression)",
        "",
        "Dataset: AutoScout24 Germany `autoscout24_germany.csv` (46 405 rows). "
        f"Rows after cleaning: **{len(cleaned)}**.",
        "Target: `price` (EUR). Reported metrics are on the original EUR scale.",
        "",
        "| Iteration | Objective | Best model | CV R² (mean ± std) | CV RMSE (EUR, mean) | CV MAE (EUR, mean) | Fit diagnosis |",
        "|---|---|---|---|---|---|---|",
    ]
    for it in iteration_results:
        m = it.best_metrics()
        lines.append(
            f"| {it.iteration} | {it.objective} | {it.best_model} | "
            f"{m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f} | "
            f"{m['cv_rmse_mean']:.0f} | "
            f"{m['cv_mae_mean']:.0f} | "
            f"{it.fit_diagnosis} |"
        )

    lines.append("")
    lines.append("## Detailed iteration breakdown")
    for it in iteration_results:
        lines.append(f"\n### Iteration {it.iteration}: {it.objective}")
        lines.append(f"- Changes vs. previous iteration: {it.changes}")
        lines.append(f"- Features used ({len(it.features_used)}): "
                     f"{', '.join('`' + f + '`' for f in it.features_used)}")
        lines.append("- Per-model 5-fold CV results:")
        lines.append("")
        lines.append("| Model | R² (mean) | R² (std) | RMSE (EUR, mean) | MAE (EUR, mean) |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, m in it.models.items():
            marker = " ⭐" if name == it.best_model else ""
            lines.append(
                f"| {name}{marker} | {m['cv_r2_mean']:.4f} | "
                f"{m['cv_r2_std']:.4f} | {m['cv_rmse_mean']:.0f} | "
                f"{m['cv_mae_mean']:.0f} |"
            )

    lines.append("")
    lines.append("## Final model")
    lines.append(
        f"- Selected: **{final_iteration.best_model}** "
        f"(R² = {final_iteration.best_metrics()['cv_r2_mean']:.4f})"
    )
    lines.append("- Trained on `log1p(price)`; the app inverts with `expm1`.")
    lines.append(
        "- Saved to `artifacts/final_model.joblib`. "
        "Metadata in `artifacts/metadata.json`."
    )

    (ARTIFACTS_DIR / "model_iterations.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"\nArtifacts written to {ARTIFACTS_DIR}")
    print(" - final_model.joblib")
    print(" - metadata.json")
    print(" - brand_defaults.csv")
    print(" - body_type_defaults.csv")
    print(" - model_iterations.md")


if __name__ == "__main__":
    main()
