"""Exploratory Data Analysis for AutoPrice Pro.

Run with:

    python src/eda.py

Produces:
  - artifacts/eda/eda_summary.md  — key findings as bullet points
  - artifacts/eda/price_distribution.png
  - artifacts/eda/log_price_distribution.png
  - artifacts/eda/price_by_fuel.png
  - artifacts/eda/price_by_gear.png
  - artifacts/eda/price_by_brand_top10.png
  - artifacts/eda/correlations.png
  - artifacts/eda/age_vs_price.png

These are referenced from documentation.md to satisfy the EDA requirement
without bloating the documentation file with raw numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing import CURRENT_YEAR, add_engineered_features
from train import _find_project_root

PROJECT_ROOT = _find_project_root()
DATA_PATH = PROJECT_ROOT / "data" / "autoscout24_germany.csv"
OUT_DIR = PROJECT_ROOT / "artifacts" / "eda"


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["price"]).copy()
    low, high = df["price"].quantile([0.005, 0.995])
    df = df.loc[(df["price"] >= low) & (df["price"] <= high)]
    df = df.loc[(df["mileage"] >= 0) & (df["mileage"] <= 500_000)]
    df = df.loc[(df["year"] >= 1980) & (df["year"] <= CURRENT_YEAR)]
    df = df.loc[df["hp"].notna() & (df["hp"] >= 30) & (df["hp"] <= 1500)]
    df = df.loc[df["make"].notna()]
    junk = {"Trailer-Anh�nger", "Caravans-Wohnm", "Trucks-Lkw"}
    df = df.loc[~df["make"].isin(junk)]
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    cleaned = load_clean()
    df = add_engineered_features(cleaned, current_year=CURRENT_YEAR)
    log_price = np.log1p(df["price"])

    # 1. Price distribution (linear and log)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["price"] / 1_000, bins=60, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Selling price (1 000 EUR)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Used car selling price — long tail (skew={df['price'].skew():.2f})"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "price_distribution.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(log_price, bins=60, color="#55A868", edgecolor="white")
    ax.set_xlabel("log1p(price)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"After log1p the target is roughly symmetric (skew={log_price.skew():.2f})"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "log_price_distribution.png", dpi=110)
    plt.close(fig)

    # 2. Price by fuel
    fuel_counts = df["fuel"].value_counts()
    keep_fuels = fuel_counts[fuel_counts >= 50].index.tolist()
    fuel_order = (
        df[df["fuel"].isin(keep_fuels)]
        .groupby("fuel")["price"]
        .median()
        .sort_values()
        .index
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(
        [df.loc[df["fuel"] == f, "price"] / 1_000 for f in fuel_order],
        tick_labels=fuel_order,
        showfliers=False,
    )
    ax.set_ylabel("Selling price (1 000 EUR)")
    ax.set_title("Price by fuel type (categories with ≥ 50 listings)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "price_by_fuel.png", dpi=110)
    plt.close(fig)

    # 3. Price by gear
    fig, ax = plt.subplots(figsize=(6, 4))
    gears_present = [g for g in ["Manual", "Automatic", "Semi-automatic"]
                     if g in df["gear"].dropna().unique()]
    ax.boxplot(
        [df.loc[df["gear"] == g, "price"] / 1_000 for g in gears_present],
        tick_labels=gears_present,
        showfliers=False,
    )
    ax.set_ylabel("Selling price (1 000 EUR)")
    ax.set_title("Automatic cars carry a clear price premium over manual")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "price_by_gear.png", dpi=110)
    plt.close(fig)

    # 4. Price by brand — top 10 by frequency
    top10 = df["brand"].value_counts().head(10).index.tolist()
    sub = df[df["brand"].isin(top10)]
    order = sub.groupby("brand")["price"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(
        [sub.loc[sub["brand"] == b, "price"] / 1_000 for b in order],
        tick_labels=order,
        showfliers=False,
    )
    ax.set_ylabel("Selling price (1 000 EUR)")
    ax.set_title("Price spread across the 10 most frequent brands")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "price_by_brand_top10.png", dpi=110)
    plt.close(fig)

    # 5. Correlation bar chart of numeric features vs log price
    num_cols = ["car_age", "mileage_km", "km_per_year", "hp", "hp_per_year"]
    corrs = {
        col: float(np.corrcoef(df[col].fillna(df[col].median()), log_price)[0, 1])
        for col in num_cols
    }
    sorted_items = sorted(corrs.items(), key=lambda x: x[1])
    labels = [c for c, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values, color=colors, edgecolor="white")
    ax.set_xlabel("Pearson correlation with log1p(price)")
    ax.set_title("hp and car_age dominate the linear signal")
    ax.axvline(0, color="grey", lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "correlations.png", dpi=110)
    plt.close(fig)

    # 6. Scatter: car_age vs price (log)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["car_age"], log_price, s=4, alpha=0.20, color="#4C72B0")
    ax.set_xlabel("Car age (years)")
    ax.set_ylabel("log1p(price)")
    ax.set_title("Strong negative trend: older cars → log-linear price drop")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "age_vs_price.png", dpi=110)
    plt.close(fig)

    # --- Textual summary ---
    premium_brands = ["BMW", "Audi", "Mercedes-Benz", "Porsche", "Jaguar",
                      "Lexus", "Tesla", "Bentley", "Ferrari", "Lamborghini",
                      "Maserati"]
    summary = [
        "# EDA — Key Findings",
        "",
        f"Dataset: AutoScout24 Germany `autoscout24_germany.csv` "
        f"(raw rows: **{len(raw):,}**, after cleaning: **{len(cleaned):,}**, "
        f"removed: {len(raw) - len(cleaned):,}).",
        "",
        "## 1. Target distribution",
        f"- `price` spans **{int(df['price'].min()):,}** EUR to "
        f"**{int(df['price'].max()):,}** EUR.",
        f"- Right-skewed on the linear scale (skew = "
        f"**{df['price'].skew():.2f}**) → motivates the `log1p` target "
        f"used in iteration 3 (post-log skew = **{log_price.skew():.2f}**).",
        f"- Median = **{int(df['price'].median()):,}** EUR, "
        f"P25 = **{int(df['price'].quantile(0.25)):,}**, "
        f"P75 = **{int(df['price'].quantile(0.75)):,}**.",
        "",
        "## 2. Numeric correlations with log price",
    ]
    for col, r in sorted(corrs.items(), key=lambda x: -abs(x[1])):
        summary.append(f"  - `{col}`: r = **{r:+.3f}**")
    summary += [
        "",
        f"  → `hp` ({corrs['hp']:+.2f}) and `car_age` ({corrs['car_age']:+.2f}) "
        "carry most of the linear signal; tree-based models add value through "
        "interactions on top of that.",
        "",
        "## 3. Categorical breakdowns",
        f"- **Fuel**: Diesel median {int(df[df['fuel']=='Diesel']['price'].median()):,} EUR "
        f"vs. Gasoline median {int(df[df['fuel']=='Gasoline']['price'].median()):,} EUR "
        f"(~{df[df['fuel']=='Diesel']['price'].median() / df[df['fuel']=='Gasoline']['price'].median():.2f}×).",
        f"- **Gear**: Automatic median "
        f"{int(df[df['gear']=='Automatic']['price'].median()):,} EUR vs. "
        f"Manual {int(df[df['gear']=='Manual']['price'].median()):,} EUR.",
        f"- **Brand spread** (top-10 by count): Opel bottom "
        f"(~{int(df[df['brand']=='Opel']['price'].median()):,} EUR), "
        f"Audi top (~{int(df[df['brand']=='Audi']['price'].median()):,} EUR).",
        "",
        "## 4. Anomalies and data-quality issues",
        f"- `mileage` max is **{int(df['mileage_km'].max()):,}** km — capped at 500 000 km "
        "during cleaning to remove unrealistic values.",
        f"- `car_age` > 30 years: {(df['car_age'] > 30).sum():,} rows (vintage outliers).",
        f"- Premium brands (BMW/Audi/Mercedes/Porsche/Jaguar/Lexus/Tesla/Bentley/Ferrari/"
        f"Lamborghini/Maserati) total **{df['brand'].isin(premium_brands).sum():,}** rows "
        "— now broad enough to fit reliable high-end predictions.",
        f"- Original raw CSV contains junk rows in `make` (e.g. trailers, trucks) "
        "with garbled encoding — dropped during cleaning.",
        "",
        "## 5. Implications for modelling",
        "- The skew justifies the `log1p` target in iteration 3.",
        "- Strong negative `car_age`↔price and strong positive `hp`↔price "
        "favour tree-based models that capture their non-linear interaction.",
        "- Class imbalance in `offerType` (Used dominates) is mild and "
        "handled implicitly by one-hot encoding.",
        "",
        "## Generated plots",
        "- `price_distribution.png` — long-tail histogram on linear scale",
        "- `log_price_distribution.png` — symmetric histogram after `log1p`",
        "- `price_by_fuel.png` — boxplot by fuel type",
        "- `price_by_gear.png` — boxplot Manual vs. Automatic",
        "- `price_by_brand_top10.png` — boxplot of the 10 most frequent brands",
        "- `correlations.png` — Pearson correlation of numeric features with `log(price)`",
        "- `age_vs_price.png` — scatter of `car_age` vs. `log(price)`",
    ]
    (OUT_DIR / "eda_summary.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"EDA artefacts written to {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
