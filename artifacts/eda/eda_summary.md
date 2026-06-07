# EDA — Key Findings

Dataset: AutoScout24 Germany `autoscout24_germany.csv` (raw rows: **46,405**, after cleaning: **43,725**, removed: 2,680).

## 1. Target distribution
- `price` spans **2,900** EUR to **112,890** EUR.
- Right-skewed on the linear scale (skew = **2.59**) → motivates the `log1p` target used in iteration 3 (post-log skew = **0.52**).
- Median = **10,990** EUR, P25 = **7,499**, P75 = **19,450**.

## 2. Numeric correlations with log price
  - `hp_per_year`: r = **+0.859**
  - `hp`: r = **+0.711**
  - `car_age`: r = **-0.678**
  - `mileage_km`: r = **-0.474**
  - `km_per_year`: r = **-0.362**

  → `hp` (+0.71) and `car_age` (-0.68) carry most of the linear signal; tree-based models add value through interactions on top of that.

## 3. Categorical breakdowns
- **Fuel**: Diesel median 12,480 EUR vs. Gasoline median 9,990 EUR (~1.25×).
- **Gear**: Automatic median 22,989 EUR vs. Manual 8,990 EUR.
- **Brand spread** (top-10 by count): Opel bottom (~8,980 EUR), Audi top (~23,890 EUR).

## 4. Anomalies and data-quality issues
- `mileage` max is **500,000** km — capped at 500 000 km during cleaning to remove unrealistic values.
- `car_age` > 30 years: 0 rows (vintage outliers).
- Premium brands (BMW/Audi/Mercedes/Porsche/Jaguar/Lexus/Tesla/Bentley/Ferrari/Lamborghini/Maserati) total **7,488** rows — now broad enough to fit reliable high-end predictions.
- Original raw CSV contains junk rows in `make` (e.g. trailers, trucks) with garbled encoding — dropped during cleaning.

## 5. Implications for modelling
- The skew justifies the `log1p` target in iteration 3.
- Strong negative `car_age`↔price and strong positive `hp`↔price favour tree-based models that capture their non-linear interaction.
- Class imbalance in `offerType` (Used dominates) is mild and handled implicitly by one-hot encoding.

## Generated plots
- `price_distribution.png` — long-tail histogram on linear scale
- `log_price_distribution.png` — symmetric histogram after `log1p`
- `price_by_fuel.png` — boxplot by fuel type
- `price_by_gear.png` — boxplot Manual vs. Automatic
- `price_by_brand_top10.png` — boxplot of the 10 most frequent brands
- `correlations.png` — Pearson correlation of numeric features with `log(price)`
- `age_vs_price.png` — scatter of `car_age` vs. `log(price)`