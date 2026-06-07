# AutoPrice Pro — Multimodal Used-Car Price Estimator

## Project Metadata

- Project title: **AutoPrice Pro — Multimodal Used-Car Price Estimator**
- Student: Adonis Shala
- GitHub repository URL: https://github.com/shalaado/ki-anwendungen-projekt
- Deployment URL: https://huggingface.co/spaces/shalaado/ki-anwendungen-projekt
- Submission date: 07.06.2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected (ML Numeric Data + NLP; Computer Vision added as bonus)
- [x] Multiple and different data sources used (see [section 2A.1](#2a1-data-sources))
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`) _(done at submission time)_

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution (choose 2):
- Primary block 1: **ML Numeric Data** (GradientBoosting price regressor)
- Primary block 2: **NLP** (LLM extraction + LLM explanation)

A third block (**Computer Vision**) is implemented and graded separately as
extra work. The vision output (body type, optional brand) is consumed by the
ML pipeline as a feature, so the integration is real — not just side-by-side
execution.

Guidance hint: Keep the project idea short and consistent. Focus most details on the selected blocks.
Evidence hint: Show where each selected block contributes to the final system.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition
- Problem statement: Private buyers and sellers of used cars rarely know
  what a fair asking price is. Existing online checkers only accept a fixed
  form: pick brand, model, year. They cannot read a free-text description
  or look at a photo.
- Goal: A single web app that turns a casual text description plus an
  optional photo into a defensible price estimate in EUR (≈ CHF) with a short
  German explanation.
- Success criteria:
  - Cross-validated R² ≥ 0.90 on the used-car regression task (achieved **0.9274**).
  - End-to-end pipeline runs in < 10 seconds per request.
  - Each block (ML, NLP, CV) contributes a feature the others cannot derive
    on their own.

### 1.2 Integration Logic
- How the selected blocks interact:
  - NLP extracts structured fields (`brand`, `model`, `year`, `mileage_km`,
    `fuel`, `gear`, `offerType`, `hp`) from German/English free text.
    See [`src/nlp_block.py`, lines 31-90](src/nlp_block.py#L31-L90).
  - CV classifies the body type from the photo and returns an optional brand
    guess. See [`src/vision_block.py`, lines 60-102](src/vision_block.py#L60-L102).
  - The two outputs are merged: CV fills in `body_type` and `brand` only when
    the text was silent on that field. See
    [`app.py`, lines 145-170](app.py#L145-L170).
  - The merged record is converted into the same 10-column feature vector
    the saved model was trained on
    ([`app.py`, lines 100-140](app.py#L100-L140)) and predicted in EUR.
  - Finally NLP is called again to produce a 2-3 sentence German explanation
    that references the inputs and adds one uncertainty note
    ([`src/nlp_block.py`, lines 122-145](src/nlp_block.py#L122-L145)).

- Data and output flow between blocks:

```
            ┌─────────────┐
   photo →  │ CV (vision) │ → body_type + brand_guess ─┐
            └─────────────┘                            │
            ┌─────────────┐                            ▼
    text →  │ NLP extract │ → JSON specs ────────► merge ─► ML predict
            └─────────────┘                                   │
                                                              ▼
                                                       price (EUR/CHF)
                                                              │
                                            ┌────────────┐    │
                                            │ NLP expl.  │ ◄──┘
                                            └────────────┘
                                                  │
                                                  ▼
                                           German answer
```

Guidance hint: This section should be short. The detailed work belongs in block sections.
Evidence hint: Include one clear pipeline overview.

---

## 2. Block Documentation

Complete only selected blocks. Mark non-selected block sections as N/A.

### 2A. ML Numeric Data (If selected)

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | AutoScout24 Germany dataset ([mirror](https://raw.githubusercontent.com/leander-ms/autoscout_Analysis/main/autoscout24-germany-dataset.csv)) | CSV / structured | 46 405 rows × 9 cols | Main training set |
| 2 | Internal `BRAND_BODY_DEFAULTS` lookup ([`src/data_processing.py`, lines 25-65](src/data_processing.py#L25-L65)) | mapping table | 41 entries | Provides a body-type default per brand for rows that have no CV input |
| 3 | Internal `BRAND_ALIASES` lookup ([`src/data_processing.py`, lines 84-148](src/data_processing.py#L84-L148)) | mapping table | 50+ entries | Maps user-typed brand spellings (e.g. "Mercedes Benz" → "Mercedes-Benz") to the canonical name the model knows |

The CSV mirrors the [Kaggle "Germany Cars Dataset"](https://www.kaggle.com/datasets/ander289386/cars-germany)
but is fetched without authentication, so the repository is fully
reproducible without any private credentials. Neither dataset is the
"apartments-canton-of-Zurich" or "dog-breeds" data used during the semester.

The dataset was deliberately chosen over a smaller Indian alternative
(CarDekho) because:
- 5.7× larger (46k vs. 8k rows) → much smaller variance,
- European market → prices in EUR ≈ CHF, no conversion fiction,
- premium brands well represented (2 354 Mercedes-Benz, 2 405 BMW, 2 684 Audi
  rows; 30 CLS listings alone).

#### 2A.2 Preprocessing and Features

- Cleaning steps
  ([`src/train.py`, lines 122-145](src/train.py#L122-L145)):
  - drop rows with missing `price`
  - clip the top and bottom 0.5 % of `price` to neutralise listings that
    are clearly mis-typed
  - restrict `mileage` to [0, 500 000] km, `year` to [1980, current_year]
    and `hp` to [30, 1 500]
  - remove junk rows where `make` is a garbled trailer/truck label
  - drop fully duplicated rows
  - net effect: 46 405 → **43 725** rows
- Preprocessing inside the sklearn pipeline
  ([`src/train.py`, lines 152-170](src/train.py#L152-L170)):
  - numeric features: median imputation + `StandardScaler`
  - categorical features: most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`
- Feature engineering and selection
  ([`src/data_processing.py`, lines 245-290](src/data_processing.py#L245-L290)):
  - `brand` extracted directly from the `make` column
  - `car_age = CURRENT_YEAR (2026) − year`
  - `km_per_year = mileage_km / car_age` (proxy for how heavily the car was used)
  - `hp_per_year = hp / car_age` (separates "old-and-fast" from "old-and-tired")
  - `body_type` from the brand lookup (or, at inference, from CV)

Final feature set (10 columns):
`car_age, mileage_km, km_per_year, hp, hp_per_year, brand, fuel, gear,
offerType, body_type`.

**Exploratory Data Analysis (EDA).** Run with `python src/eda.py`. The full
report and plots are auto-generated into
[`artifacts/eda/eda_summary.md`](artifacts/eda/eda_summary.md) and the
sibling PNGs. Key findings:

- **Target distribution**: after cleaning `price` spans **2 900 EUR to
  112 890 EUR** with median **10 990 EUR** (P25 = 7 499, P75 = 19 450).
  Right-skewed (skew = **2.59**) on the linear scale; after `log1p` the
  skew drops to **0.52** (near-symmetric), which motivates the log-target
  in iteration 3.
  ![Linear price distribution](price_distribution.png)
  ![After log1p](log_price_distribution.png)

- **Strongest numeric predictors** (Pearson r with `log(price)`):
  `hp_per_year` = **+0.86**, `hp` = **+0.71**, `car_age` = **−0.68**,
  `mileage_km` = **−0.47**, `km_per_year` = **−0.36**. The engineered
  `hp_per_year` is the single strongest linear signal.
  ![Correlations](correlations.png)
  ![car_age vs log(price)](age_vs_price.png)

- **Categorical breakdowns**:
  - Diesel median **12 480 EUR** vs. Gasoline median **9 990 EUR** (~1.25×).
  - Automatic median **22 989 EUR** vs. Manual **8 990 EUR** (~2.56×) —
    a price-premium signal so strong that the model relies on it heavily.
  - Brand medians across the top-10 (by count) range from Opel
    (~8 980 EUR) to Audi (~23 890 EUR).
  ![Price by fuel](price_by_fuel.png)
  ![Price by gear](price_by_gear.png)
  ![Top-10 brands](price_by_brand_top10.png)

- **Anomalies and data-quality issues**:
  - `mileage` values capped at 500 000 km during cleaning to remove
    unrealistic listings.
  - Premium brands (BMW, Audi, Mercedes-Benz, Porsche, Jaguar, Lexus,
    Tesla, Bentley, Ferrari, Lamborghini, Maserati) total **7 488** rows
    — broad enough to fit reliable predictions at the high end.
  - Junk `make` values from the raw CSV (e.g. "Trailer-Anh�nger",
    "Caravans-Wohnm") removed during cleaning.

These findings justify (a) the `log1p` target, (b) the preference for
tree-based models over linear ones, and (c) the brand→body fallback chain
used at inference time.

#### 2A.3 Model Selection

- Models tested: `Ridge`, `RandomForestRegressor` (200 and 300 trees),
  `RandomForestRegressor` (tuned, 400 trees, max_depth 24),
  `GradientBoostingRegressor` (tuned, 500 trees, depth 5, lr 0.05).
- Why these models were chosen:
  - Ridge is the simplest linear baseline and shows what a one-hot linear
    model can do on this data.
  - RandomForest handles the mixed feature types and the non-linear
    interaction between `car_age`, `hp`, and `mileage_km` well.
  - GradientBoosting on a `log1p(price)` target was added to close the gap
    on the long tail of premium cars (Porsche, Bentley, Mercedes-Benz),
    where RMSE on the linear scale is dominated by the upper quantiles.

#### 2A.4 Model Comparison and Iterations

The full iteration markdown is generated automatically by the training run
and lives at [`artifacts/model_iterations.md`](artifacts/model_iterations.md).
Open it for the per-model 5-fold CV numbers; the table below is a summary.

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline with raw structured features | Only `year`, `mileage_km`, `hp` + categoricals | Ridge, RandomForest | R² 0.9246 (RF), Ridge 0.8539 | — |
| 2 | Engineered numeric features | Add `car_age`, `km_per_year`, `hp_per_year`, body-type lookup | Ridge, RandomForest | R² 0.9237 (RF), Ridge **0.8964** (+0.04 R²) | improves Ridge substantially; RF stays at ~0.924 |
| 3 | Log target + tuning | `log1p(price)`, tuned RF (n=400, depth=24) and GradientBoosting (lr=0.05, n=500) | RF tuned, GBM tuned | **R² 0.9274 (GBM)** | +0.0028 R² vs it. 2 — best overall, chosen as final model |

Detailed per-model numbers (auto-regenerated on every training run) are in
[`artifacts/model_iterations.md`](artifacts/model_iterations.md#detailed-iteration-breakdown).

#### 2A.5 Evaluation and Error Analysis

- Metrics used: 5-fold CV with `R²`, `RMSE (EUR)`, `MAE (EUR)`. RMSE is
  always reported on the original euro scale even when the model trained on
  `log1p(price)` (RMSE is computed after `expm1` back-transform — see
  [`src/train.py`, lines 173-210](src/train.py#L173-L210)).
- Final results (5-fold CV, n = 43 725):
  - **R² = 0.9274 ± 0.0036**
  - **RMSE = 3 587 EUR**
  - **MAE = 1 950 EUR**
  - All metrics are also persisted in `artifacts/metadata.json`.
- Error patterns and likely causes:
  - The biggest absolute errors are on premium cars in the > 100 k EUR
    range (Bentley, Porsche 911 GT3, Mercedes-AMG). The model under-predicts
    the very high end because the long tail thins out.
  - Listings with `offerType ∈ {Pre-registered, Demonstration}` and very
    low `mileage_km` overlap with "almost new" cars; the model handles
    them by combining low age with the one-hot category, but residuals
    are slightly larger than for the bulk of Used listings.

#### 2A.6 Integration with Other Block(s)

- Inputs received from other block(s):
  - From NLP (string→struct extraction):
    `brand, year, mileage_km, fuel, gear, offerType, hp`.
  - From CV (image→struct): `body_type`, optional `brand_guess`. CV only
    overrides the text when the text was silent on that field
    ([`app.py`, lines 145-170](app.py#L145-L170)).
- Outputs provided to other block(s):
  - `prediction_eur` and `prediction_chf` are fed into the NLP explanation
    prompt ([`app.py`, lines 220-235](app.py#L220-L235)).

Guidance hint: Keep entries practical and evidence-based.
Evidence hint: Add values, not only claims.

### 2B. NLP (If selected)

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | User text input | free text | 1 per request | The actual extraction target |
| 2 | OpenAI `gpt-4o-mini` (Responses API) | hosted LLM | – | Performs both the extraction and the explanation |
| 3 | In-prompt few-shot examples ([`src/nlp_block.py`, lines 38-69](src/nlp_block.py#L38-L69)) | hand-written | 2 examples | Anchors the JSON schema and unit conventions |

#### 2B.2 Preprocessing and Prompt Design

- Text preprocessing: only minimal trimming; the LLM receives the raw text
  so that language quirks (e.g. Swiss apostrophes `95'000 km` or German
  fuel terms like "Benzin") are part of the few-shot context.
- Prompt design:
  - **System prompt (structured strategy)**: lists the allowed values for
    `fuel`, `gear`, `offerType`; pins unit conventions for km and PS;
    explicitly translates German→English terms (Benzin→Gasoline,
    Schaltgetriebe→Manual, Gebrauchtwagen→Used, …); demands strict JSON
    with all required keys, `null` for missing fields, and explicitly
    forbids invention of values
    ([`src/nlp_block.py`, lines 38-69](src/nlp_block.py#L38-L69)).
  - **System prompt (simple strategy)**: same goal, much shorter, no allowed
    values, no examples ([`src/nlp_block.py`, lines 29-34](src/nlp_block.py#L29-L34)).
  - **Explanation prompt**: forbids the LLM from re-computing the price,
    demands one uncertainty note, and forces strict JSON with key `answer`
    ([`src/nlp_block.py`, lines 122-135](src/nlp_block.py#L122-L135)).
- Output coercion: even when the LLM returns slightly wrong casings
  ("benzin" vs `"Gasoline"`), [`coerce_extracted`](src/nlp_block.py#L83-L117)
  snaps to the exact strings the ML pipeline saw during training. The
  brand is also passed through
  [`normalize_brand`](src/data_processing.py#L150-L185) which handles
  "Mercedes Benz" → "Mercedes-Benz" type aliases.

#### 2B.3 Approach Selection

- Approach used: **Prompt engineering on a hosted LLM** (OpenAI `gpt-4o-mini`,
  Responses API). The same model is used for extraction and explanation —
  one billing relationship, one auth, one place to swap a model id.
- Alternatives considered:
  - Classical NER with spaCy/HuggingFace: heavier infra, multilingual NER
    on this kind of slang ("Bj. 2017, Tageszulassung, 95'000 km") is brittle.
  - RAG over a car-spec catalogue: not needed because the only knowledge we
    want from the model is light extraction + plausible language; the
    structured catalogue (brand defaults) lives in CSV and is queried in
    code.

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Make extraction work at all | "Simple" prompt, no examples, no allowed values | `gpt-4o-mini`, T=0 | Most fields filled correctly on the German example sentences; `fuel` sometimes returned as `"Benzin"` instead of `"Gasoline"`, `gear` as `"Schaltgetriebe"` instead of `"Manual"` | baseline |
| 2 | Constrain schema | Add allowed-values list (fuel/gear/offerType), enforce unit normalisation, explicit German→English term mapping | same model | Allowed-value violations stopped; numbers now arrive as numbers, not strings | qualitative improvement; ML pipeline no longer needs string-cleanup |
| 3 | Anchor on edge cases | Add two few-shot examples (one German Mercedes CLS, one mixed VW Golf) and require `null` for missing fields | same model | LLM stops inventing `offerType` when the user did not mention it; brand mapping snaps "Mercedes Benz" → "Mercedes-Benz" via `normalize_brand` | far fewer downstream defaults triggered |

Both prompt strategies are exposed in the Gradio app via a radio button so
the evaluator can compare them at runtime (see screenshots in
[section 3](#3-deployment)).

#### 2B.5 Evaluation and Error Analysis

- Evaluation strategy: qualitative comparison on a fixed set of German
  example sentences (see [`EXAMPLES`](app.py#L260-L283)), inspecting the
  extracted JSON for (a) correctness, (b) presence of unexpected fields,
  (c) value normalisation.
- Results:
  - On the four example prompts, the **structured** strategy reliably
    matches the ML pipeline's expectations (categorical values exactly as
    one-hot keys, numerics as floats).
  - The **simple** strategy works most of the time but occasionally returns
    `"Schaltgetriebe"` for `gear` instead of `"Manual"`, which would then
    go through `coerce_extracted` and end up as `None`. The Gradio radio
    button lets the user reproduce this.
- Error patterns and likely causes:
  - Ambiguous brands (e.g. "Land Rover" vs the canonical "Land") — solved
    by the alias table that maps both to "Land".
  - When the user gives no brand at all, the structured prompt returns
    `null` and the app falls back to CV's `brand_guess` if available.

#### 2B.6 Integration with Other Block(s)

- Inputs received from other block(s):
  - For the **explanation** call: prediction in EUR/CHF from ML and
    `body_type` from CV. Merged inputs are passed as a Python dict
    ([`src/nlp_block.py`, lines 137-150](src/nlp_block.py#L137-L150)).
- Outputs provided to other block(s):
  - Extraction JSON is consumed directly by ML (after coercion).
  - The explanation text is the user-visible bottom of the pipeline.

Guidance hint: Show concrete prompt or retrieval decisions.
Evidence hint: Include representative outputs or failure cases.

### 2C. Computer Vision (If selected)

> Implemented as **bonus / third block** with full integration into the ML
> pipeline. Not used in isolation.

#### 2C.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | User-uploaded car photo (Gradio Image component) | image (PIL.Image) | 1 image per request | The actual classification input |
| 2 | OpenAI `gpt-4o-mini` vision endpoint | multimodal LLM | – | Classifies body type and optionally the brand |
| 3 | Local list `VALID_BODY_TYPES` ([`src/data_processing.py`, line 75](src/data_processing.py#L75)) | enum | 8 values | Constrains the model's output to one of our known categories |

The dog-breed image dataset used during the semester is **not** used.

#### 2C.2 Preprocessing and Augmentation

Implemented in [`preprocess_image`](src/vision_block.py#L38-L59):
- Reject images smaller than 64 px on the shorter side (`MIN_IMAGE_SIDE`).
- Respect EXIF orientation via `PIL.ImageOps.exif_transpose` so phone
  photos do not arrive rotated.
- Force-convert to RGB.
- Resize the longest side to 1 024 px (`MAX_IMAGE_SIDE`) to keep the API
  payload small while preserving detail for body-type classification.
- Re-encode as PNG before sending to the API ([`src/llm_client.py`, lines
  84-115](src/llm_client.py#L84-L115)).

No augmentation is applied — this is an *application* of a vision model at
inference time, not a training run.

#### 2C.3 Model Selection

- Vision model used: OpenAI `gpt-4o-mini` (multimodal). Same model id as
  the text block, but exposed via the `image_url` content type
  ([`src/llm_client.py`, lines 84-115](src/llm_client.py#L84-L115)).
- Why this model was chosen:
  - Lightweight: no torch / transformers / GPU footprint, so the Gradio
    Space can boot in seconds on CPU-only hardware.
  - Same API surface as the NLP block: one shared client, one shared
    JSON-parser, one OPENAI key in the Hugging Face secret.
  - GPT-4o-mini is strong on "which of these 8 categories matches?", which
    is the only call we ever make. We constrain the answer to a closed list
    in the system prompt
    ([`src/vision_block.py`, lines 23-35](src/vision_block.py#L23-L35)) so
    hallucination space is small.

#### 2C.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Get a body-type label out of the LLM | Free-form prompt: "What body type is this car?" | `gpt-4o-mini` vision | answers like "It looks like a hatchback." — unparseable | baseline |
| 2 | Constrain to JSON | System prompt with enum + strict JSON requirement | `gpt-4o-mini` vision | clean JSON, but model sometimes invents `body_type` values not in the enum | every output is JSON; need to snap to allowed list |
| 3 | Constrain output + add safety | Explicit `is_car` boolean; snap any out-of-enum value to `Sedan` in code; force `confidence ∈ [0, 1]`; `brand_guess` is routed through `normalize_brand` before use | `gpt-4o-mini` vision | predicts plausibly on car photos uploaded through the UI, returns `is_car=false` for an obvious non-car — exactly the behaviour the app needs | the merge logic in app.py can safely skip CV outputs with `is_car=false` |

#### 2C.5 Evaluation and Error Analysis

- Metrics and/or visual checks: visual inspection on a handful of car
  photos (sedan, SUV, hatchback) and one negative example (a photo without
  a car). For each, we check that `body_type ∈ VALID_BODY_TYPES`,
  `0 ≤ confidence ≤ 1`, and that `is_car` is `false` for non-car images.
- Final results:
  - On test photos covering SUV / Sedan / Hatchback, the vision model
    returned the correct enum value with `confidence ≥ 0.7`.
  - On a non-car negative example, the model returned `is_car = false` and
    the app correctly ignored the CV output, so the pipeline still produced
    a price based on the text alone.
- Error patterns and limitations:
  - On stylised renders (3D car configurator screenshots), `brand_guess` is
    sometimes wrong even when `body_type` is correct — that is fine because
    `brand_guess` is only used when the user said nothing about the brand.
  - The model has a bias toward `SUV` for ambiguous front-quarter shots of
    crossovers. This is a known limitation that surfaces in the uncertainty
    note the explanation prompt always asks for.

#### 2C.6 Integration with Other Block(s)

- Inputs received from other block(s): none — the CV block only receives
  the raw image.
- Outputs provided to other block(s):
  - `body_type` → consumed by the ML pipeline as a categorical feature.
  - `brand_guess` → fills in the ML `brand` feature only when the NLP
    extraction returned `null` for `brand`.

Guidance hint: Use concise examples from real predictions.
Evidence hint: Include sample outputs and observed failure cases.

---

## 3. Deployment

- Deployment URL: https://huggingface.co/spaces/shalaado/ki-anwendungen-projekt
- Main user flow:
  1. Open the Space.
  2. Paste a free-text description of the car (German or English).
  3. Optionally drop a photo into the image component.
  4. Choose a prompt strategy (default: *Structured (with examples)*).
  5. Click "Preis schätzen". The app shows
     - the predicted price in EUR and an approximate CHF figure,
     - the German LLM explanation,
     - and the raw NLP / CV intermediate outputs in an expandable accordion.

**Screenshot 1 — Text-only request.** Same Mercedes-Benz CLS 350 description
is processed end-to-end, the LLM produces the German explanation, and the
ML pipeline returns ~42 818 EUR. No photo uploaded, so the CV block is not
triggered.

![Text-only request](screenshot_text_only.png)

**Screenshot 2 — Combined text + photo request.** With a Mercedes-Benz CLS
photo dropped in, the CV block fires (see the bottom accordion: `body_type:
"Sedan"`, `brand_guess: "Mercedes"`, `confidence: 0.9`). Its output is
merged into the ML feature row alongside the NLP extraction
(`brand: "Mercedes-Benz"`, `model: "CLS 350"`, `year: 2019`, …) and the
final prediction is again ~42 818 EUR.

![Text and image request — full integration of ML + NLP + CV](screenshot_text_and_image.png)

Guidance hint: Deployment must be usable.
Evidence hint: Add screenshots or short demo references.

---

## 4. Execution Instructions

- Environment setup:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate            # Windows / PowerShell
  pip install -r requirements.txt
  ```
- Data setup: the AutoScout24 CSV is committed under `data/`. If it is
  missing, the training run downloads it automatically from GitHub via
  [`ensure_dataset`](src/train.py#L88-L102). Manual download:
  ```bash
  curl -L -o data/autoscout24_germany.csv \
       "https://raw.githubusercontent.com/leander-ms/autoscout_Analysis/main/autoscout24-germany-dataset.csv"
  ```
- Training command:
  ```bash
  python src/train.py
  # writes artifacts/final_model.joblib, metadata.json, brand_defaults.csv,
  # body_type_defaults.csv, model_iterations.md
  ```
- EDA command (optional, regenerates plots + key findings):
  ```bash
  python src/eda.py
  # writes artifacts/eda/eda_summary.md and seven PNG plots
  ```
- Inference / run command:
  ```bash
  $env:OPENAI_API_KEY = "sk-..."          # PowerShell
  python app.py                            # http://127.0.0.1:7860/
  ```
- Reproducibility notes:
  - Python 3.13.13 was used during development.
  - `RANDOM_STATE = 42` is set in [`src/train.py`, line 76](src/train.py#L76).
  - `requirements.txt` pins lower bounds only; the exact versions verified
    locally are `pandas 3.0.1`, `numpy 2.4.2`, `scikit-learn 1.8.0`,
    `joblib 1.5.3`, `openai 2.33.0`, `pillow 12.1.1`, `matplotlib 3.10.8`.
  - On Hugging Face Spaces the Gradio runtime is pinned via
    `sdk_version: 6.13.0` in [`README.md`](README.md).
  - The app self-bootstraps: if the artefacts are missing on a fresh
    Space, `app.py` triggers a one-off training run (~30-90 s).

Guidance hint: Another person should be able to run your project from this section.
Evidence hint: Include exact commands and versions.

---

## 5. Optional Bonus Evidence

Use this section for exceptional work beyond the core requirements.

- [x] Third selected block implemented with strong quality
- [x] More than two data sources used with clear added value
- [x] A core section is done exceptionally well (3 documented training iterations,
  metrics in EUR back-transformed from `log1p`, automatic `model_iterations.md`
  generation by `train.py`)
- [x] Extended evaluation (per-iteration table + per-model 5-fold CV scores
  in `artifacts/model_iterations.md`, plus EDA report and seven plots in
  `artifacts/eda/`)
- [x] Ethics, bias, or fairness analysis (see below)
- [ ] Creative or exceptional use case

**Bonus 1 — Computer Vision integration.**
The CV block is not just running side-by-side; its output is wired into the
ML feature row (`body_type` directly, optional `brand_guess` only when text
was silent). Disabling the image input only degrades the prediction when
the text was also missing those fields, which is the right behaviour.

**Bonus 2 — Multiple data sources, with role separation.**
One structured CSV (AutoScout24 Germany, 46k rows), two LLM calls with
different prompt designs (extraction + explanation), and a user-supplied
image. Each plays a different role in the pipeline. Two internal lookup
tables (`BRAND_BODY_DEFAULTS`, `BRAND_ALIASES`) handle the brand /
body-type bridge between blocks.

**Bonus 3 — Extended evaluation.**
Each training iteration is reported with R², RMSE, and MAE — all on the
original EUR scale even when the model trained on `log1p(price)`. The
iteration markdown is **regenerated automatically** by `src/train.py`, so
the documentation cannot drift away from the actual model.

**Bonus 4 — Self-bootstrapping deployment.**
The app is robust against missing artefacts on the Hugging Face Space:
[`_load_artifacts`](app.py#L80-L102) detects missing files and triggers
a one-off training run that also auto-downloads the CSV. This makes the
Space reproducible by anyone who only checks out the source files.

**Bonus 5 — Ethics / responsible-use note.**
- The training data is the German used-car market (prices in EUR). At
  current EUR/CHF rates these numbers are practically directly usable in
  Switzerland, but micro-pricing factors (specific dealer reputation,
  recent local trends, condition issues not visible in metadata) are not
  captured.
- The LLM explanation is forced to add one uncertainty note in every reply
  ([`src/nlp_block.py`, lines 122-135](src/nlp_block.py#L122-L135)), so the
  user is reminded that the prediction is an estimate, not a quote.
- Premium brands are sparsely represented at the extreme top end (single
  digits of Bentley/Lamborghini/Maserati rows). Predictions there are
  necessarily noisier — this is documented in section
  [2A.5](#2a5-evaluation-and-error-analysis).
- No personal data is collected. The OpenAI calls send only the user-typed
  description and the user-uploaded image, never any identifier.
