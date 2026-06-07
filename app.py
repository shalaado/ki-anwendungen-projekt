"""AutoPrice Pro — used-car price estimator combining ML + NLP + Computer Vision.

Pipeline (see documentation.md section 1.2 for the diagram):

  1. The user uploads an optional car photo and types a free-form description
     in German or English.
  2. NLP block: an LLM extracts a JSON record (brand, year, mileage, fuel, …)
     from the description. Two prompt strategies (simple vs. structured) are
     compared and the structured one is used as the default.
  3. CV block: if a photo was provided, an OpenAI vision call classifies the
     body type (Sedan / SUV / Hatchback / …) and may also return a brand guess.
  4. The two outputs are merged. The body type fills in the categorical
     feature when the user did not state it, and the brand from CV is used
     as a fallback when text was ambiguous.
  5. ML block: the merged record is fed into the saved scikit-learn pipeline
     (GradientBoosting on log-target). The output is converted from EUR
     back to a euro figure and a rough CHF equivalent.
  6. The LLM generates a short German explanation that references the inputs
     and adds one uncertainty note.

Training data: AutoScout24 Germany (46 405 listings, 2011-2021, EUR).
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Pin the project root for src/train.py and src/eda.py — see the comment
# in src/train.py:_find_project_root for why.
os.environ.setdefault("AUTOPRICE_PROJECT_ROOT", str(PROJECT_ROOT))

from data_processing import (  # noqa: E402
    ALL_FEATURES,
    BRAND_BODY_DEFAULTS,
    CURRENT_YEAR,
    NUMERIC_FEATURES,
    VALID_BODY_TYPES,
    body_type_from_brand,
    normalize_brand,
)
from llm_client import active_model, active_vision_model, has_api_key  # noqa: E402
from nlp_block import (  # noqa: E402
    EXTRACTION_KEYS,
    coerce_extracted,
    explain_prediction,
    extract_simple,
    extract_structured,
)
from vision_block import classify_body_type  # noqa: E402


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "final_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
BRAND_DEFAULTS_PATH = ARTIFACTS_DIR / "brand_defaults.csv"

# Indicative EUR → CHF conversion, kept as a constant for transparency.
# 1.0 is intentional: as of mid-2026 the EUR/CHF rate hovers around parity.
EUR_TO_CHF = 1.0


def _load_artifacts():
    """Load the saved model. If anything is missing — typically on a fresh
    Hugging Face Space where only the source files were uploaded — trigger
    a one-off training run that also downloads the dataset on demand.
    """
    artefacts_present = (
        MODEL_PATH.exists()
        and METADATA_PATH.exists()
        and BRAND_DEFAULTS_PATH.exists()
    )
    if not artefacts_present:
        print(
            "[startup] Model artefacts missing — running first-time training. "
            "This usually takes ~30-90 s on a free HF Space."
        )
        import train  # local import to avoid a long boot when artefacts exist

        train.main()

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    brand_defaults = pd.read_csv(BRAND_DEFAULTS_PATH).set_index("brand")
    return model, metadata, brand_defaults


MODEL, METADATA, BRAND_DEFAULTS = _load_artifacts()
ALL_BRANDS = METADATA["valid_brands"]
GLOBAL_DEFAULTS = BRAND_DEFAULTS.median(numeric_only=True).to_dict()


def _build_feature_row(merged: dict) -> pd.DataFrame:
    """Materialise a single-row DataFrame in the same shape the model was trained on."""
    brand = merged.get("brand") or "Volkswagen"
    if brand not in ALL_BRANDS:
        brand = "Volkswagen"  # safe fallback — most common training brand

    body_type = merged.get("body_type") or body_type_from_brand(brand)
    if body_type not in VALID_BODY_TYPES:
        body_type = "Sedan"

    year = merged.get("year") or CURRENT_YEAR - 5
    car_age = max(0, CURRENT_YEAR - int(year))

    brand_row = BRAND_DEFAULTS.loc[brand] if brand in BRAND_DEFAULTS.index else None

    def use(value, fallback_key):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if brand_row is not None and fallback_key in brand_row.index:
                return float(brand_row[fallback_key])
            return float(GLOBAL_DEFAULTS[fallback_key])
        return float(value)

    mileage_km = use(merged.get("mileage_km"), "mileage_km")
    km_per_year = mileage_km / max(1, car_age)
    hp = use(merged.get("hp"), "hp")
    hp_per_year = hp / max(1, car_age)

    row = {
        "car_age": car_age,
        "mileage_km": mileage_km,
        "km_per_year": km_per_year,
        "hp": hp,
        "hp_per_year": hp_per_year,
        "brand": brand,
        "fuel": merged.get("fuel") or "Gasoline",
        "gear": merged.get("gear") or "Manual",
        "offerType": merged.get("offerType") or "Used",
        "body_type": body_type,
    }
    return pd.DataFrame([row])[ALL_FEATURES]


def _merge_text_and_vision(text_extracted: dict, vision_result: dict | None) -> dict:
    merged: dict = dict(text_extracted)

    # Vision fills gaps but never overrides what the user explicitly said.
    if vision_result and vision_result.get("is_car"):
        merged.setdefault("body_type", vision_result.get("body_type"))
        if merged.get("body_type") in (None, ""):
            merged["body_type"] = vision_result.get("body_type")

        if merged.get("brand") in (None, ""):
            brand_guess = normalize_brand(vision_result.get("brand_guess"))
            if brand_guess and brand_guess in ALL_BRANDS:
                merged["brand"] = brand_guess

    # Final safety net: normalise once more here in case the text path
    # bypassed the alias step.
    normalized_brand = normalize_brand(merged.get("brand"))
    if normalized_brand:
        merged["brand"] = normalized_brand

    if merged.get("body_type") in (None, ""):
        merged["body_type"] = body_type_from_brand(merged.get("brand"))

    return merged


def run_pipeline(description: str, image, prompt_strategy: str):
    """End-to-end inference. Returns the tuple Gradio components expect."""
    if not description or not description.strip():
        return (
            {"error": "Bitte beschreibe das Auto kurz in Worten."},
            {},
            None,
            None,
            "",
            "Bitte gib eine kurze Auto-Beschreibung ein.",
        )

    try:
        if prompt_strategy == "Structured (with examples)":
            text_extracted_raw = extract_structured(description)
        else:
            text_extracted_raw = extract_simple(description)
        text_extracted = coerce_extracted(text_extracted_raw)
    except Exception as exc:
        return (
            {"error": f"NLP extraction failed: {exc}"},
            {},
            None,
            None,
            "",
            f"Fehler bei der Textanalyse: {exc}",
        )

    vision_result = None
    if image is not None:
        try:
            vision_result = classify_body_type(image)
        except Exception as exc:
            vision_result = {
                "body_type": None,
                "confidence": 0.0,
                "is_car": False,
                "brand_guess": None,
                "reasoning": "",
                "error": str(exc),
            }

    merged = _merge_text_and_vision(text_extracted, vision_result)

    try:
        features = _build_feature_row(merged)
        pred_log = MODEL.predict(features)[0]
        prediction_eur = float(np.expm1(pred_log))
        prediction_chf = prediction_eur * EUR_TO_CHF
    except Exception as exc:
        return (
            text_extracted,
            vision_result or {},
            None,
            None,
            "",
            f"Fehler bei der Preisvorhersage: {exc}\n{traceback.format_exc()}",
        )

    try:
        explanation = explain_prediction(
            extracted={k: v for k, v in merged.items() if v is not None},
            body_type=merged.get("body_type"),
            prediction_eur=prediction_eur,
            prediction_chf=prediction_chf,
        )
    except Exception as exc:
        explanation = f"(Erklärung nicht verfügbar: {exc})"

    summary = (
        f"**Preis:** {prediction_eur:,.0f} EUR (ca. {prediction_chf:,.0f} CHF)\n\n"
        f"**Karosserie:** {merged.get('body_type', '–')} · "
        f"**Marke:** {merged.get('brand', '–')} · "
        f"**Baujahr:** {merged.get('year', '–')}\n\n"
        f"_Modell:_ {METADATA['model_name']} · "
        f"_CV-R² (training):_ {METADATA['final_metrics']['cv_r2_mean']:.3f}"
    )

    return (
        text_extracted,
        vision_result or {},
        round(prediction_eur, 0),
        round(prediction_chf, 0),
        explanation,
        summary,
    )


HEADER_DESCRIPTION = (
    "**AutoPrice Pro** schätzt den Preis eines Gebrauchtwagens aus drei "
    "Signalen: einem Foto, einer Freitext-Beschreibung und einem trainierten "
    "Regressionsmodell.\n\n"
    "• **Computer Vision** ermittelt den Karosserietyp aus dem Bild.\n"
    "• **NLP/LLM** extrahiert Marke, Baujahr, Kilometerstand u.a. aus dem Text.\n"
    "• **ML (GradientBoosting)** kombiniert beides zu einer Preisvorhersage.\n"
    f"\n_Modell-Datenquelle:_ AutoScout24 Germany ({METADATA['rows_after_cleaning']:,} "
    "Zeilen nach Cleaning, Preise in EUR — bei einem aktuellen EUR/CHF ≈ 1 sind die Werte "
    "praktisch direkt CHF)."
)


def _status_text() -> str:
    parts = []
    parts.append(f"Text-/Vision-Modell: `{active_model()}` / `{active_vision_model()}`")
    parts.append(f"OpenAI-Schlüssel gesetzt: {'✅' if has_api_key() else '❌ (NLP/CV deaktiviert)'}")
    parts.append(
        f"ML-Modell: `{METADATA['model_name']}` "
        f"(CV R² = {METADATA['final_metrics']['cv_r2_mean']:.3f}, "
        f"RMSE = {METADATA['final_metrics']['cv_rmse_mean']:,.0f} EUR)"
    )
    return "  •  ".join(parts)


EXAMPLES = [
    [
        "Mercedes-Benz CLS 350 aus 2019, Benzin, Automatik, 95'000 km, "
        "286 PS, Gebrauchtwagen.",
        None,
        "Structured (with examples)",
    ],
    [
        "VW Golf TSI, Bj. 2017, 65000 km, Schaltgetriebe, 115 PS, Diesel.",
        None,
        "Structured (with examples)",
    ],
    [
        "BMW 320i Touring, 2018, 80000 km, Automatik, 184 PS, Benzin.",
        None,
        "Simple",
    ],
    [
        "Tesla Model 3 Long Range aus 2020, Elektro, Automatik, 45000 km, 351 PS.",
        None,
        "Structured (with examples)",
    ],
]


with gr.Blocks(title="AutoPrice Pro — KI-Preisschätzer") as demo:
    gr.Markdown("# AutoPrice Pro — KI-Preisschätzer für Gebrauchtwagen")
    gr.Markdown(HEADER_DESCRIPTION)
    status_box = gr.Markdown(_status_text())

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 1. Eingabe")
            description = gr.Textbox(
                label="Beschreibung des Autos (Deutsch oder Englisch)",
                lines=4,
                placeholder=(
                    "z.B. Mercedes-Benz CLS 350, Bj. 2019, 95'000 km, "
                    "Benzin, Automatik, 286 PS."
                ),
            )
            image = gr.Image(
                label="Optional: Foto des Autos (für die Computer-Vision-Klassifikation)",
                type="pil",
                height=240,
            )
            prompt_strategy = gr.Radio(
                ["Structured (with examples)", "Simple"],
                value="Structured (with examples)",
                label="NLP Prompt-Strategie",
                info=(
                    "‚Structured‘ verwendet Few-Shot-Beispiele und strenge "
                    "Wertebereiche — empfohlen. ‚Simple‘ dient nur zum Vergleich."
                ),
            )
            run = gr.Button("Preis schätzen", variant="primary")

        with gr.Column(scale=5):
            gr.Markdown("### 2. Ergebnis")
            summary = gr.Markdown()
            with gr.Row():
                price_eur = gr.Number(label="Preis (EUR)", interactive=False)
                price_chf = gr.Number(label="Preis (≈ CHF)", interactive=False)
            explanation = gr.Textbox(
                label="LLM-Erklärung (Deutsch, mit Unsicherheits-Hinweis)",
                lines=5,
                interactive=False,
            )

    with gr.Accordion("Zwischenergebnisse der Blöcke", open=False):
        with gr.Row():
            extracted_view = gr.JSON(label="NLP — Extrahierte Spezifikationen")
            vision_view = gr.JSON(label="CV — Bildanalyse")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[description, image, prompt_strategy],
    )

    run.click(
        fn=run_pipeline,
        inputs=[description, image, prompt_strategy],
        outputs=[extracted_view, vision_view, price_eur, price_chf, explanation, summary],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", ssr_mode=False)
