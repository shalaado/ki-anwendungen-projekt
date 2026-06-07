"""NLP block: extract structured car specs from free text and explain the price.

Two prompt strategies are implemented so we can compare them in the
documentation. Both produce strict JSON that the rest of the pipeline can
consume directly. The vocabulary (allowed fuel/gear/offerType values) is
aligned with the AutoScout24 Germany schema.
"""

from __future__ import annotations

from typing import Optional

from data_processing import (
    VALID_FUELS,
    VALID_GEARS,
    VALID_OFFER_TYPES,
    normalize_brand,
    normalize_fuel,
    normalize_gear,
    normalize_offer_type,
)
from llm_client import call_text, parse_json


EXTRACTION_KEYS = (
    "brand",
    "model",
    "year",
    "mileage_km",
    "fuel",
    "gear",
    "offerType",
    "hp",
)


SIMPLE_EXTRACTION_SYSTEM = (
    "You extract used-car attributes from a free-text description written "
    "by a private seller. Reply with strict JSON only — no markdown, no "
    "code fences, no commentary. Use null whenever the user did not state "
    "the value explicitly. Never invent values."
)


STRUCTURED_EXTRACTION_SYSTEM = (
    "You are an information extractor for the AutoPrice Pro app.\n"
    "Goal: read a free-text car description (often in German) and return "
    "strict JSON with the requested keys.\n\n"
    "Rules:\n"
    "1. Reply with one JSON object only — no markdown, no commentary.\n"
    "2. Use null for any field the user did not clearly state.\n"
    "3. Allowed values:\n"
    f"   - fuel: {sorted(VALID_FUELS)}\n"
    f"   - gear: {sorted(VALID_GEARS)}\n"
    f"   - offerType: {sorted(VALID_OFFER_TYPES)}\n"
    "4. Numbers must be numbers, not strings (e.g. 95000 not \"95'000 km\").\n"
    "5. Normalise units: mileage_km in kilometres, hp in PS/horsepower.\n"
    "6. German terms map to English: Benzin→Gasoline, Schaltgetriebe→Manual, "
    "Automatik→Automatic, Gebrauchtwagen→Used, Tageszulassung→Pre-registered, "
    "Vorführwagen→Demonstration, Jahreswagen→Employee's car.\n\n"
    "Few-shot examples follow.\n\n"
    "Input: \"Ich verkaufe meinen Mercedes-Benz CLS 350 aus 2019, Benzin, "
    "Automatik, 95'000 km, 286 PS, Gebrauchtwagen.\"\n"
    "Output: {\"brand\": \"Mercedes-Benz\", \"model\": \"CLS 350\", \"year\": 2019, "
    "\"mileage_km\": 95000, \"fuel\": \"Gasoline\", \"gear\": \"Automatic\", "
    "\"offerType\": \"Used\", \"hp\": 286}\n\n"
    "Input: \"VW Golf TSI Bj. 2017, 65000 km, Schaltgetriebe, 115 PS, Diesel.\"\n"
    "Output: {\"brand\": \"Volkswagen\", \"model\": \"Golf\", \"year\": 2017, "
    "\"mileage_km\": 65000, \"fuel\": \"Diesel\", \"gear\": \"Manual\", "
    "\"offerType\": null, \"hp\": 115}"
)


def extract_simple(user_text: str) -> dict:
    """Iteration 1 prompt: short instruction, no examples, no constraints."""
    user_prompt = (
        "Extract the car attributes as JSON. Keys: "
        f"{', '.join(EXTRACTION_KEYS)}.\n\nText: {user_text}"
    )
    raw = call_text(SIMPLE_EXTRACTION_SYSTEM, user_prompt)
    return parse_json(raw, required_keys=())


def extract_structured(user_text: str) -> dict:
    """Iteration 2 prompt: explicit allowed values, units, and few-shot examples."""
    user_prompt = (
        "Extract the requested fields from the following German or English "
        "car description. Required JSON keys: "
        f"{', '.join(EXTRACTION_KEYS)}.\n\nText: {user_text}"
    )
    raw = call_text(STRUCTURED_EXTRACTION_SYSTEM, user_prompt)
    return parse_json(raw, required_keys=())


def coerce_extracted(parsed: dict) -> dict:
    """Snap LLM output to the values the ML pipeline expects."""

    def to_float(value) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def to_int(value) -> Optional[int]:
        flt = to_float(value)
        return int(flt) if flt is not None else None

    cleaned = {key: parsed.get(key) for key in EXTRACTION_KEYS}
    cleaned["year"] = to_int(cleaned.get("year"))
    cleaned["mileage_km"] = to_int(cleaned.get("mileage_km"))
    cleaned["hp"] = to_float(cleaned.get("hp"))

    cleaned["fuel"] = normalize_fuel(cleaned.get("fuel"))
    cleaned["gear"] = normalize_gear(cleaned.get("gear"))
    cleaned["offerType"] = normalize_offer_type(cleaned.get("offerType"))

    for str_field in ("brand", "model"):
        value = cleaned.get(str_field)
        cleaned[str_field] = value.strip() if isinstance(value, str) and value.strip() else None

    # Snap brand to canonical name (handles "Mercedes Benz" vs "Mercedes-Benz")
    normalized_brand = normalize_brand(cleaned.get("brand"))
    if normalized_brand:
        cleaned["brand"] = normalized_brand

    return cleaned


EXPLANATION_SYSTEM = (
    "You explain car-price predictions for a Gradio app. "
    "Reply with strict JSON only containing the key `answer`. "
    "The value must be a 2-3 sentence German explanation that:\n"
    " 1. references the inputs in a natural way,\n"
    " 2. uses the provided prediction (do NOT recompute it),\n"
    " 3. names one specific source of uncertainty.\n"
    "No markdown, no code fences."
)


def explain_prediction(extracted: dict, body_type: Optional[str],
                       prediction_eur: float, prediction_chf: float) -> str:
    """Use the LLM to produce a short German explanation of the price."""
    user_prompt = (
        "Extrahierte Spezifikationen: "
        + str({k: v for k, v in extracted.items() if v is not None})
        + f"\nKarosserietyp (aus Bild): {body_type or 'unbekannt'}"
        + f"\nVorhersage: {prediction_eur:,.0f} EUR (entspricht ca. {prediction_chf:,.0f} CHF)."
        + "\nErstelle die JSON-Antwort mit Schlüssel 'answer'."
    )
    raw = call_text(EXPLANATION_SYSTEM, user_prompt, temperature=0.2)
    parsed = parse_json(raw, required_keys=("answer",))
    return str(parsed["answer"]).strip()
