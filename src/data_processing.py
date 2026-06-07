"""Shared data-cleaning helpers used by training and inference.

Schema source: AutoScout24 Germany used car listings
(``data/autoscout24_germany.csv``, 46 405 rows, 2011-2021, EUR prices).

Keeping cleaning logic in a single module guarantees that the feature
representation the model was trained on is identical to what the live app
sends in at inference time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


CURRENT_YEAR = 2026


# --------------------------------------------------------------------------- #
# Brand / body type lookups
# --------------------------------------------------------------------------- #

# Brand → typical body type. Used as a fallback when the user did not
# upload a photo and the LLM did not give us a body type either.
BRAND_BODY_DEFAULTS = {
    "Volkswagen": "Hatchback",
    "Opel": "Hatchback",
    "Ford": "Hatchback",
    "Skoda": "Hatchback",
    "Renault": "Hatchback",
    "Audi": "Sedan",
    "BMW": "Sedan",
    "Mercedes-Benz": "Sedan",
    "SEAT": "Hatchback",
    "Hyundai": "Hatchback",
    "Fiat": "Hatchback",
    "Toyota": "SUV",
    "Peugeot": "Hatchback",
    "Kia": "SUV",
    "smart": "Hatchback",
    "Citroen": "Hatchback",
    "Volvo": "Sedan",
    "Nissan": "Hatchback",
    "Dacia": "Hatchback",
    "Mazda": "Hatchback",
    "MINI": "Hatchback",
    "Mitsubishi": "SUV",
    "Suzuki": "Hatchback",
    "Porsche": "Coupe",
    "Chevrolet": "Sedan",
    "Honda": "Hatchback",
    "Land": "SUV",
    "Jeep": "SUV",
    "Alfa": "Sedan",
    "Jaguar": "Sedan",
    "Cupra": "Hatchback",
    "Subaru": "SUV",
    "Lexus": "Sedan",
    "Abarth": "Hatchback",
    "SsangYong": "SUV",
    "Tesla": "Sedan",
    "Maserati": "Sedan",
    "Bentley": "Sedan",
    "Ferrari": "Coupe",
    "Lamborghini": "Coupe",
    "McLaren": "Coupe",
}

BODY_TYPE_SEATS_DEFAULT = {
    "Hatchback": 5,
    "Sedan": 5,
    "SUV": 5,
    "Coupe": 4,
    "Pickup": 5,
    "Convertible": 4,
    "Van": 7,
    "Wagon": 5,
}

VALID_BODY_TYPES = list(BODY_TYPE_SEATS_DEFAULT.keys())


# Common spelling / abbreviation variants → canonical brand as stored in
# the training data. Keys must be lower-case. Without these aliases an
# input like "Mercedes Benz" (no hyphen) would miss the model's vocabulary.
BRAND_ALIASES = {
    # Mercedes-Benz
    "mercedes": "Mercedes-Benz",
    "mercedes benz": "Mercedes-Benz",
    "mercedes-benz": "Mercedes-Benz",
    "benz": "Mercedes-Benz",
    "mb": "Mercedes-Benz",
    # Volkswagen
    "vw": "Volkswagen",
    "volkswagen": "Volkswagen",
    # Land Rover (stored as "Land" because of the first-token rule)
    "land": "Land",
    "land rover": "Land",
    "landrover": "Land",
    "range rover": "Land",
    "rover": "Land",
    # Mini
    "mini": "MINI",
    # SEAT
    "seat": "SEAT",
    # SsangYong
    "ssangyong": "SsangYong",
    # Alfa Romeo (stored as "Alfa")
    "alfa": "Alfa",
    "alfa romeo": "Alfa",
    "alfaromeo": "Alfa",
    # Common single tokens — explicit so casing never trips us up
    "bmw": "BMW",
    "audi": "Audi",
    "toyota": "Toyota",
    "honda": "Honda",
    "hyundai": "Hyundai",
    "ford": "Ford",
    "opel": "Opel",
    "renault": "Renault",
    "skoda": "Skoda",
    "nissan": "Nissan",
    "fiat": "Fiat",
    "peugeot": "Peugeot",
    "citroen": "Citroen",
    "kia": "Kia",
    "porsche": "Porsche",
    "tesla": "Tesla",
    "volvo": "Volvo",
    "jaguar": "Jaguar",
    "jeep": "Jeep",
    "lexus": "Lexus",
    "mazda": "Mazda",
    "mitsubishi": "Mitsubishi",
    "suzuki": "Suzuki",
    "chevrolet": "Chevrolet",
    "subaru": "Subaru",
    "dacia": "Dacia",
    "smart": "smart",
    "cupra": "Cupra",
    "abarth": "Abarth",
    "maserati": "Maserati",
    "bentley": "Bentley",
    "ferrari": "Ferrari",
    "lamborghini": "Lamborghini",
    "mclaren": "McLaren",
}


def normalize_brand(raw):
    """Map a free-text brand string to a canonical brand the model knows.

    Returns the canonical brand (e.g. ``"Mercedes-Benz"``) if it can be
    matched and ``None`` otherwise. The caller should apply a sensible
    fallback (e.g. the most common training brand) when the result is
    ``None``.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    key = text.lower().replace("_", " ").strip()

    # 1. direct alias hit
    if key in BRAND_ALIASES:
        return BRAND_ALIASES[key]

    # 2. exact canonical-name match (case-insensitive)
    for brand in BRAND_BODY_DEFAULTS:
        if brand.lower() == key:
            return brand

    # 3. first-token match — covers "Mercedes-Benz CLS 350" style inputs
    if " " in key or "-" in key:
        first_token = key.replace("-", " ").split()[0]
        if first_token in BRAND_ALIASES:
            return BRAND_ALIASES[first_token]
        for brand in BRAND_BODY_DEFAULTS:
            if brand.lower() == first_token:
                return brand

    return None


# --------------------------------------------------------------------------- #
# Fuel / gear / offerType normalisation
# --------------------------------------------------------------------------- #

VALID_FUELS = {
    "Gasoline",
    "Diesel",
    "Electric/Gasoline",
    "Electric",
    "LPG",
    "CNG",
    "Electric/Diesel",
    "Ethanol",
    "Hydrogen",
    "Others",
}

VALID_GEARS = {"Manual", "Automatic", "Semi-automatic"}

VALID_OFFER_TYPES = {
    "Used",
    "Pre-registered",
    "Demonstration",
    "Employee's car",
    "New",
}


def normalize_fuel(raw):
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    mapping = {
        "benzin": "Gasoline",
        "petrol": "Gasoline",
        "gasoline": "Gasoline",
        "super": "Gasoline",
        "diesel": "Diesel",
        "electric": "Electric",
        "elektro": "Electric",
        "elektrisch": "Electric",
        "hybrid": "Electric/Gasoline",
        "plug-in hybrid": "Electric/Gasoline",
        "electric/gasoline": "Electric/Gasoline",
        "electric/diesel": "Electric/Diesel",
        "lpg": "LPG",
        "autogas": "LPG",
        "cng": "CNG",
        "erdgas": "CNG",
        "ethanol": "Ethanol",
        "e85": "Ethanol",
        "hydrogen": "Hydrogen",
        "wasserstoff": "Hydrogen",
    }
    return mapping.get(text, text.title() if text.title() in VALID_FUELS else None)


def normalize_gear(raw):
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if text in {"manual", "schaltgetriebe", "schalter", "manuell"}:
        return "Manual"
    if text in {"automatic", "automatik", "automat"}:
        return "Automatic"
    if text in {"semi-automatic", "semi automatic", "halbautomatik", "dsg", "dct"}:
        return "Semi-automatic"
    return None


def normalize_offer_type(raw):
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if text in {"used", "gebraucht", "gebrauchtwagen"}:
        return "Used"
    if text in {"new", "neu", "neuwagen"}:
        return "New"
    if text in {"pre-registered", "pre registered", "tageszulassung"}:
        return "Pre-registered"
    if text in {"demonstration", "vorführwagen", "vorfuehrwagen", "demo"}:
        return "Demonstration"
    if text in {"employee's car", "employees car", "jahreswagen"}:
        return "Employee's car"
    return None


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def add_engineered_features(df: pd.DataFrame, current_year: int = CURRENT_YEAR) -> pd.DataFrame:
    """Apply identical feature engineering during training and inference.

    Expected input columns (AutoScout24 schema): ``mileage``, ``make``,
    ``model``, ``fuel``, ``gear``, ``offerType``, ``price``, ``hp``,
    ``year``.
    """
    df = df.copy()

    # Rename make → brand and mileage → mileage_km for clarity
    if "make" in df.columns and "brand" not in df.columns:
        df["brand"] = df["make"].astype(str).str.strip()
    if "mileage" in df.columns and "mileage_km" not in df.columns:
        df["mileage_km"] = df["mileage"]

    # Car age
    df["car_age"] = (current_year - df["year"]).clip(lower=0)

    # Kilometres driven per year of life — proxy for how heavily the car was used
    df["km_per_year"] = df["mileage_km"] / df["car_age"].replace(0, 1)

    # Power per year (helps separate "old-but-fast" from "old-and-tired")
    df["hp_per_year"] = df["hp"] / df["car_age"].replace(0, 1)

    # Body type fallback from brand if not present
    if "body_type" not in df.columns:
        df["body_type"] = df["brand"].map(BRAND_BODY_DEFAULTS).fillna("Sedan")
    else:
        df["body_type"] = df["body_type"].fillna(
            df["brand"].map(BRAND_BODY_DEFAULTS)
        ).fillna("Sedan")

    return df


def body_type_from_brand(brand: Optional[str]) -> str:
    if not brand:
        return "Sedan"
    return BRAND_BODY_DEFAULTS.get(brand, "Sedan")


def seats_from_body_type(body_type: Optional[str]) -> int:
    if not body_type:
        return 5
    return BODY_TYPE_SEATS_DEFAULT.get(body_type, 5)


# --------------------------------------------------------------------------- #
# Feature lists (single source of truth for both training and inference)
# --------------------------------------------------------------------------- #

NUMERIC_FEATURES = [
    "car_age",
    "mileage_km",
    "km_per_year",
    "hp",
    "hp_per_year",
]

CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
    "gear",
    "offerType",
    "body_type",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price"
