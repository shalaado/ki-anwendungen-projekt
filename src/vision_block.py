"""Computer Vision block: identify the car body type from a user photo.

We use the OpenAI multimodal model in a constrained-choice setup: the model
must pick one of our seven body-type categories and may optionally return a
brand guess. The image is first normalised locally (resized, RGB-converted,
basic sanity-check) before being sent over the API.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image, ImageOps

from data_processing import VALID_BODY_TYPES
from llm_client import call_vision, parse_json


MAX_IMAGE_SIDE = 1024
MIN_IMAGE_SIDE = 64


VISION_SYSTEM = (
    "You are a vehicle classifier. Look at the image and decide which of the "
    "following body types best matches the main vehicle.\n"
    f"Allowed body types: {VALID_BODY_TYPES}.\n"
    "If the image does not clearly show a car (e.g. a person or a landscape), "
    "set is_car to false and use Sedan as body_type with confidence 0.\n\n"
    "Reply with strict JSON, no markdown, with these keys:\n"
    "  - body_type: one of the allowed body types\n"
    "  - confidence: number between 0 and 1\n"
    "  - is_car: boolean\n"
    "  - brand_guess: best guess of the brand, or null if unsure\n"
    "  - reasoning: max one short sentence in English"
)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Validate dimensions, fix orientation, resize for upload."""
    if image is None:
        raise ValueError("No image provided.")
    if min(image.size) < MIN_IMAGE_SIDE:
        raise ValueError(
            f"Image is too small ({image.size[0]}x{image.size[1]}). "
            f"Minimum side length is {MIN_IMAGE_SIDE}px."
        )

    # Honour EXIF orientation so phone photos don't arrive sideways
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    longest = max(image.size)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.LANCZOS)
    return image


def classify_body_type(image: Image.Image) -> dict:
    """Return a dict with body_type, confidence, is_car, brand_guess."""
    prepared = preprocess_image(image)
    user_prompt = (
        "Classify the body type of the vehicle in the picture. "
        f"You must pick one value from {VALID_BODY_TYPES}. "
        "Provide your confidence and an optional brand guess."
    )
    raw = call_vision(VISION_SYSTEM, user_prompt, prepared)
    parsed = parse_json(raw, required_keys=("body_type", "confidence", "is_car"))

    body_type = str(parsed.get("body_type", "Sedan")).strip().title()
    if body_type not in VALID_BODY_TYPES:
        body_type = "Sedan"

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    is_car = bool(parsed.get("is_car", True))

    brand_guess = parsed.get("brand_guess")
    if isinstance(brand_guess, str) and brand_guess.strip():
        brand_guess = brand_guess.strip()
    else:
        brand_guess = None

    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = ""

    return {
        "body_type": body_type,
        "confidence": confidence,
        "is_car": is_car,
        "brand_guess": brand_guess,
        "reasoning": reasoning.strip(),
        "image_size_px": prepared.size,
    }


def image_summary(image: Optional[Image.Image]) -> str:
    if image is None:
        return "no image"
    return f"{image.size[0]}x{image.size[1]}px, mode={image.mode}"
