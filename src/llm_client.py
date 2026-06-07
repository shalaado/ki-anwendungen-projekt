"""OpenAI helpers used by both the NLP and the Computer Vision blocks.

The two blocks share a single client and a single JSON-parsing routine so
the failure modes (missing key, malformed JSON, etc.) behave the same way
regardless of whether we are extracting structured fields from text or
asking the vision model to classify a car body.
"""

from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from typing import Optional

from PIL import Image


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", OPENAI_MODEL)


def _get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def _strip_json(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return text
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[: -len("```")]
        text = text.strip()
    if not text.startswith("{") and not text.startswith("["):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
    return text.strip()


def parse_json(raw: str, required_keys: tuple[str, ...]) -> dict:
    cleaned = _strip_json(raw)
    if not cleaned:
        raise ValueError("LLM returned an empty response.")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {cleaned[:200]}") from exc
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"LLM JSON is missing required keys: {', '.join(missing)}")
    return data


def call_text(system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
    client = _get_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not set. The NLP/Vision blocks need it.")
    response = client.responses.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = getattr(response, "output_text", "")
    if not text or not text.strip():
        raise ValueError("LLM response was empty.")
    return text.strip()


def call_vision(system_prompt: str, user_prompt: str, image: Image.Image,
                *, temperature: float = 0.0) -> str:
    client = _get_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not set. The Computer Vision block needs it.")

    # Normalise the image to PNG and base64 — works regardless of the
    # original source format (uploaded JPEG, screenshot PNG, etc.).
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = client.responses.create(
        model=OPENAI_VISION_MODEL,
        temperature=temperature,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                ],
            },
        ],
    )
    text = getattr(response, "output_text", "")
    if not text or not text.strip():
        raise ValueError("Vision response was empty.")
    return text.strip()


def has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def active_model() -> str:
    return OPENAI_MODEL


def active_vision_model() -> str:
    return OPENAI_VISION_MODEL
