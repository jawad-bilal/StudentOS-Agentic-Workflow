"""Shared neutral JSON parsing for LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)


class JsonParseError(ValueError):
    """Raised when LLM output cannot be parsed as JSON."""


def strip_optional_fence(raw_text: str) -> str:
    text = raw_text.strip()
    match = _FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text


def parse_llm_json(raw_text: str) -> Any:
    """
    Neutral parser used by baseline and agent workflow.

    1. json.loads(raw_text)
    2. If that fails, strip one outer ```json fence and try once more
    3. Otherwise fail (no partial extraction, repair, or LLM retry)
    """
    text = raw_text.strip()
    if not text:
        raise JsonParseError("Empty LLM response")

    try:
        return json.loads(text)
    except json.JSONDecodeError as first_err:
        unfenced = strip_optional_fence(text)
        if unfenced == text:
            raise JsonParseError(f"Invalid JSON: {first_err}") from first_err
        try:
            return json.loads(unfenced)
        except json.JSONDecodeError as second_err:
            raise JsonParseError(
                f"Invalid JSON after fence removal: {second_err}"
            ) from second_err
