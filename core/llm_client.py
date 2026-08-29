"""Shared LLM client for baseline and agent workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    model: str
    api_key: str

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip())


def get_llm_config() -> LlmConfig:
    return LlmConfig(
        provider=(os.getenv("LLM_PROVIDER") or "gemini").strip().lower(),
        model=(os.getenv("LLM_MODEL") or "gemini-2.5-flash").strip(),
        api_key=(os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY") or "").strip(),
    )


class LlmError(RuntimeError):
    pass


class LlmNotConfiguredError(LlmError):
    pass


async def complete_json(system_prompt: str, user_prompt: str) -> str:
    """
    Single LLM call returning raw text (expected JSON).

    Uses temperature 0 and responseMimeType application/json for Gemini.
    """
    config = get_llm_config()
    if not config.is_configured:
        raise LlmNotConfiguredError(
            "GEMINI_API_KEY is not set. Use --fixtures for offline runs or set the key in .env"
        )

    if config.provider != "gemini":
        raise LlmError(f"Unsupported LLM_PROVIDER: {config.provider}")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.model}:generateContent"
    )
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]},
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, params={"key": config.api_key}, json=body)

    if response.status_code == 429:
        raise LlmError(f"LLM quota exceeded for model {config.model}")
    if response.status_code >= 400:
        raise LlmError(f"LLM request failed ({response.status_code}): {response.text[:500]}")

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError("LLM returned an empty response") from exc
