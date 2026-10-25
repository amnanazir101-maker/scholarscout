"""LLM providers behind one small interface: Gemini (free tier, the default) or Claude."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scholarscout.llm.base import LLM, PRICES, LLMError, Usage, UsageMeter
from scholarscout.llm.claude import ClaudeLLM
from scholarscout.llm.gemini import GeminiLLM

if TYPE_CHECKING:
    from scholarscout.config import Settings

DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "claude": "claude-opus-5-5"}


def make_llm(settings: Settings, *, model: str | None = None, api_key: str | None = None) -> LLM:
    """The LLM for the configured provider. `model` and `api_key` override the settings."""
    provider = settings.provider.strip().lower()
    if provider not in DEFAULT_MODELS:
        raise LLMError(f"Unknown provider {settings.provider!r}: use 'gemini' or 'claude'.")
    name = model or settings.model or DEFAULT_MODELS[provider]
    if provider == "claude":
        return ClaudeLLM(name, settings.effort, settings.refusal_fallback, api_key=api_key)
    return GeminiLLM(
        name,
        settings.effort,
        api_key=api_key,
        max_parallel=settings.gemini_max_parallel,
        fallback_models=[
            m.strip() for m in settings.gemini_fallback_models.split(",") if m.strip()
        ],
    )


__all__ = [
    "DEFAULT_MODELS",
    "LLM",
    "PRICES",
    "ClaudeLLM",
    "GeminiLLM",
    "LLMError",
    "Usage",
    "UsageMeter",
    "make_llm",
]
