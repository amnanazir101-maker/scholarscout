"""Runtime settings, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

# Fields whose environment variable is not SCHOLARSCOUT_<FIELD>.
_ENV_NAMES = {
    "openalex_api_key": "OPENALEX_API_KEY",
    "openalex_email": "OPENALEX_EMAIL",
}


def _coerce(raw: str, default: object) -> object:
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    return raw


@dataclass(frozen=True)
class Settings:
    """Every knob of the pipeline. Set any field with the env var SCHOLARSCOUT_<FIELD_NAME>."""

    # LLM
    provider: str = "gemini"  # "gemini" (free tier) or "claude" (paid)
    model: str = ""  # empty: the provider's default (gemini-3.8-flash or claude-opus-5-5)
    effort: str = "medium"  # thinking effort: low | medium | high (Claude also xhigh | max)
    refusal_fallback: bool = True  # Claude only: retry a declined request on a fallback model
    gemini_max_parallel: int = 2  # Gemini only: requests at once (free limits are per minute)
    # Gemini only: models to switch to when the main one is rate-limited or overloaded
    gemini_fallback_models: str = "gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"

    # Search and reading
    max_papers: int = 15  # papers kept after ranking
    results_per_query: int = 10  # results requested from each source per query
    full_text_papers: int = 4  # top arXiv papers whose PDF is downloaded and read
    chunks_per_question: int = 8  # passages each researcher agent receives

    # Agent loops
    max_follow_up_searches: int = 1  # extra searches a researcher may run when evidence is thin
    max_revisions: int = 1  # critic -> reviser rounds

    # Retrieval
    embedding_model: str = "snowflake/snowflake-arctic-embed-xs"  # local ONNX model, CPU only
    qdrant_location: str = ":memory:"  # ":memory:", a folder path, or a Qdrant server URL

    # OpenAlex (optional, raises the free daily request budget)
    openalex_api_key: str | None = None
    openalex_email: str | None = None

    @classmethod
    def from_env(cls, **overrides: object) -> Settings:
        """Build settings from the environment; keyword overrides that are not None win."""
        values: dict[str, object] = {}
        for field in fields(cls):
            raw = os.getenv(_ENV_NAMES.get(field.name, f"SCHOLARSCOUT_{field.name.upper()}"))
            if raw not in (None, ""):
                values[field.name] = _coerce(raw, field.default)
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)
