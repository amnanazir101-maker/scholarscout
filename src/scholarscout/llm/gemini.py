"""Gemini through the official Google Gen AI SDK. A free AI Studio key is enough."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from scholarscout.llm.base import LLMError, T, UsageMeter

log = logging.getLogger(__name__)

# Gemini 3 models take a thinking level (2.5 models used token budgets instead).
_THINKING_LEVELS = {
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
    "xhigh": types.ThinkingLevel.HIGH,
    "max": types.ThinkingLevel.HIGH,
}

# Server errors get a few quick retries. A rate-limited model (429) is not retried: every model
# has its own free quota, so the run moves straight to the next fallback model.
_HTTP_OPTIONS = types.HttpOptions(
    timeout=180_000,  # milliseconds
    retry_options=types.HttpRetryOptions(
        attempts=4,
        initial_delay=5.0,
        max_delay=30.0,
        exp_base=2.0,
        http_status_codes=[500, 502, 503, 504],
    ),
)
_BUSY = {429: "its free-tier limit was reached", 503: "it is overloaded"}


class GeminiLLM:
    """Calls Gemini through the Google Gen AI SDK.

    At most `max_parallel` requests run at once, so the parallel agents stay under the free
    tier's per-minute limits. Each model has its own free quota: when a model is rate-limited
    or overloaded, the rest of the run moves to the next model in `fallback_models`.
    """

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        effort: str | None = "medium",
        client: genai.Client | None = None,
        api_key: str | None = None,
        max_parallel: int = 2,
        fallback_models: Sequence[str] = (),
        rate_limit_waits: Sequence[float] = (15, 30, 60),
    ) -> None:
        self.model = model
        self.effort = effort
        self.rate_limit_waits = tuple(rate_limit_waits)  # pauses before retrying the last model
        try:
            self.client = client or genai.Client(api_key=api_key, http_options=_HTTP_OPTIONS)
        except ValueError as exc:  # raised when no API key is configured
            raise LLMError(
                "No Gemini API key found. Get a free key at https://aistudio.google.com/apikey "
                "and set GEMINI_API_KEY (see .env.example)."
            ) from exc
        self.models = list(dict.fromkeys([model, *fallback_models]))
        self._active = 0  # index of the model new requests start with
        self._switch = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_parallel)
        self.usage = UsageMeter()

    def text(self, system: str, prompt: str, *, effort: str | None = None) -> str:
        return self._generate(system, prompt, effort).text.strip()

    def structured(
        self, system: str, prompt: str, schema: type[T], *, effort: str | None = None
    ) -> T:
        response = self._generate(system, prompt, effort, schema)
        if isinstance(response.parsed, schema):
            return response.parsed
        try:
            return schema.model_validate_json(response.text)
        except ValidationError as exc:
            raise LLMError(f"Gemini returned no valid {schema.__name__}.") from exc

    def _generate(
        self, system: str, prompt: str, effort: str | None, schema: type[T] | None = None
    ) -> Any:
        index, waits = self._active, list(self.rate_limit_waits)
        while True:
            model = self.models[index]
            try:
                with self._slots:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=self._config(model, system, effort, schema),
                    )
            except errors.APIError as exc:
                if exc.code in _BUSY and index + 1 < len(self.models):
                    self._fall_back(index, _BUSY[exc.code])
                    index = max(index + 1, self._active)
                    continue
                if exc.code == 429 and waits:  # last model: wait out a per-minute limit
                    time.sleep(waits.pop(0))
                    continue
                raise self._error(exc) from exc
            self._check(response, model)
            return response

    def _config(
        self, model: str, system: str, effort: str | None, schema: type[T] | None
    ) -> types.GenerateContentConfig:
        config = types.GenerateContentConfig(system_instruction=system)
        level = _THINKING_LEVELS.get((effort or self.effort or "").lower())
        if level and model.startswith("gemini-3"):
            config.thinking_config = types.ThinkingConfig(thinking_level=level)
        if schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = schema
        return config

    def _fall_back(self, index: int, reason: str) -> None:
        with self._switch:
            if self._active <= index:
                self._active = index + 1
                log.warning(
                    "%s is unavailable (%s); switching to %s.",
                    self.models[index],
                    reason,
                    self.models[index + 1],
                )

    def _error(self, exc: errors.APIError) -> LLMError:
        tried = ", ".join(self.models)
        if exc.code == 429:
            return LLMError(
                f"Gemini's free-tier limit was reached on every model tried ({tried}). "
                "Wait a few minutes (daily limits reset after midnight Pacific time) and retry."
            )
        if exc.code == 503:
            return LLMError(
                f"Gemini is overloaded on every model tried ({tried}). Try again in a few minutes."
            )
        return LLMError(f"Gemini API error {exc.code}: {exc.message}")

    def _check(self, response: Any, model: str) -> None:
        usage = response.usage_metadata
        if usage is not None:
            self.usage.add(
                usage.prompt_token_count or 0,
                (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
                model,
            )
        feedback = response.prompt_feedback
        if feedback is not None and feedback.block_reason:
            raise LLMError(f"Gemini blocked the request ({feedback.block_reason}).")
        candidate = response.candidates[0] if response.candidates else None
        reason = candidate.finish_reason if candidate else None
        if reason == types.FinishReason.MAX_TOKENS:
            raise LLMError("Gemini's answer was cut off by the output token limit.")
        if not response.text:
            raise LLMError(f"Gemini returned an empty answer (finish reason: {reason}).")
