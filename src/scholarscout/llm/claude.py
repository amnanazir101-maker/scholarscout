"""Claude through the official Anthropic SDK (paid API)."""

from __future__ import annotations

from typing import Any

import anthropic

from scholarscout.llm.base import LLMError, T, UsageMeter

# Models that support the server-side refusal fallback (`fallbacks="default"`).
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeLLM:
    """Calls Claude through the official Anthropic SDK.

    Thinking is adaptive (the default on current models); `effort` sets how hard Claude thinks.
    On models that support it, a declined request is retried server-side on a fallback model.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str | None = "medium",
        refusal_fallback: bool = True,
        client: anthropic.Anthropic | None = None,
        api_key: str | None = None,
        max_tokens: int = 16000,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic(api_key=api_key, max_retries=4)
        self.use_fallback = refusal_fallback and model in _FALLBACK_MODELS
        self.usage = UsageMeter()

    def text(self, system: str, prompt: str, *, effort: str | None = None) -> str:
        params = self._params(system, prompt, effort)
        if self.use_fallback:
            response = self._send(
                self.client.beta.messages.create,
                **params,
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        else:
            response = self._send(self.client.messages.create, **params)
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def structured(
        self, system: str, prompt: str, schema: type[T], *, effort: str | None = None
    ) -> T:
        params = self._params(system, prompt, effort)
        if self.use_fallback:
            response = self._send(
                self.client.beta.messages.parse,
                **params,
                output_format=schema,
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        else:
            response = self._send(self.client.messages.parse, **params, output_format=schema)
        if response.parsed_output is None:
            raise LLMError(f"Claude returned no valid {schema.__name__}.")
        return response.parsed_output

    def _params(self, system: str, prompt: str, effort: str | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.effort:  # an empty effort setting turns the parameter off entirely
            params["output_config"] = {"effort": effort or self.effort}
        return params

    def _send(self, method: Any, **params: Any) -> Any:
        try:
            response = method(**params)
        except TypeError as exc:  # how the SDK reports that no credentials were found
            if "authentication" in str(exc):
                raise LLMError(
                    "No Claude credentials found. Set ANTHROPIC_API_KEY (see .env.example)."
                ) from exc
            raise
        self._check(response)
        return response

    def _check(self, response: Any) -> None:
        usage = response.usage
        self.usage.add(
            (usage.input_tokens or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0),
            usage.output_tokens or 0,
            response.model or self.model,  # a fallback model may have answered
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMError(f"Claude declined the request (category: {category or 'unknown'}).")
        if response.stop_reason == "max_tokens":
            raise LLMError("Claude's answer was cut off by max_tokens.")
