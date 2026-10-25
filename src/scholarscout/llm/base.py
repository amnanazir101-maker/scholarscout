"""The interface every LLM provider implements, plus token accounting."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# USD per million (input, output) tokens, for cost estimates only. Models that are not listed
# (such as Gemini on the free tier) show no cost.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class LLMError(RuntimeError):
    """The model did not return a usable answer."""


class LLM(Protocol):
    model: str

    def text(self, system: str, prompt: str, *, effort: str | None = None) -> str: ...

    def structured(
        self, system: str, prompt: str, schema: type[T], *, effort: str | None = None
    ) -> T: ...


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    models: dict[str, int] = field(default_factory=dict)  # calls answered by each model

    def __sub__(self, other: Usage) -> Usage:
        models = {
            name: count - other.models.get(name, 0)
            for name, count in self.models.items()
            if count > other.models.get(name, 0)
        }
        return Usage(
            self.calls - other.calls,
            self.input_tokens - other.input_tokens,
            self.output_tokens - other.output_tokens,
            models,
        )

    def cost(self, model: str) -> float | None:
        if model not in PRICES:
            return None
        price_in, price_out = PRICES[model]
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000


class UsageMeter:
    """Thread-safe token counter (researcher agents call the model in parallel)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = Usage()

    def add(self, input_tokens: int, output_tokens: int, model: str) -> None:
        with self._lock:
            self._total.calls += 1
            self._total.input_tokens += input_tokens
            self._total.output_tokens += output_tokens
            self._total.models[model] = self._total.models.get(model, 0) + 1

    def snapshot(self) -> Usage:
        with self._lock:
            total = self._total
            return Usage(total.calls, total.input_tokens, total.output_tokens, dict(total.models))
