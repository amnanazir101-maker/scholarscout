"""Paper search sources (arXiv, OpenAlex) behind one small interface."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from scholarscout.config import Settings
from scholarscout.schemas import Paper
from scholarscout.sources.arxiv import ArxivSource
from scholarscout.sources.openalex import OpenAlexSource

log = logging.getLogger(__name__)


class PaperSource(Protocol):
    name: str

    def search(self, query: str, max_results: int = 10) -> list[Paper]: ...


def default_sources(settings: Settings) -> list[PaperSource]:
    return [
        ArxivSource(),
        OpenAlexSource(api_key=settings.openalex_api_key, email=settings.openalex_email),
    ]


def search_all(
    sources: Sequence[PaperSource], queries: Iterable[str], max_results: int
) -> list[Paper]:
    """Run every query on every source in parallel. A failing source is logged and skipped."""
    jobs = [(source, query) for query in queries for source in sources]

    def run(job: tuple[PaperSource, str]) -> list[Paper]:
        source, query = job
        try:
            return source.search(query, max_results)
        except Exception as exc:  # network errors, rate limits, malformed responses
            log.warning("%s search failed for %r: %s", source.name, query, exc)
            return []

    with ThreadPoolExecutor(max_workers=8) as pool:
        return [paper for batch in pool.map(run, jobs) for paper in batch]


__all__ = ["ArxivSource", "OpenAlexSource", "PaperSource", "default_sources", "search_all"]
