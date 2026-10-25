"""Merging duplicate search results, ranking papers, and numbering the ones a run uses."""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from scholarscout.schemas import Paper

if TYPE_CHECKING:
    from scholarscout.store import Embedder


def paper_keys(paper: Paper) -> list[str]:
    """Identity keys: two records sharing any key are the same paper."""
    keys = [f"title:{paper.title_key}"]
    if paper.arxiv_id:
        keys.append(f"arxiv:{paper.arxiv_id.lower()}")
    if paper.doi:
        keys.append(f"doi:{paper.doi.lower()}")
    return keys


def _is_preprint(venue: str | None) -> bool:
    return venue is None or "arxiv" in venue.lower()


def _absorb(target: Paper, other: Paper) -> None:
    """Fill gaps in `target` with what a duplicate record knows."""
    if len(other.abstract) > len(target.abstract):
        target.abstract = other.abstract
    if other.citations is not None:
        target.citations = max(target.citations or 0, other.citations)
    if _is_preprint(target.venue) and not _is_preprint(other.venue):
        target.venue = other.venue  # prefer the journal or conference over arXiv
    target.arxiv_id = target.arxiv_id or other.arxiv_id
    target.doi = target.doi or other.doi
    target.pdf_url = target.pdf_url or other.pdf_url
    target.year = target.year or other.year
    target.authors = target.authors or other.authors


def merge_papers(papers: Iterable[Paper]) -> list[Paper]:
    """Drop duplicates found by several queries or sources, keeping the richest record."""
    merged: list[Paper] = []
    index: dict[str, Paper] = {}
    for paper in papers:
        existing = next((index[key] for key in paper_keys(paper) if key in index), None)
        if existing is None:
            existing = paper.model_copy(deep=True)
            merged.append(existing)
        else:
            _absorb(existing, paper)
        for key in paper_keys(existing):
            index[key] = existing
    return merged


def rank_papers(
    question: str,
    papers: list[Paper],
    embedder: Embedder,
    current_year: int | None = None,
) -> list[Paper]:
    """Sort papers by 0.75 x relevance + 0.15 x citations + 0.10 x recency (each scaled to 0-1).

    Relevance is the cosine similarity between the question and the title + abstract.
    """
    if not papers:
        return []
    current_year = current_year or date.today().year
    query = np.asarray(embedder.embed_query(question))
    docs = np.asarray(embedder.embed_documents([f"{p.title}. {p.abstract}" for p in papers]))
    similarity = docs @ query / (np.linalg.norm(docs, axis=1) * np.linalg.norm(query) + 1e-9)
    spread = similarity.max() - similarity.min()
    relevance = (similarity - similarity.min()) / spread if spread > 0 else np.ones(len(papers))

    most_cited = max(p.citations or 0 for p in papers)
    ranked = []
    for paper, rel in zip(papers, relevance, strict=True):
        cited = math.log1p(paper.citations or 0) / math.log1p(most_cited) if most_cited else 0.0
        age = current_year - (paper.year or current_year - 10)
        recency = min(1.0, max(0.0, 1 - age / 10))
        score = 0.75 * float(rel) + 0.15 * cited + 0.10 * recency
        ranked.append(paper.model_copy(update={"score": round(score, 4)}))
    return sorted(ranked, key=lambda p: p.score, reverse=True)


def id_number(paper_id: str) -> int:
    """'P12' -> 12, for sorting citation IDs naturally."""
    return int(paper_id[1:]) if paper_id[1:].isdigit() else 10**9


class PaperRegistry:
    """Gives every paper used in a run a short citation ID (P1, P2, ...).

    Thread-safe: researcher agents add papers in parallel when they run follow-up searches.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, Paper] = {}
        self._by_key: dict[str, Paper] = {}

    def find(self, paper: Paper) -> Paper | None:
        with self._lock:
            return next((self._by_key[k] for k in paper_keys(paper) if k in self._by_key), None)

    def add(self, paper: Paper) -> tuple[Paper, bool]:
        """Register a paper. Returns (registered paper, whether it is new)."""
        with self._lock:
            for key in paper_keys(paper):
                if key in self._by_key:
                    return self._by_key[key], False
            registered = paper.model_copy(update={"id": f"P{len(self._by_id) + 1}"})
            self._by_id[registered.id] = registered
            for key in paper_keys(registered):
                self._by_key[key] = registered
            return registered, True

    def get(self, paper_id: str) -> Paper | None:
        return self._by_id.get(paper_id)

    def ids(self) -> set[str]:
        with self._lock:
            return set(self._by_id)

    def all(self) -> list[Paper]:
        with self._lock:
            return sorted(self._by_id.values(), key=lambda p: id_number(p.id))

    def __len__(self) -> int:
        return len(self._by_id)
