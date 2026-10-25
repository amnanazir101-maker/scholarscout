"""Single-agent RAG baseline for the evaluation: one search, one retrieval, one Claude call.

It sees the same sources, ranking, full-text reading and evidence budget as ScholarScout, so the
comparison measures what the multi-agent design adds (planning, parallel research, follow-up
searches and the critic loop).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from scholarscout import prompts
from scholarscout.agents import MAX_SUB_QUESTIONS, format_excerpts
from scholarscout.citations import normalize_citations
from scholarscout.config import Settings
from scholarscout.llm import LLM
from scholarscout.papers import PaperRegistry, merge_papers, rank_papers
from scholarscout.reader import FullTextReader
from scholarscout.sources import PaperSource, search_all
from scholarscout.store import Embedder, PaperStore


@dataclass
class BaselineRun:
    draft: str  # Markdown with [P#] markers
    registry: PaperRegistry
    store: PaperStore
    seconds: float


def run_baseline(
    question: str,
    settings: Settings,
    llm: LLM,
    sources: Sequence[PaperSource],
    embedder: Embedder,
    reader: FullTextReader,
) -> BaselineRun:
    started = time.perf_counter()
    registry = PaperRegistry()
    store = PaperStore(embedder)

    candidates = merge_papers(search_all(sources, [question], settings.results_per_query))
    ranked = rank_papers(question, candidates, embedder)[: settings.max_papers]
    papers = [registry.add(paper)[0] for paper in ranked]
    to_read = [paper for paper in papers if paper.arxiv_id][: settings.full_text_papers]
    texts = {paper.id: reader.read(paper) for paper in to_read}
    for paper in papers:
        store.add_paper(paper, texts.get(paper.id))

    # Same evidence budget as all researcher agents together.
    chunks = store.search(question, k=settings.chunks_per_question * MAX_SUB_QUESTIONS)
    prompt = prompts.BASELINE_PROMPT.format(
        question=question, excerpts=format_excerpts(chunks, registry)
    )
    draft = normalize_citations(llm.text(prompts.BASELINE_SYSTEM, prompt))
    return BaselineRun(draft, registry, store, time.perf_counter() - started)
