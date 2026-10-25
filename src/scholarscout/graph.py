"""The LangGraph workflow and the public `ResearchAssistant` API.

planner -> searcher -> reader -> researcher x N (parallel) -> writer -> critic
critic -> reviser -> critic (until the report passes or the revision budget is spent)
critic -> finalizer
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from scholarscout.agents import ResearchTeam, ordered_findings
from scholarscout.config import Settings
from scholarscout.llm import LLM, Usage, make_llm
from scholarscout.reader import FullTextReader
from scholarscout.schemas import Finding, Paper, Review, SubQuestion
from scholarscout.sources import PaperSource, default_sources
from scholarscout.state import ResearchState
from scholarscout.store import Embedder, FastEmbedEmbedder, PaperStore


def build_graph(team: ResearchTeam):
    graph = StateGraph(ResearchState)
    graph.add_node("planner", team.plan)
    graph.add_node("searcher", team.search)
    graph.add_node("reader", team.read)
    graph.add_node("researcher", team.research)
    graph.add_node("writer", team.write)
    graph.add_node("critic", team.critique)
    graph.add_node("reviser", team.revise)
    graph.add_node("finalizer", team.finalize)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "reader")
    graph.add_conditional_edges("reader", team.dispatch, ["researcher"])
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic", team.after_review, {"revise": "reviser", "finalize": "finalizer"}
    )
    graph.add_edge("reviser", "critic")
    graph.add_edge("finalizer", END)
    return graph.compile()


@dataclass
class ResearchResult:
    question: str
    report: str  # final Markdown: numbered citations, fact-check notes, references
    draft: str  # the last draft with [P#] markers, as the critic checked it
    plan: list[SubQuestion]
    findings: list[Finding]
    papers: list[Paper]  # every paper the run read
    cited: list[Paper]  # the reference list, in order
    review: Review
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "report": self.report,
            "plan": [sq.model_dump() for sq in self.plan],
            "findings": [f.model_dump() for f in self.findings],
            "papers": [p.model_dump() for p in self.papers],
            "cited": [p.id for p in self.cited],
            "review": self.review.model_dump(),
            "stats": self.stats,
        }


class ResearchAssistant:
    """Multi-agent research assistant: `run(question)` returns a cited, fact-checked review."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLM | None = None,
        sources: Iterable[PaperSource] | None = None,
        embedder: Embedder | None = None,
        reader: FullTextReader | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.llm = llm or make_llm(self.settings)
        self.sources = list(sources) if sources is not None else default_sources(self.settings)
        self.reader = reader or FullTextReader()
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:  # loading the ONNX model takes a few seconds
            self._embedder = FastEmbedEmbedder(self.settings.embedding_model)
        return self._embedder

    def new_team(self) -> ResearchTeam:
        store = PaperStore(self.embedder, self.settings.qdrant_location)
        return ResearchTeam(self.settings, self.llm, self.sources, store, self.reader)

    def stream(self, question: str, team: ResearchTeam | None = None) -> Iterator[tuple[str, Any]]:
        """Yield (agent name, state update) after every agent step, then ("done", result)."""
        team = team or self.new_team()
        meter = getattr(self.llm, "usage", None)
        before = meter.snapshot() if meter else None
        started = time.perf_counter()
        final: dict[str, Any] = {}
        for mode, chunk in build_graph(team).stream(
            {"question": question},
            stream_mode=["updates", "values"],
            config={"recursion_limit": 50},
        ):
            if mode == "updates":
                for agent, update in chunk.items():
                    yield agent, update or {}
            else:
                final = chunk
        usage = meter.snapshot() - before if meter else None
        yield "done", self._result(question, final, team, time.perf_counter() - started, usage)

    def run(self, question: str, team: ResearchTeam | None = None) -> ResearchResult:
        for agent, payload in self.stream(question, team):
            if agent == "done":
                return payload
        raise RuntimeError("The workflow ended without a result.")

    def _result(
        self,
        question: str,
        state: dict[str, Any],
        team: ResearchTeam,
        seconds: float,
        usage: Usage | None,
    ) -> ResearchResult:
        review = state.get("review") or Review()
        cited = [team.registry.get(pid) for pid in state.get("cited", [])]
        stats = dict(state.get("stats", {}))
        stats.update(
            provider=self.settings.provider,
            model=getattr(self.llm, "model", ""),
            papers_read=len(team.registry),
            papers_cited=len(cited),
            claims_checked=len(review.claims),
            claims_supported=review.supported_claims,
            support_rate=round(review.support_rate, 3),
            citation_coverage=round(review.citation_coverage, 3),
            revisions=state.get("revisions", 0),
            seconds=round(seconds, 1),
        )
        if usage is not None:
            cost = usage.cost(getattr(self.llm, "model", ""))
            stats.update(
                llm_calls=usage.calls,
                models_used=usage.models,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=round(cost, 4) if cost is not None else None,
            )
        return ResearchResult(
            question=question,
            report=state.get("report", ""),
            draft=state.get("draft", ""),
            plan=state.get("plan", []),
            findings=ordered_findings(state),
            papers=team.registry.all(),
            cited=cited,
            review=review,
            stats=stats,
        )


def describe(agent: str, update: dict[str, Any]) -> str:
    """One readable line per agent step, for the CLI and the web app."""
    stats = update.get("stats", {})
    if agent == "planner":
        return f"Planner split the question into {len(update['plan'])} sub-questions."
    if agent == "searcher":
        return (
            f"Searcher screened {stats.get('candidates', 0)} papers from arXiv and OpenAlex "
            f"and kept the best {len(update.get('papers', {}))}."
        )
    if agent == "reader":
        return (
            f"Reader indexed {stats.get('passages', 0)} passages "
            f"({stats.get('full_texts', 0)} full-text PDFs)."
        )
    if agent == "researcher":
        finding = update["findings"][0]
        extra = (
            f' Ran a follow-up search: "{finding.follow_up_query}".'
            if finding.follow_up_query
            else ""
        )
        return (
            f'Researcher answered "{finding.sub_question}" '
            f"citing {len(finding.paper_ids)} papers.{extra}"
        )
    if agent == "writer":
        return "Writer drafted the literature review."
    if agent == "critic":
        review = update["review"]
        invalid = (
            f" and {len(review.invalid_citations)} invalid citations"
            if review.invalid_citations
            else ""
        )
        return (
            f"Critic confirmed {review.supported_claims}/{len(review.claims)} cited statements"
            f"{invalid}."
        )
    if agent == "reviser":
        return "Reviser fixed the flagged statements."
    if agent == "finalizer":
        return "Finalizer numbered the citations and built the reference list."
    return f"{agent} finished."
