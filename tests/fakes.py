"""Offline stand-ins for Claude, the search APIs, the embedding model and the PDF reader."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from scholarscout import prompts
from scholarscout.llm import UsageMeter
from scholarscout.schemas import (
    ClaimVerdict,
    ClaimVerdicts,
    Paper,
    ResearchNote,
    ResearchPlan,
    SubQuestion,
)


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts sharing words get similar vectors."""

    dim = 256

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dim)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return (vector / np.linalg.norm(vector)).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def make_paper(title: str, abstract: str, **fields) -> Paper:
    return Paper(title=title, abstract=abstract, authors=["Ada Lovelace", "Alan Turing"], **fields)


PAPERS = [
    make_paper(
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "Selective state space models scale linearly with sequence length and match "
        "transformers on language modeling.",
        year=2023,
        arxiv_id="2312.00752",
        citations=900,
        source="arxiv",
    ),
    make_paper(
        "Transformers Are Better at Copying Than State Space Models",
        "Transformers outperform state space models on copying and retrieval from long context.",
        year=2024,
        arxiv_id="2402.01032",
        citations=80,
        source="arxiv",
    ),
    make_paper(
        "Long Range Arena: A Benchmark for Efficient Transformers",
        "A benchmark suite for evaluating efficient transformers on long sequences.",
        year=2020,
        doi="10.5555/lra",
        citations=600,
        source="openalex",
    ),
]

FOLLOW_UP_PAPER = make_paper(
    "Hardware-Aware Selective Scan Kernels",
    "A hardware-aware parallel scan makes selective state space models fast on GPUs.",
    year=2024,
    arxiv_id="2405.00001",
    citations=5,
    source="arxiv",
)


class FakeSource:
    name = "fake"

    def __init__(self, papers: list[Paper] | None = None, follow_up: Paper | None = None):
        self.papers = papers if papers is not None else PAPERS
        self.follow_up = follow_up
        self.queries: list[str] = []

    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        self.queries.append(query)
        if self.follow_up and "selective scan" in query:
            return [self.follow_up]
        return [paper.model_copy() for paper in self.papers][:max_results]


class FakeReader:
    def read(self, paper: Paper) -> str | None:
        if not paper.arxiv_id:
            return None
        return f"{paper.title}. Full text. " + " ".join(["The method is evaluated."] * 60)


WRITER_DRAFT = """\
# State space models versus transformers

## TL;DR
Selective state space models scale linearly with sequence length [P1]. Transformers are \
stronger at copying from context [P2].

## Key findings
### Efficiency
Mamba matches transformers on language modeling [P1]. This sentence is UNSUPPORTED by its \
source [P2]. A citation to a paper that does not exist [P99].

## Comparison of approaches
| Approach | Core idea | Strengths | Limitations | Papers |
|---|---|---|---|---|
| Mamba | Selective state spaces | Linear time | Weaker copying | [P1][P2] |

## Open problems
- Hybrid architectures are still poorly understood.
"""

REVISED_DRAFT = WRITER_DRAFT.replace(
    " This sentence is UNSUPPORTED by its source [P2]. A citation to a paper that does not "
    "exist [P99].",
    "",
)


class FakeLLM:
    """Scripted Claude: answers by prompt type so the whole graph runs offline."""

    model = "fake-model"

    def __init__(self, thin_evidence: bool = False, writer_draft: str = WRITER_DRAFT):
        self.thin_evidence = thin_evidence  # first researcher note asks for a follow-up search
        self.writer_draft = writer_draft
        self.calls: list[str] = []
        self.usage = UsageMeter()

    def structured(self, system, prompt, schema, *, effort=None):
        self.calls.append(schema.__name__)
        if schema is ResearchPlan:
            return ResearchPlan(
                sub_questions=[
                    SubQuestion(
                        question="What are selective state space models?",
                        search_query="selective state space models",
                    ),
                    SubQuestion(
                        question="How do they compare with transformers?",
                        search_query="mamba transformer comparison",
                    ),
                ]
            )
        if schema is ResearchNote:
            ids = re.findall(r"^\[(P\d+)\]", prompt, re.MULTILINE)
            asks_follow_up = self.thin_evidence and prompts.RESEARCHER_LAST_TRY not in prompt
            return ResearchNote(
                finding=f"State space models scale linearly [{ids[0]}].",
                evidence_sufficient=not asks_follow_up,
                follow_up_query="selective scan kernels" if asks_follow_up else "",
            )
        if schema is ClaimVerdicts:
            claims = prompt.split("CLAIMS:", 1)[1]
            return ClaimVerdicts(
                verdicts=[
                    ClaimVerdict(
                        claim_id=int(number),
                        supported="UNSUPPORTED" not in text,
                        reason="Checked against the abstract.",
                    )
                    for number, text in re.findall(r"^(\d+)\. (.*)$", claims, re.MULTILINE)
                ]
            )
        raise AssertionError(f"unexpected schema {schema}")

    def text(self, system, prompt, *, effort=None):
        if system == prompts.WRITER_SYSTEM:
            self.calls.append("writer")
            return self.writer_draft
        if system == prompts.REVISER_SYSTEM:
            self.calls.append("reviser")
            return REVISED_DRAFT
        if system == prompts.BASELINE_SYSTEM:
            self.calls.append("baseline")
            return "# Baseline\n\nState space models scale linearly [P1]."
        raise AssertionError("unexpected prompt")
