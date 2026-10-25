"""Data models shared by the agents."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, computed_field


class Paper(BaseModel):
    """A paper returned by one of the search sources."""

    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    url: str = ""
    pdf_url: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    citations: int | None = None  # citation count, when the source knows it
    source: str = ""
    id: str = ""  # short citation ID such as "P3", assigned when a paper joins a run
    score: float = 0.0  # ranking score

    @property
    def title_key(self) -> str:
        return re.sub(r"[^a-z0-9]+", " ", self.title.lower()).strip()

    def author_label(self) -> str:
        if not self.authors:
            return "Unknown authors"
        if len(self.authors) == 1:
            return self.authors[0]
        if len(self.authors) == 2:
            return f"{self.authors[0]} and {self.authors[1]}"
        return f"{self.authors[0]} et al."

    def reference(self) -> str:
        venue = f" *{self.venue}*." if self.venue else ""
        link = f" {self.url}" if self.url else ""
        title = self.title.rstrip(".")
        return f"{self.author_label()} ({self.year or 'n.d.'}). {title}.{venue}{link}"


# --- Structured outputs requested from Claude -------------------------------------------


class SubQuestion(BaseModel):
    question: str = Field(description="A focused sub-question of the research question.")
    search_query: str = Field(
        description="2-6 technical keywords for arXiv/OpenAlex. No boolean operators or quotes."
    )


class ResearchPlan(BaseModel):
    sub_questions: list[SubQuestion]


class ResearchNote(BaseModel):
    finding: str = Field(
        description="Answer to the sub-question in 120-220 words, "
        "every factual sentence cited like [P3]."
    )
    evidence_sufficient: bool = Field(
        description="False if important evidence was missing from the excerpts."
    )
    follow_up_query: str = Field(
        description="When evidence is insufficient: a better 2-6 keyword search query. "
        "Otherwise an empty string."
    )


class ClaimVerdict(BaseModel):
    claim_id: int
    supported: bool
    reason: str = Field(description="One short sentence explaining the verdict.")


class ClaimVerdicts(BaseModel):
    verdicts: list[ClaimVerdict]


# --- Internal results --------------------------------------------------------------------


class Finding(BaseModel):
    """What one researcher agent found for its sub-question."""

    sub_question: str
    text: str
    paper_ids: list[str] = Field(default_factory=list)
    follow_up_query: str | None = None  # the extra search the agent ran, if any


class Claim(BaseModel):
    """A cited statement in a report."""

    id: int
    text: str
    citations: list[str]


class CheckedClaim(Claim):
    supported: bool
    reason: str


class Review(BaseModel):
    """The critic agent's verdict on a report."""

    claims: list[CheckedClaim] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)  # IDs that match no paper
    cited_statements: int = 0
    total_statements: int = 0

    @property
    def unsupported(self) -> list[CheckedClaim]:
        return [claim for claim in self.claims if not claim.supported]

    @computed_field
    @property
    def supported_claims(self) -> int:
        return sum(claim.supported for claim in self.claims)

    @computed_field
    @property
    def support_rate(self) -> float:
        """Share of cited statements that the cited papers actually support."""
        return self.supported_claims / len(self.claims) if self.claims else 0.0

    @computed_field
    @property
    def citation_coverage(self) -> float:
        """Share of statements that carry at least one citation."""
        return self.cited_statements / self.total_statements if self.total_statements else 0.0

    @property
    def passed(self) -> bool:
        return not self.invalid_citations and not self.unsupported
