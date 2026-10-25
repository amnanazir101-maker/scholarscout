"""State shared by the agents in the LangGraph workflow."""

import operator
from typing import Annotated, Any, TypedDict

from scholarscout.schemas import Finding, Paper, Review, SubQuestion


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class ResearchState(TypedDict, total=False):
    question: str
    plan: list[SubQuestion]
    papers: Annotated[dict[str, Paper], merge_dicts]  # by citation ID
    findings: Annotated[list[Finding], operator.add]  # parallel researchers append here
    draft: str  # report with [P#] markers
    review: Review
    revisions: int
    report: str  # final report with numbered references
    cited: list[str]  # paper IDs in reference order
    stats: Annotated[dict[str, Any], merge_dicts]


class ResearchTask(TypedDict):
    """Input of one researcher agent (sent by the fan-out step)."""

    question: str
    sub_question: SubQuestion
