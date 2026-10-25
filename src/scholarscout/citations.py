"""Citation markers ([P3]): parsing, checking, and turning them into numbered references."""

from __future__ import annotations

import re
from collections.abc import Iterable

from scholarscout.schemas import Claim, Paper

_GROUP = re.compile(r"\[\s*(P\d+(?:\s*[,;]\s*P\d+)+)\s*\]")  # [P1, P3] or [P1; P3]
_MARKER = re.compile(r"\[(P\d+)\]")
_MARKER_RUN = re.compile(r"\[P\d+\](?:\s*\[P\d+\])*")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'*])")
_ABBREVIATION = re.compile(r"\b(e\.g|i\.e|et al|vs|cf|approx|Fig|Eq|Sec)\.", re.IGNORECASE)
_BULLET = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_TABLE_RULE = re.compile(r":?-+:?")
_UNCOUNTED_SECTIONS = ("open problem", "future")  # opinions, not claims that need a source
_DOT = "․"  # stands in for abbreviation dots while splitting sentences


def normalize_citations(text: str) -> str:
    """Rewrite grouped markers such as [P1, P3] as [P1][P3]."""
    return _GROUP.sub(
        lambda m: "".join(f"[{pid}]" for pid in re.findall(r"P\d+", m.group(1))), text
    )


def cited_ids(text: str) -> list[str]:
    """Paper IDs in order of first appearance."""
    return list(dict.fromkeys(_MARKER.findall(normalize_citations(text))))


def strip_citations(text: str) -> str:
    """The statement without its markers, e.g. for showing it to the fact-checker."""
    without = " ".join(_MARKER.sub("", text).split())
    return re.sub(r"\s+([.,;:!?])", r"\1", without)


def remove_citations(text: str, ids: Iterable[str]) -> str:
    """Delete the given markers, e.g. IDs that match no paper."""
    drop = set(ids)
    if not drop:
        return text
    cleaned = _MARKER.sub(lambda m: "" if m.group(1) in drop else m.group(0), text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)


def _sentences(line: str) -> list[str]:
    protected = _ABBREVIATION.sub(lambda m: m.group(1) + _DOT, line)
    sentences: list[str] = []
    for part in _SENTENCE_BREAK.split(protected):
        part = part.replace(_DOT, ".")
        if sentences and not strip_citations(part).strip(" ."):
            sentences[-1] += " " + part  # a trailing "[P2]." belongs to the sentence before it
        else:
            sentences.append(part)
    return sentences


def extract_claims(markdown: str) -> tuple[list[Claim], int, int]:
    """Split a Markdown report into statements and keep the cited ones as claims.

    A statement is a prose sentence or a table data row. Returns
    (claims, cited statements, all statements); headings and open-problem bullets are not
    counted as statements, but any citation in them is still checked.
    """
    claims: list[Claim] = []
    counts = {"cited": 0, "total": 0}
    section = ""
    headers: list[str] | None = None

    def record(text: str, ids: list[str], countable: bool) -> None:
        if countable:
            counts["total"] += 1
            counts["cited"] += bool(ids)
        if ids:
            claims.append(Claim(id=len(claims) + 1, text=text, citations=list(dict.fromkeys(ids))))

    for raw_line in normalize_citations(markdown).splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            headers = None
        if not line:
            continue
        if line.startswith("#"):
            section = line.lstrip("#").strip().lower()
            continue
        countable = not any(word in section for word in _UNCOUNTED_SECTIONS)

        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(_TABLE_RULE.fullmatch(cell) for cell in cells):
                continue
            if headers is None:  # the first row of a table is its header
                headers = [strip_citations(cell) for cell in cells]
                continue
            text = "; ".join(
                f"{header}: {strip_citations(cell)}"
                for header, cell in zip(headers, cells, strict=False)  # rows can be ragged
                if strip_citations(cell) and not header.lower().startswith("paper")
            )
            record(text, _MARKER.findall(line), countable)
            continue

        for sentence in _sentences(_BULLET.sub("", line)):
            text = strip_citations(sentence)
            if len(text.split()) >= 3:  # skip labels such as "**Speed:**"
                record(text, _MARKER.findall(sentence), countable)

    return claims, counts["cited"], counts["total"]


def number_citations(text: str, order: list[str]) -> str:
    """Replace marker runs such as [P7][P2] with reference numbers: [1, 3]."""
    numbers = {pid: n for n, pid in enumerate(order, 1)}

    def replace(match: re.Match[str]) -> str:
        found = sorted({numbers[pid] for pid in _MARKER.findall(match.group(0)) if pid in numbers})
        return f"[{', '.join(map(str, found))}]" if found else ""

    return _MARKER_RUN.sub(replace, normalize_citations(text))


def reference_list(papers: list[Paper]) -> str:
    return "\n".join(f"{n}. {paper.reference()}" for n, paper in enumerate(papers, 1))
