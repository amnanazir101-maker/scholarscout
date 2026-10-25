"""OpenAlex search (https://docs.openalex.org): 250M+ works, with citation counts and venues."""

from __future__ import annotations

import re
import time
from typing import Any

import httpx

from scholarscout.schemas import Paper
from scholarscout.sources.http import new_client

API_URL = "https://api.openalex.org/works"
_FIELDS = (
    "id,doi,display_name,publication_year,cited_by_count,authorships,"
    "abstract_inverted_index,primary_location,open_access"
)
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)$", re.IGNORECASE)


def rebuild_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; put the words back in order."""
    if not inverted_index:
        return ""
    positions = [(pos, word) for word, poses in inverted_index.items() for pos in poses]
    return " ".join(word for _, word in sorted(positions))


def parse_works(data: dict[str, Any]) -> list[Paper]:
    papers = []
    for work in data.get("results", []):
        title = " ".join((work.get("display_name") or "").split())
        abstract = rebuild_abstract(work.get("abstract_inverted_index"))
        if not title or not abstract:
            continue
        doi_url = work.get("doi") or ""
        doi = doi_url.removeprefix("https://doi.org/") or None
        arxiv = _ARXIV_DOI.search(doi or "")
        location = work.get("primary_location") or {}
        open_access = work.get("open_access") or {}
        papers.append(
            Paper(
                title=title,
                abstract=abstract,
                authors=[
                    a["author"]["display_name"]
                    for a in work.get("authorships", [])
                    if (a.get("author") or {}).get("display_name")
                ],
                year=work.get("publication_year"),
                venue=(location.get("source") or {}).get("display_name"),
                url=doi_url or location.get("landing_page_url") or work.get("id", ""),
                pdf_url=location.get("pdf_url") or open_access.get("oa_url"),
                arxiv_id=arxiv.group(1) if arxiv else None,
                doi=doi,
                citations=work.get("cited_by_count"),
                source="openalex",
            )
        )
    return papers


class OpenAlexSource:
    name = "openalex"

    def __init__(
        self,
        http: httpx.Client | None = None,
        api_key: str | None = None,
        email: str | None = None,
    ) -> None:
        self.http = http or new_client()
        self.api_key = api_key
        self.email = email

    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        params: dict[str, Any] = {
            "search": " ".join(re.sub(r"[^\w\s-]", " ", query).split()),  # "?" etc. give a 400
            "per-page": max_results,
            "filter": "has_abstract:true",
            "select": _FIELDS,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["mailto"] = self.email
        response = self.http.get(API_URL, params=params)
        if response.status_code == 429:  # rate limited: wait once, then give up
            time.sleep(2)
            response = self.http.get(API_URL, params=params)
        response.raise_for_status()
        return parse_works(response.json())
