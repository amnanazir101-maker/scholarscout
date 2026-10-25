"""arXiv search through the public Atom API (https://info.arxiv.org/help/api)."""

from __future__ import annotations

import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx

from scholarscout.schemas import Paper
from scholarscout.sources.http import new_client

API_URL = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_STOPWORDS = set(
    "a an and are as at be by can do does for from how in is it of on or such than that the "
    "their these this to used using vs versus what which who why with".split()
)


def build_query(text: str, max_terms: int = 6) -> str:
    """Turn keywords into an arXiv AND query: "all:mamba AND all:transformer".

    A plain space-separated query is an OR search on arXiv, which returns mostly noise.
    """
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]
    terms = list(dict.fromkeys(w for w in words if len(w) > 1))[:max_terms]
    return " AND ".join(f"all:{term}" for term in terms)


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def parse_feed(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", _NS):
        entry_id = entry.findtext("atom:id", "", _NS)
        if "/abs/" not in entry_id:  # error entries point to .../api/errors
            continue
        arxiv_id = re.sub(r"v\d+$", "", entry_id.split("/abs/")[-1])
        links = entry.findall("atom:link", _NS)
        pdf_url = next((link.get("href") for link in links if link.get("title") == "pdf"), None)
        published = entry.findtext("atom:published", "", _NS)
        primary = entry.find("arxiv:primary_category", _NS)
        authors = [
            _clean(author.findtext("atom:name", "", _NS))
            for author in entry.findall("atom:author", _NS)
        ]
        papers.append(
            Paper(
                title=_clean(entry.findtext("atom:title", "", _NS)),
                abstract=_clean(entry.findtext("atom:summary", "", _NS)),
                authors=authors,
                year=int(published[:4]) if published[:4].isdigit() else None,
                venue=f"arXiv ({primary.get('term')})" if primary is not None else "arXiv",
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                arxiv_id=arxiv_id,
                doi=entry.findtext("arxiv:doi", None, _NS),
                source="arxiv",
            )
        )
    return papers


class ArxivSource:
    name = "arxiv"

    def __init__(self, http: httpx.Client | None = None, min_interval: float = 3.0) -> None:
        self.http = http or new_client()
        self.min_interval = min_interval  # arXiv asks API clients to wait 3 s between calls
        self._lock = threading.Lock()
        self._last_call = 0.0

    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        search_query = build_query(query)
        if not search_query:
            return []
        papers = self._fetch(search_query, max_results)
        terms = search_query.split(" AND ")
        if len(papers) < 3 and len(terms) > 3:
            # Too strict: also try the three most important keywords.
            papers += self._fetch(" AND ".join(terms[:3]), max_results)
        return papers

    def _fetch(self, search_query: str, max_results: int) -> list[Paper]:
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.http.get(
                    API_URL,
                    params={
                        "search_query": search_query,
                        "start": 0,
                        "max_results": max_results,
                        "sortBy": "relevance",
                        "sortOrder": "descending",
                    },
                )
            finally:
                self._last_call = time.monotonic()
        response.raise_for_status()
        return parse_feed(response.text)
