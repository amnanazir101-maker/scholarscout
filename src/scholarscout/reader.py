"""Full-text reading: download arXiv PDFs, extract the text, split it into passages."""

from __future__ import annotations

import io
import logging
import re

import httpx
from pypdf import PdfReader

from scholarscout.schemas import Paper
from scholarscout.sources.http import new_client

log = logging.getLogger(__name__)

_REFERENCES_HEADING = re.compile(
    r"\n\s*(?:\d+\.?\s*)?(?:references|bibliography)\s*\n", re.IGNORECASE
)


def extract_pdf_text(data: bytes, max_pages: int = 30) -> str:
    """Extract the text of a PDF and drop the bibliography (it only adds retrieval noise)."""
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages[:max_pages]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # pypdf can fail on unusual pages; skip them
            continue
    text = "\n".join(pages)
    headings = list(_REFERENCES_HEADING.finditer(text))
    if headings and headings[-1].start() > len(text) * 0.5:
        text = text[: headings[-1].start()]
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines


def chunk_text(text: str, words_per_chunk: int = 180, overlap: int = 30) -> list[str]:
    """Split text into overlapping passages of about `words_per_chunk` words."""
    words = text.split()
    if not words:
        return []
    step = words_per_chunk - overlap
    return [
        " ".join(words[start : start + words_per_chunk])
        for start in range(0, max(len(words) - overlap, 1), step)
    ]


class FullTextReader:
    """Reads open-access arXiv PDFs. Other papers fall back to their abstract."""

    def __init__(self, http: httpx.Client | None = None, max_pages: int = 30) -> None:
        self.http = http or new_client(timeout=60.0)
        self.max_pages = max_pages

    def read(self, paper: Paper) -> str | None:
        if not paper.arxiv_id:
            return None
        url = f"https://arxiv.org/pdf/{paper.arxiv_id}"
        try:
            response = self.http.get(url)
            response.raise_for_status()
            if "pdf" not in response.headers.get("content-type", ""):
                return None
            return extract_pdf_text(response.content, self.max_pages) or None
        except Exception as exc:  # a missing PDF should never stop the run
            log.warning("Could not read %s: %s", url, exc)
            return None
