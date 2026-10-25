"""Shared HTTP client settings for the search sources and the PDF reader."""

from __future__ import annotations

import httpx

USER_AGENT = "ScholarScout/0.1 (multi-agent research assistant)"


def new_client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT})
