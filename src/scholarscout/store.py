"""Vector store for paper passages: local FastEmbed embeddings + Qdrant."""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient, models

from scholarscout.reader import chunk_text
from scholarscout.schemas import Paper

MAX_CHUNKS_PER_PAPER = 40  # about 6,000 words: intro, method and results of most papers


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


DEFAULT_EMBEDDING_MODEL = "snowflake/snowflake-arctic-embed-xs"

# Retrieval models trained with this instruction in front of every query.
_RETRIEVAL_PREFIX = "Represent this sentence for searching relevant passages: "
_QUERY_PREFIXES = {
    "snowflake/snowflake-arctic-embed-xs": _RETRIEVAL_PREFIX,
    "snowflake/snowflake-arctic-embed-s": _RETRIEVAL_PREFIX,
    "BAAI/bge-small-en-v1.5": _RETRIEVAL_PREFIX,
}


class FastEmbedEmbedder:
    """ONNX embeddings on the CPU: no API key, no GPU.

    The default, arctic-embed-xs (384-d), retrieves about as well as bge-small-en-v1.5 on MTEB
    and runs about twice as fast on a laptop CPU, which matters when indexing full papers.
    """

    cache_size = 20_000

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # noisy on Windows
        from fastembed import TextEmbedding  # imported here so tests never download the model

        self._model = TextEmbedding(model_name)
        self._query_prefix = _QUERY_PREFIXES.get(model_name, "")
        self._cache: dict[str, np.ndarray] = {}  # ranking and indexing embed the same abstracts
        self._lock = threading.Lock()
        self.dim = len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
            if missing:
                for text, vector in zip(missing, self._model.passage_embed(missing), strict=True):
                    self._cache[text] = vector
            vectors = [self._cache[text].tolist() for text in texts]
            while len(self._cache) > self.cache_size:
                self._cache.pop(next(iter(self._cache)))  # drop the oldest entry
            return vectors

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            return next(iter(self._model.query_embed(self._query_prefix + text))).tolist()


@dataclass
class Chunk:
    paper_id: str
    text: str
    kind: str  # "abstract" or "full_text"
    score: float = 0.0


def _connect(location: str) -> QdrantClient:
    if location == ":memory:":
        return QdrantClient(location=":memory:")
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location)
    return QdrantClient(path=location)


class PaperStore:
    """Passages of the papers in one run, searchable by meaning.

    In-memory by default; point `location` at a Qdrant server to keep the index.
    """

    def __init__(self, embedder: Embedder, location: str = ":memory:") -> None:
        self.embedder = embedder
        self.client = _connect(location)
        self.collection = f"run_{uuid.uuid4().hex[:12]}"
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE),
        )
        self._lock = threading.Lock()  # the local Qdrant client is not thread-safe
        self._next_id = 0
        self._indexed: set[str] = set()

    def has_paper(self, paper_id: str) -> bool:
        return paper_id in self._indexed

    def add_paper(self, paper: Paper, full_text: str | None = None) -> int:
        """Index the abstract (and full text, if given). Returns the number of passages added."""
        texts = [f"{paper.title}. {paper.abstract}"]
        if full_text:
            texts += chunk_text(full_text)[:MAX_CHUNKS_PER_PAPER]
        vectors = self.embedder.embed_documents(texts)
        with self._lock:
            start, self._next_id = self._next_id, self._next_id + len(texts)
            points = [
                models.PointStruct(
                    id=start + i,
                    vector=vector,
                    payload={
                        "paper_id": paper.id,
                        "text": text,
                        "kind": "abstract" if i == 0 else "full_text",
                    },
                )
                for i, (text, vector) in enumerate(zip(texts, vectors, strict=True))
            ]
            self.client.upsert(self.collection, points=points)
            self._indexed.add(paper.id)
        return len(points)

    def search(
        self,
        query: str,
        k: int = 8,
        paper_ids: Iterable[str] | None = None,
        kind: str | None = None,
    ) -> list[Chunk]:
        conditions: list[models.Condition] = []
        if paper_ids is not None:
            conditions.append(
                models.FieldCondition(key="paper_id", match=models.MatchAny(any=list(paper_ids)))
            )
        if kind:
            conditions.append(
                models.FieldCondition(key="kind", match=models.MatchValue(value=kind))
            )
        vector = self.embedder.embed_query(query)
        with self._lock:
            result = self.client.query_points(
                self.collection,
                query=vector,
                limit=k,
                query_filter=models.Filter(must=conditions) if conditions else None,
                with_payload=True,
            )
        return [
            Chunk(
                paper_id=point.payload["paper_id"],
                text=point.payload["text"],
                kind=point.payload["kind"],
                score=point.score,
            )
            for point in result.points
        ]
