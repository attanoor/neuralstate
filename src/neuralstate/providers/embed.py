"""Embedding adapters. All return L2-normalized vectors so cosine == dot product."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Any

import httpx

from neuralstate.providers.base import ProviderError, check_dims

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have i in is it its of on or that the this to was were will with".split()
)


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashEmbedder:
    """Deterministic, dependency-free embedder (feature hashing over words and bigrams).

    Not semantic, but texts sharing vocabulary land close together, which is enough for
    tests and the offline demo.
    """

    name = "fake"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOP]
        feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = hashlib.blake2b(f.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            v[idx] += 1.0 if h[4] & 1 else -1.0
        if not any(v):
            v[0] = 1.0
        return _normalize(v)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (default: BAAI/bge-small-en-v1.5, 384-dim)."""

    name = "fastembed"

    def __init__(self, model: str | None = None, dim: int = 384):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model or "BAAI/bge-small-en-v1.5")
        self.dim = dim
        self._lock = asyncio.Lock()  # the ONNX session is not re-entrant

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with self._lock:
            vectors = await asyncio.to_thread(lambda: [list(map(float, v)) for v in self.model.embed(texts)])
        return check_dims([_normalize(v) for v in vectors], self.dim, self.name)


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str | None = None, dim: int = 384, base_url: str | None = None, client: Any = None):
        if client is None:
            import openai

            client = openai.AsyncOpenAI(base_url=base_url)
        self.client = client
        self.model = model or "text-embedding-3-small"
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        r = await self.client.embeddings.create(model=self.model, input=texts, dimensions=self.dim)
        return check_dims([_normalize(d.embedding) for d in r.data], self.dim, self.name)


class OllamaEmbedder:
    name = "ollama"

    def __init__(self, model: str | None = None, dim: int = 768, url: str = "http://localhost:11434"):
        self.model = model or "nomic-embed-text"
        self.dim = dim
        self.url = url.rstrip("/")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts})
        if r.status_code >= 400:
            raise ProviderError(f"ollama {r.status_code}: {r.text[:300]}")
        return check_dims([_normalize(v) for v in r.json()["embeddings"]], self.dim, self.name)
