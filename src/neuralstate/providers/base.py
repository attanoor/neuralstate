from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """A provider call failed in a way the caller should surface (not retry blindly)."""


class LLMRefusal(ProviderError):
    """The model declined the request."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def complete_json(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON object conforming to `schema` (a JSON Schema object)."""
        ...


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one unit-length vector of size `dim` per input text."""
        ...


def check_dims(vectors: list[list[float]], dim: int, provider: str) -> list[list[float]]:
    for v in vectors:
        if len(v) != dim:
            raise ProviderError(
                f"{provider} returned {len(v)}-dim embeddings but NS_EMBED_DIM={dim}. "
                "Set NS_EMBED_DIM to the model's dimension before running migrations."
            )
    return vectors
