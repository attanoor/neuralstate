from __future__ import annotations

from dataclasses import dataclass

from neuralstate.config import Settings
from neuralstate.consolidation.reasoner import HeuristicReasoner, LLMReasoner, Reasoner
from neuralstate.providers.base import Embedder


@dataclass
class Providers:
    embedder: Embedder
    reasoner: Reasoner


def build_embedder(s: Settings) -> Embedder:
    from neuralstate.providers import embed

    match s.embed_provider:
        case "fake":
            return embed.HashEmbedder(dim=s.embed_dim)
        case "fastembed":
            return embed.FastEmbedEmbedder(model=s.embed_model, dim=s.embed_dim)
        case "openai":
            return embed.OpenAIEmbedder(model=s.embed_model, dim=s.embed_dim, base_url=s.openai_base_url)
        case "ollama":
            return embed.OllamaEmbedder(model=s.embed_model, dim=s.embed_dim, url=s.ollama_url)
    raise ValueError(f"unknown embed provider {s.embed_provider!r}")


def build_reasoner(s: Settings) -> Reasoner:
    from neuralstate.providers import llm

    match s.llm_provider:
        case "fake":
            return HeuristicReasoner()
        case "anthropic":
            return LLMReasoner(llm.AnthropicLLM(model=s.llm_model, effort=s.llm_effort,
                                                max_tokens=s.llm_max_tokens, fallbacks=s.anthropic_fallbacks))
        case "openai":
            return LLMReasoner(llm.OpenAILLM(model=s.llm_model, base_url=s.openai_base_url,
                                             max_tokens=s.llm_max_tokens))
        case "ollama":
            return LLMReasoner(llm.OllamaLLM(model=s.llm_model, url=s.ollama_url))
    raise ValueError(f"unknown llm provider {s.llm_provider!r}")


def build_providers(s: Settings) -> Providers:
    return Providers(embedder=build_embedder(s), reasoner=build_reasoner(s))
