from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration, read from environment variables prefixed ``NS_`` (or a ``.env`` file)."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="NS_", extra="ignore")

    database_url: str = "postgresql+asyncpg://neuralstate:neuralstate@127.0.0.1:5432/neuralstate"
    db_schema: str | None = None  # optional search_path override (used by tests)

    # Auth: when set, the REST API requires `X-API-Key` and the dashboard requires a login.
    api_key: str | None = None

    # --- Providers ---------------------------------------------------------
    llm_provider: Literal["anthropic", "openai", "ollama", "fake"] = "anthropic"
    llm_model: str | None = None  # provider default when unset
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"  # Anthropic only
    llm_max_tokens: int = 16000
    anthropic_fallbacks: bool = True  # server-side refusal fallbacks
    openai_base_url: str | None = None
    ollama_url: str = "http://localhost:11434"

    embed_provider: Literal["fastembed", "openai", "ollama", "fake"] = "fastembed"
    embed_model: str | None = None
    embed_dim: int = 384  # must match the model; fixed at migration time

    # --- Consolidation tunables -------------------------------------------
    extract_batch_size: int = 10
    max_episodes_per_cycle: int = 500
    reconcile_neighbor_threshold: float = 0.80  # facts this similar are judged: duplicate / contradiction / consistent
    # Relative strength margin needed to auto-resolve a contradiction. 0.20 resolves e.g. 4:1
    # evidence, or a fresh observation against a month-old single one; closer calls go to review.
    contradiction_margin: float = 0.20
    min_rule_evidence: int = 5  # failures needed before a heuristic rule is synthesized
    cluster_threshold: float = 0.70  # cosine similarity for grouping failures
    rule_merge_threshold: float = 0.88
    synthesis_window_days: int = 60
    rule_archive_min_trials: int = 3
    rule_archive_effectiveness: float = 0.30
    decay_lambda: float = 0.02  # per day, after the grace period
    decay_grace_days: float = 7.0
    archive_threshold: float = 0.15
    archive_after_days: float = 30.0
    compact_episodes: bool = False
    compact_after_days: float = 30.0

    # --- Worker -------------------------------------------------------------
    cycle_episode_threshold: int = 20
    cycle_interval_seconds: int = 900
    worker_poll_seconds: float = 2.0
    job_max_attempts: int = 5
    job_stale_seconds: int = 1800

    # --- Recall ---------------------------------------------------------------
    recall_k: int = 8
    # Below this cosine similarity nothing is recalled. Tuned for sentence-embedding models (bge, OpenAI),
    # where unrelated texts still score ~0.4.
    recall_min_similarity: float = 0.50


@lru_cache
def get_settings() -> Settings:
    return Settings()
