"""SQLAlchemy models.

Statuses and kinds are plain strings with CHECK constraints rather than Postgres
ENUM types, so adding a value never needs an ``ALTER TYPE`` migration.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from neuralstate.config import get_settings

EMBED_DIM = get_settings().embed_dim


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Outcome(StrEnum):
    success = "success"
    failure = "failure"
    partial = "partial"
    unknown = "unknown"


class NodeKind(StrEnum):
    entity = "entity"
    fact = "fact"
    concept = "concept"
    rule = "rule"


class NodeStatus(StrEnum):
    active = "active"
    superseded = "superseded"
    archived = "archived"
    pending_review = "pending_review"
    disabled = "disabled"  # rules switched off by an operator


class ContradictionStatus(StrEnum):
    auto = "auto"  # resolved automatically by score margin
    pending = "pending"  # waiting in the review queue
    resolved = "resolved"  # resolved by an operator


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


def _check(column: str, enum: type[StrEnum]) -> CheckConstraint:
    values = ", ".join(f"'{v.value}'" for v in enum)
    return CheckConstraint(f"{column} IN ({values})", name=f"ck_{column}_valid")


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONB, list[Any]: JSONB}


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Episode(Base):
    __tablename__ = "episodes"
    __table_args__ = (
        _check("outcome", Outcome),
        Index("ix_episodes_agent_unconsolidated", "agent_id", "created_at",
              postgresql_where=text("consolidated_at IS NULL")),
        Index("ix_episodes_agent_outcome", "agent_id", "outcome", "created_at"),
        Index("ix_episodes_content_gin", "content", postgresql_using="gin"),
        Index("ix_episodes_embedding_hnsw", "embedding", postgresql_using="hnsw",
              postgresql_ops={"embedding": "vector_cosine_ops"}),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    session_id: Mapped[str | None] = mapped_column(String(200))
    task_type: Mapped[str | None] = mapped_column(String(200))
    content: Mapped[dict[str, Any]] = mapped_column(default=dict)
    summary: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(String(16), default=Outcome.unknown.value)
    outcome_detail: Mapped[str | None] = mapped_column(Text)
    # Rules the agent had in context when it acted; used to measure rule effectiveness.
    applied_rule_ids: Mapped[list[Any]] = mapped_column(default=list)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    consolidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    compacted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Node(Base):
    __tablename__ = "nodes"
    __table_args__ = (
        _check("kind", NodeKind),
        _check("status", NodeStatus),
        Index("ix_nodes_agent_kind_status", "agent_id", "kind", "status"),
        Index("ix_nodes_agent_key", "agent_id", "key"),
        Index("ix_nodes_embedding_hnsw", "embedding", postgresql_using="hnsw",
              postgresql_ops={"embedding": "vector_cosine_ops"}),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(16))
    # Normalized lookup key: entity name, or "subject|predicate" for facts.
    key: Mapped[str] = mapped_column(String(500))
    label: Mapped[str] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column(Text)
    attrs: Mapped[dict[str, Any]] = mapped_column(default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    # `confidence` = `base_confidence` after time decay; recomputed (not compounded) each cycle.
    base_confidence: Mapped[float] = mapped_column(Float, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    status: Mapped[str] = mapped_column(String(16), default=NodeStatus.active.value)
    evidence_count: Mapped[int] = mapped_column(Integer, default=1)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_evidence_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    rule: Mapped[HeuristicRule | None] = relationship(back_populates="node", uselist=False, lazy="raise")


class Edge(Base):
    __tablename__ = "edges"
    __table_args__ = (
        UniqueConstraint("src_id", "dst_id", "relation", name="uq_edges_src_dst_relation"),
        Index("ix_edges_src", "src_id"),
        Index("ix_edges_dst", "dst_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    src_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"))
    dst_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"))
    relation: Mapped[str] = mapped_column(String(200))
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    attrs: Mapped[dict[str, Any]] = mapped_column(default=dict)
    status: Mapped[str] = mapped_column(String(16), default=NodeStatus.active.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Provenance(Base):
    """Evidence link: which episode supports (or contradicts) which node."""

    __tablename__ = "provenance"
    __table_args__ = (
        CheckConstraint("role IN ('supports', 'contradicts')", name="ck_role_valid"),
        Index("ix_provenance_episode", "episode_id"),
    )

    node_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"), primary_key=True)
    episode_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), primary_key=True)
    role: Mapped[str] = mapped_column(String(16), primary_key=True, default="supports")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class HeuristicRule(Base):
    """Rule-specific fields. The rule's text, embedding, confidence and status live on its Node."""

    __tablename__ = "heuristic_rules"

    node_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"), primary_key=True)
    task_type: Mapped[str | None] = mapped_column(String(200))
    condition: Mapped[str] = mapped_column(Text)
    avoid: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str] = mapped_column(Text)
    success_after: Mapped[int] = mapped_column(Integer, default=0)
    failure_after: Mapped[int] = mapped_column(Integer, default=0)

    node: Mapped[Node] = relationship(back_populates="rule", lazy="raise")


class Contradiction(Base):
    __tablename__ = "contradictions"
    __table_args__ = (
        _check("status", ContradictionStatus),
        UniqueConstraint("node_a", "node_b", name="uq_contradictions_pair"),
        Index("ix_contradictions_agent_status", "agent_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    node_a: Mapped[uuid.UUID] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"))
    node_b: Mapped[uuid.UUID] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"))
    winner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    margin: Mapped[float] = mapped_column(Float)
    reason: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))
    resolution: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemoryEvent(Base):
    """Append-only audit log of every memory mutation; `before` makes each one revertible."""

    __tablename__ = "memory_events"
    __table_args__ = (Index("ix_memory_events_agent_created", "agent_id", "created_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    agent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    cycle_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    op: Mapped[str] = mapped_column(String(40))
    target_type: Mapped[str] = mapped_column(String(40))
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    note: Mapped[str | None] = mapped_column(Text)
    reverted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ConsolidationJob(Base):
    __tablename__ = "consolidation_jobs"
    __table_args__ = (
        _check("status", JobStatus),
        CheckConstraint("kind IN ('cycle', 'synthesize')", name="ck_kind_valid"),
        # At most one queued job per (agent, kind): enqueueing is idempotent.
        Index("uq_jobs_one_queued", "agent_id", "kind", unique=True,
              postgresql_where=text("status = 'queued'")),
        Index("ix_jobs_claim", "status", "run_after"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.queued.value)
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConsolidationCycle(Base):
    __tablename__ = "consolidation_cycles"
    __table_args__ = (Index("ix_cycles_agent_started", "agent_id", "started_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    kind: Mapped[str] = mapped_column(String(16), default="cycle")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stats: Mapped[dict[str, Any]] = mapped_column(default=dict)
    error: Mapped[str | None] = mapped_column(Text)
