"""Small graph-write helpers shared by ingest and consolidation."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from neuralstate.db.models import Edge, Provenance


async def add_provenance(session: AsyncSession, node_id: uuid.UUID, episode_ids: list[uuid.UUID],
                         role: str = "supports") -> None:
    if not episode_ids:
        return
    stmt = insert(Provenance).values(
        [{"node_id": node_id, "episode_id": e, "role": role} for e in dict.fromkeys(episode_ids)]
    ).on_conflict_do_nothing()
    await session.execute(stmt)


async def copy_provenance(session: AsyncSession, src_node: uuid.UUID, dst_node: uuid.UUID) -> None:
    rows = (await session.execute(
        select(Provenance.episode_id, Provenance.role).where(Provenance.node_id == src_node)
    )).all()
    for role in {r.role for r in rows}:
        await add_provenance(session, dst_node, [r.episode_id for r in rows if r.role == role], role)


async def add_edge(session: AsyncSession, agent_id: uuid.UUID, src: uuid.UUID, dst: uuid.UUID, relation: str,
                   weight: float = 1.0) -> None:
    if src == dst:
        return
    stmt = insert(Edge).values(
        id=uuid.uuid4(), agent_id=agent_id, src_id=src, dst_id=dst, relation=relation[:200], weight=weight,
        attrs={}, status="active",
    ).on_conflict_do_nothing(constraint="uq_edges_src_dst_relation")
    await session.execute(stmt)
