"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-10-19
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from neuralstate.config import get_settings

# Embedding width is fixed when the tables are created; changing models later needs a migration.
EMBED_DIM = get_settings().embed_dim

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.create_table('agents',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('consolidation_cycles',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('agent_id', sa.UUID(), nullable=False),
    sa.Column('job_id', sa.UUID(), nullable=True),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('stats', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_cycles_agent_started', 'consolidation_cycles', ['agent_id', 'started_at'], unique=False)
    op.create_table('consolidation_jobs',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('agent_id', sa.UUID(), nullable=False),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('attempts', sa.Integer(), nullable=False),
    sa.Column('last_error', sa.Text(), nullable=True),
    sa.Column('run_after', sa.DateTime(timezone=True), nullable=False),
    sa.Column('locked_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("kind IN ('cycle', 'synthesize')", name='ck_kind_valid'),
    sa.CheckConstraint("status IN ('queued', 'running', 'done', 'failed')", name='ck_status_valid'),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_jobs_claim', 'consolidation_jobs', ['status', 'run_after'], unique=False)
    op.create_index('uq_jobs_one_queued', 'consolidation_jobs', ['agent_id', 'kind'], unique=True, postgresql_where=sa.text("status = 'queued'"))
    op.create_table('episodes',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('agent_id', sa.UUID(), nullable=False),
    sa.Column('session_id', sa.String(length=200), nullable=True),
    sa.Column('task_type', sa.String(length=200), nullable=True),
    sa.Column('content', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('summary', sa.Text(), nullable=False),
    sa.Column('outcome', sa.String(length=16), nullable=False),
    sa.Column('outcome_detail', sa.Text(), nullable=True),
    sa.Column('applied_rule_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('embedding', Vector(EMBED_DIM), nullable=True),
    sa.Column('consolidated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('compacted', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("outcome IN ('success', 'failure', 'partial', 'unknown')", name='ck_outcome_valid'),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_episodes_agent_outcome', 'episodes', ['agent_id', 'outcome', 'created_at'], unique=False)
    op.create_index('ix_episodes_agent_unconsolidated', 'episodes', ['agent_id', 'created_at'], unique=False, postgresql_where=sa.text('consolidated_at IS NULL'))
    op.create_index('ix_episodes_content_gin', 'episodes', ['content'], unique=False, postgresql_using='gin')
    op.create_index('ix_episodes_embedding_hnsw', 'episodes', ['embedding'], unique=False, postgresql_using='hnsw', postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.create_table('memory_events',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('agent_id', sa.UUID(), nullable=False),
    sa.Column('cycle_id', sa.UUID(), nullable=True),
    sa.Column('op', sa.String(length=40), nullable=False),
    sa.Column('target_type', sa.String(length=40), nullable=False),
    sa.Column('target_id', sa.UUID(), nullable=False),
    sa.Column('before', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('after', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('note', sa.Text(), nullable=True),
    sa.Column('reverted', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_memory_events_agent_created', 'memory_events', ['agent_id', 'created_at'], unique=False)
    op.create_table('nodes',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('agent_id', sa.UUID(), nullable=False),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('key', sa.String(length=500), nullable=False),
    sa.Column('label', sa.Text(), nullable=False),
    sa.Column('body', sa.Text(), nullable=True),
    sa.Column('attrs', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('embedding', Vector(EMBED_DIM), nullable=True),
    sa.Column('base_confidence', sa.Float(), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('evidence_count', sa.Integer(), nullable=False),
    sa.Column('access_count', sa.Integer(), nullable=False),
    sa.Column('last_accessed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_evidence_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("kind IN ('entity', 'fact', 'concept', 'rule')", name='ck_kind_valid'),
    sa.CheckConstraint("status IN ('active', 'superseded', 'archived', 'pending_review', 'disabled')", name='ck_status_valid'),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_nodes_agent_key', 'nodes', ['agent_id', 'key'], unique=False)
    op.create_index('ix_nodes_agent_kind_status', 'nodes', ['agent_id', 'kind', 'status'], unique=False)
    op.create_index('ix_nodes_embedding_hnsw', 'nodes', ['embedding'], unique=False, postgresql_using='hnsw', postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.create_table('contradictions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('agent_id', sa.UUID(), nullable=False),
    sa.Column('node_a', sa.UUID(), nullable=False),
    sa.Column('node_b', sa.UUID(), nullable=False),
    sa.Column('winner_id', sa.UUID(), nullable=True),
    sa.Column('margin', sa.Float(), nullable=False),
    sa.Column('reason', sa.Text(), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('resolution', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("status IN ('auto', 'pending', 'resolved')", name='ck_status_valid'),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['node_a'], ['nodes.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['node_b'], ['nodes.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('node_a', 'node_b', name='uq_contradictions_pair')
    )
    op.create_index('ix_contradictions_agent_status', 'contradictions', ['agent_id', 'status'], unique=False)
    op.create_table('edges',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('agent_id', sa.UUID(), nullable=False),
    sa.Column('src_id', sa.UUID(), nullable=False),
    sa.Column('dst_id', sa.UUID(), nullable=False),
    sa.Column('relation', sa.String(length=200), nullable=False),
    sa.Column('weight', sa.Float(), nullable=False),
    sa.Column('attrs', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['dst_id'], ['nodes.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['src_id'], ['nodes.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('src_id', 'dst_id', 'relation', name='uq_edges_src_dst_relation')
    )
    op.create_index('ix_edges_dst', 'edges', ['dst_id'], unique=False)
    op.create_index('ix_edges_src', 'edges', ['src_id'], unique=False)
    op.create_table('heuristic_rules',
    sa.Column('node_id', sa.UUID(), nullable=False),
    sa.Column('task_type', sa.String(length=200), nullable=True),
    sa.Column('condition', sa.Text(), nullable=False),
    sa.Column('avoid', sa.Text(), nullable=False),
    sa.Column('action', sa.Text(), nullable=False),
    sa.Column('rationale', sa.Text(), nullable=False),
    sa.Column('success_after', sa.Integer(), nullable=False),
    sa.Column('failure_after', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['node_id'], ['nodes.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('node_id')
    )
    op.create_table('provenance',
    sa.Column('node_id', sa.UUID(), nullable=False),
    sa.Column('episode_id', sa.UUID(), nullable=False),
    sa.Column('role', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("role IN ('supports', 'contradicts')", name='ck_role_valid'),
    sa.ForeignKeyConstraint(['episode_id'], ['episodes.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['node_id'], ['nodes.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('node_id', 'episode_id', 'role')
    )
    op.create_index('ix_provenance_episode', 'provenance', ['episode_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_provenance_episode', table_name='provenance')
    op.drop_table('provenance')
    op.drop_table('heuristic_rules')
    op.drop_index('ix_edges_src', table_name='edges')
    op.drop_index('ix_edges_dst', table_name='edges')
    op.drop_table('edges')
    op.drop_index('ix_contradictions_agent_status', table_name='contradictions')
    op.drop_table('contradictions')
    op.drop_index('ix_nodes_embedding_hnsw', table_name='nodes', postgresql_using='hnsw', postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.drop_index('ix_nodes_agent_kind_status', table_name='nodes')
    op.drop_index('ix_nodes_agent_key', table_name='nodes')
    op.drop_table('nodes')
    op.drop_index('ix_memory_events_agent_created', table_name='memory_events')
    op.drop_table('memory_events')
    op.drop_index('ix_episodes_embedding_hnsw', table_name='episodes', postgresql_using='hnsw', postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.drop_index('ix_episodes_content_gin', table_name='episodes', postgresql_using='gin')
    op.drop_index('ix_episodes_agent_unconsolidated', table_name='episodes', postgresql_where=sa.text('consolidated_at IS NULL'))
    op.drop_index('ix_episodes_agent_outcome', table_name='episodes')
    op.drop_table('episodes')
    op.drop_index('uq_jobs_one_queued', table_name='consolidation_jobs', postgresql_where=sa.text("status = 'queued'"))
    op.drop_index('ix_jobs_claim', table_name='consolidation_jobs')
    op.drop_table('consolidation_jobs')
    op.drop_index('ix_cycles_agent_started', table_name='consolidation_cycles')
    op.drop_table('consolidation_cycles')
    op.drop_table('agents')
