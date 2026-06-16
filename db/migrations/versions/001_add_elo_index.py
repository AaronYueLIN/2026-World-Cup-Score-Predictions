"""Add elo_ratings composite index

Revision ID: 001
Revises: None
Create Date: 2026-06-16
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "idx_elo_snapshot_team",
        "elo_ratings",
        ["snapshot_date", "team_name"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("idx_elo_snapshot_team", table_name="elo_ratings", if_exists=True)
