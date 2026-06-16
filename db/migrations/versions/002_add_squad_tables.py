"""Add squad_players and squad_coaches tables"""
from __future__ import annotations

import importlib
import sys

from alembic import op
import sqlalchemy as sa


revision = "002_add_squad_tables"
down_revision = "001_add_elo_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # squad_players
    op.create_table(
        "squad_players",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("team_code", sa.String(3), nullable=False, index=True),
        sa.Column("team_name", sa.String(100), nullable=False),
        sa.Column("jersey_number", sa.Integer(), nullable=False),
        sa.Column("position", sa.String(3), nullable=False),
        sa.Column("player_name", sa.String(200), nullable=False),
        sa.Column("first_name", sa.String(200), nullable=False),
        sa.Column("last_name", sa.String(200), nullable=False),
        sa.Column("name_on_shirt", sa.String(100), nullable=False),
        sa.Column("dob", sa.String(20), nullable=True),
        sa.Column("club", sa.String(200), nullable=False),
        sa.Column("club_country", sa.String(3), nullable=True),
        sa.Column("height_cm", sa.Integer(), nullable=True),
        sa.Column("caps", sa.Integer(), nullable=True),
        sa.Column("goals", sa.Integer(), nullable=True),
        sa.Column("top5_league", sa.Boolean(), default=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("team_code", "jersey_number", name="uq_squad_player_number"),
    )

    # squad_coaches
    op.create_table(
        "squad_coaches",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("team_code", sa.String(3), nullable=False, index=True),
        sa.Column("team_name", sa.String(100), nullable=False),
        sa.Column("role", sa.String(50), nullable=False),
        sa.Column("coach_name", sa.String(200), nullable=False),
        sa.Column("first_name", sa.String(200), nullable=False),
        sa.Column("last_name", sa.String(200), nullable=False),
        sa.Column("nationality", sa.String(100), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Index for team lookup
    op.create_index("ix_squad_players_team", "squad_players", ["team_code"])
    op.create_index("ix_squad_coaches_team", "squad_coaches", ["team_code"])


def downgrade() -> None:
    op.drop_table("squad_players")
    op.drop_table("squad_coaches")
