"""SQLAlchemy ORM models -- International football match data

Tables:
  teams         -- Teams (current name + FIFA code)
  tournaments   -- Tournaments (name + tier)
  matches       -- Matches (score, date, venue)
  odds          -- Odds (multiple bookmakers)
  team_names    -- Historical team name mappings (Soviet Union -> Russia, etc.)
"""
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------
class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    fifa_code: Mapped[str | None] = mapped_column(String(5))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp()
    )

    # relationships
    home_matches: Mapped[list["Match"]] = relationship(
        back_populates="home_team_rel", foreign_keys="Match.home_team_id"
    )
    away_matches: Mapped[list["Match"]] = relationship(
        back_populates="away_team_rel", foreign_keys="Match.away_team_id"
    )

    def __repr__(self):
        return f"<Team {self.name}>"


# ---------------------------------------------------------------------------
# Tournaments
# ---------------------------------------------------------------------------
class Tournament(Base):
    __tablename__ = "tournaments"
    __table_args__ = (
        CheckConstraint(
            "tier IN ('friendly', 'qualifier', 'continental', 'world_cup', 'other')",
            name="ck_tournament_tier",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    tier: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp()
    )

    matches: Mapped[list["Match"]] = relationship(back_populates="tournament_rel")

    def __repr__(self):
        return f"<Tournament {self.name} ({self.tier})>"


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------
class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("home_team_id", "away_team_id", "date", name="uq_match_teams_date"),
        CheckConstraint("home_score >= 0", name="ck_home_score_nonneg"),
        CheckConstraint("away_score >= 0", name="ck_away_score_nonneg"),
        CheckConstraint(
            "venue IN ('home', 'away', 'neutral')",
            name="ck_match_venue",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    home_team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id"), nullable=False, index=True
    )
    away_team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id"), nullable=False, index=True
    )
    tournament_id: Mapped[int | None] = mapped_column(
        ForeignKey("tournaments.id"), index=True
    )

    home_score: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    away_score: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    date: Mapped[datetime] = mapped_column(Date, nullable=False, index=True)

    city: Mapped[str | None] = mapped_column(String(100))
    country: Mapped[str | None] = mapped_column(String(100))
    neutral: Mapped[bool] = mapped_column(Boolean, default=False)
    venue: Mapped[str] = mapped_column(
        String(50), default="neutral"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    # relationships
    home_team_rel: Mapped["Team"] = relationship(
        back_populates="home_matches", foreign_keys=[home_team_id]
    )
    away_team_rel: Mapped["Team"] = relationship(
        back_populates="away_matches", foreign_keys=[away_team_id]
    )
    tournament_rel: Mapped["Tournament | None"] = relationship(
        back_populates="matches"
    )
    odds: Mapped[list["Odds"]] = relationship(back_populates="match_rel")

    def __repr__(self):
        return (
            f"<Match {self.date} {self.home_team_rel.name if self.home_team_rel else '?'}"
            f" {self.home_score}-{self.away_score}"
            f" {self.away_team_rel.name if self.away_team_rel else '?'}>"
        )


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------
class Odds(Base):
    __tablename__ = "odds"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    home_odds: Mapped[float | None] = mapped_column()
    draw_odds: Mapped[float | None] = mapped_column()
    away_odds: Mapped[float | None] = mapped_column()
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp()
    )

    match_rel: Mapped["Match"] = relationship(back_populates="odds")

    def __repr__(self):
        return (
            f"<Odds {self.provider} "
            f"H={self.home_odds} D={self.draw_odds} A={self.away_odds}>"
        )


# ---------------------------------------------------------------------------
# Historical team name mappings
# ---------------------------------------------------------------------------
class TeamName(Base):
    __tablename__ = "team_names"
    __table_args__ = (
        UniqueConstraint("former_name", "start_date", name="uq_team_name_period"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    current_name: Mapped[str] = mapped_column(String(100), nullable=False)
    former_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    start_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    end_date: Mapped[datetime] = mapped_column(Date, nullable=False)

    def __repr__(self):
        return f"<TeamName {self.former_name} → {self.current_name}>"
