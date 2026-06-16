"""ETL core functions -- CSV import, API fetch, team name normalization, Upsert

Four phases:
  1. CSV Loader   -- pandas reads raw CSV from archive
  2. API Fetcher   -- httpx calls football-data.org
  3. Normalizer    -- former_names.csv maps historical names to current names
  4. Upsert Engine -- SQLAlchemy ORM upsert (compatible with SQLite + PostgreSQL)
"""
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from db.config import IS_SQLITE
from db.models import Base, Match, Odds, Team, TeamName, Tournament
from db.reader import _get_engine

logger = logging.getLogger(__name__)

engine = _get_engine()


# ---------------------------------------------------------------------------
# 0. Database initialization
# ---------------------------------------------------------------------------
def init_db():
    """Create all tables (idempotent, skips existing tables)"""
    Base.metadata.create_all(engine)
    logger.info("Database table initialization complete")


# ---------------------------------------------------------------------------
# 1. CSV Loader
# ---------------------------------------------------------------------------
def load_csv(filepath: str | Path) -> pd.DataFrame:
    """Read raw Kaggle results.csv and do basic cleaning.

    Returns:
        Cleaned DataFrame with columns: date, home_team, away_team,
        home_score, away_score, tournament, city, country, neutral
    """
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["date"] = pd.to_datetime(df["date"])
    # normalize venue column
    if "neutral" in df.columns:
        df["venue"] = np.where(df["neutral"].astype(bool), "neutral", "home")
    else:
        df["venue"] = "neutral"

    logger.info("CSV loaded: %d rows, %s ~ %s", len(df), df["date"].min().date(), df["date"].max().date())
    return df


# ---------------------------------------------------------------------------
# 2. Team name normalization
# ---------------------------------------------------------------------------
_NAME_MAP: dict[str, dict[str, str]] | None = None


def _load_name_map() -> dict[str, dict[str, str]]:
    """Load former_names.csv -> {former: {current, start, end}}"""
    global _NAME_MAP
    if _NAME_MAP is not None:
        return _NAME_MAP

    csv_path = Path(__file__).parent / "former_names.csv"
    if not csv_path.exists():
        logger.warning("former_names.csv not found, skipping team name normalization")
        _NAME_MAP = {}
        return _NAME_MAP

    df = pd.read_csv(csv_path)
    _NAME_MAP = {}
    for _, row in df.iterrows():
        _NAME_MAP[row["former"]] = {
            "current": row["current"],
            "start": row["start_date"],
            "end": row["end_date"],
        }
    logger.info("Team name mapping loaded: %d entries", len(_NAME_MAP))
    return _NAME_MAP


def normalize_team(name: str, match_date: date | None = None) -> str:
    """Map historical team name to current team name.

    Args:
        name:       original team name
        match_date: match date (to check if mapping is within the time window)

    Returns:
        Normalized team name
    """
    name_map = _load_name_map()
    if name in name_map:
        entry = name_map[name]
        if match_date:
            start = datetime.strptime(entry["start"], "%Y-%m-%d").date()
            end = datetime.strptime(entry["end"], "%Y-%m-%d").date()
            if start <= match_date <= end:
                return entry["current"]
        # No date info: map directly
        return entry["current"]
    return name


# ---------------------------------------------------------------------------
# 3. Upsert engine
# ---------------------------------------------------------------------------
def _get_or_create_team(session: Session, name: str) -> int:
    """Look up or insert a team, return team_id"""
    stmt = select(Team.id).where(Team.name == name)
    team_id = session.execute(stmt).scalar()
    if team_id is None:
        team = Team(name=name)
        session.add(team)
        session.flush()
        team_id = team.id
    return team_id


def _get_or_create_tournament(session: Session, name: str) -> int | None:
    """Look up or insert a tournament, return tournament_id. Returns None for empty name."""
    if pd.isna(name) or not name:
        return None

    # Infer tier
    name_lower = str(name).lower()
    if "world cup" in name_lower and "qualif" in name_lower:
        tier = "qualifier"
    elif "world cup" in name_lower:
        tier = "world_cup"
    elif "euro" in name_lower and "qualif" in name_lower:
        tier = "qualifier"
    elif "euro" in name_lower:
        tier = "continental"
    elif "copa" in name_lower or "africa" in name_lower or "asian" in name_lower:
        tier = "continental"
    elif "friendly" in name_lower:
        tier = "friendly"
    elif "qualif" in name_lower:
        tier = "qualifier"
    elif "nation" in name_lower:
        tier = "continental"
    else:
        tier = "other"

    stmt = select(Tournament.id).where(Tournament.name == name)
    tourney_id = session.execute(stmt).scalar()
    if tourney_id is None:
        t = Tournament(name=name, tier=tier)
        session.add(t)
        session.flush()
        tourney_id = t.id
    return tourney_id


def upsert_matches(df: pd.DataFrame, batch_size: int = 1000):
    """Batch upsert match data (ORM approach, compatible with SQLite + PostgreSQL).

    Args:
        df:  DataFrame with home_team, away_team, home_score, away_score, date,
             tournament, city, country, neutral, venue
        batch_size: rows per commit batch
    """
    name_map = _load_name_map()
    df = df.copy()

    # Team name normalization
    df["home_team"] = df["home_team"].apply(lambda x: name_map.get(x, {}).get("current", x))
    df["away_team"] = df["away_team"].apply(lambda x: name_map.get(x, {}).get("current", x))

    total = len(df)

    with Session(engine) as session:
        team_cache: dict[str, int] = {}
        tourney_cache: dict[str, int | None] = {}

        for start in range(0, total, batch_size):
            batch = df.iloc[start : start + batch_size]
            batch_rows = []

            for _, row in batch.iterrows():
                ht = row["home_team"]
                at = row["away_team"]
                tn = row.get("tournament", "")

                if ht not in team_cache:
                    team_cache[ht] = _get_or_create_team(session, ht)
                if at not in team_cache:
                    team_cache[at] = _get_or_create_team(session, at)
                if tn not in tourney_cache:
                    tourney_cache[tn] = _get_or_create_tournament(session, tn)

                match_date = row["date"]
                if isinstance(match_date, datetime):
                    match_date = match_date.date()

                venue = row.get("venue", "neutral")
                if venue not in ("home", "away", "neutral"):
                    venue = "neutral"

                batch_rows.append({
                    "home_team_id": team_cache[ht],
                    "away_team_id": team_cache[at],
                    "tournament_id": tourney_cache[tn],
                    "home_score": int(row["home_score"]),
                    "away_score": int(row["away_score"]),
                    "date": match_date,
                    "city": row.get("city") if not pd.isna(row.get("city")) else None,
                    "country": row.get("country") if not pd.isna(row.get("country")) else None,
                    "neutral": bool(row.get("neutral", False)),
                    "venue": venue,
                })

            # Upsert: SQLite and PostgreSQL take different paths
            if IS_SQLITE:
                stmt = sqlite_insert(Match).values(batch_rows)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["home_team_id", "away_team_id", "date"],
                    set_={
                        "home_score": stmt.excluded.home_score,
                        "away_score": stmt.excluded.away_score,
                    },
                )
                session.execute(stmt)
            else:
                stmt = pg_insert(Match).values(batch_rows)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["home_team_id", "away_team_id", "date"],
                    set_={
                        "home_score": stmt.excluded.home_score,
                        "away_score": stmt.excluded.away_score,
                        "updated_at": datetime.utcnow(),
                    },
                )
                session.execute(stmt)

            session.commit()
            logger.info("  Batch write: %d/%d", min(start + batch_size, total), total)

    logger.info("Upsert complete: total %d rows", total)


# ---------------------------------------------------------------------------
# 4. API Fetcher (placeholder, football-data.org)
# ---------------------------------------------------------------------------
def fetch_api(api_token: str | None = None, days_back: int = 7) -> list[dict[str, Any]]:
    """Fetch recent match results from football-data.org.

    Note: Requires a football-data.org free API key (X-Auth-Token header).
    Free tier is rate-limited to 10 requests/minute.
    """
    import os

    import httpx

    token = api_token or os.getenv("FOOTBALL_DATA_API_KEY")
    if not token:
        logger.warning("FOOTBALL_DATA_API_KEY not set, skipping API fetch")
        return []

    from_date = (datetime.now().date() - pd.Timedelta(days=days_back)).isoformat()
    to_date = datetime.now().date().isoformat()

    url = "https://api.football-data.org/v4/matches"
    params = {"dateFrom": from_date, "dateTo": to_date, "status": "FINISHED"}

    try:
        resp = httpx.get(url, headers={"X-Auth-Token": token}, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("matches", [])
        logger.info("API fetched: %d matches (%s ~ %s)", len(matches), from_date, to_date)
        return [
            {
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_score": m["score"]["fullTime"]["home"],
                "away_score": m["score"]["fullTime"]["away"],
                "date": m["utcDate"][:10],
                "tournament": m["competition"]["name"],
                "venue": "neutral",
            }
            for m in matches
            if m["score"]["fullTime"]["home"] is not None
        ]
    except Exception:
        logger.exception("API fetch failed")
        return []
