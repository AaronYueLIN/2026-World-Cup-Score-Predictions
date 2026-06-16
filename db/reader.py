"""Data reader -- query interface for prediction models

Returns pandas DataFrames, compatible with existing Bayesian DC / GBM code.

Environment separation:
  APP_ENV=development -> only last 5 years of data (fast iteration)
  APP_ENV=production  -> full historical data (full training)
"""
from __future__ import annotations

import time
from datetime import date as date_type

import pandas as pd
from prometheus_client import Histogram
from sqlalchemy import create_engine, text

from db.config import APP_ENV, DATABASE_URL, ENGINE_KWARGS
from db.settings import settings

# Prometheus: DB query duration
DB_QUERY_DURATION = Histogram(
    "db_query_duration_seconds",
    "Database query duration",
    labelnames=["query"],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)


def _query_label(sql: str) -> str:
    """Generate Prometheus label from SQL summary (first 60 chars)."""
    label = sql.strip().split("\n")[0].strip().rstrip(",")[:60]
    return label.replace('"', "'")


def _timed_query(sql: str, conn, params: dict | None = None):
    """Execute SQL and record duration to Prometheus."""
    label = _query_label(sql)
    t0 = time.time()
    result = conn.execute(text(sql), params or {})
    DB_QUERY_DURATION.labels(query=label).observe(time.time() - t0)
    return result


def _read_sql_timed(sql: str, conn, params: dict | None = None) -> "pd.DataFrame":
    """pandas.read_sql with timing."""
    label = _query_label(sql)
    t0 = time.time()
    df = pd.read_sql(text(sql), conn, params=params or {})
    DB_QUERY_DURATION.labels(query=label).observe(time.time() - t0)
    return df

# Global engine lazy initialization
_engine: object | None = None
_engine_ro: object | None = None


def _get_engine(read_only: bool = False):
    """Lazily create engine (supports read/write splitting).

    When read_only=True, production environment prefers the read replica (DATABASE_URL_READ_ONLY),
    development environment falls back to the primary.
    """
    global _engine, _engine_ro
    if read_only and settings.is_production:
        if _engine_ro is None:
            _engine_ro = create_engine(
                settings.database_url_read,
                **settings.engine_kwargs
            )
        return _engine_ro
    if _engine is None:
        _engine = create_engine(DATABASE_URL, **ENGINE_KWARGS)
    return _engine


# Development mode defaults to only last 5 years
_TRAINING_YEARS = 5 if APP_ENV == "development" else None

# ---------------------------------------------------------------------------
# tournament name -> competition key (hooks into Bayesian DC's COMPETITION_WEIGHTS)
# dict mapping is 100x faster than apply()
# ---------------------------------------------------------------------------
_COMPETITION_MAP: dict[str, str] = {}


def _build_competition_map() -> None:
    """Read tier from the database tournaments table, build tournament_name -> competition_key mapping."""
    if _COMPETITION_MAP:
        return
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT name, tier FROM tournaments")).fetchall()
    for name, tier in rows:
        name_lower = name.lower()
        if tier == "world_cup":
            _COMPETITION_MAP[name] = "world_cup"
        elif "world cup" in name_lower and "qualif" in name_lower:
            _COMPETITION_MAP[name] = "world_cup_qualifying"
        elif tier == "continental":
            _COMPETITION_MAP[name] = "nations_league" if "nation" in name_lower else "continental_championship"
        elif tier == "qualifier":
            _COMPETITION_MAP[name] = "continental_qualifying"
        else:
            _COMPETITION_MAP[name] = "friendly"
    _COMPETITION_MAP.setdefault("Friendly", "friendly")


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def get_team_history(
    team: str,
    before_date: str | date_type | None = None,
) -> pd.DataFrame:
    """Get all historical matches for a team (optionally filtered by date), returns DataFrame"""
    sql = """
        SELECT m.date, ht.name AS home_team, at.name AS away_team,
               m.home_score AS home_goals, m.away_score AS away_goals,
               COALESCE(t.name, 'Friendly') AS tournament,
               m.neutral, m.venue
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        LEFT JOIN tournaments t ON m.tournament_id = t.id
        WHERE ht.name = :team OR at.name = :team
    """
    params = {"team": team}
    if before_date:
        sql += " AND m.date < :before_date"
        params["before_date"] = before_date
    sql += " ORDER BY m.date"

    engine = _get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    _build_competition_map()
    df["competition"] = df["tournament"].map(_COMPETITION_MAP).fillna("friendly")
    return df


def _derive_result(row, team: str) -> str:
    if row["home_goals"] > row["away_goals"]:
        return "H" if row["home_team"] == team else "A"
    elif row["home_goals"] < row["away_goals"]:
        return "A" if row["home_team"] == team else "H"
    return "D"


def get_all_matches(
    from_date: str | date_type | None = None,
    to_date: str | date_type | None = None,
) -> pd.DataFrame:
    """Get all match data, returns DataFrame"""
    sql = """
        SELECT m.date, ht.name AS home_team, at.name AS away_team,
               m.home_score AS home_goals, m.away_score AS away_goals,
               COALESCE(t.name, 'Friendly') AS tournament,
               m.neutral, m.venue
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        LEFT JOIN tournaments t ON m.tournament_id = t.id
    """
    params = {}
    conditions = []
    if from_date:
        conditions.append("m.date >= :from_date")
        params["from_date"] = from_date
    if to_date:
        conditions.append("m.date <= :to_date")
        params["to_date"] = to_date
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY m.date"

    engine = _get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    _build_competition_map()
    df["competition"] = df["tournament"].map(_COMPETITION_MAP).fillna("friendly")
    return df


# ---------------------------------------------------------------------------
# Training data -- one-shot fetch of all columns needed by Bayesian DC
# ---------------------------------------------------------------------------

def get_training_data(
    from_date: str | date_type | None = None,
    include_archive: bool = False,
) -> pd.DataFrame:
    """Get training data, columns fully matching ``BayesianDixonColesModel.fit()`` requirements.

    Automatic environment-based trimming:
      APP_ENV=development -> only last 5 years (fast iteration)
      APP_ENV=production  -> last 20 years (all meaningful matches)
      include_archive=True -> full history (since 1872, ~49,000 matches)

    Args:
        from_date: start date (overrides automatic trimming)
        include_archive: whether to include data older than 20 years

    Returns:
        DataFrame with columns:
        home_team, away_team, home_goals, away_goals, date, venue, competition
    """
    if from_date is None and _TRAINING_YEARS:
        from_date = f"{date_type.today().year - _TRAINING_YEARS}-01-01"
    if from_date is None and not include_archive:
        # Production default: 20 years
        from_date = f"{date_type.today().year - 20}-01-01"
    df = get_all_matches(from_date=from_date)
    return df[["home_team", "away_team", "home_goals", "away_goals",
               "date", "venue", "competition"]]


def get_match_count() -> int:
    """Return the total number of matches in the database"""
    engine = _get_engine()
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM matches")).scalar()


def get_team_list() -> list[str]:
    """Return a list of all team names"""
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT name FROM teams ORDER BY name")).fetchall()
        return [r[0] for r in rows]
