"""ELO rating queries -- SQL interface, called by model fit()"""
from __future__ import annotations

from datetime import date as date_type

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.reader import _get_engine


def _get_latest_date() -> str:
    """Return the latest snapshot_date from elo_ratings."""
    engine = _get_engine()
    with engine.connect() as c:
        row = c.execute(
            text("SELECT MAX(snapshot_date) FROM elo_ratings")
        ).scalar()
    return row if row else "1970-01-01"


def get_latest_elo() -> dict[str, float]:
    """Get the ELO rating dict for the latest snapshot {team: elo}."""
    latest = _get_latest_date()
    engine = _get_engine()
    with engine.connect() as c:
        rows = c.execute(
            text("SELECT team_name, rating FROM elo_ratings WHERE snapshot_date = :d"),
            {"d": latest},
        ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def get_latest_momentum() -> tuple[dict[str, int], dict[str, float]]:
    """Get the latest 1-year momentum data."""
    latest = _get_latest_date()
    engine = _get_engine()
    with engine.connect() as c:
        rows = c.execute(
            text("SELECT team_name, rank_chg_1y, rating_chg_1y "
                 "FROM elo_ratings WHERE snapshot_date = :d"),
            {"d": latest},
        ).fetchall()
    rank_chg = {r[0]: int(r[1]) for r in rows}
    rating_chg = {r[0]: float(r[2]) for r in rows}
    return rank_chg, rating_chg


def get_elo_dataframe(
    snapshot_date: str | date_type | None = None,
) -> pd.DataFrame:
    """Return elo_ratings DataFrame."""
    latest = _get_latest_date() if snapshot_date is None else str(snapshot_date)
    engine = _get_engine()
    with engine.connect() as c:
        df = pd.read_sql(
            text("SELECT * FROM elo_ratings WHERE snapshot_date = :d"),
            c,
            params={"d": latest},
        )
    return df


# ============================================================
#  Model interface: return standardized ELO vector in teams order
# ============================================================

def get_standardized_elo(teams: list[str]) -> np.ndarray:
    """Return standardized ELO vector (z-score), order matches teams.

    Teams missing ELO use the mean of existing values.
    """
    elo_dict = get_latest_elo()
    r = np.array([elo_dict.get(t, np.nan) for t in teams], dtype=np.float64)
    missing = np.isnan(r)
    if missing.any():
        r[missing] = np.nanmean(r) if not np.isnan(np.nanmean(r)) else 1500.0
    return (r - r.mean()) / (r.std() if r.std() > 1e-12 else 1.0)


def get_standardized_momentum(teams: list[str]) -> np.ndarray:
    """Return standardized 1-year ELO momentum vector (z-score), order matches teams."""
    _, rating_chg = get_latest_momentum()
    r = np.array([rating_chg.get(t, np.nan) for t in teams], dtype=np.float64)
    missing = np.isnan(r)
    if missing.any():
        r[missing] = 0.0
    std = r.std() if r.std() > 1e-12 else 1.0
    return (r - r.mean()) / std


# ============================================================
#  SQL query examples
# ============================================================

SQL_SAMPLE = """
-- Latest snapshot, descending by ELO
SELECT team_name, rating, rank, rank_chg_1y, rating_chg_1y
FROM elo_ratings
WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM elo_ratings)
ORDER BY rating DESC
LIMIT 10;

-- Check a team's historical trend
SELECT snapshot_date, rating, rank, rating_chg_1y
FROM elo_ratings
WHERE team_name = 'United States'
ORDER BY snapshot_date DESC;

-- Max/min momentum
SELECT team_name, rating, rating_chg_1y, rank_chg_1y
FROM elo_ratings
WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM elo_ratings)
ORDER BY rating_chg_1y DESC
LIMIT 5;
"""
