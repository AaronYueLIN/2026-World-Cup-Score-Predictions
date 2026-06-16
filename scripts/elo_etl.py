#!/usr/bin/env python3
"""ELO rating ETL -- scrape eloratings.net -> write to database

Schema:
  elo_ratings
    team_name      TEXT NOT NULL      -- Team name (compatible with matches.teams.name)
    rating         REAL NOT NULL      -- Current ELO rating
    rank           INTEGER NOT NULL   -- Current rank
    rank_chg_1y    INTEGER DEFAULT 0  -- Rank change over 1 year
    rating_chg_1y  REAL DEFAULT 0.0   -- ELO change over 1 year
    snapshot_date  TEXT NOT NULL      -- Data date ISO (YYYY-MM-DD)

  Unique constraint (team_name, snapshot_date), daily upsert.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from datetime import date

import requests
from sqlalchemy import create_engine, text

from db.config import DATABASE_URL, ENGINE_KWARGS

logger = logging.getLogger(__name__)

# Manual team name mapping (eloratings 2-letter code -> project team name)
CODE_MAP = {
    "US": "United States", "PY": "Paraguay", "DE": "Germany",
    "ES": "Spain", "FR": "France", "EN": "England", "PT": "Portugal",
    "NL": "Netherlands", "BR": "Brazil", "AR": "Argentina",
    "UY": "Uruguay", "CO": "Colombia", "CL": "Chile",
    "MX": "Mexico", "CA": "Canada", "JP": "Japan", "KR": "South Korea",
    "AU": "Australia", "TN": "Tunisia", "MA": "Morocco", "DZ": "Algeria",
    "NG": "Nigeria", "SN": "Senegal", "GH": "Ghana", "EG": "Egypt",
    "ZA": "South Africa", "RU": "Russia", "UA": "Ukraine",
    "TR": "Turkey", "HR": "Croatia", "RS": "Serbia", "CH": "Switzerland",
    "BE": "Belgium", "DK": "Denmark", "SE": "Sweden", "NO": "Norway",
    "AT": "Austria", "PL": "Poland", "CZ": "Czech Republic",
    "HU": "Hungary", "RO": "Romania", "GR": "Greece",
    "IE": "Republic of Ireland", "SC": "Scotland", "WA": "Wales",
    "IT": "Italy", "CW": "Curaçao",
}

ELO_URL = "https://eloratings.net/World.tsv"
TEAMS_URL = "https://eloratings.net/en.teams.tsv"


def _fetch_code_map() -> dict[str, str]:
    """Download team name mapping from en.teams.tsv, supplement with manual corrections."""
    r = requests.get(TEAMS_URL, timeout=10)
    m = {}
    for line in r.content.decode("utf-8").strip().split("\n"):
        p = line.split("\t")
        if len(p) >= 2:
            m[p[0].strip()] = p[1].strip()
    m.update(CODE_MAP)
    return m


def _to_float(s: str) -> float:
    s = s.strip().replace("−", "-")
    return float(s) if s and s != "-" else 0.0


def _to_int(s: str) -> int:
    return int(_to_float(s))


def fetch_and_upsert(engine=None, today: str | None = None) -> int:
    """Download latest ELO data and upsert into the database.

    Args:
        engine: SQLAlchemy engine, None=auto create.
        today:  ISO date, None=today.

    Returns:
        Number of upserted rows.
    """
    if engine is None:
        engine = create_engine(DATABASE_URL, **ENGINE_KWARGS)
    today = today or date.today().isoformat()

    # 1. Download
    r = requests.get(ELO_URL, timeout=10)
    raw = r.content.decode("utf-8").replace("−", "-")

    code_map = _fetch_code_map()

    # 2. Parse
    rows: list[dict] = []
    for line in raw.strip().split("\n"):
        cols = line.split("\t")
        if len(cols) < 22:
            continue
        team_name = code_map.get(cols[2].strip(), cols[2].strip())
        rows.append({
            "team_name": team_name,
            "rating": _to_float(cols[3]),
            "rank": _to_int(cols[0]),
            "rank_chg_1y": _to_int(cols[14]),
            "rating_chg_1y": _to_float(cols[15]),
            "snapshot_date": today,
        })

    # 3. Upsert (pure SQL, compatible with SQLite + PostgreSQL)
    with engine.begin() as conn:
        for row in rows:
            conn.execute(
                text("""
                    INSERT INTO elo_ratings (team_name, rating, rank, rank_chg_1y, rating_chg_1y, snapshot_date)
                    VALUES (:t, :r, :rk, :chg_r, :chg_e, :d)
                    ON CONFLICT (team_name, snapshot_date)
                    DO UPDATE SET
                        rating = EXCLUDED.rating,
                        rank = EXCLUDED.rank,
                        rank_chg_1y = EXCLUDED.rank_chg_1y,
                        rating_chg_1y = EXCLUDED.rating_chg_1y
                """),
                {
                    "t": row["team_name"],
                    "r": row["rating"],
                    "rk": row["rank"],
                    "chg_r": row["rank_chg_1y"],
                    "chg_e": row["rating_chg_1y"],
                    "d": row["snapshot_date"],
                },
            )

    logger.info("ELO upsert: %d teams @ %s", len(rows), today)
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    fetch_and_upsert()
