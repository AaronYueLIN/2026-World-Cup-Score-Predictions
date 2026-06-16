"""FastAPI routes — v1 endpoints (Enterprise)

Existing data query routes + prediction routes, all mounted under /api/v1 prefix.
"""
from __future__ import annotations

import time
from datetime import date as date_type
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from prometheus_client import Counter, Histogram
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatData,
    ExplainResponse,
    HealthResponse,
    LayerInfo,
    MatchOut,
    MomentumEntry,
    MomentumResponse,
    PredictionRequest,
    PredictionResponse,
    TeamHistoryOut,
    TeamOut,
    TournamentOut,
)
from db.config import IS_SQLITE
from models.exceptions import PredictionError
from models.registry import load_model

_log = structlog.get_logger(__name__)

# Prometheus metrics
PREDICT_DURATION = Histogram(
    "predict_duration_seconds", "Time per predict call",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)
PREDICT_TOTAL = Counter("predictions_total", "Total predict calls", ["outcome"])

router = APIRouter()

# SQLite uses LIKE, PostgreSQL uses ILIKE
_LIKE_OP = "LIKE" if IS_SQLITE else "ILIKE"

# Model lazy loading (registry has built-in lru_cache)
_model_loaded = False


def _get_model():
    global _model_loaded
    try:
        dc = load_model()
        _model_loaded = True
        return dc
    except Exception:
        _model_loaded = False
        raise


# ============================================================================
#  Prediction
# ============================================================================

@router.post("/predict", response_model=PredictionResponse)
def predict(body: PredictionRequest):
    """Full-match 1X2 prediction + derived markets (NB+Frank + GAS)."""
    dc = _get_model()
    home, away = body.home_team, body.away_team
    venue = body.venue
    dt = body.date

    if home not in dc.team_idx:
        raise PredictionError(f"Unknown home team: {home!r}")
    if away not in dc.team_idx:
        raise PredictionError(f"Unknown away team: {away!r}")

    t0 = time.time()
    r = dc.predict(home, away, venue=venue, date=dt)
    duration = time.time() - t0

    PREDICT_DURATION.observe(duration)
    outcome = "home" if r["home_win_prob"] > r["away_win_prob"] else "away"
    PREDICT_TOTAL.labels(outcome=outcome).inc()

    gas_b = float(dc._gas_hyper.get("B", 0)) if getattr(dc, "_gas", None) else 0.0

    _log.info("predict", home=home, away=away, venue=venue,
              prob_home=round(r["home_win_prob"], 4),
              prob_draw=round(r["draw_prob"], 4),
              prob_away=round(r["away_win_prob"], 4),
              duration_ms=round(duration * 1000, 1))

    return PredictionResponse(
        home_team=home,
        away_team=away,
        venue=venue,
        home_win_prob=round(r["home_win_prob"], 4),
        draw_prob=round(r["draw_prob"], 4),
        away_win_prob=round(r["away_win_prob"], 4),
        expected_home_goals=round(r["expected_home_goals"], 3),
        expected_away_goals=round(r["expected_away_goals"], 3),
        over_25=round(r["over_25"], 4),
        btts=round(r["btts"], 4),
        pool_method=r.get("pool_method"),
    )


# ============================================================================
#  Chat / LLM Agent Gateway
# ============================================================================

# ============================================================================
#  Chat / LLM Agent Gateway
# ============================================================================

def _get_llm_client():
    """Lazy initialization of LLM client (DeepSeek / OpenAI compatible)."""
    from db.settings import settings
    if not settings.llm_api_key:
        return None
    from openai import OpenAI
    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


# Function tools for LLM
_LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict",
            "description": "Predict match outcome (1X2 + expected goals + over/under + BTTS)",
            "parameters": {
                "type": "object",
                "properties": {
                    "home_team": {"type": "string", "description": "Home team name"},
                    "away_team": {"type": "string", "description": "Away team name"},
                    "venue": {"type": "string", "enum": ["home", "neutral", "away"]},
                },
                "required": ["home_team", "away_team"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_momentum",
            "description": "Get GAS momentum ranking (teams deviating most from Elo anchor)",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    """Natural language -> LLM/rule engine -> prediction/query -> structured reply.

    Set LLM_API_KEY env var to use DeepSeek LLM for intent parsing.
    Falls back to regex-based rule engine when LLM is unavailable.
    """
    import json
    dc = _get_model()
    client = _get_llm_client()

    # ---- LLM path (DeepSeek with Function Calling) ----
    if client is not None:
        system_prompt = (
            "You are a football prediction assistant. Parse the user's intent and call the appropriate tool.\n\n"
            "Available tools:\n"
            "1. predict(home_team, away_team, venue) - predict match\n"
            "2. get_momentum() - GAS momentum ranking\n\n"
            'For "Spain vs Cape Verde", "who wins", "predict" -> call predict\n'
            'For "momentum", "form", "hot teams" -> call get_momentum\n'
            "Reply in Chinese briefly."
        )
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": body.message},
                ],
                tools=_LLM_TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=500,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                tc = msg.tool_calls[0]
                args = json.loads(tc.function.arguments)
                if tc.function.name == "predict":
                    return _do_predict(args["home_team"], args["away_team"], args.get("venue", "neutral"))
                elif tc.function.name == "get_momentum":
                    return _do_momentum()
            return ChatResponse(reply=msg.content or "Sorry, I didn't understand.")

        except Exception as e:
            _log.warning("llm_call_failed", error=str(e))
            # fall through to rule engine

    # ---- Rule engine path (LLM unavailable) ----
    return _rule_chat(body.message)


def _do_predict(home_raw: str, away_raw: str, venue: str = "neutral") -> ChatResponse:
    dc = _get_model()
    def resolve(raw: str) -> str | None:
        if raw in dc.team_idx:
            return raw
        for name in dc.teams:
            if raw.lower() == name.lower():
                return name
        cands = [n for n in dc.teams if n.lower().startswith(raw.lower())]
        return min(cands, key=len) if cands else None
    home, away = resolve(home_raw) or home_raw, resolve(away_raw) or away_raw
    if home not in dc.team_idx or away not in dc.team_idx:
        return ChatResponse(reply=f"Team not found: {home} or {away}")
    r = dc.predict(home, away, venue=venue)
    h, d, a = r["home_win_prob"], r["draw_prob"], r["away_win_prob"]
    winner = home if h > a else (away if a > h else "Draw")
    return ChatResponse(
        reply=(
            f"**{home} vs {away}**\n\n"
            f"{winner} ({max(h,d,a):.0%})\n\n"
            f"Win {h:.1%} / Draw {d:.1%} / Loss {a:.1%}\n"
            f"Expected {r['expected_home_goals']:.1f}-{r['expected_away_goals']:.1f}\n"
            f"O2.5 {r['over_25']:.0%} / BTTS {r['btts']:.0%}"
        ),
        data=ChatData(
            type="prediction", home_team=home, away_team=away,
            home_win_prob=round(h, 4), draw_prob=round(d, 4),
            away_win_prob=round(a, 4),
            expected_home_goals=round(r["expected_home_goals"], 2),
            expected_away_goals=round(r["expected_away_goals"], 2),
            over_25=round(r["over_25"], 4), btts=round(r["btts"], 4),
        ),
    )


def _do_momentum() -> ChatResponse:
    dc = _get_model()
    table = dc.gas_momentum_table().head(10)
    lines = ["**GAS Momentum Top 10:**\n"]
    for _, row in table.iterrows():
        lines.append(f"  {row['team']}: {row['momentum']:+.3f}")
    return ChatResponse(reply="\n".join(lines))


def _rule_chat(message: str) -> ChatResponse:
    """RegExp-based rule engine (fallback when LLM unavailable)."""
    import re
    msg = message.lower().strip()
    vs_match = re.match(r"(?:predict)?\s*(.+?)\s*(?:vs|[vV][sS]|v)\s*(.+?)\s*$", msg)
    if vs_match:
        return _do_predict(vs_match.group(1).strip(), vs_match.group(2).strip())
    if any(kw in msg for kw in ["momentum", "form", "hot"]):
        return _do_momentum()
    return ChatResponse(
        reply="Hi! I'm QuantBet assistant.\n\n"
              "Try:\n"
              "  `Spain vs Cape Verde` - predict\n"
              "  `momentum` - GAS ranking\n\n"
              "Set LLM_API_KEY for natural language chat."
    )


# ============================================================================
#  Model interpretation
# ============================================================================

@router.get("/model/momentum", response_model=MomentumResponse)
def get_momentum():
    """GAS momentum ranking (teams deviating most from Elo anchor)."""
    dc = _get_model()
    table = dc.gas_momentum_table().head(20)
    teams = []
    for _, row in table.iterrows():
        teams.append(MomentumEntry(
            team=row["team"],
            momentum=round(float(row["momentum"]), 4),
            d_att=round(float(row.get("d_att", 0)), 4),
            d_def=round(float(row.get("d_def", 0)), 4),
        ))
    _log.info("momentum_returned", count=len(teams))
    return MomentumResponse(teams=teams)


# ============================================================================
#  Health check
# ============================================================================

@router.get("/health", response_model=HealthResponse)
def health():
    dc = _get_model()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        gas_active=getattr(dc, "_gas", None) is not None,
        teams=len(dc.teams),
        database="sqlite" if IS_SQLITE else "postgresql",
    )


# ============================================================================
#  Matches
# ============================================================================

@router.get("/matches", response_model=list[MatchOut])
def list_matches(
    team: Optional[str] = Query(None, description="Team name"),
    from_date: Optional[date_type] = Query(None, alias="from"),
    to_date: Optional[date_type] = Query(None, alias="to"),
    tournament: Optional[str] = Query(None, description="Tournament name"),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Query match list, filterable by team, date, tournament"""
    sql = """
        SELECT m.id, m.date,
               ht.name AS home_team, m.home_score,
               at.name AS away_team, m.away_score,
               t.name AS tournament,
               m.city, m.country, m.neutral, m.venue
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        LEFT JOIN tournaments t ON m.tournament_id = t.id
        WHERE 1=1
    """
    params = {}

    if team:
        sql += " AND (ht.name = :team OR at.name = :team)"
        params["team"] = team
    if from_date:
        sql += " AND m.date >= :from_date"
        params["from_date"] = from_date
    if to_date:
        sql += " AND m.date <= :to_date"
        params["to_date"] = to_date
    if tournament:
        sql += f" AND t.name {_LIKE_OP} :tournament"
        params["tournament"] = f"%{tournament}%"

    sql += " ORDER BY m.date DESC LIMIT :limit"
    params["limit"] = limit

    rows = db.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]


# ============================================================================
#  Teams
# ============================================================================

@router.get("/teams/{name}", response_model=TeamOut)
def get_team(name: str, db: Session = Depends(get_db)):
    """Query team info"""
    row = db.execute(
        text("SELECT id, name, fifa_code FROM teams WHERE name = :name"),
        {"name": name},
    ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Team not found")
    return dict(row)


@router.get("/teams/{name}/history", response_model=TeamHistoryOut)
def get_team_history(
    name: str,
    from_date: Optional[date_type] = Query(None, alias="from"),
    to_date: Optional[date_type] = Query(None, alias="to"),
    limit: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    """Query a team's historical matches"""
    sql = """
        SELECT m.id, m.date,
               ht.name AS home_team, m.home_score,
               at.name AS away_team, m.away_score,
               t.name AS tournament
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        LEFT JOIN tournaments t ON m.tournament_id = t.id
        WHERE ht.name = :name OR at.name = :name
    """
    params = {"name": name}
    if from_date:
        sql += " AND m.date >= :from_date"
        params["from_date"] = from_date
    if to_date:
        sql += " AND m.date <= :to_date"
        params["to_date"] = to_date

    sql += " ORDER BY m.date DESC LIMIT :limit"
    params["limit"] = limit

    rows = db.execute(text(sql), params).mappings().all()
    matches = [dict(r) for r in rows]

    wins = draws = losses = 0
    for m in matches:
        if m["home_team"] == name:
            wins += m["home_score"] > m["away_score"]
            draws += m["home_score"] == m["away_score"]
            losses += m["home_score"] < m["away_score"]
        else:
            wins += m["away_score"] > m["home_score"]
            draws += m["away_score"] == m["home_score"]
            losses += m["away_score"] < m["home_score"]

    return {
        "team": name,
        "total_matches": len(matches),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "matches": matches,
    }


# ============================================================================
#  Tournaments
# ============================================================================

@router.get("/tournaments", response_model=list[TournamentOut])
def list_tournaments(db: Session = Depends(get_db)):
    rows = db.execute(
        text("SELECT id, name, tier FROM tournaments ORDER BY name")
    ).mappings().all()
    return [dict(r) for r in rows]


# ============================================================================
#  Raw SQL — Quick queries for prediction model
# ============================================================================

@router.get("/data/all-matches")
def get_all_matches_as_dataframe(
    from_date: Optional[date_type] = Query(None, alias="from"),
    db: Session = Depends(get_db),
):
    """Return all match data (JSON lines), for model training use"""
    sql = """
        SELECT m.date, ht.name AS home_team, at.name AS away_team,
               m.home_score, m.away_score, t.name AS tournament,
               m.neutral, m.venue
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        LEFT JOIN tournaments t ON m.tournament_id = t.id
    """
    params = {}
    if from_date:
        sql += " WHERE m.date >= :from_date"
        params["from_date"] = from_date
    sql += " ORDER BY m.date"

    rows = db.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]
