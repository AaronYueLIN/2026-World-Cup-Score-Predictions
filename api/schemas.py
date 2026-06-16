"""Pydantic response/request models"""
from __future__ import annotations

from datetime import date as date_type
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Existing data query models (keep unchanged)
# ---------------------------------------------------------------------------

class MatchOut(BaseModel):
    id: int
    date: date_type
    home_team: str
    home_score: int
    away_team: str
    away_score: int
    tournament: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    neutral: bool = False
    venue: str = "neutral"

    model_config = {"from_attributes": True}


class TeamOut(BaseModel):
    id: int
    name: str
    fifa_code: Optional[str] = None

    model_config = {"from_attributes": True}


class MatchBrief(BaseModel):
    id: int
    date: date_type
    home_team: str
    home_score: int
    away_team: str
    away_score: int
    tournament: Optional[str] = None

    model_config = {"from_attributes": True}


class TeamHistoryOut(BaseModel):
    team: str
    total_matches: int
    wins: int
    draws: int
    losses: int
    matches: list[MatchBrief]


class TournamentOut(BaseModel):
    id: int
    name: str
    tier: str

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Phase 2: Prediction API models
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Prediction request body."""
    home_team: str = Field(..., min_length=1, max_length=100, description="Home team name")
    away_team: str = Field(..., min_length=1, max_length=100, description="Away team name")
    venue: str = Field(default="neutral", pattern="^(home|neutral|away)$")
    date: Optional[str] = Field(default=None, description="ISO date (YYYY-MM-DD), used for GAS decay")


class PredictionResponse(BaseModel):
    """Prediction response body — type-safe, auto-generated OpenAPI docs."""
    home_team: str
    away_team: str
    venue: str
    home_win_prob: float = Field(..., ge=0, le=1)
    draw_prob: float = Field(..., ge=0, le=1)
    away_win_prob: float = Field(..., ge=0, le=1)
    expected_home_goals: float = Field(..., ge=0)
    expected_away_goals: float = Field(..., ge=0)
    over_25: float = Field(..., ge=0, le=1)
    btts: float = Field(..., ge=0, le=1)
    model_version: str = "v9"
    gas_b: float = 0.985
    pool_method: Optional[str] = None
    handicap_2: Optional[dict] = None
    top10_scores: Optional[dict] = None
    recent_matches: Optional[dict] = None
    trace: Optional[dict] = None

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    model_loaded: bool = False
    gas_active: bool = False
    teams: int = 0
    database: str = "unknown"

    model_config = {"protected_namespaces": ()}


# ---------------------------------------------------------------------------
# Phase 7: LLM Chat + Model Explain API
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatData(BaseModel):
    type: str = "text"
    home_team: str | None = None
    away_team: str | None = None
    home_win_prob: float | None = None
    draw_prob: float | None = None
    away_win_prob: float | None = None
    expected_home_goals: float | None = None
    expected_away_goals: float | None = None


class ChatResponse(BaseModel):
    reply: str
    data: ChatData | None = None


class MomentumEntry(BaseModel):
    team: str
    momentum: float
    d_att: float
    d_def: float


class MomentumResponse(BaseModel):
    teams: list[MomentumEntry]


class LayerInfo(BaseModel):
    name: str
    value: str
    description: str


class ExplainResponse(BaseModel):
    home_team: str
    away_team: str
    layers: list[LayerInfo]
    summary: str

