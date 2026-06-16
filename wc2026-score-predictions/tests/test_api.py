"""Test API schemas — Pydantic model validation"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from pydantic import ValidationError


class TestPredictionRequest:
    def test_valid_request(self) -> None:
        req = PredictionRequest(home_team="Spain", away_team="Cape Verde")
        assert req.home_team == "Spain"
        assert req.venue == "neutral"
        assert req.date is None

    def test_valid_full_request(self) -> None:
        req = PredictionRequest(
            home_team="Spain",
            away_team="Cape Verde",
            venue="home",
            date="2026-06-13",
        )
        assert req.venue == "home"
        assert req.date == "2026-06-13"

    def test_invalid_venue(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest(home_team="A", away_team="B", venue="moon")

    def test_empty_team_name(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest(home_team="", away_team="B")

    def test_minimal_length(self) -> None:
        req = PredictionRequest(home_team="A", away_team="B")
        assert req.home_team == "A"


class TestPredictionResponse:
    def test_valid_response(self) -> None:
        resp = PredictionResponse(
            home_team="Spain",
            away_team="Cape Verde",
            venue="neutral",
            home_win_prob=0.81,
            draw_prob=0.13,
            away_win_prob=0.06,
            expected_home_goals=3.1,
            expected_away_goals=0.5,
            over_25=0.6,
            btts=0.3,
        )
        assert resp.home_win_prob == 0.81
        assert 0 < resp.home_win_prob + resp.draw_prob + resp.away_win_prob < 1.01
        assert resp.model_version == "v9"
        assert resp.gas_b == 0.985

    def test_prob_sum(self) -> None:
        resp = PredictionResponse(
            home_team="A", away_team="B", venue="neutral",
            home_win_prob=0.5, draw_prob=0.3, away_win_prob=0.2,
            expected_home_goals=1.5, expected_away_goals=1.0,
            over_25=0.45, btts=0.5,
        )
        total = resp.home_win_prob + resp.draw_prob + resp.away_win_prob
        assert abs(total - 1.0) < 0.01

    def test_invalid_prob_range(self) -> None:
        with pytest.raises(ValidationError):
            PredictionResponse(
                home_team="A", away_team="B", venue="neutral",
                home_win_prob=1.5, draw_prob=0.3, away_win_prob=-0.1,
                expected_home_goals=1.0, expected_away_goals=1.0,
                over_25=0.5, btts=0.5,
            )


class TestHealthResponse:
    def test_default(self) -> None:
        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.model_loaded is False
        assert resp.teams == 0

    def test_full(self) -> None:
        resp = HealthResponse(
            status="ok", model_loaded=True, gas_active=True,
            teams=336, database="sqlite",
        )
        assert resp.gas_active is True
        assert resp.teams == 336
