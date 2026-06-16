"""Test registry.py — Model loading, description, available versions, metadata"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from registry import (
    ModelNotFoundError,
    current_version,
    describe,
    list_available,
    load_model,
    model_metadata,
    paths,
)


class TestRegistry:
    def test_current_version(self) -> None:
        v = current_version()
        assert isinstance(v, str)
        assert v == "v9"

    def test_paths_known(self) -> None:
        p = paths("v9")
        assert "dc" in p
        assert p["dc"] is not None
        assert p["dc"].endswith(".pkl")

    def test_paths_unknown(self) -> None:
        with pytest.raises(KeyError):
            paths("v99")

    def test_load_model(self) -> None:
        dc = load_model()
        assert dc is not None
        assert hasattr(dc, "teams")
        assert len(dc.teams) > 100
        assert hasattr(dc, "predict")

    def test_load_model_with_gas(self) -> None:
        dc = load_model()
        assert getattr(dc, "_gas", None) is not None

    def test_describe_keys(self) -> None:
        d = describe()
        for key in ("selected_version", "n_teams", "has_gas", "has_elo_prior", "scoreline_engine"):
            assert key in d, f"Missing key: {key}"
        assert d["selected_version"] == "v9"
        assert d["has_gas"] is True
        assert d["has_elo_prior"] is True

    def test_describe_metadata(self) -> None:
        d = describe()
        assert "file_size_mb" in d
        assert d["file_size_mb"] > 1
        assert "file_hash_sha256" in d
        assert len(d["file_hash_sha256"]) == 16

    def test_list_available(self) -> None:
        avail = list_available()
        assert len(avail) >= 2
        v9 = [a for a in avail if a["version"] == "v9"][0]
        assert v9["artifacts"]["dc"]["exists"] is True

    def test_model_metadata(self) -> None:
        load_model()  # ensure a load record exists
        meta = model_metadata()
        assert "loaded_at" in meta
        assert "file_size_mb" in meta

    def test_predict_known_teams(self) -> None:
        dc = load_model()
        cap = [t for t in dc.teams if "Cape" in t][0]
        r = dc.predict("Spain", cap, venue="neutral")
        assert 0 < r["home_win_prob"] < 1
        assert 0 < r["draw_prob"] < 1
        assert 0 < r["away_win_prob"] < 1
        assert abs(r["home_win_prob"] + r["draw_prob"] + r["away_win_prob"] - 1) < 0.01
        assert "over_25" in r
        assert "btts" in r

    def test_predict_unknown_team(self) -> None:
        dc = load_model()
        from models.exceptions import PredictionError
        with pytest.raises(PredictionError):
            dc.predict("Mars", "Venus", venue="neutral")
