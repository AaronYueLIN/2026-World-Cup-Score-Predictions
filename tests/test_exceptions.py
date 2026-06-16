"""Test exceptions.py — Exception hierarchy and status codes"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from models.exceptions import (
    DataError,
    GasFitError,
    ModelNotFoundError,
    PredictionError,
    QuantBetError,
)


class TestExceptions:
    def test_hierarchy_prediction(self) -> None:
        assert issubclass(PredictionError, QuantBetError)
        err = PredictionError("test")
        assert err.status_code == 422
        assert err.detail == "test"

    def test_hierarchy_model_not_found(self) -> None:
        assert issubclass(ModelNotFoundError, QuantBetError)
        err = ModelNotFoundError()
        assert err.status_code == 503
        assert "not available" in err.detail

    def test_hierarchy_gas_fit(self) -> None:
        assert issubclass(GasFitError, QuantBetError)
        err = GasFitError("gas failed")
        assert err.status_code == 500
        assert err.detail == "gas failed"

    def test_hierarchy_data(self) -> None:
        assert issubclass(DataError, QuantBetError)
        err = DataError("db error")
        assert err.status_code == 500

    def test_quant_bet_error_default(self) -> None:
        err = QuantBetError()
        assert err.status_code == 500
        assert err.detail == "Internal error"

    def test_prediction_error_default(self) -> None:
        err = PredictionError()
        assert err.detail == "Prediction failed"

    def test_custom_message(self) -> None:
        err = PredictionError("unknown team")
        assert err.detail == "unknown team"
