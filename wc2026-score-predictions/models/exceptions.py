"""QuantBet-EV Exception Hierarchy (Enterprise)

Usage:
    raise PredictionError("Team not found in model")
    Caught by global exception handler in FastAPI → 422
"""
from __future__ import annotations


class QuantBetError(Exception):
    """Base class for all QuantBet exceptions."""

    status_code: int = 500
    detail: str = "Internal error"


class ModelNotFoundError(QuantBetError):
    """Model artifact not found or unable to load."""

    status_code: int = 503

    def __init__(self, detail: str = "Model not available") -> None:
        self.detail = detail


class PredictionError(QuantBetError):
    """Prediction request error (team not found, invalid venue, etc.)."""

    status_code: int = 422

    def __init__(self, detail: str = "Prediction failed") -> None:
        self.detail = detail


class GasFitError(QuantBetError):
    """GAS fitting failed."""

    status_code: int = 500

    def __init__(self, detail: str = "GAS fitting failed") -> None:
        self.detail = detail


class DataError(QuantBetError):
    """Database query exception."""

    status_code: int = 500

    def __init__(self, detail: str = "Data query failed") -> None:
        self.detail = detail
