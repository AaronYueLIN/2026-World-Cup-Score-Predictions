"""
QuantBet-EV Model Layer
Core: BayesianDixonColesModel + Registry
"""
from .bayesian_dixon_coles import BayesianDixonColesModel
from .registry import load_model, load_ensemble, describe, current_version, list_available, model_metadata

__all__ = [
    "BayesianDixonColesModel",
    "load_model",
    "load_ensemble",
    "describe",
    "current_version",
    "list_available",
    "model_metadata",
]
