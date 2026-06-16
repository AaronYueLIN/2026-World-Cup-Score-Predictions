"""
quantbet — QuantBet-EV v6.1 enhancement modules
================================================

On top of v6.0 (Bayesian DC + HistGBM), fills in five areas: market implied
probability extraction, uncertainty quantification, risk-constrained
staking/portfolio, logarithmic pooling, and rigorous evaluation — drawing
from the financial economics and convex optimisation literature.

Sub-modules:
  devig            Shin (1992/1993) de-vig + proportional/power methods
  dc_utils         Dixon-Coles score matrix
  markets          Market predicates + exact joint probability (same-game parlay)
  staking          Kelly / fractional Kelly / posterior lower-quantile Kelly
  portfolio        Busseti-Ryu-Boyd risk-constrained Kelly portfolio (drawdown bound)
  posterior        Laplace posterior + posterior predictive probabilities
  pooling          Linear vs logarithmic pooling + RPS optimal weight
  evaluation       RPS/logloss/Brier + bootstrap CI + CLV + calibration
  value_engine_v2  Value layer orchestration (de-vig → EV → staking)
"""
from . import (
    dc_utils,
    devig,
    evaluation,
    markets,
    pooling,
    portfolio,
    posterior,
    staking,
    value_engine_v2,
)

__version__ = "6.1.0"

__all__ = [
    "devig", "dc_utils", "markets", "staking", "portfolio",
    "posterior", "pooling", "evaluation", "value_engine_v2",
]
