"""
staking.py — Staking (Kelly and its variants)
==============================================

Single-bet Kelly:
    edge b = o - 1 (net odds), q = 1 - p
    f* = (b·p - q) / b = (p·o - 1) / (o - 1)         (positive when EV>0, otherwise 0)

An important and often miscommunicated fact about "uncertainty shrinkage"
----------------------------------------------------------------------
For **single-bet** log utility, the expected log growth
    g(f) = p·log(1+bf) + (1-p)·log(1-f)
is **linear** in p. Therefore, after integrating over the posterior
probability vector, the sufficient statistic is still the posterior mean p̄.
At the single-bet level, there is **no** automatic shrinkage due to the
variance of p. This must be stated honestly.

Two **correct** mechanisms that truly shrink stakes, each handled by this
module/portfolio:
  (1) Use the "posterior predictive mean" p̄ instead of the MAP plug-in —
      it is already shrunk by the prior and less extreme
      (see posterior.py output; feed directly into kelly_fraction);
  (2) Drawdown/risk constraints — see portfolio.risk_constrained_kelly,
      which replaces the fixed 1/4 Kelly with a strict mechanism offering
      "guaranteed drawdown bounds".

Also provides a practically common and literature-supported **conservative**
approach: use the lower-confidence edge (posterior quantile of the edge)
rather than the mean for staking, capturing the spirit of Baker & McHale
(2013) — "bet less under parameter uncertainty" — labelled as heuristic.

References
----------
Kelly, J.L. (1956) "A New Interpretation of Information Rate", Bell System
    Technical Journal 35, 917-926.
Baker, R.D. & McHale, I.G. (2013) "Optimal betting under parameter
    uncertainty: improving the Kelly criterion", JRSS-A.
MacLean, Thorp & Ziemba (2011) "The Kelly Capital Growth Investment Criterion".
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "edge",
    "expected_value",
    "kelly_fraction",
    "fractional_kelly",
    "lower_confidence_kelly",
]


def edge(p: float, odds: float) -> float:
    """Edge = true probability - break-even probability (1/odds)."""
    return float(p - 1.0 / odds)


def expected_value(p: float, odds: float, stake: float = 1.0) -> float:
    """Expected return per unit stake EV = p·(o-1) - (1-p)."""
    return float(stake * (p * (odds - 1.0) - (1.0 - p)))


def kelly_fraction(p: float, odds: float) -> float:
    """Full Kelly fraction, returns 0 when EV≤0."""
    b = odds - 1.0
    f = (b * p - (1.0 - p)) / b
    return float(max(f, 0.0))


def fractional_kelly(p: float, odds: float, fraction: float = 0.25) -> float:
    """Fractional Kelly (default 1/4). Multiplicative shrinkage, equivalent to additional risk aversion."""
    return float(fraction * kelly_fraction(p, odds))


def lower_confidence_kelly(
    p_samples: np.ndarray, odds: float, quantile: float = 0.25, cap: float = 1.0
) -> float:
    """
    Conservative Kelly under parameter uncertainty (heuristic, Baker-McHale spirit):
    Use the lower quantile (default 25%) of posterior probability samples for Kelly,
    rather than the posterior mean.
    Wider posterior → lower quantile → smaller stake; high-confidence matches are
    almost unaffected.

    p_samples : posterior probability samples for this outcome (1D array)
    quantile  : lower quantile to use (0.25 → conservative)
    cap       : maximum stake fraction
    """
    p_samples = np.asarray(p_samples, dtype=float)
    p_lcb = float(np.quantile(p_samples, quantile))
    f = kelly_fraction(p_lcb, odds)
    return float(min(f, cap))
