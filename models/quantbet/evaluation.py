"""
evaluation.py — Evaluation suite (proper scoring + bootstrap CI + CLV + calibration)
====================================================================================

Fixes two methodological issues in your current evaluation:

  1. n=5 "accuracy 40% vs 60%" is statistically meaningless (one match difference).
     Every metric should come with a bootstrap confidence interval; model comparison
     looks at whether intervals overlap, not point estimates.

  2. Introduces Closing Line Value (CLV) as the gold standard. Betting markets are
     efficient (Levitt 2004); the closing line beats most models. The most reliable
     leading indicator of long-term profitability is "your odds beat the closing odds"
     (positive CLV), not short-term ROI.

Ordinal convention (RPS): categories are ordered [home, draw, away], draw is middle.

References
----------
Epstein, E.S. (1969) RPS; Gneiting & Raftery (2007) JASA, proper scoring rules;
Levitt, S.D. (2004) "Why are gambling markets organised so differently from
    financial markets?", The Economic Journal 114, 223-246.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

__all__ = [
    "rps_score", "mean_rps", "log_loss_score", "brier_score",
    "bootstrap_ci", "compare_models_ci",
    "clv", "clv_summary", "reliability",
]


# ----- proper scoring rules ------------------------------------------------
def rps_score(p: Sequence[float], y: int) -> float:
    """Single-sample Ranked Probability Score (ordinal), lower is better."""
    p = np.asarray(p, dtype=float)
    r = p.size
    e = np.zeros(r); e[int(y)] = 1.0
    cp = np.cumsum(p); ce = np.cumsum(e)
    return float(np.sum((cp - ce) ** 2) / (r - 1))


def mean_rps(P: np.ndarray, y: Sequence[int]) -> float:
    P = np.asarray(P, dtype=float); y = np.asarray(y, dtype=int)
    return float(np.mean([rps_score(P[i], y[i]) for i in range(len(y))]))


def log_loss_score(p: Sequence[float], y: int) -> float:
    p = np.asarray(p, dtype=float)
    return float(-np.log(max(p[int(y)], 1e-15)))


def brier_score(p: Sequence[float], y: int) -> float:
    p = np.asarray(p, dtype=float)
    e = np.zeros(p.size); e[int(y)] = 1.0
    return float(np.sum((p - e) ** 2))


# ----- Uncertainty: bootstrap ------------------------------------------------
def bootstrap_ci(values: Sequence[float], stat: Callable = np.mean,
                 n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap per-sample metric, returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    boots = np.array([stat(values[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(values)), float(lo), float(hi)


def compare_models_ci(per_sample_a: Sequence[float], per_sample_b: Sequence[float],
                      n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """
    Paired bootstrap comparison of two models per-sample metric differences (a - b).
    Returns (mean_diff, lo, hi). If CI contains 0 → difference is not significant
    (almost guaranteed when n=5).
    """
    a = np.asarray(per_sample_a, dtype=float)
    b = np.asarray(per_sample_b, dtype=float)
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diff.mean()), float(lo), float(hi)


# ----- Closing Line Value --------------------------------------------------
def clv(bet_odds: float, closing_odds: float) -> float:
    """Single-bet CLV = bet_odds/closing_odds - 1 (>0 means bet price beat closing price)."""
    return float(bet_odds / closing_odds - 1.0)


def clv_summary(bet_odds: Sequence[float], closing_odds: Sequence[float]):
    """CLV summary: mean CLV, beat-close rate, sample size."""
    bet = np.asarray(bet_odds, dtype=float)
    close = np.asarray(closing_odds, dtype=float)
    edges = bet / close - 1.0
    return {
        "mean_clv": float(edges.mean()),
        "beat_close_rate": float((edges > 0).mean()),
        "n": int(len(edges)),
    }


# ----- Calibration ----------------------------------------------------------------
def reliability(prob_event: Sequence[float], outcome: Sequence[int], n_bins: int = 10):
    """
    Reliability curve + ECE (Expected Calibration Error).
    prob_event: predicted event probability; outcome: 0/1 actual.
    Returns (bin_centers, bin_acc, bin_conf, ece).
    """
    p = np.asarray(prob_event, dtype=float)
    y = np.asarray(outcome, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, acc, conf = [], [], []
    ece = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        centers.append((edges[i] + edges[i + 1]) / 2)
        if m.sum() > 0:
            a = float(y[m].mean()); c = float(p[m].mean())
            acc.append(a); conf.append(c)
            ece += (m.sum() / len(p)) * abs(a - c)
        else:
            acc.append(np.nan); conf.append(np.nan)
    return np.array(centers), np.array(acc), np.array(conf), float(ece)
