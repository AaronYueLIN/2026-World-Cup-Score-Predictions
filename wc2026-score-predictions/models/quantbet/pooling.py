"""
pooling.py — Probability aggregation (linear vs logarithmic pooling)
=====================================================================

Your current soft-vote is linear pooling P = w·P_dc + (1-w)·P_gbm. Linear
pooling over-disperses the combined distribution and hurts calibration
(Ranjan & Gneiting 2010). Logarithmic (geometric) pooling is the only
"externally Bayesian" pooling rule (Genest & Zidek 1986), and is typically
sharper and better calibrated for independent information sources:

    P_k ∝ Π_j P_{j,k}^{w_j}              (normalised)

This module provides both pooling methods + RPS-targeted weight optimisation,
for direct A/B comparison with your current 0.70/0.30 split.

References
----------
Genest, C. & Zidek, J.V. (1986) "Combining Probability Distributions: A
    Critique and an Annotated Bibliography", Statistical Science 1(1).
Ranjan, R. & Gneiting, T. (2010) "Combining probability forecasts", JRSS-B.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from .evaluation import rps_score

__all__ = ["linear_pool", "log_pool", "optimize_weight"]


def _normalize(P: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    P = np.asarray(P, dtype=float)
    return P / P.sum(axis=-1, keepdims=True)


def linear_pool(
    P_list: Sequence[npt.NDArray[np.float64]],
    weights: Sequence[float],
) -> npt.NDArray[np.float64]:
    """Linear pooling (arithmetic mean)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    P = sum(wi * np.asarray(p, dtype=float) for wi, p in zip(w, P_list))
    return _normalize(P)


def log_pool(
    P_list: Sequence[npt.NDArray[np.float64]],
    weights: Sequence[float],
    eps: float = 1e-12,
) -> npt.NDArray[np.float64]:
    """Logarithmic (geometric) pooling. Recommended."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    logP = sum(wi * np.log(np.asarray(p, dtype=float) + eps) for wi, p in zip(w, P_list))
    logP = logP - logP.max(axis=-1, keepdims=True)
    return _normalize(np.exp(logP))


def optimize_weight(
    P1: npt.NDArray[np.float64],
    P2: npt.NDArray[np.float64],
    y: Sequence[int],
    method: str = "log",
) -> float:
    """
    On the validation set, minimise average RPS to find the optimal weight w (assigned to P1, P2 gets 1-w).

    P1, P2 : (N, C) probability predictions from two models
    y      : (N,)   true class indices (ordinal, consistent with RPS ordinality)
    method : 'log' (default) or 'linear'
    """
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)
    y = np.asarray(y, dtype=int)
    pool = log_pool if method == "log" else linear_pool

    def loss(w):
        P = pool([P1, P2], [w, 1.0 - w])
        return float(np.mean([rps_score(P[i], y[i]) for i in range(len(y))]))

    res = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
    return float(res.x)
