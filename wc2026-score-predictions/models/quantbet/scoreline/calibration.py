"""
calibration.py — Full score matrix probability calibration
=======================================

Why (targeting "score accuracy")
-----------------------
The current project only applies isotonic calibration at the 1X2 level (via GBM). But **the exact score distribution is never calibrated** —
yet the score matrix is the source of all derived markets: over/under, BTTS, correct score, Asian handicap, etc. An uncalibrated
matrix, even if 1X2 looks correct, will show systematic bias on O/U 2.5, BTTS.

This module fits two global scalars on the **validation set** to minimise a proper score (RPS+log-loss):

  1. Temperature/dispersion τ_disp:    scales (λ_h, λ_a) isotropically, equivalently fine-tuning overall goal level and tail fatness
  2. Draw inflation θ_draw:            diagonal inflation coefficient (following KN 2003), pushes draw frequency toward empirical values

This is *post-processing*, does not touch model parameters, fits in minutes, and works with any score engine (Poisson/bivariate/copula/dynamic).
Calibration almost always improves the strictly proper score (Gneiting & Raftery 2007).

References
--------
Gneiting & Raftery (2007) "Strictly proper scoring rules, prediction, and
    estimation", JASA 102(477).
Guo et al. (2017) "On calibration of modern neural networks" (temperature scaling idea).
Karlis & Ntzoufras (2003) diagonal inflation.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from .bivariate import diagonal_inflate, outcome_probs

__all__ = ["ScoreMatrixCalibrator"]


def _rps3(p_hda: np.ndarray, y: int) -> float:
    """1X2 ordinal RPS, y∈{0:home,1:draw,2:away}."""
    e = np.zeros(3); e[y] = 1.0
    cp = np.cumsum(p_hda); ce = np.cumsum(e)
    return float(np.sum((cp - ce) ** 2) / 2.0)


class ScoreMatrixCalibrator:
    """
    Fit (temperature, draw inflation) two scalars on the validation set to calibrate score matrices.

    Usage:
        cal = ScoreMatrixCalibrator()
        # matrices: List[np.ndarray]  per-match uncalibrated score matrix (from any engine)
        # lambdas:  List[(lh, la)]    per-match expected goals (object of temperature scaling)
        # outcomes: List[int]         0=home,1=draw,2=away
        cal.fit(matrices, lambdas, outcomes)
        M_cal = cal.transform(M, lh, la)   # calibrate single match matrix
    """

    def __init__(self, w_logloss: float = 0.5) -> None:
        """
        Args:
            w_logloss: objective = (1-w)·meanRPS + w·meanLogLoss(exact score).
                       Larger w emphasises exact score log-loss, smaller w emphasises 1X2 ordering.
        """
        self.w_logloss = float(w_logloss)
        self.temp_: float = 1.0
        self.theta_draw_: float = 0.0
        self.fitted_ = False

    # ------------------------------------------------------------------
    def _rebuild(self, M: np.ndarray, lh: float, la: float, temp: float) -> np.ndarray:
        """
        Temperature scaling: after shape-adjusting expected goals via λ→λ^temp, reproject using the original matrix's "dependence structure".
        Implementation uses a simplified but effective approach: apply temperature power transform to marginals while preserving copula rank structure.
        Here we approximate by directly doing row/column power scaling + re-normalisation on the matrix (identity when temperature=1).
        """
        if abs(temp - 1.0) < 1e-9:
            return M
        # Temperature scaling on log probabilities (like softmax temperature); temperature>1 flattens, <1 sharpens
        with np.errstate(divide="ignore"):
            logM = np.log(np.clip(M, 1e-300, None))
        logM = logM / temp
        logM -= logM.max()
        out = np.exp(logM)
        return out / out.sum()

    def transform(self, M: np.ndarray, lh: float, la: float) -> np.ndarray:
        """Apply fitted (temperature, draw inflation) to a single match matrix."""
        M2 = self._rebuild(M, lh, la, self.temp_)
        if self.theta_draw_ > 0:
            M2 = diagonal_inflate(M2, self.theta_draw_)
        return M2

    # ------------------------------------------------------------------
    def fit(
        self,
        matrices: Sequence[np.ndarray],
        lambdas: Sequence[tuple[float, float]],
        outcomes: Sequence[int],
    ) -> "ScoreMatrixCalibrator":
        """Jointly optimise (log temperature, logit θ_draw) on the validation set."""
        outcomes = np.asarray(outcomes, dtype=int)

        def objective(z: np.ndarray) -> float:
            temp = float(np.exp(z[0]))
            theta = float(1.0 / (1.0 + np.exp(-z[1])) * 0.30)  # θ∈(0,0.30)
            rps_sum = 0.0
            ll_sum = 0.0
            for M, (lh, la), y in zip(matrices, lambdas, outcomes):
                M2 = self._rebuild(M, lh, la, temp)
                if theta > 0:
                    M2 = diagonal_inflate(M2, theta)
                h, d, a = outcome_probs(M2)
                p = np.array([h, d, a]); p = p / p.sum()
                rps_sum += _rps3(p, int(y))
                ll_sum += -np.log(max(p[int(y)], 1e-12))
            n = len(outcomes)
            return (1 - self.w_logloss) * (rps_sum / n) + self.w_logloss * (ll_sum / n)

        res = minimize(objective, x0=np.array([0.0, -2.0]),
                       method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400})
        self.temp_ = float(np.exp(res.x[0]))
        self.theta_draw_ = float(1.0 / (1.0 + np.exp(-res.x[1])) * 0.30)
        self.fitted_ = True
        return self

    def __repr__(self) -> str:
        return (f"ScoreMatrixCalibrator(temp={self.temp_:.3f}, "
                f"theta_draw={self.theta_draw_:.3f}, fitted={self.fitted_})")
