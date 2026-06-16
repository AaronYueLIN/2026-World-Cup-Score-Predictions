"""
posterior.py — Upgrade from MAP to posterior (Laplace approximation)
=====================================================================

Currently you estimate the MAP point. At the optimum, take the Hessian H of
the negative log posterior, and approximate the posterior as
    θ ~ N( θ̂_MAP , H^{-1} )
at almost zero extra cost, unlocking "posterior predictive distributions"
and "uncertainty-aware staking".

Constraint handling (Σ attack_i = 0): this equality constraint makes H
singular in one direction. This module eigendecomposes the covariance,
zeroes out the variance in zero/negative eigenvalue directions (no jitter
in that direction), so sampling naturally falls on the constraint manifold;
you can also pass constraint_proj to project each sample (e.g. demean the
attack block) to strictly satisfy the constraint.

Production note: for large n, the numerical Hessian is slow (O(n^2)
function evaluations). Swap to NumPyro/Pyro ADVI (variational inference)
for a more reliable posterior; this Laplace version is for fast,
zero-dependency deployment.

Reference
---------
Gelman et al. (2013) Bayesian Data Analysis, 3rd Ed, Ch.4 (Laplace / normal
    approximation to the posterior).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

__all__ = ["numerical_hessian", "laplace_covariance", "posterior_predictive"]


def numerical_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central difference numerical Hessian. f is the negative log posterior (scalar)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps
            fpp = f(x + ei + ej)
            fpm = f(x + ei - ej)
            fmp = f(x - ei + ej)
            fmm = f(x - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
    return 0.5 * (H + H.T)


def laplace_covariance(H: np.ndarray, ridge: float = 1e-8):
    """
    Posterior covariance = H^{-1} from Hessian (for positive eigenvalues).
    Singular/negative directions (including constraint nullspace) have zero variance.
    Returns (cov, eigvals).
    """
    H = 0.5 * (np.asarray(H, float) + np.asarray(H, float).T)
    w, V = np.linalg.eigh(H)
    inv = np.where(w > ridge, 1.0 / w, 0.0)
    cov = (V * inv) @ V.T
    return 0.5 * (cov + cov.T), w


def posterior_predictive(
    theta_map: np.ndarray,
    cov: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 500,
    constraint_proj: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    seed: int = 0,
):
    """
    Sample from N(θ̂, cov), push through predict_fn to probability space, returns (mean_pred, samples).

    predict_fn(theta) -> probability vector (e.g. [p_home, p_draw, p_away]).
    The returned mean_pred is the "posterior predictive probability", less extreme than MAP
    plug-in and more suitable for staking.
    samples has shape (n_samples, n_classes), can be fed to staking.lower_confidence_kelly.
    """
    theta_map = np.asarray(theta_map, dtype=float)
    cov = np.asarray(cov, dtype=float)
    rng = np.random.default_rng(seed)

    # Construct samples via eigendecomposition (robust for singular cov)
    w, V = np.linalg.eigh(0.5 * (cov + cov.T))
    w = np.clip(w, 0.0, None)
    A = V * np.sqrt(w)  # (n, n)
    z = rng.standard_normal((n_samples, theta_map.size))
    thetas = theta_map[None, :] + z @ A.T

    preds = []
    for th in thetas:
        if constraint_proj is not None:
            th = constraint_proj(th)
        preds.append(np.asarray(predict_fn(th), dtype=float))
    P = np.vstack(preds)
    return P.mean(axis=0), P
