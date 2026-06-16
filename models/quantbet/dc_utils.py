"""
dc_utils.py — Dixon-Coles score matrix utilities
=================================================

Converts (λ_home, λ_away, ρ) into a (max+1)×(max+1) joint score matrix
M[h,a] = P(home=h, away=a), including the Dixon-Coles τ low-score correction.
This is what your Bayesian DC engine already produces — this module only
provides an independent, testable reference implementation for consumption
by markets / portfolio modules.

τ(h,a; λ,μ,ρ):
    (0,0): 1 - λμρ      (0,1): 1 + λρ
    (1,0): 1 + μρ       (1,1): 1 - ρ
    else : 1

Reference
---------
Dixon, M.J. & Coles, S.G. (1997) "Modelling Association Football Scores and
    Inefficiencies in the Football Betting Market", Applied Statistics 46(2).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

__all__ = ["tau", "dixon_coles_matrix"]


def tau(h: int, a: int, lam: float, mu: float, rho: float) -> float:
    if h == 0 and a == 0:
        return 1.0 - lam * mu * rho
    if h == 0 and a == 1:
        return 1.0 + lam * rho
    if h == 1 and a == 0:
        return 1.0 + mu * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def dixon_coles_matrix(lam: float, mu: float, rho: float = 0.0, max_goals: int = 10) -> np.ndarray:
    """Returns a normalised (max_goals+1)^2 joint score probability matrix."""
    h = np.arange(max_goals + 1)
    ph = poisson.pmf(h, lam)
    pa = poisson.pmf(h, mu)
    M = np.outer(ph, pa)
    # Apply τ to the four low-score cells
    for hh, aa in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        M[hh, aa] *= tau(hh, aa, lam, mu, rho)
    M = np.clip(M, 0.0, None)
    return M / M.sum()
