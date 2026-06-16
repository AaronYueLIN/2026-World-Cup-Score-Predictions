"""
bivariate.py — Bivariate goal joint distribution (replaces independent Poisson + 4-cell τ patch)
=================================================================================================

Why (targeting "score accuracy")
---------------------------------
Current engine = independent Poisson outer product + τ correction on only 4 cells
(0-0/1-0/0-1/1-1). It has two weaknesses:

  1. **Dependence touches only 4 cells**. Real goal dependence (both teams
     attacking/parking the bus) pervades the entire score table; τ cannot
     capture the co-movement in 2-1 / 2-2 / 3-1 etc.
  2. **Draw calibration is patched via τ** — the systematic bias is never
     eliminated (as noted by Karlis-Ntzoufras 2003).

This module provides three *full-table* dependence models, all returning a
normalised (kmax+1)×(kmax+1) score matrix that can directly replace the
`np.outer + τ` two-step in BayesianDixonColesModel.predict():

  A. bivariate_poisson_matrix  —— Karlis-Ntzoufras shared component λ₃ (positive dependence)
  B. diagonal_inflate          —— diagonal inflation, calibrating draws (KN 2003's proper fix, replacing τ)
  C. frank_copula_matrix       —— Sklar/copula gluing arbitrary marginals (incl. NB/Weibull),
                                  supports positive and negative dependence (Boshnakov 2017 estimates weak negative)

Recommended combinations:
  · League/club: Frank-copula(Weibull marginals)  — closest to BKM 2017 frontier
  · National team/small sample: bivariate_poisson + diagonal_inflate — stable, few parameters

References
----------
Karlis & Ntzoufras (2003) "Analysis of sports data by using bivariate Poisson
    models", JRSS-D 52, 381-393.  (bivariate Poisson + diagonal inflation)
McHale & Scarf (2007/2011) copula gluing of goal marginals.
Boshnakov, Kharrat & McHale (2017) IJF 33(2).  (Weibull marginal + Frank copula)
Sklar (1959) copula theorem.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

__all__ = [
    "independent_matrix",
    "bivariate_poisson_matrix",
    "diagonal_inflate",
    "frank_copula_matrix",
    "outcome_probs",
]


def _norm(M: np.ndarray) -> np.ndarray:
    s = M.sum()
    return M / s if s > 0 else M


def independent_matrix(pmf_h: np.ndarray, pmf_a: np.ndarray) -> np.ndarray:
    """Independent outer product (baseline). pmf_h[i] = P(home=i)."""
    return _norm(np.outer(pmf_h, pmf_a))


# ----------------------------------------------------------------------
#  A. Bivariate Poisson (Karlis-Ntzoufras 2003)
# ----------------------------------------------------------------------
def bivariate_poisson_matrix(
    lambda_h: float,
    lambda_a: float,
    lambda3: float,
    kmax: int,
) -> np.ndarray:
    """
    Bivariate Poisson joint pmf. Introduces a shared component λ₃ creating *positive* dependence:

        X = X1 + X3,  Y = X2 + X3,  X1~Poi(λ1), X2~Poi(λ2), X3~Poi(λ3)
        Cov(X, Y) = λ3 ≥ 0

    where λ1 = λ_h − λ3, λ2 = λ_a − λ3 (requires λ3 < min(λ_h, λ_a)).

        P(X=x, Y=y) = e^{-(λ1+λ2+λ3)} (λ1^x/x!)(λ2^y/y!)
                      · Σ_{k=0}^{min(x,y)} C(x,k)C(y,k) k! (λ3/(λ1 λ2))^k

    λ3=0 degenerates to independent Poisson. In football λ3 is typically small
    but significant (KN: even small λ3 improves draw count prediction).

    Args:
        lambda_h, lambda_a: home/away expected goals
        lambda3:            shared component (covariance), auto-clipped to [0, 0.95·min(λh,λa))
        kmax:               truncation
    """
    l3 = float(np.clip(lambda3, 0.0, 0.95 * min(lambda_h, lambda_a)))
    l1 = max(lambda_h - l3, 1e-9)
    l2 = max(lambda_a - l3, 1e-9)

    x = np.arange(kmax + 1)
    log_fact = gammaln(x + 1.0)

    # marginal log components
    log_px = x * np.log(l1) - l1 - log_fact          # home (no λ3)
    log_py = x * np.log(l2) - l2 - log_fact          # away (no λ3)
    base = np.exp(log_px[:, None] + log_py[None, :] - l3)  # e^{-λ3} term

    M = base.copy()
    if l3 > 0:
        # convolution sum term Σ_k ...
        ratio = l3 / (l1 * l2)
        conv = np.zeros((kmax + 1, kmax + 1))
        for xx in range(kmax + 1):
            for yy in range(kmax + 1):
                kk = np.arange(min(xx, yy) + 1)
                logterm = (
                    gammaln(xx + 1) - gammaln(kk + 1) - gammaln(xx - kk + 1)
                    + gammaln(yy + 1) - gammaln(yy - kk + 1)   # C(x,k)C(y,k)k! combined
                    + kk * np.log(ratio + 1e-300)
                )
                conv[xx, yy] = np.exp(logterm).sum()
        M = base * conv
    return _norm(M)


# ----------------------------------------------------------------------
#  B. Diagonal inflation (Karlis-Ntzoufras 2003) —— the proper replacement for τ
# ----------------------------------------------------------------------
def diagonal_inflate(
    M: np.ndarray,
    theta: float,
    diag_dist: np.ndarray | None = None,
) -> np.ndarray:
    """
    Diagonal inflation: injects extra mass θ onto the diagonal (draws), precisely calibrating draw frequency.

        P*(x, y) = (1 − θ) · P(x, y)               , x ≠ y
        P*(x, x) = (1 − θ) · P(x, x) + θ · D(x)

    D is a discrete distribution on the diagonal (default: normalised diagonal of the original matrix). This is the core of KN 2003
    "diagonal-inflated bivariate Poisson", cleaner than Dixon-Coles τ:
    τ only touches 4 cells and can produce negative probabilities; diagonal inflation applies to *all* draw scores and always yields valid probabilities.

    Args:
        M:         normalised joint matrix
        theta:     inflation proportion ∈ [0, 1) (to be MLE/calibration estimated, football range 0.0~0.12)
        diag_dist: diagonal distribution D(x); None uses M's own diagonal normalised
    """
    theta = float(np.clip(theta, 0.0, 0.999))
    n = M.shape[0]
    out = (1.0 - theta) * M
    diag_idx = np.arange(n)
    if diag_dist is None:
        d = np.diag(M).astype(float)
        d = d / d.sum() if d.sum() > 0 else np.ones(n) / n
    else:
        d = diag_dist / diag_dist.sum()
    out[diag_idx, diag_idx] += theta * d
    return _norm(out)


# ----------------------------------------------------------------------
#  C. Frank copula glue (arbitrary marginals, supports positive and negative dependence)
# ----------------------------------------------------------------------
def _frank_cdf(u: np.ndarray, v: np.ndarray, kappa: float) -> np.ndarray:
    """Frank copula CDF C(u,v;κ). κ→0 yields independence."""
    if abs(kappa) < 1e-8:
        return u * v
    eu = np.exp(-kappa * u)
    ev = np.exp(-kappa * v)
    e1 = np.exp(-kappa)
    return -1.0 / kappa * np.log(1.0 + (eu - 1.0) * (ev - 1.0) / (e1 - 1.0))


def frank_copula_matrix(
    pmf_h: np.ndarray,
    pmf_a: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """
    Glue two *arbitrary* univariate count pmfs into a joint score matrix via Frank copula.
    This is the dependence mechanism of Boshnakov-Kharrat-McHale (2017) (they used Weibull marginals).

    For discrete marginals, use Sklar's theorem via "rectangle differencing":
        P(X=x, Y=y) = C(F_h(x),   F_a(y))
                    − C(F_h(x−1), F_a(y))
                    − C(F_h(x),   F_a(y−1))
                    + C(F_h(x−1), F_a(y−1))

    κ > 0 → positive dependence; κ < 0 → negative dependence (BKM estimated κ≈−0.46 for the EPL, weak negative).
    κ = 0 → independence, degenerates to outer product.

    Args:
        pmf_h, pmf_a: marginal pmfs of the two teams (from count_dists, can be Poisson/NB/Weibull)
        kappa:        Frank dependence parameter (to be MLE estimated)
    """
    Fh = np.clip(np.cumsum(pmf_h), 0.0, 1.0)
    Fa = np.clip(np.cumsum(pmf_a), 0.0, 1.0)
    Fh0 = np.concatenate([[0.0], Fh[:-1]])  # F_h(x-1)
    Fa0 = np.concatenate([[0.0], Fa[:-1]])

    # mesh grid
    U1, V1 = np.meshgrid(Fh, Fa, indexing="ij")
    U0, V0 = np.meshgrid(Fh0, Fa0, indexing="ij")

    M = (
        _frank_cdf(U1, V1, kappa)
        - _frank_cdf(U0, V1, kappa)
        - _frank_cdf(U1, V0, kappa)
        + _frank_cdf(U0, V0, kappa)
    )
    M = np.clip(M, 0.0, None)
    return _norm(M)


# ----------------------------------------------------------------------
#  Read 1X2 from score matrix
# ----------------------------------------------------------------------
def outcome_probs(M: np.ndarray) -> tuple[float, float, float]:
    """Returns (home_win, draw, away_win)."""
    home = float(np.tril(M, -1).sum())
    draw = float(np.trace(M))
    away = float(np.triu(M, 1).sum())
    return home, draw, away
