"""
count_dists.py — Flexible univariate goal count distributions (Poisson / NegBin / Weibull-count)
==========================================================================

Why this is needed (targeting "score accuracy")
------------------------------
The current model uses **Poisson distribution** for each team's goals. Poisson is *equidispersed* (mean == var), but real
football goals in most league/national-team contexts are **mildly overdispersed** (var > mean), leading to:
  - systematic underestimation of high-score tail probabilities (scores like 3-2 / 4-3);
  - overestimation of low scores like 0-0 / 1-0 — exactly the patch Dixon-Coles τ tries to fix.

Switching the marginal distribution from Poisson to a **distribution with a dispersion parameter** (NegBin or Weibull-count)
effectively corrects dispersion at the data level, rather than patching it with 4-cell τ.

  - **Negative Binomial**: introduces dispersion r, var = μ + μ²/r. r→∞ degenerates to Poisson.
    Numerically stable, fast MLE, the safest production default.
  - **Weibull-count** (Boshnakov-Kharrat-McHale 2017, IJF): goal inter-arrival times follow
    Weibull instead of exponential; shape c controls "hazard rate over time" (c=1 degenerates to Poisson).
    The paper shows it fits score distributions better and yields positive out-of-sample betting returns.
    This file provides its exact pmf (McShane et al. 2008 polynomial expansion); c=1 is numerically consistent with Poisson.

References
--------
McShane, Adrian, Bradlow & Fader (2008) "Count Models Based on Weibull
    Interarrival Times", JBES 26(3).
Boshnakov, Kharrat & McHale (2017) "A bivariate Weibull count model for
    forecasting association football scores", Int. J. Forecasting 33(2), 458-466.
McHale & Scarf (2011) "Modelling the dependence of goals scored ...",
    Statistical Modelling 11(3), 219-236.  (negative binomial marginal)
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

__all__ = [
    "poisson_pmf_vec",
    "negbin_pmf_vec",
    "weibull_count_pmf_vec",
]


# ----------------------------------------------------------------------
#  Poisson (baseline, equivalent to current implementation, just for unified interface)
# ----------------------------------------------------------------------
def poisson_pmf_vec(mu: float, kmax: int) -> np.ndarray:
    """Returns vector P(N=k), k=0..kmax (Poisson)."""
    k = np.arange(kmax + 1)
    logp = k * np.log(max(mu, 1e-12)) - mu - gammaln(k + 1)
    p = np.exp(logp)
    return p / p.sum()


# ----------------------------------------------------------------------
#  Negative Binomial (overdispersed; production default)
# ----------------------------------------------------------------------
def negbin_pmf_vec(mu: float, r: float, kmax: int) -> np.ndarray:
    """
    Negative binomial pmf parameterised by mean μ and dispersion r (r = size).

        var = μ + μ² / r        (as r → ∞ → Poisson)
        P(N=k) = C(k+r-1, k) p^r (1-p)^k,  p = r / (r+μ)

    Args:
        mu:   expected goals (>0)
        r:    dispersion (>0). Smaller = more overdispersion; suggested initial value r≈8 (football empirical value).
        kmax: maximum goals to truncate at
    """
    mu = max(float(mu), 1e-12)
    r = max(float(r), 1e-6)
    k = np.arange(kmax + 1)
    logp = (
        gammaln(k + r) - gammaln(r) - gammaln(k + 1)
        + r * np.log(r / (r + mu))
        + k * np.log(mu / (r + mu))
    )
    p = np.exp(logp)
    return p / p.sum()


# ----------------------------------------------------------------------
#  Weibull-count (Boshnakov-Kharrat-McHale 2017 marginal)
# ----------------------------------------------------------------------
def _weibull_count_alpha(c: float, jmax: int) -> np.ndarray:
    """
    Recursion coefficients α_j^n from McShane et al. (2008), returns lower triangular matrix alpha[n, j].

        α_j^0 = Γ(c·j + 1) / Γ(j + 1)
        α_j^{n+1} = Σ_{m=n}^{j-1} α_m^n · Γ(c·j − c·m + 1) / Γ(j − m + 1)
    """
    j = np.arange(jmax + 1)
    log_gamma_cj = gammaln(c * j + 1.0)
    alpha = np.zeros((jmax + 1, jmax + 1))
    # n = 0 row
    alpha[0, :] = np.exp(log_gamma_cj - gammaln(j + 1.0))
    # recursion n = 1..jmax
    for n in range(jmax):
        for jj in range(n + 1, jmax + 1):
            m = np.arange(n, jj)
            coef = np.exp(gammaln(c * (jj - m) + 1.0) - gammaln(jj - m + 1.0))
            alpha[n + 1, jj] = np.dot(alpha[n, m], coef)
    return alpha


def weibull_count_pmf_vec(
    mu: float,
    c: float,
    kmax: int,
    jmax: int | None = None,
) -> np.ndarray:
    """
    Weibull-count distribution pmf, P(N=k), k=0..kmax.

    Parameterised by *mean* μ (internally solves for the Weibull rate λ such that E[N]≈μ), shape=c.
    c=1 strictly degenerates to Poisson; c<1 overdispersed; c>1 underdispersed.

        P(N=n) = Σ_{j=n}^{J} (−1)^{n+j} λ^j α_j^n / Γ(c·j + 1)

    Note: this is an alternating series that can be numerically unstable for extreme parameters.
    Within the football goal range (μ∈[0.2,3.5], c∈[0.7,1.3]), J = max(kmax+15, 40) is
    double-precision stable. In production, prefer NegBin for MLE; switch to this function only
    when you need the same model as BKM.

    Args:
        mu:   target mean goals
        c:    Weibull shape (1=Poisson)
        kmax: maximum goals to truncate at
        jmax: series truncation (default max(kmax+15, 40))
    """
    if abs(c - 1.0) < 1e-6:
        return poisson_pmf_vec(mu, kmax)

    if jmax is None:
        jmax = max(kmax + 15, 40)

    alpha = _weibull_count_alpha(c, jmax)
    j = np.arange(jmax + 1)
    log_gamma_cj = gammaln(c * j + 1.0)

    def pmf_for_lambda(lam: float) -> np.ndarray:
        log_lam = np.log(max(lam, 1e-12))
        out = np.zeros(kmax + 1)
        for n in range(kmax + 1):
            jj = np.arange(n, jmax + 1)
            sign = (-1.0) ** (n + jj)
            terms = sign * np.exp(jj * log_lam + np.log(np.abs(alpha[n, jj]) + 1e-300)
                                  - log_gamma_cj[jj]) * np.sign(alpha[n, jj] + 1e-300)
            out[n] = terms.sum()
        out = np.clip(out, 0.0, None)
        s = out.sum()
        return out / s if s > 0 else poisson_pmf_vec(mu, kmax)

    # Solve for λ via binary search to match the mean to μ (monotonic relationship)
    lo, hi = 1e-3, 50.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        p = pmf_for_lambda(mid)
        mean = float(np.dot(np.arange(kmax + 1), p))
        if mean < mu:
            lo = mid
        else:
            hi = mid
    return pmf_for_lambda(np.sqrt(lo * hi))
