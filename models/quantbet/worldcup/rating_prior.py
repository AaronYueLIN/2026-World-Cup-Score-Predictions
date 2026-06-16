"""
rating_prior.py — Elo / ranking-informed hierarchical prior (World Cup module 1).

WHY (World Cup specific)
------------------------
National teams play ~10 competitive matches/year. After temporal decay
w_i = exp(-xi * days_i), most teams have too few effective samples to
estimate free attack/defence parameters. A zero-mean prior
    beta_att_i ~ N(0, sigma_att^2)
shrinks every under-sampled team toward the GLOBAL mean, which is wrong:
a weak team and a strong team with equal (small) sample sizes both get
pulled to 0. The fix is to anchor the prior MEAN to a standardised team
rating r_i (Elo / FIFA points / Bayesian Bradley-Terry log-strength):
    beta_att_i ~ N(eta_att * r_i, sigma_att^2)
    beta_def_i ~ N(eta_def * r_i, sigma_def^2)
so a data-starved team is pulled toward "a strength consistent with its
rating" rather than toward the field average.

Literature
----------
- Groll, Ley, Schauberger & Van Eetvelde (2018), arXiv:1806.03208 — ranking
  is the single most important covariate for World Cup score prediction.
- Macri Demartino et al. (2024), arXiv:2405.10247 — Bayesian Bradley-Terry-
  Davidson log-strength posterior median outperforms raw FIFA points.
- Ekstrom et al. (2021), JSA — Elo-based Bradley-Terry beats flat priors.

HOW TO WIRE INTO QuantBet-EV v7.0
---------------------------------
This module returns the prior MEAN vectors and the extra NLP terms for
eta. It is designed to *replace* the zero-mean attack/defence prior in
section 1.5 of the spec. In your NLP objective:

    OLD:
        nlp_att = (beta_att**2).sum() / (2*sigma_att**2) + n*log(sigma_att)
    NEW:
        prior = RatingPrior(ratings_std)
        mu_att, mu_def = prior.prior_means(eta_att, eta_def)
        nlp_att = ((beta_att - mu_att)**2).sum() / (2*sigma_att**2) + n*log(sigma_att)
        nlp_def = ((beta_def - mu_def)**2).sum() / (2*sigma_def**2) + n*log(sigma_def)
        nlp += prior.nlp_eta(eta_att, eta_def)

`eta_att`, `eta_def` become two extra free parameters in your theta vector
(append them; remember to bump k in AIC/BIC: k -> k + 2). The zero-sum
constraints in section 1.6 still apply to beta (NOT to mu); see
`projected_prior_means` if you want the prior mean itself to satisfy the
zero-sum constraint, which keeps identifiability clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def standardize_ratings(
    raw_ratings: Sequence[float] | FloatArray,
    *,
    method: str = "zscore",
    clip_sd: float = 4.0,
) -> FloatArray:
    """Standardise raw ratings (Elo, FIFA points, BT log-strength) to ~N(0,1).

    Parameters
    ----------
    raw_ratings
        One scalar per team, in the SAME team order as your beta vectors.
    method
        'zscore'   -> (r - mean) / std   (default, recommended)
        'minmax'   -> map to [-1, 1]
        'rank'     -> rank-transform then z-score (robust to Elo outliers)
    clip_sd
        Clip standardised values to +/- clip_sd to stop a single freak
        rating (e.g. a brand-new team with a default Elo) dominating.

    Returns
    -------
    FloatArray of shape (n_teams,), mean ~0, std ~1.
    """
    r = np.asarray(raw_ratings, dtype=np.float64).ravel()
    if r.size == 0:
        raise ValueError("raw_ratings is empty")
    if np.any(~np.isfinite(r)):
        raise ValueError("raw_ratings contains non-finite values")

    if method == "zscore":
        std = r.std()
        out = (r - r.mean()) / (std if std > 1e-12 else 1.0)
    elif method == "minmax":
        lo, hi = r.min(), r.max()
        span = hi - lo
        out = 2.0 * (r - lo) / (span if span > 1e-12 else 1.0) - 1.0
    elif method == "rank":
        order = r.argsort().argsort().astype(np.float64)
        order = (order - order.mean()) / (order.std() if order.std() > 1e-12 else 1.0)
        out = order
    else:
        raise ValueError(f"unknown method {method!r}")

    return np.clip(out, -clip_sd, clip_sd)


@dataclass
class RatingPrior:
    """Ranking-informed Gaussian prior on attack/defence parameters.

    The prior is
        beta_att_i ~ N(eta_att * r_i, sigma_att^2)
        beta_def_i ~ N(eta_def * r_i, sigma_def^2)
    where r_i is the standardised rating of team i.

    A higher rating should mean MORE attack and LESS conceded. With the
    log-linear convention of the spec (log lambda_h = att_h + def_a + ...),
    a stronger defence means a *more negative* def parameter. So in
    practice you expect eta_att > 0 and eta_def < 0. We do not hard-code
    the sign; eta is estimated freely and the hyper-prior is symmetric.

    Parameters
    ----------
    ratings_std
        Standardised ratings, shape (n_teams,), team-ordered.
    eta_prior_sd
        SD of the N(0, .) hyper-prior on eta_att and eta_def. 1.0 is a
        sensible weakly-informative default; lower it (e.g. 0.5) if you
        see eta blowing up on small dev datasets.
    enforce_zero_sum
        If True, the returned prior means are projected to satisfy
        sum(mu) = 0, matching the zero-sum identifiability constraint in
        spec section 1.6. Recommended True.
    """

    ratings_std: FloatArray
    eta_prior_sd: float = 1.0
    enforce_zero_sum: bool = True
    _r: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._r = np.asarray(self.ratings_std, dtype=np.float64).ravel()
        if self._r.size == 0:
            raise ValueError("ratings_std is empty")
        if self.eta_prior_sd <= 0:
            raise ValueError("eta_prior_sd must be positive")

    @property
    def n_teams(self) -> int:
        return self._r.size

    def _center(self, v: FloatArray) -> FloatArray:
        if self.enforce_zero_sum:
            return v - v.mean()
        return v

    def prior_means(
        self, eta_att: float, eta_def: float
    ) -> tuple[FloatArray, FloatArray]:
        """Return (mu_att, mu_def), the prior mean vectors for given eta."""
        mu_att = self._center(eta_att * self._r)
        mu_def = self._center(eta_def * self._r)
        return mu_att, mu_def

    def nlp_eta(self, eta_att: float, eta_def: float) -> float:
        """Negative-log-prior contribution of the eta hyperparameters.

        Add this scalar to your total NLP objective. It is the Gaussian
        hyper-prior eta ~ N(0, eta_prior_sd^2), constant terms dropped.
        """
        s2 = self.eta_prior_sd**2
        return float((eta_att**2 + eta_def**2) / (2.0 * s2))

    def nlp_beta(
        self,
        beta_att: FloatArray,
        beta_def: FloatArray,
        sigma_att: float,
        sigma_def: float,
        eta_att: float,
        eta_def: float,
    ) -> float:
        """Full NLP contribution of the attack/defence priors + eta hyper-prior.

        Drop-in replacement for the two prior terms in spec section 1.5.
        Returns NLP_att + NLP_def + NLP_eta (constants kept consistent with
        the spec: + n*log(sigma)).
        """
        beta_att = np.asarray(beta_att, dtype=np.float64).ravel()
        beta_def = np.asarray(beta_def, dtype=np.float64).ravel()
        mu_att, mu_def = self.prior_means(eta_att, eta_def)

        n = beta_att.size
        sa2 = sigma_att**2
        sd2 = sigma_def**2

        nlp_att = float(((beta_att - mu_att) ** 2).sum() / (2.0 * sa2)
                        + n * np.log(sigma_att))
        nlp_def = float(((beta_def - mu_def) ** 2).sum() / (2.0 * sd2)
                        + n * np.log(sigma_def))
        return nlp_att + nlp_def + self.nlp_eta(eta_att, eta_def)

    def grad_beta(
        self,
        beta_att: FloatArray,
        beta_def: FloatArray,
        sigma_att: float,
        sigma_def: float,
        eta_att: float,
        eta_def: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Analytic gradient of nlp_beta wrt (beta_att, beta_def).

        Useful if you pass jac to SLSQP. Gradient wrt eta is returned by
        `grad_eta`. (Centering for zero-sum is linear, so its effect on the
        gradient is a constant shift that cancels under the zero-sum
        constraint; we ignore it here, which is exact when beta is also
        zero-summed, as enforced in section 1.6.)
        """
        beta_att = np.asarray(beta_att, dtype=np.float64).ravel()
        beta_def = np.asarray(beta_def, dtype=np.float64).ravel()
        mu_att, mu_def = self.prior_means(eta_att, eta_def)
        g_att = (beta_att - mu_att) / (sigma_att**2)
        g_def = (beta_def - mu_def) / (sigma_def**2)
        return g_att, g_def

    def grad_eta(
        self,
        beta_att: FloatArray,
        beta_def: FloatArray,
        sigma_att: float,
        sigma_def: float,
        eta_att: float,
        eta_def: float,
    ) -> tuple[float, float]:
        """Analytic gradient of nlp_beta wrt (eta_att, eta_def)."""
        beta_att = np.asarray(beta_att, dtype=np.float64).ravel()
        beta_def = np.asarray(beta_def, dtype=np.float64).ravel()
        mu_att, mu_def = self.prior_means(eta_att, eta_def)
        r = self._center(self._r) if self.enforce_zero_sum else self._r
        # d/d eta_att [ sum (beta_att - eta_att*r_c)^2 / (2 sa2) ] + eta_att/s2
        g_eta_att = float(-(r * (beta_att - mu_att)).sum() / (sigma_att**2)
                          + eta_att / (self.eta_prior_sd**2))
        g_eta_def = float(-(r * (beta_def - mu_def)).sum() / (sigma_def**2)
                          + eta_def / (self.eta_prior_sd**2))
        return g_eta_att, g_eta_def


# ---------------------------------------------------------------------------
# Optional: Bayesian Bradley-Terry-Davidson rating producer.
# Use this to MANUFACTURE a better r_i than raw FIFA points, per
# Macri Demartino et al. (2024). Output feeds standardize_ratings().
# ---------------------------------------------------------------------------
def bradley_terry_log_strength(
    home_idx: Sequence[int],
    away_idx: Sequence[int],
    outcomes: Sequence[int],  # 1 home win, 0 draw, -1 away win
    n_teams: int,
    *,
    nu: float = 1.0,            # Davidson draw parameter (>0); 1.0 = neutral
    ridge: float = 1e-3,        # L2 shrinkage on log-strengths
    max_iter: int = 500,
    tol: float = 1e-8,
) -> FloatArray:
    """Estimate Bradley-Terry-Davidson log-strengths by MAP (Newton/IRLS-ish).

    Returns one log-strength per team (zero-summed). These are MORE
    informative priors than raw FIFA points because they account for
    strength-of-schedule. Standardise the output before passing to
    RatingPrior.

    This is a lightweight pure-numpy implementation (no SciPy needed for
    the fit itself) using projected gradient ascent on the penalised
    log-likelihood. It is intended for the ~hundreds of teams / tens of
    thousands of matches scale and converges in well under max_iter.
    """
    h = np.asarray(home_idx, dtype=np.int64)
    a = np.asarray(away_idx, dtype=np.int64)
    y = np.asarray(outcomes, dtype=np.int64)
    if not (h.shape == a.shape == y.shape):
        raise ValueError("home_idx, away_idx, outcomes must have equal length")

    theta = np.zeros(n_teams, dtype=np.float64)
    nu = max(nu, 1e-6)
    log_nu = np.log(nu)
    lr = 0.1

    prev_obj = -np.inf
    for _ in range(max_iter):
        d = theta[h] - theta[a]  # strength diff
        # Davidson: P(home), P(draw), P(away)
        eh = np.exp(d / 2.0)
        ea = np.exp(-d / 2.0)
        ed = nu  # exp(log_nu + 0) since draw term ~ nu * sqrt(pi_h pi_a)
        # use the standard Davidson parameterisation on differences:
        z = eh + ea + ed
        p_home = eh / z
        p_draw = ed / z
        p_away = ea / z

        # gradient of log-lik wrt d for each match
        # outcome encoded: +1 home, 0 draw, -1 away
        g = np.zeros_like(d)
        is_home = y == 1
        is_draw = y == 0
        is_away = y == -1
        # d log p_home / d d = 0.5 * (1 - p_home) - 0.5*p_away ... derive via softmax
        # Using softmax over scores s_home=d/2, s_draw=log_nu, s_away=-d/2:
        # grad wrt d of log p_outcome = (indicator chain). Compute cleanly:
        # dscore/dd: home +0.5, draw 0, away -0.5
        dscore_dd = np.where(is_home, 0.5, np.where(is_away, -0.5, 0.0))
        exp_dscore_dd = 0.5 * p_home - 0.5 * p_away  # E[dscore/dd]
        g = dscore_dd - exp_dscore_dd

        # accumulate to per-team gradient
        grad = np.zeros(n_teams, dtype=np.float64)
        np.add.at(grad, h, g)
        np.add.at(grad, a, -g)
        grad -= ridge * theta  # L2 penalty

        theta = theta + lr * grad / max(len(y), 1) * len(y)
        theta -= theta.mean()  # zero-sum

        obj = (
            np.where(is_home, np.log(p_home + 1e-12),
                     np.where(is_draw, np.log(p_draw + 1e-12),
                              np.log(p_away + 1e-12))).sum()
            - 0.5 * ridge * (theta**2).sum()
        )
        if abs(obj - prev_obj) < tol:
            break
        prev_obj = obj

    return theta - theta.mean()


__all__ = [
    "standardize_ratings",
    "RatingPrior",
    "bradley_terry_log_strength",
]
