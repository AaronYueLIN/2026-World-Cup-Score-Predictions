"""
knockout.py — Extra time + penalty shootout resolution (World Cup module 2).

WHY (World Cup specific)
------------------------
Your 11x11 score matrix and the P(Home)/P(Draw)/P(Away) summary in spec
sections 2.1-2.3 only cover 90 minutes. In a knockout match a draw is NOT
a terminal outcome: it goes to 30' extra time (ET), then a penalty
shootout. So "probability team advances" != "P(win in 90 min)". This
module turns a 90-minute score matrix into a clean advancement probability.

Two-layer model (Groll et al., UEFA EURO 2024, arXiv:2410.09068):
  1. If the 90-min result is decisive -> that team advances.
  2. If drawn after 90 min -> simulate 30' ET with lambda scaled by 1/3
     (shorter time to score). Standard scale factors in the literature:
     1/3 for 30 min vs 90 min (Groll EURO 2024); some use 1/6 for the
     10-min handball periods. We default to 1/3 and make it configurable.
  3. If still drawn after ET -> penalty shootout. Default 50/50 (a coin
     flip is the robust choice absent shootout-specific data). You may
     pass a custom p_home_shootout if you have a keeper/penalty model.

Everything is computed ANALYTICALLY from the score matrices (no Monte
Carlo needed for a single tie), so it is exact and fast. The tournament
simulator (module 3) calls into this per knockout tie.

Literature
----------
- Groll et al. (2024) EURO 2024, arXiv:2410.09068 — ET = 1/3 lambda, then 50/50.
- Groll et al. (2019) handball, arXiv:1901.05722 — ET scaling + coin-flip pens.

HOW TO WIRE INTO QuantBet-EV v7.0
---------------------------------
Your model already produces lambda_h, lambda_a and a 90-min score matrix
(with the Dixon-Coles tau correction applied to the 2x2 corner). Feed
those here:

    res = KnockoutResolver()
    adv = res.advancement_prob(
        score_matrix_90=P90,         # 11x11, already tau-corrected + normalised
        lambda_h=lh, lambda_a=la,    # for building the ET matrix
        tau_fn=model.tau,            # optional: your DC tau callable
        rho=model.rho,               # optional: DC dependence param
    )
    # adv.p_home_advance, adv.p_away_advance  (sum to 1)

For the win/draw/loss MARKET you still report P90 as before; this module
is specifically for advancement / "to qualify" / outright markets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _poisson_pmf_vector(lam: float, k_max: int) -> FloatArray:
    """Poisson pmf for k = 0..k_max, numerically stable via log-space."""
    k = np.arange(k_max + 1, dtype=np.float64)
    lam = float(np.clip(lam, 1e-9, None))
    log_p = -lam + k * np.log(lam) - _log_factorial(k)
    return np.exp(log_p)


def _log_factorial(k: FloatArray) -> FloatArray:
    from scipy.special import gammaln  # local import keeps top-level light
    return gammaln(k + 1.0)


def build_score_matrix(
    lambda_h: float,
    lambda_a: float,
    *,
    size: int = 11,
    rho: float = 0.0,
    tau_fn: Optional[Callable[[int, int, float, float, float], float]] = None,
    normalize: bool = True,
) -> FloatArray:
    """Build a (size x size) double-Poisson score matrix, optional DC tau.

    Mirrors spec sections 2.1-2.3 so the ET matrix is built the same way as
    your 90-min matrix. If you already have a builder, pass your own matrix
    to the resolver instead and ignore this helper.
    """
    ph = _poisson_pmf_vector(lambda_h, size - 1)
    pa = _poisson_pmf_vector(lambda_a, size - 1)
    m = np.outer(ph, pa)

    if tau_fn is not None:
        for x in (0, 1):
            for y in (0, 1):
                m[x, y] *= tau_fn(x, y, lambda_h, lambda_a, rho)

    if normalize:
        s = m.sum()
        if s > 0:
            m = m / s
    return m


def split_home_draw_away(score_matrix: FloatArray) -> tuple[float, float, float]:
    """Return (P_home_win, P_draw, P_away_win) from a score matrix."""
    p_home = float(np.tril(score_matrix, -1).sum())  # x > y
    p_away = float(np.triu(score_matrix, 1).sum())   # x < y
    p_draw = float(np.trace(score_matrix))           # x == y
    return p_home, p_draw, p_away


@dataclass
class AdvancementResult:
    """Outcome of resolving a single knockout tie."""

    p_home_advance: float
    p_away_advance: float
    # Decomposition (useful for diagnostics / "method of advancement" markets)
    p_home_reg: float   # decided in 90'
    p_away_reg: float
    p_home_et: float     # decided in extra time
    p_away_et: float
    p_home_pens: float   # decided on penalties
    p_away_pens: float

    def as_dict(self) -> dict[str, float]:
        return {
            "p_home_advance": self.p_home_advance,
            "p_away_advance": self.p_away_advance,
            "p_home_reg": self.p_home_reg,
            "p_away_reg": self.p_away_reg,
            "p_home_et": self.p_home_et,
            "p_away_et": self.p_away_et,
            "p_home_pens": self.p_home_pens,
            "p_away_pens": self.p_away_pens,
        }


@dataclass
class KnockoutResolver:
    """Resolve knockout ties: 90' -> extra time -> penalties.

    Parameters
    ----------
    et_scale
        lambda multiplier for the 30' extra-time period. Default 1/3
        (Groll EURO 2024). Use 1/6 only for 10-min periods.
    matrix_size
        Score-matrix dimension used to build the ET matrix (kept = 11 to
        match the spec; ET goals are tiny so this is more than enough).
    """

    et_scale: float = 1.0 / 3.0
    matrix_size: int = 11

    def advancement_prob(
        self,
        score_matrix_90: FloatArray,
        lambda_h: float,
        lambda_a: float,
        *,
        rho: float = 0.0,
        tau_fn: Optional[Callable[[int, int, float, float, float], float]] = None,
        p_home_shootout: float = 0.5,
    ) -> AdvancementResult:
        """Compute advancement probability for one knockout tie.

        Parameters
        ----------
        score_matrix_90
            The 90-minute score matrix (tau-corrected + normalised), exactly
            what your spec section 2.3 already produces.
        lambda_h, lambda_a
            90-minute expected goals; used to build the ET matrix at
            et_scale * lambda.
        rho, tau_fn
            Optional Dixon-Coles dependence and tau callable, applied to the
            ET matrix too for consistency.
        p_home_shootout
            P(home wins the shootout | it reaches penalties). Default 0.5.
            Plug a keeper/penalty model here if you have one.
        """
        ph_reg, p_draw90, pa_reg = split_home_draw_away(score_matrix_90)

        # Extra time matrix, scaled lambdas.
        m_et = build_score_matrix(
            self.et_scale * lambda_h,
            self.et_scale * lambda_a,
            size=self.matrix_size,
            rho=rho,
            tau_fn=tau_fn,
            normalize=True,
        )
        ph_et_cond, p_draw_et_cond, pa_et_cond = split_home_draw_away(m_et)

        # Probabilities of the tie being decided in ET (conditional on 90' draw)
        p_home_et = p_draw90 * ph_et_cond
        p_away_et = p_draw90 * pa_et_cond

        # Probability of reaching penalties = drew 90' AND drew ET
        p_pens = p_draw90 * p_draw_et_cond
        p_home_pens = p_pens * p_home_shootout
        p_away_pens = p_pens * (1.0 - p_home_shootout)

        p_home_adv = ph_reg + p_home_et + p_home_pens
        p_away_adv = pa_reg + p_away_et + p_away_pens

        # Renormalise against tiny floating error (should already sum to 1).
        total = p_home_adv + p_away_adv
        if total > 0:
            p_home_adv /= total
            p_away_adv /= total

        return AdvancementResult(
            p_home_advance=float(p_home_adv),
            p_away_advance=float(p_away_adv),
            p_home_reg=float(ph_reg),
            p_away_reg=float(pa_reg),
            p_home_et=float(p_home_et),
            p_away_et=float(p_away_et),
            p_home_pens=float(p_home_pens),
            p_away_pens=float(p_away_pens),
        )

    def sample_winner(
        self,
        rng: np.random.Generator,
        score_matrix_90: FloatArray,
        lambda_h: float,
        lambda_a: float,
        *,
        rho: float = 0.0,
        tau_fn: Optional[Callable[[int, int, float, float, float], float]] = None,
        p_home_shootout: float = 0.5,
    ) -> int:
        """Sample a single knockout winner: returns +1 (home) or -1 (away).

        For Monte Carlo tournament simulation (module 3) it is cheaper to
        sample one Bernoulli from the analytic advancement prob than to
        sample full scorelines for ET; that is what this does.
        """
        adv = self.advancement_prob(
            score_matrix_90, lambda_h, lambda_a,
            rho=rho, tau_fn=tau_fn, p_home_shootout=p_home_shootout,
        )
        return 1 if rng.random() < adv.p_home_advance else -1


__all__ = [
    "KnockoutResolver",
    "AdvancementResult",
    "build_score_matrix",
    "split_home_draw_away",
]
