"""
devig.py — Infer "fair" probabilities from betting odds (de-vigging / overround removal)
========================================================================================

Implements three methods with Shin as the recommended default:

  * proportional  : p_i = (1/o_i) / Σ(1/o_j)            —— baseline, has favorite-longshot bias
  * power         : p_i ∝ (1/o_i)^k, solve k so Σp=1    —— empirical correction
  * shin          : Shin (1992,1993) insider-trading pricing model  —— recommended, best-calibrated

Shin inversion formula (as used in Štrumbelj 2014):
  Let π_i = 1/o_i, B = Σ_j π_j (overround, >1),
      p_i(z) = ( sqrt( z^2 + 4(1-z) π_i^2 / B ) - z ) / ( 2(1-z) )
  where z ∈ [0,1) is solved numerically via Σ_i p_i(z) = 1.
  z is an observable estimate of "the degree of insider/smart-money penetration" of the market.

References
----------
Shin, H.S. (1992) "Prices of state-contingent claims with insider traders,
    and the favourite-longshot bias", The Economic Journal.
Shin, H.S. (1993) "Measuring the incidence of insider trading in a market
    for state-contingent claims", The Economic Journal 103, 1141-1153.
Štrumbelj, E. (2014) "On determining probability forecasts from betting
    odds", International Journal of Forecasting 30(4), 934-943.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

__all__ = ["deviq_proportional", "deviq_power", "deviq_shin"]

__all__ = [
    "implied_probabilities",
    "overround",
    "devig_proportional",
    "devig_power",
    "devig_shin",
    "devig",
]


def _as_odds(odds: npt.ArrayLike) -> npt.NDArray[np.float64]:
    o = np.asarray(odds, dtype=float)
    if np.any(o <= 1.0):
        raise ValueError("decimal odds must all be > 1.0")
    return o


def implied_probabilities(odds: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Raw implied probabilities 1/o_i (unnormalised, includes overround)."""
    return 1.0 / _as_odds(odds)


def overround(odds: npt.ArrayLike) -> float:
    """Booksum / overround B = Σ(1/o_i). B-1 is the bookmaker's gross margin (vig)."""
    return float(np.sum(1.0 / _as_odds(odds)))


def devig_proportional(odds: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Proportional method (basic / multiplicative). Baseline, known to overestimate underdogs and underestimate favourites."""
    inv = 1.0 / _as_odds(odds)
    return inv / inv.sum()


def devig_power(odds: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Power method: p_i ∝ (1/o_i)^k, solve k so Σp = 1."""
    inv = 1.0 / _as_odds(odds)

    def f(k):
        return np.sum(inv ** k) - 1.0

    # f(small k)>0 (each term→1, sum>1); f(large k)<0 (each inv<1, →0)
    k = brentq(f, 1e-6, 200.0, xtol=1e-12)
    p = inv ** k
    return p / p.sum()


def devig_shin(
    odds: npt.ArrayLike,
    return_z: bool = False,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], float]:
    """
    Shin (1992/1993) method. Returns fair probability vector; if return_z=True also returns the insider proportion z.
    """
    inv = 1.0 / _as_odds(odds)
    B = inv.sum()

    def p_of_z(z):
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * inv * inv / B) - z) / (2.0 * (1.0 - z))

    def f(z):
        return p_of_z(z).sum() - 1.0

    try:
        # f(0+) = sqrt(B) - 1 > 0 ; f(1-) = Σinv^2/B - 1 (usually < 0)
        z = brentq(f, 1e-12, 1.0 - 1e-9, xtol=1e-12)
        p = p_of_z(z)
        p = p / p.sum()  # numerical cleanup
    except ValueError:
        # Degenerate case (extreme odds) → fall back to proportional method
        z, p = 0.0, devig_proportional(odds)
    if return_z:
        return p, float(z)
    return p


def devig(odds: npt.ArrayLike, method: str = "shin") -> npt.NDArray[np.float64]:
    """Unified entry point. method ∈ {'shin','proportional','power'}."""
    m = method.lower()
    if m == "shin":
        return devig_shin(odds)
    if m in ("proportional", "basic", "multiplicative"):
        return devig_proportional(odds)
    if m == "power":
        return devig_power(odds)
    raise ValueError(f"unknown method: {method}")
