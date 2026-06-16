"""
trps.py — Tournament Rank Probability Score (World Cup module 4).

WHY (World Cup specific)
------------------------
Your spec section 5.1 uses the ordinary RPS to score a single 1X2 match.
That is fine per-match, but the World Cup is a ONE-SHOT tournament: you
make a pre-tournament forecast of where every team will finish, the event
happens once, and you need a proper score for the WHOLE forecast. RPS does
not generalise to that. The Tournament Rank Probability Score (TRPS) does:
it scores a predicted distribution over finishing-rank intervals against
the realised finishing rank, summing squared cumulative differences across
ranks (a multi-rank generalisation of RPS).

For 2026 (48 teams) the natural rank intervals double each round, with
relative weights 32, 16, 8, 4, 2, 1 from "group exit" up to "winner"
(Ekstrom et al. 2021). Lower TRPS = better.

Literature
----------
- Ekstrom, Van Eetvelde, Ley & Brefeld (2021), Journal of Sports Analytics,
  DOI:10.3233/JSA-200454. R package `socceR` on CRAN.

HOW TO WIRE INTO QuantBet-EV v7.0
---------------------------------
Take the per-team stage probabilities from the tournament simulator
(module 3, TournamentProbabilities) and convert them to a per-team
distribution over rank buckets, then call `trps`. After the tournament,
pass each team's realised bucket to score your forecast. Use this instead
of (or alongside) RPS for tournament-level model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# Default 2026 rank buckets, finest (best) first. Each maps to the deepest
# stage a team reached. Weights double per bucket per Ekstrom et al.
WC2026_BUCKETS: tuple[str, ...] = (
    "Winner",     # champion
    "Final",      # runner-up (lost final)
    "SF",         # lost semi-final (3rd/4th)
    "QF",         # lost quarter-final
    "R16",        # lost round of 16
    "R32",        # lost round of 32
    "group",      # eliminated in group stage
)
WC2026_WEIGHTS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def trps(
    pred_probs: Sequence[float] | FloatArray,
    realised_rank: int,
    *,
    weights: Sequence[float] | None = None,
) -> float:
    """Tournament Rank Probability Score for ONE team's forecast.

    Parameters
    ----------
    pred_probs
        Predicted probability of finishing in each rank bucket, ordered
        best-first (index 0 = best possible finish). Must sum to ~1.
    realised_rank
        Index (0-based, into the same bucket ordering) of the bucket the
        team actually finished in.
    weights
        Optional per-rank weights (best-first). If given, the cumulative
        differences are weighted (Ekstrom's weighted TRPS / wTRPS). If
        None, an unweighted TRPS is returned.

    Returns
    -------
    float TRPS for this team (0 = perfect, higher = worse).
    """
    p = np.asarray(pred_probs, dtype=np.float64).ravel()
    R = p.size
    if not (0 <= realised_rank < R):
        raise ValueError(f"realised_rank {realised_rank} out of range [0,{R})")
    s = p.sum()
    if s <= 0:
        raise ValueError("pred_probs sum to zero")
    p = p / s

    # Cumulative forecast vs realised one-hot, summed squared differences.
    cum_p = np.cumsum(p)
    cum_a = np.cumsum(_onehot(realised_rank, R))
    diff2 = (cum_p - cum_a) ** 2

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != R:
            raise ValueError("weights length must match pred_probs")
        # normalise weights to sum to (R-1) following Ekstrom's scaling
        w = w / w.sum() * (R - 1)
        diff2 = diff2 * w

    return float(diff2.sum() / (R - 1))


def _onehot(idx: int, n: int) -> FloatArray:
    v = np.zeros(n, dtype=np.float64)
    v[idx] = 1.0
    return v


@dataclass
class TRPSEvaluator:
    """Compute TRPS across all teams from simulator stage probabilities.

    Parameters
    ----------
    buckets
        Rank buckets best-first. Default WC2026_BUCKETS.
    weights
        Optional per-bucket weights for wTRPS. Default WC2026_WEIGHTS.
    """

    buckets: tuple[str, ...] = WC2026_BUCKETS
    weights: tuple[float, ...] | None = WC2026_WEIGHTS

    def stage_probs_to_bucket_probs(
        self, team_stage_probs: Mapping[str, float]
    ) -> FloatArray:
        """Convert cumulative 'reached at least stage X' probs to bucket probs.

        The simulator gives P(reach >= stage). The probability a team's
        DEEPEST stage is exactly bucket b is:
            P(reach b) - P(reach next-deeper bucket).
        Buckets are best-first, so 'next deeper' is index-1.
        """
        # team_stage_probs uses simulator stage names: group,R32,R16,QF,SF,Final,Winner
        # Our buckets are best-first: Winner,Final,SF,QF,R16,R32,group
        reach = team_stage_probs
        out = np.zeros(len(self.buckets), dtype=np.float64)
        for i, b in enumerate(self.buckets):
            p_reach_b = reach.get(b, 0.0)
            if i == 0:
                # Winner: deepest possible; exact = P(reach Winner)
                out[i] = p_reach_b
            else:
                deeper = reach.get(self.buckets[i - 1], 0.0)
                out[i] = max(p_reach_b - deeper, 0.0)
        total = out.sum()
        if total > 0:
            out = out / total
        return out

    def evaluate(
        self,
        sim_probs: Mapping[str, Mapping[str, float]],
        realised: Mapping[str, str],
    ) -> dict[str, float]:
        """Score every team's forecast against realised finishing buckets.

        Parameters
        ----------
        sim_probs
            {team: {stage: P(reach >= stage)}} from TournamentProbabilities.probs
        realised
            {team: bucket_name} actual deepest stage per team (post-tournament).

        Returns
        -------
        dict with per-team TRPS plus a 'mean' key (mean over all teams).
        """
        bucket_index = {b: i for i, b in enumerate(self.buckets)}
        scores: dict[str, float] = {}
        for team, stage_probs in sim_probs.items():
            if team not in realised:
                continue
            bp = self.stage_probs_to_bucket_probs(stage_probs)
            r = bucket_index[realised[team]]
            scores[team] = trps(bp, r, weights=self.weights)
        if scores:
            scores["mean"] = float(np.mean([v for k, v in scores.items()
                                            if k != "mean"]))
        return scores


__all__ = ["trps", "TRPSEvaluator", "WC2026_BUCKETS", "WC2026_WEIGHTS"]
