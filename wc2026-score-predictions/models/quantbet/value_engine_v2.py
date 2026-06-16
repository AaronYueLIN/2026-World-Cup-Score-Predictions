"""
value_engine_v2.py — Value layer orchestration (replaces original value_engine.py)
==================================================================================

Chains de-vig → EV/edge → staking into a table. Key changes from v1:
  * Default de-vig is Shin method (corrects favourite-longshot bias)
  * edge = model probability - market fair probability (not - raw implied probability)
  * Staking defaults to fractional Kelly, switchable to posterior lower-quantile Kelly
    (pass p_samples)

Composition layer (parlay portfolio) is in portfolio.risk_constrained_kelly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import devig as _devig
from . import staking

__all__ = ["Selection", "evaluate_market", "build_card"]


@dataclass
class Selection:
    name: str
    model_prob: float          # model's true probability (posterior predictive mean)
    offered_odds: float        # bookmaker odds
    market_fair_prob: float    # fair probability of this outcome after de-vig
    ev: float = 0.0
    edge: float = 0.0
    kelly: float = 0.0
    stake_fraction: float = 0.0
    p_samples: Optional[np.ndarray] = field(default=None, repr=False)


def evaluate_market(
    names: Sequence[str],
    model_probs: Sequence[float],
    offered_odds: Sequence[float],
    devig_method: str = "shin",
    kelly_fraction: float = 0.25,
    p_samples: Optional[Sequence[np.ndarray]] = None,
    lcb_quantile: float = 0.25,
) -> List[Selection]:
    """
    Evaluate all outcomes of a mutually exclusive market (e.g. 1X2).

    If p_samples (posterior probability samples per outcome) are provided,
    staking switches to lower_confidence_kelly.
    """
    model_probs = np.asarray(model_probs, dtype=float)
    offered_odds = np.asarray(offered_odds, dtype=float)
    fair = _devig.devig(offered_odds, method=devig_method)

    sels: List[Selection] = []
    for i, nm in enumerate(names):
        p = float(model_probs[i]); o = float(offered_odds[i])
        ev = staking.expected_value(p, o)
        ed = staking.edge(p, o) - (fair[i] - 1.0 / o)  # model vs market fair excess
        ed = float(p - fair[i])                        # more intuitive: how much model exceeds market fair prob
        if p_samples is not None:
            f = staking.lower_confidence_kelly(p_samples[i], o, quantile=lcb_quantile)
            stake = f  # already a shrunk fraction
            kelly_full = staking.kelly_fraction(p, o)
        else:
            kelly_full = staking.kelly_fraction(p, o)
            stake = kelly_fraction * kelly_full
        sels.append(Selection(
            name=nm, model_prob=p, offered_odds=o, market_fair_prob=float(fair[i]),
            ev=ev, edge=ed, kelly=kelly_full, stake_fraction=float(stake),
            p_samples=(np.asarray(p_samples[i]) if p_samples is not None else None),
        ))
    return sels


def build_card(selections: List[Selection], min_edge: float = 0.0, min_ev: float = 0.0):
    """Filter legs worth betting (edge>min_edge and EV>min_ev), sorted by EV descending."""
    picks = [s for s in selections if s.edge > min_edge and s.ev > min_ev]
    return sorted(picks, key=lambda s: s.ev, reverse=True)
