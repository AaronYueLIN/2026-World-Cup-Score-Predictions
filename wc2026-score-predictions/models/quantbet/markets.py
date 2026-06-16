"""
markets.py — Compute exact market / joint-market probabilities from the score matrix
=====================================================================================

Core idea: your DC model outputs an 11x11 joint score matrix M. Any market
is a sum over a subset of (h,a) cells; the joint probability of any
"same-match multi-leg parlay" is the sum over cells where all leg predicates
**simultaneously** hold — exact, no independence assumption. This eliminates
the systematic EV inflation from multiplying marginal probabilities for
same-game parlays.

Each market selection is represented as a predicate: (h:int, a:int) -> bool.
joint_prob(M, p1, p2, ...) returns the exact probability of all predicates
holding simultaneously.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Predicate = Callable[[int, int], bool]

__all__ = [
    "Predicate",
    "joint_prob",
    "one_x_two",
    "home_win", "draw", "away_win",
    "over", "under", "btts",
    "double_chance", "exact_score", "team_total_over",
]


def joint_prob(M: np.ndarray, *predicates: Predicate) -> float:
    """Exact probability of ∧ all predicates holding simultaneously (sum of matrix cells)."""
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    total = 0.0
    for h in range(n):
        for a in range(n):
            if all(p(h, a) for p in predicates):
                total += M[h, a]
    return float(total)


def one_x_two(M: np.ndarray):
    """Returns (p_home, p_draw, p_away), ordered consistently with RPS ordinality."""
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    idx = np.arange(n)
    lower = np.tril(np.ones((n, n)), -1)  # h>a
    upper = np.triu(np.ones((n, n)), 1)   # h<a
    p_home = float((M * lower).sum())
    p_draw = float(np.trace(M))
    p_away = float((M * upper).sum())
    s = p_home + p_draw + p_away
    return p_home / s, p_draw / s, p_away / s


# ----- Predicate factories ----------------------------------------------------
def home_win() -> Predicate:
    return lambda h, a: h > a


def draw() -> Predicate:
    return lambda h, a: h == a


def away_win() -> Predicate:
    return lambda h, a: h < a


def over(line: float) -> Predicate:
    """Total goals greater than line (e.g. over(2.5) → h+a ≥ 3)."""
    return lambda h, a: (h + a) > line


def under(line: float) -> Predicate:
    return lambda h, a: (h + a) < line


def btts(yes: bool = True) -> Predicate:
    """Both teams to score."""
    return lambda h, a: (h >= 1 and a >= 1) == yes


def double_chance(outcomes: str) -> Predicate:
    """Double chance: '1X' (home or draw), 'X2' (draw or away), '12' (no draw)."""
    o = outcomes.upper()

    def pred(h, a):
        res = "1" if h > a else ("X" if h == a else "2")
        return res in o

    return pred


def exact_score(h_goals: int, a_goals: int) -> Predicate:
    return lambda h, a: h == h_goals and a == a_goals


def team_total_over(side: str, line: float) -> Predicate:
    """Single team goals greater than line. side ∈ {'home','away'}."""
    if side == "home":
        return lambda h, a: h > line
    return lambda h, a: a > line
