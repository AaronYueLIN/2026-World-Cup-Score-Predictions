"""
portfolio.py — Risk-constrained Kelly portfolio optimisation (Busseti-Ryu-Boyd 2016)
====================================================================================

Upgrades the "optimal 20-yuan 2-leg/4-leg parlay portfolio" from a heuristic
enumeration to a convex optimisation solution with guaranteed drawdown bounds,
and uses the score matrix to **precisely** handle same-game/shared-leg parlay
correlations.

Model
-----
Variable b ∈ R^m_+ is the bet fraction for each wager (including parlays);
cash = 1 - Σb_i.
Scenario k (joint outcome of all correlated matches) has probability π_k.
Net return of wager i in scenario k is G[k,i] = (o_i - 1) if won, otherwise -1.
Wealth multiplier R_k = 1 + Σ_i b_i G[k,i].

    maximize   Σ_k π_k log R_k                         (expected logarithmic growth, concave)
    s.t.       b ≥ 0,  Σ_i b_i ≤ 1                     (budget)
               Σ_k π_k R_k^{-λ} ≤ 1                    (risk constraint)

Risk constraint guarantees a drawdown bound: P( inf_t W_t ≤ α·W_0 ) ≤ α^λ.
λ→0 degenerates to pure Kelly (most aggressive); larger λ is more conservative.

Convexity: the objective is concave in b; R_k^{-λ} (λ>0) is convex in R_k
and R_k is linear in b → the constraint is convex.
Solved with SciPy SLSQP (consistent with your existing optimiser, no CVXPY needed).

Parlay correlation: each match's correlated scoreline is compressed into
"atoms" — the minimal partition of truth values of all referenced predicates
on that match, with probabilities exactly summed from the score matrix.
The joint scenario = Cartesian product of atoms per match; a bet's win/loss
is determined by the truth of its constituent leg predicates in the
corresponding atom. Same-match multi-leg parlays are therefore handled
jointly and analytically, without an independence approximation.

Reference
---------
Busseti, E., Ryu, E.K. & Boyd, S. (2016) "Risk-Constrained Kelly Gambling",
    The Journal of Investing / arXiv:1603.06183.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize

Predicate = Callable[[int, int], bool]

__all__ = ["Leg", "Bet", "MatchModel", "PortfolioResult", "risk_constrained_kelly"]


@dataclass
class Leg:
    """A leg of a parlay: specifies the match + a predicate on that match."""
    match_id: str
    label: str
    predicate: Predicate


@dataclass
class Bet:
    """A wager (single or parlay). odds is the combined decimal odds offered by the bookmaker."""
    name: str
    legs: List[Leg]
    odds: float


@dataclass
class MatchModel:
    """Model output for one match: score matrix M[h,a]."""
    match_id: str
    score_matrix: np.ndarray


@dataclass
class PortfolioResult:
    stakes: Dict[str, float]                 # optimal fraction per bet
    cash: float                              # cash fraction
    expected_log_growth: float               # expected logarithmic growth rate
    growth_per_bet: Dict[str, float] = field(default_factory=dict)
    n_scenarios: int = 0
    lam: float = 0.0
    drawdown_bound: Callable[[float], float] | None = None  # α -> P(drawdown ≤ α) upper bound


def _match_atoms(M: np.ndarray, preds: List[Predicate]):
    """
    Compress one match into atoms: {truth-tuple: probability}. The truth-tuple is aligned with the preds order.
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    atoms: Dict[tuple, float] = {}
    for h in range(n):
        for a in range(n):
            key = tuple(bool(p(h, a)) for p in preds)
            atoms[key] = atoms.get(key, 0.0) + M[h, a]
    return [(k, v) for k, v in atoms.items() if v > 1e-12]


def _build_scenarios(bets: List[Bet], models: Dict[str, MatchModel]):
    """
    Returns (probs[K], G[K, m]) — joint scenario probabilities and net return matrix for K scenarios.
    """
    # 1) Collect unique predicates referenced per match (de-duplicate by object id, preserve order)
    match_preds: Dict[str, List[Predicate]] = {}
    pred_index: Dict[str, Dict[int, int]] = {}  # match_id -> {id(pred): pos}
    for bet in bets:
        for leg in bet.legs:
            if leg.match_id not in models:
                raise KeyError(f"no MatchModel for match_id={leg.match_id!r}")
            preds = match_preds.setdefault(leg.match_id, [])
            idx = pred_index.setdefault(leg.match_id, {})
            if id(leg.predicate) not in idx:
                idx[id(leg.predicate)] = len(preds)
                preds.append(leg.predicate)

    match_ids = list(match_preds.keys())

    # 2) Atoms per match
    atoms_per_match = [
        _match_atoms(models[mid].score_matrix, match_preds[mid]) for mid in match_ids
    ]
    n_scen = int(np.prod([len(a) for a in atoms_per_match])) if atoms_per_match else 1
    if n_scen > 20000:
        warnings.warn(
            f"{n_scen} joint scenarios — consider fewer correlated matches per portfolio."
        )

    m = len(bets)
    probs: List[float] = []
    G_rows: List[List[float]] = []

    # 3) Cartesian product to enumerate scenarios
    for combo in itertools.product(*atoms_per_match):
        # combo[j] = (truth_tuple, prob) for match_ids[j]
        prob = 1.0
        truth_by_match: Dict[str, tuple] = {}
        for mid, (truth, p) in zip(match_ids, combo):
            prob *= p
            truth_by_match[mid] = truth
        probs.append(prob)

        row = []
        for bet in bets:
            won = True
            for leg in bet.legs:
                pos = pred_index[leg.match_id][id(leg.predicate)]
                if not truth_by_match[leg.match_id][pos]:
                    won = False
                    break
            row.append((bet.odds - 1.0) if won else -1.0)
        G_rows.append(row)

    return np.asarray(probs), np.asarray(G_rows).reshape(-1, m)


def risk_constrained_kelly(
    bets: Sequence[Bet],
    models: Sequence[MatchModel],
    lam: float = 1.0,
    max_total: float = 1.0,
) -> PortfolioResult:
    """
    Solve the risk-constrained Kelly.

    bets     : Candidate wagers (singles + parlays)
    models   : MatchModel for the involved matches
    lam      : Risk aversion λ>0 (larger = more conservative; 0.0 → pure Kelly, no drawdown constraint)
    max_total: Total bet fraction upper limit (default 1.0 = can go all-in)
    """
    bets = list(bets)
    model_map = {mm.match_id: mm for mm in models}
    probs, G = _build_scenarios(bets, model_map)
    K, m = G.shape
    pi = probs

    def R(b):
        return 1.0 + G @ b  # (K,)

    def neg_growth(b):
        r = R(b)
        if np.any(r <= 1e-9):
            return 1e6
        return -float(pi @ np.log(r))

    def neg_growth_grad(b):
        r = R(b)
        r = np.clip(r, 1e-9, None)
        # d/db_j [-Σ π_k log R_k] = -Σ π_k G[k,j]/R_k
        return -(G.T @ (pi / r))

    constraints = []
    # Budget: Σ b ≤ max_total
    constraints.append({
        "type": "ineq",
        "fun": lambda b: max_total - b.sum(),
        "jac": lambda b: -np.ones_like(b),
    })
    # Domain: R_k ≥ ε  (ensures log is defined)
    eps = 1e-6
    constraints.append({
        "type": "ineq",
        "fun": lambda b: R(b) - eps,
        "jac": lambda b: G,
    })
    # Risk constraint: 1 - Σ π_k R_k^{-λ} ≥ 0
    if lam and lam > 0:
        def risk_con(b):
            r = np.clip(R(b), eps, None)
            return 1.0 - float(pi @ (r ** (-lam)))

        def risk_jac(b):
            r = np.clip(R(b), eps, None)
            # d/db_j [1 - Σ π_k R_k^{-λ}] = λ Σ π_k R_k^{-λ-1} G[k,j]
            return lam * (G.T @ (pi * r ** (-lam - 1.0)))

        constraints.append({"type": "ineq", "fun": risk_con, "jac": risk_jac})

    bounds = [(0.0, max_total)] * m
    b0 = np.full(m, min(0.02, max_total / max(m, 1)))

    res = minimize(
        neg_growth, b0, jac=neg_growth_grad, bounds=bounds,
        constraints=constraints, method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    b = np.clip(res.x, 0.0, None)

    stakes = {bet.name: float(bi) for bet, bi in zip(bets, b)}
    # Per-bet marginal contribution (diagnostic): π·log(R_with) - π·log(R_without)
    r_full = np.clip(R(b), 1e-9, None)
    base_growth = float(pi @ np.log(r_full))
    per_bet = {}
    for j, bet in enumerate(bets):
        b_off = b.copy()
        b_off[j] = 0.0
        r_off = np.clip(R(b_off), 1e-9, None)
        per_bet[bet.name] = base_growth - float(pi @ np.log(r_off))

    return PortfolioResult(
        stakes=stakes,
        cash=float(max(0.0, 1.0 - b.sum())),
        expected_log_growth=base_growth,
        growth_per_bet=per_bet,
        n_scenarios=K,
        lam=lam,
        drawdown_bound=(lambda alpha, _l=lam: alpha ** _l) if lam and lam > 0 else None,
    )
