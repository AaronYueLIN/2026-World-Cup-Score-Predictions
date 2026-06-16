"""
fit_scoreline_shape.py -- Fit v9's NB+Frank scoreline shape parameters (r, kappa, theta_draw)
========================================================================

Why
------
registry.describe() shows v9's `_scoreline_model.shape_` is
    {nb_r: 8.0, wc_c: 1.0, dep: 0.0, theta_draw: 0.0}
all defaults -- NB+Frank engine is hooked up, but **dependence kappa and diagonal inflation theta were never fitted on data**
(kappa=0 degenerates to independent NB, theta=0 means no draw inflation). The engine is idling.

This script takes **given DC lambda_h, lambda_a** as fixed (does not touch attack/Elo priors), uses earlier
data to fit the 3 global scalars (r, kappa, theta_draw), then evaluates their impact on
**derived markets** (exact score / over-under 2.5 / BTTS / 1X2) on a later held-out set,
and finally writes the fitted shape_ back to the model and saves a version.

Key: only fit 3 global scalars; lambda still comes from DC (with Elo prior). fit/eval time split isolates
the shape contribution -- even though DC has seen this data, the relative comparison "fitted shape vs default
shape" is still clean because both use the same DC lambda, differing only in shape.

Usage
-----
    python models/fit_scoreline_shape.py --data /path/to/results.csv --write
data must include: date, home_team, away_team, home_goals, away_goals [, venue]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
# project root quantbet_ev/ (containing db/, models/)
_PARENT = os.path.dirname(HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_MODELS = os.path.join(_PARENT, "models")
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

import registry  # noqa: E402


# ----------------------------------------------------- Derived market metrics
def _ou25(M: np.ndarray) -> float:
    """P(total goals > 2.5)."""
    n = M.shape[0]
    idx = np.add.outer(np.arange(n), np.arange(n))
    return float(M[idx >= 3].sum())


def _btts(M: np.ndarray) -> float:
    """P(both teams to score)."""
    return float(M[1:, 1:].sum())


def _outcome(M: np.ndarray):
    return (float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum()))


def _rps3(p, y) -> float:
    e = np.zeros(3); e[y] = 1.0
    cp = np.cumsum(p); ce = np.cumsum(e)
    return float(np.sum((cp - ce) ** 2) / 2.0)


def _evaluate(fsm, eval_df, shape: dict) -> dict:
    """Evaluate metrics with given shape on eval set (lower is better, except acc)."""
    K = fsm.max_goals
    hi = eval_df["home_team"].map(fsm.team_idx).values
    ai = eval_df["away_team"].map(fsm.team_idx).values
    venue = fsm._encode_venue(eval_df)
    lh_all, la_all = fsm._lambdas_for(hi, ai, venue)
    gh = eval_df["home_goals"].values.astype(int)
    ga = eval_df["away_goals"].values.astype(int)

    ll_exact = ll_ou = ll_btts = rps = 0.0
    n = len(eval_df)
    cache = {}
    for i in range(n):
        key = (round(lh_all[i], 2), round(la_all[i], 2))
        M = cache.get(key)
        if M is None:
            M = fsm._build_matrix(lh_all[i], la_all[i], shape)
            cache[key] = M
        x, y = min(gh[i], K), min(ga[i], K)
        ll_exact += -np.log(max(M[x, y], 1e-12))
        # Over-under 2.5
        p_over = _ou25(M)
        over = 1 if (gh[i] + ga[i]) > 2 else 0
        ll_ou += -np.log(max(p_over if over else 1 - p_over, 1e-12))
        # BTTS
        p_btts = _btts(M)
        btts = 1 if (gh[i] >= 1 and ga[i] >= 1) else 0
        ll_btts += -np.log(max(p_btts if btts else 1 - p_btts, 1e-12))
        # 1X2 RPS
        h, d, a = _outcome(M)
        s = h + d + a
        p = np.array([h, d, a]) / (s if s > 0 else 1.0)
        yo = 0 if gh[i] > ga[i] else (1 if gh[i] == ga[i] else 2)
        rps += _rps3(p, yo)
    return {
        "exact_logloss": ll_exact / n,
        "ou25_logloss": ll_ou / n,
        "btts_logloss": ll_btts / n,
        "rps_1x2": rps / n,
        "n": n,
    }


# ----------------------------------------------------- Fit shape
def fit_shape(dc, fit_df: pd.DataFrame, max_fit_rows: int = 8000, recency_damping: float = 0.0):
    """Fit dc._scoreline_model's (r, kappa, theta_draw) on fit_df. Modifies shape_ in place."""
    if not hasattr(dc, "_scoreline_model") or dc._scoreline_model is None:
        from bayesian_dixon_coles import install_scoreline  # type: ignore
        install_scoreline(dc)
    fsm = dc._scoreline_model
    if getattr(fsm, "team_idx", None) is None:
        fsm.set_strengths(
            {t: float(v) for t, v in zip(dc.teams, dc.params["attack"])},
            {t: float(v) for t, v in zip(dc.teams, dc.params["defense"])},
            float(dc.params["home_adj"]), float(dc.params["neutral_adj"]),
        )
    fsm.damping = recency_damping

    df = fit_df[fit_df["home_team"].isin(fsm.team_idx) & fit_df["away_team"].isin(fsm.team_idx)].copy()
    if max_fit_rows and len(df) > max_fit_rows:
        df = df.sort_values("date").tail(max_fit_rows)   # global scalars, latest N matches is sufficient and faster

    hi = df["home_team"].map(fsm.team_idx).values
    ai = df["away_team"].map(fsm.team_idx).values
    gh = df["home_goals"].values.astype(float)
    ga = df["away_goals"].values.astype(float)
    venue = fsm._encode_venue(df)
    w = fsm._weights(df)

    fsm._fit_shape(df, hi, ai, gh, ga, venue, w)
    return dict(fsm.shape_)


# ----------------------------------------------------- Data loading
def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Data missing columns {miss}; actual columns: {list(df.columns)}")
    df = df.dropna(subset=list(need)).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "venue" not in df.columns:
        df["venue"] = "neutral"
    return df


def load_matches_sql() -> pd.DataFrame:
    """Fetch all matches from SQL for shape fitting."""
    from sqlalchemy import create_engine, text
    from db.config import DATABASE_URL, ENGINE_KWARGS
    engine = create_engine(DATABASE_URL, **ENGINE_KWARGS)
    q = """
        SELECT m.date, ht.name AS home_team, at.name AS away_team,
               m.home_score AS home_goals, m.away_score AS away_goals,
               m.venue
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE m.home_score IS NOT NULL AND m.away_score IS NOT NULL
        ORDER BY m.date
    """
    df = pd.read_sql(q, engine)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["venue"] = df["venue"].fillna("neutral")
    return df


def run(df: pd.DataFrame, eval_frac: float = 0.2, max_fit_rows: int = 8000,
        write: bool = False, out_name: str = "bayesian_dc_v9_shapefit.pkl"):
    dc = registry.load_model()
    fsm = dc._scoreline_model
    shape_default = dict(fsm.shape_)

    n = len(df); cut = int(n * (1 - eval_frac))
    fit_df, eval_df = df.iloc[:cut], df.iloc[cut:]
    eval_df = eval_df[eval_df["home_team"].isin(fsm.team_idx) & eval_df["away_team"].isin(fsm.team_idx)].copy()

    print(f"[data] fit={len(fit_df)} eval={len(eval_df)} (split={fit_df['date'].max().date()})")
    print(f"[before] shape_ = {shape_default}")

    before = _evaluate(fsm, eval_df, shape_default)
    shape_fitted = fit_shape(dc, fit_df, max_fit_rows=max_fit_rows)
    print(f"[after ] shape_ = { {k: round(v,4) for k,v in shape_fitted.items()} }")
    after = _evaluate(fsm, eval_df, shape_fitted)

    print("\n  metric           default      fitted      Δ")
    for k in ["exact_logloss", "ou25_logloss", "btts_logloss", "rps_1x2"]:
        d, a = before[k], after[k]
        print(f"  {k:<15} {d:>9.5f}  {a:>9.5f}  {a-d:>+8.5f}  {'improved' if a < d else ''}")

    if write:
        out_path = os.path.join(HERE, out_name)
        with open(out_path, "wb") as f:
            pickle.dump(dc, f)
        print(f"\n  Written back -> {out_path}")
        print(f"  Next: add an entry in registry.REGISTRY pointing to {out_name}, and switch MODEL_VERSION over.")
    return shape_fitted, before, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="results.csv (date,home_team,away_team,home_goals,away_goals[,venue])")
    ap.add_argument("--sql", action="store_true", help="Pull data from SQL database (replaces --data)")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--max-fit-rows", type=int, default=8000)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.sql:
        df = load_matches_sql()
    elif a.data:
        df = load_matches(a.data)
    else:
        ap.error("Provide --data path or --sql")
    run(df, a.eval_frac, a.max_fit_rows, a.write)


if __name__ == "__main__":
    main()
