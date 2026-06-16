"""
ab_static_vs_dynamic.py — Step-zero A/B: static MAP strengths vs dynamic filter
================================================================================

GOAL
----
Quantify whether time-varying (adaptive) team strengths beat static MAP
strengths *on match-result accuracy*, holding everything else fixed.

THE ONE THING THAT MATTERS: both arms use the SAME lambda -> 1X2 mapping
(plain Poisson outer product). The ONLY difference between arms is where
(lambda_h, lambda_a) come from:
    - STATIC   : one train-only Poisson MLE fit, frozen over the test window.
    - DYNAMIC  : DynamicStrengthFilter, leakage-free one-step-ahead lambdas.
This isolates strength estimation from margin/copula choice (which the 49k
benchmark already showed does not move RPS).

NO LEAKAGE
----------
- Static fit uses train only.
- process_sd is tuned on a validation slice carved from the TAIL of train.
- Dynamic test lambdas are the filter's *pre-observation* one-step-ahead
  predictions (DynamicStrengthFilter.run(collect_oos=True)).

VERDICT
-------
Paired bootstrap on per-match RPS (compare_models_ci(dyn, static)):
diff = mean(RPS_dyn - RPS_static). diff < 0 means dynamic is better.
    hi < 0   -> dynamic SIGNIFICANTLY better
    lo > 0   -> static  SIGNIFICANTLY better
    else     -> indistinguishable (do NOT invest in GAS/VB-SSM yet)

Run:
    python ab_static_vs_dynamic.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

# ---- repo wiring (same convention as tests/full_49k_benchmark.py) -----------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "models"))

from quantbet.scoreline.dynamic_strength import DynamicStrengthFilter
from quantbet.scoreline.score_model import FlexibleScoreModel
from quantbet.scoreline.count_dists import poisson_pmf_vec
from quantbet.evaluation import rps_score, bootstrap_ci, compare_models_ci

# ============================== CONFIG =======================================
# TODO(agent): point DATA_PATH at the full 49k historical results table
# (same file tests/full_49k_benchmark.py loads). Must have columns:
#   date, home_team, away_team, home_goals, away_goals  [, venue]
DATA_PATH = os.path.join(ROOT, "data", "wc_2026_kaggle_clean.csv")

TEST_FRAC = 0.15          # newest fraction held out for evaluation
VAL_FRAC_OF_TRAIN = 0.15  # tail of train used ONLY to tune process_sd
KMAX = 10
HALFLIFE_DAYS = 540.0
PROCESS_SD_CANDIDATES = (0.0, 0.15, 0.25, 0.40, 0.60, 0.85)  # 0.0 == static
OUT_JSON = os.path.join(ROOT, "output", "ab_static_vs_dynamic.json")
# =============================================================================


def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset missing columns: {missing}. Got: {list(df.columns)}")
    df = df.dropna(subset=list(need)).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "venue" not in df.columns:
        df["venue"] = "neutral"
    return df


def poisson_1x2(lh: float, la: float, kmax: int = KMAX) -> np.ndarray:
    """SHARED mapping for BOTH arms. p = [home, draw, away]."""
    ph = poisson_pmf_vec(lh, kmax)
    pa = poisson_pmf_vec(la, kmax)
    M = np.outer(ph, pa)
    h = float(np.tril(M, -1).sum())
    d = float(np.trace(M))
    a = float(np.triu(M, 1).sum())
    p = np.array([h, d, a], dtype=float)
    return p / p.sum()


def outcome_label(gh: int, ga: int) -> int:
    return 0 if gh > ga else (1 if gh == ga else 2)


def venue_for_predict(v: str) -> str:
    return v if v in ("home", "neutral") else "neutral"


def main() -> None:
    df = load_matches(DATA_PATH)
    n = len(df)
    cut = int(n * (1 - TEST_FRAC))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    print(f"[data] total={n}  train={len(train)}  test={len(test)}  "
          f"split_date={train['date'].max().date()}")

    # ---- STATIC: train-only Poisson MLE, frozen ----------------------------
    static = FlexibleScoreModel(margin="poisson", dependence="none",
                                diagonal_inflation=False, damping=0.0)
    static.fit(train)
    known = set(static.teams)

    # restrict test to matches both teams seen in train (so static is defined)
    mask = test["home_team"].isin(known) & test["away_team"].isin(known)
    dropped = int((~mask).sum())
    test = test[mask].reset_index(drop=True)
    print(f"[data] test usable={len(test)}  dropped_unknown_team={dropped}")

    # ---- tune process_sd on a TAIL slice of train (no test leakage) --------
    tcut = int(len(train) * (1 - VAL_FRAC_OF_TRAIN))
    train_inner, val_inner = train.iloc[:tcut], train.iloc[tcut:]
    attack0 = dict(zip(static.teams, static.params["attack"]))
    defense0 = dict(zip(static.teams, static.params["defense"]))
    home_adj = static.params["home_adj"]
    neutral_adj = static.params["neutral_adj"]

    best_sd = DynamicStrengthFilter.tune_process_sd(
        attack0, defense0, home_adj, neutral_adj,
        train_inner, val_inner,
        candidates=PROCESS_SD_CANDIDATES, halflife_days=HALFLIFE_DAYS,
    )
    print(f"[dynamic] tuned process_sd_per_year = {best_sd}  "
          f"(0.0 would mean 'static wins on validation')")

    # ---- DYNAMIC: warm up on FULL train, then leakage-free OOS on test -----
    flt = DynamicStrengthFilter(
        attack0, defense0, home_adj, neutral_adj,
        process_sd_per_year=best_sd, mean_reversion_halflife_days=HALFLIFE_DAYS,
    )
    flt.run(train, collect_oos=False)               # warm-up only
    oos = flt.run(test, collect_oos=True)           # one-step-ahead lambdas
    oos = oos.reset_index(drop=True)

    # ---- score both arms through the SAME mapping --------------------------
    rps_static, rps_dyn = [], []
    ll_static, ll_dyn = [], []
    acc_static, acc_dyn = [], []
    for i, r in test.iterrows():
        y = outcome_label(int(r["home_goals"]), int(r["away_goals"]))
        v = venue_for_predict(str(r["venue"]))

        sp = static.predict(r["home_team"], r["away_team"], venue=v)
        p_s = poisson_1x2(sp["expected_home_goals"], sp["expected_away_goals"])

        p_d = poisson_1x2(float(oos.loc[i, "pred_lambda_h"]),
                          float(oos.loc[i, "pred_lambda_a"]))

        rps_static.append(rps_score(p_s, y)); rps_dyn.append(rps_score(p_d, y))
        ll_static.append(-np.log(max(p_s[y], 1e-12)))
        ll_dyn.append(-np.log(max(p_d[y], 1e-12)))
        acc_static.append(int(np.argmax(p_s) == y))
        acc_dyn.append(int(np.argmax(p_d) == y))

    # ---- bootstrap CIs + paired comparison ---------------------------------
    s_rps, s_lo, s_hi = bootstrap_ci(rps_static)
    d_rps, d_lo, d_hi = bootstrap_ci(rps_dyn)
    diff, dlo, dhi = compare_models_ci(rps_dyn, rps_static)  # dyn - static

    if dhi < 0:
        verdict = "DYNAMIC significantly better -> proceed to GAS / VB-SSM"
    elif dlo > 0:
        verdict = "STATIC significantly better -> drop the dynamic track"
    else:
        verdict = "INDISTINGUISHABLE -> do NOT invest in GAS/VB-SSM yet"

    report = {
        "n_test": len(test), "dropped_unknown_team": dropped,
        "tuned_process_sd": best_sd,
        "static": {"rps": round(s_rps, 5), "rps_ci": [round(s_lo, 5), round(s_hi, 5)],
                   "logloss": round(float(np.mean(ll_static)), 4),
                   "acc": round(float(np.mean(acc_static)), 4)},
        "dynamic": {"rps": round(d_rps, 5), "rps_ci": [round(d_lo, 5), round(d_hi, 5)],
                    "logloss": round(float(np.mean(ll_dyn)), 4),
                    "acc": round(float(np.mean(acc_dyn)), 4)},
        "paired_diff_dyn_minus_static": {"mean": round(diff, 5),
                                         "ci": [round(dlo, 5), round(dhi, 5)]},
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print(f"  STATIC   RPS={s_rps:.5f}  CI[{s_lo:.5f},{s_hi:.5f}]  "
          f"acc={np.mean(acc_static):.3f}")
    print(f"  DYNAMIC  RPS={d_rps:.5f}  CI[{d_lo:.5f},{d_hi:.5f}]  "
          f"acc={np.mean(acc_dyn):.3f}")
    print(f"  PAIRED   dyn-static = {diff:+.5f}  CI[{dlo:+.5f}, {dhi:+.5f}]")
    print(f"  VERDICT  {verdict}")
    print("=" * 64)
    print(f"  saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
