"""Full pipeline integration: DynamicStrengthFilter + NB/Frank copula + ScoreMatrixCalibrator

Injects dynamic filter priors using Bayesian DC v7 MAP parameters, runs Kalman
updates match by match, generates one-step-ahead lambda -> NB+Frank score matrix
-> calibration -> outputs 1X2 probabilities.

Compares out-of-sample RPS against the existing Bayesian DC independent Poisson baseline.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "models", "quantbet"))

import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from db.config import DATABASE_URL, ENGINE_KWARGS

# ============================================================================
# 1. Load v7 MAP params + SQL pull full data (last 5 years to match filter state)
# ============================================================================
dc = pickle.load(open(os.path.join(HERE, "..", "models", "bayesian_dc_v7.pkl"), "rb"))
teams = dc.teams
att0 = {t: float(v) for t, v in zip(teams, dc.params["attack"])}
def0 = {t: float(v) for t, v in zip(teams, dc.params["defense"])}
ha = float(dc.params["home_adj"])
na = float(dc.params["neutral_adj"])

print(f"v7 loaded: {len(teams)} teams")

engine = create_engine(DATABASE_URL, **ENGINE_KWARGS)
with engine.connect() as c:
    df = pd.read_sql(text("""
        SELECT m.date, ht.name AS home_team, at.name AS away_team,
               m.home_score AS home_goals, m.away_score AS away_goals,
               m.neutral, m.venue
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE m.date >= '2021-01-01'
        ORDER BY m.date
    """), c)

df["date"] = pd.to_datetime(df["date"])
df["venue"] = df["venue"].fillna("neutral")
# Only keep teams known to v7
df = df[df.home_team.isin(teams) & df.away_team.isin(teams)]
print(f"SQL pulled: {len(df)} matches, {df.home_team.nunique()} teams, {df.date.min().date()} ~ {df.date.max().date()}")

# ============================================================================
# 2. Time split: train (first 70%) / val (middle 15%) / test (last 15%)
# ============================================================================
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)
train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()
print(f"Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

# ============================================================================
# 3. tune_process_sd (select optimal process noise on val set)
# ============================================================================
from quantbet.scoreline.dynamic_strength import DynamicStrengthFilter
print("tune_process_sd...")
t0 = time.time()
best_sd = DynamicStrengthFilter.tune_process_sd(
    att0, def0, ha, na, train_df, val_df,
    candidates=(0.0, 0.15, 0.25, 0.4, 0.6, 0.85),
    halflife_days=540.0,
)
print(f"  best process_sd={best_sd} (took {time.time()-t0:.0f}s)")

# ============================================================================
# 4. Dynamic filter + NB/Frank collect matrices on val -> calibrate
# ============================================================================
from quantbet.scoreline import FlexibleScoreModel, ScoreMatrixCalibrator

dyn = DynamicStrengthFilter(att0, def0, ha, na, process_sd_per_year=best_sd,
                            mean_reversion_halflife_days=540.0)
dyn.run(train_df, collect_oos=False)

fsm = FlexibleScoreModel(margin="nb", dependence="frank", diagonal_inflation=True)
fsm.set_strengths(att0, def0, ha, na)

val_matrices, val_lambdas, val_outcomes = [], [], []
val_df_sorted = val_df.sort_values("date")
for r in val_df_sorted.itertuples():
    venue = getattr(r, "venue", "neutral")
    lh, la = dyn.expected_goals(r.home_team, r.away_team, venue, as_of=r.date)
    M = fsm.score_matrix(float(lh), float(la))
    M = M / M.sum()
    val_matrices.append(M)
    val_lambdas.append((float(lh), float(la)))
    y = 0 if r.home_goals > r.away_goals else (1 if r.home_goals == r.away_goals else 2)
    val_outcomes.append(y)
    # Update state
    dyn.step(r.home_team, r.away_team, int(r.home_goals), int(r.away_goals), venue, r.date)

cal = ScoreMatrixCalibrator(w_logloss=0.5)
cal.fit(val_matrices, val_lambdas, val_outcomes)
print(f"Calibration: temp={cal.temp_:.3f} theta_draw={cal.theta_draw_:.4f}")

# ============================================================================
# 5. Test: Compare two pipelines
# ============================================================================
from quantbet.scoreline.count_dists import poisson_pmf_vec as pois_vec

# Reset filter (restart from end of train set)
dyn2 = DynamicStrengthFilter(att0, def0, ha, na, process_sd_per_year=best_sd,
                             mean_reversion_halflife_days=540.0)
dyn2.run(train_df, collect_oos=False)

baseline_rps, proposed_rps = [], []
test_df_sorted = test_df.sort_values("date")

for r in test_df_sorted.itertuples():
    venue = getattr(r, "venue", "neutral")
    lh, la = dyn2.expected_goals(r.home_team, r.away_team, venue, as_of=r.date)
    y = 0 if r.home_goals > r.away_goals else (1 if r.home_goals == r.away_goals else 2)

    # -- Baseline: independent Poisson (simulates existing DC behavior) --
    ph_p = pois_vec(lh, 10); pa_p = pois_vec(la, 10)
    Mp = np.outer(ph_p, pa_p); Mp /= Mp.sum()
    hp = float(np.tril(Mp, -1).sum()); dp = float(np.trace(Mp)); ap = float(np.triu(Mp, 1).sum())
    bp = np.array([hp, dp, ap]); bp /= bp.sum()

    # -- Proposed: NB+Frank+dynamic+calibration --
    Mf = fsm.score_matrix(float(lh), float(la))
    Mf = cal.transform(Mf, float(lh), float(la))
    Mf /= Mf.sum()
    hf = float(np.tril(Mf, -1).sum()); df_ = float(np.trace(Mf)); af = float(np.triu(Mf, 1).sum())
    bp2 = np.array([hf, df_, af]); bp2 /= bp2.sum()

    e = np.zeros(3); e[y] = 1.0
    baseline_rps.append(float(((np.cumsum(bp) - np.cumsum(e)) ** 2).sum() / 2.0))
    proposed_rps.append(float(((np.cumsum(bp2) - np.cumsum(e)) ** 2).sum() / 2.0))

    dyn2.step(r.home_team, r.away_team, int(r.home_goals), int(r.away_goals), venue, r.date)

# ============================================================================
# 6. Report
# ============================================================================
b_mean = np.mean(baseline_rps); p_mean = np.mean(proposed_rps)
diff = b_mean - p_mean  # positive = proposed is better

# Bootstrap CI on ΔRPS
rng = np.random.default_rng(42)
diffs = []
for _ in range(2000):
    idx = rng.integers(0, len(baseline_rps), len(baseline_rps))
    diffs.append(np.mean(np.array(baseline_rps)[idx] - np.array(proposed_rps)[idx]))
diffs = np.sort(diffs)
lo, hi = float(diffs[50]), float(diffs[1949])

print(f"\n{'='*70}")
print(f"  Test set: {len(test_df_sorted)} matches")
print(f"  Baseline (independent Poisson+static):    RPS = {b_mean:.4f}")
print(f"  Proposed (dynamic+NB+Frank+calibration): RPS = {p_mean:.4f}")
print(f"  ΔRPS = {diff:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
print(f"  {'SIGNIFICANT (0 not in CI)' if lo > 0 or hi < 0 else 'not significant'}")
print(f"{'='*70}")
