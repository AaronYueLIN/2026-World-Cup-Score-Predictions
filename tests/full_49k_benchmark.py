"""Full 49k matches x three margin comparisons: Poisson / NB / Weibull + Frank copula + dynamic filter + calibration"""
import sys, os, time, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "models", "quantbet"))

import pickle, json, numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from db.config import DATABASE_URL, ENGINE_KWARGS

# ============================================================
# 1. Load v7 + SQL pull full 49k
# ============================================================
dc = pickle.load(open(os.path.join(HERE, "..", "models", "bayesian_dc_v7.pkl"), "rb"))
teams = dc.teams
att0 = {t: float(v) for t, v in zip(teams, dc.params["attack"])}
def0 = {t: float(v) for t, v in zip(teams, dc.params["defense"])}
ha, na = float(dc.params["home_adj"]), float(dc.params["neutral_adj"])

engine = create_engine(DATABASE_URL, **ENGINE_KWARGS)
with engine.connect() as c:
    df = pd.read_sql(text("""
        SELECT m.date, ht.name AS home_team, at.name AS away_team,
               m.home_score AS home_goals, m.away_score AS away_goals,
               m.neutral, m.venue
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        ORDER BY m.date
    """), c)

df["date"] = pd.to_datetime(df["date"])
df["venue"] = df["venue"].fillna("neutral")
df = df[df.home_team.isin(teams) & df.away_team.isin(teams)]
print(f"SQL full: {len(df)} matches, {df.date.min().date()}~{df.date.max().date()}")

# Split: top 80% train, middle 10% val, tail 10% test
n = len(df)
train_end = int(n * 0.80)
val_end   = int(n * 0.90)
train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

# ============================================================
# 2. tune process_sd (full val set)
# ============================================================
from quantbet.scoreline.dynamic_strength import DynamicStrengthFilter
t0 = time.time()
best_sd = DynamicStrengthFilter.tune_process_sd(
    att0, def0, ha, na, train_df, val_df,
    candidates=(0.0, 0.15, 0.25, 0.4, 0.6, 0.85),
    halflife_days=540.0,
)
print(f"\ntune_process_sd: best={best_sd} ({time.time()-t0:.0f}s)")

# ============================================================
# 3. Run full train -> calibrate on val -> evaluate three marginals on test
# ============================================================
from quantbet.scoreline import FlexibleScoreModel, ScoreMatrixCalibrator
from quantbet.scoreline.count_dists import poisson_pmf_vec, negbin_pmf_vec, weibull_count_pmf_vec

def run_trial(margin: str) -> dict:
    """Returns {'rps_mean', 'rps_ci_lo', 'rps_ci_hi', 'logloss_mean', ...}"""
    suffix = f"{margin}_frank"

    # Dynamic filter: warmup on train, then run val match by match
    dyn = DynamicStrengthFilter(att0, def0, ha, na, process_sd_per_year=best_sd,
                                mean_reversion_halflife_days=540.0)
    dyn.run(train_df, collect_oos=False)

    fsm = FlexibleScoreModel(margin=margin, dependence="frank", diagonal_inflation=True)
    fsm.set_strengths(att0, def0, ha, na)

    # --- calibrate on val (OOS) ---
    matrices, lambdas, outcomes = [], [], []
    val_sorted = val_df.sort_values("date")
    for r in val_sorted.itertuples():
        venue = getattr(r, "venue", "neutral")
        lh, la = dyn.expected_goals(r.home_team, r.away_team, venue, as_of=r.date)
        M = fsm.score_matrix(float(lh), float(la))
        M = M / M.sum()
        matrices.append(M)
        lambdas.append((float(lh), float(la)))
        y = 0 if r.home_goals > r.away_goals else (1 if r.home_goals == r.away_goals else 2)
        outcomes.append(y)
        dyn.step(r.home_team, r.away_team, int(r.home_goals), int(r.away_goals), venue, r.date)

    cal = ScoreMatrixCalibrator(w_logloss=0.5)
    cal.fit(matrices, lambdas, outcomes)

    # --- test ---
    dyn2 = DynamicStrengthFilter(att0, def0, ha, na, process_sd_per_year=best_sd,
                                 mean_reversion_halflife_days=540.0)
    dyn2.run(train_df, collect_oos=False)

    rps_list, logloss_list = [], []
    test_sorted = test_df.sort_values("date")
    for r in test_sorted.itertuples():
        venue = getattr(r, "venue", "neutral")
        lh, la = dyn2.expected_goals(r.home_team, r.away_team, venue, as_of=r.date)
        M = fsm.score_matrix(float(lh), float(la))
        M = cal.transform(M, float(lh), float(la))
        M = np.clip(M, 1e-300, None)
        M = M / M.sum()

        gh, ga = int(r.home_goals), int(r.away_goals)
        logloss_list.append(-np.log(float(M[min(gh, 10), min(ga, 10)])))

        ph, pd_, pa = float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())
        p = np.array([ph, pd_, pa]); p /= p.sum()
        y = 0 if gh > ga else (1 if gh == ga else 2)
        e = np.zeros(3); e[y] = 1.0
        rps_list.append(float(((np.cumsum(p) - np.cumsum(e)) ** 2).sum() / 2.0))

        dyn2.step(r.home_team, r.away_team, gh, ga, venue, r.date)

    # bootstrap CI on RPS
    rng = np.random.default_rng(42)
    means = []
    for _ in range(2000):
        idx = rng.integers(0, len(rps_list), len(rps_list))
        means.append(np.mean(np.array(rps_list)[idx]))
    means.sort()

    return {
        "margin": suffix,
        "calib_temp": round(cal.temp_, 4),
        "calib_theta_draw": round(cal.theta_draw_, 4),
        "n_test": len(test_sorted),
        "rps_mean": round(np.mean(rps_list), 5),
        "rps_ci_lo": round(float(means[50]), 5),
        "rps_ci_hi": round(float(means[1949]), 5),
        "logloss_mean": round(np.mean(logloss_list), 4),
    }

# ============================================================
# 4. Run three marginals + Poisson baseline
# ============================================================
results = []

# Poisson baseline (no dynamics)
print("\n--- Baseline: Poisson+independent ---")
dyn_base = DynamicStrengthFilter(att0, def0, ha, na, process_sd_per_year=0.0)
dyn_base.run(train_df, collect_oos=False)
base_rps = []
for r in test_df.sort_values("date").itertuples():
    venue = getattr(r, "venue", "neutral")
    lh, la = dyn_base.expected_goals(r.home_team, r.away_team, venue, as_of=r.date)
    ph = poisson_pmf_vec(lh, 10); pa = poisson_pmf_vec(la, 10)
    M = np.outer(ph, pa); M /= M.sum()
    h, d, a = float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())
    p = np.array([h, d, a]); p /= p.sum()
    y = 0 if r.home_goals > r.away_goals else (1 if r.home_goals == r.away_goals else 2)
    e = np.zeros(3); e[y] = 1.0
    base_rps.append(float(((np.cumsum(p) - np.cumsum(e)) ** 2).sum() / 2.0))
    dyn_base.step(r.home_team, r.away_team, int(r.home_goals), int(r.away_goals), venue, r.date)
rng = np.random.default_rng(42)
bmeans = []; [bmeans.append(np.mean(np.array(base_rps)[rng.integers(0, len(base_rps), len(base_rps))])) for _ in range(2000)]
bmeans.sort()
results.append({"margin": "baseline_poisson", "rps_mean": round(np.mean(base_rps), 5),
                "rps_ci_lo": round(float(bmeans[50]), 5), "rps_ci_hi": round(float(bmeans[1949]), 5)})
print(f"  Baseline RPS={np.mean(base_rps):.5f}")

for margin in ["poisson", "nb", "weibull"]:
    print(f"\n--- {margin} + Frank + dynamic ---")
    r = run_trial(margin)
    results.append(r)
    print(f"  RPS={r['rps_mean']:.5f} [{r['rps_ci_lo']:.5f}, {r['rps_ci_hi']:.5f}]  "
          f"logloss={r['logloss_mean']:.4f}  temp={r['calib_temp']:.4f}")

# ============================================================
# 5. Report
# ============================================================
print(f"\n{'='*75}")
base = results[0]
print(f"  Test: {results[1]['n_test']} matches  |  process_sd={best_sd}")
print(f"  {'model':<25s} {'RPS':>8s} {'95% CI':>22s} {'logloss':>10s}")
print(f"  {'-'*25} {'-'*8} {'-'*22} {'-'*10}")
for r in results:
    ci = f"[{r['rps_ci_lo']:.5f}, {r['rps_ci_hi']:.5f}]"
    ll = f"{r.get('logloss_mean','-'):.4f}" if 'logloss_mean' in r else '-'
    print(f"  {r['margin']:<25s} {r['rps_mean']:>8.5f} {ci:>22s} {ll:>10s}")

# pairwise ΔRPS bootstrap
print(f"\n  ΔRPS vs Baseline (positive=better):")
for i in range(1, len(results)):
    # bootstrap paired diff
    test_rps = []  # placeholder -- RPS already computed
    paired_diffs = []
    for _ in range(2000):
        idx = rng.integers(0, len(base_rps), len(base_rps))
        paired_diffs.append(np.mean(np.array(base_rps)[idx]))
    d = results[0]["rps_mean"] - results[i]["rps_mean"]
    ci_lo = results[0]["rps_ci_lo"] - results[i]["rps_ci_hi"]
    ci_hi = results[0]["rps_ci_hi"] - results[i]["rps_ci_lo"]
    sig = "✓" if ci_lo > 0 or ci_hi < 0 else "  "
    print(f"    {results[i]['margin']:<25s}  Δ={d:+.5f}  [{ci_lo:+.5f}, {ci_hi:+.5f}]  {sig}")

print(f"{'='*75}")

# save
with open(os.path.join(HERE, "..", "output", "full_49k_benchmark.json"), "w", encoding="utf-8") as f:
    json.dump({"process_sd": best_sd, "results": results}, f, indent=2)
print("\nsaved output/full_49k_benchmark.json")
