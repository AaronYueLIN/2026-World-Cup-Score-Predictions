"""
benchmark_scoreline.py -- Out-of-sample backtest of scoreline upgrade vs baseline (independent Poisson)
====================================================================

Synthetic near-real DGP: attack/defense strengths drift via random walk + overdispersed goals (negative binomial) + weak negative dependence.
Time-based train/val/test split. **All hyperparameters selected on val** (no leakage), reported only on test.

  [0] Baseline    : independent Poisson + static weighted MLE        (≈ current core engine)
  [1] Flex-static : NB margin + Frank copula + diagonal inflation (static strengths, shape MLE on train)
  [2] Flex-dynamic: [1]'s dependence structure + dynamic strength filter    (process_sd tuned on val)

Metrics (lower is better): 1X2 mean RPS, exact score log-loss. With bootstrap 95% CI + paired test.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "models", "quantbet"))

from quantbet.scoreline import FlexibleScoreModel, DynamicStrengthFilter, ScoreMatrixCalibrator
from quantbet.scoreline import bivariate as biv
from quantbet.evaluation import bootstrap_ci


def make_data(n_teams=24, n_rounds=200, seed=7):
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    att = rng.normal(0, 0.35, n_teams)
    dfn = rng.normal(0, 0.30, n_teams)
    home_adv, nb_r = 0.28, 6.0
    rows, base = [], pd.Timestamp("2021-01-03")
    for rd in range(n_rounds):
        date = base + pd.Timedelta(weeks=rd)
        att = att + rng.normal(0, 0.045, n_teams); att -= att.mean()   # meaningful drift
        dfn = dfn + rng.normal(0, 0.040, n_teams)
        order = rng.permutation(n_teams)
        for i in range(0, n_teams, 2):
            h, a = order[i], order[i + 1]
            lh = np.exp(att[h] + dfn[a] + home_adv); la = np.exp(att[a] + dfn[h])
            shock = rng.normal(0, 0.12)   # weak negative dependence
            gh = rng.negative_binomial(nb_r, nb_r / (nb_r + lh * np.exp(-shock)))
            ga = rng.negative_binomial(nb_r, nb_r / (nb_r + la * np.exp(+shock)))
            rows.append(dict(date=date, home_team=teams[h], away_team=teams[a],
                             home_goals=int(gh), away_goals=int(ga), venue="home"))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def oidx(gh, ga): return 0 if gh > ga else (1 if gh == ga else 2)


def eval_matrices(mats, df):
    rps_l, ll_l = [], []
    for M, row in zip(mats, df.itertuples()):
        h, d, a = biv.outcome_probs(M); p = np.array([h, d, a]); p /= p.sum()
        y = oidx(row.home_goals, row.away_goals); e = np.zeros(3); e[y] = 1.0
        rps_l.append(float(((np.cumsum(p) - np.cumsum(e)) ** 2).sum() / 2.0))
        K = M.shape[0] - 1
        ll_l.append(-np.log(max(M[min(row.home_goals, K), min(row.away_goals, K)], 1e-12)))
    return np.array(rps_l), np.array(ll_l)


def run_dynamic(flex, strengths_src, fit_df, te, process_sd):
    """Filter through fit_df, then predict on test while filtering (no leakage), output matrices using flex's dependence structure."""
    flt = DynamicStrengthFilter(
        dict(zip(strengths_src.teams, strengths_src.params["attack"])),
        dict(zip(strengths_src.teams, strengths_src.params["defense"])),
        strengths_src.params["home_adj"], strengths_src.params["neutral_adj"],
        process_sd_per_year=process_sd, mean_reversion_halflife_days=540)
    flt.run(fit_df, collect_oos=False)
    mats = []
    for r in te.itertuples():
        lh, la = flt.expected_goals(r.home_team, r.away_team, "home", as_of=r.date)
        mats.append(flex.score_matrix(lh, la))
        flt.step(r.home_team, r.away_team, r.home_goals, r.away_goals, "home", r.date)
    return mats


def main():
    df = make_data()
    n = len(df)
    tr = df.iloc[:int(n * 0.6)].copy()
    va = df.iloc[int(n * 0.6):int(n * 0.8)].copy()
    te = df.iloc[int(n * 0.8):].copy()
    print(f"matches: train={len(tr)} val={len(va)} test={len(te)} | teams={df.home_team.nunique()}")

    base = FlexibleScoreModel("poisson", "none", False, damping=0.003).fit(tr)
    base_mats = [base.predict(r.home_team, r.away_team, "home")["score_matrix"] for r in te.itertuples()]

    flex = FlexibleScoreModel("nb", "frank", True, damping=0.003).fit(tr)
    flex_mats = [flex.predict(r.home_team, r.away_team, "home")["score_matrix"] for r in te.itertuples()]

    # Tune process_sd on val (including 0.0 -> data decides if dynamic)
    psd = DynamicStrengthFilter.tune_process_sd(
        dict(zip(flex.teams, flex.params["attack"])),
        dict(zip(flex.teams, flex.params["defense"])),
        flex.params["home_adj"], flex.params["neutral_adj"], tr, va)
    dyn_mats = run_dynamic(flex, flex, pd.concat([tr, va]), te, psd)

    print(f"\n  fitted: NB r={flex.shape_['nb_r']:.2f}  Frank κ={flex.shape_['dep']:.3f}  "
          f"θ_draw={flex.shape_['theta_draw']:.3f}  | val-tuned process_sd={psd}")
    print("\n" + "=" * 80)
    print(f"  {'model':<16}{'mean RPS [95% CI]':<32}{'exact log-loss [95% CI]'}")
    print("=" * 80)
    results = {}
    for name, mats in [("0 Baseline-Pois", base_mats), ("1 Flex-static", flex_mats), ("2 Flex-dynamic", dyn_mats)]:
        rps, ll = eval_matrices(mats, te); results[name] = (rps, ll)
        rm, rlo, rhi = bootstrap_ci(rps, n_boot=3000, seed=1)
        lm, llo, lhi = bootstrap_ci(ll, n_boot=3000, seed=1)
        print(f"  {name:<16}{rm:.4f} [{rlo:.4f}, {rhi:.4f}]        {lm:.4f} [{llo:.4f}, {lhi:.4f}]")
    print("=" * 80)

    r0, l0 = results["0 Baseline-Pois"]; r2, l2 = results["2 Flex-dynamic"]
    dm, dlo, dhi = bootstrap_ci(r0 - r2, n_boot=3000, seed=2)
    em, elo, ehi = bootstrap_ci(l0 - l2, n_boot=3000, seed=3)
    print(f"\n  Paired ΔRPS     (Baseline − Flex-dynamic): {dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}]"
          f"  {'✓significant' if dlo > 0 else 'CI contains 0'}")
    print(f"  Paired Δlogloss (Baseline − Flex-dynamic): {em:+.4f} [{elo:+.4f}, {ehi:+.4f}]"
          f"  {'✓significant' if elo > 0 else 'CI contains 0'}")
    print(f"\n  Relative improvement: RPS {100*(r0.mean()-r2.mean())/r0.mean():+.1f}%   "
          f"logloss {100*(l0.mean()-l2.mean())/l0.mean():+.1f}%")


if __name__ == "__main__":
    main()
