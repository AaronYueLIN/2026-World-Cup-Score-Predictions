"""Smoke tests for quantbet.worldcup — run: python tests/test_worldcup_smoke.py"""
from __future__ import annotations

import numpy as np

from quantbet.worldcup import (
    RatingPrior, standardize_ratings, bradley_terry_log_strength,
    KnockoutResolver, build_score_matrix, split_home_draw_away,
    TournamentSimulator, make_config, TournamentProbabilities,
    trps, TRPSEvaluator, WC2026_BUCKETS,
    ConfederationPrior, suggest_friendly_weight,
    LaplacePropagator,
)


def test_rating_prior():
    raw = [1800, 1600, 2000, 1500, 1750]
    r = standardize_ratings(raw)
    assert abs(r.mean()) < 1e-9 and abs(r.std() - 1.0) < 1e-6
    prior = RatingPrior(r, eta_prior_sd=1.0)
    mu_att, mu_def = prior.prior_means(0.5, -0.5)
    assert abs(mu_att.sum()) < 1e-9  # zero-sum enforced
    ba = np.array([0.1, -0.2, 0.3, -0.1, 0.0])
    bd = np.array([-0.1, 0.2, -0.3, 0.1, 0.0])
    nlp = prior.nlp_beta(ba, bd, 0.4, 0.4, 0.5, -0.5)
    assert np.isfinite(nlp)
    # gradient finite-difference check on eta
    g_att, g_def = prior.grad_eta(ba, bd, 0.4, 0.4, 0.5, -0.5)
    eps = 1e-6
    f0 = prior.nlp_beta(ba, bd, 0.4, 0.4, 0.5, -0.5)
    f1 = prior.nlp_beta(ba, bd, 0.4, 0.4, 0.5 + eps, -0.5)
    assert abs((f1 - f0) / eps - g_att) < 1e-2, ((f1 - f0) / eps, g_att)
    print("  rating_prior OK")


def test_bradley_terry():
    rng = np.random.default_rng(0)
    n = 8
    true = rng.standard_normal(n)
    h = rng.integers(0, n, 400); a = rng.integers(0, n, 400)
    mask = h != a; h, a = h[mask], a[mask]
    d = true[h] - true[a]
    p = 1 / (1 + np.exp(-d))
    y = np.where(rng.random(len(d)) < p, 1, -1)
    est = bradley_terry_log_strength(h, a, y, n, ridge=1e-2)
    # rank correlation should be strongly positive
    corr = np.corrcoef(est.argsort().argsort(), true.argsort().argsort())[0, 1]
    assert corr > 0.6, corr
    print(f"  bradley_terry OK (rank corr={corr:.2f})")


def test_knockout():
    m = build_score_matrix(1.6, 1.1, rho=0.0)
    assert abs(m.sum() - 1.0) < 1e-9
    ph, pd, pa = split_home_draw_away(m)
    assert abs(ph + pd + pa - 1.0) < 1e-9
    res = KnockoutResolver()
    adv = res.advancement_prob(m, 1.6, 1.1)
    assert abs(adv.p_home_advance + adv.p_away_advance - 1.0) < 1e-9
    # stronger team should advance more often than its 90-min win prob
    assert adv.p_home_advance > ph
    # decomposition sums correctly
    s = (adv.p_home_reg + adv.p_away_reg + adv.p_home_et + adv.p_away_et
         + adv.p_home_pens + adv.p_away_pens)
    assert abs(s - 1.0) < 1e-6, s
    print(f"  knockout OK (adv home={adv.p_home_advance:.3f}, "
          f"reg={adv.p_home_reg:.3f}, et={adv.p_home_et:.3f}, "
          f"pens={adv.p_home_pens:.3f})")


def _toy_model_closure(strengths: dict[str, float]):
    """A toy match_prob_fn: lambda from strength difference."""
    def fn(home: str, away: str):
        sh, sa = strengths[home], strengths[away]
        lh = float(np.exp(0.2 + 0.6 * (sh - sa) / 2))
        la = float(np.exp(0.2 - 0.6 * (sh - sa) / 2))
        m = build_score_matrix(lh, la)
        return m, lh, la
    return fn


def test_tournament():
    rng = np.random.default_rng(1)
    teams = [f"T{i:02d}" for i in range(48)]
    strengths = {t: rng.standard_normal() for t in teams}
    groups = {g: teams[i*4:(i+1)*4] for i, g in enumerate("ABCDEFGHIJKL")}
    cfg = make_config(groups)
    sim = TournamentSimulator(_toy_model_closure(strengths), cfg)
    tp = sim.simulate(n_sims=300, seed=2)
    title = tp.title_odds()
    total_win = sum(p for _, p in title)
    assert abs(total_win - 1.0) < 1e-6, total_win
    # every team reached 'group' with prob 1
    assert all(abs(tp.probs[t]["group"] - 1.0) < 1e-9 for t in teams)
    # monotonic: P(reach R16) >= P(reach QF)
    for t in teams:
        assert tp.probs[t]["R16"] + 1e-9 >= tp.probs[t]["QF"]
    print(f"  tournament OK (sum P(win)={total_win:.4f}, "
          f"top={title[0][0]} {title[0][1]:.3f})")
    return tp, sim, strengths, cfg


def test_trps(tp: TournamentProbabilities):
    # toy: pick realised buckets = each team's argmax stage
    ev = TRPSEvaluator()
    realised = {}
    for t in tp.probs:
        bp = ev.stage_probs_to_bucket_probs(tp.probs[t])
        realised[t] = WC2026_BUCKETS[int(bp.argmax())]
    scores = ev.evaluate(tp.probs, realised)
    assert "mean" in scores and np.isfinite(scores["mean"])
    # perfect-ish forecast vs worst-case sanity
    perfect = trps([1, 0, 0, 0, 0, 0, 0], 0)
    worst = trps([1, 0, 0, 0, 0, 0, 0], 6)
    assert perfect < worst and abs(perfect) < 1e-9
    print(f"  trps OK (mean wTRPS={scores['mean']:.4f}, "
          f"perfect={perfect:.3f}, worst={worst:.3f})")


def test_confederation():
    confs = ["UEFA", "CONMEBOL", "UEFA", "CAF", "AFC", "CONMEBOL"]
    cp = ConfederationPrior(confs)
    assert cp.n_conf == 4
    mu = np.array([0.2, -0.1, 0.0, 0.05])
    expanded = cp.expand(mu)
    assert expanded.shape == (6,)
    ba = np.array([0.1, -0.2, 0.15, -0.05, 0.0, -0.1])
    bd = -ba
    nlp = cp.nlp(ba, bd, mu, -mu, 0.4, 0.4, 0.5, 0.5)
    assert np.isfinite(nlp)
    r = standardize_ratings([1800, 1900, 1700, 1500, 1400, 1850])
    dm = cp.demean_rating_within_conf(r)
    # within-conf demeaning: UEFA teams (idx 0,2) should have zero mean
    assert abs(dm[[0, 2]].mean()) < 1e-9
    w = suggest_friendly_weight(0.35, is_inter_confederation=True)
    assert w > 0.35
    print(f"  confederation OK (inter-conf friendly weight={w:.2f})")


def test_laplace(sim, strengths, cfg):
    # toy theta = stacked strengths; closure rebuilds model from theta
    teams = list(strengths.keys())
    mean = np.array([strengths[t] for t in teams])
    cov = np.eye(len(teams)) * 0.01

    def factory(theta):
        s = {t: float(theta[i]) for i, t in enumerate(teams)}
        return _toy_model_closure(s)

    prop = LaplacePropagator(mean, cov, factory, cfg)
    summary = prop.run(n_param_draws=5, n_sims_per_draw=80, seed=3)
    top_team = summary.title_table()[0][0]
    lo, m, hi = summary.title_credible_interval(top_team)
    assert lo <= m <= hi
    print(f"  laplace OK (top {top_team}: P(win) {m:.3f} "
          f"[{lo:.3f},{hi:.3f}])")


if __name__ == "__main__":
    print("Running quantbet.worldcup smoke tests...")
    test_rating_prior()
    test_bradley_terry()
    test_knockout()
    tp, sim, strengths, cfg = test_tournament()
    test_trps(tp)
    test_confederation()
    test_laplace(sim, strengths, cfg)
    print("\nAll smoke tests passed.")
