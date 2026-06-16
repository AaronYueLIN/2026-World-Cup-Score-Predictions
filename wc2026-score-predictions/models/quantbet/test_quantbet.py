"""
test_quantbet.py — Correctness tests
    python -m pytest tests/ -v
"""
import numpy as np
import pytest

from . import devig, dc_utils, markets, staking, portfolio
from . import posterior, pooling, evaluation as ev


# ----- devig ---------------------------------------------------------------
def test_devig_normalizes():
    odds = [1.95, 3.60, 4.20]
    for m in ("proportional", "power", "shin"):
        p = devig.devig(odds, method=m)
        assert abs(p.sum() - 1.0) < 1e-9
        assert np.all(p > 0)


def test_shin_corrects_flb():
    # Shin should lower underdog (longest odds) relative to proportional method
    odds = [1.50, 4.50, 7.00]
    p_prop = devig.devig_proportional(odds)
    p_shin = devig.devig_shin(odds)
    assert p_shin[-1] < p_prop[-1]        # underdog lowered
    assert p_shin[0] > p_prop[0]          # favourite raised


def test_overround_gt_one():
    assert devig.overround([1.95, 3.6, 4.2]) > 1.0


# ----- score matrix / markets ----------------------------------------------
def test_matrix_sums_to_one():
    M = dc_utils.dixon_coles_matrix(1.7, 1.0, rho=-0.16)
    assert abs(M.sum() - 1.0) < 1e-9


def test_1x2_partition():
    M = dc_utils.dixon_coles_matrix(1.4, 1.1, rho=-0.1)
    pH, pD, pA = markets.one_x_two(M)
    assert abs(pH + pD + pA - 1.0) < 1e-9


def test_joint_le_marginal():
    # P(A∧B) ≤ min(P(A),P(B))
    M = dc_utils.dixon_coles_matrix(1.7, 1.0, rho=-0.16)
    pH = markets.joint_prob(M, markets.home_win())
    pO = markets.joint_prob(M, markets.over(2.5))
    pHO = markets.joint_prob(M, markets.home_win(), markets.over(2.5))
    assert pHO <= min(pH, pO) + 1e-12
    assert pHO > 0


def test_over_under_complement():
    M = dc_utils.dixon_coles_matrix(1.5, 1.2)
    assert abs(markets.joint_prob(M, markets.over(2.5)) +
               markets.joint_prob(M, markets.under(2.5)) - 1.0) < 1e-9


# ----- staking -------------------------------------------------------------
def test_kelly_zero_when_no_edge():
    # Fair odds p=0.5, o=2.0 → edge 0 → kelly 0
    assert staking.kelly_fraction(0.5, 2.0) == pytest.approx(0.0, abs=1e-12)


def test_kelly_positive_with_edge():
    f = staking.kelly_fraction(0.60, 2.0)  # b=1, f=2*0.6-1=0.2
    assert f == pytest.approx(0.2, abs=1e-9)


def test_fractional_shrinks():
    assert staking.fractional_kelly(0.6, 2.0, 0.25) == pytest.approx(0.05, abs=1e-9)


def test_lcb_kelly_le_mean_kelly():
    rng = np.random.default_rng(0)
    samples = np.clip(rng.normal(0.60, 0.06, 2000), 1e-3, 1 - 1e-3)
    f_lcb = staking.lower_confidence_kelly(samples, 2.0, 0.25)
    f_mean = staking.kelly_fraction(samples.mean(), 2.0)
    assert f_lcb <= f_mean + 1e-9


# ----- portfolio -----------------------------------------------------------
def test_portfolio_feasible_and_bounded():
    M = dc_utils.dixon_coles_matrix(1.7, 1.0, rho=-0.16)
    mm = portfolio.MatchModel("m1", M)
    bets = [
        portfolio.Bet("H", [portfolio.Leg("m1", "1", markets.home_win())], 1.95),
        portfolio.Bet("O", [portfolio.Leg("m1", "O", markets.over(2.5))], 1.90),
    ]
    res = portfolio.risk_constrained_kelly(bets, [mm], lam=1.0)
    assert all(v >= -1e-9 for v in res.stakes.values())
    assert sum(res.stakes.values()) <= 1.0 + 1e-6
    assert 0.0 <= res.cash <= 1.0 + 1e-6


def test_risk_aversion_reduces_stakes():
    M = dc_utils.dixon_coles_matrix(2.0, 0.8, rho=-0.16)  # Strong home team → has edge
    mm = portfolio.MatchModel("m1", M)
    bets = [portfolio.Bet("H", [portfolio.Leg("m1", "1", markets.home_win())], 2.50)]
    low = portfolio.risk_constrained_kelly(bets, [mm], lam=0.0)
    high = portfolio.risk_constrained_kelly(bets, [mm], lam=5.0)
    assert high.stakes["H"] <= low.stakes["H"] + 1e-6


# ----- posterior -----------------------------------------------------------
def test_laplace_cov_psd():
    H = np.array([[40.0, 2.0], [2.0, 30.0]])
    cov, eigs = posterior.laplace_covariance(H)
    assert np.all(np.linalg.eigvalsh(cov) >= -1e-9)


def test_posterior_predictive_shape():
    H = np.array([[40.0, 2.0], [2.0, 30.0]])
    cov, _ = posterior.laplace_covariance(H)

    def pf(theta):
        v = np.array([theta[0], 0.0, -theta[0]])
        e = np.exp(v - v.max())
        return e / e.sum()

    mean, samp = posterior.posterior_predictive(np.array([0.5, 2.0]), cov, pf, n_samples=200)
    assert mean.shape == (3,)
    assert samp.shape == (200, 3)
    assert abs(mean.sum() - 1.0) < 1e-9


# ----- pooling -------------------------------------------------------------
def test_pools_normalize():
    P1 = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
    P2 = np.array([[0.4, 0.4, 0.2], [0.3, 0.4, 0.3]])
    for fn in (pooling.linear_pool, pooling.log_pool):
        P = fn([P1, P2], [0.6, 0.4])
        assert np.allclose(P.sum(1), 1.0)


def test_optimize_weight_in_range():
    rng = np.random.default_rng(1)
    N = 200
    true = rng.dirichlet([5, 3, 4], size=N)
    y = np.array([rng.choice(3, p=true[i]) for i in range(N)])
    P1 = true / true.sum(1, keepdims=True)
    P2 = np.clip(true + rng.normal(0, 0.1, true.shape), 1e-3, None)
    P2 /= P2.sum(1, keepdims=True)
    w = pooling.optimize_weight(P1, P2, y, method="log")
    assert 0.0 <= w <= 1.0


# ----- evaluation ----------------------------------------------------------
def test_rps_perfect_zero():
    assert ev.rps_score([1.0, 0.0, 0.0], 0) == pytest.approx(0.0, abs=1e-12)


def test_rps_ordinal_penalty():
    # Predict away but actual home: penalised more heavily than predicting draw (ordinal)
    far = ev.rps_score([0.0, 0.0, 1.0], 0)
    near = ev.rps_score([0.0, 1.0, 0.0], 0)
    assert far > near


def test_bootstrap_ci_brackets_mean():
    vals = np.random.default_rng(0).normal(0.2, 0.05, 300)
    m, lo, hi = ev.bootstrap_ci(vals)
    assert lo <= m <= hi


def test_clv():
    assert ev.clv(2.10, 2.00) == pytest.approx(0.05, abs=1e-9)
    s = ev.clv_summary([2.1, 1.9], [2.0, 2.0])
    assert s["n"] == 2
