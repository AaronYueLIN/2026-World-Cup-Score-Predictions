"""
demo.py — End-to-end demo (synthetic data, runs without your private model files)
==================================================================================
    python -m quantbet.demo
"""
from __future__ import annotations

import numpy as np

from . import devig, dc_utils, markets, staking, portfolio
from . import posterior, pooling, evaluation as ev
from .value_engine_v2 import evaluate_market, build_card


def hr(t):
    print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


def main():
    rng = np.random.default_rng(7)

    # ---------------------------------------------------------------
    hr("1) Shin de-vig vs proportional method (1X2)")
    odds = [1.95, 3.60, 4.20]  # Home / Draw / Away
    p_prop = devig.devig_proportional(odds)
    p_shin, z = devig.devig_shin(odds, return_z=True)
    print(f"  overround B = {devig.overround(odds):.4f}  (vig = {devig.overround(odds)-1:.2%})")
    print(f"  proportional : H={p_prop[0]:.4f} D={p_prop[1]:.4f} A={p_prop[2]:.4f}")
    print(f"  shin (z={z:.3f}): H={p_shin[0]:.4f} D={p_shin[1]:.4f} A={p_shin[2]:.4f}")
    print("  → Shin reduces underdog (Away) probability, raises favourite (Home), correcting FLB")

    # ---------------------------------------------------------------
    hr("2) DC score matrix → exact market / same-game parlay joint probability")
    M = dc_utils.dixon_coles_matrix(lam=1.7, mu=1.0, rho=-0.16, max_goals=10)
    pH, pD, pA = markets.one_x_two(M)
    p_over = markets.joint_prob(M, markets.over(2.5))
    p_btts = markets.joint_prob(M, markets.btts(True))
    # Same-game parlay: Home WIN AND Over 2.5 —— exact vs independent approximation
    p_joint = markets.joint_prob(M, markets.home_win(), markets.over(2.5))
    p_indep = pH * p_over
    print(f"  1X2: H={pH:.3f} D={pD:.3f} A={pA:.3f}")
    print(f"  Over2.5={p_over:.3f}  BTTS={p_btts:.3f}")
    print(f"  Same-game [Home & Over2.5] exact joint = {p_joint:.4f}")
    print(f"               independent product approx = {p_indep:.4f}  (diff {(p_indep/p_joint-1):+.1%})")
    print("  → Using independence approximation systematically mis-estimates same-game parlay EV")

    # ---------------------------------------------------------------
    hr("3) Value layer: Shin de-vig → EV/edge → fractional Kelly")
    model_probs = [pH, pD, pA]
    sels = evaluate_market(["Home", "Draw", "Away"], model_probs, odds,
                           devig_method="shin", kelly_fraction=0.25)
    for s in sels:
        print(f"  {s.name:5s} model={s.model_prob:.3f} fair={s.market_fair_prob:.3f} "
              f"edge={s.edge:+.3f} EV={s.ev:+.3f} kelly={s.kelly:.3f} stake={s.stake_fraction:.3f}")
    card = build_card(sels, min_edge=0.0, min_ev=0.0)
    print("  Recommended bets:", [s.name for s in card] or "none")

    # ---------------------------------------------------------------
    hr("4) Risk-constrained Kelly portfolio (singles + same-game parlays, drawdown bound)")
    mm = portfolio.MatchModel("BRA_MAR", M)
    bets = [
        portfolio.Bet("Home", [portfolio.Leg("BRA_MAR", "1", markets.home_win())], odds=1.95),
        portfolio.Bet("Over2.5", [portfolio.Leg("BRA_MAR", "O2.5", markets.over(2.5))], odds=1.90),
        portfolio.Bet("Home&Over", [
            portfolio.Leg("BRA_MAR", "1", markets.home_win()),
            portfolio.Leg("BRA_MAR", "O2.5", markets.over(2.5)),
        ], odds=3.10),
    ]
    for lam in (0.0, 1.0, 3.0):
        res = portfolio.risk_constrained_kelly(bets, [mm], lam=lam)
        dd = f"  P(drawdown≤30%)≤{res.drawdown_bound(0.3):.2f}" if res.drawdown_bound else ""
        print(f"  λ={lam:>3}: stakes=" +
              ", ".join(f"{k}={v:.3f}" for k, v in res.stakes.items()) +
              f"  cash={res.cash:.3f}  g={res.expected_log_growth:.4f}{dd}")
    print("  → λ↑ shrinks stakes, tightens drawdown bound; λ=0 is pure Kelly")

    # ---------------------------------------------------------------
    hr("5) Laplace posterior → posterior predictive probabilities (less extreme than MAP plug-in)")

    # Toy: 2-D parameter (atk_diff, total) → 1X2 via DC; negative log posterior is a simple quadratic
    theta_map = np.array([0.55, 2.7])  # [log attack difference, total goal rate]
    H = np.array([[40.0, 2.0], [2.0, 30.0]])  # Assumed Hessian (data information)

    def predict_fn(theta):
        d, tot = theta
        lam = np.exp(np.log(tot) / 2 + d / 2)
        mu = np.exp(np.log(tot) / 2 - d / 2)
        Mi = dc_utils.dixon_coles_matrix(lam, mu, rho=-0.16, max_goals=8)
        return np.array(markets.one_x_two(Mi))

    cov, eigs = posterior.laplace_covariance(H)
    p_map = predict_fn(theta_map)
    p_pp, samples = posterior.posterior_predictive(theta_map, cov, predict_fn, n_samples=800)
    print(f"  MAP plug-in   : H={p_map[0]:.3f} D={p_map[1]:.3f} A={p_map[2]:.3f}")
    print(f"  Posterior predictive mean: H={p_pp[0]:.3f} D={p_pp[1]:.3f} A={p_pp[2]:.3f}")
    print(f"  Posterior P(Home) 5%-95% interval: "
          f"[{np.percentile(samples[:,0],5):.3f}, {np.percentile(samples[:,0],95):.3f}]")
    # Posterior lower-quantile Kelly
    f_lcb = staking.lower_confidence_kelly(samples[:, 0], odds=1.95, quantile=0.25)
    f_mean = staking.kelly_fraction(p_pp[0], 1.95)
    print(f"  Home Kelly: mean={f_mean:.3f}  lower quantile(25%)={f_lcb:.3f}  → uncertainty shrinks stake")

    # ---------------------------------------------------------------
    hr("6) Aggregation: linear vs logarithmic pooling + RPS optimal weight")
    N = 400
    true = rng.dirichlet([6, 3, 4], size=N)
    y = np.array([rng.choice(3, p=true[i]) for i in range(N)])
    P_dc = np.clip(true + rng.normal(0, 0.05, true.shape), 1e-3, None)
    P_dc /= P_dc.sum(1, keepdims=True)
    P_gbm = np.clip(true + rng.normal(0, 0.09, true.shape), 1e-3, None)
    P_gbm /= P_gbm.sum(1, keepdims=True)
    w_lin = pooling.optimize_weight(P_dc, P_gbm, y, method="linear")
    w_log = pooling.optimize_weight(P_dc, P_gbm, y, method="log")
    rps_dc = ev.mean_rps(P_dc, y)
    rps_lin = ev.mean_rps(pooling.linear_pool([P_dc, P_gbm], [w_lin, 1 - w_lin]), y)
    rps_log = ev.mean_rps(pooling.log_pool([P_dc, P_gbm], [w_log, 1 - w_log]), y)
    print(f"  DC only        RPS={rps_dc:.4f}")
    print(f"  linear (w={w_lin:.2f})  RPS={rps_lin:.4f}")
    print(f"  log    (w={w_log:.2f})  RPS={rps_log:.4f}")

    # ---------------------------------------------------------------
    hr("7) Rigorous evaluation: bootstrap CI + paired comparison + CLV")
    per_dc = [ev.rps_score(P_dc[i], y[i]) for i in range(N)]
    per_gbm = [ev.rps_score(P_gbm[i], y[i]) for i in range(N)]
    m, lo, hi = ev.bootstrap_ci(per_dc)
    print(f"  DC mean RPS = {m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    d, dlo, dhi = ev.compare_models_ci(per_dc, per_gbm)
    sig = "significant" if (dlo > 0 or dhi < 0) else "not significant (CI contains 0)"
    print(f"  DC - GBM RPS diff = {d:+.4f}  CI [{dlo:+.4f}, {dhi:+.4f}] → {sig}")
    # n=5 control: demonstrates why small samples are incomparable
    m5, lo5, hi5 = ev.bootstrap_ci(per_dc[:5])
    print(f"  If only n=5: RPS={m5:.4f} CI [{lo5:.4f}, {hi5:.4f}] (interval too wide to be meaningful)")
    bet_odds = np.array([2.10, 1.95, 3.40, 2.05, 1.80])
    close_odds = np.array([2.00, 1.98, 3.10, 2.05, 1.72])
    print("  CLV summary:", ev.clv_summary(bet_odds, close_odds))

    print("\nAll modules passed.\n")


if __name__ == "__main__":
    main()
