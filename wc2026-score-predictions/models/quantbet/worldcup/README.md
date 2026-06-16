# quantbet.worldcup — World Cup extension modules for QuantBet-EV v7.0

Drop-in modules adapting the Bayesian Dixon-Coles engine to international
tournament football. Built for the real bottleneck of World Cup modelling:
**data scarcity**, not model expressiveness. Pure `numpy` + `scipy`, OOP +
type annotations, same style as `BayesianDixonColesModel`. No PyMC, no
heavy deps (avoids the notebook OOM you already hit).

## Install / placement

Copy the `quantbet/worldcup/` directory into your existing `quantbet/`
package. That's it — it's a sub-package:

```
quantbet/
├── __init__.py
├── posterior.py            # your existing Laplace code
├── ...                     # your existing v7.0 modules
└── worldcup/               # <-- drop this in
    ├── __init__.py
    ├── rating_prior.py
    ├── knockout.py
    ├── tournament.py
    ├── trps.py
    ├── confederation_prior.py
    └── laplace_propagation.py
```

Run the smoke test from your repo root:

```bash
PYTHONPATH=. python tests/test_worldcup_smoke.py
```

## Module map (priority for sparse WC data)

| Module | File | Priority | What it does |
|---|---|:--:|---|
| 1. Rating prior | `rating_prior.py` | ★★★ | Anchors attack/defence prior MEAN to Elo/FIFA/Bradley-Terry rating instead of zero. Replaces the zero-mean prior in spec §1.5. |
| 2. Knockout | `knockout.py` | ★★★ | Extra-time (×1/3 λ) + penalty shootout. Turns a 90-min score matrix into an advancement probability. Fixes the §2.3 structural gap. |
| 3. Tournament | `tournament.py` | ★★★ | 48-team WC2026 Monte Carlo: groups → best-thirds → R32 → final, with FIFA tiebreakers. Outright/stage-reach probabilities. |
| 4. TRPS | `trps.py` | ★★ | Tournament Rank Probability Score for one-shot tournament evaluation. Use instead of RPS at tournament level. |
| 5. Confederation prior | `confederation_prior.py` | ★★ | Confederation-level random effect for cross-confederation identifiability. Composes with module 1. |
| Laplace helper | `laplace_propagation.py` | — | Propagates your §Rule-7 Laplace posterior draws into tournament probabilities with credible intervals. |

The high-expressiveness **league** extensions (Score-Driven/GAS, CMP/Weibull
marginals, xG channel) are deliberately **excluded** — they overfit at
national-team sample sizes. See the proposal report for the reasoning.

## Wiring into the MAP/NLP objective (modules 1 & 5)

These return extra NLP terms (and analytic gradients) you splice into your
existing SLSQP objective. Append the new free parameters (`eta_att`,
`eta_def` for module 1; `mu_conf_*`, `tau_*` for module 5) to your `theta`
vector, and bump `k` in AIC/BIC accordingly.

```python
from quantbet.worldcup import RatingPrior, standardize_ratings

ratings_std = standardize_ratings(fifa_or_elo_points)   # team-ordered
prior = RatingPrior(ratings_std, eta_prior_sd=1.0)

# inside your NLP(theta):
#   ... unpack beta_att, beta_def, sigma_att, sigma_def, eta_att, eta_def ...
nlp += prior.nlp_beta(beta_att, beta_def, sigma_att, sigma_def,
                      eta_att, eta_def)   # replaces the two §1.5 prior terms
# optional jac:
g_att, g_def = prior.grad_beta(beta_att, beta_def, sigma_att, sigma_def,
                               eta_att, eta_def)
g_eta_att, g_eta_def = prior.grad_eta(beta_att, beta_def, sigma_att,
                                      sigma_def, eta_att, eta_def)
```

To use both priors together, let module 5 set the between-confederation
level and module 1 set the within-confederation ordering:

```python
from quantbet.worldcup import ConfederationPrior
cp = ConfederationPrior(conf_ids)                  # team-ordered labels
ratings_within = cp.demean_rating_within_conf(ratings_std)
# prior mean = mu_conf(i) + eta_within * ratings_within(i)
```

## Wiring the simulator & knockout (modules 2 & 3)

These never touch your parameters. You supply a closure over your fitted
model:

```python
from quantbet.worldcup import make_config, TournamentSimulator, KnockoutResolver

def match_prob_fn(home_id, away_id):
    # use your fitted model to produce a tau-corrected, normalised 11x11
    m = model.score_matrix(home_id, away_id)        # spec §2.3 output
    lh, la = model.lambdas(home_id, away_id)
    return m, lh, la

groups = {"A": ["Qatar", "Ecuador", ...], ..., "L": [...]}   # 12 groups x 4
config = make_config(groups)            # uses placeholder R32 bracket — see below
sim = TournamentSimulator(match_prob_fn, config, KnockoutResolver(et_scale=1/3))
probs = sim.simulate(n_sims=50_000, seed=0)

probs.title_odds()[:10]                 # top-10 title chances
probs.reach_stage("QF")                 # P(reach quarter-final) per team
```

**⚠️ 2026 bracket:** `default_2026_pairings()` is a *structurally valid
placeholder* (12 winners + 12 runners-up + 8 best-thirds = 32). Replace it
with the official 2026 R32 slot map once finalised — pass your own
`r32_pairings` to `make_config(groups, r32_pairings=...)`. The simulation
logic is correct; only this static map needs the official slot codes.

For a single knockout tie (advancement / "to qualify" markets):

```python
res = KnockoutResolver(et_scale=1/3)
adv = res.advancement_prob(score_matrix_90, lambda_h, lambda_a,
                           rho=model.rho, tau_fn=model.tau,
                           p_home_shootout=0.5)
adv.p_home_advance      # includes regulation + ET + penalties
adv.as_dict()           # full method-of-advancement decomposition
```

## Tournament evaluation (module 4)

```python
from quantbet.worldcup import TRPSEvaluator
ev = TRPSEvaluator()                    # WC2026 buckets + doubling weights
# pre-tournament: convert simulator probs to bucket distributions internally
# post-tournament: realised[team] = "Winner"|"Final"|"SF"|"QF"|"R16"|"R32"|"group"
scores = ev.evaluate(probs.probs, realised)
scores["mean"]                          # mean wTRPS (lower = better)
```

## Uncertainty propagation (Rule 7 / Laplace)

```python
from quantbet.posterior import laplace_mean_cov     # your existing code
from quantbet.worldcup import LaplacePropagator

mean, cov = laplace_mean_cov(model)     # N(theta_hat, H^{-1})

def factory(theta_draw):
    m = model.with_parameters(theta_draw)
    def fn(h, a): return m.score_matrix(h, a), *m.lambdas(h, a)
    return fn

prop = LaplacePropagator(mean, cov, factory, config)
summary = prop.run(n_param_draws=200, n_sims_per_draw=2000, seed=0)
summary.title_credible_interval("Brazil")   # (lo, mean, hi)
```

Same call works with a true posterior via `prop.from_samples(theta_samples)`.

## Notes & caveats

- **Group tiebreakers** implement points → GD → GF → random-lot. FIFA's full
  ladder also has head-to-head and fair-play; head-to-head storage is wired
  but not yet applied as a tiebreak step (left as a clearly marked spot in
  `_simulate_group`). Add it if your edge depends on tight groups.
- **Penalty shootout** defaults to 50/50, the robust choice absent shootout
  data. Pass `p_home_shootout` from a keeper/penalty model if you have one.
- **`bradley_terry_log_strength`** is a lightweight pure-numpy BT-Davidson
  rating producer (Macrì Demartino et al. 2024) — use it to manufacture a
  better `r_i` than raw FIFA points, then feed `standardize_ratings`.
