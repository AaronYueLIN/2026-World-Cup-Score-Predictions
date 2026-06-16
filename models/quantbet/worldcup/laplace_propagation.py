"""
laplace_propagation.py — propagate Laplace posterior into WC outputs.

WHY
---
Your spec Rule 7 quantifies parameter uncertainty via a Laplace
approximation at the MAP estimate (quantbet/posterior.py): the posterior is
approximated N(theta_hat, H^{-1}) where H is the Hessian (observed
information) of the NLP at the optimum. Point-estimate tournament
probabilities ignore this uncertainty. This helper draws parameter samples
from the Laplace posterior, rebuilds your match-probability closure for
each draw, and aggregates tournament probabilities WITH credible intervals.

It is engine-agnostic: it only needs (1) the Laplace mean/cov (or a sampler)
and (2) a factory that turns a parameter draw into a match_prob_fn closure.
Works identically whether the underlying fit was SLSQP/MAP or, later, a
Bayesian posterior — for a true posterior, pass your posterior draws via
`from_samples` instead of the Gaussian Laplace approximation.

HOW TO WIRE INTO QuantBet-EV v7.0
---------------------------------
    from quantbet.posterior import laplace_mean_cov   # your existing code
    mean, cov = laplace_mean_cov(model)               # N(theta_hat, H^{-1})

    def make_closure(theta_draw):
        m = model.with_parameters(theta_draw)          # your model rebuild
        def fn(home, away):
            return m.score_matrix(home, away), *m.lambdas(home, away)
        return fn

    prop = LaplacePropagator(mean, cov, make_closure, config)
    summary = prop.run(n_param_draws=200, n_sims_per_draw=2000, seed=0)
    summary.title_credible_interval("Brazil")   # (lo, median, hi)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt

from .tournament import (
    TournamentConfig,
    TournamentSimulator,
    TournamentProbabilities,
)
from .knockout import KnockoutResolver

FloatArray = npt.NDArray[np.float64]

# theta draw -> match_prob_fn closure
ClosureFactory = Callable[[FloatArray], Callable[[str, str], tuple]]


@dataclass
class TournamentCredibleSummary:
    """Per-team stage probabilities with credible intervals across draws."""

    stages: list[str]
    teams: list[str]
    # mean[team][stage], lo[team][stage], hi[team][stage]
    mean: dict[str, dict[str, float]]
    lo: dict[str, dict[str, float]]
    hi: dict[str, dict[str, float]]
    credible_mass: float

    def title_credible_interval(self, team: str) -> tuple[float, float, float]:
        """Return (lo, mean, hi) of P(win) for a team across parameter draws."""
        return (self.lo[team]["Winner"],
                self.mean[team]["Winner"],
                self.hi[team]["Winner"])

    def title_table(self) -> list[tuple[str, float, float, float]]:
        """All teams sorted by mean P(win): (team, lo, mean, hi)."""
        rows = [(t, self.lo[t]["Winner"], self.mean[t]["Winner"], self.hi[t]["Winner"])
                for t in self.teams]
        return sorted(rows, key=lambda r: r[2], reverse=True)


@dataclass
class LaplacePropagator:
    """Propagate Laplace (or arbitrary) parameter draws into tournament probs.

    Parameters
    ----------
    mean
        MAP estimate theta_hat (1-D).
    cov
        Laplace covariance H^{-1} (2-D). Ignored if you use `from_samples`.
    closure_factory
        Maps a parameter draw to a match_prob_fn closure (see module 3).
    config
        TournamentConfig (48-team structure).
    resolver
        Shared KnockoutResolver.
    """

    mean: FloatArray
    cov: FloatArray
    closure_factory: ClosureFactory
    config: TournamentConfig
    resolver: KnockoutResolver = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64).ravel()
        self.cov = np.asarray(self.cov, dtype=np.float64)
        if self.resolver is None:
            self.resolver = KnockoutResolver()
        # Cholesky once for fast Gaussian sampling; jitter if needed.
        self._chol = _safe_cholesky(self.cov)

    def _sample_theta(self, rng: np.random.Generator) -> FloatArray:
        z = rng.standard_normal(self.mean.size)
        return self.mean + self._chol @ z

    def run(
        self,
        n_param_draws: int = 200,
        n_sims_per_draw: int = 2000,
        *,
        credible_mass: float = 0.90,
        seed: Optional[int] = None,
    ) -> TournamentCredibleSummary:
        """Draw parameters from the Laplace posterior and propagate.

        Total tournament simulations = n_param_draws * n_sims_per_draw.
        For 200 x 2000 = 400k that is plenty; tune down for speed.
        """
        rng = np.random.default_rng(seed)
        per_draw: list[TournamentProbabilities] = []

        for _ in range(n_param_draws):
            theta = self._sample_theta(rng)
            fn = self.closure_factory(theta)
            sim = TournamentSimulator(fn, self.config, self.resolver)
            tp = sim.simulate(n_sims=n_sims_per_draw,
                              seed=int(rng.integers(0, 2**31 - 1)))
            per_draw.append(tp)

        return self._aggregate(per_draw, credible_mass)

    def from_samples(
        self,
        theta_samples: Sequence[FloatArray] | FloatArray,
        n_sims_per_draw: int = 2000,
        *,
        credible_mass: float = 0.90,
        seed: Optional[int] = None,
    ) -> TournamentCredibleSummary:
        """Use externally supplied parameter draws (e.g. true posterior)."""
        rng = np.random.default_rng(seed)
        samples = np.asarray(theta_samples, dtype=np.float64)
        per_draw: list[TournamentProbabilities] = []
        for theta in samples:
            fn = self.closure_factory(theta)
            sim = TournamentSimulator(fn, self.config, self.resolver)
            tp = sim.simulate(n_sims=n_sims_per_draw,
                              seed=int(rng.integers(0, 2**31 - 1)))
            per_draw.append(tp)
        return self._aggregate(per_draw, credible_mass)

    def _aggregate(
        self,
        per_draw: list[TournamentProbabilities],
        credible_mass: float,
    ) -> TournamentCredibleSummary:
        stages = per_draw[0].stages
        teams = list(per_draw[0].probs.keys())
        alpha = (1.0 - credible_mass) / 2.0

        mean: dict[str, dict[str, float]] = {}
        lo: dict[str, dict[str, float]] = {}
        hi: dict[str, dict[str, float]] = {}

        for t in teams:
            mean[t], lo[t], hi[t] = {}, {}, {}
            for s in stages:
                vals = np.array([d.probs[t][s] for d in per_draw], dtype=np.float64)
                mean[t][s] = float(vals.mean())
                lo[t][s] = float(np.quantile(vals, alpha))
                hi[t][s] = float(np.quantile(vals, 1.0 - alpha))

        return TournamentCredibleSummary(
            stages=stages, teams=teams,
            mean=mean, lo=lo, hi=hi, credible_mass=credible_mass,
        )


def _safe_cholesky(cov: FloatArray, max_tries: int = 6) -> FloatArray:
    """Cholesky with progressive jitter for near-singular Laplace covariances."""
    cov = np.asarray(cov, dtype=np.float64)
    n = cov.shape[0]
    base = np.trace(cov) / max(n, 1) * 1e-9
    for k in range(max_tries):
        try:
            return np.linalg.cholesky(cov + np.eye(n) * base * (10**k))
        except np.linalg.LinAlgError:
            continue
    # final fallback: diagonal-only (uncorrelated) approximation
    return np.diag(np.sqrt(np.clip(np.diag(cov), 1e-12, None)))


__all__ = ["LaplacePropagator", "TournamentCredibleSummary"]
