"""
quantbet.worldcup — World-Cup-specific extensions for QuantBet-EV v7.0.

These modules adapt the Bayesian Dixon-Coles engine to the structural
realities of international tournament football, where the bottleneck is
data scarcity (not model expressiveness). They are drop-in: each module
either (a) returns extra NLP/gradient terms you splice into your existing
SLSQP/MAP objective, or (b) consumes a closure over your fitted model and
adds new outputs (advancement, tournament, evaluation) without touching
your parameters.

Modules
-------
rating_prior          Module 1 — Elo/ranking-informed hierarchical prior.
knockout              Module 2 — extra time + penalty shootout resolution.
tournament            Module 3 — 48-team WC2026 Monte Carlo simulator.
trps                  Module 4 — Tournament Rank Probability Score.
confederation_prior   Module 5 — confederation hierarchical prior.
laplace_propagation   Helper — propagate Laplace posterior draws to outputs.

Priority for sparse World Cup data (see report): rating_prior (***),
knockout (***), tournament (***), trps (**), confederation_prior (**).
The high-expressiveness league extensions (Score-Driven, CMP/Weibull, xG)
are NOT included here — they overfit on national-team sample sizes.
"""

from __future__ import annotations

from .rating_prior import (
    RatingPrior,
    standardize_ratings,
    bradley_terry_log_strength,
)
from .knockout import (
    KnockoutResolver,
    AdvancementResult,
    build_score_matrix,
    split_home_draw_away,
)
from .tournament import (
    TournamentSimulator,
    TournamentConfig,
    TournamentProbabilities,
    GroupResult,
    make_config,
    default_2026_pairings,
)
from .trps import (
    trps,
    TRPSEvaluator,
    WC2026_BUCKETS,
    WC2026_WEIGHTS,
)
from .confederation_prior import (
    ConfederationPrior,
    suggest_friendly_weight,
)
from .laplace_propagation import (
    LaplacePropagator,
    TournamentCredibleSummary,
)

__all__ = [
    # module 1
    "RatingPrior",
    "standardize_ratings",
    "bradley_terry_log_strength",
    # module 2
    "KnockoutResolver",
    "AdvancementResult",
    "build_score_matrix",
    "split_home_draw_away",
    # module 3
    "TournamentSimulator",
    "TournamentConfig",
    "TournamentProbabilities",
    "GroupResult",
    "make_config",
    "default_2026_pairings",
    # module 4
    "trps",
    "TRPSEvaluator",
    "WC2026_BUCKETS",
    "WC2026_WEIGHTS",
    # module 5
    "ConfederationPrior",
    "suggest_friendly_weight",
    # laplace
    "LaplacePropagator",
    "TournamentCredibleSummary",
]

__version__ = "7.0-wc"
