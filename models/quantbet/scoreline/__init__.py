"""
scoreline — Score prediction accuracy upgrade package (frontier scoreline modelling)
====================================================================================

Targets three bottlenecks in QuantBet-EV's current "independent Poisson + 4-cell tau + static MAP",
providing deployable, numpy/scipy-only frontier alternatives:

  count_dists       flexible marginals (NegBin / Weibull-count) —— fixes goal dispersion
  bivariate         full-table dependence (bivariate Poisson / Frank copula) + diagonal inflation —— fixes dependence & draws
  dynamic_strength  time-varying strength online filtering (Koopman-Lit/Owen approx) —— fixes strength drift
  calibration       full score matrix temperature + draw calibration —— fixes derived market bias
  score_model      FlexibleScoreModel: assembles the above components, drop-in replacement for DC.predict()

Recommended frontier combination:
    margin='weibull'(or 'nb') + dependence='frank' + diagonal_inflation=True
    + DynamicStrengthFilter provides lambda + ScoreMatrixCalibrator post-processing
"""
from .score_model import FlexibleScoreModel
from .dynamic_strength import DynamicStrengthFilter
from .calibration import ScoreMatrixCalibrator
from . import count_dists, bivariate

__all__ = [
    "FlexibleScoreModel", "DynamicStrengthFilter", "ScoreMatrixCalibrator",
    "count_dists", "bivariate",
]
