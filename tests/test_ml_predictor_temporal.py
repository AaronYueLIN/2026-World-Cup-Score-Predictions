"""
Temporal feature correctness tests — Fixing OOF data leakage

Covers:
  1. ``EnsemblePredictor.fit_weights`` must use train_df + val_df matches
     earlier than the current match as feature history, **must not** let
     val_df leak as a whole.
  2. The ``OUTCOME_ENCODE`` category order ``[A=0, D=1, H=2]`` is a project
     hard constraint. All code depending on A/D/H order such as ``_pool_two``
     relies on this order.

Note: This file does not depend on ``bayesian_dixon_coles`` or ``histgbm``,
runs fast (< 1s), CI-friendly.
"""
import numpy as np
import pandas as pd

from models.ml_predictor import EnsemblePredictor, OUTCOME_ENCODE


# ---------------------------------------------------------------------------
# Dummy / Spy test fixtures
# ---------------------------------------------------------------------------

class DummyDC:
    """Dummy DC with no dependencies: fixed returns [0.25, 0.25, 0.50]"""
    def predict(self, home_team, away_team, venue="neutral"):
        return {
            "away_win_prob": 0.25,
            "draw_prob": 0.25,
            "home_win_prob": 0.50,
            "expected_home_goals": 1.4,
            "expected_away_goals": 0.9,
        }


class DummyGBM:
    """Dummy GBM with no dependencies: fixed returns [0.30, 0.30, 0.40]"""
    def predict_proba_from_features(self, feat):
        return np.array([0.30, 0.30, 0.40], dtype=float)


class SpyFeatureEngineer:
    """Records history length and before_date for each call to ``get_match_features``"""
    def __init__(self):
        self.history_lengths = []
        self.before_dates = []

    def get_match_features(self, df, home_team, away_team, before_date, dc_model=None, venue="neutral"):
        self.history_lengths.append(len(df))
        self.before_dates.append(pd.to_datetime(before_date))
        return {"x": 1.0}


def _df():
    """5 consecutive matches over 5 days"""
    return pd.DataFrame([
        {"date": "2024-01-01", "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0, "result": "H", "venue": "neutral"},
        {"date": "2024-01-02", "home_team": "C", "away_team": "D", "home_goals": 0, "away_goals": 0, "result": "D", "venue": "neutral"},
        {"date": "2024-01-03", "home_team": "A", "away_team": "C", "home_goals": 0, "away_goals": 1, "result": "A", "venue": "neutral"},
        {"date": "2024-01-04", "home_team": "B", "away_team": "D", "home_goals": 2, "away_goals": 1, "result": "H", "venue": "neutral"},
        {"date": "2024-01-05", "home_team": "A", "away_team": "D", "home_goals": 1, "away_goals": 1, "result": "D", "venue": "neutral"},
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fit_weights_uses_train_plus_earlier_validation_history():
    """
    OOF core assertion: the feature history for the i-th row of val_df
    must strictly equal train_df ∪ val_df[date < val_df[i].date].

    Data:
      train_df = first 3 matches (2024-01-01 ~ 2024-01-03)
      val_df   = last 2 matches (2024-01-04, 2024-01-05)

    Expected history_lengths:
      2024-01-04 (1st val): 3 + 0 = 3  (no earlier val matches)
      2024-01-05 (2nd val): 3 + 1 = 4  (01-04 is earlier in val)
    """
    df = _df()
    train_df = df.iloc[:3].copy()
    val_df   = df.iloc[3:].copy()

    spy = SpyFeatureEngineer()
    ens = EnsemblePredictor(
        dc_model=DummyDC(),
        gbm_model=DummyGBM(),
        feature_engineer=spy,
        dc_weight=0.5,
        pool_method="log",
    )

    ens.fit_weights(train_df, val_df)

    assert spy.history_lengths == [3, 4], (
        f"Expected feature history lengths [3, 4] (train=3, val i=0→+0, val i=1→+1), "
        f"got {spy.history_lengths}. This indicates OOF data leakage or "
        f"missing before_date filtering."
    )
    assert ens.pool_method == "log"
    assert 0.0 <= ens.dc_weight <= 1.0
    assert abs(ens.dc_weight + ens.gbm_weight - 1.0) < 1e-12


def test_outcome_encoding_order_is_away_draw_home():
    """
    Project hard constraint: OUTCOME_ENCODE must preserve ``[A, D, H]`` order.
    The P(A) / P(D) / P(H) order in ``_pool_two`` and ``_pool_two``'s
    blend both depend on this order. Changing this constant would cause
    ``dc_utils``, ``_pool_two``, and ``predict`` to all misalign.
    """
    assert OUTCOME_ENCODE == {"A": 0, "D": 1, "H": 2}


def test_fit_weights_requires_train_and_val():
    """fit_weights no longer supports passing only val_df (breaking signature)"""
    ens = EnsemblePredictor(
        dc_model=DummyDC(),
        gbm_model=DummyGBM(),
        feature_engineer=SpyFeatureEngineer(),
        dc_weight=0.5,
    )
    # Use empty DataFrame as train_df to trigger val_df=None validation
    train_df = pd.DataFrame(columns=[
        "date", "home_team", "away_team", "home_goals", "away_goals", "result", "venue"
    ])
    try:
        ens.fit_weights(train_df=train_df, val_df=None)
    except ValueError as e:
        assert "train_df" in str(e) and "val_df" in str(e)
    else:
        raise AssertionError("fit_weights(val_df=None) should raise ValueError")
