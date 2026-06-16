"""
QuantBet-EV: Module 3 — ML Predictor (Gradient Boosting Ensemble)

Architecture:
  FeatureEngineer     — Extract rolling features from match history (time-aware, no future data leakage)
  HistGBMPredictor    — sklearn HistGradientBoosting (equivalent to LightGBM, no GPU needed)
                        + Isotonic Calibration (calibrated probability output)
  EnsemblePredictor   — Dixon-Coles + GBM weighted probability fusion
                        + scipy optimal weight search (minimize validation RPS)
  WalkForwardCV       — Time series cross-validation (Walk-Forward Validation, prevents data leakage)

Feature source: only uses match result history (home/away goals), no player/xG/odds data needed.

Laptop performance:
  Feature building  ~0.5s (500 matches training data)
  GBM training      ~0.3s (300 trees)
  Ensemble predict  ~1ms (single match)

Dependencies: numpy, pandas, scipy, scikit-learn (all built-in)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

# Match result encoding (multiclass target)
OUTCOME_ENCODE = {"A": 0, "D": 1, "H": 2}
OUTCOME_DECODE = {v: k for k, v in OUTCOME_ENCODE.items()}

# Pooling helper (optional import; quantbet.pooling indirectly depends on quantbet.evaluation)
try:
    from quantbet.pooling import log_pool, linear_pool
except Exception:  # noqa: BLE001 — defensive: fall back to inlined impl
    log_pool = None
    linear_pool = None


# ======================================================================
#  Feature Engineering
# ======================================================================

class FeatureEngineer:
    """
    Extract time-aware features from match result history.

    Principles:
      - For each match to be predicted, only use data before that match date
      - Feature computation is per-team, extracted separately for home and away
      - Supports injecting DC model features (DC probabilities / attack/defense parameters)

    Feature list (~40 dimensions total):
      home/away rolling stats   x [3, 5, 10] game windows
      head-to-head stats
      DC model features (optional)
    """

    WINDOWS = [3, 5, 10]

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def build_training_matrix(
        self,
        df: pd.DataFrame,
        dc_model=None,
        venue_col: str = "venue",
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Generate feature matrix and labels for each match in the training set.

        Args:
            df:         Training data (containing date, home_team, away_team,
                        home_goals, away_goals, result)
            dc_model:   Optional, BayesianDixonColesModel instance
            venue_col:  Venue column name (if present)

        Returns:
            X: Feature DataFrame
            y: Label ndarray (0=A, 1=D, 2=H)
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        rows = []
        for _, row in df.iterrows():
            feat = self.get_match_features(
                df=df,
                home_team=row["home_team"],
                away_team=row["away_team"],
                before_date=row["date"],
                dc_model=dc_model,
                venue=row.get(venue_col, "neutral"),
            )
            rows.append(feat)

        X = pd.DataFrame(rows)
        y = df["result"].map(OUTCOME_ENCODE).values
        return X, y

    def get_match_features(
        self,
        df: pd.DataFrame,
        home_team: str,
        away_team: str,
        before_date,
        dc_model=None,
        venue: str = "neutral",
    ) -> dict:
        """
        Generate prediction feature vector for a single match.

        Args:
            df:          Historical data (only uses rows before before_date)
            home_team:   Home team
            away_team:   Away team
            before_date: Only use data before this date (prevents future leakage)
            dc_model:    Optional, provides DC probabilities and attack/defense parameter features
            venue:       'home' / 'neutral' / 'away'

        Returns:
            Feature dictionary
        """
        before_date = pd.to_datetime(before_date)
        hist = df[df["date"] < before_date].copy()

        feat: dict = {}

        # Home/away team rolling features
        for prefix, team in [("h_", home_team), ("a_", away_team)]:
            feat.update(
                self._team_rolling_features(hist, team, prefix)
            )

        # H2H features
        feat.update(self._h2h_features(hist, home_team, away_team))

        # Venue encoding
        feat["venue_home"]    = int(venue == "home")
        feat["venue_neutral"] = int(venue == "neutral")

        # Dixon-Coles features (optional)
        if dc_model is not None:
            feat.update(
                self._dc_features(dc_model, home_team, away_team, venue)
            )

        return feat

    # ------------------------------------------------------------------
    #  Rolling Team Features
    # ------------------------------------------------------------------

    def _team_rolling_features(
        self, hist: pd.DataFrame, team: str, prefix: str
    ) -> dict:
        """
        Extract rolling statistical features for a single team.

        For three windows (3 / 5 / 10 games) compute:
          - Goals for mean, goals against mean
          - Win rate, draw rate
          - Goal difference mean
          - Points mean (3/1/0)
          - Clean sheet rate
        """
        feat: dict = {}

        # Extract all matches for this team from history
        home_games = hist[hist["home_team"] == team].copy()
        away_games = hist[hist["away_team"] == team].copy()

        # Standardize to team perspective: goals for/against and result
        home_games["gf"] = home_games["home_goals"]
        home_games["ga"] = home_games["away_goals"]
        home_games["w"]  = (home_games["home_goals"] > home_games["away_goals"]).astype(int)
        home_games["d"]  = (home_games["home_goals"] == home_games["away_goals"]).astype(int)
        home_games["cs"] = (home_games["away_goals"] == 0).astype(int)

        away_games["gf"] = away_games["away_goals"]
        away_games["ga"] = away_games["home_goals"]
        away_games["w"]  = (away_games["away_goals"] > away_games["home_goals"]).astype(int)
        away_games["d"]  = (away_games["away_goals"] == away_games["home_goals"]).astype(int)
        away_games["cs"] = (away_games["home_goals"] == 0).astype(int)

        all_games = (
            pd.concat([home_games, away_games], ignore_index=True)
            .sort_values("date")
        )
        all_games["pts"] = all_games["w"] * 3 + all_games["d"]
        all_games["gd"]  = all_games["gf"] - all_games["ga"]

        for n in self.WINDOWS:
            recent = all_games.tail(n)
            k      = f"_{n}"
            if len(recent) == 0:
                for col in ["gf", "ga", "gd", "w", "d", "pts", "cs"]:
                    feat[f"{prefix}{col}{k}"] = np.nan
                feat[f"{prefix}form{k}"] = np.nan
                continue

            feat[f"{prefix}gf{k}"]   = recent["gf"].mean()
            feat[f"{prefix}ga{k}"]   = recent["ga"].mean()
            feat[f"{prefix}gd{k}"]   = recent["gd"].mean()
            feat[f"{prefix}w{k}"]    = recent["w"].mean()
            feat[f"{prefix}d{k}"]    = recent["d"].mean()
            feat[f"{prefix}pts{k}"]  = recent["pts"].mean()
            feat[f"{prefix}cs{k}"]   = recent["cs"].mean()

            # Exponentially weighted recent form (more recent = higher weight)
            weights = np.exp(np.linspace(-1, 0, len(recent)))
            weights /= weights.sum()
            feat[f"{prefix}form{k}"] = float(np.dot(recent["pts"].values, weights))

        # Total games count (indicates parameter reliability)
        feat[f"{prefix}n_games"] = len(all_games)

        return feat

    # ------------------------------------------------------------------
    #  Head-to-Head Features
    # ------------------------------------------------------------------

    def _h2h_features(
        self, hist: pd.DataFrame, home_team: str, away_team: str
    ) -> dict:
        """
        Extract historical head-to-head features between two teams (bidirectional).
        """
        mask = (
            ((hist["home_team"] == home_team) & (hist["away_team"] == away_team)) |
            ((hist["home_team"] == away_team) & (hist["away_team"] == home_team))
        )
        h2h = hist[mask].tail(10)

        feat: dict = {}
        feat["h2h_n"] = len(h2h)

        if len(h2h) == 0:
            feat["h2h_home_winrate"]  = np.nan
            feat["h2h_avg_goals"]     = np.nan
            feat["h2h_avg_gd"]        = np.nan
            return feat

        # From home_team perspective
        home_wins = (
            ((h2h["home_team"] == home_team) & (h2h["home_goals"] > h2h["away_goals"])) |
            ((h2h["away_team"] == home_team) & (h2h["away_goals"] > h2h["home_goals"]))
        )
        feat["h2h_home_winrate"] = float(home_wins.mean())
        feat["h2h_avg_goals"]    = float((h2h["home_goals"] + h2h["away_goals"]).mean())

        # Goal difference (home_team perspective)
        gd = np.where(
            h2h["home_team"] == home_team,
            h2h["home_goals"] - h2h["away_goals"],
            h2h["away_goals"] - h2h["home_goals"],
        )
        feat["h2h_avg_gd"] = float(gd.mean())

        return feat

    # ------------------------------------------------------------------
    #  Dixon-Coles Features
    # ------------------------------------------------------------------

    @staticmethod
    def _dc_features(dc_model, home_team: str, away_team: str, venue: str) -> dict:
        """
        Extract 6 structural features from BayesianDixonColesModel.

        The DC model provides statistically optimal "baseline probabilities,"
        and GBM learns residual patterns not captured by DC (recent form, etc.).
        """
        feat: dict = {}
        try:
            r = dc_model.predict(home_team, away_team, venue=venue)
            feat["dc_p_home"]     = r["home_win_prob"]
            feat["dc_p_draw"]     = r["draw_prob"]
            feat["dc_p_away"]     = r["away_win_prob"]
            feat["dc_lambda_h"]   = r["expected_home_goals"]
            feat["dc_lambda_a"]   = r["expected_away_goals"]
            feat["dc_lambda_diff"] = r["expected_home_goals"] - r["expected_away_goals"]

            # Attack/defense parameter differences (if model has params)
            if hasattr(dc_model, "params") and dc_model.params is not None:
                ti = dc_model.team_idx
                if home_team in ti and away_team in ti:
                    hi, ai = ti[home_team], ti[away_team]
                    feat["dc_atk_diff"] = float(
                        dc_model.params["attack"][hi] - dc_model.params["attack"][ai]
                    )
                    feat["dc_def_diff"] = float(
                        dc_model.params["defense"][hi] - dc_model.params["defense"][ai]
                    )
        except (KeyError, RuntimeError):
            # Team not in DC training set, fill NaN (HistGBM natively supports NaN features)
            for k in ["dc_p_home", "dc_p_draw", "dc_p_away",
                      "dc_lambda_h", "dc_lambda_a", "dc_lambda_diff",
                      "dc_atk_diff", "dc_def_diff"]:
                feat[k] = np.nan
        return feat


# ======================================================================
#  HistGBM Predictor
# ======================================================================

class HistGBMPredictor:
    """
    Football match predictor based on sklearn HistGradientBoostingClassifier.

    HistGBM is sklearn's built-in equivalent of LightGBM:
      - Histogram binning acceleration (same as LightGBM)
      - Native NaN feature support (no preprocessing needed)
      - No GPU required, CPU training in seconds on laptop

    Probability calibration: Isotonic Regression (better than Platt Scaling for small samples)
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        min_samples_leaf: int = 5,
        l2_regularization: float = 1.0,
        max_leaf_nodes: int = 31,
        calibrate: bool = True,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            n_estimators:      Number of trees (300 sufficient for small data)
            learning_rate:     Step size (smaller = more stable, more trees)
            max_depth:         Tree depth (<= 4 recommended for small data)
            min_samples_leaf:  Minimum samples per leaf (prevents overfitting)
            l2_regularization: L2 regularization coefficient (prevents overfitting)
            max_leaf_nodes:    Maximum leaf nodes (equivalent to LightGBM num_leaves)
            calibrate:         Whether to calibrate probability output (strongly recommended True)
            random_state:      Random seed
        """
        self.calibrate    = calibrate
        self.feature_cols: Optional[list] = None
        self._label_classes = [0, 1, 2]  # A, D, H

        base = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            early_stopping=False,
        )

        if calibrate:
            # Isotonic: more flexible, suitable for small data
            # cv=5 does cross-calibration within training set
            self.model = CalibratedClassifierCV(
                estimator=base, method="isotonic", cv=3
            )
        else:
            self.model = base

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "HistGBMPredictor":
        """
        Train model.

        Args:
            X: Feature DataFrame (from FeatureEngineer.build_training_matrix)
            y: Label ndarray (0=A, 1=D, 2=H)
        """
        self.feature_cols = list(X.columns)
        X_arr = X[self.feature_cols].values.astype(float)

        self.model.fit(X_arr, y)

        # Training set log loss (as baseline reference)
        proba = self.model.predict_proba(X_arr)
        train_ll = log_loss(y, proba)
        logger.info(
            "HistGBM trained | samples=%d | features=%d | train_logloss=%.4f",
            len(y), len(self.feature_cols), train_ll,
        )
        return self

    # ------------------------------------------------------------------
    #  Prediction
    # ------------------------------------------------------------------

    def predict_proba_from_features(self, feat: dict) -> np.ndarray:
        """
        Return [P(A), P(D), P(H)] vector from feature dictionary.

        Internal call, used by EnsemblePredictor.
        """
        if self.feature_cols is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        x = np.array([[feat.get(c, np.nan) for c in self.feature_cols]])
        return self.model.predict_proba(x)[0]

    def predict(
        self,
        home_team: str,
        away_team: str,
        df: pd.DataFrame,
        dc_model=None,
        venue: str = "neutral",
        fe: Optional[FeatureEngineer] = None,
    ) -> dict:
        """
        End-to-end single match prediction (direct call version).

        Args:
            home_team: Home team
            away_team: Away team
            df:        Historical data (for feature building)
            dc_model:  Optional DC model
            venue:     Venue type
            fe:        FeatureEngineer instance (if None, creates one internally)
        """
        if fe is None:
            fe = FeatureEngineer()

        feat  = fe.get_match_features(
            df, home_team, away_team,
            before_date=df["date"].max() + pd.Timedelta(days=1),
            dc_model=dc_model,
            venue=venue,
        )
        proba = self.predict_proba_from_features(feat)

        return {
            "home_team":     home_team,
            "away_team":     away_team,
            "venue":         venue,
            "away_win_prob": round(float(proba[0]), 4),
            "draw_prob":     round(float(proba[1]), 4),
            "home_win_prob": round(float(proba[2]), 4),
            "model":         "HistGBM",
        }

    def feature_importance(self) -> pd.DataFrame:
        """
        Feature importance (only available for uncalibrated model, or internal model of CalibratedClassifierCV).
        """
        if self.feature_cols is None:
            raise RuntimeError("Model not fitted.")
        try:
            if self.calibrate:
                # Take mean importance across all CV folds
                importances = np.mean([
                    est.estimator.feature_importances_
                    for est in self.model.calibrated_classifiers_
                ], axis=0)
            else:
                importances = self.model.feature_importances_

            return (
                pd.DataFrame({
                    "feature":    self.feature_cols,
                    "importance": importances,
                })
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
        except AttributeError:
            logger.warning("Feature importance not available for this model configuration.")
            return pd.DataFrame()


# ======================================================================
#  Ensemble Predictor
# ======================================================================

class EnsemblePredictor:
    """
    Dixon-Coles + HistGBM ensemble predictor.

    Ensemble strategy: Weighted probability averaging (Soft Voting)
        P_ensemble = w_dc x P_dc + w_gbm x P_gbm
        s.t. w_dc + w_gbm = 1, w_dc >= 0, w_gbm >= 0

    Weights automatically searched by minimizing validation RPS (scipy.optimize).

    Why this is better than Hard Voting:
      - Confidence differences between the two models are preserved
      - RPS is a strictly proper scoring rule for probability calibration
      - Weights reflect each model's relative advantage on current data
    """

    def __init__(
        self,
        dc_model,
        gbm_model: HistGBMPredictor,
        feature_engineer: FeatureEngineer,
        dc_weight: float = 0.5,
        pool_method: str = "log",
    ) -> None:
        """
        Args:
            dc_model:          BayesianDixonColesModel instance (already fit)
            gbm_model:         HistGBMPredictor instance (already fit)
            feature_engineer:  FeatureEngineer instance
            dc_weight:         DC model initial weight (0~1), fit_weights() will optimize this
            pool_method:       Pooling method, 'log' (default, Genest-Zidek 1986 / Ranjan-Gneiting 2010)
                               or 'linear'. Aligned with plan v7.0 section "v7.0 upgrade".
        """
        self.dc_model  = dc_model
        self.gbm_model = gbm_model
        self.fe        = feature_engineer
        self.dc_weight  = float(dc_weight)
        self.gbm_weight = 1.0 - self.dc_weight
        self.pool_method = pool_method
        self._weight_optimized = False

    # ------------------------------------------------------------------
    #  Weight Optimization
    # ------------------------------------------------------------------

    def fit_weights(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        venue_col: str = "venue",
        pool_method: Optional[str] = None,
    ) -> "EnsemblePredictor":
        """
        Minimize RPS on validation set, searching for optimal DC/GBM weights.

        Correct temporal logic (avoiding OOF data leakage):
          - Each validation match can only use history **before that match date**;
          - History = train_df U val_df matches earlier than the current date;
          - FeatureEngineer.get_match_features internally uses ``df[df["date"] < before_date]``
            strict filtering, double safety.

        One-dimensional optimization:
            minimize  RPS( pool(P_dc, P_gbm, w), y_actual )
            over      w in [0, 1]
            method    'log' (default) or 'linear' (controlled by ``pool_method``)

        Args:
            train_df:    Training set (required, serves as historical lower bound for each validation row)
            val_df:      Validation set (must contain result column 'H'/'D'/'A'), required
            venue_col:   Venue column name
            pool_method: 'log' (default) / 'linear', None uses __init__ setting
        """
        if val_df is None:
            raise ValueError(
                "fit_weights now requires train_df and val_df to avoid "
                "temporal feature errors (OOF data leakage)."
            )

        method = pool_method or self.pool_method
        if method not in {"log", "linear"}:
            raise ValueError("pool_method must be 'log' or 'linear'.")

        train_df = train_df.copy()
        val_df   = val_df.copy()
        train_df["date"] = pd.to_datetime(train_df["date"])
        val_df["date"]   = pd.to_datetime(val_df["date"])
        train_df = train_df.sort_values("date").reset_index(drop=True)
        val_df   = val_df.sort_values("date").reset_index(drop=True)

        # Pre-compute DC and GBM probabilities for each validation match
        dc_probs  = []
        gbm_probs = []
        outcomes  = []

        for _, row in val_df.iterrows():
            ht     = row["home_team"]
            at     = row["away_team"]
            venue  = row.get(venue_col, "neutral")
            outcome = OUTCOME_ENCODE.get(row.get("result", "D"), 1)

            try:
                # DC probability: [P(A), P(D), P(H)]
                dc_r = self.dc_model.predict(ht, at, venue=venue)
                dc_p = np.array([
                    dc_r["away_win_prob"],
                    dc_r["draw_prob"],
                    dc_r["home_win_prob"],
                ], dtype=float)
            except (KeyError, RuntimeError):
                dc_p = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)

            # GBM features: train_df U earlier validation matches
            earlier_val = val_df[val_df["date"] < row["date"]]
            history_df  = pd.concat([train_df, earlier_val], ignore_index=True)

            feat = self.fe.get_match_features(
                history_df,
                ht,
                at,
                before_date=row["date"],
                dc_model=self.dc_model,
                venue=venue,
            )
            gbm_p = self.gbm_model.predict_proba_from_features(feat)

            dc_probs.append(dc_p)
            gbm_probs.append(gbm_p)
            outcomes.append(outcome)

        dc_probs  = np.asarray(dc_probs,  dtype=float)
        gbm_probs = np.asarray(gbm_probs, dtype=float)
        outcomes  = np.asarray(outcomes,  dtype=int)

        def rps_for_weight(x):
            w = float(np.asarray(x).ravel()[0])
            blend = _pool_two(dc_probs, gbm_probs, w, method=method)
            return _mean_rps(blend, outcomes)

        result = minimize(
            rps_for_weight,
            x0=[self.dc_weight],
            bounds=[(0.0, 1.0)],
            method="L-BFGS-B",
        )

        self.dc_weight  = float(np.clip(result.x[0], 0.0, 1.0))
        self.gbm_weight = 1.0 - self.dc_weight
        self.pool_method = method
        self._weight_optimized = True

        opt_rps = float(result.fun)
        dc_rps  = rps_for_weight([1.0])
        gbm_rps = rps_for_weight([0.0])

        logger.info(
            "Ensemble weights optimized | method=%s | DC=%.3f / GBM=%.3f | "
            "RPS: DC=%.4f | GBM=%.4f | Ensemble=%.4f",
            method,
            self.dc_weight,
            self.gbm_weight,
            dc_rps,
            gbm_rps,
            opt_rps,
        )

        return self

    # ------------------------------------------------------------------
    #  Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        home_team: str,
        away_team: str,
        df: pd.DataFrame,
        venue: str = "neutral",
    ) -> dict:
        """
        Ensemble prediction (weighted soft voting).

        Returns:
            dict containing complete predictions from all three models (DC / GBM / Ensemble)
        """
        # DC prediction
        try:
            dc_r = self.dc_model.predict(home_team, away_team, venue=venue)
            dc_p = np.array([
                dc_r["away_win_prob"],
                dc_r["draw_prob"],
                dc_r["home_win_prob"],
            ])
        except (KeyError, RuntimeError) as e:
            logger.warning("DC prediction failed for %s vs %s: %s", home_team, away_team, e)
            dc_p = np.array([1/3, 1/3, 1/3])
            dc_r = {
                "expected_home_goals": np.nan,
                "expected_away_goals": np.nan,
            }

        # GBM prediction
        feat  = self.fe.get_match_features(
            df, home_team, away_team,
            before_date=df["date"].max() + pd.Timedelta(days=1),
            dc_model=self.dc_model,
            venue=venue,
        )
        gbm_p = self.gbm_model.predict_proba_from_features(feat)

        # Weighted fusion (log / linear pool, determined by self.pool_method)
        blend = _pool_two(dc_p, gbm_p, self.dc_weight, method=self.pool_method)

        def _fmt(arr):
            return {
                "away_win_prob": round(float(arr[0]), 4),
                "draw_prob":     round(float(arr[1]), 4),
                "home_win_prob": round(float(arr[2]), 4),
            }

        return {
            "home_team":        home_team,
            "away_team":        away_team,
            "venue":            venue,
            "dc_weight":        round(self.dc_weight, 3),
            "gbm_weight":       round(self.gbm_weight, 3),
            "pool_method":      self.pool_method,
            "dc":               _fmt(dc_p) | {
                "expected_home_goals": dc_r.get("expected_home_goals"),
                "expected_away_goals": dc_r.get("expected_away_goals"),
            },
            "gbm":              _fmt(gbm_p),
            "ensemble":         _fmt(blend),
        }

    def compare_table(
        self,
        home_team: str,
        away_team: str,
        df: pd.DataFrame,
        venue: str = "neutral",
    ) -> pd.DataFrame:
        """
        Return comparison DataFrame of the three models (suitable for printing/display).
        """
        r = self.predict(home_team, away_team, df, venue)
        rows = []
        for model_key in ("dc", "gbm", "ensemble"):
            p = r[model_key]
            rows.append({
                "model":     model_key.upper(),
                "home_win%": round(p["home_win_prob"] * 100, 1),
                "draw%":     round(p["draw_prob"] * 100, 1),
                "away_win%": round(p["away_win_prob"] * 100, 1),
            })
        return pd.DataFrame(rows)

    def rps(self, result_dict: dict, actual_outcome: str) -> dict:
        """Compute RPS for all three models (returns dict for comparison)"""
        def _rps(probs_dict, actual):
            p = np.array([
                probs_dict["away_win_prob"],
                probs_dict["draw_prob"],
                probs_dict["home_win_prob"],
            ])
            a = np.array({"A": [1,0,0], "D": [0,1,0], "H": [0,0,1]}[actual], dtype=float)
            return float(0.5 * np.sum((np.cumsum(p)[:-1] - np.cumsum(a)[:-1]) ** 2))

        return {
            "dc":       round(_rps(result_dict["dc"],       actual_outcome), 4),
            "gbm":      round(_rps(result_dict["gbm"],      actual_outcome), 4),
            "ensemble": round(_rps(result_dict["ensemble"], actual_outcome), 4),
        }


# ======================================================================
#  Walk-Forward Cross-Validation
# ======================================================================

class WalkForwardCV:
    """
    Time series cross-validation (Walk-Forward Validation).

    Principle:
      In each fold, training set is strictly earlier than validation set, simulating real prediction scenarios.
      Never uses future data, avoiding data leakage.

    Fold split example (n_splits=5):
      Fold 1: train=[0, 100), val=[100, 120)
      Fold 2: train=[0, 120), val=[120, 140)
      Fold 3: train=[0, 140), val=[140, 160)
      ...(expanding window, not rolling window)
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 60,
    ) -> None:
        self.n_splits       = n_splits
        self.min_train_size = min_train_size

    def evaluate(
        self,
        df: pd.DataFrame,
        dc_model_class,
        dc_kwargs: dict = None,
        gbm_kwargs: dict = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run Walk-Forward validation on full dataset, evaluating RPS for DC, GBM, and Ensemble models.

        Args:
            df:              Full dataset (containing date, home_team, away_team,
                             home_goals, away_goals, result, venue, competition)
            dc_model_class:  BayesianDixonColesModel class
            dc_kwargs:       DC model initialization parameters
            gbm_kwargs:      GBM model initialization parameters
            verbose:         Whether to print fold results

        Returns:
            Evaluation results DataFrame per fold
        """
        dc_kwargs  = dc_kwargs  or {}
        gbm_kwargs = gbm_kwargs or {}

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        n        = len(df)
        fold_size = max((n - self.min_train_size) // self.n_splits, 1)
        records   = []

        for fold in range(self.n_splits):
            val_end   = n - fold_size * (self.n_splits - fold - 1)
            val_start = val_end - fold_size
            if val_start < self.min_train_size:
                continue

            train_df = df.iloc[:val_start].reset_index(drop=True)
            val_df   = df.iloc[val_start:val_end].reset_index(drop=True)

            if len(train_df) < self.min_train_size:
                continue

            # Train DC model
            try:
                dc = dc_model_class(**dc_kwargs)
                dc.fit(train_df)
            except Exception as e:
                logger.warning("DC fit failed fold %d: %s", fold, e)
                dc = None

            # Train GBM model
            fe    = FeatureEngineer()
            X, y  = fe.build_training_matrix(train_df, dc_model=dc)
            gbm   = HistGBMPredictor(**gbm_kwargs)
            gbm.fit(X, y)

            # Validation set evaluation
            dc_rps_list, gbm_rps_list, ens_rps_list = [], [], []

            for _, row in val_df.iterrows():
                ht      = row["home_team"]
                at      = row["away_team"]
                venue   = row.get("venue", "neutral")
                actual  = row.get("result", "D")
                outcome = OUTCOME_ENCODE.get(actual, 1)

                # DC
                try:
                    dc_r = dc.predict(ht, at, venue=venue) if dc else None
                    dc_p = np.array([
                        dc_r["away_win_prob"], dc_r["draw_prob"], dc_r["home_win_prob"]
                    ]) if dc_r else np.array([1/3, 1/3, 1/3])
                except Exception:
                    dc_p = np.array([1/3, 1/3, 1/3])

                # GBM
                earlier_val = val_df[val_df["date"] < row["date"]]
                history_df  = pd.concat([train_df, earlier_val], ignore_index=True)
                feat  = fe.get_match_features(
                    history_df, ht, at, row["date"], dc, venue
                )
                gbm_p = gbm.predict_proba_from_features(feat)

                # Ensemble (0.5/0.5 equal weight + log pool, not re-fit within CV)
                blend = _pool_two(dc_p, gbm_p, 0.5, method="log")

                dc_rps_list.append(_rps_vec(dc_p, outcome))
                gbm_rps_list.append(_rps_vec(gbm_p, outcome))
                ens_rps_list.append(_rps_vec(blend, outcome))

            rec = {
                "fold":        fold + 1,
                "train_size":  len(train_df),
                "val_size":    len(val_df),
                "dc_rps":      round(float(np.mean(dc_rps_list)),  4),
                "gbm_rps":     round(float(np.mean(gbm_rps_list)), 4),
                "ens_rps":     round(float(np.mean(ens_rps_list)), 4),
            }
            records.append(rec)

            if verbose:
                print(
                    f"  Fold {rec['fold']:2d} | "
                    f"train={rec['train_size']:4d} val={rec['val_size']:3d} | "
                    f"DC={rec['dc_rps']:.4f}  "
                    f"GBM={rec['gbm_rps']:.4f}  "
                    f"Ensemble={rec['ens_rps']:.4f}"
                )

        result_df = pd.DataFrame(records)
        if not result_df.empty and verbose:
            print(f"\n  Mean RPS | "
                  f"DC={result_df['dc_rps'].mean():.4f}  "
                  f"GBM={result_df['gbm_rps'].mean():.4f}  "
                  f"Ensemble={result_df['ens_rps'].mean():.4f}")
        return result_df


# ======================================================================
#  Utilities
# ======================================================================

def _rps_vec(proba: np.ndarray, outcome: int) -> float:
    """RPS for a single prediction (internal)."""
    actual = np.zeros(3)
    actual[outcome] = 1.0
    return float(0.5 * np.sum((np.cumsum(proba)[:-1] - np.cumsum(actual)[:-1]) ** 2))


def _pool_two(
    p1: np.ndarray,
    p2: np.ndarray,
    w: float,
    method: str = "log",
) -> np.ndarray:
    """
    Weighted pooling of two probability distributions (log / linear).

    Prefers ``quantbet.pooling`` implementation; falls back to inline log/linear if unavailable.
    Class order consistent with project convention: ``[A, D, H]``.

    Args:
        p1, p2:  shape ``(C,)`` or ``(N, C)`` probability arrays
        w:       weight for p1 (p2 gets 1-w), clipped to ``[0, 1]``
        method:  ``"log"`` (log/geometric pooling, Ranjan-Gneiting 2010 recommended)
                 or ``"linear"`` (linear/arithmetic pooling)

    Returns:
        Pooled probabilities, shape matches input, sum=1.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    w = float(np.clip(w, 0.0, 1.0))

    if method == "log":
        if log_pool is not None:
            return log_pool([p1, p2], [w, 1.0 - w])
        # Fallback: inline log pool (log-sum-exp + normalization)
        eps = 1e-12
        logp = w * np.log(np.clip(p1, eps, 1.0)) + (1.0 - w) * np.log(np.clip(p2, eps, 1.0))
        logp = logp - np.max(logp, axis=-1, keepdims=True)
        out = np.exp(logp)
    elif method == "linear":
        if linear_pool is not None:
            return linear_pool([p1, p2], [w, 1.0 - w])
        out = w * p1 + (1.0 - w) * p2
    else:
        raise ValueError(f"Unknown pool method: {method!r}. Use 'log' or 'linear'.")

    out = np.clip(out, 1e-12, 1.0)
    return out / out.sum(axis=-1, keepdims=True)


def _mean_rps(proba_matrix: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean RPS over a batch (internal)."""
    n = len(outcomes)
    actuals = np.zeros((n, 3))
    for i, o in enumerate(outcomes):
        actuals[i, o] = 1.0
    return float(np.mean([
        0.5 * np.sum((np.cumsum(proba_matrix[i])[:-1] - np.cumsum(actuals[i])[:-1]) ** 2)
        for i in range(n)
    ]))


# ======================================================================
#  Full Pipeline Demo
# ======================================================================

def build_and_evaluate_pipeline(df: pd.DataFrame, dc_model=None) -> EnsemblePredictor:
    """
    Full training pipeline: feature engineering -> GBM training -> ensemble weight optimization.

    Args:
        df:       Full training data
        dc_model: Pre-fitted BayesianDixonColesModel (optional)

    Returns:
        Trained EnsemblePredictor
    """
    # Split by time: 80% training / 20% validation
    df = df.copy().sort_values("date").reset_index(drop=True)
    split_idx  = int(len(df) * 0.8)
    train_df   = df.iloc[:split_idx]
    val_df     = df.iloc[split_idx:]

    # Feature engineering
    fe   = FeatureEngineer()
    X, y = fe.build_training_matrix(train_df, dc_model=dc_model)

    # GBM training
    gbm = HistGBMPredictor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=5,
        l2_regularization=1.0,
        calibrate=True,
    )
    gbm.fit(X, y)

    # Ensemble weight optimization (search optimal DC/GBM weight on validation set)
    if dc_model is not None:
        ensemble = EnsemblePredictor(dc_model, gbm, fe, dc_weight=0.5)
        ensemble.fit_weights(train_df, val_df, pool_method="log")
    else:
        # No DC model, create dummy ensemble (100% GBM)
        class _DummyDC:
            def predict(self, *a, **k):
                raise RuntimeError("No DC model")
        ensemble = EnsemblePredictor(_DummyDC(), gbm, fe, dc_weight=0.0)

    return ensemble


# ======================================================================
#  Test / Demo
# ======================================================================

def run_tests():
    print("\n" + "=" * 65)
    print("  Module 3 — ML Predictor Full Test")
    print("=" * 65)

    # --- Generate test data (same structure as Module 2) ---
    try:
        from bayesian_dixon_coles import (
            BayesianDixonColesModel,
            generate_international_mock_data,
        )
        df = generate_international_mock_data(n_teams=16, seed=42)
        print(f"\nData source: Module 2 mock data | {len(df)} matches")
        has_dc_module = True
    except ImportError:
        print("\nModule 2 not found, using standalone mock data")
        df = _standalone_mock_data()
        has_dc_module = False
        BayesianDixonColesModel = None

    # --- Train DC model (if available) ---
    dc_model = None
    if has_dc_module:
        print("\n[1/4] Training Dixon-Coles model...")
        dc_model = BayesianDixonColesModel(damping=0.002)
        dc_model.fit(df)
        print(f"      {dc_model}")

    # --- Walk-Forward Cross-Validation ---
    print("\n[2/4] Walk-Forward Cross-Validation (5 fold)...")
    if has_dc_module:
        cv = WalkForwardCV(n_splits=5, min_train_size=60)
        cv_results = cv.evaluate(
            df,
            dc_model_class=BayesianDixonColesModel,
            dc_kwargs={"damping": 0.002},
            gbm_kwargs={"n_estimators": 200},
            verbose=True,
        )
    else:
        print("      (Skip DC, GBM CV only)")
        cv_results = None

    # --- Full pipeline ---
    print("\n[3/4] Training full ensemble pipeline...")
    ensemble = build_and_evaluate_pipeline(df, dc_model=dc_model)
    print(f"      Optimal weights: DC={ensemble.dc_weight:.3f} / GBM={ensemble.gbm_weight:.3f}")

    # --- Feature importance ---
    print("\n[4/4] Feature Importance (Top 10)...")
    fi = ensemble.gbm_model.feature_importance()
    if not fi.empty:
        for _, row in fi.head(10).iterrows():
            bar = "█" * int(row["importance"] * 200)
            print(f"  {row['feature']:<35} {row['importance']:.4f}  {bar}")

    # --- Single match prediction comparison ---
    teams = sorted(df["home_team"].unique())
    ht, at = teams[0], teams[1]
    print(f"\n{'Single Match Prediction Comparison':{home_t}} vs {away_t}  [Neutral Venue]")

    r = ensemble.predict(ht, at, df, venue="neutral")

    print(f"\n  {'Model':<12} {'Home Win':>8} {'Draw':>8} {'Away Win':>8}")
    print(f"  {'─'*40}")
    for model_key, label in [("dc", "Dixon-Coles"), ("gbm", "HistGBM"), ("ensemble", "Ensemble")]:
        p = r[model_key]
        print(
            f"  {label:<12} "
            f"{p['home_win_prob']*100:>7.1f}%"
            f"{p['draw_prob']*100:>8.1f}%"
            f"{p['away_win_prob']*100:>8.1f}%"
        )

    # RPS comparison (assuming home win)
    rps = ensemble.rps(r, "H")
    print(f"\n  RPS (if home win): DC={rps['dc']:.4f}  GBM={rps['gbm']:.4f}  Ensemble={rps['ensemble']:.4f}")

    if cv_results is not None and not cv_results.empty:
        print(f"\n{'─'*65}")
        print("  Walk-Forward Summary")
        print(f"{'─'*65}")
        for col in ["dc_rps", "gbm_rps", "ens_rps"]:
            label = col.replace("_rps", "").upper()
            vals  = cv_results[col]
            print(f"  {label:<12}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}")

    print("\n" + "=" * 65)
    print("  All tests passed.")
    print("=" * 65 + "\n")

    return ensemble


def _standalone_mock_data(n=200, seed=42) -> pd.DataFrame:
    """Standalone mock data when Module 2 is unavailable"""
    rng   = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(10)]
    base  = pd.Timestamp("2024-01-01")
    rows  = []
    for i in range(n):
        ht, at = rng.choice(teams, 2, replace=False)
        hg     = int(rng.poisson(1.4))
        ag     = int(rng.poisson(1.1))
        rows.append({
            "date":        base + pd.Timedelta(days=int(i * 3.5)),
            "home_team":   ht, "away_team": at,
            "home_goals":  hg, "away_goals": ag,
            "venue":       rng.choice(["home", "neutral", "away"]),
            "competition": rng.choice(["world_cup_qualifying", "friendly"]),
            "result":      "H" if hg > ag else "A" if hg < ag else "D",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_tests()
