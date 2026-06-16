"""
QuantBet-EV: Module 2 - Mathematical Probability Engine (Improved)
Dixon-Coles Bivariate Poisson Model — Enhanced Edition

Fix items (corresponding to code review):
  [FIX-1] Tau low-score correction factor + rho parameter MLE estimation
  [FIX-2] Model identifiability constraint sum(attack) = 0
  [FIX-3] Temporal decay weight formally written into likelihood function
  [FIX-4] Vectorized log-likelihood (gammaln), replacing for loop
  [FIX-5] Analytical score matrix prediction, replacing Monte Carlo

Academic paper new features:
  [NEW-1] xG mode (Wilkens 2026) — expected goals replacing actual goals
  [NEW-2] DynamicDixonColesModel — incremental parameter updates (Rue & Salvesen 2000)
  [NEW-3] AIC / BIC / RPS model evaluation metrics
  [NEW-4] team_ratings() — attack/defense strength visualization interface

References:
  Dixon & Coles (1997) Applied Statistics 46, 265-280
  Rue & Salvesen (2000) Scandinavian Journal of Statistics 27, 385-402
  Karlis & Ntzoufras (2003) JRSS Series D 52, 381-393
  Koopman & Lit (2015) JRSS Series A 178, 167-186
  Wilkens (2026) SAGE Sports Analytics
  Epstein (1969) Journal of Applied Meteorology 8, 985-987
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

logger = logging.getLogger(__name__)


# ======================================================================
#  Core Model
# ======================================================================

class DixonColesModel:
    """
    Improved Dixon-Coles bivariate Poisson model.

    Core improvements over the original implementation:
      - Tau correction (low-score joint probability correction)
      - Parameter identifiability constraint (sum-to-zero)
      - Temporal decay weights (formally integrated into likelihood function)
      - Vectorized MLE (gammaln)
      - Analytical prediction (no Monte Carlo)
      - AIC / BIC / RPS evaluation
      - xG input support

    Usage:
        model = DixonColesModel(damping=0.002)
        model.fit(df)
        result = model.predict("Manchester City", "Liverpool")
    """

    def __init__(
        self,
        damping: float = 0.0,
        use_xg: bool = False,
        max_goals: int = 10,
    ) -> None:
        """
        Args:
            damping:   Temporal decay factor xi, weight w_i = exp(-xi * days_ago).
                       0 = no decay; recommended starting value 0.002 (weight drops to 50% after ~1 year).
                       Requires df to contain 'date' column to take effect.
            use_xg:    When True, uses home_xg / away_xg as lambda target instead of actual goals.
                       Requires df to contain 'home_xg', 'away_xg' columns.
            max_goals: Score matrix truncation value, suggest 10 (probability of >10 goals < 0.001%).
        """
        self.damping   = damping
        self.use_xg    = use_xg
        self.max_goals = max_goals

        self.params:    Optional[dict] = None
        self.teams:     Optional[list] = None
        self.team_idx:  Optional[dict] = None
        self._fit_info: dict = {}

    # ------------------------------------------------------------------
    #  [FIX-1] τ (Tau) Correction
    #  Dixon & Coles (1997), Section 3
    # ------------------------------------------------------------------

    @staticmethod
    def _tau(
        goals_h: int,
        goals_a: int,
        lambda_h: float,
        lambda_a: float,
        rho: float,
    ) -> float:
        """
        Dixon-Coles tau correction factor.

        Corrects systematic estimation bias of the independent Poisson model
        for low-score joint probabilities.
        Only applies correction to four low-score outcomes; tau = 1.0 for all others.

        Formula (original paper Section 3):
            tau(0,0) = 1 - lambda_h * lambda_a * rho
            tau(1,0) = 1 + lambda_a * rho
            tau(0,1) = 1 + lambda_h * rho
            tau(1,1) = 1 - rho
            tau(x,y) = 1  for all x,y >= 2

        When rho is estimated negative, corrects 0-0 downward;
        when estimated positive, corrects 1-1 downward.
        """
        if goals_h == 0 and goals_a == 0:
            return 1.0 - lambda_h * lambda_a * rho
        elif goals_h == 1 and goals_a == 0:
            return 1.0 + lambda_a * rho
        elif goals_h == 0 and goals_a == 1:
            return 1.0 + lambda_h * rho
        elif goals_h == 1 and goals_a == 1:
            return 1.0 - rho
        return 1.0

    # ------------------------------------------------------------------
    #  Data Preparation
    # ------------------------------------------------------------------

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract team list, build name-to-index mapping"""
        all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
        self.teams    = sorted(all_teams)
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        return df

    # ------------------------------------------------------------------
    #  [FIX-3] Temporal Decay Weights
    #  Dixon & Coles (1997), Section 4
    # ------------------------------------------------------------------

    def _compute_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute temporal decay weight vector.

        When damping > 0 and df contains 'date' column:
            w_i = exp(-xi * days_ago_i)
        Weights are normalized to mean 1 (keeping likelihood magnitude consistent).

        Otherwise returns all-ones vector (no decay).
        """
        if self.damping > 0.0 and "date" in df.columns:
            # Compatible with multiple date formats (ISO + 11v11 "09 Jun 2025")
            dates = pd.to_datetime(df["date"], format="mixed", dayfirst=True, errors="coerce")
            # Remove unparseable ones
            valid_mask = dates.notna()
            if not valid_mask.all():
                logger.warning("Dropped %d rows with unparseable dates", (~valid_mask).sum())
                dates = dates[valid_mask]
            t_max   = dates.max()
            days_ago = (t_max - dates).dt.days.values.astype(float)
            weights  = np.exp(-self.damping * days_ago)
            # Normalize: weight mean = 1, does not affect parameter estimation scale
            weights  = weights / weights.mean()
            return weights

        return np.ones(len(df))

    # ------------------------------------------------------------------
    #  [FIX-4] Vectorized Log-Likelihood with τ Correction
    # ------------------------------------------------------------------

    def _log_likelihood(
        self,
        params:    np.ndarray,
        goals_h:   np.ndarray,
        goals_a:   np.ndarray,
        home_idx:  np.ndarray,
        away_idx:  np.ndarray,
        weights:   np.ndarray,
    ) -> float:
        """
        Weighted negative log-likelihood function (vectorized + tau correction).

        Poisson log-probability (numerically stable form):
            log P(k; lambda) = k*log(lambda) - lambda - gammaln(k+1)

        tau correction only appended to subset where goals_h <= 1 AND goals_a <= 1:
            log L_i += log tau(x_i, y_i, lambda_h, lambda_a, rho)

        Parameter vector layout:
            params[0 : n]      -> attack  (n teams)
            params[n : 2n]     -> defense (n teams)
            params[-2]         -> home_adj
            params[-1]         -> rho
        """
        n = len(self.teams)

        attack   = params[:n]
        defense  = params[n:2 * n]
        home_adj = params[-2]
        rho      = params[-1]

        # Expected goals (log-linear model)
        lambda_h = np.exp(attack[home_idx] + defense[away_idx] + home_adj)
        lambda_a = np.exp(attack[away_idx] + defense[home_idx])

        # [FIX-4] Vectorized Poisson log-probability — replaces original for loop
        log_lik = (
            goals_h * np.log(lambda_h) - lambda_h - gammaln(goals_h + 1)
            + goals_a * np.log(lambda_a) - lambda_a - gammaln(goals_a + 1)
        )

        # [FIX-1] Tau correction: only process low-score subset (4 combinations), negligible global performance impact
        low_mask = (goals_h <= 1) & (goals_a <= 1)
        if low_mask.any():
            lh  = lambda_h[low_mask]
            la  = lambda_a[low_mask]
            gh  = goals_h[low_mask].astype(int)
            ga  = goals_a[low_mask].astype(int)

            tau_vals = np.array([
                self._tau(int(h), int(a), float(lh_), float(la_), rho)
                for h, a, lh_, la_ in zip(gh, ga, lh, la)
            ])

            # Prevent tau <= 0 causing log domain error (very rare after constraining rho range)
            tau_vals = np.clip(tau_vals, 1e-12, None)
            log_lik[low_mask] += np.log(tau_vals)

        # Take negative after weighted sum (minimization)
        return float(-np.dot(weights, log_lik))

    # ------------------------------------------------------------------
    #  [FIX-2] Model Fitting with Identification Constraint
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DixonColesModel":
        """
        Maximum likelihood estimation with constraints.

        Fix points:
          [FIX-2] Equality constraint sum attack_i = 0, resolving parameter non-identifiability.
                  Uses SLSQP optimizer (supports equality constraints), replacing original L-BFGS-B.
          [FIX-3] Temporal decay weights formally participate in weighted likelihood.
          rho constrained within (-0.99, 0.99) to ensure tau correction values are positive.

        Args:
            df: Training data DataFrame.
                Required columns: home_team, away_team, home_goals, away_goals
                Optional columns: date (needed for damping), home_xg / away_xg (xG mode)

        Returns:
            self (supports chaining)
        """
        df = self._prepare_data(df)
        n  = len(self.teams)

        # Data extraction
        home_idx = df["home_team"].map(self.team_idx).values
        away_idx = df["away_team"].map(self.team_idx).values
        weights  = self._compute_weights(df)

        # [NEW-1] xG mode: use expected goals instead of actual goals
        if self.use_xg and {"home_xg", "away_xg"}.issubset(df.columns):
            goals_h = df["home_xg"].values.astype(float)
            goals_a = df["away_xg"].values.astype(float)
            logger.info("xG mode: using home_xg / away_xg as target variable")
        else:
            goals_h = df["home_goals"].values.astype(float)
            goals_a = df["away_goals"].values.astype(float)

        # Initial parameter vector: [attack x n, defense x n, home_adj, rho]
        init_params = np.concatenate([
            np.zeros(n),   # attack
            np.zeros(n),   # defense
            [0.1, -0.1],   # home_adj, rho
        ])

        # [FIX-2] Equality constraint: sum(attack) = 0
        constraints = [{
            "type": "eq",
            "fun": lambda p: np.sum(p[:n]),
        }]

        # Parameter bounds: only apply hard bounds to rho
        bounds = (
            [(None, None)] * n       # attack: unbounded
            + [(None, None)] * n     # defense: unbounded
            + [(None, None)]         # home_adj: unbounded
            + [(-0.99, 0.99)]        # rho: bounded, ensure tau > 0
        )

        result = minimize(
            self._log_likelihood,
            init_params,
            args=(goals_h, goals_a, home_idx, away_idx, weights),
            method="SLSQP",           # Supports equality constraints
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning("Optimization warning: %s", result.message)

        # Store fitted parameters
        self.params = {
            "attack":   result.x[:n],
            "defense":  result.x[n:2 * n],
            "home_adj": float(result.x[-2]),
            "rho":      float(result.x[-1]),
        }

        # [NEW-3] Compute AIC / BIC
        #   Free parameter count = 2n + 2 - 1 (minus sum-to-zero constraint)
        n_free   = 2 * n + 2 - 1
        n_obs    = len(df)
        log_lik  = -result.fun

        self._fit_info = {
            "n_teams":   n,
            "n_matches": n_obs,
            "log_lik":   round(log_lik, 4),
            "aic":       round(2 * n_free - 2 * log_lik, 4),
            "bic":       round(np.log(n_obs) * n_free - 2 * log_lik, 4),
            "converged": result.success,
            "rho":       self.params["rho"],
            "home_adj":  self.params["home_adj"],
        }

        logger.info(
            "Model fitted | teams=%d | matches=%d | LL=%.2f | "
            "AIC=%.2f | rho=%.4f | home_adj=%.4f",
            n, n_obs, log_lik,
            self._fit_info["aic"],
            self.params["rho"],
            self.params["home_adj"],
        )

        return self

    # ------------------------------------------------------------------
    #  [FIX-5] Analytical Prediction (Score Matrix)
    # ------------------------------------------------------------------

    def predict(self, home_team: str, away_team: str) -> dict:
        """
        Computes exact score probability matrix using analytical method.

        Calculation steps:
          1. Compute expected goals lambda_h, lambda_a for both teams using fitted parameters
          2. Build the (max_goals+1)^2 Poisson marginal distribution outer product matrix
          3. Apply tau correction to 4 low-score cells
          4. Normalize the matrix (truncation error correction)
          5. Read home win / draw / away win probabilities from triangular matrix

        The result is deterministic and approximately 100x faster than Monte Carlo.

        Args:
            home_team: Home team name (must have appeared in training data)
            away_team: Away team name (must have appeared in training data)

        Returns:
            dict containing:
              - expected_home_goals / expected_away_goals
              - home_win_prob / draw_prob / away_win_prob
              - score_probs: {score string: probability} (filtering scores < 0.5%)
              - score_matrix: numpy matrix (raw data, for further analysis)
        """
        self._check_fitted()
        self._check_team(home_team)
        self._check_team(away_team)

        h_idx = self.team_idx[home_team]
        a_idx = self.team_idx[away_team]

        lambda_h = float(np.exp(
            self.params["attack"][h_idx]
            + self.params["defense"][a_idx]
            + self.params["home_adj"]
        ))
        lambda_a = float(np.exp(
            self.params["attack"][a_idx]
            + self.params["defense"][h_idx]
        ))
        rho = self.params["rho"]

        # Build score matrix
        goals = np.arange(self.max_goals + 1)
        home_pmf = poisson.pmf(goals, lambda_h)   # shape: (max_goals+1,)
        away_pmf = poisson.pmf(goals, lambda_a)   # shape: (max_goals+1,)
        score_matrix = np.outer(home_pmf, away_pmf)

        # Apply tau correction (only 4 cells)
        for h in range(2):
            for a in range(2):
                tau = self._tau(h, a, lambda_h, lambda_a, rho)
                score_matrix[h, a] *= max(tau, 1e-12)

        # Normalize (truncation + tau correction causes slight deviation)
        total = score_matrix.sum()
        if total > 0:
            score_matrix /= total

        # Three-way classification probabilities
        home_win = float(np.tril(score_matrix, -1).sum())
        draw     = float(np.trace(score_matrix))
        away_win = float(np.triu(score_matrix, 1).sum())

        # Score dictionary (filter very low probability scores)
        score_probs = {}
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                p = score_matrix[h, a]
                if p >= 0.005:
                    score_probs[f"{h}-{a}"] = round(float(p), 4)

        return {
            "home_team":           home_team,
            "away_team":           away_team,
            "expected_home_goals": round(lambda_h, 3),
            "expected_away_goals": round(lambda_a, 3),
            "home_win_prob":       round(home_win, 4),
            "draw_prob":           round(draw, 4),
            "away_win_prob":       round(away_win, 4),
            "score_probs":         score_probs,
            "score_matrix":        score_matrix,  # Raw matrix, for downstream analysis
        }

    # ------------------------------------------------------------------
    #  [NEW-3] Model Evaluation
    # ------------------------------------------------------------------

    def rps(self, result: dict, actual_outcome: str) -> float:
        """
        Compute Ranked Probability Score (RPS).

        RPS is the standard evaluation metric for football predictions, lower is better:
          - Perfect prediction: RPS = 0.0
          - Random prediction: RPS ≈ 0.333
          - Uniform prediction (1/3, 1/3, 1/3): RPS = 0.222

        Formula (three-class):
            RPS = 0.5 * sum_{k=1}^{2} (CDF_pred[k] - CDF_actual[k])^2

        Reference: Epstein (1969); Constantinou & Fenton (2012)

        Args:
            result:         Return value of predict()
            actual_outcome: Actual result, 'H' (home win) / 'D' (draw) / 'A' (away win)

        Returns:
            RPS value (float, lower is better)
        """
        pred_probs  = np.array([
            result["home_win_prob"],
            result["draw_prob"],
            result["away_win_prob"],
        ])
        outcome_map = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}
        actual_probs = np.array(outcome_map[actual_outcome], dtype=float)

        pred_cdf   = np.cumsum(pred_probs)[:-1]    # [P(H), P(H)+P(D)]
        actual_cdf = np.cumsum(actual_probs)[:-1]

        return float(0.5 * np.sum((pred_cdf - actual_cdf) ** 2))

    def evaluate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch compute prediction RPS for a set of matches.

        df must include: home_team, away_team, result ('H'/'D'/'A')

        Returns:
            DataFrame with columns: home_win_prob, draw_prob, away_win_prob, rps
        """
        self._check_fitted()

        records = []
        for _, row in df.iterrows():
            try:
                pred = self.predict(row["home_team"], row["away_team"])
                rps  = self.rps(pred, row["result"])
                records.append({
                    "home_team":     row["home_team"],
                    "away_team":     row["away_team"],
                    "home_win_prob": pred["home_win_prob"],
                    "draw_prob":     pred["draw_prob"],
                    "away_win_prob": pred["away_win_prob"],
                    "actual":        row["result"],
                    "rps":           round(rps, 4),
                })
            except KeyError:
                logger.warning("Skipping unknown team pair: %s vs %s",
                               row["home_team"], row["away_team"])

        result_df = pd.DataFrame(records)
        if not result_df.empty:
            logger.info("Batch evaluation | matches=%d | mean_rps=%.4f",
                        len(result_df), result_df["rps"].mean())
        return result_df

    # ------------------------------------------------------------------
    #  Team Ratings
    # ------------------------------------------------------------------

    def team_ratings(self) -> pd.DataFrame:
        """
        Return attack/defense strength rating DataFrame for all teams.

        Column descriptions:
          attack:  Log attack strength (higher = stronger goal-scoring ability)
          defense: Log defense strength (lower / more negative = harder to score against)
          overall: attack - defense (higher = stronger overall)

        Returns:
            DataFrame sorted by overall in descending order
        """
        self._check_fitted()

        return (
            pd.DataFrame({
                "team":    self.teams,
                "attack":  self.params["attack"].round(4),
                "defense": self.params["defense"].round(4),
                "overall": (self.params["attack"] - self.params["defense"]).round(4),
            })
            .sort_values("overall", ascending=False)
            .reset_index(drop=True)
        )

    def fit_summary(self) -> dict:
        """Return fit summary: sample size, log-likelihood, AIC, BIC, rho, home_adj, convergence status"""
        self._check_fitted()
        return self._fit_info.copy()

    # ------------------------------------------------------------------
    #  Validation Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def _check_team(self, team: str) -> None:
        if team not in self.team_idx:
            raise KeyError(
                f"Unknown team: '{team}'. "
                f"Known teams: {self.teams}"
            )

    def __repr__(self) -> str:
        if self.params is None:
            return "DixonColesModel(not fitted)"
        i = self._fit_info
        return (
            f"DixonColesModel("
            f"teams={i['n_teams']}, matches={i['n_matches']}, "
            f"LL={i['log_lik']:.2f}, AIC={i['aic']:.2f}, "
            f"rho={i['rho']:.4f}, home_adj={i['home_adj']:.4f}, "
            f"converged={i['converged']})"
        )


# ======================================================================
#  [NEW-2] Dynamic Model — Rue & Salvesen (2000) Inspired
# ======================================================================

class DynamicDixonColesModel(DixonColesModel):
    """
    Dynamic Dixon-Coles model: supports incremental parameter updates mid-season.

    Core idea (from Rue & Salvesen 2000):
      Team attack/defense abilities are not fixed over a season, but evolve dynamically
      with match results. The original paper uses MCMC for true Bayesian dynamics;
      this implementation uses a lightweight approximation:
        - Save historical match window
        - Each update() re-runs weighted MLE on new data
        - damping ensures old data weights gradually decay

    Reference:
        Rue, H. & Salvesen, O. (2000). Prediction and retrospective analysis
        of soccer matches in a league. Scandinavian Journal of Statistics, 27, 385-402.

    Usage:
        # Before season start: fit base model on historical data
        model = DynamicDixonColesModel(damping=0.002)
        model.fit(historical_df)

        # After each round: incremental update
        model.update(round_1_results)
        model.update(round_2_results)
        ...
    """

    def __init__(
        self,
        damping: float = 0.002,
        window_matches: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            damping:        Temporal decay factor (dynamic model suggest 0.001-0.005)
            window_matches: Rolling window size (keep latest N matches), None = keep all history
            **kwargs:       Other parameters passed to DixonColesModel
        """
        super().__init__(damping=damping, **kwargs)
        self.window_matches = window_matches
        self._history: list[pd.DataFrame] = []

    def fit(self, df: pd.DataFrame) -> "DynamicDixonColesModel":
        """Initial fit, also initializes history"""
        self._history = [df.copy()]
        super().fit(df)
        return self

    def update(self, new_matches: pd.DataFrame) -> "DynamicDixonColesModel":
        """
        Incremental update: call after each round of new match results.

        Process:
          1. Append new matches to history list
          2. (Optional) Trim history window by window_matches
          3. Merge history, re-run weighted MLE with temporal decay

        Args:
            new_matches: New round of match results, same format as fit()

        Returns:
            self (supports chaining)
        """
        self._check_fitted()

        self._history.append(new_matches.copy())

        # Rolling window trim (by match count, not by round)
        if self.window_matches is not None:
            full = pd.concat(self._history, ignore_index=True)
            if len(full) > self.window_matches:
                full = full.iloc[-self.window_matches:]
                self._history = [full]

        full_df = pd.concat(self._history, ignore_index=True)
        super().fit(full_df)

        logger.info(
            "Dynamic update | new=%d | total_history=%d matches",
            len(new_matches), len(full_df),
        )
        return self

    @property
    def history_size(self) -> int:
        """Current history dataset size (number of matches)"""
        if not self._history:
            return 0
        return sum(len(d) for d in self._history)


# ======================================================================
#  Utilities
# ======================================================================

def generate_mock_data(
    n_teams: int = 10,
    n_rounds: int = 18,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate fake league data for testing.

    Args:
        n_teams:  Number of teams (must be even)
        n_rounds: Number of rounds (each team plays once per round)
        seed:     Random seed

    Returns:
        DataFrame containing date, home_team, away_team, home_goals, away_goals,
        home_xg, away_xg, result columns
    """
    assert n_teams % 2 == 0, "n_teams must be even"
    rng   = np.random.default_rng(seed)
    teams = [f"Team_{chr(65 + i)}" for i in range(n_teams)]

    # True strength differences (used to generate statistically meaningful data)
    true_attack  = rng.normal(0.0, 0.4, n_teams)
    true_defense = rng.normal(0.0, 0.3, n_teams)
    home_adv     = 0.3

    records = []
    base_date = pd.Timestamp("2024-08-10")

    for round_num in range(n_rounds):
        match_date = base_date + pd.Timedelta(weeks=round_num)
        shuffled   = rng.permutation(n_teams)

        for i in range(0, n_teams, 2):
            h, a = shuffled[i], shuffled[i + 1]
            ht, at = teams[h], teams[a]

            lh = np.exp(true_attack[h] + true_defense[a] + home_adv)
            la = np.exp(true_attack[a] + true_defense[h])

            hg = int(rng.poisson(lh))
            ag = int(rng.poisson(la))
            hx = float(round(rng.gamma(shape=lh, scale=1.0), 2))
            ax = float(round(rng.gamma(shape=la, scale=1.0), 2))

            if hg > ag:
                res = "H"
            elif hg < ag:
                res = "A"
            else:
                res = "D"

            records.append({
                "date":       match_date,
                "home_team":  ht,
                "away_team":  at,
                "home_goals": hg,
                "away_goals": ag,
                "home_xg":    hx,
                "away_xg":    ax,
                "result":     res,
            })

    return pd.DataFrame(records)


# ======================================================================
#  Tests / Demo
# ======================================================================

def test_static_model(df: pd.DataFrame) -> DixonColesModel:
    """Test static improved Dixon-Coles model"""
    print("\n" + "=" * 60)
    print("  Static DixonColesModel Test")
    print("=" * 60)

    model = DixonColesModel(damping=0.002, use_xg=False, max_goals=10)
    model.fit(df)
    print(f"\n{model}")

    # Fit summary
    s = model.fit_summary()
    print("\n── Fit Summary ──────────────────────────────────")
    print(f"  Log-Likelihood : {s['log_lik']:.4f}")
    print(f"  AIC            : {s['aic']:.4f}")
    print(f"  BIC            : {s['bic']:.4f}")
    print(f"  rho (τ)        : {s['rho']:.4f}")
    print(f"  Home Advantage : {s['home_adj']:.4f}")
    print(f"  Converged      : {s['converged']}")

    # Team ratings (Top 5)
    print("\n── Team Ratings (Top 5) ─────────────────────────")
    print(model.team_ratings().head(5).to_string(index=False))

    # Predict a match
    teams = model.teams
    home, away = teams[0], teams[1]
    result = model.predict(home, away)

    print(f"\n── Prediction: {home} vs {away} ──")
    print(f"  xGoals         : {result['expected_home_goals']:.2f} — {result['expected_away_goals']:.2f}")
    print(f"  Home Win       : {result['home_win_prob'] * 100:.1f}%")
    print(f"  Draw           : {result['draw_prob'] * 100:.1f}%")
    print(f"  Away Win       : {result['away_win_prob'] * 100:.1f}%")

    top_scores = sorted(result["score_probs"].items(), key=lambda x: -x[1])[:5]
    print(f"\n  Top 5 Scorelines:")
    for score, prob in top_scores:
        bar = "█" * int(prob * 100)
        print(f"    {score:>5}  {prob * 100:5.1f}%  {bar}")

    # RPS demo
    rps_h = model.rps(result, "H")
    rps_d = model.rps(result, "D")
    rps_a = model.rps(result, "A")
    print(f"\n  RPS if Home wins : {rps_h:.4f}")
    print(f"  RPS if Draw      : {rps_d:.4f}")
    print(f"  RPS if Away wins : {rps_a:.4f}")

    # Batch evaluation
    eval_df = df.head(20).copy()
    eval_results = model.evaluate_batch(eval_df)
    if not eval_results.empty:
        print(f"\n── Batch Evaluation (first 20 matches) ─────────")
        print(f"  Mean RPS : {eval_results['rps'].mean():.4f}")
        print(f"  Min RPS  : {eval_results['rps'].min():.4f}")
        print(f"  Max RPS  : {eval_results['rps'].max():.4f}")

    return model


def test_dynamic_model(df: pd.DataFrame) -> DynamicDixonColesModel:
    """Test dynamic incremental update model"""
    print("\n" + "=" * 60)
    print("  DynamicDixonColesModel Test (Rue & Salvesen 2000)")
    print("=" * 60)

    # Fit initial model with first 60% of matches
    split  = int(len(df) * 0.6)
    train  = df.iloc[:split].reset_index(drop=True)
    new_1  = df.iloc[split:split + 10].reset_index(drop=True)
    new_2  = df.iloc[split + 10:].reset_index(drop=True)

    dyn_model = DynamicDixonColesModel(damping=0.002, window_matches=500)
    dyn_model.fit(train)
    print(f"\nInitial fit ({len(train)} matches): {dyn_model}")

    # First incremental update
    dyn_model.update(new_1)
    print(f"After update 1 (+{len(new_1)} matches): {dyn_model}")

    # Second incremental update
    dyn_model.update(new_2)
    print(f"After update 2 (+{len(new_2)} matches): {dyn_model}")
    print(f"History size: {dyn_model.history_size} matches")

    return dyn_model


def test_xg_mode(df: pd.DataFrame) -> DixonColesModel:
    """Test parameter differences between xG mode and actual goals mode"""
    print("\n" + "=" * 60)
    print("  xG Mode vs Actual Goals Comparison")
    print("=" * 60)

    model_goals = DixonColesModel(damping=0.002, use_xg=False).fit(df)
    model_xg    = DixonColesModel(damping=0.002, use_xg=True).fit(df)

    teams = model_goals.teams
    home, away = teams[0], teams[1]

    r_goals = model_goals.predict(home, away)
    r_xg    = model_xg.predict(home, away)

    print(f"\n{home} vs {away}")
    print(f"{'Metric':<25} {'Actual Goals':>15} {'xG Mode':>15}")
    print("-" * 57)
    print(f"{'Expected Home Goals':<25} {r_goals['expected_home_goals']:>15.3f} {r_xg['expected_home_goals']:>15.3f}")
    print(f"{'Expected Away Goals':<25} {r_goals['expected_away_goals']:>15.3f} {r_xg['expected_away_goals']:>15.3f}")
    print(f"{'Home Win %':<25} {r_goals['home_win_prob']*100:>14.1f}% {r_xg['home_win_prob']*100:>14.1f}%")
    print(f"{'Draw %':<25} {r_goals['draw_prob']*100:>14.1f}% {r_xg['draw_prob']*100:>14.1f}%")
    print(f"{'Away Win %':<25} {r_goals['away_win_prob']*100:>14.1f}% {r_xg['away_win_prob']*100:>14.1f}%")
    print(f"{'rho':<25} {model_goals.fit_summary()['rho']:>15.4f} {model_xg.fit_summary()['rho']:>15.4f}")

    return model_xg


def run_all_tests() -> None:
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  QuantBet-EV Module 2 — Improved Dixon-Coles Tests")
    print("=" * 60)

    # Load or generate test data
    try:
        df = pd.read_csv("quantbet_ev/data/mock_matches.csv")
        df["date"] = pd.to_datetime(df["date"])
        print(f"\nLoaded CSV: {len(df)} matches")
    except FileNotFoundError:
        df = generate_mock_data(n_teams=10, n_rounds=18, seed=42)
        print(f"\nGenerated mock data: {len(df)} matches, "
              f"{df['home_team'].nunique()} teams")

    # Ensure result column exists
    if "result" not in df.columns:
        def _result(row):
            if row["home_goals"] > row["away_goals"]:
                return "H"
            elif row["home_goals"] < row["away_goals"]:
                return "A"
            return "D"
        df["result"] = df.apply(_result, axis=1)

    # Run tests
    test_static_model(df)
    test_dynamic_model(df)
    test_xg_mode(df)

    print("\n" + "=" * 60)
    print("  All tests completed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    run_all_tests()
