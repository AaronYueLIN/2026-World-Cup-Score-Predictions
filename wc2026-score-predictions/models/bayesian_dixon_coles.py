"""
QuantBet-EV: Module 2 - Bayesian Hierarchical Dixon-Coles Model
A Bayesian hierarchical improved version for international football (national teams)

Core architectural changes (not band-aids):
  [A] MLE -> MAP estimation: apply Bayesian priors to rho / attack / defense
      Cures the rho boundary problem (MLE underdetermined with small samples)
  [B] Hierarchical shrinkage prior: attack_i ~ N(0, sigma_atk), sigma_atk estimated from data (empirical Bayes)
      Cures extreme attack/defense parameter values from few matches
  [C] Three-level venue covariate: home / neutral / away, estimated separately
      Cures the problem of using home advantage parameters on neutral World Cup venues
  [D] Match importance weights: friendly 0.35, World Cup proper 1.00
      Cures equal weighting of matches with different significance

Parameter vector layout (total 2n+5 dimensions):
  [0:n]      attack_i       — Attack strength (shrunk by hierarchical prior)
  [n:2n]     defense_i      — Defense strength (shrunk by hierarchical prior)
  [2n]       home_adj       — Home advantage (true home)
  [2n+1]     neutral_adj    — Neutral venue slight advantage (World Cup etc.)
  [2n+2]     rho            — Tau correction parameter (prior prevents boundary)
  [2n+3]     log_sigma_atk  — Attack hierarchical std (log scale, empirical Bayes estimated)
  [2n+4]     log_sigma_def  — Defense hierarchical std (log scale, empirical Bayes estimated)

Constraint: sum(attack_i) = 0 (model identifiability)

References:
  Dixon & Coles (1997) Applied Statistics 46, 265-280
  Rue & Salvesen (2000) Scandinavian Journal of Statistics 27, 385-402
  Baio & Blangiardo (2010) Journal of Applied Statistics 37, 253-270
  Pollard & Pollard (2005) Journal of Sports Sciences 23, 487-492
  Gelman et al. (2013) Bayesian Data Analysis, Chapter 5
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.special import gammaln

import structlog

from models.exceptions import GasFitError, ModelNotFoundError, PredictionError

logger = logging.getLogger(__name__)
_log = structlog.get_logger(__name__)


# ======================================================================
#  Constants
# ======================================================================

COMPETITION_WEIGHTS: dict[str, float] = {
    "world_cup":                 1.00,
    "world_cup_qualifying":      0.85,
    "continental_championship":  0.85,
    "continental_qualifying":    0.70,
    "nations_league":            0.60,
    "friendly":                  0.35,
}

VENUE_HOME    = 2
VENUE_NEUTRAL = 1
VENUE_AWAY    = 0

_VENUE_MAP: dict[str, int] = {
    "home":    VENUE_HOME,
    "neutral": VENUE_NEUTRAL,
    "away":    VENUE_AWAY,
}


# ======================================================================
#  BayesianDixonColesModel
# ======================================================================

class BayesianDixonColesModel:
    """
    Bayesian Hierarchical Dixon-Coles Model (MAP estimation).

    Fundamental architectural differences from the previous version:

    1. Objective function changed from "negative log-likelihood" to "negative log-posterior"
       loss = -log L(data|theta) - log pi(theta)
       Posterior = likelihood x prior -> MAP uses prior when samples are few, dominated by likelihood when abundant

    2. Hierarchical prior attack_i ~ N(0, sigma_atk)
       sigma_atk is a learnable parameter (empirical Bayes / Type-II ML)
       Effect: teams with 5 matches shrink toward global mean, teams with 50 matches determined by data

    3. Three-level venue: home_adj (true home) / neutral_adj (neutral) / 0 (away)
       neutral_adj has independent prior N(0.10, 0.20), about 30-40% of home advantage

    4. Competition weights: friendly 0.35 x temporal decay, World Cup 1.00 x temporal decay
       Combined with damping to form composite weight

    Training data must contain columns:
        home_team, away_team, home_goals, away_goals, date, venue, competition
    """

    def __init__(
        self,
        damping: float = 0.002,
        max_goals: int = 10,
        # Prior hyperparameters (all statistically grounded, see file header comments)
        prior_rho_std: float = 0.15,
        prior_home_mean: float = 0.25,
        prior_home_std: float = 0.20,
        prior_neutral_mean: float = 0.10,
        prior_neutral_std: float = 0.20,
        prior_log_sigma_std: float = 1.0,
    ) -> None:
        """
        Args:
            damping:              Temporal decay factor xi, w_temporal = exp(-xi x days_ago)
            max_goals:            Score matrix truncation value
            prior_rho_std:        rho prior std, N(0, prior_rho_std)
                                  Smaller -> rho more constrained near 0
                                  Larger -> closer to MLE, more prone to boundary issues
            prior_home_mean:      home_adj prior mean (literature ~0.25)
            prior_home_std:       home_adj prior std
            prior_neutral_mean:   neutral_adj prior mean (~40% of home)
            prior_neutral_std:    neutral_adj prior std
            prior_log_sigma_std:  log sigma prior std (preventing sigma from degenerating to 0 or infinity)
        """
        self.damping              = damping
        self.max_goals            = max_goals
        self.prior_rho_std        = prior_rho_std
        self.prior_home_mean      = prior_home_mean
        self.prior_home_std       = prior_home_std
        self.prior_neutral_mean   = prior_neutral_mean
        self.prior_neutral_std    = prior_neutral_std
        self.prior_log_sigma_std  = prior_log_sigma_std

        self.params:   Optional[dict] = None
        self.teams:    Optional[list] = None
        self.team_idx: Optional[dict] = None
        self._fit_info: dict = {}
        self._ensemble = None  # HistGBM ensemble (lazy loaded)
        self._fit_df = None   # Most recent training df (for ensemble rolling features)
        # Pooling weight: 0.556 comes from v7 OOF validation set RPS optimum (log pool, fit_weights)
        # To re-optimize: EnsemblePredictor.fit_weights(train_df, val_df, pool_method="log")
        self._ensemble_weight = 0.556

    # ------------------------------------------------------------------
    #  Data Preparation
    # ------------------------------------------------------------------

    def _prepare_teams(self, df: pd.DataFrame) -> None:
        all_teams     = pd.concat([df["home_team"], df["away_team"]]).unique()
        self.teams    = sorted(all_teams)
        self.team_idx = {t: i for i, t in enumerate(self.teams)}

    def _encode_venues(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert venue column (string) to integer encoding:
          'home' -> 2, 'neutral' -> 1, 'away' -> 0
        Unknown values default to neutral (most conservative assumption).
        """
        if "venue" not in df.columns:
            logger.warning(
                "'venue' column missing — treating all matches as neutral. "
                "Add venue='home'/'neutral'/'away' for correct predictions."
            )
            return np.full(len(df), VENUE_NEUTRAL, dtype=int)
        return df["venue"].map(_VENUE_MAP).fillna(VENUE_NEUTRAL).values.astype(int)

    def _compute_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Composite weight = temporal decay x match importance

        w_i = exp(-xi x days_ago_i) x competition_weight_i

        This stacks two independent weight dimensions:
          - Temporal decay: more recent matches are more important
          - Competition weight: World Cup more important than friendlies (time-independent)

        Normalized to mean = 1, keeping likelihood magnitude stable.
        """
        # Temporal decay
        if self.damping > 0.0 and "date" in df.columns:
            t_max    = pd.to_datetime(df["date"]).max()
            days_ago = (t_max - pd.to_datetime(df["date"])).dt.days.values.astype(float)
            w_time   = np.exp(-self.damping * days_ago)
        else:
            w_time = np.ones(len(df))

        # Match importance weights
        if "competition" in df.columns:
            w_comp = (
                df["competition"]
                .map(COMPETITION_WEIGHTS)
                .fillna(0.50)          # Unknown competition defaults to 0.5 (conservative)
                .values.astype(float)
            )
        else:
            logger.warning(
                "'competition' column missing — treating all matches as equal weight. "
                "Add competition='world_cup'/'friendly'/... for correct weighting."
            )
            w_comp = np.ones(len(df))

        # Composite weight normalization
        weights = w_time * w_comp
        return weights / weights.mean()

    # ------------------------------------------------------------------
    #  Fitting (PyMC find_MAP)
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "BayesianDixonColesModel":
        """
        MAP fitting (PyMC find_MAP — automatic differentiation, ZeroSumNormal built-in constraint, NaN-safe).

        Priors:
          beta_att ~ ZeroSumNormal(sigma_att)    sum(att)=0 built-in
          beta_def ~ ZeroSumNormal(sigma_def)    sum(def)=0 built-in
          sigma_att, sigma_def ~ HalfNormal(1)
          gamma_home      ~ N(0.25, 0.20)
          gamma_neutral   ~ N(0.10, 0.20)
          rho           ~ N(0, 0.15)

        Uncertainty: no MCMC (memory safe), uses quantbet.posterior Laplace approximation.
        """
        import pymc as pm
        import pytensor.tensor as pt

        self._prepare_teams(df)
        n = len(self.teams)

        home_idx  = df["home_team"].map(self.team_idx).values.astype(int)
        away_idx  = df["away_team"].map(self.team_idx).values.astype(int)
        goals_h   = df["home_goals"].values.astype(int)
        goals_a   = df["away_goals"].values.astype(int)
        venue_enc = self._encode_venues(df).astype(int)
        weights   = self._compute_weights(df)

        with pm.Model(coords={"team": self.teams}) as model:
            # --- Shared data ---
            h_idx = pm.Data("h_idx", home_idx)
            a_idx = pm.Data("a_idx", away_idx)
            v_enc = pm.Data("v_enc", venue_enc)
            g_h   = pm.Data("g_h", goals_h)
            g_a   = pm.Data("g_a", goals_a)
            wgt   = pm.Data("wgt", weights)

            # --- Rating prior: ELO (database, eloratings.net) — anchor = pure Elo+confederation ---
            from quantbet.worldcup.confederations import CONFEDERATION_MAP as _CONF_MAP

            # ELO ratings — read from SQL (daily auto-update, cross-confederation comparable)
            from db.elo_reader import get_standardized_elo as _get_elo
            try:
                ratings = _get_elo(self.teams)
                if np.all(np.abs(ratings) < 1e-9):
                    raise ValueError("ELO all zero")
            except Exception as e:
                logger.warning("ELO load failed (%s), falling back to Bradley-Terry", e)
                from quantbet.worldcup import bradley_terry_log_strength, standardize_ratings
                y_dc = np.where(goals_h > goals_a, 1, np.where(goals_h < goals_a, -1, 0))
                bt_log = bradley_terry_log_strength(home_idx, away_idx, y_dc, n, ridge=1e-2)
                ratings = standardize_ratings(bt_log)

            # Confederation labels + prior
            conf_labels = [_CONF_MAP.get(t, "Other") for t in self.teams]
            from quantbet.worldcup.confederation_prior import ConfederationPrior
            cp = ConfederationPrior(conf_labels)
            n_conf = cp.n_conf
            tau_att = pm.HalfNormal("tau_att", sigma=1.0)
            tau_def = pm.HalfNormal("tau_def", sigma=1.0)
            mu_conf_att_raw = pm.ZeroSumNormal("mu_conf_att_raw", sigma=tau_att, shape=n_conf)
            mu_conf_def_raw = pm.ZeroSumNormal("mu_conf_def_raw", sigma=tau_def, shape=n_conf)
            mu_conf_att_expanded = mu_conf_att_raw[cp._index]
            mu_conf_def_expanded = mu_conf_def_raw[cp._index]

            # Elo ratings: within-conf ranking, sign(x)*log1p(|x|) compress tails
            ratings_within = cp.demean_rating_within_conf(ratings)
            ratings_compressed = np.sign(ratings_within) * np.log1p(np.abs(ratings_within))
            elo_tensor = pt.as_tensor_variable(ratings_compressed, dtype="float64")
            eta_att = pm.Normal("eta_att", mu=0.0, sigma=1.0)
            eta_def = pm.Normal("eta_def", mu=0.0, sigma=1.0)

            # Synthetic anchor mean: att_i = mu_conf(i) + eta_att * elo_i (no static momentum)
            mu_att = mu_conf_att_expanded + eta_att * elo_tensor
            mu_def = mu_conf_def_expanded + eta_def * elo_tensor

            # Hierarchical residual noise (ZeroSumNormal built-in sum=0)
            sigma_att = pm.HalfNormal("sigma_att", sigma=1.0)
            sigma_def = pm.HalfNormal("sigma_def", sigma=1.0)
            att_raw = pm.ZeroSumNormal("att_raw", sigma=sigma_att, dims="team")
            dfn_raw = pm.ZeroSumNormal("dfn_raw", sigma=sigma_def, dims="team")

            # Final attack/defense parameters
            att = pm.Deterministic("att", mu_att + att_raw)
            dfn = pm.Deterministic("dfn", mu_def + dfn_raw)

            # Venue + rho (unchanged)
            home_adj    = pm.Normal("home_adj", mu=self.prior_home_mean, sigma=self.prior_home_std)
            neutral_adj = pm.Normal("neutral_adj", mu=self.prior_neutral_mean, sigma=self.prior_neutral_std)
            rho         = pm.Normal("rho", mu=0.0, sigma=0.15)

            # --- Log-linear predictor (PyTensor auto overflow protection) ---
            v = home_adj * pt.eq(v_enc, 2) + neutral_adj * pt.eq(v_enc, 1)
            log_lh = att[h_idx] + dfn[a_idx] + v
            log_la = att[a_idx] + dfn[h_idx]

            log_lh = pt.clip(log_lh, -10.0, 10.0)
            log_la = pt.clip(log_la, -10.0, 10.0)

            lam_h = pt.exp(log_lh)
            lam_a = pt.exp(log_la)

            # --- Poisson log-likelihood ---
            ll  = g_h * log_lh - lam_h - pt.gammaln(g_h + 1)
            ll += g_a * log_la - lam_a - pt.gammaln(g_a + 1)

            # Tau correction
            tau = pt.switch(
                pt.eq(g_h,0) & pt.eq(g_a,0), pt.log(pt.clip(1.0 - lam_h*lam_a*rho, 1e-12, 2.0)),
            pt.switch(
                pt.eq(g_h,1) & pt.eq(g_a,0), pt.log(pt.clip(1.0 + lam_a*rho, 1e-12, 2.0)),
            pt.switch(
                pt.eq(g_h,0) & pt.eq(g_a,1), pt.log(pt.clip(1.0 + lam_h*rho, 1e-12, 2.0)),
            pt.switch(
                pt.eq(g_h,1) & pt.eq(g_a,1), pt.log(pt.clip(1.0 - rho, 1e-12, 2.0)),
                pt.zeros_like(ll)
            ))))
            pm.Potential("log_lik", pt.sum(wgt * (ll + tau)))

            # MAP estimation (PyMC built-in optimizer, more stable than SLSQP)
            map_est = pm.find_MAP(progressbar=True, maxeval=10000)

            # Save model reference (for subsequent Laplace approximation)
            self._pymc_model = model

        # --- Extract parameters (v9.1: GAS anchor = pure Elo+confederation, no static momentum) ---
        self.params = {
            "attack":        map_est["att"],
            "defense":       map_est["dfn"],
            "home_adj":      float(map_est["home_adj"]),
            "neutral_adj":   float(map_est["neutral_adj"]),
            "rho":           float(map_est["rho"]),
            "sigma_attack":  float(map_est["sigma_att"]),
            "sigma_defense": float(map_est["sigma_def"]),
            "eta_att":       float(map_est["eta_att"]),
            "eta_def":       float(map_est["eta_def"]),
            "tau_att":       float(map_est.get("tau_att", 0.0)),
            "tau_def":       float(map_est.get("tau_def", 0.0)),
            "ratings":       ratings,
            "conf_labels":   cp.labels,
        }
        # Confederation means
        for att_key, def_key in [("mu_conf_att_raw", "mu_conf_def_raw"),
                                  ("mu_conf_att", "mu_conf_def")]:
            if att_key in map_est:
                self.params["mu_conf_att"] = map_est[att_key]
                self.params["mu_conf_def"] = map_est[def_key]
                break

        # --- Diagnostics (v9.1: anchor = pure Elo+confederation, GAS post-fit, not counted in MAP degrees of freedom) ---
        n_obs  = len(df)
        n_free = 2 * n + 3 + 2 + (n_conf - 1) * 2  # no mom_att/mom_def (GAS post-fit)
        ll_at_map = float(pt.sum(ll + tau).eval({
            att: map_est["att"], dfn: map_est["dfn"],
            home_adj: map_est["home_adj"], neutral_adj: map_est["neutral_adj"],
            rho: map_est["rho"],
        }))
        log_lik = ll_at_map

        self._fit_info = {
            "n_teams": n, "n_matches": n_obs,
            "log_lik": round(log_lik, 4),
            "aic": round(2*n_free - 2*log_lik, 4),
            "bic": round(np.log(n_obs)*n_free - 2*log_lik, 4),
            "converged": True,
            "rho": self.params["rho"], "home_adj": self.params["home_adj"],
            "neutral_adj": self.params["neutral_adj"],
            "sigma_attack": self.params["sigma_attack"],
            "sigma_defense": self.params["sigma_defense"],
            "eta_att": self.params["eta_att"],
            "eta_def": self.params["eta_def"],
            "tau_att": self.params["tau_att"],
            "tau_def": self.params["tau_def"],
            "n_conf": n_conf,
            "conf_labels": cp.labels,
        }

        _log.info(
            "map_fit",
            teams=n, matches=n_obs, log_lik=round(log_lik, 2),
            eta_att=round(self.params["eta_att"], 3),
            eta_def=round(self.params["eta_def"], 3),
            tau_att=round(self.params["tau_att"], 3),
            tau_def=round(self.params["tau_def"], 3),
            sigma_att=round(self.params["sigma_attack"], 3),
            sigma_def=round(self.params["sigma_defense"], 3),
        )

        # Auto-install NB+Frank score matrix layer
        self._init_scoreline()

        # Auto dynamic tuning + GAS post-fit (does not contaminate MAP)
        try:
            self._tune_dynamics_and_calibrate(df)
        except Exception as e:
            _log.warning("dynamic_tuning_skipped", error=str(e))
            self._dynamic_sd = 0.0
            self._calibrator = None

        try:
            self._fit_gas(df)
            if self._gas is None:
                raise GasFitError("GAS not fitted, cannot save — check _fit_gas logs")
        except Exception as e:
            _log.error("gas_fit_failed", error=str(e))
            raise

        # Auto dynamic optimization of ensemble pooling weight (RPS optimal on validation set)
        try:
            self._optimize_ensemble_weight(df)
        except Exception as e:
            logger.warning("Ensemble weight opt skipped: %s", e)

        self._fit_df = df.copy()

        # Save training metadata (v9.meta.json)
        try:
            from registry import save_training_metadata
            fi = self._fit_info
            save_training_metadata(
                "v9", self, training_matches=fi.get("n_matches", len(df)),
                log_lik=fi.get("log_lik", 0), aic=fi.get("aic", 0), bic=fi.get("bic", 0),
            )
        except Exception:
            pass

        return self

    def _tune_dynamics_and_calibrate(self, df: pd.DataFrame):
        """Tune process_sd + fit calibrator (ScoreMatrixCalibrator) on validation set."""
        from quantbet.scoreline.dynamic_strength import DynamicStrengthFilter
        from quantbet.scoreline import ScoreMatrixCalibrator

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        n_val = max(n // 5, 50)  # 20%
        train_df = df.iloc[:n - n_val].copy()
        val_df = df.iloc[n - n_val:].copy()

        # Build att0/def0 dict
        att0 = {t: float(v) for t, v in zip(self.teams, self.params["attack"])}
        def0 = {t: float(v) for t, v in zip(self.teams, self.params["defense"])}
        ha = float(self.params["home_adj"])
        na = float(self.params["neutral_adj"])

        # tune process_sd
        best_sd = DynamicStrengthFilter.tune_process_sd(
            att0, def0, ha, na, train_df, val_df,
            candidates=(0.0, 0.15, 0.25, 0.4, 0.6, 0.85),
            halflife_days=540.0,
        )
        self._dynamic_sd = best_sd
        logger.info("DynamicStrengthFilter: tuned process_sd=%.2f", best_sd)

        # If process_sd > 0, run one full pass to collect matrices -> calibrate
        if best_sd > 0:
            dyn = DynamicStrengthFilter(
                att0, def0, ha, na, process_sd_per_year=best_sd,
                mean_reversion_halflife_days=540.0,
            )
            dyn.run(train_df, collect_oos=False)
            matrices, lambdas, outcomes = [], [], []
            for r in val_df.itertuples():
                venue = getattr(r, "venue", "neutral")
                lh, la = dyn.expected_goals(r.home_team, r.away_team, venue, as_of=r.date)
                M = self._scoreline_model.score_matrix(float(lh), float(la))
                M = M / M.sum()
                matrices.append(M)
                lambdas.append((float(lh), float(la)))
                y = 0 if r.home_goals > r.away_goals else (1 if r.home_goals == r.away_goals else 2)
                outcomes.append(y)
                dyn.step(r.home_team, r.away_team, int(r.home_goals), int(r.away_goals), venue, r.date)

            cal = ScoreMatrixCalibrator(w_logloss=0.5)
            cal.fit(matrices, lambdas, outcomes)
            self._calibrator = cal
            logger.info("Calibrator: temp=%.4f theta_draw=%.4f", cal.temp_, cal.theta_draw_)
        else:
            self._calibrator = None

    def _init_scoreline(self):
        """Install FlexibleScoreModel (NB+Frank copula + diagonal inflation)."""
        from quantbet.scoreline import FlexibleScoreModel

        teams = self.teams
        params = self.params
        fsm = FlexibleScoreModel(margin="nb", dependence="frank", diagonal_inflation=True)
        fsm.set_strengths(
            {t: float(v) for t, v in zip(teams, params["attack"])},
            {t: float(v) for t, v in zip(teams, params["defense"])},
            float(params["home_adj"]),
            float(params["neutral_adj"]),
        )
        self._scoreline_model = fsm

    # ------------------------------------------------------------------
    #  HistGBM Ensemble Lazy Loading
    # ------------------------------------------------------------------

    def _lazy_load_ensemble(self):
        """Load ensemble_v3.pkl (HistGBM, 40-dim rolling features). Uses registry unified path."""
        if self._ensemble is not None:
            return
        try:
            from registry import load_ensemble
            ens = load_ensemble()
            if ens is None:
                logger.warning("Ensemble pkl not found in registry — skipping")
                return
            # Attach scoreline to old dc_model inside ensemble (didn't exist at pickle time)
            if hasattr(ens, "dc_model") and not hasattr(ens.dc_model, "_scoreline_model"):
                try:
                    ens.dc_model._init_scoreline()
                except Exception:
                    pass
            self._ensemble = ens
            logger.info("Ensemble loaded via registry")
        except Exception as exc:
            logger.warning("Ensemble load failed (%s) — running DC only", exc)

    # ------------------------------------------------------------------
    #  Ensemble Pooling Weight Dynamic Optimization (minimize RPS on validation set)
    # ------------------------------------------------------------------

    def _optimize_ensemble_weight(self, df: pd.DataFrame):
        """Load ensemble, search optimal DC/GBM pooling weight on validation set.

        Split out 20% time tail as validation set, get DC probability (self) + GBM probability (ensemble)
        for each match, minimize average RPS to find w*. Store result in self._ensemble_weight.
        """
        self._lazy_load_ensemble()
        if self._ensemble is None:
            logger.warning("No ensemble loaded, keeping default weight=%.3f", self._ensemble_weight)
            return

        from scipy.optimize import minimize_scalar

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        n_val = max(n // 5, 100)
        val_df = df.iloc[n - n_val:].copy()
        train_df = df.iloc[:n - n_val].copy()

        from quantbet.scoreline.score_driven import ScoreDrivenStrength
        sd = ScoreDrivenStrength(
            attack0={t: float(v) for t, v in zip(self.teams, self.params["attack"])},
            defense0={t: float(v) for t, v in zip(self.teams, self.params["defense"])},
            home_adj=float(self.params["home_adj"]),
            neutral_adj=float(self.params["neutral_adj"]),
        )
        sd.run(train_df, collect_oos=False)

        dc_probs, gbm_probs, outcomes = [], [], []
        for _, r in val_df.iterrows():
            ht, at = r["home_team"], r["away_team"]
            venue = r.get("venue", "neutral")
            gh, ga = int(r["home_goals"]), int(r["away_goals"])
            y = 0 if gh > ga else (1 if gh == ga else 2)

            try:
                lh, la = sd.expected_goals(ht, at, venue=venue, as_of=r["date"])
                from quantbet.scoreline.score_model import FlexibleScoreModel
                fsm = FlexibleScoreModel(margin="nb", dependence="frank", diagonal_inflation=True)
                fsm.set_strengths(
                    {t: float(v) for t, v in zip(self.teams, self.params["attack"])},
                    {t: float(v) for t, v in zip(self.teams, self.params["defense"])},
                    float(self.params["home_adj"]), float(self.params["neutral_adj"]),
                )
                M = fsm.score_matrix(float(lh), float(la))
                M = M / M.sum()
                dc_p = np.array([float(np.tril(M,-1).sum()), float(np.trace(M)), float(np.triu(M,1).sum())])
            except Exception:
                dc_p = np.array([1/3, 1/3, 1/3])
            dc_probs.append(dc_p)
            outcomes.append(y)

            # GBM
            earlier = val_df[val_df["date"] < r["date"]]
            hist = pd.concat([train_df, earlier], ignore_index=True)
            try:
                gbm_r = self._ensemble.predict(ht, at, df=hist, venue=venue)
                gbm_p = np.array([gbm_r["ensemble"]["home_win_prob"],
                                  gbm_r["ensemble"]["draw_prob"],
                                  gbm_r["ensemble"]["away_win_prob"]])
            except Exception:
                gbm_p = np.array([1/3, 1/3, 1/3])
            gbm_probs.append(gbm_p)

            sd.step(ht, at, gh, ga, venue, r["date"])

        dc_probs = np.asarray(dc_probs)
        gbm_probs = np.asarray(gbm_probs)
        outcomes = np.asarray(outcomes)

        def _mean_rps_arr(p, y):
            e = np.zeros((len(y), 3)); e[np.arange(len(y)), y] = 1.0
            return float(np.mean([0.5*np.sum((np.cumsum(p[i])-np.cumsum(e[i]))**2) for i in range(len(y))]))

        from quantbet.pooling import log_pool
        def rps_for_w(w):
            blend = log_pool([dc_probs, gbm_probs], [float(w), 1.0-float(w)])
            return _mean_rps_arr(blend, outcomes)

        res = minimize_scalar(rps_for_w, bounds=(0.05, 0.95), method="bounded")
        w_opt = float(res.x)

        self._ensemble_weight = round(w_opt, 3)
        dc_only = _mean_rps_arr(dc_probs, outcomes)
        gbm_only = _mean_rps_arr(gbm_probs, outcomes)
        opt_rps = float(res.fun)

        logger.info("Ensemble weight optimized | DC=%.3f GBM=%.3f | RPS: DC=%.4f GBM=%.4f blend=%.4f",
                     self._ensemble_weight, 1-self._ensemble_weight, dc_only, gbm_only, opt_rps)
    # ------------------------------------------------------------------
    #  GAS Post-Fit — Fit time-varying dynamics on top of MAP anchor (ScoreDrivenStrength)
    # ------------------------------------------------------------------

    def _fit_gas(self, df: "pd.DataFrame | None" = None, fix_B: float | None = 0.985):
        """Fit GAS time-varying dynamics on top of MAP anchor (ScoreDrivenStrength).

        When df is not provided, automatically pulls full match data from SQL.
        fix_B=0.985: B is fixed to held-out RPS optimal value, only A is optimized (source locked, prevents sliding to B->1 edge solution)
        """
        from quantbet.scoreline.score_driven import ScoreDrivenStrength

        if df is None:
            from sqlalchemy import create_engine, text
            from db.config import DATABASE_URL, ENGINE_KWARGS
            import pandas as pd
            engine = create_engine(DATABASE_URL, **ENGINE_KWARGS)
            with engine.connect() as c:
                df = pd.read_sql(text("""
                    SELECT m.date, ht.name AS home_team, at.name AS away_team,
                           m.home_score AS home_goals, m.away_score AS away_goals,
                           m.venue
                    FROM matches m
                    JOIN teams ht ON m.home_team_id = ht.id
                    JOIN teams at ON m.away_team_id = at.id
                    WHERE m.date >= :cut
                    ORDER BY m.date
                """), c, params={"cut": f"{pd.Timestamp.now().year - 10}-01-01"})
            df["venue"] = df["venue"].fillna("neutral")
            df = df[df.home_team.isin(self.teams) & df.away_team.isin(self.teams)]

        sd = ScoreDrivenStrength(
            attack0={t: float(v) for t, v in zip(self.teams, self.params["attack"])},
            defense0={t: float(v) for t, v in zip(self.teams, self.params["defense"])},
            home_adj=float(self.params["home_adj"]),
            neutral_adj=float(self.params["neutral_adj"]),
        )

        hyper = sd.fit_hyperparams(df, share_gain=True, fix_B=fix_B)
        self._gas_hyper = hyper
        _log.info("gas_fit", A_att=hyper["A_att"], B=hyper["B"],
                   neg_ll=round(hyper["neg_ll"], 1), n_matches=hyper["n_matches"])

        sd.run(df, collect_oos=False)
        self._gas = sd

    def gas_momentum_table(self) -> "pd.DataFrame":
        """Return per-team real-time GAS momentum relative to Elo anchor.

        momentum = d_att - d_def (>0 = currently stronger than Elo predicted).
        """
        if getattr(self, "_gas", None) is None:
            raise GasFitError("No GAS fit. Call fit() with GAS enabled first.")
        return self._gas.momentum_table()

    # ------------------------------------------------------------------
    #  Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        home_team: str,
        away_team: str,
        venue: str = "neutral",
        date = None,
    ) -> dict:
        """
        Score matrix prediction (NB / Frank copula + diagonal inflation, Boshnakov et al. 2017, IJF).

        Args:
            home_team: Home team (or first-listed team at neutral venue)
            away_team: Away team
            venue:     'home' / 'neutral' / 'away'
            date:      Prediction date (passed to GAS as_of, determines momentum decay point)

        Returns:
            dict containing home_win_prob / draw_prob / away_win_prob / score_probs
        """
        self._check_fitted()
        self._check_team(home_team)
        self._check_team(away_team)

        venue_code = _VENUE_MAP.get(venue, VENUE_NEUTRAL)
        if venue not in _VENUE_MAP:
            logger.warning("Unknown venue '%s', defaulting to 'neutral'", venue)

        h_idx = self.team_idx[home_team]
        a_idx = self.team_idx[away_team]

        # GAS real-time dynamic lambda (persisted into pkl, does not pull SQL at prediction time)
        gas = getattr(self, "_gas", None)
        if gas is not None:
            lh, la = gas.expected_goals(home_team, away_team, venue=venue, as_of=date)
        else:
            _log.warning("predict_no_gas", home_team=home_team, away_team=away_team)
            if venue_code == VENUE_HOME:
                venue_adj = self.params["home_adj"]
            elif venue_code == VENUE_NEUTRAL:
                venue_adj = self.params["neutral_adj"]
            else:
                venue_adj = 0.0

            lh = float(np.exp(np.clip(
                self.params["attack"][h_idx] + self.params["defense"][a_idx] + venue_adj,
                -10.0, 10.0,
            )))
            la = float(np.exp(np.clip(
                self.params["attack"][a_idx] + self.params["defense"][h_idx],
                -10.0, 10.0,
            )))

        lambda_h, lambda_a = lh, la

        # NB + Frank copula + diagonal inflation (Boshnakov-Kharrat-McHale 2017)
        score_matrix = self._scoreline_model.score_matrix(lambda_h, lambda_a)
        score_matrix = score_matrix / score_matrix.sum()

        # Apply calibrator if fitted (temperature + diagonal inflation on val)
        if getattr(self, "_calibrator", None) is not None:
            score_matrix = self._calibrator.transform(score_matrix, lambda_h, lambda_a)
        score_matrix = score_matrix / score_matrix.sum()

        home_win = float(np.tril(score_matrix, -1).sum())
        draw     = float(np.trace(score_matrix))
        away_win = float(np.triu(score_matrix, 1).sum())

        # Derived markets: directly read from score matrix M (architecture [4R])
        n = self.max_goals + 1
        idx_sum = np.add.outer(np.arange(n), np.arange(n))
        over_25 = float(score_matrix[idx_sum >= 3].sum())
        btts    = float(score_matrix[1:, 1:].sum())

        # HistGBM ensemble + log pooling (Genest-Zidek 1986 / Ranjan-Gneiting 2010)
        pool_method = None
        dc_weight = 1.0
        if not hasattr(self, "_ensemble"):
            self._ensemble = None
        if self._ensemble is None and hasattr(self, "max_goals") and self.max_goals:
            try:
                self._lazy_load_ensemble()
            except Exception:
                pass

        if self._ensemble is not None:
            P_gbm = None
            try:
                # Ensemble needs historical df to compute rolling features + h2h
                df_ens = getattr(self, "_fit_df", None)
                if df_ens is None:
                    logger.warning("predict: _fit_df not persisted, skipping ensemble")
                    raise RuntimeError("No persisted df for ensemble")
                gbm_r = self._ensemble.predict(home_team, away_team, df=df_ens, venue=venue)
                P_gbm = np.array([gbm_r["ensemble"]["home_win_prob"],
                                  gbm_r["ensemble"]["draw_prob"],
                                  gbm_r["ensemble"]["away_win_prob"]], dtype=float)
            except Exception as exc:
                logger.warning("Ensemble predict failed: %s", exc)

            if P_gbm is not None:
                from quantbet.pooling import log_pool
                P_dc = np.array([home_win, draw, away_win], dtype=float)
                w = float(self._ensemble_weight)
                P_pooled = log_pool([P_dc, P_gbm], [w, 1.0 - w])
                home_win, draw, away_win = float(P_pooled[0]), float(P_pooled[1]), float(P_pooled[2])
                pool_method = "log"
                dc_weight = w

        score_probs = {
            f"{h}-{a}": round(float(score_matrix[h, a]), 4)
            for h in range(self.max_goals + 1)
            for a in range(self.max_goals + 1)
            if score_matrix[h, a] >= 0.005
        }

        return {
            "home_team":           home_team,
            "away_team":           away_team,
            "venue":               venue,
            "expected_home_goals": round(lambda_h, 3),
            "expected_away_goals": round(lambda_a, 3),
            "home_win_prob":       round(home_win, 4),
            "draw_prob":           round(draw, 4),
            "away_win_prob":       round(away_win, 4),
            "pool_method":         pool_method,
            "dc_weight":           round(dc_weight, 3),
            "over_25":             round(over_25, 4),
            "btts":                round(btts, 4),
            "score_probs":         score_probs,
            "score_matrix":        score_matrix,
        }

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    def rps(self, result: dict, actual_outcome: str) -> float:
        """
        Ranked Probability Score (RPS). Lower is better.
        actual_outcome: 'H' (home win) / 'D' (draw) / 'A' (away win)
        """
        pred = np.array([
            result["home_win_prob"],
            result["draw_prob"],
            result["away_win_prob"],
        ])
        actual = np.array({"H": [1,0,0], "D": [0,1,0], "A": [0,0,1]}[actual_outcome], dtype=float)
        return float(0.5 * np.sum((np.cumsum(pred)[:-1] - np.cumsum(actual)[:-1]) ** 2))

    def evaluate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch evaluation. df must contain: home_team, away_team, venue, result ('H'/'D'/'A')
        """
        self._check_fitted()
        records = []
        for _, row in df.iterrows():
            try:
                venue = row.get("venue", "neutral")
                pred  = self.predict(row["home_team"], row["away_team"], venue=venue)
                records.append({
                    "home_team":     row["home_team"],
                    "away_team":     row["away_team"],
                    "venue":         venue,
                    "home_win_prob": pred["home_win_prob"],
                    "draw_prob":     pred["draw_prob"],
                    "away_win_prob": pred["away_win_prob"],
                    "actual":        row["result"],
                    "rps":           round(self.rps(pred, row["result"]), 4),
                })
            except KeyError:
                logger.warning("Skipping unknown team: %s vs %s",
                               row["home_team"], row["away_team"])
        result_df = pd.DataFrame(records)
        if not result_df.empty:
            logger.info("Batch eval | n=%d | mean_rps=%.4f",
                        len(result_df), result_df["rps"].mean())
        return result_df

    def team_ratings(self) -> pd.DataFrame:
        """Attack/defense rating DataFrame, sorted by overall strength descending"""
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
        """Return fitting diagnostic info"""
        self._check_fitted()
        return self._fit_info.copy()

    # ------------------------------------------------------------------
    #  Diagnostics: Prior vs Posterior Comparison
    # ------------------------------------------------------------------

    def diagnose_shrinkage(self) -> pd.DataFrame:
        """
        Shrinkage diagnosis: compare each team's parameter distance from hierarchical prior mean (0).

        shrinkage_ratio = |posterior_mean| / prior_std
        Closer to 0 means prior dominates (few data), larger means data dominates (more data).
        Used to determine which team's parameter estimates are reliable, and which rely on prior shrinkage.
        """
        self._check_fitted()
        sigma_atk = self.params["sigma_attack"]
        sigma_def = self.params["sigma_defense"]
        atk  = self.params["attack"]
        defn = self.params["defense"]

        return pd.DataFrame({
            "team":           self.teams,
            "attack":         atk.round(4),
            "defense":        defn.round(4),
            "atk_shrinkage":  (np.abs(atk) / sigma_atk).round(3),
            "def_shrinkage":  (np.abs(defn) / sigma_def).round(3),
            "note": [
                "data-driven" if (abs(a) / sigma_atk) > 1.0 else "prior-dominated"
                for a in atk
            ],
        }).sort_values("atk_shrinkage", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    #  Validation Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.params is None:
            raise ModelNotFoundError("Model not fitted. Call fit() first.")

    def _check_team(self, team: str) -> None:
        if team not in self.team_idx:
            raise PredictionError(
                f"Unknown team: '{team}'. "
                f"Known: {self.teams}"
            )

    def __repr__(self) -> str:
        if self.params is None:
            return "BayesianDixonColesModel(not fitted)"
        i = self._fit_info
        return (
            f"BayesianDixonColesModel("
            f"teams={i['n_teams']}, matches={i['n_matches']}, "
            f"LL={i['log_lik']:.2f}, AIC={i['aic']:.2f}, "
            f"rho={i['rho']:.4f}, "
            f"σ_atk={i['sigma_attack']:.3f}, σ_def={i['sigma_defense']:.3f}, "
            f"home={i['home_adj']:.3f}, neutral={i['neutral_adj']:.3f})"
        )


# ======================================================================
#  Test / Demo
# ======================================================================

def generate_international_mock_data(
    n_teams: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate simulated international football data (including venue and competition columns).

    Match setup:
      - World Cup qualifiers (home/away)
      - Continental championship (neutral venue)
      - Friendlies (home/away)
    """
    rng   = np.random.default_rng(seed)
    teams = [f"NTL_{chr(65 + i)}" for i in range(n_teams)]

    # True strengths (used to generate differentiated data)
    true_attack  = rng.normal(0.0, 0.4, n_teams)
    true_defense = rng.normal(0.0, 0.3, n_teams)

    records = []
    base_date = pd.Timestamp("2024-09-01")

    competition_schedule = [
        # (competition, venue_type, weeks_offset, n_rounds)
        ("world_cup_qualifying", "home_away", 0,  8),
        ("friendly",             "home_away", 10, 3),
        ("continental_championship", "neutral", 16, 4),
        ("world_cup_qualifying", "home_away", 22, 6),
        ("friendly",             "home_away", 30, 2),
    ]

    for comp, venue_type, week_offset, n_rounds in competition_schedule:
        for round_num in range(n_rounds):
            match_date = base_date + pd.Timedelta(weeks=week_offset + round_num * 2)
            shuffled   = rng.permutation(n_teams)

            for i in range(0, n_teams, 2):
                h, a = shuffled[i], shuffled[i + 1]

                if venue_type == "neutral":
                    venue    = "neutral"
                    adj      = 0.08  # Small neutral venue advantage
                elif rng.random() < 0.5:
                    venue    = "home"
                    adj      = 0.25
                else:
                    venue    = "away"
                    adj      = 0.0

                lh = np.exp(true_attack[h] + true_defense[a] + adj)
                la = np.exp(true_attack[a] + true_defense[h])

                hg = int(rng.poisson(lh))
                ag = int(rng.poisson(la))

                records.append({
                    "date":        match_date,
                    "home_team":   teams[h],
                    "away_team":   teams[a],
                    "home_goals":  hg,
                    "away_goals":  ag,
                    "venue":       venue,
                    "competition": comp,
                    "result":      "H" if hg > ag else "A" if hg < ag else "D",
                })

    return pd.DataFrame(records)


def test_bayesian_model() -> None:
    print("\n" + "=" * 65)
    print("  BayesianDixonColesModel — Full Test")
    print("=" * 65)

    df = generate_international_mock_data(n_teams=16, seed=42)
    print(f"\nTraining data: {len(df)} matches, {df['home_team'].nunique()} teams")
    print(f"Venue distribution: {df['venue'].value_counts().to_dict()}")
    print(f"Competition distribution: {df['competition'].value_counts().to_dict()}")

    # --- Fitting ---
    model = BayesianDixonColesModel(
        damping=0.002,
        prior_rho_std=0.15,
        prior_home_mean=0.25,
        prior_neutral_mean=0.10,
    )
    model.fit(df)
    print(f"\n{model}")

    # --- Fit summary ---
    s = model.fit_summary()
    print("\n-- Fit Summary ---------------------------------------------")
    print(f"  Log-Likelihood  : {s['log_lik']:.4f}")
    print(f"  AIC             : {s['aic']:.4f}")
    print(f"  BIC             : {s['bic']:.4f}")
    print(f"  rho (tau)       : {s['rho']:.4f}  <- should be in (-0.3, 0)")
    print(f"  home_adj        : {s['home_adj']:.4f}  <- should be in (0.15, 0.40)")
    print(f"  neutral_adj     : {s['neutral_adj']:.4f}  <- should be < home_adj")
    print(f"  sigma_attack    : {s['sigma_attack']:.4f}  <- hierarchical std")
    print(f"  sigma_defense   : {s['sigma_defense']:.4f}  <- hierarchical std")
    print(f"  converged        : {s['converged']}")

    # --- Team ratings (Top 5) ---
    print("\n-- Team Ratings Top 5 --------------------------------------")
    print(model.team_ratings().head(5).to_string(index=False))

    # --- Shrinkage diagnosis ---
    print("\n-- Shrinkage Diagnosis (data-driven or prior-dominated?) -------")
    shrink = model.diagnose_shrinkage()
    print(shrink.head(8).to_string(index=False))

    # --- Venue comparison prediction ---
    teams   = model.teams
    home_t  = teams[0]
    away_t  = teams[1]

    print(f"\n-- Venue Comparison Prediction: {home_t} vs {away_t} ---------")
    print(f"{'Metric':<28} {'Home':>10} {'Neutral':>10} {'Away':>10}")
    print("-" * 60)

    for venue in ["home", "neutral", "away"]:
        r = model.predict(home_t, away_t, venue=venue)
        label = {"home": "(home)", "neutral": "(neutral)", "away": "(away)"}[venue]
        if venue == "home":
            print(f"{'λ_' + home_t[:6]:<28} {r['expected_home_goals']:>10.3f}", end="")
        elif venue == "neutral":
            print(f" {r['expected_home_goals']:>10.3f}", end="")
        else:
            print(f" {r['expected_home_goals']:>10.3f}")

    for venue in ["home", "neutral", "away"]:
        r = model.predict(home_t, away_t, venue=venue)
        if venue == "home":
            print(f"{'Home Win%':<28} {r['home_win_prob']*100:>9.1f}%", end="")
        elif venue == "neutral":
            print(f" {r['home_win_prob']*100:>9.1f}%", end="")
        else:
            print(f" {r['home_win_prob']*100:>9.1f}%")

    for venue in ["home", "neutral", "away"]:
        r = model.predict(home_t, away_t, venue=venue)
        if venue == "home":
            print(f"{'Draw%':<28} {r['draw_prob']*100:>9.1f}%", end="")
        elif venue == "neutral":
            print(f" {r['draw_prob']*100:>9.1f}%", end="")
        else:
            print(f" {r['draw_prob']*100:>9.1f}%")

    for venue in ["home", "neutral", "away"]:
        r = model.predict(home_t, away_t, venue=venue)
        if venue == "home":
            print(f"{'Away Win%':<28} {r['away_win_prob']*100:>9.1f}%", end="")
        elif venue == "neutral":
            print(f" {r['away_win_prob']*100:>9.1f}%", end="")
        else:
            print(f" {r['away_win_prob']*100:>9.1f}%")

    # --- Neutral venue prediction details ---
    r_ntl = model.predict(home_t, away_t, venue="neutral")
    print(f"\n-- Neutral Top 5 Scores: {home_t} vs {away_t} --------------")
    for score, prob in sorted(r_ntl["score_probs"].items(), key=lambda x: -x[1])[:5]:
        bar = "█" * int(prob * 80)
        print(f"  {score:>5}  {prob*100:5.1f}%  {bar}")

    # --- Batch RPS ---
    eval_df = df.sample(30, random_state=1).copy()
    batch   = model.evaluate_batch(eval_df)
    if not batch.empty:
        print(f"\n-- Batch Evaluation (30 matches) ---------------------------")
        print(f"  Mean RPS : {batch['rps'].mean():.4f}")
        by_venue = batch.groupby("venue")["rps"].mean()
        for v, rps_v in by_venue.items():
            print(f"  RPS [{v:>7}] : {rps_v:.4f}")
        by_comp = batch.groupby("actual")["rps"].mean()
        for out, rps_c in by_comp.items():
            print(f"  RPS [actual={out}] : {rps_c:.4f}")

    print("\n" + "=" * 65)
    print("  All tests passed.")
    print("=" * 65 + "\n")


# ======================================================================
#  Scoreline — NB/Frank copula + diagonal inflation (v8 default integration)
# ======================================================================

def install_scoreline(model: "BayesianDixonColesModel") -> "BayesianDixonColesModel":
    """Install FlexibleScoreModel score matrix layer (manually activate for old loaded pkl).

    Newly trained models automatically have this installed by fit(), no need to call this manually.
    Only used to upgrade old pickle.load pkl versions to NB+Frank.
    """
    model._init_scoreline()
    return model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    test_bayesian_model()
