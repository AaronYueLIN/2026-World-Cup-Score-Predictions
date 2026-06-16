"""
score_model.py — Flexible score model (marginal x dependence x inflation, drop-in replacement for DC predict)
=============================================================================================================

Assembles components from the scoreline sub-package into a predictor whose
**return structure is fully consistent** with the existing
`BayesianDixonColesModel.predict()`, enabling seamless downstream integration
(value_engine_v2 / markets.joint_prob / report).

Core pipeline (per match):
    lambda_h, lambda_a  ---(from static MAP or dynamic filter)--->
    marginal pmf_h, pmf_a  (Poisson / NegBin / Weibull-count)        <- fixes dispersion
    joint matrix M         (independent / bivariate-Poisson / Frank)  <- fixes full-table dependence
    diagonal inflation M*  (Karlis-Ntzoufras)                         <- fixes draws
    (optional) calibration M** (ScoreMatrixCalibrator)                <- fixes overall probability

Two ways to obtain attack/defence strengths:
  . fit(df)                : built-in lightweight weighted Poisson MLE (numpy/scipy, no pymc dependency)
  . set_strengths(...)     : inject external strengths (Bayesian DC MAP, or dynamic filter means)

References: see docstring headers in count_dists.py / bivariate.py.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from . import bivariate as biv
from . import count_dists as cd

logger = logging.getLogger(__name__)

__all__ = ["FlexibleScoreModel"]


class FlexibleScoreModel:
    def __init__(
        self,
        margin: str = "nb",          # 'poisson' | 'nb' | 'weibull'
        dependence: str = "frank",   # 'none' | 'bivpoi' | 'frank'
        diagonal_inflation: bool = True,
        max_goals: int = 10,
        damping: float = 0.002,
    ) -> None:
        assert margin in ("poisson", "nb", "weibull")
        assert dependence in ("none", "bivpoi", "frank")
        self.margin = margin
        self.dependence = dependence
        self.diagonal_inflation = diagonal_inflation
        self.max_goals = max_goals
        self.damping = damping

        # attack/defence / venue
        self.teams: Optional[list] = None
        self.team_idx: Optional[dict] = None
        self.params: dict = {}
        # global shape parameters (MLE estimated)
        self.shape_: dict = {"nb_r": 8.0, "wc_c": 1.0, "dep": 0.0, "theta_draw": 0.0}
        self.calibrator = None  # can attach ScoreMatrixCalibrator

    # ==================================================================
    #  Strength source 1: inject external (recommended with dynamic filter / Bayesian MAP)
    # ==================================================================
    def set_strengths(
        self, attack: dict[str, float], defense: dict[str, float],
        home_adj: float, neutral_adj: float,
    ) -> "FlexibleScoreModel":
        self.teams = sorted(set(attack) | set(defense))
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        self.params = {
            "attack": np.array([attack.get(t, 0.0) for t in self.teams]),
            "defense": np.array([defense.get(t, 0.0) for t in self.teams]),
            "home_adj": float(home_adj),
            "neutral_adj": float(neutral_adj),
        }
        return self

    # ==================================================================
    #  Strength source 2: built-in weighted Poisson MLE (no pymc)
    # ==================================================================
    def _weights(self, df: pd.DataFrame) -> np.ndarray:
        if self.damping > 0 and "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce")
            days = (d.max() - d).dt.days.values.astype(float)
            w = np.exp(-self.damping * np.nan_to_num(days, nan=0.0))
        else:
            w = np.ones(len(df))
        return w / w.mean()

    def fit(self, df: pd.DataFrame) -> "FlexibleScoreModel":
        """Weighted independent Poisson MLE (sum-to-zero) → then MLE global shape parameters."""
        all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
        self.teams = sorted(all_teams)
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        hi = df["home_team"].map(self.team_idx).values
        ai = df["away_team"].map(self.team_idx).values
        gh = df["home_goals"].values.astype(float)
        ga = df["away_goals"].values.astype(float)
        venue = self._encode_venue(df)
        w = self._weights(df)

        def nll(p):
            att, dfn = p[:n], p[n:2 * n]
            home_adj, neutral_adj = p[2 * n], p[2 * n + 1]
            vadj = home_adj * (venue == 2) + neutral_adj * (venue == 1)
            llh = np.clip(att[hi] + dfn[ai] + vadj, -10, 10)
            lla = np.clip(att[ai] + dfn[hi], -10, 10)
            lh, la = np.exp(llh), np.exp(lla)
            ll = (gh * llh - lh - gammaln(gh + 1) + ga * lla - la - gammaln(ga + 1))
            return float(-np.dot(w, ll))

        x0 = np.concatenate([np.zeros(2 * n), [0.25, 0.10]])
        cons = [{"type": "eq", "fun": lambda p: np.sum(p[:n])}]
        bounds = [(None, None)] * (2 * n) + [(-1, 2), (-1, 2)]
        res = minimize(nll, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-9})
        self.params = {
            "attack": res.x[:n], "defense": res.x[n:2 * n],
            "home_adj": float(res.x[2 * n]), "neutral_adj": float(res.x[2 * n + 1]),
        }
        # global shape parameters (dispersion + dependence + draw inflation)
        self._fit_shape(df, hi, ai, gh, ga, venue, w)
        logger.info("FlexibleScoreModel fitted | margin=%s dep=%s shape=%s",
                    self.margin, self.dependence, self.shape_)
        return self

    def _encode_venue(self, df: pd.DataFrame) -> np.ndarray:
        if "venue" not in df.columns:
            return np.full(len(df), 1, dtype=int)
        m = {"home": 2, "neutral": 1, "away": 0}
        return df["venue"].map(m).fillna(1).values.astype(int)

    def _lambdas_for(self, hi, ai, venue):
        att, dfn = self.params["attack"], self.params["defense"]
        vadj = self.params["home_adj"] * (venue == 2) + self.params["neutral_adj"] * (venue == 1)
        lh = np.exp(np.clip(att[hi] + dfn[ai] + vadj, -10, 10))
        la = np.exp(np.clip(att[ai] + dfn[hi], -10, 10))
        return lh, la

    def _fit_shape(self, df, hi, ai, gh, ga, venue, w) -> None:
        """
        Minimise weighted NLL of the joint pmf to estimate (dispersion, dependence, theta_draw).

        Low-dimensional (<=3 dims), but NLL is flat near boundaries (r->inf / c=1 / kappa=0); pure Nelder-Mead
        easily stalls. Here we use **coordinate grid scan + local polishing** to robustly find the global optimum.
        """
        lh_all, la_all = self._lambdas_for(hi, ai, venue)
        ghi, gai = gh.astype(int), ga.astype(int)
        K = self.max_goals

        def joint_nll(z):
            shape = self._unpack_shape(z)
            cache: dict = {}
            ll = 0.0
            for i in range(len(ghi)):
                key = (round(lh_all[i], 2), round(la_all[i], 2))
                M = cache.get(key)
                if M is None:
                    M = self._build_matrix(lh_all[i], la_all[i], shape)
                    cache[key] = M
                ll += w[i] * np.log(max(M[min(ghi[i], K), min(gai[i], K)], 1e-12))
            return float(-ll)

        z0, _ = self._shape_init_bounds()
        if len(z0) == 0:
            return

        # --- Coordinate grid (candidate z values per active parameter) ---
        grids = self._shape_grids()
        best_z, best_f = z0.copy(), joint_nll(z0)
        # Greedy scan per coordinate twice (enough for low-dimensional coupling)
        for _ in range(2):
            for d in range(len(best_z)):
                for cand in grids[d]:
                    z = best_z.copy(); z[d] = cand
                    f = joint_nll(z)
                    if f < best_f:
                        best_f, best_z = f, z
        # --- Local polishing ---
        res = minimize(joint_nll, best_z, method="Nelder-Mead",
                       options={"xatol": 1e-2, "fatol": 1e-2, "maxiter": 200})
        if res.fun < best_f:
            best_z = res.x
        self.shape_.update(self._unpack_shape(best_z))

    def _shape_grids(self) -> list:
        """Candidate z grid per active parameter (consistent with the unpacking order in _unpack_shape)."""
        grids = []
        if self.margin == "nb":
            grids.append(np.log(np.array([2, 3, 4, 6, 9, 15, 30, 100])))     # r
        elif self.margin == "weibull":
            # c in (0.6,1.4) via logit; pick candidates covering over/under dispersion
            cs = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
            grids.append(np.log((cs - 0.6) / (1.4 - cs)))                     # inverse logit→z
        if self.dependence == "frank":
            grids.append(np.array([-1.5, -0.8, -0.4, 0.0, 0.4, 0.8, 1.5]))   # kappa
        elif self.dependence == "bivpoi":
            grids.append(np.log(np.array([0.02, 0.05, 0.1, 0.2, 0.35])))      # lambda3
        if self.diagonal_inflation:
            ths = np.array([0.0, 0.02, 0.05, 0.08, 0.12, 0.18])
            ths = np.clip(ths, 1e-4, 0.299)
            grids.append(np.log(ths / (0.30 - ths)))                          # logit θ
        return grids

    def _shape_init_bounds(self):
        z, b = [], []
        if self.margin == "nb":
            z.append(np.log(8.0))         # log r
        elif self.margin == "weibull":
            z.append(0.0)                 # logit-ish for c around 1
        if self.dependence == "frank":
            z.append(0.0)                 # kappa
        elif self.dependence == "bivpoi":
            z.append(np.log(0.1))         # log lambda3
        if self.diagonal_inflation:
            z.append(-2.0)                # logit theta
        return np.array(z, dtype=float), b

    def _unpack_shape(self, z) -> dict:
        s = dict(self.shape_)
        idx = 0
        if self.margin == "nb":
            s["nb_r"] = float(np.exp(np.clip(z[idx], -20, 20))); idx += 1
        elif self.margin == "weibull":
            s["wc_c"] = float(0.6 + 0.8 / (1 + np.exp(-np.clip(z[idx], -30, 30)))); idx += 1
        if self.dependence == "frank":
            s["dep"] = float(z[idx]); idx += 1
        elif self.dependence == "bivpoi":
            s["dep"] = float(np.exp(np.clip(z[idx], -20, 5))); idx += 1
        if self.diagonal_inflation:
            s["theta_draw"] = float(0.30 / (1 + np.exp(-np.clip(z[idx], -30, 30)))); idx += 1
        return s

    # ==================================================================
    #  Core: construct score matrix given lambda
    # ==================================================================
    def _margin_pmf(self, mu: float, shape: dict) -> np.ndarray:
        if self.margin == "poisson":
            return cd.poisson_pmf_vec(mu, self.max_goals)
        if self.margin == "nb":
            return cd.negbin_pmf_vec(mu, shape["nb_r"], self.max_goals)
        return cd.weibull_count_pmf_vec(mu, shape["wc_c"], self.max_goals)

    def _build_matrix(self, lh: float, la: float, shape: dict) -> np.ndarray:
        if self.dependence == "bivpoi":
            # Bivariate Poisson uses lambda directly (its marginals are Poisson; incompatible with NB/Weibull marginals)
            M = biv.bivariate_poisson_matrix(lh, la, shape["dep"], self.max_goals)
        else:
            ph = self._margin_pmf(lh, shape)
            pa = self._margin_pmf(la, shape)
            if self.dependence == "frank":
                M = biv.frank_copula_matrix(ph, pa, shape["dep"])
            else:
                M = biv.independent_matrix(ph, pa)
        if self.diagonal_inflation and shape.get("theta_draw", 0.0) > 0:
            M = biv.diagonal_inflate(M, shape["theta_draw"])
        return M

    def score_matrix(self, lh: float, la: float) -> np.ndarray:
        """Public API: given expected goals, directly returns the calibrated score matrix (for use in dynamic filter pipeline)."""
        M = self._build_matrix(lh, la, self.shape_)
        if self.calibrator is not None and getattr(self.calibrator, "fitted_", False):
            M = self.calibrator.transform(M, lh, la)
        return M

    # ==================================================================
    #  drop-in predict (same structure as BayesianDixonColesModel.predict)
    # ==================================================================
    def predict(self, home_team: str, away_team: str, venue: str = "neutral") -> dict:
        hi, ai = self.team_idx[home_team], self.team_idx[away_team]
        vadj = {"home": self.params["home_adj"], "neutral": self.params["neutral_adj"]}.get(venue, 0.0)
        lh = float(np.exp(np.clip(self.params["attack"][hi] + self.params["defense"][ai] + vadj, -10, 10)))
        la = float(np.exp(np.clip(self.params["attack"][ai] + self.params["defense"][hi], -10, 10)))
        M = self.score_matrix(lh, la)
        h, d, a = biv.outcome_probs(M)
        score_probs = {
            f"{x}-{y}": round(float(M[x, y]), 4)
            for x in range(self.max_goals + 1) for y in range(self.max_goals + 1)
            if M[x, y] >= 0.005
        }
        return {
            "home_team": home_team, "away_team": away_team, "venue": venue,
            "expected_home_goals": round(lh, 3), "expected_away_goals": round(la, 3),
            "home_win_prob": round(h, 4), "draw_prob": round(d, 4), "away_win_prob": round(a, 4),
            "score_probs": score_probs, "score_matrix": M,
        }
