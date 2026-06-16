"""
dynamic_strength.py — Time-varying team strength online filtering (a deployable approximation of Koopman-Lit)
====================================================================

Why (this is the single biggest lever for score accuracy)
-----------------------------------
The current main model is a *one-shot static* MLE/MAP + exponential time decay weights. But decay weights only "down-weight old
data"; they still give the *average strength over the entire window*, reacting slowly to **inflection points**
(manager changes, signings, injuries, form collapse) — yet inflection points are exactly where the odds market is
most wrong and the model can profit most.

The academic frontier (Koopman & Lit 2015, JRSS-A; Rue & Salvesen 2000; Owen 2011) models attack/defence
strength as a **time-evolving latent state**, filtered/smoothed via state-space models. The full version requires
non-Gaussian simulation smoothers, which are heavy. This module implements an **extended Kalman / linear Bayesian**
approximation (West & Harrison DLM style), O(matches) complexity, no MCMC, online-incremental:

  State:   each team (att_i, def_i) as Gaussian belief  N(m_i, P_i)
  Evolution: random walk + mean reversion toward prior mean (Owen 2011's OU idea, prevents drift)
             m ← μ_prior + φ·(m − μ_prior),   P ← φ·P + σ_w·Δdays
  Observation: goals ~ Poisson(λ), log λ = att_h + def_a + venue
  Update:   for log-link Poisson, score = (g − λ), information = λ
             P_post = 1/(1/P + λ),  m_post = m + P_post·(g − λ)
           (single-step Newton = Kalman gain, Gaussian approximation for Poisson observation)

Outputs λ_h, λ_a as *one-step-ahead* predictions, can be fed directly to scoreline.bivariate
to generate score matrices, forming a complete frontier pipeline of "dynamic strength → flexible dependent marginals".

References
--------
Koopman & Lit (2015) "A dynamic bivariate Poisson model ... EPL", JRSS-A 178.
Rue & Salvesen (2000) Scand. J. Statist. 27, 385-402.
Owen (2011) "Dynamic Bayesian forecasting models of football match outcomes",
    IMA J. Management Mathematics.  (discrete-time random walk att/def)
West & Harrison (1997) "Bayesian Forecasting and Dynamic Models", Ch.13-14.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .count_dists import poisson_pmf_vec as cd_poisson

__all__ = ["DynamicStrengthFilter"]

_VENUE_ADJ_KEY = {"home": "home_adj", "neutral": "neutral_adj", "away": None}


@dataclass
class _TeamState:
    att_m: float
    att_P: float
    def_m: float
    def_P: float
    last_date: Optional[pd.Timestamp] = None
    n_matches: int = 0


class DynamicStrengthFilter:
    """
    Online filter for time-varying attack/defence strength.

    Typical usage (backtesting / production):

        # 1) First get prior means (att0/def0) and venue parameters from static Bayesian DC
        flt = DynamicStrengthFilter(
            attack0=dict(zip(dc.teams, dc.params["attack"])),
            defense0=dict(zip(dc.teams, dc.params["defense"])),
            home_adj=dc.params["home_adj"],
            neutral_adj=dc.params["neutral_adj"],
            process_sd_per_year=0.30,   # annual strength drift magnitude (tune)
            mean_reversion_halflife_days=365,
        )

        # 2) Feed history chronologically (online filtering), simultaneously collecting one-step-ahead predictions for backtesting
        preds = flt.run(history_df, collect_oos=True)

        # 3) Predict expected goals for a future match at any point in time
        lh, la = flt.expected_goals("Brazil", "Morocco", venue="neutral")

    All strength parameters share the same scale (log-additive) as the static model, so they can seamlessly plug into the existing predict pipeline.
    """

    def __init__(
        self,
        attack0: dict[str, float],
        defense0: dict[str, float],
        home_adj: float = 0.25,
        neutral_adj: float = 0.10,
        process_sd_per_year: float = 0.30,
        mean_reversion_halflife_days: Optional[float] = 365.0,
        init_sd: float = 0.20,
        max_goals_cap: float = 12.0,
    ) -> None:
        """
        Args:
            attack0/defense0: prior means (from static Bayesian DC fit). Also the mean reversion target.
            home_adj/neutral_adj: venue log adjustments (carried over from static estimate, not filtered).
            process_sd_per_year: annual standard deviation of the strength random walk σ_w. Larger = more sensitive, more jittery.
                                 Football empirical range 0.2~0.4; =0 degenerates to static.
            mean_reversion_halflife_days: half-life of mean reversion toward prior mean (Owen OU).
                                 None = pure random walk (no reversion).
            init_sd: initial state standard deviation (reflects uncertainty in att0/def0).
            max_goals_cap: lambda upper limit, numerical explosion prevention.
        """
        self.mu_att = dict(attack0)
        self.mu_def = dict(defense0)
        self.home_adj = float(home_adj)
        self.neutral_adj = float(neutral_adj)
        self.sigma_w2_per_day = (process_sd_per_year ** 2) / 365.0
        self.init_P = init_sd ** 2
        self.max_log = np.log(max_goals_cap)

        if mean_reversion_halflife_days is not None and mean_reversion_halflife_days > 0:
            # φ_per_day = 0.5 ** (1/halflife)
            self.phi_per_day = 0.5 ** (1.0 / mean_reversion_halflife_days)
        else:
            self.phi_per_day = 1.0  # no reversion

        self.state: dict[str, _TeamState] = {}
        for t in set(self.mu_att) | set(self.mu_def):
            self.state[t] = _TeamState(
                att_m=self.mu_att.get(t, 0.0), att_P=self.init_P,
                def_m=self.mu_def.get(t, 0.0), def_P=self.init_P,
            )

    # ------------------------------------------------------------------
    def _ensure_team(self, t: str) -> _TeamState:
        if t not in self.state:
            # new team: prior mean 0, larger uncertainty
            self.state[t] = _TeamState(0.0, self.init_P * 4, 0.0, self.init_P * 4)
            self.mu_att.setdefault(t, 0.0)
            self.mu_def.setdefault(t, 0.0)
        return self.state[t]

    def _venue_adj(self, venue: str) -> float:
        if venue == "home":
            return self.home_adj
        if venue == "neutral":
            return self.neutral_adj
        return 0.0

    # ------------------------------------------------------------------
    #  Time update (predict step): advance state to date
    # ------------------------------------------------------------------
    def _evolve(self, st: _TeamState, date: pd.Timestamp) -> None:
        if st.last_date is None:
            st.last_date = date
            return
        dt = max((date - st.last_date).days, 0)
        if dt == 0:
            return
        phi = self.phi_per_day ** dt
        # mean reversion toward prior
        # (note: prior means are fetched per team outside _evolve; placeholder here, actual reversion inside step)
        st.att_P = phi * phi * st.att_P + self.sigma_w2_per_day * dt
        st.def_P = phi * phi * st.def_P + self.sigma_w2_per_day * dt
        st.last_date = date

    def _revert_mean(self, st: _TeamState, team: str, date: pd.Timestamp) -> None:
        if st.last_date is None:
            return
        dt = max((date - st.last_date).days, 0)
        if dt == 0:
            return
        phi = self.phi_per_day ** dt
        st.att_m = self.mu_att[team] + phi * (st.att_m - self.mu_att[team])
        st.def_m = self.mu_def[team] + phi * (st.def_m - self.mu_def[team])

    # ------------------------------------------------------------------
    #  One-step-ahead expected goals (using current filtered means)
    # ------------------------------------------------------------------
    def expected_goals(
        self, home: str, away: str, venue: str = "neutral",
        as_of: Optional[pd.Timestamp] = None,
    ) -> tuple[float, float]:
        sh, sa = self._ensure_team(home), self._ensure_team(away)
        if as_of is not None:
            self._revert_mean(sh, home, as_of)
            self._revert_mean(sa, away, as_of)
        log_lh = np.clip(sh.att_m + sa.def_m + self._venue_adj(venue), -self.max_log, self.max_log)
        log_la = np.clip(sa.att_m + sh.def_m, -self.max_log, self.max_log)
        return float(np.exp(log_lh)), float(np.exp(log_la))

    # ------------------------------------------------------------------
    #  Observation update (one match)
    # ------------------------------------------------------------------
    def step(
        self, home: str, away: str, gh: int, ga: int,
        venue: str = "neutral", date: Optional[pd.Timestamp] = None,
    ) -> None:
        sh, sa = self._ensure_team(home), self._ensure_team(away)
        if date is not None:
            # 1) Mean reversion + variance inflation (time update)
            self._revert_mean(sh, home, date); self._revert_mean(sa, away, date)
            self._evolve(sh, date); self._evolve(sa, date)

        va = self._venue_adj(venue)
        log_lh = np.clip(sh.att_m + sa.def_m + va, -self.max_log, self.max_log)
        log_la = np.clip(sa.att_m + sh.def_m, -self.max_log, self.max_log)
        lh, la = np.exp(log_lh), np.exp(log_la)

        # 2) Kalman/linear Bayesian update (Poisson observation, log-link)
        #    home goals information → update att_home & def_away
        #    away goals information → update att_away & def_home
        self._kalman_update(sh, "att", gh - lh, lh)
        self._kalman_update(sa, "def", gh - lh, lh)
        self._kalman_update(sa, "att", ga - la, la)
        self._kalman_update(sh, "def", ga - la, la)

        sh.n_matches += 1; sa.n_matches += 1

    @staticmethod
    def _kalman_update(st: _TeamState, which: str, resid: float, info: float) -> None:
        """Gaussian-approximated Poisson update for a single parameter. resid=(g-lambda), info=lambda."""
        if which == "att":
            P = st.att_P
            P_post = 1.0 / (1.0 / P + info)
            st.att_m = st.att_m + P_post * resid
            st.att_P = P_post
        else:
            P = st.def_P
            P_post = 1.0 / (1.0 / P + info)
            st.def_m = st.def_m + P_post * resid
            st.def_P = P_post

    # ------------------------------------------------------------------
    #  Batch online filtering (main backtesting entry point)
    # ------------------------------------------------------------------
    def run(
        self, df: pd.DataFrame, collect_oos: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Online filter over the entire table in chronological order.

        df must contain: date, home_team, away_team, home_goals, away_goals[, venue].

        If collect_oos=True, returns the **pre-observation** one-step-ahead predictions lambda_h/lambda_a
        for each match (no data leakage, can be used directly for OOS backtesting vs the static model).
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        rows = []
        for _, r in df.iterrows():
            venue = r.get("venue", "neutral")
            date = r["date"]
            if collect_oos:
                lh, la = self.expected_goals(r["home_team"], r["away_team"], venue, as_of=date)
                rows.append({
                    "date": date, "home_team": r["home_team"], "away_team": r["away_team"],
                    "pred_lambda_h": lh, "pred_lambda_a": la,
                    "home_goals": r["home_goals"], "away_goals": r["away_goals"], "venue": venue,
                })
            self.step(
                r["home_team"], r["away_team"],
                int(r["home_goals"]), int(r["away_goals"]),
                venue=venue, date=date,
            )
        return pd.DataFrame(rows) if collect_oos else None

    @staticmethod
    def tune_process_sd(
        attack0, defense0, home_adj, neutral_adj,
        train_df: pd.DataFrame, val_df: pd.DataFrame,
        candidates=(0.0, 0.15, 0.25, 0.4, 0.6, 0.85),
        halflife_days=540.0,
    ) -> float:
        """
        Select the optimal process_sd_per_year via one-step-ahead RPS on the validation set (no leakage).
        candidates includes 0.0 (= degenerate to static), so "dynamic is never worse than static" is determined by the data.
        """
        train_df = train_df.copy(); val_df = val_df.copy()
        train_df["date"] = pd.to_datetime(train_df["date"])
        val_df["date"] = pd.to_datetime(val_df["date"])
        best_sd, best_rps = candidates[0], np.inf
        for sd in candidates:
            flt = DynamicStrengthFilter(attack0, defense0, home_adj, neutral_adj,
                                        process_sd_per_year=sd,
                                        mean_reversion_halflife_days=halflife_days)
            flt.run(train_df, collect_oos=False)
            rps = []
            for r in val_df.sort_values("date").itertuples():
                venue = getattr(r, "venue", "neutral")
                lh, la = flt.expected_goals(r.home_team, r.away_team, venue, as_of=r.date)
                # Evaluate RPS with simple Poisson 1X2 (only for selecting sd, decoupled from the final engine)
                K = 10
                ph = cd_poisson(lh, K); pa = cd_poisson(la, K)
                M = np.outer(ph, pa)
                h = float(np.tril(M, -1).sum()); d = float(np.trace(M)); a = float(np.triu(M, 1).sum())
                p = np.array([h, d, a]); p /= p.sum()
                y = 0 if r.home_goals > r.away_goals else (1 if r.home_goals == r.away_goals else 2)
                e = np.zeros(3); e[y] = 1.0
                rps.append(float(((np.cumsum(p) - np.cumsum(e)) ** 2).sum() / 2.0))
                flt.step(r.home_team, r.away_team, int(r.home_goals), int(r.away_goals), venue, r.date)
            m = float(np.mean(rps))
            if m < best_rps:
                best_rps, best_sd = m, sd
        return best_sd

    def current_ratings(self) -> pd.DataFrame:
        """Current filtered attack/defence strengths (including uncertainty)."""
        recs = []
        for t, st in self.state.items():
            recs.append({
                "team": t, "attack": st.att_m, "attack_sd": np.sqrt(st.att_P),
                "defense": st.def_m, "defense_sd": np.sqrt(st.def_P),
                "overall": st.att_m - st.def_m, "n_matches": st.n_matches,
            })
        return pd.DataFrame(recs).sort_values("overall", ascending=False).reset_index(drop=True)
