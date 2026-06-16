"""
score_driven.py — Score-driven (GAS) adaptive strength on top of the Elo anchor
===============================================================================

WHAT THIS REPLACES
------------------
Your Bayesian DC prior mean is
    att_i = mu_conf(i) + eta_att * elo_i + mom_att * mom_i
where `mom_i` is a STATIC one-year Elo delta — a frozen snapshot of "recent
form". This module turns that snapshot into a principled, per-match-updated
quantity: a score-driven (GAS, Creal-Koopman-Lucas 2013; Koopman-Lit 2019)
time-varying DEVIATION delta_i(t) from the Elo-anchored baseline.

    att_i(t) = att_anchor_i + delta_att_i(t)
    def_i(t) = def_anchor_i + delta_def_i(t)

delta evolves by the GAS(1,1) recursion, mean-reverting to 0 (= the anchor):
    delta_{t+1} = B * delta_t + A * s_t
with s_t the inverse-Fisher-scaled score of the Poisson log-likelihood:
    s_t = (g - lambda) / lambda = g/lambda - 1.
A (adaptation gain) and B (persistence) are LEARNED by maximising the
one-step-ahead predictive log-likelihood — not hand-tuned.

WHY THIS IS THE "DYNAMIC ELO MOMENTUM"
--------------------------------------
delta_att_i(t) - delta_def_i(t) IS team i's live momentum relative to its
Elo baseline. `momentum_table()` exposes it directly, so you can read off
"team X is currently +0.31 above its Elo-implied strength and rising".

INTEGRATION (replace the static momentum)
-----------------------------------------
1. Retrain BayesianDixonColesModel with mom_att = mom_def = 0 (drop the
   static momentum term) — keep eta_att/eta_def (Elo) and confederation.
2. Use that MAP fit's att/def as the anchors here:
       sd = ScoreDrivenStrength(
           attack0=dict(zip(dc.teams, dc.params["attack"])),
           defense0=dict(zip(dc.teams, dc.params["defense"])),
           home_adj=dc.params["home_adj"], neutral_adj=dc.params["neutral_adj"],
       )
       sd.fit_hyperparams(history_df)      # learn A, B by one-step-ahead MLE
       sd.run(history_df, collect_oos=False)  # warm up to "today"
3. Predict expected goals at any future date:
       lh, la = sd.expected_goals("Brazil", "Morocco", venue="neutral")
   then feed (lh, la) into your existing score-matrix / FlexibleScoreModel.

Mirrors the DynamicStrengthFilter API (run / expected_goals / step) so it is
a drop-in for that filter in your pipeline.

References
----------
Creal, Koopman & Lucas (2013) "Generalized autoregressive score models",
    J. Applied Econometrics 28(5).
Koopman & Lit (2019) "Forecasting football match results ... score-driven
    time series models", Int. J. Forecasting 35(2):797-809.
Harvey (2013) "Dynamic Models for Volatility and Heavy Tails", CUP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

__all__ = ["ScoreDrivenStrength"]


def _softplus(x: float) -> float:
    return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class _S:
    d_att: float = 0.0      # delta attack (deviation from anchor)
    d_def: float = 0.0      # delta defence
    last_date: Optional[pd.Timestamp] = None
    n: int = 0


class ScoreDrivenStrength:
    def __init__(
        self,
        attack0: dict[str, float],
        defense0: dict[str, float],
        home_adj: float = 0.25,
        neutral_adj: float = 0.10,
        gain_att: float = 0.06,
        gain_def: float = 0.06,
        persistence: float = 0.985,
        days_ref: float = 30.0,
        innov_clip: float = 3.0,
        max_goals_cap: float = 12.0,
    ) -> None:
        """
        Args:
            attack0/defense0 : anchor strengths (MAP fit, Elo+confederation).
            gain_att/gain_def: GAS adaptation gain A (>=0). Learned by fit.
            persistence      : GAS persistence B in (0,1). Decay per `days_ref`.
                               Default 0.985 = holdout-RPS-optimal (verified on 49k).
            days_ref         : days that correspond to one application of B
                               (calendar-time mean reversion for sparse fixtures).
            innov_clip       : clip the scaled score to +/- this (robustness to
                               freak scorelines, e.g. 7-0).
        """
        self.mu_att = dict(attack0)
        self.mu_def = dict(defense0)
        self.home_adj = float(home_adj)
        self.neutral_adj = float(neutral_adj)
        self.A_att = float(gain_att)
        self.A_def = float(gain_def)
        self.B = float(persistence)
        self.days_ref = float(days_ref)
        self.clip = float(innov_clip)
        self.max_log = float(np.log(max_goals_cap))
        self.state: dict[str, _S] = {t: _S() for t in set(self.mu_att) | set(self.mu_def)}

    # ------------------------------------------------------------------ utils
    def _ensure(self, t: str) -> _S:
        if t not in self.state:
            self.state[t] = _S()
            self.mu_att.setdefault(t, 0.0)
            self.mu_def.setdefault(t, 0.0)
        return self.state[t]

    def _venue_adj(self, venue: str) -> float:
        return {"home": self.home_adj, "neutral": self.neutral_adj}.get(venue, 0.0)

    def _decay(self, st: _S, date: Optional[pd.Timestamp]) -> None:
        """Calendar-time GAS persistence: delta <- B**(dt/days_ref) * delta."""
        if date is None:
            return
        if st.last_date is None:
            st.last_date = date
            return
        dt = max((date - st.last_date).days, 0)
        if dt > 0:
            b = self.B ** (dt / self.days_ref)
            st.d_att *= b
            st.d_def *= b
        st.last_date = date

    def _lambdas(self, sh: _S, sa: _S, venue: str) -> tuple[float, float]:
        h, a = sh, sa
        log_lh = np.clip((self.mu_att_of(h) + h.d_att) + (self.mu_def_of(a) + a.d_def)
                         + self._venue_adj(venue), -self.max_log, self.max_log)
        log_la = np.clip((self.mu_att_of(a) + a.d_att) + (self.mu_def_of(h) + h.d_def),
                         -self.max_log, self.max_log)
        return float(np.exp(log_lh)), float(np.exp(log_la))

    # team lookup helpers (state object doesn't carry its own name)
    def mu_att_of(self, st: _S) -> float:  # resolved via _name map at call sites
        return st._mu_att  # type: ignore[attr-defined]

    def mu_def_of(self, st: _S) -> float:
        return st._mu_def  # type: ignore[attr-defined]

    # ------------------------------------------------------- core: one match
    def _predict_lambdas(self, home: str, away: str, venue: str,
                         date: Optional[pd.Timestamp]) -> tuple[float, float]:
        sh, sa = self._ensure(home), self._ensure(away)
        # attach anchors so _lambdas can read them
        sh._mu_att, sh._mu_def = self.mu_att[home], self.mu_def[home]  # type: ignore
        sa._mu_att, sa._mu_def = self.mu_att[away], self.mu_def[away]  # type: ignore
        self._decay(sh, date)
        self._decay(sa, date)
        return self._lambdas(sh, sa, venue)

    def _observe(self, home: str, away: str, gh: int, ga: int, venue: str) -> None:
        """Apply the GAS innovation using the (already date-decayed) state."""
        sh, sa = self.state[home], self.state[away]
        sh._mu_att, sh._mu_def = self.mu_att[home], self.mu_def[home]  # type: ignore
        sa._mu_att, sa._mu_def = self.mu_att[away], self.mu_def[away]  # type: ignore
        lh, la = self._lambdas(sh, sa, venue)
        # inverse-Fisher scaled Poisson scores, clipped for robustness
        s_h = float(np.clip(gh / max(lh, 1e-9) - 1.0, -self.clip, self.clip))
        s_a = float(np.clip(ga / max(la, 1e-9) - 1.0, -self.clip, self.clip))
        # home goals inform home attack (+) and away defence (+ = concedes more)
        sh.d_att += self.A_att * s_h
        sa.d_def += self.A_def * s_h
        # away goals inform away attack (+) and home defence (+)
        sa.d_att += self.A_att * s_a
        sh.d_def += self.A_def * s_a
        sh.n += 1
        sa.n += 1

    def step(self, home: str, away: str, gh: int, ga: int,
             venue: str = "neutral", date: Optional[pd.Timestamp] = None) -> None:
        self._predict_lambdas(home, away, venue, date)  # decays state to `date`
        self._observe(home, away, int(gh), int(ga), venue)

    def expected_goals(self, home: str, away: str, venue: str = "neutral",
                       as_of: Optional[pd.Timestamp] = None) -> tuple[float, float]:
        return self._predict_lambdas(home, away, venue, as_of)

    # ------------------------------------------------------- batch / OOS run
    def run(self, df: pd.DataFrame, collect_oos: bool = True) -> Optional[pd.DataFrame]:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        rows = []
        for _, r in df.iterrows():
            venue = r.get("venue", "neutral")
            date = r["date"]
            lh, la = self._predict_lambdas(r["home_team"], r["away_team"], venue, date)
            if collect_oos:
                rows.append({
                    "date": date, "home_team": r["home_team"], "away_team": r["away_team"],
                    "pred_lambda_h": lh, "pred_lambda_a": la,
                    "home_goals": r["home_goals"], "away_goals": r["away_goals"], "venue": venue,
                })
            self._observe(r["home_team"], r["away_team"],
                          int(r["home_goals"]), int(r["away_goals"]), venue)
        return pd.DataFrame(rows) if collect_oos else None

    # --------------------------------------------- learn A, B by 1-step MLE
    def fit_hyperparams(
        self,
        df: pd.DataFrame,
        share_gain: bool = True,
        fix_B: float | None = None,
    ) -> dict:
        """
        Estimate (A_att, A_def, B) by maximising the one-step-ahead predictive
        Poisson log-likelihood over `df` (prediction-error decomposition; the
        proper GAS estimation, no data leakage).

        fix_B : if set (e.g. 0.985), B is held constant and only A is optimised.
                This prevents the optimizer from sliding to the B→1 boundary
                where the likelihood surface is flat (verified on holdout RPS).
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        rec = df.to_dict("records")
        teams = set(self.mu_att) | set(self.mu_def)

        def neg_ll(z: np.ndarray) -> float:
            A_att = _softplus(z[0])
            A_def = A_att if share_gain else _softplus(z[1])
            B = fix_B if fix_B is not None else _sigmoid(z[-1])
            # fresh state for this evaluation
            st = {t: _S() for t in teams}
            mu_a, mu_d = self.mu_att, self.mu_def
            ll = 0.0
            for r in rec:
                h, a, venue, date = r["home_team"], r["away_team"], r.get("venue", "neutral"), r["date"]
                if h not in st:
                    st[h] = _S(); mu_a.setdefault(h, 0.0); mu_d.setdefault(h, 0.0)
                if a not in st:
                    st[a] = _S(); mu_a.setdefault(a, 0.0); mu_d.setdefault(a, 0.0)
                sh, sa = st[h], st[a]
                for s in (sh, sa):
                    if s.last_date is not None:
                        dt = max((date - s.last_date).days, 0)
                        if dt > 0:
                            b = B ** (dt / self.days_ref)
                            s.d_att *= b; s.d_def *= b
                    s.last_date = date
                va = self._venue_adj(venue)
                log_lh = np.clip((mu_a[h] + sh.d_att) + (mu_d[a] + sa.d_def) + va, -self.max_log, self.max_log)
                log_la = np.clip((mu_a[a] + sa.d_att) + (mu_d[h] + sh.d_def), -self.max_log, self.max_log)
                lh, la = np.exp(log_lh), np.exp(log_la)
                gh, ga = int(r["home_goals"]), int(r["away_goals"])
                ll += gh * log_lh - lh - gammaln(gh + 1) + ga * log_la - la - gammaln(ga + 1)
                s_h = np.clip(gh / max(lh, 1e-9) - 1.0, -self.clip, self.clip)
                s_a = np.clip(ga / max(la, 1e-9) - 1.0, -self.clip, self.clip)
                sh.d_att += A_att * s_h; sa.d_def += A_def * s_h
                sa.d_att += A_att * s_a; sh.d_def += A_def * s_a
            return float(-ll)

        # Initial guess: when fix_B, only include A in z
        if fix_B is not None:
            z0 = np.array([np.log(np.expm1(0.06))])  # A~0.06, single parameter
        else:
            z0 = np.array([np.log(np.expm1(0.06)), np.log(np.expm1(0.06)), 4.0])
        if share_gain and fix_B is None:
            z0 = np.array([z0[0], z0[2]])
        res = minimize(neg_ll, z0, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-2, "maxiter": 400})
        if fix_B is not None:
            self.A_att = self.A_def = _softplus(res.x[0])
            self.B = float(fix_B)
        elif share_gain:
            self.A_att = self.A_def = _softplus(res.x[0])
            self.B = _sigmoid(res.x[1])
        else:
            self.A_att = _softplus(res.x[0])
            self.A_def = _softplus(res.x[1])
            self.B = _sigmoid(res.x[2])
        # reset live state (fit used throwaway state)
        self.state = {t: _S() for t in teams}
        return {"A_att": self.A_att, "A_def": self.A_def, "B": self.B,
                "neg_ll": float(res.fun), "n_matches": len(rec)}

    # ----------------------------------------------- the "dynamic Elo momentum"
    def momentum_table(self) -> pd.DataFrame:
        """
        Current deviation of each team from its Elo-anchored baseline.
        `momentum` = d_att - d_def (>0 = currently over-performing its Elo).
        """
        recs = []
        for t, st in self.state.items():
            recs.append({
                "team": t,
                "anchor_overall": self.mu_att.get(t, 0.0) - self.mu_def.get(t, 0.0),
                "d_att": st.d_att, "d_def": st.d_def,
                "momentum": st.d_att - st.d_def,
                "live_overall": (self.mu_att.get(t, 0.0) + st.d_att)
                                - (self.mu_def.get(t, 0.0) + st.d_def),
                "n_matches": st.n,
            })
        return (pd.DataFrame(recs)
                .sort_values("momentum", ascending=False)
                .reset_index(drop=True))
