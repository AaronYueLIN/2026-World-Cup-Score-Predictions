"""
confederation_prior.py — Confederation hierarchical prior (module 5).

WHY (World Cup specific)
------------------------
Your zero-sum constraints (spec 1.6) assume all teams sit on one
comparable scale. But UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC teams play
each other almost only at World Cups and in sparse friendlies, so the
cross-confederation anchoring is weak and the attack/defence scale can
drift between confederations. A confederation-level random effect supplies
a shared anchor per confederation:

    beta_att_i ~ N(mu_conf(i)^att, sigma_att^2)
    mu_conf^att ~ N(0, tau_att^2)

This partial-pools each confederation's teams toward a confederation mean
that is itself estimated from whatever cross-confederation games exist
(World Cup history + inter-confederation friendlies). It composes with the
rating prior (module 1): you can use BOTH — rating sets the within-conf
ordering, confederation sets the between-conf level.

Also handles the related point: in a World Cup model, friendlies are one of
the FEW cross-confederation information sources, so their tournament weight
(spec 1.4, currently 0.35) should not be crushed. `suggest_friendly_weight`
gives a principled bump for *inter-confederation* friendlies specifically.

Literature
----------
- Standard hierarchical/partial-pooling random effects (Gelman & Hill).
- Groll et al. World Cup models use confederation/region effects as covariates.

HOW TO WIRE INTO QuantBet-EV v7.0
---------------------------------
Like module 1, this returns prior-mean vectors + extra NLP terms. Append
the confederation means mu_conf^att, mu_conf^def and the hyper-SDs
tau_att, tau_def to your theta vector. In the NLP:

    cp = ConfederationPrior(conf_ids)
    nlp += cp.nlp(beta_att, beta_def, mu_conf_att, mu_conf_def,
                  sigma_att, sigma_def, tau_att, tau_def)

Combine with module 1 by using the rating prior for the WITHIN-conf mean
and this for the BETWEEN-conf mean (see `combined_prior_mean`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass
class ConfederationPrior:
    """Confederation-level hierarchical (partial-pooling) prior.

    Parameters
    ----------
    conf_ids
        Per-team confederation label, team-ordered, same order as beta.
        e.g. ['UEFA','CONMEBOL','UEFA','CAF',...]. Any hashable labels work.
    tau_prior_sd
        SD of the half-normal-ish hyper-prior on tau (the between-conf SD).
        Implemented as a soft log-normal penalty; default 1.0.
    """

    conf_ids: Sequence[str]
    tau_prior_sd: float = 1.0
    _labels: list[str] = field(init=False, repr=False)
    _index: IntArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._labels = sorted(set(self.conf_ids))
        lut = {c: i for i, c in enumerate(self._labels)}
        self._index = np.array([lut[c] for c in self.conf_ids], dtype=np.int64)

    @property
    def n_conf(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def expand(self, mu_conf: FloatArray) -> FloatArray:
        """Broadcast per-confederation means to per-team vector."""
        mu_conf = np.asarray(mu_conf, dtype=np.float64).ravel()
        if mu_conf.size != self.n_conf:
            raise ValueError(f"expected {self.n_conf} conf means, got {mu_conf.size}")
        return mu_conf[self._index]

    def nlp(
        self,
        beta_att: FloatArray,
        beta_def: FloatArray,
        mu_conf_att: FloatArray,
        mu_conf_def: FloatArray,
        sigma_att: float,
        sigma_def: float,
        tau_att: float,
        tau_def: float,
    ) -> float:
        """Full NLP contribution: team-level + confederation-level + hyper.

        Replaces the zero-mean attack/defence prior terms (spec 1.5) with a
        two-level hierarchy. Returns a single scalar to add to total NLP.
        """
        beta_att = np.asarray(beta_att, dtype=np.float64).ravel()
        beta_def = np.asarray(beta_def, dtype=np.float64).ravel()
        n = beta_att.size

        mu_att_team = self.expand(mu_conf_att)
        mu_def_team = self.expand(mu_conf_def)

        # team-level: beta ~ N(mu_conf, sigma^2)
        nlp_att = (((beta_att - mu_att_team) ** 2).sum() / (2.0 * sigma_att**2)
                   + n * np.log(sigma_att))
        nlp_def = (((beta_def - mu_def_team) ** 2).sum() / (2.0 * sigma_def**2)
                   + n * np.log(sigma_def))

        # confederation-level: mu_conf ~ N(0, tau^2)
        nc = self.n_conf
        nlp_mu_att = ((np.asarray(mu_conf_att) ** 2).sum() / (2.0 * tau_att**2)
                      + nc * np.log(tau_att))
        nlp_mu_def = ((np.asarray(mu_conf_def) ** 2).sum() / (2.0 * tau_def**2)
                      + nc * np.log(tau_def))

        # hyper-prior on tau (log-normal, prevents tau -> 0 degeneracy)
        nlp_tau = ((np.log(tau_att) ** 2 + np.log(tau_def) ** 2)
                   / (2.0 * self.tau_prior_sd**2))

        return float(nlp_att + nlp_def + nlp_mu_att + nlp_mu_def + nlp_tau)

    def combined_prior_mean(
        self,
        mu_conf: FloatArray,
        rating_within: Optional[FloatArray] = None,
        eta_within: float = 0.0,
    ) -> FloatArray:
        """Compose confederation level + rating within-conf ordering.

        prior_mean_i = mu_conf(i) + eta_within * rating_i
        where rating_i should be CONFEDERATION-DEMEANED so it only encodes
        within-confederation ordering (the between-conf level comes from
        mu_conf). Use this to feed module 1 and module 5 together.
        """
        base = self.expand(mu_conf)
        if rating_within is None or eta_within == 0.0:
            return base
        r = np.asarray(rating_within, dtype=np.float64).ravel()
        return base + eta_within * r

    def demean_rating_within_conf(self, ratings_std: FloatArray) -> FloatArray:
        """Remove each confederation's mean rating, leaving within-conf signal.

        Pass standardised ratings; returns ratings with per-confederation
        mean subtracted, so module 1's eta only captures within-conf ordering
        and does not double-count the between-conf level handled here.
        """
        r = np.asarray(ratings_std, dtype=np.float64).ravel()
        out = r.copy()
        for c in range(self.n_conf):
            mask = self._index == c
            if mask.any():
                out[mask] = r[mask] - r[mask].mean()
        return out


def suggest_friendly_weight(
    base_friendly_weight: float = 0.35,
    *,
    is_inter_confederation: bool = False,
    bump: float = 0.20,
    cap: float = 0.60,
) -> float:
    """Principled tournament weight for a friendly (spec 1.4 C_tournament).

    Inter-confederation friendlies are rare cross-conf information, so they
    get a bump over intra-confederation friendlies (which are plentiful and
    less informative for World Cup cross-conf calibration).

    Returns a weight to use in place of the flat 0.35 for friendlies.
    """
    if not is_inter_confederation:
        return base_friendly_weight
    return float(min(base_friendly_weight + bump, cap))


__all__ = [
    "ConfederationPrior",
    "suggest_friendly_weight",
]
