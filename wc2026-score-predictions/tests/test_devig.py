"""Test devig.py — Shin de-vig, proportional method, power method"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from quantbet.devig import (
    devig,
    devig_power,
    devig_proportional,
    devig_shin,
    implied_probabilities,
    overround,
)


class TestDevig:
    def test_overround(self) -> None:
        odds = [1.5, 4.0, 6.0]
        b = overround(odds)
        assert b > 0
        assert b > 1.0  # has overround

    def test_implied_probabilities(self) -> None:
        odds = [2.0, 3.0, 6.0]
        probs = implied_probabilities(odds)
        assert np.all(probs > 0)
        # 1/2+1/3+1/6 = 1.0 exactly no overround
        assert np.allclose(probs.sum(), 1.0)

    def test_devig_proportional(self) -> None:
        odds = [1.5, 4.0, 6.0]
        probs = devig_proportional(odds)
        assert np.allclose(probs.sum(), 1.0)
        assert probs[0] > probs[1] > probs[2]

    def test_devig_power(self) -> None:
        odds = [1.5, 4.0, 6.0]
        probs = devig_power(odds)
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs > 0)

    def test_devig_shin(self) -> None:
        odds = [1.5, 4.0, 6.0]
        probs = devig_shin(odds)
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs > 0)

    def test_devig_shin_with_z(self) -> None:
        odds = [1.5, 4.0, 6.0]
        probs, z = devig_shin(odds, return_z=True)
        assert np.allclose(probs.sum(), 1.0)
        assert 0 <= z < 1

    def test_devig_unit(self) -> None:
        odds = [1.5, 4.0, 6.0]
        for method in ("shin", "proportional", "power"):
            probs = devig(odds, method=method)
            assert np.allclose(probs.sum(), 1.0), f"method={method}"

    def test_devig_shin_balanced(self) -> None:
        """Balanced odds, Shin should be close to proportional method."""
        odds = [2.0, 3.0, 6.0]
        s = devig_shin(odds)
        p = devig_proportional(odds)
        assert np.allclose(s, p, atol=0.05)

    def test_devig_shin_extreme(self) -> None:
        """Extreme odds should not crash."""
        odds = [1.1, 10.0, 20.0]
        probs = devig_shin(odds)
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs > 0)

    def test_invalid_odds(self) -> None:
        with pytest.raises(ValueError):
            devig_proportional([1.0, 2.0, 3.0])
