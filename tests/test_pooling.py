"""Test pooling.py — Linear pooling, log pooling, weight optimization"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from quantbet.pooling import linear_pool, log_pool, optimize_weight
from quantbet.evaluation import rps_score


class TestPooling:
    def test_linear_pool_basic(self) -> None:
        P1 = np.array([0.8, 0.1, 0.1])
        P2 = np.array([0.2, 0.6, 0.2])
        result = linear_pool([P1, P2], [0.5, 0.5])
        assert np.allclose(result.sum(), 1.0)
        assert result[0] == pytest.approx(0.5, abs=0.01)

    def test_log_pool_basic(self) -> None:
        P1 = np.array([0.8, 0.1, 0.1])
        P2 = np.array([0.2, 0.6, 0.2])
        result = log_pool([P1, P2], [0.5, 0.5])
        assert np.allclose(result.sum(), 1.0)

    def test_log_pool_extreme(self) -> None:
        """Extreme probabilities should not cause numerical issues."""
        P1 = np.array([1.0, 0.0, 0.0])
        P2 = np.array([0.0, 0.0, 1.0])
        result = log_pool([P1, P2], [0.5, 0.5], eps=1e-12)
        assert np.allclose(result.sum(), 1.0)

    def test_log_pool_single_model(self) -> None:
        P = np.array([0.7, 0.2, 0.1])
        result = log_pool([P], [1.0])
        assert np.allclose(result, P, atol=1e-6)

    def test_log_pool_weight_effect(self) -> None:
        P1 = np.array([0.9, 0.05, 0.05])
        P2 = np.array([0.1, 0.8, 0.1])
        r1 = log_pool([P1, P2], [1.0, 0.0])
        r2 = log_pool([P1, P2], [0.0, 1.0])
        assert r1[0] > 0.8  # weight on P1 -> close to P1
        assert r2[1] > 0.6  # weight on P2 -> close to P2

    def test_optimize_weight(self) -> None:
        rng = np.random.default_rng(42)
        P1 = rng.dirichlet([5, 2, 1], size=20)
        P2 = rng.dirichlet([1, 2, 5], size=20)
        y = np.argmax(P1, axis=1)  # P1 is better
        w = optimize_weight(P1, P2, y, method="log")
        assert 0 <= w <= 1
        assert w > 0.5  # P1 more accurate -> weight should be > 0.5

    def test_optimize_weight_linear(self) -> None:
        rng = np.random.default_rng(42)
        P1 = rng.dirichlet([5, 2, 1], size=10)
        P2 = rng.dirichlet([1, 5, 2], size=10)
        y = np.argmax(P1, axis=1)
        w = optimize_weight(P1, P2, y, method="linear")
        assert 0 <= w <= 1
