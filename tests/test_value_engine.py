"""
Unit tests for Value Engine
Verify EV / Kelly formula correctness
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import pytest
from value_engine import ValueEngine


class TestValueEngine:
    """Value Engine test suite"""

    def setup_method(self):
        self.engine = ValueEngine(bankroll=1000, kelly_fraction=0.25)

    # ===== De-vigging Tests =====

    def test_devig_balanced_odds(self):
        """Test: standard odds (no margin) normalize after de-vig"""
        # Three-way, odds exactly match true probabilities
        odds = {'home': 2.0, 'draw': 3.0, 'away': 6.0}
        fair, margin = self.engine.devig_odds(odds)
        # The three fair_probs should equal 1/odds normalized
        assert abs(margin) < 0.01  # no margin
        assert abs(sum(fair.values()) - 1.0) < 0.001

    def test_devig_with_margin(self):
        """Test: odds with margin normalize after de-vig"""
        # Bookmaker added ~4% margin
        odds = {'home': 1.8, 'draw': 3.5, 'away': 5.0}
        fair, margin = self.engine.devig_odds(odds)
        assert margin > 0.03  # has positive margin
        assert abs(sum(fair.values()) - 1.0) < 0.001

    # ===== Expected Value (EV) Tests =====

    def test_ev_positive_when_value(self):
        """Test: model win prob > implied prob -> EV > 0"""
        # True win rate 60%, odds 2.0 (implied 50%)
        ev = self.engine.calculate_ev(0.60, 2.0)
        # 60% * 1.0 - 40% * 1.0 = 0.20
        assert abs(ev - 0.20) < 0.001

    def test_ev_zero_at_fair_odds(self):
        """Test: at fair odds, EV = 0"""
        # True win rate 50%, odds 2.0
        ev = self.engine.calculate_ev(0.50, 2.0)
        assert abs(ev) < 0.001

    def test_ev_negative_when_overpriced(self):
        """Test: low odds -> EV < 0"""
        ev = self.engine.calculate_ev(0.30, 2.0)  # 30% vs 50% implied
        assert ev < 0

    # ===== Kelly Criterion Tests =====

    def test_kelly_zero_at_fair(self):
        """Test: at fair odds, Kelly = 0"""
        k = self.engine.kelly_criterion(0.50, 2.0)
        assert abs(k) < 0.001

    def test_kelly_positive_when_value(self):
        """Test: when there's value, Kelly > 0"""
        k = self.engine.kelly_criterion(0.60, 2.0)
        # Full Kelly = (1*0.6 - 0.4)/1 = 0.2
        # 1/4 Kelly = 0.05
        assert abs(k - 0.05) < 0.001

    def test_kelly_zero_when_negative_ev(self):
        """Test: negative EV -> Kelly = 0 (no bet)"""
        k = self.engine.kelly_criterion(0.30, 2.0)
        assert k == 0.0

    def test_kelly_capped_by_fraction(self):
        """Test: Kelly does not exceed set fraction"""
        engine_full = ValueEngine(bankroll=1000, kelly_fraction=1.0)
        engine_quarter = ValueEngine(bankroll=1000, kelly_fraction=0.25)
        # 60% win rate, odds 2.0: b=1, Full Kelly = (1*0.6 - 0.4)/1 = 0.2
        # But @staticmethod default kelly_fraction=0.25 has been overridden by the instance
        # Actually: engine_full.kelly_criterion(0.60, 2.0) defaults to 0.25 -> 0.05
        # This is an API misuse, use evaluate_match instead
        match = {
            'home_team': 'A', 'away_team': 'B',
            'home_odds': 2.0, 'draw_odds': 3.0, 'away_odds': 4.0
        }
        model_probs = {
            'home_win_prob': 0.60,
            'draw_prob': 0.20,
            'away_win_prob': 0.20
        }
        results = engine_full.evaluate_match(match, model_probs)
        home_bet = [r for r in results if r['outcome'] == 'HOME'][0]
        # 1/4 Kelly should be 1/4 of full Kelly
        results_q = engine_quarter.evaluate_match(match, model_probs)
        home_bet_q = [r for r in results_q if r['outcome'] == 'HOME'][0]
        assert abs(home_bet_q['kelly_pct'] - home_bet['kelly_pct'] * 0.25) < 0.001

    # ===== Integration Tests =====

    def test_evaluate_match_returns_three_outcomes(self):
        """Test: single match evaluation returns 3 outcomes"""
        match = {
            'home_team': 'A', 'away_team': 'B',
            'home_odds': 2.0, 'draw_odds': 3.0, 'away_odds': 4.0
        }
        model_probs = {
            'home_win_prob': 0.50,
            'draw_prob': 0.30,
            'away_win_prob': 0.20
        }
        results = self.engine.evaluate_match(match, model_probs)
        assert len(results) == 3
        outcomes = {r['outcome'] for r in results}
        assert outcomes == {'HOME', 'DRAW', 'AWAY'}

    def test_filter_value_bets(self):
        """Test: value bet filtering"""
        import pandas as pd
        df = pd.DataFrame({
            'match': ['A vs B', 'A vs B', 'C vs D'],
            'outcome': ['HOME', 'AWAY', 'HOME'],
            'odds': [2.0, 1.5, 3.0],
            'model_prob': [0.6, 0.3, 0.4],
            'implied_prob': [0.5, 0.667, 0.333],
            'EV': [0.20, -0.05, 0.05],
            'EV_pct': [20, -5, 5],
            'kelly_pct': [5, 0, 2.5],
            'bet_amount': [50, 0, 25],
            'margin': [0.05, 0.05, 0.05]
        })
        filtered = self.engine.filter_value_bets(df)
        # Should have 2 (EV > 0)
        assert len(filtered) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])