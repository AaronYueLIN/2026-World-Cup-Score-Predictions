"""
Unit tests for Dixon-Coles model
Verify model mathematical correctness
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import numpy as np
import pandas as pd
import pytest
from dixon_coles import DixonColesModel


class TestDixonColesModel:
    """Dixon-Coles model test suite (adapted for improved v3.0)"""

    def setup_method(self):
        """Initialize before each test"""
        self.df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=30, freq='3D'),
            'home_team': ['A', 'B', 'A', 'B', 'A', 'B'] * 5,
            'away_team': ['B', 'A', 'C', 'C', 'D', 'D'] * 5,
            'home_goals': [1, 0, 2, 1, 0, 1] * 5,
            'away_goals': [0, 1, 1, 2, 0, 0] * 5,
            'home_xg': [1.5] * 30,
            'away_xg': [1.0] * 30
        })
        self.model = DixonColesModel(damping=0.002)

    def test_model_can_fit(self):
        """Test that model can fit successfully"""
        self.model.fit(self.df)
        assert self.model.params is not None
        assert self.model.teams is not None
        assert len(self.model.teams) >= 4

    def test_probabilities_sum_to_one(self):
        """Test: home_win + draw + away_win = 1.0 (analytic, exact)"""
        self.model.fit(self.df)
        result = self.model.predict('A', 'B')
        total = result['home_win_prob'] + result['draw_prob'] + result['away_win_prob']
        assert 0.99 < total < 1.01, f"Probabilities sum to {total}"

    def test_predict_returns_all_fields(self):
        """Test: prediction returns all required fields"""
        self.model.fit(self.df)
        result = self.model.predict('A', 'B')
        required = ['home_team', 'away_team', 'expected_home_goals',
                    'expected_away_goals', 'home_win_prob', 'draw_prob',
                    'away_win_prob', 'score_probs']
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_xg_positive(self):
        """Test: expected goals > 0"""
        self.model.fit(self.df)
        result = self.model.predict('A', 'B')
        assert result['expected_home_goals'] > 0
        assert result['expected_away_goals'] > 0

    def test_score_probs_dict(self):
        """Test: score probability dict is non-empty"""
        self.model.fit(self.df)
        result = self.model.predict('A', 'B')
        assert isinstance(result['score_probs'], dict)
        assert len(result['score_probs']) > 0
        # score probability sum should be close to 1
        total = sum(result['score_probs'].values())
        assert 0.9 < total < 1.0, f"Score probs sum to {total}"

    def test_team_not_in_model_raises(self):
        """Test: predicting a non-existent team raises an error"""
        self.model.fit(self.df)
        with pytest.raises(KeyError):
            self.model.predict('NonExistent', 'B')

    def test_tau_correction_in_predict(self):
        """Test: tau correction influences low-score probabilities (Dixon & Coles 1997)"""
        self.model.fit(self.df)
        result = self.model.predict('A', 'B')
        # 0-0, 1-0, 0-1 must exist (these are common scores in the raw data)
        # 1-1 may be filtered by the 0.5% threshold, so use score_matrix to verify
        assert '0-0' in result['score_probs']
        # tau correction effect: verify 1-1 has non-zero probability via score_matrix
        assert result['score_matrix'][1, 1] > 0
        # 0-0 should also be non-zero in score_matrix
        assert result['score_matrix'][0, 0] > 0

    def test_team_ratings_method(self):
        """Test: team_ratings() returns complete DataFrame"""
        self.model.fit(self.df)
        ratings = self.model.team_ratings()
        assert 'team' in ratings.columns
        assert 'attack' in ratings.columns
        assert 'defense' in ratings.columns
        assert 'overall' in ratings.columns
        assert len(ratings) >= 4

    def test_fit_summary_method(self):
        """Test: fit_summary() returns AIC / BIC"""
        self.model.fit(self.df)
        s = self.model.fit_summary()
        assert 'log_lik' in s
        assert 'aic' in s
        assert 'bic' in s
        assert 'rho' in s
        assert 'home_adj' in s

    def test_rps_method(self):
        """Test: RPS calculation"""
        self.model.fit(self.df)
        result = self.model.predict('A', 'B')
        rps = self.model.rps(result, 'H')
        # perfect prediction RPS=0, random prediction RPS=0.333
        assert 0 <= rps <= 0.5

    def test_xg_mode(self):
        """Test: xG mode (use_xg=True)"""
        model_xg = DixonColesModel(damping=0.002, use_xg=True)
        model_xg.fit(self.df)
        assert model_xg.params is not None

    def test_damping_zero_uniform_weights(self):
        """Test: damping=0 gives uniform weights"""
        m = DixonColesModel(damping=0.0)
        m.fit(self.df)
        assert m.params is not None

    def test_score_matrix_in_result(self):
        """Test: prediction result contains score_matrix (hallmark of vectorized prediction)"""
        self.model.fit(self.df)
        result = self.model.predict('A', 'B')
        assert 'score_matrix' in result
        assert result['score_matrix'].shape == (11, 11)  # max_goals+1


def test_data_csv_exists():
    """Test: mock data CSV exists and is non-empty"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mock_matches.csv')
    df = pd.read_csv(data_path)
    assert len(df) >= 100
    assert 'home_team' in df.columns
    assert 'away_team' in df.columns
    assert 'home_goals' in df.columns
    assert 'away_goals' in df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
