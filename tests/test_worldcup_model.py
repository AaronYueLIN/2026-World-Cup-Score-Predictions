"""
Unit tests for World Cup model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import pytest
from worldcup_model import (
    get_country_recent_xg,
    predict_match,
    build_world_cup_dataset,
    ALL_WORLD_CUP_DATA
)


class TestWorldCupModel:
    """World Cup model tests"""

    def test_country_data_has_teams(self):
        """Test: country data is non-empty"""
        data = get_country_recent_xg()
        assert len(data) >= 20  # at least 20 national teams
        assert 'Brazil' in data
        assert 'Argentina' in data
        assert 'Mexico' in data
        assert 'South Africa' in data

    def test_predict_match_returns_dict(self):
        """Test: prediction returns a dict"""
        data = get_country_recent_xg()
        pred = predict_match('Mexico', 'South Africa', data)
        assert pred is not None
        assert 'expected_home_goals' in pred
        assert 'expected_away_goals' in pred
        assert 0.3 < pred['expected_home_goals'] < 3.5
        assert 0.3 < pred['expected_away_goals'] < 3.0

    def test_strong_team_higher_xg(self):
        """Test: strong team (Brazil) xG > weak team (South Africa)"""
        data = get_country_recent_xg()
        brazil_pred = predict_match('Brazil', 'Japan', data)
        sa_pred = predict_match('South Africa', 'Ecuador', data)
        # Brazil (home) should score more than South Africa (home)
        assert brazil_pred['expected_home_goals'] > sa_pred['expected_home_goals']

    def test_worldcup_dataset_has_128_matches(self):
        """Test: dataset contains 128 matches (64+64)"""
        assert len(ALL_WORLD_CUP_DATA) == 128
        df = build_world_cup_dataset()
        assert len(df) == 128

    def test_xg_bounded(self):
        """Test: xG in reasonable range (0.3-3.5)"""
        data = get_country_recent_xg()
        for home in ['Brazil', 'Germany', 'Mexico', 'South Africa', 'Japan']:
            for away in ['Argentina', 'France', 'Iran', 'Ecuador', 'USA']:
                if home in data and away in data:
                    pred = predict_match(home, away, data)
                    assert 0.3 <= pred['expected_home_goals'] <= 3.5
                    assert 0.3 <= pred['expected_away_goals'] <= 3.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])