"""
QuantBet-EV: Real-time Odds Data Collector (The-Odds-API)
Fetch decimal odds for 1X2 market from The-Odds-API
"""

import requests
import pandas as pd
import os
import time
from datetime import datetime


class OddsAPIClient:
    """
    The-Odds-API client
    Free tier docs: https://the-odds-api.com/liveapi/guides/v4/
    Register for API Key: https://the-odds-api.com/
    """

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "soccer_epl"  # Premier League (can change to soccer_uefa_champs_league, etc.)

    def __init__(self, api_key=None):
        """
        Args:
            api_key: The-Odds-API API Key
                     Can be set via environment variable THE_ODDS_API_KEY
        """
        self.api_key = api_key or os.environ.get('THE_ODDS_API_KEY')
        if not self.api_key:
            print("[!] Warning: No API key provided. Set THE_ODDS_API_KEY env var or pass api_key.")
            print("    Get a free key at: https://the-odds-api.com/")

    def fetch_odds(self, sport=None, regions='uk', markets='h2h', odds_format='decimal', max_retries=3):
        """
        Fetch live odds
        Args:
            sport: sport type (default soccer_epl)
                   options: soccer_uefa_champs_league, soccer_spain_la_liga, etc.
            regions: region (uk, us, eu, au)
            markets: market type (h2h = 1X2)
            odds_format: decimal | american
            max_retries: number of retries on network failure
        Returns:
            DataFrame
        """
        if not self.api_key:
            raise ValueError("API key required")

        sport = sport or self.SPORT
        url = f"{self.BASE_URL}/sports/{sport}/odds/"
        params = {
            'apiKey': self.api_key,
            'regions': regions,
            'markets': markets,
            'oddsFormat': odds_format,
            'dateFormat': 'iso'
        }

        # Use Session + retry logic
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        print(f"  Fetching odds: {sport}, region={regions}, market={markets}")
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
        except requests.exceptions.SSLError:
            # Try disabling SSL verification (security risk, last resort only)
            print("  [WARN] SSL error, retrying with verify=False")
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = session.get(url, params=params, timeout=30, verify=False)
            response.raise_for_status()

        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text}")

        data = response.json()
        print(f"  Got {len(data)} matches from API")
        print(f"  Remaining requests: {response.headers.get('x-requests-remaining', '?')}")
        print(f"  Used requests: {response.headers.get('x-requests-used', '?')}")

        return self._parse_response(data)

    def _parse_response(self, data):
        """
        Parse API response into DataFrame
        Columns: home_team, away_team, commence_time, home_odds, draw_odds, away_odds
        """
        rows = []
        for match in data:
            try:
                # Find odds for h2h market
                for bookmaker in match.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            outcomes = {o['name']: o['price'] for o in market['outcomes']}
                            home_team = match['home_team']
                            away_team = match['away_team']
                            home_odds = outcomes.get(home_team)
                            draw_odds = outcomes.get('Draw')
                            away_odds = outcomes.get(away_team)

                            if home_odds and draw_odds and away_odds:
                                rows.append({
                                    'date': match.get('commence_time', '')[:10],
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'home_odds': home_odds,
                                    'draw_odds': draw_odds,
                                    'away_odds': away_odds,
                                    'bookmaker': bookmaker['title']
                                })
                            break
                    if rows and rows[-1].get('home_team') == home_team:
                        break
            except Exception as e:
                print(f"  Skip match: {e}")
                continue

        return pd.DataFrame(rows)

    def save_odds(self, df, output_path=None):
        """Save odds to CSV"""
        if output_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_path = os.path.join(base, 'data', 'live_odds.csv')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df)} odds rows to {output_path}")
        return output_path


def fetch_odds_safe(api_key, sport='soccer_epl', fallback_to_mock=True):
    """
    Safely fetch odds - falls back to mock data on failure
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mock_path = os.path.join(base, 'data', 'mock_odds.csv')

    try:
        client = OddsAPIClient(api_key=api_key)
        df = client.fetch_odds(sport=sport)
        if len(df) == 0:
            raise Exception("Empty response")
        return df
    except Exception as e:
        print(f"[!] API fetch failed: {e}")
        if fallback_to_mock and os.path.exists(mock_path):
            print(f"    Falling back to mock data: {mock_path}")
            return pd.read_csv(mock_path)
        return pd.DataFrame()


if __name__ == '__main__':
    # Test (must have THE_ODDS_API_KEY environment variable set first)
    api_key = os.environ.get('THE_ODDS_API_KEY')

    if api_key:
        client = OddsAPIClient(api_key=api_key)
        try:
            df = client.fetch_odds()
            print(df.head())
            client.save_odds(df)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No API key set. Using mock data fallback for testing.")
        df = fetch_odds_safe(api_key=None, fallback_to_mock=True)
        print(f"Loaded {len(df)} mock odds rows")