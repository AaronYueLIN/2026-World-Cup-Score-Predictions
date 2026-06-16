#!/usr/bin/env python3
"""Fetch current ELO ratings from eloratings.net, write to database + cache file.

World.tsv format (tab-separated):
  0:rank  1:rank_chg  2:code  3:rating  4:best_rank  5:best_rating  ...

en.teams.tsv format:
  code<TAB>name[<TAB>aliases...]

Only needs to be updated once daily.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import csv, io, json, logging, pickle
from datetime import date
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

ELO_BASE = "https://eloratings.net"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"

# Manual mapping: eloratings code -> full name (overrides inconsistencies in en.teams.tsv)
_MANUAL_MAP: dict[str, str] = {
    "DE": "Germany",
    "EN": "England",
    "ES": "Spain",
    "FR": "France",
    "IT": "Italy",
    "NL": "Netherlands",
    "PT": "Portugal",
    "BR": "Brazil",
    "AR": "Argentina",
    "UY": "Uruguay",
    "CO": "Colombia",
    "CL": "Chile",
    "EC": "Ecuador",
    "PE": "Peru",
    "PY": "Paraguay",
    "BO": "Bolivia",
    "VE": "Venezuela",
    "MX": "Mexico",
    "US": "United States",
    "CR": "Costa Rica",
    "HN": "Honduras",
    "SV": "El Salvador",
    "PA": "Panama",
    "JM": "Jamaica",
    "TT": "Trinidad and Tobago",
    "CA": "Canada",
    "CW": "Curaçao",
    "HR": "Croatia",
    "RS": "Serbia",
    "BA": "Bosnia and Herzegovina",
    "SI": "Slovenia",
    "MK": "North Macedonia",
    "ME": "Montenegro",
    "DK": "Denmark",
    "SE": "Sweden",
    "NO": "Norway",
    "FI": "Finland",
    "IS": "Iceland",
    "BE": "Belgium",
    "CH": "Switzerland",
    "AT": "Austria",
    "CZ": "Czech Republic",
    "SK": "Slovakia",
    "HU": "Hungary",
    "PL": "Poland",
    "RO": "Romania",
    "BG": "Bulgaria",
    "GR": "Greece",
    "TR": "Turkey",
    "UA": "Ukraine",
    "RU": "Russia",
    "IE": "Republic of Ireland",
    "NI": "Northern Ireland",
    "WA": "Wales",
    "SC": "Scotland",
    "JP": "Japan",
    "KR": "South Korea",
    "IR": "Iran",
    "SA": "Saudi Arabia",
    "AU": "Australia",
    "QA": "Qatar",
    "AE": "United Arab Emirates",
    "IQ": "Iraq",
    "JO": "Jordan",
    "KW": "Kuwait",
    "OM": "Oman",
    "BH": "Bahrain",
    "SY": "Syria",
    "LB": "Lebanon",
    "UZ": "Uzbekistan",
    "CN": "China",
    "TH": "Thailand",
    "VN": "Vietnam",
    "ID": "Indonesia",
    "MY": "Malaysia",
    "SG": "Singapore",
    "IN": "India",
    "BD": "Bangladesh",
    "KP": "North Korea",
    "EG": "Egypt",
    "MA": "Morocco",
    "TN": "Tunisia",
    "DZ": "Algeria",
    "NG": "Nigeria",
    "CM": "Cameroon",
    "GH": "Ghana",
    "CI": "Ivory Coast",
    "SN": "Senegal",
    "ML": "Mali",
    "BF": "Burkina Faso",
    "ZA": "South Africa",
    "CD": "DR Congo",
    "ZM": "Zambia",
    "ZW": "Zimbabwe",
    "AO": "Angola",
    "ET": "Ethiopia",
    "SD": "Sudan",
    "KE": "Kenya",
    "UG": "Uganda",
    "TZ": "Tanzania",
    "GA": "Gabon",
    "GQ": "Equatorial Guinea",
    "CG": "Congo",
    "BJ": "Benin",
    "TG": "Togo",
    "LR": "Liberia",
    "SL": "Sierra Leone",
    "GM": "Gambia",
    "GW": "Guinea-Bissau",
    "GN": "Guinea",
    "LY": "Libya",
    "MR": "Mauritania",
    "NE": "Niger",
    "TD": "Chad",
    "CF": "Central African Republic",
    "RW": "Rwanda",
    "BI": "Burundi",
    "SO": "Somalia",
    "DJ": "Djibouti",
    "SZ": "Eswatini",
    "LS": "Lesotho",
    "BW": "Botswana",
    "NA": "Namibia",
    "MW": "Malawi",
    "MZ": "Mozambique",
    "MG": "Madagascar",
    "MU": "Mauritius",
    "SC": "Seychelles",
    "KM": "Comoros",
    "CV": "Cape Verde",
    "ST": "São Tomé and Príncipe",
    "GQ": "Equatorial Guinea",
    "NZ": "New Zealand",
    "FJ": "Fiji",
    "PG": "Papua New Guinea",
    "NC": "New Caledonia",
    "PF": "Tahiti",
    "SB": "Solomon Islands",
    "VU": "Vanuatu",
    "WS": "Samoa",
    "TO": "Tonga",
    "CK": "Cook Islands",
    "AM": "Armenia",
    "AZ": "Azerbaijan",
    "BY": "Belarus",
    "EE": "Estonia",
    "GE": "Georgia",
    "IL": "Israel",
    "KZ": "Kazakhstan",
    "KG": "Kyrgyzstan",
    "LV": "Latvia",
    "LT": "Lithuania",
    "MD": "Moldova",
    "TJ": "Tajikistan",
    "TM": "Turkmenistan",
    "XK": "Kosovo",
    "CY": "Cyprus",
    "LU": "Luxembourg",
    "MT": "Malta",
    "AD": "Andorra",
    "LI": "Liechtenstein",
    "SM": "San Marino",
    "GI": "Gibraltar",
    "FO": "Faroe Islands",
    "SU": "Soviet Union",
    "CS": "Czechoslovakia",
    "YU": "Yugoslavia",
    "DD": "German DR",
    "HT": "Haiti",
    "CU": "Cuba",
    "DO": "Dominican Republic",
    "PR": "Puerto Rico",
    "GD": "Grenada",
    "LC": "Saint Lucia",
    "VC": "Saint Vincent and the Grenadines",
    "AG": "Antigua and Barbuda",
    "BB": "Barbados",
    "GY": "Guyana",
    "SR": "Suriname",
    "BM": "Bermuda",
    "KY": "Cayman Islands",
    "AW": "Aruba",
    "SX": "Sint Maarten",
    "BQ": "Bonaire",
    "FK": "Falkland Islands",
    "GL": "Greenland",
    "MM": "Myanmar",
    "LA": "Laos",
    "KH": "Cambodia",
    "PH": "Philippines",
    "TL": "Timor-Leste",
    "BN": "Brunei",
    "LK": "Sri Lanka",
    "NP": "Nepal",
    "PK": "Pakistan",
    "AF": "Afghanistan",
    "MV": "Maldives",
    "MN": "Mongolia",
    "TW": "Taiwan",
    "HK": "Hong Kong",
    "MO": "Macau",
    "GU": "Guam",
    "BT": "Bhutan",
    "YE": "Yemen",
    "PS": "Palestine",
}


def _download_tsv(path: str) -> str:
    resp = requests.get(f"{ELO_BASE}/{path}", timeout=15)
    resp.raise_for_status()
    return resp.text


def fetch_elo_ratings() -> dict[str, float]:
    """Return {team_name: elo_rating}, keys use full team names from database/model."""

    # 1. Team name mapping: code -> name
    teams_text = _download_tsv("en.teams.tsv")
    code_to_name: dict[str, str] = {}
    for line in teams_text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            code_to_name[parts[0].strip()] = parts[1].strip()
    # Manually override known inconsistencies
    code_to_name.update(_MANUAL_MAP)

    # 2. ELO data
    world_text = _download_tsv("World.tsv")
    ratings: dict[str, float] = {}
    for line in world_text.strip().split("\n"):
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        code = cols[2].strip()
        elo = float(cols[3])
        name = code_to_name.get(code, code)  # fallback: use code directly
        ratings[name] = elo

    # 3. Check how many teams in DB can be matched
    try:
        from db.reader import get_team_list
        db_teams = set(get_team_list())
        matched = sum(1 for t in ratings if t in db_teams)
        logger.info("ELO: %d teams total, %d matched to DB", len(ratings), matched)
    except Exception:
        pass

    return ratings


def save_elo_cache(ratings: dict[str, float]) -> Path:
    """Save local cache + timestamp, returns path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today_iso = date.today().isoformat()

    payload = {
        "date": today_iso,
        "source": "eloratings.net/World.tsv",
        "n_teams": len(ratings),
        "ratings": ratings,
    }
    cache_path = CACHE_DIR / "elo_ratings.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    logger.info("ELO cache saved: %s (%d teams)", cache_path, len(ratings))
    return cache_path


def load_elo_cache(ttl_days: int = 1) -> dict[str, float] | None:
    """Load local cache, return if within ttl_days, otherwise return None."""
    cache_path = CACHE_DIR / "elo_ratings.json"
    if not cache_path.exists():
        return None
    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)
    cached_date = date.fromisoformat(data["date"])
    if (date.today() - cached_date).days < ttl_days:
        return data["ratings"]
    return None


def get_elo_ratings(force_refresh: bool = False) -> dict[str, float]:
    """Get ELO ratings (prefer cache)."""
    if not force_refresh:
        cached = load_elo_cache()
        if cached is not None:
            return cached

    logger.info("Fetching fresh ELO ratings from eloratings.net...")
    ratings = fetch_elo_ratings()
    save_elo_cache(ratings)
    return ratings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    r = get_elo_ratings(force_refresh=True)
    top = sorted(r.items(), key=lambda x: -x[1])[:10]
    for name, elo in top:
        print(f"  {name:<25s} {elo:.0f}")
    print(f"  ... ({len(r)} teams)")
