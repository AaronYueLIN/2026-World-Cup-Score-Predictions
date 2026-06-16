"""Load WC 2026 squad list from converted PDF text into SQL."""
import re, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from db.config import DATABASE_URL, ENGINE_KWARGS

SRC = r"C:\Users\Aaronlin\Downloads\SquadLists-English_converted_text.txt"

TOP5_COUNTRIES = {"ENG": True, "ESP": True, "GER": True, "ITA": True, "FRA": True}

# FIFA squad names -> DC model team names
TEAM_NAME_MAP = {
    "Côte D'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "USA": "United States",
    "Curaçao": "Curaçao",
}

# CLUB country codes: "Lille OSC (FRA)" -> "FRA"
CLUB_COUNTRY_RE = re.compile(r"\(([A-Z]{3})\)\s*$")


def extract_club_country(club: str) -> tuple[str, str | None, bool]:
    """Return (clean_club, country_code, is_top5)."""
    m = CLUB_COUNTRY_RE.search(club)
    if m:
        cc = m.group(1)
        return club[:m.start()].strip(), cc, cc in TOP5_COUNTRIES
    return club, None, False


def parse_squads(path: str) -> tuple[list[dict], list[dict]]:
    """Parse the converted squad text into player and coach records."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on page markers
    pages = content.split("=" * 100)
    raw_teams = []
    for page in pages:
        lines = [l.strip() for l in page.strip().split("\n")]
        team_name = None
        team_code = None
        header_found = False
        players = []
        coaches = []
        for line in lines:
            if not line or line.startswith("Page") or line.startswith("Tuesday") or line.startswith("FIFA") or line.startswith("DOB") or line.startswith("POS") or line.startswith("GK") or line.startswith("DF") or line.startswith("MF") or line.startswith("FW"):
                continue
            # Team header: "Algeria (ALG)"
            m = re.match(r"^([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s\-\.']+)\(([A-Z]{3})\)$", line)
            if m and not team_name:
                raw_name = m.group(1).strip()
                team_name = TEAM_NAME_MAP.get(raw_name, raw_name)
                team_code = m.group(2)
                continue
            # Column header
            if line.startswith("#	POS	PLAYER NAME"):
                header_found = True
                continue
            # Coach
            if line.startswith("Head coach"):
                parts = [p.strip() for p in line.split("\t")]
                if len(parts) >= 5:
                    coaches.append({
                        "role": "Head coach",
                        "coach_name": parts[1],
                        "first_name": parts[2],
                        "last_name": parts[3],
                        "nationality": parts[4],
                    })
                continue
            # Player line: "#	POS	NAME	FIRST	LAST	SHIRT	DOB	CLUB	HEIGHT	CAPS	GOALS"
            if header_found and team_name:
                parts = [p.strip() for p in line.split("\t")]
                if len(parts) >= 11 and parts[0].isdigit():
                    try:
                        jersey = int(parts[0])
                        position = parts[1]
                        player_name = parts[2]
                        first_name = parts[3]
                        last_name = parts[4]
                        name_on_shirt = parts[5]
                        dob = parts[6] if parts[6] != "N/A" else None
                        club_raw = parts[7]
                        club, club_country, top5 = extract_club_country(club_raw)
                        height = int(parts[8]) if parts[8].isdigit() else None
                        caps = int(parts[9]) if parts[9].isdigit() else None
                        goals = int(parts[10]) if parts[10].isdigit() else None

                        players.append({
                            "team_code": team_code,
                            "team_name": team_name,
                            "jersey_number": jersey,
                            "position": position,
                            "player_name": player_name,
                            "first_name": first_name,
                            "last_name": last_name,
                            "name_on_shirt": name_on_shirt,
                            "dob": dob,
                            "club": club,
                            "club_country": club_country,
                            "height_cm": height,
                            "caps": caps,
                            "goals": goals,
                            "top5_league": top5,
                        })
                    except (ValueError, IndexError):
                        pass

        if players and team_code:
            raw_teams.append((team_code, team_name, players, coaches))

    all_players = []
    all_coaches = []
    for tc, tn, pl, co in raw_teams:
        all_players.extend(pl)
        all_coaches.extend(coach | {"team_code": tc, "team_name": tn} for coach in co)

    return all_players, all_coaches


def load_squads(players: list[dict], coaches: list[dict], db_url: str):
    """Clear and reload squad tables."""
    engine = create_engine(db_url, **ENGINE_KWARGS)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM squad_players"))
        conn.execute(text("DELETE FROM squad_coaches"))

        for p in players:
            conn.execute(
                text("""INSERT INTO squad_players
                    (team_code, team_name, jersey_number, position, player_name,
                     first_name, last_name, name_on_shirt, dob, club,
                     club_country, height_cm, caps, goals, top5_league)
                    VALUES (:team_code, :team_name, :jersey_number, :position, :player_name,
                     :first_name, :last_name, :name_on_shirt, :dob, :club,
                     :club_country, :height_cm, :caps, :goals, :top5_league)"""),
                p,
            )

        for c in coaches:
            conn.execute(
                text("""INSERT INTO squad_coaches
                    (team_code, team_name, role, coach_name, first_name, last_name, nationality)
                    VALUES (:team_code, :team_name, :role, :coach_name, :first_name, :last_name, :nationality)"""),
                c,
            )

    print(f"Loaded: {len(players)} players, {len(coaches)} coaches")


if __name__ == "__main__":
    print("Parsing squad list...")
    players, coaches = parse_squads(SRC)
    print(f"  Found {len(players)} players, {len(coaches)} coaches across {len(set(p['team_code'] for p in players))} teams")
    load_squads(players, coaches, DATABASE_URL)
    print("Done.")
