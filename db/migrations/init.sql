# QuantBet-EV database initialization script
# Executed on startup: CREATE TABLE IF NOT EXISTS

# --- Teams ---
CREATE TABLE IF NOT EXISTS teams (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    fifa_code VARCHAR(5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index: search by team name (most common query)
CREATE INDEX IF NOT EXISTS idx_teams_name ON teams(name);

# --- Tournaments ---
CREATE TABLE IF NOT EXISTS tournaments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL UNIQUE,
    tier VARCHAR(20) NOT NULL CHECK (tier IN ('friendly', 'qualifier', 'continental', 'world_cup', 'other')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tournaments_tier ON tournaments(tier);

# --- Matches ---
CREATE TABLE IF NOT EXISTS matches (
    id SERIAL PRIMARY KEY,
    home_team_id INTEGER NOT NULL REFERENCES teams(id),
    away_team_id INTEGER NOT NULL REFERENCES teams(id),
    tournament_id INTEGER REFERENCES tournaments(id),
    home_score SMALLINT NOT NULL,
    away_score SMALLINT NOT NULL,
    date DATE NOT NULL,
    city VARCHAR(100),
    country VARCHAR(100),
    neutral BOOLEAN DEFAULT FALSE,
    venue VARCHAR(50) DEFAULT 'neutral' CHECK (venue IN ('home', 'away', 'neutral')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Composite unique constraint: prevents duplicate imports
    UNIQUE (home_team_id, away_team_id, date)
);

-- Index: by date (time series queries)
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date DESC);
-- Index: by home team + date
CREATE INDEX IF NOT EXISTS idx_matches_home_team_date ON matches(home_team_id, date DESC);
-- Index: by away team + date
CREATE INDEX IF NOT EXISTS idx_matches_away_team_date ON matches(away_team_id, date DESC);
-- Index: by tournament
CREATE INDEX IF NOT EXISTS idx_matches_tournament ON matches(tournament_id);
-- Foreign key indexes (automatic already, but explicit creation ensures)
CREATE INDEX IF NOT EXISTS idx_matches_home_team ON matches(home_team_id);
CREATE INDEX IF NOT EXISTS idx_matches_away_team ON matches(away_team_id);

# --- Odds ---
CREATE TABLE IF NOT EXISTS odds (
    id SERIAL PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    home_odds DECIMAL(7, 2),
    draw_odds DECIMAL(7, 2),
    away_odds DECIMAL(7, 2),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_odds_match ON odds(match_id);
CREATE INDEX IF NOT EXISTS idx_odds_provider ON odds(provider);

# --- Team name change history ---
CREATE TABLE IF NOT EXISTS team_names (
    id SERIAL PRIMARY KEY,
    current_name VARCHAR(100) NOT NULL,
    former_name VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    UNIQUE (former_name, start_date)
);

CREATE INDEX IF NOT EXISTS idx_team_names_former ON team_names(former_name);
