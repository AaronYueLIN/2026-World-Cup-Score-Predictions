# 2026 World Cup Score Predictions

Production-style football scoreline forecasting and value-analysis platform for international football and 2026 World Cup scenarios.

This repository combines a **Bayesian Dixon-Coles football model**, a **dynamic strength update layer**, a **scoreline probability engine**, a **FastAPI service**, and optional **market de-vig / value-betting utilities**. The main purpose is to turn historical international match data and Elo-style priors into interpretable match probabilities: 1X2, expected goals, Over/Under 2.5, BTTS, and exact-score distributions.

> Research and education project. It is not betting, financial, or legal advice.

---

## Why this project exists

Most football prediction demos stop at a single win/draw/loss probability. This project is built around a fuller modelling pipeline:

1. estimate team attack and defense strength;
2. adjust those strengths through a dynamic form/momentum layer;
3. produce a calibrated score matrix rather than only a class label;
4. derive football markets from the same probability matrix;
5. keep prediction logic separated from bookmaker odds and staking logic.

That separation matters: **odds are never used as an input to the prediction model**. Odds only enter the optional value-analysis layer after model probabilities have already been produced.

---

## What the system returns

For a match such as `Germany vs Japan`, the model can return:

- home win / draw / away win probabilities;
- expected home and away goals;
- Over 2.5 probability;
- BTTS probability;
- exact-score probabilities through the scoreline matrix;
- GAS momentum diagnostics for teams;
- optional EV and Kelly-style staking outputs when odds are supplied separately.

---

## Core architecture

```text
Historical international matches + Elo ratings
        |
        v
Bayesian Dixon-Coles model
(team attack, defense, venue/home effects)
        |
        v
GAS dynamic-strength layer
(short-term team momentum around long-term anchors)
        |
        v
Flexible scoreline engine
Negative Binomial margins + dependence/calibration
        |
        +--> 1X2 probabilities
        +--> expected goals
        +--> Over/Under, BTTS, exact scores
        |
        v
FastAPI / Streamlit consumers


Optional and separate:

Bookmaker odds
        |
        v
Shin / proportional de-vig
        |
        v
Model probability vs market probability
        |
        v
EV filter + risk-constrained Kelly sizing
```

The project has a deliberate **prediction firewall**: market odds can be evaluated against model probabilities, but they should not leak into the model training or prediction path.

---

## Repository layout

```text
.
├── api/                    # FastAPI app, routes, schemas, auth, rate limit, metrics
├── db/                     # SQLAlchemy models, settings, ETL, Alembic migrations
├── docs/                   # Design notes and implementation plans
├── frontend/               # Streamlit workbench for chat, momentum, and batch prediction
├── models/
│   ├── bayesian_dixon_coles.py
│   ├── bayesian_dc_v9.pkl  # current packaged model artifact
│   ├── registry.py         # single source of truth for model loading
│   └── quantbet/           # de-vig, staking, portfolio, scoreline, World Cup simulation
├── scripts/                # ETL, Elo fetching, odds API utilities, scheduled jobs
├── tests/                  # unit, integration, smoke, and benchmark-oriented tests
├── docker-compose.yml      # production-style Postgres + API stack
├── docker-compose.override.yml
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Model pipeline in plain English

### 1. Bayesian Dixon-Coles base model

The base model estimates team attack and defense parameters and produces match-level scoring intensities. It follows the Dixon-Coles idea of modelling football scores with low-score dependence adjustments, then extends the implementation with Bayesian/PyMC fitting utilities and serialized model artifacts.

### 2. Elo-informed anchoring

Elo-style ratings provide an external prior / anchor for team strength. This is useful for international football because many national teams play uneven schedules and have sparse head-to-head data.

### 3. GAS dynamic-strength update

The v9 path includes a GAS-style dynamic update layer. In practical terms, it tracks how a team is currently deviating from its long-term attack/defense anchor. The API exposes this through the momentum endpoint.

### 4. Scoreline matrix

Instead of returning only "home/draw/away", the prediction path builds a score matrix. Derived markets such as Over 2.5 and BTTS are calculated from that same matrix, keeping outputs internally consistent.

### 5. Optional value layer

The `models/quantbet` package contains de-vigging, expected value, staking, pooling, portfolio, and evaluation utilities. This layer is optional and should be treated as analysis code, not as a guarantee of profit.

---

## Quick start: local API

Python 3.10+ is recommended.

```bash
git clone <your-repo-url>
cd wc2026-score-predictions

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the API in development mode:

```bash
export APP_ENV=development
export API_TOKEN=dev-token
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

Prediction request:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Germany",
    "away_team": "Japan",
    "venue": "neutral"
  }'
```

Natural-language chat endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"message": "Spain vs Cape Verde"}'
```

Momentum endpoint:

```bash
curl -H "Authorization: Bearer dev-token" \
  http://localhost:8000/api/v1/model/momentum
```

OpenAPI docs are available while the server is running:

```text
http://localhost:8000/docs
```

---

## Quick start: direct model usage

```python
from models.registry import load_model, describe

model = load_model()
print(describe())

prediction = model.predict("Germany", "Japan", venue="neutral")
print(prediction)

# Common keys:
# home_win_prob, draw_prob, away_win_prob
# expected_home_goals, expected_away_goals
# over_25, btts
```

The current selected model version is controlled in `models/registry.py`.

```python
MODEL_VERSION = "v9"
```

The packaged archive includes the v9 Dixon-Coles artifact, `models/bayesian_dc_v9.pkl`. Some older registry entries or optional ensemble references may remain for compatibility; they are not required for the core `/predict` path.

---

## Docker usage

Development-style API with mounted source and SQLite:

```bash
docker compose up --build
```

Production-style compose without the development override:

```bash
export DB_PASSWORD=<strong-password>
export API_TOKEN=<strong-api-token>
docker compose -f docker-compose.yml up --build
```

The base `docker-compose.yml` uses PostgreSQL. The development override switches the API toward SQLite and hot reload.

---

## Streamlit frontend

The repository includes a Streamlit workbench in `frontend/app.py`.

```bash
pip install streamlit
export QUANTBET_API=http://localhost:8000
export QUANTBET_TOKEN=dev-token
streamlit run frontend/app.py
```

The frontend is designed as a data-scientist workbench with:

- natural-language prediction;
- GAS momentum table;
- batch prediction workflow.

If you change API request schemas, keep the frontend payload fields aligned with `api/schemas.py`.

---

## Configuration

The main environment variables are:

| Variable | Purpose | Typical development value |
| --- | --- | --- |
| `APP_ENV` | `development`, `production`, or `test` | `development` |
| `API_TOKEN` | Bearer token for protected endpoints | `dev-token` |
| `DATABASE_URL` | SQLAlchemy database URL | SQLite URL or Postgres URL |
| `DATABASE_URL_READ_ONLY` | Optional read replica URL | empty |
| `DB_PASSWORD` | Postgres password for Docker Compose | local secret |
| `LLM_API_KEY` | Optional DeepSeek/OpenAI-compatible chat key | empty |
| `LLM_BASE_URL` | Optional LLM base URL | `https://api.deepseek.com` |
| `LLM_MODEL` | Optional LLM model name | `deepseek-chat` |

Notes:

- `/api/v1/health` and `/metrics` skip token verification.
- Other API routes require `Authorization: Bearer <API_TOKEN>`.
- In development, the server defaults to `dev-token` if `API_TOKEN` is not exported.
- Do not commit real API keys, production database URLs, or downloaded private data.

---

## Data and ETL

The database layer supports SQLite for development and PostgreSQL for production-like deployment.

Important files:

- `db/models.py` defines teams, tournaments, matches, odds, and team-name mappings.
- `db/etl.py` initializes the schema and upserts match data.
- `db/reader.py` provides read helpers for training and API queries.
- `scripts/elo_fetcher.py` and `scripts/elo_etl.py` handle Elo-related inputs.
- `db/migrations/` contains Alembic migration files.

Large historical datasets and production databases are not bundled in this archive. The packaged model artifact can be used for prediction, while database query endpoints depend on whatever local database you initialize and populate.

---

## Testing and quality checks

Run the test suite:

```bash
pytest
```

Run the same style of checks used by CI:

```bash
ruff check models/quantbet/ db/ api/
mypy models/quantbet/pooling.py models/quantbet/devig.py models/registry.py --ignore-missing-imports
```

Some tests and training scripts are heavier than normal unit tests because they touch PyMC, model fitting, or benchmark workflows. For small API or utility changes, start with targeted tests.

---

## Key design rules

1. **Use `models.registry.load_model()` as the model entry point.**  
   Avoid ad-hoc pickle loading in new code.

2. **Keep odds out of the prediction path.**  
   Odds belong only in de-vig, EV, staking, and portfolio analysis.

3. **Keep derived markets consistent.**  
   1X2, Over/Under, BTTS, and exact scores should all come from the same score matrix whenever possible.

4. **Preserve API contracts.**  
   Request and response shapes live in `api/schemas.py`.

5. **Be explicit about model artifacts.**  
   If you change `MODEL_VERSION`, update the registry and documentation together.

---

## Limitations

- A prediction model is not a guarantee of match outcomes or market profit.
- Model quality depends on the training data, rating inputs, calibration, and update schedule.
- International football data can be sparse and uneven across confederations.
- Optional odds and LLM integrations require external credentials.
- The included artifact is intended for demonstration and research workflows, not automated wagering.

---

## License

See `LICENSE`.

