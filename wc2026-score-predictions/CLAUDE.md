# CLAUDE.md

This file gives Claude Code and other AI coding assistants the project context needed to work safely in this repository.

## Project identity

`wc2026-score-predictions` is a football scoreline prediction and value-analysis platform for international football and 2026 World Cup scenarios.

The core product is **not** a generic betting bot. It is a modelling system that:

1. loads a packaged Bayesian Dixon-Coles model artifact;
2. predicts football scoreline distributions;
3. derives internally consistent match markets from the score matrix;
4. serves those predictions through FastAPI;
5. optionally compares model probabilities with de-vigged market odds in a separate analysis layer.

The most important invariant is:

> **Prediction and odds must stay separated.**  
> The model may produce probabilities. The value layer may compare those probabilities with market odds. Market odds should not leak into training, calibration, or prediction logic unless the user explicitly asks for a research experiment and the documentation marks it clearly.

---

## Current high-level architecture

```text
api/
  FastAPI app, auth/rate limit middleware, Pydantic schemas, prediction and data routes

db/
  SQLAlchemy settings, models, ETL, read helpers, migrations

models/
  Bayesian Dixon-Coles implementation, current model artifact, registry entry point

models/quantbet/
  de-vigging, staking, portfolio, scoreline, calibration, World Cup simulation utilities

frontend/
  Streamlit workbench that calls the API

scripts/
  Elo fetching, ETL pipeline, odds API helper, scheduled jobs

tests/
  unit tests, integration tests, smoke tests, benchmark/training-oriented tests
```

---

## Core runtime path

For normal predictions, follow this path:

```python
from models.registry import load_model

dc = load_model()
result = dc.predict("Germany", "Japan", venue="neutral")
```

The API uses the same idea:

```text
POST /api/v1/predict
        |
        v
api.routes.predict()
        |
        v
models.registry.load_model()
        |
        v
dc.predict(...)
```

`models.registry.py` is the single source of truth for model selection. The current default is:

```python
MODEL_VERSION = "v9"
```

The packaged model artifact is:

```text
models/bayesian_dc_v9.pkl
```

The registry may still contain older model versions or optional ensemble references for compatibility. Do not assume every historical artifact is present in the repository.

---

## Prediction response contract

`api/schemas.py` defines the public API contract. The main request shape is:

```json
{
  "home_team": "Germany",
  "away_team": "Japan",
  "venue": "neutral",
  "date": "2026-06-16"
}
```

`venue` must be one of:

```text
home | neutral | away
```

The response includes:

```text
home_win_prob
draw_prob
away_win_prob
expected_home_goals
expected_away_goals
over_25
btts
model_version
gas_b
pool_method
```

When changing model outputs, update `PredictionResponse`, route code, tests, frontend payloads, and README examples together.

---

## API and security notes

The app entry point is:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Protected routes require:

```text
Authorization: Bearer <API_TOKEN>
```

Exceptions:

```text
/api/v1/health
/health
/metrics
```

Those skip token verification.

Important implementation detail: `api/server.py` currently reads `API_TOKEN` with `os.getenv`. In development it defaults to `dev-token` when `APP_ENV=development`. In production, export `API_TOKEN` as a real environment variable rather than relying only on a local `.env` file.

Do not remove the security middleware, rate limiter, structured logging, or unified exception handling unless the user explicitly asks.

---

## Database notes

The database layer is designed to work with both SQLite and PostgreSQL.

Use these files as the main references:

```text
db/settings.py       typed settings and default database URL logic
db/config.py         compatibility layer for existing imports
db/models.py         SQLAlchemy ORM models
db/etl.py            schema initialization and upsert helpers
db/reader.py         read/query helpers
db/migrations/       Alembic migration files
```

Development commonly uses SQLite. Production-style Docker Compose uses PostgreSQL.

Large historical match datasets are not necessarily bundled. Avoid writing documentation or tests that assume a fully populated database unless you create or load the fixture in that test.

---

## Model notes

Key files:

```text
models/bayesian_dixon_coles.py
models/dixon_coles.py
models/registry.py
models/rebuild_v9_gas.py
models/fit_scoreline_shape.py
models/ml_predictor.py
models/quantbet/scoreline/
models/quantbet/worldcup/
```

Guidelines:

- Use `load_model()` from `models.registry`.
- Do not load pickle files directly in API or frontend code.
- If changing `MODEL_VERSION`, update the registry, tests, and README.
- If adding a new artifact, make sure the filename is present in `REGISTRY`.
- Do not commit huge generated artifacts unless the user explicitly wants them in Git history.
- If model artifacts become large, recommend Git LFS or release assets.

The current v9 pipeline should be described as:

```text
Bayesian Dixon-Coles base strength
+ Elo-informed anchors
+ GAS dynamic-strength update
+ flexible scoreline matrix
+ calibration
```

Do not claim live accuracy, guaranteed profit, or production betting performance unless there is a cited evaluation file in the repository.

---

## Quant/value layer notes

The value-analysis utilities live in `models/quantbet/`.

Important concepts:

```text
devig.py            bookmaker margin removal, including Shin method
markets.py          derive football markets from a score matrix
value_engine_v2.py  compare model probabilities to market probabilities
staking.py          Kelly and fractional Kelly helpers
portfolio.py        portfolio sizing/risk constraints
evaluation.py       RPS, log loss, Brier, CLV and reliability utilities
```

Rules for this area:

1. De-vig odds before comparing against model probabilities.
2. Keep EV calculations transparent and testable.
3. Apply positive-edge filters explicitly.
4. Use fractional or risk-constrained Kelly rather than full Kelly defaults.
5. Keep disclaimers in user-facing documentation.

---

## Frontend notes

The Streamlit app is:

```bash
streamlit run frontend/app.py
```

It reads:

```text
QUANTBET_API
QUANTBET_TOKEN
```

The frontend is an API client. Do not duplicate model-loading logic inside Streamlit. When the API schema changes, verify that frontend payload names still match `api/schemas.py`.

---

## LLM chat notes

`POST /api/v1/chat` supports two modes:

1. DeepSeek/OpenAI-compatible function calling when `LLM_API_KEY` is configured.
2. Regex/rule fallback when no LLM key is available.

The LLM client is optional. The prediction API must keep working without it.

The current system prompt asks the assistant to reply briefly in Chinese. If the user wants the public GitHub project to be English-first, change user-facing docs and README examples first; only change runtime chat language if requested separately.

---

## Common development commands

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run API:

```bash
export APP_ENV=development
export API_TOKEN=dev-token
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Run tests:

```bash
pytest
```

Run targeted tests:

```bash
pytest tests/test_api.py -q
pytest tests/test_registry.py -q
pytest tests/test_devig.py -q
```

Run lint/type checks similar to CI:

```bash
ruff check models/quantbet/ db/ api/
mypy models/quantbet/pooling.py models/quantbet/devig.py models/registry.py --ignore-missing-imports
```

Run Docker development stack:

```bash
docker compose up --build
```

Run production-style compose without the development override:

```bash
docker compose -f docker-compose.yml up --build
```

Run frontend:

```bash
pip install streamlit
export QUANTBET_API=http://localhost:8000
export QUANTBET_TOKEN=dev-token
streamlit run frontend/app.py
```

---

## Testing guidance

When editing:

- API routes: run `tests/test_api.py`.
- Registry/model loading: run `tests/test_registry.py`.
- De-vig or staking: run `tests/test_devig.py`, `tests/test_value_engine.py`, and `models/quantbet/test_quantbet.py`.
- Database/ETL: add fixtures or temporary SQLite databases; do not rely on a local production database.
- World Cup simulation: run smoke tests before deeper benchmark tests.

Some tests are intentionally heavier because they touch model fitting, benchmarks, or PyMC. Prefer targeted tests during small edits, then full `pytest` before final delivery.

---

## Documentation rules

For GitHub-facing docs:

- Keep `README.md` English-first.
- Explain the project in terms of prediction, architecture, and reproducible usage.
- Do not over-index on betting language.
- Include the research/education disclaimer.
- Be explicit about what is bundled and what is optional.
- Avoid stale references to old `quantbet_ev/main.py` style layouts.

For AI-agent docs:

- Keep `CLAUDE.md` focused on how to modify the repository safely.
- Document invariants and gotchas.
- Keep commands copy-pasteable.
- Update this file whenever architecture or default model version changes.

---

## Known gotchas

- `requirements.txt` includes API/model dependencies but not every optional UI/LLM dependency. `streamlit` and an OpenAI-compatible client may need to be installed separately depending on the workflow.
- The packaged archive includes `bayesian_dc_v9.pkl`; older or optional ensemble artifacts may not be present.
- Database query endpoints need an initialized/populated local database. The prediction endpoint can work from the model artifact.
- The API request schema uses `home_team` and `away_team`, not `home` and `away`.
- Do not commit `.env`, database files, private API keys, or large downloaded datasets.

---

## Safe change checklist

Before returning work to the user, check:

- Did README still describe the real repository layout?
- Did CLAUDE still point to the correct entry points?
- Did API examples use `home_team` / `away_team`?
- Did the change keep odds separate from prediction?
- Did model version references match `models/registry.py`?
- Did any new docs avoid unsupported accuracy/profit claims?
- Did tests or at least targeted checks run, or did you clearly state they were not run?

