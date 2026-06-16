# 2026 World Cup Score Predictions

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Model](https://img.shields.io/badge/Model-v9.1-orange)](models/registry.py)

Bayesian Dixon-Coles scoreline forecasting with Elo-informed priors, GAS dynamic strengths, and gradient-boosted ensemble via log pooling. 49,415 international matches, 336 teams, FastAPI + Streamlit serving.

> Research and education project. Not betting, financial, or legal advice.

---

## Demo

```python
>>> from models.registry import load_model
>>> dc = load_model()
>>> dc.predict("Spain", "Cape Verde", venue="neutral")
```

```
Spain vs Cape Verde
H=55.1%  D=25.1%  A=19.8%  |  λ 3.10 − 0.56  |  O2.5 58.6%  BTTS 30.9%
Handicap -2: H=42.8%  Push=19.5%  A=37.7%
pool_method=log  dc_weight=0.621

Top 10 scores:
 2-0   13.46%   1-0   11.81%   3-0   11.60%
 4-0    8.44%   0-0    6.23%   2-1    5.83%
 1-1    5.48%   5-0    5.48%   3-1    5.03%
 4-1    3.66%

Spain recent 10 (newest first):
 2026-06-08  3-1  vs Peru         (Spain)
 2026-06-04  1-1  vs Iraq         (draw)
 2026-03-31  0-0  vs Egypt        (draw)
 2026-03-27  3-0  vs Serbia       (Spain)
 2025-11-18  2-2  vs Turkey       (draw)
 2025-11-15  4-0  vs Georgia      (Spain)
 2025-10-14  4-0  vs Bulgaria     (Spain)
 2025-10-11  2-0  vs Georgia      (Spain)
 2025-09-07  6-0  vs Turkey       (Spain)
 2025-09-04  3-0  vs Bulgaria     (Spain)

Cape Verde recent 10 (newest first):
 2026-06-06  3-0  vs Bermuda     (Cape Verde)
 2026-05-31  3-0  vs Serbia      (Cape Verde)
 2026-03-30  1-1  vs Finland     (draw)
 2026-03-27  2-4  vs Chile       (Chile)
 2025-11-17  1-1  vs Egypt       (draw)
 2025-11-13  0-0  vs Iran        (draw)
 2025-10-13  3-0  vs Eswatini    (Cape Verde)
 2025-10-08  3-3  vs Libya       (draw)
 2025-09-09  1-0  vs Cameroon    (Cape Verde)
 2025-09-04  2-0  vs Mauritius   (Cape Verde)

Trace:
  att_home=1.673  def_home=-1.447  att_away=0.590  def_away=-0.669
  gas_δ_home=-0.034  gas_δ_away=0.302
  λ_home=3.10  λ_away=0.56  calibrator_temp=0.846
```

---

## Quick Start

Python 3.10+ required.

```bash
git clone https://github.com/AaronYueLIN/2026-World-Cup-Score-Predictions.git
cd 2026-World-Cup-Score-Predictions
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start the API:

```bash
export APP_ENV=development API_TOKEN=dev-token
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` for the interactive OpenAPI explorer.

---

## Core Architecture

```text
Historical matches (49k) + Elo ratings
        │
        ▼
Bayesian Dixon-Coles (PyMC MAP)
  team attack / defense / venue effects
        │
        ▼
GAS dynamic-strength layer
  DoubleGAS: slow B=0.985 + fast B=0.96 (blended 70/30)
        │
        ▼
Flexible scoreline engine
  NB marginals + calibrator (temp = 0.846)
        │
        ├──► DC 1X2
        │
        ▼
HistGBM ensemble + log pooling
  64 features (rolling form, h2h, momentum)
  DC_weight ≈ 0.62, GBM_weight ≈ 0.38
        │
        ▼
Final 1X2 + O2.5 + BTTS + exact scores

────── prediction firewall ──────

Bookmaker odds → Shin de-vig → EV filter → Kelly staking  (optional, separate)
```

---

## Model Pipeline

### 1. Dixon-Coles Base Model

Estimates team attack/defense parameters from match results. Uses PyMC `find_MAP` with `ZeroSumNormal` constraints and `HalfNormal` priors. Serialized as `models/bayesian_dc_v9.pkl`.

### 2. Elo-Informed Anchoring

Elo ratings (from `eloratings.net`) are log-compressed via `sign(x) * log1p(|x|)` and fed as priors on team strength: `mu_att_i = mu_conf(i) + eta * elo_i`.

### 3. GAS Dynamic Strength

Score-driven update `delta(t+1) = B * delta(t) + A * s_t` tracks short-term deviation from the Elo anchor. B = 0.985 fixed; A learned via MLE. Exposed at `GET /api/v1/model/momentum`.

### 4. Scoreline Matrix

Negative Binomial marginals produce an 11×11 joint score matrix. Calibrated via `ScoreMatrixCalibrator(temp=0.846)`. 1X2, O2.5, BTTS, and exact-score probabilities all read from the same matrix.

### 5. HistGBM Ensemble + Log Pooling

`HistGradientBoostingClassifier` (300 trees, 64 features) trained on the full 49k match history. DC and GBM probabilities are fused via log pooling with weight optimized to minimize RPS on a temporal validation split. Artifacts: `models/ensemble_v3.pkl`, `models/ensemble_data.parquet`. Retrain: `python scripts/train_ensemble.py` (~45 s).

### 6. Optional Value Layer

`models/quantbet/` contains Shin de-vigging, expected value, Kelly staking, and portfolio utilities. Separate from the prediction path.

---

## Repository Layout

```text
.
├── api/                 FastAPI app, routes, schemas, auth, rate limit, /metrics
├── db/                  SQLAlchemy models, settings, ETL, Alembic migrations
├── frontend/            Streamlit workbench (chat, momentum, batch prediction)
├── models/
│   ├── bayesian_dc_v9.pkl          current DC model artifact (8.7 MB)
│   ├── ensemble_v3.pkl             trained GBM + pooling weight (4.7 MB)
│   ├── ensemble_data.parquet       49k companion data for feature engineering
│   ├── bayesian_dixon_coles.py     DC fit + predict (ensemble branch)
│   ├── registry.py                 unified load_model() entry point
│   └── quantbet/                   de-vig, staking, portfolio, scoreline, worldcup
├── scripts/
│   ├── train_ensemble.py           one-shot ensemble retraining
│   ├── elo_fetcher.py              Elo scraper
│   └── etl_pipeline.py             match-data ingestion
├── tests/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Configuration

| Variable | Purpose | Dev default |
|----------|---------|-------------|
| `APP_ENV` | `development` / `production` / `test` | `development` |
| `API_TOKEN` | Bearer token for protected routes | `dev-token` |
| `DATABASE_URL` | SQLAlchemy database URL | SQLite local |
| `LLM_API_KEY` | DeepSeek / OpenAI key for chat endpoint | *(none)* |

`/health` and `/metrics` skip auth. All other routes require `Authorization: Bearer <API_TOKEN>`.

---

## Testing

```bash
pytest

# targeted
pytest tests/test_api.py -q
pytest tests/test_registry.py -q

# lint
ruff check models/quantbet/ db/ api/
```

---

## License

MIT. See `LICENSE`.

---

*This project was built by LLM agents operating on ideas and direction from the repository owner.*
