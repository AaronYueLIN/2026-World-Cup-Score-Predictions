# wc2026-score-predictions

Bayesian Dixon-Coles score prediction + HistGBM ensemble + log pooling + FastAPI serving.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

```python
from models.registry import load_model
dc = load_model()
r = dc.predict("Germany", "Japan", venue="neutral")
# pool_method=log  dc_weight≈0.62  → H=xx% D=xx% A=xx%
```

## Pipeline

```
SQL 49k matches + Elo ratings
  → Bayesian DC (PyMC MAP) → att/def anchors
    → GAS(1,1) B=0.985 A≈0.059 → match λ_h, λ_a
      → NB score matrix + Calibrator(temp=0.846) → 11×11 M → DC_1X2
                                                  │
        [firewall — no odds in prediction track]   │
                                                  ▼
                              ┌────── log_pool(DC_1X2, GBM_1X2; w=0.62)
                              │
      FeatureEngineer ──→ HistGBM (64-dim rolling form/h2h/momentum)
      (49k companion data)

Odds → Shin de-vig → market prob
  → EV = model − market (filter EV>0)
    → risk-constrained Kelly
```

## Key Files

```
models/
├── bayesian_dixon_coles.py   — DC fit + predict (contains ensemble branch)
├── bayesian_dc_v9.pkl        — trained DC model (8.7 MB)
├── ensemble_v3.pkl           — trained HistGBM + weight (4.7 MB)
├── ensemble_data.parquet     — 49k companion data for FE (0.4 MB)
├── registry.py               — unified load_model()
├── ml_predictor.py           — FeatureEngineer, HistGBMPredictor, build_training_matrix_fast
└── quantbet/
    ├── scoreline/            — NB, copula, calibration, GAS
    ├── worldcup/             — tournament sim, knockout, TRPS
    ├── devig.py              — Shin de-vig
    ├── pooling.py            — log_pool / linear_pool
    ├── portfolio.py          — Kelly staking
    └── value_engine_v2.py    — EV filter
db/            — SQLAlchemy + Alembic
api/           — FastAPI + Bearer auth + rate limit
frontend/      — Streamlit + DeepSeek chat
scripts/
├── train_ensemble.py         — (re)train HistGBM + optimize weight (~45s with vectorized FE)
├── elo_fetcher.py            — ELO scraper
└── etl_pipeline.py           — match data ingestion
```

## Runtime Predict Flow

```
dc.predict("Spain", "Cape Verde")
  ├── DC core (always runs)
  │     GAS λ → NB matrix → calibrator → DC_1X2
  │
  ├── Ensemble branch (if ensemble_v3.pkl + companion present)
  │     _loading_ensemble guard → prevent recursion
  │     load ensemble_v3.pkl (dict: gbm_model, feature_engineer, dc_weight)
  │     load ensemble_data.parquet (49k matches for rolling features)
  │     FeatureEngineer.get_match_features() → 64-dim vector
  │       └── _dc_features() calls dc.predict() again ← _loading_ensemble=True skips
  │     HistGBM.predict_proba() → GBM_1X2
  │     log_pool([DC_1X2, GBM_1X2], [dc_weight, 1-dc_weight])
  │     → final 1X2 with pool_method="log"
  │
  └── Fallback (silent)
        ensemble files missing → pool_method=None → pure DC output
```

## Retrain Ensemble

```bash
python scripts/train_ensemble.py
# ~45s: load 49k → vectorized FE (build_training_matrix_fast) → HistGBM fit → weight optimize → save
```

Add incremental data to SQLite first (`db/etl.py`), then rerun. The feature matrix is cached as `ensemble_data.parquet`; future runs only compute features for new rows.

## Stack

| Layer | Tool |
|-------|------|
| Data | SQLite/PostgreSQL + SQLAlchemy + Alembic |
| DC Model | PyMC find_MAP, NB + ScoreMatrixCalibrator |
| Ensemble | HistGradientBoosting (sklearn) + log pooling |
| Value | Shin de-vig, risk-constrained Kelly |
| API | FastAPI, Pydantic v2, structlog, Prometheus |
| Frontend | Streamlit, httpx, DeepSeek LLM |
| Deploy | Docker Compose (dev/prod profiles) |

## References

Dixon-Coles (1997), Karlis-Ntzoufras (2003), Boshnakov-Kharrat-McHale (2017),
Groll et al. (2024), Busseti-Ryu-Boyd (2016), Koopman-Lit (2015),
Genest-Zidek (1986), Ranjan-Gneiting (2010)
