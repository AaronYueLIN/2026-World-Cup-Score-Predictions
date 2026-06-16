"""Train HistGBM ensemble on full 49k data, optimize pooling weight, save artifacts.

One-shot. Run once after major DC model updates.
~25 min on laptop.

Usage:
    python scripts/train_ensemble.py
"""
from __future__ import annotations

import os, sys, time, pickle, logging, warnings

import numpy as np
import pandas as pd
from sqlalchemy import text

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5s] %(message)s")
logger = logging.getLogger("train_ensemble")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "models") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "models"))

t0_total = time.time()

# =========================================================================
# Step 1: Load full 49k data (bypass dev-mode 5-year cap)
# =========================================================================
logger.info("Step 1: Loading full 49k from SQLite ...")
from db.reader import _get_engine, _COMPETITION_MAP, _build_competition_map

engine = _get_engine()
with engine.connect() as conn:
    df = pd.read_sql(text("""
        SELECT m.date, ht.name AS home_team, at.name AS away_team,
               m.home_score AS home_goals, m.away_score AS away_goals,
               COALESCE(t.name, 'Friendly') AS tournament,
               m.neutral, m.venue
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        LEFT JOIN tournaments t ON m.tournament_id = t.id
        ORDER BY m.date
    """), conn)

_build_competition_map()
df["competition"] = df["tournament"].map(_COMPETITION_MAP).fillna("friendly")
df["date"] = pd.to_datetime(df["date"])

logger.info("  Loaded %d rows, %d teams, %s ~ %s",
    len(df), df["home_team"].nunique(),
    df["date"].min().date(), df["date"].max().date())

# =========================================================================
# Step 2: Build labels
# =========================================================================
logger.info("Step 2: Building result labels ...")

def make_result(row):
    if row["home_goals"] > row["away_goals"]:
        return "H"
    if row["home_goals"] == row["away_goals"]:
        return "D"
    return "A"

df["result"] = df.apply(make_result, axis=1)

# =========================================================================
# Step 3: Load DC model
# =========================================================================
logger.info("Step 3: Loading v9 DC model ...")
from models.registry import load_model

dc = load_model()
dc._ensemble = None          # prevent ensemble code during training
dc._fit_df = None            # will be set after training
logger.info("  %d teams loaded", len(dc.teams))

# =========================================================================
# Step 4: Temporal split
# =========================================================================
logger.info("Step 4: Temporal split ...")
df = df.sort_values("date").reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
val_df = df.iloc[split_idx:].copy()
logger.info("  train=%d (before %s)  val=%d (%s~%s)",
    len(train_df), train_df["date"].max().date(),
    len(val_df), val_df["date"].min().date(), val_df["date"].max().date())

# =========================================================================
# Step 5: Feature engineering (vectorized, ~30s)
# =========================================================================
logger.info("Step 5: Feature engineering on %d train rows ...", len(train_df))
t0_fe = time.time()

# Prevent ensemble loading during training (dc.predict() is called by _dc_features)
dc._loading_ensemble = True

from ml_predictor import FeatureEngineer

fe = FeatureEngineer()
X_train, y_train = fe.build_training_matrix_fast(train_df, dc_model=dc)

elapsed_fe = time.time() - t0_fe
logger.info("  X_train=%s, y_train=%s (%.0fs)", X_train.shape, y_train.shape, elapsed_fe)

# =========================================================================
# Step 6: Train HistGBM
# =========================================================================
logger.info("Step 6: Training HistGBM ...")
t0_gbm = time.time()

from ml_predictor import HistGBMPredictor

gbm = HistGBMPredictor(n_estimators=300, learning_rate=0.05, max_depth=4)
gbm.fit(X_train, y_train)

elapsed_gbm = time.time() - t0_gbm
logger.info("  Trained in %.1fs", elapsed_gbm)

# Keep _loading_ensemble = True through Step 7 (fit_weights calls dc.predict)
# =========================================================================
# Step 7: Optimize pooling weight (use vectorized features, skip fit_weights loop)
# =========================================================================
logger.info("Step 7: Optimizing DC/GBM pooling weight on %d val rows ...", len(val_df))
t0_w = time.time()

# Build validation features with the same fast method
X_val, y_val = fe.build_training_matrix_fast(val_df, dc_model=dc)
logger.info("  X_val=%s, y_val=%s (%.0fs)", X_val.shape, y_val.shape, time.time() - t0_w)

# Get GBM probabilities on validation set
gbm_probs_val = gbm.model.predict_proba(X_val[gbm.feature_cols].values.astype(float))
# GBM output: [A, D, H] per OUTCOME_ENCODE

# Get DC probabilities on validation set (from cached DC features)
# dc_p_home/dc_p_draw/dc_p_away -> reorder to [A, D, H] = [dc_p_away, dc_p_draw, dc_p_home]
dc_probs_val = X_val[["dc_p_away", "dc_p_draw", "dc_p_home"]].values

# Grid search for optimal weight
from scipy.optimize import minimize_scalar
from ml_predictor import _pool_two, _mean_rps

def rps_for_w(w):
    blended = _pool_two(dc_probs_val, gbm_probs_val, w, method="log")
    return _mean_rps(blended, y_val)

result = minimize_scalar(rps_for_w, bounds=(0.0, 1.0), method="bounded")
w_opt = float(result.x)
dc_rps = rps_for_w(1.0)
gbm_rps = rps_for_w(0.0)
opt_rps = result.fun
logger.info("  Optimized: DC=%.4f  GBM=%.4f  Ensemble=%.4f  RPS: DC=%.4f  GBM=%.4f  Opt=%.4f",
    1-w_opt, w_opt, opt_rps, dc_rps, gbm_rps, opt_rps)

ens_dc_weight = w_opt

elapsed_w = time.time() - t0_w
logger.info("  Optimized in %.0fs  DC_weight=%.4f  GBM_weight=%.4f",
    elapsed_w, 1-w_opt, w_opt)

# Restore normal predict behavior
dc._loading_ensemble = False

# =========================================================================
# Step 8: Save artifacts
# =========================================================================
logger.info("Step 8: Saving artifacts ...")

artifact = {
    "gbm_model": gbm,
    "feature_engineer": fe,
    "dc_weight": w_opt,
    "pool_method": "log",
}
pkl_path = os.path.join(ROOT, "models", "ensemble_v3.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(artifact, f)
logger.info("  Saved %s (%.1f MB)", pkl_path, os.path.getsize(pkl_path) / 1e6)

parquet_path = os.path.join(ROOT, "models", "ensemble_data.parquet")
df.to_parquet(parquet_path, index=False)
logger.info("  Saved %s (%.1f MB)", parquet_path, os.path.getsize(parquet_path) / 1e6)

# Also update the companion data on the loaded model
dc._fit_df = df
if "date" in dc._fit_df.columns:
    dc._fit_df["date"] = pd.to_datetime(dc._fit_df["date"])

total = time.time() - t0_total
logger.info("")
logger.info("Done. Total: %.0fs  DC_weight=%.4f  GBM_weight=%.4f",
    total, w_opt, 1-w_opt)
