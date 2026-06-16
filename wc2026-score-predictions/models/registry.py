"""
registry.py — Model single source of truth
============================================

Usage
-----
    from registry import load_model, load_ensemble, describe, current_version, list_available

    dc  = load_model()              # Currently selected Bayesian DC
    ens = load_ensemble()           # Corresponding DC+GBM ensemble (may be None)
    print(describe())               # Print which module is currently in use
    print(list_available())         # List all available versions and their status

Command-line self-check:
    python models/registry.py
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any

# -- CLI compatibility -------------------------------------------------
# When running `python models/registry.py`, models/ is not in sys.path,
# so we need to add the parent directory before importing.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_SCRIPT_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
# -----------------------------------------------------------------------

from models.exceptions import GasFitError, ModelNotFoundError, PredictionError, QuantBetError

# Re-export exceptions (unified entry point)
__all__ = [
    "load_model", "load_ensemble", "describe", "current_version",
    "list_available", "model_metadata",
    "ModelNotFoundError", "PredictionError", "GasFitError", "QuantBetError",
]
logger = logging.getLogger(__name__)

# Path anchored to this file, independent of cwd
_THIS = os.path.abspath(__file__)
MODELS_DIR = os.path.dirname(_THIS)
PROJECT_ROOT = os.path.dirname(MODELS_DIR)

# ====================================================================
#  Currently selected version — change model by editing this line only
# ====================================================================
MODEL_VERSION = "v9"

# Version -> artifact filename (relative to models/). Set to None if no ensemble.
REGISTRY: dict[str, dict] = {
    "v9":       {"dc": "bayesian_dc_v9.pkl",      "ensemble": "ensemble_v3.pkl"},
    "v7":       {"dc": "bayesian_dc_v7.pkl",       "ensemble": "ensemble_v3.pkl"},
    "v6_clean": {"dc": "bayesian_dc_v6_clean.pkl", "ensemble": "ensemble_v3.pkl"},
}


# ---------------------------------------------------------------- Internal
_cache_info: dict[str, dict[str, Any]] = {}


def _ensure_import_path() -> None:
    if MODELS_DIR not in sys.path:
        sys.path.insert(0, MODELS_DIR)


def _load_pickle(path: str | None) -> Any:
    if not path or not os.path.exists(path):
        raise ModelNotFoundError(
            f"Model artifact missing: {path}\n"
            f"  -> Check registry.MODEL_VERSION / REGISTRY, or retrain and export the pkl."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- Public API

def current_version() -> str:
    return MODEL_VERSION


def paths(version: str | None = None) -> dict[str, str | None]:
    """Return absolute paths for current (or specified) version artifacts."""
    v = version or MODEL_VERSION
    if v not in REGISTRY:
        raise KeyError(f"Unknown model version {v!r}; available: {list(REGISTRY)}")
    return {
        k: (os.path.join(MODELS_DIR, fn) if fn else None)
        for k, fn in REGISTRY[v].items()
    }


@lru_cache(maxsize=None)
def load_model(version: str | None = None, with_scoreline: bool = True) -> Any:
    """Load current (or specified) version of Bayesian DC model (cached, loads once per process)."""
    v = version or MODEL_VERSION
    _ensure_import_path()
    p = paths(v)["dc"]
    dc = _load_pickle(p)

    # Record metadata
    _record_metadata(v, p)

    if with_scoreline and not hasattr(dc, "_scoreline_model"):
        try:
            from bayesian_dixon_coles import install_scoreline  # type: ignore
            install_scoreline(dc)
            logger.info("registry: old %s auto-installed NB+Frank score engine", v)
        except Exception as e:
            logger.warning("registry: scoreline install failed (%s); predict may fall back to old path", e)
    if getattr(dc, "_gas", None) is None:
        logger.warning("registry: loaded %s has no persisted GAS, predict will fall back to static!", v)
    return dc


def _record_metadata(version: str, path: str | None) -> None:
    """Record model loading metadata (load time, file size, file hash)."""
    meta: dict[str, Any] = {
        "version": version,
        "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if path and os.path.exists(path):
        meta["file_size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 2)
        with open(path, "rb") as f:
            meta["file_hash_sha256"] = hashlib.sha256(f.read()).hexdigest()[:16]
    _cache_info[version] = meta


def model_metadata(version: str | None = None) -> dict[str, Any]:
    """Return loading metadata for current (or specified) version model."""
    v = version or MODEL_VERSION
    return _cache_info.get(v, {"version": v, "loaded_at": None})


def list_available() -> list[dict[str, Any]]:
    """List all registered versions in REGISTRY and their pkl file status."""
    result = []
    for v, artifacts in REGISTRY.items():
        entry: dict[str, Any] = {"version": v, "artifacts": {}}
        for kind, fn in artifacts.items():
            full = os.path.join(MODELS_DIR, fn) if fn else None
            entry["artifacts"][kind] = {
                "path": fn,
                "exists": full is not None and os.path.exists(full),
            }
        result.append(entry)
    return result


def describe(version: str | None = None) -> dict[str, Any]:
    """See at a glance 'which module is currently in use'."""
    v = version or MODEL_VERSION
    dc = load_model(v)
    sm = getattr(dc, "_scoreline_model", None)
    meta = model_metadata(v)
    return {
        "selected_version": v,
        "dc_file": os.path.relpath(paths(v)["dc"], PROJECT_ROOT),
        "dc_class": type(dc).__name__,
        "n_teams": len(getattr(dc, "teams", [])),
        "scoreline_engine": (
            f"{getattr(sm, 'margin', None)}+{getattr(sm, 'dependence', None)}"
            if sm is not None else "legacy Poisson+tau"
        ),
        "scoreline_shape": getattr(sm, "shape_", None) if sm is not None else None,
        "has_calibrator": hasattr(dc, "_calibrator"),
        "has_gas": getattr(dc, "_gas", None) is not None,
        "has_elo_prior": "eta_att" in getattr(dc, "params", {}),
        "loaded_at": meta.get("loaded_at"),
        "file_size_mb": meta.get("file_size_mb"),
        "file_hash_sha256": meta.get("file_hash_sha256"),
    }


def load_ensemble(version: str | None = None) -> Any:
    """Load current (or specified) version DC+GBM ensemble (cached)."""
    v = version or MODEL_VERSION
    _ensure_import_path()
    ep = paths(v).get("ensemble")
    if ep is None:
        return None
    if "ensemble" not in sys.modules:
        try:
            sys.modules["ensemble"] = importlib.import_module("ml_predictor")
        except Exception as e:
            logger.warning("registry: alias 'ensemble' creation failed (%s)", e)
    return _load_pickle(ep)


def save_training_metadata(
    version: str,
    dc: Any,
    *,
    training_matches: int = 0,
    log_lik: float = 0.0,
    aic: float = 0.0,
    bic: float = 0.0,
    rps_holdout: float = 0.0,
    git_commit: str | None = None,
    extra: dict | None = None,
) -> str:
    """Save training metadata to models/<version>.meta.json."""
    if git_commit is None:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            git_commit = "unknown"

    meta = {
        "version": version,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "training_matches": training_matches,
        "n_teams": len(getattr(dc, "teams", [])),
        "log_lik": round(log_lik, 2),
        "aic": round(aic, 2),
        "bic": round(bic, 2),
        "rps_holdout": round(rps_holdout, 4),
        "gas_b": round(float(dc._gas_hyper.get("B", 0)), 4) if getattr(dc, "_gas", None) else 0.0,
        "gas_a": round(float(dc._gas_hyper.get("A_att", 0)), 4) if getattr(dc, "_gas", None) else 0.0,
        "shape_nb_r": dc._scoreline_model.shape_.get("nb_r", 0) if hasattr(dc, "_scoreline_model") else 0,
        "git_commit": git_commit,
    }
    if extra:
        meta.update(extra)

    path = os.path.join(MODELS_DIR, f"{version}.meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Training metadata saved: %s", path)
    return path


def compare_models(v1: str = "v8", v2: str = "v9") -> dict[str, Any]:
    """Compare description info for two versions (RPS needs external input)."""
    # Load metadata
    meta_path_v1 = os.path.join(MODELS_DIR, f"{v1}.meta.json")
    meta_path_v2 = os.path.join(MODELS_DIR, f"{v2}.meta.json")
    meta_v1 = json.load(open(meta_path_v1)) if os.path.exists(meta_path_v1) else {"version": v1}
    meta_v2 = json.load(open(meta_path_v2)) if os.path.exists(meta_path_v2) else {"version": v2}

    d1 = describe(v1)
    d2 = describe(v2)

    return {
        "v1": {
            "version": v1,
            "n_teams": d1.get("n_teams"),
            "has_gas": d1.get("has_gas"),
            "has_elo_prior": d1.get("has_elo_prior"),
            "log_lik": meta_v1.get("log_lik"),
            "aic": meta_v1.get("aic"),
            "rps_holdout": meta_v1.get("rps_holdout"),
            "file_size_mb": d1.get("file_size_mb"),
        },
        "v2": {
            "version": v2,
            "n_teams": d2.get("n_teams"),
            "has_gas": d2.get("has_gas"),
            "has_elo_prior": d2.get("has_elo_prior"),
            "log_lik": meta_v2.get("log_lik"),
            "aic": meta_v2.get("aic"),
            "rps_holdout": meta_v2.get("rps_holdout"),
            "file_size_mb": d2.get("file_size_mb"),
        },
        "delta_log_lik": (meta_v2.get("log_lik") or 0) - (meta_v1.get("log_lik") or 0),
        "better": v2 if (meta_v2.get("log_lik") or -1e9) > (meta_v1.get("log_lik") or -1e9) else v1,
    }


if __name__ == "__main__":
    result = describe()
    result["available_versions"] = list_available()
    print(json.dumps(result, ensure_ascii=False, indent=2))
