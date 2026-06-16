"""
rebuild_v9_gas.py — Safely persist GAS into v9 (no retraining, atomic write, with backup)
=======================================================================================

Why this way (learning from last OOM + lost pkl)
------------------------------------------------
· Never call dc.fit() — that runs PyMC MAP from scratch (336 teams x 49k), which will OOM on a 32GB laptop.
  This script reuses v9's **existing** MAP attack/defense parameters, doing only pure numpy GAS fitting.
· v9's anchor has static momentum baked in (params contain mom_att/mom_def/mom_vec). Stacking GAS directly
  would double-count recent form. The script first subtracts the momentum contribution from attack/defense,
  yielding a pure Elo+confederation anchor, then stacks GAS on top.
· Atomic write: write .tmp -> verify loadable and _gas is in place -> os.replace rename. Backup before overwrite.
  This way any mid-crash won't destroy the usable pkl again.

Usage
-----
    python models/rebuild_v9_gas.py --data results.csv
data must contain: date, home_team, away_team, home_goals, away_goals [, venue]
(Export a CSV from SQL once; script never connects to SQL or runs PyMC)
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Data missing columns {miss}; actual columns {list(df.columns)}")
    df = df.dropna(subset=list(need)).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "venue" not in df.columns:
        df["venue"] = "neutral"
    return df


def decontaminate(dc) -> str:
    """Subtract baked-in static momentum contribution from attack/defense, yielding pure Elo+confederation anchor. Returns description."""
    p = dc.params
    if "mom_att" not in p or "mom_vec" not in p:
        return "No mom_* keys, anchor already pure, skip"
    mom_att = float(p.get("mom_att", 0.0))
    mom_def = float(p.get("mom_def", 0.0))
    if abs(mom_att) + abs(mom_def) < 1e-9:
        return "mom coefficients are 0, no decontamination needed"
    mom_vec = np.asarray(p["mom_vec"], dtype=float)
    p["attack"] = np.asarray(p["attack"], dtype=float) - mom_att * mom_vec
    p["defense"] = np.asarray(p["defense"], dtype=float) - mom_def * mom_vec
    p["mom_att"] = 0.0
    p["mom_def"] = 0.0
    # Refresh score engine strengths (using pure anchor)
    if getattr(dc, "_scoreline_model", None) is not None:
        dc._scoreline_model.set_strengths(
            {t: float(v) for t, v in zip(dc.teams, p["attack"])},
            {t: float(v) for t, v in zip(dc.teams, p["defense"])},
            float(p["home_adj"]), float(p["neutral_adj"]),
        )
    return f"Decontaminated: subtracted mom_att={mom_att:.3f}, mom_def={mom_def:.3f} contribution"


def rebuild(in_path: str, out_path: str, data_path: str, decon: bool = True):
    # 0) Backup (keep a usable copy before overwrite)
    if os.path.exists(out_path):
        bak = f"{out_path}.backup-{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(out_path, bak)
        print(f"[backup] {bak}")

    # 1) Load + assert MAP complete (no retraining)
    with open(in_path, "rb") as f:
        dc = pickle.load(f)
    assert len(getattr(dc, "teams", [])) > 50, "MAP team count abnormal"
    assert float(np.std(dc.params["attack"])) > 1e-3, "attack has no variance, MAP may be corrupted"
    assert hasattr(dc, "_scoreline_model"), "Missing _scoreline_model"
    print(f"[load ] {os.path.basename(in_path)}  teams={len(dc.teams)}  eta_att={float(dc.params.get('eta_att',0)):.3f}")

    # 2) Decontaminate (subtract static momentum baked into anchor, avoid double-counting with GAS)
    if decon:
        print(f"[decon] {decontaminate(dc)}")

    # 3) Fit GAS (pure numpy, pass df to bypass SQL, never trigger PyMC)
    #    fix_B=0.985 -> B fixed to held-out RPS optimal value, only optimize A (prevents sliding to B->1 edge solution)
    df = load_matches(data_path)
    df = df[df.home_team.isin(dc.team_idx) & df.away_team.isin(dc.team_idx)].reset_index(drop=True)
    print(f"[gas  ] Using {len(df)} matches to fit GAS (fix_B=0.985)...")
    dc._fit_gas(df=df, fix_B=0.985)
    print(f"[gas  ] A={dc._gas_hyper['A_att']:.4f} B={dc._gas_hyper['B']:.4f} n={dc._gas_hyper['n_matches']}")

    # 4) Clean up potentially heavy/non-serializable residuals
    if hasattr(dc, "_pymc_model"):
        del dc._pymc_model

    # 5) Atomic write: tmp -> verify -> rename
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(dc, f)
    with open(tmp, "rb") as f:
        chk = pickle.load(f)
    assert getattr(chk, "_gas", None) is not None, "tmp verification: _gas missing"
    # Offline predict must use GAS (not fall back to static)
    a, b = chk.teams[0], chk.teams[1]
    r = chk.predict(a, b, venue="neutral")
    assert getattr(chk, "_gas", None) is not None, "_gas should not disappear after predict"
    os.replace(tmp, out_path)
    print(f"[save ] {os.path.basename(out_path)}  [OK] Atomic write complete")
    print(f"[verify] sample predict {a} vs {b}: H={r['home_win_prob']:.3f} lh={r['expected_home_goals']:.3f} (GAS active)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--in", dest="in_path", default=os.path.join(HERE, "bayesian_dc_v9.pkl"))
    ap.add_argument("--out", dest="out_path", default=os.path.join(HERE, "bayesian_dc_v9.pkl"))
    ap.add_argument("--no-decontaminate", action="store_true")
    a = ap.parse_args()
    rebuild(a.in_path, a.out_path, a.data, decon=not a.no_decontaminate)


if __name__ == "__main__":
    main()
