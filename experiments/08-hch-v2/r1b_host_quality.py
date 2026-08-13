"""R1B §7 host-quality sanity screen -> host_quality_by_domain.csv.

For every market x host domain (cache must exist), compute host error
regimes on frozen H0-fit host predictions:

  - MAE by segment (S1R / S2V / S4)
  - transformed host error  E|zY - z0|  (host identity baseline on S2V)
  - residual mean / std / IQR
  - lag-1 residual ACF
  - negative-residual rate
  - large-positive / large-negative residual rates

Purpose: determine whether R1B spans heterogeneous host-error regimes.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from common import load_dataset
from eval_manifest import ExperimentManifest
import r1a_run as R

MARKETS = ["LAGO_DE", "LAGO_PJM", "NEM_SA1", "NORD_DK1"]
HOSTS = ["Linear", "MLP", "LSTM", "PatchTST"]


def _lag1_acf(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 4 or np.std(x) < 1e-12:
        return 0.0
    x = x - x.mean()
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def quality_for(ds_key: str, bb: str) -> dict:
    ds = load_dataset(ds_key)
    y_full = ds["price"].astype(np.float64)
    ts = ds["ts"]
    exp = ExperimentManifest.from_dataset(ds, np.isfinite(y_full),
                                          dataset_id=ds_key)
    cache = HERE / "results" / "cache" / ds_key / bb
    seg = json.load(open(cache / "seg.json"))
    yhat_full = np.load(cache / "pred_full.npy").astype(np.float64)

    def mae(split: str):
        rows = exp.valid_indices_in_split(split)
        y = y_full[rows]
        yh = yhat_full[rows]
        m = np.isfinite(y) & np.isfinite(yh)
        if m.sum() == 0:
            return None
        return float(np.mean(np.abs(y[m] - yh[m])))

    # residual stats over S2V (segment used for candidate selection / transfer)
    rows = exp.valid_indices_in_split("S2V")
    y = y_full[rows]
    yh = yhat_full[rows]
    m = np.isfinite(y) & np.isfinite(yh)
    resid = y[m] - yh[m]
    # scale-free transformed host error E|zY - z0| on S2V
    s2v_mae_raw = float(np.mean(np.abs(resid))) if len(resid) else None
    resid_mean = float(np.mean(resid)) if len(resid) else None
    resid_std = float(np.std(resid)) if len(resid) else None
    resid_iqr = float(np.percentile(resid, 75) - np.percentile(resid, 25)) if len(resid) else None
    acf1 = _lag1_acf(resid) if len(resid) else None
    neg_rate = float(np.mean(resid < 0)) if len(resid) else None
    q99 = np.percentile(np.abs(resid), 99) if len(resid) else 0.0
    large_pos = float(np.mean(resid > q99)) if len(resid) else None
    large_neg = float(np.mean(resid < -q99)) if len(resid) else None

    # transformed host error (identity corrector) = mean |zY - z0| over S2V
    z0, s = R.precompute_scale_free(yhat_full, ts)
    zm = np.isfinite(z0) & np.isfinite(y_full) & (s > 0)
    rows_s = exp.valid_indices_in_split("S2V")
    sub = zm & (np.isin(np.arange(len(y_full)), rows_s))
    if sub.sum() > 0:
        s_safe = np.maximum(s[sub], 1e-12)
        tr_err = float(np.mean(np.abs(
            np.arcsinh(y_full[sub] / s_safe) - z0[sub])))
    else:
        tr_err = None

    return {
        "market": ds_key, "host": bb,
        "n_H0": seg["n_H0"], "n_S1R": seg["n_S1R"], "n_S2V": seg["n_S2V"],
        "s1r_mae": round(mae("S1R"), 4) if mae("S1R") is not None else None,
        "s2v_mae": round(mae("S2V"), 4) if mae("S2V") is not None else None,
        "s4_mae": round(mae("S4"), 4) if mae("S4") is not None else None,
        "s2v_mae_raw": round(s2v_mae_raw, 4) if s2v_mae_raw is not None else None,
        "transformed_host_error": round(tr_err, 4) if tr_err is not None else None,
        "resid_mean": round(resid_mean, 4) if resid_mean is not None else None,
        "resid_std": round(resid_std, 4) if resid_std is not None else None,
        "resid_iqr": round(resid_iqr, 4) if resid_iqr is not None else None,
        "resid_lag1_acf": round(acf1, 4) if acf1 is not None else None,
        "neg_resid_rate": round(neg_rate, 4) if neg_rate is not None else None,
        "large_pos_resid_rate": round(large_pos, 4) if large_pos is not None else None,
        "large_neg_resid_rate": round(large_neg, 4) if large_neg is not None else None,
    }


def main():
    rows = []
    for mk in MARKETS:
        for bb in HOSTS:
            try:
                r = quality_for(mk, bb)
                rows.append(r)
                print(f"[r1b/hq] {mk} x {bb}: "
                      f"S2V_mae={r['s2v_mae']} tr_err={r['transformed_host_error']} "
                      f"neg_res={r['neg_resid_rate']} acf1={r['resid_lag1_acf']}")
            except FileNotFoundError:
                print(f"[r1b/hq] {mk} x {bb}: cache missing (skip)")
    out = HERE / "results" / "host_quality_by_domain.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["market", "host"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[r1b/hq] saved {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
