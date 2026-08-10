"""Exp1: Shandong full comparison — HCH vs all 6 baselines × 5 backbones.

Output: results/exp1_shandong.json, docs/exp1_shandong_results.md
"""
from __future__ import annotations
import sys, os, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "07-route-e"))
sys.path.insert(0, str(ROOT / "experiments" / "07-route-e" / "peers"))
sys.path.insert(0, str(ROOT / "experiments" / "01-comparative" / "src"))

import numpy as np
import pandas as pd
from backbones import BACKBONES, make_backbone, needs_seq
from common import load_shandong, build_tabular, build_sequences, four_segment_split, assert_no_leakage, episode_metrics
from selective_hurdle import HurdleCorrectionHead, build_corrector_features
from metrics import all_metrics, summary_table

# Import baselines
from quantile import QuantileCorrection
from vahedi_style import VahediStyle
from crc_impl import CRC
from spike_reg import SpikeRegularization

# PIR-style (simplified: failure ID + Ridge correction)
class PIR_Simple:
    name = "PIR"
    def __init__(self, seed=0): self.seed = seed; self.model = None
    def fit(self, Z, yhat, y):
        from sklearn.linear_model import Ridge
        resid = y - yhat
        self.model = Ridge(alpha=1.0).fit(Z, resid.mean(axis=1) if resid.ndim > 1 else resid)
    def predict(self, Z, yhat):
        if self.model is None: return yhat
        return yhat + self.model.predict(Z)

# Identity
class Identity:
    name = "Base"
    def fit(self, Z, yhat, y): pass
    def predict(self, Z, yhat): return yhat

# ---------- setup ----------
BB_LIST = list(BACKBONES)  # 5 backbones
BASELINES = [
    Identity(),
    QuantileCorrection(seed=0),
    VahediStyle(seed=0),
    CRC(seed=0),
    PIR_Simple(seed=0),
    SpikeRegularization(seed=0),
]
OUT_DIR = ROOT / "experiments" / "01-comparative" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- run ----------
print("Loading Shandong...")
ds = load_shandong("日前电价")
ts = ds["ts"]
y_full = ds["price"]
X, y, names, valid = build_tabular(ds)
n = len(valid)
seg = four_segment_split(n)

all_res = {}
for bb_name in BB_LIST:
    print(f"\n{'='*60}\nBackbone: {bb_name}")
    t0 = time.time()
    seed = hash(bb_name) % 1000

    # Train backbone on S1
    bb = make_backbone(bb_name, seed=seed)
    if needs_seq(bb_name):
        seq_full = build_sequences(ds, valid)
        bb.fit(X[seg["S1"]], y[seg["S1"]], seq_full[seg["S1"]])
        yhat_full = bb.predict(X, seq_full)
    else:
        bb.fit(X[seg["S1"]], y[seg["S1"]])
        yhat_full = bb.predict(X)

    s4 = seg["S4"]
    y_true = y[s4]
    y_base = yhat_full[s4]

    # Build corrector features Z (for S2 onwards)
    oos = seg["S1"][-1] + 1
    hour = ts.dt.hour.to_numpy()
    dayid = (ts - ts.min()).dt.days.to_numpy()
    Z_corr, _ = build_corrector_features(
        X, names, yhat_full, y, hour[valid], dayid[valid], oos,
    )
    n_corr = len(np.arange(oos, n))
    cut = int(n_corr * 0.75)
    s2_fit = np.arange(oos, oos + cut)
    s4_z = Z_corr[seg["S4"]]

    # HCH
    hch = HurdleCorrectionHead(neg_thr=0.0, seed=seed)
    hch.fit(Z_corr[s2_fit], yhat_full[s2_fit], y[s2_fit])
    cal_idx = np.concatenate([seg["S3"], np.arange(oos + cut, n)])
    hch.calibrate(Z_corr[cal_idx], yhat_full[cal_idx], y[cal_idx])
    y_hch, hch_diag = hch.apply(s4_z, y_base)

    bb_results = {}
    # Base
    m = all_metrics(y_true, y_base, y_base)
    m.update({"n_train_S1": len(seg["S1"]), "n_test_S4": len(s4), "duration_s": round(time.time()-t0,1)})
    bb_results["Base"] = m

    # HCH
    m = all_metrics(y_true, y_hch, y_base)
    m.update({"fire_rate": hch_diag["fire_rate"], "lam_neg": hch_diag["lam_neg"],
              "lam_pos": hch_diag["lam_pos"]})
    bb_results["HCH"] = m

    # Other baselines
    for bl in BASELINES:
        if bl.name == "Base": continue
        try:
            bl.fit(Z_corr[s2_fit], yhat_full[s2_fit], y[s2_fit])
            y_peer = bl.predict(s4_z, y_base)
            if y_peer.ndim > 1 and y_peer.shape[1] > 1:
                y_peer = y_peer.mean(axis=1)
            m = all_metrics(y_true, y_peer, y_base)
            bb_results[bl.name] = m
            print(f"  {bl.name:20s} MAE={m['mae']:.2f} ep_recall={m.get('ep_our_episode_recall',0):.1%}")
        except Exception as e:
            bb_results[bl.name] = {"error": str(e)}
            print(f"  {bl.name:20s} FAILED: {e}")

    all_res[bb_name] = bb_results

# Save
out_path = OUT_DIR / "exp1_shandong.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_res, f, indent=2, default=str, ensure_ascii=False)

# Pretty print
print(f"\n{'='*60}\nSUMMARY")
print(summary_table({"Shandong": all_res}))
print(f"\nSaved: {out_path}")
