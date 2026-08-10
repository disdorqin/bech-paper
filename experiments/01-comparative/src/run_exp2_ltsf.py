"""Exp2: LTSF comparison — HCH vs post-processing baselines × 3 datasets × 2 backbones.

Datasets: ETTh1, Weather, Electricity (ECL)
Backbones: Linear, GBDT
Baselines: Base, HCH, Quantile, CRC, PIR (5 methods)
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
from common import load_dataset, build_tabular, build_sequences, four_segment_split
from backbones import make_backbone, needs_seq
from selective_hurdle import HurdleCorrectionHead, build_corrector_features
from metrics import all_metrics, summary_table
from quantile import QuantileCorrection
from crc_impl import CRC

class PIR_Simple:
    name = "PIR"
    def __init__(self, seed=0): self.seed = seed; self.model = None
    def fit(self, Z, yhat, y):
        from sklearn.linear_model import Ridge; resid = y - yhat
        self.model = Ridge(alpha=1.0).fit(Z, resid.mean(axis=1) if resid.ndim > 1 else resid)
    def predict(self, Z, yhat):
        if self.model is None: return yhat
        return yhat + self.model.predict(Z)

class Identity:
    name = "Base"
    def fit(self, Z, yhat, y): pass
    def predict(self, Z, yhat): return yhat

LTSF_DATASETS = ["LAGO_DE", "GEFCOM14P"]  # 用电价公开集代替纯 LTSF（有价格语义）
BB_LIST = ["Linear", "GBDT"]
BASELINES = [Identity(), QuantileCorrection(seed=0), CRC(seed=0), PIR_Simple(seed=0)]
OUT_DIR = ROOT / "experiments" / "01-comparative" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

all_res = {}
for ds_key in LTSF_DATASETS:
    print(f"\n{'='*60}\nDataset: {ds_key}")
    ds = load_dataset(ds_key)
    ts = ds["ts"]
    y_full = ds["price"]
    X, y, names, valid = build_tabular(ds)
    n = len(valid)
    seg = four_segment_split(n)
    s4 = seg["S4"]
    y_true = y[s4]
    all_res[ds_key] = {}

    for bb_name in BB_LIST:
        print(f"  Backbone: {bb_name}")
        seed = hash(f"{ds_key}{bb_name}") % 1000
        bb = make_backbone(bb_name, seed=seed)
        if needs_seq(bb_name):
            sq = build_sequences(ds, valid)
            bb.fit(X[seg["S1"]], y[seg["S1"]], sq[seg["S1"]])
            yhat_full = bb.predict(X, sq)
        else:
            bb.fit(X[seg["S1"]], y[seg["S1"]])
            yhat_full = bb.predict(X)
        y_base = yhat_full[s4]

        oos = seg["S1"][-1] + 1
        hour = ts.dt.hour.to_numpy()
        dayid = (ts - ts.min()).dt.days.to_numpy()
        Z_corr, _ = build_corrector_features(X, names, yhat_full, y, hour[valid], dayid[valid], oos)
        n_corr = len(np.arange(oos, n))
        cut = int(n_corr * 0.75)
        s2_fit = np.arange(oos, oos + cut)
        s4_z = Z_corr[s4]

        # HCH
        hch = HurdleCorrectionHead(neg_thr=0.0, seed=seed)
        hch.fit(Z_corr[s2_fit], yhat_full[s2_fit], y[s2_fit])
        cal_idx = np.concatenate([seg["S3"], np.arange(oos + cut, n)])
        hch.calibrate(Z_corr[cal_idx], yhat_full[cal_idx], y[cal_idx])
        y_hch, hch_diag = hch.apply(s4_z, y_base)

        bb_res = {}
        bb_res["Base"] = all_metrics(y_true, y_base, y_base)
        bb_res["HCH"] = all_metrics(y_true, y_hch, y_base)
        bb_res["HCH"]["fire_rate"] = hch_diag["fire_rate"]

        for bl in BASELINES:
            if bl.name == "Base": continue
            try:
                bl.fit(Z_corr[s2_fit], yhat_full[s2_fit], y[s2_fit])
                y_peer = bl.predict(s4_z, y_base)
                if y_peer.ndim > 1 and y_peer.shape[1] > 1: y_peer = y_peer.mean(axis=1)
                bb_res[bl.name] = all_metrics(y_true, y_peer, y_base)
            except Exception as e:
                bb_res[bl.name] = {"error": str(e)}
        all_res[ds_key][bb_name] = bb_res

out_path = OUT_DIR / "exp2_ltsf.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_res, f, indent=2, default=str, ensure_ascii=False)
print(f"\nSaved: {out_path}")
print(summary_table(all_res))
