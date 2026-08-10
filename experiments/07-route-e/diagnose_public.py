"""Diagnostic: why does HCH not fire on public datasets?"""
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
from common import load_dataset, build_tabular, four_segment_split
from backbones import make_backbone
from selective_hurdle import HurdleCorrectionHead, build_corrector_features

for ds_key in ["LAGO_DE", "LAGO_BE", "LAGO_FR", "GEFCOM14P"]:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_key}")
    ds = load_dataset(ds_key)
    ts = ds["ts"]
    X, y, names, valid = build_tabular(ds)
    seg = four_segment_split(len(valid))
    s2_y = y[seg["S2"]]
    s4_y = y[seg["S4"]]
    spike_thr = float(np.quantile(s2_y, 0.99))

    bb = make_backbone("Linear", seed=0)
    bb.fit(X[seg["S1"]], y[seg["S1"]])
    yhat = bb.predict(X)

    print(f"S2 p99 spike_thr = {spike_thr:.1f}")
    print(f"S4 neg%={(s4_y < 0).mean():.2%}  spike%={(s4_y > spike_thr).mean():.1%}")

    oos = seg["S1"][-1] + 1
    hour = ts.dt.hour.to_numpy()
    dayid = (ts - ts.min()).dt.days.to_numpy()
    n_valid = len(valid)
    corr_full = np.arange(oos, n_valid)
    Z_corr, z_names = build_corrector_features(
        X, names, yhat, y, hour[valid], dayid[valid], oos,
    )

    n_corr = len(corr_full)
    cut = int(n_corr * 0.75)
    s2_fit = corr_full[:cut]
    hch = HurdleCorrectionHead(neg_thr=0.0, seed=0)
    hch.fit(Z_corr[s2_fit], yhat[s2_fit], y[s2_fit])

    Z_s4 = Z_corr[seg["S4"]]
    for br in ["neg", "pos"]:
        p = hch._prob(Z_s4, br)
        n_trigger = int((p > hch.tau[br]).sum())
        print(f"  {br}: max P={p.max():.4f}  mean P={p.mean():.4f}  n>tau={n_trigger}  tau={hch.tau[br]}")
