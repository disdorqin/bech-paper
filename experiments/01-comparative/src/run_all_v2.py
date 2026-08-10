"""全量重跑 — 所有10数据集 × 新metrics模块."""
import sys, os, json
ROOT = r"D:\作业\science\solar_leak_price_model"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "07-route-e"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "07-route-e", "peers"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "01-comparative", "src"))

import numpy as np
from common import load_dataset, load_shandong, build_tabular, build_sequences, four_segment_split
from backbones import make_backbone, needs_seq
from selective_hurdle import HurdleCorrectionHead, build_corrector_features
from metrics import all_metrics, summary_row
from quantile import QuantileCorrection
from vahedi_style import VahediStyle
from crc_impl import CRC
from spike_reg import SpikeRegularization

class PIR_Simple:
    name = "PIR"
    def __init__(self, seed=0): self.seed = seed; self.model = None
    def fit(self, Z, yhat, y):
        from sklearn.linear_model import Ridge
        r = y - yhat; self.model = Ridge(alpha=1.0).fit(Z, r.mean(axis=1) if r.ndim > 1 else r)
    def predict(self, Z, yhat): return yhat + self.model.predict(Z) if self.model is not None else yhat

BASELINES = [
    ("Base", None), ("HCH", None),
    ("Quantile", lambda s: QuantileCorrection(seed=s)),
    ("Vahedi", lambda s: VahediStyle(seed=s)),
    ("CRC", lambda s: CRC(seed=s)),
    ("PIR", lambda s: PIR_Simple(seed=s)),
    ("SpikeReg", lambda s: SpikeRegularization(seed=s)),
]

CONFIG = [
    ("shandong", "日前电价", None, ["Linear","MLP","LSTM","Transformer","GBDT"]),
    ("NEM_SA1", None, None, ["Linear","GBDT"]),
    ("DE_EPEX", None, None, ["Linear","GBDT"]),
    ("PJM_2020", None, None, ["Linear","GBDT"]),
    ("EPEX_FR", None, None, ["MLP","LSTM"]),
    ("EPEX_BE", None, None, ["Transformer","GBDT"]),
    ("EPEX_NL", None, None, ["Linear","MLP"]),
    ("NORD_FI", None, None, ["LSTM","Transformer"]),
    ("NORD_NO", None, None, ["Linear","GBDT"]),
    ("NORD_SE3", None, None, ["MLP","Transformer"]),
    ("NORD_DK1", None, None, ["LSTM","GBDT"]),
]

RESULTS = ROOT + "/experiments/01-comparative/results"

for ds_key, sd_col, _extra, bb_list in CONFIG:
    print(f"\n{'='*60}\n{ds_key} × {bb_list}")
    sys.stdout.flush()

    # Load data
    if ds_key == "shandong":
        ds = load_shandong(sd_col)
    else:
        ds = load_dataset(ds_key)
    ts = ds["ts"]
    X, y, names, valid = build_tabular(ds)
    n = len(valid)
    seg = four_segment_split(n)
    s4 = seg["S4"]
    y_true = y[s4]

    table = {}
    for bb_name in bb_list:
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

        hch = HurdleCorrectionHead(neg_thr=0.0, seed=seed)
        hch.fit(Z_corr[s2_fit], yhat_full[s2_fit], y[s2_fit])
        cal_idx = np.concatenate([seg["S3"], np.arange(oos + cut, n)])
        hch.calibrate(Z_corr[cal_idx], yhat_full[cal_idx], y[cal_idx])
        y_hch, ddiag = hch.apply(s4_z, y_base)

        row = {}
        row["Base"] = all_metrics(y_true, y_base, y_base)
        m = all_metrics(y_true, y_hch, y_base)
        m["fire_rate"] = ddiag["fire_rate"]
        row["HCH"] = m

        for name, factory in BASELINES:
            if name in ("Base", "HCH"):
                continue
            try:
                bl = factory(seed)
                bl.fit(Z_corr[s2_fit], yhat_full[s2_fit], y[s2_fit])
                yp = bl.predict(s4_z, y_base)
                if yp.ndim > 1 and yp.shape[1] > 1:
                    yp = yp.mean(axis=1)
                row[name] = all_metrics(y_true, yp, y_base)
            except Exception as e:
                row[name] = {"error": str(e)}

        table[bb_name] = row

        # Print summary
        print(f"\n  {bb_name}:")
        hdr = f"  {'Method':14s} {'MAE':>7s} {'RMSE':>7s} {'WAPE':>6s} {'ΔMAE%':>7s} {'Recall':>7s} {'F1':>7s} {'FPR':>7s} {'MAEext':>7s} {'Fire':>6s}"
        print(hdr)
        print("  " + "-" * len(hdr))
        for nm in ["Base", "HCH", "Quantile", "Vahedi", "CRC", "PIR", "SpikeReg"]:
            if nm in row and "mae" in row[nm]:
                print("  " + summary_row(nm, row[nm], include_fire=(nm == "HCH")))

    with open(f"{RESULTS}/v2_{ds_key}.json", "w") as f:
        json.dump(table, f, indent=2, default=str)

print("\n=== All done ===")
