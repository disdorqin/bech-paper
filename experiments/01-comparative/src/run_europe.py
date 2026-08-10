"""Europe experiment: 7 new markets × 2 rotating backbones × 7 methods."""
import sys, os, json
ROOT = r"D:\作业\science\solar_leak_price_model"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "07-route-e"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "07-route-e", "peers"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "01-comparative", "src"))

import numpy as np
from common import load_dataset, build_tabular, build_sequences, four_segment_split
from backbones import make_backbone, needs_seq
from selective_hurdle import HurdleCorrectionHead, build_corrector_features
from metrics import all_metrics
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
    ("Base", None),
    ("HCH", None),
    ("Quantile", lambda s: QuantileCorrection(seed=s)),
    ("Vahedi", lambda s: VahediStyle(seed=s)),
    ("CRC", lambda s: CRC(seed=s)),
    ("PIR", lambda s: PIR_Simple(seed=s)),
    ("SpikeReg", lambda s: SpikeRegularization(seed=s)),
]

# Rotating backbones: different pair per dataset
DATASETS = {
    "EPEX_FR":  ("MLP", "LSTM"),
    "EPEX_BE":  ("Transformer", "GBDT"),
    "EPEX_NL":  ("Linear", "MLP"),
    "NORD_FI":  ("LSTM", "Transformer"),
    "NORD_NO":  ("Linear", "GBDT"),
    "NORD_SE3": ("MLP", "Transformer"),
    "NORD_DK1": ("LSTM", "GBDT"),
}

OUT = os.path.join(ROOT, "experiments", "01-comparative", "results")
os.makedirs(OUT, exist_ok=True)

all_results = {}
for ds_key, (bb1, bb2) in DATASETS.items():
    print(f"\n{'='*60}\n{ds_key} × {bb1} + {bb2}")
    ds = load_dataset(ds_key); ts = ds["ts"]
    X, y, names, valid = build_tabular(ds); n = len(valid); seg = four_segment_split(n)
    s4 = seg["S4"]; y_true = y[s4]
    all_results[ds_key] = {}

    for bb_name in [bb1, bb2]:
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
        hour = ts.dt.hour.to_numpy(); dayid = (ts - ts.min()).dt.days.to_numpy()
        Z_corr, _ = build_corrector_features(X, names, yhat_full, y, hour[valid], dayid[valid], oos)
        n_corr = len(np.arange(oos, n)); cut = int(n_corr * 0.75)
        s2_fit = np.arange(oos, oos + cut); s4_z = Z_corr[s4]

        # HCH
        hch = HurdleCorrectionHead(neg_thr=0.0, seed=seed)
        hch.fit(Z_corr[s2_fit], yhat_full[s2_fit], y[s2_fit])
        cal_idx = np.concatenate([seg["S3"], np.arange(oos + cut, n)])
        hch.calibrate(Z_corr[cal_idx], yhat_full[cal_idx], y[cal_idx])
        y_hch, ddiag = hch.apply(s4_z, y_base)

        bb_res = {}
        bb_res["Base"] = all_metrics(y_true, y_base, y_base)

        m = all_metrics(y_true, y_hch, y_base)
        m["fire"] = ddiag["fire_rate"]; m["lam_neg"] = ddiag["lam_neg"]; m["lam_pos"] = ddiag["lam_pos"]
        bb_res["HCH"] = m

        for name, factory in BASELINES:
            if name in ("Base", "HCH"): continue
            try:
                bl = factory(seed)
                bl.fit(Z_corr[s2_fit], yhat_full[s2_fit], y[s2_fit])
                yp = bl.predict(s4_z, y_base)
                if yp.ndim > 1 and yp.shape[1] > 1: yp = yp.mean(axis=1)
                bb_res[name] = all_metrics(y_true, yp, y_base)
            except Exception as e:
                bb_res[name] = {"error": str(e)}

        all_results[ds_key][bb_name] = bb_res
        h = bb_res["HCH"]; b = bb_res["Base"]
        er = h.get("ep_our_episode_recall", 0)
        print(f"  {bb_name:12s} Base={b['mae']:7.2f} HCH={h['mae']:7.2f} NDR={h.get('ndr',0):.1%} EpR={er:.1%} Fire={h.get('fire',0):.1%}")

with open(os.path.join(OUT, "europe_7mkts.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print("\nDone: europe_7mkts.json")
