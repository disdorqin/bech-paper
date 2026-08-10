"""Final: DE_EPEX + PJM × 5 backbones × all baselines vs HCH."""
import sys, os, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "07-route-e"))
sys.path.insert(0, str(ROOT / "experiments" / "07-route-e" / "peers"))
sys.path.insert(0, str(ROOT / "experiments" / "01-comparative" / "src"))

import numpy as np
from common import load_dataset, build_tabular, build_sequences, four_segment_split
from backbones import make_backbone, needs_seq, BACKBONES
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

class Identity:
    name = "Base"
    def fit(self, Z, yhat, y): pass
    def predict(self, Z, yhat): return yhat

BASELINES = [Identity(), QuantileCorrection(seed=0), VahediStyle(seed=0),
             CRC(seed=0), PIR_Simple(seed=0), SpikeRegularization(seed=0)]
OUT = ROOT / "experiments" / "01-comparative" / "results"
OUT.mkdir(parents=True, exist_ok=True)

for ds_key in ["DE_EPEX", "PJM_2020"]:
    print(f"\n{'='*60}\n{ds_key}")
    ds = load_dataset(ds_key); ts = ds["ts"]
    X, y, names, valid = build_tabular(ds); n = len(valid); seg = four_segment_split(n)
    s4 = seg["S4"]; y_true = y[s4]
    all_res = {}

    for bb_name in BACKBONES:
        if bb_name in ("LSTM","Transformer") and ds_key != "shandong":
            # LSTM/Transformer need sequences
            pass  # proceed anyway
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

        hch = HurdleCorrectionHead(neg_thr=0.0, seed=seed)
        hch.fit(Z_corr[s2_fit], yhat_full[s2_fit], y[s2_fit])
        cal_idx = np.concatenate([seg["S3"], np.arange(oos + cut, n)])
        hch.calibrate(Z_corr[cal_idx], yhat_full[cal_idx], y[cal_idx])
        y_hch, ddiag = hch.apply(s4_z, y_base)

        bb_res = {}
        bb_res["Base"] = all_metrics(y_true, y_base, y_base)
        bb_res["Base"]["mae_on_neg"] = bb_res["Base"].get("mae_on_neg", None)

        m = all_metrics(y_true, y_hch, y_base)
        m["fire_rate"] = ddiag["fire_rate"]; m["lam_neg"] = ddiag["lam_neg"]; m["lam_pos"] = ddiag["lam_pos"]
        bb_res["HCH"] = m

        for bl in BASELINES:
            if bl.name == "Base": continue
            try:
                bl.fit(Z_corr[s2_fit], yhat_full[s2_fit], y[s2_fit])
                yp = bl.predict(s4_z, y_base)
                if yp.ndim > 1 and yp.shape[1] > 1: yp = yp.mean(axis=1)
                bb_res[bl.name] = all_metrics(y_true, yp, y_base)
            except Exception as e:
                bb_res[bl.name] = {"error": str(e)}

        all_res[bb_name] = bb_res
        # Quick print
        h = bb_res["HCH"]; b = bb_res["Base"]
        print(f"  {bb_name:12s} Base MAE={b['mae']:7.2f} HCH MAE={h['mae']:7.2f} NDR={h.get('ndr',0):.1%} EpR={h.get('ep_our_episode_recall',0):.1%}")

    with open(OUT / f"final_{ds_key}_5bb.json", "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, default=str, ensure_ascii=False)
print("\nDone")
