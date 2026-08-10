import sys, os, importlib.util

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/07-route-e")

import numpy as np
from common import load_dataset, build_tabular, four_segment_split, evaluate, episode_metrics
from backbones import make_backbone
from selective_hurdle import HurdleCorrectionHead, build_corrector_features

spec = importlib.util.spec_from_file_location(
    "vahedi_style", "experiments/07-route-e/peers/vahedi_style.py"
)
vahedi_style = importlib.util.module_from_spec(spec)
sys.modules["vahedi_style"] = vahedi_style
spec.loader.exec_module(vahedi_style)
VahediStyle = vahedi_style.VahediStyle

ds = load_dataset("NEM_SA1")
ts = ds["ts"]
X, y, names, valid = build_tabular(ds)
seg = four_segment_split(len(valid))
s4 = seg["S4"]

bb = make_backbone("Linear", seed=0)
bb.fit(X[seg["S1"]], y[seg["S1"]])
yhat = bb.predict(X)

oos = seg["S1"][-1] + 1
n = len(valid)
hour = ts.dt.hour.to_numpy()
dayid = (ts - ts.min()).dt.days.to_numpy()
Z_corr, _ = build_corrector_features(X, names, yhat, y, hour[valid], dayid[valid], oos)
corr_full = np.arange(oos, n)
cut = int(len(corr_full) * 0.75)
s2_fit = corr_full[:cut]

vah = VahediStyle(neg_thr=0.0, seed=0)
vah.fit(Z_corr[s2_fit], yhat[s2_fit], y[s2_fit])
vah_pred = vah.predict(Z_corr[s4], yhat[s4])

base_eval = evaluate(y[s4], yhat[s4], None, neg_thr=0.0)
vah_eval = evaluate(y[s4], vah_pred, None, neg_thr=0.0)
base_ep = episode_metrics(y[s4], yhat[s4], yhat[s4])
vah_ep = episode_metrics(y[s4], vah_pred, yhat[s4])

row = lambda nm, ev, ep: (
    f"{nm:15s} MAE={ev['mae']:7.1f}  "
    f"neg_miss={ev.get('neg_miss_rate', 0):5.1%}  "
    f"ep_recall={ep['our_episode_recall'] if 'our_episode_recall' in ep else ep['base_episode_recall']:5.1%}  "
    f"complete_miss={ep['our_complete_miss'] if 'our_complete_miss' in ep else ep['base_complete_miss']:5.1%}"
)

print(row("Base", base_eval, base_ep))
print(row("Vahedi-style", vah_eval, vah_ep))
print("\nPaper: 98% neg-event recall (NEM-SA 5min 2024, LightGBM direct)")
print(f"Our Vahedi-style: {vah_ep['our_episode_recall']:.1%} episode recall (hourly SA1 2024, Linear base + classification-regression)")
