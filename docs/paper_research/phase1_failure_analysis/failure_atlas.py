"""Phase-2 failure atlas harness (ready for 2.5 Shandong 96-min model results).

INPUT CONTRACT (when real data arrives):
  A single long/wide CSV with at least:
    - `timestamp`  : datetime (96-min granularity for Shandong 2.5 project)
    - `actual`     : realized price (yuan/MWh, KEEP negatives, no floor clip)
    - one column per base model prediction, e.g. `pred_lightgbm`, `pred_timesfm`, ...
  Pass via --data path.csv. Output: per-model failure metrics (tail RMSE,
  negative-price miss, spike miss, by-horizon) -> printed + written CSV.

Run `--demo` to self-test on synthetic data (no real data needed yet).
"""
import argparse, sys
import numpy as np
import pandas as pd

NEG_TH = 0.0        # negative price threshold
DEEP_NEG_TH = -50.0
SPIKE_TH = 500.0

def build_atlas(preds: pd.DataFrame, actual: pd.Series, hours=None):
    """preds: DataFrame index=ts, cols=models. actual: Series aligned. hours: optional hour array."""
    rows = []
    a = actual.values.astype(float)
    neg_mask = a < NEG_TH
    deep_mask = a < DEEP_NEG_TH
    spike_mask = a > SPIKE_TH
    # extreme subsets by actual percentile
    lo_q = np.quantile(a, 0.05); hi_q = np.quantile(a, 0.95)
    lo_mask = a <= lo_q; hi_mask = a >= hi_q
    for m in preds.columns:
        p = preds[m].values.astype(float)
        err = p - a
        rmse = float(np.sqrt(np.mean(err**2)))
        # tail RMSE
        def rmse_on(mask):
            if mask.sum() == 0: return float("nan")
            return float(np.sqrt(np.mean(err[mask]**2)))
        # negative-price miss: actual<0 but pred>=0 (wrong sign) OR large under-prediction
        neg_miss = float(np.mean((a < NEG_TH) & (p >= NEG_TH))) if neg_mask.sum() else float("nan")
        spike_miss = float(np.mean((a > SPIKE_TH) & (p <= SPIKE_TH))) if spike_mask.sum() else float("nan")
        rows.append({
            "model": m, "RMSE": round(rmse, 3),
            "tail_RMSE_low5%": round(rmse_on(lo_mask), 3),
            "tail_RMSE_high5%": round(rmse_on(hi_mask), 3),
            "neg_price_miss_rate": round(neg_miss, 4),
            "spike_miss_rate": round(spike_miss, 4),
            "mean_err": round(float(np.mean(err)), 3),
        })
        # by-horizon (if available)
        if hours is not None:
            for h in sorted(set(hours)):
                hm = np.array(hours) == h
                if hm.sum() == 0: continue
                he = err[hm]
                rows.append({"model": f"{m}@h{h:02d}", "RMSE": round(float(np.sqrt(np.mean(he**2))),3),
                             "tail_RMSE_low5%": "", "tail_RMSE_high5%": "",
                             "neg_price_miss_rate": "", "spike_miss_rate": "", "mean_err": round(float(np.mean(he)),3)})
    return pd.DataFrame(rows)

def demo():
    rng = np.random.default_rng(0)
    n = 4000
    ts = pd.date_range("2026-01-01", periods=n, freq="15min")
    base = 250 + 80*np.sin(np.arange(n)/40.0)
    a = base.copy()
    # inject negatives (solar hours) + spikes
    neg_i = ((ts.hour >= 9) & (ts.hour <= 16)) & (rng.random(n) < 0.05)
    a[neg_i] = rng.uniform(-120, -10, neg_i.sum())
    sp_i = (rng.random(n) < 0.02)
    a[sp_i] = rng.uniform(550, 1200, sp_i.sum())
    # model A: naive (under-corrects extremes)
    pa = a + rng.normal(0, 15, n)
    # model B: partially fixes extremes (shrinks extreme error)
    pb = a + rng.normal(0, 15, n)
    pb[neg_i] = a[neg_i] + rng.normal(0, 8, neg_i.sum())   # better on negatives
    preds = pd.DataFrame({"model_naive": pa, "model_partial": pb}, index=ts)
    df = build_atlas(preds, pd.Series(a, index=ts), hours=ts.hour.values)
    print("=== DEMO failure atlas ===")
    print(df.to_string())
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="wide CSV: timestamp,actual,pred_*")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", help="output csv path")
    args = ap.parse_args()
    if args.demo:
        df = demo()
    elif args.data:
        d = pd.read_csv(args.data, parse_dates=["timestamp"])
        pred_cols = [c for c in d.columns if c.startswith("pred_")]
        preds = d[pred_cols].rename(columns=lambda c: c[len("pred_"):]); preds.index = d["timestamp"]
        actual = d.set_index("timestamp")["actual"]
        hours = d["timestamp"].dt.hour.values if "timestamp" in d else None
        df = build_atlas(preds, actual, hours=hours)
        print(df.to_string())
        if args.out:
            df.to_csv(args.out, index=False, encoding="utf-8-sig"); print("WROTE", args.out)
    else:
        print("use --demo or --data path.csv"); sys.exit(1)

if __name__ == "__main__":
    main()
