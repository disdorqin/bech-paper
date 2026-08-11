"""HCH v2 data audit — check time resolution, DST, field availability, exog leakage.

Produces: experiments/08-hch-v2/results/data_audit.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import DATASETS, load_dataset, load_shandong

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def audit_dst(ts: pd.Series):
    dates = ts.dt.date
    counts = dates.value_counts().sort_index()
    abnormal = counts[(counts != 24)]
    total_days = len(counts)
    return {
        "n_days": total_days,
        "n_23h": int((counts == 23).sum()),
        "n_25h": int((counts == 25).sum()),
        "n_abnormal": int(len(abnormal)),
        "abnormal_pct": round(len(abnormal) / total_days * 100, 2) if total_days else 0,
        "abnormal_dates": ";".join(str(d) for d in abnormal.index[:10]),
    }


def audit_dataset(key: str, ds: dict) -> dict:
    ts = ds["ts"]
    price = ds["price"]
    exog_fc = ds["exog_fc"]
    exog_act = ds["exog_act"]

    n = len(price)
    nan_price = int(np.isnan(price).sum())
    neg_n = int((price < 0).sum())
    neg_pct = round(neg_n / n * 100, 2) if n else 0

    time_delta = ts.diff().dropna()
    unique_deltas = time_delta.value_counts()
    primary_res = str(unique_deltas.index[0]) if len(unique_deltas) > 0 else "unknown"

    row = {
        "dataset": key,
        "n_hours": n,
        "time_start": str(ts.iloc[0]),
        "time_end": str(ts.iloc[-1]),
        "primary_resolution": primary_res,
        "unique_timedeltas": len(unique_deltas),
        "nan_price": nan_price,
        "neg_n": neg_n,
        "neg_pct": neg_pct,
        "n_exog_fc": len(exog_fc.columns) if hasattr(exog_fc, "columns") else 0,
        "n_exog_act": len(exog_act.columns) if hasattr(exog_act, "columns") else 0,
        "exog_fc_cols": ";".join(str(c) for c in (exog_fc.columns if hasattr(exog_fc, "columns") else [])),
        "exog_act_cols": ";".join(str(c) for c in (exog_act.columns if hasattr(exog_act, "columns") else [])),
    }

    dst = audit_dst(ts)
    row.update({f"dst_{k}": v for k, v in dst.items()})

    meta = ds.get("meta", {})
    row.update({
        "currency": meta.get("currency", "?"),
        "tier": meta.get("tier", "?"),
    })

    return row


def main():
    rows = []

    for key in DATASETS:
        try:
            ds = load_dataset(key)
            rows.append(audit_dataset(key, ds))
            print(f"  {key}: {rows[-1]['n_hours']}h, neg={rows[-1]['neg_pct']}%, "
                  f"DST={rows[-1]['dst_n_abnormal']}d, exog_fc={rows[-1]['n_exog_fc']}")
        except Exception as e:
            print(f"  {key}: FAILED — {e}")

    try:
        ds_sd = load_shandong(price_col="日前电价", encoding="gbk")
        rows.append(audit_dataset("shandong_DA", ds_sd))
        print(f"  shandong_DA: {rows[-1]['n_hours']}h, neg={rows[-1]['neg_pct']}%, "
              f"DST={rows[-1]['dst_n_abnormal']}d, exog_fc={rows[-1]['n_exog_fc']}")

        ds_rt = load_shandong(price_col="实时电价", encoding="gbk")
        rows.append(audit_dataset("shandong_RT", ds_rt))
        print(f"  shandong_RT: {rows[-1]['n_hours']}h, neg={rows[-1]['neg_pct']}%, "
              f"DST={rows[-1]['dst_n_abnormal']}d, exog_fc={rows[-1]['n_exog_fc']}")
    except Exception as e:
        print(f"  shandong: FAILED — {e}")

    out = OUT / "data_audit.csv"
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out} ({len(rows)} datasets)")
    else:
        print("\nNo data audited")


if __name__ == "__main__":
    main()
