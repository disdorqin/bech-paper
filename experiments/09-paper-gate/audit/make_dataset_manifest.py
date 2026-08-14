"""WP-0 asset baseline — generate 02_DATASET_MANIFEST.csv / 03_DOMESTIC_DATA_AUDIT.csv / 04_HOST_MANIFEST.csv.

Protocol: hch_v2_paper_benchmark_gate §5.3 (domestic audit fields) + §4 (host provenance).
Dataset loading reuses src/common.py load_dataset / load_shandong (P0-B: single loader).
User decision 2026-08-14: Shandong headline target = the 24h-hourly file only
  (shandong_pmos_hourly.csv, DA=日前电价 / RT=实时电价). The 96-point xlsx is EXCLUDED.
Other provinces (NX/GS/SX/QH) enter the secondary domestic table only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # repo root
DATA = ROOT / "data"                          # spec["path"] already starts with "raw/"
PROV = ROOT / "data" / "raw" / "provinces"
sys.path.insert(0, str(ROOT / "src"))

from common import DATASETS, load_dataset, load_shandong  # noqa: E402

CACHE_ROOT = ROOT / "experiments" / "08-hch-v2" / "results" / "cache"
HOSTS = ["Linear", "MLP", "LSTM", "PatchTST"]

# order as in protocol §5.1 (headline) then §5.2 (extended)
HEADLINE = ["LAGO_DE", "LAGO_BE", "LAGO_FR", "LAGO_PJM", "LAGO_NP",
            "NEM_SA1", "GEFCOM14P", "NORD_DK1"]
EXTENDED = ["DE_EPEX", "PJM_2020", "EPEX_FR", "EPEX_BE", "EPEX_NL",
            "NORD_FI", "NORD_NO", "NORD_SE3"]

PROVINCES = {  # display name -> file (24h hourly panel files)
    "ningxia": "宁夏24h电价数据集.xlsx",
    "gansu": "甘肃24h电价数据集.xlsx",
    "shaanxi": "陕西24h电价数据集(1).xlsx",
    "qinghai": "青海24h电价数据集.xlsx",
}


def freq_h(ts: pd.Series) -> float:
    d = pd.Series(ts).sort_values().diff().dropna()
    if len(d) == 0:
        return np.nan
    return float(np.median(d.dt.total_seconds() / 3600.0))


def is_cached(ds_key: str) -> bool:
    return all((CACHE_ROOT / ds_key / bb / "pred.npy").exists() for bb in HOSTS)


def foreign_row(key: str) -> dict:
    spec = DATASETS[key]
    raw_path = DATA / spec["path"]
    # missingness on the raw file before load_dataset drops rows
    df = pd.read_csv(raw_path)
    if key.startswith("LAGO_"):
        first = df.columns[0]
        df = df.rename(columns={first: "timestamp"})
        pc = [c for c in df.columns if "price" in c.lower()][0]
    elif key == "GEFCOM14P":
        pc = "price"
    elif key.startswith("NEM_"):
        pc = "price"
    else:
        pc = "price"
    n_raw = len(df)
    missing = float(pd.to_numeric(df[pc], errors="coerce").isna().mean())

    ds = load_dataset(key)
    ts, price = pd.Series(ds["ts"]), np.asarray(ds["price"])
    neg = float((price < 0).mean())
    return {
        "dataset": key,
        "panel": "HEADLINE" if key in HEADLINE else "EXTENDED",
        "currency": spec.get("currency"),
        "tier": spec.get("tier"),
        "freq_h": round(freq_h(ts), 3),
        "start": str(ts.min().date()),
        "end": str(ts.max().date()),
        "n_hours": int(len(price)),
        "info_cutoff": str(ts.max().date()),
        "missing_rate": round(missing, 4),
        "neg_price_rate": round(neg, 4),
        "valid_days": int(len(price) / 24),
        "da_rt": "DA" if key not in ("NORD_",) else "DA/RT",
        "host_cached": is_cached(key),
        "note": spec.get("note", ""),
    }


def province_row(pname: str, path: str) -> dict:
    """Read a province 24h panel xlsx, audit semantics for secondary table."""
    p = PROV / path
    df = pd.read_excel(p)
    df.columns = [str(c).strip() for c in df.columns]
    ts_col = next((c for c in df.columns
                   if any(k in c for k in ("时刻", "时间", "time", "Time"))), None)
    price_cols = [c for c in df.columns if "电价" in c]
    out = {"dataset": f"CN_{pname}", "file": path,
           "columns": ";".join(df.columns[:6]),
           "freq_h": np.nan, "start": "", "end": "", "n_hours": int(len(df)),
           "info_cutoff": "", "has_da": any("日前" in c for c in price_cols),
           "has_rt": any("实时" in c for c in price_cols),
           "missing_rate": np.nan, "neg_price_rate": np.nan,
           "headline_eligible": False, "note": ""}
    if ts_col is None:
        out["note"] = "no time column located"
        return out
    ts = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)
    out.update(freq_h=round(freq_h(pd.Series(ts)), 3),
               start=str(ts.min().date()), end=str(ts.max().date()),
               n_hours=int(len(df)), info_cutoff=str(ts.max().date()))
    pc = next((c for c in price_cols if "实时" in c), None) or \
        (price_cols[0] if price_cols else None)
    if pc:
        pr = pd.to_numeric(df[pc], errors="coerce")
        out["missing_rate"] = round(float(pr.isna().mean()), 4)
        out["neg_price_rate"] = round(float((pr < 0).mean()), 4)
    out["headline_eligible"] = bool(out["has_da"] and out["neg_price_rate"]
                                    and out["neg_price_rate"] > 0.01)
    out["note"] = ("headline candidate" if out["headline_eligible"] else
                   "secondary only (no neg-price / semantics unclear)")
    return out


def shandong_rows() -> list[dict]:
    """Shandong hourly 24pt: DA (日前电价) + RT (实时电价) — user-decided headline targets."""
    base = {"dataset": "shandong", "panel": "DOMESTIC", "currency": "CNY",
            "tier": "L1", "freq_h": 1.0, "da_rt": "DA/RT", "note": "Shandong spot, 24h hourly",
            "host_cached": is_cached("shandong_DA") and is_cached("shandong_RT")}
    rows = []
    for target, pc in (("shandong_DA", "日前电价"), ("shandong_RT", "实时电价")):
        ds = load_shandong(price_col=pc, encoding="gbk")
        ts, price = pd.Series(ds["ts"]), np.asarray(ds["price"])
        rows.append({**base, "dataset": target, "target": "DA" if "DA" in target else "RT",
                    "start": str(ts.min().date()), "end": str(ts.max().date()),
                    "n_hours": int(len(price)), "info_cutoff": str(ts.max().date()),
                    "neg_price_rate": round(float((price < 0).mean()), 4),
                    "valid_days": int(len(price) / 24)})
    return rows


def main():
    out_dir = HERE / ".."
    # ---- 02 foreign + domestic manifest
    frows = [foreign_row(k) for k in HEADLINE + EXTENDED]
    frows += shandong_rows()
    cols = ["dataset", "panel", "currency", "tier", "freq_h", "start", "end",
            "n_hours", "info_cutoff", "missing_rate", "neg_price_rate",
            "valid_days", "da_rt", "host_cached", "note"]
    df_manifest = pd.DataFrame(frows)[cols]
    df_manifest.to_csv(out_dir / "02_DATASET_MANIFEST.csv", index=False, encoding="utf-8-sig")

    # ---- 03 domestic audit (Shandong DA/RT mandatory headline candidates, §5.3)
    sd = {}
    for r in shandong_rows():
        sd[r["dataset"]] = r
    prow = [province_row(pn, p) for pn, p in PROVINCES.items()]
    prow += [{"dataset": "shandong_pmos_96_full_v2", "file": "shandong_pmos_96_full_v2.xlsx",
              "columns": "(not read)", "freq_h": np.nan, "start": "", "end": "",
              "n_hours": np.nan, "info_cutoff": "", "has_da": np.nan, "has_rt": np.nan,
              "missing_rate": np.nan, "neg_price_rate": np.nan,
              "headline_eligible": False,
              "note": "EXCLUDED by user decision 2026-08-14 (only the 24pt hourly file is tested)"}]
    prow += [
        {"dataset": "shandong_DA", "file": "shandong_pmos_hourly.csv",
         "columns": "时刻;日前电价;实时电价;exog", "freq_h": 1.0,
         "start": sd["shandong_DA"]["start"], "end": sd["shandong_DA"]["end"],
         "n_hours": sd["shandong_DA"]["n_hours"],
         "info_cutoff": sd["shandong_DA"]["info_cutoff"],
         "has_da": True, "has_rt": True,
         "missing_rate": 0.0, "neg_price_rate": sd["shandong_DA"]["neg_price_rate"],
         "headline_eligible": True,
         "note": "MANDATORY headline target (protocol §5.3), DA 11.1% neg"},
        {"dataset": "shandong_RT", "file": "shandong_pmos_hourly.csv",
         "columns": "时刻;日前电价;实时电价;exog", "freq_h": 1.0,
         "start": sd["shandong_RT"]["start"], "end": sd["shandong_RT"]["end"],
         "n_hours": sd["shandong_RT"]["n_hours"],
         "info_cutoff": sd["shandong_RT"]["info_cutoff"],
         "has_da": True, "has_rt": True,
         "missing_rate": 0.0, "neg_price_rate": sd["shandong_RT"]["neg_price_rate"],
         "headline_eligible": True,
         "note": "MANDATORY headline target (protocol §5.3), RT 13.4% neg"},
    ]
    pcols = ["dataset", "file", "columns", "freq_h", "start", "end", "n_hours",
             "info_cutoff", "has_da", "has_rt", "missing_rate", "neg_price_rate",
             "headline_eligible", "note"]
    df_dom = pd.DataFrame(prow)[pcols]
    df_dom.to_csv(out_dir / "03_DOMESTIC_DATA_AUDIT.csv", index=False, encoding="utf-8-sig")

    # ---- 04 host manifest
    host_rows = []
    for bb in HOSTS:
        cached_ds = [k for k in HEADLINE + EXTENDED + ["shandong_DA", "shandong_RT"]
                     if (CACHE_ROOT / k / bb / "pred.npy").exists()]
        host_rows.append({"host": bb, "family": {"Linear": "Ridge-style", "MLP": "DNN",
                                                 "LSTM": "SeqRNN", "PatchTST": "patch-transformer"}[bb],
                          "n_cached_datasets": len(cached_ds), "cached_datasets": "|".join(cached_ds)})
    pd.DataFrame(host_rows).to_csv(out_dir / "04_HOST_MANIFEST.csv", index=False, encoding="utf-8-sig")

    print(f"[manifest] foreign+domestic rows: {len(frows)}; province rows: {len(prow)}")
    print(df_manifest[["dataset", "panel", "freq_h", "start", "end",
                       "neg_price_rate", "host_cached"]].to_string(index=False))
    print("\n--- domestic secondary ---")
    print(df_dom[["dataset", "freq_h", "neg_price_rate", "headline_eligible"]].to_string(index=False))


if __name__ == "__main__":
    main()
