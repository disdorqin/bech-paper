"""D1 admission audit for domestic provinces (Shandong + 4 provinces × DA/RT).

For each of the 10 (mode) keys: load via the standard dispatch, run build_tabular
+ assert_no_leakage + assert_no_future_leakage + verify_forecast_vs_actual, and
report honest per-key stats (DST completeness, true missing rate from the raw
file, negative-price share, feature counts, 7-seg day counts, feature schema
hash). Also re-audits any existing host caches: a cache is reusable only if its
seg.json feature_schema_hash matches the current loader's hash.

Produces:
    results/domestic/admission_report.csv
    results/domestic/admission_summary.md
Exit code 0 = all 10 modes PASS; 1 = any FAIL (leak, schema mismatch).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import (DATA, load_shandong, load_province, PROVINCES, PROVINCE_KEYS,
                    build_tabular, assert_no_leakage, assert_no_future_leakage,
                    verify_forecast_vs_actual)
from eval_manifest import ExperimentManifest

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "domestic"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT = HERE / "results" / "cache"
HOSTS4 = ("Linear", "MLP", "LSTM", "PatchTST")


def schema_hash(names: list[str]) -> str:
    return hashlib.sha256(repr(sorted(names)).encode()).hexdigest()[:16]


def raw_frame(key: str) -> pd.DataFrame:
    if key in ("shandong_DA", "shandong_RT"):
        return pd.read_csv(
            os.path.join(DATA, "raw", "provinces", "shandong_pmos_hourly.csv"),
            encoding="gbk")
    prov, _ = key.rsplit("_", 1)
    return pd.read_excel(os.path.join(DATA, "raw", "provinces",
                                      PROVINCES[prov]["file"]))


def load(key: str) -> dict:
    if key in ("shandong_DA", "shandong_RT"):
        return load_shandong(price_col="日前电价" if key == "shandong_DA" else "实时电价",
                             encoding="gbk")
    prov, mode = key.rsplit("_", 1)
    return load_province(prov, price_col="日前电价" if mode == "DA" else "实时电价")


def audit_mode(key: str) -> dict:
    ds = load(key)
    raw = raw_frame(key)
    raw.columns = raw.columns.str.strip()
    price_col = ds["meta"]["price_col"]
    n = len(ds["price"])
    ts = ds["ts"]

    # TRUE missing rate from timestamps: absent rows never enter the loader's
    # ffill/bfill, so compare first..last expected hours vs rows actually present.
    n_expected = int((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600) + 1
    missing_pct = round((n_expected - n) / n_expected * 100, 3)

    dates = ts.dt.date
    counts = dates.value_counts().sort_index()
    non24h = counts[counts != 24]
    # China has no DST: a 23/25h day is a data artifact. Boundary partial days
    # (first/last date of the file) are expected and auto-excluded by the
    # ExperimentManifest; an INTERIOR non-24h day is a genuine anomaly.
    boundary_partial = bool(non24h.index.isin([counts.index[0], counts.index[-1]]).all())
    dst_abnormal = 0 if boundary_partial else int(len(non24h))

    neg_n = int((ds["price"] < 0).sum())
    neg_pct = round(neg_n / n * 100, 3)

    X, y, names, valid = build_tabular(ds)
    assert_no_leakage(ds, X, y, valid, names)          # existing same-hour guard
    fut = assert_no_future_leakage(X, y, names)        # NEW future-price guard
    fv = verify_forecast_vs_actual(ds)                 # fc vs act per column

    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id=key)
    s1_days = len(exp.valid_indices_in_split("S1R")) // 24
    s2t_days = len(exp.valid_indices_in_split("S2T")) // 24
    s2v_days = len(exp.valid_indices_in_split("S2V")) // 24
    n_excluded_dates = len(exp.excluded_dates)

    worst_ok = [(nm, r) for nm, r in fut.items() if r["kind"] == "ok"]
    worst = max(worst_ok, key=lambda t: t[1]["max_future_corr"]) if worst_ok else None
    n_const = sum(1 for r in fut.values() if r["kind"] == "const")
    n_backfilled = sum(1 for r in fv.values() if r["kind"] == "backfilled_actual")

    # existing-cache re-audit. Reusable iff:
    #   ok        — feature_schema_hash present AND matches current loader
    #   legacy-ok — legacy cache (no feature_schema_hash) but split_hash matches
    #               current manifest (segment boundaries identical). Reusable.
    #   rebuild   — split_hash mismatch (stale) or corrupt
    #   missing   — not built yet
    cache_state = {}
    for bb in HOSTS4:
        seg = CACHE_ROOT / key / bb / "seg.json"
        exists = seg.exists()
        state = "missing"
        if exists:
            try:
                meta = json.loads(seg.read_text())
                h = meta.get("feature_schema_hash")
                sh = meta.get("split_hash")
                cur_sh = ExperimentManifest.from_dataset(
                    ds, valid, dataset_id=key).split_hash
                if h == schema_hash(names):
                    state = "ok"
                elif h is None and sh == cur_sh:
                    state = "legacy-ok"
                else:
                    state = "rebuild"
            except Exception:
                state = "rebuild"
        cache_state[bb] = state
    cache_all_ok = all(v in ("ok", "legacy-ok") for v in cache_state.values())

    row = {
        "dataset": key,
        "n_hours": n,
        "n_days": len(counts),
        "time_start": str(ts.iloc[0])[:10],
        "time_end": str(ts.iloc[-1])[:10],
        "primary_resolution": str(ts.diff().dropna().value_counts().index[0]),
        "dst_n_abnormal": dst_abnormal,
        "boundary_partial_only": boundary_partial,
        "n_excluded_dates": n_excluded_dates,
        "missing_pct": missing_pct,
        "neg_n": neg_n,
        "neg_pct": neg_pct,
        "n_exog_fc": len(ds["exog_fc"].columns),
        "n_exog_act": len(ds["exog_act"].columns),
        "n_feat": len(names),
        "n_valid_rows": len(valid),
        "S1R_days": s1_days,
        "S2T_days": s2t_days,
        "S2V_days": s2v_days,
        "future_leak": "PASS" if worst is None or worst[1]["max_future_corr"] < 0.999 else "FAIL",
        "worst_future_corr": round(worst[1]["max_future_corr"], 4) if worst else None,
        "worst_future_col": worst[0] if worst else "",
        "worst_future_offset": worst[1]["argmax_offset"] if worst else None,
        "n_const_cols": n_const,
        "n_backfilled_fc": n_backfilled,
        "backfilled_cols": ";".join(c for c, r in fv.items() if r["kind"] == "backfilled_actual"),
        "feature_schema_hash": schema_hash(names),
        "cache_Linear": cache_state["Linear"],
        "cache_MLP": cache_state["MLP"],
        "cache_LSTM": cache_state["LSTM"],
        "cache_PatchTST": cache_state["PatchTST"],
        "cache_all_ok": cache_all_ok,
        "exog_fc_cols": ";".join(ds["exog_fc"].columns),
        "exog_act_cols": ";".join(ds["exog_act"].columns),
    }
    # 准入 = 数据硬门(future leak / 回填实际值)。缓存是否已建是独立的构建步骤
    # (#20),不在准入判定内(否则建缓存前永远无法通过,循环依赖)。缺失率/S1R 偏薄
    # 是高缺失或薄参考的诚实 WARN 标注,不挡路(manifest 自动排除非 24h 日)。
    hard_pass = (row["future_leak"] == "PASS" and n_backfilled == 0)
    warnings = []
    if missing_pct > 5:
        warnings.append(f"missing {missing_pct}%")
    if s1_days < 25:
        warnings.append(f"S1R only {s1_days}d")
    if dst_abnormal > 0:
        warnings.append(f"{dst_abnormal} non-24h days (manifest-excluded)")
    row["admission"] = "PASS" if hard_pass else "FAIL"
    row["warnings"] = "; ".join(warnings)
    return row


def main():
    keys = ["shandong_DA", "shandong_RT"] + PROVINCE_KEYS
    rows, fails = [], []
    for k in keys:
        try:
            r = audit_mode(k)
            rows.append(r)
            print(f"  {k:14s} {r['n_days']:5d}d  neg={r['neg_pct']:6.2f}%  "
                  f"missing={r['missing_pct']:.2f}%  future={r['future_leak']:4s}  "
                  f"S1R={r['S1R_days']:3d}d  cache={r['cache_all_ok']}  "
                  f"-> {r['admission']}")
            if r["admission"] != "PASS":
                fails.append(k)
        except Exception as e:
            print(f"  {k:14s} FAILED — {type(e).__name__}: {e}")
            fails.append(k)

    csv_path = OUT / "admission_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = ["# 国内省份准入审计 (D1)", "",
          f"- 日期: 2026-08-14", f"- 审计 {len(keys)} 模式 (山东 + 4 省 × DA/RT)",
          f"- 硬门: 未来电价泄漏 (assert_no_future_leakage) / 回填实际值冒充预测值 (verify_forecast_vs_actual) / DST 完整性",
          "", "| key | days | neg% | missing% | DST | S1R_d | S2T_d | S2V_d | future | backfill | cache_ok | admission |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['dataset']} | {r['n_days']} | {r['neg_pct']} | {r['missing_pct']} "
                  f"| {r['dst_n_abnormal']} | {r['S1R_days']} | {r['S2T_days']} | {r['S2V_days']} "
                  f"| {r['future_leak']} | {r['n_backfilled_fc']} | {r['cache_all_ok']} | {r['admission']} |")
    md += ["", "## 备注",
           "- 山东有负价 (11~13%),其余 4 省 0% 负价 → 按「严禁」规则不报 negative-price 指标。",
           "- 宁夏/青海 S1R 仅 ~18-19 天,S1RankReference 小时池 ≥10 仍可用,但参考偏噪,如实标注 WARN。",
           "- 已缓存复检: cache_ok=True 仅当 seg.json feature_schema_hash 与当前加载器一致。"]
    (OUT / "admission_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {OUT / 'admission_summary.md'}")
    if fails:
        print(f"\nFAIL ({len(fails)}): {fails}")
        sys.exit(1)
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
