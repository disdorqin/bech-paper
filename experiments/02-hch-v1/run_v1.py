"""HCH v1 — unified experiment runner.

用法:
  python run_v1.py --dataset LAGO_DE --backbone Linear          # smoke test
  python run_v1.py --dataset all --backbone all                  # full matrix

数据流 (四段严格隔离):
  S1(50%) -> backbone train/freeze + spike threshold computed & frozen
  S2(20%) -> all methods fit (HCH + 5 peers)
  S3(10%) -> HCH SCARR calibrate ONLY (peers ignore)
  S4(20%) -> locked evaluation (NO method touches S4 before eval)

所有方法共享同一宿主预测、切分、阈值和指标。seed=0 固定。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "07-route-e" / "peers"))

from backbones import BACKBONES, make_backbone, needs_seq
from common import (
    DATASETS,
    load_dataset,
    load_shandong,
    build_tabular,
    build_sequences,
    assert_no_leakage,
    four_segment_split,
    evaluate,
    weekly_naive,
)
from selective_hurdle import HurdleCorrectionHead, build_corrector_features

from base import Identity
from quantile import QuantileCorrection
from vahedi_style import VahediStyle
from spike_reg import SpikeRegularization
from crc_impl import CRC
from delta_adapter import DeltaAdapter

PY = r"D:/computer_download/environment/conda/epf-2/python.exe"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
SEED = 0
NEG_THR = 0.0

ALL_DATASETS = list(DATASETS.keys()) + ["shandong"]

METHOD_NAMES = [
    "Identity",
    "QuantileCorrection",
    "VahediStyle",
    "SpikeRegularization",
    "CRC",
    "DeltaAdapter",
    "HCH",
]


def parse_args():
    p = argparse.ArgumentParser(description="HCH v1 unified runner")
    p.add_argument("--dataset", default="LAGO_DE",
                   help="dataset key | all")
    p.add_argument("--backbone", default="Linear",
                   help="backbone name | all")
    p.add_argument("--shandong_price_col", default="日前电价")
    p.add_argument("--resume", type=str, default=None,
                   help="resume from existing JSON results file")
    return p.parse_args()


def _load_ds(key: str, args) -> dict:
    if key == "shandong":
        return load_shandong(price_col=args.shandong_price_col, encoding="gbk")
    return load_dataset(key)


def _compute_naive(y_full, valid, s4):
    t4 = valid[s4]
    naive = y_full[t4 - 168]
    return naive


def _touch_rate(pred, yhat_base):
    return float((np.abs(pred - yhat_base) > 1e-9).mean())


def _method_eval(name, pred, y, naive, yhat_base, neg_thr, spike_thr):
    ev = evaluate(y, pred, naive, neg_thr=neg_thr, spike_thr=spike_thr)
    out = {
        "method": name,
        "n": ev["n"],
        "mae": ev["mae"],
        "rmse": ev["rmse"],
        "rmae": ev["rmae"],
        "neg_n": ev["neg_n"],
        "mae_on_neg": ev["mae_on_neg"],
        "neg_miss_rate": ev["neg_miss_rate"],
        "spike_thr": ev["spike_thr"],
        "spike_n": ev["spike_n"],
        "mae_on_spike": ev["mae_on_spike"],
        "spike_miss_rate": ev["spike_miss_rate"],
        "normal_n": _normal_n(y, neg_thr, spike_thr),
        "mae_on_normal": ev["mae_on_normal"],
        "touch_rate": _touch_rate(pred, yhat_base),
    }
    return out


def _normal_n(y, neg_thr, spike_thr):
    neg = y < neg_thr
    sp = y > spike_thr
    normal = (~neg) & (~sp)
    return int(normal.sum())


def run_one(ds_key: str, bb_name: str, args) -> list[dict]:
    t0 = time.time()
    np.random.seed(SEED)

    ds = _load_ds(ds_key, args)
    meta = ds["meta"]
    y_full = ds["price"]
    ts = ds["ts"]
    n_full = len(y_full)

    X, y, names, valid = build_tabular(ds)
    n = len(valid)
    seg = four_segment_split(n)
    s1, s2, s3, s4 = seg["S1"], seg["S2"], seg["S3"], seg["S4"]

    assert_no_leakage(ds, X, y, valid, names)

    spike_thr = float(np.quantile(y[s1], 0.99))

    bb = make_backbone(bb_name, seed=SEED)
    if needs_seq(bb_name):
        seq_full = build_sequences(ds, valid)
        bb.fit(X[s1], y[s1], seq_full[s1])
        yhat_full = bb.predict(X, seq_full)
    else:
        bb.fit(X[s1], y[s1])
        yhat_full = bb.predict(X)

    yhat = yhat_full
    y_true = y
    naive = _compute_naive(y_full, valid, s4)

    hour = ts.dt.hour.to_numpy()
    dayid = (ts - ts.min()).dt.days.to_numpy()
    oos = s1[-1] + 1
    Z_all, z_names = build_corrector_features(
        X, names, yhat, y_true,
        hour[valid], dayid[valid], oos,
    )

    Z_s2 = Z_all[s2]
    Z_s3 = Z_all[s3]
    Z_s4 = Z_all[s4]
    yhat_s2 = yhat[s2]
    yhat_s3 = yhat[s3]
    yhat_s4 = yhat[s4]
    y_s2 = y_true[s2]
    y_s3 = y_true[s3]
    y_s4 = y_true[s4]

    identity = Identity()
    y_id = identity.predict(Z_s4, yhat_s4)
    id_eval = _method_eval("Identity", y_id, y_s4, naive, yhat_s4,
                           NEG_THR, spike_thr)
    id_eval["status"] = "ok"

    results = [id_eval]

    methods: list[tuple[str, object]] = [
        ("QuantileCorrection", QuantileCorrection(seed=SEED)),
        ("VahediStyle", VahediStyle(neg_thr=NEG_THR, seed=SEED)),
        ("SpikeRegularization", SpikeRegularization(seed=SEED)),
        ("CRC", CRC(seed=SEED)),
        ("DeltaAdapter", DeltaAdapter(seed=SEED)),
    ]

    for mname, m in methods:
        try:
            m.fit(Z_s2, yhat_s2, y_s2)
            pred = m.predict(Z_s4, yhat_s4)
            ev = _method_eval(mname, pred, y_s4, naive, yhat_s4,
                              NEG_THR, spike_thr)
            ev["status"] = "ok"
            results.append(ev)
        except Exception as e:
            results.append({
                "method": mname,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    try:
        hch = HurdleCorrectionHead(neg_thr=NEG_THR, seed=SEED)
        hch.fit(Z_s2, yhat_s2, y_s2, spike_thr=spike_thr)
        hch.calibrate(Z_s3, yhat_s3, y_s3)
        y_corr, diag = hch.apply(Z_s4, yhat_s4)
        ev = _method_eval("HCH", y_corr, y_s4, naive, yhat_s4,
                          NEG_THR, spike_thr)
        ev["fire_rate"] = diag["fire_rate"]
        ev["lam_neg"] = diag["lam_neg"]
        ev["lam_pos"] = diag["lam_pos"]
        ev["status"] = "ok"
        results.append(ev)
    except Exception as e:
        results.append({
            "method": "HCH",
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })

    shared = {
        "dataset": ds_key,
        "backbone": bb_name,
        "seed": SEED,
        "neg_thr": NEG_THR,
        "spike_thr": float(spike_thr),
        "spike_thr_source": "S1",
        "n_full": int(n_full),
        "n_valid": int(n),
        "n_S1": int(len(s1)),
        "n_S2": int(len(s2)),
        "n_S3": int(len(s3)),
        "n_S4": int(len(s4)),
        "s1_range": [int(valid[s1][0]), int(valid[s1][-1])],
        "s2_range": [int(valid[s2][0]), int(valid[s2][-1])],
        "s3_range": [int(valid[s3][0]), int(valid[s3][-1])],
        "s4_range": [int(valid[s4][0]), int(valid[s4][-1])],
        "duration_s": round(time.time() - t0, 1),
    }
    for r in results:
        r.update(shared)

    return results


def _compute_deltas(results_by_combo: list[dict]):
    for entry in results_by_combo:
        methods = entry.get("methods", [])
        base = next((m for m in methods if m.get("method") == "Identity"), None)
        if base is None:
            continue
        for m in methods:
            if m.get("status") != "ok" or m.get("method") == "Identity":
                continue
            for metric in ["mae", "mae_on_neg", "mae_on_spike", "mae_on_normal"]:
                bv = base.get(metric)
                mv = m.get(metric)
                if bv is not None and mv is not None:
                    m[f"delta_{metric}"] = round(mv - bv, 6)
                else:
                    m[f"delta_{metric}"] = None


def _build_csv_rows(results_by_combo: list[dict]) -> list[dict]:
    rows = []
    for entry in results_by_combo:
        shared = {k: v for k, v in entry.items() if k != "methods"}
        for m in entry.get("methods", []):
            row = dict(shared)
            row.update(m)
            rows.append(row)
    return rows


def _write_csv(rows: list[dict], path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _build_report(results_by_combo: list[dict], failures: list[dict]) -> str:
    n_combos = len(results_by_combo)
    n_methods_total = n_combos * len(METHOD_NAMES)
    n_ok = sum(
        1 for e in results_by_combo
        for m in e.get("methods", [])
        if m.get("status") == "ok"
    )
    n_failed = sum(
        1 for e in results_by_combo
        for m in e.get("methods", [])
        if m.get("status") == "failed"
    )

    lines = [
        f"# HCH v1 Full Benchmark Report",
        f"",
        f"**Generated**: {TIMESTAMP}",
        f"**Seed**: {SEED}  **neg_thr**: {NEG_THR}  **spike_thr source**: S1 p99",
        f"**Split**: S1(50%) / S2(20%) / S3(10%) / S4(20%)",
        f"",
        f"## 1. Summary",
        f"",
        f"- Datasets: {len({e['dataset'] for e in results_by_combo})}",
        f"- Backbones: {len({e['backbone'] for e in results_by_combo})}",
        f"- Total combinations: {n_combos}",
        f"- Total method evaluations: {n_methods_total}",
        f"- OK: {n_ok}  Failed: {n_failed}",
        f"",
        f"## 2. Per-Method Aggregate (Delta vs Base)",
        f"",
    ]

    agg: dict[str, list[float]] = {}
    for entry in results_by_combo:
        base = next((m for m in entry.get("methods", [])
                     if m.get("method") == "Identity" and m.get("status") == "ok"), None)
        if base is None:
            continue
        for m in entry.get("methods", []):
            if m.get("status") != "ok" or m.get("method") == "Identity":
                continue
            nm = m["method"]
            if nm not in agg:
                agg[nm] = []
            for metric in ["delta_mae", "delta_mae_on_neg", "delta_mae_on_spike",
                           "delta_mae_on_normal"]:
                v = m.get(metric)
                if v is not None and np.isfinite(v):
                    agg.setdefault(f"{nm}:{metric}", []).append(v)

    lines.append("| Method | Δ-MAE (mean) | Δ-MAE-on-neg | Δ-MAE-on-spike | Δ-MAE-on-normal | N combos |")
    lines.append("|---|---|---|---|---|---|")
    for nm in METHOD_NAMES:
        if nm == "Identity":
            continue
        vals = {k: agg.get(f"{nm}:{k}", []) for k in
                ["delta_mae", "delta_mae_on_neg", "delta_mae_on_spike", "delta_mae_on_normal"]}
        n = len(vals["delta_mae"])
        if n == 0:
            lines.append(f"| {nm} | — | — | — | — | 0 |")
            continue
        d_mae = np.mean(vals["delta_mae"])
        d_neg = np.mean([v for v in vals["delta_mae_on_neg"] if v is not None]) if any(v is not None for v in vals["delta_mae_on_neg"]) else None
        d_sp = np.mean([v for v in vals["delta_mae_on_spike"] if v is not None]) if any(v is not None for v in vals["delta_mae_on_spike"]) else None
        d_norm = np.mean([v for v in vals["delta_mae_on_normal"] if v is not None]) if any(v is not None for v in vals["delta_mae_on_normal"]) else None
        d_neg_s = f"{d_neg:.4f}" if d_neg is not None else "—"
        d_sp_s = f"{d_sp:.4f}" if d_sp is not None else "—"
        d_norm_s = f"{d_norm:.4f}" if d_norm is not None else "—"
        lines.append(f"| {nm} | {d_mae:.4f} | {d_neg_s} | {d_sp_s} | {d_norm_s} | {n} |")

    lines += [
        "",
        "## 3. HCH Behaviour Summary",
        "",
    ]
    hch_entries = []
    for entry in results_by_combo:
        for m in entry.get("methods", []):
            if m.get("method") == "HCH" and m.get("status") == "ok":
                hch_entries.append((entry["dataset"], entry["backbone"], m))
    if hch_entries:
        lines.append("| Dataset | Backbone | Δ-MAE | Δ-neg | Δ-spike | Δ-normal | fire_rate | λ_neg | λ_pos | neg_n | spike_n |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for ds, bb, m in hch_entries:
            lines.append(
                f"| {ds} | {bb} | {m.get('delta_mae', '—')} | {m.get('delta_mae_on_neg', '—')} | "
                f"{m.get('delta_mae_on_spike', '—')} | {m.get('delta_mae_on_normal', '—')} | "
                f"{m.get('fire_rate', '—')} | {m.get('lam_neg', '—')} | {m.get('lam_pos', '—')} | "
                f"{m.get('neg_n', '—')} | {m.get('spike_n', '—')} |"
            )

    lines += [
        "",
        "## 4. Rankings (by mean Δ-MAE across all combos)",
        "",
    ]
    ranked = sorted(
        [(nm, np.mean(agg[f"{nm}:delta_mae"])) for nm in METHOD_NAMES
         if nm != "Identity" and f"{nm}:delta_mae" in agg and agg[f"{nm}:delta_mae"]],
        key=lambda x: x[1],
    )
    for i, (nm, d) in enumerate(ranked, 1):
        lines.append(f"{i}. **{nm}**: {d:.4f}")

    lines += [
        "",
        "## 5. Next Steps (top 3-5 issues)",
        "",
        "1. (to be filled from results)",
        "2. ",
        "3. ",
        "4. ",
        "5. ",
    ]

    return "\n".join(lines)


def main():
    args = parse_args()

    ds_list = ALL_DATASETS if args.dataset == "all" else [args.dataset]
    bb_list = list(BACKBONES) if args.backbone == "all" else [args.backbone]

    ds_list = [d for d in ds_list if d == "shandong" or d in DATASETS]

    results_by_combo: list[dict] = []
    failures: list[dict] = []

    if args.resume:
        with open(args.resume, "r", encoding="utf-8") as f:
            existing = json.load(f)
        results_by_combo = existing.get("results", [])
        completed = {(e["dataset"], e["backbone"]) for e in results_by_combo}
        print(f"Resuming: {len(completed)} combos already done")

    total = len(ds_list) * len(bb_list)
    done = 0

    for ds_key in ds_list:
        for bb_name in bb_list:
            if args.resume and (ds_key, bb_name) in completed:
                done += 1
                continue

            print(f"\n[{done+1}/{total}] {ds_key} × {bb_name} ", end="", flush=True)
            try:
                methods = run_one(ds_key, bb_name, args)
                combo = {
                    "dataset": ds_key,
                    "backbone": bb_name,
                    "methods": methods,
                }
                results_by_combo.append(combo)
                ok = sum(1 for m in methods if m.get("status") == "ok")
                fail = sum(1 for m in methods if m.get("status") == "failed")
                print(f"OK={ok} FAIL={fail}")
            except Exception as e:
                tb = traceback.format_exc()
                print(f"COMBO FAILED: {e}")
                failures.append({
                    "dataset": ds_key,
                    "backbone": bb_name,
                    "error": str(e),
                    "traceback": tb,
                })
                results_by_combo.append({
                    "dataset": ds_key,
                    "backbone": bb_name,
                    "error": str(e),
                    "methods": [],
                })

            done += 1

            if done % 3 == 0:
                _compute_deltas(results_by_combo)
                tmp = OUT / f"_autosave_{TIMESTAMP}.json"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"results": results_by_combo, "failures": failures},
                              f, indent=2, default=str, ensure_ascii=False)

    _compute_deltas(results_by_combo)

    n_ok = sum(1 for e in results_by_combo
               for m in e.get("methods", []) if m.get("status") == "ok")
    n_failed = sum(1 for e in results_by_combo
                   for m in e.get("methods", []) if m.get("status") == "failed")
    expected = len(results_by_combo) * len(METHOD_NAMES)
    if n_ok + n_failed != expected:
        print(f"WARNING: task count mismatch: ok={n_ok} + failed={n_failed} != {expected}")

    json_path = OUT / f"v1_full_matrix_{TIMESTAMP}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {"seed": SEED, "neg_thr": NEG_THR, "split": "50/20/10/20",
                     "spike_thr_source": "S1 p99", "timestamp": TIMESTAMP},
            "results": results_by_combo,
            "failures": failures,
        }, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nJSON: {json_path}")

    csv_rows = _build_csv_rows(results_by_combo)
    csv_path = OUT / f"v1_full_matrix_{TIMESTAMP}.csv"
    _write_csv(csv_rows, str(csv_path))
    print(f"CSV:  {csv_path}")

    report = _build_report(results_by_combo, failures)
    md_path = OUT / f"v1_full_report_{TIMESTAMP}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"MD:   {md_path}")

    log_path = OUT / f"v1_full_failures_{TIMESTAMP}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        if failures:
            for fl in failures:
                f.write(f"[{fl['dataset']}×{fl['backbone']}] {fl['error']}\n")
                f.write(f"{fl['traceback']}\n\n")
            for entry in results_by_combo:
                for m in entry.get("methods", []):
                    if m.get("status") == "failed":
                        f.write(f"[{entry['dataset']}×{entry['backbone']}::{m['method']}] {m.get('error')}\n")
                        f.write(f"{m.get('traceback', '')}\n\n")
        else:
            f.write("0 failures\n")
    print(f"LOG:  {log_path}")

    print(f"\nDONE: {len(results_by_combo)} combos, {n_ok} ok, {n_failed} failed")


if __name__ == "__main__":
    main()
