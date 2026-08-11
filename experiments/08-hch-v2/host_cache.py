"""HCH v2 host cache generator — train 5 backbones on S1, cache S1-S4 predictions.

Produces: experiments/08-hch-v2/results/host_cache_manifest.csv
           experiments/08-hch-v2/results/cache/{dataset}/{backbone}/pred.npy
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from backbones import make_backbone, needs_seq
from common import (
    DATASETS, load_dataset, load_shandong,
    build_tabular, build_sequences, assert_no_leakage,
    four_segment_split,
)

SEED = 0
    V2_BACKBONES = ("Linear", "MLP", "LSTM", "TCN", "PatchTST")
    HERE = Path(__file__).resolve().parent
    CACHE_ROOT = HERE / "results" / "cache"
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                   help="skip already-cached combos")
    ap.add_argument("--backbone", type=str, default=None,
                   help="single backbone name")
    ap.add_argument("--dataset", type=str, default=None,
                   help="single dataset key")
    args = ap.parse_args()

    if args.backbone:
        backbones = [args.backbone]
    else:
        backbones = list(V2_BACKBONES)


def _load_ds(key: str) -> dict:
    if key == "shandong_DA":
        return load_shandong(price_col="日前电价", encoding="gbk")
    if key == "shandong_RT":
        return load_shandong(price_col="实时电价", encoding="gbk")
    return load_dataset(key)


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def cache_one(ds_key: str, bb_name: str) -> dict:
    t0 = time.time()
    ds = _load_ds(ds_key)
    y_full = ds["price"]
    n_full = len(y_full)

    X, y, names, valid = build_tabular(ds)
    n = len(valid)
    seg = four_segment_split(n)
    s1, s2, s3, s4 = seg["S1"], seg["S2"], seg["S3"], seg["S4"]

    assert_no_leakage(ds, X, y, valid, names)

    bb = make_backbone(bb_name, seed=SEED)
    if needs_seq(bb_name):
        seq_full = build_sequences(ds, valid)
        bb.fit(X[s1], y[s1], seq_full[s1])
        yhat = bb.predict(X, seq_full)
    else:
        bb.fit(X[s1], y[s1])
        yhat = bb.predict(X)

    cache_dir = CACHE_ROOT / ds_key / bb_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    yhat_full_arr = np.full(len(y_full), np.nan, dtype=np.float32)
    yhat_full_arr[valid] = yhat.astype(np.float32)

    np.save(cache_dir / "pred_full.npy", yhat_full_arr)  # full-length with NaN warmup
    np.save(cache_dir / "pred.npy", yhat.astype(np.float32))  # valid-only
    np.save(cache_dir / "y.npy", y.astype(np.float32))
    np.save(cache_dir / "valid.npy", valid.astype(np.int32))

    seg_info = {k: [int(v[0]), int(v[-1])] for k, v in seg.items()}
    with open(cache_dir / "seg.json", "w") as f:
        json.dump(seg_info, f)

    pred_hash = _hash_array(yhat.astype(np.float32))

    n_params = "N/A"
    if hasattr(bb, "m") and hasattr(bb.m, "parameters"):
        n_params = sum(p.numel() for p in bb.m.parameters())

    record = {
        "dataset": ds_key,
        "backbone": bb_name,
        "seed": SEED,
        "n_full": n_full,
        "n_valid": n,
        "n_S1": int(len(s1)),
        "n_S2": int(len(s2)),
        "n_S3": int(len(s3)),
        "n_S4": int(len(s4)),
        "pred_hash": pred_hash,
        "n_params": n_params,
        "duration_s": round(time.time() - t0, 1),
    }
    return record


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None, help="single dataset or all")
    ap.add_argument("--backbone", type=str, default=None, help="single backbone or all")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = list(DATASETS.keys()) + ["shandong_DA", "shandong_RT"]

    if args.backbone:
        backbones = [args.backbone]
    else:
        backbones = list(V2_BACKBONES)

    records = []

    total = len(datasets) * len(backbones)
    done = 0
    for ds_key in datasets:
        for bb_name in backbones:
            done += 1
            cache_dir = CACHE_ROOT / ds_key / bb_name
            if args.resume and (cache_dir / "pred_full.npy").exists():
                print(f"[{done}/{total}] {ds_key} x {bb_name} SKIP (cached)")
                continue
            print(f"[{done}/{total}] {ds_key} x {bb_name} ", end="", flush=True)
            try:
                rec = cache_one(ds_key, bb_name)
                records.append(rec)
                print(f"OK ({rec['duration_s']}s)")
            except Exception as e:
                print(f"FAILED: {e}")
                records.append({
                    "dataset": ds_key, "backbone": bb_name,
                    "seed": SEED, "error": str(e),
                })

    out = HERE / "results" / "host_cache_manifest.csv"
    if records:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
        print(f"\nSaved: {out} ({len(records)} records)")

    n_ok = sum(1 for r in records if "error" not in r)
    print(f"OK: {n_ok}/{len(records)}")


if __name__ == "__main__":
    main()
