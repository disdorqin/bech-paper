"""HCH v2 host cache generator — train 5 backbones on H0 (P0-2), cache predictions.

Per protocol §6: fit host on H0 ONLY, freeze, predict S1R-S4, and record
required cache metadata (split bounds, feature schema hash, git commit).
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
    DATASETS, load_dataset, load_shandong, load_province, PROVINCE_KEYS,
    build_tabular, build_sequences, assert_no_leakage,
    four_segment_split,
)

ALL_DATASETS = list(DATASETS) + ["shandong_DA", "shandong_RT"] + PROVINCE_KEYS
from eval_manifest import ExperimentManifest

SEED = 0
V2_BACKBONES = ("Linear", "MLP", "LSTM", "TCN", "PatchTST")
HERE = Path(__file__).resolve().parent
CACHE_ROOT = HERE / "results" / "cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _load_ds(key: str) -> dict:
    if key == "shandong_DA":
        return load_shandong(price_col="日前电价", encoding="gbk")
    if key == "shandong_RT":
        return load_shandong(price_col="实时电价", encoding="gbk")
    if key in PROVINCE_KEYS:
        prov, mode = key.rsplit("_", 1)
        return load_province(prov, price_col="日前电价" if mode == "DA" else "实时电价")
    return load_dataset(key)


def _cache_is_valid(ds_key: str, bb_name: str) -> bool:
    """Cache is reusable only if it has the current split_hash metadata (P0-B).

    Legacy R1B caches (seg.json = {S1,S2,S3,S4}, no split_hash) use a 4-segment
    split and are NOT compatible with the 7-segment protocol split -> regenerate.
    """
    cache_dir = CACHE_ROOT / ds_key / bb_name
    if not (cache_dir / "pred_full.npy").exists():
        return False
    seg_path = cache_dir / "seg.json"
    if not seg_path.exists():
        return False
    try:
        with open(seg_path) as f:
            seg = json.load(f)
    except Exception:
        return False
    return isinstance(seg, dict) and "split_hash" in seg


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _git_head() -> str:
    """Short git HEAD for provenance (protocol §6 cache metadata)."""
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _segment_bounds(exp: ExperimentManifest) -> dict:
    """Protocol §6: per-segment start/end dates for cache metadata."""
    bounds = {}
    for seg in ("H0", "S1R", "S2T", "S2V", "S3M", "S3C", "S4"):
        dates = sorted(exp.dates_in_split(seg))
        bounds[f"{seg}_start"] = dates[0] if dates else ""
        bounds[f"{seg}_end"] = dates[-1] if dates else ""
    return bounds


def cache_one(ds_key: str, bb_name: str, seed: int = 0) -> dict:
    t0 = time.time()
    ds = _load_ds(ds_key)
    y_full = ds["price"]
    n_full = len(y_full)

    X, y, names, valid = build_tabular(ds)
    n = len(valid)
    assert_no_leakage(ds, X, y, valid, names)

    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id=ds_key)
    # P0-2: host is fitted on H0 ONLY; S1R predictions are out-of-sample and
    # are what build the rank/signature reference.
    # P0-B: X/y from build_tabular and seq_full from build_sequences are
    # VALID-ROW compressed (len = n_valid). valid_indices_in_split() returns
    # RAW indices into the full array; using them here silently misfits the
    # host (H0 rows shifted ~warm hours forward + leaks ~warm hours of S1R).
    # Always index X/y/seq_full by valid_row_in_split().
    s1_rows = exp.valid_row_in_split("H0")

    bb = make_backbone(bb_name, seed=seed)
    if needs_seq(bb_name):
        seq_full = build_sequences(ds, valid)
        bb.fit(X[s1_rows], y[s1_rows], seq_full[s1_rows])
        yhat = bb.predict(X, seq_full)
    else:
        bb.fit(X[s1_rows], y[s1_rows])
        yhat = bb.predict(X)

    cache_dir = CACHE_ROOT / ds_key / bb_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    yhat_full_arr = np.full(len(y_full), np.nan, dtype=np.float32)
    yhat_full_arr[valid] = yhat.astype(np.float32)

    np.save(cache_dir / "pred_full.npy", yhat_full_arr)
    np.save(cache_dir / "pred.npy", yhat.astype(np.float32))
    np.save(cache_dir / "y.npy", y.astype(np.float32))
    np.save(cache_dir / "valid.npy", valid.astype(np.int32))

    seg_info = {
        "split_hash": exp.split_hash,
        "n_H0": int(len(exp.valid_indices_in_split("H0"))),
        "n_S1R": int(len(exp.valid_indices_in_split("S1R"))),
        "n_S2T": int(len(exp.valid_indices_in_split("S2T"))),
        "n_S2V": int(len(exp.valid_indices_in_split("S2V"))),
        "n_S3M": int(len(exp.valid_indices_in_split("S3M"))),
        "n_S3C": int(len(exp.valid_indices_in_split("S3C"))),
        "n_S4": int(len(exp.valid_indices_in_split("S4"))),
        "n_excluded_dates": len(exp.excluded_dates),
        **_segment_bounds(exp),  # protocol §6: per-segment start/end dates
        "feature_schema_hash": hashlib.sha256(
            (repr(sorted(names)) if isinstance(names, list) else repr(names)).encode()
        ).hexdigest()[:16],  # D1: make the cache self-describing for re-audit
    }
    with open(cache_dir / "seg.json", "w") as f:
        json.dump(seg_info, f)

    pred_hash = _hash_array(yhat.astype(np.float32))
    feature_schema_hash = hashlib.sha256(
        (repr(sorted(names)) if isinstance(names, list) else repr(names)).encode()
    ).hexdigest()[:16]

    n_params = "N/A"
    if hasattr(bb, "m") and hasattr(bb.m, "parameters"):
        n_params = sum(p.numel() for p in bb.m.parameters())

    record = {
        "dataset": ds_key,
        "backbone": bb_name,
        "seed": seed,
        "n_full": n_full,
        "n_valid": n,
        "n_H0": int(len(exp.valid_indices_in_split("H0"))),
        "n_S1R": int(len(exp.valid_indices_in_split("S1R"))),
        "n_S2T": int(len(exp.valid_indices_in_split("S2T"))),
        "n_S2V": int(len(exp.valid_indices_in_split("S2V"))),
        "n_S3M": int(len(exp.valid_indices_in_split("S3M"))),
        "n_S3C": int(len(exp.valid_indices_in_split("S3C"))),
        "n_S4": int(len(exp.valid_indices_in_split("S4"))),
        "n_excluded_dates": len(exp.excluded_dates),
        "split_hash": exp.split_hash,
        "pred_hash": pred_hash,
        "feature_schema_hash": feature_schema_hash,
        "git_commit": _git_head(),
        **_segment_bounds(exp),
        "n_params": n_params,
        "duration_s": round(time.time() - t0, 1),
    }
    return record


def reconstruct_manifest() -> list[dict]:
    """Rebuild the full cache manifest from on-disk caches (provenance restore).

    Walks results/cache/{ds}/{bb}/ and emits one record per existing cache by
    reading seg.json + pred.npy and recomputing what the file lacks
    (feature_schema_hash, pred_hash, n_params). Marks reused=True. Used to
    regenerate host_cache_manifest.csv without retraining anything.
    """
    from backbones import make_backbone
    records = []
    if not CACHE_ROOT.exists():
        return records
    for ds_dir in sorted(CACHE_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_key = ds_dir.name
        for bb_dir in sorted(ds_dir.iterdir()):
            if not bb_dir.is_dir():
                continue
            bb_name = bb_dir.name
            seg_path = bb_dir / "seg.json"
            pred_path = bb_dir / "pred.npy"
            if not seg_path.exists() or not pred_path.exists():
                continue
            try:
                seg = json.loads(seg_path.read_text())
            except Exception:
                continue
            # recompute feature schema from the loader (schema is deterministic)
            names = []
            try:
                ds = _load_ds(ds_key)
                X, y, nms, valid = build_tabular(ds)
                names = nms
            except Exception:
                pass
            feature_schema_hash = (hashlib.sha256(
                repr(sorted(names)).encode()).hexdigest()[:16]) if names else seg.get(
                "feature_schema_hash")
            pred_hash = _hash_array(
                np.load(pred_path, mmap_mode="r").astype(np.float32)
            ) if pred_path.exists() else None
            n_params = "N/A"
            try:
                bb = make_backbone(bb_name, seed=0)
                if hasattr(bb, "m") and hasattr(bb.m, "parameters"):
                    n_params = sum(p.numel() for p in bb.m.parameters())
            except Exception:
                pass
            records.append({
                "dataset": ds_key, "backbone": bb_name, "seed": 0,
                "n_full": seg.get("n_full"),
                "n_valid": seg.get("n_valid"),
                "n_H0": seg.get("n_H0"), "n_S1R": seg.get("n_S1R"),
                "n_S2T": seg.get("n_S2T"), "n_S2V": seg.get("n_S2V"),
                "n_S3M": seg.get("n_S3M"), "n_S3C": seg.get("n_S3C"),
                "n_S4": seg.get("n_S4"),
                "n_excluded_dates": seg.get("n_excluded_dates"),
                "split_hash": seg.get("split_hash"),
                "pred_hash": pred_hash,
                "feature_schema_hash": feature_schema_hash,
                "git_commit": _git_head(),
                **{f"{s}_start": seg.get(f"{s}_start", "") for s in
                   ("H0", "S1R", "S2T", "S2V", "S3M", "S3C", "S4")},
                **{f"{s}_end": seg.get(f"{s}_end", "") for s in
                   ("H0", "S1R", "S2T", "S2V", "S3M", "S3C", "S4")},
                "n_params": n_params,
                "duration_s": None,
                "reused": True,
            })
    return records


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None, help="single dataset or all")
    ap.add_argument("--backbone", type=str, default=None, help="single backbone or all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reconstruct", action="store_true",
                    help="rebuild host_cache_manifest.csv from on-disk caches only")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.reconstruct:
        records = reconstruct_manifest()
        out = HERE / "results" / "host_cache_manifest.csv"
        if records:
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                w.writeheader()
                w.writerows(records)
        print(f"Reconstructed manifest: {len(records)} caches -> {out}")
        return

    seed = args.seed

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = list(ALL_DATASETS)

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
            if args.resume and _cache_is_valid(ds_key, bb_name):
                print(f"[{done}/{total}] {ds_key} x {bb_name} SKIP (valid cache)")
                continue
            print(f"[{done}/{total}] {ds_key} x {bb_name} ", end="", flush=True)
            try:
                rec = cache_one(ds_key, bb_name, seed=seed)
                records.append(rec)
                print(f"OK ({rec['duration_s']}s)")
            except Exception as e:
                print(f"FAILED: {e}")
                records.append({
                    "dataset": ds_key, "backbone": bb_name,
                    "seed": seed, "error": str(e),
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
