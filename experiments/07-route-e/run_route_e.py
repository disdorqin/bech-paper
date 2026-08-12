"""Route-E 统一实验 runner.

用法:
  python run_route_e.py --dataset shandong --backbone Linear     # 单个组合
  python run_route_e.py --dataset all --backbone all              # 全矩阵
  python run_route_e.py --dataset shandong --backbone all --mode base  # 仅基座

数据流:
  S1(50%)→backbone train/freeze → S2(20%)→HCH train → S3(10%)→SCARR calibrate → S4(20%)→evaluate
  所有报告仅来自 S4。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from backbones import BACKBONES, make_backbone, needs_seq
from common import (
    DATASETS, PRICE_LAGS, SEQ_LEN,
    load_dataset, load_shandong,
    build_tabular, build_sequences, assert_no_leakage,
    four_segment_split, evaluate, weekly_naive,
    episode_metrics,
)
from _legacy.selective_hurdle import (
    HurdleCorrectionHead,
    build_corrector_features,
)

# conda epf-2
PY = r"D:/computer_download/environment/conda/epf-2/python.exe"
RESULTS = ROOT / "experiments" / "07-route-e" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------- parse args
def parse_args():
    p = argparse.ArgumentParser(description="Route-E experiment runner")
    p.add_argument("--dataset", default="shandong",
                   help="shandong | LAGO_DE | NEM_SA1 | ... | all")
    p.add_argument("--backbone", default="Linear",
                   help="Linear|MLP|LSTM|Transformer|GBDT | all")
    p.add_argument("--mode", default="full",
                   help="full (Base+Ours) | base (Base only)")
    p.add_argument("--neg_thr", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shandong_price_col", default="日前电价")
    return p.parse_args()


# ------------------------------------------------------------- data loaders
def _load_ds(key: str, args) -> dict:
    if key == "shandong":
        return load_shandong(price_col=args.shandong_price_col, encoding="gbk")
    return load_dataset(key)


# ---------------------------------------------------------- single run ----
def run_one(ds_key: str, bb_name: str, args) -> dict:
    t0 = time.time()
    seed = args.seed + hash(f"{ds_key}{bb_name}") % 1000
    np.random.seed(seed)

    # 1. load
    ds = _load_ds(ds_key, args)
    meta = ds["meta"]
    y_full = ds["price"]
    ts = ds["ts"]
    n_full = len(y_full)

    # 2. build features & split
    X, y, names, valid = build_tabular(ds)
    n = len(valid)
    seg = four_segment_split(n)

    # 3. backbone
    bb = make_backbone(bb_name, seed=seed)
    if needs_seq(bb_name):
        seq_full = build_sequences(ds, valid)
        bb.fit(X[seg["S1"]], y[seg["S1"]], seq_full[seg["S1"]])
        yhat_full = bb.predict(X, seq_full)
    else:
        bb.fit(X[seg["S1"]], y[seg["S1"]])
        yhat_full = bb.predict(X)

    yhat = yhat_full  # (N,)
    y_true = y        # (N,)
    valid_idx = valid  # indices into ds["price"]

    # 4. base evaluate (S4)
    s4 = seg["S4"]
    base_eval = evaluate(y_true[s4], yhat[s4], None, neg_thr=args.neg_thr)
    ep_true = episode_metrics(y_true[s4], yhat[s4], yhat[s4], neg_thr=args.neg_thr)
    base_eval.update({f"ep_{k}": v for k, v in ep_true.items()})

    result = {
        "dataset": ds_key, "backbone": bb_name, "seed": seed,
        "n_train_S1": int(len(seg["S1"])), "n_train_S2": int(len(seg["S2"])),
        "n_calib_S3": int(len(seg["S3"])), "n_test_S4": int(len(seg["S4"])),
        "base": base_eval,
    }

    if args.mode == "base":
        result["duration_s"] = round(time.time() - t0, 1)
        return result

    # 5. Hurdle Correction Head
    # build corrector features Z
    hour = ts.dt.hour.to_numpy()
    dayid = (ts - ts.min()).dt.days.to_numpy()
    oos = seg["S1"][-1] + 1

    # corrector uses S2 onwards
    corr_full = np.arange(oos, n)
    Z_corr, z_names = build_corrector_features(
        X, names, yhat, y_true,
        hour[valid_idx], dayid[valid_idx], oos,
    )

    hch = HurdleCorrectionHead(neg_thr=args.neg_thr, seed=seed)

    # S2 (first 80% of corrector range -> fit; last 20% -> held-out joins S3)
    n_corr = len(corr_full)
    cut_s2 = int(n_corr * 0.75)
    s2_fit = corr_full[:cut_s2]
    s2_hold = corr_full[cut_s2:]
    hch.fit(Z_corr[s2_fit], yhat[s2_fit], y_true[s2_fit])
    # S3 + held-out S2
    cal_idx = np.concatenate([seg["S3"], s2_hold]) if len(s2_hold) else seg["S3"]
    hch.calibrate(Z_corr[cal_idx], yhat[cal_idx], y_true[cal_idx])

    # S4
    y_corr, diag = hch.apply(Z_corr[seg["S4"]], yhat[seg["S4"]])
    our_eval = evaluate(y_true[s4], y_corr, None, neg_thr=args.neg_thr)
    our_ep = episode_metrics(y_true[s4], y_corr, yhat[s4], neg_thr=args.neg_thr)
    our_eval.update({f"ep_{k}": v for k, v in our_ep.items()})
    our_eval["fire_rate"] = diag["fire_rate"]
    our_eval["lam_neg"] = diag["lam_neg"]
    our_eval["lam_pos"] = diag["lam_pos"]

    result["ours"] = our_eval
    result["hch_info"] = hch.info
    result["duration_s"] = round(time.time() - t0, 1)

    # print summary
    imp = f'MAE: {base_eval["mae"]:.3f}→{our_eval["mae"]:.3f}'
    ep_rec = f'ep_recall: {ep_true["base_episode_recall"]:.1%}→{our_ep["our_episode_recall"]:.1%}'
    print(f"  [{ds_key}×{bb_name}] {imp} | {ep_rec} | fire={diag['fire_rate']:.1%} | λ=({diag['lam_neg']:.2f},{diag['lam_pos']:.2f}) | {result['duration_s']:.0f}s")
    return result


# ---------------------------------------------------------------- main ----
def main():
    args = parse_args()
    ds_list = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    bb_list = list(BACKBONES) if args.backbone == "all" else [args.backbone]

    all_results = []
    for ds_key in ds_list:
        for bb_name in bb_list:
            print(f"\n=== {ds_key} × {bb_name} ===")
            try:
                r = run_one(ds_key, bb_name, args)
                all_results.append(r)
            except Exception as e:
                print(f"  FAILED: {e}")
                all_results.append(dict(
                    dataset=ds_key, backbone=bb_name, error=str(e)))

    # save
    tag = f"{args.dataset}_{args.backbone}_{args.mode}"
    out = RESULTS / f"results_{tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nSaved: {out} ({len(all_results)} runs)")


if __name__ == "__main__":
    main()
