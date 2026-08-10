"""Peer baseline runner — LTSF datasets + EPF datasets.

用法:
  python run_peers.py --dataset ETTh1 --backbone Linear
  python run_peers.py --dataset all_ltsf --backbone all --peers quantile,crc
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "07-route-e"))

from backbones import make_backbone, DEVICE
from common import four_segment_split, evaluate, episode_metrics
from selective_hurdle import HurdleCorrectionHead, build_corrector_features

from peers.base import Identity
from peers.quantile import QuantileCorrection
from peers.vahedi_style import VahediStyle
from peers.spike_reg import SpikeRegularization
from peers.crc_impl import CRC
from peers.delta_adapter import DeltaAdapter

# registry
PEER_REGISTRY = {
    "identity": lambda args: Identity(),
    "quantile": lambda args: QuantileCorrection(seed=args.seed),
    "vahedi": lambda args: VahediStyle(seed=args.seed),
    "spike_reg": lambda args: SpikeRegularization(seed=args.seed),
    "crc": lambda args: CRC(seed=args.seed),
    "delta_adapter": lambda args: DeltaAdapter(seed=args.seed),
}

RESULTS = ROOT / "experiments" / "07-route-e" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

LTSF_DATA = ROOT / "data" / "raw" / "ts_benchmarks"
LTSF_TARGETS = {
    "ETTh1": "OT", "ETTh2": "OT", "ETTm1": "OT", "ETTm2": "OT",
    "weather": "OT", "electricity": "MT_001",
    "traffic": "sensor_0", "exchange_rate": "Australia",
}
LTSF_DATASETS = list(LTSF_TARGETS.keys())


def load_ltsf(name: str) -> dict:
    name_lower = name.lower()
    fn_map_lower = {
        "weather": "weather.csv", "electricity": "electricity.csv",
        "traffic": "traffic.csv", "exchange_rate": "exchange_rate.csv",
    }
    fname = fn_map_lower.get(name_lower, f"{name}.csv")
    path = LTSF_DATA / fname
    for enc in ["utf-8", "latin-1", "cp1252", "ISO-8859-1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        df = pd.read_csv(path, encoding="latin-1", errors="replace")
    df["timestamp"] = pd.to_datetime(df["date"])
    target = LTSF_TARGETS.get(name_lower, LTSF_TARGETS.get(name, df.columns[1]))
    price = df[target].astype(float).to_numpy()
    exog_cols = [c for c in df.columns if c not in ("date", "timestamp", target)]
    exog_fc = pd.DataFrame({c: df[c].astype(float) for c in exog_cols})
    return dict(
        key=name, ts=df["timestamp"], price=price,
        exog_fc=exog_fc, exog_act=pd.DataFrame(index=df.index),
        meta=dict(path=str(path), target=target, tier="LTSF"),
    )


def build_ltsf_features(ds: dict) -> tuple:
    y = ds["price"]
    ts = ds["ts"]
    n = len(y)
    cols, names = [], []

    s = pd.Series(y)
    for L in (1, 2, 3, 7, 14, 28, 56, 168, 336):
        cols.append(s.shift(L).to_numpy()); names.append(f"lag{L}")
    # sin/cos encoding
    hr = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    mon = ts.dt.month.to_numpy()
    cols += [np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24),
             np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
             np.sin(2*np.pi*mon/12), np.cos(2*np.pi*mon/12)]
    names += ["hour_sin","hour_cos","dow_sin","dow_cos","mon_sin","mon_cos"]

    X = np.column_stack(cols).astype(np.float64)
    warm = 336
    valid = np.arange(warm, n)
    ok = np.isfinite(X[valid]).all(axis=1)
    valid = valid[ok]
    return X[valid], y[valid], names, valid


def run_peer_comparison(ds, ds_key: str, bb_name: str, peers: list, args):
    """Returns dict of per-method metrics on S4."""
    t0 = time.time()
    seed = args.seed + hash(f"{ds_key}{bb_name}") % 1000
    np.random.seed(seed)
    ts = ds["ts"]

    X, y, names, valid = build_ltsf_features(ds)
    n = len(valid)
    seg = four_segment_split(n)

    bb = make_backbone(bb_name, seed=seed)
    bb.fit(X[seg["S1"]], y[seg["S1"]])
    yhat = bb.predict(X)

    hour = ts.dt.hour.to_numpy()
    dayid = (ts - ts.min()).dt.days.to_numpy()
    oos = seg["S1"][-1] + 1
    corr_full = np.arange(oos, n)

    Z_corr, _ = build_corrector_features(
        X, names, yhat, y, hour[valid], dayid[valid], oos,
    )
    n_corr = len(corr_full)
    cut = int(n_corr * 0.75)
    s2_fit = corr_full[:cut]
    s2_hold = corr_full[cut:]

    s4 = seg["S4"]
    results = {}
    base_ref = yhat[s4]
    base_err = np.abs(base_ref - y[s4])

    for peer in peers:
        nm = peer.name
        t1 = time.time()
        try:
            peer.fit(Z_corr[s2_fit], yhat[s2_fit], y[s2_fit])
            pred = peer.predict(Z_corr[s4], yhat[s4])
            ev = evaluate(y[s4], pred, None, neg_thr=0.0)
            # also compute MSE
            ev["mse"] = float(((pred - y[s4]) ** 2).mean())
            base_mse = float((base_err ** 2).mean())
            ev["mse_reduction"] = -round((ev["mse"] - base_mse) / max(base_mse, 1e-9) * 100, 1)
            results[nm] = ev
            dur = time.time() - t1
            mae_str = f"{ev['mae']:.4f}"
            print(f"  {nm:25s} MAE={mae_str:>10s}  MSE_rel={ev['mse_reduction']:+.1f}%  [{dur:.0f}s]")
        except Exception as e:
            results[nm] = dict(error=str(e))
            print(f"  {nm:25s} FAILED: {e}")

    base_mse = float((base_err ** 2).mean())
    print(f"  {'Base':25s} MSE={base_mse:.4f}  MAE={np.mean(base_err):.4f}")
    return {ds_key: {bb_name: results}}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--backbone", default="Linear")
    p.add_argument("--peers", default="quantile", help="comma-separated: quantile,crc,...")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    ds_list = LTSF_DATASETS if args.dataset == "all_ltsf" else [args.dataset]
    peer_names = [p.strip() for p in args.peers.split(",")]

    peers = []
    for pn in peer_names:
        factory = PEER_REGISTRY.get(pn)
        if factory is None:
            print(f"Unknown peer: {pn}. Available: {list(PEER_REGISTRY.keys())}")
            return
        peers.append(factory(args))

    for ds_key in ds_list:
        print(f"\n=== {ds_key} × {args.backbone} ===")
        ds = load_ltsf(ds_key)
        run_peer_comparison(ds, ds_key, args.backbone, peers, args)


if __name__ == "__main__":
    main()
