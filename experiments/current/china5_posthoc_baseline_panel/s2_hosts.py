"""Stage 2 — train and freeze the two Hosts per market under the frozen recipe.

The architecture, loss and hyperparameters are the ones already accepted by the
strict GANSU work (`run_canary.host_cfg`), reused verbatim.  Nothing here is
searched per province: only the dataset adapter and the legal feature mapping
differ, which is permitted.

Emits, per market and Host:
  02_hosts/<MARKET>/<HOST>/host_predictions.npz + .json   (frozen HostPredictionCache)
  02_hosts/<MARKET>/<HOST>/FREEZE_MANIFEST.json           (config/commit/hash/split/access)

Run:  python experiments/current/china5_posthoc_baseline_panel/s2_hosts.py [MARKET ...]
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
for p in (HERE, ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import panel_contract as pc                       # noqa: E402
import panel_metrics as pm                        # noqa: E402
from backbones.base import HostConfig             # noqa: E402
from backbones.factory import make_host           # noqa: E402
from utils.benchmark_cache import HostPredictionCache, stable_hash  # noqa: E402
from utils.benchmark_contracts import ForecastWindows               # noqa: E402

OUT = ROOT / "experiments/evidence/china5_posthoc_baseline_panel_20260912"
HOSTS = ("PatchTST", "TimeMixer")


# --------------------------------------------------------- frozen Host recipe
def host_cfg(name: str) -> HostConfig:
    """Verbatim copy of the accepted GANSU_DA Host recipe. Do not edit per province."""
    common = dict(seq_len=pc.SEQ, pred_len=pc.H, channels=1, optimizer="Adam",
                  weight_decay=0.0, scheduler="OneCycleLR", pct_start=.3,
                  patience=100 if name == "PatchTST" else 10, seed=7, device="cpu")
    if name == "PatchTST":
        return HostConfig(**common, epochs=100, batch_size=128, lr=1e-4, official_params={
            "patch_len": 16, "stride": 8, "padding_patch": "end", "revin": 1, "affine": 0,
            "subtract_last": 0, "decomposition": 0, "d_model": 512, "n_heads": 8,
            "e_layers": 2, "d_ff": 2048, "dropout": .05, "fc_dropout": .05, "head_dropout": 0.0})
    return HostConfig(**common, epochs=10, batch_size=128, lr=.01, official_params={
        "d_model": 16, "d_ff": 32, "e_layers": 2, "dropout": .1, "down_sampling_layers": 3,
        "down_sampling_window": 2, "down_sampling_method": "avg", "moving_avg": 25,
        "decomp_method": "moving_avg", "use_norm": 1, "channel_independence": 1,
        "use_future_temporal_feature": 0, "label_len": 0, "embed": "timeF", "freq": "h", "top_k": 5})


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def state_dict_sha(host) -> str:
    h = hashlib.sha256()
    for k, v in sorted(host.model.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest().upper()


def _sub(w: ForecastWindows, seg: str) -> ForecastWindows:
    z = w.segment.astype(str) == seg
    return ForecastWindows(w.context[z], w.target[z], w.timestamp[z], w.segment[z],
                           w.market_id, w.physical_market_group)


# ------------------------------------------------------------------ stage core
def build_host(market: str, name: str, w: ForecastWindows, split_hash: str,
               audit: dict) -> tuple[HostPredictionCache, Path, dict]:
    dest = OUT / "02_hosts" / market / name
    path = dest / "host_predictions.npz"
    if path.exists() and (dest / "FREEZE_MANIFEST.json").exists():
        c = HostPredictionCache.load(path)
        m = c.metadata
        if (m.get("split_hash") != split_hash or not m.get("frozen")
                or str(m.get("array_sha256", "")).upper() != pc.sha256_file(path)
                or m.get("host_recipe_id") != "gansu_da_frozen_recipe_v1"):
            raise RuntimeError(f"{market}/{name}: incompatible frozen Host cache")
        manifest = json.loads((dest / "FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
        return c, path, manifest

    tr, va = _sub(w, "HOST_TRAIN"), _sub(w, "HOST_VAL")
    cfg = host_cfg(name)
    saved = list(sys.path)
    sys.path[:] = [p for p in sys.path if not (p and (Path(p) / "models.py").exists())]
    t0 = time.perf_counter()
    try:
        host = make_host(name, cfg)
        host.fit(tr, va)
        host.freeze()
        pred = host.predict(w)
    finally:
        sys.path[:] = saved
    train_seconds = time.perf_counter() - t0
    if pred.shape != w.target.shape or not np.isfinite(pred).all():
        raise RuntimeError(f"{market}/{name}: non-finite or mis-shaped Host prediction")

    meta = dict(host.metadata)
    meta.update({
        "target_column": pc.MARKETS[market]["target"],
        "track": "china5_DA_to_DA",
        "host_recipe_id": "gansu_da_frozen_recipe_v1",
        "source_path": pc.MARKETS[market]["path"],
        "source_sha256": pc.sha256_file(pc.source_path(market)),
        "split_hash": split_hash,
        "split_manifest": pc.split_manifest(pc.build_day_contract(market), market),
        "host_fit_segment": "HOST_TRAIN", "host_valid_segment": "HOST_VAL",
        "evaluation_segment": "DEV_EVAL", "postprocessor_fit_segment": "POST_TRAIN",
        "fit_n": int(len(tr.context)), "valid_n": int(len(va.context)),
        "sealed_roles_materialized": False,
        "fixed_host_recipe": True, "per_province_search": False,
        "host_config": cfg.__dict__,
        "train_seconds": train_seconds,
        "source_commit": git_commit(),
        "checkpoint_sha256": state_dict_sha(host),
        "access_audit": audit,
    })
    cache = HostPredictionCache(pc.MARKETS[market]["dataset_id"], market, name,
                                w.timestamp, w.context, w.target, pred, w.segment, meta)
    cache.save(dest / "host_predictions.npz")
    saved = json.loads((dest / "host_predictions.json").read_text(encoding="utf-8"))
    cache.metadata.update(saved)

    manifest = {
        "schema": "china5_host_freeze.v1",
        "physical_market_group": market, "dataset_id": pc.MARKETS[market]["dataset_id"],
        "backbone": name, "host_recipe_id": "gansu_da_frozen_recipe_v1",
        "source_commit": meta["source_commit"],
        "source_path": meta["source_path"], "source_sha256": meta["source_sha256"],
        "array_sha256": saved["array_sha256"], "checkpoint_sha256": meta["checkpoint_sha256"],
        "split_hash": split_hash, "split_manifest": meta["split_manifest"],
        "effective_host_config": cfg.__dict__,
        "fit_n": meta["fit_n"], "valid_n": meta["valid_n"],
        "train_seconds": train_seconds,
        "access_manifest": {
            "roles_read": list(pc.OPEN_ROLES),
            "roles_sealed": list(pc.CLOSED_ROLES),
            "target_values_read": audit.get("target_day_values_read", 0),
            "causal_context_values_read": audit.get("causal_context_values_read", 0),
            "protected_final_target_values_read": 0,
            "imputation_used": False,
        },
    }
    pc.dump_json(dest / "FREEZE_MANIFEST.json", manifest)
    return cache, path, manifest


def host_gate(market: str, cache: HostPredictionCache, host_train_mean: float) -> dict:
    """Pre-registered Host sanity/fidelity gate (PROTOCOL_FREEZE.md section 4)."""
    ev = cache.subset("DEV_EVAL")
    tr = cache.subset("HOST_TRAIN")
    va = cache.subset("HOST_VAL")
    mae_ev = float(np.mean(np.abs(ev.y_true - ev.host_pred)))
    mae_tr = float(np.mean(np.abs(tr.y_true - tr.host_pred)))
    mae_va = float(np.mean(np.abs(va.y_true - va.host_pred)))
    mae_const = float(np.mean(np.abs(ev.y_true - host_train_mean)))
    checks = {
        "finite_and_shaped": bool(np.isfinite(cache.host_pred).all()
                                  and cache.host_pred.shape == cache.y_true.shape),
        "beats_train_mean_constant_on_DEV_EVAL": bool(mae_ev < mae_const),
        "HOST_VAL_not_worse_than_3x_HOST_TRAIN": bool(mae_va <= 3.0 * mae_tr),
    }
    return {
        "physical_market_group": market, "backbone": cache.backbone,
        "DEV_EVAL_MAE": mae_ev, "HOST_TRAIN_MAE": mae_tr, "HOST_VAL_MAE": mae_va,
        "DEV_EVAL_MAE_train_mean_constant": mae_const,
        "checks": checks, "passed": all(checks.values()),
        "n_dev_eval_days": int(len(ev.timestamp)), "n_host_train_days": int(len(tr.timestamp)),
        "n_host_val_days": int(len(va.timestamp)),
    }


def main(argv: list[str]) -> int:
    markets = [m for m in (argv or list(pc.MARKETS)) if m in pc.MARKETS]
    registry, gates = [], {}
    for market in markets:
        contract = pc.build_day_contract(market)
        split_hash = stable_hash(pc.split_manifest(contract, market))
        ctx, y, stamps, segs, ds_id, phys, audit = pc.load_target_windows(market, contract)
        w = ForecastWindows(ctx, y, stamps, segs, ds_id, phys)
        train_mean = float(y[segs == "HOST_TRAIN"].mean())
        for name in HOSTS:
            t0 = time.perf_counter()
            cache, path, manifest = build_host(market, name, w, split_hash, audit)
            gate = host_gate(market, cache, train_mean)
            gates[f"{market}__{name}"] = gate
            ev = cache.subset("DEV_EVAL")
            registry.append({
                "physical_market_group": market, "dataset_id": ds_id, "backbone": name,
                "host_recipe_id": manifest["host_recipe_id"],
                "source_commit": manifest["source_commit"],
                "source_sha256": manifest["source_sha256"],
                "array_sha256": manifest["array_sha256"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "split_hash": split_hash,
                "fit_n": manifest["fit_n"], "valid_n": manifest["valid_n"],
                "n_dev_eval_days": int(len(ev.timestamp)),
                "DEV_EVAL_MAE": gate["DEV_EVAL_MAE"],
                "HOST_TRAIN_MAE": gate["HOST_TRAIN_MAE"],
                "HOST_VAL_MAE": gate["HOST_VAL_MAE"],
                "DEV_EVAL_MAE_train_mean_constant": gate["DEV_EVAL_MAE_train_mean_constant"],
                "gate_passed": gate["passed"],
                "host_cache": str(path.relative_to(ROOT)).replace("\\", "/"),
                "wall_seconds_this_run": time.perf_counter() - t0,
            })
            tag = "GATE_PASS" if gate["passed"] else "GATE_FAIL"
            print(f"[S2] {market:9s} {name:10s} {tag}  DEV_EVAL_MAE={gate['DEV_EVAL_MAE']:9.4f} "
                  f"(const {gate['DEV_EVAL_MAE_train_mean_constant']:9.4f})  "
                  f"TRAIN={gate['HOST_TRAIN_MAE']:8.4f} VAL={gate['HOST_VAL_MAE']:8.4f}  "
                  f"{time.perf_counter()-t0:6.1f}s")
    pc.dump_json(OUT / "03_host_gate" / "HOST_GATE.json", {"schema": "china5_host_gate.v1", "cells": gates})
    import csv
    with (OUT / "02_hosts" / "HOST_REGISTRY.csv").open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(registry[0]))
        wtr.writeheader()
        wtr.writerows(registry)
    print(f"[S2] wrote {OUT/'02_hosts'/'HOST_REGISTRY.csv'} and 03_host_gate/HOST_GATE.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
