"""P2 — retrain all 20 Hosts under the common-benchmark 70/10/20 split.

`COMMON_BENCHMARK_701020_V1` changes the data budget, so every prior Host checkpoint is
protocol-mismatched and is not reusable.  This module trains each of the 20
market x Host coordinates from scratch with the *already-frozen* architecture/config
recipe and nothing else:

* scaler fit on TRAIN only (`TorchHostAdapter._scale_fit(training windows)`);
* training windows = TRAIN only;
* checkpoint selection / early stopping = VAL only (`fit(train, valid=VAL)`);
* no hyperparameter, width, epoch-budget or seed choice is made from a VAL outcome —
  the config is a module-level constant per Host, identical in all five markets;
* prediction is taken on the TRAIN+VAL windows only, so a TEST target value is never
  materialised (the loader is `cb_contracts.load_pretest_windows`).

Recipe provenance is unchanged from the two admitted authorities: PatchTST/TimeMixer use
the frozen `gansu_da_frozen_recipe_v1` (`run_canary.host_cfg`, reused verbatim by
`s2_hosts.host_cfg`); iTransformer and LSTM use the breadth stage's `new_host_cfg`
(source-native official iTransformer / shared-contract LSTM).  Only the *partition
labels* change, which is the point of the new protocol.

The Host gate of PROTOCOL §8 E2 is applied unchanged, with one disclosed adaptation:
the old contract's held-out region for the gate was `DEV_EVAL`, which the new
three-role contract does not have.  The gate therefore runs on VAL, which under the
common benchmark is *also* the selection region — so the constant-predictor check is
weaker than it was and is recorded as a scope caveat, not silently redefined.

Run:  python .../cb_p2_hosts.py [--force] [MARKET ...]
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

IMPL = Path(__file__).resolve().parent
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))

import cb_contracts as CB                                                    # noqa: E402

# Bound before any vendor snapshot can displace the top-level name `utils`.
from utils.benchmark_cache import HostPredictionCache                        # noqa: E402

BREADTH_IMPL = CB.BREADTH_IMPL
if str(BREADTH_IMPL) not in sys.path:
    sys.path.insert(0, str(BREADTH_IMPL))

from vendor_guard import vendor_bindings                                     # noqa: E402
import hosts as H                                                            # noqa: E402
from backbones.base import HostConfig                                        # noqa: E402
from backbones.factory import make_host                                      # noqa: E402

EV = CB.EV
HOST_ROOT = EV / "02_hosts"
CONTRACT_THREADS = 1

# Which frozen recipe authority supplies each Host's config, and where the Host's
# declared width comes from (the frozen transfer interface reads
# `official_config.d_model`, so it must be present for every Host).
RECIPE = {
    "PatchTST": ("gansu_da_frozen_recipe_v1", 512,
                 "run_canary.host_cfg / s2_hosts.host_cfg, verbatim (official_params.d_model)"),
    "TimeMixer": ("gansu_da_frozen_recipe_v1", 16,
                  "run_canary.host_cfg / s2_hosts.host_cfg, verbatim (official_params.d_model)"),
    "iTransformer": ("breadth_new_host_itransformer_v1", 256,
                     "hosts.new_host_cfg('iTransformer') (official_params.d_model = 256)"),
    "LSTM": ("breadth_new_host_lstm_v1", 64,
             "hosts.new_host_cfg('LSTM'); declared width := LSTM hidden size 64"),
}

GATE_FIELDS = ["protocol_id", "market", "dataset_id", "host", "family",
               "split_manifest_hash", "train_id_sha256", "val_id_sha256",
               "test_target_blind_id_sha256", "source_path", "source_sha256",
               "host_recipe_id", "n_train_days", "n_val_days",
               "HOST_TRAIN_MAE", "HOST_VAL_MAE", "VAL_MAE_train_mean_constant",
               "finite_and_shaped", "beats_train_mean_constant_on_VAL",
               "HOST_VAL_not_worse_than_3x_HOST_TRAIN", "gate_passed",
               "gate_held_out_region", "gate_scope_caveat",
               "scaler_fit_partition", "selection_partition", "test_target_values_read",
               "array_sha256", "checkpoint_sha256", "host_cache_path",
               "train_seconds", "torch_num_threads", "reuse", "note"]


def _rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(CB.ROOT)).replace("\\", "/")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()


def _sub(w, seg: str):
    from utils.benchmark_contracts import ForecastWindows
    z = w.segment.astype(str) == seg
    if int(z.sum()) == 0:
        raise RuntimeError(f"role {seg} is empty in the loaded windows")
    return ForecastWindows(w.context[z], w.target[z], w.timestamp[z], w.segment[z],
                           w.market_id, w.physical_market_group)


def state_dict_sha(host) -> str:
    h = hashlib.sha256()
    for k, v in sorted(host.model.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest().upper()


def frozen_cfg(host_name: str, market: str) -> HostConfig:
    """The frozen recipe for one Host.  Identical in every market — no per-market search."""
    import run_canary as gs
    if host_name in ("PatchTST", "TimeMixer"):
        return gs.host_cfg(host_name)
    if host_name == "iTransformer":
        return H.new_host_cfg("iTransformer")
    if host_name == "LSTM":
        return H.new_host_cfg("LSTM")
    raise KeyError(f"host not in the common-benchmark coordinate system: {host_name!r}")


def official_config_for(host_name: str, cfg: HostConfig) -> dict:
    oc = dict(cfg.official_params)
    oc["d_model"] = int(RECIPE[host_name][1])
    return oc


def split_hashes(market: str) -> tuple[dict, str]:
    """The frozen P0 manifest plus its hash, cross-checked against `SPLIT_INDEX.json`."""
    man = CB.split_manifest(market)
    h = CB.manifest_hash(man)
    idx = json.loads((EV / "01_splits/SPLIT_INDEX.json").read_text(encoding="utf-8"))
    rec = idx["markets"][market]
    if rec["split_manifest_sha256"] != h:
        raise RuntimeError(f"{market}: split manifest hash {h} does not match the frozen "
                           f"P0 index {rec['split_manifest_sha256']}")
    return man, h


def host_gate(cache, market: str, w, train_mean: float) -> dict:
    """PROTOCOL §8 E2, applied unchanged on the new roles (TRAIN / VAL)."""
    seg = w.segment.astype(str)
    tr, va = seg == "TRAIN", seg == "VAL"
    y = np.asarray(w.target, dtype=np.float64)
    p = np.asarray(cache.host_pred, dtype=np.float64)
    if p.shape != y.shape:
        raise RuntimeError(f"{market}/{cache.backbone}: gate received {p.shape}, expected {y.shape}")
    mae_tr = float(np.mean(np.abs(y[tr] - p[tr])))
    mae_va = float(np.mean(np.abs(y[va] - p[va])))
    mae_const = float(np.mean(np.abs(y[va] - train_mean)))
    checks = {
        "finite_and_shaped": bool(np.isfinite(p).all() and p.shape == y.shape),
        "beats_train_mean_constant_on_VAL": bool(mae_va < mae_const),
        "HOST_VAL_not_worse_than_3x_HOST_TRAIN": bool(mae_va <= 3.0 * mae_tr),
    }
    return {"n_train_days": int(tr.sum()), "n_val_days": int(va.sum()),
            "HOST_TRAIN_MAE": mae_tr, "HOST_VAL_MAE": mae_va,
            "VAL_MAE_train_mean_constant": mae_const, **checks,
            "gate_passed": all(checks.values()),
            "gate_held_out_region": "VAL",
            "gate_scope_caveat": (
                "under the old five-role contract the gate's held-out region (DEV_EVAL) was "
                "never used for selection; the common benchmark has no fourth open role, so "
                "the gate runs on VAL, which is also the checkpoint-selection region — the "
                "constant-predictor check is therefore weaker and no TEST value is used")}


def train_one(market: str, host_name: str, w, man: dict, split_h: str,
              audit: dict, force: bool = False) -> dict:
    import torch

    import run_canary as gs
    dest = HOST_ROOT / market / host_name
    npz = dest / "host_predictions.npz"
    freeze = dest / "FREEZE_MANIFEST.json"
    recipe_id, _width, width_basis = RECIPE[host_name]

    if npz.exists() and freeze.exists() and not force:
        rec = json.loads(freeze.read_text(encoding="utf-8"))
        cache = HostPredictionCache.load(npz)
        stale = (rec.get("protocol_id") != CB.PROTOCOL_ID
                 or rec.get("split_manifest_hash") != split_h
                 or rec.get("host_recipe_id") != recipe_id
                 or str(rec.get("array_sha256", "")).upper() != _sha(npz))
        note = ("reused: hash-verified protocol-scoped checkpoint "
                "(no retraining, no TEST read)")
        if stale and (rec.get("protocol_id") != CB.PROTOCOL_ID
                      or rec.get("split_manifest_hash") != split_h):
            raise RuntimeError(f"{market}/{host_name}: existing artifact is protocol-mismatched; "
                               f"refusing to reuse (re-run with --force)")
        if not stale:
            gate = host_gate(cache, market, w, float(np.asarray(
                w.target, dtype=np.float64)[w.segment.astype(str) == "TRAIN"].mean()))
            gate.update({"protocol_id": CB.PROTOCOL_ID, "market": market,
                         "dataset_id": man["dataset_id"], "host": host_name,
                         "family": man["family"], "split_manifest_hash": split_h,
                         "train_id_sha256": man["TRAIN_id_sha256"],
                         "val_id_sha256": man["VAL_id_sha256"],
                         "test_target_blind_id_sha256": man["test_target_blind_id_sha256"],
                         "source_path": man["source"], "source_sha256": man["source_sha256"],
                         "host_recipe_id": recipe_id,
                         "scaler_fit_partition": "TRAIN", "selection_partition": "VAL",
                         "test_target_values_read": int(audit["test_target_values_read"]),
                         "array_sha256": _sha(npz),
                         "checkpoint_sha256": rec.get("checkpoint_sha256", ""),
                         "host_cache_path": _rel(npz),
                         "train_seconds": rec.get("train_seconds", 0.0),
                         "torch_num_threads": CONTRACT_THREADS, "reuse": True, "note": note})
            print(f"[P2] {market:12s} {host_name:12s} REUSED   "
                  f"VAL_MAE={gate['HOST_VAL_MAE']:9.4f} gate={gate['gate_passed']}")
            return gate

    tr, va = _sub(w, "TRAIN"), _sub(w, "VAL")
    cfg = frozen_cfg(host_name, market)
    train_mean = float(np.asarray(w.target, dtype=np.float64)[
        w.segment.astype(str) == "TRAIN"].mean())

    torch.set_num_threads(CONTRACT_THREADS)
    assert torch.get_num_threads() == CONTRACT_THREADS, "thread pinning contract violated"
    saved_path = list(sys.path)
    sys.path[:] = [p for p in sys.path if not (p and (Path(p) / "models.py").exists())]
    t0 = time.perf_counter()
    try:
        with vendor_bindings():
            host = (H.make_new_host(host_name, cfg) if host_name in ("iTransformer", "LSTM")
                    else make_host(host_name, cfg))
            host.fit(tr, va)          # scaler + training = TRAIN; early stop = VAL
            host.freeze()
            pred = np.asarray(host.predict(w))   # w = TRAIN+VAL only
    finally:
        sys.path[:] = saved_path
    train_seconds = time.perf_counter() - t0

    if pred.shape != w.target.shape or not np.isfinite(pred).all():
        raise RuntimeError(f"{market}/{host_name}: non-finite or mis-shaped Host prediction")

    meta = dict(host.metadata)
    meta["backbone"] = host_name          # source-native subclass would record its own class name
    meta.update({
        "schema_version": "host_prediction_cache.v1",
        "protocol_id": CB.PROTOCOL_ID,
        "target_column": man["target_column"],
        "track": "china5_common_benchmark_DA_to_DA",
        "frozen": True, "new_execution": True,
        "source_path": man["source"], "source_sha256": man["source_sha256"],
        "split_protocol_id": CB.PROTOCOL_ID,
        "split_hash": split_h, "split_manifest_hash": split_h,
        "split_manifest": man,
        "role_vocabulary": "COMMON_BENCHMARK_TRAIN_VAL_TEST",
        "host_fit_segment": "TRAIN", "host_valid_segment": "VAL",
        "scaler_fit_partition": "TRAIN", "selection_partition": "VAL",
        "evaluation_segment": "TEST (sealed during PRETEST)",
        "fit_n": int(len(tr.context)), "valid_n": int(len(va.context)),
        "host_train_budget": int(len(tr.context)), "host_val_budget": int(len(va.context)),
        "sealed_roles_materialized": False,
        "test_target_values_read": int(audit["test_target_values_read"]),
        "protected_final_read": 0,
        "host_recipe_id": recipe_id,
        "host_config": dict(cfg.__dict__),
        "official_config": official_config_for(host_name, cfg),
        "official_config_width_basis": width_basis,
        "torch_num_threads": CONTRACT_THREADS,
        "thread_pinning_contract": "torch.set_num_threads(1), the canonical frozen runner setting",
        "train_seconds": train_seconds,
        "source_commit": _git_commit(),
        "source_file_sha256": (H.source_native_source_hashes()
                               if host_name == "iTransformer" else {}),
        "access_audit": audit,
        "per_market_hyperparameter_search": False,
        "old_low_shot_checkpoint_reused": False,
    })
    dest.mkdir(parents=True, exist_ok=True)
    cache = HostPredictionCache(w.market_id, w.physical_market_group, host_name,
                                w.timestamp, w.context, w.target, pred, w.segment, meta)
    data_path, _ = cache.save(dest / "host_predictions.npz")
    saved_json = json.loads((dest / "host_predictions.json").read_text(encoding="utf-8"))
    checkpoint_sha = state_dict_sha(host)

    manifest = {
        "schema": "china5_common_benchmark_host_freeze.v1",
        "protocol_id": CB.PROTOCOL_ID,
        "market": market, "dataset_id": man["dataset_id"], "family": man["family"],
        "backbone": host_name, "host_recipe_id": recipe_id,
        "new_execution": True, "imported_frozen": False,
        "old_low_shot_checkpoint_reused": False,
        "source_path": man["source"], "source_sha256": man["source_sha256"],
        "split_protocol_id": CB.PROTOCOL_ID,
        "split_manifest_hash": split_h,
        "train_id_sha256": man["TRAIN_id_sha256"], "val_id_sha256": man["VAL_id_sha256"],
        "test_target_blind_id_sha256": man["test_target_blind_id_sha256"],
        "host_train_budget": int(len(tr.context)), "host_val_budget": int(len(va.context)),
        "scaler_fit_partition": "TRAIN", "selection_partition": "VAL",
        "array_sha256": str(saved_json["array_sha256"]).upper(),
        "hash_definition": "sha256 of the whole .npz container bytes as written",
        "canonical_array_sha256": hashlib.sha256(
            np.ascontiguousarray(pred).tobytes()).hexdigest().upper(),
        "checkpoint_sha256": checkpoint_sha,
        "effective_host_config": dict(cfg.__dict__),
        "official_config": meta["official_config"],
        "official_config_width_basis": width_basis,
        "fit_n": meta["fit_n"], "valid_n": meta["valid_n"],
        "train_seconds": train_seconds,
        "torch_num_threads": CONTRACT_THREADS,
        "source_commit": meta["source_commit"],
        "access_manifest": {
            "roles_read": ["TRAIN", "VAL"],
            "roles_sealed": ["TEST"],
            "partitions_never_touched": ["TEST", "PROTECTED_FINAL", "S3", "S4"],
            "test_target_values_read": int(audit["test_target_values_read"]),
            "sealed_final_target_values_read": int(audit["sealed_final_target_values_read"]),
            "imputation_used": False,
        },
    }
    CB.dump_json(freeze, manifest)

    gate = host_gate(cache, market, w, train_mean)
    gate.update({"protocol_id": CB.PROTOCOL_ID, "market": market,
                 "dataset_id": man["dataset_id"], "host": host_name, "family": man["family"],
                 "split_manifest_hash": split_h,
                 "train_id_sha256": man["TRAIN_id_sha256"],
                 "val_id_sha256": man["VAL_id_sha256"],
                 "test_target_blind_id_sha256": man["test_target_blind_id_sha256"],
                 "source_path": man["source"], "source_sha256": man["source_sha256"],
                 "host_recipe_id": recipe_id,
                 "scaler_fit_partition": "TRAIN", "selection_partition": "VAL",
                 "test_target_values_read": int(audit["test_target_values_read"]),
                 "array_sha256": _sha(data_path), "checkpoint_sha256": checkpoint_sha,
                 "host_cache_path": _rel(data_path),
                 "train_seconds": round(train_seconds, 2),
                 "torch_num_threads": CONTRACT_THREADS, "reuse": False,
                 "note": "NEW_EXECUTION under COMMON_BENCHMARK_701020_V1"})
    tag = "GATE_PASS" if gate["gate_passed"] else "GATE_FAIL"
    print(f"[P2] {market:12s} {host_name:12s} {tag}  VAL_MAE={gate['HOST_VAL_MAE']:9.4f} "
          f"(const {gate['VAL_MAE_train_mean_constant']:9.4f})  "
          f"TRAIN={gate['HOST_TRAIN_MAE']:8.4f}  {train_seconds:6.1f}s")
    return gate


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=CB.ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def main(argv: list[str]) -> int:
    markets = [a for a in argv if a in CB.MARKETS] or list(CB.MARKETS)
    force = "--force" in argv
    rows = []
    for market in markets:
        man, split_h = split_hashes(market)
        w, audit = CB.load_pretest_windows(market)
        if int(audit["test_target_values_read"]) != 0:
            raise RuntimeError(f"{market}: the PRETEST loader materialised a TEST target value")
        if int(audit["sealed_final_target_values_read"]) != 0:
            raise RuntimeError(f"{market}: the PRETEST loader touched the sealed final role")
        for host_name in CB.HOSTS:
            rows.append(train_one(market, host_name, w, man, split_h, audit, force=force))

    rows.sort(key=lambda r: (r["market"], r["host"]))
    out = HOST_ROOT / "EXECUTION_HOST_GATE.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=GATE_FIELDS)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r.get(k, "") for k in GATE_FIELDS})

    n_pass = sum(1 for r in rows if r["gate_passed"])
    CB.dump_json(HOST_ROOT / "HOST_FREEZE_INDEX.json", {
        "schema": "china5_common_benchmark_host_freeze_index.v1",
        "protocol_id": CB.PROTOCOL_ID,
        "stage_id": "hch_china5_common_benchmark_701020",
        "n_cells": len(rows),
        "n_gate_pass": n_pass,
        "test_label_read_count": CB.PROCESS_SEAL.test_label_read_count,
        "cells": [{"market": r["market"], "host": r["host"],
                   "host_recipe_id": r["host_recipe_id"],
                   "split_manifest_hash": r["split_manifest_hash"],
                   "array_sha256": r["array_sha256"],
                   "checkpoint_sha256": r["checkpoint_sha256"],
                   "host_cache_path": r["host_cache_path"],
                   "gate_passed": bool(r["gate_passed"]),
                   "HOST_VAL_MAE": r["HOST_VAL_MAE"]} for r in rows],
    })
    print(f"\n[P2] {n_pass}/{len(rows)} Host gates PASS; "
          f"TEST-label read count = {CB.PROCESS_SEAL.test_label_read_count}")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
