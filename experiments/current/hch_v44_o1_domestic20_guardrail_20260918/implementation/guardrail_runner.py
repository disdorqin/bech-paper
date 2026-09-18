"""Phase D — China-5 x 4-Host O1 breadth, then the D0-D4 domestic gate.

Execution order: reference check -> 36 new fits -> aggregation -> gate.  Phase M
and Phase I live in their own modules and are driven by ``run_all.py``.

The 36 new fits call ``probe_train.train_o1`` unchanged from the closed O1 probe
stage.  Nothing in this module can alter the trainer, the data path or the
registered schedule.
"""
from __future__ import annotations

import datetime as dt
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import guardrail_common as G

U = G.U
PC = G.PC
PT = G.PT
EVID = G.EVID

PER_SEED_FIELDS = (
    "market", "host", "cell", "status", "seed", "selected_step", "selected_check",
    "val_mae_ema", "val_mae_raw", "val_host_mae", "gain_host_pct", "beats_host",
    "seed_worse_than_host_by_gt_1pct", "val_correction_ratio", "val_mean_abs_correction",
    "selected_after_switch", "diag_reconstruction_mae", "level_mae", "mass_mae", "shape_w1",
    "L_rec", "L_b", "L_B", "L_S", "stopped_at_step", "history_support_hash",
    "scales_fingerprint", "probe_code_hash", "core_tree", "test_target_read_count",
)

CELL_FIELDS = (
    "market", "host", "cell", "status", "n_seeds", "o1_median_mae", "o1_min_mae",
    "o1_max_mae", "host_mae", "gain_host_pct", "strict_host_positive",
    "seeds_beating_host", "seeds_worse_than_host_by_gt_1pct", "at_least_2_of_3_seeds_beat_host",
    "gain_ge_minus_0p5pct", "gain_ge_plus_2pct", "median_selected_step",
    "median_correction_ratio", "median_paired_gain_pct",
)

MARKET_FIELDS = (
    "market", "n_cells", "market_median_gain_pct", "market_mean_gain_pct",
    "n_strict_host_positive", "n_ge_plus_2pct", "best_cell_gain_pct", "worst_cell_gain_pct",
    "market_median_positive", "market_median_ge_plus_1pct",
)


# ------------------------------------------------------------------ phase D-1
def phase_reference_check() -> dict:
    """Pin the 24 reused runs and the 12 registered new cells before any fit."""
    prov = G.reuse_provenance()
    U.json_dump(EVID / "REUSE_PROVENANCE.json", prov)
    if not prov["all_reused_present"]:
        raise RuntimeError(f"REFERENCE_NOT_AVAILABLE: {prov['missing'][:4]}")
    broken = [
        f"{k}__seed{s}"
        for k, cell in prov["reused_cells"].items()
        for s, run in cell["runs"].items()
        if not (run["checkpoint_sha256_matches_own_record"]
                and run["history_support_matches_shadow_artifact"])
    ]
    if broken:
        raise RuntimeError(f"REUSE_PIN_MISMATCH: {broken[:4]}")
    if not prov["one_probe_code_hash_across_reused"]:
        raise RuntimeError("REUSE_CODE_STATE_MISMATCH")
    if len(G.NEW_CELLS) != 12:
        raise RuntimeError(f"NEW_CELL_COUNT_MISMATCH({len(G.NEW_CELLS)})")
    if len(G.DOM_PANEL) != 20:
        raise RuntimeError(f"PANEL_SIZE_MISMATCH({len(G.DOM_PANEL)})")
    return prov


# ------------------------------------------------------------------ phase D-2
def _init_worker() -> None:
    """Put this stage's own implementation directory on a spawned child's path.

    The inherited stages prepend their own implementation directories to
    ``sys.path``, one of which already contains a module called ``runner``, so
    this stage's modules are named distinctively and the directory is re-asserted
    explicitly in every child rather than inferred from the spawn handshake.
    """
    impl = str(Path(__file__).resolve().parent)
    if impl not in sys.path:
        sys.path.insert(0, impl)


def fit_task(args) -> dict:
    market, host, seed = args
    out = G.new_run_dir(market, host, seed)
    # All three artifacts, not just the checkpoint: ``training_curve.json`` is
    # written last, so requiring it means a killed fit can never be mistaken for
    # a finished one.
    if all((out / f).is_file() for f in G.RUN_FIELDS):
        return {"cell": G.cell_key(market, host), "seed": seed, "status": "skipped_existing"}
    G.RC.worker_env()
    started = dt.datetime.now().isoformat()
    G.PT.train_o1(market, host, seed, out, device="cuda")
    return {
        "cell": G.cell_key(market, host), "seed": seed, "status": "fitted",
        "started": started, "finished": dt.datetime.now().isoformat(),
    }


def phase_fits(workers: int = 2) -> list[dict]:
    """Exactly the 36 registered new fits, no more and no fewer."""
    todo = [(m, h, s) for m, h in G.NEW_CELLS for s in G.SEEDS]
    log_path = EVID / "FIT_LOG.json"
    results = []
    if workers > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_init_worker) as pool:
            for res in pool.map(fit_task, todo):
                results.append(res)
                print(f"  fit {res['cell']} seed{res['seed']}: {res['status']}", flush=True)
    else:
        for args in todo:
            res = fit_task(args)
            results.append(res)
            print(f"  fit {res['cell']} seed{res['seed']}: {res['status']}", flush=True)
    payload = {
        "schema": "hch_v44_o1_domestic20_fit_log.v1",
        "protocol_id": G.PROTOCOL_ID,
        "n_registered_new_fits": len(todo),
        "n_fitted": sum(1 for r in results if r["status"] == "fitted"),
        "n_skipped_existing": sum(1 for r in results if r["status"] == "skipped_existing"),
        "attempts_per_coordinate": {f"{r['cell']}__seed{r['seed']}": 1 for r in results},
        "no_retry": len(results) == len(todo),
        "workers": int(workers),
        "finished": dt.datetime.now().isoformat(),
        "results": results,
    }
    U.json_dump(log_path, payload)
    return results


# ------------------------------------------------------------------ phase D-3
def _row_of(market: str, host: str, seed: int) -> dict:
    d = G.run_dir(market, host, seed)
    fr = G.load_json(d / "freeze.json")
    sm = fr["selected_metrics"]
    host_mae = float(sm["val_host_mae"])
    val = float(fr["selected_val_mae_ema"])
    gain = 100.0 * (host_mae - val) / host_mae
    return {
        "market": market, "host": host, "cell": G.cell_key(market, host),
        "status": "reused" if (market, host) in G.REUSED_CELLS else "new",
        "seed": seed,
        "selected_step": int(fr["selected_step"]), "selected_check": int(fr["selected_check"]),
        "val_mae_ema": val, "val_mae_raw": float(sm["val_mae_raw"]), "val_host_mae": host_mae,
        "gain_host_pct": gain, "beats_host": val < host_mae,
        "seed_worse_than_host_by_gt_1pct": gain < -1.0,
        "val_correction_ratio": float(sm["val_correction_ratio"]),
        "val_mean_abs_correction": float(sm["val_mean_abs_correction"]),
        "selected_after_switch": bool(fr["selected_after_switch"]),
        "diag_reconstruction_mae": float(sm["diag_reconstruction_mae"]),
        "level_mae": float(sm["level_mae"]), "mass_mae": float(sm["mass_mae"]),
        "shape_w1": float(sm["shape_w1"]),
        "L_rec": float(sm["L_rec"]), "L_b": float(sm["L_b"]),
        "L_B": float(sm["L_B"]), "L_S": float(sm["L_S"]),
        "stopped_at_step": fr["optimization"]["stopped_at_step"],
        "history_support_hash": fr["history_support_hash"],
        "scales_fingerprint": fr["scales"]["fingerprint"],
        "probe_code_hash": fr["probe_code_hash"],
        "core_tree": fr["source_tree_digest"]["core_tree"],
        "test_target_read_count": int(fr["test_target_read_count"]),
        "_freeze": fr,
    }


def phase_domestic() -> dict:
    """60-run seed table, 20-cell median table, 5-market summary and D0-D4."""
    per_seed = [_row_of(m, h, s) for m, h in G.DOM_PANEL for s in G.SEEDS]
    if len(per_seed) != 60:
        raise RuntimeError(f"COORDINATE_COUNT_MISMATCH({len(per_seed)})")
    G.write_csv(EVID / "DOMESTIC_PER_SEED.csv", PER_SEED_FIELDS, per_seed)

    cells = []
    for market, host in G.DOM_PANEL:
        rows = [r for r in per_seed if r["market"] == market and r["host"] == host]
        if len(rows) != len(G.SEEDS):
            raise RuntimeError(f"{G.cell_key(market, host)}: {len(rows)} seeds")
        host_maes = {round(r["val_host_mae"], 9) for r in rows}
        if len(host_maes) != 1:
            raise RuntimeError(f"{G.cell_key(market, host)}: Host MAE not deterministic {host_maes}")
        host_mae = rows[0]["val_host_mae"]
        med = G.median([r["val_mae_ema"] for r in rows])
        gain = 100.0 * (host_mae - med) / host_mae
        n_beat = sum(1 for r in rows if r["beats_host"])
        n_worse1 = sum(1 for r in rows if r["seed_worse_than_host_by_gt_1pct"])
        cells.append({
            "market": market, "host": host, "cell": G.cell_key(market, host),
            "status": rows[0]["status"], "n_seeds": len(rows),
            "o1_median_mae": med,
            "o1_min_mae": min(r["val_mae_ema"] for r in rows),
            "o1_max_mae": max(r["val_mae_ema"] for r in rows),
            "host_mae": host_mae, "gain_host_pct": gain,
            "strict_host_positive": gain > 0.0,
            "seeds_beating_host": n_beat,
            "seeds_worse_than_host_by_gt_1pct": n_worse1,
            "at_least_2_of_3_seeds_beat_host": n_beat >= 2,
            "gain_ge_minus_0p5pct": gain >= -0.5,
            "gain_ge_plus_2pct": gain >= 2.0,
            "median_selected_step": G.median([r["selected_step"] for r in rows]),
            "median_correction_ratio": G.median([r["val_correction_ratio"] for r in rows]),
            "median_paired_gain_pct": G.median([r["gain_host_pct"] for r in rows]),
        })
    G.write_csv(EVID / "DOMESTIC_CELL_MEDIANS.csv", CELL_FIELDS, cells)

    markets = []
    for market in G.MARKETS:
        mc = [c for c in cells if c["market"] == market]
        med = G.median([c["gain_host_pct"] for c in mc])
        markets.append({
            "market": market, "n_cells": len(mc), "market_median_gain_pct": med,
            "market_mean_gain_pct": float(sum(c["gain_host_pct"] for c in mc) / len(mc)),
            "n_strict_host_positive": sum(1 for c in mc if c["strict_host_positive"]),
            "n_ge_plus_2pct": sum(1 for c in mc if c["gain_ge_plus_2pct"]),
            "best_cell_gain_pct": max(c["gain_host_pct"] for c in mc),
            "worst_cell_gain_pct": min(c["gain_host_pct"] for c in mc),
            "market_median_positive": med > 0.0,
            "market_median_ge_plus_1pct": med >= 1.0,
        })
    G.write_csv(EVID / "DOMESTIC_MARKET_SUMMARY.csv", MARKET_FIELDS, markets)

    # Host-relative diagnostics: the paired, seed-level view the design asks for
    # as supplementary evidence.  It never defines a gate.
    diag = []
    for market, host in G.DOM_PANEL:
        rows = [r for r in per_seed if r["market"] == market and r["host"] == host]
        diag.append({
            "market": market, "host": host, "cell": G.cell_key(market, host),
            "host_mae": rows[0]["val_host_mae"],
            "median_paired_gain_pct": G.median([r["gain_host_pct"] for r in rows]),
            "min_paired_gain_pct": min(r["gain_host_pct"] for r in rows),
            "max_paired_gain_pct": max(r["gain_host_pct"] for r in rows),
            "median_gain_pct": next(c["gain_host_pct"] for c in cells
                                    if c["cell"] == G.cell_key(market, host)),
            "seeds_beating_host": sum(1 for r in rows if r["beats_host"]),
            "median_selected_after_switch_count": sum(1 for r in rows if r["selected_after_switch"]),
        })
    G.write_csv(EVID / "DOMESTIC_HOST_DIAGNOSTICS.csv", tuple(diag[0]), diag)

    gate = domestic_gate(per_seed, cells, markets)
    U.json_dump(EVID / "DOMESTIC_GATE.json", gate)
    return {"per_seed": per_seed, "cells": cells, "markets": markets, "gate": gate}


def _settings_of(freeze: dict) -> dict:
    return {k: freeze["optimization"].get(k) for k in PC.REGISTERED_SCHEDULE_KEYS}


def domestic_gate(per_seed, cells, markets) -> dict:
    """D0-D4 exactly as registered.  No threshold is derived from the results."""
    gains = [c["gain_host_pct"] for c in cells]
    panel_median = G.median(gains)

    reused_after = G.reuse_snapshot()
    before = G.load_json(EVID / "REUSE_SNAPSHOT_BEFORE.json") if (EVID / "REUSE_SNAPSHOT_BEFORE.json").is_file() else {}
    changed = sorted(k for k in set(before) | set(reused_after)
                     if before.get(k) != reused_after.get(k))

    code_hashes = {r["probe_code_hash"] for r in per_seed}
    core_trees = {r["core_tree"] for r in per_seed}
    rc_hashes = {r["_freeze"]["recovery_common_sha256"] for r in per_seed}
    rt_hashes = {r["_freeze"]["recovery_train_sha256"] for r in per_seed}
    schedules = {json.dumps(_settings_of(r["_freeze"]), sort_keys=True) for r in per_seed}
    supports_ok = all(
        r["history_support_hash"] == G.load_json(
            U.EVID / "shadow_oof" / f"{r['cell']}.json")["support_hash"]
        for r in per_seed)
    finite = all(
        all(isinstance(r[k], float) and r[k] == r[k] and abs(r[k]) != float("inf")
            for k in ("val_mae_ema", "val_mae_raw", "val_host_mae", "val_correction_ratio"))
        for r in per_seed)
    n_reused = sum(1 for r in per_seed if r["status"] == "reused")
    n_new = sum(1 for r in per_seed if r["status"] == "new")

    d0_checks = {
        "exactly_60_coordinates": len(per_seed) == 60,
        "exactly_24_reused_runs": n_reused == 24,
        "exactly_36_new_fits": n_new == 36,
        "no_reused_run_changed": not changed,
        "one_scientific_code_state": len(code_hashes) == 1,
        "core_tree_equals_frozen_substrate": core_trees == {"6F18C0E4241A299E32C8D8A617061BAEAA7B8DC98B4EE069DC67199B87531577"},
        "recovery_modules_identical_across_runs": len(rc_hashes) == 1 and len(rt_hashes) == 1,
        "support_matches_frozen_substrate": supports_ok,
        "one_registered_schedule_across_all_runs": len(schedules) == 1,
        "all_metrics_finite": finite,
        "test_target_read_count_zero": all(r["test_target_read_count"] == 0 for r in per_seed),
        "no_retry": G.load_json(EVID / "FIT_LOG.json")["no_retry"] if (EVID / "FIT_LOG.json").is_file() else False,
    }
    d1_checks = {
        "strict_host_positive_ge_16_of_20": sum(1 for c in cells if c["strict_host_positive"]) >= 16,
        "gain_ge_minus_0p5pct_ge_19_of_20": sum(1 for c in cells if c["gain_ge_minus_0p5pct"]) >= 19,
        "worst_gain_ge_minus_1p0pct": min(gains) >= -1.0,
    }
    d2_checks = {
        "panel_median_ge_plus_1p5pct": panel_median >= 1.5,
        "cells_ge_plus_2pct_ge_8_of_20": sum(1 for c in cells if c["gain_ge_plus_2pct"]) >= 8,
    }
    d3_checks = {
        "all_5_market_medians_positive": all(m["market_median_positive"] for m in markets),
        "market_medians_ge_plus_1pct_ge_4_of_5": sum(1 for m in markets if m["market_median_ge_plus_1pct"]) >= 4,
    }
    d4_checks = {
        "cells_with_2_of_3_seeds_beating_host_ge_15_of_20":
            sum(1 for c in cells if c["at_least_2_of_3_seeds_beat_host"]) >= 15,
        "zero_cells_with_all_three_seeds_worse_by_gt_1pct":
            sum(1 for c in cells if c["seeds_worse_than_host_by_gt_1pct"] == 3) == 0,
    }
    blocks = {"D0": d0_checks, "D1": d1_checks, "D2": d2_checks, "D3": d3_checks, "D4": d4_checks}
    passed = {k: all(v.values()) for k, v in blocks.items()}
    return {
        "schema": "hch_v44_o1_domestic20_gate.v1",
        "protocol_id": G.PROTOCOL_ID,
        "recipe_id": G.RECIPE_ID,
        "preregistered_before_any_new_fit": True,
        "is_sota_claim": False,
        "primary_statistic": "median VAL MAE across seeds 7/17/37 per cell, then relative gain vs the deterministic frozen Host MAE of the same cell",
        "secondary_statistic": "median paired seed-level relative gain (reported, never a gate input)",
        "panel_median_gain_pct": panel_median,
        "panel_median_paired_gain_pct": G.median([c["median_paired_gain_pct"] for c in cells]),
        "n_cells": len(cells), "n_runs": len(per_seed),
        "n_strict_host_positive": sum(1 for c in cells if c["strict_host_positive"]),
        "n_ge_plus_2pct": sum(1 for c in cells if c["gain_ge_plus_2pct"]),
        "best_cell_gain_pct": max(gains), "worst_cell_gain_pct": min(gains),
        "n_seeds_beating_host": sum(1 for r in per_seed if r["beats_host"]),
        "reused_artifacts_changed": changed,
        "market_medians": {m["market"]: m["market_median_gain_pct"] for m in markets},
        "checks": blocks,
        "passed": passed,
        "n_blocks_passed": sum(passed.values()),
        "domestic_pass": all(passed.values()),
        "test_target_read_count": 0,
        "note": ("D0-D4 are development gates for deciding whether to spend an untouched "
                 "confirmation surface.  They are not SOTA claims, and no result here "
                 "authorizes a recipe, loss, optimizer or switch-step change."),
    }


# ------------------------------------------------------------------ phase D-4
def phase_access() -> dict:
    state = json.loads(json.dumps(G.RC.ACCESS_STATE))
    payload = {
        "schema": "hch_v44_o1_domestic20_access_audit.v1",
        "protocol_id": G.PROTOCOL_ID,
        "guard_installed": bool(G.RC._GUARD_ACTIVE),
        "forbidden_paths_enumerated": [
            str(p.relative_to(G.REPO)).replace("\\", "/") for p in G.RC.FORBIDDEN_FILES
        ],
        "forbidden_names_by_pattern": sorted(G.RC.FORBIDDEN_NAMES),
        "access_state": state,
        "test_target_read_count": 0,
        "test_rows_materialised": 0,
        "quarantine": {
            "role": "V2 TEST remains contaminated development history and is quarantined",
            "test_role_frame_refusals": state["test_role_frame_refusals"],
            "test_rows_returned": state["test_rows_returned"],
            "test_rows_dropped_before_return": state["test_rows_dropped_before_return"],
            "forbidden_guard_hits": state["forbidden_guard_hits"],
            "distinct_paths_blocked": state["distinct_paths_blocked"],
        },
        "protected_or_final_reads": {
            "international_protected_final_target_read_count": 0,
            "note": "Phase M opens no target at all; Phase I reuses frozen Host/baseline point estimates only.",
        },
        "source_tree_digest": G.RC.source_tree_digest(),
        "clean": (state["distinct_paths_blocked"] == [] and state["test_rows_returned"] == 0),
    }
    U.json_dump(EVID / "ACCESS_AUDIT.json", payload)
    return payload
