"""Orchestration for the HCH v4.4 optimization-recovery stage (TRAIN+VAL only).

Subcommands, in the order the protocol allows them:

  --smoke      plumbing smoke on the smallest panel cell, written outside evidence
  --r0         R0 diagnostics rebuilt from the completed stage's TRAIN/VAL freeze
  --r1         24 registered panel fits (8 cells x seeds 7/17/37)
  --r1-gate    recompute the preregistered 8-cell promotion gate
  --expand     36 remaining cells x seeds, only after the panel gate passed
  --full-gate  recompute the preregistered 20-cell gate
  --r2a/--r2b  conditional single-factor R2 arms, only when legally triggered
  --finalize   RECIPE_FREEZE.json + RESULTS.md + the access/provenance audits

No subcommand opens TEST.  ``--r2a``/``--r2b`` additionally refuse unless the
recomputed optimization-limited trigger is on disk and satisfied.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import statistics as st
from pathlib import Path

import numpy as np

import recovery_common as RC
import recovery_train as RT

EVID = RC.EVID
U = RC.U
PRIOR_EVID = RC.PRIOR_EVID

PANEL = [
    ("GANSU_DA", "PatchTST"),
    ("GANSU_DA", "LSTM"),
    ("SHANDONG_DA", "PatchTST"),
    ("SHANDONG_DA", "iTransformer"),
    ("SHAANXI_DA", "TimeMixer"),
    ("SHAANXI_DA", "PatchTST"),
    ("NINGXIA_DA", "iTransformer"),
    ("QINGHAI_DA", "TimeMixer"),
]
PANEL_KEYS = {U.cell_key(m, h) for m, h in PANEL}

RECIPE_R1 = {
    "recipe_id": "R1_STEP_BUDGET_2000",
    "optimizer": "AdamW", "lr": 1e-3, "batch_size": 32, "weight_decay": 1e-4,
    "grad_clip": 1.0, "ema_decay": 0.995, "eval_weights": "ema",
    "max_steps": 2000, "val_every": 50, "no_stop_before": 800, "patience_checks": 8,
    "readout_init": None, "selection": "min EMA VAL MAE over all checks including step 0",
}
RECIPE_R2A = {**RECIPE_R1, "recipe_id": "R2A_LR_3E-3", "lr": 3e-3}
RECIPE_R2B = {**RECIPE_R1, "recipe_id": "R2B_LEVEL_AFFINE_N0_1E-3",
              "readout_init": "level_affine_n0_1e-3"}


def _recipe(arm: str) -> dict:
    return {"R1": RECIPE_R1, "R2A": RECIPE_R2A, "R2B": RECIPE_R2B}[arm.upper()]


def run_dir(arm: str, market: str, host: str, seed: int) -> Path:
    return EVID / f"{arm.lower()}_runs" / U.cell_key(market, host) / f"seed{seed}"


# --------------------------------------------------------------------------- R0
def phase_r0() -> dict:
    """Rebuild the old epoch-budget trajectory from the completed stage's freeze.

    Reads only ``primary/*/seed*/freeze.json``, which is a TRAIN/VAL artifact.
    The adjacent ``result.json`` / ``test_predictions.npz`` are TEST artifacts and
    are blocked by the access guard.
    """
    RC.install_access_guard()
    rows, failures = [], []
    for m, h in U.cell_list():
        for s in U.SEEDS:
            p = PRIOR_EVID / "primary" / U.cell_key(m, h) / f"seed{s}" / "freeze.json"
            if not p.is_file():
                failures.append(f"{U.cell_key(m, h)} seed{s}: prior freeze missing")
                continue
            rec = json.loads(p.read_text(encoding="utf-8"))
            cfg = rec["train_config"]
            n = int(rec["n_train_rows"])
            spe = int(np.ceil(n / cfg["batch_size"]))
            hist = rec["epoch_history"]
            sel = int(rec["selected_epoch"])
            steps_done = len(hist) * spe
            rows.append({
                "cell": U.cell_key(m, h), "market": m, "host": h, "seed": s,
                "n_train_rows": n, "n_val_rows": int(rec["n_val_rows"]),
                "batch_size": cfg["batch_size"], "max_epochs": cfg["max_epochs"],
                "patience": cfg["patience"], "lr": cfg["lr"], "ema_decay": cfg["ema_decay"],
                "steps_per_epoch": spe,
                "old_total_optimizer_steps": steps_done,
                "epochs_run": len(hist),
                "selected_epoch": sel,
                "estimated_selected_optimizer_step": (sel + 1) * spe,
                "epoch0_selected": bool(sel == 0),
                "step0_val_mae_ema": float(hist[0]["val_mae_ema"]),
                "best_old_val_mae_ema": float(rec["selected_val_mae_ema"]),
                "val_improvement_from_training_pct": float(
                    (hist[0]["val_mae_ema"] - rec["selected_val_mae_ema"])
                    / hist[0]["val_mae_ema"] * 100.0
                ),
                "selected_ema_parameter_hash": rec["selected_ema_parameter_hash"],
                "scales_fingerprint": rec["scales"]["fingerprint"],
                "history_support_hash": rec["history_support_hash"],
                "prior_freeze_sha256": U.sha256(p),
                "in_recovery_panel": U.cell_key(m, h) in PANEL_KEYS,
            })

    csv_path = EVID / "R0_DIAGNOSTICS.csv"
    cols = [
        "cell", "market", "host", "seed", "n_train_rows", "n_val_rows", "batch_size",
        "max_epochs", "patience", "lr", "ema_decay", "steps_per_epoch",
        "old_total_optimizer_steps", "epochs_run", "selected_epoch",
        "estimated_selected_optimizer_step", "epoch0_selected", "step0_val_mae_ema",
        "best_old_val_mae_ema", "val_improvement_from_training_pct",
        "selected_ema_parameter_hash", "scales_fingerprint", "history_support_hash",
        "prior_freeze_sha256", "in_recovery_panel",
    ]
    EVID.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")

    spe = [r["steps_per_epoch"] for r in rows]
    sel0 = [r for r in rows if r["epoch0_selected"]]
    panel_rows = [r for r in rows if r["in_recovery_panel"]]
    summary = {
        "schema": "hch_v44_recovery_r0.v1",
        "n_runs": len(rows), "failures": failures,
        "steps_per_epoch": {
            "min": int(min(spe)), "max": int(max(spe)),
            "ratio_max_over_min": float(max(spe) / min(spe)),
            "per_market": {m: sorted({r["steps_per_epoch"] for r in rows if r["market"] == m})
                           for m in U.MARKETS},
        },
        "old_optimizer_steps": {
            "min": int(min(r["old_total_optimizer_steps"] for r in rows)),
            "max": int(max(r["old_total_optimizer_steps"] for r in rows)),
            "median": float(st.median([r["old_total_optimizer_steps"] for r in rows])),
        },
        "epoch0_selected_count": len(sel0),
        "epoch0_selected_fraction": float(len(sel0) / len(rows)) if rows else None,
        "epoch0_selected_by_market": {
            m: sum(1 for r in sel0 if r["market"] == m) for m in U.MARKETS
        },
        "val_improvement_from_training_pct": {
            "median": float(st.median([r["val_improvement_from_training_pct"] for r in rows])),
            "max": float(max(r["val_improvement_from_training_pct"] for r in rows)),
            "n_positive": sum(1 for r in rows if r["val_improvement_from_training_pct"] > 0),
        },
        "recovery_panel": {
            "keys": sorted(PANEL_KEYS),
            "epoch0_selected_count": sum(1 for r in panel_rows if r["epoch0_selected"]),
            "n_runs": len(panel_rows),
            "median_old_optimizer_steps": float(
                st.median([r["old_total_optimizer_steps"] for r in panel_rows])
            ) if panel_rows else None,
        },
        "correction_magnitude_stored_in_prior_freeze": False,
        "note": (
            "The completed stage's freeze record stores the EMA VAL MAE per epoch but not "
            "correction magnitude; that diagnostic is produced by the R1 step-0 check, which "
            "is the first check of every recovery run."
        ),
        "test_target_read_count": 0,
    }
    U.json_dump(EVID / "R0_SUMMARY.json", summary)
    _write_r0_markdown(rows, summary)
    RC.write_access_audit(EVID / "ACCESS_AUDIT.json", {"phase": "R0", "n_runs_read": len(rows)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("R0_INCOMPLETE: " + "; ".join(failures[:3]))
    return summary


def _write_r0_markdown(rows: list[dict], summary: dict) -> None:
    lines = [
        "# R0 — why the frozen epoch budget stopped near the Host",
        "",
        "Rebuilt read-only from the completed stage's TRAIN/VAL freeze records; **0 new fits, "
        "0 TEST reads**. Authority: `PROTOCOL.md` §6.",
        "",
        "## Budget asymmetry (the registered reason for this stage)",
        "",
        f"- optimizer steps per epoch: **{summary['steps_per_epoch']['min']} .. "
        f"{summary['steps_per_epoch']['max']}** "
        f"(max/min = {summary['steps_per_epoch']['ratio_max_over_min']:.1f}x)",
        f"- total optimizer steps in a full 50-epoch run: **"
        f"{summary['old_optimizer_steps']['min']} .. {summary['old_optimizer_steps']['max']}** "
        f"(median {summary['old_optimizer_steps']['median']:.0f})",
        f"- epoch 0 selected in **{summary['epoch0_selected_count']}/{summary['n_runs']}** runs "
        f"({summary['epoch0_selected_fraction'] * 100:.1f}%)",
        f"- steps per epoch by market: "
        + ", ".join(f"{m}={v}" for m, v in summary["steps_per_epoch"]["per_market"].items()),
        "",
        "## Did training ever improve VAL?",
        "",
        f"- VAL improvement from step 0 to the selected epoch: median "
        f"**{summary['val_improvement_from_training_pct']['median']:.4f}%**, best "
        f"**{summary['val_improvement_from_training_pct']['max']:.4f}%**, positive in "
        f"**{summary['val_improvement_from_training_pct']['n_positive']}/{summary['n_runs']}** runs",
        "- correction magnitude is not stored in the prior freeze; it is measured at step 0 of "
        "every R1 run (same evaluation path), which is the honest pre-training reference.",
        "",
        "## Recovery panel (8 cells, optimization-development only)",
        "",
        f"- {summary['recovery_panel']['epoch0_selected_count']}/"
        f"{summary['recovery_panel']['n_runs']} panel runs selected epoch 0",
        f"- median old optimizer steps inside the panel: "
        f"{summary['recovery_panel']['median_old_optimizer_steps']}",
        "",
        "| cell | host | n_train | steps/epoch | old total steps | selected epoch | epoch0? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["cell"], x["seed"])):
        if not r["in_recovery_panel"]:
            continue
        lines.append(
            f"| {r['market']} | {r['host']} | {r['n_train_rows']} | {r['steps_per_epoch']} | "
            f"{r['old_total_optimizer_steps']} | {r['selected_epoch']} | "
            f"{'yes' if r['epoch0_selected'] else 'no'} |"
        )
    lines += ["", "Full per-run table: `R0_DIAGNOSTICS.csv`.", ""]
    (EVID / "R0_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- fits
def fit_task(task: dict) -> dict:
    RC.worker_env()
    arm = task["arm"]
    m, h, s = task["market"], task["host"], task["seed"]
    recipe = _recipe(arm)
    out = run_dir(arm, m, h, s)
    rec = RT.train_step_based(
        m, h, s, out,
        max_steps=recipe["max_steps"], val_every=recipe["val_every"],
        no_stop_before=recipe["no_stop_before"], patience_checks=recipe["patience_checks"],
        lr=recipe["lr"], readout_init=recipe["readout_init"],
        device=task.get("device", "cuda"), recipe_id=recipe["recipe_id"],
    )
    print(f"[{arm.lower()}] {U.cell_key(m, h)} seed{s} step={rec['selected_step']} "
          f"val={rec['selected_val_mae_ema']:.4f} "
          f"stopped={rec['optimization']['stopped_at_step']}", flush=True)
    return rec


def _pending(arm: str, cells: list[tuple[str, str]]) -> list[dict]:
    tasks = []
    for m, h in cells:
        for s in U.SEEDS:
            if not (run_dir(arm, m, h, s) / "freeze.json").is_file():
                tasks.append({"arm": arm, "market": m, "host": h, "seed": s})
    return tasks


def phase_fits(arm: str, cells: list[tuple[str, str]], workers: int) -> None:
    RC.install_access_guard()
    tasks = _pending(arm, cells)
    print(f"[{arm.lower()}] {len(tasks)} fits pending on {workers} GPU lanes", flush=True)
    if tasks:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for _ in pool.map(fit_task, tasks):
                pass
    records = _load_runs(arm, cells)
    print(f"[{arm.lower()}] {len(records)} freeze records present", flush=True)


def _load_runs(arm: str, cells: list[tuple[str, str]]) -> list[dict]:
    out = []
    for m, h in cells:
        for s in U.SEEDS:
            p = run_dir(arm, m, h, s) / "freeze.json"
            if p.is_file():
                out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _curves(arm: str, cells: list[tuple[str, str]]) -> dict:
    out = {}
    for m, h in cells:
        for s in U.SEEDS:
            p = run_dir(arm, m, h, s) / "training_curve.json"
            if p.is_file():
                out[f"{U.cell_key(m, h)}__seed{s}"] = json.loads(p.read_text(encoding="utf-8"))
    return out


# --------------------------------------------------------------------------- gates
def _prior_frozen(cell: str, seed: int) -> dict:
    p = PRIOR_EVID / "primary" / cell / f"seed{seed}" / "freeze.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    return {"val_mae": float(rec["selected_val_mae_ema"]), "selected_epoch": int(rec["selected_epoch"]),
            "n_val_rows": int(rec["n_val_rows"])}


def _cell_table(arm: str, cells: list[tuple[str, str]]) -> list[dict]:
    """Per-cell seed-median comparison against the frozen v4.4 recipe."""
    table = []
    for m, h in cells:
        cell = U.cell_key(m, h)
        runs = [r for r in _load_runs(arm, [(m, h)])]
        if len(runs) != len(U.SEEDS):
            raise SystemExit(f"{cell}: {len(runs)} recovery runs, expected {len(U.SEEDS)}")
        priors = [_prior_frozen(cell, s) for s in U.SEEDS]
        for r in runs:
            pr = _prior_frozen(cell, r["seed"])
            if int(r["n_val_rows"]) != pr["n_val_rows"]:
                raise SystemExit(f"{cell} seed{r['seed']}: VAL row count differs from the frozen stage")
        rec_med = float(st.median([r["selected_val_mae_ema"] for r in runs]))
        old_med = float(st.median([p["val_mae"] for p in priors]))
        host_med = float(st.median([r["selected_metrics"]["val_host_mae"] for r in runs]))
        corr_med = float(st.median([r["selected_metrics"]["val_correction_ratio"] for r in runs]))
        n_pos_step = sum(1 for r in runs if int(r["selected_step"]) > 0
                         and float(r["selected_metrics"]["val_correction_ratio"]) >= 0.05)
        finite = all(np.isfinite([r["selected_val_mae_ema"], r["selected_metrics"]["val_host_mae"]]).all()
                     for r in runs)
        table.append({
            "cell": cell, "market": m, "host": h,
            "n_seeds": len(runs),
            "recovery_val_mae_median": rec_med,
            "frozen_val_mae_median": old_med,
            "host_val_mae_median": host_med,
            "improvement_vs_frozen_pct": (old_med - rec_med) / old_med * 100.0,
            "gain_vs_host_pct_recovery": (host_med - rec_med) / host_med * 100.0,
            "gain_vs_host_pct_frozen": (host_med - old_med) / host_med * 100.0,
            "selected_correction_ratio_median": corr_med,
            "n_seeds_step_gt0_and_ratio_ge_0.05": n_pos_step,
            "selected_steps": [int(r["selected_step"]) for r in runs],
            "selected_epoch0_equivalents": sum(1 for r in runs if int(r["selected_step"]) == 0),
            "all_finite": bool(finite),
        })
    return table


def _write_table_csv(path: Path, table: list[dict]) -> None:
    cols = list(table[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for row in table:
            fh.write(",".join(
                json.dumps(row[c]) if isinstance(row[c], (list, dict)) else str(row[c])
                for c in cols
            ) + "\n")


def _access_clean(arm: str, cells: list[tuple[str, str]]) -> tuple[bool, dict]:
    bad = {}
    for m, h in cells:
        for s in U.SEEDS:
            p = run_dir(arm, m, h, s) / "freeze.json"
            if not p.is_file():
                continue
            rec = json.loads(p.read_text(encoding="utf-8"))
            st_ = rec.get("access_state", {})
            if rec.get("test_target_read_count", 0) != 0 or rec.get("test_rows_materialised", 0) != 0:
                bad[f"{U.cell_key(m, h)}__seed{s}"] = "test_target_read_count != 0"
            if st_.get("distinct_paths_blocked"):
                bad[f"{U.cell_key(m, h)}__seed{s}"] = "forbidden path opened"
            if st_.get("test_rows_returned", 0) != 0:
                bad[f"{U.cell_key(m, h)}__seed{s}"] = "TEST row returned"
    return (not bad), bad


def gate(arm: str, cells: list[tuple[str, str]], kind: str) -> dict:
    """Preregistered gate arithmetic, recomputed from the raw per-seed runs."""
    table = _cell_table(arm, cells)
    n = len(table)
    imps = [c["improvement_vs_frozen_pct"] for c in table]
    clean, bad = _access_clean(arm, cells)
    finite = all(c["all_finite"] for c in table)
    if kind == "panel":
        checks = {
            "gate_1_at_least_6_of_8_cells_improve": sum(1 for v in imps if v > 0) >= 6,
            "gate_2_panel_median_improvement_ge_0.5pct": float(st.median(imps)) >= 0.5,
            "gate_3_at_least_5_of_8_step_gt0_and_ratio_ge_0.05":
                sum(1 for c in table if c["n_seeds_step_gt0_and_ratio_ge_0.05"] >= 2) >= 5,
            "gate_4_worst_degradation_le_1.0pct": min(imps) >= -1.0,
            "gate_5_no_leakage_or_numerical_failure": clean and finite,
        }
        cells_improving = sum(1 for v in imps if v > 0)
        detail = {"n_cells": n, "cells_improving": cells_improving,
                  "panel_median_improvement_pct": float(st.median(imps)),
                  "worst_improvement_pct": float(min(imps))}
    else:
        checks = {
            "gate_1_at_least_15_of_20_non_worse": sum(1 for v in imps if v >= 0) >= 15,
            "gate_2_at_least_12_of_20_improve_ge_0.5pct": sum(1 for v in imps if v >= 0.5) >= 12,
            "gate_3_panel_median_gain_vs_host_gt_0_and_materially_above_frozen":
                float(st.median([c["gain_vs_host_pct_recovery"] for c in table])) > 0
                and float(st.median([c["gain_vs_host_pct_recovery"] for c in table]))
                >= float(st.median([c["gain_vs_host_pct_frozen"] for c in table])) + 0.5,
            "gate_4_no_market_median_degradation_worse_than_1.0pct": min(imps) >= -1.0,
            "gate_5_no_leakage_or_numerical_failure": clean and finite,
        }
        detail = {
            "n_cells": n,
            "cells_non_worse": sum(1 for v in imps if v >= 0),
            "cells_improving_ge_0.5pct": sum(1 for v in imps if v >= 0.5),
            "panel_median_improvement_pct": float(st.median(imps)),
            "panel_median_gain_vs_host_recovery_pct": float(
                st.median([c["gain_vs_host_pct_recovery"] for c in table])),
            "panel_median_gain_vs_host_frozen_pct": float(
                st.median([c["gain_vs_host_pct_frozen"] for c in table])),
            "worst_improvement_pct": float(min(imps)),
            "gate_3_operationalization": "recovery panel-median gain vs Host must exceed the frozen "
                                         "recipe's by at least 0.5 pp AND be positive",
        }
    payload = {
        "schema": f"hch_v44_recovery_{kind}_gate.v1",
        "arm": arm, "recipe": _recipe(arm),
        "cells": [c["cell"] for c in table],
        "per_cell": table,
        "checks": checks,
        "detail": detail,
        "n_passed": sum(1 for v in checks.values() if v),
        "n_failed": sum(1 for v in checks.values() if not v),
        "failed": [k for k, v in checks.items() if not v],
        "access_violations": bad,
        "passed": all(checks.values()),
        "token": None,
    }
    if kind == "panel":
        payload["token"] = (
            "HCH_V44_RECOVERY_R1_PANEL_PROMOTED" if payload["passed"]
            else "HCH_V44_RECOVERY_R1_PANEL_NOT_PROMOTED"
        )
    else:
        payload["token"] = (
            "HCH_V44_OPTIMIZATION_RECOVERY_READY_FOR_UNTOUCHED_CONFIRMATION" if payload["passed"]
            else "HCH_V44_OPTIMIZATION_RECOVERY_NOT_SUPPORTED"
        )
    if arm == "R1":
        gate_path = EVID / ("R1_GATE.json" if kind == "panel" else "FULL_PANEL_GATE.json")
        cells_path = EVID / ("R1_PANEL_CELL_MEDIANS.csv" if kind == "panel"
                             else "FULL_PANEL_CELL_MEDIANS.csv")
        seeds_path = EVID / ("R1_PANEL_PER_SEED.csv" if kind == "panel"
                             else "FULL_PANEL_PER_SEED.csv")
    else:
        gate_path = EVID / f"{arm}_GATE.json"
        cells_path = EVID / f"{arm}_PANEL_CELL_MEDIANS.csv"
        seeds_path = EVID / f"{arm}_PANEL_PER_SEED.csv"
    U.json_dump(gate_path, payload)
    _write_table_csv(cells_path, table)
    _per_seed_csv(arm, cells, seeds_path)
    _curves_out(arm, cells)
    print(json.dumps({"check": checks, "detail": detail, "token": payload["token"]},
                     ensure_ascii=False, indent=2))
    return payload


def _per_seed_csv(arm: str, cells: list[tuple[str, str]], path: Path) -> None:
    rows = []
    for m, h in cells:
        cell = U.cell_key(m, h)
        for s in U.SEEDS:
            p = run_dir(arm, m, h, s) / "freeze.json"
            if not p.is_file():
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            sel = r["selected_metrics"]
            pr = _prior_frozen(cell, s)
            rows.append({
                "cell": cell, "market": m, "host": h, "seed": s,
                "n_train_rows": r["n_train_rows"], "n_val_rows": r["n_val_rows"],
                "selected_step": r["selected_step"], "selected_check": r["selected_check"],
                "stopped_at_step": r["optimization"]["stopped_at_step"],
                "val_mae_ema": r["selected_val_mae_ema"],
                "val_mae_raw_at_selected_step": sel["val_mae_raw"],
                "val_host_mae": sel["val_host_mae"],
                "val_gain_vs_host_pct": sel["val_gain_vs_host_pct"],
                "val_correction_ratio": sel["val_correction_ratio"],
                "train_correction_ratio": sel["train_correction_ratio"],
                "diag_reconstruction_mae": sel["diag_reconstruction_mae"],
                "level_mae": sel["level_mae"], "mass_mae": sel["mass_mae"],
                "shape_w1": sel["shape_w1"],
                "frozen_val_mae_ema": pr["val_mae"], "frozen_selected_epoch": pr["selected_epoch"],
                "improvement_vs_frozen_pct": (pr["val_mae"] - r["selected_val_mae_ema"])
                / pr["val_mae"] * 100.0,
                "checkpoint_parameter_hash": r["selected_ema_parameter_hash"],
            })
    cols = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for row in rows:
            fh.write(",".join(str(row[c]) for c in cols) + "\n")


def _curves_out(arm: str, cells: list[tuple[str, str]]) -> None:
    d = EVID / ("R1_TRAINING_CURVES" if arm == "R1" else f"{arm}_TRAINING_CURVES")
    d.mkdir(parents=True, exist_ok=True)
    for key, curve in _curves(arm, cells).items():
        U.json_dump(d / f"{key}.json", curve)


# --------------------------------------------------------------------------- smoke
def smoke() -> dict:
    """Plumbing smoke: 60 steps on the smallest panel cell, written outside evidence."""
    RC.install_access_guard()
    out = RC.STAGE / "build" / "smoke" / "QINGHAI_DA__TimeMixer"
    rec = RT.train_step_based(
        "QINGHAI_DA", "TimeMixer", 7, out, max_steps=60, val_every=20,
        no_stop_before=0, patience_checks=99, device="cuda", recipe_id="SMOKE_60_STEPS",
    )
    payload = {
        "cell": "QINGHAI_DA__TimeMixer", "seed": 7, "max_steps": 60,
        "n_checks": rec["epoch_history_check_count"],
        "selected_step": rec["selected_step"],
        "selected_val_mae_ema": rec["selected_val_mae_ema"],
        "step0_val_mae_ema": _step0(rec),
        "n_train_rows": rec["n_train_rows"], "n_val_rows": rec["n_val_rows"],
        "steps_per_epoch_old_semantics": rec["steps_per_epoch_old_semantics"],
        "test_target_read_count": rec["test_target_read_count"],
        "test_rows_dropped_from_joint_cache": rec["test_rows_dropped_from_joint_cache"],
        "is_registered_fit": False,
    }
    U.json_dump(RC.STAGE / "build" / "smoke" / "SMOKE_REPORT.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _step0(rec: dict) -> float:
    curve = json.loads((RC.STAGE / "build" / "smoke" / "QINGHAI_DA__TimeMixer"
                        / "training_curve.json").read_text(encoding="utf-8"))
    return float(curve[0]["val_mae_ema"])


# --------------------------------------------------------------------------- R2 gate
def r2_trigger(arm: str = "R1") -> dict:
    """Recompute PROTOCOL §10.1 from the R1 curves.  R2 is refused unless it holds."""
    curves = _curves(arm, PANEL)
    n = len(curves)
    hits = {k: 0 for k in ("ceiling", "train_improving_val_flat", "tiny_ratio_with_grad", "ema_lags_raw")}
    for key, curve in curves.items():
        last = curve[-1]
        sel = [e for e in curve if e["selected"]][-1]
        if int(sel["step"]) >= 2000:
            hits["ceiling"] += 1
        tail = [e for e in curve if e["step"] >= 1600]
        if len(tail) >= 2 and tail[-1]["diag_reconstruction_mae"] < tail[0]["diag_reconstruction_mae"] - 1e-6:
            if last["val_mae_ema"] > min(e["val_mae_ema"] for e in tail) - 1e-9:
                hits["train_improving_val_flat"] += 1
        if float(sel["val_correction_ratio"]) < 0.05 and (last["grad_norms_preclip"]["level"] or 0) > 0:
            hits["tiny_ratio_with_grad"] += 1
        lags = [e for e in curve if e["val_mae_raw"] < e["val_mae_ema"] - 1e-6]
        if len(lags) > 8:
            hits["ema_lags_raw"] += 1
    frac = {k: v / n for k, v in hits.items()} if n else {}
    payload = {
        "schema": "hch_v44_recovery_r2_trigger.v1",
        "n_runs": n, "hits": hits, "fractions": frac,
        "threshold": "at least one condition holds in >=50% of the 24 R1 runs",
        "satisfied": any(v >= 0.5 for v in frac.values()),
        "r2_authorized": any(v >= 0.5 for v in frac.values()),
    }
    U.json_dump(EVID / "R2_TRIGGER.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


# --------------------------------------------------------------------------- finalize
def finalize() -> dict:
    RC.install_access_guard()
    panel_gate = json.loads((EVID / "R1_GATE.json").read_text(encoding="utf-8"))
    full_path = EVID / "FULL_PANEL_GATE.json"
    full_gate = json.loads(full_path.read_text(encoding="utf-8")) if full_path.is_file() else None
    manifest = RC.build_evidence_digest()
    U.json_dump(EVID / "REUSED_SUPPORT_HASHES.json", {
        "schema": "hch_v44_recovery_reused_support.v1",
        "shadow_recipe": RC.SHADOW_RECIPE,
        "prior_stage_evidence_root":
            str(PRIOR_EVID.relative_to(RC.REPO)).replace("\\", "/"),
        "fields_reused": ["shadow OOF support npz/json", "TRAIN-only scales (recomputed from "
                          "TRAIN frame)", "TRAIN/VAL day ids", "prior TRAIN/VAL freeze records",
                          "frozen Host joint cache (TRAIN/VAL segments only)"],
        "fields_forbidden": ["TEST targets", "TEST predictions", "TEST metrics",
                             "CELL_MEDIAN_RESULTS.csv", "PER_SEED_RESULTS.csv",
                             "BASELINE_COMPARISON.csv", "RESULTS_LONG.csv",
                             "HCH_V44_MAIN_20260917.csv", "BASELINE_COMPLETION_20260917.csv"],
        **manifest,
    })
    # The gate tokens are *intermediate* promotion tokens; the stage may only report
    # one of the three registered terminal tokens.  Map explicitly and refuse to
    # guess: an unmapped state is a bug, not a verdict.
    if full_gate is not None:
        token = ("HCH_V44_OPTIMIZATION_RECOVERY_READY_FOR_UNTOUCHED_CONFIRMATION"
                 if full_gate["passed"] else "HCH_V44_OPTIMIZATION_RECOVERY_NOT_SUPPORTED")
        route = "full_20_cell_panel"
    elif panel_gate["passed"]:
        raise RuntimeError("R1 panel promoted but the full panel was never run; "
                           "run --expand and --full-gate before finalizing")
    else:
        trigger_path = EVID / "R2_TRIGGER.json"
        trigger = (json.loads(trigger_path.read_text(encoding="utf-8"))
                   if trigger_path.is_file() else None)
        r2 = next((a for a in ("R2A", "R2B") if (EVID / f"{a}_GATE.json").is_file()), None)
        if r2 is not None:
            token = ("HCH_V44_OPTIMIZATION_RECOVERY_READY_FOR_UNTOUCHED_CONFIRMATION"
                     if json.loads((EVID / f"{r2}_GATE.json").read_text(encoding="utf-8"))["passed"]
                     else "HCH_V44_OPTIMIZATION_RECOVERY_NOT_SUPPORTED")
            route = f"r1_panel_not_promoted_then_{r2.lower()}"
        elif trigger is None:
            raise RuntimeError("R1 panel did not promote and no R2 trigger was computed; "
                               "run --r2-trigger before finalizing")
        elif trigger["satisfied"]:
            raise RuntimeError("R2 is authorized but neither R2A nor R2B was run; the stage "
                               "cannot terminate while an authorized arm is untested")
        else:
            token = "HCH_V44_OPTIMIZATION_RECOVERY_NOT_SUPPORTED"
            route = "r1_panel_not_promoted_r2_trigger_unsatisfied"
    freeze = {
        "route_to_terminal_token": route,
        "schema": "hch_v44_recovery_recipe_freeze.v1",
        "protocol_id": "HCH_V44_OPTIMIZATION_RECOVERY_20260917",
        "recommended_recipe": _recipe("R1"),
        "panel_gate": {"passed": panel_gate["passed"], "detail": panel_gate["detail"]},
        "full_panel_gate": ({"passed": full_gate["passed"], "detail": full_gate["detail"]}
                            if full_gate else None),
        "frozen_scientific_object": {
            "variant": RC.VARIANT,
            "changed_vs_prior_stage": ["optimizer-step budget semantics"],
            "unchanged": ["src/core/**", "b/B/S+/S- geometry", "decoder", "feature set",
                          "loss and weights", "W=7 history", "G2 routing", "seeds 7/17/37",
                          "batch 32", "weight_decay 1e-4", "dropout 0.1", "grad_clip 1.0",
                          "EMA 0.995", "EMA-only primary selection"],
        },
        "source_tree_digest": RC.source_tree_digest(),
        "terminal_token": token,
        "test_target_read_count": 0,
        "not_a_sota_claim": True,
        "note": "V2 TEST was observed before this stage and is quarantined; the token only "
                "authorizes requesting a separately sealed untouched confirmation surface.",
    }
    U.json_dump(EVID / "RECIPE_FREEZE.json", freeze)
    RC.write_access_audit(EVID / "ACCESS_AUDIT.json", {
        "phase": "finalize",
        "reused_support_manifest": "REUSED_SUPPORT_HASHES.json",
        "recipe_freeze": "RECIPE_FREEZE.json",
    })
    _write_results(panel_gate, full_gate, freeze)
    print(json.dumps({"terminal_token": token}, indent=2))
    return freeze


def _write_results(panel_gate, full_gate, freeze) -> None:
    p = panel_gate["detail"]
    lines = [
        "# HCH v4.4 optimization recovery — TRAIN+VAL only, TEST quarantined",
        "",
        f"Terminal token: **{freeze['terminal_token']}**",
        "",
        "## What changed",
        "",
        "Exactly one thing: the training budget is registered in **optimizer steps** with a "
        "burn-in, instead of in epochs. Architecture, geometry, features, loss, initialization, "
        "seeds, batch size, weight decay, dropout, gradient clip and EMA are unchanged, and "
        "`src/core/**` was not edited.",
        "",
        "## R0 — the budget asymmetry that motivated the stage",
        "",
        "See `R0_SUMMARY.md` / `R0_DIAGNOSTICS.csv`.",
        "",
        "## R1 — 8-cell panel (24 fits)",
        "",
        f"- cells improving over the frozen recipe: {p['cells_improving']}/{p['n_cells']}",
        f"- panel median improvement: {p['panel_median_improvement_pct']:.4f}%",
        f"- worst improvement: {p['worst_improvement_pct']:.4f}%",
        "",
        "| gate | result |",
        "|---|---|",
    ]
    for k, v in panel_gate["checks"].items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines += ["", f"Panel token: `{panel_gate['token']}`", ""]
    if full_gate:
        f = full_gate["detail"]
        lines += [
            "## Full 20-cell panel",
            "",
            f"- cells non-worse: {f['cells_non_worse']}/{f['n_cells']}",
            f"- cells improving >=0.5%: {f['cells_improving_ge_0.5pct']}/{f['n_cells']}",
            f"- panel median improvement: {f['panel_median_improvement_pct']:.4f}%",
            f"- panel median gain vs Host — recovery "
            f"{f['panel_median_gain_vs_host_recovery_pct']:.4f}% vs frozen "
            f"{f['panel_median_gain_vs_host_frozen_pct']:.4f}%",
            "",
            "| gate | result |",
            "|---|---|",
        ]
        for k, v in full_gate["checks"].items():
            lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
        lines += ["", f"Full-panel token: `{full_gate['token']}`", ""]
    else:
        lines += ["## Full 20-cell panel", "",
                  "Not run: the 8-cell promotion gate did not pass, so the protocol forbids "
                  "expanding the recipe.", ""]
    lines += [
        "## What this is not",
        "",
        "This is **not** a SOTA or generalization claim. V2 TEST was observed before this stage "
        "and directly motivated it, so it is quarantined from every decision here. The recovery "
        "process read no TEST target, no TEST prediction and no TEST metric. Any SOTA claim "
        "requires a separately authorized untouched confirmation surface.",
        "",
        "## Provenance",
        "",
        "- `ACCESS_AUDIT.json` — guard state and TEST-read counters;",
        "- `REUSED_SUPPORT_HASHES.json` — every prior-stage artifact reused, with hashes;",
        "- `RECIPE_FREEZE.json` — the frozen recipe and the terminal token;",
        "- `INDEPENDENT_VERIFICATION_REPORT.json` — the independent verifier's verdict.",
        "",
    ]
    (EVID / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--r0", action="store_true")
    ap.add_argument("--r1", action="store_true")
    ap.add_argument("--r1-gate", action="store_true")
    ap.add_argument("--expand", action="store_true")
    ap.add_argument("--full-gate", action="store_true")
    ap.add_argument("--r2-trigger", action="store_true")
    ap.add_argument("--r2a", action="store_true")
    ap.add_argument("--r2b", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    if args.smoke:
        smoke()
    elif args.r0:
        phase_r0()
    elif args.r1:
        phase_fits("R1", PANEL, args.workers)
    elif args.r1_gate:
        gate("R1", PANEL, "panel")
    elif args.expand:
        g = json.loads((EVID / "R1_GATE.json").read_text(encoding="utf-8"))
        if not g["passed"]:
            raise SystemExit("EXPAND_REFUSED: the 8-cell promotion gate did not pass")
        rest = [(m, h) for m, h in U.cell_list() if U.cell_key(m, h) not in PANEL_KEYS]
        phase_fits("R1", rest, args.workers)
    elif args.full_gate:
        gate("R1", U.cell_list(), "full")
    elif args.r2_trigger:
        r2_trigger("R1")
    elif args.r2a or args.r2b:
        trig = json.loads((EVID / "R2_TRIGGER.json").read_text(encoding="utf-8"))
        if not trig["r2_authorized"]:
            raise SystemExit("R2_REFUSED: the optimization-limited trigger is not satisfied")
        arm = "R2A" if args.r2a else "R2B"
        phase_fits(arm, PANEL, args.workers)
        gate(arm, PANEL, "panel")
    elif args.finalize:
        finalize()
    else:
        ap.error("choose one subcommand")


if __name__ == "__main__":
    main()
