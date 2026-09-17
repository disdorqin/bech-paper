"""Objective-alignment probe runtime: TRAIN+VAL only, TEST quarantined.

This stage adds **no new data path**.  It imports the closed recovery stage's
``recovery_common`` read-only and inherits its identical quarantine, causal
conventions, shadow-OOF support, TRAIN-only scales and Host caches.  The only
thing this module adds is the probe's own evidence root and the registered
reference/recipe bookkeeping.

The closed recovery stage's directory is deliberately never edited: its 24 freeze
records carry a directory-level ``code_hash``, so writing into it would invalidate
frozen provenance.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]

RECOVERY_STAGE = REPO / "experiments/current/hch_v44_optimization_recovery_20260917"
RECOVERY_IMPL = RECOVERY_STAGE / "implementation"
RECOVERY_EVID = REPO / "experiments/evidence/hch_v44_optimization_recovery_20260917"

import sys  # noqa: E402

if str(RECOVERY_IMPL) not in sys.path:
    sys.path.insert(0, str(RECOVERY_IMPL))

import recovery_common as RC  # noqa: E402  (closed stage, read-only)

U = RC.U
EVID = REPO / "experiments/evidence/hch_v44_objective_alignment_probe_20260917"
R1_RUNS = RECOVERY_EVID / "r1_runs"
R1_GATE = RECOVERY_EVID / "R1_GATE.json"

PROTOCOL_ID = "HCH_V44_OBJECTIVE_ALIGNMENT_PROBE_20260917"
VARIANT = RC.VARIANT
SEEDS = list(U.SEEDS)

#: The fixed, non-tunable objective switch: updates 1..400 use the frozen full
#: loss, updates 401..max_steps use the reconstruction term alone.
SWITCH_STEP = 400

PANEL = [
    ("GANSU_DA", "PatchTST"),
    ("SHANDONG_DA", "iTransformer"),
    ("SHAANXI_DA", "TimeMixer"),
    ("NINGXIA_DA", "iTransformer"),
]
PANEL_KEYS = [U.cell_key(m, h) for m, h in PANEL]

#: Registered schedule.  Identical to R1 except for the loss schedule.
RECIPE_O1 = {
    "recipe_id": "O1_AUX_WARMUP_400_THEN_MAE",
    "optimizer": "AdamW", "lr": 1e-3, "batch_size": 32, "weight_decay": 1e-4,
    "grad_clip": 1.0, "ema_decay": 0.995, "eval_weights": "ema",
    "max_steps": 2000, "val_every": 50, "no_stop_before": 800, "patience_checks": 8,
    "readout_init": None,
    "loss_schedule": {
        "switch_step": SWITCH_STEP,
        "updates_1_to_400": "L_rec + (L_b + L_B + L_S)/3",
        "updates_401_to_2000": "L_rec",
    },
    "selection": "min EMA VAL MAE over all checks including step 0",
}

#: The registered schedule keys.  ``stopped_at_step`` is deliberately excluded: it
#: is an outcome, not a setting, so it legitimately differs between runs.
REGISTERED_SCHEDULE_KEYS = (
    "optimizer", "lr", "batch_size", "weight_decay", "grad_clip", "ema_decay",
    "eval_weights", "max_steps", "val_every", "no_stop_before", "patience_checks",
    "readout_init",
)


def registered_schedule_of(optimization: dict) -> dict:
    return {k: optimization.get(k) for k in REGISTERED_SCHEDULE_KEYS}


def registered_recipe() -> dict:
    return {k: RECIPE_O1[k] for k in REGISTERED_SCHEDULE_KEYS}


#: The conditions that must be *identical* to the R1 reference.  The verifier
#: compares every one of these against the reference freeze records.
IDENTITY_FIELDS = (
    "n_train_rows", "n_val_rows", "train_day_ids_hash", "val_day_ids_hash",
    "history_support_hash", "steps_per_epoch_old_semantics",
)


def probe_tree_digest() -> str:
    """Digest of this stage's own source, so a freeze record pins its own runner."""
    items = []
    for p in sorted(HERE.parent.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".md"):
            items.append(f"{p.relative_to(REPO).as_posix()}|{U.sha256(p)}")
    return U.sha256_bytes("\n".join(items).encode())


def run_dir(market: str, host: str, seed: int) -> Path:
    return EVID / "o1_runs" / U.cell_key(market, host) / f"seed{seed}"


def reference_dir(market: str, host: str, seed: int) -> Path:
    return R1_RUNS / U.cell_key(market, host) / f"seed{seed}"


def reference_available() -> tuple[bool, list[str], list[str]]:
    """Is a legal step-based R1 full-loss reference present for all 12 coordinates?"""
    missing, present = [], []
    for m, h in PANEL:
        for s in SEEDS:
            d = reference_dir(m, h, s)
            if (d / "freeze.json").is_file() and (d / "training_curve.json").is_file():
                present.append(f"{U.cell_key(m, h)}__seed{s}")
            else:
                missing.append(f"{U.cell_key(m, h)}__seed{s}")
    return (not missing), present, missing


def load_reference(market: str, host: str, seed: int) -> dict:
    """The reused R1 run: the freeze record plus its recorded provenance."""
    d = reference_dir(market, host, seed)
    rec = load_json(d / "freeze.json")
    return {
        "record": rec,
        "freeze_sha256": U.sha256(d / "freeze.json"),
        "curve_sha256": U.sha256(d / "training_curve.json"),
        "checkpoint_sha256": U.sha256(d / "selected_ema.pt"),
    }


def sealed_reference_recipe() -> dict:
    """The reference recipe, read off disk rather than re-imported from source."""
    return load_json(R1_GATE)["recipe"]


# --------------------------------------------------------------------------- gate
def paired_rows() -> list[dict]:
    """One row per (cell, seed): O1 vs its paired R1 reference."""
    rows = []
    for m, h in PANEL:
        cell = U.cell_key(m, h)
        for s in SEEDS:
            o1 = load_json(run_dir(m, h, s) / "freeze.json")
            ref = load_reference(m, h, s)
            r1 = ref["record"]
            mae_o1 = float(o1["selected_val_mae_ema"])
            mae_r1 = float(r1["selected_val_mae_ema"])
            sel = o1["selected_metrics"]
            rows.append({
                "cell": cell, "market": m, "host": h, "seed": s,
                "mae_o1": mae_o1, "mae_r1": mae_r1,
                "relative_val_gain_pct": (mae_r1 - mae_o1) / mae_r1 * 100.0,
                "abs_delta": mae_r1 - mae_o1,
                "selected_step_o1": int(o1["selected_step"]),
                "selected_step_r1": int(r1["selected_step"]),
                "stopped_at_step_o1": o1["optimization"]["stopped_at_step"],
                "stopped_at_step_r1": r1["optimization"]["stopped_at_step"],
                "correction_ratio_o1": float(sel["val_correction_ratio"]),
                "correction_ratio_r1": float(r1["selected_metrics"]["val_correction_ratio"]),
                "val_mae_raw_o1": float(sel["val_mae_raw"]),
                "raw_minus_ema_gap_o1": float(sel["val_mae_raw"]) - mae_o1,
                "train_recon_mae_o1": float(sel["diag_reconstruction_mae"]),
                "train_recon_mae_r1": float(r1["selected_metrics"]["diag_reconstruction_mae"]),
                "val_host_mae_o1": float(sel["val_host_mae"]),
                "n_val_rows_o1": int(o1["n_val_rows"]),
                "n_val_rows_r1": int(r1["n_val_rows"]),
                "r1_freeze_sha256": ref["freeze_sha256"],
                "step0_retreat": bool(int(o1["selected_step"]) == 0
                                      and int(r1["selected_step"]) > 0),
            })
    return rows


def val_optimum_position(market: str, host: str, seed: int) -> dict:
    """Where the best post-switch VAL check sits, relative to the switch."""
    curve = load_json(run_dir(market, host, seed) / "training_curve.json")
    post = [e for e in curve if int(e["step"]) > SWITCH_STEP]
    best_post = min(post, key=lambda e: float(e["val_mae_ema"])) if post else None
    best_all = min(curve, key=lambda e: float(e["val_mae_ema"]))
    step0 = next(e for e in curve if int(e["step"]) == 0)
    return {
        "best_post_switch_step": int(best_post["step"]) if best_post else None,
        "best_post_switch_val_mae_ema": float(best_post["val_mae_ema"]) if best_post else None,
        "argmin_over_all_steps": int(best_all["step"]),
        "step0_val_mae_ema": float(step0["val_mae_ema"]),
        "post_switch_better_than_step0": bool(
            best_post is not None and float(best_post["val_mae_ema"]) < float(step0["val_mae_ema"])
        ),
        "post_switch_better_than_at_switch": bool(
            best_post is not None
            and float(best_post["val_mae_ema"])
            < float(next(e for e in curve if int(e["step"]) == SWITCH_STEP)["val_mae_ema"])
        ),
    }


def cell_medians(rows: list[dict]) -> list[dict]:
    out = []
    for m, h in PANEL:
        cell = U.cell_key(m, h)
        sub = [r for r in rows if r["cell"] == cell]
        if len(sub) != len(SEEDS):
            raise RuntimeError(f"{cell}: {len(sub)} paired rows, expected {len(SEEDS)}")
        gains = [r["relative_val_gain_pct"] for r in sub]
        out.append({
            "cell": cell, "market": m, "host": h, "n_seeds": len(sub),
            "mae_o1_median": float(st.median([r["mae_o1"] for r in sub])),
            "mae_r1_median": float(st.median([r["mae_r1"] for r in sub])),
            "cell_median_relative_gain_pct": float(st.median(gains)),
            "per_seed_gains_pct": [round(g, 6) for g in gains],
            "cells_seeds_improving": sum(1 for g in gains if g > 0),
            "selected_correction_ratio_median": float(
                st.median([r["correction_ratio_o1"] for r in sub])),
            "selected_step_median": float(st.median([r["selected_step_o1"] for r in sub])),
        })
    return out


def gate(rows: list[dict], cells: list[dict], compliance: dict) -> dict:
    """PROTOCOL.md section 8, recomputed from the raw per-seed numbers.

    ``compliance`` carries the audited g5 facts (access clean, all finite, objective
    switch exactly after ``SWITCH_STEP``, registered schedule unchanged); it is
    computed from the freeze records and curves, never asserted by hand.
    """
    gains = [c["cell_median_relative_gain_pct"] for c in cells]
    panel_median = float(st.median(gains))
    step0_retreats = sum(1 for r in rows if r["step0_retreat"])
    ratio_median = float(st.median([r["correction_ratio_o1"] for r in rows]))
    g5 = bool(
        compliance["access_clean"] and compliance["all_finite"]
        and compliance["switch_exactly_after_400"] and compliance["schedule_unchanged"]
    )
    checks = {
        "g1_at_least_3_of_4_cell_medians_improve": sum(1 for g in gains if g > 0) >= 3,
        "g2_panel_median_relative_improvement_ge_0.5pct": panel_median >= 0.5,
        "g3_worst_cell_degradation_le_0.5pct": min(gains) >= -0.5,
        "g4_not_a_step0_or_near_zero_retreat": (step0_retreats <= 2 and ratio_median >= 0.05),
        "g5_no_leakage_anomaly_or_protocol_violation": g5,
    }
    detail = {
        "g5_compliance": compliance,
        "n_cells": len(cells), "n_seeds": len(rows),
        "cells_improving": sum(1 for g in gains if g > 0),
        "panel_median_relative_gain_pct": panel_median,
        "panel_median_over_12_seed_gains_pct": float(
            st.median([r["relative_val_gain_pct"] for r in rows])),
        "worst_cell_gain_pct": float(min(gains)),
        "best_cell_gain_pct": float(max(gains)),
        "seeds_improving": sum(1 for r in rows if r["relative_val_gain_pct"] > 0),
        "step0_retreat_seeds": step0_retreats,
        "step0_retreat_rule": "<=2 of 12",
        "selected_correction_ratio_median": ratio_median,
        "non_degeneracy_threshold": 0.05,
        "median_gain_excluding_step0_retreats_pct": (
            float(st.median([r["relative_val_gain_pct"] for r in rows if not r["step0_retreat"]]))
            if any(not r["step0_retreat"] for r in rows) else None
        ),
    }
    return {"checks": checks, "detail": detail}
