"""Orchestration for the HCH v4.4 objective-alignment probe (TRAIN+VAL only).

Subcommands, in the order the protocol allows them:

  --reference-check  prove a legal step-based R1 reference exists for all 12 coords
  --smoke            plumbing smoke that crosses the step-400 switch, outside evidence
  --fits             the 12 registered O1 fits (4 cells x seeds 7/17/37)
  --gate             preregistered gate arithmetic, recomputed from the raw runs
  --plots            the four required diagnostic figures as PNG and SVG
  --finalize         PROBE_CONFIG.json + RESULTS.md + the provenance audits

No subcommand opens TEST.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import statistics as st
from pathlib import Path

import numpy as np

import probe_common as PC
import probe_train as PT
import recovery_common as RC

U = RC.U
EVID = PC.EVID

CURVE_FIELDS = (
    "check", "step", "epoch_equivalent", "samples_seen", "lr",
    "train_reconstruction_mae", "step_loss_optimized", "L_rec", "L_b", "L_B", "L_S",
    "L_total_frozen", "aux_mean",
    "val_mae_ema", "val_mae_raw", "val_host_mae", "val_gain_vs_host_pct",
    "val_correction_ratio", "train_correction_ratio",
    "val_mean_abs_correction", "train_mae_ema", "train_host_mae",
    "used_full_loss_at_this_step", "loss_schedule", "objective_switch_here",
    "best_val_mae_ema_so_far", "best_step_so_far", "selected",
)


# --------------------------------------------------------------------------- reference
def phase_reference_check() -> dict:
    RC.install_access_guard()
    ok, present, missing = PC.reference_available()
    recipe = PC.sealed_reference_recipe()
    per_coord = {}
    schedules = set()
    for m, h in PC.PANEL:
        for s in PC.SEEDS:
            key = f"{U.cell_key(m, h)}__seed{s}"
            if key in missing:
                continue
            ref = PC.load_reference(m, h, s)
            r = ref["record"]
            schedules.add(json.dumps(PC.registered_schedule_of(r["optimization"]),
                                     sort_keys=True))
            per_coord[key] = {
                "recipe_id": r["recipe_id"],
                "selected_val_mae_ema": float(r["selected_val_mae_ema"]),
                "selected_step": int(r["selected_step"]),
                "n_train_rows": int(r["n_train_rows"]),
                "n_val_rows": int(r["n_val_rows"]),
                "val_day_ids_hash": r["val_day_ids_hash"],
                "train_day_ids_hash": r["train_day_ids_hash"],
                "history_support_hash": r["history_support_hash"],
                "scales_fingerprint": r["scales"]["fingerprint"],
                "test_target_read_count": int(r.get("test_target_read_count", 0)),
                "freeze_sha256": ref["freeze_sha256"],
                "curve_sha256": ref["curve_sha256"],
                "checkpoint_sha256": ref["checkpoint_sha256"],
                "is_full_loss_run": bool(r["recipe_id"] == "R1_STEP_BUDGET_2000"),
            }
    payload = {
        "schema": "hch_v44_objective_probe_reference.v1",
        "protocol_id": PC.PROTOCOL_ID,
        "role": "reused reference; no retraining",
        "reference_stage": "hch_v44_optimization_recovery_20260917",
        "reference_path": str(PC.R1_RUNS.relative_to(PC.REPO)).replace("\\", "/"),
        "reference_recipe_as_registered": recipe,
        "reference_schedules_distinct": len(schedules),
        "reference_schedule": json.loads(next(iter(schedules))) if len(schedules) == 1 else None,
        "reference_schedule_note": (
            "compared on the registered keys only; stopped_at_step is an outcome and "
            "legitimately differs between runs"
        ),
        "reference_matches_o1_registered_schedule": bool(
            len(schedules) == 1
            and json.loads(next(iter(schedules))) == PC.registered_recipe()
        ),
        "all_12_present": bool(ok),
        "present": present, "missing": missing,
        "per_coordinate": per_coord,
        "note": ("Reference availability is decided before any fit.  If it had been "
                 "incomplete the probe would have reported REFERENCE_NOT_AVAILABLE and "
                 "stopped without training anything."),
    }
    U.json_dump(EVID / "REFERENCE_PROVENANCE.json", payload)
    print(json.dumps({k: payload[k] for k in
                      ("all_12_present", "missing", "reference_schedules_distinct")},
                     ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit("REFERENCE_NOT_AVAILABLE: " + ", ".join(missing))
    return payload


# --------------------------------------------------------------------------- fits
def fit_task(task: dict) -> dict:
    RC.worker_env()
    m, h, s = task["market"], task["host"], task["seed"]
    res = PT.train_o1(m, h, s, PC.run_dir(m, h, s), device=task.get("device", "cuda"))
    rec = res["record"]
    print(f"[o1] {U.cell_key(m, h)} seed{s} step={rec['selected_step']} "
          f"val={rec['selected_val_mae_ema']:.4f} "
          f"stopped={rec['optimization']['stopped_at_step']}", flush=True)
    return rec


def phase_fits(workers: int) -> None:
    RC.install_access_guard()
    ok, _, missing = PC.reference_available()
    if not ok:
        raise SystemExit("REFERENCE_NOT_AVAILABLE: " + ", ".join(missing))
    tasks = [
        {"market": m, "host": h, "seed": s}
        for m, h in PC.PANEL for s in PC.SEEDS
        if not (PC.run_dir(m, h, s) / "freeze.json").is_file()
    ]
    print(f"[o1] {len(tasks)} fits pending on {workers} GPU lanes", flush=True)
    if tasks:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for _ in pool.map(fit_task, tasks):
                pass
    n = sum(1 for m, h in PC.PANEL for s in PC.SEEDS
            if (PC.run_dir(m, h, s) / "freeze.json").is_file())
    print(f"[o1] {n}/12 freeze records present", flush=True)


def load_curve(market: str, host: str, seed: int) -> list[dict]:
    return json.loads(
        (PC.run_dir(market, host, seed) / "training_curve.json").read_text(encoding="utf-8"))


def phase_per_step_csv() -> Path:
    path = EVID / "PER_STEP_CURVES.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("cell,market,host,seed," + ",".join(CURVE_FIELDS) + "\n")
        for m, h in PC.PANEL:
            for s in PC.SEEDS:
                for e in load_curve(m, h, s):
                    fh.write(",".join(
                        [U.cell_key(m, h), m, h, str(s)]
                        + ["" if e[f] is None else str(e[f]) for f in CURVE_FIELDS]
                    ) + "\n")
    return path


# --------------------------------------------------------------------------- gate
def compliance() -> dict:
    """The audited g5 facts, read off the freeze records and curves."""
    access_bad, nonfinite, switch_bad, schedule_bad = {}, {}, {}, {}
    schedules = set()
    for m, h in PC.PANEL:
        for s in PC.SEEDS:
            key = f"{U.cell_key(m, h)}__seed{s}"
            rec = PC.load_json(PC.run_dir(m, h, s) / "freeze.json")
            st_ = rec.get("access_state", {})
            if (int(rec.get("test_target_read_count", 0)) != 0
                    or int(rec.get("test_rows_materialised", 0)) != 0
                    or st_.get("distinct_paths_blocked")
                    or int(st_.get("test_rows_returned", 0)) != 0):
                access_bad[key] = st_
            if not np.isfinite([rec["selected_val_mae_ema"],
                                rec["selected_metrics"]["val_host_mae"],
                                rec["selected_metrics"]["L_rec"]]).all():
                nonfinite[key] = "non-finite metric"
            curve = load_curve(m, h, s)
            marks = [e for e in curve if e["objective_switch_here"]]
            if len(marks) != 1 or int(marks[0]["step"]) != PC.SWITCH_STEP:
                switch_bad[key] = f"switch markers at {[e['step'] for e in marks]}"
            for e in curve:
                if not e["is_optimization_step"]:
                    continue  # the step-0 calibration check precedes any update
                expect_full = int(e["step"]) <= PC.SWITCH_STEP
                if bool(e["used_full_loss_at_this_step"]) != expect_full:
                    switch_bad[key] = f"step {e['step']} used_full={e['used_full_loss_at_this_step']}"
                    break
            schedules.add(json.dumps(PC.registered_schedule_of(rec["optimization"]),
                                     sort_keys=True))
    integrity = {"n_distinct_schedules": len(schedules),
                 "compared_keys": list(PC.REGISTERED_SCHEDULE_KEYS),
                 "stopped_at_step_excluded": "an outcome, not a setting"}
    if len(schedules) == 1:
        sched, registered = json.loads(next(iter(schedules))), PC.registered_recipe()
        for k in PC.REGISTERED_SCHEDULE_KEYS:
            if sched.get(k) != registered[k]:
                schedule_bad[k] = f"{sched.get(k)!r} != registered {registered[k]!r}"
    else:
        schedule_bad["n_distinct_schedules"] = len(schedules)
    return {
        "access_clean": not access_bad, "access_violations": access_bad,
        "all_finite": not nonfinite, "non_finite": nonfinite,
        "switch_exactly_after_400": not switch_bad, "switch_violations": switch_bad,
        "schedule_unchanged": not schedule_bad, "schedule_violations": schedule_bad,
        "schedule_integrity": integrity,
    }


def phase_gate() -> dict:
    RC.install_access_guard()
    rows = PC.paired_rows()
    cells = PC.cell_medians(rows)
    comp = compliance()
    g = PC.gate(rows, cells, comp)
    verdict = ("AUXILIARY_PERSISTENCE_LIKELY_HARMFUL" if all(g["checks"].values())
               else "AUXILIARY_PERSISTENCE_NOT_SHOWN_HARMFUL")
    positions = {
        f"{U.cell_key(m, h)}__seed{s}": PC.val_optimum_position(m, h, s)
        for m, h in PC.PANEL for s in PC.SEEDS
    }
    payload = {
        "schema": "hch_v44_objective_probe_gate.v1",
        "protocol_id": PC.PROTOCOL_ID,
        "recipe": PC.RECIPE_O1,
        "reference": {
            "path": str(PC.R1_RUNS.relative_to(PC.REPO)).replace("\\", "/"),
            "role": "reused, not retrained",
            "n_paired": len(rows),
        },
        "per_seed": rows,
        "per_cell": cells,
        "val_optimum_position": positions,
        "compliance": comp,
        "checks": g["checks"],
        "detail": g["detail"],
        "n_passed": sum(1 for v in g["checks"].values() if v),
        "n_failed": sum(1 for v in g["checks"].values() if not v),
        "failed": [k for k, v in g["checks"].items() if not v],
        "verdict": verdict,
        "is_paper_promotion_gate": False,
        "not_a_sota_claim": True,
        "test_target_read_count": 0,
    }
    U.json_dump(EVID / "GATE.json", payload)
    _write_csv(EVID / "PAIRED_PER_SEED.csv", rows)
    _write_csv(EVID / "CELL_MEDIANS.csv", cells)
    _write_per_step_csv_alias()
    print(json.dumps({"checks": g["checks"], "detail": g["detail"], "verdict": verdict},
                     ensure_ascii=False, indent=2))
    return payload


def _write_per_step_csv_alias() -> None:
    phase_per_step_csv()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for row in rows:
            fh.write(",".join(
                json.dumps(row[c]) if isinstance(row[c], (list, dict)) else str(row[c])
                for c in cols) + "\n")


# --------------------------------------------------------------------------- plots
def phase_plots() -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RC.install_access_guard()
    out = EVID / "PLOTS"
    out.mkdir(parents=True, exist_ok=True)
    panels = [
        ("train_reconstruction_mae", "TRAIN reconstruction MAE (diagnostic subset)",
         "train_curve_reconstruction_mae", False),
        ("val_mae_ema", "VAL MAE (EMA = primary, raw = diagnostic)",
         "val_mae_vs_step", True),
        ("loss_components", "loss components", "loss_components_vs_step", False),
        ("val_correction_ratio", "correction / residual ratio", "correction_ratio_vs_step", False),
    ]
    written = []
    for field, ylabel, stem, overlay_raw in panels:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
        for ax, (m, h) in zip(axes.ravel(), PC.PANEL):
            cell = U.cell_key(m, h)
            for s in PC.SEEDS:
                o1 = load_curve(m, h, s)
                steps = [e["step"] for e in o1]
                if field == "loss_components":
                    for key, style in (("L_rec", "-"), ("L_b", "--"), ("L_B", ":"),
                                       ("L_S", "-.")):
                        ax.plot(steps, [e[key] for e in o1], style, lw=1.0, alpha=0.8,
                                label=f"seed{s} {key}" if s == PC.SEEDS[0] else None)
                else:
                    ax.plot(steps, [e[field] for e in o1], "-", lw=1.2,
                            label=f"O1 seed{s}")
                if overlay_raw:
                    ax.plot(steps, [e["val_mae_raw"] for e in o1], "--", lw=0.9, alpha=0.6,
                            label=f"O1 raw seed{s}")
                    r1 = json.loads((PC.reference_dir(m, h, s) / "training_curve.json")
                                    .read_text(encoding="utf-8"))
                    ax.plot([e["step"] for e in r1], [e["val_mae_ema"] for e in r1],
                            "-", lw=0.9, color="0.45", alpha=0.8,
                            label=f"R1 seed{s}")
            ax.axvline(PC.SWITCH_STEP, color="crimson", lw=1.1, ls="--")
            ax.set_title(f"{cell}   (dashed red = objective switch at step {PC.SWITCH_STEP})",
                         fontsize=9)
            ax.set_ylabel(ylabel if field != "loss_components" else "loss (log)", fontsize=8)
            ax.set_xlabel("optimizer step", fontsize=8)
            if field == "loss_components":
                ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6, ncol=2)
        fig.suptitle(f"HCH v4.4 O1 objective probe — {ylabel}", fontsize=11)
        fig.tight_layout()
        for ext in ("png", "svg"):
            p = out / f"{stem}.{ext}"
            fig.savefig(p, dpi=150)
            written.append(str(p.relative_to(PC.REPO)).replace("\\", "/"))
        plt.close(fig)
    print(json.dumps(written, indent=2))
    return written


# --------------------------------------------------------------------------- finalize
def phase_finalize() -> dict:
    RC.install_access_guard()
    gate = json.loads((EVID / "GATE.json").read_text(encoding="utf-8"))
    reference = json.loads((EVID / "REFERENCE_PROVENANCE.json").read_text(encoding="utf-8"))
    runs = {}
    for m, h in PC.PANEL:
        for s in PC.SEEDS:
            rec = PC.load_json(PC.run_dir(m, h, s) / "freeze.json")
            runs[f"{U.cell_key(m, h)}__seed{s}"] = {
                "selected_step": rec["selected_step"],
                "selected_val_mae_ema": rec["selected_val_mae_ema"],
                "selected_ema_parameter_hash": rec["selected_ema_parameter_hash"],
                "freeze_sha256": U.sha256(PC.run_dir(m, h, s) / "freeze.json"),
                "curve_sha256": U.sha256(PC.run_dir(m, h, s) / "training_curve.json"),
                "test_target_read_count": rec["test_target_read_count"],
            }
    config = {
        "schema": "hch_v44_objective_probe_config.v1",
        "protocol_id": PC.PROTOCOL_ID,
        "recipe": PC.RECIPE_O1,
        "panel": [U.cell_key(m, h) for m, h in PC.PANEL],
        "seeds": PC.SEEDS,
        "n_registered_fits": len(PC.PANEL) * len(PC.SEEDS),
        "reference": {
            "source": reference["reference_path"],
            "all_12_present": reference["all_12_present"],
            "n_distinct_schedules": reference["reference_schedules_distinct"],
            "reused_not_retrained": True,
        },
        "source_tree_digest": RC.source_tree_digest(),
        "runs": runs,
        "test_target_read_count": 0,
        "verdict": gate["verdict"],
        "n_gate_passed": gate["n_passed"],
        "not_a_sota_claim": True,
    }
    U.json_dump(EVID / "PROBE_CONFIG.json", config)
    RC.write_access_audit(EVID / "ACCESS_AUDIT.json", {
        "phase": "finalize", "probe_config": "PROBE_CONFIG.json",
        "gate": "GATE.json", "reference_provenance": "REFERENCE_PROVENANCE.json",
    })
    _write_results(gate, config)
    print(json.dumps({"verdict": gate["verdict"], "n_passed": gate["n_passed"]}, indent=2))
    return config


def _write_results(gate: dict, config: dict) -> None:
    d = gate["detail"]
    lines = [
        "# HCH v4.4 objective-alignment probe (O1) — TRAIN+VAL only, TEST quarantined",
        "",
        f"Gate verdict: **{gate['verdict']}** ({gate['n_passed']}/{len(gate['checks'])} conditions)",
        "",
        "## What changed",
        "",
        "Exactly one thing: updates 1..400 use the frozen full loss "
        "`L_rec + (L_b + L_B + L_S)/3`, updates 401..2000 use `L_rec` alone. "
        "Architecture, geometry, features, initialization, optimizer, learning rate, batch size, "
        "weight decay, gradient clip, EMA, seeds, data split, history support and the stopping "
        "rule are unchanged, and `src/core/**` was not edited.",
        "",
        "## Reference",
        "",
        "The same-cell, same-seed full-loss step-based R1 runs of the closed recovery stage were "
        "**reused, not retrained** "
        f"({config['reference']['n_distinct_schedules']} distinct reference schedule, "
        "all 12 coordinates present). See `REFERENCE_PROVENANCE.json`.",
        "",
        "## Four-cell medians",
        "",
        "| cell | O1 median VAL MAE | R1 median VAL MAE | median relative gain |",
        "|---|---|---|---|",
    ]
    for c in gate["per_cell"]:
        lines.append(
            f"| {c['cell']} | {c['mae_o1_median']:.4f} | {c['mae_r1_median']:.4f} | "
            f"{c['cell_median_relative_gain_pct']:+.4f}% |"
        )
    lines += [
        "",
        f"- panel median relative gain (median of the four cell medians): "
        f"**{d['panel_median_relative_gain_pct']:+.4f}%**",
        f"- cells improving: {d['cells_improving']}/{d['n_cells']}; seeds improving: "
        f"{d['seeds_improving']}/{d['n_seeds']}",
        f"- worst / best cell: {d['worst_cell_gain_pct']:+.4f}% / {d['best_cell_gain_pct']:+.4f}%",
        f"- step-0 retreat seeds: {d['step0_retreat_seeds']} (rule <=2); "
        f"median selected correction ratio {d['selected_correction_ratio_median']:.4f} "
        f"(rule >=0.05)",
        "",
        "## Gate",
        "",
        "| condition | result |",
        "|---|---|",
    ]
    for k, v in gate["checks"].items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines += [
        "",
        "This is a **diagnostic** verdict, not a paper-promotion gate.",
        "",
        "## Not a SOTA claim",
        "",
        "V2 TEST was observed before this probe and stays quarantined from every decision here. "
        "The probe read no TEST target, prediction or metric. No paper claim follows from it.",
        "",
        "## Evidence",
        "",
        "- `REFERENCE_PROVENANCE.json` — the reused reference, with per-coordinate hashes;",
        "- `PER_STEP_CURVES.csv` — the full per-check logging;",
        "- `PAIRED_PER_SEED.csv`, `CELL_MEDIANS.csv` — the paired comparison and the medians;",
        "- `GATE.json` — gate arithmetic, compliance audit and the verdict;",
        "- `PLOTS/` — the four required figures as PNG and SVG;",
        "- `PROBE_CONFIG.json`, `ACCESS_AUDIT.json` — the exact config and the access audit;",
        "- `INDEPENDENT_VERIFICATION_REPORT.json` — the independent verifier's verdict.",
        "",
    ]
    (EVID / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- smoke
def phase_smoke() -> dict:
    """Plumbing smoke that crosses the switch, written outside evidence."""
    RC.install_access_guard()
    out = PC.STAGE / "build" / "smoke" / "QINGHAI_DA__TimeMixer"
    res = PT.train_o1("QINGHAI_DA", "TimeMixer", 7, out, max_steps=460, val_every=20,
                      no_stop_before=0, patience_checks=99, device="cuda",
                      recipe_id="SMOKE_CROSSES_SWITCH_460")
    curve = res["curve"]
    marks = [e["step"] for e in curve if e["objective_switch_here"]]
    optim = [e for e in curve if e["is_optimization_step"]]
    full = [e["step"] for e in optim if e["used_full_loss_at_this_step"]]
    light = [e["step"] for e in optim if not e["used_full_loss_at_this_step"]]
    payload = {
        "cell": "QINGHAI_DA__TimeMixer", "seed": 7, "max_steps": 460,
        "n_checks": len(curve),
        "switch_marker_steps": marks,
        "checks_using_full_loss": full,
        "checks_using_reconstruction_only": light,
        "step0_is_calibration_check": bool(curve[0]["is_initial"]),
        "switch_correct": bool(marks == [PC.SWITCH_STEP]
                               and full and max(full) == PC.SWITCH_STEP
                               and light and min(light) == PC.SWITCH_STEP + 20),
        "step0_val_mae_ema": float(curve[0]["val_mae_ema"]),
        "selected_step": res["record"]["selected_step"],
        "test_target_read_count": res["record"]["test_target_read_count"],
        "test_rows_dropped_from_joint_cache":
            res["record"]["test_rows_dropped_from_joint_cache"],
        "is_registered_fit": False,
    }
    U.json_dump(PC.STAGE / "build" / "smoke" / "SMOKE_REPORT.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["switch_correct"]:
        raise SystemExit("SMOKE_FAILED: the objective switch did not land exactly after step 400")
    return payload


# --------------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fits", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    if args.reference_check:
        phase_reference_check()
    elif args.smoke:
        phase_smoke()
    elif args.fits:
        phase_fits(args.workers)
    elif args.gate:
        phase_gate()
    elif args.plots:
        phase_plots()
    elif args.finalize:
        phase_finalize()
    else:
        ap.error("choose one subcommand")


if __name__ == "__main__":
    main()
