"""Orchestration for the HCH v4.4 O1 full-8 completion (TRAIN+VAL only).

Subcommands, in the order the protocol allows them:

  --reference-check  prove the 24 R1 references and the 12 reused O1 runs are present
  --smoke            plumbing smoke on a new cell, written outside evidence
  --fits             the 12 registered new fits (4 new cells x seeds 7/17/37)
  --gate             the preregistered 8-cell gate, recomputed from the raw runs
  --plots            the four required diagnostic figures as PNG and SVG
  --finalize         PROBE_CONFIG.json + ACCESS_AUDIT.json + RESULTS.md

The 12 new fits call the O1 stage's own ``probe_train.train_o1`` (imported
read-only), so no trainer is re-implemented here.  No subcommand opens TEST.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
from pathlib import Path

import continuation_common as CC
import probe_common as PC
import probe_train as PT
import recovery_common as RC

U = RC.U
EVID = CC.EVID

CURVE_FIELDS = (
    "check", "step", "is_initial", "is_optimization_step",
    "epoch_equivalent", "samples_seen", "lr",
    "train_reconstruction_mae", "step_loss_optimized", "L_rec", "L_b", "L_B", "L_S",
    "L_total_frozen", "aux_mean",
    "val_mae_ema", "val_mae_raw", "val_host_mae", "val_gain_vs_host_pct",
    "val_correction_ratio", "train_correction_ratio",
    "val_mean_abs_correction", "train_mae_ema", "train_host_mae",
    "used_full_loss_at_this_step", "loss_schedule", "objective_switch_here",
    "best_val_mae_ema_so_far", "best_step_so_far", "selected",
)

MECHANISM_FIELDS = (
    "cell", "market", "host", "seed", "source",
    "o1_selected_step", "r1_selected_step",
    "o1_selected_correction_ratio", "r1_selected_correction_ratio",
    "selected_is_post_switch", "final_check_step",
    "at_400_L_rec", "at_400_L_b", "at_400_L_B", "at_400_L_S", "at_400_aux_mean",
    "at_400_val_mae_ema",
    "at_selected_L_rec", "at_selected_L_b", "at_selected_L_B", "at_selected_L_S",
    "at_selected_aux_mean", "at_selected_val_mae_ema",
    "at_final_check_L_rec", "at_final_check_L_b", "at_final_check_L_B",
    "at_final_check_L_S", "at_final_check_aux_mean", "at_final_check_val_mae_ema",
    "delta_selected_minus_400_L_rec", "delta_selected_minus_400_L_b",
    "delta_selected_minus_400_L_B", "delta_selected_minus_400_L_S",
    "delta_selected_minus_400_aux_mean", "delta_selected_minus_400_val_mae_ema",
    "delta_selected_minus_400_L_rec_direction", "delta_selected_minus_400_L_b_direction",
    "delta_selected_minus_400_L_B_direction", "delta_selected_minus_400_L_S_direction",
    "delta_final_minus_400_L_rec", "delta_final_minus_400_L_b",
    "delta_final_minus_400_L_B", "delta_final_minus_400_L_S",
    "delta_final_minus_400_aux_mean", "delta_final_minus_400_val_mae_ema",
    "delta_final_minus_400_L_rec_direction", "delta_final_minus_400_L_b_direction",
    "delta_final_minus_400_L_B_direction", "delta_final_minus_400_L_S_direction",
    "aux_mean_direction_final", "val_mae_direction_final",
    "aux_better_val_worse_at_final", "aux_worse_val_better_at_final",
    "aux_better_val_worse_at_selected", "aux_worse_val_better_at_selected",
)


def load_curve(market: str, host: str, seed: int) -> list[dict]:
    return json.loads(
        (CC.run_dir(market, host, seed) / "training_curve.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- reference
def phase_reference_check() -> dict:
    RC.install_access_guard()
    ok, present, missing = CC.reference_available()
    ok_o1, o1_present, o1_missing = CC.reused_o1_available()
    if not ok_o1:
        raise SystemExit("REUSED_O1_NOT_AVAILABLE: " + ", ".join(o1_missing))
    prov = CC.reuse_provenance()
    schedules = {json.dumps(CC.schedule_of(
        json.loads((CC.reference_dir(m, h, s) / "freeze.json").read_text(encoding="utf-8"))),
        sort_keys=True) for m, h in CC.PANEL for s in CC.SEEDS}
    prov.update({
        "all_24_references_present": bool(ok),
        "reference_present": present, "reference_missing": missing,
        "all_12_reused_o1_present": bool(ok_o1),
        "reused_o1_present": o1_present, "reused_o1_missing": o1_missing,
        "reference_recipe_as_registered": CC.sealed_reference_recipe(),
        "reference_schedules_distinct": len(schedules),
        "reference_schedule_matches_o1_registered": bool(
            len(schedules) == 1
            and json.loads(next(iter(schedules))) == PC.registered_recipe()),
    })
    U.json_dump(EVID / "REUSE_PROVENANCE.json", prov)
    print(json.dumps({k: prov[k] for k in (
        "all_24_references_present", "all_12_reused_o1_present", "reference_missing",
        "reused_o1_missing", "reused_o1_hashes_match_o1_stage_record",
        "r1_reference_hashes_match_recorded_provenance",
        "n_r1_references_pinned_by_o1_stage", "n_r1_references_pinned_by_published_table",
        "reference_schedules_distinct", "reference_schedule_matches_o1_registered")},
        ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit("REFERENCE_NOT_AVAILABLE: " + ", ".join(missing))
    if not prov["reused_o1_hashes_match_o1_stage_record"]:
        raise SystemExit("REUSED_O1_HASH_MISMATCH: " + "; ".join(prov["reused_o1_mismatches"]))
    if not prov["r1_reference_hashes_match_recorded_provenance"]:
        raise SystemExit("REFERENCE_HASH_MISMATCH: " + "; ".join(prov["r1_reference_mismatches"]))
    return prov


# --------------------------------------------------------------------------- fits
def fit_task(task: dict) -> dict:
    RC.worker_env()
    m, h, s = task["market"], task["host"], task["seed"]
    res = PT.train_o1(m, h, s, CC.run_dir(m, h, s), device=task.get("device", "cuda"))
    rec = res["record"]
    print(f"[o1-full8] {U.cell_key(m, h)} seed{s} step={rec['selected_step']} "
          f"val={rec['selected_val_mae_ema']:.4f} "
          f"stopped={rec['optimization']['stopped_at_step']}", flush=True)
    return rec


def phase_fits(workers: int) -> None:
    RC.install_access_guard()
    ok, _, missing = CC.reference_available()
    if not ok:
        raise SystemExit("REFERENCE_NOT_AVAILABLE: " + ", ".join(missing))
    ok_o1, _, o1_missing = CC.reused_o1_available()
    if not ok_o1:
        raise SystemExit("REUSED_O1_NOT_AVAILABLE: " + ", ".join(o1_missing))
    tasks = [
        {"market": m, "host": h, "seed": s}
        for m, h in CC.NEW_CELLS for s in CC.SEEDS
        if not (CC.run_dir(m, h, s) / "freeze.json").is_file()
    ]
    print(f"[o1-full8] {len(tasks)} new fits pending on {workers} GPU lanes "
          f"(the 12 reused O1 runs are never refit)", flush=True)
    if tasks:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for _ in pool.map(fit_task, tasks):
                pass
    n = sum(1 for m, h in CC.NEW_CELLS for s in CC.SEEDS
            if (CC.run_dir(m, h, s) / "freeze.json").is_file())
    print(f"[o1-full8] {n}/12 new freeze records present", flush=True)


# --------------------------------------------------------------------------- csv
def _csv_cell(value) -> str:
    """RFC4180 quoting: some fields legitimately contain commas (e.g. float lists)."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else (
        json.dumps(value) if isinstance(value, (list, dict)) else str(value))
    if any(ch in text for ch in ',"\r\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(_csv_cell(row.get(c)) for c in header) + "\n")


def phase_per_step_csv() -> Path:
    path = EVID / "PER_STEP_CURVES.csv"
    rows = [
        {"cell": U.cell_key(m, h), "market": m, "host": h, "seed": s,
         "source": CC.source_of(m, h), **e}
        for m, h in CC.PANEL for s in CC.SEEDS for e in load_curve(m, h, s)
    ]
    _write_csv(path, ["cell", "market", "host", "seed", "source", *CURVE_FIELDS], rows)
    return path


# --------------------------------------------------------------------------- gate
def phase_gate() -> dict:
    RC.install_access_guard()
    rows = CC.paired_rows()
    cells = CC.cell_medians(rows)
    comp = CC.compliance()
    g = CC.gate(rows, cells, comp)
    verdict = ("AUXILIARY_PERSISTENCE_BROADLY_HARMFUL_ON_RECOVERY_PANEL"
               if all(g["checks"].values())
               else "AUXILIARY_PERSISTENCE_NOT_BROADLY_HARMFUL_ON_RECOVERY_PANEL")
    positions = {f"{U.cell_key(m, h)}__seed{s}": CC.val_optimum_position(m, h, s)
                 for m, h in CC.PANEL for s in CC.SEEDS}
    payload = {
        "schema": "hch_v44_o1_full8_gate.v1",
        "protocol_id": CC.PROTOCOL_ID,
        "recipe": PC.RECIPE_O1,
        "panel": CC.PANEL_KEYS,
        "new_cells": [U.cell_key(m, h) for m, h in CC.NEW_CELLS],
        "reused_cells": [U.cell_key(m, h) for m, h in CC.REUSED_CELLS],
        "reference": {"path": str(CC.R1_RUNS.relative_to(CC.REPO)).replace("\\", "/"),
                      "role": "reused, not retrained", "n_paired": len(rows)},
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
        "preregistered_before_any_new_fit": True,
        "is_paper_promotion_gate": False,
        "not_a_sota_claim": True,
        "test_target_read_count": 0,
    }
    U.json_dump(EVID / "GATE.json", payload)
    _write_csv(EVID / "PAIRED_PER_SEED.csv", list(rows[0].keys()), rows)
    _write_csv(EVID / "CELL_MEDIANS.csv", list(cells[0].keys()), cells)
    mech = CC.mechanism_rows()
    for row in mech:
        missing = [f for f in MECHANISM_FIELDS if f not in row]
        if missing:
            raise RuntimeError(f"mechanism schema drift: {missing} absent from the row")
    _write_csv(EVID / "MECHANISM_SUMMARY.csv", list(MECHANISM_FIELDS), mech)
    phase_per_step_csv()
    print(json.dumps({"checks": g["checks"], "detail": g["detail"], "verdict": verdict},
                     ensure_ascii=False, indent=2))
    return payload


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
        ("val_mae_ema", "VAL MAE (EMA = primary, raw = diagnostic)", "val_mae_vs_step", True),
        ("loss_components", "loss components", "loss_components_vs_step", False),
        ("val_correction_ratio", "correction / residual ratio", "correction_ratio_vs_step", False),
    ]
    written = []
    for field, ylabel, stem, overlay_raw in panels:
        fig, axes = plt.subplots(4, 2, figsize=(13, 16), sharex=True)
        for ax, (m, h) in zip(axes.ravel(), CC.PANEL):
            cell = U.cell_key(m, h)
            tag = "new" if CC.is_new_cell(m, h) else "reused"
            for s in CC.SEEDS:
                curve = load_curve(m, h, s)
                steps = [e["step"] for e in curve]
                if field == "loss_components":
                    for key, style in (("L_rec", "-"), ("L_b", "--"), ("L_B", ":"),
                                       ("L_S", "-.")):
                        ax.plot(steps, [e[key] for e in curve], style, lw=1.0, alpha=0.8,
                                label=f"seed{s} {key}" if s == CC.SEEDS[0] else None)
                else:
                    ax.plot(steps, [e[field] for e in curve], "-", lw=1.2, label=f"O1 seed{s}")
                if overlay_raw:
                    ax.plot(steps, [e["val_mae_raw"] for e in curve], "--", lw=0.9, alpha=0.6,
                            label=f"O1 raw seed{s}")
                    r1 = json.loads((CC.reference_dir(m, h, s) / "training_curve.json")
                                    .read_text(encoding="utf-8"))
                    ax.plot([e["step"] for e in r1], [e["val_mae_ema"] for e in r1],
                            "-", lw=0.9, color="0.45", alpha=0.8, label=f"R1 seed{s}")
            ax.axvline(CC.SWITCH_STEP, color="crimson", lw=1.1, ls="--")
            ax.set_title(f"{cell} [{tag}]   (dashed red = objective switch at step "
                         f"{CC.SWITCH_STEP})", fontsize=9)
            ax.set_ylabel(ylabel if field != "loss_components" else "loss (log)", fontsize=8)
            ax.set_xlabel("optimizer step", fontsize=8)
            if field == "loss_components":
                ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6, ncol=2)
        fig.suptitle(f"HCH v4.4 O1 full-8 completion — {ylabel}", fontsize=11)
        fig.tight_layout()
        for ext in ("png", "svg"):
            p = out / f"{stem}.{ext}"
            fig.savefig(p, dpi=130)
            written.append(str(p.relative_to(CC.REPO)).replace("\\", "/"))
        plt.close(fig)
    print(json.dumps(written, indent=2))
    return written


# --------------------------------------------------------------------------- finalize
def phase_finalize() -> dict:
    RC.install_access_guard()
    gate = json.loads((EVID / "GATE.json").read_text(encoding="utf-8"))
    reuse = json.loads((EVID / "REUSE_PROVENANCE.json").read_text(encoding="utf-8"))
    runs = {}
    for m, h in CC.PANEL:
        for s in CC.SEEDS:
            key = f"{U.cell_key(m, h)}__seed{s}"
            rec = CC.load_json(CC.run_dir(m, h, s) / "freeze.json")
            runs[key] = {
                "source": CC.source_of(m, h),
                "selected_step": rec["selected_step"],
                "selected_val_mae_ema": rec["selected_val_mae_ema"],
                "selected_ema_parameter_hash": rec["selected_ema_parameter_hash"],
                "freeze_sha256": U.sha256(CC.run_dir(m, h, s) / "freeze.json"),
                "curve_sha256": U.sha256(CC.run_dir(m, h, s) / "training_curve.json"),
                "probe_code_hash": rec["probe_code_hash"],
                "test_target_read_count": rec["test_target_read_count"],
            }
    config = {
        "schema": "hch_v44_o1_full8_config.v1",
        "protocol_id": CC.PROTOCOL_ID,
        "recipe": PC.RECIPE_O1,
        "panel": CC.PANEL_KEYS,
        "new_cells": [U.cell_key(m, h) for m, h in CC.NEW_CELLS],
        "reused_cells": [U.cell_key(m, h) for m, h in CC.REUSED_CELLS],
        "seeds": CC.SEEDS,
        "n_registered_runs": CC.N_RUNS,
        "n_new_fits": CC.N_NEW_FITS,
        "n_reused_o1_runs": len(CC.REUSED_CELLS) * len(CC.SEEDS),
        "trainer": ("probe_train.train_o1 imported read-only from "
                    "experiments/current/hch_v44_objective_alignment_probe_20260917/implementation"),
        "reference": {"source": reuse.get("r1_reference_hashes_match_recorded_provenance"),
                      "all_24_present": reuse.get("all_24_references_present"),
                      "reused_not_retrained": True},
        "source_tree_digest": RC.source_tree_digest(),
        "runs": runs,
        "test_target_read_count": 0,
        "verdict": gate["verdict"],
        "n_gate_passed": gate["n_passed"],
        "not_a_sota_claim": True,
    }
    U.json_dump(EVID / "PROBE_CONFIG.json", config)
    CC.write_access_audit(EVID / "ACCESS_AUDIT.json", {
        "phase": "finalize", "probe_config": "PROBE_CONFIG.json", "gate": "GATE.json",
        "reuse_provenance": "REUSE_PROVENANCE.json",
    })
    _write_results(gate, config)
    print(json.dumps({"verdict": gate["verdict"], "n_passed": gate["n_passed"]}, indent=2))
    return config


def _write_results(gate: dict, config: dict) -> None:
    d = gate["detail"]
    lines = [
        "# HCH v4.4 O1 full-8 completion — TRAIN+VAL only, TEST quarantined",
        "",
        f"Gate verdict: **{gate['verdict']}** ({gate['n_passed']}/{len(gate['checks'])} conditions)",
        "",
        "## What changed",
        "",
        "Nothing in the method. The frozen O1 objective schedule is extended to the full 8-cell R1",
        "recovery panel: updates 1..400 use `L_rec + (L_b + L_B + L_S)/3`, updates 401..2000 use",
        "`L_rec` alone. The 12 new fits execute the O1 stage's own `train_o1` imported read-only;",
        "architecture, geometry, features, initialization, optimizer, learning rate, batch size,",
        "weight decay, gradient clip, EMA, seeds, data split, history support and the stopping rule",
        "are unchanged and `src/core/**` was not edited.",
        "",
        "## Reuse",
        "",
        f"- the 24 R1 full-loss references (`R1_STEP_BUDGET_2000`) are reused, not retrained, and",
        "  pinned by hash in `REUSE_PROVENANCE.json`;",
        "- the 12 O1 runs of the four original cells are reused **in place** from the prior probe's",
        "  evidence root and are not rerun or copied.",
        "",
        "## Eight-cell medians",
        "",
        "| cell | source | O1 median VAL MAE | R1 median VAL MAE | median relative gain |",
        "|---|---|---|---|---|",
    ]
    for c in gate["per_cell"]:
        lines.append(
            f"| {c['cell']} | {c['source']} | {c['mae_o1_median']:.4f} | {c['mae_r1_median']:.4f} | "
            f"{c['cell_median_relative_gain_pct']:+.4f}% |")
    lines += [
        "",
        f"- panel median relative gain (median of the eight cell medians): "
        f"**{d['panel_median_relative_gain_pct']:+.4f}%**",
        f"- cells improving: {d['cells_improving']}/{d['n_cells']} "
        f"(need {d['cells_improving_needed']}); seeds improving: "
        f"{d['seeds_improving']}/{d['n_seeds']}",
        f"- worst / best cell: {d['worst_cell_gain_pct']:+.4f}% / {d['best_cell_gain_pct']:+.4f}%",
        f"- step-0 retreat seeds: {d['step0_retreat_seeds']} "
        f"({d['step0_retreat_rule_proportional']} proportional reading: "
        f"{'PASS' if d['step0_retreat_rule_proportional_pass'] else 'FAIL'}; "
        f"{d['step0_retreat_rule_absolute_reading']} absolute reading: "
        f"{'PASS' if d['step0_retreat_rule_absolute_pass'] else 'FAIL'})",
        f"- median selected correction ratio {d['selected_correction_ratio_median']:.4f} "
        f"(rule >= {d['non_degeneracy_threshold']})",
        f"- new-cell sub-panel median {d['new_cell_subpanel_median_gain_pct']:+.4f}%; "
        f"reused-cell sub-panel median {d['reused_cell_subpanel_median_gain_pct']:+.4f}%",
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
        "This is a **diagnostic** verdict, not a paper-promotion gate. The thresholds were",
        "preregistered in `PROTOCOL.md` before any new fit.",
        "",
        "## Not a SOTA claim",
        "",
        "V2 TEST was observed before this line of work and stays quarantined from every decision",
        "here. The stage read no TEST target, prediction or metric. No paper claim follows.",
        "",
        "## Evidence",
        "",
        "- `REUSE_PROVENANCE.json` — the reused O1 runs and the R1 references, by hash;",
        "- `PER_STEP_CURVES.csv` — the full per-check logging for all 24 runs;",
        "- `PAIRED_PER_SEED.csv`, `CELL_MEDIANS.csv` — the paired comparison and the medians;",
        "- `MECHANISM_SUMMARY.csv` — the post-switch auxiliary directions (interpretation only);",
        "- `GATE.json` — gate arithmetic, compliance audit and the verdict;",
        "- `PLOTS/` — the four required figures as PNG and SVG;",
        "- `PROBE_CONFIG.json`, `ACCESS_AUDIT.json` — the exact config and the access audit;",
        "- `INDEPENDENT_VERIFICATION_REPORT.json` — the independent verifier's verdict.",
        "",
    ]
    (EVID / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- smoke
def phase_smoke() -> dict:
    """Plumbing smoke on a new cell, crossing the switch, written outside evidence."""
    RC.install_access_guard()
    m, h = CC.NEW_CELLS[0]
    out = CC.STAGE / "build" / "smoke" / U.cell_key(m, h)
    res = PT.train_o1(m, h, 7, out, max_steps=460, val_every=20, no_stop_before=0,
                      patience_checks=99, device="cuda",
                      recipe_id="SMOKE_CROSSES_SWITCH_460")
    curve = res["curve"]
    marks = [e["step"] for e in curve if e["objective_switch_here"]]
    optim = [e for e in curve if e["is_optimization_step"]]
    full = [e["step"] for e in optim if e["used_full_loss_at_this_step"]]
    light = [e["step"] for e in optim if not e["used_full_loss_at_this_step"]]
    payload = {
        "cell": U.cell_key(m, h), "seed": 7, "max_steps": 460, "n_checks": len(curve),
        "switch_marker_steps": marks,
        "checks_using_full_loss": full,
        "checks_using_reconstruction_only": light,
        "step0_is_calibration_check": bool(curve[0]["is_initial"]),
        "switch_correct": bool(marks == [CC.SWITCH_STEP] and full
                               and max(full) == CC.SWITCH_STEP
                               and light and min(light) == CC.SWITCH_STEP + 20),
        "step0_val_mae_ema": float(curve[0]["val_mae_ema"]),
        "selected_step": res["record"]["selected_step"],
        "test_target_read_count": res["record"]["test_target_read_count"],
        "test_rows_dropped_from_joint_cache":
            res["record"]["test_rows_dropped_from_joint_cache"],
        "probe_code_hash": res["record"]["probe_code_hash"],
        "is_registered_fit": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    U.json_dump(out.parent / "SMOKE_REPORT.json", payload)
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
    ap.add_argument("--all", action="store_true",
                    help="reference-check, fits, gate, plots, finalize")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    if args.all:
        phase_reference_check()
        phase_fits(args.workers)
        phase_gate()
        phase_plots()
        phase_finalize()
    elif args.reference_check:
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
