"""O1 full-8 completion runtime: TRAIN+VAL only, TEST quarantined.

This stage adds **no new data path and no new trainer**.  It imports the O1
stage's ``probe_common``/``probe_train`` read-only, so the 12 new fits are
produced by the identical code bytes that produced the 12 reused ones, and it
imports the closed recovery stage's ``recovery_common`` through them for the
identical quarantine, causal conventions, shadow-OOF support, TRAIN-only scales
and Host caches.

What this module adds is only the 8-cell panel bookkeeping, the reuse
provenance, the 8-cell gate arithmetic generalized from the O1 protocol, the
mechanism summary and this stage's own evidence root.

Neither the O1 stage nor the recovery stage is edited.  ``sys.dont_write_bytecode``
is set before the imports so that importing the O1 implementation cannot even
touch its ``__pycache__``.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]

O1_STAGE = REPO / "experiments/current/hch_v44_objective_alignment_probe_20260917"
O1_IMPL = O1_STAGE / "implementation"
O1_EVID = REPO / "experiments/evidence/hch_v44_objective_alignment_probe_20260917"

RECOVERY_STAGE = REPO / "experiments/current/hch_v44_optimization_recovery_20260917"
RECOVERY_IMPL = RECOVERY_STAGE / "implementation"
RECOVERY_EVID = REPO / "experiments/evidence/hch_v44_optimization_recovery_20260917"

if str(O1_IMPL) not in sys.path:
    sys.path.insert(0, str(O1_IMPL))

import probe_common as PC  # noqa: E402  (O1 stage, read-only)
import recovery_common as RC  # noqa: E402  (closed recovery stage, read-only)

U = RC.U
EVID = REPO / "experiments/evidence/hch_v44_o1_full8_completion_20260918"
R1_RUNS = RECOVERY_EVID / "r1_runs"

PROTOCOL_ID = "HCH_V44_O1_FULL8_COMPLETION_20260918"
VARIANT = RC.VARIANT
SEEDS = list(U.SEEDS)
SWITCH_STEP = PC.SWITCH_STEP  # 400, fixed and non-tunable

#: The registered R1 recovery-panel cell order, preserved exactly.
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
PANEL_KEYS = [U.cell_key(m, h) for m, h in PANEL]

#: The four O1 cells of the prior probe: reused in place, never rerun.
REUSED_CELLS = [
    ("GANSU_DA", "PatchTST"),
    ("SHANDONG_DA", "iTransformer"),
    ("SHAANXI_DA", "TimeMixer"),
    ("NINGXIA_DA", "iTransformer"),
]
#: The four cells this continuation is allowed to fit.
NEW_CELLS = [
    ("GANSU_DA", "LSTM"),
    ("SHANDONG_DA", "PatchTST"),
    ("SHAANXI_DA", "PatchTST"),
    ("QINGHAI_DA", "TimeMixer"),
]

N_RUNS = len(PANEL) * len(SEEDS)          # 24
N_NEW_FITS = len(NEW_CELLS) * len(SEEDS)  # 12

#: Preregistered 8-cell gate constants.  See PROTOCOL.md section 7.
GATE_MIN_CELLS_IMPROVING = 6              # of 8
GATE_PANEL_MEDIAN_MIN_PCT = 0.5
GATE_WORST_CELL_MIN_PCT = -0.5
GATE_NON_DEGENERACY_RATIO_MIN = 0.05
GATE_STEP0_RETREAT_MAX_PROPORTIONAL = 4   # of 24 == 2 of 12, the proportional extension
GATE_STEP0_RETREAT_MAX_ABSOLUTE = 2       # of 24 == O1's literal "at most 2" retained

COMPONENTS = ("L_rec", "L_b", "L_B", "L_S")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_new_cell(market: str, host: str) -> bool:
    return (market, host) in NEW_CELLS


def source_of(market: str, host: str) -> str:
    return "new_fit" if is_new_cell(market, host) else "reused_o1"


def run_dir(market: str, host: str, seed: int) -> Path:
    """Where this run lives: reused O1 runs stay in the O1 evidence root."""
    root = EVID if is_new_cell(market, host) else O1_EVID
    return root / "o1_runs" / U.cell_key(market, host) / f"seed{seed}"


def reference_dir(market: str, host: str, seed: int) -> Path:
    return R1_RUNS / U.cell_key(market, host) / f"seed{seed}"


def run_dirs(market: str, host: str, seed: int) -> tuple[Path, Path, Path]:
    d = run_dir(market, host, seed)
    return d / "freeze.json", d / "training_curve.json", d / "selected_ema.pt"


def freeze_paths_digest(market: str, host: str, seed: int) -> dict:
    f, c, k = run_dirs(market, host, seed)
    return {"freeze_sha256": U.sha256(f), "curve_sha256": U.sha256(c),
            "checkpoint_sha256": U.sha256(k)}


def reference_digest(market: str, host: str, seed: int) -> dict:
    d = reference_dir(market, host, seed)
    return {"freeze_sha256": U.sha256(d / "freeze.json"),
            "curve_sha256": U.sha256(d / "training_curve.json"),
            "checkpoint_sha256": U.sha256(d / "selected_ema.pt")}


# --------------------------------------------------------------------------- reuse
def reference_available() -> tuple[bool, list[str], list[str]]:
    """A legal step-based R1 full-loss reference for all 24 coordinates?"""
    missing, present = [], []
    for m, h in PANEL:
        for s in SEEDS:
            d = reference_dir(m, h, s)
            key = f"{U.cell_key(m, h)}__seed{s}"
            if (d / "freeze.json").is_file() and (d / "training_curve.json").is_file():
                present.append(key)
            else:
                missing.append(key)
    return (not missing), present, missing


def reused_o1_available() -> tuple[bool, list[str], list[str]]:
    """The 12 prior O1 runs that must be reused rather than rerun."""
    missing, present = [], []
    for m, h in REUSED_CELLS:
        for s in SEEDS:
            f, c, k = run_dirs(m, h, s)
            key = f"{U.cell_key(m, h)}__seed{s}"
            if f.is_file() and c.is_file() and k.is_file():
                present.append(key)
            else:
                missing.append(key)
    return (not missing), present, missing


def _o1_stage_config() -> dict:
    return load_json(O1_EVID / "PROBE_CONFIG.json")


def _o1_stage_reference() -> dict:
    return load_json(O1_EVID / "REFERENCE_PROVENANCE.json")


def r1_published_table() -> dict:
    """The recovery stage's own published per-seed table, all 24 coordinates.

    This is the authoritative numeric record of the R1 runs and the only one that
    covers the four cells the O1 probe never used.
    """
    import csv
    path = RECOVERY_EVID / "R1_PANEL_PER_SEED.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        return {f"{r['cell']}__seed{r['seed']}": r for r in csv.DictReader(fh)}


def reuse_provenance() -> dict:
    """Pin every reused artifact by hash, against the stage that froze it.

    Three independent pins are applied, because the two evidence roots cover
    different subsets:

    * the 12 reused O1 runs are checked against the O1 stage's own
      ``PROBE_CONFIG.json`` (which records their freeze/curve hashes, selected step,
      selected VAL MAE and checkpoint parameter hash);
    * the R1 references are checked against the O1 stage's
      ``REFERENCE_PROVENANCE.json`` where that covers them (the 12 O1 cells) and
      against the recovery stage's published ``R1_PANEL_PER_SEED.csv`` everywhere
      (all 24);
    * every checkpoint is re-hashed and compared to the sha256 the run's *own*
      freeze record claims, so a swapped ``selected_ema.pt`` cannot pass.
    """
    cfg_runs = _o1_stage_config()["runs"]
    o1_ref = _o1_stage_reference()["per_coordinate"]
    published = r1_published_table()

    runs, mismatched = {}, []
    for m, h in REUSED_CELLS:
        for s in SEEDS:
            key = f"{U.cell_key(m, h)}__seed{s}"
            got = freeze_paths_digest(m, h, s)
            fz = load_json(run_dir(m, h, s) / "freeze.json")
            rec = cfg_runs.get(key)
            if rec is None:
                mismatched.append(f"{key}: absent from the O1 stage's PROBE_CONFIG.json")
            else:
                for k in ("freeze_sha256", "curve_sha256"):
                    if rec[k] != got[k]:
                        mismatched.append(f"{key}: {k} {got[k]} != O1-stage {rec[k]}")
                for f in ("selected_step",):
                    if int(rec[f]) != int(fz[f]):
                        mismatched.append(f"{key}: {f} {fz[f]!r} != O1-stage {rec[f]!r}")
                if float(rec["selected_val_mae_ema"]) != float(fz["selected_val_mae_ema"]):
                    mismatched.append(f"{key}: selected_val_mae_ema differs from the O1 stage")
                if fz["selected_ema_parameter_hash"] != rec["selected_ema_parameter_hash"]:
                    mismatched.append(f"{key}: selected checkpoint parameter hash differs from "
                                      "the O1 stage")
            if fz["selected_ema_checkpoint_sha256"] != got["checkpoint_sha256"]:
                mismatched.append(f"{key}: selected_ema.pt does not hash to the sha256 its own "
                                  "freeze record claims")
            if fz.get("test_target_read_count") != 0:
                mismatched.append(f"{key}: reused O1 run reports a TEST read")
            runs[key] = {
                **got,
                "selected_step": int(fz["selected_step"]),
                "selected_val_mae_ema": float(fz["selected_val_mae_ema"]),
                "selected_ema_parameter_hash": fz["selected_ema_parameter_hash"],
                "probe_code_hash": fz["probe_code_hash"],
                "source_evidence_root": str(O1_EVID.relative_to(REPO)).replace("\\", "/"),
                "reused_not_rerun": True,
            }

    refs, ref_bad = {}, []
    for m, h in PANEL:
        for s in SEEDS:
            key = f"{U.cell_key(m, h)}__seed{s}"
            got = reference_digest(m, h, s)
            fz = load_json(reference_dir(m, h, s) / "freeze.json")
            pinned_by = []
            rec = o1_ref.get(key)
            if rec is not None:
                pinned_by.append("o1_stage_reference_provenance")
                for k in ("freeze_sha256", "curve_sha256", "checkpoint_sha256"):
                    if rec[k] != got[k]:
                        ref_bad.append(f"{key}: {k} differs from the O1-stage reference record")
            if key in published:
                pinned_by.append("recovery_stage_published_table")
                pub = published[key]
                if float(pub["val_mae_ema"]) != float(fz["selected_val_mae_ema"]):
                    ref_bad.append(f"{key}: published val_mae_ema {pub['val_mae_ema']} != freeze "
                                   f"{fz['selected_val_mae_ema']}")
                if int(pub["selected_step"]) != int(fz["selected_step"]):
                    ref_bad.append(f"{key}: published selected_step {pub['selected_step']} != freeze "
                                   f"{fz['selected_step']}")
                if pub["checkpoint_parameter_hash"] != fz["selected_ema_parameter_hash"]:
                    ref_bad.append(f"{key}: published checkpoint parameter hash != freeze")
            else:
                ref_bad.append(f"{key}: absent from R1_PANEL_PER_SEED.csv")
            if fz["selected_ema_checkpoint_sha256"] != got["checkpoint_sha256"]:
                ref_bad.append(f"{key}: selected_ema.pt does not hash to the sha256 its own freeze "
                               "record claims")
            if fz["recipe_id"] != "R1_STEP_BUDGET_2000":
                ref_bad.append(f"{key}: recipe_id {fz['recipe_id']!r} is not the R1 step-budget run")
            refs[key] = {
                **got, "recipe_id": fz["recipe_id"],
                "selected_step": int(fz["selected_step"]),
                "selected_val_mae_ema": float(fz["selected_val_mae_ema"]),
                "selected_ema_parameter_hash": fz["selected_ema_parameter_hash"],
                "pinned_by": pinned_by, "reused_not_retrained": True,
            }
    return {
        "schema": "hch_v44_o1_full8_reuse_provenance.v1",
        "protocol_id": PROTOCOL_ID,
        "reused_o1_runs": runs,
        "reused_o1_hashes_match_o1_stage_record": not mismatched,
        "reused_o1_mismatches": mismatched,
        "r1_references": refs,
        "r1_reference_hashes_match_recorded_provenance": not ref_bad,
        "r1_reference_mismatches": ref_bad,
        "n_reused_o1": len(runs), "n_r1_references": len(refs),
        "n_r1_references_pinned_by_o1_stage": sum(
            1 for v in refs.values() if "o1_stage_reference_provenance" in v["pinned_by"]),
        "n_r1_references_pinned_by_published_table": sum(
            1 for v in refs.values() if "recovery_stage_published_table" in v["pinned_by"]),
        "note": ("Reused artifacts are referenced in place, never copied or rerun.  The O1 cells' "
                 "reference is byte-identical to the one the four-cell probe used, so the four-cell "
                 "and eight-cell comparisons share one reference set; the four new cells' reference "
                 "was frozen by the recovery stage and is pinned against its published table."),
    }


def sealed_reference_recipe() -> dict:
    return load_json(RECOVERY_EVID / "R1_GATE.json")["recipe"]


# --------------------------------------------------------------------------- pairing
def paired_rows() -> list[dict]:
    """One row per (cell, seed): O1 (new or reused) vs its paired R1 reference."""
    rows = []
    for m, h in PANEL:
        cell = U.cell_key(m, h)
        for s in SEEDS:
            fz = load_json(run_dir(m, h, s) / "freeze.json")
            r1 = load_json(reference_dir(m, h, s) / "freeze.json")
            mae_o1 = float(fz["selected_val_mae_ema"])
            mae_r1 = float(r1["selected_val_mae_ema"])
            sel = fz["selected_metrics"]
            rows.append({
                "cell": cell, "market": m, "host": h, "seed": s,
                "source": source_of(m, h),
                "mae_o1": mae_o1, "mae_r1": mae_r1,
                "relative_val_gain_pct": (mae_r1 - mae_o1) / mae_r1 * 100.0,
                "abs_delta": mae_r1 - mae_o1,
                "selected_step_o1": int(fz["selected_step"]),
                "selected_step_r1": int(r1["selected_step"]),
                "stopped_at_step_o1": fz["optimization"]["stopped_at_step"],
                "stopped_at_step_r1": r1["optimization"]["stopped_at_step"],
                "correction_ratio_o1": float(sel["val_correction_ratio"]),
                "correction_ratio_r1": float(r1["selected_metrics"]["val_correction_ratio"]),
                "val_mae_raw_o1": float(sel["val_mae_raw"]),
                "raw_minus_ema_gap_o1": float(sel["val_mae_raw"]) - mae_o1,
                "train_recon_mae_o1": float(sel["diag_reconstruction_mae"]),
                "train_recon_mae_r1": float(r1["selected_metrics"]["diag_reconstruction_mae"]),
                "val_host_mae_o1": float(sel["val_host_mae"]),
                "n_val_rows_o1": int(fz["n_val_rows"]),
                "n_val_rows_r1": int(r1["n_val_rows"]),
                "step0_retreat": bool(int(fz["selected_step"]) == 0
                                      and int(r1["selected_step"]) > 0),
                "probe_code_hash": fz["probe_code_hash"],
            })
    return rows


def cell_medians(rows: list[dict]) -> list[dict]:
    out = []
    for m, h in PANEL:
        cell = U.cell_key(m, h)
        sub = [r for r in rows if r["cell"] == cell]
        if len(sub) != len(SEEDS):
            raise RuntimeError(f"{cell}: {len(sub)} paired rows, expected {len(SEEDS)}")
        gains = [r["relative_val_gain_pct"] for r in sub]
        out.append({
            "cell": cell, "market": m, "host": h, "source": source_of(m, h),
            "n_seeds": len(sub),
            "mae_o1_median": float(st.median([r["mae_o1"] for r in sub])),
            "mae_r1_median": float(st.median([r["mae_r1"] for r in sub])),
            "cell_median_relative_gain_pct": float(st.median(gains)),
            "per_seed_gains_pct": [round(g, 6) for g in gains],
            "seeds_improving": sum(1 for g in gains if g > 0),
            "selected_correction_ratio_median": float(
                st.median([r["correction_ratio_o1"] for r in sub])),
            "selected_step_median": float(st.median([r["selected_step_o1"] for r in sub])),
        })
    return out


def val_optimum_position(market: str, host: str, seed: int) -> dict:
    curve = load_json(run_dir(market, host, seed) / "training_curve.json")
    post = [e for e in curve if int(e["step"]) > SWITCH_STEP]
    best_post = min(post, key=lambda e: float(e["val_mae_ema"])) if post else None
    best_all = min(curve, key=lambda e: float(e["val_mae_ema"]))
    step0 = next(e for e in curve if int(e["step"]) == 0)
    at_switch = next(e for e in curve if int(e["step"]) == SWITCH_STEP)
    post_selected = [e for e in post if e.get("selected")]
    return {
        "best_post_switch_step": int(best_post["step"]) if best_post else None,
        "best_post_switch_val_mae_ema": float(best_post["val_mae_ema"]) if best_post else None,
        "argmin_over_all_steps": int(best_all["step"]),
        "step0_val_mae_ema": float(step0["val_mae_ema"]),
        "at_switch_val_mae_ema": float(at_switch["val_mae_ema"]),
        "post_switch_checks": len(post),
        "selected_after_switch": bool(post_selected),
        "post_switch_better_than_step0": bool(
            best_post is not None and float(best_post["val_mae_ema"]) < float(step0["val_mae_ema"])),
        "post_switch_better_than_at_switch": bool(
            best_post is not None
            and float(best_post["val_mae_ema"]) < float(at_switch["val_mae_ema"])),
        "post_switch_checks_setting_new_best": sum(1 for e in post if e.get("selected")),
        "fraction_of_post_switch_checks_setting_new_best": (
            sum(1 for e in post if e.get("selected")) / len(post) if post else 0.0),
    }


# --------------------------------------------------------------------------- mechanism
def _components_at(curve: list[dict], step: int) -> dict:
    e = next(x for x in curve if int(x["step"]) == step)
    return {k: float(e[k]) for k in COMPONENTS + ("aux_mean", "val_mae_ema")}


def mechanism_row(market: str, host: str, seed: int) -> dict:
    """Post-switch direction of every coordinate auxiliary term.

    Reported for interpretation only; it never enters selection or the gate.
    """
    curve = load_json(run_dir(market, host, seed) / "training_curve.json")
    r1 = load_json(reference_dir(market, host, seed) / "freeze.json")
    o1 = load_json(run_dir(market, host, seed) / "freeze.json")
    at400 = _components_at(curve, SWITCH_STEP)
    final = curve[-1]
    sel_step = int(o1["selected_step"])
    at_sel = _components_at(curve, sel_step) if sel_step > SWITCH_STEP else None

    def _delta(a: dict, b: dict) -> dict:
        return {k: float(b[k] - a[k]) for k in COMPONENTS + ("aux_mean", "val_mae_ema")}

    row = {
        "cell": U.cell_key(market, host), "market": market, "host": host, "seed": seed,
        "source": source_of(market, host),
        "o1_selected_step": sel_step,
        "r1_selected_step": int(r1["selected_step"]),
        "o1_selected_correction_ratio": float(
            o1["selected_metrics"]["val_correction_ratio"]),
        "r1_selected_correction_ratio": float(
            r1["selected_metrics"]["val_correction_ratio"]),
        "selected_is_post_switch": bool(sel_step > SWITCH_STEP),
        "final_check_step": int(final["step"]),
    }
    for name, comp in (("at_400", at400), ("at_final_check", _components_at(curve, int(final["step"])))):
        row.update({f"{name}_{k}": v for k, v in comp.items()})
    if at_sel is not None:
        row.update({f"at_selected_{k}": v for k, v in at_sel.items()})
        row.update({f"delta_selected_minus_400_{k}": v
                    for k, v in _delta(at400, at_sel).items()})
    else:
        for k in COMPONENTS + ("aux_mean", "val_mae_ema"):
            row[f"at_selected_{k}"] = None
            row[f"delta_selected_minus_400_{k}"] = None
    dfin = _delta(at400, _components_at(curve, int(final["step"])))
    row.update({f"delta_final_minus_400_{k}": v for k, v in dfin.items()})

    # The two cross-over patterns this stage exists to detect, at both horizons.
    horizons = {"final": (dfin["aux_mean"], dfin["val_mae_ema"])}
    if at_sel is not None:
        horizons["selected"] = (row["delta_selected_minus_400_aux_mean"],
                                row["delta_selected_minus_400_val_mae_ema"])
    # The "selected" horizon exists only where the selection is post-switch; the
    # columns are still emitted, empty, so the CSV schema is stable across runs.
    for tag, (aux_d, val_d) in horizons.items():
        row[f"aux_better_val_worse_at_{tag}"] = bool(aux_d < 0 < val_d)
        row[f"aux_worse_val_better_at_{tag}"] = bool(aux_d > 0 > val_d)
    for tag in ("final", "selected"):
        if f"aux_better_val_worse_at_{tag}" not in row:
            row[f"aux_better_val_worse_at_{tag}"] = None
            row[f"aux_worse_val_better_at_{tag}"] = None
    for k in COMPONENTS:
        row[f"delta_final_minus_400_{k}_direction"] = (
            "down" if dfin[k] < 0 else ("up" if dfin[k] > 0 else "flat"))
        d = row[f"delta_selected_minus_400_{k}"]
        row[f"delta_selected_minus_400_{k}_direction"] = (
            None if d is None else ("down" if d < 0 else ("up" if d > 0 else "flat")))
    row["aux_mean_direction_final"] = (
        "down" if dfin["aux_mean"] < 0 else ("up" if dfin["aux_mean"] > 0 else "flat"))
    row["val_mae_direction_final"] = (
        "down" if dfin["val_mae_ema"] < 0 else ("up" if dfin["val_mae_ema"] > 0 else "flat"))
    return row


def mechanism_rows() -> list[dict]:
    return [mechanism_row(m, h, s) for m, h in PANEL for s in SEEDS]


# --------------------------------------------------------------------------- compliance
def compliance() -> dict:
    """The audited g5 facts for all 24 runs, read off the freeze records and curves."""
    access_bad, nonfinite, switch_bad, schedule_bad = {}, {}, {}, {}
    support_bad, code_bad = {}, {}
    schedules, code_hashes, support_hashes = set(), {}, {}
    for m, h in PANEL:
        for s in SEEDS:
            key = f"{U.cell_key(m, h)}__seed{s}"
            rec = load_json(run_dir(m, h, s) / "freeze.json")
            ref = load_json(reference_dir(m, h, s) / "freeze.json")
            st_ = rec.get("access_state", {})
            if (int(rec.get("test_target_read_count", 0)) != 0
                    or int(rec.get("test_rows_materialised", 0)) != 0
                    or st_.get("distinct_paths_blocked")
                    or int(st_.get("test_rows_returned", 0)) != 0):
                access_bad[key] = st_
            if not all(bool(v) for v in np_isfinite_checks(rec)):
                nonfinite[key] = "non-finite metric"
            curve = load_json(run_dir(m, h, s) / "training_curve.json")
            marks = [e for e in curve if e["objective_switch_here"]]
            if len(marks) != 1 or int(marks[0]["step"]) != SWITCH_STEP:
                switch_bad[key] = f"switch markers at {[e['step'] for e in marks]}"
            for e in curve:
                if not e["is_optimization_step"]:
                    continue
                expect_full = int(e["step"]) <= SWITCH_STEP
                if bool(e["used_full_loss_at_this_step"]) != expect_full:
                    switch_bad[key] = f"step {e['step']} used_full={e['used_full_loss_at_this_step']}"
                    break
            schedules.add(json.dumps(schedule_of(rec), sort_keys=True))
            code_hashes[key] = rec["probe_code_hash"]
            support_hashes[key] = rec["history_support_hash"]
            for field in ("n_train_rows", "n_val_rows", "train_day_ids_hash",
                          "val_day_ids_hash", "history_support_hash",
                          "steps_per_epoch_old_semantics"):
                if rec[field] != ref[field]:
                    support_bad.setdefault(key, {})[field] = [rec[field], ref[field]]
            if int(ref.get("test_target_read_count", 0)) != 0:
                support_bad.setdefault(key, {})["reference_read_test"] = True
    integrity = {"n_distinct_schedules": len(schedules),
                 "compared_keys": list(PC.REGISTERED_SCHEDULE_KEYS),
                 "stopped_at_step_excluded": "an outcome, not a setting"}
    if len(schedules) == 1:
        sched = json.loads(next(iter(schedules)))
        for k in PC.REGISTERED_SCHEDULE_KEYS:
            if sched.get(k) != PC.registered_recipe()[k]:
                schedule_bad[k] = f"{sched.get(k)!r} != registered {PC.registered_recipe()[k]!r}"
    else:
        schedule_bad["n_distinct_schedules"] = len(schedules)
    distinct_codes = sorted(set(code_hashes.values()))
    if len(distinct_codes) != 1:
        code_bad["n_distinct_probe_code_hashes"] = len(distinct_codes)
        by_hash: dict = {}
        for k, v in code_hashes.items():
            by_hash.setdefault(v, []).append(k)
        code_bad["runs_by_hash"] = by_hash
        bad = set(max(by_hash.values(), key=len)) if by_hash else set()
        code_bad["minority_runs"] = sorted(set(code_hashes) - bad)
    return {
        "access_clean": not access_bad, "access_violations": access_bad,
        "all_finite": not nonfinite, "non_finite": nonfinite,
        "switch_exactly_after_400": not switch_bad, "switch_violations": switch_bad,
        "schedule_unchanged": not schedule_bad, "schedule_violations": schedule_bad,
        "schedule_integrity": integrity,
        "support_matches_reference": not support_bad, "support_violations": support_bad,
        "one_code_state": len(distinct_codes) == 1, "code_mutation": code_bad,
        "probe_code_hash": distinct_codes[0] if len(distinct_codes) == 1 else None,
        "n_runs_checked": len(code_hashes),
        "distinct_support_hashes": len(set(support_hashes.values())),
    }


def np_isfinite_checks(rec: dict) -> list[bool]:
    """Finiteness of every reported metric, without importing numpy here."""
    vals = [rec["selected_val_mae_ema"],
            rec["selected_metrics"]["val_host_mae"],
            rec["selected_metrics"]["val_mae_raw"],
            rec["selected_metrics"]["val_correction_ratio"],
            rec["selected_metrics"]["L_rec"], rec["selected_metrics"]["L_b"],
            rec["selected_metrics"]["L_B"], rec["selected_metrics"]["L_S"],
            rec["scales"]["s_r"], rec["scales"]["s_b"], rec["scales"]["s_B"]]
    out = []
    for v in vals:
        f = float(v)
        out.append(f == f and f not in (float("inf"), float("-inf")))
    return out


def schedule_of(record: dict) -> dict:
    return PC.registered_schedule_of(record["optimization"])


# --------------------------------------------------------------------------- gate
def gate(rows: list[dict], cells: list[dict], comp: dict) -> dict:
    """PROTOCOL.md section 7, recomputed from the raw per-seed numbers."""
    gains = [c["cell_median_relative_gain_pct"] for c in cells]
    panel_median = float(st.median(gains))
    step0_retreats = sum(1 for r in rows if r["step0_retreat"])
    ratio_median = float(st.median([r["correction_ratio_o1"] for r in rows]))
    g5 = bool(
        comp["access_clean"] and comp["all_finite"]
        and comp["switch_exactly_after_400"] and comp["schedule_unchanged"]
        and comp["support_matches_reference"] and comp["one_code_state"]
    )
    c4_ratio = ratio_median >= GATE_NON_DEGENERACY_RATIO_MIN
    c4_prop = step0_retreats <= GATE_STEP0_RETREAT_MAX_PROPORTIONAL
    c4_abs = step0_retreats <= GATE_STEP0_RETREAT_MAX_ABSOLUTE
    checks = {
        "g1_at_least_6_of_8_cell_medians_improve": sum(1 for g in gains if g > 0) >= GATE_MIN_CELLS_IMPROVING,
        "g2_panel_median_relative_improvement_ge_0.5pct": panel_median >= GATE_PANEL_MEDIAN_MIN_PCT,
        "g3_worst_cell_degradation_le_0.5pct": min(gains) >= GATE_WORST_CELL_MIN_PCT,
        "g4_not_a_step0_or_near_zero_retreat": bool(c4_ratio and c4_prop and c4_abs),
        "g5_no_leakage_anomaly_support_or_code_violation": g5,
    }
    detail = {
        "g5_compliance": comp,
        "n_cells": len(cells), "n_seeds": len(rows),
        "cells_improving": sum(1 for g in gains if g > 0),
        "cells_improving_needed": GATE_MIN_CELLS_IMPROVING,
        "panel_median_relative_gain_pct": panel_median,
        "panel_median_over_24_seed_gains_pct": float(
            st.median([r["relative_val_gain_pct"] for r in rows])),
        "worst_cell_gain_pct": float(min(gains)),
        "best_cell_gain_pct": float(max(gains)),
        "seeds_improving": sum(1 for r in rows if r["relative_val_gain_pct"] > 0),
        "step0_retreat_seeds": step0_retreats,
        "step0_retreat_rule_proportional": f"<={GATE_STEP0_RETREAT_MAX_PROPORTIONAL} of 24",
        "step0_retreat_rule_proportional_pass": bool(c4_prop),
        "step0_retreat_rule_absolute_reading": f"<={GATE_STEP0_RETREAT_MAX_ABSOLUTE} of 24",
        "step0_retreat_rule_absolute_pass": bool(c4_abs),
        "step0_retreat_rule_readings_agree": bool(c4_prop == c4_abs),
        "selected_correction_ratio_median": ratio_median,
        "non_degeneracy_threshold": GATE_NON_DEGENERACY_RATIO_MIN,
        "non_degeneracy_ratio_pass": bool(c4_ratio),
        "new_cell_subpanel_median_gain_pct": float(st.median(
            [c["cell_median_relative_gain_pct"] for c in cells if c["source"] == "new_fit"])),
        "reused_cell_subpanel_median_gain_pct": float(st.median(
            [c["cell_median_relative_gain_pct"] for c in cells if c["source"] == "reused_o1"])),
    }
    return {"checks": checks, "detail": detail}


# --------------------------------------------------------------------------- audit
def write_access_audit(path: Path, extra: dict | None = None) -> dict:
    payload = {
        "schema": "hch_v44_o1_full8_access_audit.v1",
        "protocol_id": PROTOCOL_ID,
        "quarantine": "V2 TEST targets/predictions/metrics forbidden to every decision here",
        "forbidden_files_enumerated": [
            str(p.relative_to(REPO)).replace("\\", "/") for p in RC.FORBIDDEN_FILES],
        "forbidden_name_patterns": sorted(RC.FORBIDDEN_NAMES),
        "guard_installed": bool(RC._GUARD_ACTIVE),
        "access_state": json.loads(json.dumps(RC.ACCESS_STATE)),
        "test_target_read_count": 0,
        "test_rows_materialised": 0,
        "note": ("Inherited unchanged from the O1 and recovery stages: pretest_joint_arrays drops "
                 "the sealed segment before any frame exists, pretest_role_frame refuses "
                 "role='TEST', and the forbidden-path guard refuses the enumerated TEST tables "
                 "at the I/O boundary."),
    }
    if extra:
        payload.update(extra)
    U.json_dump(path, payload)
    return payload
