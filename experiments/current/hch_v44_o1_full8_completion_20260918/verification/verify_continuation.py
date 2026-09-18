"""Independent verification of the HCH v4.4 O1 full-8 completion.

Deliberately does **not** import ``continuation_runner``, ``continuation_common``,
``probe_train``, ``probe_common`` or ``recovery_common``.  Every causal rule, hash,
identity, gate number and provenance claim below is recomputed here from the frozen
artifacts, so a shared bug cannot pass unnoticed.

Checks (PROTOCOL.md sections 2-9):

 1. ``registered_coordinates``      exactly 8 cells x 3 seeds, and exactly the 12
                                    registered new fits, with no extra runs
 2. ``new_fits_identity``           the 12 new fits carry the registered protocol,
                                    recipe, support surface and one code state
 3. ``reused_o1_identity``          the 12 reused O1 runs are byte-identical to what
                                    the O1 stage recorded, and were not rerun
 4. ``r1_reference_identity``       all 24 R1 references match the recorded
                                    provenance and the published per-seed table
 5. ``val_surface_rebuilt``         the four new cells' eligible TRAIN/VAL rows are
                                    rebuilt from the raw cache and the plan
 6. ``pre_switch_identity``         every O1 run reproduces its R1 reference bit for
                                    bit up to and including step 400
 7. ``schedule_identical``          one registered schedule across all 24 runs
 8. ``only_loss_schedule_changed``  the inherited trainer still differs from the
                                    reference loop by exactly the registered lines
 9. ``objective_switch_after_400``  the switch lands exactly after update 400
10. ``selection_rule``              the running-best rule with its 1e-7 tolerance
11. ``core_unchanged``              src/core + src/backbones digest unchanged
12. ``test_quarantine``             TEST reads = 0 and no forbidden path opened
13. ``gate_arithmetic``             the 8-cell gate recomputed from the raw rows
14. ``one_code_state``              one probe code hash across all 24 runs
15. ``no_new_trainer``              this stage imported the trainer, it did not
                                    write its own
16. ``terminal_token``              the reported token is one of the two registered
"""
from __future__ import annotations

import datetime as dt
import difflib
import hashlib
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]
IMPL = STAGE / "implementation"
EVID = REPO / "experiments/evidence/hch_v44_o1_full8_completion_20260918"

O1_STAGE = REPO / "experiments/current/hch_v44_objective_alignment_probe_20260917"
O1_IMPL = O1_STAGE / "implementation"
O1_EVID = REPO / "experiments/evidence/hch_v44_objective_alignment_probe_20260917"

RECOVERY_IMPL = REPO / "experiments/current/hch_v44_optimization_recovery_20260917/implementation"
RECOVERY_EVID = REPO / "experiments/evidence/hch_v44_optimization_recovery_20260917"
PRIOR_EVID = REPO / "experiments/evidence/hch_v44_china5_fullpanel_main_eval_20260917"
V2_EVID = REPO / "experiments/evidence/hch_china5_common_benchmark_701020_full_v2_20260913"

PANEL = [("GANSU_DA", "PatchTST"), ("GANSU_DA", "LSTM"),
         ("SHANDONG_DA", "PatchTST"), ("SHANDONG_DA", "iTransformer"),
         ("SHAANXI_DA", "TimeMixer"), ("SHAANXI_DA", "PatchTST"),
         ("NINGXIA_DA", "iTransformer"), ("QINGHAI_DA", "TimeMixer")]
NEW_CELLS = [("GANSU_DA", "LSTM"), ("SHANDONG_DA", "PatchTST"),
             ("SHAANXI_DA", "PatchTST"), ("QINGHAI_DA", "TimeMixer")]
REUSED_CELLS = [c for c in PANEL if c not in NEW_CELLS]
SEEDS = [7, 17, 37]
SWITCH_STEP = 400
SELECTION_TOL = 1e-7
HORIZON = 24
WINDOW = 7

GATE_MIN_CELLS = 6
GATE_PANEL_MEDIAN_MIN = 0.5
GATE_WORST_CELL_MIN = -0.5
GATE_RATIO_MIN = 0.05
GATE_RETREAT_MAX_PROPORTIONAL = 4
GATE_RETREAT_MAX_ABSOLUTE = 2

REGISTERED_SCHEDULE = {
    "optimizer": "AdamW", "lr": 1e-3, "batch_size": 32, "weight_decay": 1e-4,
    "grad_clip": 1.0, "ema_decay": 0.995, "eval_weights": "ema",
    "max_steps": 2000, "val_every": 50, "no_stop_before": 800, "patience_checks": 8,
    "readout_init": None,
}
#: The historical TEST-bearing tables named by the standing quarantine.  This
#: stage's own RESULTS.md is a *registered output* and is deliberately not on the
#: list -- the forbidden RESULTS.md belongs to the prior main-eval evidence root.
FORBIDDEN_NAMES = ("CELL_MEDIAN_RESULTS.csv", "PER_SEED_RESULTS.csv", "BASELINE_COMPARISON.csv",
                   "HCH_V44_MAIN_20260917.csv", "RESULTS_LONG.csv",
                   "BASELINE_COMPLETION_20260917.csv", "result.json", "test_predictions.npz")

TOKENS = ("HCH_V44_O1_FULL8_COMPLETION_COMPLETE_FOR_ADJUDICATION",
          "HCH_V44_O1_FULL8_COMPLETION_BLOCKED_")


# --------------------------------------------------------------------------- utils
def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_key(market: str, host: str) -> str:
    return f"{market}__{host}"


def is_new(market: str, host: str) -> bool:
    return (market, host) in NEW_CELLS


def run_dir(market: str, host: str, seed: int) -> Path:
    root = EVID if is_new(market, host) else O1_EVID
    return root / "o1_runs" / cell_key(market, host) / f"seed{seed}"


def reference_dir(market: str, host: str, seed: int) -> Path:
    return RECOVERY_EVID / "r1_runs" / cell_key(market, host) / f"seed{seed}"


def freeze(market: str, host: str, seed: int) -> dict:
    return load_json(run_dir(market, host, seed) / "freeze.json")


def curve(market: str, host: str, seed: int) -> list:
    return load_json(run_dir(market, host, seed) / "training_curve.json")


def ref_freeze(market: str, host: str, seed: int) -> dict:
    return load_json(reference_dir(market, host, seed) / "freeze.json")


def source_tree_digest() -> dict:
    out = {}
    for name, pattern in (("core_tree", "src/core"), ("backbones_tree", "src/backbones")):
        paths = sorted(p for p in (REPO / pattern).rglob("*")
                       if p.is_file() and p.suffix in (".py", ".md"))
        out[name] = sha256_bytes(
            "\n".join(f"{p.relative_to(REPO).as_posix()}|{sha256(p)}" for p in paths).encode())
        out[name + "_n_files"] = len(paths)
    return out


# ------------------------------------------------------------------- causal rules
def split_plan(market: str):
    v2_impl = REPO / "experiments/current/hch_china5_common_benchmark_701020_full_v2/implementation"
    if str(v2_impl) not in sys.path:
        sys.path.insert(0, str(v2_impl))
    import cb_contracts as C  # noqa: PLC0415

    return C.split_plan(market)


def to_date(value):
    if isinstance(value, dt.date):
        return value
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]").astype(object)
    return dt.date.fromisoformat(str(value)[:10])


def origin_of(day):
    return dt.datetime.combine(day, dt.time(1, 0)) - dt.timedelta(hours=15, minutes=30)


def reveal_of(day):
    """Moment ``day``'s full 24h target is known: hour-ending 24:00, i.e. next day 01:00."""
    return dt.datetime.combine(day + dt.timedelta(days=1), dt.time(1, 0))


def history_window(target_day, available: set):
    """The 7 most recent admissible available days, or None if there are fewer.

    The registered predicate is ``day < target`` and ``reveal_time < origin``, so it
    is **gap-tolerant**: a missing intermediate day pulls the window further back
    rather than voiding it.  The window is not necessarily a contiguous block.
    """
    target = to_date(target_day)
    origin = origin_of(target)
    chosen = []
    for day in sorted((d for d in available if d < target), reverse=True):
        if not reveal_of(day) < origin:
            continue
        chosen.append(day)
        if len(chosen) == WINDOW:
            break
    if len(chosen) != WINDOW:
        return None
    chosen.reverse()
    return chosen


def pretest_joint(market: str, host: str) -> dict:
    """The joint Host cache with TEST rows dropped before anything is returned."""
    p = V2_EVID / "02_hosts" / market / host / "host_predictions_joint.npz"
    with np.load(p, allow_pickle=False) as z:
        segment = z["segment"].astype(str)
        keep = (segment == "TRAIN") | (segment == "VAL")
        return {
            "timestamp": z["timestamp"][keep],
            "y_true": z["y_true"].reshape(len(z["y_true"]), HORIZON)[keep],
            "host_pred": z["host_pred"].reshape(len(z["host_pred"]), HORIZON)[keep],
            "segment": segment[keep],
            "n_test_rows_dropped": int((~keep).sum()),
            "cache_days_total": int(len(segment)),
        }


def rebuild_rows(market: str, host: str) -> dict:
    """Independently rebuild the eligible TRAIN/VAL row sets from the raw cache."""
    plan = split_plan(market)
    z = pretest_joint(market, host)
    test_ids = {to_date(d) for d in plan.ids("TEST")}
    out = {}
    support = load_json(PRIOR_EVID / "shadow_oof" / f"{cell_key(market, host)}.json")
    # The support file carries both ``oof_days`` (an int *count*) and ``oof_day_ids``
    # (the day list).  Only the latter is iterable.
    oof_days = {to_date(d) for d in support["oof_day_ids"]}
    if len(oof_days) != int(support["oof_days"]):
        raise RuntimeError(f"{cell_key(market, host)}: oof_day_ids/oof_days disagree")
    for role in ("TRAIN", "VAL"):
        ids = [to_date(d) for d in plan.ids(role)]
        mask = z["segment"] == role
        cache_days = [to_date(t) for t in z["timestamp"][mask]]
        if cache_days != ids:
            raise RuntimeError(f"{cell_key(market, host)} {role}: cache/plan day mismatch")
        y = z["y_true"][mask]
        out[role] = {"n_registered": len(ids), "n_scorable": int(np.isfinite(y).all(axis=1).sum()),
                     "days": ids, "finite": np.isfinite(y).all(axis=1)}
    # Eligible rows.  TRAIN and VAL share one starting history index: the revealed
    # residuals of the causal shadow-OOF support.  TRAIN days are NOT added to it.
    index_days = set(oof_days)
    train_eligible, train_excluded = [], {"no_history": 0, "non_finite_current_day": 0}
    for d, f in zip(out["TRAIN"]["days"], out["TRAIN"]["finite"]):
        if not f:
            train_excluded["non_finite_current_day"] += 1
            continue
        if history_window(d, index_days) is None:
            train_excluded["no_history"] += 1
            continue
        train_eligible.append(d)
    out["TRAIN"]["eligible_days"] = train_eligible
    out["TRAIN"]["excluded"] = train_excluded
    out["TRAIN"]["day_ids_hash"] = sha256_bytes(
        json.dumps([d.isoformat() for d in train_eligible]).encode())
    val_days, val_index = [], set(index_days)
    for d, f in zip(out["VAL"]["days"], out["VAL"]["finite"]):
        if not f:
            continue
        if history_window(d, val_index) is None:
            continue
        val_days.append(d)
        val_index.add(d)  # prequential reveal
    out["VAL"]["eligible_days"] = val_days
    out["VAL"]["day_ids_hash"] = sha256_bytes(
        json.dumps([d.isoformat() for d in val_days]).encode())
    out["test_days_in_train_or_val"] = sum(
        1 for d in train_eligible + val_days if d in test_ids)
    out["n_test_rows_dropped"] = z["n_test_rows_dropped"]
    out["cache_days_total"] = z["cache_days_total"]
    out["support_hash_recomputed"] = None
    return out


def replay_selection(curve_rows: list, field: str = "val_mae_ema") -> tuple:
    """The registered running-best rule, re-implemented."""
    best_v, best_step = float("inf"), -1
    for e in curve_rows:
        v = float(e[field])
        if v < best_v - SELECTION_TOL:
            best_v, best_step = v, int(e["step"])
    return best_step, best_v


# ---------------------------------------------------------------------- loop diff
def loop_region(text: str) -> list:
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == "while step < max_steps:")
    except StopIteration:
        raise SystemExit("verifier: could not locate the training loop")
    try:
        end = next(i for i, l in enumerate(lines)
                   if l.strip().startswith('if best["shadow"] is None'))
    except StopIteration:
        raise SystemExit("verifier: could not locate the loop end")
    return [l.rstrip() for l in lines[start:end]]


EXPECTED_REMOVED = {
    'loss = model.compute_loss(out, sub_target, sub_res, data["scales"])["total"]',
    "record_check(step, samples_seen, False, norms)",
}
EXPECTED_ADDED = {
    "loss, used_full = objective(model.compute_loss(out, sub_target, sub_res, "
    'data["scales"]), step + 1)',
    "record_check(step, samples_seen, False, norms, float(loss.detach()), used_full)",
}


def check_only_loss_schedule_changed() -> dict:
    ref = loop_region((RECOVERY_IMPL / "recovery_train.py").read_text(encoding="utf-8"))
    new = loop_region((O1_IMPL / "probe_train.py").read_text(encoding="utf-8"))
    diff = [l for l in difflib.unified_diff(ref, new, lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    removed = {l[1:].strip() for l in diff if l.startswith("-")}
    added = {l[1:].strip() for l in diff if l.startswith("+")}
    return {"n_removed_lines": len(removed), "n_added_lines": len(added),
            "removed": sorted(removed), "added": sorted(added),
            "removed_is_exactly_expected": removed == EXPECTED_REMOVED,
            "added_is_exactly_expected": added == EXPECTED_ADDED}


def check_no_direct_readers() -> list:
    """No module of this stage may call the prior stage's TEST-materialising readers."""
    pattern = re.compile(
        r"\bU\.role_frame\s*\(|\bU\.joint_arrays\s*\(|\bU\.SEAL_STATE\b|\bscore_test\s*\("
        r"|(?<!pretest_)\brole_frame\s*\("
    )
    hits = []
    for p in sorted(IMPL.rglob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{p.name}:{i}: {line.strip()}")
    return hits


# --------------------------------------------------------------------------- main
def main() -> dict:
    checks, notes, issues = {}, {}, []
    payload = {"schema": "hch_v44_o1_full8_verification.v1"}

    gate = load_json(EVID / "GATE.json")
    config = load_json(EVID / "PROBE_CONFIG.json")
    reuse = load_json(EVID / "REUSE_PROVENANCE.json")
    audit = load_json(EVID / "ACCESS_AUDIT.json")
    payload["gate_verdict"] = gate["verdict"]

    # --- 1. registered coordinates, split across two evidence roots
    expected_new = {f"{cell_key(m, h)}__seed{s}" for m, h in NEW_CELLS for s in SEEDS}
    expected_reused = {f"{cell_key(m, h)}__seed{s}" for m, h in REUSED_CELLS for s in SEEDS}

    def scan(root: Path) -> set:
        found = set()
        run_root = root / "o1_runs"
        if not run_root.is_dir():
            return found
        for cell_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in cell_dir.iterdir() if p.is_dir()):
                found.add(f"{cell_dir.name}__{seed_dir.name}")
        return found

    found_new, found_reused = scan(EVID), scan(O1_EVID)
    no_extra = found_new <= expected_new and found_reused <= expected_reused
    coord_bad = []
    if found_new != expected_new:
        coord_bad.append(f"new-fit set mismatch: missing={sorted(expected_new - found_new)} "
                         f"extra={sorted(found_new - expected_new)}")
    if not expected_reused <= found_reused:
        coord_bad.append(f"reused O1 runs missing: {sorted(expected_reused - found_reused)}")
    if not no_extra:
        coord_bad.append("an unregistered run directory exists")
    all_files = all((run_dir(m, h, s) / f).is_file()
                    for m, h in PANEL for s in SEEDS
                    for f in ("freeze.json", "training_curve.json", "selected_ema.pt"))
    if not all_files:
        coord_bad.append("a registered run is missing freeze/curve/checkpoint")
    checks["registered_coordinates"] = not coord_bad
    notes["registered_coordinates"] = {
        "n_new_fits_expected": len(expected_new), "n_new_fits_found": len(found_new),
        "n_reused_o1_expected": len(expected_reused),
        "n_reused_o1_found_in_o1_root": len(found_reused),
        "n_panel_runs": len(PANEL) * len(SEEDS),
        "all_runs_have_freeze_curve_checkpoint": all_files, "violations": coord_bad,
    }

    # --- 2. the 12 new fits carry the registered identity
    new_bad, new_notes = [], {}
    reused_hashes = {freeze(m, h, s)["probe_code_hash"] for m, h in REUSED_CELLS for s in SEEDS}
    if len(reused_hashes) != 1:
        new_bad.append(f"the reused O1 runs do not share one code hash: {sorted(reused_hashes)}")
    reused_hash = sorted(reused_hashes)[0] if reused_hashes else None
    for m, h in NEW_CELLS:
        for s in SEEDS:
            key = f"{cell_key(m, h)}__seed{s}"
            r, ref = freeze(m, h, s), ref_freeze(m, h, s)
            if r.get("protocol_id") != "HCH_V44_OBJECTIVE_ALIGNMENT_PROBE_20260917":
                new_bad.append(f"{key}: protocol_id {r.get('protocol_id')!r}")
            if r.get("recipe_id") != "O1_AUX_WARMUP_400_THEN_MAE":
                new_bad.append(f"{key}: recipe_id {r.get('recipe_id')!r}")
            if r.get("probe_code_hash") != reused_hash:
                new_bad.append(f"{key}: probe_code_hash {r.get('probe_code_hash')} != the reused "
                               f"runs' {reused_hash}")
            if r.get("is_registered_fit") is not True:
                new_bad.append(f"{key}: is_registered_fit is not True")
            if int(r["objective_probe"]["switch_step"]) != SWITCH_STEP:
                new_bad.append(f"{key}: switch_step != 400")
            for f in ("n_train_rows", "n_val_rows", "train_day_ids_hash", "val_day_ids_hash",
                      "history_support_hash", "steps_per_epoch_old_semantics"):
                if r[f] != ref[f]:
                    new_bad.append(f"{key}: {f} {r[f]!r} != the paired R1 reference {ref[f]!r}")
            h_on_disk = sha256(RECOVERY_IMPL / "recovery_common.py")
            if r["recovery_common_sha256"] != h_on_disk:
                new_bad.append(f"{key}: recovery_common_sha256 is not the module on disk")
            if r["source_tree_digest"]["core_tree"] != ref["source_tree_digest"]["core_tree"]:
                new_bad.append(f"{key}: core_tree digest differs from the paired reference")
            new_notes[key] = {"selected_step": int(r["selected_step"]),
                              "selected_val_mae_ema": float(r["selected_val_mae_ema"]),
                              "stopped_at_step": r["optimization"]["stopped_at_step"],
                              "n_train_rows": int(r["n_train_rows"]),
                              "n_val_rows": int(r["n_val_rows"])}
    checks["new_fits_identity"] = not new_bad
    notes["new_fits_identity"] = {"n_new_fits": len(new_notes), "shared_code_hash": reused_hash,
                                 "violations": new_bad, "per_run": new_notes}

    # --- 3. the 12 reused O1 runs are byte-identical to the O1 stage's record
    o1_cfg = load_json(O1_EVID / "PROBE_CONFIG.json")["runs"]
    re_bad, re_notes = [], {}
    for m, h in REUSED_CELLS:
        for s in SEEDS:
            key = f"{cell_key(m, h)}__seed{s}"
            d = run_dir(m, h, s)
            got = {"freeze_sha256": sha256(d / "freeze.json"),
                   "curve_sha256": sha256(d / "training_curve.json"),
                   "checkpoint_sha256": sha256(d / "selected_ema.pt")}
            rec = o1_cfg.get(key)
            if rec is None:
                re_bad.append(f"{key}: absent from the O1 stage's PROBE_CONFIG.json")
            else:
                for k in ("freeze_sha256", "curve_sha256"):
                    if rec[k] != got[k]:
                        re_bad.append(f"{key}: {k} no longer matches the O1 stage")
                if float(rec["selected_val_mae_ema"]) != float(freeze(m, h, s)["selected_val_mae_ema"]):
                    re_bad.append(f"{key}: selected_val_mae_ema no longer matches the O1 stage")
                if rec["selected_ema_parameter_hash"] != freeze(m, h, s)["selected_ema_parameter_hash"]:
                    re_bad.append(f"{key}: checkpoint parameter hash no longer matches the O1 stage")
            fz = freeze(m, h, s)
            if fz["selected_ema_checkpoint_sha256"] != got["checkpoint_sha256"]:
                re_bad.append(f"{key}: selected_ema.pt does not hash to its own record's sha256")
            re_notes[key] = {**got, "selected_step": int(fz["selected_step"]),
                             "matches_o1_stage_record": rec is not None}
    if config["runs"] and any(config["runs"][k]["probe_code_hash"] != reused_hash
                              for k in config["runs"] if k in expected_reused):
        re_bad.append("PROBE_CONFIG.json disagrees with the on-disk probe_code_hash")
    checks["reused_o1_identity"] = not re_bad
    notes["reused_o1_identity"] = {"n_reused": len(re_notes), "violations": re_bad,
                                   "per_run": re_notes,
                                   "source_evidence_root": str(O1_EVID.relative_to(REPO)).replace("\\", "/")}

    # --- 4. the 24 R1 references match every recorded provenance
    o1_ref = load_json(O1_EVID / "REFERENCE_PROVENANCE.json")["per_coordinate"]
    import csv as _csv
    with (RECOVERY_EVID / "R1_PANEL_PER_SEED.csv").open(encoding="utf-8", newline="") as fh:
        published = {f"{r['cell']}__seed{r['seed']}": r for r in _csv.DictReader(fh)}
    ref_bad, ref_notes = [], {}
    for m, h in PANEL:
        for s in SEEDS:
            key = f"{cell_key(m, h)}__seed{s}"
            d = reference_dir(m, h, s)
            fz = ref_freeze(m, h, s)
            got = {"freeze_sha256": sha256(d / "freeze.json"),
                   "curve_sha256": sha256(d / "training_curve.json"),
                   "checkpoint_sha256": sha256(d / "selected_ema.pt")}
            if fz["recipe_id"] != "R1_STEP_BUDGET_2000":
                ref_bad.append(f"{key}: recipe_id {fz['recipe_id']!r}")
            if fz["selected_ema_checkpoint_sha256"] != got["checkpoint_sha256"]:
                ref_bad.append(f"{key}: selected_ema.pt does not hash to its own record's sha256")
            rec = o1_ref.get(key)
            if rec is not None:
                for k in ("freeze_sha256", "curve_sha256", "checkpoint_sha256"):
                    if rec[k] != got[k]:
                        ref_bad.append(f"{key}: {k} differs from the O1-stage reference record")
            pub = published.get(key)
            if pub is None:
                ref_bad.append(f"{key}: absent from R1_PANEL_PER_SEED.csv")
            else:
                if float(pub["val_mae_ema"]) != float(fz["selected_val_mae_ema"]):
                    ref_bad.append(f"{key}: published val_mae_ema != freeze")
                if int(pub["selected_step"]) != int(fz["selected_step"]):
                    ref_bad.append(f"{key}: published selected_step != freeze")
                if pub["checkpoint_parameter_hash"] != fz["selected_ema_parameter_hash"]:
                    ref_bad.append(f"{key}: published checkpoint parameter hash != freeze")
            if int(fz.get("test_target_read_count", 0)) != 0:
                ref_bad.append(f"{key}: the reference read TEST")
            ref_notes[key] = {**got, "pinned_by_o1_stage_record": rec is not None,
                              "pinned_by_published_table": pub is not None,
                              "selected_step": int(fz["selected_step"])}
    checks["r1_reference_identity"] = not ref_bad
    notes["r1_reference_identity"] = {
        "n_references": len(ref_notes), "violations": ref_bad,
        "n_pinned_by_o1_stage_record": sum(1 for v in ref_notes.values()
                                           if v["pinned_by_o1_stage_record"]),
        "n_pinned_by_published_table": sum(1 for v in ref_notes.values()
                                           if v["pinned_by_published_table"]),
        "note": "each reference is pinned both by hash and by its own recorded checkpoint sha256",
    }

    # --- 5. the four new cells' measured partition, rebuilt from the raw cache
    rebuilt_bad, rebuilt = [], {}
    for m, h in NEW_CELLS:
        r = rebuild_rows(m, h)
        rebuilt[cell_key(m, h)] = {
            "n_train_eligible": len(r["TRAIN"]["eligible_days"]),
            "n_val_eligible": len(r["VAL"]["eligible_days"]),
            "n_train_registered": r["TRAIN"]["n_registered"],
            "n_val_registered": r["VAL"]["n_registered"],
            "test_days_in_train_or_val": r["test_days_in_train_or_val"],
            "test_rows_dropped_from_joint_cache": r["n_test_rows_dropped"],
            "train_day_ids_hash": r["TRAIN"]["day_ids_hash"],
            "val_day_ids_hash": r["VAL"]["day_ids_hash"],
        }
        for s in SEEDS:
            fz = freeze(m, h, s)
            for key, want in (("n_train_rows", len(r["TRAIN"]["eligible_days"])),
                              ("n_val_rows", len(r["VAL"]["eligible_days"])),
                              ("train_day_ids_hash", r["TRAIN"]["day_ids_hash"]),
                              ("val_day_ids_hash", r["VAL"]["day_ids_hash"])):
                if fz[key] != want:
                    rebuilt_bad.append(f"{cell_key(m, h)}__seed{s}: {key} {fz[key]!r} != "
                                       f"rebuilt {want!r}")
            if dict(fz["train_rows_excluded"]) != r["TRAIN"]["excluded"]:
                rebuilt_bad.append(
                    f"{cell_key(m, h)}__seed{s}: train_rows_excluded {fz['train_rows_excluded']} "
                    f"!= rebuilt {r['TRAIN']['excluded']}")
        if r["test_days_in_train_or_val"]:
            rebuilt_bad.append(f"{cell_key(m, h)}: TEST day inside TRAIN/VAL")
        if r["n_test_rows_dropped"] <= 0:
            rebuilt_bad.append(f"{cell_key(m, h)}: joint cache dropped no TEST rows")
    checks["val_surface_rebuilt"] = not rebuilt_bad
    notes["val_surface_rebuilt"] = {
        "rule": ("eligible = fully-finite 24h target AND a legal W=7 gap-tolerant window whose "
                 "days satisfy d < target and reveal_of(d) < origin_of(target), with history "
                 "taken from the shadow-OOF support days only; TRAIN days are never added and "
                 "VAL days are added prequentially.  Reimplemented here from the split plan and "
                 "the joint cache, independently of the runner."),
        "cells_rebuilt": [cell_key(m, h) for m, h in NEW_CELLS],
        "violations": rebuilt_bad, "per_cell": rebuilt,
    }

    # --- 6. pre-switch identity: every O1 run reproduces its R1 reference to step 400
    id_bad, identity = [], {}
    for m, h in PANEL:
        for s in SEEDS:
            key = f"{cell_key(m, h)}__seed{s}"
            o1 = {e["step"]: e for e in curve(m, h, s)}
            r1 = {e["step"]: e for e in load_json(reference_dir(m, h, s) / "training_curve.json")}
            pre = sorted(x for x in set(o1) & set(r1) if x <= SWITCH_STEP)
            if not pre or pre[0] != 0 or max(pre) != SWITCH_STEP:
                id_bad.append(f"{key}: pre-switch checks {pre[:3]}..{pre[-3:]}")
            worst = 0.0
            for step in pre:
                for f in ("val_mae_ema", "val_mae_raw", "train_mae_ema",
                          "diag_reconstruction_mae", "level_mae", "mass_mae", "shape_w1",
                          "train_correction_ratio"):
                    a, b = float(o1[step][f]), float(r1[step][f])
                    worst = max(worst, abs(a - b) / max(1.0, abs(b)))
                if o1[step]["grad_norms_preclip"] != r1[step]["grad_norms_preclip"]:
                    id_bad.append(f"{key} step{step}: grad norms differ from the reference")
            if worst > 0.0:
                id_bad.append(f"{key}: pre-switch relative deviation {worst}")
            identity[key] = {"shared_pre_switch_steps": len(pre), "max_relative_deviation": worst,
                             "source": "new_fit" if is_new(m, h) else "reused_o1"}
    checks["pre_switch_identity"] = not id_bad
    notes["pre_switch_identity"] = {
        "note": ("updates 1..400 use the reference's own loss and the same random stream, so "
                 "every pre-switch checkpoint must reproduce the R1 reference exactly; any "
                 "deviation would mean a second, unregistered change.  This is the strongest "
                 "available check that the 12 new fits ran the inherited trainer."),
        "n_coords_compared": len(identity), "violations": id_bad, "per_coord": identity,
    }

    # --- 7. one schedule across all 24 runs
    sched_bad, schedules = [], set()
    for m, h in PANEL:
        for s in SEEDS:
            key = f"{cell_key(m, h)}__seed{s}"
            r = freeze(m, h, s)
            sched = {k: r["optimization"].get(k) for k in REGISTERED_SCHEDULE}
            schedules.add(json.dumps(sched, sort_keys=True))
            if sched != REGISTERED_SCHEDULE:
                sched_bad.append(f"{key}: {sched}")
            if r["diag_subset"]["rule"] != ("np.linspace over registered TRAIN order, "
                                            "capped at 256"):
                sched_bad.append(f"{key}: diagnostic subset changed")
            if r["interleaver_audit"]["reweighting"] or r["interleaver_audit"]["oversampling"]:
                sched_bad.append(f"{key}: interleaver changed")
            if r["rescue_used"] is not False or r["optional_features_active"] is not False:
                sched_bad.append(f"{key}: a rescue or optional feature was active")
            if r["objective_probe"]["loss_full"] != "L_rec + (L_b + L_B + L_S)/3":
                sched_bad.append(f"{key}: warm-up loss is not the frozen full loss")
            if r["objective_probe"]["loss_after_switch"] != "L_rec":
                sched_bad.append(f"{key}: post-switch loss is not L_rec")
            if r["objective_probe"]["changed_vs_reference"] != ["loss schedule after update 400"]:
                sched_bad.append(f"{key}: changed_vs_reference {r['objective_probe']['changed_vs_reference']}")
    ref_scheds = {json.dumps({k: ref_freeze(m, h, s)["optimization"].get(k)
                              for k in REGISTERED_SCHEDULE}, sort_keys=True)
                  for m, h in PANEL for s in SEEDS}
    if ref_scheds != {json.dumps(REGISTERED_SCHEDULE, sort_keys=True)}:
        sched_bad.append("the reference schedule is not the registered schedule")
    extra_dirs = sorted(p.name for p in EVID.iterdir() if p.is_dir())
    checks["schedule_identical"] = not sched_bad and len(schedules) == 1
    notes["schedule_identical"] = {
        "n_distinct_schedules_over_24_runs": len(schedules), "violations": sched_bad,
        "registered": REGISTERED_SCHEDULE,
        "reference_schedule_matches": ref_scheds == {json.dumps(REGISTERED_SCHEDULE,
                                                                sort_keys=True)},
        "evidence_root_dirs": extra_dirs,
    }

    # --- 8. the inherited trainer still differs from the reference loop by exactly
    #        the registered lines (nothing was added to it by this stage)
    diff = check_only_loss_schedule_changed()
    stage_impl = "\n".join(p.read_text(encoding="utf-8") for p in sorted(IMPL.rglob("*.py")))
    checks["only_loss_schedule_changed"] = bool(
        diff["removed_is_exactly_expected"] and diff["added_is_exactly_expected"]
        and "train_o1" in stage_impl)
    notes["only_loss_schedule_changed"] = {
        **diff, "stage_calls_train_o1": "train_o1" in stage_impl,
        "o1_trainer_sha256": sha256(O1_IMPL / "probe_train.py"),
        "note": ("the 12 new fits import the O1 stage's train_o1; this check confirms the "
                 "imported trainer is still the registered one and was not edited"),
    }

    # --- 9. objective switch exactly after 400
    switch_bad = []
    for m, h in PANEL:
        for s in SEEDS:
            key = f"{cell_key(m, h)}__seed{s}"
            rows = curve(m, h, s)
            marks = [e["step"] for e in rows if e["objective_switch_here"]]
            if marks != [SWITCH_STEP]:
                switch_bad.append(f"{key}: switch markers {marks}")
            for e in rows:
                if not e["is_optimization_step"]:
                    if e["used_full_loss_at_this_step"] is not None:
                        switch_bad.append(f"{key}: step-0 check carries a loss schedule")
                    continue
                expect = int(e["step"]) <= SWITCH_STEP
                if bool(e["used_full_loss_at_this_step"]) != expect:
                    switch_bad.append(f"{key} step{e['step']}: used_full="
                                      f"{e['used_full_loss_at_this_step']}")
                if e["loss_schedule"] != ("L_rec_plus_aux" if expect else "L_rec_only"):
                    switch_bad.append(f"{key} step{e['step']}: {e['loss_schedule']}")
            if not rows or not rows[-1]["samples_seen"]:
                switch_bad.append(f"{key}: empty curve")
    checks["objective_switch_after_400"] = not switch_bad
    notes["objective_switch_after_400"] = {"violations": switch_bad[:10],
                                           "n_violations": len(switch_bad),
                                           "n_runs_checked": len(PANEL) * len(SEEDS)}

    # --- 10. selection rule
    sel_bad = []
    for m, h in PANEL:
        for s in SEEDS:
            key = f"{cell_key(m, h)}__seed{s}"
            rows, rec = curve(m, h, s), freeze(m, h, s)
            step, val = replay_selection(rows)
            if step != int(rec["selected_step"]):
                sel_bad.append(f"{key}: replay step {step} != {rec['selected_step']}")
            if abs(val - float(rec["selected_val_mae_ema"])) > 1e-9 * max(1.0, abs(val)):
                sel_bad.append(f"{key}: replay value {val} != {rec['selected_val_mae_ema']}")
            if min(float(e["val_mae_ema"]) for e in rows) < val - 1e-9:
                sel_bad.append(f"{key}: selection is not the curve minimum")
            if not any(e["selected"] and int(e["step"]) == step for e in rows):
                sel_bad.append(f"{key}: no selected curve entry at the chosen step")
    checks["selection_rule"] = not sel_bad
    notes["selection_rule"] = {"violations": sel_bad, "tol": SELECTION_TOL,
                               "n_runs_checked": len(PANEL) * len(SEEDS)}

    # --- 11. core unchanged
    digest = source_tree_digest()
    prior_digests = {ref_freeze(m, h, s)["source_tree_digest"]["core_tree"]
                     for m, h in PANEL for s in SEEDS}
    for f in sorted(PRIOR_EVID.glob("primary/*/seed*/freeze.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if "core_tree_digest" in rec:
            prior_digests.add(rec["core_tree_digest"])
    run_digests = {freeze(m, h, s)["source_tree_digest"]["core_tree"]
                   for m, h in PANEL for s in SEEDS}
    run_backbones = {freeze(m, h, s)["source_tree_digest"]["backbones_tree"]
                     for m, h in PANEL for s in SEEDS}
    checks["core_unchanged"] = bool(digest["core_tree"] in prior_digests
                                    and run_digests == {digest["core_tree"]}
                                    and run_backbones == {digest["backbones_tree"]})
    notes["core_unchanged"] = {
        "working_tree_core_tree": digest["core_tree"],
        "working_tree_backbones_tree": digest["backbones_tree"],
        "n_files": {"core": digest["core_tree_n_files"],
                    "backbones": digest["backbones_tree_n_files"]},
        "matches_a_frozen_stage_digest": digest["core_tree"] in prior_digests,
        "n_distinct_run_core_digests": len(run_digests),
    }
    if not checks["core_unchanged"]:
        issues.append("src/core or src/backbones digest does not match the recorded digest")

    # --- 12. TEST quarantine
    access_bad = []
    for m, h in PANEL:
        for s in SEEDS:
            key = f"{cell_key(m, h)}__seed{s}"
            r = freeze(m, h, s)
            st_ = r.get("access_state", {})
            if int(r["test_target_read_count"]) != 0 or int(r["test_rows_materialised"]) != 0:
                access_bad.append(f"{key}: test_target_read_count != 0")
            if st_.get("distinct_paths_blocked"):
                access_bad.append(f"{key}: blocked path opened {st_['distinct_paths_blocked']}")
            if int(st_.get("test_rows_returned", 0)) != 0:
                access_bad.append(f"{key}: TEST row returned")
            if int(r["test_rows_dropped_from_joint_cache"]) <= 0:
                access_bad.append(f"{key}: joint cache dropped no TEST rows")
            if r["is_registered_fit"] is not True:
                access_bad.append(f"{key}: is_registered_fit is not True")
    scan = check_no_direct_readers()
    if audit["access_state"]["distinct_paths_blocked"]:
        access_bad.append("access audit: a forbidden path was opened")
    if int(audit.get("test_target_read_count", 0)) != 0:
        access_bad.append("access audit: test_target_read_count != 0")
    opened = sorted(str(p.relative_to(EVID)).replace("\\", "/") for p in EVID.rglob("*")
                    if p.is_file() and p.name in FORBIDDEN_NAMES)
    checks["test_quarantine"] = not access_bad and not scan and not opened
    notes["test_quarantine"] = {
        "test_target_read_count": 0, "violations": access_bad, "direct_reader_scan_hits": scan,
        "forbidden_artifacts_written_here": opened,
        "guard_installed": audit["guard_installed"],
        "forbidden_files_enumerated": len(audit["forbidden_files_enumerated"]),
        "forbidden_name_patterns": audit["forbidden_name_patterns"],
        "distinct_paths_blocked_in_audit": audit["access_state"]["distinct_paths_blocked"],
    }

    # --- 13. one code state across all 24 runs (checked before the gate, which
    #        consumes it as one of its g5 facts)
    code_hashes = {freeze(m, h, s)["probe_code_hash"] for m, h in PANEL for s in SEEDS}
    rec_hashes = {freeze(m, h, s)["recovery_common_sha256"] for m, h in PANEL for s in SEEDS}
    if len(code_hashes) != 1:
        issues.append(f"{len(code_hashes)} distinct probe code hashes across the 24 runs")
    checks["one_code_state"] = bool(
        len(code_hashes) == 1 and rec_hashes == {sha256(RECOVERY_IMPL / "recovery_common.py")})
    notes["one_code_state"] = {
        "n_distinct_probe_code_hashes": len(code_hashes), "probe_code_hash": sorted(code_hashes),
        "recovery_common_sha256_matches_disk":
            rec_hashes == {sha256(RECOVERY_IMPL / "recovery_common.py")},
        "note": ("no run straddled a source edit, and the imported quarantine module is the one "
                 "on disk; the 12 new fits and the 12 reused runs share one trainer identity"),
    }

    # --- 14. gate arithmetic, recomputed from the raw per-seed rows
    rows = []
    for m, h in PANEL:
        for s in SEEDS:
            r, ref = freeze(m, h, s), ref_freeze(m, h, s)
            mae_o1, mae_r1 = float(r["selected_val_mae_ema"]), float(ref["selected_val_mae_ema"])
            rows.append({"cell": cell_key(m, h), "seed": s, "mae_o1": mae_o1, "mae_r1": mae_r1,
                         "gain": (mae_r1 - mae_o1) / mae_r1 * 100.0,
                         "step0_retreat": bool(int(r["selected_step"]) == 0
                                               and int(ref["selected_step"]) > 0),
                         "ratio": float(r["selected_metrics"]["val_correction_ratio"]),
                         "source": "new_fit" if is_new(m, h) else "reused_o1"})
    cells = []
    for m, h in PANEL:
        sub = [r for r in rows if r["cell"] == cell_key(m, h)]
        cells.append({"cell": cell_key(m, h),
                      "source": "new_fit" if is_new(m, h) else "reused_o1",
                      "median_gain": float(st.median([r["gain"] for r in sub])),
                      "mae_o1": float(st.median([r["mae_o1"] for r in sub])),
                      "mae_r1": float(st.median([r["mae_r1"] for r in sub])),
                      "per_seed_gains": [r["gain"] for r in sub]})
    gains = [c["median_gain"] for c in cells]
    retreats = sum(1 for r in rows if r["step0_retreat"])
    ratio_med = float(st.median([r["ratio"] for r in rows]))
    panel_median = float(st.median(gains))
    recomputed = {
        "g1_at_least_6_of_8_cell_medians_improve": sum(1 for g in gains if g > 0) >= GATE_MIN_CELLS,
        "g2_panel_median_relative_improvement_ge_0.5pct": panel_median >= GATE_PANEL_MEDIAN_MIN,
        "g3_worst_cell_degradation_le_0.5pct": min(gains) >= GATE_WORST_CELL_MIN,
        "g4_not_a_step0_or_near_zero_retreat": bool(
            ratio_med >= GATE_RATIO_MIN
            and retreats <= GATE_RETREAT_MAX_PROPORTIONAL
            and retreats <= GATE_RETREAT_MAX_ABSOLUTE),
        "g5_no_leakage_anomaly_support_or_code_violation": bool(
            checks["test_quarantine"] and checks["objective_switch_after_400"]
            and checks["schedule_identical"] and checks["core_unchanged"]
            and checks["one_code_state"] and checks["val_surface_rebuilt"]),
    }
    verdict = ("AUXILIARY_PERSISTENCE_BROADLY_HARMFUL_ON_RECOVERY_PANEL"
               if all(recomputed.values())
               else "AUXILIARY_PERSISTENCE_NOT_BROADLY_HARMFUL_ON_RECOVERY_PANEL")
    gate_match = recomputed == gate["checks"] and verdict == gate["verdict"]
    numbers_match = (
        abs(panel_median - float(gate["detail"]["panel_median_relative_gain_pct"])) < 1e-9
        and retreats == int(gate["detail"]["step0_retreat_seeds"])
        and abs(ratio_med - float(gate["detail"]["selected_correction_ratio_median"])) < 1e-12
        and abs(min(gains) - float(gate["detail"]["worst_cell_gain_pct"])) < 1e-9
        and abs(max(gains) - float(gate["detail"]["best_cell_gain_pct"])) < 1e-9
        and {c["cell"]: round(c["median_gain"], 9) for c in cells}
        == {c["cell"]: round(c["cell_median_relative_gain_pct"], 9)
            for c in gate["per_cell"]}
    )
    checks["gate_arithmetic"] = bool(gate_match and numbers_match)
    notes["gate_arithmetic"] = {
        "recomputed_checks": recomputed, "recomputed_verdict": verdict,
        "recorded_checks": gate["checks"], "recorded_verdict": gate["verdict"],
        "panel_median_recomputed": panel_median, "step0_retreats_recomputed": retreats,
        "step0_retreat_rule_proportional": f"<={GATE_RETREAT_MAX_PROPORTIONAL} of 24",
        "step0_retreat_rule_absolute": f"<={GATE_RETREAT_MAX_ABSOLUTE} of 24",
        "selected_correction_ratio_median_recomputed": ratio_med,
        "per_cell_recomputed": cells, "checks_match": gate_match, "numbers_match": numbers_match,
    }

    # --- 15. this stage imported the trainer, it did not write one
    own_loops = []
    for p in sorted(IMPL.rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if re.match(r"\s*def (train_step_based|train_o1|fit_cell|_train)\s*\(", line):
                own_loops.append(f"{p.name}:{i}: {line.strip()}")
    checks["no_new_trainer"] = not own_loops
    notes["no_new_trainer"] = {
        "violations": own_loops,
        "imports_train_o1": "train_o1" in (IMPL / "continuation_runner.py").read_text(
            encoding="utf-8"),
        "note": "implementation inheritance is by import, not by copy",
    }

    # --- 16. terminal token
    token_path = EVID / "STAGE_TOKEN.json"
    token = load_json(token_path).get("terminal_token") if token_path.is_file() else None
    checks["terminal_token"] = bool(token is not None
                                    and (token == TOKENS[0] or token.startswith(TOKENS[1])))
    notes["terminal_token"] = {"written_token": token, "artifact": "STAGE_TOKEN.json",
                               "allowed": list(TOKENS)}

    payload.update({
        "checks": checks, "notes": notes,
        "n_passed": sum(1 for v in checks.values() if v), "n_checks": len(checks),
        "failed": [k for k, v in checks.items() if not v], "issues": issues,
        "source_tree_digest": digest, "verifier_imports_runner": False,
        "verdict": ("HCH_V44_O1_FULL8_INDEPENDENTLY_VERIFIED" if all(checks.values())
                    else "HCH_V44_O1_FULL8_VERIFICATION_FAILED"),
    })
    return payload


if __name__ == "__main__":
    report = main()
    out = EVID / "INDEPENDENT_VERIFICATION_REPORT.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("verdict", "n_passed", "n_checks", "failed", "issues")},
                     ensure_ascii=False, indent=2))
