"""Independent verification of the HCH v4.4 objective-alignment probe.

Deliberately does **not** import ``runner``, ``probe_train`` or ``probe_common``.
Every causal rule, hash, gate number and provenance claim below is recomputed here
from the frozen artifacts, so a shared bug cannot pass unnoticed.

Checks (PROTOCOL.md sections 2, 5, 6, 8, 9):

 1. ``registered_coordinates``      exactly 4 cells x 3 seeds, no extra runs
 2. ``reference_pairing``           the reused R1 reference exists, matches its
                                    recorded hashes, and shares the O1 VAL surface
 3. ``schedule_identical``          the registered schedule is one schedule, equal
                                    to the reference's, with no sweep
 4. ``only_loss_schedule_changed``  the O1 training loop differs from the reference
                                    loop by exactly the registered lines
 5. ``pre_switch_identity``         O1 reproduces the reference checkpoints bit for
                                    bit up to and including step 400
 6. ``objective_switch_after_400``  the switch lands exactly after update 400
 7. ``selection_rule``              the running-best rule with its 1e-7 tolerance
 8. ``core_unchanged``              src/core + src/backbones digest unchanged
 9. ``test_quarantine``             TEST reads = 0 and no forbidden path opened
10. ``gate_arithmetic``             the gate recomputed from the raw per-seed rows
11. ``terminal_token``              the reported token is one of the two registered
"""
from __future__ import annotations

import difflib
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]
IMPL = STAGE / "implementation"
EVID = REPO / "experiments/evidence/hch_v44_objective_alignment_probe_20260917"
RECOVERY_STAGE = REPO / "experiments/current/hch_v44_optimization_recovery_20260917"
RECOVERY_IMPL = RECOVERY_STAGE / "implementation"
RECOVERY_EVID = REPO / "experiments/evidence/hch_v44_optimization_recovery_20260917"
PRIOR_EVID = REPO / "experiments/evidence/hch_v44_china5_fullpanel_main_eval_20260917"

PANEL = [("GANSU_DA", "PatchTST"), ("SHANDONG_DA", "iTransformer"),
         ("SHAANXI_DA", "TimeMixer"), ("NINGXIA_DA", "iTransformer")]
SEEDS = [7, 17, 37]
SWITCH_STEP = 400
SELECTION_TOL = 1e-7
HORIZON = 24
WINDOW = 7

REGISTERED_SCHEDULE = {
    "optimizer": "AdamW", "lr": 1e-3, "batch_size": 32, "weight_decay": 1e-4,
    "grad_clip": 1.0, "ema_decay": 0.995, "eval_weights": "ema",
    "max_steps": 2000, "val_every": 50, "no_stop_before": 800, "patience_checks": 8,
    "readout_init": None,
}
FORBIDDEN_NAMES = ("CELL_MEDIAN_RESULTS.csv", "PER_SEED_RESULTS.csv", "BASELINE_COMPARISON.csv",
                   "HCH_V44_MAIN_20260917.csv", "RESULTS_LONG.csv",
                   "BASELINE_COMPLETION_20260917.csv", "result.json", "test_predictions.npz")

TOKENS = ("HCH_V44_OBJECTIVE_ALIGNMENT_PROBE_COMPLETE_FOR_ADJUDICATION",
          "HCH_V44_OBJECTIVE_ALIGNMENT_PROBE_BLOCKED_")


# --------------------------------------------------------------------------- utils
def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_key(market: str, host: str) -> str:
    return f"{market}__{host}"


def run_dir(market: str, host: str, seed: int) -> Path:
    return EVID / "o1_runs" / cell_key(market, host) / f"seed{seed}"


def reference_dir(market: str, host: str, seed: int) -> Path:
    return RECOVERY_EVID / "r1_runs" / cell_key(market, host) / f"seed{seed}"


def source_tree_digest() -> dict:
    """Digest of the frozen scientific source, recomputed here from scratch."""
    out = {}
    for name, pattern in (("core_tree", "src/core"), ("backbones_tree", "src/backbones")):
        paths = sorted(p for p in (REPO / pattern).rglob("*")
                       if p.is_file() and p.suffix in (".py", ".md"))
        items = [f"{p.relative_to(REPO).as_posix()}|{sha256(p)}" for p in paths]
        out[name] = sha256_bytes("\n".join(items).encode())
        out[name + "_n_files"] = len(paths)
    return out


# ------------------------------------------------------------------- causal rules
def split_plan(market: str):
    """The registered split plan, imported from the V2 stage independently."""
    v2_impl = REPO / "experiments/current/hch_china5_common_benchmark_701020_full_v2/implementation"
    if str(v2_impl) not in sys.path:
        sys.path.insert(0, str(v2_impl))
    import cb_contracts as C  # noqa: PLC0415

    return C.split_plan(market)


def to_date(value):
    import datetime as dt

    if isinstance(value, dt.date):
        return value
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]").astype(object)
    return dt.date.fromisoformat(str(value)[:10])


def origin_of(day):
    import datetime as dt

    return dt.datetime.combine(day, dt.time(1, 0)) - dt.timedelta(hours=15, minutes=30)


def reveal_of(day):
    """Moment ``day``'s full 24h target is known: hour-ending 24:00, i.e. next day 01:00."""
    import datetime as dt

    return dt.datetime.combine(day + dt.timedelta(days=1), dt.time(1, 0))


def history_window(target_day, available: set):
    """The 7 most recent admissible available days, or None if there are fewer.

    Mirrors ``src.core.history.select_revealed_history``'s predicate
    (``day < target`` and ``reveal_time < origin``) exactly: it is a
    **gap-tolerant** rule, so a missing intermediate day pulls the window further
    back rather than voiding it, and the window need not be a contiguous
    ``[d-8, d-2]`` block.
    """
    import datetime as dt

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
    v2 = REPO / "experiments/evidence/hch_china5_common_benchmark_701020_full_v2_20260913"
    p = v2 / "02_hosts" / market / host / "host_predictions_joint.npz"
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
    # NB: the support file carries both ``oof_days`` (an int *count*) and
    # ``oof_day_ids`` (the day list).  Only the latter is iterable.
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
        finite = np.isfinite(y).all(axis=1)
        out[role] = {
            "n_registered": len(ids), "n_scorable": int(finite.sum()),
            "days": ids, "finite": finite, "y_true": y,
            "host_pred": z["host_pred"][mask], "day_ids_hash": None,
        }
    # eligible rows.  TRAIN and VAL share the same starting history index: the
    # revealed residuals of the causal shadow-OOF support (``index`` in
    # ``recovery_train.build_cell_data``).  TRAIN days are NOT added to it.
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
    leaked = [d for d in train_eligible + val_days if d in test_ids]
    out["test_days_in_train_or_val"] = len(leaked)
    out["n_test_rows_dropped"] = z["n_test_rows_dropped"]
    out["cache_days_total"] = z["cache_days_total"]
    return out


def replay_selection(curve: list[dict], field: str = "val_mae_ema") -> tuple:
    """The registered running-best rule, re-implemented."""
    best_v, best_step = float("inf"), -1
    for e in curve:
        v = float(e[field])
        if v < best_v - SELECTION_TOL:
            best_v, best_step = v, int(e["step"])
    return best_step, best_v


# ---------------------------------------------------------------------- loop diff
def loop_region(text: str) -> list[str]:
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
    new = loop_region((IMPL / "probe_train.py").read_text(encoding="utf-8"))
    diff = [l for l in difflib.unified_diff(ref, new, lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    removed = {l[1:].strip() for l in diff if l.startswith("-")}
    added = {l[1:].strip() for l in diff if l.startswith("+")}
    ref_src = (RECOVERY_IMPL / "recovery_train.py").read_text(encoding="utf-8")
    new_src = (IMPL / "probe_train.py").read_text(encoding="utf-8")
    return {
        "n_removed_lines": len(removed), "n_added_lines": len(added),
        "removed": sorted(removed), "added": sorted(added),
        "removed_is_exactly_expected": removed == EXPECTED_REMOVED,
        "added_is_exactly_expected": added == EXPECTED_ADDED,
        "reference_has_original_calls": (
            'loss = model.compute_loss(out, sub_target, sub_res, data["scales"])["total"]' in ref_src
            and "record_check(step, samples_seen, False, norms)" in ref_src
        ),
        "probe_has_replacement_calls": (
            "objective(model.compute_loss(out, sub_target, sub_res" in new_src
            and "record_check(step, samples_seen, False, norms, float(loss.detach()), used_full)"
            in new_src
        ),
    }


def check_no_direct_readers() -> list[str]:
    """No probe module may call the prior stage's TEST-materialising readers."""
    import re

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
    checks: dict = {}
    notes: dict = {}
    issues: list[str] = []
    payload: dict = {"schema": "hch_v44_objective_probe_verification.v1"}

    gate = load_json(EVID / "GATE.json")
    config = load_json(EVID / "PROBE_CONFIG.json")
    reference = load_json(EVID / "REFERENCE_PROVENANCE.json")
    payload["gate_verdict"] = gate["verdict"]

    # --- 1. registered coordinates
    expected = {f"{cell_key(m, h)}__seed{s}" for m, h in PANEL for s in SEEDS}
    found = set()
    run_root = EVID / "o1_runs"
    for cell_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in cell_dir.iterdir() if p.is_dir()):
            found.add(f"{cell_dir.name}__{seed_dir.name}")
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra or len(found) != 12:
        issues.append(f"run set mismatch: missing={missing} extra={extra}")
    checks["registered_coordinates"] = not missing and not extra and len(found) == 12
    notes["registered_coordinates"] = {
        "expected": sorted(expected), "n_found": len(found),
        "missing": missing, "extra": extra,
        "each_run_has_freeze_curve_checkpoint": all(
            (run_dir(m, h, s) / f).is_file()
            for m, h in PANEL for s in SEEDS
            for f in ("freeze.json", "training_curve.json", "selected_ema.pt")
        ),
    }

    # --- 2. reference pairing and VAL-surface identity
    pair_bad = []
    pairing = {}
    for m, h in PANEL:
        for s in SEEDS:
            o1 = load_json(run_dir(m, h, s) / "freeze.json")
            rd = reference_dir(m, h, s)
            r1 = load_json(rd / "freeze.json")
            rec = reference["per_coordinate"][f"{cell_key(m, h)}__seed{s}"]
            got = {"freeze_sha256": sha256(rd / "freeze.json"),
                   "curve_sha256": sha256(rd / "training_curve.json"),
                   "checkpoint_sha256": sha256(rd / "selected_ema.pt")}
            for k, v in got.items():
                if rec[k] != v:
                    pair_bad.append(f"{cell_key(m, h)}__seed{s}: {k} differs from the registered reference hash")
            if int(o1["n_val_rows"]) != int(r1["n_val_rows"]):
                pair_bad.append(f"{cell_key(m, h)}__seed{s}: n_val_rows {o1['n_val_rows']} != {r1['n_val_rows']}")
            if o1["val_day_ids_hash"] != r1["val_day_ids_hash"]:
                pair_bad.append(f"{cell_key(m, h)}__seed{s}: VAL day ids differ from the reference")
            if o1["history_support_hash"] != r1["history_support_hash"]:
                pair_bad.append(f"{cell_key(m, h)}__seed{s}: history support differs")
            if int(r1["test_target_read_count"]) != 0:
                pair_bad.append(f"{cell_key(m, h)}__seed{s}: reference read TEST")
            pairing[f"{cell_key(m, h)}__seed{s}"] = {
                "r1_recipe_id": r1["recipe_id"],
                "r1_selected_step": int(r1["selected_step"]),
                "o1_selected_step": int(o1["selected_step"]),
                "n_val_rows": int(o1["n_val_rows"]),
            }
    checks["reference_pairing"] = not pair_bad
    notes["reference_pairing"] = {"n_paired": len(pairing), "violations": pair_bad,
                                  "all_reference_runs_are_R1_full_loss": all(
                                      v["r1_recipe_id"] == "R1_STEP_BUDGET_2000"
                                      for v in pairing.values()),
                                  "reference_all_12_present": reference["all_12_present"]}

    # --- 2b. the measured partition, rebuilt from the raw cache
    rebuilt_bad, rebuilt = [], {}
    for m, h in PANEL:
        r = rebuild_rows(m, h)
        rebuilt[cell_key(m, h)] = {
            "n_train_eligible": len(r["TRAIN"]["eligible_days"]),
            "n_val_eligible": len(r["VAL"]["eligible_days"]),
            "test_days_in_train_or_val": r["test_days_in_train_or_val"],
            "test_rows_dropped_from_joint_cache": r["n_test_rows_dropped"],
            "cache_days_total": r["cache_days_total"],
            "train_day_ids_hash": r["TRAIN"]["day_ids_hash"],
            "val_day_ids_hash": r["VAL"]["day_ids_hash"],
        }
        for s in SEEDS:
            o1 = load_json(run_dir(m, h, s) / "freeze.json")
            for key, want in (("n_train_rows", len(r["TRAIN"]["eligible_days"])),
                              ("n_val_rows", len(r["VAL"]["eligible_days"])),
                              ("train_day_ids_hash", r["TRAIN"]["day_ids_hash"]),
                              ("val_day_ids_hash", r["VAL"]["day_ids_hash"])):
                if o1[key] != want:
                    rebuilt_bad.append(f"{cell_key(m, h)}__seed{s}: {key} {o1[key]!r} != rebuilt {want!r}")
            if dict(o1["train_rows_excluded"]) != r["TRAIN"]["excluded"]:
                rebuilt_bad.append(
                    f"{cell_key(m, h)}__seed{s}: train_rows_excluded "
                    f"{o1['train_rows_excluded']} != rebuilt {r['TRAIN']['excluded']}")
        if r["test_days_in_train_or_val"]:
            rebuilt_bad.append(f"{cell_key(m, h)}: TEST day inside TRAIN/VAL")
        if r["n_test_rows_dropped"] <= 0:
            rebuilt_bad.append(f"{cell_key(m, h)}: joint cache dropped no TEST rows")
    checks["val_surface_rebuilt"] = not rebuilt_bad
    notes["val_surface_rebuilt"] = {
        "rule": ("eligible = fully-finite 24h target AND a legal W=7 window [d-8, d-2] "
                 "with history from the shadow-OOF support; VAL residuals revealed "
                 "prequentially.  Reimplemented here from the split plan and the joint "
                 "cache, independently of the runner."),
        "violations": rebuilt_bad, "per_cell": rebuilt,
    }

    # --- 3. schedule identical, no sweep
    sched_bad, schedules = [], set()
    for m, h in PANEL:
        for s in SEEDS:
            r = load_json(run_dir(m, h, s) / "freeze.json")
            sched = {k: r["optimization"].get(k) for k in REGISTERED_SCHEDULE}
            schedules.add(json.dumps(sched, sort_keys=True))
            if sched != REGISTERED_SCHEDULE:
                sched_bad.append(f"{cell_key(m, h)}__seed{s}: {sched}")
            if r["recipe_id"] != "O1_AUX_WARMUP_400_THEN_MAE":
                sched_bad.append(f"{cell_key(m, h)}__seed{s}: recipe_id {r['recipe_id']!r}")
            if int(r["objective_probe"]["switch_step"]) != SWITCH_STEP:
                sched_bad.append(f"{cell_key(m, h)}__seed{s}: switch_step != 400")
            if r["diag_subset"]["rule"] != ("np.linspace over registered TRAIN order, "
                                            "capped at 256"):
                sched_bad.append(f"{cell_key(m, h)}__seed{s}: diagnostic subset changed")
            if r["interleaver_audit"]["reweighting"] or r["interleaver_audit"]["oversampling"]:
                sched_bad.append(f"{cell_key(m, h)}__seed{s}: interleaver changed")
    ref_scheds = {json.dumps({k: load_json(reference_dir(m, h, s) / "freeze.json")
                              ["optimization"].get(k) for k in REGISTERED_SCHEDULE},
                             sort_keys=True)
                  for m, h in PANEL for s in SEEDS}
    if ref_scheds != {json.dumps(REGISTERED_SCHEDULE, sort_keys=True)}:
        sched_bad.append("reference schedule is not the registered schedule")
    checks["schedule_identical"] = not sched_bad and len(schedules) == 1
    notes["schedule_identical"] = {
        "n_distinct_o1_schedules": len(schedules), "violations": sched_bad,
        "registered": REGISTERED_SCHEDULE,
        "reference_schedule_matches": ref_scheds == {json.dumps(REGISTERED_SCHEDULE, sort_keys=True)},
        "no_extra_arms": sorted(p.name for p in EVID.iterdir() if p.is_dir()) == ["o1_runs", "PLOTS"],
    }

    # --- 4. only the loss schedule changed
    diff = check_only_loss_schedule_changed()
    checks["only_loss_schedule_changed"] = bool(
        diff["removed_is_exactly_expected"] and diff["added_is_exactly_expected"]
        and diff["reference_has_original_calls"] and diff["probe_has_replacement_calls"])
    notes["only_loss_schedule_changed"] = diff

    # --- 5. pre-switch identity against the reference
    identity = {}
    id_bad = []
    for m, h in PANEL:
        for s in SEEDS:
            o1 = {e["step"]: e for e in load_json(run_dir(m, h, s) / "training_curve.json")}
            r1 = {e["step"]: e for e in load_json(reference_dir(m, h, s) / "training_curve.json")}
            shared = sorted(set(o1) & set(r1))
            pre = [x for x in shared if x <= SWITCH_STEP]
            worst = 0.0
            for step in pre:
                for f in ("val_mae_ema", "val_mae_raw", "train_mae_ema", "diag_reconstruction_mae",
                          "level_mae", "mass_mae", "shape_w1", "train_correction_ratio"):
                    a, b = float(o1[step][f]), float(r1[step][f])
                    worst = max(worst, abs(a - b) / max(1.0, abs(b)))
                if o1[step]["grad_norms_preclip"] != r1[step]["grad_norms_preclip"]:
                    id_bad.append(f"{cell_key(m, h)}__seed{s} step{step}: grad norms differ")
            step0_dev = abs(float(o1[0]["val_mae_ema"]) - float(r1[0]["val_mae_ema"]))
            if step0_dev > 1e-6:
                id_bad.append(f"{cell_key(m, h)}__seed{s}: step-0 deviation {step0_dev}")
            if worst > 0.0:
                id_bad.append(f"{cell_key(m, h)}__seed{s}: pre-switch deviation {worst}")
            identity[f"{cell_key(m, h)}__seed{s}"] = {
                "shared_pre_switch_steps": pre, "max_relative_deviation": worst,
                "step0_abs_deviation": step0_dev,
            }
    checks["pre_switch_identity"] = not id_bad
    notes["pre_switch_identity"] = {
        "note": ("O1's updates 1..400 use the reference's own loss and the same random "
                 "stream, so the pre-switch checkpoints must reproduce the reference "
                 "exactly; any deviation would mean a second, unregistered change."),
        "n_coords_compared": len(identity), "violations": id_bad, "per_coord": identity,
    }

    # --- 6. objective switch exactly after 400
    switch_bad = []
    for m, h in PANEL:
        for s in SEEDS:
            curve = load_json(run_dir(m, h, s) / "training_curve.json")
            marks = [e["step"] for e in curve if e["objective_switch_here"]]
            if marks != [SWITCH_STEP]:
                switch_bad.append(f"{cell_key(m, h)}__seed{s}: switch markers {marks}")
            for e in curve:
                if not e["is_optimization_step"]:
                    if e["used_full_loss_at_this_step"] is not None:
                        switch_bad.append(f"{cell_key(m, h)}__seed{s}: step-0 misuse")
                    continue
                expect = int(e["step"]) <= SWITCH_STEP
                if bool(e["used_full_loss_at_this_step"]) != expect:
                    switch_bad.append(
                        f"{cell_key(m, h)}__seed{s} step{e['step']}: used_full="
                        f"{e['used_full_loss_at_this_step']}")
                want = "L_rec_plus_aux" if expect else "L_rec_only"
                if e["loss_schedule"] != want:
                    switch_bad.append(f"{cell_key(m, h)}__seed{s} step{e['step']}: {e['loss_schedule']}")
            if not curve[-1]["samples_seen"]:
                switch_bad.append(f"{cell_key(m, h)}__seed{s}: empty curve")
    checks["objective_switch_after_400"] = not switch_bad
    notes["objective_switch_after_400"] = {"violations": switch_bad[:10],
                                           "n_violations": len(switch_bad)}

    # --- 7. selection rule
    sel_bad = []
    for m, h in PANEL:
        for s in SEEDS:
            curve = load_json(run_dir(m, h, s) / "training_curve.json")
            rec = load_json(run_dir(m, h, s) / "freeze.json")
            step, val = replay_selection(curve)
            if step != int(rec["selected_step"]):
                sel_bad.append(f"{cell_key(m, h)}__seed{s}: replay step {step} != {rec['selected_step']}")
            if abs(val - float(rec["selected_val_mae_ema"])) > 1e-9 * max(1.0, abs(val)):
                sel_bad.append(f"{cell_key(m, h)}__seed{s}: replay value {val} != {rec['selected_val_mae_ema']}")
            if min(float(e["val_mae_ema"]) for e in curve) < val - 1e-9:
                sel_bad.append(f"{cell_key(m, h)}__seed{s}: selection is not the curve minimum")
            if not any(e["selected"] and int(e["step"]) == step for e in curve):
                sel_bad.append(f"{cell_key(m, h)}__seed{s}: no selected curve entry at the chosen step")
    checks["selection_rule"] = not sel_bad
    notes["selection_rule"] = {"violations": sel_bad, "tol": SELECTION_TOL}

    # --- 8. core unchanged
    digest = source_tree_digest()
    prior_digests = set()
    for m, h in PANEL:
        for s in SEEDS:
            prior_digests.add(load_json(reference_dir(m, h, s) / "freeze.json")
                              ["source_tree_digest"]["core_tree"])
    for f in sorted(PRIOR_EVID.glob("primary/*/seed*/freeze.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if "core_tree_digest" in rec:
            prior_digests.add(rec["core_tree_digest"])
    o1_digests = {load_json(run_dir(m, h, s) / "freeze.json")
                  ["source_tree_digest"]["core_tree"] for m, h in PANEL for s in SEEDS}
    o1_backbones = {load_json(run_dir(m, h, s) / "freeze.json")
                    ["source_tree_digest"]["backbones_tree"] for m, h in PANEL for s in SEEDS}
    checks["core_unchanged"] = bool(
        digest["core_tree"] in prior_digests and o1_digests == {digest["core_tree"]}
        and o1_backbones == {digest["backbones_tree"]})
    notes["core_unchanged"] = {
        "working_tree_core_tree": digest["core_tree"],
        "working_tree_backbones_tree": digest["backbones_tree"],
        "n_files": {"core": digest["core_tree_n_files"], "backbones": digest["backbones_tree_n_files"]},
        "prior_recorded_digests": sorted(prior_digests),
        "matches_a_prior_stage_digest": digest["core_tree"] in prior_digests,
    }
    if not checks["core_unchanged"]:
        issues.append("src/core or src/backbones digest does not match the recorded frozen digest")

    # --- 9. TEST quarantine
    access_bad = []
    for m, h in PANEL:
        for s in SEEDS:
            r = load_json(run_dir(m, h, s) / "freeze.json")
            st_ = r.get("access_state", {})
            if int(r["test_target_read_count"]) != 0 or int(r["test_rows_materialised"]) != 0:
                access_bad.append(f"{cell_key(m, h)}__seed{s}: test_target_read_count != 0")
            if st_.get("distinct_paths_blocked"):
                access_bad.append(f"{cell_key(m, h)}__seed{s}: blocked path opened "
                                  f"{st_['distinct_paths_blocked']}")
            if int(st_.get("test_rows_returned", 0)) != 0:
                access_bad.append(f"{cell_key(m, h)}__seed{s}: TEST row returned")
            if int(r["test_rows_dropped_from_joint_cache"]) <= 0:
                access_bad.append(f"{cell_key(m, h)}__seed{s}: joint cache dropped no TEST rows")
            if r["is_registered_fit"] is not True:
                access_bad.append(f"{cell_key(m, h)}__seed{s}: is_registered_fit is not True")
    scan = check_no_direct_readers()
    audit = load_json(EVID / "ACCESS_AUDIT.json")
    if audit["access_state"]["distinct_paths_blocked"]:
        access_bad.append("access audit: a forbidden path was opened")
    checks["test_quarantine"] = not access_bad and not scan
    notes["test_quarantine"] = {
        "test_target_read_count": 0, "violations": access_bad,
        "direct_reader_scan_hits": scan,
        "guard_installed": audit["guard_installed"],
        "forbidden_files_enumerated": len(audit["forbidden_files_enumerated"]),
        "forbidden_name_patterns": audit["forbidden_name_patterns"],
        "distinct_paths_blocked_in_audit": audit["access_state"]["distinct_paths_blocked"],
    }

    # --- 10. gate arithmetic, recomputed from the raw per-seed table
    rows = []
    for m, h in PANEL:
        for s in SEEDS:
            o1 = load_json(run_dir(m, h, s) / "freeze.json")
            r1 = load_json(reference_dir(m, h, s) / "freeze.json")
            mae_o1, mae_r1 = float(o1["selected_val_mae_ema"]), float(r1["selected_val_mae_ema"])
            rows.append({
                "cell": cell_key(m, h), "seed": s, "mae_o1": mae_o1, "mae_r1": mae_r1,
                "gain": (mae_r1 - mae_o1) / mae_r1 * 100.0,
                "step0_retreat": bool(int(o1["selected_step"]) == 0 and int(r1["selected_step"]) > 0),
                "ratio": float(o1["selected_metrics"]["val_correction_ratio"]),
            })
    cells = []
    for m, h in PANEL:
        sub = [r for r in rows if r["cell"] == cell_key(m, h)]
        cells.append({"cell": cell_key(m, h),
                      "median_gain": float(st.median([r["gain"] for r in sub])),
                      "mae_o1": float(st.median([r["mae_o1"] for r in sub])),
                      "mae_r1": float(st.median([r["mae_r1"] for r in sub]))})
    gains = [c["median_gain"] for c in cells]
    retreats = sum(1 for r in rows if r["step0_retreat"])
    ratio_med = float(st.median([r["ratio"] for r in rows]))
    recomputed = {
        "g1_at_least_3_of_4_cell_medians_improve": sum(1 for g in gains if g > 0) >= 3,
        "g2_panel_median_relative_improvement_ge_0.5pct": float(st.median(gains)) >= 0.5,
        "g3_worst_cell_degradation_le_0.5pct": min(gains) >= -0.5,
        "g4_not_a_step0_or_near_zero_retreat": retreats <= 2 and ratio_med >= 0.05,
        "g5_no_leakage_anomaly_or_protocol_violation": bool(
            checks["test_quarantine"] and checks["objective_switch_after_400"]
            and checks["schedule_identical"]),
    }
    verdict = ("AUXILIARY_PERSISTENCE_LIKELY_HARMFUL" if all(recomputed.values())
               else "AUXILIARY_PERSISTENCE_NOT_SHOWN_HARMFUL")
    gate_match = recomputed == gate["checks"] and verdict == gate["verdict"]
    numbers_match = (
        abs(float(st.median(gains)) - float(gate["detail"]["panel_median_relative_gain_pct"])) < 1e-9
        and retreats == int(gate["detail"]["step0_retreat_seeds"])
        and abs(ratio_med - float(gate["detail"]["selected_correction_ratio_median"])) < 1e-12
    )
    checks["gate_arithmetic"] = bool(gate_match and numbers_match)
    notes["gate_arithmetic"] = {
        "recomputed_checks": recomputed, "recomputed_verdict": verdict,
        "recorded_checks": gate["checks"], "recorded_verdict": gate["verdict"],
        "panel_median_recomputed": float(st.median(gains)),
        "step0_retreats_recomputed": retreats,
        "selected_correction_ratio_median_recomputed": ratio_med,
        "per_cell_recomputed": cells,
    }

    # --- 10b. all 12 fits ran under one identical code state
    code_hashes = {load_json(run_dir(m, h, s) / "freeze.json")["probe_code_hash"]
                   for m, h in PANEL for s in SEEDS}
    rec_hashes = {load_json(run_dir(m, h, s) / "freeze.json")["recovery_common_sha256"]
                  for m, h in PANEL for s in SEEDS}
    ref_hashes = {sha256(RECOVERY_IMPL / "recovery_common.py")}
    checks["one_code_state"] = bool(len(code_hashes) == 1 and rec_hashes == ref_hashes)
    notes["one_code_state"] = {
        "n_distinct_probe_code_hashes": len(code_hashes),
        "probe_code_hash": sorted(code_hashes),
        "recovery_common_sha256_matches_disk": rec_hashes == ref_hashes,
        "note": ("no run straddled a source edit, and the imported quarantine module "
                 "is the one on disk"),
    }

    # --- 11. terminal token
    token_path = EVID / "STAGE_TOKEN.json"
    token = load_json(token_path).get("terminal_token") if token_path.is_file() else None
    checks["terminal_token"] = bool(token is not None
                                    and (token == TOKENS[0] or token.startswith(TOKENS[1])))
    notes["terminal_token"] = {
        "written_token": token, "artifact": "STAGE_TOKEN.json",
        "allowed": list(TOKENS),
        "note": "the probe's terminal token is fixed by the protocol, not by the verdict",
    }

    payload.update({
        "checks": checks,
        "notes": notes,
        "n_passed": sum(1 for v in checks.values() if v),
        "n_checks": len(checks),
        "failed": [k for k, v in checks.items() if not v],
        "issues": issues,
        "source_tree_digest": digest,
        "verdict": ("HCH_V44_OBJECTIVE_PROBE_INDEPENDENTLY_VERIFIED"
                    if all(checks.values()) else "HCH_V44_OBJECTIVE_PROBE_VERIFICATION_FAILED"),
    })
    return payload


if __name__ == "__main__":
    report = main()
    out = EVID / "INDEPENDENT_VERIFICATION_REPORT.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("verdict", "n_passed", "n_checks", "failed", "issues")},
                     ensure_ascii=False, indent=2))
