"""Independent verifier for the HCH v4.4 optimization-recovery stage.

This script deliberately does **not** import ``runner``, ``recovery_train`` or any
verdict function from the implementation.  Every claim is re-derived from the raw
artifacts on disk, and the causal rules are re-implemented here from the contract
text rather than reused:

* the W=7 revealed-history window and the eligible TRAIN/VAL row sets are rebuilt
  independently -- this is the one place the rule is deliberately written twice;
* checkpoint selection is replayed as the registered running-best rule, including
  its 1e-7 strict-improvement tolerance; plain ``min()`` is *not* equivalent and
  any case where the two disagree is reported;
* the promotion-gate arithmetic is recomputed from raw per-seed numbers;
* ``src/core`` identity is proved against the digest the *completed* stage
  recorded before this stage existed, so a silent core edit cannot pass.

Run: ``python verification/verify_recovery.py``
"""
from __future__ import annotations

import datetime as dt
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
EVID = REPO / "experiments/evidence/hch_v44_optimization_recovery_20260917"
PRIOR_EVID = REPO / "experiments/evidence/hch_v44_china5_fullpanel_main_eval_20260917"
V2 = REPO / "experiments/evidence/hch_china5_common_benchmark_701020_full_v2_20260913"
IMPL = STAGE / "implementation"

MARKETS = ["GANSU_DA", "SHANDONG_DA", "SHAANXI_DA", "NINGXIA_DA", "QINGHAI_DA"]
HOSTS = ["PatchTST", "TimeMixer", "iTransformer", "LSTM"]
SEEDS = [7, 17, 37]
HORIZON = 24
HISTORY_DAYS = 7
R1_PANEL = {
    "GANSU_DA__PatchTST", "GANSU_DA__LSTM", "SHANDONG_DA__PatchTST",
    "SHANDONG_DA__iTransformer", "SHAANXI_DA__TimeMixer", "SHAANXI_DA__PatchTST",
    "NINGXIA_DA__iTransformer", "QINGHAI_DA__TimeMixer",
}
# The registered step-budget recipe.  Any deviation is a per-market recipe.
REGISTERED_SCHEDULE = {
    "optimizer": "AdamW", "lr": 1e-3, "batch_size": 32, "weight_decay": 1e-4,
    "grad_clip": 1.0, "ema_decay": 0.995, "eval_weights": "ema",
    "max_steps": 2000, "val_every": 50, "no_stop_before": 800, "patience_checks": 8,
}
SELECTION_TOL = 1e-7  # the tolerance inside the registered running-best rule

TOKEN_FULL_OK = "HCH_V44_OPTIMIZATION_RECOVERY_READY_FOR_UNTOUCHED_CONFIRMATION"
TOKEN_FULL_NO = "HCH_V44_OPTIMIZATION_RECOVERY_NOT_SUPPORTED"

# --------------------------------------------------------------------------- rules
_GATE = dt.time(9, 30)
_REVEAL = dt.time(1, 0)


def to_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def origin_of(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day - dt.timedelta(days=1), _GATE)


def reveal_of(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day + dt.timedelta(days=1), _REVEAL)


def history_window(target_day, available):
    """W=7 window [d-8, d-2], rebuilt from the contract text."""
    target = to_date(target_day)
    origin = origin_of(target)
    chosen = []
    for day in sorted((d for d in available if d < target), reverse=True):
        if not reveal_of(day) < origin:
            continue
        chosen.append(day)
        if len(chosen) == HISTORY_DAYS:
            break
    return chosen[::-1] if len(chosen) == HISTORY_DAYS else None


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def source_tree_digest() -> dict:
    out = {}
    for name, sub in (("core_tree", "src/core"), ("backbones_tree", "src/backbones")):
        paths = sorted(p for p in (REPO / sub).rglob("*")
                       if p.is_file() and p.suffix in (".py", ".md"))
        items = [f"{p.relative_to(REPO).as_posix()}|{sha256(p)}" for p in paths]
        out[name] = sha256_bytes("\n".join(items).encode())
    return out


def split_ids(market: str, role: str) -> list:
    sys.path.insert(0, str(REPO / "experiments/current/"
                          "hch_china5_common_benchmark_701020_full_v2/implementation"))
    import cb_contracts as C  # noqa: PLC0415

    return [to_date(d) for d in C.split_plan(market).ids(role)]


# --------------------------------------------------------------------------- rebuild
def rebuild_rows(market: str, host: str) -> dict:
    """Rebuild the TRAIN/VAL eligible rows and their day ids from scratch."""
    key = f"{market}__{host}"
    with np.load(PRIOR_EVID / "shadow_oof" / f"{key}.npz", allow_pickle=False) as z:
        residual = z["residual"].astype(np.float64)
        oof_days = [to_date(d) for d in z["day_id"]]
    index = {d for i, d in enumerate(oof_days) if np.isfinite(residual[i]).all()}

    with np.load(V2 / "02_hosts" / market / host / "host_predictions_joint.npz",
                 allow_pickle=False) as z:
        segment = z["segment"].astype(str)
        ts = z["timestamp"]
        y = z["y_true"].reshape(len(z["y_true"]), HORIZON)
        hp = z["host_pred"].reshape(len(z["host_pred"]), HORIZON)

    plan = {r: split_ids(market, r) for r in ("TRAIN", "VAL", "TEST")}
    for role in ("TRAIN", "VAL"):
        mask = segment == role
        if [to_date(t) for t in ts[mask]] != plan[role]:
            raise AssertionError(f"{key} {role}: joint cache days differ from the split plan")

    train_mask = segment == "TRAIN"
    train_days = [to_date(t) for t in ts[train_mask]]
    train_finite = np.isfinite(y[train_mask]).all(axis=1)
    train_rows = [d for i, d in enumerate(train_days)
                  if train_finite[i] and history_window(d, index) is not None]

    val_mask = segment == "VAL"
    val_days = [to_date(t) for t in ts[val_mask]]
    val_finite = np.isfinite(y[val_mask]).all(axis=1)
    val_index = set(index)
    val_rows = []
    for i, day in enumerate(val_days):
        if not val_finite[i] or history_window(day, val_index) is None:
            continue
        val_rows.append(day)
        val_index.add(day)  # prequential reveal, exactly as the registered loop does

    test_ids = set(plan["TEST"])
    return {
        "n_train_rows": len(train_rows), "n_val_rows": len(val_rows),
        "train_day_ids_hash": sha256_bytes(json.dumps([d.isoformat() for d in train_rows]).encode()),
        "val_day_ids_hash": sha256_bytes(json.dumps([d.isoformat() for d in val_rows]).encode()),
        "test_days_in_train": sum(1 for d in train_rows if d in test_ids),
        "test_days_in_val": sum(1 for d in val_rows if d in test_ids),
        "host_val_mae": float(np.abs(y[val_mask][val_finite] - hp[val_mask][val_finite]).mean()),
    }


def replay_selection(curve: list, field: str):
    """Replay the registered running-best rule; returns (step, value)."""
    best_v, best_step = float("inf"), -1
    for e in curve:
        v = float(e[field])
        if v < best_v - SELECTION_TOL:
            best_v, best_step = v, int(e["step"])
    return best_step, best_v


def runs_of(arm: str) -> list:
    root = EVID / f"{arm.lower()}_runs"
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(root.glob("*/seed*/freeze.json"))]


def curve_of(arm: str, cell: str, seed: int) -> list:
    p = EVID / f"{arm.lower()}_runs" / cell / f"seed{seed}" / "training_curve.json"
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- main
def main() -> int:
    checks: dict = {}

    def record(name: str, _passed: bool, **payload) -> None:
        payload["passed"] = bool(_passed)
        checks[name] = payload

    arm = "R2B" if (EVID / "R2B_GATE.json").is_file() else (
        "R2A" if (EVID / "R2A_GATE.json").is_file() else "R1")
    runs = runs_of(arm)
    per_seed = {(f"{r['market']}__{r['host']}", r["seed"]): r for r in runs}
    cells = sorted({(r["market"], r["host"]) for r in runs})

    # 1. coordinates exactly as registered
    seen: dict = {}
    for r in runs:
        k = (f"{r['market']}__{r['host']}", r["seed"])
        seen[k] = seen.get(k, 0) + 1
    dup = [f"{k[0]}/seed{k[1]}" for k, v in seen.items() if v > 1]
    bad_seed = sorted({r["seed"] for r in runs} - set(SEEDS))
    bad_host = sorted({r["host"] for r in runs} - set(HOSTS))
    bad_mkt = sorted({r["market"] for r in runs} - set(MARKETS))
    record("registered_coordinates", arm=arm, n_runs=len(runs), duplicates=dup,
           seeds_outside_registered=bad_seed, hosts_outside_registered=bad_host,
           markets_outside_registered=bad_mkt,
           _passed=not dup and not bad_seed and not bad_host and not bad_mkt)

    # 2. one registered recipe for every run -- no per-market recipe selection
    mismatches, readouts = [], set()
    for r in runs:
        o = r["optimization"]
        readouts.add(str(o.get("readout_init")))
        for k, want in REGISTERED_SCHEDULE.items():
            if o.get(k) != want:
                mismatches.append(
                    f"{r['market']}__{r['host']} seed{r['seed']}: {k}={o.get(k)!r} != {want!r}")
    record("registered_schedule", n_mismatches=len(mismatches), mismatches=mismatches[:10],
           distinct_readout_inits=sorted(readouts),
           _passed=not mismatches and readouts == {str(None)})

    # 3. selection is the registered running-best on EMA, not the raw trajectory
    viol, raw_explains = [], []
    for r in runs:
        cell = f"{r['market']}__{r['host']}"
        curve = curve_of(arm, cell, r["seed"])
        tag = f"{cell} seed{r['seed']}"
        if int(curve[0]["step"]) != 0 or not curve[0]["is_initial"]:
            viol.append(f"{tag}: curve does not begin with the step-0 checkpoint")
            continue
        ema_step, ema_val = replay_selection(curve, "val_mae_ema")
        raw_step, _ = replay_selection(curve, "val_mae_raw")
        if ema_step != int(r["selected_step"]):
            viol.append(f"{tag}: replay gives step {ema_step}, record says {r['selected_step']}")
        if abs(ema_val - float(r["selected_val_mae_ema"])) > 1e-9:
            viol.append(f"{tag}: replay value {ema_val} != recorded {r['selected_val_mae_ema']}")
        if abs(float(r["selected_metrics"]["val_mae_ema"]) - ema_val) > 1e-9:
            viol.append(f"{tag}: selected_metrics disagrees with the record")
        if raw_step != ema_step and int(r["selected_step"]) == raw_step:
            raw_explains.append(tag)
        if int(min(curve, key=lambda e: e["val_mae_ema"])["step"]) != ema_step:
            viol.append(f"{tag}: plain argmin would pick a different step than the rule")
    record("selection_rule", rule="running-best on val_mae_ema, strict improvement > 1e-7",
           n_violations=len(viol), violations=viol[:10],
           runs_explained_by_raw_trajectory=raw_explains,
           _passed=not viol and not raw_explains)

    # 4. the step schedule was actually obeyed
    issues = []
    for r in runs:
        cell = f"{r['market']}__{r['host']}"
        curve = curve_of(arm, cell, r["seed"])
        steps = [int(e["step"]) for e in curve]
        tag = f"{cell} seed{r['seed']}"
        if steps[0] != 0 or any(b - a != 50 for a, b in zip(steps, steps[1:])):
            issues.append(f"{tag}: check grid is not 0,50,100,... ({steps[:4]}...)")
        if steps[-1] > 2000:
            issues.append(f"{tag}: ran past the 2000-step ceiling")
        # Stop rule: patience counts CHECKS since the best checkpoint, so the first
        # legal stop is max(no_stop_before, best_step + patience_checks * val_every).
        stopped = r["optimization"]["stopped_at_step"]
        best = int(r["selected_step"])
        if stopped is None:
            if steps[-1] != 2000:
                issues.append(f"{tag}: ended at {steps[-1]} without stopping or reaching 2000")
        else:
            floor = max(800, best + 400)
            if stopped % 50 or not (floor <= stopped <= 2000):
                issues.append(f"{tag}: stopped at {stopped}, but the registered rule "
                              f"permits the first stop at {floor}")
        if best > int(r["optimization"]["max_steps"]):
            issues.append(f"{tag}: selected step beyond the ceiling")
    record("schedule_obeyed", n_violations=len(issues), violations=issues[:10],
           _passed=not issues)

    # 5. TEST quarantine: runtime counters, then a static scan of the direct readers
    leak = []
    for r in runs:
        a = r.get("access_state", {})
        tag = f"{r['market']}__{r['host']} seed{r['seed']}"
        if r.get("test_target_read_count", 0) != 0:
            leak.append(f"{tag}: test_target_read_count != 0")
        if r.get("test_rows_materialised", 0) != 0:
            leak.append(f"{tag}: test_rows_materialised != 0")
        if a.get("test_rows_returned", 0) != 0:
            leak.append(f"{tag}: a TEST row was returned to the trainer")
        if a.get("distinct_paths_blocked"):
            leak.append(f"{tag}: opened a quarantined table {a['distinct_paths_blocked']}")
        if a.get("test_role_frame_refusals", 0) != 0:
            leak.append(f"{tag}: asked for a TEST role frame {a['test_role_frame_refusals']}x")
    direct = re.compile(r"\bU\.role_frame\s*\(|\bU\.joint_arrays\s*\(|\bU\.SEAL_STATE\b"
                        r"|\bscore_test\s*\(|(?<!pretest_)\brole_frame\s*\(")
    static = {}
    for f in sorted(IMPL.glob("*.py")):
        hits = [m.group(0).strip() for m in direct.finditer(f.read_text(encoding="utf-8"))]
        if hits:
            static[f.name] = sorted(set(hits))
    joint = sorted({f.name for f in sorted(IMPL.glob("*.py"))
                    if "host_predictions_joint" in f.read_text(encoding="utf-8")})
    audit = json.loads((EVID / "ACCESS_AUDIT.json").read_text(encoding="utf-8"))
    record("test_quarantine", n_runtime_violations=len(leak), runtime_violations=leak[:10],
           direct_reader_call_sites=static, files_opening_the_joint_cache=joint,
           audit_test_target_read_count=audit.get("test_target_read_count"),
           _passed=(not leak and not static and joint == ["recovery_common.py"]
                    and audit.get("test_target_read_count") == 0))

    # 6. TRAIN/VAL-only row identity, rebuilt from the causal rules
    day_viol, host_viol = [], []
    for m, h in cells:
        rb = rebuild_rows(m, h)
        tag0 = f"{m}__{h}"
        if rb["test_days_in_train"] or rb["test_days_in_val"]:
            day_viol.append(f"{tag0}: a TEST day entered a legal row set")
        for s in SEEDS:
            r = per_seed.get((tag0, s))
            if r is None:
                continue
            tag = f"{tag0} seed{s}"
            if r["train_day_ids_hash"] != rb["train_day_ids_hash"]:
                day_viol.append(f"{tag}: TRAIN day-id hash mismatch")
            if r["val_day_ids_hash"] != rb["val_day_ids_hash"]:
                day_viol.append(f"{tag}: VAL day-id hash mismatch")
            if int(r["n_train_rows"]) != rb["n_train_rows"] or int(r["n_val_rows"]) != rb["n_val_rows"]:
                day_viol.append(f"{tag}: row counts {r['n_train_rows']}/{r['n_val_rows']} "
                                f"!= rebuilt {rb['n_train_rows']}/{rb['n_val_rows']}")
            hm = float(r["selected_metrics"]["val_host_mae"])
            if abs(hm - rb["host_val_mae"]) / rb["host_val_mae"] > 5e-4:
                host_viol.append(f"{tag}: val_host_mae {hm} != rebuilt {rb['host_val_mae']}")
    record("train_val_identity", n_violations=len(day_viol), violations=day_viol[:10],
           host_mae_mismatches=host_viol[:5],
           rule="W=7 window [d-8,d-2]; VAL residuals revealed prequentially; TEST never a row",
           _passed=not day_viol and not host_viol)

    # 7. reused shadow support hashes still match the completed legal stage
    manifest = json.loads((EVID / "REUSED_SUPPORT_HASHES.json").read_text(encoding="utf-8"))
    sup = manifest["shadow_support"]
    sup_viol = []
    for key, rec in sup.items():
        if sha256(PRIOR_EVID / "shadow_oof" / f"{key}.npz") != rec["npz_sha256"]:
            sup_viol.append(f"{key}: shadow npz has drifted")
        if rec.get("blocks_used_val_segment"):
            sup_viol.append(f"{key}: support blocks consumed the VAL segment")
    for r in runs:
        key = f"{r['market']}__{r['host']}"
        if key in sup and r["history_support_hash"] != sup[key]["support_hash"]:
            sup_viol.append(f"{key} seed{r['seed']}: history support hash mismatch")
    record("reused_support", n_cells=len(sup), n_violations=len(sup_viol),
           violations=sup_viol[:10], _passed=not sup_viol)

    # 8. src/core identity, proved against the digest the completed stage recorded
    mine = source_tree_digest()
    prior_digests = {json.loads(p.read_text(encoding="utf-8"))["core_tree_digest"]
                     for p in PRIOR_EVID.glob("primary/*/seed*/freeze.json")}
    run_digests = {r["source_tree_digest"]["core_tree"] for r in runs}
    record("core_unchanged", independent_core_tree_digest=mine["core_tree"],
           prior_stage_core_tree_digests=sorted(prior_digests),
           recovery_run_digests=sorted(run_digests),
           prior_matches_independent=prior_digests == {mine["core_tree"]},
           recovery_matches_independent=run_digests == {mine["core_tree"]},
           _passed=(len(prior_digests) == 1 and prior_digests == run_digests
                    == {mine["core_tree"]}))

    # 9. checkpoints present and hashed exactly as recorded
    ck = []
    for r in runs:
        p = (EVID / f"{arm.lower()}_runs" / f"{r['market']}__{r['host']}"
             / f"seed{r['seed']}" / "selected_ema.pt")
        if not p.is_file():
            ck.append(f"{r['market']}__{r['host']} seed{r['seed']}: checkpoint missing")
        elif sha256(p) != r["selected_ema_checkpoint_sha256"]:
            ck.append(f"{r['market']}__{r['host']} seed{r['seed']}: checkpoint hash drift")
    record("checkpoints", n_violations=len(ck), violations=ck[:10], _passed=not ck)

    # 10. promotion-gate arithmetic, recomputed from the raw per-seed numbers
    def prior_val(cell: str, seed: int) -> float:
        p = PRIOR_EVID / "primary" / cell / f"seed{seed}" / "freeze.json"
        return float(json.loads(p.read_text(encoding="utf-8"))["selected_val_mae_ema"])

    table = []
    for m, h in cells:
        cell = f"{m}__{h}"
        rs = [per_seed[(cell, s)] for s in SEEDS if (cell, s) in per_seed]
        rec_med = float(st.median([r["selected_val_mae_ema"] for r in rs]))
        old_med = float(st.median([prior_val(cell, s) for s in SEEDS]))
        host_med = float(st.median([r["selected_metrics"]["val_host_mae"] for r in rs]))
        table.append({
            "cell": cell,
            "improvement_vs_frozen_pct": (old_med - rec_med) / old_med * 100.0,
            "gain_vs_host_pct_recovery": (host_med - rec_med) / host_med * 100.0,
            "gain_vs_host_pct_frozen": (host_med - old_med) / host_med * 100.0,
            "n_seeds_step_gt0_and_ratio_ge_0.05": sum(
                1 for r in rs if int(r["selected_step"]) > 0
                and float(r["selected_metrics"]["val_correction_ratio"]) >= 0.05),
            "all_finite": all(
                np.isfinite([r["selected_metrics"][k] for k in (
                    "val_mae_ema", "val_mae_raw", "val_host_mae", "val_correction_ratio")]).all()
                for r in rs),
        })
    imps = [c["improvement_vs_frozen_pct"] for c in table]
    panel = [c for c in table if c["cell"] in R1_PANEL]
    panel_imps = [c["improvement_vs_frozen_pct"] for c in panel]
    clean = not leak and not static
    finite = all(c["all_finite"] for c in table)

    if len(table) == 20:
        scope, gate_name = imps, ("FULL_PANEL_GATE.json" if arm == "R1" else f"{arm}_GATE.json")
        recomputed = {
            "gate_1_at_least_15_of_20_non_worse": sum(1 for v in imps if v >= 0) >= 15,
            "gate_2_at_least_12_of_20_improve_ge_0.5pct": sum(1 for v in imps if v >= 0.5) >= 12,
            "gate_3_panel_median_gain_vs_host_gt_0_and_materially_above_frozen":
                float(st.median([c["gain_vs_host_pct_recovery"] for c in table])) > 0
                and float(st.median([c["gain_vs_host_pct_recovery"] for c in table]))
                >= float(st.median([c["gain_vs_host_pct_frozen"] for c in table])) + 0.5,
            "gate_4_no_market_median_degradation_worse_than_1.0pct": min(imps) >= -1.0,
            "gate_5_no_leakage_or_numerical_failure": clean and finite,
        }
        ok_token, no_token = TOKEN_FULL_OK, TOKEN_FULL_NO
        want_token = ok_token if all(recomputed.values()) else no_token
    else:
        scope, gate_name = panel_imps, ("R1_GATE.json" if arm == "R1" else f"{arm}_GATE.json")
        recomputed = {
            "gate_1_at_least_6_of_8_cells_improve": sum(1 for v in panel_imps if v > 0) >= 6,
            "gate_2_panel_median_improvement_ge_0.5pct": float(st.median(panel_imps)) >= 0.5,
            "gate_3_at_least_5_of_8_step_gt0_and_ratio_ge_0.05":
                sum(1 for c in panel if c["n_seeds_step_gt0_and_ratio_ge_0.05"] >= 2) >= 5,
            "gate_4_worst_degradation_le_1.0pct": min(panel_imps) >= -1.0,
            "gate_5_no_leakage_or_numerical_failure": clean and finite,
        }
        want_token = ("HCH_V44_RECOVERY_R1_PANEL_PROMOTED" if all(recomputed.values())
                      else "HCH_V44_RECOVERY_R1_PANEL_NOT_PROMOTED")

    stored = json.loads((EVID / gate_name).read_text(encoding="utf-8"))
    record("gate_arithmetic", gate_file=gate_name, scope=("full_20_cell" if len(table) == 20
                                                          else "r1_panel_8_cell"), arm=arm,
           n_cells=len(table), stored=stored["checks"], recomputed=recomputed,
           stored_passed=stored["passed"], recomputed_passed=all(recomputed.values()),
           stored_token=stored.get("token"), expected_token=want_token,
           median_improvement_pct=float(st.median(scope)),
           _passed=(stored["checks"] == recomputed
                    and stored["passed"] == all(recomputed.values())
                    and stored.get("token") == want_token))

    # 11. no rescue path was reachable
    rescue = [f"{r['market']}__{r['host']} seed{r['seed']}" for r in runs
              if r.get("rescue_used") or r.get("optional_features_active")]
    record("no_rescue", n_rescue_runs=len(rescue), runs=rescue[:10], _passed=not rescue)

    # 12. the STAGE terminal token is one of the three registered ones and follows
    #     from the gate results -- intermediate promotion tokens must not leak out
    freeze = json.loads((EVID / "RECIPE_FREEZE.json").read_text(encoding="utf-8"))
    token = freeze.get("terminal_token")
    trigger_path = EVID / "R2_TRIGGER.json"
    trigger = (json.loads(trigger_path.read_text(encoding="utf-8"))
               if trigger_path.is_file() else None)
    if len(table) == 20:
        expected = TOKEN_FULL_OK if stored["passed"] else TOKEN_FULL_NO
    else:
        expected = (TOKEN_FULL_OK if stored["passed"]
                    else TOKEN_FULL_NO if trigger is not None and not trigger["satisfied"]
                    else None)
    allowed = {TOKEN_FULL_OK, TOKEN_FULL_NO}
    token_ok = token in allowed or (token or "").startswith(
        "HCH_V44_OPTIMIZATION_RECOVERY_BLOCKED_")
    record("terminal_token", token=token, expected=expected,
           route=freeze.get("route_to_terminal_token"),
           r2_trigger_satisfied=(trigger or {}).get("satisfied"),
           is_registered_token=token_ok,
           not_a_sota_claim=freeze.get("not_a_sota_claim"),
           _passed=token_ok and (expected is None or token == expected)
                   and freeze.get("not_a_sota_claim") is True)

    payload = {
        "schema": "hch_v44_optimization_recovery_verification.v1",
        "protocol_id": "HCH_V44_OPTIMIZATION_RECOVERY_20260917",
        "arm": arm,
        "verifier_imports_runner": False,
        "n_checks": len(checks),
        "failed_checks": [k for k, v in checks.items() if not v["passed"]],
        "checks": checks,
        "all_passed": all(v["passed"] for v in checks.values()),
    }
    payload["verdict"] = ("HCH_V44_RECOVERY_INDEPENDENTLY_VERIFIED" if payload["all_passed"]
                          else "HCH_V44_RECOVERY_VERIFICATION_FAILED")
    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / "INDEPENDENT_VERIFICATION_REPORT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "arm": arm,
                      "n_checks": payload["n_checks"],
                      "failed": payload["failed_checks"]}, ensure_ascii=False, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
