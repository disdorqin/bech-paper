"""Runner for the HCH O1 matched-control closure stage.

Execution order, one explicit step at a time:

    source            E0  source digest parity, 24 reuse pins, variant parity, access
    phase_a  [n]      E1  24 G1_GEOM_SHARED_CONTEXT fits under the O1 schedule
    gate_a            E1  seed-median-first aggregation and the A1/A2/A3 verdict
    phase_b  [n]      E2  24 matched direct-control fits (only if A1 or A2 held)
    gate_b            E2  seed-median-first aggregation and the B1/B2 verdict
    access            E3  access/quarantine audit over every run of the stage
    finalize          E3  RESULTS.md and the single terminal token

Every fit is produced by ``matched_train.train_matched``, which differs from the
closed ``probe_train.train_o1`` in exactly the three ways that module documents.
Nothing here can alter the trainer, the data path or the registered schedule.

There is no TEST path in this module.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matched_common as MC
import matched_train as MT

U = MC.U
PC = MC.PC
RC = MC.RC
RT = MC.RT
EVID = MC.EVID

PHASE_A = "g1"
PHASE_B = "direct"

G1_FIELDS = (
    "market", "host", "cell", "seed", "variant",
    "mae_g1", "mae_g2", "gain_g2_vs_g1_pct",
    "host_mae_g1", "host_mae_g2", "g1_host_gain_pct", "g2_host_gain_pct",
    "selected_step_g1", "selected_step_g2", "stopped_at_step_g1", "stopped_at_step_g2",
    "step0_val_mae_g1", "step0_val_mae_g2", "step0_identity_exact",
    "g1_selected_check", "g1_check_count",
    "history_support_hash_g1", "history_support_hash_g2", "support_matched",
    "n_train_rows", "n_val_rows", "data_identity_matched", "scales_fingerprint_matched",
    "reuse_freeze_sha256", "g1_checkpoint_sha256", "g1_parameter_hash",
    "probe_code_hash", "matched_code_hash", "core_tree", "test_target_read_count",
)

DIRECT_FIELDS = (
    "market", "host", "cell", "seed", "variant", "struct_variant",
    "mae_struct", "mae_direct", "gain_struct_vs_direct_pct",
    "host_mae_struct", "host_mae_direct", "struct_host_gain_pct", "direct_host_gain_pct",
    "selected_step_struct", "selected_step_direct",
    "stopped_at_step_struct", "stopped_at_step_direct",
    "direct_l_rec_only_throughout", "direct_selected_check", "direct_check_count",
    "history_support_hash_struct", "history_support_hash_direct", "support_matched",
    "n_train_rows", "n_val_rows", "data_identity_matched",
    "direct_checkpoint_sha256", "direct_parameter_hash",
    "probe_code_hash", "matched_code_hash", "core_tree", "test_target_read_count",
)

CELL_A_FIELDS = (
    "cell", "market", "host", "n_seeds",
    "mae_g2_median", "mae_g1_median", "gain_g2_vs_g1_pct", "per_seed_gain_pct",
    "g2_beats_g1", "g2_host_positive", "g1_host_positive",
    "host_mae", "g2_host_gain_pct", "g1_host_gain_pct",
    "selected_step_g2_median", "selected_step_g1_median",
)

CELL_B_FIELDS = (
    "cell", "market", "host", "n_seeds",
    "mae_struct_median", "mae_direct_median", "gain_struct_vs_direct_pct",
    "per_seed_gain_pct", "struct_beats_direct", "struct_host_positive",
    "direct_host_positive", "host_mae", "struct_host_gain_pct", "direct_host_gain_pct",
    "selected_step_struct_median", "selected_step_direct_median",
)

CURVE_FIELDS = (
    "cell", "seed", "variant", "check", "step", "val_mae_ema", "val_mae_raw",
    "train_mae_ema", "lr", "L_rec", "L_b", "L_B", "L_S", "L_total_frozen", "aux_mean",
    "used_full_loss_at_this_step", "loss_schedule", "selected",
)


# ------------------------------------------------------------------ helpers
def _init_worker() -> None:
    """Re-assert this stage's implementation directory on a spawned child's path.

    The inherited stages prepend their own implementation directories to
    ``sys.path``; this stage's modules are named distinctively and the directory
    is re-asserted explicitly rather than inferred from the spawn handshake.
    """
    impl = str(Path(__file__).resolve().parent)
    if impl not in sys.path:
        sys.path.insert(0, impl)


def _run(phase: str, market: str, host: str, seed: int) -> Path:
    return MC.new_run_dir(phase, market, host, seed)


def _complete(d: Path) -> bool:
    return all((d / f).is_file() for f in MC.RUN_FIELDS)


def _load(d: Path) -> tuple:
    return MC.load_json(d / "freeze.json"), MC.load_json(d / "training_curve.json")


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _check0(curve) -> float:
    return float(next(e for e in curve if int(e["step"]) == 0)["val_mae_ema"])


def _registered_schedule_ok(rec: dict) -> bool:
    return PC.registered_schedule_of(rec["optimization"]) == PC.registered_recipe()


def _digest_ok(rec: dict) -> bool:
    d = rec.get("source_tree_digest") or {}
    return (d.get("core_tree") == MC.FROZEN_CORE_DIGEST
            and d.get("backbones_tree") == MC.FROZEN_BACKBONES_DIGEST)


# ------------------------------------------------------------------ E0
def phase_source() -> dict:
    """E0 -- source and reuse audit.  Blocks before any fit if parity fails."""
    RC.install_access_guard()
    RC.worker_env()

    digest = RC.source_tree_digest()
    core_ok = digest.get("core_tree") == MC.FROZEN_CORE_DIGEST
    backbone_ok = digest.get("backbones_tree") == MC.FROZEN_BACKBONES_DIGEST

    core_files = sorted(
        p.relative_to(RC.REPO).as_posix()
        for p in (RC.REPO / "src/core").rglob("*")
        if p.is_file() and p.suffix in (".py", ".md"))
    backbone_files = sorted(
        p.relative_to(RC.REPO).as_posix()
        for p in (RC.REPO / "src/backbones").rglob("*")
        if p.is_file() and p.suffix in (".py", ".md"))

    audit = {
        "schema": "hch_v44_o1_matched_source_audit.v1",
        "protocol_id": MC.PROTOCOL_ID,
        "digest_rule": "sha256 over sorted 'relpath|sha256(file)' lines, .py and .md, "
                       "per tree; identical to recovery_common.source_tree_digest",
        "observed": digest,
        "frozen_expected": {"core_tree": MC.FROZEN_CORE_DIGEST,
                            "backbones_tree": MC.FROZEN_BACKBONES_DIGEST},
        "core_tree_matches_frozen_o1_g2": core_ok,
        "backbones_tree_matches_frozen_o1_g2": backbone_ok,
        "source_parity_pass": bool(core_ok and backbone_ok),
        "core_file_count": len(core_files), "core_files": core_files,
        "backbones_file_count": len(backbone_files), "backbones_files": backbone_files,
        "src_core_modified": not core_ok,
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    U.json_dump(EVID / "SOURCE_AUDIT.json", audit)
    if not audit["source_parity_pass"]:
        raise RuntimeError("HCH_O1_MATCHED_CONTROL_BLOCKED_SOURCE_PARITY")

    # --- the 24 reused O1/G2 runs, hash-pinned before any fit -----------------
    prov = MC.reuse_provenance()
    U.json_dump(EVID / "O1_G2_REUSE_PROVENANCE.json", prov)
    if not prov["all_reused_present"]:
        raise RuntimeError("HCH_O1_MATCHED_CONTROL_BLOCKED_REFERENCE_NOT_AVAILABLE")
    broken = [
        f"{k}__seed{s}"
        for k, cell in prov["reused_cells"].items()
        for s, run in cell["runs"].items()
        if not (run["checkpoint_sha256_matches_own_record"]
                and run["history_support_matches_registered"])
    ]
    if broken:
        raise RuntimeError("HCH_O1_MATCHED_CONTROL_BLOCKED_REUSE_PIN_MISMATCH")
    if not prov["one_probe_code_hash_across_reused"]:
        raise RuntimeError("HCH_O1_MATCHED_CONTROL_BLOCKED_REUSE_CODE_STATE_MISMATCH")
    if prov["n_reused_runs"] != len(MC.PANEL) * len(MC.SEEDS):
        raise RuntimeError("HCH_O1_MATCHED_CONTROL_BLOCKED_REUSE_COUNT_MISMATCH")

    # --- variant parity, on the real registered scales ------------------------
    parity = _variant_parity()
    U.json_dump(EVID / "VARIANT_PARITY.json", parity)
    if not parity["all_pass"]:
        raise RuntimeError("HCH_O1_MATCHED_CONTROL_BLOCKED_VARIANT_PARITY")

    MC.write_csv(EVID / "PARAMETER_COUNTS.csv", PARAM_FIELDS, parity["parameter_table"])

    # --- byte snapshot of every reused artifact, before the fits --------------
    snap = MC.reuse_snapshot()
    U.json_dump(EVID / "REUSE_SNAPSHOT_BEFORE.json",
                {"schema": "hch_v44_o1_matched_reuse_snapshot.v1",
                 "n_artifacts": len(snap), "sha256": snap})
    return {"source": audit, "reuse": prov, "parity": parity, "n_snapshot": len(snap)}


PARAM_FIELDS = ("variant", "role", "n_trainable_parameters", "ratio_to_g2",
                "within_1.25x_budget", "tied_identity", "structured_decoder",
                "n_named_parameter_tensors", "note")


def _variant_parity() -> dict:
    """G1/G2 exact equality, the only-tie delta, and the G3 <=1.25x budget.

    Beyond ``src/core.assert_variant_parity``'s parameter counts this checks the
    things that actually make the two runs matched:

    * G1 and G2 expose the *same* named parameter tensors with the same shapes;
    * constructed under the same seed they are **bitwise identical**, which is
      possible only because ``coord_embed`` is one shared ``(3, token_dim)``
      parameter and no construction order differs;
    * their forward outputs on one real registered batch are bitwise identical in
      ``eval`` mode -- the tie is exact at initialization, so any later
      divergence is caused by the released embeddings receiving distinct
      gradients, not by a different network;
    * ``G3_SHARED_DIRECT_CONTEXT_CTRL`` adds exactly zero parameters to G3.
    """
    import torch
    from core.model import HCHV44Core, assert_variant_parity
    from core.training_support import build_primary_train_config

    market, host = MC.PANEL[0]
    data = RT.build_cell_data(market, host)
    scales = data["scales"]
    # The registered primary profile, exactly as the trainer asserts it.
    profile = build_primary_train_config().profile

    def build(variant: str, seed: int = 1234):
        torch.manual_seed(seed)
        m = HCHV44Core(variant, scales, profile=profile)
        m.eval()
        return m

    g1, g2, g3 = build(MC.G1), build(MC.G2), build(MC.G3)
    torch.manual_seed(1234)
    g3s = HCHV44Core(MC.G3, scales, profile=profile)
    g3s.tied_identity = True
    g3s.eval()

    core = assert_variant_parity({MC.G1: g1, MC.G2: g2, MC.G3: g3})

    s1, s2 = g1.state_dict(), g2.state_dict()
    keys_equal = sorted(s1) == sorted(s2)
    bitwise = bool(keys_equal) and all(
        s1[k].shape == s2[k].shape and torch.equal(s1[k], s2[k]) for k in s1)
    shapes_equal = bool(keys_equal) and all(s1[k].shape == s2[k].shape for k in s1)

    g3s_g3_keys_equal = sorted(g3s.state_dict()) == sorted(g3.state_dict())
    g3s_g3_bitwise = bool(g3s_g3_keys_equal) and all(
        torch.equal(g3s.state_dict()[k], g3.state_dict()[k]) for k in g3.state_dict())

    P_ = RT._bind_prior()
    idx = torch.as_tensor([0, 1, 2, 3], dtype=torch.long)
    batch = P_.select_batch(data["train_batch"], idx)
    with torch.no_grad():
        c1, c2 = g1(batch).correction, g2(batch).correction

    def count(m) -> int:
        return int(sum(p.numel() for p in m.parameters() if p.requires_grad))

    n = {name: count(m) for name, m in
         ((MC.G1, g1), (MC.G2, g2), (MC.G3, g3), (MC.G3_SHARED, g3s))}
    ratio = n[MC.G3] / max(n[MC.G2], 1)
    table = [
        {"variant": MC.G1, "role": "Phase-A candidate (shared context)",
         "n_trainable_parameters": n[MC.G1], "ratio_to_g2": n[MC.G1] / max(n[MC.G2], 1),
         "within_1.25x_budget": True, "tied_identity": True, "structured_decoder": True,
         "n_named_parameter_tensors": len(g1.state_dict()),
         "note": "parameter-for-parameter identical to G2"},
        {"variant": MC.G2, "role": "Phase-A baseline (reused, never retrained)",
         "n_trainable_parameters": n[MC.G2], "ratio_to_g2": 1.0,
         "within_1.25x_budget": True, "tied_identity": False, "structured_decoder": True,
         "n_named_parameter_tensors": len(g2.state_dict()),
         "note": "reference; fitted by the closed O1 stages"},
        {"variant": MC.G3, "role": "Phase-B direct control when G2 is selected",
         "n_trainable_parameters": n[MC.G3], "ratio_to_g2": ratio,
         "within_1.25x_budget": ratio <= 1.25, "tied_identity": False,
         "structured_decoder": False, "n_named_parameter_tensors": len(g3.state_dict()),
         "note": "shares encoders, allocator and the three router calls"},
        {"variant": MC.G3_SHARED, "role": "Phase-B direct control when G1 is selected",
         "n_trainable_parameters": n[MC.G3_SHARED], "ratio_to_g2": n[MC.G3_SHARED] / max(n[MC.G2], 1),
         "within_1.25x_budget": n[MC.G3_SHARED] / max(n[MC.G2], 1) <= 1.25,
         "tied_identity": True, "structured_decoder": False,
         "n_named_parameter_tensors": len(g3s.state_dict()),
         "note": "existing G3 model with tied_identity=True; adds no parameter"},
    ]
    g3_ratio = float(n[MC.G3]) / float(n[MC.G2])
    checks = {
        "g1_g2_equal_parameter_count": bool(core["checks"]["g1_g2_equal_parameter_count"]),
        "g1_g2_identical_named_parameter_keys": keys_equal,
        "g1_g2_identical_named_parameter_shapes": shapes_equal,
        "g1_g2_bitwise_identical_at_init": bitwise,
        "g1_g2_forward_bitwise_identical_at_init": bool(torch.equal(c1, c2)),
        "g1_g2_only_tied_identity_flag_differs":
            (g1.tied_identity is True and g2.tied_identity is False
             and g1.structured is True and g2.structured is True),
        "g3_within_budget": bool(core["checks"]["g3_within_budget"]),
        "g3_shared_adds_no_parameter_to_g3": bool(
            n[MC.G3_SHARED] == n[MC.G3] and g3s_g3_keys_equal and g3s_g3_bitwise),
        "g3_shared_only_tied_identity_flag_differs":
            (g3s.tied_identity is True and g3.tied_identity is False
             and g3s.structured is False and g3.structured is False),
    }
    return {
        "schema": "hch_v44_o1_matched_variant_parity.v1",
        "protocol_id": MC.PROTOCOL_ID,
        "scales_source_cell": U.cell_key(market, host),
        "scales_fingerprint": scales.fingerprint(),
        "init_seed": 1234,
        "core_assert_variant_parity": core,
        "g3_over_g2_ratio": g3_ratio,
        "g3_over_g2_ratio_le_1.25": g3_ratio <= 1.25,
        "checks": checks,
        "all_pass": all(checks.values()),
        "tie_mechanism": (
            "G1 and G2 differ only in MaskedSoftmaxAllocator.coordinate_vectors(tied="
            "self.tied_identity): tied returns coord_embed.mean(dim=0).expand(3,-1), "
            "released returns coord_embed. coord_embed is one (3, token_dim) parameter "
            "that starts at zero, so the tie is exact at initialization and only the "
            "released variant can break it. No other module reads tied_identity."),
        "parameter_table": table,
    }


# ------------------------------------------------------------------ E1 Phase A
def fit_task_a(args) -> dict:
    market, host, seed = args
    out = _run(PHASE_A, market, host, seed)
    if _complete(out):
        return {"cell": MC.cell_key(market, host), "seed": seed, "status": "skipped_existing"}
    MT.train_matched(market, host, seed, out, variant=MC.G1, device="cuda")
    return {"cell": MC.cell_key(market, host), "seed": seed, "status": "fitted"}


def _fit_phase(phase: str, variant: str, task, workers: int, log_name: str) -> list[dict]:
    todo = [(m, h, s) for m, h in MC.PANEL for s in MC.SEEDS]
    results = []
    if workers > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_init_worker) as pool:
            for res in pool.map(task, todo):
                results.append(res)
                print(f"  {phase} fit {res['cell']} seed{res['seed']}: {res['status']}", flush=True)
    else:
        for args in todo:
            res = task(args)
            results.append(res)
            print(f"  {phase} fit {res['cell']} seed{res['seed']}: {res['status']}", flush=True)
    U.json_dump(EVID / log_name, {
        "schema": "hch_v44_o1_matched_fit_log.v1",
        "protocol_id": MC.PROTOCOL_ID,
        "phase": phase, "variant": variant,
        "n_registered_new_fits": len(todo),
        "n_fitted": sum(1 for r in results if r["status"] == "fitted"),
        "n_skipped_existing": sum(1 for r in results if r["status"] == "skipped_existing"),
        "attempts_per_coordinate": {f"{r['cell']}__seed{r['seed']}": 1 for r in results},
        "no_retry": len(results) == len(todo),
        "workers": int(workers),
        "finished": dt.datetime.now(dt.timezone.utc).isoformat(),
        "results": results,
    })
    return results


def phase_a(workers: int = 2) -> list[dict]:
    return _fit_phase(PHASE_A, MC.G1, fit_task_a, workers, "FIT_LOG_PHASE_A.json")


def _phase_a_rows() -> list[dict]:
    rows = []
    for market, host in MC.PANEL:
        cell = MC.cell_key(market, host)
        for seed in MC.SEEDS:
            d1 = _run(PHASE_A, market, host, seed)
            d2 = MC.reused_run_dir(market, host, seed)
            if not _complete(d1):
                raise RuntimeError(f"{cell}__seed{seed}: G1 run incomplete")
            r1, c1 = _load(d1)
            r2, c2 = _load(d2)
            mae_g1, mae_g2 = float(r1["selected_val_mae_ema"]), float(r2["selected_val_mae_ema"])
            s1, s2 = r1["selected_metrics"], r2["selected_metrics"]
            data_identity = all([
                r1["n_train_rows"] == r2["n_train_rows"],
                r1["n_val_rows"] == r2["n_val_rows"],
                r1["train_day_ids_hash"] == r2["train_day_ids_hash"],
                r1["val_day_ids_hash"] == r2["val_day_ids_hash"],
            ])
            rows.append({
                "market": market, "host": host, "cell": cell, "seed": seed,
                "variant": MC.G1,
                "mae_struct": mae_g2, "mae_ctrl": mae_g1,
                "host_mae_struct": float(s2["val_host_mae"]),
                "host_mae_ctrl": float(s1["val_host_mae"]),
                "gain_pct": MC.gain_pct(mae_g2, mae_g1),
                "selected_step_struct": int(r2["selected_step"]),
                "selected_step_ctrl": int(r1["selected_step"]),
                # --- csv-only fields
                "mae_g1": mae_g1, "mae_g2": mae_g2,
                "gain_g2_vs_g1_pct": MC.gain_pct(mae_g2, mae_g1),
                "host_mae_g1": float(s1["val_host_mae"]), "host_mae_g2": float(s2["val_host_mae"]),
                "g1_host_gain_pct": MC.gain_pct(mae_g1, float(s1["val_host_mae"])),
                "g2_host_gain_pct": MC.gain_pct(mae_g2, float(s2["val_host_mae"])),
                "selected_step_g1": int(r1["selected_step"]),
                "selected_step_g2": int(r2["selected_step"]),
                "stopped_at_step_g1": r1["optimization"]["stopped_at_step"],
                "stopped_at_step_g2": r2["optimization"]["stopped_at_step"],
                "step0_val_mae_g1": _check0(c1), "step0_val_mae_g2": _check0(c2),
                "step0_identity_exact": bool(_check0(c1) == _check0(c2)),
                "g1_selected_check": int(r1["selected_check"]),
                "g1_check_count": int(r1["epoch_history_check_count"]),
                "history_support_hash_g1": r1["history_support_hash"],
                "history_support_hash_g2": r2["history_support_hash"],
                "support_matched": r1["history_support_hash"] == r2["history_support_hash"],
                "n_train_rows": r1["n_train_rows"], "n_val_rows": r1["n_val_rows"],
                "data_identity_matched": bool(data_identity),
                "scales_fingerprint_matched": (
                    r1["scales"]["fingerprint"] == r2["scales"]["fingerprint"]),
                "reuse_freeze_sha256": MC.sha256_file(d2 / "freeze.json"),
                "g1_checkpoint_sha256": MC.sha256_file(d1 / "selected_ema.pt"),
                "g1_parameter_hash": r1["selected_ema_parameter_hash"],
                "probe_code_hash": r1["probe_code_hash"],
                "matched_code_hash": r1["matched_code_hash"],
                "core_tree": (r1["source_tree_digest"] or {}).get("core_tree"),
                "test_target_read_count": r1["test_target_read_count"],
                # --- internal, used by the compliance audit
                "_run_dir": str(d1), "_record": r1, "_curve": c1,
                "_ref_record": r2, "_ref_curve": c2,
            })
    return rows


def _compliance_a(rows: list[dict], snapshot_after: dict) -> dict:
    recs = [r["_record"] for r in rows]
    curves = [r["_curve"] for r in rows]
    refs = [r["_ref_record"] for r in rows]
    facts = {
        "n_new_runs": len(rows),
        "access_clean": all(
            int(r["test_target_read_count"]) == 0 and int(r["test_rows_materialised"]) == 0
            for r in recs + refs),
        "all_finite": all(
            _finite(r["selected_val_mae_ema"])
            and all(_finite(e["val_mae_ema"]) for e in c)
            for r, c in zip(recs, curves)),
        "source_digest_frozen": all(_digest_ok(r) for r in recs + refs),
        "schedule_registered": all(_registered_schedule_ok(r) for r in recs + refs),
        "variant_as_registered": all(
            r["variant"] == MC.G1 and r["tied_identity"] is True
            and r["structured_decoder"] is True for r in recs),
        "reference_variant_is_g2": all(r["variant"] == MC.G2 for r in refs),
        "history_support_matched_to_reference": all(r["support_matched"] for r in rows),
        "data_identity_matched_to_reference": all(r["data_identity_matched"] for r in rows),
        "scales_fingerprint_matched_to_reference": all(
            r["scales_fingerprint_matched"] for r in rows),
        "step0_val_mae_bitwise_equal_to_reference": all(
            r["step0_identity_exact"] for r in rows),
        "reused_artifacts_unmodified": snapshot_after == MC.reuse_snapshot(),
        "loss_switched_after_400_not_before": all(
            all((bool(e["used_full_loss_at_this_step"]) if int(e["step"]) <= MC.SWITCH_STEP
                 else (not bool(e["used_full_loss_at_this_step"])))
                for e in c if e["used_full_loss_at_this_step"] is not None)
            for c in curves),
    }
    facts["clean"] = all(v for k, v in facts.items() if k != "n_new_runs")
    return facts


def gate_a() -> dict:
    rows = _phase_a_rows()
    snapshot_after = MC.reuse_snapshot()
    before = MC.load_json(EVID / "REUSE_SNAPSHOT_BEFORE.json")["sha256"] \
        if (EVID / "REUSE_SNAPSHOT_BEFORE.json").is_file() else {}
    compliance = _compliance_a(rows, snapshot_after)
    compliance["reused_artifacts_unmodified_vs_before_file"] = (snapshot_after == before)
    compliance["clean"] = bool(compliance["clean"]
                               and compliance["reused_artifacts_unmodified_vs_before_file"])
    cells = MC.cell_medians(rows)
    gate = MC.gate_a(rows, cells, compliance)
    MC.write_csv(EVID / "G1_PER_SEED.csv", G1_FIELDS, rows)
    MC.write_csv(EVID / "PHASE_A_CELL_MEDIANS.csv", CELL_A_FIELDS, [
        {**c, "mae_g2_median": c["mae_struct_median"], "mae_g1_median": c["mae_ctrl_median"],
         "gain_g2_vs_g1_pct": c["gain_pct_from_medians"],
         "g2_beats_g1": c["struct_beats_ctrl"], "g2_host_positive": c["struct_host_positive"],
         "g1_host_positive": c["ctrl_host_positive"],
         "host_mae": c["host_mae_struct_median"],
         "g2_host_gain_pct": c["struct_host_gain_pct"],
         "g1_host_gain_pct": c["ctrl_host_gain_pct"],
         "selected_step_g2_median": c["selected_step_struct_median"],
         "selected_step_g1_median": c["selected_step_ctrl_median"]}
        for c in cells])
    U.json_dump(EVID / "PHASE_A_GATE.json", gate)
    U.json_dump(EVID / "REUSE_SNAPSHOT_AFTER_PHASE_A.json",
                {"schema": "hch_v44_o1_matched_reuse_snapshot.v1",
                 "n_artifacts": len(snapshot_after), "sha256": snapshot_after})
    return gate


# ------------------------------------------------------------------ E2 Phase B
def fit_task_b(args) -> dict:
    market, host, seed, variant = args
    out = _run(PHASE_B, market, host, seed)
    if _complete(out):
        return {"cell": MC.cell_key(market, host), "seed": seed, "status": "skipped_existing"}
    MT.train_matched(market, host, seed, out, variant=variant, device="cuda")
    return {"cell": MC.cell_key(market, host), "seed": seed, "status": "fitted"}


def phase_b(direct_variant: str, workers: int = 2) -> list[dict]:
    if direct_variant not in (MC.G3, MC.G3_SHARED):
        raise ValueError(f"illegal direct control {direct_variant!r}")
    return _fit_phase(PHASE_B, direct_variant,
                      lambda a: fit_task_b((a[0], a[1], a[2], direct_variant)),
                      workers, "FIT_LOG_PHASE_B.json")


def _struct_run_for(selected: str, market: str, host: str, seed: int) -> tuple:
    if selected == MC.G2:
        d = MC.reused_run_dir(market, host, seed)
    else:
        d = _run(PHASE_A, market, host, seed)
    if not _complete(d):
        raise RuntimeError(f"{MC.cell_key(market, host)}__seed{seed}: structured run missing")
    return d, *_load(d)


def _phase_b_rows(selected: str, direct_variant: str) -> list[dict]:
    rows = []
    for market, host in MC.PANEL:
        cell = MC.cell_key(market, host)
        for seed in MC.SEEDS:
            ds, rs, cs = _struct_run_for(selected, market, host, seed)
            dd = _run(PHASE_B, market, host, seed)
            if not _complete(dd):
                raise RuntimeError(f"{cell}__seed{seed}: direct run incomplete")
            rd, cd = _load(dd)
            if rd["variant"] != direct_variant:
                raise RuntimeError(
                    f"{cell}__seed{seed}: direct run variant {rd['variant']!r} "
                    f"!= selected {direct_variant!r}")
            mae_s, mae_d = float(rs["selected_val_mae_ema"]), float(rd["selected_val_mae_ema"])
            ss, sd = rs["selected_metrics"], rd["selected_metrics"]
            rows.append({
                "market": market, "host": host, "cell": cell, "seed": seed,
                "variant": direct_variant,
                "mae_struct": mae_s, "mae_ctrl": mae_d,
                "host_mae_struct": float(ss["val_host_mae"]),
                "host_mae_ctrl": float(sd["val_host_mae"]),
                "gain_pct": MC.gain_pct(mae_s, mae_d),
                "selected_step_struct": int(rs["selected_step"]),
                "selected_step_ctrl": int(rd["selected_step"]),
                "struct_variant": selected,
                "mae_direct": mae_d,
                "gain_struct_vs_direct_pct": MC.gain_pct(mae_s, mae_d),
                "host_mae_direct": float(sd["val_host_mae"]),
                "direct_host_gain_pct": MC.gain_pct(mae_d, float(sd["val_host_mae"])),
                "selected_step_direct": int(rd["selected_step"]),
                "stopped_at_step_struct": rs["optimization"]["stopped_at_step"],
                "stopped_at_step_direct": rd["optimization"]["stopped_at_step"],
                "direct_l_rec_only_throughout": all(
                    e["L_b"] is None and e["L_B"] is None and e["L_S"] is None for e in cd),
                "direct_selected_check": int(rd["selected_check"]),
                "direct_check_count": int(rd["epoch_history_check_count"]),
                "history_support_hash_struct": rs["history_support_hash"],
                "history_support_hash_direct": rd["history_support_hash"],
                "support_matched": rs["history_support_hash"] == rd["history_support_hash"],
                "n_train_rows": rd["n_train_rows"], "n_val_rows": rd["n_val_rows"],
                "data_identity_matched": bool(all([
                    rd["n_train_rows"] == rs["n_train_rows"],
                    rd["n_val_rows"] == rs["n_val_rows"],
                    rd["train_day_ids_hash"] == rs["train_day_ids_hash"],
                    rd["val_day_ids_hash"] == rs["val_day_ids_hash"],
                ])),
                "direct_checkpoint_sha256": MC.sha256_file(dd / "selected_ema.pt"),
                "direct_parameter_hash": rd["selected_ema_parameter_hash"],
                "probe_code_hash": rd["probe_code_hash"],
                "matched_code_hash": rd["matched_code_hash"],
                "core_tree": (rd["source_tree_digest"] or {}).get("core_tree"),
                "test_target_read_count": rd["test_target_read_count"],
                "_run_dir": str(dd), "_record": rd, "_curve": cd,
            })
    return rows


def _compliance_b(rows: list[dict], direct_variant: str) -> dict:
    recs = [r["_record"] for r in rows]
    curves = [r["_curve"] for r in rows]
    facts = {
        "n_new_runs": len(rows),
        "access_clean": all(
            int(r["test_target_read_count"]) == 0 and int(r["test_rows_materialised"]) == 0
            for r in recs),
        "all_finite": all(
            _finite(r["selected_val_mae_ema"])
            and all(_finite(e["val_mae_ema"]) for e in c)
            for r, c in zip(recs, curves)),
        "source_digest_frozen": all(_digest_ok(r) for r in recs),
        "schedule_registered": all(_registered_schedule_ok(r) for r in recs),
        "direct_variant_as_registered": all(r["variant"] == direct_variant for r in recs),
        "direct_reconstruction_only_throughout": all(
            r["direct_l_rec_only_throughout"] for r in rows),
        "history_support_matched_to_structured": all(r["support_matched"] for r in rows),
        "data_identity_matched_to_structured": all(r["data_identity_matched"] for r in rows),
        "no_auxiliary_term_exists": all(
            r["objective_probe"]["loss_effective_for_direct_control"].startswith("L_rec")
            for r in recs),
    }
    facts["clean"] = all(v for k, v in facts.items() if k != "n_new_runs")
    return facts


def gate_b(selected: str, direct_variant: str) -> dict:
    rows = _phase_b_rows(selected, direct_variant)
    compliance = _compliance_b(rows, direct_variant)
    cells = MC.cell_medians(rows)
    gate = MC.gate_b(rows, cells, compliance, selected, direct_variant)
    MC.write_csv(EVID / "DIRECT_PER_SEED.csv", DIRECT_FIELDS, rows)
    MC.write_csv(EVID / "PHASE_B_CELL_MEDIANS.csv", CELL_B_FIELDS, [
        {**c, "mae_struct_median": c["mae_struct_median"],
         "mae_direct_median": c["mae_ctrl_median"],
         "gain_struct_vs_direct_pct": c["gain_pct_from_medians"],
         "struct_beats_direct": c["struct_beats_ctrl"],
         "direct_host_positive": c["ctrl_host_positive"],
         "host_mae": c["host_mae_struct_median"],
         "direct_host_gain_pct": c["ctrl_host_gain_pct"],
         "selected_step_direct_median": c["selected_step_ctrl_median"]}
        for c in cells])
    U.json_dump(EVID / "PHASE_B_GATE.json", gate)
    return gate


# ------------------------------------------------------------------ shared curves
def write_curves() -> int:
    rows = []
    for phase, variant in ((PHASE_A, MC.G1), (PHASE_B, None)):
        for market, host in MC.PANEL:
            cell = MC.cell_key(market, host)
            for seed in MC.SEEDS:
                d = _run(phase, market, host, seed)
                if not _complete(d):
                    continue
                rec, curve = _load(d)
                for e in curve:
                    rows.append({
                        "cell": cell, "seed": seed, "variant": rec["variant"],
                        **{k: e.get(k) for k in CURVE_FIELDS if k in e},
                    })
    MC.write_csv(EVID / "TRAINING_CURVES.csv", CURVE_FIELDS, rows)
    return len(rows)


# ------------------------------------------------------------------ E3 access
def phase_access() -> dict:
    runs, per_variant = [], {}
    total_reads = 0
    for phase in (PHASE_A, PHASE_B):
        for market, host in MC.PANEL:
            for seed in MC.SEEDS:
                d = _run(phase, market, host, seed)
                if not _complete(d):
                    continue
                rec, curve = _load(d)
                n = int(rec["test_target_read_count"])
                total_reads += n
                per_variant.setdefault(rec["variant"], 0)
                per_variant[rec["variant"]] += 1
                runs.append({
                    "cell": MC.cell_key(market, host), "seed": seed,
                    "variant": rec["variant"], "phase": phase,
                    "test_target_read_count": n,
                    "test_rows_materialised": rec["test_rows_materialised"],
                    "test_rows_dropped_from_joint_cache":
                        rec["test_rows_dropped_from_joint_cache"],
                    "access_state": rec["access_state"],
                    "core_tree": (rec["source_tree_digest"] or {}).get("core_tree"),
                    "backbones_tree": (rec["source_tree_digest"] or {}).get("backbones_tree"),
                    "matched_code_hash": rec["matched_code_hash"],
                })
    reused = MC.reuse_provenance()
    reused_reads = [
        run["test_target_read_count"]
        for cell in reused["reused_cells"].values() for run in cell["runs"].values()]
    snap_before = (MC.load_json(EVID / "REUSE_SNAPSHOT_BEFORE.json")["sha256"]
                   if (EVID / "REUSE_SNAPSHOT_BEFORE.json").is_file() else {})
    snap_after = MC.reuse_snapshot()
    audit = {
        "schema": "hch_v44_o1_matched_access_audit.v1",
        "protocol_id": MC.PROTOCOL_ID,
        "n_new_runs": len(runs), "runs_by_variant": per_variant,
        "test_target_read_count_total_new": total_reads,
        "test_target_read_count_max_new": max([r["test_target_read_count"] for r in runs] or [0]),
        "all_new_runs_zero_test_reads": all(r["test_target_read_count"] == 0 for r in runs),
        "test_target_read_count_reused": reused_reads,
        "all_reused_runs_zero_test_reads": all(int(v) == 0 for v in reused_reads),
        "test_rows_materialised_total": sum(r["test_rows_materialised"] for r in runs),
        "test_rows_dropped_before_return_total":
            sum(r["test_rows_dropped_from_joint_cache"] for r in runs),
        "access_states_observed": sorted({
            json.dumps(r["access_state"], sort_keys=True) for r in runs}),
        "reused_artifacts_unmodified": snap_after == snap_before,
        "n_reused_artifacts": len(snap_after),
        "runs": runs,
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    U.json_dump(EVID / "ACCESS_AUDIT.json", audit)
    return audit


# ------------------------------------------------------------------ CLI
def main(argv: list[str]) -> int:
    """One explicit step per invocation; no step is implied by another."""
    if not argv:
        print(__doc__)
        return 2
    step, rest = argv[0], argv[1:]
    if step == "source":
        out = phase_source()
        a = out["source"]
        print(f"source parity: core={a['core_tree_matches_frozen_o1_g2']} "
              f"backbones={a['backbones_tree_matches_frozen_o1_g2']}")
        p = out["parity"]["checks"]
        print(json.dumps(p, indent=2))
        print(f"reused runs pinned: {out['reuse']['n_reused_runs']} "
              f"(one probe_code_hash: {out['reuse']['one_probe_code_hash_across_reused']})")
        return 0
    if step == "phase_a":
        res = phase_a(int(rest[0]) if rest else 2)
        return 0 if res else 1
    if step == "gate_a":
        g = gate_a()
        print(json.dumps({"verdict": g["verdict"],
                          "selected": g["selected_structured_variant"],
                          "a1_checks": g["a1_checks"],
                          "a2_checks": g["a2_checks"]}, indent=2))
        return 0
    if step == "phase_b":
        selected = MC.load_json(EVID / "PHASE_A_GATE.json")["selected_structured_variant"]
        if selected is None:
            raise RuntimeError("PHASE_B_BLOCKED: Phase A selected no structured variant")
        variant = MC.G3 if selected == MC.G2 else MC.G3_SHARED
        print(f"Phase A selected {selected}; direct control = {variant}")
        phase_b(variant, int(rest[0]) if rest else 2)
        return 0
    if step == "gate_b":
        selected = MC.load_json(EVID / "PHASE_A_GATE.json")["selected_structured_variant"]
        if selected is None:
            raise RuntimeError("PHASE_B_BLOCKED: Phase A selected no structured variant")
        variant = MC.G3 if selected == MC.G2 else MC.G3_SHARED
        g = gate_b(selected, variant)
        print(json.dumps({"verdict": g["verdict"], "b1_checks": g["b1_checks"]}, indent=2))
        return 0
    if step == "curves":
        print(f"curve rows written: {write_curves()}")
        return 0
    if step == "access":
        a = phase_access()
        print(json.dumps({k: v for k, v in a.items()
                          if k not in ("runs", "access_states_observed")}, indent=2))
        return 0
    raise SystemExit(f"unknown step {step!r}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main(sys.argv[1:]))


__all__ = [name for name in dir() if not name.startswith("_")]
