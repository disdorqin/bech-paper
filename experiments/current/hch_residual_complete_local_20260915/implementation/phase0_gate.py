"""The Phase-0 gate record: what was checked, and what each check could see.

Phase 0 is the last boundary before any scientific fit.  Three of its conditions
cannot be established by a test that passes — they are *absences*:

* **scientific fit count = 0** — established by the absence of every artifact a fit
  would have produced, not by a counter a fit could have incremented.
* **TEST target read count = 0** — established by the loader's structure (only
  TRAIN/VAL rows are ever materialized) and by the boundary scanners, not by a
  number the run reports about itself.
* **``src/core`` unchanged** — established by a manifest of per-file SHA-256 plus
  the filesystem mtime, taken before the round and recomputed in Phase 2.

**On the digest.**  ``tree_digest`` is defined here and used unchanged by Phase 2;
the comparison is only meaningful between two evaluations of *this* function.  An
earlier ad-hoc one-liner quoted a different value for the same tree over the same
12 files, and no reformulation of it reproduced that value, so it is not carried
forward as a baseline — a digest whose recipe is lost is not evidence.  What the
recipe cannot be checked against is supplied independently by the mtime: every file
under ``src/core`` was last written at 2026-09-15 21:48:19, before the final-patch
round opened at 22:39, and no file carries a later one.

This module reads no dataset, builds no model and fits nothing.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["ROOT", "SRC_CORE", "EVIDENCE_ROOT", "GATE_PATH", "AUDIT_ITEMS",
           "GATE_SUITES", "core_manifest", "tree_digest", "gate_record",
           "write_gate"]

ROOT = Path(__file__).resolve().parents[4]
SRC_CORE = ROOT / "src/core"
EVIDENCE_ROOT = ROOT / "experiments/evidence/hch_residual_complete_local_v4_1_20260915"
GATE_PATH = EVIDENCE_ROOT / "00_protocol/PHASE0_GATE.json"

#: The audit's A-H list, as the record's own statement of scope.  A patch item not
#: named here is out of scope by construction.
AUDIT_ITEMS = (
    "A_stage1_oof_and_final_normalization_fitted_from_fit_ids_only",
    "B_level_loss_uses_an_independent_s_b_not_the_raw_residual_scale",
    "C_stage2_seeds_before_build_candidate",
    "D_final_level_keeps_three_frozen_refits_and_val_uses_the_oof_median",
    "E_b1_is_causal_q_history_only_with_b2_b3_b4_targets",
    "F_r2_trains_on_native_signed_mass_coordinates",
    "G_r0_parity_r3_rho_history_q_history_level_condition_no_dead_parameters",
    "H_val_metrics_arithmetic_in_residual_error_space",
)

#: The suites the gate is decided on.  ``test_package_equivalence`` is deliberately
#: absent: it is a pre-RCL script that imports the retired ``hch_v2`` package and
#: cannot run at all, so it is not a gate suite and its non-collection is not a pass.
GATE_SUITES = {
    "core_rcl_model": "experiments/foundation/tests/test_core_rcl_model.py",
    "core_rcl_losses": "experiments/foundation/tests/test_core_rcl_losses.py",
    "core_rcl_targets": "experiments/foundation/tests/test_core_rcl_targets.py",
    "core_rcl_masked_horizon":
        "experiments/foundation/tests/test_core_rcl_masked_horizon.py",
    "core_rcl_preprocessing":
        "experiments/foundation/tests/test_core_rcl_preprocessing.py",
    "core_rcl_purity": "experiments/foundation/tests/test_core_rcl_purity.py",
    "core_experiment_parity":
        "experiments/foundation/tests/test_rcl_core_experiment_parity.py",
    "execution_contract":
        "experiments/foundation/tests/test_rcl_execution_contract.py",
    "execution_harness": "experiments/foundation/tests/test_rcl_execution_harness.py",
    "final_patch_adversarial":
        "experiments/foundation/tests/test_rcl_v4_1_final_patch.py",
}

#: Suites that exist in the tree but are outside the gate, with the reason.  Kept in
#: the record so that "the gate ran 10 suites" cannot be read as "the tree has 10".
NON_GATE_SUITES = {
    "test_package_equivalence":
        "pre-RCL script comparing flat legacy modules against the retired hch_v2 "
        "package; ModuleNotFoundError: hch_v2 on import, so it cannot run and "
        "collects no pytest tests",
    "test_action_lattice, test_action_shield, test_action_value_boundaries":
        "legacy suites that abort collection with ModuleNotFoundError: archive",
}


def core_manifest(root: Path = SRC_CORE) -> list[dict]:
    """Every ``.py`` under ``src/core`` with its digest and mtime."""
    base = Path(root)
    out = []
    for path in sorted(base.rglob("*.py")):
        stat = path.stat()
        out.append({
            "path": path.relative_to(base.parent.parent).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mtime": stat.st_mtime,
        })
    return out


def tree_digest(manifest) -> str:
    """A digest of the manifest's *content*, independent of mtime and of order.

    Deliberately not a digest over the file bytes in directory order: the claim is
    that the same set of file contents is present, and a manifest that changed only
    by a touch must not read as a modified core.
    """
    payload = "\n".join(f"{item['path']}:{item['sha256']}"
                        for item in sorted(manifest, key=lambda i: i["path"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def gate_record(*, test_counts: dict, src_core: list[dict],
                fits_completed: int = 0, test_target_reads: int = 0,
                evidence_root_present: bool = False,
                non_gate_suites: dict | None = None) -> dict:
    """The gate's own record.

    ``fits_completed`` and ``test_target_reads`` are not counters the run
    maintains — they are the values the round is required to leave at zero, and the
    record carries them so Phase 2 checks the claim against the artifacts rather
    than against this file.
    """
    return {
        "schema": "rcl_v4_1_phase0_gate.v1",
        "audit_items": list(AUDIT_ITEMS),
        "gate_suites": dict(test_counts),
        "gate_total_tests": int(sum(test_counts.values())),
        "gate_all_passed": all(int(v) > 0 for v in test_counts.values()),
        "non_gate_suites": dict(non_gate_suites if non_gate_suites is not None
                                else NON_GATE_SUITES),
        "scientific_fits_completed": int(fits_completed),
        "test_target_values_read": int(test_target_reads),
        "evidence_root_present_at_gate_time": bool(evidence_root_present),
        "src_core": {
            "n_files": len(src_core),
            "tree_digest": tree_digest(src_core),
            "files": {item["path"]: item["sha256"] for item in src_core},
            "latest_mtime": max((item["mtime"] for item in src_core), default=0.0),
        },
    }


def write_gate(record: dict, path: Path = GATE_PATH) -> str:
    """Write the record, refusing to overwrite a differing one."""
    from result_schema import write_json

    return write_json(path, record)
