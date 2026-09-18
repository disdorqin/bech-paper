"""Phase driver for the repairability-spectrum diagnostic.

    python run_all.py r0        # exact per-day repairability atlas
    python run_all.py summary   # cell / market-host headroom + taxonomy inputs
    python run_all.py r1        # legal descriptors + Host-residual analogue
    python run_all.py r2        # five-fold leave-one-market-out Ridge probes
    python run_all.py gate      # diagnosis, gate A-E, RESULTS.md, token
    python run_all.py all

Every phase reads only frozen artifacts (the 60 O1 runs, the frozen TRAIN/VAL
joint cache, the shadow-OOF support).  No phase trains anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import repair_common as R  # noqa: E402


def phase_r0() -> dict:
    import repair_r0 as M

    before = R.run_snapshot()
    provenance = R.reuse_provenance()
    R.json_dump(R.EVID / "REUSE_PROVENANCE.json", provenance)
    if not provenance["all_present"]:
        raise RuntimeError(f"frozen run set incomplete: {provenance['missing'][:5]}")

    out = M.run()
    M.write_parquet(R.EVID / "PER_DAY_REPAIRABILITY.parquet", out["rows"])
    recon = M.check_reconciliation(out["reconciliation"])
    after = R.run_snapshot()
    recon["run_snapshot_unchanged"] = before == after
    recon["n_run_files_hashed"] = len(before)
    recon["rows_written"] = len(out["rows"])
    recon["n_days_with_partial_hours"] = sum(1 for r in out["rows"] if r["n_valid_hours"] != R.HORIZON)
    recon["max_decoder_gap"] = max(r["decoder_gap"] for r in out["rows"])
    recon["n_alpha_zero_direction"] = sum(1 for r in out["rows"] if r["alpha_status"] != "ok")
    R.json_dump(R.EVID / "R0_RECONCILIATION.json", recon)
    if not recon["passed"]:
        raise RuntimeError("R0 reconciliation FAILED -- refusing to continue")
    if not recon["run_snapshot_unchanged"]:
        raise RuntimeError("a frozen O1 run changed bytes during R0")
    return recon


def phase_summary() -> dict:
    import repair_summary as M

    return M.run()


def phase_r1() -> dict:
    import repair_r1 as M

    return M.run()


def phase_r2() -> dict:
    import repair_r2 as M

    return M.run()


def phase_gate() -> dict:
    import repair_gate as M

    return M.run()


#: Artifact each phase leaves behind, so the audit can state which phases have
#: run in this stage even when they ran in an earlier invocation of this driver.
PHASE_ARTIFACTS = (
    ("r0", "PER_DAY_REPAIRABILITY.parquet"),
    ("summary", "CELL_HEADROOM_SUMMARY.csv"),
    ("r1", "HOST_RESIDUAL_ANALOGUE.csv"),
    ("r2", "LOMO_PROBE_RESULTS.csv"),
    ("gate", "DIAGNOSIS_BY_CELL.csv"),
)


def phase_access(phases: list[str]) -> dict:
    evidenced = [name for name, art in PHASE_ARTIFACTS if (R.EVID / art).exists()]
    return R.RC.write_access_audit(
        R.EVID / "ACCESS_AUDIT.json",
        extra={
            "stage": "hch_v44_repairability_spectrum_20260918",
            "phases_executed_this_invocation": phases,
            "phases_evidenced_in_this_stage": evidenced,
            "new_neural_candidates": 0,
            "optimizer_steps": 0,
            "new_fits": 0,
            "o1_runs_reused": 60,
            "o1_checkpoints_mutated": 0,
            "src_core_edited": False,
            "paper_edited": False,
            "foreign_fits": 0,
            "foreign_preflight": 0,
            "test_reads": R.RC.ACCESS_STATE.get("test_role_frame_refusals", 0) * 0
            + R.RC.ACCESS_STATE.get("test_rows_returned", 0),
            "forbidden_guard_hits": R.RC.ACCESS_STATE.get("forbidden_guard_hits", 0),
            "note": (
                "R0/R1/R2 read only the frozen TRAIN/VAL frames, the shadow-OOF support and the "
                "60 selected EMA checkpoints under no_grad; the stage takes no optimizer step."
            ),
        },
    )


PHASES = {
    "r0": phase_r0, "summary": phase_summary, "r1": phase_r1, "r2": phase_r2, "gate": phase_gate,
}


def main(argv: list[str]) -> int:
    requested = argv[1:] or ["all"]
    if requested == ["all"]:
        requested = ["r0", "summary", "r1", "r2", "gate"]
    done = []
    for name in requested:
        if name not in PHASES:
            raise SystemExit(f"unknown phase {name!r}; choose from {sorted(PHASES)}")
        print(f"[run_all] phase {name} ...", flush=True)
        PHASES[name]()
        done.append(name)
    audit = phase_access(done)
    # The per-read guard counters live under ``access_state``; the top level carries
    # the stage-level declarations (zero TEST reads, zero optimizer steps, ...).
    state = audit["access_state"]
    print(json.dumps({k: state[k] for k in ("test_rows_returned", "test_role_frame_refusals",
                                            "forbidden_guard_hits", "test_rows_dropped_before_return")}
                     | {"test_reads": audit["test_reads"]}, indent=1))
    print(f"[run_all] phases completed: {done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
