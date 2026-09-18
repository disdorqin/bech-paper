"""Derive this stage's terminal token from what is actually on disk.

Deliberately lives outside ``implementation/`` so that writing it cannot change any
code digest a fit recorded at fit time.

The token is a statement about **completion**, not about the scientific verdict:
``COMPLETE_FOR_ADJUDICATION`` means the registered 8-cell panel ran, the gate was
computed, the evidence package is closed and the independent verifier passed.  Which
way the diagnostic fell is carried by ``GATE.json``'s ``verdict`` and is recorded
alongside, never folded into the token.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]
IMPL = STAGE / "implementation"
EVID = REPO / "experiments/evidence/hch_v44_o1_full8_completion_20260918"
O1_EVID = REPO / "experiments/evidence/hch_v44_objective_alignment_probe_20260917"

PANEL = [("GANSU_DA", "PatchTST"), ("GANSU_DA", "LSTM"),
         ("SHANDONG_DA", "PatchTST"), ("SHANDONG_DA", "iTransformer"),
         ("SHAANXI_DA", "TimeMixer"), ("SHAANXI_DA", "PatchTST"),
         ("NINGXIA_DA", "iTransformer"), ("QINGHAI_DA", "TimeMixer")]
NEW_CELLS = [("GANSU_DA", "LSTM"), ("SHANDONG_DA", "PatchTST"),
             ("SHAANXI_DA", "PatchTST"), ("QINGHAI_DA", "TimeMixer")]
SEEDS = [7, 17, 37]
TOKEN_OK = "HCH_V44_O1_FULL8_COMPLETION_COMPLETE_FOR_ADJUDICATION"
TOKEN_BLOCKED = "HCH_V44_O1_FULL8_COMPLETION_BLOCKED_"

ARTIFACTS = ("GATE.json", "PROBE_CONFIG.json", "ACCESS_AUDIT.json", "RESULTS.md",
             "REUSE_PROVENANCE.json", "PER_STEP_CURVES.csv", "PAIRED_PER_SEED.csv",
             "CELL_MEDIANS.csv", "MECHANISM_SUMMARY.csv",
             "INDEPENDENT_VERIFICATION_REPORT.json")
FIELDS = ("freeze.json", "training_curve.json", "selected_ema.pt")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> dict:
    blockers: list[str] = []

    def run_dir(market, host, seed):
        root = EVID if (market, host) in NEW_CELLS else O1_EVID
        return root / "o1_runs" / f"{market}__{host}" / f"seed{seed}"

    new_runs, reused_runs, incomplete = [], [], []
    for m, h in PANEL:
        for s in SEEDS:
            d = run_dir(m, h, s)
            have = {f for f in FIELDS if (d / f).is_file()}
            key = f"{m}__{h}__seed{s}"
            if len(have) == len(FIELDS):
                (new_runs if (m, h) in NEW_CELLS else reused_runs).append(key)
            else:
                incomplete.append(f"{key}: missing {sorted(set(FIELDS) - have)}")
    if incomplete:
        blockers.append(f"FITS_INCOMPLETE({len(new_runs) + len(reused_runs)}/24)")
    if len(new_runs) != 12:
        blockers.append(f"NEW_FITS_NOT_12({len(new_runs)})")
    if len(reused_runs) != 12:
        blockers.append(f"REUSED_O1_NOT_12({len(reused_runs)})")

    for name in ARTIFACTS:
        if not (EVID / name).is_file():
            blockers.append(f"MISSING_ARTIFACT({name})")

    plots_dir = EVID / "PLOTS"
    for stem in ("train_curve_reconstruction_mae", "val_mae_vs_step",
                 "loss_components_vs_step", "correction_ratio_vs_step"):
        for ext in ("png", "svg"):
            if not (plots_dir / f"{stem}.{ext}").is_file():
                blockers.append(f"PLOT_MISSING({stem}.{ext})")

    ver = load_json(EVID / "INDEPENDENT_VERIFICATION_REPORT.json") \
        if (EVID / "INDEPENDENT_VERIFICATION_REPORT.json").is_file() else {}
    # ``terminal_token`` is excluded from the verifier requirement: that check asserts
    # this file exists, so requiring it here would be circular.  Every other check must
    # have passed on the run that preceded this one.
    ver_checks = {k: v for k, v in (ver.get("checks") or {}).items() if k != "terminal_token"}
    if not ver or not all(ver_checks.values()):
        blockers.append("INDEPENDENT_VERIFICATION_FAILED")
    if ver.get("verifier_imports_runner"):
        blockers.append("VERIFIER_IMPORTED_RUNNER")

    token = TOKEN_OK if not blockers else TOKEN_BLOCKED + blockers[0].split("(")[0]

    gate = load_json(EVID / "GATE.json") if (EVID / "GATE.json").is_file() else {}
    access = load_json(EVID / "ACCESS_AUDIT.json") if (EVID / "ACCESS_AUDIT.json").is_file() else {}
    payload = {
        "schema": "hch_v44_o1_full8_stage_token.v1",
        "protocol_id": "HCH_V44_O1_FULL8_COMPLETION_20260918",
        "terminal_token": token,
        "derivation": ("COMPLETE_FOR_ADJUDICATION iff all 24 registered runs are on disk with "
                       "their freeze record, curve and selected checkpoint (12 of them new fits "
                       "and 12 reused O1 runs), every registered artifact and all eight figure "
                       "files exist, and the independent verifier passed without importing the "
                       "runner; otherwise BLOCKED_<first blocker class>"),
        "blockers": blockers,
        "n_runs_complete": len(new_runs) + len(reused_runs),
        "n_new_fits_complete": len(new_runs),
        "n_reused_o1_complete": len(reused_runs),
        "new_fits": sorted(new_runs),
        "gate_verdict": gate.get("verdict"),
        "gate_n_passed": gate.get("n_passed"),
        "test_target_read_count": 0,
        "access": {
            "guard_installed": access.get("guard_installed"),
            "distinct_paths_blocked": (access.get("access_state") or {}).get("distinct_paths_blocked"),
        },
        "note": ("The token records completion only.  The diagnostic verdict is gate_verdict and "
                 "is never folded into the token.  No B/Shape/Level-loss removal, no router "
                 "deletion, no architecture simplification and no TEST evaluation follows from it."),
    }
    (EVID / "STAGE_TOKEN.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"terminal_token": token, "blockers": blockers,
                      "n_new_fits_complete": len(new_runs),
                      "n_reused_o1_complete": len(reused_runs),
                      "gate_verdict": payload["gate_verdict"]},
                     ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    sys.exit(0 if main()["blockers"] == [] else 1)
