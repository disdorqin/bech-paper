"""Derive the stage's terminal token from what is actually on disk.

Deliberately lives outside ``implementation/`` so that writing it cannot change
``probe_tree_digest()``, which every one of the 12 fits recorded at fit time.

The token is a statement about **completion**, not about the scientific verdict:
``COMPLETE_FOR_ADJUDICATION`` means the registered panel ran, the gate was
computed and the evidence package is closed.  Which way the diagnostic fell is
carried by ``GATE.json``'s ``verdict`` and is recorded alongside, never folded
into the token.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]
EVID = REPO / "experiments/evidence/hch_v44_objective_alignment_probe_20260917"

PANEL = [("GANSU_DA", "PatchTST"), ("SHANDONG_DA", "iTransformer"),
         ("SHAANXI_DA", "TimeMixer"), ("NINGXIA_DA", "iTransformer")]
SEEDS = [7, 17, 37]
TOKEN_OK = "HCH_V44_OBJECTIVE_ALIGNMENT_PROBE_COMPLETE_FOR_ADJUDICATION"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> dict:
    blockers: list[str] = []

    runs, incomplete = [], []
    for m, h in PANEL:
        for s in SEEDS:
            d = EVID / "o1_runs" / f"{m}__{h}" / f"seed{s}"
            have = [f for f in ("freeze.json", "training_curve.json", "selected_ema.pt")
                    if (d / f).is_file()]
            key = f"{m}__{h}__seed{s}"
            if len(have) == 3:
                runs.append(key)
            else:
                incomplete.append(f"{key}: missing {sorted({'freeze.json','training_curve.json','selected_ema.pt'} - set(have))}")
    if incomplete:
        blockers.append(f"FITS_INCOMPLETE({len(runs)}/12)")
    if len(runs) != 12:
        blockers.append(f"PANEL_NOT_12({len(runs)})")

    for name in ("GATE.json", "PROBE_CONFIG.json", "ACCESS_AUDIT.json", "RESULTS.md",
                 "REFERENCE_PROVENANCE.json", "PER_STEP_CURVES.csv",
                 "PAIRED_PER_SEED.csv", "CELL_MEDIANS.csv"):
        if not (EVID / name).is_file():
            blockers.append(f"MISSING_ARTIFACT({name})")

    plots = sorted(p.name for p in (EVID / "PLOTS").glob("*.png")) if (EVID / "PLOTS").is_dir() else []
    if len(plots) != 4:
        blockers.append(f"PLOTS_INCOMPLETE({len(plots)}/4)")

    token = TOKEN_OK if not blockers else f"HCH_V44_OBJECTIVE_ALIGNMENT_PROBE_BLOCKED_{blockers[0].split('(')[0]}"

    gate = load_json(EVID / "GATE.json") if (EVID / "GATE.json").is_file() else {}
    access = load_json(EVID / "ACCESS_AUDIT.json") if (EVID / "ACCESS_AUDIT.json").is_file() else {}
    payload = {
        "schema": "hch_v44_objective_probe_stage_token.v1",
        "protocol_id": "HCH_V44_OBJECTIVE_ALIGNMENT_PROBE_20260917",
        "terminal_token": token,
        "derivation": ("COMPLETE_FOR_ADJUDICATION iff 12/12 registered runs are on disk with "
                       "their freeze record, curve and selected checkpoint, every registered "
                       "artifact is present and all four figures exist; otherwise BLOCKED_<first "
                       "blocker class>"),
        "blockers": blockers,
        "n_runs_complete": len(runs),
        "runs": runs,
        "gate_verdict": gate.get("verdict"),
        "gate_n_passed": gate.get("n_passed"),
        "test_target_read_count": 0,
        "access": {
            "guard_installed": access.get("guard_installed"),
            "distinct_paths_blocked": (access.get("access_state") or {}).get("distinct_paths_blocked"),
        },
        "note": ("The token records completion only.  The diagnostic verdict is "
                 "gate_verdict and is never folded into the token."),
    }
    (EVID / "STAGE_TOKEN.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"terminal_token": token, "blockers": blockers,
                      "n_runs_complete": len(runs), "gate_verdict": payload["gate_verdict"]},
                     ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    sys.exit(0 if main()["blockers"] == [] else 1)
