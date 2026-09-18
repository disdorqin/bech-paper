"""Phase I — zero-fit international preflight, plus the fits it gates.

Only reachable when D0-D4 all pass; ``run_all.py`` enforces that.  The preflight
opens no target at all -- it resolves identities in the v4.4 information setting
and re-reads the five frozen 2026-09-11 transfer authority files.

PROTOCOL.md section 5.2 is explicit that if legal O1 history support is absent
and cannot be reconstructed from already-authorized pre-eval support without
changing the information setting, Phase I stops with an explicit blocker rather
than inventing a workaround.  This module therefore treats "absent support" as a
first-class outcome and never substitutes a proxy baseline or retrains one.
"""
from __future__ import annotations

import hashlib
import json
import sys

import guardrail_common as G

U = G.U
PT = G.PT
EVID = G.EVID
TRANSFER = G.TRANSFER_EVID

#: The five files PROTOCOL.md section 5.2 names as the comparison authority.
AUTHORITY_FILES = (
    "baseline_admission_matrix.csv", "best_baseline_gap_by_cell.csv",
    "metrics_by_cell.csv", "provenance.json", "transfer_fidelity_audit.md",
)

V2 = U.V2


def _authority_pins() -> dict:
    pins = {}
    for name in AUTHORITY_FILES:
        p = TRANSFER / name
        pins[name] = {
            "present": p.is_file(),
            "sha256": G.sha256_file(p) if p.is_file() else None,
            "bytes": p.stat().st_size if p.is_file() else None,
        }
    return pins


def _rows(path, key_fields) -> list[dict]:
    lines = (TRANSFER / path).read_text(encoding="utf-8").strip().splitlines()
    head = lines[0].split(",")
    out = []
    for line in lines[1:]:
        cells, cur, quoted = [], "", False
        for ch in line:
            if ch == '"':
                quoted = not quoted
            elif ch == "," and not quoted:
                cells.append(cur)
                cur = ""
            else:
                cur += ch
        cells.append(cur)
        row = dict(zip(head, cells))
        out.append({k: row.get(k) for k in key_fields})
    return out


def _split_plan_probe(market: str) -> dict:
    try:
        plan = U.split_plan(market)
    except Exception as exc:  # noqa: BLE001 - the failure mode is the finding
        return {"resolved": False, "roles": None, "error": f"{type(exc).__name__}: {exc}"}
    roles = None
    for attr in ("roles", "ROLES", "role_names"):
        obj = getattr(plan, attr, None)
        if obj is None:
            continue
        try:
            roles = sorted(obj() if callable(obj) else obj)
        except Exception:  # noqa: BLE001
            roles = None
        if roles:
            break
    return {"resolved": True, "roles": roles, "error": None,
            "plan_api": [a for a in dir(plan) if not a.startswith("_")]}


def preflight() -> dict:
    admission = _rows("baseline_admission_matrix.csv",
                      ("method", "role", "admitted_to_best_baseline", "fidelity_label",
                       "source_commit", "reason"))
    best = _rows("best_baseline_gap_by_cell.csv",
                 ("market", "host", "HCH_MAE", "Host_MAE", "best_baseline",
                  "best_baseline_MAE", "gap_to_best_baseline_pct", "hch_gain_vs_host_pct"))
    metrics_host = _rows("metrics_by_cell.csv", ("market", "host", "method", "Overall_MAE",
                                                 "Normal_MAE", "fidelity", "source"))

    cells = []
    for market, host in G.INTL_CELLS:
        key = G.cell_key(market, host)
        shadow_npz = U.EVID / "shadow_oof" / f"{key}.npz"
        shadow_js = U.EVID / "shadow_oof" / f"{key}.json"
        joint = V2 / "02_hosts" / market / host / "host_predictions_joint.npz"
        b = next((r for r in best if r["market"] == market and r["host"] == host), None)
        h = next((r for r in metrics_host
                  if r["market"] == market and r["host"] == host and r["method"] == "Host"), None)
        cells.append({
            "market": market, "host": host, "cell": key,
            "in_v44_market_universe": market in U.MARKETS,
            "v44_split_plan": _split_plan_probe(market),
            "v44_frozen_host_joint_cache": {
                "path": str(joint.relative_to(G.REPO)).replace("\\", "/"),
                "present": joint.is_file(),
            },
            "v44_legal_o1_history_support": {
                "npz_present": shadow_npz.is_file(), "json_present": shadow_js.is_file(),
                "recipe_required": U.SHADOW_RECIPE,
            },
            "o1_legal_evidence_complete": bool(
                joint.is_file() and shadow_npz.is_file() and shadow_js.is_file()),
            "frozen_transfer_host_point_estimate": (
                None if h is None else {"Overall_MAE": h["Overall_MAE"],
                                        "Normal_MAE": h["Normal_MAE"],
                                        "fidelity": h["fidelity"], "source": h["source"]}),
            "frozen_best_admitted_baseline": (
                None if b is None else {"best_baseline": b["best_baseline"],
                                        "best_baseline_MAE": b["best_baseline_MAE"],
                                        "recorded_gap_pct": b["gap_to_best_baseline_pct"]}),
        })

    admissible = [r for r in admission if r["admitted_to_best_baseline"] == "True"]
    blocked = [r for r in admission if r["admitted_to_best_baseline"] == "True"
               and "FORBIDDEN" in (r["fidelity_label"] or "")]
    complete = [c for c in cells if c["o1_legal_evidence_complete"]]

    payload = {
        "schema": "hch_v44_o1_domestic20_international_preflight.v1",
        "protocol_id": G.PROTOCOL_ID,
        "phase": "I",
        "n_new_fits_authorized_by_this_preflight": 0,
        "fits_executed": 0,
        "authority_root": str(TRANSFER.relative_to(G.REPO)).replace("\\", "/"),
        "authority_file_pins": _authority_pins(),
        "authority_honoured": {
            "files_read": list(AUTHORITY_FILES),
            "no_proxy_baseline_substituted": True,
            "no_baseline_retrained": True,
            "admitted_baseline_methods": sorted({r["method"] for r in admissible}),
            "forbidden_methods_carried_as_forbidden": sorted(
                {r["method"] for r in admission if "FORBIDDEN" in (r["fidelity_label"] or "")}),
            "admitted_but_forbidden_labelled": [r["method"] for r in blocked],
        },
        "information_setting": {
            "v44_protocol": "COMMON_BENCHMARK_701020_FULL_V2",
            "v44_history_recipe": U.SHADOW_RECIPE,
            "v44_market_universe": list(U.MARKETS),
            "transfer_role_system": "S1 / DIAG_FIT / DIAG_EVAL (2026-09-11 protocol)",
            "commensurable": False,
            "commensurability_note": (
                "The frozen baseline and Host point estimates are DIAG_EVAL-role Overall/Normal MAE "
                "under the 2026-09-11 transfer protocol, while O1 produces a VAL-role MAE under "
                "COMMON_BENCHMARK_701020_FULL_V2. The two are not like-for-like, so even with legal "
                "support an O1-vs-baseline gap would need an explicit role reconciliation rather "
                "than a direct subtraction."),
        },
        "cells": cells,
        "n_cells_with_complete_o1_legal_evidence": len(complete),
        "protected_or_final_target_reads": 0,
        "test_target_read_count": 0,
        "blocked": len(complete) < len(G.INTL_CELLS),
    }

    if payload["blocked"]:
        missing = [c["cell"] for c in cells if not c["o1_legal_evidence_complete"]]
        payload["blocker"] = {
            "reason_code": "NO_V44_LEGAL_O1_HISTORY_SUPPORT_FOR_LAGO",
            "cells_missing_legal_evidence": missing,
            "detail": (
                "The v4.4 information setting has no LAGO split plan, no frozen Host joint cache and "
                "no V44_SHADOW_OOF_PREFIX_90_10_V1 support for any of the four registered foreign "
                "cells. Reconstructing that support would define a new information setting, which "
                "PROTOCOL.md section 5.2 forbids; a workaround, proxy baseline or baseline retrain "
                "is likewise forbidden, so no international fit is executed."),
        }
    U.json_dump(EVID / "INTERNATIONAL_PREFLIGHT.json", payload)
    return payload


def run_fits(workers: int = 2) -> dict:
    """The 12 registered international fits.  Unreachable while the preflight blocks."""
    if not preflight()["cells"]:
        raise RuntimeError("PREFLIGHT_EMPTY")
    if preflight()["blocked"]:
        raise RuntimeError("INTERNATIONAL_PREFLIGHT_BLOCKED")
    from guardrail_runner import fit_task  # pragma: no cover - blocked path
    return {"fits": [fit_task((m, h, s)) for m, h in G.INTL_CELLS for s in G.SEEDS]}


if __name__ == "__main__":
    G.RC.install_access_guard()
    p = preflight()
    print(json.dumps({"blocked": p["blocked"],
                      "n_complete": p["n_cells_with_complete_o1_legal_evidence"],
                      "blocker": p.get("blocker", {}).get("reason_code")}, indent=2))
    sys.exit(0)
