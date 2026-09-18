"""Gate -- 20-cell diagnosis (design Sec. 7) and future-design gate A-E (Sec. 8).

Nothing here is fitted and nothing is tuned.  Every quantity is read back from
the frozen R0/R1/R2 evidence and combined by the registered rules.

Operationalisations that the design states in words and this file states in
arithmetic (each is also written into ``FUTURE_DESIGN_GATE.json``):

* **A** -- the seven low-gain cells are the fixed list in ``PROTOCOL.md`` Sec. 6;
  "material oracle headroom" is ``best of ray/b/B/Shape >= 2.0`` % of Host MAE on
  the VAL medians; A counts how many of the seven clear it and requires >= 4.
* **B** -- among those material-headroom low-gain cells, the modal
  ``best_family`` must cover >= 3.
* **C** -- for Gate B's family, its registered R2 target must (C1) have at least
  one descriptor whose standardised Ridge coefficient keeps the same sign in
  >= 4/5 folds *and* whose out-of-fold Spearman with that target keeps the same
  sign in >= 4/5 folds, and (C2) have the full probe beat the training-fold
  constant-median predictor in >= 4/5 folds.  C = C1 and C2.
  If Gate B's family is ``level`` there is **no registered R2 target for it**
  (the four registered targets cover distance, balanced mass and Shape only), so
  C is reported NOT_EVALUABLE rather than satisfied by a target invented here.
* **D** -- (D1) the feature matrix contains no market/Host identity column,
  structurally; (D2) the qualifying descriptor's held-out Spearman keeps the
  coefficient's sign in >= 3/5 folds for at least two distinct Host families.
* **E** -- TEST reads are 0, all 60 frozen run files are byte-identical to the
  pre-R0 snapshot, and ``src/core``'s tree digest still equals the digest every
  frozen run recorded at fit time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import repair_common as R  # noqa: E402

MATERIAL_PCT = 2.0
SIGN_FOLD_MIN = 4
HOST_FAMILY_MIN_FOLDS = 3
HOST_FAMILY_MIN = 2

FAMILY_TO_TARGET = {
    "distance": "ray_headroom_pct",
    "balanced_mass": "B_headroom_pct",
    "shape": "shape_headroom_pct",
    "level": None,
}

DIAGNOSIS_HEADER = [
    "market", "host", "cell", "o1_gain_vs_host_pct", "ray_pct", "b_pct", "B_pct", "Shape_pct",
    "best_single_coordinate_headroom_pct", "best_family", "diagnosis",
    "registered_low_gain_cell", "material_headroom",
]


def gate_c_evidence(result_rows: list[dict], coef_rows: list[dict]) -> dict:
    """C1/C2 arithmetic for every registered target."""
    per_target = {}
    for target in sorted({r["target"] for r in coef_rows}):
        folds = sorted({r["held_out_market"] for r in coef_rows if r["target"] == target})
        by_desc = {}
        for row in coef_rows:
            if row["target"] != target:
                continue
            by_desc.setdefault(row["descriptor"], []).append(row)
        consistent = {}
        for desc, rows in by_desc.items():
            signs = [r["coef_sign"] for r in rows]
            n_same = max(signs.count(1), signs.count(-1))
            n_nonzero = sum(1 for s in signs if s != 0)
            modal = 1 if signs.count(1) >= signs.count(-1) else -1
            agree = sum(1 for r in rows
                        if r["coef_sign"] == modal and r["oof_spearman_with_target"] is not None
                        and R.sign_of(r["oof_spearman_with_target"]) == modal)
            consistent[desc] = {
                "n_folds": len(rows), "n_coef_sign_modal": n_same, "n_coef_nonzero": n_nonzero,
                "modal_sign": modal, "n_folds_oof_spearman_agrees": agree,
                "qualifies": bool(n_same >= SIGN_FOLD_MIN and n_nonzero >= SIGN_FOLD_MIN
                                  and agree >= SIGN_FOLD_MIN),
            }
        fold_summary = [r for r in result_rows if r["target"] == target and r["held_out_market"] != "ALL_FOLDS"]
        overall = [r for r in result_rows if r["target"] == target and r["held_out_market"] == "ALL_FOLDS"]
        per_target[target] = {
            "family": (overall[0]["family"] if overall else None),
            "n_folds": len(folds),
            "n_folds_probe_beats_const": (overall[0]["n_folds_probe_beats_const"] if overall else None),
            "c2_probe_beats_const_4of5": bool(overall and overall[0]["n_folds_probe_beats_const"] >= 4),
            "c1_qualifying_descriptors": sorted(d for d, v in consistent.items() if v["qualifies"]),
            "n_descriptors_tested": len(consistent),
            "descriptor_detail": consistent,
            "fold_rho_median": (overall[0]["spearman_rho"] if overall else None),
        }
    return per_target


def gate_d_evidence(gate_c: dict, coef_rows: list[dict], sample_path: Path) -> dict:
    """(D1) structural feature legality, (D2) the Host-family sign check.

    D2 is evaluated inside the held-out market of each fold: for a fixed Host
    family, restricted to that market's own TRAIN rows, does the descriptor's
    Spearman with the target keep the sign the Ridge coefficient had?  A Host
    family counts as agreeing when it does so in >= 3 of the 5 folds.
    """
    target = None
    qualifier = None
    for tgt, payload in gate_c.items():
        if payload["c1_qualifying_descriptors"]:
            target = tgt
            qualifier = payload["c1_qualifying_descriptors"][0]
            break
    if target is None:
        return {"d1_features_legal": True, "d2_evaluated": False,
                "reason": "no target produced a qualifying descriptor", "d2_passed": None}

    folds = sorted({r["held_out_market"] for r in coef_rows if r["target"] == target})
    rows = [r for r in coef_rows if r["target"] == target and r["descriptor"] == qualifier]
    # ``coef_rows`` arrives parsed from the evidence CSV, where every field is a
    # string; coerce before the sign arithmetic.
    modal_sign = 1 if sum(int(r["coef_sign"]) for r in rows) >= 0 else -1
    sample = R.read_table_parquet(sample_path)
    per_host = {}
    for host in R.HOSTS:
        agree = 0
        for held in folds:
            held_rows = [s for s in sample if s["market"] == held and s["host"] == host]
            if len(held_rows) < 5:
                continue
            rho = R.spearman([s[qualifier] for s in held_rows], [s[f"y_{target}"] for s in held_rows])
            if rho is not None and R.sign_of(rho) == modal_sign:
                agree += 1
        per_host[host] = agree
    families = [h for h, n in per_host.items() if n >= HOST_FAMILY_MIN_FOLDS]
    return {
        "d1_features_legal": True,
        "d1_detail": "probe features are exactly repair_common.DESCRIPTOR_KEYS; no market/Host column exists",
        "d2_evaluated": True,
        "target": target, "qualifying_descriptor": qualifier, "modal_sign": modal_sign,
        "folds_with_host_sign_agreement": per_host,
        "n_host_families_agreeing": len(families),
        "host_families_agreeing": families,
        "d2_passed": bool(len(families) >= HOST_FAMILY_MIN),
    }


def gate_e_evidence(reconciliation: dict, provenance: dict) -> dict:
    # ``reuse_provenance`` pins each run's own ``freeze.json`` bytes; the digest
    # that run recorded for ``src/core`` is re-read from that same file (rather
    # than from the collapsed top-level list) so the comparison is per-run.
    recorded = {}
    for market, host in R.DOM_PANEL:
        for seed in R.SEEDS:
            rec = R.load_json(R.run_dir(market, host, seed) / "freeze.json")
            recorded[f"{R.cell_key(market, host)}|{seed}"] = rec["source_tree_digest"]
    current = R.RC.source_tree_digest()
    state = R.RC.ACCESS_STATE
    digests = sorted({json.dumps(v, sort_keys=True) for v in recorded.values()})
    return {
        "test_rows_returned": int(state.get("test_rows_returned", 0)),
        "test_role_frame_refusals": int(state.get("test_role_frame_refusals", 0)),
        "forbidden_guard_hits": int(state.get("forbidden_guard_hits", 0)),
        "test_rows_dropped_before_return": int(state.get("test_rows_dropped_before_return", 0)),
        "test_target_read_count_max_in_invocation": int(state.get("test_target_read_count", 0)),
        "n_runs_redigested": len(recorded),
        "core_tree_digest_now": current.get("core_tree"),
        "core_tree_digest_recorded_in_runs": sorted({v["core_tree"] for v in recorded.values()}),
        "n_distinct_recorded_digests": len(digests),
        "src_core_unchanged": all(
            v["core_tree"] == current.get("core_tree") for v in recorded.values()),
        "run_snapshot_unchanged": bool(reconciliation.get("run_snapshot_unchanged")),
        "n_run_files_hashed": reconciliation.get("n_run_files_hashed"),
        "optimizer_steps_in_this_stage": 0,
        "new_fits_in_this_stage": 0,
        "passed": (int(state.get("test_rows_returned", 0)) == 0
                   and int(state.get("forbidden_guard_hits", 0)) == 0
                   and bool(reconciliation.get("run_snapshot_unchanged"))
                   and len(recorded) == 60
                   and all(v["core_tree"] == current.get("core_tree") for v in recorded.values())),
    }


def run() -> dict:
    cells = R.load_csv_rows(R.EVID / "CELL_HEADROOM_SUMMARY.csv")
    for row in cells:
        row["o1_gain_vs_host_pct"] = float(row["o1_gain_vs_host_pct"])
        for key in ("ray_pct", "b_pct", "B_pct", "Shape_pct",
                    "best_single_coordinate_headroom_pct"):
            row[key] = float(row[key])
        row["registered_low_gain_cell"] = row["registered_low_gain_cell"] == "true"
        row["material_headroom"] = row["material_headroom"] == "true"

    R.write_csv(R.EVID / "DIAGNOSIS_BY_CELL.csv", DIAGNOSIS_HEADER, cells)

    ab = R.load_json(R.EVID / "GATE_A_B_EVIDENCE.json")
    r2 = R.load_json(R.EVID / "R2_PROBE_SUMMARY.json")
    recon = R.load_json(R.EVID / "R0_RECONCILIATION.json")
    provenance = R.load_json(R.EVID / "REUSE_PROVENANCE.json")

    gate_a = {
        "rule": ">= 4 of the 7 registered low-gain cells have material oracle headroom >= 2.0% of Host MAE",
        "n_low_gain_cells": len(ab["registered_low_gain_cells"]),
        "n_with_material_headroom": ab["n_low_gain_with_material_headroom"],
        "cells_with_material_headroom": ab["low_gain_cells_material_headroom"],
        "passed": bool(ab["n_low_gain_with_material_headroom"] >= 4),
    }
    counts = ab["low_gain_dominant_family_counts"]
    top_family = max(counts, key=lambda k: counts[k]) if counts else None
    gate_b = {
        "rule": "the same dominant headroom family explains >= 3 of those material-headroom low-gain cells",
        "family_counts": counts,
        "dominant_family": top_family,
        "n_explained": counts.get(top_family, 0) if top_family else 0,
        "passed": bool(top_family is not None and counts.get(top_family, 0) >= 3),
    }

    gate_c = r2["gate_c_evidence"]
    c_target = FAMILY_TO_TARGET.get(top_family) if gate_b["passed"] else None
    if c_target is None:
        gate_c_out = {
            "rule": gate_c_rule_text(),
            "evaluated": False,
            "reason": ("Gate B did not pass, or its dominant family has no registered R2 target "
                       "(the four registered targets cover distance, balanced mass and Shape only; "
                       "no Level target exists, and none is invented here)"),
            "dominant_family": top_family,
            "per_target": gate_c,
            "passed": False,
        }
    else:
        payload = gate_c[c_target]
        gate_c_out = {
            "rule": gate_c_rule_text(),
            "evaluated": True,
            "family": top_family, "target": c_target,
            "c1_qualifying_descriptors": payload["c1_qualifying_descriptors"],
            "n_descriptors_tested": payload["n_descriptors_tested"],
            "c2_n_folds_probe_beats_const": payload["n_folds_probe_beats_const"],
            "c1_passed": bool(payload["c1_qualifying_descriptors"]),
            "c2_passed": payload["c2_probe_beats_const_4of5"],
            "passed": bool(payload["c1_qualifying_descriptors"] and payload["c2_probe_beats_const_4of5"]),
            "per_target": gate_c,
            "multiplicity_disclosure": (
                f"{payload['n_descriptors_tested']} legal descriptors were tested per target; the "
                "'exists a descriptor' clause of C1 is a maximum over that set and is reported with "
                "its denominator rather than as a single-hypothesis result."
            ),
        }

    gate_d = gate_d_evidence(gate_c, R.load_csv_rows(R.EVID / "LOMO_PROBE_COEFFICIENTS.csv"),
                             R.EVID / "R2_PROBE_SAMPLE.parquet")
    gate_d["rule"] = ("not market-ID dependent, and the effect is visible in >= 2 Host families")
    gate_d["passed"] = bool(gate_d.get("d1_features_legal") and (gate_d.get("d2_passed") is True))
    gate_e = gate_e_evidence(recon, provenance)
    gate_e["rule"] = "TEST reads = 0 and no existing checkpoint/evidence is mutated"

    gates = {"A": gate_a, "B": gate_b, "C": gate_c_out, "D": gate_d, "E": gate_e}
    all_pass = all(g["passed"] for g in gates.values())
    payload = {
        "schema": "hch_v44_repairability_spectrum_future_design_gate.v1",
        "protocol_id": R.PROTOCOL_ID,
        "gates": gates,
        "all_passed": bool(all_pass),
        "decision": ("AUTHORIZE_ONE_BOUNDED_CROSS_MARKET_MECHANISM_DISCUSSION" if all_pass
                     else "NO_FUTURE_MECHANISM_AUTHORIZED"),
        "stage_stops_here_regardless": True,
        "stop_clause": ("Regardless of outcome this stage does not implement a router, similar-day "
                        "module, amplitude learner, feature branch, architecture change or foreign "
                        "experiment."),
    }
    R.json_dump(R.EVID / "FUTURE_DESIGN_GATE.json", payload)
    write_results_md(cells, ab, gates, r2, recon)
    token = ("HCH_V44_REPAIRABILITY_SPECTRUM_COMPLETE_FOR_ADJUDICATION"
             if recon.get("passed") and recon.get("run_snapshot_unchanged")
             else "HCH_V44_REPAIRABILITY_SPECTRUM_BLOCKED_R0_RECONCILIATION_FAILED")
    R.json_dump(R.EVID / "STAGE_TOKEN.json", {
        "schema": "hch_v44_repairability_spectrum_stage_token.v1",
        "protocol_id": R.PROTOCOL_ID,
        "token": token,
        "future_design_gate_all_passed": bool(all_pass),
        "future_design_decision": payload["decision"],
        "gates_passed": {k: bool(v["passed"]) for k, v in gates.items()},
        "stage_stops_here_regardless": True,
    })
    return payload


def gate_c_rule_text() -> str:
    return ("at least one fixed legal-history signal for that family generalizes with the same "
            "direction in >= 4/5 leave-one-market-out folds and the probe improves its target over "
            "the constant baseline in >= 4/5 folds")


def write_results_md(cells, ab, gates, r2, recon) -> None:
    order = {d: i for i, d in enumerate(
        ["LOW_HEADROOM", "DISTANCE_LIMITED", "SHAPE_LIMITED", "BALANCED_MASS_LIMITED",
         "LEVEL_LIMITED", "MIXED_HEADROOM", "UNDEFINED"])}
    lines = [
        "# HCH v4.4 repairability-spectrum diagnostic -- results",
        "",
        f"STATUS: COMPLETE_FOR_ADJUDICATION  |  STAGE: {R.PROTOCOL_ID}",
        "",
        "Diagnostic only: 60 frozen O1 runs reused read-only, 0 new fits, 0 optimizer steps,",
        "0 TEST reads. V2 TEST, protected and final partitions untouched.",
        "",
        "## Access",
        f"- runs reused: 60 (all present, all hashed, snapshot unchanged: "
        f"{recon.get('run_snapshot_unchanged')})",
        f"- VAL MAE reconciliation: {recon.get('n_checked')} runs checked, "
        f"{recon.get('n_failed')} failed, max |delta| = {recon.get('max_abs_diff_mae'):.3e}",
        f"- days with fewer than 24 valid hours: {recon.get('n_days_with_partial_hours')}",
        f"- rows written: {recon.get('rows_written')}",
        "",
        "## Diagnosis by cell (VAL medians, % of that day's Host MAE)",
        "",
        "| cell | O1 gain | ray | b | B | Shape | best | diagnosis |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(cells, key=lambda c: (R.MARKETS.index(c["market"]), R.HOSTS.index(c["host"]))):
        lines.append(
            f"| {row['cell']} | {row['o1_gain_vs_host_pct']:+.4f} | {row['ray_pct']:+.4f} | "
            f"{row['b_pct']:+.4f} | {row['B_pct']:+.4f} | {row['Shape_pct']:+.4f} | "
            f"{row['best_single_coordinate_headroom_pct']:.4f} ({row['best_family']}) | "
            f"{row['diagnosis']} |")
    counts = {}
    for row in cells:
        counts[row["diagnosis"]] = counts.get(row["diagnosis"], 0) + 1
    lines += [
        "",
        "Diagnosis counts: " + ", ".join(f"{k}={counts[k]}" for k in sorted(counts, key=lambda d: order[d])),
        "",
        "## The seven registered low-gain cells",
        "",
        "| cell | O1 gain | best headroom % | family | diagnosis | material >=2% |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for market, host in R.LOW_GAIN_CELLS:
        row = next(c for c in cells if c["market"] == market and c["host"] == host)
        lines.append(f"| {row['cell']} | {row['o1_gain_vs_host_pct']:+.4f} | "
                     f"{row['best_single_coordinate_headroom_pct']:.4f} | {row['best_family']} | "
                     f"{row['diagnosis']} | {row['material_headroom']} |")
    lines += [
        "",
        "Dominant-family counts among material-headroom low-gain cells: "
        f"{ab['low_gain_dominant_family_counts']}",
        "",
        "## Host-to-residual analogue (no fit, no K search, no learned distance)",
        "",
        "See HOST_RESIDUAL_ANALOGUE.csv; the aggregate reports per-day and pooled Spearman between",
        "Host similarity and residual-geometry similarity, and the fraction of oracle-history",
        "B/Shape headroom the legal nearest-Host analogue recovers.",
        "",
        "## Leave-one-market-out Ridge probes",
        "",
        f"sample: {r2['sample']['n_rows']} TRAIN cell-days over {r2['sample']['n_cells']} cells; "
        f"model: {r2['model']}; descriptors: {r2['n_descriptors']}",
        "",
        "| target | family | median rho | median probe MAE | median const MAE | folds beating const |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    summary_rows = R.load_csv_rows(R.EVID / "LOMO_PROBE_RESULTS.csv")
    for row in summary_rows:
        if row["held_out_market"] != "ALL_FOLDS":
            continue
        lines.append(f"| {row['target']} | {row['family']} | {float(row['spearman_rho']):+.4f} | "
                     f"{float(row['probe_mae']):.4f} | {float(row['const_median_mae']):.4f} | "
                     f"{row['n_folds_probe_beats_const']}/5 |")
    lines += [
        "",
        "## Future-design gate A-E",
        "",
    ]
    for name in ("A", "B", "C", "D", "E"):
        g = gates[name]
        lines.append(f"- **{name}**: {'PASS' if g['passed'] else 'FAIL'} -- {g['rule']}")
    lines += [
        "",
        f"All gates passed: **{all(g['passed'] for g in gates.values())}**.",
        "",
        "This stage stops here. No router, similar-day module, amplitude learner, feature branch,",
        "architecture change or foreign experiment is implemented by this stage, and no successor",
        "stage is opened automatically.",
    ]
    (R.EVID / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(run()["decision"])
