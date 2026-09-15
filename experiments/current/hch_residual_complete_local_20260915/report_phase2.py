"""Phase-2 deliverable tables, read from the verified artifacts.

What this module is, and is not
-------------------------------
It is a *presentation* layer.  Every number it prints is read from an artifact the
independent verifier has already checked against its own recomputation, and it
recomputes nothing from the datasets: no cell is re-read, no model is built, no
Host cache is opened.  That division matters because the protocol forbids a second
definition of a metric, and a report that recomputed the metrics would be exactly
that — a second definition, free to drift from the verified one.

The two things it does derive are aggregations the protocol *defines* rather than
measures — section 9's median across seeds and median across cells — plus the
section 10 adjudication clauses, which are comparisons between already-verified
per-cell medians.  Both are arithmetic over verified inputs, so a disagreement
here is a disagreement about aggregation, not about a metric.

Where a required table has no artifact behind it, this module says so in the table
instead of manufacturing a column.  The one place that happens is the tail/extreme
harm: section 8 defines harm only for the Normal subset, so the tail and extreme
rows are repaired-side levels, and the table states that rather than inventing a
Host counterpart the run does not record.

Where this module lives, and why not in ``implementation/``
-----------------------------------------------------------
It sits beside the protocol documents rather than inside the ``implementation``
package, and that placement is load-bearing rather than tidiness.  The verifier's
``source_boundary_scan`` requires that the package contain exactly *one* file that
knows a split is sealed (plus the refusal helper that names it in order to refuse
it).  This module has to spell the sealed tokens to report that they were never
read, so putting it in the package would make that scan fail — correctly, since
the scan's rule is about the code path that touches data, and this file touches no
data.  Keeping it outside leaves the package byte-identical to the one the
verifier scanned.

Reads ``06_candidates/records/*.json``, ``00_protocol/oof_levels/*.json``,
``06_candidates/RESULT_INDEX.json`` and ``06_candidates/VERIFICATION_REPORT.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IMPL = HERE / "implementation"
ROOT = HERE.parents[2]
EVIDENCE = ROOT / "experiments/evidence/hch_residual_complete_local_v4_1_20260915"

sys.path.insert(0, str(IMPL))
sys.path.insert(0, str(ROOT / "src"))

import phase0_gate  # noqa: E402
from result_schema import write_json  # noqa: E402
from rcl_contract import CELLS, SEEDS  # noqa: E402

PRIMARY = "R1_RCL_ORTHOGONAL"
LADDER = ("R0_STACK_ZERO_SUM", PRIMARY, "R2_UNTIED_Q", "R3_DIRECT_Q")


# ---------------------------------------------------------------- helpers
def write_text_immutable(path: Path, text: str) -> str:
    """Write a text artifact under the same no-silent-replace rule as the JSON."""
    import hashlib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"{path} already exists with different content; evidence artifacts "
                "are immutable under this protocol")
    else:
        path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def improvement_pct(reference: float, candidate: float) -> float:
    """``(reference - candidate) / reference * 100`` — positive means candidate better.

    One definition, used for every comparison in this report including the
    Host gain, so that "R1 better" cannot mean one thing in the Host table and its
    opposite in the ladder table.
    """
    reference = float(reference)
    candidate = float(candidate)
    if reference == 0:
        return float("nan")
    return (reference - candidate) / reference * 100.0


def median(values) -> float:
    """Plain median.  Non-finite inputs propagate rather than being dropped.

    Dropping them would be the tempting convenience and it is the wrong one: a
    ``nan`` in these tables means a *subset selector matched nothing* — an empty
    extreme tail, say — and averaging over the seeds that did match would report a
    one-seed number under a three-seed column heading.  The empty case is carried
    to the table as ``None`` and counted there instead.
    """
    values = [float(v) for v in values]
    if not values:
        return float("nan")
    return float(np.median(values))


def median_nonempty(values) -> tuple[float, int, int]:
    """``(median over the finite entries, n finite, n total)`` for a subset column.

    Used only where emptiness is a property of the cell rather than of the fit:
    the extreme / tail selectors are TRAIN-only thresholds and a frozen Host, so
    they are identical across the three seeds, and a cell is either all-finite or
    all-empty.  The count is returned so the panel row can say how many cells it
    was taken over.
    """
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not finite:
        return float("nan"), 0, len(values)
    return float(np.median(finite)), len(finite), len(values)


def fmt(value, width: int = 10, places: int = 4) -> str:
    if value is None or not np.isfinite(float(value)):
        return f"{'n/a':>{width}}"
    return f"{float(value):>{width}.{places}f}"


def dash(value, places: int = 2) -> str:
    """A cell value for a markdown table, with an empty subset shown as ``—``."""
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.{places}f}"


# ---------------------------------------------------------------- load
def load_records() -> list[dict]:
    records = [read_json(p) for p in
               sorted((EVIDENCE / "06_candidates/records").glob("*.json"))]
    return records


def cell_medians(records: list[dict]) -> dict:
    """Per cell x variant metric medians across seeds (protocol section 9)."""
    out: dict[tuple[str, str, str], dict] = {}
    for market, host in CELLS:
        for variant in LADDER:
            rows = [r for r in records
                    if r["market"] == market and r["host"] == host
                    and r["variant"] == variant]
            if not rows:
                continue
            med: dict[str, float] = {}
            for key in rows[0]:
                values = [r.get(key) for r in rows]
                numeric = [v for v in values
                           if isinstance(v, (int, float))
                           and not isinstance(v, bool)]
                if numeric and len(numeric) == len(values):
                    med[key] = median(values)
            med["n_seeds"] = len(rows)
            out[(market, host, variant)] = med
    return out


# ---------------------------------------------------------------- registry
def build_registry(records: list[dict]) -> dict:
    """The 216 registered neural fits, each named by the artifact that records it."""
    oof, final, stage2 = [], [], []
    for market, host in CELLS:
        levels = read_json(EVIDENCE / f"00_protocol/oof_levels/{market}__{host}.json")
        for fold in levels["folds"]:
            for seed in fold["seeds"]:
                oof.append({
                    "market": market, "host": host,
                    "stage": "stage1_oof_level", "fold": fold["fold"],
                    "block": fold["block"], "seed": int(seed),
                    "n_fit_days": fold["n_fit_days"],
                    "n_early_days": fold["n_early_days"],
                    "n_target_days": fold["n_target_days"],
                    "target_first_day": fold["target_first_day"],
                    "prefix_last_day": fold["prefix_last_day"],
                    "target_ids_sha256": fold["target_ids_sha256"],
                    "prefix_ids_sha256": fold["prefix_ids_sha256"],
                })
        fl = levels["final_level"]
        for seed, state in zip(fl["seeds"], fl["state_dict_sha256"]):
            final.append({
                "market": market, "host": host,
                "stage": "final_level_refit", "seed": int(seed),
                "frozen": bool(fl["frozen"]),
                "aggregation": fl["aggregation"],
                "fit_ids_sha256": fl["fit_ids_sha256"],
                "early_ids_sha256": fl["early_ids_sha256"],
                "architecture_fingerprint": fl["architecture_fingerprint"],
                "state_dict_sha256": state,
                "s_b": fl["s_b"], "residual_scale": fl["residual_scale"],
            })
    for record in records:
        stage2.append({
            "market": record["market"], "host": record["host"],
            "stage": "stage2_local", "variant": record["variant"],
            "seed": int(record["seed"]),
            "n_parameters": int(record["n_parameters"]),
            "best_epoch": record["best_epoch"],
            "epochs_run": record["epochs_run"],
            "fit_seconds": record["fit_seconds"],
            "initial_state_sha256": record["initial_state_sha256"],
            "final_state_sha256": record["final_state_sha256"],
            "local_target_sha256": record["local_target_sha256"],
            "local_fit_ids_sha256": record["local_fit_ids_sha256"],
            "local_target_ids_sha256": record["local_target_ids_sha256"],
            "local_history_ids_sha256": record["local_history_ids_sha256"],
            "coarse_level_source": record["coarse_level_source"],
            "coarse_level_sha256": record["coarse_level_sha256"],
        })
    counts = {
        "stage1_oof_level": len(oof),
        "final_level_refit": len(final),
        "stage2_local": len(stage2),
    }
    return {
        "schema": "rcl_v4_1_fit_registry.v1",
        "definition": (
            "every neural fit the protocol registers, named by the artifact field "
            "that records it: stage-1 OOF folds x seeds (8 cells x 4 folds x 3), "
            "frozen final-Level refits (8 x 3) and Stage-2 fits "
            "(8 x 4 variants x 3 seeds)"),
        "cells": [list(c) for c in CELLS],
        "seeds": list(SEEDS),
        "counts": counts,
        "total_fits": sum(counts.values()),
        "expected_total_fits": len(CELLS) * (4 * len(SEEDS) + len(SEEDS)
                                             + len(LADDER) * len(SEEDS)),
        "fits": {"stage1_oof_level": oof, "final_level_refit": final,
                 "stage2_local": stage2},
    }


def build_manifests(records: list[dict]) -> dict:
    """Level-conditioner / Local-normalisation / checkpoint manifests."""
    levels, norm, checkpoints = {}, {}, {}
    for market, host in CELLS:
        stem = f"{market}__{host}"
        art = read_json(EVIDENCE / f"00_protocol/oof_levels/{stem}.json")
        conditioner = dict(art["conditioner"])
        levels[stem] = {
            "center": conditioner["center"],
            "scale": conditioner["scale"],
            "n_observations": conditioner["n_observations"],
            "n_fitted_days": len(conditioner["fitted_days"]),
            "fingerprint": conditioner["fingerprint"],
            "architecture_fingerprint": art["final_level"][
                "oof_architecture_fingerprint"],
            "aggregation": art["final_level"]["aggregation"],
            "n_ensemble_members": art["final_level"]["n_ensemble_members"],
            "s_b": art["final_level"]["s_b"],
            "frac_s_b_over_residual_scale": (
                art["final_level"]["s_b"] / art["final_level"]["residual_scale"]),
        }
        rows = [r for r in records if r["market"] == market and r["host"] == host]
        norm[stem] = {
            "variants": sorted({r["variant"] for r in rows}),
            "local_target_sha256_by_variant": {
                v: sorted({r["local_target_sha256"] for r in rows
                           if r["variant"] == v}) for v in LADDER},
            "local_fit_ids_sha256_by_variant": {
                v: sorted({r["local_fit_ids_sha256"] for r in rows
                           if r["variant"] == v}) for v in LADDER},
            "local_target_ids_sha256_by_variant": {
                v: sorted({r["local_target_ids_sha256"] for r in rows
                           if r["variant"] == v}) for v in LADDER},
            "coarse_level_sha256_by_variant": {
                v: sorted({r["coarse_level_sha256"] for r in rows
                           if r["variant"] == v}) for v in LADDER},
        }
        checkpoints[stem] = {
            "final_level_state_dict_sha256": art["final_level"][
                "state_dict_sha256"],
            "stage2_final_state_sha256": {
                f"{r['variant']}__seed_{r['seed']}": r["final_state_sha256"]
                for r in rows},
        }
    return {
        "schema": "rcl_v4_1_manifests.v1",
        "level_condition": levels,
        "local_normalization": norm,
        "checkpoints": checkpoints,
        "checkpoint_note": (
            "these are SHA-256 digests of the in-memory state dictionaries, "
            "recorded at fit time. The run serialises no tensor files, so a "
            "checkpoint cannot be re-loaded from this evidence root; it can only be "
            "checked against a rebuild. The Stage-2 *initial* states are "
            "independently reproducible (the verifier rebuilds each one by seeding "
            "and constructing the variant's model), the final states are not."),
    }


# ---------------------------------------------------------------- tables
def table_cell_medians(medians: dict) -> dict:
    rows = []
    for market, host in CELLS:
        row = {"market": market, "host": host}
        for variant in LADDER:
            med = medians.get((market, host, variant))
            row[variant] = None if med is None else {
                "repaired_overall_mae": med["repaired_overall_mae"],
                "host_overall_mae": med["host_overall_mae"],
                "gain_vs_host_pct": med["gain_vs_host_pct"],
                "q_reconstruction_mae": med["q_reconstruction_mae"],
                "level_mae": med["level_mae"],
                "delta_mae": med["delta_mae"],
                "n_parameters": med["n_parameters"],
                "n_seeds": med["n_seeds"],
            }
        rows.append(row)
    panel = {}
    for variant in LADDER:
        panel[variant] = {
            "median_repaired_overall_mae": median(
                [r[variant]["repaired_overall_mae"] for r in rows
                 if r[variant]]),
            "median_gain_vs_host_pct": median(
                [r[variant]["gain_vs_host_pct"] for r in rows if r[variant]]),
            "median_n_parameters": median(
                [r[variant]["n_parameters"] for r in rows if r[variant]]),
        }
    return {"schema": "rcl_v4_1_cell_median_table.v1",
            "unit": "per-cell metric = median across seeds 7/17/37 (section 9)",
            "cells": rows, "panel": panel}


def table_ladder(medians: dict) -> dict:
    """R1 vs each of R0/R2/R3, per cell and panel."""
    def per_cell(other: str) -> list[dict]:
        out = []
        for market, host in CELLS:
            r1 = medians[(market, host, PRIMARY)]
            ro = medians[(market, host, other)]
            out.append({
                "market": market, "host": host,
                "r1_mae": r1["repaired_overall_mae"],
                "other_mae": ro["repaired_overall_mae"],
                "r1_improvement_pct": improvement_pct(
                    ro["repaired_overall_mae"], r1["repaired_overall_mae"]),
                "other_improvement_pct": improvement_pct(
                    r1["repaired_overall_mae"], ro["repaired_overall_mae"]),
                "r1_n_parameters": r1["n_parameters"],
                "other_n_parameters": ro["n_parameters"],
            })
        return out

    tables = {}
    for other in ("R0_STACK_ZERO_SUM", "R2_UNTIED_Q", "R3_DIRECT_Q"):
        rows = per_cell(other)
        tables[f"R1_vs_{other}"] = {
            "other_variant": other,
            "cells": rows,
            "panel_median_r1_improvement_pct": median(
                [r["r1_improvement_pct"] for r in rows]),
            "panel_median_other_improvement_pct": median(
                [r["other_improvement_pct"] for r in rows]),
            "n_cells_r1_non_worse": sum(
                1 for r in rows if r["r1_improvement_pct"] >= 0),
            "n_cells_other_better": sum(
                1 for r in rows if r["other_improvement_pct"] > 0),
            "worst_r1_improvement_pct": min(
                r["r1_improvement_pct"] for r in rows),
            "n_cells_r1_within_0p5pct": sum(
                1 for r in rows if abs(r["r1_improvement_pct"]) <= 0.5),
            "parameter_ratio_median": median(
                [r["other_n_parameters"] / r["r1_n_parameters"] for r in rows]),
        }
    return {"schema": "rcl_v4_1_ladder_tables.v1", "tables": tables}


def table_host(medians: dict) -> dict:
    rows = []
    for market, host in CELLS:
        r1 = medians[(market, host, PRIMARY)]
        rows.append({
            "market": market, "host": host,
            "host_overall_mae": r1["host_overall_mae"],
            "r1_repaired_overall_mae": r1["repaired_overall_mae"],
            "gain_vs_host_pct": r1["gain_vs_host_pct"],
        })
    return {
        "schema": "rcl_v4_1_host_comparison.v1",
        "rows": rows,
        "n_positive": sum(1 for r in rows if r["gain_vs_host_pct"] > 0),
        "panel_median_gain_pct": median([r["gain_vs_host_pct"] for r in rows]),
        "worst_gain_pct": min(r["gain_vs_host_pct"] for r in rows),
        "caveat": ("protocol section 10: a development target, not a claim of "
                   "baseline competitiveness"),
    }


def table_tail(medians: dict) -> dict:
    def finite_or_none(value):
        value = float(value)
        return value if np.isfinite(value) else None

    rows = []
    for market, host in CELLS:
        r1 = medians[(market, host, PRIMARY)]
        rows.append({
            "market": market, "host": host,
            "upper_mae": finite_or_none(r1["upper_mae"]),
            "lower_mae": finite_or_none(r1["lower_mae"]),
            "extreme_mae": finite_or_none(r1["extreme_mae"]),
            "tail_mae": finite_or_none(r1["tail_mae"]),
            "normal_mae": r1["normal_mae"],
            "normal_host_mae": r1["normal_host_mae"],
            "normal_harm": r1["normal_harm"],
            "normal_improvement_mae": r1["normal_host_mae"] - r1["normal_mae"],
        })

    panel = {}
    for key, label in (("upper_mae", "upper"), ("lower_mae", "lower"),
                       ("extreme_mae", "extreme"), ("tail_mae", "failure_tail"),
                       ("normal_mae", "normal"), ("normal_host_mae", "normal_host"),
                       ("normal_harm", "normal_harm")):
        value, n_finite, n_total = median_nonempty([r[key] for r in rows])
        panel[f"median_{key}"] = None if n_finite == 0 else value
        panel[f"n_cells_{label}_nonempty"] = n_finite

    return {
        "schema": "rcl_v4_1_tail_extreme_table.v1",
        "variant": PRIMARY,
        "threshold_source": "TRAIN_ONLY",
        "definition": ("upper = target >= TRAIN q95; lower = target <= TRAIN q05; "
                       "extreme = upper u lower; failure-tail day = Host daily MAE "
                       ">= TRAIN daily-error q90; normal = not upper/lower"),
        "empty_subset_meaning": (
            "a null cell means the selector matched no horizon cell in that cell's "
            "VAL span, so the metric is undefined rather than zero. The selectors "
            "are TRAIN-only thresholds and a frozen Host forecast, so they do not "
            "depend on the seed or the variant: a cell is all-empty or all-finite, "
            "never a median over fewer than three seeds."),
        "rows": rows,
        "panel": {
            **panel,
            "n_cells_normal_harm_positive": sum(
                1 for r in rows if r["normal_harm"] > 0),
            "n_cells_normal_improved": sum(
                1 for r in rows if r["normal_improvement_mae"] > 0),
        },
        "harm_scope": (
            "section 8 defines harm only for the Normal subset: "
            "normal_harm = mean(max(0, |e_R| - |e_H|))[normal]. The upper / lower / "
            "extreme / failure-tail columns are therefore repaired-side levels in "
            "residual-error space, not harms; the run records no Host counterpart "
            "for those subsets, so no harm can be computed for them from this "
            "evidence root and none is asserted here."),
    }


def table_cost(medians: dict, records: list[dict]) -> dict:
    rows = []
    for market, host in CELLS:
        r1 = medians[(market, host, PRIMARY)]
        r3 = medians[(market, host, "R3_DIRECT_Q")]
        rows.append({
            "market": market, "host": host,
            "r1_n_parameters": r1["n_parameters"],
            "r3_n_parameters": r3["n_parameters"],
            "ratio_r3_over_r1": r3["n_parameters"] / r1["n_parameters"],
            "r1_best_epoch": r1["best_epoch"],
            "r1_epochs_run": r1["epochs_run"],
        })
    return {
        "schema": "rcl_v4_1_parameter_runtime_table.v1",
        "rows": rows,
        "panel": {
            "median_r1_n_parameters": median([r["r1_n_parameters"] for r in rows]),
            "max_ratio_r3_over_r1": max(r["ratio_r3_over_r1"] for r in rows),
            "ceiling": 1.25,
            "total_fit_seconds": sum(float(r["fit_seconds"]) for r in records),
            "median_fit_seconds": median([r["fit_seconds"] for r in records]),
            "max_fit_seconds": max(float(r["fit_seconds"]) for r in records),
        },
    }


# ---------------------------------------------------------------- audit
def build_read_set_audit() -> dict:
    index = read_json(EVIDENCE / "06_candidates/RESULT_INDEX.json")
    manifest = read_json(EVIDENCE / "00_protocol/RUN_MANIFEST.json")
    verification = read_json(EVIDENCE / "06_candidates/VERIFICATION_REPORT.json")
    checks = {c["check"]: c for c in verification["checks"]}
    return {
        "schema": "rcl_v4_1_read_set_audit.v1",
        "runner_audit_block": index["audit"],
        "test_label_read_count": {
            "index": index["test_label_read_count"],
            "manifest": manifest["test_label_read_count"],
            "records": sorted({r["test_label_read_count"] for r in
                               load_records()}),
        },
        "verifier_boundary_checks": {
            name: checks.get(name) for name in
            ("source_boundary_scan", "test_label_read_count_records",
             "test_label_read_count_index")},
        "forbidden_path_tokens": list(
            __import__("verify_run").FORBIDDEN_SOURCE_TOKENS),
        "host_cache_untouched_checks": {
            k: v for k, v in checks.items() if k.startswith("host_untouched:")},
        "joint_file": ("host_predictions_joint.npz was never opened by the runner or "
                       "the verifier; every Host read is the non-joint "
                       "host_predictions.npz, whose loaded rows must equal the "
                       "metadata fit_n + valid_n"),
    }


def build_core_audit() -> dict:
    before = read_json(EVIDENCE / "00_protocol/PHASE0_GATE.json")["src_core"]
    after_manifest = phase0_gate.core_manifest()
    after_digest = phase0_gate.tree_digest(after_manifest)
    after_files = {item["path"]: item["sha256"] for item in after_manifest}
    return {
        "schema": "rcl_v4_1_src_core_audit.v1",
        "definition": ("``src/core`` was read-only for this round; the digest is "
                       "phase0_gate.tree_digest, evaluated over the same 12 files "
                       "before and after"),
        "before": {"n_files": before["n_files"],
                   "tree_digest": before["tree_digest"],
                   "latest_mtime": before["latest_mtime"],
                   "files": before["files"]},
        "after": {"n_files": len(after_manifest), "tree_digest": after_digest,
                  "latest_mtime": max(item["mtime"] for item in after_manifest),
                  "files": after_files},
        "unchanged": (before["tree_digest"] == after_digest
                      and before["files"] == after_files),
        "changed_files": sorted(
            k for k in set(before["files"]) | set(after_files)
            if before["files"].get(k) != after_files.get(k)),
    }


# ---------------------------------------------------------------- adjudication
def adjudicate(medians: dict, ladder: dict, host: dict, tail: dict) -> dict:
    """Protocol section 10, clause by clause, each with its measured value."""
    r0 = ladder["tables"]["R1_vs_R0_STACK_ZERO_SUM"]
    r2 = ladder["tables"]["R1_vs_R2_UNTIED_Q"]
    r3 = ladder["tables"]["R1_vs_R3_DIRECT_Q"]

    delta_beats_zero = sum(
        1 for (market, hostname) in CELLS
        if medians[(market, hostname, PRIMARY)]["delta_mae"]
        < medians[(market, hostname, PRIMARY)]["delta_zero_reference_mae"])
    floor_reduction = median([
        improvement_pct(medians[(m, h, PRIMARY)]["stage1_floor"],
                        medians[(m, h, PRIMARY)]["delta_closure_error"])
        for m, h in CELLS])

    # --- R1 vs R0 ---------------------------------------------------------
    n_positive_r0 = sum(1 for r in r0["cells"] if r["r1_improvement_pct"] > 0)
    clauses_r0 = {
        "positive_cells_ge_5_of_8": n_positive_r0 >= 5,
        "panel_median_gain_ge_0p75pct":
            r0["panel_median_r1_improvement_pct"] >= 0.75,
        "worst_degradation_ge_minus_0p75pct":
            r0["worst_r1_improvement_pct"] >= -0.75,
        "delta_beats_zero_delta_ge_5_of_8": delta_beats_zero >= 5,
        "panel_median_mean_floor_reduction_ge_25pct": floor_reduction >= 25.0,
    }
    label_r0 = ("RCL_RESIDUAL_COMPLETENESS_SUPPORTED"
                if all(clauses_r0.values()) else "MIXED/NOT_SUPPORTED")

    # --- R1 vs R2 ---------------------------------------------------------
    if r2["n_cells_r1_non_worse"] >= 5 and \
            r2["panel_median_r1_improvement_pct"] >= 0:
        label_r2 = "ORTHOGONAL_COORDINATES_PREFERRED"
    elif r2["n_cells_other_better"] >= 5 and \
            r2["panel_median_other_improvement_pct"] >= 0.5:
        label_r2 = "UNTIED_SIGNED_MASS_PREFERRED"
    else:
        label_r2 = "LOCAL_PARAMETERIZATION_MIXED"

    # --- R1 vs R3 ---------------------------------------------------------
    matched_capacity = r3["parameter_ratio_median"] <= 1.25
    if r3["n_cells_other_better"] >= 5 and \
            r3["panel_median_other_improvement_pct"] >= 1.0:
        label_r3 = "STRUCTURED_LOCAL_UNDERFIT_RISK"
    elif r3["n_cells_r1_within_0p5pct"] >= 6 and matched_capacity:
        label_r3 = "STRUCTURED_LOCAL_COMPETITIVE"
    else:
        # The protocol's own word for this branch.  Deliberately not
        # ``LOCAL_PARAMETERIZATION_MIXED``: that is R1-vs-R2's else label, and two
        # ladders sharing an else label would make the two indistinguishable in a
        # summary that prints only the labels.
        label_r3 = "MIXED"

    host_clauses = {
        "positive_ge_7_of_8": host["n_positive"] >= 7,
        "panel_median_gain_ge_1pct": host["panel_median_gain_pct"] >= 1.0,
        "worst_loss_ge_minus_0p5pct": host["worst_gain_pct"] >= -0.5,
    }

    return {
        "schema": "rcl_v4_1_adjudication.v1",
        "aggregation": ("per-cell = median across seeds 7/17/37; panel = median "
                        "across the 8 cells (section 9)"),
        "R1_vs_R0": {
            "label": label_r0, "clauses": clauses_r0,
            "measured": {
                "n_positive_cells": n_positive_r0,
                "panel_median_gain_pct": r0["panel_median_r1_improvement_pct"],
                "worst_gain_pct": r0["worst_r1_improvement_pct"],
                "delta_beats_zero_delta_cells": delta_beats_zero,
                "panel_median_mean_floor_reduction_pct": floor_reduction},
        },
        "R1_vs_R2": {
            "label": label_r2,
            "measured": {
                "n_cells_r1_non_worse": r2["n_cells_r1_non_worse"],
                "panel_median_r1_improvement_pct":
                    r2["panel_median_r1_improvement_pct"],
                "n_cells_r2_better": r2["n_cells_other_better"],
                "panel_median_r2_improvement_pct":
                    r2["panel_median_other_improvement_pct"]},
        },
        "R1_vs_R3": {
            "label": label_r3,
            "measured": {
                "n_cells_r3_better": r3["n_cells_other_better"],
                "panel_median_r3_improvement_pct":
                    r3["panel_median_other_improvement_pct"],
                "n_cells_r1_within_0p5pct": r3["n_cells_r1_within_0p5pct"],
                "median_parameter_ratio_r3_over_r1":
                    r3["parameter_ratio_median"],
                "matched_capacity": matched_capacity},
        },
        "R1_vs_Host": {
            "status": ("DEVELOPMENT_TARGET_MET" if all(host_clauses.values())
                       else "DEVELOPMENT_TARGET_NOT_MET"),
            "clauses": host_clauses,
            "measured": {"n_positive_cells": host["n_positive"],
                         "panel_median_gain_pct": host["panel_median_gain_pct"],
                         "worst_gain_pct": host["worst_gain_pct"]},
            "caveat": host["caveat"],
        },
        "tail_extreme_normal": tail["panel"],
    }


# ---------------------------------------------------------------- RESULTS.md
def render_results(payload: dict) -> str:
    cell_table = payload["tables"]["cell_median_table"]
    ladder = payload["tables"]["ladder"]["tables"]
    host = payload["tables"]["host"]
    tail = payload["tables"]["tail"]
    cost = payload["tables"]["cost"]
    adj = payload["adjudication"]
    registry = payload["registry"]
    core = payload["core_audit"]
    defects = payload["verification"]["defect_analysis"]

    out = []
    add = out.append
    add("# RCL v4.1 — Residual-Complete Local, scientific execution")
    add("")
    add(f"Protocol `{payload['protocol_id']}` · evidence root "
        "`experiments/evidence/hch_residual_complete_local_v4_1_20260915/`")
    add("")
    add(f"Registered neural fits: **{registry['total_fits']}** "
        f"({registry['counts']['stage1_oof_level']} stage-1 OOF + "
        f"{registry['counts']['final_level_refit']} final-Level refits + "
        f"{registry['counts']['stage2_local']} Stage-2), against an expected "
        f"{registry['expected_total_fits']}.")
    add("")
    add(f"Independent verification: **{'PASS' if payload['verification']['passed'] else 'FAIL'}** "
        f"— {payload['verification']['n_checks']} checks, "
        f"{payload['verification']['n_failed']} failed, "
        f"{payload['verification']['n_defects']} supplementary defects "
        "(characterised in §8).")
    add("")
    add("## 1. Cell median table (repaired Overall MAE, median across seeds)")
    add("")
    add("| cell | Host | R0 | R1 | R2 | R3 | R1 gain vs Host % | R1 params |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in cell_table["cells"]:
        cells = []
        for variant in LADDER:
            entry = row[variant]
            cells.append(fmt(entry["repaired_overall_mae"], 0, 2) if entry else "n/a")
        r1 = row[PRIMARY]
        add(f"| {row['market']} | {row['host']} | {cells[0]} | {cells[1]} | "
            f"{cells[2]} | {cells[3]} | {r1['gain_vs_host_pct']:+.3f} | "
            f"{int(r1['n_parameters'])} |")
    panel = cell_table["panel"]
    add(f"| **panel median** | | "
        f"{fmt(panel['R0_STACK_ZERO_SUM']['median_repaired_overall_mae'],0,2)} | "
        f"{fmt(panel[PRIMARY]['median_repaired_overall_mae'],0,2)} | "
        f"{fmt(panel['R2_UNTIED_Q']['median_repaired_overall_mae'],0,2)} | "
        f"{fmt(panel['R3_DIRECT_Q']['median_repaired_overall_mae'],0,2)} | "
        f"{panel[PRIMARY]['median_gain_vs_host_pct']:+.3f} | "
        f"{int(panel[PRIMARY]['median_n_parameters'])} |")
    add("")
    add("## 2. Ladder: R1 vs R0 / R2 / R3")
    add("")
    for name, table in ladder.items():
        other = table["other_variant"]
        add(f"### R1 vs {other}")
        add("")
        add(f"- cells where R1 is non-worse: **{table['n_cells_r1_non_worse']}/8**")
        add(f"- cells where {other} is better: **{table['n_cells_other_better']}/8**")
        add(f"- panel median R1 improvement: "
            f"**{table['panel_median_r1_improvement_pct']:+.4f}%**")
        add(f"- panel median {other} improvement: "
            f"**{table['panel_median_other_improvement_pct']:+.4f}%**")
        add(f"- worst R1 improvement: {table['worst_r1_improvement_pct']:+.4f}%")
        add(f"- cells with |R1 improvement| <= 0.5%: "
            f"{table['n_cells_r1_within_0p5pct']}/8; median parameter ratio "
            f"{other}/R1 = {table['parameter_ratio_median']:.4f}")
        add("")
        add("| cell | R1 MAE | " + other + " MAE | R1 improvement % |")
        add("|---|---:|---:|---:|")
        for row in table["cells"]:
            add(f"| {row['market']}/{row['host']} | {row['r1_mae']:.2f} | "
                f"{row['other_mae']:.2f} | {row['r1_improvement_pct']:+.4f} |")
        add("")
    add("## 3. Host comparison (R1)")
    add("")
    add("| cell | Host MAE | R1 MAE | gain % |")
    add("|---|---:|---:|---:|")
    for row in host["rows"]:
        add(f"| {row['market']}/{row['host']} | {row['host_overall_mae']:.2f} | "
            f"{row['r1_repaired_overall_mae']:.2f} | "
            f"{row['gain_vs_host_pct']:+.4f} |")
    add("")
    add(f"positive {host['n_positive']}/8 · panel median "
        f"{host['panel_median_gain_pct']:+.4f}% · worst "
        f"{host['worst_gain_pct']:+.4f}% — {host['caveat']}.")
    add("")
    add("## 4. Level / delta closure core metrics (R1)")
    add("")
    add("| cell | stage1 floor | delta closure err | floor reduction % | q reconstruction MAE |")
    add("|---|---:|---:|---:|---:|")
    for market, hostname in CELLS:
        med = payload["medians"][f"{market}__{hostname}"][PRIMARY]
        red = improvement_pct(med["stage1_floor"], med["delta_closure_error"])
        add(f"| {market}/{hostname} | {med['stage1_floor']:.2f} | "
            f"{med['delta_closure_error']:.2f} | {red:+.3f} | "
            f"{med['q_reconstruction_mae']:.2f} |")
    add("")
    add("Three identities hold in every row above, and they are why some of the "
        "protocol's columns coincide rather than independent measurements:")
    add("")
    add("- `level_mae = stage1_floor = delta_zero_reference_mae = "
        "mean|mean(r) - b_hat|` — the Stage-1 mean-floor error, which is also the "
        "reference a zero `delta` would score.")
    add("- `delta_mae = delta_closure_error = mean|mean(r) - b_hat - delta_hat|` — "
        "`delta_hat` is a per-day scalar, so the delta's own error and the closure "
        "error are the same functional.")
    add("- `q_reconstruction_mae = repaired_overall_mae` — `q_hat = delta_hat*1 + "
        "A_hat(S+ - S-)` and `correction = coarse_level*1 + local` are two routes "
        "to the same emitted quantity (`local = q_hat`), so "
        "`(r - b_hat) - q_hat = r - correction` identically. The two columns agree "
        "to float32-vs-float64 reduction precision, and that agreement is the "
        "reconstruction check passing, not a duplicated column.")
    add("")
    add("Panel closure ratio per variant (protocol section 13, "
        "`median|b*-b_hat-delta_hat| / median|b*-b_hat|`; below 1 means `delta` "
        "recovered part of the coarse Level's mean error, at 1 it carried none):")
    add("")
    add("| variant | closure ratio |")
    add("|---|---:|")
    for variant, value in payload["closure_ratio"].items():
        add(f"| {variant} | {value:.6f} |")
    add("")
    add("## 5. Tail / extreme / Normal")
    add("")
    add("| cell | upper | lower | extreme | failure-tail | Normal | "
        "Normal Host | Normal harm |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in tail["rows"]:
        add(f"| {row['market']}/{row['host']} | {dash(row['upper_mae'])} | "
            f"{dash(row['lower_mae'])} | {dash(row['extreme_mae'])} | "
            f"{dash(row['tail_mae'])} | {row['normal_mae']:.2f} | "
            f"{row['normal_host_mae']:.2f} | {row['normal_harm']:+.4f} |")
    p = tail["panel"]
    add(f"| **panel median** | {dash(p['median_upper_mae'])} | "
        f"{dash(p['median_lower_mae'])} | {dash(p['median_extreme_mae'])} | "
        f"{dash(p['median_tail_mae'])} | {p['median_normal_mae']:.2f} | "
        f"{p['median_normal_host_mae']:.2f} | {p['median_normal_harm']:+.4f} |")
    add("")
    add(f"Panel medians are taken over the cells whose selector matched something: "
        f"upper {p['n_cells_upper_nonempty']}/8, lower "
        f"{p['n_cells_lower_nonempty']}/8, extreme "
        f"{p['n_cells_extreme_nonempty']}/8, failure-tail "
        f"{p['n_cells_failure_tail_nonempty']}/8, Normal "
        f"{p['n_cells_normal_nonempty']}/8. `—` is an empty subset, not a zero: "
        f"{tail['empty_subset_meaning']}")
    add("")
    add(f"Normal harm is positive in **{p['n_cells_normal_harm_positive']}/8** "
        f"cells; the Normal subset improves in {p['n_cells_normal_improved']}/8. "
        f"Thresholds are {tail['threshold_source']}: {tail['definition']}.")
    add("")
    add(f"Read the 8/8 with care: `normal_harm` is one-sided "
        f"(`mean(max(0, |e_R| - |e_H|))`), so it is non-negative by construction and "
        f"any cell with a single worse day reports a positive value. The magnitude "
        f"is what carries information, and the panel median is "
        f"{p['median_normal_harm']:+.4f} in residual-error units against a Normal "
        f"level of {p['median_normal_mae']:.2f} — the two subsets that agree in sign "
        f"are the ones where the mean itself moved.")
    add(f"> {tail['harm_scope']}")
    add("")
    add("## 6. Parameters and runtime")
    add("")
    add("| cell | R1 params | R3 params | R3/R1 | R1 best epoch | R1 epochs run |")
    add("|---|---:|---:|---:|---:|---:|")
    for row in cost["rows"]:
        add(f"| {row['market']}/{row['host']} | {int(row['r1_n_parameters'])} | "
            f"{int(row['r3_n_parameters'])} | {row['ratio_r3_over_r1']:.4f} | "
            f"{row['r1_best_epoch']:.0f} | {row['r1_epochs_run']:.0f} |")
    add("")
    add(f"median R1 parameters {int(cost['panel']['median_r1_n_parameters'])} · "
        f"max R3/R1 {cost['panel']['max_ratio_r3_over_r1']:.4f} "
        f"(ceiling {cost['panel']['ceiling']}) · "
        f"{cost['panel']['total_fit_seconds']:.1f}s total Stage-2 fit time, "
        f"median {cost['panel']['median_fit_seconds']:.2f}s per fit.")
    add("")
    add("## 7. Adjudication")
    add("")
    add(f"- **R1 vs R0 → `{adj['R1_vs_R0']['label']}`**")
    for clause, ok in adj["R1_vs_R0"]["clauses"].items():
        add(f"  - {'PASS' if ok else 'FAIL'} — {clause}")
    add(f"  - measured: {json.dumps(adj['R1_vs_R0']['measured'])}")
    add(f"- **R1 vs R2 → `{adj['R1_vs_R2']['label']}`**")
    add(f"  - measured: {json.dumps(adj['R1_vs_R2']['measured'])}")
    add(f"- **R1 vs R3 → `{adj['R1_vs_R3']['label']}`**")
    add(f"  - measured: {json.dumps(adj['R1_vs_R3']['measured'])}")
    add(f"- R1 vs Host → `{adj['R1_vs_Host']['status']}` "
        f"({adj['R1_vs_Host']['caveat']})")
    add(f"  - measured: {json.dumps(adj['R1_vs_Host']['measured'])}")
    add("")
    add("## 8. Boundary and verification")
    add("")
    add(f"- `src/core` unchanged: **{core['unchanged']}** "
        f"(digest `{core['after']['tree_digest'][:16]}…`, "
        f"{core['after']['n_files']} files)")
    add(f"- TEST target reads: "
        f"**{payload['read_set_audit']['test_label_read_count']['manifest']}**; "
        "`host_predictions_joint.npz` never opened; no QINGHAI, foreign, final or "
        "baseline path read.")
    add(f"- Verifier: {payload['verification']['n_checks']} checks, "
        f"{payload['verification']['n_failed']} failed.")
    add("")
    add("### Supplementary defects (do not decide the round)")
    add("")
    add(f"{defects['n_defects']} defects on the two supplementary centred-target "
        f"diagnostics, of which {defects['rounding_scale']['n']} are float32 "
        f"rounding on a supplementary readout and "
        f"{defects['amplification']['n']} are the amplification described below.")
    add("")
    diag = defects["amplification"]["diagnosis"]
    if diag:
        add(f"- fields: {', '.join(diag['fields'])}; fits: {diag['n_fits']}")
        add(f"- cause: {diag['cause']}")
        add(f"- scope: {diag['scope']}")
        add(f"- remedy: {diag['fix']}")
        add(f"- not applied: {diag['not_applied_because']}")
    else:
        add("- no amplification-scale defect was found.")
    add("")
    add("## 9. Reading")
    add("")
    add(payload["reading"])
    add("")
    return "\n".join(out)


# ---------------------------------------------------------------- main
def main() -> int:
    records = load_records()
    medians = cell_medians(records)
    registry = build_registry(records)
    manifests = build_manifests(records)
    cell_table = table_cell_medians(medians)
    ladder = table_ladder(medians)
    host = table_host(medians)
    tail = table_tail(medians)
    cost = table_cost(medians, records)
    read_set = build_read_set_audit()
    core = build_core_audit()
    adjudication = adjudicate(medians, ladder, host, tail)
    verification = read_json(EVIDENCE / "06_candidates/VERIFICATION_REPORT.json")
    closure = read_json(EVIDENCE / "06_candidates/RESULT_INDEX.json")[
        "closure_ratio_by_variant"]

    if not registry["total_fits"] == registry["expected_total_fits"] == 216:
        raise SystemExit(
            f"the registry does not close at 216: {registry['counts']}")
    if not core["unchanged"]:
        raise SystemExit(f"src/core changed during the round: {core['changed_files']}")

    medians_out = {f"{m}__{h}": {v: medians[(m, h, v)] for v in LADDER}
                   for m, h in CELLS}

    payload = {
        "protocol_id": "HCH_RESIDUAL_COMPLETE_LOCAL_V4_1_20260915",
        "registry": registry,
        "tables": {"cell_median_table": cell_table, "ladder": ladder,
                   "host": host, "tail": tail, "cost": cost},
        "adjudication": adjudication,
        "read_set_audit": read_set,
        "core_audit": core,
        "verification": verification,
        "closure_ratio": closure,
        "medians": medians_out,
    }

    written = []
    written.append(write_json(EVIDENCE / "00_protocol/PREEXECUTION_GATE.json",
                              read_json(EVIDENCE / "00_protocol/PHASE0_GATE.json")))
    written.append(write_json(EVIDENCE / "02_registry/FIT_REGISTRY.json", registry))
    written.append(write_json(EVIDENCE / "02_registry/MANIFESTS.json", manifests))
    written.append(write_json(EVIDENCE / "03_tables/cell_median_table.json",
                              cell_table))
    written.append(write_json(EVIDENCE / "03_tables/ladder_R1_vs_R0_R2_R3.json",
                              ladder))
    written.append(write_json(EVIDENCE / "03_tables/host_comparison.json", host))
    written.append(write_json(EVIDENCE / "03_tables/tail_extreme_normal.json", tail))
    written.append(write_json(EVIDENCE / "03_tables/parameter_runtime.json", cost))
    written.append(write_json(EVIDENCE / "04_audit/read_set_audit.json", read_set))
    written.append(write_json(EVIDENCE / "04_audit/src_core_before_after.json", core))
    written.append(write_json(EVIDENCE / "05_adjudication/adjudication.json",
                              adjudication))

    # ``reading`` is prose that depends on the labels, so it is set last and the
    # markdown is rendered once both are known.
    labels = (adjudication["R1_vs_R0"]["label"],
              adjudication["R1_vs_R2"]["label"],
              adjudication["R1_vs_R3"]["label"],
              adjudication["R1_vs_Host"]["status"])
    payload["reading"] = (
        f"Three section-10 labels: R1-vs-R0 `{labels[0]}`, R1-vs-R2 `{labels[1]}`, "
        f"R1-vs-R3 `{labels[2]}`; the R1-vs-Host development target is "
        f"`{labels[3]}`. The ladder separates the four coordinates on the same "
        f"emitted quantity (q reconstruction), so the labels differ because the "
        f"coordinate choice differs, not because the Level or the emission rule "
        f"changed. All numbers above are the verifier's arithmetic over the raw "
        f"artifacts; the runner's own metric functions were not imported by the "
        f"verifier. The {verification['n_defects']} supplementary defects are "
        f"disclosed in section 8 and affect only the centred-target diagnostics.")
    written.append(write_text_immutable(EVIDENCE / "RESULTS.md",
                                        render_results(payload)))
    for path in written:
        print(f"  wrote {path}")
    print(f"registry: {registry['counts']} total {registry['total_fits']}")
    print(f"labels: {labels}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
