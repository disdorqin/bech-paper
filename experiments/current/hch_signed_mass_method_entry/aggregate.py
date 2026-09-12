"""Aggregation, component adjudication and the development gate (PART E-H).

Everything here reads persisted evidence and decides; nothing here fits, tunes or
re-runs a model.  That separation is the point: a decision module that could also
produce its own inputs would be free to make the evidence agree with it.

Four rules shape the code, each of them a requirement rather than a preference.

**Seeds aggregate by cell median.**  The registered seeds are replicates, not
candidates.  A configuration is summarised by the median over ``{7, 17, 37}`` of
each cell, and no step ever selects a best seed -- selecting one would turn a
replicate into a hyperparameter and quietly shrink the effective sample.  This is
why :func:`cell_table` collapses seeds before anything else looks at the numbers.

**A component is deleted unless the pre-registered evidence supports it.**  The
five retention conditions are evaluated from the primary and secondary metrics
each component was registered with, before any of them was run.  There is no
rescue variant, no per-market exemption and no post-hoc metric substitution: a
component that fails condition 1 on its own pre-registered primary metric is
deleted even if some other metric moved favourably, because that other metric was
not the claim.

**The frozen method contains only retained components.**  It is one switch
vector, derived mechanically from the decisions.  If it coincides with an
already-executed registered configuration the persisted raw predictions may be
reused, but only after an identity check that the switch vectors and seeds agree
exactly -- a coincidence of names is not evidence of a coincidence of recipes.

**Comparators are joined, never re-run.**  The strict offline table is frozen
upstream evidence.  Our method is added to it as one more column of numbers, at
the same setting, and the ranking is computed from the table's own values.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import config as C
from . import raw_evidence as RE
from .contracts import HarnessError, write_csv

__all__ = [
    "read_csv", "cell_table", "component_decisions", "frozen_vector",
    "baseline_comparison", "supplementary_view", "development_gates",
    "verdict_token", "run",
    "COMPONENT_SPECS", "DECISION_FIELDS", "COMPARISON_FIELDS",
    "CELL_METRIC_FIELDS", "DELTA_FIELDS", "SUPPLEMENTARY_FIELDS",
]


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def read_csv(path: Any) -> List[Dict[str, str]]:
    """Read a CSV into dicts, tolerating a missing file by returning nothing.

    The harness writes its own tables, so a missing one is a real absence rather
    than an error to raise here; the caller decides whether absence is fatal.
    """
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "n/a"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.median(present)) if present else None


def _fmt(value: Any) -> str:
    number = _number(value)
    return "" if number is None else repr(number)


# --------------------------------------------------------------------------
# Seed aggregation: cell medians, never a best seed
# --------------------------------------------------------------------------
#: Fields that are identity or provenance rather than a measurement.  They are
#: never medianed, because the median of three seeds' hashes is not a hash.
_NON_NUMERIC_FIELDS = frozenset({
    "market", "host", "cell", "config", "seed", "partition", "status",
    "failure", "raw_path", "raw_sha256", "raw_bytes", "switches", "seeds",
    "branch_arrays_used", "inference_setting", "amplitude_scale_rule",
    "high_mass_status_positive", "high_mass_status_negative",
})


def _numeric_fields(records: Sequence[Mapping[str, Any]]) -> List[str]:
    """The numeric columns the records actually carry.

    Derived from the records rather than from a fixed list because the two
    callers do not hand over the same shape: :func:`run` passes rows read from
    ``GRID_STATUS.csv``, while the independent verifier passes records it rebuilt
    from the raw arrays.  A registered list would silently drop whichever columns
    the caller happened not to have, and a summariser that quietly loses a metric
    is indistinguishable from a metric that was never computed.

    A field qualifies when at least one record carries a finite number for it.
    Some seeds legitimately report ``None`` for a metric with an empty subset
    (``NOT_APPLICABLE_N0``); that is a fact about those seeds, not a reason to
    erase the column for the seeds that do report it, so the gaps survive to
    :func:`_median`, which skips them.
    """
    numeric: Dict[str, bool] = {}
    for record in records:
        for key, value in record.items():
            if key in _NON_NUMERIC_FIELDS:
                continue
            numeric[key] = numeric.get(key, False) or _number(value) is not None
    return [key for key, ok in numeric.items() if ok]


def cell_table(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse the seed axis: one row per ``(cell, config)``, seeds medianed.

    Only successful records take part.  A failed fit is not a zero and not a
    missing value to be imputed; it is carried as a blocker, and a cell whose
    grid is incomplete is reported as incomplete rather than silently summarised
    from the seeds that happened to finish.
    """
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    failures: Dict[Tuple[str, str], List[str]] = {}
    for record in records:
        key = (str(record.get("cell")), str(record.get("config")))
        if str(record.get("status", "")).startswith("OK"):
            grouped.setdefault(key, []).append(record)
        else:
            failures.setdefault(key, []).append(str(record.get("failure")))

    numeric = _numeric_fields(records)
    out: List[Dict[str, Any]] = []
    for (cell, config), rows in sorted(grouped.items()):
        row: Dict[str, Any] = {
            "cell": cell, "config": config,
            "market": rows[0].get("market"), "host": rows[0].get("host"),
            "n_seeds": len(rows),
            "seeds": "|".join(str(int(float(r["seed"]))) for r in rows),
            "n_failed_seeds": len(failures.get((cell, config), [])),
            "complete": len(rows) == len(C.TRAINING["seeds"])
            and not failures.get((cell, config)),
            "switches": _switches_text(rows[0].get("switches"), config),
        }
        for field in numeric:
            row[field] = _median(_number(r.get(field)) for r in rows)
        out.append(row)
    return out


def _switches_text(value: Any, config: str) -> str:
    """The switch vector of a cell row, canonically serialised.

    The record's own value wins, because the artifact is the evidence and the
    registry is only a name table -- the smallest surviving method is generally
    *not* a registered name, so a lookup by config name would either fail or,
    worse, silently substitute a different vector.  The registry is the fallback
    for a record that predates the column.
    """
    if isinstance(value, Mapping) and value:
        return json.dumps({str(k): bool(v) for k, v in value.items()},
                          sort_keys=True, separators=(",", ":"))
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        if isinstance(parsed, Mapping):
            return json.dumps({str(k): bool(v) for k, v in parsed.items()},
                              sort_keys=True, separators=(",", ":"))
        return value
    registered = C.CONFIG_VARIANTS.get(config)
    return json.dumps(dict(registered or {}), sort_keys=True,
                      separators=(",", ":"))


def _cell_map(cells: Sequence[Mapping[str, Any]], config: str
              ) -> Dict[str, Mapping[str, Any]]:
    return {str(r["cell"]): r for r in cells if r["config"] == config}


# --------------------------------------------------------------------------
# PART E -- component adjudication
# --------------------------------------------------------------------------
#: Each component's pre-registered evidence.  ``primary`` is the claim; the
#: component succeeds or fails on those metrics alone.  ``secondary`` carries the
#: co-registered tail/upper/lower evidence, and ``diagnostic`` is reported but
#: never decides anything -- a diagnostic that could flip a decision would be a
#: second, undeclared claim.  Lower is better for every metric named here.
COMPONENT_SPECS: Dict[str, Dict[str, Any]] = {
    "shape_semantic_context": {
        "on": "FULL", "off": "NO_SHAPE_CONTEXT",
        "primary": ("shape_w1_positive", "shape_w1_negative"),
        "secondary": ("tail_mae",),
        "diagnostic": ("mass_positive_l1", "mass_negative_l1", "overall_mae"),
        "claim": ("Shape semantic context improves the direct per-sign Shape "
                  "Wasserstein distance against the realised signed-mass shape."),
    },
    "use_tcn": {
        "on": "FULL", "off": "NO_TCN",
        "primary": ("tail_mae",),
        "secondary": ("tail_mae",),
        "diagnostic": ("shape_w1_positive", "shape_w1_negative",
                       "mass_positive_l1", "mass_negative_l1", "overall_mae"),
        "claim": ("The TCN is an executor, not a contribution: it is retained only "
                  "if the temporal convolution improves the tail it exists to "
                  "execute."),
    },
    "untied_amplitude_heads": {
        "on": "FULL", "off": "TIED_AMPLITUDE",
        "primary": ("mass_positive_l1", "mass_negative_l1"),
        "secondary": ("upper_tail_mae", "lower_tail_mae"),
        "diagnostic": ("overall_mae", "tail_mae"),
        "claim": ("Untied +/- Amplitude heads improve the direct per-sign mass "
                  "error, and separately the upper and lower tails."),
    },
    "rare_mass_sampling": {
        "on": "FULL", "off": "NO_RARE_MASS",
        "primary": ("high_mass_positive_l1", "high_mass_negative_l1"),
        "secondary": ("tail_mae",),
        "diagnostic": ("negative_price_mae", "normal_relative_harm_pct",
                       "overall_mae"),
        "claim": ("Rare-mass spread batching improves the direct mass error on the "
                  "high-realised-mass subset without harming the normal region."),
    },
}

#: The registered margins.  They are named here rather than inlined so the
#: decision table can print the rule it applied.
NORMAL_HARM_LIMIT_PCT = 1.0
MAJORITY_CELLS = 10  # of 20: strictly more than this must improve

DECISION_FIELDS = (
    "component", "on_config", "off_config", "claim",
    "primary_metrics", "secondary_metrics", "diagnostic_metrics",
    "n_cells", "n_cells_primary_improved", "n_cells_primary_worsened",
    "n_cells_primary_tied", "per_primary_cell_counts",
    "median_delta_primary", "per_primary_median_delta",
    "median_delta_secondary", "per_secondary_median_delta",
    "median_delta_overall_mae", "median_delta_normal_harm_pct",
    "max_delta_normal_harm_pct", "worst_normal_harm_cell",
    "leave_one_cell_out_min_majority",
    "n_seeds_with_majority_improvement", "per_seed_majority",
    "condition_1_majority_cells", "condition_2_tail_or_relevant_improves",
    "condition_3_overall_non_worse", "condition_4_normal_harm_bounded",
    "condition_5_not_one_cell_or_seed",
    "decision", "reason",
)


def _cells_for(cells: Sequence[Mapping[str, Any]]) -> List[str]:
    return sorted({str(r["cell"]) for r in cells})


def _per_seed_improvement(records: Sequence[Mapping[str, Any]], config_on: str,
                          config_off: str, metric: str) -> Dict[str, int]:
    """How many cells improve, per seed, on one metric.

    The cell median can hide a seed that disagrees with the other two.  This
    counts the cells where the ON arm beats the OFF arm *within the same seed*,
    which is the only comparison in which the two arms saw identical data.
    """
    pairs: Dict[Tuple[str, int], Dict[str, Optional[float]]] = {}
    for record in records:
        config = str(record.get("config"))
        if config not in (config_on, config_off):
            continue
        if not str(record.get("status", "")).startswith("OK"):
            continue
        key = (str(record.get("cell")), int(float(record["seed"])))
        pairs.setdefault(key, {})[config] = _number(record.get(metric))
    per_seed: Dict[int, int] = {}
    for (cell, seed), arms in pairs.items():
        on, off = arms.get(config_on), arms.get(config_off)
        if on is None or off is None:
            continue
        per_seed[seed] = per_seed.get(seed, 0) + (1 if on < off else 0)
    return per_seed


def _adjudicate(records: Sequence[Mapping[str, Any]],
                cells: Sequence[Mapping[str, Any]], component: str,
                spec: Mapping[str, Any]) -> Dict[str, Any]:
    on_map = _cell_map(cells, spec["on"])
    off_map = _cell_map(cells, spec["off"])
    shared = sorted(set(on_map) & set(off_map))

    def deltas(metric: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for cell in shared:
            on, off = _number(on_map[cell].get(metric)), _number(off_map[cell].get(metric))
            if on is None or off is None:
                continue
            out[cell] = on - off
        return out

    primary_counts: Dict[str, Dict[str, int]] = {}
    primary_medians: Dict[str, Optional[float]] = {}
    per_primary_cells: Dict[str, int] = {}
    for metric in spec["primary"]:
        d = deltas(metric)
        improved = sum(1 for v in d.values() if v < 0.0)
        worsened = sum(1 for v in d.values() if v > 0.0)
        primary_counts[metric] = {
            "improved": improved, "worsened": worsened,
            "tied": len(shared) - improved - worsened}
        primary_medians[metric] = _median(d.values())
        per_primary_cells[metric] = improved

    secondary_medians = {m: _median(deltas(m).values()) for m in spec["secondary"]}
    diagnostic_medians = {m: _median(deltas(m).values()) for m in spec["diagnostic"]}

    overall = deltas("overall_mae")
    harm = deltas("normal_relative_harm_pct")
    worst_cell = max(harm, key=lambda c: harm[c]) if harm else None

    # Condition 5a -- leave one cell out; the majority must survive every deletion.
    loo_min = None
    if shared:
        loo = []
        for metric in spec["primary"]:
            d = deltas(metric)
            for dropped in d:
                remaining = [v for c, v in d.items() if c != dropped]
                if remaining:
                    loo.append(sum(1 for v in remaining if v < 0.0) > len(remaining) / 2)
        loo_min = bool(loo) and all(loo)

    # Condition 5b -- per seed, a majority of cells must improve.
    per_seed: Dict[str, Dict[str, int]] = {}
    seeds_with_majority = 0
    for metric in spec["primary"]:
        counts = _per_seed_improvement(records, spec["on"], spec["off"], metric)
        per_seed[metric] = {str(k): v for k, v in sorted(counts.items())}
    if per_seed:
        seeds = sorted({int(s) for counts in per_seed.values() for s in counts})
        agreeing = [s for s in seeds
                    if all(counts.get(str(s), 0) > MAJORITY_CELLS
                           for counts in per_seed.values())]
        seeds_with_majority = len(agreeing)

    c1 = bool(spec["primary"]) and all(
        counts["improved"] > MAJORITY_CELLS for counts in primary_counts.values())
    c2 = bool(secondary_medians) and all(
        v is not None and v < 0.0 for v in secondary_medians.values())
    c3 = _median(overall.values())
    c3_ok = c3 is not None and c3 <= 0.0
    c4_ok = all(v <= NORMAL_HARM_LIMIT_PCT for v in harm.values()) if harm else True
    c5_ok = bool(loo_min) and seeds_with_majority >= 2

    reasons: List[str] = []
    if not c1:
        reasons.append("primary mechanism does not improve in a majority of cells")
    if not c2:
        reasons.append("median relevant tail/upper/lower metric does not improve")
    if not c3_ok:
        reasons.append("median Overall-MAE is worse")
    if not c4_ok:
        reasons.append(f"a cell gains more than {NORMAL_HARM_LIMIT_PCT}% normal-region "
                       "relative harm from this component alone")
    if not c5_ok:
        reasons.append("the effect does not survive leave-one-cell-out or is a "
                       "single-seed artifact")

    keep = c1 and c2 and c3_ok and c4_ok and c5_ok
    return {
        "component": component,
        "on_config": spec["on"], "off_config": spec["off"],
        "claim": spec["claim"],
        "primary_metrics": "|".join(spec["primary"]),
        "secondary_metrics": "|".join(spec["secondary"]),
        "diagnostic_metrics": "|".join(spec["diagnostic"]),
        "n_cells": len(shared),
        "n_cells_primary_improved": min(per_primary_cells.values(), default=0),
        "n_cells_primary_worsened": min(
            (c["worsened"] for c in primary_counts.values()), default=0),
        "n_cells_primary_tied": min(
            (c["tied"] for c in primary_counts.values()), default=0),
        "per_primary_cell_counts": json.dumps(
            {m: primary_counts[m]["improved"] for m in spec["primary"]}, sort_keys=True),
        "median_delta_primary": _fmt(min(
            (v for v in primary_medians.values() if v is not None), default=None)),
        "per_primary_median_delta": json.dumps(
            {m: _fmt(primary_medians[m]) for m in spec["primary"]}, sort_keys=True),
        "median_delta_secondary": _fmt(_median(
            [v for v in secondary_medians.values() if v is not None])),
        "per_secondary_median_delta": json.dumps(
            {m: _fmt(secondary_medians[m]) for m in spec["secondary"]}, sort_keys=True),
        "median_delta_overall_mae": _fmt(c3),
        "median_delta_normal_harm_pct": _fmt(_median(harm.values())),
        "max_delta_normal_harm_pct": _fmt(max(harm.values()) if harm else None),
        "worst_normal_harm_cell": worst_cell,
        "leave_one_cell_out_min_majority": loo_min,
        "n_seeds_with_majority_improvement": seeds_with_majority,
        "per_seed_majority": json.dumps(per_seed, sort_keys=True),
        "condition_1_majority_cells": c1,
        "condition_2_tail_or_relevant_improves": c2,
        "condition_3_overall_non_worse": c3_ok,
        "condition_4_normal_harm_bounded": c4_ok,
        "condition_5_not_one_cell_or_seed": c5_ok,
        "decision": "RETAINED" if keep else "DELETED",
        "reason": ("all five pre-registered conditions hold" if keep
                   else "; ".join(reasons)),
        "diagnostics": json.dumps(
            {m: _fmt(diagnostic_medians[m]) for m in spec["diagnostic"]}, sort_keys=True),
    }


def component_decisions(records: Sequence[Mapping[str, Any]],
                        cells: Optional[Sequence[Mapping[str, Any]]] = None
                        ) -> List[Dict[str, Any]]:
    """Adjudicate all four components against their registered evidence."""
    table = list(cells) if cells is not None else cell_table(records)
    for cell in table:
        if not cell.get("complete"):
            raise HarnessError(
                f"{cell['cell']} / {cell['config']}: the seed grid is incomplete "
                f"({cell['n_seeds']} of {len(C.TRAINING['seeds'])} seeds, "
                f"{cell['n_failed_seeds']} failed); a component decision cannot be "
                "taken on a partial panel")
    return [_adjudicate(records, table, name, spec)
            for name, spec in COMPONENT_SPECS.items()]


def frozen_vector(decisions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The single switch vector containing only retained components.

    Derived from the decisions, never chosen: a component that was deleted is
    switched off and nothing else changes.  A vector that matches a registered
    configuration is reported as such so the caller can reuse persisted
    predictions *after* an identity check rather than re-running them.
    """
    vector = dict(C.FULL_SWITCHES)
    deleted = []
    for decision in decisions:
        if decision["decision"] != "RETAINED":
            name = str(decision["component"])
            if name not in C.SWITCH_NAMES:
                raise HarnessError(f"decision names an unknown switch {name!r}")
            vector[name] = False
            deleted.append(name)
    matches = [name for name, registered in C.CONFIG_VARIANTS.items()
               if dict(registered) == vector]
    return {"switches": vector, "deleted_components": sorted(deleted),
            "retained_components": sorted(n for n in C.SWITCH_NAMES if vector[n]),
            "matches_registered_config": matches[0] if matches else None,
            "registered_config_match_count": len(matches),
            "n_components_retained": sum(1 for n in C.SWITCH_NAMES if vector[n])}


# --------------------------------------------------------------------------
# PART G -- the strict offline comparison
# --------------------------------------------------------------------------
COMPARISON_FIELDS = (
    "coordinate_id", "market", "host",
    "our_method_Overall_MAE", "our_method_Tail_MAE",
    "our_method_Upper_tail_MAE", "our_method_Lower_tail_MAE",
    "our_method_Normal_region_MAE", "our_method_negative_price_MAE",
    "our_method_n_days", "our_method_seeds",
    "host_overall_mae", "host_tail_mae", "host_normal_mae",
    "our_gain_vs_host_pct", "our_tail_gain_vs_host_pct",
    "our_normal_harm_pct", "host_nonworse",
    "best_strict_comparator", "best_strict_comparator_setting",
    "best_strict_comparator_Overall_MAE", "best_strict_comparator_Tail_MAE",
    "best_strict_comparator_status", "strict_comparator_count",
    "strict_win", "strict_margin_pct", "strict_margin_abs",
    "control_method", "control_Overall_MAE", "control_status",
    "our_gain_vs_control_pct", "control_excluded_from_rank",
    "inference_setting_disclosure",
)

#: The registered column order of ``CELL_METRICS.csv``.  Declared rather than
#: discovered, so a metric that stops being produced leaves a hole in a known
#: place instead of quietly disappearing from the table.
CELL_METRIC_FIELDS = (
    "cell", "market", "host", "config", "seeds", "n_seeds", "n_failed_seeds",
    "complete", "switches",
    "n_days", "n_entries_evaluated", "n_days_any_valid",
    "overall_mae", "host_mae", "mae_gain_abs", "mae_gain_pct",
    "tail_mae", "tail_mae_host", "upper_tail_mae", "lower_tail_mae",
    "normal_mae", "normal_mae_host", "normal_relative_harm_pct",
    "negative_price_mae", "negative_price_mae_host",
    "shape_w1_positive", "shape_w1_negative",
    "mass_positive_l1", "mass_negative_l1",
    "high_mass_positive_l1", "high_mass_negative_l1",
    "high_mass_positive_n_days", "high_mass_negative_n_days",
    "alpha", "amplitude_scale", "n_parameters", "fit_seconds",
    "inference_seconds", "n_folds", "final_epochs",
    "branch_reconstruction_max_abs_err", "alpha_consistency_max_abs_err",
)

#: The per-cell evidence table behind the decisions.  ``delta_on_minus_off`` is
#: signed the same way as the adjudication reads it: negative favours the
#: component.  It is emitted per cell rather than per median so a reader can see
#: which cell carried a decision, and so the median in
#: ``COMPONENT_DECISIONS.csv`` can be checked against its own members.
DELTA_FIELDS = (
    "component", "role", "metric", "cell", "market", "host",
    "on_config", "off_config", "on_value", "off_value",
    "delta_on_minus_off", "favours",
)

#: The unpooled online view.  It carries no rank and no win flag: the moment a
#: column named "win" existed here, the two inference settings would start being
#: compared as if they were the same measurement.
SUPPLEMENTARY_FIELDS = (
    "coordinate_id", "market", "host",
    "supplementary_method", "supplementary_setting", "supplementary_parity_role",
    "supplementary_Overall_MAE",
    "our_method_Overall_MAE", "our_method_seeds",
    "pooled_with_strict_rank", "note",
)


def _host_rows(strict: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, str], Mapping[str, str]]:
    return {(r["market"], r["host"]): r for r in strict if r.get("method") == "Host"}


def baseline_comparison(cells: Sequence[Mapping[str, Any]], config: str,
                        strict: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    """One row per cell: our cell-median method against the frozen strict table.

    The comparator is the **best available numeric, setting-matched strict
    offline** row for that cell.  "Setting-matched" is the frozen panel's own
    category (:data:`config.SETTING_MATCHED_STRICT`): a static offline post-hoc
    corrector, which is what our method is.  ``MatchedDirectResidual`` is
    registered as a *control* in a different setting, so it is reported in its own
    columns and is never ranked -- the panel's protocol forbids a mixed best
    baseline, and the control's numbers would only ever make our margin look
    larger, so admitting it could not be defended as neutral.

    Neither does a non-numeric row enter the rank: UEC-STD, OMPB and PIR outside
    its qualified cells are blockers, and a blocker is an outcome.  Substituting
    a number for one would invent evidence.  COSA is online supplementary and is
    deliberately absent here; it is reported, unpooled, in its own view.
    """
    ours = _cell_map(cells, config)
    hosts = _host_rows(strict)

    by_cell: Dict[Tuple[str, str], List[Mapping[str, str]]] = {}
    controls: Dict[Tuple[str, str], Mapping[str, str]] = {}
    for row in strict:
        if row.get("method") == "Host":
            continue
        if not str(row.get("result_status", "")).startswith("NUMERIC"):
            continue
        if _number(row.get("Overall_MAE")) is None:
            continue
        key = (row["market"], row["host"])
        if str(row.get("setting")) != C.SETTING_MATCHED_STRICT:
            if str(row.get("setting")) == C.CONTROL_SETTING:
                controls[key] = row
            continue
        by_cell.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for coordinate in sorted(ours):
        market, host = coordinate.split("::", 1)
        row = ours[coordinate]
        host_row = hosts.get((market, host), {})
        host_mae = _number(host_row.get("Overall_MAE"))
        host_tail = _number(host_row.get("Tail_MAE"))
        host_normal = _number(host_row.get("Normal_region_MAE"))

        candidates = by_cell.get((market, host), [])
        best = min(candidates, key=lambda r: _number(r["Overall_MAE"])) if candidates else None
        control = controls.get((market, host))

        our_mae = _number(row.get("overall_mae"))
        our_tail = _number(row.get("tail_mae"))
        our_normal = _number(row.get("normal_mae"))
        gain = (None if (our_mae is None or not host_mae)
                else 100.0 * (host_mae - our_mae) / host_mae)
        tail_gain = (None if (our_tail is None or not host_tail)
                     else 100.0 * (host_tail - our_tail) / host_tail)
        harm = (None if (our_normal is None or not host_normal)
                else 100.0 * (our_normal - host_normal) / host_normal)

        best_mae = _number(best["Overall_MAE"]) if best is not None else None
        out.append({
            "coordinate_id": f"{market}::{host}",
            "market": market, "host": host,
            "our_method_Overall_MAE": _fmt(our_mae),
            "our_method_Tail_MAE": _fmt(our_tail),
            "our_method_Upper_tail_MAE": _fmt(row.get("upper_tail_mae")),
            "our_method_Lower_tail_MAE": _fmt(row.get("lower_tail_mae")),
            "our_method_Normal_region_MAE": _fmt(our_normal),
            "our_method_negative_price_MAE": _fmt(row.get("negative_price_mae")),
            "our_method_n_days": _fmt(row.get("n_days")),
            "our_method_seeds": row.get("seeds", ""),
            "host_overall_mae": _fmt(host_mae),
            "host_tail_mae": _fmt(host_tail),
            "host_normal_mae": _fmt(host_normal),
            "our_gain_vs_host_pct": _fmt(gain),
            "our_tail_gain_vs_host_pct": _fmt(tail_gain),
            "our_normal_harm_pct": _fmt(harm),
            "host_nonworse": (None if (our_mae is None or host_mae is None)
                              else bool(our_mae <= host_mae)),
            "best_strict_comparator": (best["method"] if best is not None else None),
            "best_strict_comparator_setting": (best["setting"] if best is not None else None),
            "best_strict_comparator_Overall_MAE": _fmt(best_mae),
            "best_strict_comparator_Tail_MAE": (
                _fmt(_number(best.get("Tail_MAE"))) if best is not None else None),
            "best_strict_comparator_status": (
                best.get("result_status") if best is not None else None),
            "strict_comparator_count": len(candidates),
            "strict_win": (None if (our_mae is None or best_mae is None)
                           else bool(our_mae < best_mae)),
            "strict_margin_pct": (None if (our_mae is None or not best_mae)
                                  else _fmt(100.0 * (best_mae - our_mae) / best_mae)),
            "strict_margin_abs": (None if (our_mae is None or best_mae is None)
                                  else _fmt(best_mae - our_mae)),
            "control_method": (control["method"] if control is not None else None),
            "control_Overall_MAE": _fmt(_number(control["Overall_MAE"])
                                        if control is not None else None),
            "control_status": (control.get("result_status")
                               if control is not None else None),
            "our_gain_vs_control_pct": (
                _fmt(100.0 * (_number(control["Overall_MAE"]) - our_mae)
                     / _number(control["Overall_MAE"]))
                if (control is not None and our_mae is not None
                    and _number(control["Overall_MAE"])) else None),
            # Derived from the two selections rather than asserted.  The pair
            # falls out of one pass, so "the control is not the comparator" is
            # the only statement about it that the pass cannot make true by
            # construction -- and it is checked independently in
            # ``verify_results``, which re-selects without this function.
            "control_excluded_from_rank": (control is None or best is None
                                           or control is not best),
            "inference_setting_disclosure": C.INFERENCE_SETTING_NOTE,
        })
    return out


def supplementary_view(cells: Sequence[Mapping[str, Any]], config: str,
                       online: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    """COSA, per cell, never pooled and never ranked against the strict table."""
    ours = _cell_map(cells, config)
    out: List[Dict[str, Any]] = []
    for row in online:
        coordinate = f"{row['market']}::{row['host']}"
        mine = ours.get(coordinate, {})
        out.append({
            "coordinate_id": coordinate,
            "market": row["market"], "host": row["host"],
            "supplementary_method": row.get("method"),
            "supplementary_setting": row.get("setting"),
            "supplementary_parity_role": row.get("parity_role"),
            "supplementary_Overall_MAE": row.get("Overall_MAE"),
            "our_method_Overall_MAE": _fmt(_number(mine.get("overall_mae"))),
            "our_method_seeds": mine.get("seeds", ""),
            "pooled_with_strict_rank": False,
            "note": ("COSA is an online test-time-adaptation result; it is shown "
                     "beside ours and never inside the strict offline ranking."),
        })
    return out


# --------------------------------------------------------------------------
# PART H -- the development paper-candidate gate
# --------------------------------------------------------------------------
def development_gates(comparison: Sequence[Mapping[str, Any]],
                      cells: Sequence[Mapping[str, Any]], config: str,
                      decisions: Sequence[Mapping[str, Any]],
                      vector: Mapping[str, Any]) -> Dict[str, Any]:
    """The six registered gates, computed from the joined comparison table."""
    n = len(comparison)
    nonworse = [r for r in comparison if r.get("host_nonworse") is True]
    gains = [v for v in (_number(r.get("our_gain_vs_host_pct")) for r in comparison)
             if v is not None]
    wins = [r for r in comparison if r.get("strict_win") is True]
    tail_gains = [v for v in (_number(r.get("our_tail_gain_vs_host_pct"))
                              for r in comparison) if v is not None]
    harms = [v for v in (_number(r.get("our_normal_harm_pct")) for r in comparison)
             if v is not None]

    # Gate 4 is restricted to cells whose frozen tail subset is non-empty: a cell
    # with no tail entries has no tail metric, and counting it as a pass would
    # be counting an absence as a success.
    tail_cells = [r for r in comparison
                  if _number(r.get("our_method_Tail_MAE")) is not None]
    tail_gain_gate = [v for v in (_number(r.get("our_tail_gain_vs_host_pct"))
                                  for r in tail_cells) if v is not None]

    # Gate 6 -- one recipe.  The switch vector is identical by construction; what
    # can still differ is which components survived, and it is checked from the
    # decision table rather than assumed.
    recipe = {
        "switch_vectors": sorted({r["switches"] for r in cells if r["config"] == config}),
        "seeds": [int(s) for s in C.TRAINING["seeds"]],
        "loss_weights": dict(C.LOSS_WEIGHTS),
        "retained_components": list(vector["retained_components"]),
        "n_cells": len({str(r["cell"]) for r in cells if r["config"] == config}),
        "legal_dimension_differences": ["n_features", "n_episodes", "n_days"],
    }
    recipe_ok = (len(recipe["switch_vectors"]) == 1
                 and recipe["n_cells"] == 20
                 and len(decisions) == len(C.SWITCH_NAMES))

    gates = {
        "gate_1_host_nonworse_19_of_20": {
            "value": len(nonworse), "required": 19, "of": n,
            "pass": len(nonworse) >= 19},
        "gate_2_median_overall_gain_vs_host_at_least_3pct": {
            "value": _fmt(_median(gains)), "required": 3.0,
            "pass": (_median(gains) is not None and _median(gains) >= 3.0)},
        "gate_3_strict_win_14_of_20": {
            "value": len(wins), "required": 14, "of": n,
            "pass": len(wins) >= 14},
        "gate_4_median_tail_gain_vs_host_at_least_5pct": {
            "value": _fmt(_median(tail_gain_gate)), "required": 5.0,
            "n_cells_with_tail_subset": len(tail_cells),
            "pass": (_median(tail_gain_gate) is not None
                     and _median(tail_gain_gate) >= 5.0
                     and len(tail_cells) > 0)},
        "gate_5_max_normal_harm_at_most_1pct": {
            "value": _fmt(max(harms) if harms else None), "required": 1.0,
            "pass": bool(harms) and max(harms) <= 1.0},
        "gate_6_one_recipe": {
            "value": recipe, "pass": recipe_ok},
    }
    all_pass = all(g["pass"] for g in gates.values())
    return {"gates": gates, "all_pass": all_pass, "n_cells": n,
            "median_gain_pct": _median(gains),
            "median_tail_gain_pct": _median(tail_gain_gate),
            "max_normal_harm_pct": max(harms) if harms else None,
            "tail_gain_values": tail_gains}


def verdict_token(gates: Mapping[str, Any]) -> str:
    return C.VERDICT_CANDIDATE if gates["all_pass"] else C.VERDICT_NOT_SUPPORTED


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(records: Sequence[Mapping[str, Any]],
        config: Optional[str] = None,
        strict: Optional[Sequence[Mapping[str, str]]] = None,
        online: Optional[Sequence[Mapping[str, str]]] = None,
        write: bool = True) -> Dict[str, Any]:
    """Adjudicate, freeze, compare and gate -- from persisted records only."""
    cells = cell_table(records)
    decisions = component_decisions(records, cells)
    vector = frozen_vector(decisions)
    frozen_config = config or vector["matches_registered_config"] or "FROZEN"

    # Which table the rows came from is recorded rather than assumed: a caller
    # that injected rows in memory would otherwise get a provenance line naming a
    # file it never read, and the digest beside it would hash that file.
    strict_source = C.STRICT_OFFLINE_TABLE if strict is None else None
    online_source = C.ONLINE_SUPPLEMENTARY_TABLE if online is None else None
    if strict is None:
        strict = read_csv(C.STRICT_OFFLINE_TABLE)
    if online is None:
        online = read_csv(C.ONLINE_SUPPLEMENTARY_TABLE)
    if not strict:
        raise HarnessError(
            f"the frozen strict offline table is missing or empty: "
            f"{C.STRICT_OFFLINE_TABLE}; the comparison cannot be invented")

    comparison = baseline_comparison(cells, frozen_config, strict)
    if len(comparison) != 20:
        raise HarnessError(
            f"the comparison covers {len(comparison)} cells, not 20; the unit of "
            "evidence is the full 5-market x 4-Host panel")
    gates = development_gates(comparison, cells, frozen_config, decisions, vector)
    token = verdict_token(gates)

    payload = {
        "schema": "signed_mass_development_verdict.v1",
        "config": frozen_config,
        "frozen_vector": vector,
        "component_decisions": decisions,
        "gates": gates,
        "verdict": token,
        "inference_setting": C.INFERENCE_SETTING,
        "inference_setting_note": C.INFERENCE_SETTING_NOTE,
        "protected_final_authorized": False,
        "cell_table_rows": len(cells),
    }
    if write:
        out = RE.evidence_dir("00_protocol", "aggregation")
        target = RE.confine(out / "AGGREGATION.json", "aggregation record")
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                       default=str), encoding="utf-8")
        _write_package(cells, decisions, vector, frozen_config, comparison,
                       gates, token, online, strict_source, online_source)

    return {"cells": cells, "decisions": decisions, "vector": vector,
            "frozen_config": frozen_config, "comparison": comparison,
            "gates": gates, "verdict": token, "payload": payload}


# --------------------------------------------------------------------------
# The package
# --------------------------------------------------------------------------
def _write_package(cells, decisions, vector, frozen_config, comparison, gates,
                   token, online, strict_source=None,
                   online_source=None) -> List[str]:
    """Write the required artifacts at the evidence root and under PART E/F/G.

    Every destination is a fixed name under the confined evidence root, and the
    set is written from the same in-memory result the caller already has, so the
    package and the return value cannot disagree.
    """
    written: List[str] = []
    root = RE.evidence_root()

    def _csv(name: str, rows: Sequence[Mapping[str, Any]],
             fields: Sequence[str]) -> None:
        path = RE.confine(root / name, f"aggregate artifact {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, path, fields)
        written.append(name)

    def _json(name: str, body: Mapping[str, Any]) -> None:
        path = RE.confine(root / name, f"aggregate artifact {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2,
                                   sort_keys=True, default=str), encoding="utf-8")
        written.append(name)

    def _text(name: str, body: str) -> None:
        path = RE.confine(root / name, f"aggregate artifact {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        written.append(name)

    # -- 00_protocol: what the frozen method is -----------------------------
    _json("METHOD_CONFIG.json", {
        "schema": "signed_mass_method_config.v1",
        "frozen_config": frozen_config,
        "switches": vector["switches"],
        "retained_components": vector["retained_components"],
        "deleted_components": vector["deleted_components"],
        "registered_config_match": vector["matches_registered_config"],
        "seeds": [int(s) for s in C.TRAINING["seeds"]],
        "loss_weights": dict(C.LOSS_WEIGHTS),
        "inference_setting": C.INFERENCE_SETTING,
        "inference_setting_note": C.INFERENCE_SETTING_NOTE,
        "knn_enabled": False,
        "high_mass_quantile": C.HIGH_MASS_QUANTILE,
        "n_cells": len({str(r["cell"]) for r in cells}),
        "cells": [{"cell": r["cell"], "config": r["config"]} for r in cells],
    })

    # -- 03_component_screen: the per-cell evidence behind each decision ----
    _csv("03_component_screen/COMPONENT_SCREEN.csv", decisions, DECISION_FIELDS)
    deltas = _component_deltas(cells, decisions)
    _csv("03_component_screen/COMPONENT_DELTAS.csv", deltas, DELTA_FIELDS)
    _json("03_component_screen/COMPONENT_DELTAS.json", {
        "schema": "signed_mass_component_deltas.v1",
        "aggregation": "cell median over the registered seeds, per cell",
        "convention": ("delta = ON - OFF, the same sign the adjudication reads: a "
                       "negative delta means the component helped on that cell"),
        "metrics": sorted({d["metric"] for d in deltas}),
        "rows": deltas,
    })

    # -- root tables --------------------------------------------------------
    _csv("CELL_METRICS.csv", cells, CELL_METRIC_FIELDS)
    _csv("COMPONENT_DECISIONS.csv", decisions, DECISION_FIELDS)
    _json("VERDICT.json", {
        "schema": "signed_mass_development_verdict.v1",
        "generated_utc": _iso_now(),
        "config": frozen_config,
        "frozen_vector": vector,
        "gates": gates,
        "verdict": token,
        "n_cells": len(comparison),
        "inference_setting": C.INFERENCE_SETTING,
        "inference_setting_note": C.INFERENCE_SETTING_NOTE,
        "protected_final_authorized": False,
        "protected_final_reads": 0,
        "development_experiment_only": True,
        "note": ("the frozen baseline comparison in "
                 "05_baseline_comparison/ joins frozen comparator rows; no Host "
                 "or baseline was retrained or re-run by this experiment"),
    })

    # -- 05_baseline_comparison --------------------------------------------
    _csv("05_baseline_comparison/STRICT_OFFLINE_WITH_OUR_METHOD.csv", comparison,
         COMPARISON_FIELDS)
    _json("05_baseline_comparison/STRICT_OFFLINE_WITH_OUR_METHOD.json", {
        "schema": "signed_mass_strict_offline_with_our_method.v1",
        "source_table": (None if strict_source is None
                         else _relative(strict_source)),
        "source_table_sha256": (None if strict_source is None
                                else _digest(strict_source)),
        "source": ("read from the frozen table"
                   if strict_source is not None
                   else "INJECTED_IN_MEMORY: the rows were supplied by the "
                        "caller and no frozen table was read"),
        "n_cells": len(comparison),
        "ranking_scope": "STRICT_OFFLINE only",
        "blocked_comparator_policy": (
            "a comparator row whose result_status is not NUMERIC* is a "
            "disclosed blocker and never enters the rank"),
        "inference_setting_disclosure": C.INFERENCE_SETTING_NOTE,
        "rows": comparison,
    })
    supplement = supplementary_view(cells, frozen_config, online)
    _csv("05_baseline_comparison/ONLINE_SUPPLEMENTARY_COSA.csv", supplement,
         SUPPLEMENTARY_FIELDS)
    _json("05_baseline_comparison/ONLINE_SUPPLEMENTARY_COSA.json", {
        "schema": "signed_mass_online_supplementary.v1",
        "source_table": (None if online_source is None
                         else _relative(online_source)),
        "source_table_sha256": (None if online_source is None
                                else _digest(online_source)),
        "source": ("read from the frozen table"
                   if online_source is not None
                   else "INJECTED_IN_MEMORY: the rows were supplied by the "
                        "caller and no frozen table was read"),
        "pooled_with_the_strict_rank": False,
        "why": ("COSA is an online test-time-adaptation result.  It is reported "
                "beside ours for completeness and is never ranked against the "
                "strict offline table, because the two were not produced under "
                "the same inference setting."),
        "rows": supplement,
    })

    # -- the two prose artifacts -------------------------------------------
    _text("METHOD_SUMMARY.md", _render_method_summary(
        cells, decisions, vector, frozen_config, comparison, gates, token))
    _text("NEW_WINDOW_HANDOFF.md", _render_handoff(
        decisions, vector, frozen_config, token, gates))
    return written


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _relative(path: Any) -> str:
    try:
        return str(Path(path).resolve().relative_to(C.REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _digest(path: Any) -> Optional[str]:
    from .contracts import sha256_file

    target = Path(path)
    return sha256_file(target) if target.exists() else None


def _component_deltas(cells: Sequence[Mapping[str, Any]],
                      decisions: Sequence[Mapping[str, Any]]
                      ) -> List[Dict[str, Any]]:
    """The per-cell ON/OFF evidence behind every decision, in one long table.

    A metric can be registered at more than one rank (``use_tcn`` registers
    ``tail_mae`` as both its primary and its secondary evidence).  It is emitted
    once, at its strongest rank, because the same number entered twice under two
    headings reads as two independent confirmations.
    """
    rank = {"primary": 0, "secondary": 1, "diagnostic": 2}
    out: List[Dict[str, Any]] = []
    for decision in decisions:
        component = str(decision["component"])
        spec = COMPONENT_SPECS[component]
        on_map = _cell_map(cells, spec["on"])
        off_map = _cell_map(cells, spec["off"])

        order: Dict[str, int] = {}
        chosen: Dict[str, str] = {}
        for role in ("primary", "secondary", "diagnostic"):
            for metric in spec[role]:
                order.setdefault(metric, len(order))
                if metric not in chosen or rank[role] < rank[chosen[metric]]:
                    chosen[metric] = role

        for metric in sorted(chosen, key=lambda m: (rank[chosen[m]], order[m])):
            role = chosen[metric]
            for cell in sorted(set(on_map) & set(off_map)):
                on = _number(on_map[cell].get(metric))
                off = _number(off_map[cell].get(metric))
                delta = None if (on is None or off is None) else on - off
                out.append({
                    "component": component, "role": role, "metric": metric,
                    "cell": cell,
                    "market": on_map[cell].get("market"),
                    "host": on_map[cell].get("host"),
                    "on_config": spec["on"], "off_config": spec["off"],
                    "on_value": _fmt(on), "off_value": _fmt(off),
                    "delta_on_minus_off": _fmt(delta),
                    "favours": (None if delta is None
                                else "ON" if delta < 0.0
                                else "OFF" if delta > 0.0 else "TIE"),
                })
    return out


# --------------------------------------------------------------------------
# The prose artifacts
# --------------------------------------------------------------------------
def _cell(x: Any, digits: int = 4) -> str:
    value = _number(x)
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(x: Any, digits: int = 3) -> str:
    value = _number(x)
    return "n/a" if value is None else f"{value:+.{digits}f}%"


def _yes(x: Any) -> str:
    return "yes" if x is True else "no" if x is False else "n/a"


def _render_method_summary(cells, decisions, vector, frozen_config, comparison,
                           gates, token) -> str:
    """The one-page account of what was run, what it found, and what it claims."""
    lines: List[str] = []
    add = lines.append

    add("# Signed-mass method — development experiment summary")
    add("")
    add(f"Generated: {_iso_now()}")
    add("")
    add("This is a **development** experiment on the China-5 panel.  Every number "
        "below was produced on `DEV_EVAL` of the 20 registered market x Host "
        "cells, under the frozen method, with the frozen Host predictions left "
        "untouched.  `PROTECTED_FINAL` was not opened and is not authorized.")
    add("")
    add("## Inference setting")
    add("")
    add(f"`{C.INFERENCE_SETTING}` — {C.INFERENCE_SETTING_NOTE}.")
    add("")
    add("This is **not** online test-time adaptation.  A frozen-parameter "
        "post-processor and an online TTA method are not the same inference "
        "setting, and their numbers are never ranked against each other.")
    add("")

    add("## The frozen method")
    add("")
    add(f"Configuration: `{frozen_config}`  ·  components retained: "
        f"{', '.join(vector['retained_components']) or 'none'}  ·  deleted: "
        f"{', '.join(vector['deleted_components']) or 'none'}")
    add("")
    add("| switch | state |")
    add("|---|---|")
    for name in C.SWITCH_NAMES:
        add(f"| `{name}` | {'ON' if vector['switches'][name] else 'OFF'} |")
    add("")
    add(f"Seeds (cell median over all three, never a best seed): "
        f"{', '.join(str(int(s)) for s in C.TRAINING['seeds'])}.  "
        f"KNN: off.  High-mass subset: q{int(round(C.HIGH_MASS_QUANTILE * 100))} "
        f"of realised mass, fitted per cell on `POST_TRAIN` only and frozen "
        f"before any `DEV_EVAL` metric was computed.")
    add("")

    add("## Component adjudication")
    add("")
    add("| component | decision | primary evidence (median delta, ON − OFF) | "
        "cells improved | reason |")
    add("|---|---|---|---|---|")
    for decision in decisions:
        add(f"| `{decision['component']}` | **{decision['decision']}** | "
            f"`{decision['per_primary_median_delta']}` | "
            f"{decision['n_cells_primary_improved']} of {decision['n_cells']} | "
            f"{decision['reason']} |")
    add("")
    add("A component is deleted unless all five pre-registered conditions held on "
        "its own registered primary metric.  No rescue variant was run, no "
        "per-market exemption was applied, and no metric was substituted after "
        "the fact.")
    add("")

    add("## Per-cell results")
    add("")
    add(f"{len(comparison)} cells.  All values are cell medians over the three "
        "registered seeds; every cell was evaluated on its complete `DEV_EVAL` "
        "partition.")
    add("")
    add("| cell | Overall MAE | Host | gain vs Host | Tail | Host tail | "
        "Upper | Lower | Normal | Host normal | normal harm | neg-price | "
        "best strict comparator | margin | control (not-ranked) | gain vs control |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in comparison:
        add("| {c} | {m} | {h} | {g} | {t} | {ht} | {u} | {l} | {n} | {hn} | "
            "{nh} | {np} | {bc} | {mg} | {ct} | {cg} |".format(
                c=row["coordinate_id"],
                m=_cell(row["our_method_Overall_MAE"]),
                h=_cell(row["host_overall_mae"]),
                g=_pct(row["our_gain_vs_host_pct"]),
                t=_cell(row["our_method_Tail_MAE"]),
                ht=_cell(row["host_tail_mae"]),
                u=_cell(row["our_method_Upper_tail_MAE"]),
                l=_cell(row["our_method_Lower_tail_MAE"]),
                n=_cell(row["our_method_Normal_region_MAE"]),
                hn=_cell(row["host_normal_mae"]),
                nh=_pct(row["our_normal_harm_pct"]),
                np=_cell(row["our_method_negative_price_MAE"]),
                bc=(f"{row['best_strict_comparator']} "
                    f"({_cell(row['best_strict_comparator_Overall_MAE'])})"
                    if row.get("best_strict_comparator") else "n/a"),
                mg=_pct(row["strict_margin_pct"]),
                ct=(f"{row['control_method']} "
                    f"({_cell(row['control_Overall_MAE'])})"
                    if row.get("control_method") else "n/a"),
                cg=_pct(row["our_gain_vs_control_pct"])))
    add("")
    add("`gain vs Host` is positive when our method is better; `normal harm` is "
        "positive when it is worse on the normal region.  `margin` is our "
        "advantage over the best numeric, setting-matched **strict offline** "
        "comparator for that cell; blocked comparator rows (a disclosed "
        "`Q-FIDELITY`/`Q-DATA`/`Q-INCOMPATIBLE` outcome) never enter that rank.")
    add("")
    add(f"The `control` column is `{C.CONTROL_SETTING}`, a different measurement "
        "setting, and it is reported *beside* the comparison rather than inside "
        "it.  Our method post-processes a frozen Host and never updates a "
        "parameter during `DEV_EVAL`, which is "
        f"`{C.SETTING_MATCHED_STRICT}`; the control is a deliberately different "
        "setting, and the comparator panel's own protocol forbids a cross-setting "
        "`mixed best baseline` claim.  The rule fixes the direction of the "
        "comparison before any of our numbers existed -- the control is far worse "
        "than its own Host in most cells, so admitting it to the rank would "
        "*loosen* the gate, not tighten it.")
    add("")

    add("## Where it fails")
    add("")
    host_worse = [r["coordinate_id"] for r in comparison
                  if r.get("host_nonworse") is False]
    strict_loss = [r["coordinate_id"] for r in comparison
                   if r.get("strict_win") is False]
    add(f"- cells worse than their own Host on Overall MAE "
        f"({len(host_worse)}): {', '.join(host_worse) if host_worse else 'none'}")
    add(f"- cells not strictly better than the best strict offline comparator "
        f"({len(strict_loss)}): {', '.join(strict_loss) if strict_loss else 'none'}")
    add("")
    for heading, coordinates in (("worse than their own Host", host_worse),
                                 ("not strictly better than the best strict "
                                  "comparator", strict_loss)):
        for label, index in (("market", 0), ("Host", 1)):
            counts: Dict[str, int] = {}
            for coordinate in coordinates:
                key = coordinate.split("::")[index]
                counts[key] = counts.get(key, 0) + 1
            add(f"- {heading}, by {label}: "
                + (", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                   if counts else "none"))
    add("")

    add("## Development gate")
    add("")
    add("| gate | value | required | result |")
    add("|---|---|---|---|")
    for name, gate in gates["gates"].items():
        value = gate["value"]
        if isinstance(value, dict):
            value = (f"{value.get('n_cells')} cells, "
                     f"{len(value.get('switch_vectors', []))} switch vector(s)")
        add(f"| `{name}` | {value} | {gate.get('required', '')} | "
            f"{'PASS' if gate['pass'] else 'FAIL'} |")
    add("")
    add(f"All gates pass: **{_yes(gates['all_pass'])}**.")
    add("")

    add("## Verdict")
    add("")
    add(f"`{token}`")
    add("")
    add("The independent audit of this run, together with the three corrections "
        "made to the audit instrument after its first pass, is reported in "
        "`FINAL_AUDIT.md` and `06_audits/RESULT_AUDIT.json`.  Those corrections "
        "changed what the audit could see and one tolerance inside it; they "
        "changed no metric, no component decision and no gate.")
    add("")
    add("Development experiment only.  No claim here is stronger than the gate "
        "table above: a component that failed is deleted rather than rescued, "
        "and a gate that failed is reported as failed rather than relabelled.")
    add("")
    return "\n".join(lines) + "\n"


def _render_handoff(decisions, vector, frozen_config, token, gates) -> str:
    """What the next window inherits, and what it must not assume."""
    lines: List[str] = []
    add = lines.append

    add("# New window handoff — signed-mass method")
    add("")
    add(f"Written: {_iso_now()}")
    add("")
    add("## What is now frozen")
    add("")
    add(f"- The method is one switch vector: `{frozen_config}`, "
        f"{vector['switches']}.")
    add(f"- Retained components: {', '.join(vector['retained_components']) or 'none'}.")
    add(f"- Deleted components: {', '.join(vector['deleted_components']) or 'none'} "
        "(deleted on pre-registered evidence; they are not candidates for rescue).")
    add(f"- Terminal development verdict: `{token}`.")
    add("- Seeds `7/17/37`, aggregated by cell median.  The recipe is identical "
        "in all 20 cells; only legal feature width differs between markets.")
    add("")
    add("## Component decisions")
    add("")
    for decision in decisions:
        add(f"- `{decision['component']}`: **{decision['decision']}** — "
            f"{decision['reason']}.")
    add("")
    add("## Hard constraints this window did not lift")
    add("")
    add("- `PROTECTED_FINAL` was **not** opened.  Read count is 0 and the "
        "authorization is still not granted.")
    add("- No Host and no baseline comparator was retrained or re-run.  The Host "
        "predictions are the frozen artifacts; the strict offline comparison "
        "joins the frozen table.")
    add("- Every statistic that could touch the evaluation partition — the OOF "
        "scalar `alpha`, the robust amplitude scale `s_A`, and the per-cell "
        "high-mass threshold — was fitted on `POST_TRAIN` only.")
    add("")
    add("## What a next window may and may not do")
    add("")
    add("- **May**: read this package, re-derive every number from the raw "
        "artifacts under `02_crossfit/raw` and `04_frozen_method/raw`, and run "
        "the independent verifier in `06_audits/`.")
    add("- **May not**: treat this development result as a paper claim.  The "
        "gates below are development gates.")
    add("- **May not**: re-run comparators to improve the comparison, or "
        "re-adjudicate a deleted component on a different metric.")
    add("")
    add("## Gate state at handoff")
    add("")
    for name, gate in gates["gates"].items():
        add(f"- `{name}`: {'PASS' if gate['pass'] else 'FAIL'}")
    add("")
    add(f"All gates pass: **{_yes(gates['all_pass'])}**.  Any claim built on this "
        "window must be no stronger than this line.")
    add("")
    return "\n".join(lines) + "\n"
