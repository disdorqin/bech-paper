"""Metric and result writer schema for the RCL v4.1 run.

One record per cell x seed x variant, plus a run-level index.  The schema is a
closed field set rather than a free-form dict: a metric the protocol requires but
the writer forgot shows up as a missing key at write time, not as an absent column
nobody notices until adjudication.

Every writer here is **pure Python/json**: it takes already-computed numbers and
writes them.  Nothing in this module reads a dataset, builds a model or fits
anything, so a test can exercise the whole schema without touching a split.

``test_label_read_count`` is carried on every record and every index.  It comes
from :class:`data_boundary.AuditCounters`, whose value has no increment path, so
the field documents the boundary rather than measuring it — the verifier is what
actually re-derives it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

__all__ = [
    "RESULT_SCHEMA",
    "INDEX_SCHEMA",
    "REQUIRED_CELL_FIELDS",
    "METRIC_FIELDS",
    "DIAGNOSTIC_FIELDS",
    "make_cell_record",
    "make_run_index",
    "write_json",
    "sha256_file",
    "ids_sha256",
    "aggregate_closure_ratio",
]

RESULT_SCHEMA = "rcl_v4_1_cell_result.v1"
INDEX_SCHEMA = "rcl_v4_1_run_index.v1"

#: Identity and provenance of one fit.  Required on every record.
REQUIRED_CELL_FIELDS = (
    "schema", "protocol_id", "market", "host", "variant", "seed",
    "n_parameters", "best_epoch", "epochs_run", "fit_seconds",
    "coarse_level_source", "coarse_level_sha256",
    "level_condition_fingerprint",
    "oof_level_architecture_fingerprint", "final_level_architecture_fingerprint",
    "local_target_sha256", "local_fit_ids_sha256",
    "local_target_ids_sha256", "local_history_ids_sha256",
    "initial_state_sha256", "final_state_sha256",
    "test_label_read_count",
)

#: Verdict-bearing metrics (protocol section 8).
#:
#: Every subset metric scores ``|e_R|`` where ``e_R = r - c``; the two elementwise
#: price comparisons the audit found — repaired price against a *residual* — are
#: gone, and the names that carried them went with them.  ``normal_harm`` is the
#: only place a difference of two absolute errors appears, and its sign is
#: ``max(0, |e_R| - |e_H|)``: harm, not improvement.
METRIC_FIELDS = (
    "level_mae", "level_zero_reference_mae", "stage1_floor",
    "delta_mae", "delta_zero_reference_mae", "delta_closure_error",
    "closure_ratio_cell",
    "amplitude_mae", "shape_w1",
    "q_reconstruction_mae",
    "host_overall_mae", "repaired_overall_mae", "gain_vs_host_pct",
    "upper_mae", "lower_mae", "extreme_mae", "tail_mae",
    "normal_mae", "normal_host_mae", "normal_harm",
    "cal_mae", "cal_raw_mae", "cal_host_mae",
)

#: Secondary, non-verdict diagnostics (protocol section 7).
DIAGNOSTIC_FIELDS = (
    "alpha_secondary", "eval_mae_calibrated_secondary",
    "centered_amplitude_gap_host", "centered_amplitude_gap_candidate",
    "centered_shape_mae_host", "centered_shape_mae_candidate",
    "native_mass_plus_mae", "native_mass_minus_mae",
    "emission_precedes_reveal", "n_eval_days",
)


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ids_sha256(ids) -> str:
    """Stable fingerprint of a day-index sequence.

    The digest is over the comma-joined indices, so it identifies the *exact*
    ordered day set a fit trained on.  Two runs whose folds differ by one day
    produce different digests even when their day counts match.
    """
    return _sha256_text(",".join(str(int(d)) for d in ids))


def make_cell_record(*, market: str, host: str, variant: str, seed: int,
                     metrics: Mapping[str, float],
                     diagnostics: Optional[Mapping[str, object]] = None,
                     provenance: Optional[Mapping[str, object]] = None,
                     test_label_read_count: int = 0) -> dict:
    """One cell x seed x variant record.

    Fails on a missing required field or a missing metric.  The failure mode this
    prevents is a partially written record that later reads as "the run did not
    report a Normal harm" rather than "the writer dropped it".
    """
    if test_label_read_count != 0:
        raise ValueError(
            "test_label_read_count must be 0 for this protocol; the run has no "
            "legal path that could raise it")
    record = {"schema": RESULT_SCHEMA, "market": market, "host": host,
              "variant": variant, "seed": int(seed),
              "test_label_read_count": 0}
    record.update(dict(provenance or {}))

    missing = [name for name in REQUIRED_CELL_FIELDS if name not in record]
    if missing:
        raise KeyError(f"cell record is missing required fields: {missing}")
    absent = [name for name in METRIC_FIELDS if name not in metrics]
    if absent:
        raise KeyError(f"cell record is missing required metrics: {absent}")
    record.update({name: metrics[name] for name in METRIC_FIELDS})
    record.update({name: (diagnostics or {}).get(name) for name in DIAGNOSTIC_FIELDS})
    return record


def make_run_index(records: Sequence[Mapping], *, protocol_id: str, counters: Mapping,
                   cells: Sequence[Sequence[str]], seeds: Sequence[int],
                   terminal_state: str,
                   closure: Optional[Mapping[str, float]] = None) -> dict:
    """The run-level index, with the audit block the verifier re-derives.

    ``closure`` carries the protocol's panel-level closure ratio per variant.  It
    is a *panel* aggregate on purpose: a per-cell ratio would let one easy cell
    speak for the mechanism, which is the reading the protocol explicitly avoids.
    """
    expected = len(cells) * len(seeds) * 4
    ordered = sorted((dict(r) for r in records),
                     key=lambda r: (r["market"], r["host"], r["variant"], r["seed"]))
    return {
        "schema": INDEX_SCHEMA,
        "protocol_id": protocol_id,
        "cells": [list(c) for c in cells],
        "seeds": list(seeds),
        "variants": ["R0_STACK_ZERO_SUM", "R1_RCL_ORTHOGONAL",
                     "R2_UNTIED_Q", "R3_DIRECT_Q"],
        "expected_fits": expected,
        "completed_fits": len(records),
        # Hashed in a canonical order rather than in write order, so the verifier
        # can recompute the digest from the record files on disk — which it can
        # only do if the digest does not depend on the order the runner happened
        # to emit them in.
        "records_sha256": _sha256_text(json.dumps(
            ordered, sort_keys=True, default=str)),
        "closure_ratio_by_variant": dict(closure or {}),
        "audit": dict(counters),
        "test_label_read_count": 0,
        "terminal_state": terminal_state,
    }


def write_json(path, payload) -> str:
    """Write a JSON artifact and return its file SHA-256.

    Refuses to overwrite a *differing* existing artifact: a rerun that changes a
    number must produce a new path, so a reported metric cannot be silently
    replaced by a later run under the same name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise FileExistsError(
                f"{path} already exists with different content; evidence artifacts "
                "are immutable under this protocol")
    else:
        path.write_text(text, encoding="utf-8")
    return sha256_file(path)


def aggregate_closure_ratio(level_truth, level_prediction, delta_prediction) -> float:
    """``median|b* - b_hat - delta_hat| / median|b* - b_hat|`` (protocol section 13).

    A ratio below 1 means the Local remainder mean recovered part of the coarse
    Level's error; a ratio at 1 means ``delta`` carried none of it.  Reported as a
    panel aggregate, never per cell, so a single easy cell cannot stand in for the
    panel.
    """
    import numpy as np

    truth = np.asarray(level_truth, dtype=np.float64).reshape(-1)
    predicted = np.asarray(level_prediction, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta_prediction, dtype=np.float64).reshape(-1)
    if not (truth.shape == predicted.shape == delta.shape):
        raise ValueError("closure ratio inputs must have the same length")
    denominator = float(np.median(np.abs(truth - predicted)))
    numerator = float(np.median(np.abs(truth - predicted - delta)))
    if denominator <= 0:
        return float("nan")
    return numerator / denominator
