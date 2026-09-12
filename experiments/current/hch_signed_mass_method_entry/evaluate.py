"""One registered fit, evaluated once on the full ``DEV_EVAL`` partition (PART D).

This module is the scientific unit of work of the experiment: it takes a cell,
a registered configuration name and a seed, runs the frozen fitting chain,
evaluates on **every** ``DEV_EVAL`` row of that cell exactly once, and persists
two artifacts that must agree:

* the **raw** arrays -- frozen Host prediction, direct branch outputs, raw
  correction, ``alpha``, repaired prediction, valid mask, identity -- written by
  :mod:`raw_evidence` through its confinement guard; and
* the **metric** record computed from those very arrays in the same call.

The split matters.  A metric file on its own cannot be checked, because the
numbers in it are the only witness that the numbers are right.  With the arrays
persisted alongside, an independent verifier recomputes every reported number
from the raw object and must land on the same value; disagreement is a finding,
not a rounding note.

Nothing here decides anything: the configuration, the seed and the epoch rule
all arrive from the caller, and no number produced here is read back into a
fitting step.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

import numpy as np

from . import config as C
from . import metrics as M
from . import oof
from . import raw_evidence as RE
from . import training as T
from .china5_adapter import CellDataset
from .contracts import HarnessError, LegalityError

__all__ = ["evaluate_fit", "parameter_count", "EVIDENCE_RECORD_FIELDS"]


#: The flat columns of ``GRID_STATUS.csv`` -- one row per registered fit.
EVIDENCE_RECORD_FIELDS = (
    "market", "host", "cell", "config", "seed", "partition",
    "n_days", "n_entries_evaluated", "n_days_any_valid",
    "overall_mae", "host_mae", "mae_gain_abs", "mae_gain_pct",
    "tail_mae", "tail_mae_host", "upper_tail_mae", "lower_tail_mae",
    "normal_mae", "normal_mae_host", "normal_relative_harm_pct",
    "negative_price_mae", "negative_price_mae_host",
    "shape_w1_positive", "shape_w1_negative",
    "mass_positive_l1", "mass_negative_l1",
    "high_mass_positive_l1", "high_mass_negative_l1",
    "high_mass_positive_n_days", "high_mass_negative_n_days",
    "alpha", "amplitude_scale", "final_epochs", "n_folds",
    "n_parameters", "fit_seconds", "inference_seconds",
    "branch_arrays_used", "branch_reconstruction_max_abs_err",
    "alpha_consistency_max_abs_err",
    # The switch vector travels with the row.  Without it the table records a
    # configuration *name*, and a name is a claim: PART F may reuse a registered
    # configuration's persisted predictions only if every row can be shown to
    # carry that exact vector, and the component screen must be able to say what
    # it ran for a label that is not one of the registered five.  A table that
    # has to look the recipe up by name cannot do either.
    "switches",
    "raw_path", "raw_sha256", "status", "failure",
)


def parameter_count(model) -> int:
    """Trainable parameter count of the fitted method."""
    return int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def dataset_provenance(dataset: CellDataset) -> Dict[str, Any]:
    """The frozen-artifact digests a raw record must carry to be joinable."""
    prov = dict(dataset.provenance)
    return {
        "market": dataset.market,
        "host": dataset.host,
        "n_features": int(dataset.n_features),
        "n_episodes": int(dataset.n_episodes),
        "contract_source": prov.get("contract_source"),
        "contract_sha256": prov.get("contract_sha256"),
        "source_table_sha256": prov.get("source_table_sha256"),
        "source_recorded_sha256": prov.get("source_recorded_sha256"),
        "host_artifact_path": prov.get("host_artifact_path"),
        "host_artifact_container_sha256": prov.get("host_artifact_container_sha256"),
        "host_artifact_array_sha256": prov.get("host_artifact_array_sha256"),
        "role_taxonomy": prov.get("role_taxonomy"),
        "lead_convention": prov.get("lead_convention"),
    }


def _source_digest() -> str:
    """One digest over every harness module that took part in this fit."""
    import hashlib

    payload = hashlib.sha256()
    for name in sorted(p.name for p in Path(__file__).resolve().parent.glob("*.py")):
        payload.update(name.encode("utf-8"))
        payload.update((Path(__file__).resolve().parent / name).read_bytes())
    return payload.hexdigest().upper()


# --------------------------------------------------------------------------
# One registered fit
# --------------------------------------------------------------------------
def evaluate_fit(dataset: CellDataset,
                 config_name: str,
                 seed: int,
                 high_mass: Optional[Mapping[str, Any]] = None,
                 raw_subdir: str = C.RAW_DIR_GRID,
                 metric_subdir: str = C.METRIC_DIR_GRID,
                 partition: str = C.ROLE_DEV_EVAL,
                 max_epochs: Optional[int] = None,
                 switches: Optional[Mapping[str, bool]] = None,
                 write: bool = True) -> Dict[str, Any]:
    """Fit one ``(cell, config, seed)`` and evaluate it on the whole partition.

    Returns the flat record; when ``write`` is set, the raw artifact, the metric
    record and the flat evidence row are also persisted under the experiment's
    own evidence root.

    ``high_mass`` is the cell's frozen ``POST_TRAIN`` q90 pair.  It is computed
    once per cell by the caller and passed in, so the threshold cannot vary
    between the configurations being compared -- which is the entire point of
    freezing it before the first ``DEV_EVAL`` number exists.

    ``switches`` is how PART F's derived vector is fitted.  When it is supplied
    the configuration name is a *label* for that vector and need not be one of the
    registered five -- after two or more components are deleted the smallest
    surviving method is generally not a registered ablation, and refusing to name
    it would force the frozen run through a different code path than the screen.
    When it is absent the name must be registered, so the ordinary grid cannot
    invent a recipe by typo.
    """
    if switches is None:
        if config_name not in C.CONFIG_VARIANTS:
            raise HarnessError(f"unknown configuration {config_name!r}")
        switches = C.CONFIG_VARIANTS[config_name]
    else:
        switches = {str(k): bool(v) for k, v in switches.items()}
        if set(switches) != set(C.SWITCH_NAMES):
            raise HarnessError(
                f"switch vector for {config_name!r} is not over the registered "
                f"switches {sorted(C.SWITCH_NAMES)}: {sorted(switches)}")

    if partition != C.ROLE_DEV_EVAL:
        raise LegalityError(
            f"the registered component grid evaluates on {C.ROLE_DEV_EVAL}; "
            f"{partition} is not an admissible development partition")
    if partition in C.SEALED_ROLES:
        raise LegalityError(f"{partition} is structurally sealed")

    rows = dataset.positions(partition)
    if rows.size == 0:
        raise LegalityError(f"{dataset.key}: {partition} has no rows")

    if high_mass is None:
        high_mass = oof.fit_high_mass_thresholds(dataset)

    started = time.time()
    method = oof.fit_method(dataset, switches, int(seed), max_epochs=max_epochs)
    fit_seconds = time.time() - started

    started = time.time()
    out = method.predict(dataset, rows)
    inference_seconds = time.time() - started

    prediction = out["prediction"]
    target = out["target"]
    host = out["host"]
    valid = out["valid_mask"]

    block = M.compute_metrics(target, host, prediction, valid,
                              market=dataset.market,
                              branch=out,
                              high_mass=high_mass,
                              alpha=method.alpha)

    model, _ = method.model_and_builder()
    n_parameters = parameter_count(model)

    # Provenance is computed once and then written both nested (for a human
    # reading the artifact) and flat under the index's own field names (so the
    # artifact can produce its RAW_PREDICTION_INDEX row by itself).  Two
    # computations of the same digest could disagree; one cannot.
    prov = dataset_provenance(dataset)
    core_prov = _core_provenance_safe()
    git_state = _git_state_safe()
    hm_label = f"q{int(round(float(high_mass['quantile']) * 100))}"

    meta: Dict[str, Any] = {
        "schema": "signed_mass_raw_prediction.v1",
        "generated_utc": _utc_now(),
        "config": config_name,
        "switches": dict(switches),
        "seed": int(seed),
        "partition": partition,
        "alpha": float(method.alpha),
        "alpha_objective": method.alpha_objective,
        "amplitude_scale": float(method.stats.amplitude_scale),
        "amplitude_scale_rule": method.stats.amplitude_scale_rule,
        "mass_center": list(method.stats.mass_center),
        "mass_scale": list(method.stats.mass_scale),
        "residual_scale": float(method.stats.residual_scale),
        "final_fit_rows_sha256": method.stats.fit_rows_sha256,
        "final_fit_rows": int(method.stats.n_fit_rows),
        "final_epochs": int(method.epoch_rule["value"]),
        "epoch_rule": method.epoch_rule,
        "n_folds": int(len(method.folds)),
        "oof_plan": method.oof_plan,
        "oof_correction_sha256": method.oof_correction_sha256,
        "high_mass_thresholds": dict(high_mass),
        "inference_setting": C.INFERENCE_SETTING,
        "inference_setting_note": C.INFERENCE_SETTING_NOTE,
        "knn_enabled": False,
        "n_parameters": n_parameters,
        "fit_seconds": round(fit_seconds, 3),
        "inference_seconds": round(inference_seconds, 6),
        "row_positions": [int(x) for x in rows],
        "identity": out["identity"],
        "dataset": prov,
        "source_sha256": _source_digest(),
        "core": core_prov,
        "git": git_state,
        "metric_definitions": M.definitions(),

        # -- flat, index-ready identity and provenance ---------------------
        # These duplicate values that are also reachable nested above.  The
        # duplication is deliberate: the raw-prediction index is built from the
        # artifact alone, so a reader holding one `.npz` must not have to know
        # the shape of `dataset` or `core` in order to place it in the panel.
        "role": C.ROLE_OUR_METHOD,
        "market": dataset.market,
        "host": dataset.host,
        "cell": dataset.key,
        "n_days": block.get("n_days"),
        "n_entries_evaluated": block.get("n_entries_evaluated"),
        "n_days_any_valid": block.get("n_days_any_valid"),
        "high_mass_q90_positive": float(high_mass.get(f"{hm_label}_positive", 0.0)),
        "high_mass_q90_negative": float(high_mass.get(f"{hm_label}_negative", 0.0)),
        "frozen_host_artifact_sha256": prov.get("host_artifact_container_sha256"),
        "dataset_contract_sha256": prov.get("contract_sha256"),
        "core_provenance_sha256": core_prov.get("core_tree_digest"),
        "git_commit": git_state.get("head"),
    }

    record: Dict[str, Any] = {
        "market": dataset.market,
        "host": dataset.host,
        "cell": dataset.key,
        "config": config_name,
        "seed": int(seed),
        "partition": partition,
        "status": "OK",
        "failure": None,
        "n_parameters": n_parameters,
        "fit_seconds": round(fit_seconds, 3),
        "inference_seconds": round(inference_seconds, 6),
        "alpha": float(method.alpha),
        "amplitude_scale": float(method.stats.amplitude_scale),
        "final_epochs": int(method.epoch_rule["value"]),
        "n_folds": int(len(method.folds)),
        "raw_path": None,
        "raw_sha256": None,
        "switches": dict(switches),
    }
    for field in EVIDENCE_RECORD_FIELDS:
        if field in block:
            record[field] = block[field]
    record["switches"] = dict(switches)

    raw_arrays = {
        "target": target,
        "host_prediction": host,
        "shape_positive": out["shape_positive"],
        "shape_negative": out["shape_negative"],
        "mass_positive": out["mass_positive"],
        "mass_negative": out["mass_negative"],
        "correction": out["correction"],
        "prediction": prediction,
        "valid_mask": valid.astype(np.uint8),
    }

    if write:
        raw_path = RE.raw_artifact_path(dataset.market, dataset.host, config_name,
                                        seed, partition, raw_subdir)
        written = RE.write_raw_artifact(raw_arrays, meta, raw_path)
        record["raw_path"] = _repo_relative(Path(written["raw_path"]))
        record["raw_sha256"] = written["raw_sha256"]
        record["raw_bytes"] = written["raw_bytes"]

        metric_path = RE.confine(
            RE.evidence_root() / metric_subdir /
            (Path(written["raw_path"]).stem + ".json"), "metric record")
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(json.dumps(
            {"schema": "signed_mass_metric_record.v1",
             "record": record, "metrics": block, "meta": meta},
            ensure_ascii=False, indent=2, sort_keys=True, default=str),
            encoding="utf-8")
        record["metric_path"] = _repo_relative(metric_path)

    return record


def _repo_relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(C.REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _core_provenance_safe() -> Dict[str, Any]:
    from .core_bridge import core_provenance

    try:
        return core_provenance()
    except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
        return {"error": repr(exc)}


def _git_state_safe() -> Dict[str, Any]:
    import subprocess

    def _run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", *args], cwd=str(C.REPO_ROOT),
                                 capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    return {"head": _run("rev-parse", "HEAD"),
            "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_run("status", "--porcelain"))}


def raw_index_row(record: Mapping[str, Any],
                  meta: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``RAW_PREDICTION_INDEX.csv`` row for one evaluated fit."""
    dataset = dict(meta.get("dataset") or {})
    core = dict(meta.get("core") or {})
    git = dict(meta.get("git") or {})
    thresholds = dict(meta.get("high_mass_thresholds") or {})
    return {
        "market": record["market"],
        "host": record["host"],
        "cell": record["cell"],
        "config": record["config"],
        "seed": record["seed"],
        "partition": record["partition"],
        "role": record["partition"],
        "n_days": record.get("n_days"),
        "n_entries_evaluated": record.get("n_entries_evaluated"),
        "n_days_any_valid": record.get("n_days_any_valid"),
        "raw_path": record.get("raw_path"),
        "raw_sha256": record.get("raw_sha256"),
        "raw_bytes": Path(record["raw_path"]).stat().st_size
        if record.get("raw_path") else None,
        "alpha": record.get("alpha"),
        "amplitude_scale": record.get("amplitude_scale"),
        "high_mass_q90_positive": thresholds.get("q90_positive"),
        "high_mass_q90_negative": thresholds.get("q90_negative"),
        "frozen_host_artifact_sha256": dataset.get("host_artifact_container_sha256"),
        "dataset_contract_sha256": dataset.get("contract_sha256"),
        "source_sha256": meta.get("source_sha256"),
        "core_provenance_sha256": core.get("core_tree_digest"),
        "git_commit": git.get("head"),
        "switches": json.dumps(dict(meta.get("switches") or {}), sort_keys=True,
                               separators=(",", ":")),
        "generated_utc": meta.get("generated_utc"),
    }
