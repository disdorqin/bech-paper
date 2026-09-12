"""The registered 20 x 5 x 3 component grid (PART D).

Three hundred fits, every one of them evaluated on the **complete** ``DEV_EVAL``
partition of its own cell.  The prompt's unit of scientific evidence is the full
5-market x 4-Host panel, so the grid is enumerated from the readiness table
rather than from a hand-written list: a cell that is not ``READY_FROZEN`` stops
the run instead of quietly shrinking the panel.

Two operational properties are part of the contract rather than conveniences:

**Thread pinning.**  Each worker sets ``torch.set_num_threads(1)`` before it
builds anything.  These models are small enough that BLAS thread contention
dominates their runtime, and -- more importantly -- a result that depends on how
many cores were free is not a reproducible result.  One thread makes a fit a
function of ``(cell, config, seed)`` alone.

**Resume by identity, not by trust.**  A fit is skipped only when its raw
artifact is already on disk *and* the artifact's own metadata records the same
cell, configuration, seed and switch vector.  A stale artifact from a different
recipe is not a resume; it is a collision, and it raises.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import config as C
from . import evaluate as E
from . import raw_evidence as RE
from .contracts import HarnessError

__all__ = ["plan_grid", "run_grid", "GRID_STATUS_FIELDS"]

GRID_STATUS_FIELDS = E.EVIDENCE_RECORD_FIELDS


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------
def plan_grid(cells: Sequence[Tuple[str, str]],
              configs: Sequence[str] = tuple(C.CONFIG_VARIANTS),
              seeds: Sequence[int] = tuple(C.TRAINING["seeds"]),
              variants: Optional[Mapping[str, Mapping[str, bool]]] = None
              ) -> List[Dict[str, Any]]:
    """Enumerate the registered grid, refusing unknown configurations.

    The order is ``cell -> config -> seed`` so that a partially completed run
    has whole cells behind it rather than a scattering of seeds, which makes an
    interrupted grid readable.

    ``variants`` exists for PART F.  After the four components are adjudicated
    the frozen method is *one switch vector derived from the decisions*, and with
    two or more components deleted it is frequently not one of the five
    registered names -- the five names are one-switch ablations, and the survivor
    need not be one.  Passing the vector here lets it be fitted through exactly
    the same code path as the grid, so the frozen run cannot differ from the
    screen in anything but the vector itself.
    """
    catalogue = _catalogue(variants)
    for name in configs:
        if name not in catalogue:
            raise HarnessError(f"unknown configuration {name!r}")
    plan: List[Dict[str, Any]] = []
    for market, host in cells:
        for name in configs:
            for seed in seeds:
                plan.append({"market": market, "host": host, "config": name,
                             "seed": int(seed)})
    return plan


def _catalogue(variants: Optional[Mapping[str, Mapping[str, bool]]] = None
               ) -> Dict[str, Mapping[str, bool]]:
    """The switch-vector catalogue a run may draw on.

    The default is the registered five.  A caller may extend it with a derived
    vector, but the vector must still be a well-formed setting of exactly the
    registered switches: an unknown switch name would mean the run had changed
    the model rather than selected from it.
    """
    if variants is None:
        return dict(C.CONFIG_VARIANTS)
    catalogue = dict(C.CONFIG_VARIANTS)
    for name, vector in variants.items():
        if set(vector) != set(C.SWITCH_NAMES):
            raise HarnessError(
                f"switch vector {name!r} is not over the registered switches "
                f"{sorted(C.SWITCH_NAMES)}: {sorted(vector)}")
        catalogue[str(name)] = {k: bool(v) for k, v in vector.items()}
    return catalogue


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
_DATASET_CACHE: Dict[str, Any] = {}


def _thread_pin() -> int:
    """Pin this process to one compute thread, once."""
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already initialised in this process; the value is what matters and it
        # is 1 by construction on first call.
        pass
    return int(torch.get_num_threads())


def _dataset(market: str, host: str):
    """Build one cell's dataset, cached for the life of the worker process."""
    key = f"{market}::{host}"
    if key not in _DATASET_CACHE:
        from .china5_adapter import build_cell_dataset, load_feature_source, \
            load_market_contract, verify_source_bytes
        from . import host_prediction_loader as H
        from .contracts import ReadAudit

        audit = ReadAudit()
        contract = load_market_contract(market)
        verify_source_bytes(contract)
        panel = H.load_host_panel(market, host, contract=contract, audit=audit)
        source = load_feature_source(contract, audit=audit)
        dataset = build_cell_dataset(market, host, contract=contract, panel=panel,
                                     source=source, audit=audit)
        if audit.protected_final_reads != 0:
            raise HarnessError(
                f"{key}: PROTECTED_FINAL read count is "
                f"{audit.protected_final_reads}, not 0")
        _DATASET_CACHE[key] = (dataset, audit)
    return _DATASET_CACHE[key]


def dataset_for(market: str, host: str):
    """One cell's cached ``(dataset, audit)``.

    The public seam onto :func:`_dataset`.  Several steps visit all 20 cells, and
    each visit re-reads the frozen Host container and the feature source;
    caching keeps the read count from growing with the number of steps while
    leaving the audit object intact -- the check that matters
    (``protected_final_reads == 0``) happens inside the loader, on the one audit
    that actually saw the reads.
    """
    return _dataset(market, host)


def _run_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    """One ``(cell, config, seed)`` fit, with its raw and metric evidence."""
    _thread_pin()
    market, host = str(task["market"]), str(task["host"])
    config_name, seed = str(task["config"]), int(task["seed"])
    raw_subdir = str(task.get("raw_subdir", C.RAW_DIR_GRID))
    metric_subdir = str(task.get("metric_subdir", C.METRIC_DIR_GRID))

    from . import protocol

    dataset, _audit = _dataset(market, host)
    # A3: the threshold is *loaded*, never fitted here.  Re-fitting would give
    # the same two numbers in almost every cell, which is precisely why it must
    # not happen -- the metric would keep looking right while the claim that the
    # subset was fixed before the evaluation had stopped being true.
    high_mass = protocol.high_mass_for(market, host)

    record = E.evaluate_fit(dataset, config_name, seed, high_mass=high_mass,
                            raw_subdir=raw_subdir, metric_subdir=metric_subdir,
                            max_epochs=task.get("max_epochs"),
                            switches=task.get("switches"))
    return {"task": dict(task), "record": record, "high_mass": high_mass,
            "index_row": _index_row(
                RE.read_raw_meta(RE.raw_artifact_path(
                    market, host, config_name, seed, C.ROLE_DEV_EVAL, raw_subdir)),
                record)}


def _index_row_from_resume(resumed: Mapping[str, Any]) -> Dict[str, Any]:
    """The index row of an already-persisted fit, from the artifact itself.

    A resumed fit is not a fit that did not happen: its artifact is on disk and
    belongs in the index exactly as much as a fresh one.  The row comes from the
    artifact's metadata, so re-indexing cannot drift from what was written.
    """
    written = resumed["raw"]
    return RE.index_rows_from_artifact(resumed["meta"], {
        "raw_path": resumed["raw_path"],
        "raw_sha256": written["raw_sha256"],
        "raw_bytes": Path(resumed["raw_path"]).stat().st_size,
    })


def _index_row(meta: Mapping[str, Any], record: Mapping[str, Any]
               ) -> Dict[str, Any]:
    """One ``RAW_PREDICTION_INDEX.csv`` row for a fit that has just been written.

    Built from the artifact's own metadata, not from the task: the row describes
    the file that exists, and a row assembled from what the caller *intended* to
    write would keep looking right after the write had changed.
    """
    row = RE.index_rows_from_artifact(meta, {
        "raw_path": record.get("raw_path"),
        "raw_sha256": record.get("raw_sha256"),
        "raw_bytes": record.get("raw_bytes"),
    })
    return row


def _read_existing(task: Mapping[str, Any], raw_subdir: str,
                   catalogue: Mapping[str, Mapping[str, bool]]
                   ) -> Optional[Dict[str, Any]]:
    """The persisted record for this fit, if one exists and matches by identity.

    A mismatch is not a miss: it means an artifact for this coordinate was
    produced by a different recipe, and silently overwriting it would destroy
    evidence.  That raises.
    """
    path = RE.raw_artifact_path(str(task["market"]), str(task["host"]),
                                str(task["config"]), int(task["seed"]),
                                C.ROLE_DEV_EVAL, raw_subdir)
    if not path.exists():
        return None
    loaded = RE.read_raw_artifact(path)
    meta = loaded["meta"]
    expected = {"config": str(task["config"]), "seed": int(task["seed"]),
                "partition": C.ROLE_DEV_EVAL}
    for field, value in expected.items():
        if meta.get(field) != value:
            raise HarnessError(
                f"raw artifact collision at {path}: metadata records "
                f"{field}={meta.get(field)!r} but the grid asked for {value!r}. "
                "Refusing to treat a different recipe's evidence as a resume.")
    recorded = dict(meta.get("switches") or {})
    wanted = dict(catalogue[str(task["config"])])
    if recorded != wanted:
        raise HarnessError(
            f"raw artifact collision at {path}: switch vector {recorded} does "
            f"not match the registered {task['config']} vector {wanted}")
    return {"task": dict(task), "meta": meta, "raw": loaded,
            "resumed": True, "raw_path": str(path)}


def _record_from_resume(resumed: Mapping[str, Any],
                        metric_subdir: str) -> Dict[str, Any]:
    """Rebuild the flat status row of an already-evaluated fit."""
    task = resumed["task"]
    meta = resumed["meta"]
    metric_path = RE.confine(
        RE.evidence_root() / metric_subdir
        / (Path(resumed["raw_path"]).stem + ".json"), "metric record read")
    record: Dict[str, Any] = {
        "market": task["market"], "host": task["host"],
        "cell": f"{task['market']}::{task['host']}",
        "config": task["config"], "seed": int(task["seed"]),
        "partition": C.ROLE_DEV_EVAL,
        "status": "OK_RESUMED", "failure": None,
        "n_parameters": meta.get("n_parameters"),
        "fit_seconds": meta.get("fit_seconds"),
        "inference_seconds": meta.get("inference_seconds"),
        "alpha": meta.get("alpha"),
        "amplitude_scale": meta.get("amplitude_scale"),
        "final_epochs": meta.get("final_epochs"),
        "n_folds": meta.get("n_folds"),
        "raw_path": _relative(resumed["raw_path"]),
        "raw_sha256": resumed["raw"]["raw_sha256"],
        "switches": dict(meta.get("switches") or {}),
    }
    if metric_path.exists():
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        record.update({k: v for k, v in payload.get("record", {}).items()})
        record["status"] = "OK_RESUMED"
    return record


def _relative(path: Any) -> str:
    try:
        return str(Path(path).resolve().relative_to(C.REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run_grid(cells: Sequence[Tuple[str, str]],
             configs: Sequence[str] = tuple(C.CONFIG_VARIANTS),
             seeds: Sequence[int] = tuple(C.TRAINING["seeds"]),
             workers: int = 1,
             resume: bool = True,
             raw_subdir: str = C.RAW_DIR_GRID,
             metric_subdir: str = C.METRIC_DIR_GRID,
             status_name: str = "GRID_STATUS.csv",
             out_subdir: str = "02_crossfit",
             max_epochs: Optional[int] = None,
             variants: Optional[Mapping[str, Mapping[str, bool]]] = None,
             progress=None) -> Dict[str, Any]:
    """Run (or resume) the grid and write its status table and summary."""
    catalogue = _catalogue(variants)
    tasks = plan_grid(cells, configs, seeds, variants=catalogue)
    started = time.time()
    records: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    n_resumed = 0

    pending: List[Dict[str, Any]] = []
    for task in tasks:
        task = dict(task, max_epochs=max_epochs, raw_subdir=raw_subdir,
                    metric_subdir=metric_subdir,
                    switches=dict(catalogue[str(task["config"])]))
        if resume:
            existing = _read_existing(task, raw_subdir, catalogue)
            if existing is not None:
                records.append(_record_from_resume(existing, metric_subdir))
                index_rows.append(_index_row_from_resume(existing))
                n_resumed += 1
                continue
        pending.append(task)

    if progress is not None:
        progress(f"grid: {len(tasks)} fits, {n_resumed} resumed, "
                 f"{len(pending)} to run, workers={workers}")

    def _collect(result: Mapping[str, Any]) -> None:
        records.append(result["record"])
        index_rows.append(result["index_row"])
        if progress is not None:
            r = result["record"]
            progress(f"  {r['cell']:<24} {r['config']:<17} seed={r['seed']:<3} "
                     f"mae_gain={_fmt(r.get('mae_gain_pct'))}% "
                     f"({r.get('fit_seconds')}s)")

    if workers <= 1:
        for task in pending:
            try:
                _collect(_run_task(task))
            except Exception as exc:  # noqa: BLE001 - recorded as a blocker
                failures.append({"task": task, "error": repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {pool.submit(_run_task, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    _collect(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append({"task": task, "error": repr(exc)})
                    if progress is not None:
                        progress(f"  FAILED {task['market']}/{task['host']} "
                                 f"{task['config']} seed={task['seed']}: {exc!r}")

    for failure in failures:
        task = failure["task"]
        records.append({
            "market": task["market"], "host": task["host"],
            "cell": f"{task['market']}::{task['host']}",
            "config": task["config"], "seed": int(task["seed"]),
            "partition": C.ROLE_DEV_EVAL, "status": "FAILED",
            "failure": failure["error"],
            "switches": dict(catalogue[str(task["config"])]),
        })

    records.sort(key=lambda r: (r["cell"], r["config"], r["seed"]))
    elapsed = time.time() - started

    # A4: the index is a merge keyed on (market, host, config, seed, partition),
    # so a resumed run refreshes its own rows instead of duplicating them.  It is
    # written once, after every artifact it names is on disk.
    index_path = RE.append_raw_index(index_rows) if index_rows else None

    from .contracts import write_csv

    status_path = RE.confine(RE.evidence_root() / out_subdir / status_name,
                             "grid status table")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv([{f: _csv(r.get(f)) for f in GRID_STATUS_FIELDS} for r in records],
              status_path, GRID_STATUS_FIELDS)

    summary = {
        "schema": "signed_mass_component_grid.v1",
        "generated_utc": E._utc_now(),
        "n_tasks": len(tasks),
        "n_records": len(records),
        "n_resumed": n_resumed,
        "n_failed": len(failures),
        "workers": int(workers),
        "threads_per_worker": 1,
        "seeds": [int(s) for s in seeds],
        "configs": list(configs),
        "switch_vectors": {name: dict(vector)
                           for name, vector in sorted(catalogue.items())},
        "cells": [f"{m}::{h}" for m, h in cells],
        "dev_eval_partition": C.ROLE_DEV_EVAL,
        "dev_eval_subsampled": False,
        "elapsed_seconds": round(elapsed, 2),
        "status_table": _relative(status_path),
        "raw_index": None if index_path is None else _relative(index_path),
        "n_index_rows": len(index_rows),
        "failures": [{"task": f["task"], "error": f["error"]} for f in failures],
        "runtime": {
            "total_fit_seconds": round(sum(
                float(r["fit_seconds"]) for r in records
                if isinstance(r.get("fit_seconds"), (int, float))), 3),
            "max_fit_seconds": max(
                (float(r["fit_seconds"]) for r in records
                 if isinstance(r.get("fit_seconds"), (int, float))), default=0.0),
            "total_inference_seconds": round(sum(
                float(r["inference_seconds"]) for r in records
                if isinstance(r.get("inference_seconds"), (int, float))), 6),
            "host_retrained": False,
            "host_inference_rerun": False,
            "note": ("Host predictions are the frozen artifacts; this run "
                     "trains only the repair method and re-runs no comparator."),
        },
    }
    summary_path = RE.confine(RE.evidence_root() / out_subdir / "GRID_SUMMARY.json",
                              "grid summary")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                       sort_keys=True, default=str),
                            encoding="utf-8")
    return {"records": records, "summary": summary,
            "status_table": str(status_path), "summary_path": str(summary_path)}


def _csv(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):+.3f}"
    return "n/a"


def default_workers() -> int:
    """A conservative default: never all cores, never a single thread."""
    return max(1, min(12, (os.cpu_count() or 2) - 1))
