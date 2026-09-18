"""Stage runtime for the HCH final geometry-coupled canary (TRAIN+VAL only).

This module is the only place the canary stage touches the repository.  It

* re-exports the completed recovery stage's quarantine runtime (``recovery_common``)
  so the access guard, the pre-TEST frame builders and the frozen-source digest are
  the *identical* objects, not a re-implementation;
* binds the completed full-panel stage's ``pipeline`` lazily, read-only, for the
  causal history index, the TRAIN-only coordinate scales and the Host features;
* registers the fixed eight-cell canary panel and the read-only O1 evidence roots;
* provides CSV/aggregation helpers so the seed-median-first statistic is computed
  once, in one place.

No module here writes to ``src/**``, to any prior stage, or to any prior evidence
root.  There is no TEST scorer and none is importable from the canary runner.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]

RECOVERY_IMPL = REPO / "experiments/current/hch_v44_optimization_recovery_20260917/implementation"
FULLPANEL_IMPL = REPO / "experiments/current/hch_v44_china5_fullpanel_main_eval_20260917/implementation"
#: The completed O1 probe stage, whose registered schedule record this stage
#: mirrors.  Read-only: the canary never retrains, copies or re-selects O1.
PROBE_IMPL = REPO / "experiments/current/hch_v44_objective_alignment_probe_20260917/implementation"

EVID = REPO / "experiments/evidence/hch_final_geometry_coupled_canary_20260918"
RUNS = EVID / "runs"

if str(RECOVERY_IMPL) not in sys.path:
    sys.path.insert(0, str(RECOVERY_IMPL))

import recovery_common as RC  # noqa: E402  (prior stage, read-only)

U = RC.U

PROTOCOL_ID = "HCH_FINAL_GEOMETRY_COUPLED_CANARY_20260918"

#: The O1 recipe the canary reuses read-only and mirrors for its own schedule.
RECIPE_O1 = "O1_AUX_WARMUP_400_THEN_MAE"
#: The canary's own recipe id (same schedule, new architecture).
RECIPE_GC = "GC_AUX_WARMUP_400_THEN_MAE"
#: Registered objective switch point.  No search is permitted around this value.
SWITCH_STEP = 400

ROLE_TRAIN, ROLE_VAL = "TRAIN", "VAL"

#: The fixed eight-cell development canary (plan §8, PROTOCOL §4).
CANARY_CELLS = (
    ("GANSU_DA", "TimeMixer"),
    ("GANSU_DA", "iTransformer"),
    ("SHANDONG_DA", "TimeMixer"),
    ("SHANDONG_DA", "iTransformer"),
    ("SHAANXI_DA", "TimeMixer"),
    ("SHAANXI_DA", "iTransformer"),
    ("SHAANXI_DA", "LSTM"),
    ("QINGHAI_DA", "PatchTST"),
)
SEEDS = (7, 17, 37)

#: The six registered Shape-limited weak cells inside the canary panel.
SHAPE_LIMITED_WEAK = (
    ("GANSU_DA", "TimeMixer"),
    ("SHANDONG_DA", "TimeMixer"),
    ("SHANDONG_DA", "iTransformer"),
    ("SHAANXI_DA", "TimeMixer"),
    ("SHAANXI_DA", "iTransformer"),
    ("SHAANXI_DA", "LSTM"),
)

#: The strong QINGHAI_DA/PatchTST control cell (G4).
STRONG_CONTROL = ("QINGHAI_DA", "PatchTST")

VARIANTS = ("DIRECT", "GEOM_FLAT", "GEOM_COUPLED", "GEOM_COUPLED_NOHIST")
#: The same four names ``final_model`` registers, bound here so the runner can
#: address a variant without importing the model (and therefore without torch).
DIRECT, GEOM_FLAT, GEOM_COUPLED, GEOM_COUPLED_NOHIST = VARIANTS
#: Variants on the promotion critical path (E2).  The rest are conditional (E3).
CRITICAL_VARIANTS = ("GEOM_FLAT", "GEOM_COUPLED")
ABLATION_VARIANTS = ("DIRECT", "GEOM_COUPLED_NOHIST")

#: China-5 markets x 4 Hosts = the full development panel (E4).
CHINA5_MARKETS = ("GANSU_DA", "SHANDONG_DA", "SHAANXI_DA", "NINGXIA_DA", "QINGHAI_DA")
CHINA5_HOSTS = ("PatchTST", "TimeMixer", "iTransformer", "LSTM")
FULL20_CELLS = tuple((m, h) for m in CHINA5_MARKETS for h in CHINA5_HOSTS)
#: The 12 cells E4 must add (the canary already covers the other eight).
FULL20_MISSING_CELLS = tuple(c for c in FULL20_CELLS if c not in CANARY_CELLS)

#: Where each cell's frozen O1 result lives.  O1 is reused read-only and is never
#: retrained, never copied and never re-selected.  The completed O1 evidence was
#: produced by three invocations of one code state; every China-5 cell is covered
#: with all three seeds, and the *canary* reuses the eight rows marked below.
O1_ROOT_PROBE = REPO / "experiments/evidence/hch_v44_objective_alignment_probe_20260917"
O1_ROOT_FULL8 = REPO / "experiments/evidence/hch_v44_o1_full8_completion_20260918"
O1_ROOT_GUARD = REPO / "experiments/evidence/hch_v44_o1_domestic20_guardrail_20260918"

_P, _F, _G = O1_ROOT_PROBE, O1_ROOT_FULL8, O1_ROOT_GUARD
O1_ROOTS = {
    # -- canary panel (E2) -------------------------------------------------------
    "GANSU_DA__TimeMixer": _G,
    "GANSU_DA__iTransformer": _G,
    "SHANDONG_DA__TimeMixer": _G,
    "SHANDONG_DA__iTransformer": _P,
    "SHAANXI_DA__TimeMixer": _P,
    "SHAANXI_DA__iTransformer": _G,
    "SHAANXI_DA__LSTM": _G,
    "QINGHAI_DA__PatchTST": _G,
    # -- remaining full-20 cells (E4) -------------------------------------------
    "GANSU_DA__PatchTST": _P,
    "GANSU_DA__LSTM": _F,
    "SHANDONG_DA__PatchTST": _F,
    "SHANDONG_DA__LSTM": _G,
    "SHAANXI_DA__PatchTST": _F,
    "NINGXIA_DA__PatchTST": _G,
    "NINGXIA_DA__TimeMixer": _G,
    "NINGXIA_DA__iTransformer": _P,
    "NINGXIA_DA__LSTM": _G,
    "QINGHAI_DA__TimeMixer": _F,
    "QINGHAI_DA__iTransformer": _G,
    "QINGHAI_DA__LSTM": _G,
}
O1_RUN_FIELDS = ("freeze.json", "training_curve.json", "selected_ema.pt")


# --------------------------------------------------------------------------- paths
def cell_key(market: str, host: str) -> str:
    return U.cell_key(market, host)


def run_dir(variant: str, market: str, host: str, seed: int) -> Path:
    return RUNS / variant / cell_key(market, host) / f"seed{int(seed)}"


def o1_run_dir(market: str, host: str, seed: int) -> Path:
    key = cell_key(market, host)
    if key not in O1_ROOTS:
        raise KeyError(f"{key}: no registered O1 evidence root for this cell")
    return O1_ROOTS[key] / "o1_runs" / key / f"seed{int(seed)}"


def lazy_pipeline():
    """Import the completed full-panel stage's read-only runtime once per process."""
    if str(FULLPANEL_IMPL) not in sys.path:
        sys.path.insert(0, str(FULLPANEL_IMPL))
    import pipeline as _p  # noqa: PLC0415

    return _p


# --------------------------------------------------------------------------- io
def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    U.json_dump(Path(path), payload)


def sha256_file(path: Path) -> str:
    return U.sha256(Path(path))


def write_csv(path: Path, rows, header=None) -> None:
    """RFC4180 CSV writer; an empty row set still writes the header."""
    rows = list(rows)
    if header is None:
        header = list(rows[0].keys()) if rows else []
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(row.get(k)) for k in header])


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if value != value else repr(round(value, 10))
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def read_csv_rows(path: Path) -> list:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- stats
def median(values) -> float:
    arr = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def quantile(values, q: float) -> float:
    arr = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, float(q)))


def gain_pct(reference: float, candidate: float) -> float:
    """``100 * (reference - candidate) / reference``; positive means candidate better."""
    ref = float(reference)
    if not np.isfinite(ref) or ref == 0.0:
        return float("nan")
    return float(100.0 * (ref - float(candidate)) / ref)


# --------------------------------------------------------------------------- O1 reuse
def o1_reuse_snapshot(cells=None) -> dict:
    """Read-only, hash-pinned snapshot of the reused O1 runs.

    Every value is read from the prior stage's own freeze record; nothing is
    retrained, copied or re-selected.  Three independent pins must hold:

    1. the checkpoint re-hash equals the hash the run recorded for itself;
    2. the reused cells were produced by one and the same scientific code
       state (one ``probe_code_hash``, and the frozen source digest matches the
       current ``src`` tree);
    3. every run read zero TEST targets.

    ``cells`` defaults to the eight-cell canary panel; E4 passes the full-20 set.
    """
    cells = tuple(cells) if cells is not None else CANARY_CELLS
    cell_records = {}
    problems = []
    for market, host in cells:
        key = cell_key(market, host)
        root = O1_ROOTS[key]
        runs = {}
        for seed in SEEDS:
            run = o1_run_dir(market, host, seed)
            missing = [f for f in O1_RUN_FIELDS if not (run / f).is_file()]
            if missing:
                problems.append(f"{key}/seed{seed}: missing {missing}")
                continue
            freeze = load_json(run / "freeze.json")
            ckpt = run / "selected_ema.pt"
            ckpt_hash = sha256_file(ckpt)
            runs[str(seed)] = {
                "root": str(root.relative_to(REPO)).replace("\\", "/"),
                "freeze_sha256": sha256_file(run / "freeze.json"),
                "curve_sha256": sha256_file(run / "training_curve.json"),
                "checkpoint_sha256": ckpt_hash,
                "checkpoint_sha256_matches_own_record": bool(
                    ckpt_hash == str(freeze.get("selected_ema_checkpoint_sha256", ""))
                ),
                "selected_ema_parameter_hash": freeze.get("selected_ema_parameter_hash"),
                "selected_step": int(freeze.get("selected_step", -1)),
                "selected_val_mae_ema": float(freeze.get("selected_val_mae_ema", float("nan"))),
                "probe_code_hash": freeze.get("probe_code_hash"),
                "history_support_hash": freeze.get("history_support_hash"),
                "source_tree_digest": freeze.get("source_tree_digest"),
                "test_target_read_count": int(freeze.get("test_target_read_count", -1)),
                "recipe_id": freeze.get("recipe_id"),
            }
        cell_records[key] = {
            "market": market,
            "host": host,
            "oof_days": None,
            "runs": runs,
        }
    code_hashes = {
        r["probe_code_hash"]
        for c in cell_records.values() for r in c["runs"].values()
    }
    digests = {
        json.dumps(r["source_tree_digest"], sort_keys=True)
        for c in cell_records.values() for r in c["runs"].values()
    }
    n_runs = sum(len(c["runs"]) for c in cell_records.values())
    expected = len(cells) * len(SEEDS)
    payload = {
        "schema": "hch_final_gc_o1_reuse_provenance.v1",
        "protocol_id": PROTOCOL_ID,
        "role": "reused by hash; never retrained, never copied, never re-selected",
        "reused_cells": [cell_key(m, h) for m, h in cells],
        "n_reused_cells": len(cells),
        "n_reused_runs": n_runs,
        "expected_runs": expected,
        "missing": problems,
        "all_reused_present": (not problems) and n_runs == expected,
        "one_probe_code_hash_across_reused": len(code_hashes) == 1,
        "distinct_probe_code_hash_across_reused": sorted(h for h in code_hashes if h),
        "one_source_tree_digest_across_reused": len(digests) == 1,
        "source_tree_digest_reused": json.loads(sorted(digests)[0]) if len(digests) == 1 else None,
        "source_tree_digest_current": RC.source_tree_digest(),
        "all_checkpoints_match_own_record": all(
            r["checkpoint_sha256_matches_own_record"] for c in cell_records.values() for r in c["runs"].values()
        ),
        "all_test_target_reads_zero": all(
            r["test_target_read_count"] == 0 for c in cell_records.values() for r in c["runs"].values()
        ),
        "cells": cell_records,
    }
    return payload


def o1_checkpoint_pins(cells=None) -> dict:
    """``{relative path: sha256}`` for every reused O1 checkpoint and freeze record.

    Written *before* the canary fits and re-checked at gate time: an unchanged map
    is the evidence that no prior evidence or checkpoint was mutated.
    """
    cells = tuple(cells) if cells is not None else CANARY_CELLS
    pins = {}
    for market, host in cells:
        for seed in SEEDS:
            run = o1_run_dir(market, host, seed)
            for name in O1_RUN_FIELDS:
                p = run / name
                pins[str(p.relative_to(REPO)).replace("\\", "/")] = sha256_file(p)
    return pins


def o1_val_mae(market: str, host: str, seed: int) -> float:
    return float(load_json(o1_run_dir(market, host, seed) / "freeze.json")["selected_val_mae_ema"])


def o1_cell_median(market: str, host: str) -> float:
    return median([o1_val_mae(market, host, s) for s in SEEDS])


def o1_cell_medians(cells=None) -> dict:
    cells = tuple(cells) if cells is not None else CANARY_CELLS
    return {cell_key(m, h): o1_cell_median(m, h) for m, h in cells}


# --------------------------------------------------------------------------- workers
def init_worker() -> None:
    """Re-assert this stage's implementation dir on a spawned child's ``sys.path``.

    ``multiprocessing`` on Windows re-imports by module name; several completed
    stages are on ``sys.path`` and more than one of them ships a module named
    ``runner``/``trainer``.  Putting our own directory first makes the canary
    import its own modules, deterministically.
    """
    import os

    if str(HERE.parent) not in sys.path:
        sys.path.insert(0, str(HERE.parent))
    RC.worker_env()
    os.environ.setdefault("PYTHONHASHSEED", "0")
