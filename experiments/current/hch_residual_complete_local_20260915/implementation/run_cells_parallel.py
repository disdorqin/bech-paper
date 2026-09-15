"""Parallel scheduler for the RCL v4.1 scientific run — **scheduling only**.

Why a separate module rather than a flag on the runner
------------------------------------------------------
The registered code path is :func:`run_rcl_v4_1.run_cell`.  This module calls that
function **verbatim** and changes nothing about it: same arguments, same seed
constants, same order of fits within a cell, same ``torch.set_num_threads(1)``,
same metric functions, same artifact writers.  The only thing that differs is
*which process* a cell runs in and *when*.

That is the whole licence.  The brief permits parallelism that changes scheduling
and forbids parallelism that changes data order, seeds, fit support, early
stopping, aggregation, model structure or metric definitions.  Running one
unmodified cell per process satisfies it exactly, because:

* **Per-cell results are bit-identical.**  A cell's entire state is its inputs,
  the fixed ``SEEDS`` tuple, ``torch.set_num_threads(1)``, and ``seed_everything``
  before each model is built.  No randomness is drawn from a process-global stream
  that another process could have advanced, and no result depends on wall-clock
  time.  Moving a cell to another process therefore cannot move a number in it.
* **Cross-cell results are bit-identical.**  Cells never read each other.  The two
  panel-level objects — the closure ratio and the run index — are assembled in the
  parent in ``rcl_contract.CELLS`` order, which is the order the sequential
  ``main`` would have produced them in.  ``make_run_index`` additionally hashes its
  records in a canonical sort, so the digest is order-free by construction.
* **No artifact has two writers.**  Every path a cell writes is namespaced by
  ``{market}__{host}``; the parent writes only the two run-level files
  (``RUN_MANIFEST.json``, ``RESULT_INDEX.json``) and the per-record files, all
  after every worker has exited.

What is *not* done, and why
---------------------------
The 12 Stage-2 fits inside a cell, the 12 Stage-1 fold fits and the 3 final-Level
refits are independent of one another and could be spread across processes too.
They are left sequential on purpose: parallelising them means re-implementing the
inside of :func:`run_cell` — the loop that pairs each fit with the model built from
its returned state dict, evaluates it, and appends its emissions and closure inputs
in order.  That is a rewrite of the scientific code path, not a change to its
schedule, and the licence above does not cover it.  Eight cells across eight
one-thread processes is the speedup that costs no correctness argument.

Device
------
CPU, matching the registered run.  CUDA is deliberately not used even though it is
available: moving a fit to the GPU changes the arithmetic (different reduction
orders, different transcendental implementations), which is a *numerical* change
and not a scheduling one.  ``run_cell`` is called with ``device="cpu"``, the same
value the sequential ``main`` passes.

Usage::

    python run_cells_parallel.py --authorized <TOKEN> [--workers 8]

The token requirement is the runner's own, re-checked here so this module cannot
become a way to start a scientific fit without it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

#: One thread per process.  Set in the environment *and* through torch, because the
#: math libraries read these before torch is imported and torch's own setter cannot
#: un-start a thread pool that BLAS already opened.  A single-threaded worker is
#: what makes a worker's arithmetic identical to the sequential run's.
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _pin_single_thread() -> None:
    for name in _THREAD_ENV:
        os.environ[name] = "1"


def _worker(payload: dict) -> dict:
    """Run one cell, in this process, through the registered code path.

    Torch is imported *after* the thread pins are set, and ``run_rcl_v4_1`` is
    imported here rather than at module scope so that a spawned child (which
    re-imports this module) does not pull torch in before the pins land.
    """
    _pin_single_thread()

    import numpy as np
    import torch

    torch.set_num_threads(1)

    sys.path.insert(0, payload["impl_dir"])
    sys.path.insert(0, payload["src_dir"])

    import run_rcl_v4_1 as runner
    from data_boundary import AuditCounters

    market, host = payload["market"], payload["host"]
    root = Path(payload["evidence_root"])
    stem = f"{market}__{host}"
    started = time.time()

    counters = AuditCounters()
    records, artifacts = runner.run_cell(market, host, counters, "cpu")

    # The same writes, to the same paths, in the same order as ``main``.
    runner.write_json(root / f"00_protocol/oof_levels/{stem}.json",
                      artifacts["oof_levels"])
    runner.write_json(root / f"01_val/{stem}.json", artifacts["val"])
    for key, emitted in artifacts["emissions"].items():
        runner.write_json(root / f"06_candidates/emissions/{stem}__{key}.json",
                          emitted)
    for variant, arrays in artifacts["closures"].items():
        runner.write_json(root / f"06_candidates/closure/{stem}__{variant}.json",
                          arrays)

    return {
        "market": market,
        "host": host,
        "records": records,
        "closures": {v: {k: list(map(float, np.asarray(x).reshape(-1)))
                         for k, x in arrays.items()}
                     for v, arrays in artifacts["closures"].items()},
        "counters": counters.as_dict(),
        "seconds": time.time() - started,
    }


def _merge_counters(parts: list[dict]) -> dict:
    """Sum the per-worker audits in ``CELLS`` order.

    ``markets_read`` is rebuilt by first appearance rather than concatenated: each
    worker saw exactly one market, so the sequential run's list — every market in
    the order its first cell was visited — is the order of ``CELLS`` itself.
    """
    markets: list[str] = []
    for part in parts:
        for market in part["markets_read"]:
            if market not in markets:
                markets.append(market)
    return {
        "cells_read": sum(int(p["cells_read"]) for p in parts),
        "seeds_run": sum(int(p["seeds_run"]) for p in parts),
        "test_label_read_count": max(int(p["test_label_read_count"]) for p in parts),
        "protected_final_read": any(bool(p["protected_final_read"]) for p in parts),
        "foreign_read": any(bool(p["foreign_read"]) for p in parts),
        "host_retrained": any(bool(p["host_retrained"]) for p in parts),
        "baseline_rerun": any(bool(p["baseline_rerun"]) for p in parts),
        "split_modified": any(bool(p["split_modified"]) for p in parts),
        "core_modified": any(bool(p["core_modified"]) for p in parts),
        "markets_read": markets,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorized", default="")
    parser.add_argument("--workers", type=int, default=0,
                        help="worker processes; 0 means one per cell")
    parser.add_argument("--evidence-root", default="")
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    root = Path(args.evidence_root).resolve() if args.evidence_root else (
        here.parents[2] / "evidence/hch_residual_complete_local_v4_1_20260915")

    # Imported only to read the token and the cell list: this does not fit.
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(repo / "src"))
    import run_rcl_v4_1 as runner
    from rcl_contract import CELLS, SEEDS

    if args.authorized != runner.AUTHORIZATION_TOKEN:
        raise PermissionError(
            "the RCL v4.1 scientific run requires the registered execution token")

    workers = args.workers or len(CELLS)
    workers = max(1, min(int(workers), len(CELLS)))
    payloads = [{
        "market": market, "host": host,
        "evidence_root": str(root),
        "impl_dir": str(here),
        "src_dir": str(repo / "src"),
    } for market, host in CELLS]

    print(f"[parallel] {len(payloads)} cells over {workers} worker process(es), "
          f"1 thread each, device=cpu -> {root}", flush=True)

    import multiprocessing as mp

    context = mp.get_context("spawn")
    results: list[dict] = []
    with context.Pool(processes=workers) as pool:
        for part in pool.imap_unordered(_worker, payloads):
            results.append(part)
            print(f"[parallel] done {part['market']}/{part['host']} "
                  f"in {part['seconds']:.1f}s", flush=True)

    # ``imap_unordered`` returns in completion order; the merge must not inherit it.
    order = {f"{m}__{h}": i for i, (m, h) in enumerate(CELLS)}
    results.sort(key=lambda p: order[f"{p['market']}__{p['host']}"])

    records = [r for part in results for r in part["records"]]
    counters = _merge_counters([p["counters"] for p in results])

    # The closure aggregate, pooled over cells in ``CELLS`` order.
    import numpy as np

    by_variant: dict[str, dict[str, list]] = {}
    for part in results:
        for variant, arrays in part["closures"].items():
            for key, values in arrays.items():
                by_variant.setdefault(variant, {}).setdefault(key, []).append(
                    np.asarray(values))
    closure = {variant: runner.aggregate_closure_ratio(
        np.concatenate(arrays["level_truth"]),
        np.concatenate(arrays["level_prediction"]),
        np.concatenate(arrays["delta_prediction"]))
        for variant, arrays in by_variant.items()}

    runner.write_json(root / "00_protocol/RUN_MANIFEST.json", {
        "protocol_id": runner.PROTOCOL_ID,
        "cells": [list(c) for c in CELLS],
        "seeds": list(SEEDS),
        "variants": list(runner.VARIANT_ORDER),
        "test_label_read_count": runner.TEST_LABEL_READ_COUNT,
        "authorization_token": runner.AUTHORIZATION_TOKEN,
    })
    index_path = runner.write_json(
        root / "06_candidates/RESULT_INDEX.json",
        runner.make_run_index(records, protocol_id=runner.PROTOCOL_ID,
                              counters=counters, cells=CELLS, seeds=SEEDS,
                              terminal_state="PENDING_INDEPENDENT_VERIFICATION",
                              closure=closure))
    for record in records:
        runner.write_json(root / "06_candidates/records" /
                          f"{record['market']}__{record['host']}__"
                          f"{record['variant']}__seed_{record['seed']}.json",
                          record)
    print(f"[parallel] wrote {len(records)} records; index {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
