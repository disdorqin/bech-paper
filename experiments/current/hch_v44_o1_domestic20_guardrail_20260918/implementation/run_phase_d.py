"""Phase D driver — reference check, 36 new fits, aggregation, D0-D4, access audit.

Run:  python run_phase_d.py [--workers N]

Spawn-safe: the ``__main__`` guard is required because the fit pool uses the
``spawn`` start method, which re-imports this module in every child.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import guardrail_common as G
from guardrail_runner import phase_access, phase_domestic, phase_fits, phase_reference_check

U = G.U
EVID = G.EVID


def main(workers: int) -> int:
    t0 = time.time()
    EVID.mkdir(parents=True, exist_ok=True)
    # The main process opens the same guarded surfaces as the workers, so the
    # audit covers aggregation too, not only training.
    G.RC.install_access_guard()

    print("[D-0] coordinate universe", flush=True)
    coords = G.coordinate_table()
    U.json_dump(EVID / "COORDINATE_REGISTRY.json", {
        "schema": "hch_v44_o1_domestic20_coordinate_registry.v1",
        "protocol_id": G.PROTOCOL_ID,
        "n_coordinates": len(coords),
        "n_reused_expected": 24, "n_new_expected": 36,
        "market_order": G.MARKETS, "host_order": G.HOSTS, "seeds": G.SEEDS,
        "reused_cells": [G.cell_key(m, h) for m, h in G.REUSED_CELLS],
        "new_cells_in_registered_order": [G.cell_key(m, h) for m, h in G.NEW_CELLS],
        "coordinates": coords,
    })
    print(f"       {len(coords)} coordinates, "
          f"{sum(1 for c in coords if c['complete'])} already complete", flush=True)

    print("[D-1] reuse reference check", flush=True)
    G.U.json_dump(EVID / "REUSE_SNAPSHOT_BEFORE.json", G.reuse_snapshot())
    prov = phase_reference_check()
    print(f"       {prov['n_reused_runs']} reused runs pinned, "
          f"one code state={prov['one_probe_code_hash_across_reused']}", flush=True)

    print(f"[D-2] {len(G.NEW_CELLS) * len(G.SEEDS)} new fits on {workers} lane(s)", flush=True)
    results = phase_fits(workers=workers)
    print(f"       fitted={sum(1 for r in results if r['status'] == 'fitted')} "
          f"skipped={sum(1 for r in results if r['status'] == 'skipped_existing')}", flush=True)

    print("[D-3] aggregation + gate", flush=True)
    out = phase_domestic()
    gate = out["gate"]
    for block, checks in gate["checks"].items():
        flag = "PASS" if gate["passed"][block] else "FAIL"
        print(f"       {block} {flag}", flush=True)
        for name, ok in checks.items():
            print(f"           {'ok ' if ok else 'XX '}{name}", flush=True)
    print(f"       panel median {gate['panel_median_gain_pct']:+.4f}%  "
          f"host-positive {gate['n_strict_host_positive']}/20  "
          f">=2% {gate['n_ge_plus_2pct']}/20", flush=True)

    print("[D-4] access audit", flush=True)
    audit = phase_access()
    print(f"       test reads={audit['test_target_read_count']} "
          f"blocked={len(audit['quarantine']['distinct_paths_blocked'])}", flush=True)

    print(f"[D] done in {time.time() - t0:.1f}s  domestic_pass={gate['domestic_pass']}", flush=True)
    return 0 if gate["domestic_pass"] else 3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    sys.exit(main(ap.parse_args().workers))
