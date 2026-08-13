"""R1B §6 host-cache batch generation — 4 markets x 4 hosts, host_seed=0.

16 cells: LAGO_DE/PJM/NEM_SA1/NORD_DK1 x {Linear, MLP, LSTM, PatchTST}.
Fit on H0 ONLY, freeze, predict S1R-S4 (cache_one contract, P0-2).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from host_cache import cache_one

MARKETS = ["LAGO_DE", "LAGO_PJM", "NEM_SA1", "NORD_DK1"]
HOSTS = ["Linear", "MLP", "LSTM", "PatchTST"]


def main():
    total = len(MARKETS) * len(HOSTS)
    done = 0
    for mk in MARKETS:
        for bb in HOSTS:
            done += 1
            print(f"[{done}/{total}] {mk} x {bb} ...", flush=True)
            try:
                rec = cache_one(mk, bb, seed=0)
                print(f"  OK split={rec['split_hash']} n_params={rec['n_params']} "
                      f"dur={rec['duration_s']}s", flush=True)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
    print(f"[r1b/cache] {done}/{total} cells attempted")


if __name__ == "__main__":
    main()
