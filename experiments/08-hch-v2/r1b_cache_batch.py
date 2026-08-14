"""Paper Gate host-cache batch — all headline/extended/domestic datasets x 4 hosts.

Fit on H0 ONLY, freeze, predict S1R-S4 (cache_one contract, protocol §6 / §12).
--resume skips only caches that carry the current split_hash (legacy 4-segment
caches are regenerated). --markets filters the dataset list.

R1B §6 canonical 16 (LAGO_DE/PJM/NEM_SA1/NORD_DK1 x 4 hosts) is a subset of this.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from host_cache import cache_one, _cache_is_valid

# protocol §5.1 headline + §5.2 extended + §5.3 domestic headline targets
ALL_DATASETS = [
    "LAGO_DE", "LAGO_BE", "LAGO_FR", "LAGO_PJM", "LAGO_NP",
    "NEM_SA1", "GEFCOM14P", "NORD_DK1",                  # headline
    "DE_EPEX", "PJM_2020", "EPEX_FR", "EPEX_BE", "EPEX_NL",
    "NORD_FI", "NORD_NO", "NORD_SE3",                    # extended
    "shandong_DA", "shandong_RT",                        # domestic headline
]
HOSTS = ["Linear", "MLP", "LSTM", "PatchTST"]            # protocol §6 H1-H4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="skip datasets/backbones with a valid split_hash cache")
    ap.add_argument("--markets", nargs="+", default=None,
                    help="restrict to these dataset keys")
    ap.add_argument("--hosts", nargs="+", default=HOSTS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    datasets = args.markets if args.markets else ALL_DATASETS
    backbones = args.hosts
    total = len(datasets) * len(backbones)
    done = n_ok = n_skip = 0
    for ds_key in datasets:
        for bb_name in backbones:
            done += 1
            if args.resume and _cache_is_valid(ds_key, bb_name):
                n_skip += 1
                print(f"[{done}/{total}] {ds_key} x {bb_name} SKIP (valid)", flush=True)
                continue
            print(f"[{done}/{total}] {ds_key} x {bb_name} ...", flush=True)
            try:
                rec = cache_one(ds_key, bb_name, seed=args.seed)
                n_ok += 1
                print(f"  OK split={rec['split_hash']} n_params={rec['n_params']} "
                      f"dur={rec['duration_s']}s", flush=True)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
    print(f"[cache-batch] attempted={done} ok={n_ok} skipped_valid={n_skip}")


if __name__ == "__main__":
    main()
