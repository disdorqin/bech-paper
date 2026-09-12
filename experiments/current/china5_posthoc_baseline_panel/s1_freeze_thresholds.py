"""Stage 1 — freeze per-market tail thresholds from HOST_TRAIN targets only.

Runs before any Host or baseline exists, so the thresholds cannot be conditioned
on a result.  Emits 00_protocol/THRESHOLD_FREEZE.json.

Run:  python experiments/current/china5_posthoc_baseline_panel/s1_freeze_thresholds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import panel_contract as pc      # noqa: E402
import panel_metrics as pm       # noqa: E402

OUT = ROOT / "experiments/evidence/china5_posthoc_baseline_panel_20260912"


def main() -> int:
    thr = {}
    for market in pc.MARKETS:
        contract = pc.build_day_contract(market)
        t = pm.compute_thresholds(market, contract)
        thr[market] = t
        print(f"[S1] {market:10s} q05={t['q05']:9.3f} q95={t['q95']:9.3f} "
              f"S90_day_spread={t['day_spread_p90']:9.3f} "
              f"(fit on {t['n_days_fit']} HOST_TRAIN days)")
        assert t["read_audit"]["protected_final_target_values_read"] == 0
    pm.dump_thresholds(OUT, thr)
    print(f"[S1] wrote {OUT/'00_protocol'/'THRESHOLD_FREEZE.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
