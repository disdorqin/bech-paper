"""Stage 4 — COSA supplementary track (strict online/TTA), if its protocol is legal here.

COSA is the only audited method whose setting is *online*: it is scored on the
same DEV_EVAL days it adapts on, in strict predict-then-update order.  It is
therefore never admitted to the strict offline main table; it goes to the
supplementary table with an explicit setting disclosure.

Legality preflight (recorded, not assumed):
  P1  the frozen Host cache must expose DEV_EVAL as a strictly increasing,
      gapless-per-day sequence (COSA's chronology is meaningless otherwise)
  P2  the online buffer may only be seeded from POST_TRAIN, which is strictly
      prior to every DEV_EVAL day
  P3  no PROTECTED_FINAL day may be materialised
If any check fails the track is recorded BLOCKED, never approximated.

Run:  python experiments/current/china5_posthoc_baseline_panel/s4_cosa.py [MARKET ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
for p in (HERE, ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import panel_contract as pc                                     # noqa: E402
import cosa_transfer                                            # noqa: E402
from src.utils.benchmark_cache import HostPredictionCache        # noqa: E402

OUT = ROOT / "experiments/evidence/china5_posthoc_baseline_panel_20260912"
HOSTS = ("PatchTST", "TimeMixer")
SEED = 2021


def preflight(market: str, name: str, cache: HostPredictionCache) -> dict:
    ev = cache.subset("DEV_EVAL")
    post = cache.subset("POST_TRAIN")
    days = np.array([str(s)[:10] for s in ev.timestamp])
    post_days = np.array([str(s)[:10] for s in post.timestamp]) if len(post.timestamp) else np.array([])
    dd = days.astype("datetime64[D]") if len(days) else np.array([], dtype="datetime64[D]")
    pd_ = post_days.astype("datetime64[D]") if len(post_days) else np.array([], dtype="datetime64[D]")
    mono = bool(np.all(np.diff(dd.astype(int)) > 0)) if len(dd) > 1 else False
    prior = bool(len(pd_) and pd_.max() < dd.min())
    checks = {
        "P1_DEV_EVAL_strictly_increasing_days": mono,
        "P2_buffer_seed_strictly_prior": prior,
        "P3_no_protected_final_materialised": bool(
            not ({"PROTECTED_FINAL"} & set(str(s) for s in cache.segment))),
        "P4_eval_days_present": bool(len(days) >= 2),
    }
    return {"checks": checks, "passed": all(checks.values()),
            "n_dev_eval_days": int(len(days)), "n_post_train_days": int(len(post_days)),
            "dev_eval_first_day": str(days[0]) if len(days) else None,
            "dev_eval_last_day": str(days[-1]) if len(days) else None,
            "post_train_last_day": str(pd_.max()) if len(pd_) else None}


def main(argv: list[str]) -> int:
    markets = [m for m in (argv or list(pc.MARKETS)) if m in pc.MARKETS]
    blocked, ran = [], []
    for market in markets:
        for name in HOSTS:
            store = OUT / "03_baselines" / market / name
            cache = HostPredictionCache.load(OUT / "02_hosts" / market / name / "host_predictions.npz")
            pre = preflight(market, name, cache)
            if not pre["passed"]:
                blocked.append({"market": market, "backbone": name, "track": "COSA_online_supplementary",
                                "reason": "PREFLIGHT_FAILED", "detail": pre})
                print(f"[S4] {market:9s} {name:10s} COSA BLOCKED preflight {pre['checks']}")
                continue
            ev, post = cache.subset("DEV_EVAL"), cache.subset("POST_TRAIN")
            hp = ev.host_pred[:, :, 0].astype(np.float64)
            y = ev.y_true[:, :, 0].astype(np.float64)
            seed_means = post.y_true[:, :, 0].mean(axis=1).astype(np.float64)
            t0 = time.perf_counter()
            res = cosa_transfer.run_cosa(hp, y, seed_means, seed=SEED)
            pred = np.asarray(res["pred"], dtype=np.float64)
            if pred.shape != y.shape or not np.isfinite(pred).all():
                raise RuntimeError(f"{market}/{name}/COSA: bad prediction array")
            store.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(store / "COSA_online.npz",
                                pred=pred.astype(np.float32), y_true=y.astype(np.float32))
            pc.dump_json(store / "COSA_online.json",
                         {"market": pc.MARKETS[market]["dataset_id"], "host": name,
                          "method": "COSA_online", "seed": SEED,
                          "eval_partition": "DEV_EVAL", "buffer_seed_partition": "POST_TRAIN",
                          "setting": "ONLINE_TTA_scored_on_adapted_days",
                          "preflight": pre, "wall_seconds": time.perf_counter() - t0,
                          **{k: v for k, v in res.items() if k != "pred"}})
            ran.append({"market": market, "backbone": name, "preflight": pre,
                        "DEV_EVAL_MAE": float(np.mean(np.abs(y - pred))),
                        "Host_DEV_EVAL_MAE": float(np.mean(np.abs(y - hp))),
                        "n_days_adapted": res["n_days_adapted"]})
            print(f"[S4] {market:9s} {name:10s} COSA ONLINE_MAE={np.mean(np.abs(y-pred)):9.4f} "
                  f"(Host {np.mean(np.abs(y-hp)):9.4f})  adapted={res['n_days_adapted']}/{len(y)}  "
                  f"{time.perf_counter()-t0:6.1f}s")
    # Merge into any existing record instead of clobbering it: a partial
    # re-invocation must never erase another market's legality proof.
    p = OUT / "03_baselines" / "COSA_PREFLIGHT.json"
    prev = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    key = lambda r: (r["market"], r["backbone"])                      # noqa: E731
    merged_ran = {key(r): r for r in prev.get("ran", [])}
    merged_ran.update({key(r): r for r in ran})
    merged_blocked = {(b["market"], b["backbone"]): b for b in prev.get("blocked", [])}
    merged_blocked.update({(b["market"], b["backbone"]): b for b in blocked})
    pc.dump_json(p, {"schema": "cosa_supplementary_preflight.v1",
                     "n_cells_recorded": len(merged_ran) + len(merged_blocked),
                     "ran": [merged_ran[k] for k in sorted(merged_ran)],
                     "blocked": [merged_blocked[k] for k in sorted(merged_blocked)]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
