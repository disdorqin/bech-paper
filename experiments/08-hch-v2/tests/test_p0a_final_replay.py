"""P0-A regression tests — protocol §4 final replay.

    x_final = s_d * sinh(z0 + pi_eff),  pi_eff = pi if DVG releases else 0

Tests (runnable as `python test_p0a_final_replay.py` or via pytest):
  1. all Identity            -> x_final == host  (deg == 0)
  2. released nonzero pi     -> x_final follows inverse-asinh exactly;
                               unreleased day ignores pi
  3. bundle reload           -> x_final effectively identical after
                               predict -> save -> reload
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _final_point import final_point_metrics, from_dvg  # noqa: E402

EPS = 1e-12


def _mk_rows(n_days=4, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        z0 = rng.normal(0, 1.0, 24)
        s = float(rng.uniform(5, 80))
        pi = rng.normal(0, 0.15, 24)
        price = s * np.sinh(z0 + pi * rng.choice([0, 1])) + rng.normal(0, 2, 24)
        rows.append({"date": f"2024-01-0{d+1}", "z0": z0, "s_day": s,
                     "pi": pi, "price": price,
                     "vm": np.ones(24, dtype=bool)})
    return rows


def test1_identity_matches_host():
    """No correction (pi=0 or no release) -> final forecast == host exactly."""
    rows = _mk_rows()
    for r in rows:
        r["pi"] = np.zeros_like(r["pi"])
    out = final_point_metrics(rows, [False] * len(rows))
    assert out["mae"] == out["host_mae"], "identity replay must equal host"
    assert abs(out["degradation"]) <= EPS, "degradation must be ~0"
    assert out["smape_nofloor"] >= 0
    # released=True with zero pi must also equal host
    out2 = final_point_metrics(rows, [True] * len(rows))
    assert abs(out2["mae"] - out2["host_mae"]) <= EPS
    print("  PASS test1 identity == host")


def test2_released_pi_follows_inverse_asinh():
    """Released nonzero pi -> pred == s*sinh(z0+pi) exactly; unreleased -> host."""
    rows = _mk_rows()
    rel = [True, False, True, True]
    out = final_point_metrics(rows, rel)
    # manual exact replay per row
    pred_all, host_all, price_all = [], [], []
    for r, rl in zip(rows, rel):
        pi_eff = r["pi"] if rl else np.zeros_like(r["pi"])
        pred_all.append(r["s_day"] * np.sinh(r["z0"] + pi_eff))
        host_all.append(r["s_day"] * np.sinh(r["z0"]))
        price_all.append(r["price"])
    p = np.concatenate(price_all)
    pred = np.concatenate(pred_all)
    host = np.concatenate(host_all)
    mae_manual = float(np.abs(p - pred).mean())
    host_mae_manual = float(np.abs(p - host).mean())
    assert abs(out["mae"] - round(mae_manual, 6)) <= 1e-6, "final mae mismatch manual replay"
    assert abs(out["host_mae"] - round(host_mae_manual, 6)) <= 1e-6
    # unreleased row 1 (index 1) must equal host on that day
    assert abs(out["day_mae_final"][1] - out["day_mae_host"][1]) <= EPS
    print("  PASS test2 inverse-asinh exact replay")


def test3_bundle_reload_identical():
    """Serialize predictions, reload, recompute -> effectively identical."""
    rows = _mk_rows()
    rel = [True, False, True, True]
    out = final_point_metrics(rows, rel)
    # rebuild x_final from (z0, pi_eff, s) exactly as the runner would store it
    preds = np.stack([r["s_day"] * np.sinh(r["z0"] + (r["pi"] if rl else 0))
                      for r, rl in zip(rows, rel)])
    np.savez_compressed(HERE / "_p0a_bundle_tmp.npz", preds=preds)
    with np.load(HERE / "_p0a_bundle_tmp.npz") as z:
        reloaded = z["preds"]
    (HERE / "_p0a_bundle_tmp.npz").unlink()
    # recompute metrics from reloaded predictions directly
    p = np.concatenate([r["price"] for r in rows])
    mae_reload = float(np.abs(p - reloaded.reshape(-1)).mean())
    assert abs(out["mae"] - mae_reload) <= 1e-6, "reloaded bundle mismatch"
    print("  PASS test3 bundle reload identical")


def test4_from_dvg_handles_empty():
    """from_dvg on empty/None -> n_hours 0; on a real dvg dict -> metrics."""
    assert from_dvg(None) == {"n_hours": 0}
    rows = _mk_rows()
    fake_dvg = {"_rows": rows, "_released": np.array([True, False, True, True])}
    o = from_dvg(fake_dvg)
    assert o["n_hours"] == 96 and o["mae"] > 0
    print("  PASS test4 from_dvg interface")


if __name__ == "__main__":
    print("P0-A final replay regression tests")
    test1_identity_matches_host()
    test2_released_pi_follows_inverse_asinh()
    test3_bundle_reload_identical()
    test4_from_dvg_handles_empty()
    print("ALL P0-A TESTS PASSED")
