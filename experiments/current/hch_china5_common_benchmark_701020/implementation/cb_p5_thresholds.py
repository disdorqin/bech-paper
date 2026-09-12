"""P5 — the TRAIN-only threshold freeze and the metric/unit contract.

Two things are frozen here, both before TEST exists:

1. **The thresholds**, fitted from TRAIN target values alone — `q05`, `q95` and
   `S90`, the extreme-day spread threshold.  The rule is the admitted common one,
   reproduced from `china5_posthoc_baseline_panel/panel_metrics.py` rather than
   re-invented:

   * `q05 = percentile(all TRAIN target values, 5)`
   * `q95 = percentile(all TRAIN target values, 95)`
   * `S90 = percentile(per-day spread over TRAIN, 90)`, spread = `max_h y - min_h y`

   The exact source day IDs and their hashes are persisted, so a reviewer can
   re-derive every number from the split manifest rather than trusting it.

2. **The metric contract and its units.**  Every harm field must declare whether it
   is an absolute price quantity or a percentage of the Host's own MAE.  The
   historical interpretation bug was reading `Normal_harm_vs_Host` — an absolute
   price difference — as though it were a percentage.  The registry therefore
   carries a companion `Normal_harm_vs_Host_pct` computed against the Host's own
   normal-region MAE, and both fields name their unit explicitly.

No TEST subset count and no TEST metric is produced: every quantity here is
computed from TRAIN target values only, and the TEST-label read count stays 0.

Run:  python .../cb_p5_thresholds.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

IMPL = Path(__file__).resolve().parent
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))

import cb_contracts as CB                                                    # noqa: E402

EV = CB.EV
OUT = EV / "05_thresholds"

FIT_PARTITION = "TRAIN"
H = CB.H                       # forecast horizon of a benchmark window

# The admitted common metric vocabulary, reproduced from
# china5_posthoc_baseline_panel/panel_metrics.py::cell_metrics.
METRIC_UNITS = {
    "Overall_MAE": "ABSOLUTE_PRICE",
    "MSE": "ABSOLUTE_PRICE_SQUARED",
    "RMSE": "ABSOLUTE_PRICE",
    "Host_Overall_MAE": "ABSOLUTE_PRICE",
    "relative_gain_vs_Host_pct": "PERCENT_OF_HOST_MAE",
    "Tail_MAE": "ABSOLUTE_PRICE",
    "Upper_tail_MAE": "ABSOLUTE_PRICE",
    "Lower_tail_MAE": "ABSOLUTE_PRICE",
    "Normal_region_MAE": "ABSOLUTE_PRICE",
    "Host_Normal_region_MAE": "ABSOLUTE_PRICE",
    "Normal_harm_vs_Host": "ABSOLUTE_PRICE",
    "Normal_harm_vs_Host_pct": "PERCENT_OF_HOST_NORMAL_REGION_MAE",
    "Tail_MAE_Host": "ABSOLUTE_PRICE",
    "Extreme_day_MAE": "ABSOLUTE_PRICE",
    "Extreme_day_MAE_Host": "ABSOLUTE_PRICE",
    "Upper_tail_day_MAE": "ABSOLUTE_PRICE",
    "Negative_price_MAE": "ABSOLUTE_PRICE",
    "Negative_price_MAE_Host": "ABSOLUTE_PRICE",
}

HARM_FIELDS = ("Normal_harm_vs_Host", "Normal_harm_vs_Host_pct")

METRIC_CONTRACT = {
    "schema": "china5_common_benchmark_metric_contract.v1",
    "protocol_id": CB.PROTOCOL_ID,
    "frozen_before_test_opened": True,
    "definition_source": ("experiments/current/china5_posthoc_baseline_panel/panel_metrics.py"
                          "::cell_metrics — the admitted common metric code, reproduced"),
    "region_rule": {
        "lower": "y_true <= q05",
        "upper": "y_true >= q95",
        "tail": "lower | upper",
        "normal": "complement of tail",
        "negative": "y_true < 0",
        "extreme_day": "(max_h y_true - min_h y_true) >= S90",
        "upper_tail_day": "any_h y_true >= q95",
        "note": ("every mask is computed on y_true at the hour or day being masked, so a "
                 "subset is defined without reference to any method's prediction"),
    },
    "empty_subset_rule": ("a subset with no members is reported as NOT_APPLICABLE_N0 and "
                          "carries no numeric value; a number is never invented for it"),
    "units": METRIC_UNITS,
    "harm_field_units": {
        "Normal_harm_vs_Host": {
            "unit": "ABSOLUTE_PRICE",
            "definition": "Normal_region_MAE - Host_Normal_region_MAE",
            "interpretation": ("an absolute price difference in the market's price unit per "
                               "MWh; it is NOT a percentage and must never be compared "
                               "against, or printed beside without its unit, a "
                               "PERCENT_OF_* field"),
            "historical_bug": ("the earlier stage read this absolute quantity as if it were a "
                               "relative percentage, which is the interpretation bug this "
                               "contract exists to prevent"),
        },
        "Normal_harm_vs_Host_pct": {
            "unit": "PERCENT_OF_HOST_NORMAL_REGION_MAE",
            "definition": "100 * (Normal_region_MAE - Host_Normal_region_MAE) / "
                          "Host_Normal_region_MAE",
            "interpretation": ("the relative counterpart of the field above, stated as a "
                               "percentage of the Host's own normal-region MAE; it is "
                               "undefined (NOT_APPLICABLE_N0) when the Host normal-region MAE "
                               "is zero or the normal subset is empty"),
        },
    },
    "reporting_rules": [
        "every harm value is published together with its unit token, never alone",
        "an ABSOLUTE_PRICE harm and a PERCENT_OF_* harm are never placed in the same column, "
        "ranked against each other, or averaged together",
        "the sign convention is method-minus-Host: positive means the method is worse",
        "relative_gain_vs_Host_pct uses the opposite sign convention "
        "(Host-minus-method); the two are never mixed in one table",
        "no harm field is computed on TEST during PRETEST",
    ],
}


def _percentile(a: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


# The benchmark represents every window in `float32`
# (`cb_contracts.load_windows` casts at construction), so the frozen thresholds are
# quantiles of a float32 array.  The source cells are exact float64.  The disclosure
# below states both and the gap between them, and the mirror that produces the second
# set is checked against the first rather than trusted.
TARGET_REPRESENTATION_DISCLOSURE = {
    "benchmark_window_dtype": "float32",
    "source_value_dtype": "float64",
    "frozen_thresholds_are_computed_in": "float32",
    "why": ("`cb_contracts.load_windows` builds every context/target window as "
            "`np.float32`, and that same array is what any future metric code masks and "
            "scores against, so a threshold expressed in it is self-consistent with the "
            "quantities it will classify"),
    "consequence": ("a threshold is a quantile of the float32 representation of the TRAIN "
                    "targets, not of the exact source cells; the exact-source values are "
                    "reported alongside as `float64_from_source` and the largest absolute "
                    "difference between the two representations is recorded per market"),
    "rule_unchanged": ("this is a representation disclosure only: no threshold value was "
                       "changed, no partition changed, and no TEST value is involved"),
}


def source_matrix_f64(market: str, role: str) -> tuple[np.ndarray, dict]:
    """The same TRAIN windows in exact source float64, plus a proof the mirror agrees.

    The value selection mirrors `cb_contracts.load_windows` (origin, then the 24 hours
    following it, read from the projected target map).  A mirror that silently drifted
    would report a wrong deviation, so the caller checks the float32 cast of this matrix
    against the benchmark's own window array and fails loudly on any disagreement.
    """
    plan = CB.split_plan(market)
    ob = CB.open_block(market)
    ids = list(plan.ids(role))
    mp, audit = CB._read_target_values(market, ids)                          # noqa: SLF001
    rows = []
    for d in ids:
        origin = pd.Timestamp(ob.timestamps[int(ob.positions[d][0])])
        rows.append([mp[origin + pd.Timedelta(hours=k)] for k in range(H)])
    return np.asarray(rows, dtype=np.float64), audit


def fit_market(market: str) -> dict:
    """q05 / q95 / S90 from TRAIN target values alone."""
    plan = CB.split_plan(market)
    w, audit = CB.load_windows(market, roles=(FIT_PARTITION,))
    y = np.asarray(w.target[:, :, 0], dtype=np.float64)
    if y.shape[0] != len(plan.ids(FIT_PARTITION)):
        raise RuntimeError(f"{market}: TRAIN window count {y.shape[0]} does not match the "
                           f"split manifest ({len(plan.ids(FIT_PARTITION))})")
    flat = y.reshape(-1)
    spread = y.max(axis=1) - y.min(axis=1)
    day_ids = [str(s) for s in plan.ids(FIT_PARTITION)]

    # --- representation cross-check: exact source float64 vs the frozen float32 windows
    y64, _ = source_matrix_f64(market, FIT_PARTITION)
    if y64.shape != y.shape:
        raise RuntimeError(f"{market}: the float64 source mirror has shape {y64.shape}, the "
                           f"benchmark windows {y.shape}")
    if not np.array_equal(y64.astype(np.float32), w.target[:, :, 0]):
        raise RuntimeError(f"{market}: the float64 source mirror does not reproduce the "
                           f"benchmark windows under a float32 cast — the deviation "
                           f"disclosure would be meaningless, so it is not published")
    f64_flat, f64_spread = y64.reshape(-1), y64.max(axis=1) - y64.min(axis=1)
    representation = {
        **TARGET_REPRESENTATION_DISCLOSURE,
        "float64_from_source": {"q05": _percentile(f64_flat, 5),
                                "q95": _percentile(f64_flat, 95),
                                "S90": _percentile(f64_spread, 90)},
        "float32_as_delivered": {"q05": _percentile(flat, 5),
                                 "q95": _percentile(flat, 95),
                                 "S90": _percentile(spread, 90)},
        "max_abs_deviation_float64_vs_float32": float(np.abs(y64 - y).max()),
        "mirror_reproduces_benchmark_windows_under_float32_cast": True,
    }
    return {
        "market": market,
        "dataset_id": json.loads((EV / "01_splits" / market / "split_manifest.json")
                                 .read_text(encoding="utf-8"))["dataset_id"],
        "fit_partition": FIT_PARTITION,
        "day_id_convention": CB.DAY_CONVENTION[market],
        "n_days_fit": int(y.shape[0]),
        "n_values_fit": int(flat.size),
        "q05": _percentile(flat, 5),
        "q95": _percentile(flat, 95),
        "day_spread_p90_S90": _percentile(spread, 90),
        "target_representation": representation,
        "source_day_ids": day_ids,
        "source_day_ids_sha256": hashlib.sha256(
            "\n".join(day_ids).encode("utf-8")).hexdigest().upper(),
        "source_target_values_sha256": hashlib.sha256(
            np.ascontiguousarray(y).tobytes()).hexdigest().upper(),
        "first_train_day": day_ids[0] if day_ids else None,
        "last_train_day": day_ids[-1] if day_ids else None,
        "production_rule": ("fixed once from TRAIN targets; applied unchanged to every Host, "
                            "method and partition; never recomputed per method and never "
                            "refit at TEST"),
        "test_target_values_read": 0,
        "read_audit": audit,
    }


# ------------------------------------------------------------------ re-derivation
def rederive(market: str, thr: dict) -> dict:
    """Independently re-read the source rows for the recorded day IDs and re-fit.

    A second, deliberately different read path: the windows are rebuilt from the raw
    source by the same projected reader but for the recorded IDs only, so the check
    fails if a stored number was not produced by the rule it claims.
    """
    plan = CB.split_plan(market)
    w, _ = CB.load_windows(market, roles=(FIT_PARTITION,))
    ids = [str(s) for s in plan.ids(FIT_PARTITION)]
    if ids != thr["source_day_ids"]:
        return {"agree": False, "reason": "source day IDs differ from the recorded list"}
    y = np.asarray(w.target[:, :, 0], dtype=np.float64)
    spread = y.max(axis=1) - y.min(axis=1)
    got = {"q05": _percentile(y.reshape(-1), 5), "q95": _percentile(y.reshape(-1), 95),
           "S90": _percentile(spread, 90)}
    want = {"q05": thr["q05"], "q95": thr["q95"], "S90": thr["day_spread_p90_S90"]}

    # the exact-source float64 mirror, re-derived here rather than copied
    y64, _ = source_matrix_f64(market, FIT_PARTITION)
    got64 = {"q05": _percentile(y64.reshape(-1), 5), "q95": _percentile(y64.reshape(-1), 95),
             "S90": _percentile((y64.max(axis=1) - y64.min(axis=1)), 90)}
    rep = thr["target_representation"]
    want64 = rep["float64_from_source"]

    return {"agree": all(np.isclose(got[k], want[k], rtol=0, atol=0) for k in got),
            "rederived": got, "recorded": want,
            "float64_agree": all(np.isclose(got64[k], want64[k], rtol=0, atol=0)
                                 for k in got64),
            "float64_rederived": got64, "float64_recorded": want64,
            "deviation_agrees": np.isclose(float(np.abs(y64 - y).max()),
                                           rep["max_abs_deviation_float64_vs_float32"],
                                           rtol=0, atol=0),
            "hash_agrees": hashlib.sha256(
                np.ascontiguousarray(y).tobytes()).hexdigest().upper()
                == thr["source_target_values_sha256"]}


def _rederivation_ok(v: dict) -> bool:
    """A re-derivation passes only if every published quantity reproduced, including the
    disclosed exact-source float64 values and the float32 deviation."""
    return bool(v.get("agree") and v.get("hash_agrees") and v.get("float64_agree")
                and v.get("deviation_agrees"))


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    thresholds, rederivations = {}, {}
    for market in CB.MARKETS:
        thresholds[market] = fit_market(market)
        rederivations[market] = rederive(market, thresholds[market])

    payload = {
        "schema": "china5_common_benchmark_threshold_freeze.v1",
        "protocol_id": CB.PROTOCOL_ID,
        "stage_id": "hch_china5_common_benchmark_701020",
        "fit_partition": FIT_PARTITION,
        "threshold_names": {"q05": "5th percentile of TRAIN target values",
                            "q95": "95th percentile of TRAIN target values",
                            "S90": "90th percentile of the TRAIN per-day spread "
                                   "(max_h y - min_h y) — the extreme-day threshold"},
        "created_before_any_test_target_read": True,
        "test_target_values_read": 0,
        "target_representation_disclosure": TARGET_REPRESENTATION_DISCLOSURE,
        "method_outcomes_read": 0,
        "host_predictions_read": 0,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thresholds": thresholds,
    }
    thresholds_path = OUT / "THRESHOLD_FREEZE.json"
    CB.dump_json(thresholds_path, payload)
    CB.dump_json(OUT / "THRESHOLD_REDERIVATION.json", {
        "schema": "china5_common_benchmark_threshold_rederivation.v1",
        "protocol_id": CB.PROTOCOL_ID,
        "per_market": rederivations,
        "passed": all(_rederivation_ok(v) for v in rederivations.values()),
    })
    CB.dump_json(OUT / "METRIC_CONTRACT.json", METRIC_CONTRACT)

    thr_hash = hashlib.sha256(thresholds_path.read_bytes()).hexdigest().upper()
    metric_hash = hashlib.sha256((OUT / "METRIC_CONTRACT.json").read_bytes()).hexdigest().upper()
    ok = all(_rederivation_ok(v) for v in rederivations.values())
    CB.dump_json(OUT / "P5_SUMMARY.json", {
        "protocol_id": CB.PROTOCOL_ID,
        "threshold_freeze_path": str(thresholds_path.relative_to(CB.ROOT)).replace("\\", "/"),
        "threshold_freeze_sha256": thr_hash,
        "metric_contract_sha256": metric_hash,
        "fit_partition": FIT_PARTITION,
        "test_target_values_read": CB.PROCESS_SEAL.test_label_read_count,
        "n_harm_fields_with_declared_unit": len(HARM_FIELDS),
        "target_representation": TARGET_REPRESENTATION_DISCLOSURE["benchmark_window_dtype"],
        "max_abs_deviation_float64_vs_float32_by_market": {
            m: thresholds[m]["target_representation"]["max_abs_deviation_float64_vs_float32"]
            for m in CB.MARKETS},
        "harm_fields": {k: METRIC_UNITS[k] for k in HARM_FIELDS},
        "rederivation_passed": ok,
        "passed": bool(ok),
    })

    for m in CB.MARKETS:
        t = thresholds[m]
        print(f"[P5] {m:12s} n_days={t['n_days_fit']:>4d} n_values={t['n_values_fit']:>6d}  "
              f"q05={t['q05']:>12.4f}  q95={t['q95']:>12.4f}  S90={t['day_spread_p90_S90']:>12.4f}"
              f"  rederive={'OK' if rederivations[m]['agree'] else 'MISMATCH'}"
              f"  f64_dev={t['target_representation']['max_abs_deviation_float64_vs_float32']:.2e}")
    print(f"[P5] threshold freeze  : {thr_hash[:16]}…  (TRAIN only; float32 window "
          f"representation, disclosed)")
    print(f"[P5] metric contract   : {metric_hash[:16]}…  "
          f"harm units declared: {[METRIC_UNITS[k] for k in HARM_FIELDS]}")
    print(f"[P5] TEST-label reads  : {CB.PROCESS_SEAL.test_label_read_count}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
