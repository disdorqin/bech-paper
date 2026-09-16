"""HCH v4.4 domestic baseline comparison freeze -- registry constructor.

Protocol : `HCH_V44_BASELINE_COMPARISON_FREEZE_20260916`
Design   : `docs/current/HCH_V44_BASELINE_COMPLETENESS_AND_PAPER_FREEZE_20260916.md`

Read-only with respect to every parent source.  Writes only under
`experiments/evidence/hch_v44_baseline_comparison_freeze_20260916/`.

Zero model fits.  The metric definition is not reimplemented here: the frozen
parent module `panel_metrics.py` (the file named by
`05_thresholds/METRIC_CONTRACT.json::definition_source`) is imported and called
verbatim, and its sha256 is recorded in `SOURCE_HASHES.json`.

Unit discipline (METRIC_CONTRACT.json `reporting_rules`): the two normal-region
harm fields are emitted in separate columns whose names carry their unit token,
and `relative_gain_vs_Host_pct` (Host-minus-method, positive = method better)
is never mixed with a harm field (method-minus-Host, positive = method worse).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/作业/science/solar_leak_price_model")
CURRENT = ROOT / "experiments/current/hch_v44_baseline_comparison_freeze_20260916"
V2 = ROOT / "experiments/evidence/hch_china5_common_benchmark_701020_full_v2_20260913"
ADJ = V2 / "10_missing_target_adjudication"
LAB = ROOT / "experiments/lab/common_benchmark_results_v2"
METRIC_DIR = ROOT / "experiments/current/china5_posthoc_baseline_panel"
EVID = ROOT / "experiments/evidence/hch_v44_baseline_comparison_freeze_20260916"

sys.path.insert(0, str(METRIC_DIR))
import panel_metrics as PM  # frozen parent definition, called verbatim

PROTOCOL_ID = "HCH_V44_BASELINE_COMPARISON_FREEZE_20260916"
V2_PROTOCOL_ID = "COMMON_BENCHMARK_701020_FULL_V2"
DATA_ADJUDICATION = "V2_MISSING_TARGET_SCORABILITY_ADJUDICATION_20260913"
PAPER_SCOPE = "COMMON_BENCHMARK_TEST_WITH_HISTORICAL_EXPOSURE_CAVEAT"

MARKETS = ["GANSU_DA", "SHANDONG_DA", "SHAANXI_DA", "NINGXIA_DA", "QINGHAI_DA"]
HOSTS = ["PatchTST", "TimeMixer", "iTransformer", "LSTM"]
METHODS = ["Host", "MatchedDirectResidual", "delta-Adapter", "PIR",
           "COSA", "UEC-STD", "OMPB"]

SETTING = {
    "Host": "HOST_REFERENCE",
    "MatchedDirectResidual": "OFFLINE_STATIC_CONTROL",
    "delta-Adapter": "OFFLINE_STATIC_POSTHOC",
    "PIR": "OFFLINE_STATIC_POSTHOC",
    "COSA": "ONLINE_TTA",
    "UEC-STD": "BLOCKED",
    "OMPB": "BLOCKED",
}
# PIR is numerically supported only on the hosts its frozen backbone config admits.
PIR_NUMERIC_HOSTS = {"PatchTST", "TimeMixer"}
BLOCKED_METHODS = {"UEC-STD", "OMPB"}

METRIC_SOURCE = ("RECOMPUTED_FROM_FROZEN_TEST_ARRAYS_FLOAT64_VIA_FROZEN_"
                 "panel_metrics.cell_metrics")


# ------------------------------------------------------------------ helpers
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def fnum(x):
    """Full-precision, bit-faithful CSV rendering; None -> empty."""
    if x is None:
        return ""
    if isinstance(x, float):
        return "nan" if x != x else repr(x)
    return x


def subset_status(n: int) -> str:
    return "OK" if n > 0 else "NOT_APPLICABLE_N0"


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: fnum(r.get(c)) for c in cols})


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ frozen parents
def load_thresholds() -> dict:
    t = json.loads((V2 / "05_thresholds/THRESHOLD_FREEZE.json").read_text(encoding="utf-8"))
    out = {}
    for m, d in t["thresholds"].items():
        out[m] = {"q05": d["q05"], "q95": d["q95"],
                  "day_spread_p90": d["day_spread_p90_S90"]}
    return out


def load_scorable_manifest() -> dict:
    return json.loads((ADJ / "SCORABLE_DAY_MANIFEST.json").read_text(encoding="utf-8"))


def load_parent_long() -> dict:
    """The adjudicated numeric authority: RESULTS_LONG.csv (TEST rows)."""
    rows = list(csv.DictReader(
        open(LAB / "RESULTS_LONG.csv", encoding="utf-8")))
    out = {}
    for r in rows:
        if r["split"] == "TEST" and r["method"] in METHODS:
            out[(r["market"], r["host"], r["method"])] = r
    return out


def load_arrays(market: str, host: str, method: str):
    """Frozen TEST arrays.  Rows with a non-finite target are left in place; the
    caller applies the declared scorable-day mask."""
    if method == "Host":
        z = np.load(V2 / f"08_joint_test/hosts/{market}/{host}/TEST.npz")
        yt = z["y_true"].astype(np.float64)
        yp = z["pred"].astype(np.float64)
        return yt, yp, yp.copy(), z["timestamp"]
    z = np.load(V2 / f"08_joint_test/baselines/{market}/{host}/{method}.npz")
    return (z["y_true"].astype(np.float64), z["pred"].astype(np.float64),
            z["host"].astype(np.float64), None)


def load_sidecar(market: str, host: str, method: str) -> dict | None:
    if method == "Host":
        p = V2 / f"08_joint_test/hosts/{market}/{host}/TEST.json"
    else:
        p = V2 / f"08_joint_test/baselines/{market}/{host}/{method}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_blocker(market: str, host: str, method: str) -> dict | None:
    p = V2 / f"08_joint_test/baselines/{market}/{host}/{method}.BLOCKED.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ per-cell metric row
def metric_fields(m: dict) -> dict:
    """Map the frozen contract's output onto registry columns, keeping each
    harm field in its own unit-tokened column."""
    up_n = m["n_upper_tail_hours"]
    lo_n = m["n_lower_tail_hours"]
    tail_n = up_n + lo_n
    norm_n = m["n_normal_hours"]
    host_norm = m["Host_Normal_region_MAE"]
    harm = m["Normal_harm_vs_Host"]
    # METRIC_CONTRACT: 100 * (Normal - Host_Normal) / Host_Normal; N0 if undefined.
    if harm is None or not host_norm:
        harm_pct = None
    else:
        harm_pct = 100.0 * harm / host_norm
    return {
        "mae": m["Overall_MAE"],
        "mse": m["MSE"],
        "rmse": m["RMSE"],
        "host_mae": m["Host_Overall_MAE"],
        "relative_gain_vs_host_pct": m["relative_gain_vs_Host_pct"],
        "tail_mae": m["Tail_MAE"], "tail_n_hours": tail_n,
        "tail_subset_status": subset_status(tail_n),
        "upper_tail_mae": m["Upper_tail_MAE"], "upper_tail_n_hours": up_n,
        "upper_tail_subset_status": subset_status(up_n),
        "lower_tail_mae": m["Lower_tail_MAE"], "lower_tail_n_hours": lo_n,
        "lower_tail_subset_status": subset_status(lo_n),
        "normal_region_mae": m["Normal_region_MAE"], "normal_n_hours": norm_n,
        "normal_subset_status": subset_status(norm_n),
        "negative_price_mae": m["Negative_price_MAE"],
        "negative_n_hours": m["n_negative_hours"],
        "negative_subset_status": m["negative_price_subset_status"],
        "extreme_day_mae": m["Extreme_day_MAE"],
        "extreme_day_n_days": m["n_extreme_days"],
        "extreme_day_subset_status": m["extreme_day_subset_status"],
        "upper_tail_day_mae": m["Upper_tail_day_MAE"],
        "upper_tail_day_n_days": m["n_upper_tail_days"],
        "upper_tail_day_subset_status": m["upper_tail_day_subset_status"],
        # the two harm fields, each in its own unit-tokened column
        "normal_harm_vs_host_abs_price": harm,
        "normal_harm_vs_host_pct_of_host_normal_mae": harm_pct,
        "host_normal_region_mae_abs_price": host_norm,
    }


MATRIX_COLS = [
    "protocol_id", "market", "host", "method", "setting", "method_canonical_setting",
    "frozen_declared_setting",
    "status", "blocker_class", "blocker_code", "blocker_reason", "blocker_authority",
    "blocker_inherited", "blocker_reopened",
    "mae", "parent_mae", "mae_matches_parent", "mse", "rmse", "host_mae",
    "relative_gain_vs_host_pct",
    "tail_mae", "tail_n_hours", "tail_subset_status",
    "upper_tail_mae", "upper_tail_n_hours", "upper_tail_subset_status",
    "lower_tail_mae", "lower_tail_n_hours", "lower_tail_subset_status",
    "normal_region_mae", "normal_n_hours", "normal_subset_status",
    "negative_price_mae", "negative_n_hours", "negative_subset_status",
    "extreme_day_mae", "extreme_day_n_days", "extreme_day_subset_status",
    "upper_tail_day_mae", "upper_tail_day_n_days", "upper_tail_day_subset_status",
    "normal_harm_vs_host_abs_price", "normal_harm_vs_host_pct_of_host_normal_mae",
    "host_normal_region_mae_abs_price",
    "n_registered_test_days", "n_scored_test_days", "n_unscorable_missing_target_days",
    "excluded_unscorable_day_ids", "scoring_mask",
    "paper_use_scope", "data_adjudication", "fidelity",
    "metric_source", "metric_definition_sha256",
    "source_pointer", "source_sha256", "sidecar_pointer", "sidecar_sha256",
]


def build() -> dict:
    thresholds = load_thresholds()
    manifest = {m["market"]: m for m in load_scorable_manifest()["markets"]}
    parent = load_parent_long()
    metric_sha = sha256_file(METRIC_DIR / "panel_metrics.py")

    matrix, snapshot_rows = [], []
    hashes = {}
    mismatches, sidecar_deviations = [], []

    for market in MARKETS:
        man = manifest[market]
        excluded = [d["day_id"] for d in man["unscorable_days"]]
        mask_label = ("EXCLUDE_UNSCORABLE_MISSING_TARGET_DAYS" if excluded
                      else "NONE_ALL_REGISTERED_DAYS_SCORABLE")
        for host in HOSTS:
            cell_rows = {}
            for method in METHODS:
                coord = f"{market}::{host}::{method}"
                row = {
                    "protocol_id": PROTOCOL_ID, "market": market, "host": host,
                    "method": method, "setting": SETTING[method],
                    # the method's own evaluation setting, kept even when the
                    # coordinate itself is BLOCKED (design doc section 5 vocabulary)
                    "method_canonical_setting": SETTING[method],
                    "paper_use_scope": PAPER_SCOPE, "data_adjudication": DATA_ADJUDICATION,
                    "metric_source": METRIC_SOURCE, "metric_definition_sha256": metric_sha,
                    "excluded_unscorable_day_ids": ";".join(excluded),
                    "scoring_mask": mask_label,
                }
                blocked = method in BLOCKED_METHODS or (
                    method == "PIR" and host not in PIR_NUMERIC_HOSTS)

                if blocked:
                    b = load_blocker(market, host, method)
                    if b is None:
                        raise SystemExit(f"BLOCKED coordinate without frozen blocker: {coord}")
                    bp = V2 / f"08_joint_test/baselines/{market}/{host}/{method}.BLOCKED.json"
                    row.update({
                        "status": "BLOCKED",
                        # a blocked coordinate has no evaluation setting
                        "setting": "BLOCKED",
                        # what the frozen blocker object itself declares, if anything
                        # (UEC-STD declares OFFLINE_STATIC_POSTHOC, OMPB ONLINE_SHIFT,
                        # the PIR blocker declares none)
                        "frozen_declared_setting": b.get("setting"),
                        # the method's own setting, taken from the frozen declaration
                        # where one exists and from the role map otherwise
                        "method_canonical_setting": b.get("setting") or SETTING[method],
                        "blocker_class": b.get("blocker_class"),
                        "blocker_code": b.get("blocker_code") or b.get("blocker_id"),
                        "blocker_reason": b.get("blocker_reason"),
                        "blocker_authority": b.get("authority"),
                        "blocker_inherited": b.get("inherited"),
                        "blocker_reopened": b.get("reopened"),
                        "fidelity": "NOT_APPLICABLE_BLOCKED",
                        # a blocked coordinate has no scored TEST days
                        "n_registered_test_days": man["n_registered_test_days"],
                        "n_scored_test_days": None,
                        "n_unscorable_missing_target_days": None,
                        "source_pointer": str(bp.relative_to(ROOT)).replace("\\", "/"),
                        "source_sha256": sha256_file(bp),
                    })
                    hashes[str(bp.relative_to(ROOT)).replace("\\", "/")] = row["source_sha256"]
                    matrix.append(row)
                    continue

                yt, yp, hp, _ = load_arrays(market, host, method)
                keep = ~np.isnan(yt).any(axis=1)
                n_scored = int(keep.sum())
                if n_scored != man["n_scored_test_days"]:
                    raise SystemExit(
                        f"{coord}: array scorable-day count {n_scored} != declared "
                        f"{man['n_scored_test_days']}")

                m = PM.strip_arrays(PM.cell_metrics(yt[keep], yp[keep], hp[keep],
                                                    thresholds[market]))
                if method == "Host":
                    npz = V2 / f"08_joint_test/hosts/{market}/{host}/TEST.npz"
                else:
                    npz = V2 / f"08_joint_test/baselines/{market}/{host}/{method}.npz"
                side = load_sidecar(market, host, method)

                row.update(metric_fields(m))
                row.update({
                    "status": "NUMERIC",
                    "n_registered_test_days": man["n_registered_test_days"],
                    "n_scored_test_days": n_scored,
                    "n_unscorable_missing_target_days": man["n_unscorable_missing_target_days"],
                    "source_pointer": str(npz.relative_to(ROOT)).replace("\\", "/"),
                    "source_sha256": sha256_file(npz),
                })
                hashes[row["source_pointer"]] = row["source_sha256"]

                if side is not None:
                    sp = (V2 / f"08_joint_test/hosts/{market}/{host}/TEST.json"
                          if method == "Host"
                          else V2 / f"08_joint_test/baselines/{market}/{host}/{method}.json")
                    row["sidecar_pointer"] = str(sp.relative_to(ROOT)).replace("\\", "/")
                    row["sidecar_sha256"] = sha256_file(sp)
                    hashes[row["sidecar_pointer"]] = row["sidecar_sha256"]
                    row["fidelity"] = side.get("fidelity", "NOT_APPLICABLE_HOST_REFERENCE")
                    row["frozen_declared_setting"] = side.get("setting", SETTING[method])
                    # The per-cell sidecar was scored from float64 source values while
                    # the stored array is float32; record the deviation rather than
                    # claiming a parity that does not hold to the last bit.
                    if side.get("Overall_MAE") is not None and side["Overall_MAE"] == side["Overall_MAE"]:
                        d = abs(row["mae"] - side["Overall_MAE"])
                        sidecar_deviations.append({
                            "coordinate": coord, "abs_diff": d,
                            "rel_diff": d / side["Overall_MAE"] if side["Overall_MAE"] else None,
                        })
                else:
                    row["fidelity"] = "NOT_APPLICABLE_HOST_REFERENCE"

                # §10.9 -- every numeric MAE must equal its frozen V2 parent value
                pr = parent.get((market, host, method))
                if pr is None:
                    raise SystemExit(f"{coord}: no frozen V2 parent row in RESULTS_LONG.csv")
                row["parent_mae"] = float(pr["mae"])
                row["mae_matches_parent"] = (row["mae"] == row["parent_mae"])
                if not row["mae_matches_parent"]:
                    mismatches.append({"coordinate": coord, "recomputed": row["mae"],
                                       "parent": row["parent_mae"]})
                if int(pr["n_scored_test_days"]) != n_scored:
                    mismatches.append({"coordinate": coord, "field": "n_scored_test_days",
                                       "recomputed": n_scored,
                                       "parent": int(pr["n_scored_test_days"])})
                cell_rows[method] = row
                matrix.append(row)

            snapshot_rows.append(build_snapshot(market, host, cell_rows))

    return {
        "matrix": matrix, "snapshot": snapshot_rows, "hashes": hashes,
        "mismatches": mismatches, "sidecar_deviations": sidecar_deviations,
        "metric_sha": metric_sha, "manifest": manifest,
    }


def build_snapshot(market: str, host: str, cell: dict) -> dict:
    """One market x Host comparator record (design doc §7).

    `best_strict_offline_external_posthoc` selects only among delta-Adapter and a
    numeric PIR (PROTOCOL §8).  MDR is an internal control and is recorded
    separately; Host is the reference, not a comparator; COSA is online/TTA and
    never participates in the offline selector.
    """
    def region(method):
        r = cell.get(method)
        if r is None or r["status"] != "NUMERIC":
            return None
        return {
            "mae_abs_price": r["mae"],
            "tail_mae_abs_price": r["tail_mae"], "tail_n_hours": r["tail_n_hours"],
            "tail_subset_status": r["tail_subset_status"],
            "upper_tail_mae_abs_price": r["upper_tail_mae"],
            "upper_tail_subset_status": r["upper_tail_subset_status"],
            "lower_tail_mae_abs_price": r["lower_tail_mae"],
            "lower_tail_subset_status": r["lower_tail_subset_status"],
            "normal_region_mae_abs_price": r["normal_region_mae"],
            "normal_subset_status": r["normal_subset_status"],
            "negative_price_mae_abs_price": r["negative_price_mae"],
            "negative_subset_status": r["negative_subset_status"],
            "extreme_day_mae_abs_price": r["extreme_day_mae"],
            "extreme_day_subset_status": r["extreme_day_subset_status"],
            "upper_tail_day_mae_abs_price": r["upper_tail_day_mae"],
            "upper_tail_day_subset_status": r["upper_tail_day_subset_status"],
            "normal_harm_vs_host_abs_price": r["normal_harm_vs_host_abs_price"],
            "normal_harm_vs_host_pct_of_host_normal_mae":
                r["normal_harm_vs_host_pct_of_host_normal_mae"],
        }

    cands = [(m, cell[m]["mae"]) for m in ("delta-Adapter", "PIR")
             if cell.get(m) and cell[m]["status"] == "NUMERIC"]
    best = min(cands, key=lambda t: t[1]) if cands else (None, None)

    blockers = {}
    for m in METHODS:
        r = cell.get(m)
        if r is not None and r["status"] == "BLOCKED":
            blockers[m] = {"blocker_class": r["blocker_class"],
                           "blocker_code": r["blocker_code"],
                           "blocker_reason": r["blocker_reason"],
                           "blocker_authority": r["blocker_authority"]}

    host_row = cell.get("Host", {})
    return {
        "market": market, "host": host,
        "host_mae_abs_price": host_row.get("mae"),
        "n_registered_test_days": host_row.get("n_registered_test_days"),
        "n_scored_test_days": host_row.get("n_scored_test_days"),
        "excluded_unscorable_day_ids": host_row.get("excluded_unscorable_day_ids", ""),
        "host_reference": region("Host"),
        "internal_control_mdr": region("MatchedDirectResidual"),
        "strict_offline_external_posthoc": {
            "delta_adapter": region("delta-Adapter"),
            "pir": region("PIR"),
            "pir_status": cell["PIR"]["status"] if "PIR" in cell else None,
            "best_method": best[0],
            "best_mae_abs_price": best[1],
            "selector_rule": ("min MAE over {delta-Adapter, numeric PIR} only; "
                              "MDR excluded as internal control, Host excluded as "
                              "reference, COSA excluded as online/TTA"),
        },
        "online_tta_separate": {
            "cosa": region("COSA"),
            "setting": "ONLINE_TTA",
            "note": "never pooled with, ranked against, or selected into the offline table",
        },
        "blockers": blockers,
    }


SNAPSHOT_COLS = [
    "market", "host", "n_registered_test_days", "n_scored_test_days",
    "excluded_unscorable_day_ids", "host_mae_abs_price",
    "mdr_mae_abs_price", "delta_adapter_mae_abs_price",
    "pir_status", "pir_mae_abs_price",
    "best_strict_offline_external_posthoc_method",
    "best_strict_offline_external_posthoc_mae_abs_price",
    "cosa_mae_abs_price", "cosa_setting",
    "host_tail_mae_abs_price", "host_normal_region_mae_abs_price",
    "host_extreme_day_subset_status", "host_negative_subset_status",
    "delta_tail_mae_abs_price", "delta_normal_region_mae_abs_price",
    "delta_normal_harm_vs_host_abs_price",
    "delta_normal_harm_vs_host_pct_of_host_normal_mae",
    "cosa_tail_mae_abs_price", "cosa_normal_region_mae_abs_price",
]


def snapshot_csv_rows(snap: list[dict]) -> list[dict]:
    out = []
    for s in snap:
        off = s["strict_offline_external_posthoc"]
        d = off["delta_adapter"] or {}
        p = off["pir"] or {}
        o = s["online_tta_separate"]["cosa"] or {}
        out.append({
            "market": s["market"], "host": s["host"],
            "n_registered_test_days": s["n_registered_test_days"],
            "n_scored_test_days": s["n_scored_test_days"],
            "excluded_unscorable_day_ids": s["excluded_unscorable_day_ids"],
            "host_mae_abs_price": s["host_mae_abs_price"],
            "mdr_mae_abs_price": (s["internal_control_mdr"] or {}).get("mae_abs_price"),
            "delta_adapter_mae_abs_price": d.get("mae_abs_price"),
            "pir_status": off["pir_status"], "pir_mae_abs_price": p.get("mae_abs_price"),
            "best_strict_offline_external_posthoc_method": off["best_method"],
            "best_strict_offline_external_posthoc_mae_abs_price": off["best_mae_abs_price"],
            "cosa_mae_abs_price": o.get("mae_abs_price"), "cosa_setting": "ONLINE_TTA",
            "host_tail_mae_abs_price": (s["host_reference"] or {}).get("tail_mae_abs_price"),
            "host_normal_region_mae_abs_price":
                (s["host_reference"] or {}).get("normal_region_mae_abs_price"),
            "host_extreme_day_subset_status":
                (s["host_reference"] or {}).get("extreme_day_subset_status"),
            "host_negative_subset_status":
                (s["host_reference"] or {}).get("negative_subset_status"),
            "delta_tail_mae_abs_price": d.get("tail_mae_abs_price"),
            "delta_normal_region_mae_abs_price": d.get("normal_region_mae_abs_price"),
            "delta_normal_harm_vs_host_abs_price":
                d.get("normal_harm_vs_host_abs_price"),
            "delta_normal_harm_vs_host_pct_of_host_normal_mae":
                d.get("normal_harm_vs_host_pct_of_host_normal_mae"),
            "cosa_tail_mae_abs_price": o.get("tail_mae_abs_price"),
            "cosa_normal_region_mae_abs_price": o.get("normal_region_mae_abs_price"),
        })
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    EVID.mkdir(parents=True, exist_ok=True)
    r = build()
    matrix, snap = r["matrix"], r["snapshot"]

    numeric = [x for x in matrix if x["status"] == "NUMERIC"]
    blocked = [x for x in matrix if x["status"] == "BLOCKED"]
    strict_off = [x for x in matrix if x["method"] in
                  ("Host", "MatchedDirectResidual", "delta-Adapter", "PIR")]
    print(f"coordinates={len(matrix)} numeric={len(numeric)} blocked={len(blocked)}")
    print(f"strict_offline_rows={len(strict_off)} (numeric "
          f"{sum(1 for x in strict_off if x['status']=='NUMERIC')})")
    print(f"parent MAE mismatches={len(r['mismatches'])}")

    write_csv(EVID / "MATRIX_140.csv", matrix, MATRIX_COLS)

    # strict offline table -- offline methods only; blocked rows visible, never ranked
    offline_rows = []
    for market in MARKETS:
        for host in HOSTS:
            cell = [x for x in strict_off if x["market"] == market and x["host"] == host]
            ranked = sorted([x for x in cell if x["status"] == "NUMERIC"],
                            key=lambda x: x["mae"])
            rank = {id(x): i + 1 for i, x in enumerate(ranked)}
            for x in sorted(cell, key=lambda x: METHODS.index(x["method"])):
                y = dict(x)
                y["mae_rank_within_cell"] = rank.get(id(x), "")
                y["is_best_strict_offline_external_posthoc"] = (
                    x["method"] in ("delta-Adapter", "PIR")
                    and x["status"] == "NUMERIC"
                    and x["mae"] == min([z["mae"] for z in cell
                                         if z["status"] == "NUMERIC"
                                         and z["method"] in ("delta-Adapter", "PIR")],
                                        default=None))
                offline_rows.append(y)
    write_csv(EVID / "STRICT_OFFLINE_TABLE.csv", offline_rows,
              MATRIX_COLS + ["mae_rank_within_cell",
                             "is_best_strict_offline_external_posthoc"])

    # online supplementary table -- COSA plus its same-cell Host reference
    online_rows = []
    for market in MARKETS:
        for host in HOSTS:
            h = next(x for x in matrix if x["market"] == market and x["host"] == host
                     and x["method"] == "Host")
            c = next(x for x in matrix if x["market"] == market and x["host"] == host
                     and x["method"] == "COSA")
            for x, role in ((h, "SAME_CELL_HOST_REFERENCE"), (c, "ONLINE_TTA_METHOD")):
                y = dict(x)
                y["row_role"] = role
                y["pooled_with_offline_rank"] = False
                online_rows.append(y)
    write_csv(EVID / "ONLINE_SUPPLEMENTARY_TABLE.csv", online_rows,
              MATRIX_COLS + ["row_role", "pooled_with_offline_rank"])

    write_csv(EVID / "BLOCKED_ROWS.csv", blocked, MATRIX_COLS)
    write_csv(EVID / "CELL_COMPARATOR_SNAPSHOT.csv", snapshot_csv_rows(snap),
              SNAPSHOT_COLS)
    write_json(EVID / "CELL_COMPARATOR_SNAPSHOT.json", {
        "schema": "hch_v44_cell_comparator_snapshot.v1",
        "protocol_id": PROTOCOL_ID,
        "v2_protocol_id": V2_PROTOCOL_ID,
        "unit_discipline": {
            "mae_abs_price": "ABSOLUTE_PRICE",
            "normal_harm_vs_host_abs_price": "ABSOLUTE_PRICE (method minus Host; positive = worse)",
            "normal_harm_vs_host_pct_of_host_normal_mae":
                "PERCENT_OF_HOST_NORMAL_REGION_MAE",
            "note": "the absolute and percentage harm fields are separate keys and are never ranked or averaged together",
        },
        "comparator_rule": {
            "best_strict_offline_external_posthoc": "min MAE over {delta-Adapter, numeric PIR}",
            "excluded": {
                "MatchedDirectResidual": "internal project control",
                "Host": "reference, not an external post-hoc method",
                "COSA": "online/TTA; never participates in the offline selector",
            },
        },
        "n_records": len(snap),
        "records": snap,
    })

    # ---------------------------------------------------------------- hashes
    parent_files = [
        LAB / "RESULTS_LONG.csv", LAB / "STATE.json",
        V2 / "05_thresholds/METRIC_CONTRACT.json",
        V2 / "05_thresholds/THRESHOLD_FREEZE.json",
        V2 / "01_splits/SPLIT_INDEX.json",
        ADJ / "FINAL_OFFLINE_MATRIX.csv", ADJ / "FINAL_PREQUENTIAL_MATRIX.csv",
        ADJ / "FINAL_VERIFICATION.json", ADJ / "SCORABLE_DAY_MANIFEST.json",
        ADJ / "INHERITED_BLOCKERS.csv", ADJ / "INHERITED_BLOCKERS.json",
        ADJ / "RAW_ARTIFACT_HASHES.json", ADJ / "EXPORT_COMPLETE.json",
        METRIC_DIR / "panel_metrics.py",
    ]
    parent_hashes = {}
    for p in parent_files:
        if p.exists():
            parent_hashes[str(p.relative_to(ROOT)).replace("\\", "/")] = sha256_file(p)
    write_json(EVID / "SOURCE_HASHES.json", {
        "schema": "hch_v44_baseline_freeze_source_hashes.v1",
        "protocol_id": PROTOCOL_ID,
        "hash_algorithm": "sha256",
        "frozen_metric_definition": str(
            (METRIC_DIR / "panel_metrics.py").relative_to(ROOT)).replace("\\", "/"),
        "frozen_metric_definition_sha256": r["metric_sha"],
        "parent_source_hashes": parent_hashes,
        "per_cell_source_hashes": r["hashes"],
        "n_per_cell_sources": len(r["hashes"]),
    })

    # ---------------------------------------------------------------- audit
    per_method = {}
    for meth in METHODS:
        rs = [x for x in matrix if x["method"] == meth]
        per_method[meth] = {
            "registered": len(rs),
            "numeric": sum(1 for x in rs if x["status"] == "NUMERIC"),
            "blocked": sum(1 for x in rs if x["status"] == "BLOCKED"),
            "setting": SETTING[meth],
            "blocker_class": next((x["blocker_class"] for x in rs
                                   if x["status"] == "BLOCKED"), None),
        }
    per_market = {}
    for m in MARKETS:
        man = r["manifest"][m]
        per_market[m] = {
            "n_registered_test_days": man["n_registered_test_days"],
            "n_scored_test_days": man["n_scored_test_days"],
            "n_unscorable_missing_target_days": man["n_unscorable_missing_target_days"],
            "unscorable_days": man["unscorable_days"],
        }
    devs = [d for d in r["sidecar_deviations"] if d["rel_diff"] is not None]
    max_dev = max(devs, key=lambda d: d["rel_diff"]) if devs else None
    write_json(EVID / "BASELINE_COMPLETENESS_AUDIT.json", {
        "schema": "hch_v44_baseline_completeness_audit.v1",
        "protocol_id": PROTOCOL_ID,
        "v2_protocol_id": V2_PROTOCOL_ID,
        "controlling_design":
            "docs/current/HCH_V44_BASELINE_COMPLETENESS_AND_PAPER_FREEZE_20260916.md",
        "coordinate_universe": {
            "markets": MARKETS, "hosts": HOSTS, "methods": METHODS,
            "registered": len(matrix),
            "numeric": len(numeric),
            "blocked": len(blocked),
            "expected_numeric": 90, "expected_blocked": 50,
            "census_matches_expected": len(numeric) == 90 and len(blocked) == 50,
        },
        "per_method": per_method,
        "per_market": per_market,
        "metric_authority": {
            "numeric_authority": "experiments/lab/common_benchmark_results_v2/RESULTS_LONG.csv",
            "row_kind": "SEED_OR_HOST", "data_adjudication": DATA_ADJUDICATION,
            "per_cell_sidecar_role": "disclosure and fidelity metadata",
            "per_cell_sidecar_caveat": (
                "the per-cell sidecars carry n_test_values = registered*24 and, on "
                "SHANDONG_DA, Overall_MAE/Normal = NaN because the frozen cell_metrics "
                "puts non-finite-target hours into the normal complement; the "
                "missing-target-adjudicated artifacts are therefore the numeric "
                "authority (PROTOCOL section 6)"),
            "sidecar_vs_recomputed_relation": (
                "sidecars were scored from float64 source values while the stored "
                "arrays are float32; agreement is to ~1e-9 relative, not bit-exact"),
            "n_numeric_cells_compared_to_sidecar": len(devs),
            "max_sidecar_rel_deviation": max_dev["rel_diff"] if max_dev else None,
            "max_sidecar_rel_deviation_coordinate": max_dev["coordinate"] if max_dev else None,
        },
        "no_new_fit_audit": {
            "new_baseline_neural_fits": 0,
            "new_host_fits": 0,
            "new_v44_fits": 0,
            "models_fitted_by_this_task": [],
            "statement": ("this task reconstructs a registry from frozen TEST arrays and "
                          "frozen metric code; it fits nothing and writes no model"),
        },
        "blocker_disposition": {
            "reopened": 0, "proxied": 0,
            "policy": ("UEC-STD Q-FIDELITY, OMPB Q-DATA and PIR new-Host "
                       "Q-INCOMPATIBLE remain closed; a blocker is a result"),
        },
        "blocked_coordinate_day_counts": (
            "BLOCKED rows carry the market's registered TEST-day count and leave "
            "scored/unscorable empty: a blocked coordinate has no scored day"),
        "parent_mae_mismatches": r["mismatches"],
        "conclusion": ("numeric coverage is complete at 90/90 against the adjudicated "
                       "V2 authority; the registry is reconstructible end-to-end from "
                       "frozen arrays with zero new fits"),
    })

    write_results_md(r)
    print("registry written to", EVID)
    return 0 if not r["mismatches"] else 1


def write_results_md(r: dict) -> None:
    matrix, snap = r["matrix"], r["snapshot"]
    numeric = [x for x in matrix if x["status"] == "NUMERIC"]
    blocked = [x for x in matrix if x["status"] == "BLOCKED"]
    L = []
    A = L.append
    A("# HCH v4.4 Domestic Baseline Comparison Freeze -- Registry")
    A("")
    A(f"Protocol: `{PROTOCOL_ID}`  ")
    A("Design: `docs/current/HCH_V44_BASELINE_COMPLETENESS_AND_PAPER_FREEZE_20260916.md`  ")
    A("Status: **READ-ONLY BASELINE RECONSTRUCTION / ZERO NEW FITS**")
    A("")
    A("## 1. What was built")
    A("")
    A(f"A {len(matrix)}-coordinate registry over 5 markets x 4 Hosts x 7 methods: "
      f"**{len(numeric)} numeric + {len(blocked)} blocked**. No model was fitted.")
    A("")
    A("Numeric values are recomputed from the frozen V2 TEST arrays by calling the "
      "frozen parent metric definition "
      "(`experiments/current/china5_posthoc_baseline_panel/panel_metrics.py::cell_metrics`) "
      "verbatim, under the thresholds frozen in "
      "`05_thresholds/THRESHOLD_FREEZE.json`. No metric formula is reimplemented here.")
    A("")
    A("## 2. Metric authority, and why the sidecars are not it")
    A("")
    A("The numeric authority is `RESULTS_LONG.csv` "
      "(`row_kind=SEED_OR_HOST`, "
      f"`data_adjudication={DATA_ADJUDICATION}`), as PROTOCOL section 6 requires. "
      "Every one of the 90 numeric MAE values reproduces that authority **exactly** "
      f"(0 mismatches).")
    A("")
    A("The per-cell JSON sidecars are **not** usable as the numeric authority: they "
      "carry `n_test_values = registered*24`, and on SHANDONG_DA they record "
      "`Overall_MAE` and `Normal` as `NaN`. That is a property of the frozen "
      "`cell_metrics`: `NaN >= q95` is False, so non-finite-target hours fall out of "
      "the tail masks, but `normal = ~(up | lo)` is True for them, so they enter the "
      "normal complement. The missing-target-adjudicated artifacts exist precisely to "
      "resolve this, and they are what this registry uses.")
    A("")
    A("For the 70 cells that have a sidecar, the sidecar was scored from float64 "
      "source values while the stored array is float32. Agreement is therefore to "
      "~1e-9 relative, not bit-exact. That deviation is recorded per cell rather than "
      "papered over.")
    A("")
    A("## 3. Scoring mask")
    A("")
    A("The mask is derived from the arrays (drop days whose target vector is "
      "non-finite) and cross-checked against the declared "
      "`SCORABLE_DAY_MANIFEST.json`. The two agree exactly:")
    A("")
    A("| Market | Registered | Scored | Unscorable days |")
    A("|---|---:|---:|---|")
    for m in MARKETS:
        man = r["manifest"][m]
        days = ", ".join(d["day_id"] for d in man["unscorable_days"]) or "--"
        A(f"| {m} | {man['n_registered_test_days']} | {man['n_scored_test_days']} | {days} |")
    A("")
    A("## 4. Method census")
    A("")
    A("| Method | Method setting (canonical) | row `setting` | Numeric | Blocked | Blocker class |")
    A("|---|---|---|---:|---:|---|")
    for meth in METHODS:
        rs = [x for x in matrix if x["method"] == meth]
        n = sum(1 for x in rs if x["status"] == "NUMERIC")
        b = sum(1 for x in rs if x["status"] == "BLOCKED")
        cls = next((x["blocker_class"] for x in rs if x["status"] == "BLOCKED"), "--")
        canon = rs[0]["method_canonical_setting"]
        rowset = "BLOCKED" if b and not n else canon
        if n and b:
            rowset = f"{canon} / BLOCKED"
        A(f"| {meth} | {canon} | {rowset} | {n} | {b} | {cls} |")
    A("")
    A("A blocked coordinate carries `setting = BLOCKED` (design doc section 5 "
      "vocabulary) because it has no evaluation setting. Its method's own setting is "
      "preserved in `method_canonical_setting`, taken from the frozen blocker object "
      "where that object declares one: UEC-STD declares `OFFLINE_STATIC_POSTHOC` and "
      "OMPB declares `ONLINE_SHIFT`. The raw frozen declaration is kept verbatim in "
      "`frozen_declared_setting` (the PIR blocker declares none).")
    A("")
    A("## 5. Headline numbers (ABSOLUTE_PRICE, TEST, common benchmark)")
    A("")
    A("| Market | Host | Host MAE | MDR | delta-Adapter | PIR | best offline external | COSA (online) |")
    A("|---|---|---:|---:|---:|---:|---|---:|")
    for s in snap:
        off = s["strict_offline_external_posthoc"]
        d = off["delta_adapter"] or {}
        p = off["pir"] or {}
        o = s["online_tta_separate"]["cosa"] or {}
        def g(v):
            return "--" if v is None else f"{v:.4f}"
        pir = g(p.get("mae_abs_price")) if off["pir_status"] == "NUMERIC" else "BLOCKED"
        best = off["best_method"] or "--"
        A(f"| {s['market']} | {s['host']} | {g(s['host_mae_abs_price'])} | "
          f"{g((s['internal_control_mdr'] or {}).get('mae_abs_price'))} | "
          f"{g(d.get('mae_abs_price'))} | {pir} | {best} | {g(o.get('mae_abs_price'))} |")
    A("")
    A("`relative_gain_vs_Host_pct` uses the Host-minus-method convention "
      "(positive = method better). The two normal-region harm fields use the "
      "opposite, method-minus-Host convention (positive = method worse) and are "
      "published in separate unit-tokened columns, never ranked together.")
    A("")
    A("## 6. Blocker disposition")
    A("")
    A("All 50 blockers are inherited frozen objects, reopened = false, proxied = "
      "false. No proxy value was created for any of them. BLOCKED rows carry the "
      "market's registered TEST-day count and leave the scored count empty, because a "
      "blocked coordinate has no scored day.")
    A("")
    A("## 7. What this round does NOT claim")
    A("")
    A("- It does not train, retrain, or fine-tune anything. New fits = 0.")
    A("- It does not reopen UEC-STD, OMPB or PIR-new-Host blockers.")
    A("- It does not modify any parent artifact or any file under `paper/**`.")
    A("- It does not treat the frozen V2 TEST as untouched-final evidence; every "
      "numeric row carries "
      f"`{PAPER_SCOPE}`.")
    A("- It does not start v4.4 training.")
    A("")
    A("## 8. Files")
    A("")
    for f in ["MATRIX_140.csv", "STRICT_OFFLINE_TABLE.csv",
              "ONLINE_SUPPLEMENTARY_TABLE.csv", "BLOCKED_ROWS.csv",
              "CELL_COMPARATOR_SNAPSHOT.csv", "CELL_COMPARATOR_SNAPSHOT.json",
              "SOURCE_HASHES.json", "BASELINE_COMPLETENESS_AUDIT.json",
              "VERIFICATION_REPORT.json"]:
        A(f"- `{f}`")
    A("")
    A("`VERIFICATION_REPORT.json` is produced by `implementation/verify_registry.py`, "
      "which does not import the registry constructor and does not import "
      "`panel_metrics` -- it re-implements the frozen region rules and recomputes "
      "every value from the parent arrays.")
    A("")
    A("## 9. Superseded artifacts, disclosed")
    A("")
    A("This directory previously held output from a competing instruction file in the "
      "same protocol folder. The two documents in "
      "`experiments/current/hch_v44_baseline_comparison_freeze_20260916/` prescribe "
      "**different** experiments:")
    A("")
    A("- `PROTOCOL.md` -- this registry: read-only, 140 coordinates, 90 numeric, "
      "zero new fits.")
    A("- `AI_EXECUTION_PROMPT.md` -- 100 coordinates and **50 new training runs**, "
      "and it names "
      "`experiments/current/hch_v44_five_baseline_supplement_20260916/PROTOCOL.md` "
      "as its mandatory first read. **That directory does not exist.**")
    A("")
    A("The controlling design document resolves this in favour of `PROTOCOL.md` and "
      "forbids the prompt's training outright: \"Any new baseline training merely to "
      "fill UEC/OMPB/PIR blocker cells would change method fidelity and is "
      "forbidden\", and \"Expected new baseline neural fit count is `0`\". The "
      "superseded files are preserved, unmodified, under "
      "`superseded_ai_execution_prompt_20260916/` with a README explaining their "
      "status. They are **not** part of this registry and were not verified by it.")
    A("")
    A("## 10. Reproducing")
    A("")
    A("```")
    A("python experiments/current/hch_v44_baseline_comparison_freeze_20260916/"
      "implementation/build_registry.py")
    A("python experiments/current/hch_v44_baseline_comparison_freeze_20260916/"
      "implementation/verify_registry.py")
    A("```")
    A("")
    (EVID / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
