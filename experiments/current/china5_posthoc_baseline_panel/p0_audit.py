"""P0 data-contract audit for the China-5 post-hoc baseline panel.

Establishes, per market, before any Host is trained or any baseline is run:
  forecast origin, time resolution, timezone, target column, legal forecast-known
  target-day features, forbidden features, missing/duplicate policy, provenance
  and hash.  A market whose contract fails here never reaches Host training.

Run:  python experiments/current/china5_posthoc_baseline_panel/p0_audit.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import panel_contract as pc  # noqa: E402

OUT = ROOT / "experiments/evidence/china5_posthoc_baseline_panel_20260912"
CONTRACTS = OUT / "01_dataset_contracts"


def hour_label_profile(ts: pd.Series) -> dict:
    """Empirically decide the hour-label convention and quantify grid gaps."""
    hours = ts.dt.hour.value_counts().sort_index()
    has_midnight = int(hours.get(0, 0))
    diffs = ts.diff().dropna()
    dt = diffs.value_counts().sort_index()
    return {
        "n_rows": int(len(ts)),
        "n_unique_timestamps": int(ts.nunique()),
        "n_duplicate_timestamps": int(ts.duplicated().sum()),
        "n_unparseable": int(ts.isna().sum()),
        "monotonic_increasing": bool(ts.is_monotonic_increasing),
        "min_timestamp": str(ts.min()),
        "max_timestamp": str(ts.max()),
        "hour_labels_present": sorted(int(h) for h in hours.index),
        "n_rows_labelled_00": has_midnight,
        "n_rows_labelled_01": int(hours.get(1, 0)),
        "step_counts": {str(k): int(v) for k, v in dt.items()},
        "dominant_step": str(dt.index[0]) if len(dt) else None,
        "resolution_consistent": bool(len(dt) == 1 and str(dt.index[0]) == "0 days 01:00:00"),
        "observed_span_hours": int((ts.max() - ts.min()).total_seconds() // 3600),
        "n_missing_grid_hours": int((ts.max() - ts.min()).total_seconds() // 3600 + 1 - len(ts)),
    }


def target_profile(market: str, contract: pc.DayContract, y: np.ndarray, segs: np.ndarray) -> dict:
    """Target statistics computed only over open roles (PROTECTED_FINAL never read)."""
    flat = y.reshape(-1)
    prof = {
        "column": pc.MARKETS[market]["target"],
        "n_values": int(flat.size),
        "n_nonfinite": int((~np.isfinite(flat)).sum()),
        "min": float(np.nanmin(flat)), "max": float(np.nanmax(flat)),
        "mean": float(np.nanmean(flat)), "median": float(np.nanmedian(flat)),
        "std": float(np.nanstd(flat)),
        "n_negative": int((flat < 0).sum()),
        "n_zero": int((flat == 0).sum()),
        "pct_negative": float(100.0 * (flat < 0).mean()),
        "pct_zero": float(100.0 * (flat == 0).mean()),
        "by_role": {},
    }
    for r in pc.ROLES:
        z = y[segs == r].reshape(-1)
        if z.size == 0:
            continue
        prof["by_role"][r] = {
            "n_days": int((segs == r).sum()), "n_values": int(z.size),
            "min": float(z.min()), "max": float(z.max()),
            "mean": float(z.mean()), "median": float(np.median(z)),
            "pct_negative": float(100.0 * (z < 0).mean()),
            "pct_zero": float(100.0 * (z == 0).mean()),
        }
    return prof


def legal_profile(market: str, contract: pc.DayContract, contract_days, origins) -> dict:
    cols = pc.legal_columns(market)
    z, meta = pc.load_legal_state(market, origins, cols)
    per_col = {}
    for i, c in enumerate(cols):
        v = z[:, :, i].reshape(-1)
        per_col[c] = {"min": float(v.min()), "max": float(v.max()),
                      "mean": float(v.mean()), "n_nonfinite": int((~np.isfinite(v)).sum())}
    return {"n_legal_columns": len(cols), "legal_columns": cols, "read_audit": meta,
            "per_column_stats": per_col,
            "realized_columns_excluded": pc.realized_columns(market),
            "excluded_by_policy": pc.EXCLUDED_FEATURES}


def audit_market(market: str) -> dict:
    path = pc.source_path(market)
    if not path.exists():
        return {"physical_market_group": market, "status": "DATA_BLOCKED",
                "reason": f"missing source {path}"}
    ts = pc.read_timestamps_only(market)
    label = hour_label_profile(ts)
    contract = pc.build_day_contract(market)
    manifest = pc.split_manifest(contract, market)
    ctx, y, stamps, segs, ds_id, phys, read_audit = pc.load_target_windows(market, contract)
    tprof = target_profile(market, contract, y, segs)
    lprof = legal_profile(market, contract, contract, stamps)

    header = pc.read_header(market)
    _row_count = len(ts)
    eligible = len(contract.positions)
    return {
        "physical_market_group": market,
        "dataset_id": ds_id,
        "status": "CONTRACT_OK",
        "provenance": {
            "source_path": pc.MARKETS[market]["path"],
            "source_sha256": pc.sha256_file(path),
            "source_bytes": int(path.stat().st_size),
            "reader": pc.MARKETS[market]["reader"],
            "encoding": pc.MARKETS[market].get("encoding", "utf-8"),
            "raw_columns": header,
            "n_raw_rows": _row_count,
            "declared_research_role": ("private_external_case" if not pc.MARKETS[market]["public"]
                                       else "supplementary_external_case"),
            "public": pc.MARKETS[market]["public"],
        },
        "timestamp_contract": label,
        "horizon_contract": {
            "seq_len": pc.SEQ, "horizon": pc.H, "forecast_origin": "00:00 UTC+8 of target day D",
            "origin_rule": "origin = first row of day D; target = origin+1h .. origin+24h",
            "timezone": pc.TZ,
            "dst_policy": "China has observed no DST since 1991-09-15; no DST branch is needed",
            "hour_label_convention": "hour_ending",
            "day_definition": "day D = rows labelled D 01:00 .. D+1 00:00 (24 contiguous hours)",
        },
        "target_contract": {
            "target_column": pc.MARKETS[market]["target"],
            "task": "DA_price -> next-24h DA_price",
            "target_day_price_is_not_an_input": True,
            "forbidden_target_day_inputs": list(pc.FORBIDDEN_TARGET_DAY_INPUTS),
            "profile": tprof,
        },
        "feature_contract": lprof,
        "missing_duplicate_policy": {
            "imputation": "none - no ffill/bfill/interpolation anywhere in the panel pipeline",
            "duplicate_timestamps": "hard error",
            "non_finite_legal_feature": "hard error",
            "incomplete_day": "excluded from the eligible-episode set (never patched)",
            "gap_policy": "gaps are preserved; a day is eligible only with 24 contiguous hours "
                          "AND a contiguous 168h causal context",
        },
        "split": manifest,
        "eligible_days": eligible,
        "raw_rows": _row_count,
        "read_audit": read_audit,
        "role_day_span": {
            r: {"n": int(sum(1 for v in contract.segments.values() if v == r)),
                "first": min([d for d, v in contract.segments.items() if v == r], default=None),
                "last": max([d for d, v in contract.segments.items() if v == r], default=None)}
            for r in pc.ROLES},
    }


def main() -> int:
    CONTRACTS.mkdir(parents=True, exist_ok=True)
    rows, all_ok = [], True
    report = {"schema": "china5_dataset_registry.v1", "markets": {},
              "non_runnable": pc.NON_RUNNABLE}
    for market in list(pc.MARKETS) + list(pc.NON_RUNNABLE):
        if market in pc.NON_RUNNABLE:
            rec = dict(pc.NON_RUNNABLE[market])
            rec["physical_market_group"] = market
            report["markets"][market] = rec
            rows.append(dict(physical_market_group=market, dataset_id=rec["dataset_id"],
                             status=rec["status"], n_eligible_days="", n_raw_rows="",
                             source_path="", source_sha256="", public="", reason=rec["reason"]))
            print(f"[P0] {market:10s} {rec['status']}")
            continue
        try:
            rec = audit_market(market)
        except Exception as exc:  # fail closed
            rec = {"physical_market_group": market, "status": "CONTRACT_INVALID",
                   "reason": f"{type(exc).__name__}: {exc}"}
            all_ok = False
        report["markets"][market] = rec
        pc.dump_json(CONTRACTS / market / "DATASET_CONTRACT.json", rec)
        if rec["status"] == "CONTRACT_OK":
            c = rec["split"]["counts"]
            print(f"[P0] {market:10s} OK  rows={rec['raw_rows']:6d} eligible={rec['eligible_days']:4d} "
                  f"roles={c}")
            rows.append(dict(physical_market_group=market, dataset_id=rec["dataset_id"],
                             status="CONTRACT_OK", n_eligible_days=rec["eligible_days"],
                             n_raw_rows=rec["raw_rows"],
                             source_path=rec["provenance"]["source_path"],
                             source_sha256=rec["provenance"]["source_sha256"],
                             public=rec["provenance"]["public"], reason=""))
        else:
            print(f"[P0] {market:10s} {rec['status']} {rec.get('reason','')}")
            rows.append(dict(physical_market_group=market, dataset_id=market + "_DA",
                             status=rec["status"], n_eligible_days="", n_raw_rows="",
                             source_path="", source_sha256="", public="", reason=rec.get("reason", "")))

    pc.dump_json(OUT / "00_protocol" / "P0_DATASET_AUDIT.json", report)
    with (OUT / "04_registry" / "DATASET_REGISTRY.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[P0] wrote {OUT/'04_registry'/'DATASET_REGISTRY.csv'}")
    print(f"[P0] overall_contract_ok={all_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
