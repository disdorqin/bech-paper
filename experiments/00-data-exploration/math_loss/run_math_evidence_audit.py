"""Math evidence audit — Phase B, C, D diagnostics from cached host predictions.

Reads host_cache from experiments/08-hch-v2/results/cache/ and produces:
  - 02_RESIDUAL_GEOMETRY.csv
  - 03_OCCURRENCE_MAGNITUDE.csv
  - 04_DAY_DEPENDENCE.csv
  - 05_DISTRIBUTION_FIT.csv
  - 06_CANDIDATE_ACTIONS.csv
  - 00_EXECUTIVE_EVIDENCE_VERDICT.md

Usage:
  python run_math_evidence_audit.py [--datasets LAGO_DE,LAGO_FR] [--backbones Linear,MLP]
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_utils import (
    residual_basic_stats, tail_stability_check,
    occ_magnitude_by_bins, day_dependence, distribution_compare,
    candidate_action_compare, day_block_bootstrap,
)

CACHE = ROOT / "experiments" / "08-hch-v2" / "results" / "cache"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DATA_META = {
    "LAGO_DE": {"currency": "EUR", "neg_price": True, "region": "EU"},
    "LAGO_BE": {"currency": "EUR", "neg_price": True, "region": "EU"},
    "LAGO_FR": {"currency": "EUR", "neg_price": True, "region": "EU"},
    "LAGO_PJM": {"currency": "USD", "neg_price": False, "region": "US"},
}


def load_cached(ds_key: str, bb_name: str) -> dict | None:
    """Load cached host predictions, truth, splits for one combo."""
    d = CACHE / ds_key / bb_name
    if not (d / "pred.npy").exists():
        return None
    pred = np.load(d / "pred.npy").astype(np.float64)
    y = np.load(d / "y.npy").astype(np.float64)
    with open(d / "seg.json") as f:
        seg = json.load(f)
    return {"pred": pred, "y": y, "seg": seg}


def get_day_and_hour(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate day_id and hour arrays for hourly data."""
    return np.arange(n) // 24, np.arange(n) % 24


def run_phase_b(ds_key: str, bb_name: str, data: dict):
    """Phase B: residual geometry statistics."""
    pred, y = data["pred"], data["y"]
    seg = data["seg"]
    s2_s, s2_e = seg["S2"][0], seg["S2"][1] + 1
    s3_s, s3_e = seg["S3"][0], seg["S3"][1] + 1

    r_s2 = pred[s2_s:s2_e] - y[s2_s:s2_e]
    r_s3 = pred[s3_s:s3_e] - y[s3_s:s3_e]
    day_s2, hour_s2 = get_day_and_hour(len(r_s2))
    day_s3, hour_s3 = get_day_and_hour(len(r_s3))

    meta = DATA_META.get(ds_key, {"currency": "?", "neg_price": False, "region": "?"})

    rows = []
    for split, r, day in [("S2", r_s2, day_s2), ("S3", r_s3, day_s3)]:
        row = {"dataset": ds_key, "backbone": bb_name, "split": split,
               "currency": meta["currency"], "neg_price_market": meta["neg_price"],
               "evidence_status": "COMPUTED"}
        row.update(residual_basic_stats(r, day))
        rows.append(row)
    return rows


def run_phase_b_tail(ds_key: str, bb_name: str, data: dict) -> dict:
    """Phase B: tail stability + outlier sensitivity."""
    pred, y = data["pred"], data["y"]
    seg = data["seg"]
    s2_s = seg["S2"][0]; s2_e = seg["S2"][1] + 1
    s3_s = seg["S3"][0]; s3_e = seg["S3"][1] + 1
    r_s2 = pred[s2_s:s2_e] - y[s2_s:s2_e]
    r_s3 = pred[s3_s:s3_e] - y[s3_s:s3_e]
    day_s2, _ = get_day_and_hour(len(r_s2))
    day_s3, _ = get_day_and_hour(len(r_s3))

    row = {"dataset": ds_key, "backbone": bb_name,
           "evidence_status": "COMPUTED"}
    row.update(tail_stability_check(r_s2, day_s2, r_s3, day_s3))

    # Sensitivity: remove top-3 largest residual events
    for rm_k, label in [(3, "_rm3"), (5, "_rm5"), (10, "_rm10")]:
        r2c = r_s2.copy()
        absr = np.abs(r2c)
        thr = np.sort(absr)[-min(rm_k, len(absr))]
        r2c[absr >= thr] = np.nan
        r2c = r2c[np.isfinite(r2c)]
        if len(r2c) > 50:
            row[f"skew{label}"] = float(np.median(np.abs(r2c)))
            row[f"M_neg{label}"] = float(np.mean(r2c[r2c < 0])) if (r2c < 0).sum() else 0.0
            row[f"M_pos{label}"] = float(np.mean(r2c[r2c > 0])) if (r2c > 0).sum() else 0.0
        else:
            row[f"skew{label}"] = None
            row[f"M_neg{label}"] = None
            row[f"M_pos{label}"] = None
    return row


def run_phase_b_occmag(ds_key: str, bb_name: str, data: dict) -> list:
    """Phase B4: occurrence-magnitude by host_pred bin."""
    pred, y = data["pred"], data["y"]
    seg = data["seg"]
    s2_s = seg["S2"][0]; s2_e = seg["S2"][1] + 1
    r_s2 = pred[s2_s:s2_e] - y[s2_s:s2_e]
    yhat_s2 = pred[s2_s:s2_e]
    day_s2, _ = get_day_and_hour(len(r_s2))

    rows = occ_magnitude_by_bins(r_s2, day_s2, yhat_s2)
    if isinstance(rows, dict) and rows.get("status"):
        return [{"dataset": ds_key, "backbone": bb_name, **rows}]
    for r in rows:
        r["dataset"] = ds_key
        r["backbone"] = bb_name
        r["evidence_status"] = "COMPUTED"
    return rows


def run_phase_b_dist(ds_key: str, bb_name: str, data: dict) -> dict:
    """Phase B5: distribution fitting (Normal, Laplace, Student-t)."""
    pred, y = data["pred"], data["y"]
    seg = data["seg"]
    s2_s = seg["S2"][0]; s2_e = seg["S2"][1] + 1
    s3_s = seg["S3"][0]; s3_e = seg["S3"][1] + 1
    r_s2 = pred[s2_s:s2_e] - y[s2_s:s2_e]
    r_s3 = pred[s3_s:s3_e] - y[s3_s:s3_e]
    day_s2, _ = get_day_and_hour(len(r_s2))
    day_s3, _ = get_day_and_hour(len(r_s3))

    row = {"dataset": ds_key, "backbone": bb_name, "evidence_status": "COMPUTED"}
    row.update(distribution_compare(r_s2, r_s3, day_s2, day_s3))
    return row


def run_phase_c(ds_key: str, bb_name: str, data: dict) -> list:
    """Phase C: candidate action comparison on S3."""
    pred, y = data["pred"], data["y"]
    seg = data["seg"]
    s3_s = seg["S3"][0]; s3_e = seg["S3"][1] + 1
    r_s3 = pred[s3_s:s3_e] - y[s3_s:s3_e]
    yhat_s3 = pred[s3_s:s3_e]
    day_s3, _ = get_day_and_hour(len(r_s3))

    # Compute simple candidates
    pi_neg = float(np.mean(r_s3 < 0))
    pi_pos = float(np.mean(r_s3 > 0))
    m_neg = float(np.mean(-r_s3[r_s3 < 0])) if (r_s3 < 0).sum() > 0 else 0.0
    m_pos = float(np.mean(r_s3[r_s3 > 0])) if (r_s3 > 0).sum() > 0 else 0.0

    candidates = {
        "identity": np.zeros_like(r_s3),
        "mean_residual": np.full_like(r_s3, np.mean(r_s3)),
        "median_residual": np.full_like(r_s3, np.median(r_s3)),
        "partial_moment_neg": np.where(r_s3 < 0, -pi_neg * m_neg, 0.0),
        "partial_moment_pos": np.where(r_s3 > 0, pi_pos * m_pos, 0.0),
        "side_cond_mean_neg": np.where(r_s3 < 0, -m_neg, 0.0),
        "side_cond_mean_pos": np.where(r_s3 > 0, m_pos, 0.0),
    }
    rows = candidate_action_compare(r_s3, candidates, day_s3)
    for r in rows:
        r["dataset"] = ds_key
        r["backbone"] = bb_name
        r["evidence_status"] = "COMPUTED"
    return rows


def run_phase_d(ds_key: str, bb_name: str, data: dict) -> dict:
    """Phase D: 24h day dependence diagnostics."""
    pred, y = data["pred"], data["y"]
    seg = data["seg"]
    s2_s = seg["S2"][0]; s2_e = seg["S2"][1] + 1
    r_s2 = pred[s2_s:s2_e] - y[s2_s:s2_e]
    day_s2, hour_s2 = get_day_and_hour(len(r_s2))

    row = {"dataset": ds_key, "backbone": bb_name, "evidence_status": "COMPUTED"}
    row.update(day_dependence(r_s2, day_s2, hour_s2))
    return row


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_verdict(all_results: dict):
    """Write executive evidence verdict."""
    lines = [
        "# Math Evidence Audit — Executive Verdict",
        f"Date: 2026-08-11",
        "",
        "## Data Scope",
    ]
    combos = all_results.get("combos", [])
    ds_set = set(c.split(" x ")[0] for c in combos)
    bb_set = set(c.split(" x ")[1] for c in combos)
    lines.append(f"- {len(ds_set)} datasets: {', '.join(sorted(ds_set))}")
    lines.append(f"- {len(bb_set)} backbones: {', '.join(sorted(bb_set))}")
    lines.append(f"- {len(combos)} dataset x backbone combinations")
    lines.append("")

    # NLL comparison
    dist_rows = all_results.get("dist_fit", [])
    t_better = sum(1 for r in dist_rows
                   if r.get("t_vs_normal_ci_hi", 0) is not None
                   and r.get("t_vs_normal_ci_hi", 0) < 0)
    t_better_lap = sum(1 for r in dist_rows
                       if r.get("t_vs_laplace_ci_hi", 0) is not None
                       and r.get("t_vs_laplace_ci_hi", 0) < 0)
    lines.append("## Distribution Fit Results")
    lines.append(f"- Student-t NLL < Normal NLL (S3, CI excludes 0): {t_better}/{len(dist_rows)}")
    lines.append(f"- Student-t NLL < Laplace NLL (S3, CI excludes 0): {t_better_lap}/{len(dist_rows)}")
    nu_vals = [r.get("t_nu") for r in dist_rows if r.get("t_nu")]
    if nu_vals:
        lines.append(f"- Fitted nu range: [{min(nu_vals):.1f}, {max(nu_vals):.1f}], median: {np.median(nu_vals):.1f}")
    lines.append("")

    # Tail stability
    tail_rows = all_results.get("tail_stability", [])
    skew_changes = []
    for r in tail_rows:
        s2 = r.get("S2_skew"); s3 = r.get("S3_skew")
        if s2 is not None and s3 is not None:
            skew_changes.append(abs(s3 - s2))
    if skew_changes:
        lines.append("## Tail Stability")
        lines.append(f"- |S3 skew - S2 skew| median: {np.median(skew_changes):.3f}")
        lines.append(f"- |S3 skew - S2 skew| max: {np.max(skew_changes):.3f}")
    lines.append("")

    # Day dependence
    dep_rows = all_results.get("day_dep", [])
    eranks = [r.get("effective_rank") for r in dep_rows if r.get("effective_rank")]
    first_evals = [r.get("first_eval_ratio") for r in dep_rows if r.get("first_eval_ratio")]
    max_corrs = [r.get("max_offdiag_corr") for r in dep_rows if r.get("max_offdiag_corr")]
    if eranks:
        lines.append("## 24h Dependence")
        lines.append(f"- Effective rank median: {np.median(eranks):.1f} / 24")
        lines.append(f"- First eigenvalue ratio median: {np.median(first_evals):.3f}")
        lines.append(f"- Max off-diagonal corr median: {np.median(max_corrs):.3f}")
        w1_ok = sum(1 for r in dep_rows
                    if (r.get("first_eval_ratio") or 1.0) < 0.4
                    and (r.get("max_offdiag_corr") or 1.0) < 0.3)
        lines.append(f"- W1 (hourly independence) appears adequate: {w1_ok}/{len(dep_rows)}")
    lines.append("")

    # Assumption verdicts
    lines.append("## Assumption Verdicts")
    verdicts = []
    if t_better >= 0.8 * len(dist_rows):
        verdicts.append("A1 (finite variance): Student-t consistently better than Normal → heavy tails confirmed")
    else:
        verdicts.append("A1: mixed evidence; check per-market")

    if skew_changes and np.median(skew_changes) < 1.0:
        verdicts.append("A4 (skewness stable): median S2→S3 skew change is small → asymmetry stable enough")
    else:
        verdicts.append("A4: skew not stable across splits; caution with M1")

    if w1_ok >= 0.7 * len(dep_rows):
        verdicts.append("A6 (W1 sufficient): day-dependence diagnostics do not require W2")
    else:
        verdicts.append("A6: significant day-level dependence; consider W2")

    for i, v in enumerate(verdicts):
        lines.append(f"{i+1}. {v}")

    lines.append("")
    lines.append("## Status")
    if t_better > 0 and len(dist_rows) > 0:
        lines.append("**PARTIAL_BLOCKED** — Core S2/S3 evidence produced.")
        lines.append("Missing: NEM/EPEX/GEFCOM markets (need host_cache), FS skew-t (M1), CAGM/DVG artifacts.")
    else:
        lines.append("**PARTIAL_BLOCKED** — insufficient data to conclude.")
    lines.append("")

    with open(OUT / "00_EXECUTIVE_EVIDENCE_VERDICT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default=None,
                    help="comma-separated, default all")
    ap.add_argument("--backbones", type=str, default=None,
                    help="comma-separated, default all")
    args = ap.parse_args()

    # Discover available combos
    combos = []
    for ds_dir in sorted(CACHE.iterdir()):
        if not ds_dir.is_dir():
            continue
        for bb_dir in sorted(ds_dir.iterdir()):
            if bb_dir.is_dir() and (bb_dir / "pred.npy").exists():
                combos.append((ds_dir.name, bb_dir.name))

    if args.datasets:
        ds_filter = set(args.datasets.split(","))
        combos = [(d, b) for d, b in combos if d in ds_filter]
    if args.backbones:
        bb_filter = set(args.backbones.split(","))
        combos = [(d, b) for d, b in combos if b in bb_filter]

    print(f"Found {len(combos)} dataset x backbone combinations")

    # Collectors
    residual_rows = []
    tail_rows = []
    occmag_rows = []
    dist_rows = []
    candidate_rows = []
    daydep_rows = []

    for i, (ds_key, bb_name) in enumerate(combos):
        data = load_cached(ds_key, bb_name)
        if data is None:
            print(f"[{i+1}/{len(combos)}] {ds_key} x {bb_name} SKIP (no cache)")
            continue

        print(f"[{i+1}/{len(combos)}] {ds_key} x {bb_name} ...", end=" ", flush=True)
        try:
            residual_rows.extend(run_phase_b(ds_key, bb_name, data))
            tail_rows.append(run_phase_b_tail(ds_key, bb_name, data))
            occmag_rows.extend(run_phase_b_occmag(ds_key, bb_name, data))
            dist_rows.append(run_phase_b_dist(ds_key, bb_name, data))
            candidate_rows.extend(run_phase_c(ds_key, bb_name, data))
            daydep_rows.append(run_phase_d(ds_key, bb_name, data))
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Write outputs
    write_csv(OUT / "02_RESIDUAL_GEOMETRY.csv", residual_rows)
    write_csv(OUT / "02b_TAIL_STABILITY.csv", tail_rows)
    write_csv(OUT / "03_OCCURRENCE_MAGNITUDE.csv", occmag_rows)
    write_csv(OUT / "05_DISTRIBUTION_FIT.csv", dist_rows)
    write_csv(OUT / "06_CANDIDATE_ACTIONS.csv", candidate_rows)
    write_csv(OUT / "04_DAY_DEPENDENCE.csv", daydep_rows)

    all_results = {
        "combos": [f"{d} x {b}" for d, b in combos],
        "dist_fit": dist_rows,
        "tail_stability": tail_rows,
        "day_dep": daydep_rows,
    }
    write_verdict(all_results)

    print(f"\nDone — outputs in {OUT}/")
    for fname in ["00_EXECUTIVE_EVIDENCE_VERDICT.md",
                  "02_RESIDUAL_GEOMETRY.csv", "02b_TAIL_STABILITY.csv",
                  "03_OCCURRENCE_MAGNITUDE.csv", "04_DAY_DEPENDENCE.csv",
                  "05_DISTRIBUTION_FIT.csv", "06_CANDIDATE_ACTIONS.csv"]:
        fp = OUT / fname
        if fp.exists():
            print(f"  {fname} ({fp.stat().st_size:,} bytes)")
        else:
            print(f"  {fname} MISSING")


if __name__ == "__main__":
    main()
