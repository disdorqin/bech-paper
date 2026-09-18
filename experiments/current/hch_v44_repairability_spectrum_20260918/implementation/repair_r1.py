"""R1 -- legal-history evidence audit (PROTOCOL.md Sec. 4, design Sec. 5).

Two products, in this order and never the other way round:

1. ``LEGAL_DESCRIPTOR_SCHEMA.json``, written **before** a single association is
   computed, fixing the descriptor list.  The list is a Python constant in
   ``repair_common``; nothing may be appended to it after this file has run.
2. The no-fit Host-to-residual analogue test.

The analogue test is deliberately not a retrieval model: no K search, no
softmax temperature, no learned distance.  For every target day with W=7
support it ranks the seven historical days twice -- once by the frozen
robust-scaled absolute Host view (L1) and once by the relative-Shape view
(cosine on the nonnegative normalised trajectory) -- and then asks how close the
nearest analogue's *residual geometry* is to the target's true geometry,
compared with two references the deployable system cannot see:

* the fixed W=7 summary (uniform mean), which is what a no-retrieval system
  would have to use;
* the oracle best historical day, chosen with target labels.

``frac_headroom_recovered`` is then defined per metric as
``(err_summary - err_legal) / (err_summary - err_oracle)`` -- the share of the
achievable gap between "use the average history" and "use the best possible
history day" that the legal nearest-Host analogue captures.  It is only defined
when the denominator is positive; the fraction of days where it is defined is
reported alongside, so an undefined-heavy aggregate can never be read as a win.

The row set and every legal input are seed-independent (``build_cell_data``
takes no seed; only the checkpoint does), so descriptors are emitted once per
(cell, day).  ``run`` verifies against the R0 atlas that the seed-invariant
columns really do agree across the three seeds before relying on that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import repair_common as R  # noqa: E402

METRICS = ("b_dist", "logB_dist", "shape_dist")
VIEWS = ("abs", "rel")
DENOM_TOL = 1e-9

#: Columns of the R0 atlas that depend only on frozen data, never on a seed.
SEED_INVARIANT_COLUMNS = (
    "host_mae", "residual_l1", "b_true", "B_true", "P_true", "N_true", "q_L", "q_B",
    "pos_mass_share", "neg_mass_share", "sign_changes", "residual_tv", "residual_roughness",
    "max_abs_resid_over_mae", "shape_entropy_pos", "shape_conc_pos",
    "shape_entropy_neg", "shape_conc_neg", "n_valid_hours",
)

DESCRIPTOR_DEFINITIONS = {
    "hd_mean": "robust-scaled mean of the frozen Host 24h trajectory, z=(v-center)/scale",
    "hd_std": "within-day population std of the robust-scaled Host trajectory",
    "hd_iqr": "within-day IQR (linear quantiles 0.75-0.25) of the robust-scaled Host trajectory",
    "hd_spread": "max-min of the robust-scaled Host trajectory",
    "hd_tv": "total absolute first difference of the robust-scaled Host trajectory",
    "hd_max_ramp": "maximum absolute hourly first difference of the robust-scaled Host trajectory",
    "hd_shape_entropy": "Shannon entropy (nats) of p=relu(host-min)/sum on the raw Host trajectory",
    "hd_shape_concentration": "max_h p_h of the same relative-Shape view",
    "hd_hour_max_sin": "sin(2*pi*argmax(host)/24)",
    "hd_hour_max_cos": "cos(2*pi*argmax(host)/24)",
    "hd_hour_min_sin": "sin(2*pi*argmin(host)/24)",
    "hd_hour_min_cos": "cos(2*pi*argmin(host)/24)",
    "hs_mae_med": "median over the 7 revealed history days of that day's mean|residual|",
    "hs_mae_mad": "MAD of the same 7 day MAEs",
    "hs_absb_med": "median over the 7 days of |b_j| (exact geometry through core.geometry)",
    "hs_absb_mad": "MAD of the same 7 |b_j|",
    "hs_logB_med": "median over the 7 days of log1p(B_j / s_B), s_B the frozen TRAIN mass scale",
    "hs_logB_mad": "MAD of the same 7 log1p(B_j / s_B)",
    "hs_qL_med": "median over the 7 days of the Level share q_L = H|b_j| / ||r_j||_1",
    "hs_qB_med": "median over the 7 days of the balanced share q_B = 2B_j / ||r_j||_1",
    "hs_trend_b": "OLS slope of b_j against the chronological index 0..6 within the window",
    "hs_trend_logB": "OLS slope of log1p(B_j/s_B) against the same index",
    "hs_signchange_med": "median over the 7 days of the residual sign-change count (zero hours skipped)",
    "hs_coh_pos": "median over the 7 days of W1(s+_j, mass-weighted barycentre of the historical s+)",
    "hs_coh_neg": "median over the 7 days of W1(s-_j, mass-weighted barycentre of the historical s-)",
    "hs_mae_iqr": "IQR of the 7 day MAEs (the registered dispersion of day MAE)",
    "hs_b_iqr": "IQR of the 7 b_j (the registered dispersion of b)",
    "hs_B_iqr": "IQR of the 7 B_j (the registered dispersion of B)",
}

LEGALITY = {
    "allowed_inputs": [
        "current frozen Host 24h trajectory for the target day",
        "the W=7 fully revealed historical residual days already carried by the target row "
        "(shadow-OOF TRAIN provenance, or prequential earlier-VAL provenance)",
        "the deterministic calendar/time origin",
        "availability masks",
        "frozen TRAIN-only scales: host RobustScaler(center, scale) and CoordinateScales.s_B",
    ],
    "forbidden_inputs_absent_by_construction": [
        "market or province identity",
        "Host family identity",
        "V2 TEST target / prediction / metric",
        "protected or final targets",
        "target-day realized covariates",
        "foreign or final outcomes",
        "any statistic computed from the target day's realized residual or from a model output",
    ],
    "seed_dependence": (
        "none: build_cell_data takes no seed and every descriptor is a function of frozen "
        "evidence fields; repair_r1 verifies the seed-invariant R0 atlas columns agree across "
        "the three seeds for every (cell, day) before deduplicating."
    ),
}


def write_schema() -> dict:
    schema = {
        "schema": "hch_v44_repairability_spectrum_legal_descriptors.v1",
        "protocol_id": R.PROTOCOL_ID,
        "status": "FROZEN BEFORE ANY ASSOCIATION WAS COMPUTED",
        "descriptor_order": list(R.DESCRIPTOR_KEYS),
        "n_descriptors": len(R.DESCRIPTOR_KEYS),
        "host_trajectory_descriptors": list(R.HOST_DESCRIPTOR_KEYS),
        "history_descriptors": list(R.HISTORY_DESCRIPTOR_KEYS),
        "definitions": DESCRIPTOR_DEFINITIONS,
        "legality": LEGALITY,
        "no_descriptor_added_after_seeing_correlations": True,
        "analogue_test_registration": {
            "absolute_view": "L1 over the frozen robust-scaled 24h Host trajectory",
            "relative_view": "1 - cosine between the nonnegative normalised Host Shape views",
            "K": "argmin over exactly the 7 revealed history days; no K search",
            "temperature": "none",
            "learned_distance": "none",
            "summary_reference_primary": "uniform mean over the 7 history days",
            "summary_reference_secondary": "mass-weighted mean, weights = the history day's ||r_j||_1",
            "oracle_reference": "argmin over the 7 history days of that metric's own distance; "
                                "target-aware upper bound only, never a deployable selector",
            "frac_headroom_recovered": "(err_summary_uniform - err_legal) / "
                                       "(err_summary_uniform - err_oracle), defined only when the "
                                       "denominator exceeds 1e-9",
            "metric_units": {
                "b_dist": "|b_j - b*| / s_b",
                "logB_dist": "|log1p(B_j/s_B) - log1p(B*/s_B)|",
                "shape_dist": "target-mass-weighted W1, weights = the target day's P and N",
            },
            "association_reported": [
                "rho_perday_median: median over days of the within-day Spearman over the 7 pairs",
                "rho_pooled_7xN: Spearman over all 7*N pairs pooled",
            ],
        },
    }
    R.json_dump(R.EVID / "LEGAL_DESCRIPTOR_SCHEMA.json", schema)
    return schema


def check_seed_invariance(atlas: list[dict]) -> dict:
    """Every frozen-data column must be identical across the three seeds."""
    seen, bad = {}, []
    for row in atlas:
        key = (row["cell"], row["day"])
        sig = tuple(float(row[c]) for c in SEED_INVARIANT_COLUMNS)
        if key in seen:
            if seen[key] != sig:
                bad.append(f"{key[0]} {key[1]}")
        else:
            seen[key] = sig
    return {
        "n_days": len(seen),
        "n_seed_invariant_columns": len(SEED_INVARIANT_COLUMNS),
        "mismatched_days": bad[:20],
        "passed": not bad,
    }


def _summary_vectors(hist_res: np.ndarray, weights: np.ndarray) -> dict:
    g = R.geometry_np(hist_res)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
    return {
        "uniform": {"b": float(np.mean(g["b"])), "B": float(np.mean(g["B"])),
                    "s_plus": g["s_plus"].mean(axis=0), "s_minus": g["s_minus"].mean(axis=0)},
        "massweighted": {"b": float((w * g["b"]).sum()), "B": float((w * g["B"]).sum()),
                         "s_plus": (w[:, None] * g["s_plus"]).sum(axis=0),
                         "s_minus": (w[:, None] * g["s_minus"]).sum(axis=0)},
    }


def _metric_distances(hist_geom: dict, target_geom: dict, s_b: float, s_B: float) -> dict:
    n = len(hist_geom["b"])
    P_t, N_t = float(target_geom["P"][0]), float(target_geom["N"][0])
    denom = P_t + N_t + 1e-12
    return {
        "b_dist": np.abs(hist_geom["b"] - float(target_geom["b"][0])) / max(s_b, 1e-12),
        "logB_dist": np.abs(
            np.log1p(np.maximum(hist_geom["B"], 0.0) / max(s_B, 1e-12))
            - np.log1p(max(float(target_geom["B"][0]), 0.0) / max(s_B, 1e-12))),
        "shape_dist": np.array([
            (P_t * R.w1_ordered(hist_geom["s_plus"][j], target_geom["s_plus"][0])
             + N_t * R.w1_ordered(hist_geom["s_minus"][j], target_geom["s_minus"][0])) / denom
            for j in range(n)]),
    }


def _one_metric(summary: dict, target_geom: dict, metric: str, s_b: float, s_B: float) -> float:
    P_t, N_t = float(target_geom["P"][0]), float(target_geom["N"][0])
    denom = P_t + N_t + 1e-12
    if metric == "b_dist":
        return abs(summary["b"] - float(target_geom["b"][0])) / max(s_b, 1e-12)
    if metric == "logB_dist":
        return abs(np.log1p(max(summary["B"], 0.0) / max(s_B, 1e-12))
                   - np.log1p(max(float(target_geom["B"][0]), 0.0) / max(s_B, 1e-12)))
    return (P_t * R.w1_ordered(summary["s_plus"], target_geom["s_plus"][0])
            + N_t * R.w1_ordered(summary["s_minus"], target_geom["s_minus"][0])) / denom


def cell_analogue(market: str, host: str) -> tuple:
    import repair_r0 as R0

    data = R.RT.build_cell_data(market, host)
    s_b = float(data["scales"].s_b)
    s_B = float(data["scales"].s_B)
    center = float(data["scaler"].center[0])
    scale = float(data["scaler"].scale[0])
    lookup, lookup_role = R0._frames_host_lookup(market, host)

    analogue, descriptors = [], []
    degenerate = {"n_missing_history_trajectory": 0, "n_zero_historical_pos_mass": 0,
                  "n_zero_historical_neg_mass": 0}
    for tag, split_rows in (("TRAIN", data["train_rows"]), ("VAL", data["val_rows"])):
        for row in split_rows:
            days = [h.day for h in row.history]
            if any(d not in lookup for d in days):
                degenerate["n_missing_history_trajectory"] += 1
                continue

            cur = np.asarray(row.host_pred, dtype=np.float64)
            hist_res = np.stack([np.asarray(h.residual, dtype=np.float64) for h in row.history])
            hist_traj = np.stack([lookup[d] for d in days])
            hist_geom = R.geometry_np(hist_res)
            target_geom = R.geometry_np(np.asarray(row.residual, dtype=np.float64)[None, :])

            if float(hist_geom["P"].sum()) <= 0:
                degenerate["n_zero_historical_pos_mass"] += 1
            if float(hist_geom["N"].sum()) <= 0:
                degenerate["n_zero_historical_neg_mass"] += 1

            descriptors.append({
                "cell": R.cell_key(market, host), "market": market, "host": host,
                "role": tag, "day": row.day.isoformat(),
                **R.host_descriptor_vector(cur, center, scale),
                **R.history_descriptor_vector(hist_res, s_B),
            })

            z_cur = (cur - center) / max(scale, 1e-12)
            z_hist = (hist_traj - center) / max(scale, 1e-12)
            p_cur = R.shape_distribution(cur)
            p_hist = np.stack([R.shape_distribution(v) for v in hist_traj])
            d_abs = np.abs(z_hist - z_cur[None, :]).mean(axis=1)
            cos = (p_hist * p_cur[None, :]).sum(axis=1) / np.maximum(
                np.linalg.norm(p_hist, axis=1) * np.linalg.norm(p_cur), 1e-12)
            d_rel = 1.0 - cos

            metric_d = _metric_distances(hist_geom, target_geom, s_b=s_b, s_B=s_B)
            summary = _summary_vectors(hist_res, weights=hist_geom["m1"])

            rec = {
                "cell": R.cell_key(market, host), "market": market, "host": host,
                "role": tag, "day": row.day.isoformat(), "n_history": len(days),
                "history_roles": sorted({lookup_role[d] for d in days}),
                "history_provenance": sorted({h.provenance for h in row.history}),
                "nearest_abs_index": int(np.argmin(d_abs)),
                "nearest_rel_index": int(np.argmin(d_rel)),
                "d_abs_min": float(d_abs.min()), "d_rel_min": float(d_rel.min()),
                "d_abs_at_rel_nearest": float(d_abs[int(np.argmin(d_rel))]),
                "d_rel_at_abs_nearest": float(d_rel[int(np.argmin(d_abs))]),
                "d_abs_vec": d_abs.tolist(), "d_rel_vec": d_rel.tolist(),
            }
            for metric in METRICS:
                rec[f"metric_{metric}_vec"] = metric_d[metric].tolist()
                rec[f"oracle_{metric}"] = float(metric_d[metric].min())
                rec[f"oracle_day_{metric}"] = int(np.argmin(metric_d[metric]))
                for name in ("uniform", "massweighted"):
                    rec[f"summary_{name}_{metric}"] = _one_metric(summary[name], target_geom, metric, s_b, s_B)
                base = rec[f"summary_uniform_{metric}"]
                denom = base - rec[f"oracle_{metric}"]
                rec[f"frac_denominator_positive_{metric}"] = bool(denom > DENOM_TOL)
                for view, dist in (("abs", d_abs), ("rel", d_rel)):
                    j = int(np.argmin(dist))
                    rec[f"legal_{view}_{metric}"] = float(metric_d[metric][j])
                    rec[f"frac_{view}_{metric}"] = (
                        (base - float(metric_d[metric][j])) / denom if denom > DENOM_TOL else None)
                    rec[f"rho_{view}_{metric}"] = R.spearman(dist.tolist(), metric_d[metric].tolist())
                    rec[f"legal_{view}_day_{metric}"] = days[j]
            analogue.append(rec)
    return analogue, descriptors, degenerate


def _aggregate(per_day: list[dict]) -> list[dict]:
    groups = [("ALL", lambda r: "ALL"),
              ("MARKET", lambda r: r["market"]),
              ("HOST_FAMILY", lambda r: r["host"]),
              ("LOW_GAIN_CELLS", lambda r: ("LOW_GAIN_CELLS"
                                            if (r["market"], r["host"]) in R.LOW_GAIN_CELLS else None))]
    rows = []
    for group_type, key_fn in groups:
        buckets = {}
        for rec in per_day:
            name = key_fn(rec)
            if name is not None:
                buckets.setdefault(name, []).append(rec)
        for name, members in sorted(buckets.items()):
            for view in VIEWS:
                for metric in METRICS:
                    rhos = [m[f"rho_{view}_{metric}"] for m in members if m[f"rho_{view}_{metric}"] is not None]
                    fracs = [m[f"frac_{view}_{metric}"] for m in members if m[f"frac_{view}_{metric}"] is not None]
                    pooled_x, pooled_y = [], []
                    for m in members:
                        pooled_x.extend(m[f"d_{view}_vec"])
                        pooled_y.extend(m[f"metric_{metric}_vec"])
                    rows.append({
                        "group_type": group_type, "group": name, "view": view, "metric": metric,
                        "n_days": len(members),
                        "rho_perday_median": R.median(rhos),
                        "rho_perday_positive_frac": (float(np.mean([r > 0 for r in rhos])) if rhos else None),
                        "rho_pooled_7xN": R.spearman(pooled_x, pooled_y),
                        "n_pairs_pooled": len(pooled_x),
                        "frac_headroom_recovered_median": R.median(fracs),
                        "frac_defined_frac": (len(fracs) / len(members) if members else None),
                        "err_summary_uniform_median": R.median([m[f"summary_uniform_{metric}"] for m in members]),
                        "err_summary_massweighted_median": R.median([m[f"summary_massweighted_{metric}"] for m in members]),
                        "err_legal_median": R.median([m[f"legal_{view}_{metric}"] for m in members]),
                        "err_oracle_median": R.median([m[f"oracle_{metric}"] for m in members]),
                    })
    return rows


def run() -> dict:
    import repair_r0 as R0

    schema = write_schema()
    atlas = R.read_table_parquet(R.EVID / "PER_DAY_REPAIRABILITY.parquet")
    invariance = check_seed_invariance(atlas)
    if not invariance["passed"]:
        raise RuntimeError("R0 atlas is not seed-invariant on frozen-data columns")

    all_rec, all_desc, degenerates = [], [], {}
    for market, host in R.DOM_PANEL:
        key = R.cell_key(market, host)
        analogue, descriptors, degen = cell_analogue(market, host)
        all_rec.extend(analogue)
        all_desc.extend(descriptors)
        degenerates[key] = degen
        print(f"[r1] {key}: {len(analogue)} analogue rows, {len(descriptors)} descriptor rows", flush=True)

    atlas_keys = {(r["cell"], r["day"]) for r in atlas}
    desc_keys = {(r["cell"], r["day"]) for r in all_desc}
    anlg_keys = {(r["cell"], r["day"]) for r in all_rec}
    coverage = {
        "n_atlas_cell_days": len(atlas_keys),
        "n_descriptor_cell_days": len(desc_keys),
        "n_analogue_cell_days": len(anlg_keys),
        "descriptors_cover_atlas": atlas_keys == desc_keys,
        "analogue_covers_atlas": atlas_keys == anlg_keys,
        "atlas_days_missing_analogue": sorted(f"{c} {d}" for c, d in (atlas_keys - anlg_keys))[:20],
    }
    if not (coverage["descriptors_cover_atlas"] and coverage["analogue_covers_atlas"]):
        raise RuntimeError(f"R1 day coverage does not match the R0 atlas: {coverage}")

    R.write_table_parquet(R.EVID / "PER_DAY_ANALOGUE.parquet", all_rec)
    R.write_table_parquet(R.EVID / "LEGAL_DESCRIPTORS.parquet", all_desc)
    agg = _aggregate(all_rec)
    R.write_csv(R.EVID / "HOST_RESIDUAL_ANALOGUE.csv", list(agg[0]), agg)

    summary = {
        "schema": "hch_v44_repairability_spectrum_analogue.v1",
        "protocol_id": R.PROTOCOL_ID,
        "n_analogue_rows": len(all_rec),
        "n_descriptor_rows": len(all_desc),
        "n_cells": len(R.DOM_PANEL),
        "k_search": 0, "temperature": None, "learned_retrieval": False,
        "seed_invariance_check": invariance,
        "coverage": coverage,
        "degenerate": degenerates,
        "descriptor_schema_sha256": R.sha256_file(R.EVID / "LEGAL_DESCRIPTOR_SCHEMA.json"),
        "n_descriptors": schema["n_descriptors"],
    }
    R.json_dump(R.EVID / "R1_ANALOGUE_SUMMARY.json", summary)
    return summary


if __name__ == "__main__":
    print(run())
