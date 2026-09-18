"""R2 -- fixed low-capacity predictability probes (PROTOCOL.md Sec. 5, design Sec. 6).

Diagnostic probes only; they are not candidate models and no threshold is chosen
for deployment.

Registered design:

* sample = the pooled **TRAIN** cell-days of all 20 cells, cell-macro-balanced
  (each cell contributes total weight 1, so a 72-row cell and a 224-row cell
  weigh the same in the fit);
* features = exactly the 28 frozen legal descriptors of
  ``LEGAL_DESCRIPTOR_SCHEMA.json`` -- no market ID, no Host ID, nothing else;
* five leave-one-market-out folds; in each fold the descriptor standardisation
  and the constant baseline are fit on the training folds only;
* model = ``sklearn.linear_model.Ridge(alpha=1.0)``, no grid, no search, no
  other estimator;
* targets = the four registered ones, each a per-cell-day median across the
  three seeds (the project's median-across-seeds-first rule);
* metrics = held-out Spearman rho, probe MAE, constant-training-median MAE, R2
  (descriptive only), and the sign of the top-vs-bottom quartile difference of
  the realized target, quartiles taken on the predicted score.

The frozen O1 correction ratio is additionally reported as a single-scalar
descriptive predictor of realized O1 Host-relative utility; it is never fitted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import repair_common as R  # noqa: E402

TARGETS = {
    "log1p_alpha_star": "log1p(median across seeds of the per-day ray-optimal alpha*)",
    "ray_headroom_pct": "median across seeds of the per-day ray-oracle headroom, % of that day's Host MAE",
    "B_headroom_pct": "median across seeds of the per-day O_B balanced-mass swap headroom, % of Host MAE",
    "shape_headroom_pct": "median across seeds of the per-day O_S Shape swap headroom, % of Host MAE",
}
TARGET_COLUMN = {
    "log1p_alpha_star": "alpha_star", "ray_headroom_pct": "increment_ray_pct",
    "B_headroom_pct": "imp_oB_pct", "shape_headroom_pct": "imp_oS_pct",
}
#: Which registered headroom family each target belongs to (design Sec. 7).
TARGET_FAMILY = {
    "log1p_alpha_star": "distance", "ray_headroom_pct": "distance",
    "B_headroom_pct": "balanced_mass", "shape_headroom_pct": "shape",
}


def _median_across_seeds(atlas: list[dict], column: str, role: str) -> dict:
    by_key = {}
    for row in atlas:
        if row["role"] != role:
            continue
        by_key.setdefault((row["cell"], row["day"]), []).append(row[column])
    return {k: R.median(v) for k, v in by_key.items()}


def build_sample(atlas: list[dict], descriptors: list[dict], role: str = "TRAIN") -> list[dict]:
    desc_index = {(d["cell"], d["day"]): d for d in descriptors if d["role"] == role}
    targets = {name: _median_across_seeds(atlas, col, role) for name, col in TARGET_COLUMN.items()}
    o1_gain = {}
    ratio = {}
    for row in atlas:
        if row["role"] != role:
            continue
        key = (row["cell"], row["day"])
        host = float(row["host_mae"])
        o1_gain.setdefault(key, []).append(100.0 * (host - float(row["o1_mae"])) / host)
        ratio.setdefault(key, []).append(float(row["o1_correction_ratio"]))

    sample = []
    for key, desc in sorted(desc_index.items()):
        if any(targets[name].get(key) is None for name in TARGETS):
            continue
        rec = {"cell": key[0], "market": desc["market"], "host": desc["host"], "day": key[1],
               "o1_gain_pct": R.median(o1_gain.get(key, [])),
               "o1_correction_ratio": R.median(ratio.get(key, []))}
        rec["y_log1p_alpha_star"] = float(np.log1p(max(float(targets["log1p_alpha_star"][key]), 0.0)))
        for name in ("ray_headroom_pct", "B_headroom_pct", "shape_headroom_pct"):
            rec[f"y_{name}"] = float(targets[name][key])
        for k in R.DESCRIPTOR_KEYS:
            rec[k] = float(desc[k])
        sample.append(rec)
    return sample


#: Explicit, because the per-fold rows and the ``ALL_FOLDS`` summary rows are not
#: the same shape (only the summary carries ``n_folds_probe_beats_const``).
RESULT_HEADER = [
    "target", "family", "held_out_market", "n_train_rows", "n_test_rows",
    "n_train_cells", "n_test_cells", "spearman_rho", "probe_mae", "const_median_mae",
    "r2_descriptive", "quartile_top_minus_bottom", "quartile_diff_sign",
    "probe_beats_const", "n_folds_probe_beats_const",
]


def _weights(sample: list[dict]) -> np.ndarray:
    counts = {}
    for row in sample:
        counts[row["cell"]] = counts.get(row["cell"], 0) + 1
    w = np.array([1.0 / counts[row["cell"]] for row in sample], dtype=np.float64)
    return w / w.mean()


def run() -> dict:
    from sklearn.linear_model import Ridge

    atlas = R.read_table_parquet(R.EVID / "PER_DAY_REPAIRABILITY.parquet")
    descriptors = R.read_table_parquet(R.EVID / "LEGAL_DESCRIPTORS.parquet")
    sample = build_sample(atlas, descriptors, role="TRAIN")
    markets = sorted({row["market"] for row in sample}, key=R.MARKETS.index)
    x_all = np.array([[row[k] for k in R.DESCRIPTOR_KEYS] for row in sample], dtype=np.float64)
    w_all = _weights(sample)
    hosts_by_row = [row["host"] for row in sample]

    result_rows, coef_rows = [], []
    for target in TARGETS:
        y_all = np.array([row[f"y_{target}"] for row in sample], dtype=np.float64)
        fold_records = []
        for held in markets:
            test = np.array([row["market"] == held for row in sample])
            train = ~test
            mu, sd = x_all[train].mean(axis=0), x_all[train].std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            xs, xt = (x_all[train] - mu) / sd, (x_all[test] - mu) / sd
            # PROTOCOL Sec. 5 registers "macro-balance cells in the training
            # loss/data weights".  The balance is a property of the fold, so the
            # training slice is renormalised to mean weight 1 *within the fold*:
            # with alpha fixed at 1.0 by the registration, a global constant on
            # the weights would silently rescale the penalty.
            sw = w_all[train]
            sw = sw / sw.mean()
            model = Ridge(alpha=1.0, fit_intercept=True)
            model.fit(xs, y_all[train], sample_weight=sw)
            pred = model.predict(xt)
            actual = y_all[test]
            const = float(np.median(y_all[train]))
            const_pred = np.full(actual.shape, const)
            ss_res = float(((actual - pred) ** 2).sum())
            ss_tot = float(((actual - actual.mean()) ** 2).sum())
            q1, q3 = np.quantile(pred, [0.25, 0.75])
            top = actual[pred >= q3]
            bottom = actual[pred <= q1]
            rho = R.spearman(pred.tolist(), actual.tolist())
            fold_records.append({
                "target": target, "family": TARGET_FAMILY[target], "held_out_market": held,
                "n_train_rows": int(train.sum()), "n_test_rows": int(test.sum()),
                "n_train_cells": len({row["cell"] for row, m in zip(sample, train) if m}),
                "n_test_cells": len({row["cell"] for row, m in zip(sample, test) if m}),
                "spearman_rho": rho,
                "probe_mae": float(np.abs(actual - pred).mean()),
                "const_median_mae": float(np.abs(actual - const_pred).mean()),
                "r2_descriptive": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
                "quartile_top_minus_bottom": (float(np.median(top) - np.median(bottom))
                                              if len(top) and len(bottom) else None),
                "quartile_diff_sign": R.sign_of(np.median(top) - np.median(bottom))
                if len(top) and len(bottom) else 0,
                "probe_beats_const": bool(np.abs(actual - pred).mean() < np.abs(actual - const_pred).mean()),
            })
            for j, key in enumerate(R.DESCRIPTOR_KEYS):
                coef_rows.append({
                    "target": target, "family": TARGET_FAMILY[target], "held_out_market": held,
                    "descriptor": key, "coef_standardised": float(model.coef_[j]),
                    "coef_sign": R.sign_of(model.coef_[j]),
                    "oof_spearman_with_target": R.spearman(
                        xt[:, j].tolist(), actual.tolist()),
                })
        probe_beats = sum(1 for f in fold_records if f["probe_beats_const"])
        summary = {
            "target": target, "family": TARGET_FAMILY[target], "held_out_market": "ALL_FOLDS",
            "n_train_rows": None, "n_test_rows": None, "n_train_cells": None, "n_test_cells": None,
            "spearman_rho": R.median([f["spearman_rho"] for f in fold_records]),
            "probe_mae": R.median([f["probe_mae"] for f in fold_records]),
            "const_median_mae": R.median([f["const_median_mae"] for f in fold_records]),
            "r2_descriptive": R.median([f["r2_descriptive"] for f in fold_records]),
            "quartile_top_minus_bottom": R.median([f["quartile_top_minus_bottom"] for f in fold_records]),
            "quartile_diff_sign": R.sign_of(R.median([f["quartile_top_minus_bottom"] for f in fold_records])),
            "probe_beats_const": bool(probe_beats >= 4),
            "n_folds_probe_beats_const": probe_beats,
        }
        result_rows.extend(fold_records + [summary])

    # --- the frozen O1 correction-ratio scalar, never fitted ------------------
    scalar_rows = []
    for group_type, key_fn in (("POOLED", lambda r: "POOLED"), ("MARKET", lambda r: r["market"])):
        buckets = {}
        for row in sample:
            buckets.setdefault(key_fn(row), []).append(row)
        for name, members in sorted(buckets.items()):
            for target in ("o1_gain_pct",):
                scalar_rows.append({
                    "group_type": group_type, "group": name, "predictor": "o1_correction_ratio",
                    "target": target, "n_rows": len(members),
                    "spearman_rho": R.spearman([m["o1_correction_ratio"] for m in members],
                                               [m[target] for m in members]),
                    "note": "descriptive only; no fit, no threshold, no gate",
                })

    R.write_table_parquet(R.EVID / "R2_PROBE_SAMPLE.parquet", sample)
    R.write_csv(R.EVID / "LOMO_PROBE_RESULTS.csv", RESULT_HEADER, result_rows)
    R.write_csv(R.EVID / "LOMO_PROBE_COEFFICIENTS.csv", list(coef_rows[0]), coef_rows)
    R.write_csv(R.EVID / "R2_O1_CORRECTION_RATIO_SCALAR.csv", list(scalar_rows[0]), scalar_rows)

    import repair_gate as G

    gate_c = G.gate_c_evidence(result_rows, coef_rows)
    out = {
        "schema": "hch_v44_repairability_spectrum_lomo_probe.v1",
        "protocol_id": R.PROTOCOL_ID,
        "model": "sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True)",
        "n_descriptors": len(R.DESCRIPTOR_KEYS),
        "descriptor_order": list(R.DESCRIPTOR_KEYS),
        "targets": TARGETS,
        "target_family": TARGET_FAMILY,
        "sample": {"role": "TRAIN", "n_rows": len(sample),
                   "n_cells": len({row["cell"] for row in sample}),
                   "markets": markets,
                   "cell_macro_balanced": True,
                   "scaling": "descriptor mean/std fit on the training folds only"},
        "folds": "leave-one-market-out, 5 folds",
        "hyperparameter_search": "none",
        "market_or_host_id_features": "none (features are exactly the 28 frozen legal descriptors)",
        "gate_c_evidence": gate_c,
        "scalar_o1_correction_ratio": scalar_rows,
    }
    R.json_dump(R.EVID / "R2_PROBE_SUMMARY.json", out)
    return out


if __name__ == "__main__":
    print(run()["sample"])
