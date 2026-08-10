"""Audit whether extreme-price forecast failures are episode-structured.

The audit reuses the paper's S1-S4 protocol and regenerates S4 predictions. It
does not tune a new method. Extreme episodes are maximal hourly runs within one
delivery day, so the measurements match the proposed 24-point correction unit.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import backbones as B  # noqa: E402
import common as C  # noqa: E402
from bech import BECH, build_corrector_features  # noqa: E402

# The repository was reorganized after the original Phase-3 scripts were run;
# keep this audit scoped to the current documented data location.
C.DATA = os.path.join(ROOT, "data", "raw")


DEFAULT_DATASETS = [
    "LAGO_DE", "LAGO_BE", "LAGO_FR", "LAGO_PJM", "LAGO_NP",
    "NEM_SA1", "NEM_VIC1", "NEM_NSW1", "GEFCOM14P",
    "UNIELEC_DE", "UNIELEC_AT", "UNIELEC_IE", "UNIELEC_BE",
    "UNIELEC_FI", "UNIELEC_DK", "UNIELEC_NL", "UNIELEC_CZ",
]

UNIELEC_COUNTRIES = {
    "UNIELEC_DE": "Germany",
    "UNIELEC_AT": "Austria",
    "UNIELEC_IE": "Ireland",
    "UNIELEC_BE": "Belgium",
    "UNIELEC_FI": "Finland",
    "UNIELEC_DK": "Denmark",
    "UNIELEC_NL": "Netherlands",
    "UNIELEC_CZ": "Czech Republic",
}


def load_for_audit(dataset: str) -> dict:
    if dataset not in UNIELEC_COUNTRIES:
        return C.load_dataset(dataset)
    country = UNIELEC_COUNTRIES[dataset]
    path = os.path.join(C.DATA, "unielecprice", "by_country", f"{country}.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (df.sort_values("timestamp")
            .drop_duplicates("timestamp")
            .dropna(subset=["price"])
            .reset_index(drop=True))
    step = df["timestamp"].diff().dropna().median()
    if step < pd.Timedelta(hours=1):
        df = (df.set_index("timestamp")[["price"]]
                .resample("h").mean()
                .dropna()
                .reset_index())
    empty = pd.DataFrame(index=df.index)
    return {
        "key": dataset,
        "ts": df["timestamp"],
        "price": pd.to_numeric(df["price"], errors="raise").to_numpy(float),
        "exog_fc": empty,
        "exog_act": empty.copy(),
        "meta": {
            "tier": "L1" if dataset == "UNIELEC_DE" else "L2",
            "currency": "EUR",
            "note": f"UniElecPrice day-ahead market: {country}",
        },
    }


def extract_episodes(mask: np.ndarray, ts: pd.Series) -> list[dict]:
    """Return maximal true runs, split at delivery-day and timestamp gaps."""
    mask = np.asarray(mask, dtype=bool)
    t = pd.Series(pd.to_datetime(ts)).reset_index(drop=True)
    out: list[dict] = []
    start: int | None = None

    def close(end: int) -> None:
        nonlocal start
        if start is None:
            return
        out.append({
            "start": start,
            "end": end,
            "duration": end - start + 1,
            "day": str(t.iloc[start].date()),
            "start_ts": str(t.iloc[start]),
            "end_ts": str(t.iloc[end]),
        })
        start = None

    for i, active in enumerate(mask):
        if not active:
            close(i - 1)
            continue
        if start is None:
            start = i
            continue
        same_day = t.iloc[i].date() == t.iloc[i - 1].date()
        hourly = t.iloc[i] - t.iloc[i - 1] == pd.Timedelta(hours=1)
        if not (same_day and hourly):
            close(i - 1)
            start = i
    close(len(mask) - 1)
    return out


def interval_iou(a: dict, b: dict) -> float:
    lo = max(a["start"], b["start"])
    hi = min(a["end"], b["end"])
    inter = max(0, hi - lo + 1)
    union = a["duration"] + b["duration"] - inter
    return inter / union if union else 0.0


def overlaps(a: dict, b: dict) -> bool:
    return a["day"] == b["day"] and max(a["start"], b["start"]) <= min(a["end"], b["end"])


def label_summary(dataset: str, regime: str, mask: np.ndarray,
                  ts: pd.Series) -> tuple[dict, list[dict]]:
    eps = extract_episodes(mask, ts)
    dur = np.asarray([e["duration"] for e in eps], dtype=float)
    row = {
        "dataset": dataset,
        "regime": regime,
        "n_test_hours": len(mask),
        "n_event_hours": int(np.sum(mask)),
        "event_hour_rate": float(np.mean(mask)) if len(mask) else np.nan,
        "n_true_events": len(eps),
        "mean_duration_h": float(dur.mean()) if len(dur) else np.nan,
        "median_duration_h": float(np.median(dur)) if len(dur) else np.nan,
        "p90_duration_h": float(np.quantile(dur, 0.9)) if len(dur) else np.nan,
        "max_duration_h": int(dur.max()) if len(dur) else 0,
        "singleton_rate": float(np.mean(dur == 1)) if len(dur) else np.nan,
        "multi_hour_rate": float(np.mean(dur >= 2)) if len(dur) else np.nan,
        "four_plus_rate": float(np.mean(dur >= 4)) if len(dur) else np.nan,
    }
    return row, eps


def score_prediction(dataset: str, backbone: str, regime: str, predictor: str,
                     y: np.ndarray, pred: np.ndarray, true_mask: np.ndarray,
                     pred_mask: np.ndarray, ts: pd.Series,
                     true_eps: list[dict]) -> tuple[dict, list[dict]]:
    pred_eps = extract_episodes(pred_mask, ts)
    details: list[dict] = []
    types = Counter()
    ious, boundaries, duration_errors = [], [], []

    for event_id, te in enumerate(true_eps):
        hit = [pe for pe in pred_eps if overlaps(te, pe)]
        if not hit:
            kind = "complete_miss"
            best_iou = 0.0
            boundary_error = np.nan
            duration_error = np.nan
        else:
            best = max(hit, key=lambda pe: interval_iou(te, pe))
            best_iou = interval_iou(te, best)
            boundary_error = abs(te["start"] - best["start"]) + abs(te["end"] - best["end"])
            duration_error = abs(te["duration"] - best["duration"])
            if len(hit) > 1:
                kind = "fragmented"
            elif boundary_error == 0:
                kind = "exact_boundary"
            else:
                kind = "boundary_mismatch"
            ious.append(best_iou)
            boundaries.append(boundary_error)
            duration_errors.append(duration_error)
        types[kind] += 1
        idx = np.arange(te["start"], te["end"] + 1)
        details.append({
            "dataset": dataset,
            "backbone": backbone,
            "regime": regime,
            "predictor": predictor,
            "event_id": event_id,
            "day": te["day"],
            "start_ts": te["start_ts"],
            "end_ts": te["end_ts"],
            "duration_h": te["duration"],
            "failure_type": kind,
            "n_overlapping_pred_events": len(hit),
            "best_iou": best_iou,
            "boundary_l1_h": boundary_error,
            "duration_abs_error_h": duration_error,
            "event_mae": float(np.mean(np.abs(pred[idx] - y[idx]))),
            "event_bias": float(np.mean(pred[idx] - y[idx])),
        })

    false_events = sum(not any(overlaps(pe, te) for te in true_eps) for pe in pred_eps)
    tp_hours = int(np.sum(true_mask & pred_mask))
    row = {
        "dataset": dataset,
        "backbone": backbone,
        "regime": regime,
        "predictor": predictor,
        "n_true_events": len(true_eps),
        "n_pred_events": len(pred_eps),
        "event_recall": 1.0 - types["complete_miss"] / len(true_eps) if true_eps else np.nan,
        "complete_miss_rate": types["complete_miss"] / len(true_eps) if true_eps else np.nan,
        "boundary_mismatch_rate": types["boundary_mismatch"] / len(true_eps) if true_eps else np.nan,
        "fragmented_rate": types["fragmented"] / len(true_eps) if true_eps else np.nan,
        "exact_boundary_rate": types["exact_boundary"] / len(true_eps) if true_eps else np.nan,
        "mean_best_iou_matched": float(np.mean(ious)) if ious else np.nan,
        "mean_boundary_l1_h_matched": float(np.mean(boundaries)) if boundaries else np.nan,
        "mean_duration_abs_error_h_matched": float(np.mean(duration_errors)) if duration_errors else np.nan,
        "false_event_rate": false_events / len(pred_eps) if pred_eps else np.nan,
        "point_recall": tp_hours / int(np.sum(true_mask)) if np.sum(true_mask) else np.nan,
        "point_precision": tp_hours / int(np.sum(pred_mask)) if np.sum(pred_mask) else np.nan,
        "mae_on_true_event_hours": float(np.mean(np.abs(pred[true_mask] - y[true_mask])))
        if np.sum(true_mask) else np.nan,
    }
    return row, details


def run_one(dataset: str, backbone: str, alpha: float, rho: float,
            seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    ds = load_for_audit(dataset)
    X, y, names, valid = C.build_tabular(ds)
    C.assert_no_leakage(ds, X, y, valid, names)
    sp = C.four_segment_split(len(y))
    S1, S2, S3, S4 = sp["S1"], sp["S2"], sp["S3"], sp["S4"]
    spike_thr = float(np.quantile(y[S1], 0.99))

    seq = C.build_sequences(ds, valid) if B.needs_seq(backbone) else None
    model = B.make_backbone(backbone, seed)
    if seq is None:
        model.fit(X[S1], y[S1])
        yhat = model.predict(X)
    else:
        model.fit(X[S1], y[S1], seq[S1])
        yhat = model.predict(X, seq)

    ts_all = ds["ts"].iloc[valid].reset_index(drop=True)
    hour = ts_all.dt.hour.to_numpy()
    dayid = ts_all.dt.floor("D").astype("int64").to_numpy()
    Z, _ = build_corrector_features(X, names, yhat, y, hour, dayid,
                                    oos_start=int(S2[0]))
    head = BECH(neg_thr=0.0, spike_thr=spike_thr, alpha=alpha,
                harm_budget_ratio=rho, seed=seed)
    head.fit(Z[S2], yhat[S2], y[S2])
    head.calibrate(Z[S3], yhat[S3], y[S3])
    corrected, _ = head.apply(Z[S4], yhat[S4])

    y4 = y[S4]
    base4 = yhat[S4]
    ts4 = ts_all.iloc[S4].reset_index(drop=True)
    day = ts4.dt.floor("D")
    counts = day.value_counts()
    complete_days = counts[counts == 24].index
    keep = day.isin(complete_days).to_numpy()
    y4 = y4[keep]
    base4 = base4[keep]
    corrected = corrected[keep]
    ts4 = ts4.loc[keep].reset_index(drop=True)
    label_rows, model_rows, detail_rows = [], [], []
    regimes = {
        "negative": (y4 < 0.0, base4 < 0.0, corrected < 0.0),
        "spike": (y4 > spike_thr, base4 > spike_thr, corrected > spike_thr),
    }
    for regime, (truth, base_mask, bech_mask) in regimes.items():
        label_row, true_eps = label_summary(dataset, regime, truth, ts4)
        label_row["spike_thr_train"] = spike_thr
        label_rows.append(label_row)
        for predictor, pred, pred_mask in (
            ("base", base4, base_mask),
            ("bech", corrected, bech_mask),
        ):
            row, details = score_prediction(
                dataset, backbone, regime, predictor, y4, pred, truth,
                pred_mask, ts4, true_eps,
            )
            row["spike_thr_train"] = spike_thr
            model_rows.append(row)
            detail_rows.extend(details)
    return label_rows, model_rows, detail_rows


def make_comparison(model_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "backbone", "regime"]
    wide = model_df.pivot(index=keys, columns="predictor")
    rows = []
    for idx in wide.index:
        row = dict(zip(keys, idx))
        for metric in (
            "event_recall", "complete_miss_rate", "boundary_mismatch_rate",
            "fragmented_rate", "exact_boundary_rate", "point_recall",
            "point_precision", "mae_on_true_event_hours",
        ):
            row[f"{metric}_base"] = wide.loc[idx, (metric, "base")]
            row[f"{metric}_bech"] = wide.loc[idx, (metric, "bech")]
            row[f"{metric}_delta"] = (
                wide.loc[idx, (metric, "bech")] - wide.loc[idx, (metric, "base")]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_rate(df: pd.DataFrame, col: str) -> float:
    ok = df[col].notna() & (df["n_true_events"] > 0)
    if not ok.any():
        return np.nan
    return float(np.average(df.loc[ok, col], weights=df.loc[ok, "n_true_events"]))


def write_report(label_df: pd.DataFrame, model_df: pd.DataFrame,
                 comparison_df: pd.DataFrame, path: str) -> None:
    label_unique = label_df.drop_duplicates(["dataset", "regime"])
    lines = [
        "# Extreme-Price Episode Audit", "",
        "> P0 audit only: no new model was tuned. Episodes are maximal hourly runs within a delivery day.", "",
        "## Label Topology", "",
        "| Dataset | Regime | Event hours | Episodes | Median h | Multi-hour | >=4h | Max h |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in label_unique.sort_values(["regime", "dataset"]).iterrows():
        lines.append(
            f"| {r.dataset} | {r.regime} | {int(r.n_event_hours)} | {int(r.n_true_events)} | "
            f"{r.median_duration_h:.1f} | {r.multi_hour_rate:.1%} | "
            f"{r.four_plus_rate:.1%} | {int(r.max_duration_h)} |"
        )

    lines += ["", "## Weighted Failure Structure", "",
              "Rates are weighted by the number of true S4 episodes.", "",
              "| Regime | Predictor | Event recall | Complete miss | Boundary mismatch | Fragmented | Exact boundary |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for regime in ("negative", "spike"):
        for predictor in ("base", "bech"):
            x = model_df[(model_df.regime == regime) & (model_df.predictor == predictor)]
            lines.append(
                f"| {regime} | {predictor} | {weighted_rate(x, 'event_recall'):.1%} | "
                f"{weighted_rate(x, 'complete_miss_rate'):.1%} | "
                f"{weighted_rate(x, 'boundary_mismatch_rate'):.1%} | "
                f"{weighted_rate(x, 'fragmented_rate'):.1%} | "
                f"{weighted_rate(x, 'exact_boundary_rate'):.1%} |"
            )

    lines += ["", "## Per-Combination Deltas", "",
              "Positive event-recall delta is better; positive fragmentation delta is worse.", "",
              "| Dataset | Backbone | Regime | Event recall delta | Fragmentation delta | Event-hour MAE delta |",
              "|---|---|---|---:|---:|---:|"]
    for _, r in comparison_df.sort_values(["regime", "dataset", "backbone"]).iterrows():
        lines.append(
            f"| {r.dataset} | {r.backbone} | {r.regime} | "
            f"{r.event_recall_delta:+.1%} | {r.fragmented_rate_delta:+.1%} | "
            f"{r.mae_on_true_event_hours_delta:+.3f} |"
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--backbones", nargs="*", default=["Linear", "GBDT"])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--rho", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.datasets = ["LAGO_DE", "NEM_SA1"]
        args.backbones = ["Linear"]

    result_dir = os.path.join(HERE, "results")
    os.makedirs(result_dir, exist_ok=True)
    all_labels, all_models, all_details = [], [], []
    for dataset in args.datasets:
        for backbone in args.backbones:
            print(f"[audit] {dataset}/{backbone}", flush=True)
            labels, models, details = run_one(
                dataset, backbone, args.alpha, args.rho, args.seed,
            )
            all_labels.extend(labels)
            all_models.extend(models)
            all_details.extend(details)

    label_df = pd.DataFrame(all_labels)
    model_df = pd.DataFrame(all_models)
    detail_df = pd.DataFrame(all_details)
    comparison_df = make_comparison(model_df)
    label_df.to_csv(os.path.join(result_dir, "label_episode_summary.csv"), index=False)
    model_df.to_csv(os.path.join(result_dir, "model_episode_summary.csv"), index=False)
    detail_df.to_csv(os.path.join(result_dir, "event_failure_detail.csv"), index=False)
    comparison_df.to_csv(os.path.join(result_dir, "bech_episode_deltas.csv"), index=False)
    write_report(label_df, model_df, comparison_df,
                 os.path.join(result_dir, "episode_audit.md"))
    print(f"[done] {result_dir}", flush=True)


if __name__ == "__main__":
    main()
