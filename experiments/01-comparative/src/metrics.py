"""统一评估指标 — 顶会标准版."""
from __future__ import annotations
import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_base: np.ndarray | None = None,
                           neg_thr: float = 0.0) -> dict:
    """Binary classification metrics: negative price detection."""
    true_neg = y_true < neg_thr
    pred_neg = y_pred < neg_thr
    TP = float((true_neg & pred_neg).sum())
    FP = float((~true_neg & pred_neg).sum())
    TN = float((~true_neg & ~pred_neg).sum())
    FN = float((true_neg & ~pred_neg).sum())
    total = len(y_true)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / total if total > 0 else 1.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    m = dict(
        precision=precision, recall=recall, f1=f1, accuracy=accuracy, fpr=fpr,
        TP=int(TP), FP=int(FP), TN=int(TN), FN=int(FN),
        neg_true=int(true_neg.sum()), neg_pred=int(pred_neg.sum()),
    )
    if y_base is not None:
        base_neg = y_base < neg_thr
        TPB = float((true_neg & base_neg).sum())
        FPB = float((~true_neg & base_neg).sum())
        FNB = true_neg.sum() - TPB
        m["base_recall"] = TPB / (TPB + FNB) if (TPB + FNB) > 0 else 1.0
        m["base_fpr"] = FPB / (FPB + float((~true_neg).sum())) if (~true_neg).sum() > 0 else 0.0
    return m


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       y_base: np.ndarray | None = None) -> dict:
    ae = np.abs(y_true - y_pred)
    se = ae ** 2
    m = dict(
        mae=float(ae.mean()),
        rmse=float(np.sqrt(se.mean())),
        wape=float(ae.sum() / (np.abs(y_true).sum() + 1e-9)),
    )
    r2_num = float((se).sum())
    r2_den = float(((y_true - y_true.mean()) ** 2).sum())
    m["r2"] = 1.0 - r2_num / r2_den if r2_den > 0 else 0.0

    if y_base is not None:
        base_ae = np.abs(y_true - y_base)
        base_se = base_ae ** 2
        m["base_mae"] = float(base_ae.mean())
        m["base_rmse"] = float(np.sqrt(base_se.mean()))
        m["delta_mae_pct"] = (m["mae"] - m["base_mae"]) / max(m["base_mae"], 1e-9) * 100
        m["delta_rmse_pct"] = (m["rmse"] - m["base_rmse"]) / max(m["base_rmse"], 1e-9) * 100

    return m


def extreme_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    neg_thr: float = 0.0) -> dict:
    neg = y_true < neg_thr
    m = dict(n_neg=int(neg.sum()))
    if neg.sum() > 0:
        m["mae_on_extremes"] = float(np.abs(y_true[neg] - y_pred[neg]).mean())
    else:
        m["mae_on_extremes"] = None
    return m


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                y_base: np.ndarray | None = None, neg_thr: float = 0.0) -> dict:
    m = {}
    m.update(regression_metrics(y_true, y_pred, y_base))
    m.update(classification_metrics(y_true, y_pred, y_base, neg_thr))
    m.update(extreme_metrics(y_true, y_pred, neg_thr))
    return m


METRIC_ORDER = ["mae", "rmse", "wape", "r2", "delta_mae_pct", "delta_rmse_pct",
                "recall", "precision", "f1", "accuracy", "fpr", "mae_on_extremes",
                "fire_rate", "base_mae", "base_rmse", "neg_true"]


def summary_row(method: str, m: dict, include_fire: bool = False) -> str:
    fire_str = f"{m.get('fire_rate',0):.1%}" if include_fire and m.get("fire_rate") is not None else ""
    dm = f"{m.get('delta_mae_pct',0):+.1f}%" if m.get('delta_mae_pct') is not None else ""
    dr = f"{m.get('delta_rmse_pct',0):+.1f}%" if m.get('delta_rmse_pct') is not None else ""
    rec = f"{m.get('recall',0):.1%}" if m.get('recall') is not None else "N/A"
    f1s = f"{m.get('f1',0):.3f}" if m.get('f1') is not None else "N/A"
    fprs = f"{m.get('fpr',0):.1%}" if m.get('fpr') is not None else "N/A"
    mae_ex = f"{m.get('mae_on_extremes',0):.1f}" if m.get('mae_on_extremes') is not None else "N/A"
    return (f"{method:14s} {m['mae']:7.2f} {m['rmse']:7.2f} {m.get('wape',0):6.1%} "
            f"{dm:>7s} {rec:>7s} {f1s:>7s} {fprs:>7s} {mae_ex:>7s} {fire_str:>6s}")
