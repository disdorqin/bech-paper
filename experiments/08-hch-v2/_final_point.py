"""P0-A — authoritative final point replay (protocol §4, Paper Benchmark Gate v0.1).

    x_final = s_d * sinh(z0 + pi_eff),     pi_eff = pi  if DVG releases the day
                                                 = 0    otherwise

This is the ONE final-output function used for every headline point metric
(MAE / sMAPE / RMSE / rMAE / neg-price / high-tail). It consumes
evaluate_days-style rows (dict with z0, s_day, pi, price, vm) plus a per-day
released mask and evaluates on ALL hours of the day: pi is exactly zero outside
the volatile mask (mm=mp=0 there), so final == host on those hours — i.e. the
standard day-ahead 24h forecast.

R1B Stage-2D previously evaluated `day["host_day"]` in its headline point
metrics. That path is retired here.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def final_point_metrics(rows, released) -> dict:
    """Point metrics for the true final forecast on all hours.

    rows     : list of dicts from evaluate_days — each has z0, s_day, pi,
               price, vm (+ date/block).
    released : iterable of bool aligned to rows (DVG release decision).

    Returns metric dict + per-day arrays (day_mae_final / day_mae_host /
    dates) so day-level DM tests can be recomputed without rerunning.
    """
    rows = list(rows)
    if not rows:
        return {"n_hours": 0}
    rel = np.asarray(released, dtype=bool)
    if len(rel) != len(rows):
        raise ValueError("released mask must align with rows")

    ae_f, ae_h, smape, price_all = [], [], [], []
    day_f, day_h, dates = [], [], []
    for r, rl in zip(rows, rel):
        pi_eff = r["pi"] if rl else np.zeros_like(r["pi"])
        pred = r["s_day"] * np.sinh(r["z0"] + pi_eff)          # x_final (§4)
        host = r["s_day"] * np.sinh(r["z0"])                   # frozen host
        p = np.asarray(r["price"], dtype=np.float64)
        ef = np.abs(p - pred)
        eh = np.abs(p - host)
        denom = np.abs(p) + np.abs(pred)
        sm = 200.0 * ef / np.maximum(denom, EPS)
        ae_f.append(ef); ae_h.append(eh); smape.append(sm)
        price_all.append(p)
        day_f.append(float(ef.mean())); day_h.append(float(eh.mean()))
        dates.append(str(r["date"]))
    ef = np.concatenate(ae_f); eh = np.concatenate(ae_h)
    sm = np.concatenate(smape)
    p = np.concatenate(price_all)
    mae = float(ef.mean()); host_mae = float(eh.mean())
    scale = float(np.mean(np.abs(p)))
    neg = p < 0
    tail = np.abs(p) >= np.quantile(np.abs(p), 0.95)
    out = {
        "n_hours": int(len(p)),
        # PRIMARY (protocol §10): true final forecast
        "mae": round(mae, 6),
        "smape_nofloor": round(float(sm.mean()), 6),
        # SECONDARY
        "rmse": round(float(np.sqrt(np.mean(ef ** 2))), 6),
        "rmae": round(mae / max(scale, EPS), 6) if scale > 0 else None,
        "neg_price_rate": round(float(neg.mean()), 4),
        "neg_price_mae": round(float(ef[neg].mean()), 6) if neg.any() else None,
        "high_tail_rate": round(float(tail.mean()), 4),
        "high_tail_mae": round(float(ef[tail].mean()), 6) if tail.any() else None,
        # HOST equivalent (old R1B 'host_day' headline becomes host_mae)
        "host_mae": round(host_mae, 6),
        "host_rmae": round(host_mae / max(scale, EPS), 6) if scale > 0 else None,
        "degradation": round(mae - host_mae, 6),
        "degradation_frac": round((mae - host_mae) / host_mae, 6) if host_mae > 0 else None,
        # per-day arrays for DM / reproducibility
        "day_mae_final": np.asarray(day_f),
        "day_mae_host": np.asarray(day_h),
        "days": dates,
    }
    return out


def from_dvg(dvg_out: dict) -> dict:
    """Convenience: read _rows/_released from a dvg_and_s4 output dict."""
    if dvg_out is None:
        return {"n_hours": 0}
    return final_point_metrics(dvg_out.get("_rows", []),
                               dvg_out.get("_released"))


def raw_metrics(pred_days, host_days, price_days) -> dict:
    """Raw-space point metrics for baselines B0-B4 (paper gate WP-4+).

    Identical metric definitions as final_point_metrics but takes RAW price-
    space day vectors (predictions are NOT replayed through z0/pi):
      pred_days/host_days/price_days : iterables of 24h arrays, aligned by day.
    Days may be shorter than 24h (partial-validity hours); hours are simply
    concatenated (per-day mean MAE recomputable via day lengths).
    """
    preds = [np.asarray(p, dtype=np.float64) for p in pred_days]
    hosts = [np.asarray(h, dtype=np.float64) for h in host_days]
    prics = [np.asarray(q, dtype=np.float64) for q in price_days]
    if not preds or not any(len(p) for p in preds):
        return {"n_hours": 0}
    ef = np.concatenate([np.abs(q - p) for p, q in zip(preds, prics)])
    eh = np.concatenate([np.abs(q - h) for h, q in zip(hosts, prics)])
    p = np.concatenate(prics)
    sm = np.concatenate([200.0 * np.abs(q - p) / np.maximum(np.abs(q) + np.abs(p), EPS)
                         for p, q in zip(preds, prics)])
    mae = float(ef.mean()); host_mae = float(eh.mean())
    scale = float(np.mean(np.abs(p)))
    neg = p < 0
    tail = np.abs(p) >= np.quantile(np.abs(p), 0.95)
    out = {
        "n_hours": int(len(p)),
        "mae": round(mae, 6),
        "smape_nofloor": round(float(sm.mean()), 6),
        "rmse": round(float(np.sqrt(np.mean(ef ** 2))), 6),
        "rmae": round(mae / max(scale, EPS), 6) if scale > 0 else None,
        "neg_price_rate": round(float(neg.mean()), 4),
        "neg_price_mae": round(float(ef[neg].mean()), 6) if neg.any() else None,
        "high_tail_rate": round(float(tail.mean()), 4),
        "high_tail_mae": round(float(ef[tail].mean()), 6) if tail.any() else None,
        "host_mae": round(host_mae, 6),
        "host_rmae": round(host_mae / max(scale, EPS), 6) if scale > 0 else None,
        "degradation": round(mae - host_mae, 6),
        "degradation_frac": round((mae - host_mae) / host_mae, 6) if host_mae > 0 else None,
    }
    return out


# canonical alias used by stage2d/2e
final_metrics = from_dvg


if __name__ == "__main__":
    # smoke: identity replay returns exactly the host forecast
    rng = np.random.default_rng(0)
    z0 = rng.normal(0, 1, 24); s = 30.0
    rows = [{"date": "2024-01-01", "z0": z0, "s_day": s,
             "pi": np.zeros(24), "price": s * np.sinh(z0) + 1.0,
             "vm": np.ones(24, dtype=bool)}]
    out = final_point_metrics(rows, [False])
    print("identity replay: mae=%.4f host_mae=%.4f" % (out["mae"], out["host_mae"]))
