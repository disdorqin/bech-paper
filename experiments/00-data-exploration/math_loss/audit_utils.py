"""Math evidence audit utilities — FS skew-t, bootstrap, residual diagnostics."""
from __future__ import annotations

import json
import numpy as np
from scipy import stats, optimize, special
from pathlib import Path

EPS = 1e-10

# ====================================================== bootstrap utilities ===
def day_block_bootstrap(values: np.ndarray, day_ids: np.ndarray,
                        n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Day-block bootstrap CI for mean of `values`.
    Returns (lower, mean, upper) at (alpha/2, 0.5, 1-alpha/2).
    """
    rng = np.random.default_rng(seed)
    unique_days = np.unique(day_ids)
    n_days = len(unique_days)
    if n_days < 10:
        return None, float(np.mean(values)), None

    m = np.zeros(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(unique_days, size=n_days, replace=True)
        mask = np.isin(day_ids, sampled)
        m[b] = np.mean(values[mask]) if mask.sum() > 0 else np.nan
    m = m[~np.isnan(m)]
    lo = np.quantile(m, alpha / 2)
    hi = np.quantile(m, 1 - alpha / 2)
    return float(lo), float(np.mean(values)), float(hi)


def day_block_bootstrap_diff(v1, day1, v2, day2, n_boot=2000, seed=0, alpha=0.05):
    """Block bootstrap CI for mean(v1 - v2), assuming same-day alignment not required."""
    rng = np.random.default_rng(seed)
    d1u = np.unique(day1)
    d2u = np.unique(day2)
    if len(d1u) < 10 or len(d2u) < 10:
        return None, float(np.mean(v1) - np.mean(v2)), None
    m = np.zeros(n_boot)
    for b in range(n_boot):
        s1 = rng.choice(d1u, size=len(d1u), replace=True)
        s2 = rng.choice(d2u, size=len(d2u), replace=True)
        m1 = np.mean(v1[np.isin(day1, s1)])
        m2 = np.mean(v2[np.isin(day2, s2)])
        m[b] = m1 - m2
    lo = np.quantile(m, alpha / 2)
    hi = np.quantile(m, 1 - alpha / 2)
    return float(lo), float(np.mean(v1) - np.mean(v2)), float(hi)


# ======================================================= FS skew-t primitives ===
def student_t_logpdf(x: np.ndarray, nu: float) -> np.ndarray:
    """Standard Student-t(nu) log-pdf."""
    c = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
         - 0.5 * np.log(max(nu, EPS) * np.pi))
    return c - ((nu + 1) / 2) * np.log1p(x * x / max(nu, EPS))


def student_t_pdf(x: np.ndarray, nu: float) -> np.ndarray:
    return np.exp(student_t_logpdf(x, nu))


def student_t_cdf(x: np.ndarray, nu: float) -> np.ndarray:
    """CDF of standard Student-t using regularized incomplete beta."""
    ratio = nu / (nu + x * x)
    I_x = special.betainc(nu / 2, 0.5, ratio)
    return 0.5 + 0.5 * np.sign(x) * I_x


def fit_student_t_mle(data: np.ndarray) -> dict:
    """Fit location-scale Student-t to data via MLE.
    Returns {mu, sigma, nu, nll, converged}.
    """
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]

    # grid over nu
    nu_candidates = np.array([2.1, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0, 100.0])
    best_nll, best_params = np.inf, None

    for nu in nu_candidates:
        mu = np.median(data)

        def nll_fn(s_log):
            s = np.exp(s_log) + EPS
            z = (data - mu) / s
            lp = student_t_logpdf(z, nu) - np.log(s)
            return float(np.mean(-lp))

        res = optimize.minimize_scalar(nll_fn, bounds=(-5, 5), method="bounded")
        sigma = np.exp(res.x) + EPS

        z = (data - mu) / sigma
        nll = float(np.mean(-(student_t_logpdf(z, nu) - np.log(sigma))))
        if nll < best_nll:
            best_nll = nll
            best_params = (mu, sigma, nu)

    mu, sigma, nu = best_params
    return {"mu": mu, "sigma": sigma, "nu": nu, "nll": best_nll, "converged": True}


def fit_normal_mle(data):
    return {"mu": float(np.mean(data)), "sigma": float(np.std(data)),
            "nll": float(-np.mean(stats.norm.logpdf(data, np.mean(data), np.std(data))))}


def fit_laplace_mle(data):
    mu = float(np.median(data))
    b = float(np.mean(np.abs(data - mu)))
    nll = float(-np.mean(stats.laplace.logpdf(data, mu, b)))
    return {"mu": mu, "b": b, "nll": nll}


# ========================================================= residual diagnostics ===
def residual_basic_stats(r: np.ndarray, day_ids: np.ndarray) -> dict:
    """Compute basic + robust residual summaries."""
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10:
        return {"n": n, "status": "INSUFFICIENT_SAMPLE"}

    qs = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50,
          0.75, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
    qvals = {f"q{p*1000:.0f}": float(np.quantile(r, p)) for p in qs if n >= 1 / (1 - p)}

    pi_neg = float(np.mean(r < 0))
    pi_pos = float(np.mean(r > 0))
    M_neg = float(np.mean(r[r < 0])) * pi_neg if (r < 0).sum() > 0 else 0.0
    M_pos = float(np.mean(r[r > 0])) * pi_pos if (r > 0).sum() > 0 else 0.0

    out = {
        "n": n, "n_days": int(len(np.unique(day_ids))),
        "mean": float(np.mean(r)), "median": float(np.median(r)),
        "std": float(np.std(r, ddof=0)),
        "mad": float(np.median(np.abs(r - np.median(r))) * 1.4826),
        "iqr": float(np.quantile(r, 0.75) - np.quantile(r, 0.25)),
        "min": float(np.min(r)), "max": float(np.max(r)),
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "pi_neg": pi_neg, "pi_pos": pi_pos,
        "M_neg": M_neg, "M_pos": M_pos,
        "m_neg": float(np.mean(-r[r < 0])) if (r < 0).sum() > 0 else 0.0,
        "m_pos": float(np.mean(r[r > 0])) if (r > 0).sum() > 0 else 0.0,
        "top1_pct_contrib": _topk_contrib(r, k_pct=0.001),
        "top10_pct_contrib": _topk_contrib(r, k_pct=0.01),
    }
    out.update(qvals)
    return out


def _topk_contrib(r, k_pct):
    n = len(r)
    k = max(1, int(n * k_pct))
    top = np.sort(np.abs(r))[-k:]
    return float(top.sum() / (r * r).sum()) if (r * r).sum() > 0 else 0.0


def tail_stability_check(r_s2, day_s2, r_s3, day_s3) -> dict:
    """Check if tail properties are stable from S2 to S3."""
    out = {}
    for name, r, d in [("S2", r_s2, day_s2), ("S3", r_s3, day_s3)]:
        r = r[np.isfinite(r)]
        out[f"{name}_skew"] = float(stats.skew(r))
        out[f"{name}_kurtosis"] = float(stats.kurtosis(r))
        out[f"{name}_pi_neg"] = float(np.mean(r < 0))
        lo, m, hi = day_block_bootstrap(r, d)
        out[f"{name}_mean_ci_lo"] = lo
        out[f"{name}_mean_ci_hi"] = hi
    return out


def occ_magnitude_by_bins(r_s2, day_s2, yhat_s2, n_bins=10):
    """Estimate pi_±(Z) and m_±(Z) over yhat bins as predictive state proxy."""
    r = r_s2[np.isfinite(r_s2)]
    yh = yhat_s2[np.isfinite(r_s2)]
    day = day_s2[np.isfinite(r_s2)]
    if len(r) < 100:
        return {"status": "INSUFFICIENT_SAMPLE"}

    edges = np.quantile(yh, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        mask = (yh >= edges[i]) & (yh < edges[i + 1])
        if mask.sum() < 5:
            continue
        rr = r[mask]
        row = {
            "bin": i, "n": int(mask.sum()),
            "yhat_lo": float(edges[i]), "yhat_hi": float(edges[i + 1]),
            "pi_neg": float(np.mean(rr < 0)),
            "pi_pos": float(np.mean(rr > 0)),
            "m_neg": float(np.mean(-rr[rr < 0])) if (rr < 0).sum() > 0 else 0.0,
            "m_pos": float(np.mean(rr[rr > 0])) if (rr > 0).sum() > 0 else 0.0,
            "M_neg": -float(np.mean(rr[rr < 0]) * np.mean(rr < 0)) if (rr < 0).sum() > 0 else 0.0,
            "M_pos": float(np.mean(rr[rr > 0]) * np.mean(rr > 0)) if (rr > 0).sum() > 0 else 0.0,
        }
        lo, m, hi = day_block_bootstrap(rr, day[mask])
        row["mean_ci_lo"] = lo
        row["mean_ci_hi"] = hi
        rows.append(row)
    return rows


def day_dependence(r_s2, day_s2, hour_s2) -> dict:
    """Compute 24h within-day residual dependence diagnostics."""
    r = r_s2[np.isfinite(r_s2)]
    d = day_s2[np.isfinite(r_s2)]
    h = hour_s2[np.isfinite(r_s2)]
    if len(r) < 24 * 5:
        return {"status": "INSUFFICIENT_SAMPLE"}

    out = {}
    # Hourly residual correlation matrix (24x24)
    unique_days = np.unique(d)
    mat = np.zeros((24, 24))
    counts = np.zeros((24, 24))
    for dd in unique_days:
        dm = d == dd
        if dm.sum() != 24:
            continue
        rh = r[dm]
        hh = h[dm].astype(int)
        for i in range(24):
            for j in range(24):
                mat[hh[i], hh[j]] += rh[i] * rh[j]
                counts[hh[i], hh[j]] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = mat / counts

    # Eigenvalue spectrum of 24x24 residual correlation
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    evals = s ** 2
    evals = evals / evals.sum() if evals.sum() > 0 else np.zeros_like(evals)
    out["effective_rank"] = float((evals > 0.01).sum())
    out["first_eval_ratio"] = float(evals[0]) if len(evals) > 0 else 0.0
    out["top3_eval_ratio"] = float(evals[:3].sum()) if len(evals) >= 3 else 0.0

    # Max off-diagonal correlation
    corr = mat / (np.sqrt(np.diag(mat)[:, None] * np.diag(mat)[None, :]) + EPS)
    offdiag = corr[~np.eye(24, dtype=bool)]
    out["max_offdiag_corr"] = float(np.max(np.abs(offdiag))) if len(offdiag) > 0 else 0.0

    # Day-level residual ACF (up to lag 6 hours)
    acf_vals = []
    for dd in unique_days:
        dm = d == dd
        if dm.sum() != 24:
            continue
        rh = r[dm]
        for lag in range(1, 7):
            if len(rh) > lag:
                c = np.corrcoef(rh[:len(rh) - lag], rh[lag:])[0, 1]
                if not np.isnan(c):
                    acf_vals.append((lag, float(c)))
    if acf_vals:
        for lag in range(1, 7):
            vals = [v for l, v in acf_vals if l == lag]
            out[f"acf_lag{lag}"] = float(np.median(vals)) if vals else np.nan
    return out


def distribution_compare(r_s2, r_s3, day_s2, day_s3) -> dict:
    """Fit Normal, Laplace, Student-t on S2; evaluate NLL on S3."""
    r2 = r_s2[np.isfinite(r_s2)]
    r3 = r_s3[np.isfinite(r_s3)]
    d2 = day_s2[np.isfinite(r_s2)]
    d3 = day_s3[np.isfinite(r_s3)]

    out = {}

    # Normal (MLE)
    nrm = fit_normal_mle(r2)
    nrm_s3_nll = float(-np.mean(stats.norm.logpdf(r3, nrm["mu"], nrm["sigma"])))
    out["normal_mu"] = nrm["mu"]
    out["normal_sigma"] = nrm["sigma"]
    out["normal_s2_nll"] = nrm["nll"]
    out["normal_s3_nll"] = nrm_s3_nll

    # Laplace (MLE)
    lap = fit_laplace_mle(r2)
    lap_s3_nll = float(-np.mean(stats.laplace.logpdf(r3, lap["mu"], lap["b"])))
    out["laplace_mu"] = lap["mu"]
    out["laplace_b"] = lap["b"]
    out["laplace_s2_nll"] = lap["nll"]
    out["laplace_s3_nll"] = lap_s3_nll

    # Student-t (MLE)
    try:
        st = fit_student_t_mle(r2)
        s3_z = (r3 - st["mu"]) / st["sigma"]
        st_s3_nll = float(-np.mean(student_t_logpdf(s3_z, st["nu"]) - np.log(st["sigma"])))
        out["t_mu"] = st["mu"]
        out["t_sigma"] = st["sigma"]
        out["t_nu"] = st["nu"]
        out["t_s2_nll"] = st["nll"]
        out["t_s3_nll"] = st_s3_nll
    except Exception as e:
        out["t_error"] = str(e)

    # Bootstrap CI for NLL differences
    _nll_boot_diff(out, "t_vs_normal", r_s3, day_s3, nrm, "normal", st, "t")
    _nll_boot_diff(out, "t_vs_laplace", r_s3, day_s3, lap, "laplace", st, "t")

    return out


def _nll_boot_diff(out_dict, key, r, day, fit1, name1, fit2, name2):
    """Bootstrap CI for NLL difference fit2 - fit1 on held-out data."""
    r = r[np.isfinite(r)]
    day = day[np.isfinite(r)]
    if len(r) < 100:
        return
    if name1 == "normal":
        nll1_arr = -stats.norm.logpdf(r, fit1["mu"], fit1["sigma"])
    elif name1 == "laplace":
        nll1_arr = -stats.laplace.logpdf(r, fit1["mu"], fit1["b"])
    else:
        return

    if name2 == "t":
        s3_z = (r - fit2["mu"]) / fit2["sigma"]
        nll2_arr = -(student_t_logpdf(s3_z, fit2["nu"]) - np.log(fit2["sigma"]))
    else:
        return

    diff = nll2_arr - nll1_arr
    lo, m, hi = day_block_bootstrap(diff, day)
    out_dict[f"{key}_diff"] = m
    out_dict[f"{key}_ci_lo"] = lo
    out_dict[f"{key}_ci_hi"] = hi


def candidate_action_compare(r, candidates, day_ids) -> list:
    """Compare candidate actions: partial moments, side-conditional means, etc.
    r: true residuals = y - host_pred
    candidates: dict of name -> array of correction deltas (add to host_pred)
    Returns list of dicts with gain statistics.
    """
    r = r[np.isfinite(r)]
    day = day_ids[np.isfinite(r)]
    rows = []
    for name, delta in candidates.items():
        delta = delta[np.isfinite(r)]
        # MAE gain = |r| - |r - delta|
        gain = np.abs(r) - np.abs(r - delta)
        lo, m, hi = day_block_bootstrap(gain, day)
        harm_rate = float(np.mean(gain < 0))
        tail_gain = float(np.mean(gain[np.abs(r) > np.quantile(np.abs(r), 0.95)]))
        normal_gain = float(np.mean(gain[np.abs(r) <= np.quantile(np.abs(r), 0.95)]))

        rows.append({
            "candidate": name,
            "mean_gain": m,
            "gain_ci_lo": lo, "gain_ci_hi": hi,
            "harm_rate": harm_rate,
            "tail_gain": tail_gain,
            "normal_gain": normal_gain,
        })
    return rows
