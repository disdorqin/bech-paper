"""Registered metric definitions for the signed-mass method experiment.

Every tail definition here is a **lookup**, not a computation: ``q05``, ``q95``
and ``day_spread_p90`` are read from the frozen breadth-stage threshold file and
are never re-derived from the data being evaluated.  Re-fitting a quantile on an
evaluation partition would silently change what "tail" means per method, which
is exactly the failure mode the threshold freeze exists to prevent.

Metrics are pure functions of

* the realised target ``y``,
* the frozen Host prediction ``y_hat_host``,
* the method prediction ``y_hat``,
* the validity mask,

so they apply identically to the Host baseline (``y_hat = y_hat_host``) and to
every configuration of the method.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from . import config as C
from .contracts import ProvenanceError, sha256_file

__all__ = ["load_thresholds", "shape_decomposition", "wasserstein1",
           "fit_high_mass_thresholds", "compute_metrics", "METRIC_DEFINITIONS"]

#: Machine-readable statement of what each reported number means.  It is emitted
#: into the evidence so a reader never has to infer a definition from a column
#: name.
METRIC_DEFINITIONS = {
    "mass_thresholds_used": "the cell's own POST_TRAIN q90(A+) and q90(A-), with "
                            "the digest of the rows they were fitted on; fitted "
                            "once per market x Host before any DEV_EVAL metric "
                            "and never recomputed",
    "high_mass_positive_l1": "mean |Ahat+ - A+| over days whose realised A+ is "
                             "at least the cell's q90(A+), using the DIRECT "
                             "positive head output",
    "high_mass_negative_l1": "same for the negative branch",
    "high_mass_*_n_days": "size of that subset; the metric is "
                          "NOT_APPLICABLE_N0 when the subset is empty",
    "branch_arrays_used": "whether Shape/mass metrics came from the direct branch "
                          "heads (DIRECT_BRANCH_HEADS) or from decomposing the net "
                          "correction (DECOMPOSED_NET_CORRECTION); only the direct "
                          "path is admissible for mechanism claims, because S+ and "
                          "S- can overlap at a horizon and cancel inside c",
    "branch_reconstruction_max_abs_err": "max |c - (Ahat+ Shat+ - Ahat- Shat-)| over "
                                         "the evaluated block; a value beyond float32 "
                                         "rounding means the persisted branch arrays "
                                         "did not produce the persisted correction, "
                                         "and the metric block is refused rather than "
                                         "reported",
    "alpha_consistency_max_abs_err": "max |(host + alpha*correction) - prediction| when "
                                     "alpha is supplied; a value beyond round-off means "
                                     "the metrics and the persisted calibration do not "
                                     "describe the same prediction",
    "overall_mae": "mean |y_hat - y| over valid entries",
    "overall_mse": "mean (y_hat - y)^2 over valid entries",
    "overall_rmse": "sqrt(overall_mse)",
    "host_mae": "mean |y_hat_host - y| over the same entries",
    "mae_gain_abs": "host_mae - overall_mae (positive means the method helps)",
    "mae_gain_pct": "100 * mae_gain_abs / host_mae",
    "shape_w1_positive": "day-wise sum_{h=1..H-1} |F_S+(h) - F_Shat+(h)|, averaged "
                         "over days where the realised positive mass is > 0; the "
                         "realised Shape comes from the exact factorization of "
                         "y - y_hat_host and the predicted Shape from the direct "
                         "positive head",
    "shape_w1_negative": "same for the negative branch",
    "shape_cosine_positive": "cosine between the realised and predicted positive "
                             "simplex vectors, averaged over the same days",
    "shape_cosine_negative": "same for the negative branch",
    "mass_positive_l1": "mean |Ahat+ - A+| per day, over days with at least one valid "
                        "entry; Ahat+ is the direct positive head",
    "mass_negative_l1": "mean |Ahat- - A-| per day, over the same days",
    "tail_mae": "MAE over entries with y <= q05 or y >= q95",
    "upper_tail_mae": "MAE over entries with y >= q95",
    "lower_tail_mae": "MAE over entries with y <= q05",
    "normal_mae": "MAE over entries that are in neither tail",
    "normal_relative_harm_pct": "100 * (normal_mae_method - normal_mae_host) / "
                                "normal_mae_host; positive means the method harms "
                                "the ordinary region",
    "negative_price_mae": "MAE over entries with y < 0 (None where the partition "
                          "contains no negative price)",
    "high_spread_day_mae": "MAE over entries belonging to days whose realised "
                           "intraday spread is >= day_spread_p90",
    "thresholds_used": "the frozen (q05, q95, day_spread_p90) triple, with its "
                       "source file and digest; never recomputed",
}


# --------------------------------------------------------------------------
# Frozen thresholds
# --------------------------------------------------------------------------
_THRESHOLD_CACHE: Dict[str, Any] = {}


def load_thresholds(market: str) -> Dict[str, Any]:
    """Read one market's frozen threshold record.  Fails closed if absent."""
    if market in _THRESHOLD_CACHE:
        return _THRESHOLD_CACHE[market]

    import json

    path = C.THRESHOLD_FREEZE
    if not path.exists():
        raise ProvenanceError(f"frozen threshold file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = payload.get("thresholds", {})
    if market not in table:
        raise ProvenanceError(
            f"no frozen threshold record for {market} in {path}; the harness "
            "will not improvise a quantile for a market")

    rec = dict(table[market])
    for field in ("q05", "q95", "day_spread_p90"):
        if field not in rec or not np.isfinite(float(rec[field])):
            raise ProvenanceError(f"{market}: frozen threshold field {field!r} is absent")
        rec[field] = float(rec[field])
    rec["threshold_source"] = str(path.relative_to(C.REPO_ROOT)).replace("\\", "/")
    rec["threshold_source_sha256"] = sha256_file(path)
    rec["rule"] = dict(C.THRESHOLD_RULE)
    rec["recomputed_for_this_evaluation"] = False
    _THRESHOLD_CACHE[market] = rec
    return rec


# --------------------------------------------------------------------------
# Shape / mass decomposition of an arbitrary residual series
# --------------------------------------------------------------------------
def shape_decomposition(residual: np.ndarray) -> Dict[str, np.ndarray]:
    """Split ``(N, H)`` residuals into nonnegative masses and unit simplices.

    ``A+ = sum_h max(r_h, 0)``, ``S+ = max(r_h, 0) / A+``.  A zero mass is
    reported with ``defined = False`` and a uniform simplex, so the caller can
    exclude the day instead of reading a fabricated shape out of ``0/0``.
    """
    r = np.asarray(residual, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("residual must be (N, H)")
    pos = np.clip(r, 0.0, None)
    neg = np.clip(-r, 0.0, None)
    a_pos = pos.sum(axis=1)
    a_neg = neg.sum(axis=1)
    h = r.shape[1]

    s_pos = np.full_like(pos, 1.0 / h)
    s_neg = np.full_like(neg, 1.0 / h)
    ok_pos = a_pos > 0.0
    ok_neg = a_neg > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        s_pos[ok_pos] = pos[ok_pos] / a_pos[ok_pos, None]
        s_neg[ok_neg] = neg[ok_neg] / a_neg[ok_neg, None]

    return {"mass_positive": a_pos, "mass_negative": a_neg,
            "shape_positive": s_pos, "shape_negative": s_neg,
            "defined_positive": ok_pos, "defined_negative": ok_neg}


def wasserstein1(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``sum_{h=1..H-1} |F_u(h) - F_v(h)|`` for ``(N, H)`` simplex rows.

    The horizon-wise cumulative form is the exact 1-Wasserstein distance between
    two distributions on ``{0..H-1}``, not an approximation of it, which is why
    the Shape metric needs no binning choice.
    """
    cu = np.cumsum(u, axis=1)[:, :-1]
    cv = np.cumsum(v, axis=1)[:, :-1]
    return np.abs(cu - cv).sum(axis=1)


#: Retained under its old private spelling for the call sites that predate the
#: rename; new code should use :func:`wasserstein1`.
_wasserstein1 = wasserstein1


def _cosine(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    num = (u * v).sum(axis=1)
    den = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0.0, num / den, np.nan)
    return out


# --------------------------------------------------------------------------
# The metric set
# --------------------------------------------------------------------------
def _masked(values: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, int]:
    v = np.asarray(values, dtype=np.float64)[mask]
    return v, int(v.size)


def _mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Optional[float]:
    v, n = _masked(np.abs(pred - target), mask)
    return float(v.mean()) if n else None


# --------------------------------------------------------------------------
# High-mass mechanism thresholds  (PART A3)
# --------------------------------------------------------------------------
def fit_high_mass_thresholds(residual: np.ndarray,
                             valid_mask: Optional[np.ndarray] = None,
                             quantile: float = C.HIGH_MASS_QUANTILE,
                             fit_rows: Optional[np.ndarray] = None
                             ) -> Dict[str, Any]:
    """``q90(A+)`` and ``q90(A-)`` for one market x Host cell.

    The **value** is Host-specific because the residual is Host-specific; the
    **rule** -- the same quantile of the realised daily mass -- is global and
    identical everywhere.  This function is pure: the caller supplies the rows,
    and the only admissible rows are that cell's own ``POST_TRAIN``.

    A subset threshold is only meaningful when it is strictly positive: if a
    cell never leaves a positive residual mass in that direction, ``q90`` is 0
    and the "high mass" subset would be every day, which is not a subset at all.
    That case is reported as ``usable=False`` rather than silently used.
    """
    r = np.asarray(residual, dtype=np.float64)
    mask = (np.ones_like(r, dtype=bool) if valid_mask is None
            else np.asarray(valid_mask, dtype=bool))
    masses = shape_decomposition(r if valid_mask is None else r * mask)
    out: Dict[str, Any] = {
        "rule": f"q{int(round(quantile * 100))}(A^pm) over the cell's POST_TRAIN rows",
        "quantile": float(quantile),
        "fit_partition": C.ROLE_POST_TRAIN,
        "n_rows": int(r.shape[0]),
        "n_positive_mass_days": int((masses["mass_positive"] > 0.0).sum()),
        "n_negative_mass_days": int((masses["mass_negative"] > 0.0).sum()),
        "fit_rows_sha256": (None if fit_rows is None
                            else _rows_digest(np.asarray(fit_rows))),
        "recomputed_for_this_evaluation": False,
    }
    for sign in ("positive", "negative"):
        key = f"mass_{sign}"
        values = masses[key]
        value = float(np.quantile(values, quantile)) if values.size else 0.0
        out[f"q{int(round(quantile * 100))}_{sign}"] = value
        out[f"usable_{sign}"] = bool(value > 0.0 and np.isfinite(value))
    return out


def _rows_digest(rows: np.ndarray) -> str:
    import hashlib

    payload = np.ascontiguousarray(rows, dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest().upper()


# --------------------------------------------------------------------------
# Direct branch outputs
# --------------------------------------------------------------------------
#: Keys a direct branch record must carry.  ``metrics`` refuses to report a
#: mechanism number from a record that is missing any of them, because the
#: fallback -- decomposing the net correction -- answers a different question.
DIRECT_BRANCH_KEYS = ("shape_positive", "shape_negative",
                      "mass_positive", "mass_negative", "correction")

DIRECT = "DIRECT_BRANCH_HEADS"
DECOMPOSED = "DECOMPOSED_NET_CORRECTION"

#: Tolerance for ``c == A+ S+ - A- S-``.  Both sides are persisted as float64
#: after a float32 forward pass, so the check is scaled by the correction's own
#: magnitude.  It exists to catch a mis-wired or substituted branch array, not
#: to certify floating-point determinism.
RECONSTRUCTION_RTOL = 1e-5
RECONSTRUCTION_ATOL = 1e-6


def _direct_branch_arrays(branch: Mapping[str, Any],
                          shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Validate a direct branch record against the evaluation grid."""
    missing = [k for k in DIRECT_BRANCH_KEYS if k not in branch]
    if missing:
        raise ProvenanceError(
            "direct branch evidence is incomplete; missing " + ", ".join(missing))
    n, h = shape
    out: Dict[str, np.ndarray] = {}
    for key in ("shape_positive", "shape_negative", "correction"):
        arr = np.asarray(branch[key], dtype=np.float64)
        if arr.shape != (n, h):
            raise ProvenanceError(
                f"direct branch array {key!r} has shape {arr.shape}, "
                f"but the evaluation grid is {(n, h)}")
        out[key] = arr
    for key in ("mass_positive", "mass_negative"):
        arr = np.asarray(branch[key], dtype=np.float64).reshape(-1)
        if arr.shape != (n,):
            raise ProvenanceError(
                f"direct branch array {key!r} has shape {arr.shape}, "
                f"but the evaluation grid has {n} day(s)")
        out[key] = arr
    return out


def branch_reconstruction_error(branch: Mapping[str, Any],
                                shape: Tuple[int, int]) -> float:
    """``max |c - (A_hat+ S_hat+ - A_hat- S_hat-)|`` over the whole block.

    This is the algebraic identity the method is defined by.  A non-zero value
    beyond float32 rounding means the persisted branch arrays did not produce
    the persisted correction, which would make every mechanism metric a
    description of some other object.
    """
    a = _direct_branch_arrays(branch, shape)
    rebuilt = (a["mass_positive"][:, None] * a["shape_positive"]
               - a["mass_negative"][:, None] * a["shape_negative"])
    if rebuilt.size == 0:
        return 0.0
    return float(np.max(np.abs(rebuilt - a["correction"])))


def _tolerance(scale: np.ndarray) -> float:
    magnitude = float(np.max(np.abs(scale))) if scale.size else 0.0
    return RECONSTRUCTION_ATOL + RECONSTRUCTION_RTOL * max(1.0, magnitude)


# --------------------------------------------------------------------------
# The metric set
# --------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, host_pred: np.ndarray, method_pred: np.ndarray,
                    valid_mask: Optional[np.ndarray] = None,
                    market: Optional[str] = None,
                    host_correction_zero: bool = False,
                    branch: Optional[Mapping[str, Any]] = None,
                    high_mass: Optional[Mapping[str, Any]] = None,
                    alpha: Optional[float] = None) -> Dict[str, Any]:
    """Everything the method protocol reports, for one evaluated cell.

    ``host_correction_zero`` is set when ``method_pred`` *is* the Host, which
    makes the shape/mass metrics degenerate by construction rather than by
    failure; the returned ``shape_undefined_days`` count says so explicitly.

    ``branch`` is the direct branch record produced by the same forward pass
    that produced ``method_pred`` (see :data:`DIRECT_BRANCH_KEYS`).  When it is
    supplied, **every** Shape/mass/high-mass number is computed from the branch
    heads themselves and ``branch_arrays_used`` says so.  When it is omitted the
    numbers are recovered by decomposing the net correction and the record is
    marked ``DECOMPOSED_NET_CORRECTION``: that path is a diagnostic only, since
    ``S_hat+`` and ``S_hat-`` may overlap at a horizon and cancel inside ``c``,
    so the decomposition describes a different object than the branches did.

    ``high_mass`` is the cell's frozen ``POST_TRAIN`` q90 pair
    (:func:`fit_high_mass_thresholds`).  It is required for the two high-mass
    mechanism numbers and is never re-derived from the evaluated partition.
    """
    y = np.asarray(y_true, dtype=np.float64)
    hp = np.asarray(host_pred, dtype=np.float64)
    mp = np.asarray(method_pred, dtype=np.float64)
    if not (y.shape == hp.shape == mp.shape) or y.ndim != 2:
        raise ValueError("y_true, host_pred and method_pred must share shape (N, H)")
    mask = (np.ones_like(y, dtype=bool) if valid_mask is None
            else np.asarray(valid_mask, dtype=bool))
    if mask.shape != y.shape:
        raise ValueError("valid_mask must match (N, H)")
    if market is None:
        raise ValueError("market is required: the frozen thresholds are per market")

    th = load_thresholds(market)
    q05, q95, spread_p90 = th["q05"], th["q95"], th["day_spread_p90"]

    err_m = np.abs(mp - y)
    err_h = np.abs(hp - y)

    lower = (y <= q05) & mask
    upper = (y >= q95) & mask
    tail = lower | upper
    normal = mask & ~tail
    negative = (y < 0.0) & mask

    day_spread = y.max(axis=1) - y.min(axis=1)
    high_spread_days = day_spread >= spread_p90
    high_spread = np.broadcast_to(high_spread_days[:, None], y.shape) & mask

    mae_m = _mae(mp, y, mask)
    mae_h = _mae(hp, y, mask)
    normal_m = _mae(mp, y, normal)
    normal_h = _mae(hp, y, normal)

    # The realised side is always the exact signed-mass factorization of the
    # Host's own residual, masked exactly as the core's training targets are.
    realised = shape_decomposition((y - hp) * mask)

    # The predicted side is the direct branch when it was handed over, and the
    # decomposition of the net correction only as an explicitly-flagged fallback.
    recon_err: Optional[float] = None
    if branch is not None:
        direct = _direct_branch_arrays(branch, y.shape)
        pred_mass_pos = direct["mass_positive"]
        pred_mass_neg = direct["mass_negative"]
        pred_shape_pos = direct["shape_positive"]
        pred_shape_neg = direct["shape_negative"]
        pred_defined_pos = np.isfinite(pred_mass_pos)
        pred_defined_neg = np.isfinite(pred_mass_neg)
        branch_arrays_used = DIRECT
        rebuilt = (pred_mass_pos[:, None] * pred_shape_pos
                   - pred_mass_neg[:, None] * pred_shape_neg)
        recon_err = float(np.max(np.abs(rebuilt - direct["correction"]))) \
            if rebuilt.size else 0.0
        if recon_err > _tolerance(direct["correction"]):
            raise ProvenanceError(
                "the persisted branch arrays do not reconstruct the persisted "
                f"correction (max |c - (A+S+ - A-S-)| = {recon_err:.3e} exceeds "
                f"tolerance {_tolerance(direct['correction']):.3e}); every "
                "mechanism metric would describe a different object")
    else:
        decomposed = shape_decomposition(mp - hp)
        pred_mass_pos = decomposed["mass_positive"]
        pred_mass_neg = decomposed["mass_negative"]
        pred_shape_pos = decomposed["shape_positive"]
        pred_shape_neg = decomposed["shape_negative"]
        pred_defined_pos = decomposed["defined_positive"]
        pred_defined_neg = decomposed["defined_negative"]
        branch_arrays_used = DECOMPOSED

    alpha_err: Optional[float] = None
    if alpha is not None and branch is not None:
        expected = hp + float(alpha) * np.asarray(branch["correction"], dtype=np.float64)
        alpha_err = float(np.max(np.abs(expected - mp))) if mp.size else 0.0
        if alpha_err > _tolerance(mp):
            raise ProvenanceError(
                "the evaluated prediction is not host + alpha * correction "
                f"(max deviation {alpha_err:.3e}); the metric block and the "
                "persisted calibration do not describe the same prediction")

    def _shape_pair(sign: str) -> Dict[str, Any]:
        real_shape = realised[f"shape_{sign}"]
        real_defined = realised[f"defined_{sign}"]
        pred_shape = pred_shape_pos if sign == "positive" else pred_shape_neg
        pred_defined = pred_defined_pos if sign == "positive" else pred_defined_neg
        ok = real_defined & pred_defined
        n_ok = int(ok.sum())
        if n_ok == 0:
            return {"w1": None, "cosine": None, "n_days_defined": 0}
        w1 = _wasserstein1(real_shape[ok], pred_shape[ok])
        cos = _cosine(real_shape[ok], pred_shape[ok])
        return {"w1": float(w1.mean()),
                "cosine": float(np.nanmean(cos)) if np.isfinite(cos).any() else None,
                "n_days_defined": n_ok}

    shape_pos = _shape_pair("positive")
    shape_neg = _shape_pair("negative")

    valid_days = mask.any(axis=1)
    mass_pos_err = np.abs(pred_mass_pos - realised["mass_positive"])[valid_days]
    mass_neg_err = np.abs(pred_mass_neg - realised["mass_negative"])[valid_days]

    high_mass_block: Dict[str, Any] = {
        "mass_thresholds_used": (None if high_mass is None else dict(high_mass)),
        "high_mass_positive_l1": None,
        "high_mass_negative_l1": None,
        "high_mass_positive_n_days": 0,
        "high_mass_negative_n_days": 0,
        "high_mass_positive_status": C.NOT_APPLICABLE_N0,
        "high_mass_negative_status": C.NOT_APPLICABLE_N0,
    }
    if high_mass is not None:
        label = f"q{int(round(float(high_mass['quantile']) * 100))}"
        for sign, pred_mass in (("positive", pred_mass_pos),
                                ("negative", pred_mass_neg)):
            realised_mass = realised[f"mass_{sign}"]
            threshold = float(high_mass.get(f"{label}_{sign}", 0.0))
            usable = bool(high_mass.get(f"usable_{sign}", False))
            picked = valid_days & (realised_mass > 0.0) & (realised_mass >= threshold)
            n_picked = int(picked.sum())
            high_mass_block[f"high_mass_{sign}_n_days"] = n_picked
            if not usable:
                high_mass_block[f"high_mass_{sign}_status"] = (
                    f"{C.NOT_APPLICABLE_N0}:THRESHOLD_NOT_USABLE")
                continue
            if n_picked == 0:
                high_mass_block[f"high_mass_{sign}_status"] = (
                    f"{C.NOT_APPLICABLE_N0}:EMPTY_DEV_SUBSET")
                continue
            high_mass_block[f"high_mass_{sign}_l1"] = float(
                np.abs(pred_mass[picked] - realised_mass[picked]).mean())
            high_mass_block[f"high_mass_{sign}_status"] = "REPORTED"

    return {
        "market": market,
        "n_days": int(y.shape[0]),
        "n_entries_evaluated": int(mask.sum()),
        "n_days_any_valid": int(valid_days.sum()),
        "overall_mae": mae_m,
        "overall_mse": _mae((mp - y) ** 2, np.zeros_like(y), mask),
        "overall_rmse": (float(np.sqrt(_mae((mp - y) ** 2, np.zeros_like(y), mask)))
                         if mask.any() else None),
        "host_mae": mae_h,
        "host_mse": _mae((hp - y) ** 2, np.zeros_like(y), mask),
        "mae_gain_abs": (None if (mae_m is None or mae_h is None) else mae_h - mae_m),
        "mae_gain_pct": (None if not mae_h else 100.0 * (mae_h - mae_m) / mae_h),
        "branch_arrays_used": branch_arrays_used,
        "branch_reconstruction_max_abs_err": recon_err,
        "alpha_consistency_max_abs_err": alpha_err,
        "shape_w1_positive": shape_pos["w1"],
        "shape_w1_negative": shape_neg["w1"],
        "shape_cosine_positive": shape_pos["cosine"],
        "shape_cosine_negative": shape_neg["cosine"],
        "shape_undefined_days_positive": int(
            (~(realised["defined_positive"] & pred_defined_pos)).sum()),
        "shape_undefined_days_negative": int(
            (~(realised["defined_negative"] & pred_defined_neg)).sum()),
        "shape_days_defined_positive": shape_pos["n_days_defined"],
        "shape_days_defined_negative": shape_neg["n_days_defined"],
        "mass_positive_l1": float(mass_pos_err.mean()) if mass_pos_err.size else None,
        "mass_negative_l1": float(mass_neg_err.mean()) if mass_neg_err.size else None,
        "tail_mae": _mae(mp, y, tail),
        "tail_mae_host": _mae(hp, y, tail),
        "tail_n_entries": int(tail.sum()),
        "upper_tail_mae": _mae(mp, y, upper),
        "upper_tail_mae_host": _mae(hp, y, upper),
        "upper_tail_n_entries": int(upper.sum()),
        "lower_tail_mae": _mae(mp, y, lower),
        "lower_tail_mae_host": _mae(hp, y, lower),
        "lower_tail_n_entries": int(lower.sum()),
        "normal_mae": normal_m,
        "normal_mae_host": normal_h,
        "normal_n_entries": int(normal.sum()),
        "normal_relative_harm_pct": (None if not normal_h else
                                     100.0 * (normal_m - normal_h) / normal_h),
        "negative_price_mae": _mae(mp, y, negative),
        "negative_price_mae_host": _mae(hp, y, negative),
        "negative_price_n_entries": int(negative.sum()),
        "negative_price_n_days": int((negative.any(axis=1)).sum()),
        "high_spread_day_mae": _mae(mp, y, high_spread),
        "high_spread_day_mae_host": _mae(hp, y, high_spread),
        "high_spread_n_days": int(high_spread_days.sum()),
        "host_correction_zero": bool(host_correction_zero),
        "thresholds_used": {
            "q05": q05, "q95": q95, "day_spread_p90": spread_p90,
            "source": th["threshold_source"],
            "source_sha256": th["threshold_source_sha256"],
            "fit_partition": th.get("fit_partition"),
            "rule": th["rule"],
            "recomputed_for_this_evaluation": False,
        },
        **high_mass_block,
    }


def definitions() -> Dict[str, str]:
    """The metric dictionary, emitted alongside every result file."""
    return dict(METRIC_DEFINITIONS)
