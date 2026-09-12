"""The pooled MAE scalar -- one global ``alpha``, fitted out-of-fold.

This module deliberately contains **no fitting mathematics**.  ``src/core``
already owns the exact solution (``fit_mae_scalar`` / ``mae_objective`` /
``apply_mae_scalar`` / ``weighted_median``); the harness only decides *which
rows are legal to fit on* and *what to record about the fit*.  Re-deriving the
weighted median here would be a second implementation of a frozen scientific
object, which the entry protocol forbids.

Registered contract:

* one scalar per (market, Host, configuration, seed) -- never per instance, per
  horizon step or per market;
* nonnegative, i.e. the method may not be inverted away from the Host;
* fitted on concatenated chronological out-of-fold corrections from
  ``POST_TRAIN`` only, never on ``DEV_EVAL`` and never on ``PROTECTED_FINAL``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .contracts import LegalityError
from .core_bridge import attr

__all__ = ["fit_pooled_alpha", "apply_pooled_alpha", "objective_at", "CONTRACT"]


CONTRACT = {
    "form": "y_hat = y_hat_host + alpha * c",
    "alpha_definition": "argmin_{alpha >= 0} sum_i |r_i - alpha * c_i|",
    "solution": "exact weighted median of r_i / c_i with weights |c_i|, c_i != 0",
    "granularity": "one global scalar per (market, host, configuration, seed)",
    "fit_rows": "chronological out-of-fold corrections from POST_TRAIN only",
    "provider": "src/core/calibration.py (imported, never re-implemented)",
}


def _flat(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def fit_pooled_alpha(candidate, residual, valid_mask=None,
                     key: str = "<cell>",
                     nonnegative: bool = True):
    """Fit the scalar and return ``(alpha, record)``.

    ``candidate`` are the raw corrections, ``residual`` the realised residuals
    ``y - y_hat_host``.  The two must describe the same rows.
    """
    fit_mae_scalar = attr("fit_mae_scalar")

    c, r = _flat(candidate), _flat(residual)
    if c.size == 0:
        raise LegalityError(f"{key}: no calibration rows; alpha is undefined")
    if c.shape != r.shape:
        raise LegalityError(
            f"{key}: candidate/residual shape mismatch ({c.shape} vs {r.shape})")

    keep = np.ones(c.shape, dtype=bool)
    if valid_mask is not None:
        vm = _flat(valid_mask)
        if vm.shape != c.shape:
            raise LegalityError(f"{key}: valid_mask shape mismatch")
        keep = vm > 0
    if not keep.any():
        raise LegalityError(f"{key}: every calibration entry is masked out")

    alpha = float(fit_mae_scalar(c, r, valid_mask, nonnegative=nonnegative))

    record = {
        "definition": CONTRACT["alpha_definition"],
        "solution": CONTRACT["solution"],
        "provider": CONTRACT["provider"],
        "granularity": CONTRACT["granularity"],
        "fit_rows": CONTRACT["fit_rows"],
        "nonnegative": bool(nonnegative),
        "n_entries": int(c.size),
        "n_valid_entries": int(keep.sum()),
        "n_zero_candidate_entries": int((c[keep] == 0.0).sum()),
        "objective_at_alpha": objective_at(c, r, alpha, keep),
        "objective_at_zero": objective_at(c, r, 0.0, keep),
        "objective_at_one": objective_at(c, r, 1.0, keep),
    }
    if nonnegative and alpha < 0.0:
        raise LegalityError(f"{key}: fitted alpha is negative ({alpha})")
    return alpha, record


def apply_pooled_alpha(host, correction, alpha: float) -> np.ndarray:
    """``y_hat = y_hat_host + alpha * c`` via the core's own applicator."""
    apply_mae_scalar = attr("apply_mae_scalar")
    a = float(alpha)
    if a < 0.0:
        raise LegalityError(f"alpha must be nonnegative, got {a}")
    return np.asarray(host, dtype=np.float64) + apply_mae_scalar(
        np.asarray(correction, dtype=np.float64), a)


def objective_at(candidate, residual, alpha: float,
                 valid_mask: Optional[np.ndarray] = None) -> float:
    """The pooled MAE objective value at a given ``alpha`` (core's definition)."""
    mae_objective = attr("mae_objective")
    mask = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    return float(mae_objective(_flat(candidate), _flat(residual), float(alpha), mask))
