"""Exact pooled MAE calibration for a fitted repair model.

For one fitted market × Host repair instance, collect its legal chronological OOF
candidate corrections ``c_i`` and residuals ``r_i`` on the calibration rows and
fit one non-negative scalar

    alpha* = argmin_{alpha >= 0} sum_i |r_i - alpha c_i|.

For nonzero ``c_i`` the unconstrained solution is a weighted median of
``r_i/c_i`` with weights ``|c_i|``; the registered historical contract then
applies the non-negative constraint ``max(0, ·)``.

This is pooled across coordinates/OOF rows *within the fitted repair instance*.
It is not per-instance/test-day adaptation and it is not one cross-market scalar.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

__all__ = ["weighted_median", "fit_mae_scalar", "apply_mae_scalar", "mae_objective"]

ArrayLike = Union[np.ndarray, torch.Tensor]


def _flat(x: ArrayLike) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def weighted_median(values: ArrayLike, weights: ArrayLike,
                    threshold: float = 0.5) -> np.ndarray:
    """Deterministic weighted median along the last axis."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.shape != w.shape or v.ndim == 0:
        raise ValueError("values and weights must have the same non-scalar shape")
    if v.shape[-1] == 0:
        raise ValueError("weighted median requires a non-empty support")
    if not np.isfinite(v).all() or not np.isfinite(w).all():
        raise ValueError("weighted median inputs must be finite")
    if (w < 0).any():
        raise ValueError("weighted median weights must be non-negative")
    mass = w.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("weighted median requires positive total mass")

    normalised = w / mass
    order = np.argsort(v, axis=-1, kind="mergesort")
    ordered_values = np.take_along_axis(v, order, axis=-1)
    ordered_weights = np.take_along_axis(normalised, order, axis=-1)
    cumulative = np.cumsum(ordered_weights, axis=-1)
    index = np.argmax(cumulative >= float(threshold), axis=-1)
    return np.take_along_axis(ordered_values, index[..., None], axis=-1)[..., 0]


def mae_objective(candidate: ArrayLike, residual: ArrayLike,
                  alpha: float, valid_mask: Optional[ArrayLike] = None) -> float:
    c, r = _flat(candidate), _flat(residual)
    if c.shape != r.shape:
        raise ValueError("candidate and residual must have the same shape")
    if valid_mask is not None:
        keep = _flat(valid_mask) > 0
        if keep.shape != c.shape:
            raise ValueError("valid_mask must match candidate and residual")
        c, r = c[keep], r[keep]
    return float(np.abs(r - float(alpha) * c).sum())


def fit_mae_scalar(candidate: ArrayLike, residual: ArrayLike,
                   valid_mask: Optional[ArrayLike] = None,
                   threshold: float = 0.5,
                   nonnegative: bool = True) -> float:
    """Fit the exact pooled MAE scalar from a legal calibration row set.

    ``nonnegative=True`` is the project default and reproduces the validated
    calibrated-ray contract. Zero candidate coordinates contribute a constant to
    MAE and therefore are removed before forming ratios.
    """
    c, r = _flat(candidate), _flat(residual)
    if c.shape != r.shape:
        raise ValueError("candidate and residual must have the same shape")
    if valid_mask is not None:
        keep = _flat(valid_mask) > 0
        if keep.shape != c.shape:
            raise ValueError("valid_mask must match candidate and residual")
        c, r = c[keep], r[keep]

    usable = np.isfinite(c) & np.isfinite(r) & (c != 0.0)
    if not usable.any():
        return 0.0

    ratios = r[usable] / c[usable]
    weights = np.abs(c[usable])
    alpha = float(weighted_median(ratios, weights, threshold=threshold))
    return max(0.0, alpha) if nonnegative else alpha


def apply_mae_scalar(candidate: ArrayLike, alpha: float) -> ArrayLike:
    if alpha < 0:
        raise ValueError("registered MAE calibration scalar must be non-negative")
    if torch.is_tensor(candidate):
        return candidate * float(alpha)
    return np.asarray(candidate) * float(alpha)
