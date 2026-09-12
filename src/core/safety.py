"""Exact MAE safe-ray primitives for Residual Alignment Safety.

Math authority:
``paper/02_math/RESIDUAL_ALIGNMENT_SAFE_RAY_DERIVATION_20260912.md``.
Scientific authority:
``docs/current/HCH_RESIDUAL_ALIGNMENT_SAFETY_DESIGN_20260912.md``.

Everything in this module is a **pure function**.  There is no learned module, no
gate, no selector, no uncertainty head and no threshold: the deployment scalar is
computed from closed-form convex geometry on the last seven honest records, and
the whole controller is therefore auditable by reading this file.

The object is one ray.  A frozen Host gives ``y_H``; a frozen candidate generator
produces ``c`` before the target is known; the realised residual is ``r = y - y_H``.
Deployment is restricted to

    y(lambda) = y_H + lambda * c,      lambda >= 0

and the question is only how much of an already-generated correction to keep.

Exact ray geometry
------------------

    L(lambda; r, c) = || r - lambda c ||_1

For coordinates with ``c_i != 0`` write ``z_i = r_i / c_i`` and ``w_i = |c_i|``::

    L(lambda) = sum_{c_i != 0} w_i |z_i - lambda| + sum_{c_i = 0} |r_i|

so the objective is piecewise linear in ``lambda`` with breakpoints exactly at the
ratios ``r_i / c_i`` and weights ``|c_i|`` (or ``kappa_s |c_i|`` for the
Shape-weighted risk).  Every radius below is therefore the right endpoint of an
explicit no-harm interval, computed by scanning sorted positive breakpoints.  No
grid, no search and no tolerance to tune.

Definitions used here, with ``Q = sum_i q_i`` and ``Q_+ = sum_{z_i > 0} q_i``::

    R(lambda)   = sum_i q_i ( |z_i - lambda| - |z_i| )        excess MAE over the Host
    A_align     = 2 Q_+ / Q - 1                               alignment support
    D'(0+)      = Q - 2 Q_+ = -Q * A_align                    right derivative of R at 0
    kappa_{d,s} = 1 - ( W1(S+_d,S+_s) + W1(S-_d,S-_s) ) / (2 (H-1))
    lambda      = min(alpha_0, lambda_U, lambda_S)

``R`` is convex and piecewise linear with ``R(0) = 0``, so its no-harm set is an
interval starting at zero and the radius is well defined.  Coordinates with
``r_i = 0`` have ``z_i = 0`` and are *not* supporting for a positive step; they
sit on the non-supporting side of ``A_align`` exactly as the derivation requires.
"""
from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch

__all__ = [
    "wasserstein1",
    "shape_relevance",
    "mae_excess_risk",
    "mae_alignment_support",
    "exact_mae_safe_radius",
    "shape_weighted_safe_radius",
    "deployment_scale",
]

ArrayLike = Union[np.ndarray, torch.Tensor]

_EPS = 1e-12


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _scalar_or_array(values: np.ndarray) -> Union[float, np.ndarray]:
    flat = np.asarray(values, dtype=np.float64)
    if flat.shape == (1,):
        return float(flat[0])
    return flat


def _valid_positions(valid_mask: Optional[ArrayLike], horizon: int) -> Optional[np.ndarray]:
    """Reduce a mask to a 1-D boolean position vector, or ``None``."""
    if valid_mask is None:
        return None
    m = _to_numpy(valid_mask)
    if m.ndim == 0:
        raise ValueError("valid_mask must have a horizon axis")
    if m.shape[-1] != horizon:
        raise ValueError("valid_mask final axis must match the horizon")
    if m.size == horizon:
        return m.reshape(horizon) > 0
    position = np.all(m > 0, axis=tuple(range(m.ndim - 1)))
    return position.reshape(horizon)


def _restrict(distribution: np.ndarray, keep: Optional[np.ndarray]) -> np.ndarray:
    if keep is None:
        return distribution
    masked = np.where(keep, distribution, 0.0)
    total = masked.sum(axis=-1, keepdims=True)
    return np.divide(masked, np.where(total > _EPS, total, 1.0))


def wasserstein1(predicted: ArrayLike, target: ArrayLike,
                 valid_mask: Optional[ArrayLike] = None) -> Union[float, np.ndarray]:
    """One-dimensional Wasserstein-1 between horizon distributions.

    ``W1 = sum_{h=1}^{H-1} |F_pred(h) - F_target(h)|`` over the valid cut points.
    For distributions on the simplex this lies in ``[0, H-1]``, which is what
    makes the Shape relevance below a bounded quantity rather than a free score.
    """
    p = _to_numpy(predicted)
    q = _to_numpy(target)
    if p.shape != q.shape:
        p, q = np.broadcast_arrays(p, q)
    if p.ndim < 1 or p.shape[-1] == 0:
        raise ValueError("distributions must have a non-empty horizon axis")
    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        raise ValueError("distributions must be finite")
    if (p < -_EPS).any() or (q < -_EPS).any():
        raise ValueError("distributions must be non-negative")

    horizon = p.shape[-1]
    keep = _valid_positions(valid_mask, horizon)
    if keep is not None and int(keep.sum()) < 2:
        raise ValueError("Wasserstein-1 needs at least two valid horizon positions")
    p = _restrict(p, keep)
    q = _restrict(q, keep)
    if keep is not None:
        p = p[..., keep]
        q = q[..., keep]
    cdf_p = np.cumsum(p, axis=-1)[..., :-1]
    cdf_q = np.cumsum(q, axis=-1)[..., :-1]
    return _scalar_or_array(np.abs(cdf_p - cdf_q).sum(axis=-1))


def shape_relevance(current_positive: ArrayLike, current_negative: ArrayLike,
                    historical_positive: ArrayLike, historical_negative: ArrayLike,
                    valid_mask: Optional[ArrayLike] = None) -> Union[float, np.ndarray]:
    """``kappa_{d,s} = 1 - (W1(S+_d,S+_s) + W1(S-_d,S-_s)) / (2 (H-1))``.

    Shape is shared as **relevance**, never as prediction: the formula says which
    past outcomes of the same frozen generator resemble the current candidate's
    geometry, and it introduces no similarity model, neighbour count, bandwidth or
    temperature.

    The result lives in ``[0, 1]``: ``W1 <= H-1`` for distributions on the simplex,
    so the ratio is a normalised disagreement and the clamp only absorbs
    float error.  With fewer than two valid horizon positions the Shape carries no
    discriminative information and the relevance is exactly ``1``.
    """
    cp = _to_numpy(current_positive)
    hp = _to_numpy(historical_positive)
    horizon = hp.shape[-1]
    keep = _valid_positions(valid_mask, horizon)
    n_valid = horizon if keep is None else int(keep.sum())
    denominator = 2.0 * (n_valid - 1)
    if denominator <= 0:
        shape = np.broadcast_shapes(np.shape(cp)[:-1], np.shape(hp)[:-1])
        return _scalar_or_array(np.ones(shape if shape else (1,)))

    distance = (np.asarray(wasserstein1(cp, hp, valid_mask))
                + np.asarray(wasserstein1(current_negative, historical_negative, valid_mask)))
    kappa = 1.0 - distance / denominator
    return _scalar_or_array(np.clip(kappa, 0.0, 1.0))


def _rays(residuals: ArrayLike, corrections: ArrayLike,
          weights: Optional[ArrayLike],
          valid_mask: Optional[ArrayLike]) -> tuple:
    """Flatten evidence into coordinate triples ``(r_i, c_i, q_i)``.

    ``q_i = |c_i|``, or ``|c_i| * w`` when weights are supplied.  The evidence is
    flattened to one coordinate list on purpose: the design's ``R^U`` and ``R^S``
    sum over **every** coordinate of **every** record in the bank, so a batched
    array is one risk, not one risk per row.  A caller wanting a per-record figure
    passes that record alone.
    """
    r = _to_numpy(residuals)
    c = _to_numpy(corrections)
    if r.shape != c.shape:
        r, c = np.broadcast_arrays(r, c)
    if r.ndim < 1:
        raise ValueError("residuals and corrections need a horizon axis")
    if not (np.isfinite(r).all() and np.isfinite(c).all()):
        raise ValueError("residuals and corrections must be finite")

    horizon = r.shape[-1]
    keep = _valid_positions(valid_mask, horizon)
    if keep is not None:
        r = np.where(keep, r, 0.0)
        c = np.where(keep, c, 0.0)

    if weights is None:
        q = np.abs(c)
    else:
        w = _to_numpy(weights)
        if not np.isfinite(w).all():
            raise ValueError("weights must be finite and non-negative")
        if (w < 0).any():
            raise ValueError("weights must be finite and non-negative")
        if w.ndim == 0:
            # One weight for the whole ray: a single record scored under one kappa.
            pass
        elif w.shape == r.shape:
            # A weight per coordinate, already aligned with the evidence.
            pass
        elif w.shape == r.shape[:-1]:
            # The design's case: one kappa per historical record, broadcast over
            # that record's horizon.
            w = w[..., None]
        elif w.shape == r.shape[:-1] + (1,):
            # The same per-record weight, already carrying its broadcast axis.
            pass
        else:
            raise ValueError(
                f"weights of shape {w.shape} do not describe the evidence of shape "
                f"{r.shape}; expected a scalar, one weight per record "
                f"{r.shape[:-1]}, or one weight per coordinate {r.shape}"
            )
        q = np.abs(c) * w
    return r.ravel(), c.ravel(), q.ravel()


def mae_excess_risk(residuals: ArrayLike, corrections: ArrayLike,
                    lam: float = 0.0,
                    weights: Optional[ArrayLike] = None,
                    valid_mask: Optional[ArrayLike] = None) -> float:
    """``R(lambda) = sum_i q_i (|z_i - lambda| - |z_i|)`` — excess MAE over the Host.

    Written in ratio form with ``z_i = r_i / c_i`` and ``q_i = |c_i| w_i``.  For a
    single record this is identically ``sum_i (|r_i - lambda c_i| - |r_i|)``,
    because ``|c_i| |z_i - lambda| = |r_i - lambda c_i|``; the weight ``w`` is the
    per-record Shape relevance, so with ``w = 1`` the expression is exactly the
    unweighted L1 excess of the design.

    Zero exactly at ``lambda = 0`` and convex piecewise linear in ``lambda``.
    Coordinates with ``c_i = 0`` carry ``q_i = 0`` and contribute nothing at any
    strength, which is the exact behaviour rather than a convention.
    """
    lam = float(lam)
    if not math.isfinite(lam):
        raise ValueError("lambda must be finite")
    r, c, q = _rays(residuals, corrections, weights, valid_mask)
    z = np.divide(r, c, out=np.zeros_like(r), where=c != 0.0)
    return float(((np.abs(z - lam) - np.abs(z)) * q).sum())


def mae_alignment_support(residuals: ArrayLike, corrections: ArrayLike,
                          weights: Optional[ArrayLike] = None,
                          valid_mask: Optional[ArrayLike] = None) -> float:
    """``A_align = 2 Q_+ / Q - 1`` — directional alignment of the correction with the residual.

    ``A_align > 0`` means a sufficiently small positive step strictly lowers MAE
    on the evidence; ``A_align < 0`` means every positive step is harmful, because
    ``R`` is convex and its derivative at zero is ``-Q * A_align``.  Coordinates
    with ``r_i = 0`` are counted on the non-supporting side, which is why they must
    not be treated as free.

    With no usable coordinate (``Q = 0``) there is no directional information and
    the statistic is ``0``, neither supporting nor opposing.
    """
    r, c, q = _rays(residuals, corrections, weights, valid_mask)
    live = (c != 0.0) & (q != 0.0)
    if not live.any():
        return 0.0
    quality = q[live]
    total = float(quality.sum())
    if total <= 0.0:
        return 0.0
    z = r[live] / c[live]
    supporting = float(quality[z > 0.0].sum())
    return 2.0 * supporting / total - 1.0


def _scan_safe_radius(r: np.ndarray, c: np.ndarray, q: np.ndarray) -> float:
    """Exact right endpoint of ``{lambda >= 0 : R(lambda) <= 0}`` for one ray.

    On the segment ``[z_k, z_{k+1})`` between two consecutive positive
    breakpoints the objective is exactly linear,

        R(lambda) = lambda * s_k - 2 * B_k,     s_k = Q - 2 Q_{>z_k},

    with ``s_k`` non-decreasing in ``k`` because ``Q_{>z_k}`` only shrinks.  Before
    the first breakpoint the slope is ``Q - 2 Q_+``; if that is already positive the
    function leaves zero upwards and the radius is exactly zero.

    The walk carries the exact value ``R(z_k)`` forward instead of using the
    intercept alone.  That matters: the line through the origin with slope ``s_k``
    is the objective only **up to** ``z_{k+1}``, so a segment whose slope has just
    turned positive can still have its line-root lying far beyond the segment.  The
    root is accepted only when it falls inside the segment it belongs to, and the
    walk continues otherwise.  Slices between the last breakpoint and infinity have
    slope ``Q > 0``, so the walk always terminates.

    Returns ``inf`` when the evidence carries no correction mass at all, i.e. when
    the design's no-harm set is the whole ray.
    """
    live = (c != 0.0) & (q > 0.0) & np.isfinite(r) & np.isfinite(c)
    if not live.any():
        return math.inf
    qq = q[live]
    total = float(qq.sum())
    if total <= 0.0:
        return math.inf
    zz = r[live] / c[live]

    positive = zz > 0.0
    breakpoints = zz[positive]
    masses = qq[positive]
    order = np.argsort(breakpoints, kind="mergesort")
    breakpoints = breakpoints[order]
    masses = masses[order]
    unique_z, inverse = np.unique(breakpoints, return_inverse=True)
    grouped = np.zeros(unique_z.shape[0], dtype=np.float64)
    np.add.at(grouped, inverse, masses)

    above = float(grouped.sum())          # Q_{>0}
    slope = total - 2.0 * above           # slope on [0, z_0)
    if slope > 0.0:
        return 0.0

    value = 0.0                           # R at the running breakpoint
    previous = 0.0
    for index in range(unique_z.shape[0]):
        position = float(unique_z[index])
        value += slope * (position - previous)
        above -= float(grouped[index])
        slope = total - 2.0 * above
        if slope > 0.0:
            crossing = position - value / slope
            if index + 1 == unique_z.shape[0] or crossing <= float(unique_z[index + 1]):
                return max(0.0, crossing)
        previous = position
    return math.inf


def exact_mae_safe_radius(residuals: ArrayLike, corrections: ArrayLike,
                          weights: Optional[ArrayLike] = None,
                          valid_mask: Optional[ArrayLike] = None) -> float:
    """Right endpoint of the empirical no-harm interval, exactly.

    ``lambda_U`` when called without ``weights``.  ``inf`` when the evidence
    carries no correction mass at all (``sum q = 0``), which means "this evidence
    imposes no restriction" rather than "any strength is safe".
    """
    r, c, q = _rays(residuals, corrections, weights, valid_mask)
    return _scan_safe_radius(r, c, q)


def shape_weighted_safe_radius(residuals: ArrayLike, corrections: ArrayLike,
                               relevance: ArrayLike,
                               valid_mask: Optional[ArrayLike] = None
                               ) -> Optional[float]:
    """``lambda_S``: the safe radius of the Shape-weighted recent risk.

    ``relevance`` holds ``kappa_{d,s}`` per historical record, broadcast over that
    record's horizon.  When the total Shape-weighted correction mass is zero the
    Shape supplies no additional information and this returns ``None``, meaning
    "impose no extra restriction"; the caller then keeps ``alpha_0`` for this
    channel rather than abstaining.
    """
    r, c, q = _rays(residuals, corrections, relevance, valid_mask)
    if not (q > 0.0).any():
        return None
    return _scan_safe_radius(r, c, q)


def deployment_scale(alpha_0: float, lambda_uniform: Optional[float] = None,
                     lambda_shape: Optional[float] = None) -> float:
    """``lambda = min(alpha_0, lambda_U, lambda_S)``.

    ``None`` and non-finite entries mean "that channel imposes no restriction" and
    are dropped rather than treated as a constraint.  The result is therefore
    always finite and never exceeds ``alpha_0``: the controller can only shrink a
    registered correction, never amplify it, and it never manufactures a
    correction for a candidate whose mass is zero.
    """
    a0 = float(alpha_0)
    if not math.isfinite(a0) or a0 < 0.0:
        raise ValueError("alpha_0 must be a finite non-negative scalar")
    candidates = [a0]
    for value in (lambda_uniform, lambda_shape):
        if value is None:
            continue
        v = float(value)
        if math.isnan(v):
            raise ValueError("safe radii must not be NaN")
        if not math.isfinite(v):
            continue
        candidates.append(max(0.0, v))
    return min(candidates)
