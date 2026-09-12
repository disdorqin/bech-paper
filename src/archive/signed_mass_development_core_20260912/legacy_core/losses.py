"""Stages 5-6 -- training objectives.

Design reference:
``docs/current/HCH_SIGNED_MASS_CONTEXT_ROUTED_NETWORK_DESIGN_20260912.md``
sections 12, 19 and 21.

Only the objectives the signed-mass factorization actually needs are here::

    L_R = (1/H) * sum_h |r_h - c_h|                    (section 19)
    L_S = w+ W1(S+, S+_hat) + w- W1(S-, S-_)hat)       (section 12)
    L_A = |A+_bar - A+_hat_bar| + |A-_bar - A-_hat_bar| (section 19)
    L   = L_R + lambda_S L_S + lambda_A L_A            (section 19)

with the training-only directional weights

    w+- = A+- / (A+ + A- + eps)

and zero-mass directions masked out of their own Shape term, since ``S+-`` is
undefined when the corresponding mass is zero.

There is no gate objective, no CORN objective, no quantile-grid objective and no
market-specific weighting.  Shape losses are computed on the simplex; when a
mask is supplied the distribution is renormalised over the valid steps first, so
a partially observed day is compared as a distribution over what was observed.

The default weights ``lambda_S = lambda_A = 1`` are the design's *discussion*
default and are **not scientifically frozen**.  Province-specific loss tuning is
not allowed.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

__all__ = [
    "repair_mae",
    "shape_wasserstein1",
    "shape_l1",
    "shape_loss",
    "fit_mass_normalization",
    "amplitude_l1",
    "amplitude_loss",
    "combined_loss",
]

_EPS = 1e-12


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"unknown reduction {reduction!r}")


def _restrict(shape: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if valid_mask is None:
        return shape
    restricted = shape * valid_mask
    return restricted / restricted.sum(dim=-1, keepdim=True).clamp_min(_EPS)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def repair_mae(prediction: torch.Tensor, target: torch.Tensor,
               valid_mask: Optional[torch.Tensor] = None,
               reduction: str = "mean") -> torch.Tensor:
    """``L_R``: masked mean absolute error, per sample before reduction."""
    error = (prediction - target).abs()
    if valid_mask is not None:
        count = valid_mask.sum(dim=-1).clamp_min(1.0)
        error = (error * valid_mask).sum(dim=-1) / count
    else:
        error = error.mean(dim=-1)
    return _reduce(error, reduction)


# ---------------------------------------------------------------------------
# Shape (ordered location error)
# ---------------------------------------------------------------------------

def shape_wasserstein1(predicted: torch.Tensor, target: torch.Tensor,
                       valid_mask: Optional[torch.Tensor] = None,
                       reduction: str = "mean") -> torch.Tensor:
    """``W1 = sum_{h=1}^{H-1} |F_pred(h) - F_target(h)|`` on the horizon."""
    p = _restrict(predicted, valid_mask)
    q = _restrict(target, valid_mask)
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    return _reduce((cdf_p - cdf_q).abs()[..., :-1].sum(dim=-1), reduction)


def shape_l1(predicted: torch.Tensor, target: torch.Tensor,
             valid_mask: Optional[torch.Tensor] = None,
             reduction: str = "mean") -> torch.Tensor:
    """Pointwise L1 between the two distributions.  Ablation alternative to W1."""
    p = _restrict(predicted, valid_mask)
    q = _restrict(target, valid_mask)
    return _reduce((p - q).abs().sum(dim=-1), reduction)


def shape_loss(positive_predicted: torch.Tensor, positive_target: torch.Tensor,
               negative_predicted: torch.Tensor, negative_target: torch.Tensor,
               mass_positive: torch.Tensor, mass_negative: torch.Tensor,
               valid_mask: Optional[torch.Tensor] = None,
               metric: str = "wasserstein") -> Dict[str, torch.Tensor]:
    """``L_S`` with training-only directional weights and zero-mass masking.

    ``mass_positive`` / ``mass_negative`` are the observed ``A+`` / ``A-`` of the
    training window; they set the direction weights
    ``w+- = A+- / (A+ + A- + eps)``.  A direction whose observed mass is zero is
    excluded from its own term, because ``S+-`` is undefined there.

    The per-sample total is divided by the sum of the weights actually applied,
    so the loss scale does not collapse when a branch is masked out.
    """
    if metric not in ("wasserstein", "l1"):
        raise ValueError(f"metric must be 'wasserstein' or 'l1', got {metric!r}")
    fn = shape_wasserstein1 if metric == "wasserstein" else shape_l1

    d_pos = fn(positive_predicted, positive_target, valid_mask, reduction="none")
    d_neg = fn(negative_predicted, negative_target, valid_mask, reduction="none")

    total_mass = (mass_positive + mass_negative).clamp_min(0.0)
    share = total_mass + _EPS
    w_pos = (mass_positive / share) * (mass_positive > 0).to(mass_positive.dtype)
    w_neg = (mass_negative / share) * (mass_negative > 0).to(mass_negative.dtype)

    numerator = w_pos * d_pos + w_neg * d_neg
    denominator = (w_pos + w_neg).clamp_min(_EPS)
    total = (numerator / denominator).mean()
    return {"total": total,
            "positive": d_pos.mean(),
            "negative": d_neg.mean(),
            "weight_positive": w_pos.mean(),
            "weight_negative": w_neg.mean()}


# ---------------------------------------------------------------------------
# Amplitude (MAE-aligned magnitude error)
# ---------------------------------------------------------------------------

def fit_mass_normalization(mass_positive: torch.Tensor,
                           mass_negative: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robust centre/scale for the mass normalization of section 19.

    **Training partition only.**  This is a fitted statistic, so it falls under
    the same rule as every other scaler: fit on ``POST_TRAIN``, never on
    ``DEV_EVAL``, never on ``PROTECTED_FINAL``.
    """
    both = torch.stack([torch.as_tensor(mass_positive, dtype=torch.float32).reshape(-1),
                        torch.as_tensor(mass_negative, dtype=torch.float32).reshape(-1)])
    center = both.median(dim=1).values
    mad = (both - center.unsqueeze(1)).abs().median(dim=1).values
    return center, (1.4826 * mad).clamp_min(_EPS)


def amplitude_l1(predicted: torch.Tensor, target: torch.Tensor,
                 center: Optional[torch.Tensor] = None,
                 scale: Optional[torch.Tensor] = None,
                 reduction: str = "mean") -> torch.Tensor:
    """``L_A = |A_bar - A_hat_bar|`` summed over both signs.

    ``center`` / ``scale`` are the frozen training-only mass normalization from
    :func:`fit_mass_normalization`; omitting them compares masses in raw units.
    """
    predicted = torch.as_tensor(predicted, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    if center is not None and scale is not None:
        predicted = (predicted - center) / scale
        target = (target - center) / scale
    return _reduce((predicted - target).abs().sum(dim=-1), reduction)


def amplitude_loss(positive_predicted: torch.Tensor, positive_target: torch.Tensor,
                   negative_predicted: torch.Tensor, negative_target: torch.Tensor,
                   center: Optional[torch.Tensor] = None,
                   scale: Optional[torch.Tensor] = None,
                   reduction: str = "mean") -> torch.Tensor:
    """``L_A`` over the two sign channels stacked as ``(B, 2)``."""
    predicted = torch.stack([positive_predicted, negative_predicted], dim=-1)
    target = torch.stack([positive_target, negative_target], dim=-1)
    return amplitude_l1(predicted, target, center, scale, reduction)


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def combined_loss(*, prediction: torch.Tensor, target: torch.Tensor,
                  shape_positive_predicted: torch.Tensor,
                  shape_positive_target: torch.Tensor,
                  shape_negative_predicted: torch.Tensor,
                  shape_negative_target: torch.Tensor,
                  amplitude_positive_predicted: torch.Tensor,
                  amplitude_positive_target: torch.Tensor,
                  amplitude_negative_predicted: torch.Tensor,
                  amplitude_negative_target: torch.Tensor,
                  mass_positive: torch.Tensor, mass_negative: torch.Tensor,
                  valid_mask: Optional[torch.Tensor] = None,
                  weight_repair: float = 1.0,
                  weight_shape: float = 1.0,
                  weight_amplitude: float = 1.0,
                  shape_metric: str = "wasserstein",
                  mass_center: Optional[torch.Tensor] = None,
                  mass_scale: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """``L = L_R + lambda_S L_S + lambda_A L_A``, returned component-wise.

    Components are returned alongside the total so a training loop can log them
    without recomputing anything.

    Note on the gradient graph (design section 20): ``L_R`` alone already
    couples the branches through ``d c_h / d A+- = +-S+-_h`` and
    ``d c_h / d S+-_h = +-A+-``.  The auxiliary terms sharpen each branch on its
    own object.  The pooled ``alpha`` calibration is outside this graph.
    """
    repair = repair_mae(prediction, target, valid_mask, reduction="mean")
    shape = shape_loss(shape_positive_predicted, shape_positive_target,
                       shape_negative_predicted, shape_negative_target,
                       mass_positive, mass_negative, valid_mask, shape_metric)
    amplitude = amplitude_loss(amplitude_positive_predicted, amplitude_positive_target,
                               amplitude_negative_predicted, amplitude_negative_target,
                               mass_center, mass_scale, reduction="mean")
    total = (weight_repair * repair
             + weight_shape * shape["total"]
             + weight_amplitude * amplitude)
    return {"total": total,
            "repair_mae": repair,
            "shape": shape["total"],
            "shape_positive": shape["positive"],
            "shape_negative": shape["negative"],
            "amplitude": amplitude}
