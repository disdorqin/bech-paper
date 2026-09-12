"""Deterministic signed-mass fusion, and the exact targets it is trained against.

The two branches meet through the factorization, never through a learned fusion
layer::

    r_h   = y_h - y_hat_h
    r_h^+ = max(r_h, 0),   r_h^- = max(-r_h, 0)
    A^+   = sum_h r_h^+,   A^-   = sum_h r_h^-
    S_h^+ = r_h^+ / A^+,   S_h^- = r_h^- / A^-
    r_h   = A^+ S_h^+ - A^- S_h^-
    c_h   = A_hat^+ S_hat_h^+ - A_hat^- S_hat_h^-

Zero mass is handled explicitly: when ``A = 0`` the corresponding Shape is set
to the uniform distribution and the mass to zero, so the product is exactly zero
and no ``0/0`` can appear.
"""
from __future__ import annotations

from typing import Optional

import torch

from .contracts import AmplitudeOutput, ShapeOutput, SignedMassTargets

__all__ = ["safe_shape", "decompose_residual", "fuse", "repair"]

_EPS = 1e-12


def safe_shape(mass_profile: torch.Tensor, total: torch.Tensor,
               eps: float = _EPS) -> torch.Tensor:
    """Normalise a non-negative mass profile, falling back to uniform at zero mass."""
    horizon = mass_profile.shape[-1]
    uniform = torch.full_like(mass_profile, 1.0 / horizon)
    positive_mass = total > eps
    normalised = mass_profile / total.clamp_min(eps).unsqueeze(-1)
    return torch.where(positive_mass.unsqueeze(-1), normalised, uniform)


def decompose_residual(residual: torch.Tensor,
                       valid_mask: Optional[torch.Tensor] = None,
                       eps: float = _EPS) -> SignedMassTargets:
    """Exact signed-mass factorization of a residual window.

    ``residual`` is ``(B, H)``, optionally ``(..., H)``.  Masked steps are zeroed
    before the masses are formed, so they contribute to neither sign.
    """
    r = residual if valid_mask is None else residual * valid_mask
    positive = torch.clamp(r, min=0.0)
    negative = torch.clamp(-r, min=0.0)
    mass_positive = positive.sum(dim=-1)
    mass_negative = negative.sum(dim=-1)
    return SignedMassTargets(
        mass_positive=mass_positive,
        mass_negative=mass_negative,
        shape_positive=safe_shape(positive, mass_positive, eps),
        shape_negative=safe_shape(negative, mass_negative, eps),
    )


def fuse(amplitude: AmplitudeOutput, shape: ShapeOutput) -> torch.Tensor:
    """``c_h = A_hat^+ S_hat_h^+ - A_hat^- S_hat_h^-``.  No learned parameters."""
    return (amplitude.positive.unsqueeze(-1) * shape.positive
            - amplitude.negative.unsqueeze(-1) * shape.negative)


def repair(host: torch.Tensor, correction: torch.Tensor,
           alpha: Optional[float] = None) -> torch.Tensor:
    """``y_hat_h = host_h + alpha * c_h``.  ``alpha`` is a pooled scalar or ``None``."""
    return host + correction if alpha is None else host + alpha * correction
