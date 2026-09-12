"""Amplitude branch: HOW MUCH correction mass the day carries, as one scalar.

The surviving generator uses one nonnegative scalar for both signs:

    A = s_A * softplus(f(z_A)),        A >= 0.

This is an **exact reparameterization** of the tied forward semantics
``A+ = A- = A`` that survived adjudication, not an approximation of it: with
``A+ = A- = A`` the fusion ``A+ S+ - A- S-`` collapses to ``A (S+ - S-)``, so a
second head would be a free parameter with no effect on the forecast.

Because the output is one scalar, this class has no sign-specific expert, no
tied/untied switch and no second readout.  It also carries no daily level or bias
term: the candidate is a within-day residual **redistribution** operator, and a
level branch would change what the object is.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BalancedAmplitudeBranch"]


class BalancedAmplitudeBranch(nn.Module):
    """One trunk plus one nonnegative scalar readout."""

    def __init__(self, d_input: int, hidden: int = 32, dropout: float = 0.0,
                 scale: float = 1.0):
        super().__init__()
        if scale < 0:
            raise ValueError("amplitude scale must be non-negative")
        layers = [nn.Linear(d_input, hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)
        self.scale = float(scale)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """``(B, d_input) -> (B,)``, every entry ``>= 0``."""
        return self.scale * F.softplus(self.head(self.trunk(latent))).squeeze(-1)
