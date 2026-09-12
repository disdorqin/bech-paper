"""Stage 2 -- the single shared feature-interaction stem.

Design reference:
``docs/current/HCH_SIGNED_MASS_CONTEXT_ROUTED_NETWORK_DESIGN_20260912.md`` section 8.

There is exactly one MLP in the whole model.  Every base legal hourly token
passes through it::

    e_{d,h} = phi_theta0(x^base_{d,h})

so the stem is applied pointwise along the hour axis and its output keeps the
hour axis.  It exists so that low-cost cross-feature interaction happens once, on
a common footing, instead of being replicated inside every branch.  There is
deliberately no per-feature MLP, no per-market MLP and no per-Host MLP, and the
single set of weights is updated by gradients from both branches.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["UnifiedFeatureStem"]


class UnifiedFeatureStem(nn.Module):
    """``Linear(D, hidden) -> GELU -> LayerNorm -> Linear(hidden, out)``, pointwise.

    The module maps ``(..., D)`` to ``(..., d_out)``, so passing ``(B, H, D)``
    yields one embedding per hour.  Widths are engineering choices, not
    scientific claims.
    """

    def __init__(self, d_in: int, d_hidden: int = 64, d_out: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(d_in, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(d_hidden, d_out))
        self.net = nn.Sequential(*layers)
        self.d_out = d_out

    def forward(self, base_tokens: torch.Tensor) -> torch.Tensor:
        """``(B, H, D) -> (B, H, d_out)``."""
        return self.net(base_tokens)
