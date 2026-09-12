"""Amplitude branch: HOW MUCH total positive/negative correction mass is needed.

The default uses one shared trunk and two untied non-negative heads.  A tied-head
mode exists only for the preregistered asymmetry ablation: the same scalar is
then used for both signs.  There are no sign-specific encoders or routers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import AmplitudeOutput

__all__ = ["AmplitudeBranch"]


class AmplitudeBranch(nn.Module):
    def __init__(self, d_input: int, hidden: int = 32, dropout: float = 0.0,
                 scale: float = 1.0, untied_heads: bool = True):
        super().__init__()
        layers = [nn.Linear(d_input, hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        self.untied_heads = bool(untied_heads)
        if self.untied_heads:
            self.head_positive = nn.Linear(hidden, 1)
            self.head_negative = nn.Linear(hidden, 1)
            self.head_shared = None
        else:
            self.head_shared = nn.Linear(hidden, 1)
            self.head_positive = None
            self.head_negative = None
        self.scale = float(scale)

    def forward(self, latent: torch.Tensor) -> AmplitudeOutput:
        z = self.trunk(latent)
        if self.untied_heads:
            positive = F.softplus(self.head_positive(z)).squeeze(-1)
            negative = F.softplus(self.head_negative(z)).squeeze(-1)
        else:
            shared = F.softplus(self.head_shared(z)).squeeze(-1)
            positive = shared
            negative = shared
        return AmplitudeOutput(
            positive=self.scale * positive,
            negative=self.scale * negative,
        )
