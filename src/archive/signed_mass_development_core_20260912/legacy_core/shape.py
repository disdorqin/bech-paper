"""Shape branch: WHERE the correction mass belongs across the horizon.

Shape emits two distributions over the horizon, one per sign.  A softmax over
the horizon axis makes the simplex invariant structural rather than something a
caller has to hope for: entries are non-negative and each row sums to one.

The head is per-horizon-step, so the model is not tied to a 24-hour day.  No tail
threshold, quantile grid, or regime cut is baked in.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .contracts import ShapeOutput

__all__ = ["ShapeBranch"]


class ShapeBranch(nn.Module):
    """Per-step logits for both signs, normalised over the horizon."""

    def __init__(self, d_input: int, hidden: int = 32, dropout: float = 0.0,
                 d_context: int = 0):
        super().__init__()
        layers = [nn.Linear(d_input + d_context, hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)
        self.d_context = d_context

    def forward(self, steps: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> ShapeOutput:
        """``steps`` is ``(B, H, d_input)``; ``context`` is ``(B, H, d_context)``.

        The optional ``context`` is the similar-window prototype.  It is an
        input only: it never becomes the correction by itself.
        """
        if self.d_context:
            if context is None:
                raise ValueError("ShapeBranch was built with context but none was given")
            steps = torch.cat([steps, context], dim=-1)
        logits = self.net(steps)                     # (B, H, 2)
        probs = torch.softmax(logits, dim=1)         # normalise over the horizon
        return ShapeOutput(positive=probs[..., 0], negative=probs[..., 1])
