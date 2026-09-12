"""Shape branch: WHERE the correction mass belongs across the horizon.

Shape emits two distributions over the horizon, one per sign.  A softmax over the
horizon axis makes the simplex invariant structural rather than something a
caller has to hope for: entries are non-negative and the row sums to one over the
valid horizon.

Two properties matter beyond the simplex.

* The head is per-horizon-step, so the model is not tied to a 24-hour day; any
  ``H >= 1`` works, including ``H = 1``.
* The same ``S+``/``S-`` pair is later reused as an *interpretable relevance
  descriptor* by the deployment safety layer, which compares the current Shapes
  with the Shapes of past honest deliveries.  That is the only reason a
  distribution rather than a raw profile is stored in the evidence bank.

There is no context argument: the Shape head takes only the horizon-local states
of its own branch.  The privileged semantic-context view and the retrieval-context
interface were both removed, and neither is present here under another name.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .contracts import ShapeOutput

__all__ = ["ShapeBranch", "masked_softmax"]


def masked_softmax(logits: torch.Tensor, valid_mask: Optional[torch.Tensor],
                   dim: int = 1) -> torch.Tensor:
    """Softmax over ``dim`` restricted to valid positions.

    Invalid positions get exactly zero probability, so the distribution is a
    simplex over the valid horizon.  A row with no valid position at all returns
    zeros rather than a uniform distribution over positions the caller declared
    absent.  No ``-inf`` is ever produced, so no ``NaN`` can appear.
    """
    if valid_mask is None:
        return torch.softmax(logits, dim=dim)
    keep = torch.as_tensor(valid_mask) > 0
    target = logits.shape[:-1]
    if keep.shape != target:
        # A bare ``(H,)`` mask applies to every row; anything that does not
        # broadcast is a caller error and is reported as one rather than surfacing
        # as an opaque shape mismatch inside ``softmax``.
        try:
            keep = torch.broadcast_to(keep, target)
        except RuntimeError as exc:
            raise ValueError(
                f"valid_mask of shape {tuple(keep.shape)} does not broadcast to "
                f"the batch shape {tuple(target)}"
            ) from exc
    keep = keep.unsqueeze(-1)
    anything_valid = keep.any(dim=dim, keepdim=True)
    effective = torch.where(anything_valid, keep, torch.ones_like(keep))
    floor = torch.finfo(logits.dtype).min
    shifted = logits.masked_fill(~effective, floor)
    probs = torch.softmax(shifted, dim=dim)
    probs = probs * effective.to(logits.dtype)
    return probs * anything_valid.to(logits.dtype)


class ShapeBranch(nn.Module):
    """Per-step logits for both signs, normalised over the horizon."""

    def __init__(self, d_input: int, hidden: int = 32, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(d_input, hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, steps: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> ShapeOutput:
        """``steps`` is ``(B, H, d_input)``; ``valid_mask`` is ``(B, H)``."""
        logits = self.net(steps)                              # (B, H, 2)
        probs = masked_softmax(logits, valid_mask, dim=1)     # normalise over the horizon
        return ShapeOutput(positive=probs[..., 0], negative=probs[..., 1])
