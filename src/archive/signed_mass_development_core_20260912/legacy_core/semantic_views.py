"""Semantic routing after the ONE shared feature stem.

The routing is intentionally minimal.  Every legal target-day absolute fact has
already interacted inside the shared MLP.  The two branches then receive:

* Shape: the shared embedding plus privileged scale-free within-day geometry;
* Amplitude: the shared absolute embedding only.

This implements "share factual representation; specialise task context" without
feature experts, learned routing, attention, market branches, or a hidden-state
Bridge.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .contracts import SemanticViews

__all__ = ["SemanticRouter"]


class SemanticRouter(nn.Module):
    """Create branch views from shared embeddings and deterministic Shape context."""

    def forward(self, embedded: torch.Tensor,
                shape_channels: torch.Tensor) -> SemanticViews:
        """Route ``(B,H,d_e)`` + ``(B,C_s,H)`` to branch day-series tensors.

        Returns tensors in ``(B,C,H)`` format for the branch TCNs.  Amplitude
        receives no hand-crafted severity inventory: the shared absolute
        embedding retains all legal forecast-known information and its own TCN
        and GRU learn the relevant severity representation.
        """
        if embedded.ndim != 3 or shape_channels.ndim != 3:
            raise ValueError("embedded and shape_channels must both be rank-3 tensors")
        if embedded.shape[0] != shape_channels.shape[0] or embedded.shape[1] != shape_channels.shape[2]:
            raise ValueError("Shape channels must align with embedded batch and horizon")
        shared = embedded.transpose(1, 2)  # (B,d_e,H)
        shape = torch.cat([shared, shape_channels], dim=1)
        amplitude = shared
        return SemanticViews(shape=shape, amplitude=amplitude)
