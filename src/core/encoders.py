"""Branch encoders: one day embedding, one chronological GRU32 over history.

The surviving executor is deliberately plain, and it is plain *by decision*
rather than by omission:

* a 1x1 (per-channel) projection of the current-day shared embedding;
* simple day pooling where a per-day vector is needed;
* one chronological GRU32 over the revealed historical daily windows;
* a horizon-local projection of the current day for the Shape decoder;
* a latent projection that the branch head consumes.

There is no temporal convolution here.  The local multi-scale convolution was one
of the four registered development components and it was deleted by the closed
adjudication; a configurable copy of it in the promoted path would be the same
component under a new flag, so the flag does not exist either.  The encoder is
also not a generic architecture-search API: GRU is the only recurrent cell and
there is no bidirectional or LSTM branch.

The Shape and Amplitude branches hold separate instances of this class with
independent weights, because their historical tensors carry different semantics
(scale-free geometry versus magnitude).

This module is self-contained on purpose.  The host-layer sequence blocks under
``backbones/`` are single-channel private helpers of the frozen Host and are not
a reusable component of this core.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .contracts import BranchRepresentation

__all__ = ["WindowGRUEncoder"]

_POOLS = ("mean", "meanmax")


class WindowGRUEncoder(nn.Module):
    """``current-day embedding -> day vector; historical windows -> GRU32 state``.

    ``forward`` takes the shared stem embedding ``(B, H, d_in)`` and the branch's
    own historical tensor ``(B, W, d_history, H)``.  When ``W == 0`` the GRU
    state is exactly zero, so a caller without revealed history still gets a
    well-defined representation rather than a silent fallback.

    ``pool`` is ``"mean"`` for Shape and ``"meanmax"`` for Amplitude, matching the
    surviving generator's construction.  It is not a search axis.
    """

    def __init__(self, d_in: int, d_history: int, d_embed: int = 16,
                 rnn_hidden: int = 32, rnn_layers: int = 1, d_latent: int = 32,
                 d_step: int = 32, pool: str = "mean", dropout: float = 0.0):
        super().__init__()
        if pool not in _POOLS:
            raise ValueError(f"pool must be one of {_POOLS}, got {pool!r}")
        if rnn_layers < 1:
            raise ValueError("rnn_layers must be at least 1")
        self.pool = pool
        self.day_embed = nn.Conv1d(d_in, d_embed, kernel_size=1)
        self.history_embed = nn.Conv1d(d_history, d_embed, kernel_size=1)
        # The recurrent layer reads the *pooled* daily summary, so its input width is
        # the pooled width: ``mean`` gives d_embed, ``meanmax`` gives twice that.
        self.pooled_dim = d_embed * (2 if pool == "meanmax" else 1)
        self.rnn = nn.GRU(self.pooled_dim, rnn_hidden, num_layers=rnn_layers,
                          batch_first=True,
                          dropout=dropout if rnn_layers > 1 else 0.0)
        self.latent_proj = nn.Linear(self.pooled_dim + rnn_hidden, d_latent)
        self.step_proj = nn.Linear(d_embed, d_step)
        self.drop = nn.Dropout(dropout)
        self.d_latent = d_latent
        self.d_step = d_step
        self.rnn_hidden = rnn_hidden

    def _pool(self, encoded: torch.Tensor) -> torch.Tensor:
        """``(N, d_embed, H) -> (N, pooled_dim)``.  Mean, or ``[mean, max]``."""
        mean = encoded.mean(dim=-1)
        if self.pool == "mean":
            return mean
        return torch.cat([mean, encoded.amax(dim=-1)], dim=-1)

    def forward(self, day: torch.Tensor,
                history: torch.Tensor) -> BranchRepresentation:
        """``day`` is ``(B, H, d_in)``; ``history`` is ``(B, W, d_history, H)``."""
        encoded = self.day_embed(day.transpose(1, 2))        # (B, d_embed, H)
        day_repr = self._pool(encoded)

        if history is not None and history.shape[1] > 0:
            batch, windows, channels, horizon = history.shape
            flat = history.reshape(batch * windows, channels, horizon)
            per_window = self._pool(self.history_embed(flat))
            window_repr = self.rnn(per_window.reshape(batch, windows, -1))[0][:, -1]
        else:
            window_repr = torch.zeros(day.shape[0], self.rnn_hidden,
                                      dtype=day.dtype, device=day.device)

        latent = self.latent_proj(
            self.drop(torch.cat([day_repr, window_repr], dim=-1)))
        steps = self.step_proj(encoded.transpose(1, 2))       # (B, H, d_step)
        return BranchRepresentation(latent=latent, steps=steps)
