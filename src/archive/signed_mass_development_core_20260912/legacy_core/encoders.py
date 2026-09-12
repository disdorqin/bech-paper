"""Lightweight branch encoders.

The design is deliberately plain: a local causal TCN reads multi-scale structure
*within* a day, one pooled vector summarises that day, and a single recurrent
layer reads the *sequence* of historical windows.  There is no attention, no
Transformer, and no market-specific subclass.

This module is self-contained on purpose.  The host-layer sequence blocks under
``backbones/`` are single-channel private helpers of the frozen host and are not
a reusable component of this core.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import BranchRepresentation

__all__ = ["LocalTCN", "WindowRNN", "BranchEncoder"]


class LocalTCN(nn.Module):
    """Causal multi-scale temporal convolution along the within-day axis.

    Every convolution is left-padded by ``(kernel - 1) * dilation`` so that step
    ``h`` depends only on steps ``<= h`` and the horizon length is preserved.
    """

    def __init__(self, channels: int, hidden: int = 32,
                 kernel_sizes: Sequence[int] = (3, 5),
                 dilations: Sequence[int] = (1, 2, 4, 8),
                 dropout: float = 0.1):
        super().__init__()
        self.pairs: Tuple[Tuple[int, int], ...] = tuple(
            (int(k), int(d)) for k in kernel_sizes for d in dilations)
        if not self.pairs:
            raise ValueError("LocalTCN needs at least one (kernel, dilation) pair")
        self.convs = nn.ModuleList(
            nn.Conv1d(channels, hidden, kernel_size=k, dilation=d)
            for k, d in self.pairs)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Conv1d(hidden * len(self.pairs), hidden, kernel_size=1)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H) -> (B, hidden, H)``."""
        feats = []
        for (kernel, dilation), conv in zip(self.pairs, self.convs):
            pad = (kernel - 1) * dilation
            feats.append(F.relu(conv(F.pad(x, (pad, 0)))))
        return self.out(self.drop(torch.cat(feats, dim=1)))


class WindowRNN(nn.Module):
    """Recurrent layer over the sequence of historical windows.

    ``cell`` selects GRU or LSTM.  ``bidirectional`` defaults to ``False``: the
    validated path is the unidirectional GRU, and a bidirectional ablation is a
    configuration change rather than a different class.
    """

    def __init__(self, d_in: int, hidden: int = 32, layers: int = 1, cell: str = "gru",
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        if cell not in ("gru", "lstm"):
            raise ValueError(f"cell must be 'gru' or 'lstm', got {cell!r}")
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(d_in, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0,
                           bidirectional=bidirectional)
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """``(B, W, d_in) -> (B, out_dim)`` using the last step."""
        out, _ = self.rnn(windows)
        return out[:, -1]


class BranchEncoder(nn.Module):
    """``embedding -> local TCN -> daily representation -> window RNN -> latent``.

    The Shape and Amplitude branches hold separate instances of this class with
    independent weights.  Day channels and history channels are embedded
    separately because they are different objects: current-day profiles versus
    revealed past residual windows.
    """

    def __init__(self, n_day_channels: int, n_history_channels: int,
                 d_embed: int = 16, tcn_hidden: int = 32,
                 kernel_sizes: Sequence[int] = (3, 5),
                 dilations: Sequence[int] = (1, 2, 4, 8),
                 dropout: float = 0.1, temporal_rnn: str = "gru",
                 rnn_hidden: int = 32, rnn_layers: int = 1,
                 rnn_bidirectional: bool = False,
                 d_latent: int = 32, d_step: int = 32,
                 pool: str = "mean", use_tcn: bool = True):
        super().__init__()
        if pool not in ("mean", "meanmax"):
            raise ValueError(f"pool must be 'mean' or 'meanmax', got {pool!r}")
        self.pool = pool
        self.use_tcn = bool(use_tcn)
        self.day_embed = nn.Conv1d(n_day_channels, d_embed, kernel_size=1)
        self.history_embed = nn.Conv1d(n_history_channels, d_embed, kernel_size=1)
        if self.use_tcn:
            self.local = LocalTCN(d_embed, tcn_hidden, kernel_sizes, dilations, dropout)
            local_hidden = tcn_hidden
        else:
            # Registered TCN ablation: retain only the 1x1 channel projection.
            # This removes all within-day temporal convolution while preserving
            # the same GRU/history semantics.
            self.local = nn.Identity()
            local_hidden = d_embed
        self.pooled_dim = local_hidden * (2 if pool == "meanmax" else 1)
        self.rnn = WindowRNN(self.pooled_dim, rnn_hidden, rnn_layers, temporal_rnn,
                             dropout, rnn_bidirectional)
        self.drop = nn.Dropout(dropout)
        self.latent_proj = nn.Linear(self.pooled_dim + self.rnn.out_dim, d_latent)
        self.step_proj = nn.Linear(local_hidden, d_step)
        self.d_latent = d_latent
        self.d_step = d_step

    def _pool(self, encoded: torch.Tensor) -> torch.Tensor:
        """``(B, tcn_hidden, H) -> (B, pooled_dim)``.  Mean, or ``[mean, max]``."""
        mean = encoded.mean(dim=-1)
        if self.pool == "mean":
            return mean
        return torch.cat([mean, encoded.amax(dim=-1)], dim=-1)

    def forward(self, day: torch.Tensor,
                history: torch.Tensor) -> BranchRepresentation:
        """``day`` is ``(B, C, H)``; ``history`` is ``(B, W, C_h, H)``."""
        encoded = self.local(self.day_embed(day))            # (B, local_hidden, H)
        day_repr = self._pool(encoded)

        if history is not None and history.shape[1] > 0:
            batch, windows, channels, horizon = history.shape
            flat = history.reshape(batch * windows, channels, horizon)
            per_window = self._pool(self.local(self.history_embed(flat)))
            window_repr = self.rnn(per_window.reshape(batch, windows, -1))
        else:
            window_repr = torch.zeros(day.shape[0], self.rnn.out_dim,
                                      dtype=day.dtype, device=day.device)

        latent = self.latent_proj(
            self.drop(torch.cat([day_repr, window_repr], dim=-1)))
        steps = self.step_proj(encoded.transpose(1, 2))       # (B, H, d_step)
        return BranchRepresentation(latent=latent, steps=steps)
