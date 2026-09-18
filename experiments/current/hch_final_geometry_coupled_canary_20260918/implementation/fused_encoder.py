"""One shared temporal encoder for the final HCH method (freeze §5.1).

Exactly one learned temporal encoder exists on the active path: a single
one-layer bidirectional GRU with total output width 32.  There is no second GRU,
no Transformer, no mixer, no attention stack and no source bundle.

Bidirectionality is legal because the complete target-day Host forecast and every
admitted forecast-origin trajectory are available at repair time.

Input channels per hour (the order is part of the frozen input contract):

    0..4   frozen Host hour channels          (host_hour_channels, 5)
    5..11  calendar hour channels             (calendar_hour, 7)
    12     Sbar^+  (deterministic W=7 prior)
    13     Sbar^-  (deterministic W=7 prior)
    14     positive-mass availability
    15     negative-mass availability

The calendar width is not a free choice: ``core.calendar.hour_channels`` returns
7 channels and ``src/core`` is immutable, so the implementation conforms to it.
Optional admitted exogenous channels append after the last core index and must
default to disabled; they cannot affect the core canary.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from core.contracts import HORIZON

WIDTH = 32
HOST_HOUR_CHANNELS = 5
#: Width authority is the frozen ``core.calendar.CALENDAR_HOUR_CHANNELS``; kept
#: as a literal so this module stays importable without pulling in ``src/core``.
CALENDAR_HOUR_CHANNELS = 7
PRIOR_CHANNELS = 4
CORE_HOUR_CHANNELS = HOST_HOUR_CHANNELS + CALENDAR_HOUR_CHANNELS + PRIOR_CHANNELS  # 16
_S_PLUS_BAR = HOST_HOUR_CHANNELS + CALENDAR_HOUR_CHANNELS
_S_MINUS_BAR = _S_PLUS_BAR + 1
_AVAIL_PLUS = _S_MINUS_BAR + 1
_AVAIL_MINUS = _AVAIL_PLUS + 1


class FinalSharedEncoder(nn.Module):
    """Input affine to width 32 plus one 1-layer BiGRU(16+16)."""

    def __init__(
        self,
        in_channels: int = CORE_HOUR_CHANNELS,
        width: int = WIDTH,
        dropout: float = 0.1,
    ):
        super().__init__()
        if width % 2 != 0:
            raise ValueError("a bidirectional GRU needs an even total output width")
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.input_affine = nn.Linear(in_channels, width)
        self.gru = nn.GRU(
            input_size=width,
            hidden_size=width // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x).float()
        if x.dim() != 3 or x.shape[1] != HORIZON:
            raise ValueError(f"hour channels must be [B, {HORIZON}, C], got {tuple(x.shape)}")
        if x.shape[-1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} hour channels, got {x.shape[-1]}")
        h = torch.nn.functional.gelu(self.input_affine(x))
        u, _ = self.gru(h)
        return self.drop(self.norm(u))

    def channel_layout(self) -> dict:
        return {
            "host_hour_channels": [0, HOST_HOUR_CHANNELS],
            "calendar_hour": [HOST_HOUR_CHANNELS, HOST_HOUR_CHANNELS + CALENDAR_HOUR_CHANNELS],
            "s_plus_bar": [_S_PLUS_BAR, _S_MINUS_BAR],
            "s_minus_bar": [_S_MINUS_BAR, _AVAIL_PLUS],
            "avail_plus": [_AVAIL_PLUS, _AVAIL_MINUS],
            "avail_minus": [_AVAIL_MINUS, CORE_HOUR_CHANNELS],
            "core_total": CORE_HOUR_CHANNELS,
            "total_in_channels": self.in_channels,
        }
