"""Thin orchestration of the signed-mass repair model.

Current data flow:

    complete legal target-day facts
      -> deterministic coordinates
      -> ONE shared MLP stem
      -> semantic routing
           -> Shape: shared embedding + scale-free geometry
              -> small TCN -> GRU32 over Shape history -> S+, S-
           -> Amplitude: shared absolute embedding
              -> small TCN -> GRU32 over magnitude history -> A+, A-
      -> c = A+ S+ - A- S-

There is no learned proposal selector, no hidden-state Bridge and no learned fusion.  The
pooled non-negative MAE scalar alpha is fitted outside the gradient graph.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .amplitude import AmplitudeBranch
from .contracts import ForecastOriginBatch, ModelConfig, RepairOutput
from .encoders import BranchEncoder
from .fusion import fuse, repair
from .preprocessing import (AMPLITUDE_HISTORY_CHANNELS, SHAPE_HISTORY_CHANNELS,
                            base_token_dim, shape_channel_dim)
from .semantic_views import SemanticRouter
from .shape import ShapeBranch
from .stem import UnifiedFeatureStem

__all__ = ["SignedMassRepairModel", "default_config"]


def default_config(n_forecast_features: int = 0,
                   n_calendar_features: int = 0, **overrides) -> ModelConfig:
    """Build a small dataset-agnostic configuration for a given legal schema.

    Different markets may expose different counts of audited forecast-known
    features; only input dimensions change.  The architecture recipe, hidden
    widths and training protocol remain common.
    """
    settings = dict(
        d_base_tokens=base_token_dim(n_forecast_features, n_calendar_features),
        n_shape_channels=shape_channel_dim(n_forecast_features, n_calendar_features),
        n_shape_history_channels=len(SHAPE_HISTORY_CHANNELS),
        n_amplitude_history_channels=len(AMPLITUDE_HISTORY_CHANNELS),
    )
    settings.update(overrides)
    return ModelConfig(**settings)


class SignedMassRepairModel(nn.Module):
    """Minimal Shape/Amplitude executor with deterministic signed-mass fusion."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.stem = UnifiedFeatureStem(config.d_base_tokens, config.stem_hidden,
                                       config.stem_out, config.dropout)
        self.router = SemanticRouter()

        encoder_kwargs = dict(
            d_embed=config.d_embed,
            tcn_hidden=config.tcn_hidden,
            kernel_sizes=config.tcn_kernel_sizes,
            dilations=config.tcn_dilations,
            dropout=config.dropout,
            temporal_rnn=config.temporal_rnn,
            rnn_hidden=config.rnn_hidden,
            rnn_layers=config.rnn_layers,
            rnn_bidirectional=config.rnn_bidirectional,
            d_latent=config.d_latent,
            d_step=config.d_step,
            use_tcn=config.use_tcn,
        )
        self.shape_encoder = BranchEncoder(
            config.stem_out + config.n_shape_channels,
            config.n_shape_history_channels,
            **encoder_kwargs,
        )
        self.amplitude_encoder = BranchEncoder(
            config.stem_out,
            config.n_amplitude_history_channels,
            pool="meanmax",
            **encoder_kwargs,
        )

        # Shape keeps horizon-local current-day TCN features and combines them
        # with the cross-day GRU state at the decoder.
        self.shape_head = ShapeBranch(
            config.d_step + config.d_latent,
            hidden=config.d_latent,
            dropout=config.dropout,
            d_context=config.shape_context_dim,
        )
        # Amplitude uses one common trunk and two untied non-negative heads.
        self.amplitude_head = AmplitudeBranch(
            config.d_latent,
            hidden=config.amplitude_hidden,
            dropout=config.dropout,
            scale=config.amplitude_scale,
            untied_heads=config.untied_amplitude_heads,
        )

    def forward(self, batch: ForecastOriginBatch,
                context: Optional[torch.Tensor] = None) -> RepairOutput:
        embedded = self.stem(batch.base_tokens)  # (B,H,d_e)
        shape_channels = (batch.shape_channels if self.config.shape_semantic_context
                          else torch.zeros_like(batch.shape_channels))
        views = self.router(embedded, shape_channels)

        shape_repr = self.shape_encoder(views.shape, batch.shape_history)
        amplitude_repr = self.amplitude_encoder(views.amplitude, batch.amplitude_history)

        horizon = shape_repr.steps.shape[1]
        history_state = shape_repr.latent.unsqueeze(1).expand(-1, horizon, -1)
        shape_steps = torch.cat([shape_repr.steps, history_state], dim=-1)
        shape_context = batch.shape_context if context is None else context
        shape = self.shape_head(shape_steps, shape_context)
        amplitude = self.amplitude_head(amplitude_repr.latent)

        return RepairOutput(
            correction=fuse(amplitude, shape),
            shape=shape,
            amplitude=amplitude,
        )

    def predict(self, batch: ForecastOriginBatch, alpha: Optional[float] = None,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return ``Host + alpha*c`` (or ``Host+c`` when alpha is omitted)."""
        output = self.forward(batch, context=context)
        return repair(batch.host, output.correction, alpha)
