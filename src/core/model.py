"""Thin orchestration of the minimal signed-redistribution candidate.

Data flow, complete::

    legal target-day facts
      -> deterministic coordinates                    (preprocessing)
      -> ONE shared MLP stem                          (stem)
      -> Shape branch:    GRU32 over Shape history    -> S+, S-
      -> Amplitude branch: GRU32 over magnitude hist  -> one scalar A >= 0
      -> c = A (S+ - S-)                              (fusion, deterministic)
      -> y = Host + lambda c                          (deployment safety layer)

The trainable module is the first five lines.  ``lambda`` is not produced here:
it is computed by :mod:`core.safety` from the seven-record bank and never enters
the gradient graph.  ``alpha_0`` is a pooled scalar fitted on an out-of-fold
calibration partition, also outside the graph.

Where the safety bank lives
---------------------------
The bank is **not** a buffer, a parameter, a module attribute or part of
``state_dict``.  It is passed to :func:`plan_deployment` as an argument.  A safety
history stored inside the model would be trained, checkpointed and silently
restored with the weights, which would make the controller's evidence a function
of which checkpoint happened to be loaded; keeping it outside makes the evidence a
function of what actually happened.  Nothing in this class remembers a past day.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .amplitude import BalancedAmplitudeBranch
from .contracts import (CandidateOutput, DeploymentDecision, ForecastOriginBatch,
                        ModelConfig)
from .encoders import WindowGRUEncoder
from .fusion import fuse, repair
from .history import SafetyEvidenceBank
from .preprocessing import (AMPLITUDE_HISTORY_CHANNELS, SHAPE_HISTORY_CHANNELS,
                            base_token_dim)
from .safety import (deployment_scale, exact_mae_safe_radius, mae_alignment_support,
                     shape_relevance, shape_weighted_safe_radius)
from .shape import ShapeBranch
from .stem import UnifiedFeatureStem

__all__ = ["MinimalSignedRedistributionModel", "default_config", "plan_deployment"]


def default_config(n_forecast_features: int = 0, n_calendar_features: int = 0,
                   **overrides) -> ModelConfig:
    """Build the dataset-agnostic reference configuration for a legal schema.

    Different markets expose different counts of audited forecast-known features;
    only input dimensions change.  Hidden widths, the RNN family and the training
    protocol are common, and there is no switch to vary them: the surviving
    generator is a single recipe.
    """
    settings = dict(
        d_base_tokens=base_token_dim(n_forecast_features, n_calendar_features),
        n_shape_history_channels=len(SHAPE_HISTORY_CHANNELS),
        n_amplitude_history_channels=len(AMPLITUDE_HISTORY_CHANNELS),
    )
    settings.update(overrides)
    return ModelConfig(**settings)


class MinimalSignedRedistributionModel(nn.Module):
    """Shape and Amplitude branches with deterministic zero-sum fusion.

    The module produces ``S+``, ``S-``, one nonnegative ``A`` and the correction
    ``c``.  It does not produce a level term, a daily bias, a trust score, a
    scale, or a second amplitude head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.stem = UnifiedFeatureStem(config.d_base_tokens, config.stem_hidden,
                                       config.stem_out, config.dropout)

        self.shape_encoder = WindowGRUEncoder(
            config.stem_out,
            config.n_shape_history_channels,
            d_embed=config.d_embed,
            rnn_hidden=config.rnn_hidden,
            rnn_layers=config.rnn_layers,
            d_latent=config.d_latent,
            d_step=config.d_step,
            pool="mean",
            dropout=config.dropout,
        )
        # Amplitude keeps absolute scale, so its pooled statistics carry magnitude
        # as well as level; the pooling mode follows the surviving configuration.
        self.amplitude_encoder = WindowGRUEncoder(
            config.stem_out,
            config.n_amplitude_history_channels,
            d_embed=config.d_embed,
            rnn_hidden=config.rnn_hidden,
            rnn_layers=config.rnn_layers,
            d_latent=config.d_latent,
            d_step=config.d_step,
            pool="meanmax",
            dropout=config.dropout,
        )

        # Shape sees horizon-local current-day states plus the cross-day GRU state.
        self.shape_head = ShapeBranch(config.d_step + config.d_latent,
                                      hidden=config.shape_hidden,
                                      dropout=config.dropout)
        # Amplitude reads the pooled cross-day latent and emits one scalar.
        self.amplitude_head = BalancedAmplitudeBranch(config.d_latent,
                                                      hidden=config.amplitude_hidden,
                                                      dropout=config.dropout,
                                                      scale=config.amplitude_scale)

    def forward(self, batch: ForecastOriginBatch) -> CandidateOutput:
        embedded = self.stem(batch.base_tokens)                    # (B, H, d_stem)

        shape_repr = self.shape_encoder(embedded, batch.shape_history)
        amplitude_repr = self.amplitude_encoder(embedded, batch.amplitude_history)

        horizon = shape_repr.steps.shape[1]
        history_state = shape_repr.latent.unsqueeze(1).expand(-1, horizon, -1)
        shape_steps = torch.cat([shape_repr.steps, history_state], dim=-1)
        shape = self.shape_head(shape_steps, batch.valid_mask)
        amplitude = self.amplitude_head(amplitude_repr.latent)

        return CandidateOutput(
            shape_positive=shape.positive,
            shape_negative=shape.negative,
            amplitude=amplitude,
            correction=fuse(amplitude, shape),
        )

    def predict(self, batch: ForecastOriginBatch,
                alpha: Optional[float] = None) -> torch.Tensor:
        """``Host + alpha * c``, or ``Host + c`` when ``alpha`` is omitted.

        ``alpha`` here is the pre-fitted pooled calibration scalar used for
        evaluation runs.  Live deployment goes through :func:`plan_deployment`,
        which derives its own bounded scalar from the safety bank.
        """
        return repair(batch.host, self.forward(batch).correction, alpha)


def plan_deployment(host: torch.Tensor, candidate: CandidateOutput,
                    bank: SafetyEvidenceBank, alpha_0: float,
                    valid_mask: Optional[torch.Tensor] = None) -> DeploymentDecision:
    """The whole deployment controller, as one auditable pure function.

    ``alpha_0`` is the global out-of-fold calibration scalar
    ``max(0, WM(r/c; |c|))``.  The returned scalar is

        lambda = min(alpha_0, lambda_U, lambda_S)

    where ``lambda_U`` is the exact no-harm radius of the uniform recent risk and
    ``lambda_S`` the same radius under Shape relevance weights.  Taking the
    minimum makes Shape **one-way conservative**: it can shorten the permitted ray
    when the current candidate resembles past outcomes that went badly, and it can
    never lengthen the ray beyond what the broad recent evidence already allows.

    No threshold, no grid, no gate and no learned component participates.  One
    delivery day is scored at a time, against one bank, because the bank is the
    evidence for that market and Host rather than for a batch of unrelated days.
    """
    correction = candidate.correction
    if correction.ndim != 2 or correction.shape[0] != 1:
        raise ValueError(
            "deployment scores exactly one delivery day against one evidence bank; "
            f"got a correction of shape {tuple(correction.shape)}"
        )
    evidence = bank.as_arrays()

    residual = evidence["residual"]
    historical = evidence["correction"]
    historical_mask = evidence["valid_mask"]

    relevance = shape_relevance(
        candidate.shape_positive, candidate.shape_negative,
        evidence["shape_positive"], evidence["shape_negative"],
        valid_mask=valid_mask,
    )
    lambda_uniform = exact_mae_safe_radius(residual, historical,
                                           valid_mask=historical_mask)
    lambda_shape = shape_weighted_safe_radius(residual, historical, relevance,
                                              valid_mask=historical_mask)
    lam = deployment_scale(alpha_0, lambda_uniform, lambda_shape)

    alignment_shape = (mae_alignment_support(residual, historical, weights=relevance,
                                             valid_mask=historical_mask)
                       if lambda_shape is not None else None)
    return DeploymentDecision(
        alpha_0=float(alpha_0),
        lambda_uniform=float(lambda_uniform),
        lambda_shape=None if lambda_shape is None else float(lambda_shape),
        lambda_final=float(lam),
        alignment_uniform=float(mae_alignment_support(residual, historical,
                                                      valid_mask=historical_mask)),
        alignment_shape=(None if alignment_shape is None else float(alignment_shape)),
        repaired=repair(host, correction, lam),
        n_evidence=len(evidence["delivery_ids"]),
    )
