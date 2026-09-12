"""Typed data contracts for the signed-mass repair core.

Pure dataclasses only. No dataset paths, no market or Host names, no experiment
registry and no evaluation logic. Dataset-specific adapters are responsible for
mapping all legally available forecast-origin covariates into the generic
``forecast_features`` tensor while preserving provenance outside this package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch

__all__ = [
    "RawInputs",
    "ForecastOriginBatch",
    "SemanticViews",
    "BranchRepresentation",
    "ShapeOutput",
    "AmplitudeOutput",
    "SignedMassTargets",
    "RepairOutput",
    "ModelConfig",
]


@dataclass(frozen=True)
class RawInputs:
    """Legal per-origin information before any learned transform.

    ``host`` / ``valid_mask`` are ``(B, H)``.
    ``forecast_features`` is the complete audited target-day forecast-known
    covariate tensor ``(B, H, F)``.  The core does not attach market-specific
    meaning to columns; the experiment adapter owns names/provenance.
    ``forecast_feature_mask`` has the same shape and distinguishes a genuinely
    observed zero from an unavailable role.  If omitted while features are
    provided, every feature entry is treated as available.

    ``calendar`` is ``(B, H, C)`` and contains only deterministic information
    known at issue time.  ``past_residual`` is ``(B, W, H)`` and must contain
    already-revealed residual windows formed from original-origin Host forecasts.
    """

    host: torch.Tensor
    valid_mask: torch.Tensor
    forecast_features: Optional[torch.Tensor] = None
    forecast_feature_mask: Optional[torch.Tensor] = None
    calendar: Optional[torch.Tensor] = None
    past_residual: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ForecastOriginBatch:
    """Deterministically canonicalised batch handed to the learned model."""

    host: torch.Tensor                 # (B, H)
    valid_mask: torch.Tensor           # (B, H)
    base_tokens: torch.Tensor          # (B, H, D_base), absolute/shared coordinates
    shape_channels: torch.Tensor       # (B, C_s, H), privileged scale-free geometry
    shape_history: torch.Tensor        # (B, W, C_hs, H), scale-free residual geometry
    amplitude_history: torch.Tensor    # (B, W, C_ha, H), scale-sensitive residual history
    shape_context: Optional[torch.Tensor] = None  # optional evidence-gated context


@dataclass(frozen=True)
class SemanticViews:
    """Branch-specific current-day series after the shared stem.

    Amplitude receives the shared absolute embedding itself. Shape receives that
    same factual embedding plus scale-free geometric coordinates. This is a
    bias, not an information firewall.
    """

    shape: torch.Tensor
    amplitude: torch.Tensor


@dataclass(frozen=True)
class BranchRepresentation:
    latent: torch.Tensor   # (B, d_latent)
    steps: torch.Tensor    # (B, H, d_step)


@dataclass(frozen=True)
class ShapeOutput:
    positive: torch.Tensor   # (B, H), >= 0, sums to 1
    negative: torch.Tensor   # (B, H), >= 0, sums to 1


@dataclass(frozen=True)
class AmplitudeOutput:
    positive: torch.Tensor   # (B,), >= 0
    negative: torch.Tensor   # (B,), >= 0


@dataclass(frozen=True)
class SignedMassTargets:
    mass_positive: torch.Tensor
    mass_negative: torch.Tensor
    shape_positive: torch.Tensor
    shape_negative: torch.Tensor


@dataclass(frozen=True)
class RepairOutput:
    correction: torch.Tensor
    shape: ShapeOutput
    amplitude: AmplitudeOutput
    context: Optional[Mapping[str, Any]] = None


@dataclass
class ModelConfig:
    """Architecture configuration.

    Dimensions are engineering settings, not contribution claims. Defaults are
    deliberately small.  The same recipe may be instantiated with different
    feature counts because different markets expose different legal covariates;
    that is a data-adapter difference, not a market-specific architecture.
    """

    d_base_tokens: int
    n_shape_channels: int
    n_shape_history_channels: int = 4
    n_amplitude_history_channels: int = 4

    stem_hidden: int = 64
    stem_out: int = 32
    tcn_hidden: int = 32
    # Keep the first candidate genuinely small. A larger multi-kernel TCN is not
    # justified before the TCN-vs-GRU ablation.
    tcn_kernel_sizes: Tuple[int, ...] = (3,)
    tcn_dilations: Tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1

    # Four preregistered removable hypotheses. These flags exist for controlled
    # leave-one-out ablations only, not architecture search.
    shape_semantic_context: bool = True
    use_tcn: bool = True
    untied_amplitude_heads: bool = True

    temporal_rnn: str = "gru"
    rnn_hidden: int = 32
    rnn_layers: int = 1
    rnn_bidirectional: bool = False

    d_embed: int = 16
    d_latent: int = 32
    d_step: int = 32
    amplitude_hidden: int = 32
    amplitude_scale: float = 1.0

    history_windows: int = 7

    # Optional similar-window context; interface only, disabled in the first
    # method experiment.
    knn_enabled: bool = False
    shape_context_dim: int = 0
    knn_k: int = 8
    knn_temperature: float = 1.0

    # Training-only rare-mass batch ordering. Disabled in the structural base
    # run and tested as one of the four preregistered removable components.
    rare_mass_sampling: bool = False

    def __post_init__(self) -> None:
        if self.temporal_rnn not in ("gru", "lstm"):
            raise ValueError(
                f"temporal_rnn must be 'gru' or 'lstm', got {self.temporal_rnn!r}")
        if self.d_base_tokens < 2:
            raise ValueError("d_base_tokens must include at least Host and valid mask")
        if self.n_shape_channels < 1:
            raise ValueError("n_shape_channels must be positive")
        if not self.knn_enabled:
            self.shape_context_dim = 0
