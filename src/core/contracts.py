"""Typed data contracts for the minimal signed-redistribution repair core.

Pure dataclasses and constants only.  No dataset paths, no market or Host names,
no experiment registry and no evaluation logic.  Dataset-specific adapters are
responsible for mapping all legally available forecast-origin covariates into the
generic ``forecast_features`` tensor while preserving provenance outside this
package.

The candidate object is the one that survived the closed signed-mass development
experiment (``HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION``):

    c_h = A * (S+_h - S-_h),      A >= 0,   S+-, S-  in  Delta^{H-1}

One nonnegative Amplitude scalar, two horizon Shapes, one deterministic fusion.
There is no shape-context tensor, no TCN, no untied ``A+/A-`` pair and no
retrieval interface in these contracts, because none of those is part of the
active design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch

__all__ = [
    "RawInputs",
    "ForecastOriginBatch",
    "BranchRepresentation",
    "ShapeOutput",
    "CandidateOutput",
    "SignedMassTargets",
    "SafetyEvidence",
    "DeploymentDecision",
    "ModelConfig",
    "PROVENANCE_OOF_PREQUENTIAL",
    "PROVENANCE_PREQUENTIAL",
    "PROVENANCE_IN_SAMPLE",
    "WARM_START_PROVENANCE",
]

#: Provenance marks for a persisted safety-evidence record.
#:
#: Only the first two may seed the deployment safety bank.  ``in_sample`` marks a
#: prediction produced with the target day inside the fitting data; such a record
#: is a fitted value, not evidence about deployment behaviour, and is refused.
PROVENANCE_OOF_PREQUENTIAL = "oof_prequential"
PROVENANCE_PREQUENTIAL = "prequential"
PROVENANCE_IN_SAMPLE = "in_sample"

WARM_START_PROVENANCE = frozenset({PROVENANCE_OOF_PREQUENTIAL, PROVENANCE_PREQUENTIAL})


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
    """Deterministically canonicalised batch handed to the learned candidate."""

    host: torch.Tensor                 # (B, H)
    valid_mask: torch.Tensor           # (B, H)
    base_tokens: torch.Tensor          # (B, H, D_base), absolute/shared coordinates
    shape_history: torch.Tensor        # (B, W, C_hs, H), scale-free residual geometry
    amplitude_history: torch.Tensor    # (B, W, C_ha, H), scale-sensitive residual history


@dataclass(frozen=True)
class BranchRepresentation:
    """Branch-local view of one origin after the shared stem and its own GRU."""

    latent: torch.Tensor   # (B, d_latent)
    steps: torch.Tensor    # (B, H, d_step), horizon-local current-day states


@dataclass(frozen=True)
class ShapeOutput:
    """Two horizon distributions, one per sign direction."""

    positive: torch.Tensor   # (B, H), >= 0, sums to 1 over the valid horizon
    negative: torch.Tensor   # (B, H), >= 0, sums to 1 over the valid horizon


@dataclass(frozen=True)
class CandidateOutput:
    """Everything the trainable candidate produces, and nothing else.

    ``amplitude`` is a single nonnegative scalar per sample, not a
    positive/negative pair: the surviving generator is exactly the tied-mass
    factorization ``A+ = A- = A``, and exposing two numbers would reintroduce a
    deleted degree of freedom.
    """

    shape_positive: torch.Tensor   # (B, H)
    shape_negative: torch.Tensor   # (B, H)
    amplitude: torch.Tensor        # (B,), >= 0
    correction: torch.Tensor       # (B, H), zero-sum over the horizon


@dataclass(frozen=True)
class SignedMassTargets:
    """Exact supervision targets for the auxiliary objectives."""

    mass_positive: torch.Tensor    # (B,)
    mass_negative: torch.Tensor    # (B,)
    shape_positive: torch.Tensor   # (B, H)
    shape_negative: torch.Tensor   # (B, H)


@dataclass(frozen=True)
class SafetyEvidence:
    """One persisted delivery-day record for the prequential safety bank.

    The candidate fields (``shape_positive``, ``shape_negative``,
    ``candidate_correction``) must have been generated and persisted **before**
    ``host_residual`` became available.  ``candidate_created_at`` and
    ``revealed_at`` are caller-supplied chronology indices; the bank enforces
    ``candidate_created_at < ordinal <= revealed_at``.

    ``host_residual`` is ``None`` while the delivery day is still pending.  It is
    attached only after the target is revealed, which is why the bank has a
    two-phase append rather than a single constructor call.

    ``ordinal`` orders the delivery days; it is the only chronology the safety
    mathematics uses, so no calendar convention leaks into this package.
    """

    delivery_id: str
    ordinal: int
    shape_positive: torch.Tensor
    shape_negative: torch.Tensor
    candidate_correction: torch.Tensor
    candidate_created_at: int
    provenance: str = PROVENANCE_OOF_PREQUENTIAL
    host_residual: Optional[torch.Tensor] = None
    revealed_at: Optional[int] = None
    origin_id: str = ""
    valid_mask: Optional[torch.Tensor] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.host_residual is not None


@dataclass(frozen=True)
class DeploymentDecision:
    """The analytic safety layer's output for one delivery day.

    Every field is a plain scalar so a reviewer can read the whole controller
    without opening the model.  ``lambda_final`` is the only number that changes
    the forecast, and it is bounded by ``alpha_0`` by construction.
    """

    alpha_0: float
    lambda_uniform: float
    lambda_shape: Optional[float]
    lambda_final: float
    alignment_uniform: float
    alignment_shape: Optional[float]
    repaired: torch.Tensor
    n_evidence: int = 0


@dataclass
class ModelConfig:
    """Architecture configuration for the minimal candidate generator.

    Dimensions are engineering settings, not contribution claims.  There is
    deliberately **no** TCN switch, no semantic-context switch, no untied-head
    switch, no sampling switch and no history-window setting: those decisions are
    settled and a configurable copy of a deleted component is the same component.
    """

    d_base_tokens: int
    n_shape_history_channels: int = 4
    n_amplitude_history_channels: int = 4

    stem_hidden: int = 64
    stem_out: int = 32
    dropout: float = 0.1

    d_embed: int = 16
    rnn_hidden: int = 32
    rnn_layers: int = 1
    d_latent: int = 32
    d_step: int = 32

    shape_hidden: int = 32
    amplitude_hidden: int = 32
    amplitude_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.d_base_tokens < 2:
            raise ValueError("d_base_tokens must include at least Host and valid mask")
        if self.n_shape_history_channels < 1 or self.n_amplitude_history_channels < 1:
            raise ValueError("history channel counts must be positive")
        if self.rnn_layers < 1:
            raise ValueError("rnn_layers must be at least 1")
        if self.amplitude_scale < 0:
            raise ValueError("amplitude_scale must be non-negative")
