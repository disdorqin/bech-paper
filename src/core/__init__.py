"""Minimal signed-redistribution core with an analytic alignment safety layer.

The active candidate object, and the only one this package implements::

    c_h = A (S+_h - S-_h),     A >= 0,   S+-, S- in Delta^{H-1}

produced by one shared MLP stem, a Shape GRU32 and an Amplitude GRU32.  Fusion is
deterministic and the correction is exactly zero-sum over the horizon, so the
candidate is a within-day residual **redistribution** rather than a level
correction.

Deployment adds no trainable component::

    y = Host + lambda * c,     lambda = min(alpha_0, lambda_U, lambda_S)

with ``alpha_0`` a pooled out-of-fold calibration scalar and ``lambda_U`` /
``lambda_S`` exact no-harm radii read off the last seven honest prequential
records.  The controller lives in :mod:`core.safety` as pure functions.

Scope
-----
This package is a source refactor of the accepted signed-mass scaffolding.  It is
**not** a scientific promotion: nothing here is a contribution claim, no result
follows from source presence, and the closed development verdict
``HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION`` is unaffected.
The four development components the adjudication deleted -- semantic Shape-context
routing, the temporal convolution, rare-mass sampling and the untied amplitude
option -- are absent from this package by name and by construction, and the
retrieval interface is absent with them.  None of them is present behind a flag.

There is no learned selector, no trust score, no gate, no dataset path, no market
or Host name and no evaluation logic here.
"""
from .amplitude import BalancedAmplitudeBranch
from .calibration import apply_mae_scalar, fit_mae_scalar, mae_objective, weighted_median
from .contracts import (PROVENANCE_IN_SAMPLE, PROVENANCE_OOF_PREQUENTIAL,
                        PROVENANCE_PREQUENTIAL, WARM_START_PROVENANCE,
                        BranchRepresentation, CandidateOutput, DeploymentDecision,
                        ForecastOriginBatch, ModelConfig, RawInputs, SafetyEvidence,
                        ShapeOutput, SignedMassTargets)
from .encoders import WindowGRUEncoder
from .fusion import correction_mass, decompose_residual, fuse, repair, safe_shape
from .history import (HISTORY_DAYS, DishonestRecordError, DuplicateDeliveryError,
                      HistoryError, IncompleteHistoryError, InSampleCandidateError,
                      InsufficientWarmStartError, SafetyEvidenceBank, select_warm_start)
from .losses import (amplitude_l1, balanced_amplitude_loss, combined_loss,
                     fit_mass_normalization, repair_mae, shape_l1, shape_loss,
                     shape_wasserstein1)
from .model import MinimalSignedRedistributionModel, default_config, plan_deployment
from .preprocessing import (AMPLITUDE_HISTORY_CHANNELS, SHAPE_HISTORY_CHANNELS,
                            DeterministicFeatureBuilder, TrainFrozenScaler,
                            base_token_dim, first_difference, fit_residual_scale,
                            masked_median, masked_summary, within_day_robust_normalize)
from .safety import (deployment_scale, exact_mae_safe_radius, mae_alignment_support,
                     mae_excess_risk, shape_relevance, shape_weighted_safe_radius,
                     wasserstein1)
from .shape import ShapeBranch, masked_softmax
from .stem import UnifiedFeatureStem

__all__ = [
    # contracts
    "RawInputs", "ForecastOriginBatch", "BranchRepresentation", "ShapeOutput",
    "CandidateOutput", "SignedMassTargets", "SafetyEvidence", "DeploymentDecision",
    "ModelConfig",
    "PROVENANCE_OOF_PREQUENTIAL", "PROVENANCE_PREQUENTIAL", "PROVENANCE_IN_SAMPLE",
    "WARM_START_PROVENANCE",
    # deterministic coordinates and the one shared stem
    "DeterministicFeatureBuilder", "TrainFrozenScaler", "base_token_dim",
    "fit_residual_scale", "masked_median", "masked_summary",
    "within_day_robust_normalize", "first_difference",
    "SHAPE_HISTORY_CHANNELS", "AMPLITUDE_HISTORY_CHANNELS", "UnifiedFeatureStem",
    # branches and fusion
    "WindowGRUEncoder", "ShapeBranch", "masked_softmax", "BalancedAmplitudeBranch",
    "safe_shape", "decompose_residual", "fuse", "correction_mass", "repair",
    # candidate generator
    "MinimalSignedRedistributionModel", "default_config", "plan_deployment",
    # pooled calibration
    "weighted_median", "fit_mae_scalar", "apply_mae_scalar", "mae_objective",
    # auxiliary objectives
    "repair_mae", "shape_wasserstein1", "shape_l1", "shape_loss",
    "amplitude_l1", "balanced_amplitude_loss", "fit_mass_normalization",
    "combined_loss",
    # analytic safety layer
    "wasserstein1", "shape_relevance", "mae_excess_risk", "mae_alignment_support",
    "exact_mae_safe_radius", "shape_weighted_safe_radius", "deployment_scale",
    # seven-record honest prequential bank
    "HISTORY_DAYS", "SafetyEvidenceBank", "select_warm_start",
    "HistoryError", "DuplicateDeliveryError", "DishonestRecordError",
    "InSampleCandidateError", "IncompleteHistoryError", "InsufficientWarmStartError",
]
