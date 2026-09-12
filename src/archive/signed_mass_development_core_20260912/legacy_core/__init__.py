"""Signed-mass extreme-price post-processing core.

Current candidate object:

    r_h = A+ S+_h - A- S-_h
    c_h = A+_hat S+_hat_h - A-_hat S-_hat_h

The package contains the current user-authorized method scaffolding only.  It is
not yet scientifically promoted: no result follows from source presence alone.
There is no learned proposal selector, no Bridge and no learned fusion operator.
"""
from .amplitude import AmplitudeBranch
from .calibration import apply_mae_scalar, fit_mae_scalar, mae_objective, weighted_median
from .contracts import (AmplitudeOutput, BranchRepresentation, ForecastOriginBatch,
                        ModelConfig, RawInputs, RepairOutput, SemanticViews,
                        ShapeOutput, SignedMassTargets)
from .encoders import BranchEncoder, LocalTCN, WindowRNN
from .fusion import decompose_residual, fuse, repair, safe_shape
from .losses import (amplitude_l1, amplitude_loss, combined_loss,
                     fit_mass_normalization, repair_mae, shape_l1, shape_loss,
                     shape_wasserstein1)
from .model import SignedMassRepairModel, default_config
from .preprocessing import (AMPLITUDE_HISTORY_CHANNELS, SHAPE_HISTORY_CHANNELS,
                            DeterministicFeatureBuilder, TrainFrozenScaler,
                            base_token_dim, first_difference, fit_residual_scale,
                            masked_median, masked_summary, shape_channel_dim,
                            within_day_robust_normalize)
from .sampling import RareMassSampler, RareMassSamplerConfig
from .semantic_views import SemanticRouter
from .shape import ShapeBranch
from .similarity import KNNShapeContext, KNNShapeContextProvider, ShapeMemoryBank
from .stem import UnifiedFeatureStem

__all__ = [
    "RawInputs", "ForecastOriginBatch", "SemanticViews", "BranchRepresentation",
    "ShapeOutput", "AmplitudeOutput", "SignedMassTargets", "RepairOutput",
    "ModelConfig",
    "DeterministicFeatureBuilder", "TrainFrozenScaler", "base_token_dim",
    "shape_channel_dim", "fit_residual_scale", "masked_median", "masked_summary",
    "within_day_robust_normalize", "first_difference",
    "SHAPE_HISTORY_CHANNELS", "AMPLITUDE_HISTORY_CHANNELS",
    "UnifiedFeatureStem", "SemanticRouter", "LocalTCN", "WindowRNN",
    "BranchEncoder", "ShapeBranch", "AmplitudeBranch", "SignedMassRepairModel",
    "default_config",
    "ShapeMemoryBank", "KNNShapeContext", "KNNShapeContextProvider",
    "safe_shape", "decompose_residual", "fuse", "repair",
    "weighted_median", "fit_mae_scalar", "apply_mae_scalar", "mae_objective",
    "repair_mae", "shape_wasserstein1", "shape_l1", "shape_loss",
    "amplitude_l1", "amplitude_loss", "fit_mass_normalization", "combined_loss",
    "RareMassSampler", "RareMassSamplerConfig",
]
