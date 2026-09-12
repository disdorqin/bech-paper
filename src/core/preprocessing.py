"""Deterministic canonicalisation for the minimal signed-redistribution core.

The core accepts the *complete audited set* of target-day forecast-known numeric
covariates as one generic tensor.  Dataset adapters keep the names/provenance;
this package stays market-agnostic.

The canonical facts feed one shared MLP embedding.  Historical residuals are
deliberately split into two representations:

* Shape history removes scale (profile, ramp, historical signed Shapes);
* Amplitude history retains scale (scaled residual and its sign parts).

This fixes the old failure mode where a normalised Shape tensor was accidentally
reused as historical magnitude information.

There is no target-day privileged geometry channel and no semantic-view
construction here.  That routing was one of the four registered development
components and it was deleted by the closed adjudication; the active path reads
legal facts through one tensor with one embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .contracts import ForecastOriginBatch, RawInputs

__all__ = [
    "SHAPE_HISTORY_CHANNELS",
    "AMPLITUDE_HISTORY_CHANNELS",
    "TrainFrozenScaler",
    "masked_median",
    "within_day_robust_normalize",
    "first_difference",
    "masked_summary",
    "fit_residual_scale",
    "DeterministicFeatureBuilder",
    "base_token_dim",
]

SHAPE_HISTORY_CHANNELS = (
    "residual_profile",
    "residual_profile_ramp",
    "positive_shape",
    "negative_shape",
)

AMPLITUDE_HISTORY_CHANNELS = (
    "scaled_residual",
    "scaled_absolute_residual",
    "scaled_positive_residual",
    "scaled_negative_residual",
)

_EPS = 1e-6


def _as_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(x)
    return (mask > 0).to(x.dtype)


def masked_median(x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                  dim: int = -1) -> torch.Tensor:
    """Median over valid entries. Returns zero when a slice has no valid entry."""
    m = _as_mask(x, mask)
    counts = m.sum(dim=dim, keepdim=True)
    ordered, _ = torch.sort(x.masked_fill(m == 0, float("inf")), dim=dim)
    idx = ((counts - 1).clamp_min(0) // 2).long()
    picked = torch.gather(ordered, dim, idx)
    return torch.where(counts > 0, picked, torch.zeros_like(picked)).squeeze(dim)


def within_day_robust_normalize(x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                                eps: float = _EPS) -> torch.Tensor:
    """Robustly normalise along the last (horizon) axis.

    ``N(x) = (x - median) / (1.4826*MAD + eps)``.  The transform works for
    ``(B,H)`` as well as ``(B,F,H)`` tensors.
    """
    m = _as_mask(x, mask)
    med = masked_median(x, mask, dim=-1).unsqueeze(-1)
    mad = masked_median((x - med).abs(), mask, dim=-1).unsqueeze(-1)
    scale = (1.4826 * mad).clamp_min(eps)
    return ((x - med) / scale) * m


def first_difference(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """First difference along the last axis; first horizon step is zero."""
    d = torch.zeros_like(x)
    d[..., 1:] = x[..., 1:] - x[..., :-1]
    return d * _as_mask(x, mask)


def masked_summary(x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                   dim: int = -1) -> torch.Tensor:
    """Return mean/std/max-absolute summary over valid entries."""
    m = _as_mask(x, mask)
    n = m.sum(dim=dim).clamp_min(1.0)
    mean = (x * m).sum(dim=dim) / n
    var = (((x - mean.unsqueeze(dim)) ** 2) * m).sum(dim=dim) / n
    peak = (x.abs() * m).amax(dim=dim)
    return torch.stack([mean, var.clamp_min(0).sqrt(), peak], dim=-1)


@dataclass
class TrainFrozenScaler:
    """Per-channel robust scaler fitted on training data only.

    ``mask`` may be supplied to prevent absent roles from affecting the fitted
    statistics.  Missing entries are returned as exact zeros after
    transformation; availability itself is carried as a separate model input.
    """

    center: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None

    @property
    def fitted(self) -> bool:
        return self.center is not None and self.scale is not None

    def fit(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> "TrainFrozenScaler":
        values = torch.as_tensor(x, dtype=torch.float32)
        flat = values.reshape(-1, values.shape[-1])
        if mask is None:
            flat_mask = torch.ones_like(flat, dtype=torch.bool)
        else:
            flat_mask = torch.as_tensor(mask).reshape(-1, values.shape[-1]) > 0
        centers, scales = [], []
        for j in range(flat.shape[1]):
            v = flat[:, j][flat_mask[:, j] & torch.isfinite(flat[:, j])]
            if v.numel() == 0:
                centers.append(torch.tensor(0.0))
                scales.append(torch.tensor(1.0))
                continue
            center = v.median()
            mad = (v - center).abs().median()
            centers.append(center)
            scales.append((1.4826 * mad).clamp_min(_EPS))
        self.center = torch.stack(centers)
        self.scale = torch.stack(scales)
        return self

    def transform(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("TrainFrozenScaler must be fitted before use")
        center = self.center.to(device=x.device, dtype=x.dtype)
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        out = (x - center) / scale
        if mask is not None:
            out = out * (mask > 0).to(out.dtype)
        return out

    def state(self) -> Dict[str, list]:
        if not self.fitted:
            raise RuntimeError("TrainFrozenScaler must be fitted before state()")
        return {"center": self.center.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state(cls, state: Dict[str, list]) -> "TrainFrozenScaler":
        return cls(center=torch.tensor(state["center"], dtype=torch.float32),
                   scale=torch.tensor(state["scale"], dtype=torch.float32))


def fit_residual_scale(residual: torch.Tensor, valid_mask: Optional[torch.Tensor] = None,
                       eps: float = _EPS) -> float:
    """Training-only robust scale for Amplitude-history residuals.

    The center remains exactly zero because sign is scientifically meaningful.
    This helper only rescales units; it never changes residual direction.
    """
    r = torch.as_tensor(residual, dtype=torch.float32)
    if valid_mask is not None:
        keep = torch.as_tensor(valid_mask) > 0
        values = r[keep]
    else:
        values = r.reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 1.0
    return float((1.4826 * values.abs().median()).clamp_min(eps))


def base_token_dim(n_forecast_features: int, n_calendar_features: int = 0) -> int:
    """Width of the shared base token ``[host, features, calendar, host_mask, feature_masks]``."""
    if n_forecast_features < 0 or n_calendar_features < 0:
        raise ValueError("feature counts must be non-negative")
    return 2 + 2 * n_forecast_features + n_calendar_features


def _safe_mass_shape(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    total = x.sum(dim=-1, keepdim=True)
    return torch.where(total > eps, x / total.clamp_min(eps), torch.zeros_like(x))


class DeterministicFeatureBuilder:
    """Create the canonical batch without market-specific feature logic.

    All target-day legal numeric covariates enter the same shared base token.
    The two historical residual views are built separately because Shape needs
    scale-free geometry while Amplitude needs magnitude.

    ``residual_scale`` is a **training-only** fitted statistic.  Like every other
    scaler it is fitted on the training partition, never on an evaluation
    partition and never on protected data.
    """

    def __init__(self, base_scaler: Optional[TrainFrozenScaler] = None,
                 residual_scale: float = 1.0):
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.base_scaler = base_scaler
        self.residual_scale = float(residual_scale)

    @staticmethod
    def _feature_tensors(raw: RawInputs) -> tuple[torch.Tensor, torch.Tensor]:
        host = raw.host
        batch, horizon = host.shape
        if raw.forecast_features is None:
            features = torch.zeros(batch, horizon, 0, dtype=host.dtype, device=host.device)
            mask = torch.zeros_like(features)
            return features, mask
        features = raw.forecast_features
        if features.ndim != 3 or features.shape[:2] != host.shape:
            raise ValueError("forecast_features must have shape (B,H,F) aligned with host")
        if raw.forecast_feature_mask is None:
            mask = torch.ones_like(features)
        else:
            mask = raw.forecast_feature_mask
            if mask.shape != features.shape:
                raise ValueError("forecast_feature_mask must match forecast_features")
            mask = (mask > 0).to(features.dtype)
        return features, mask

    @staticmethod
    def _calendar(raw: RawInputs) -> torch.Tensor:
        host = raw.host
        batch, horizon = host.shape
        if raw.calendar is None:
            return torch.zeros(batch, horizon, 0, dtype=host.dtype, device=host.device)
        calendar = raw.calendar
        if calendar.ndim == 2:
            calendar = calendar.unsqueeze(-1)
        if calendar.ndim != 3 or calendar.shape[:2] != host.shape:
            raise ValueError("calendar must have shape (B,H,C) aligned with host")
        return calendar

    @staticmethod
    def build_shape_history(past_residual: torch.Tensor) -> torch.Tensor:
        r = past_residual
        profile = within_day_robust_normalize(r)
        ramp = first_difference(profile)
        pos = torch.clamp(r, min=0.0)
        neg = torch.clamp(-r, min=0.0)
        return torch.stack([profile, ramp, _safe_mass_shape(pos), _safe_mass_shape(neg)], dim=-2)

    def build_amplitude_history(self, past_residual: torch.Tensor) -> torch.Tensor:
        r = past_residual / self.residual_scale
        return torch.stack([r, r.abs(), torch.clamp(r, min=0.0),
                            torch.clamp(-r, min=0.0)], dim=-2)

    def build(self, raw: RawInputs) -> ForecastOriginBatch:
        host = raw.host
        valid = (raw.valid_mask > 0).to(host.dtype)
        if host.ndim != 2 or valid.shape != host.shape:
            raise ValueError("host and valid_mask must both have shape (B,H)")
        batch, horizon = host.shape
        features, feature_mask = self._feature_tensors(raw)
        calendar = self._calendar(raw)
        n_features = features.shape[-1]

        # Absolute numeric block. Unavailable roles are exactly zero after
        # scaling, while feature_mask tells the MLP that the zero means missing.
        numeric = torch.cat([host.unsqueeze(-1), features], dim=-1)
        numeric_mask = torch.cat([valid.unsqueeze(-1), feature_mask], dim=-1)
        numeric = numeric * numeric_mask
        if self.base_scaler is not None:
            numeric = self.base_scaler.transform(numeric, numeric_mask)

        base_tokens = torch.cat([
            numeric,
            calendar,
            valid.unsqueeze(-1),
            feature_mask,
        ], dim=-1)

        expected = base_token_dim(n_features, calendar.shape[-1])
        if base_tokens.shape[-1] != expected:
            raise RuntimeError("base token dimension contract violated")

        if raw.past_residual is None:
            shape_history = torch.zeros(batch, 0, len(SHAPE_HISTORY_CHANNELS), horizon,
                                        dtype=host.dtype, device=host.device)
            amplitude_history = torch.zeros(
                batch, 0, len(AMPLITUDE_HISTORY_CHANNELS), horizon,
                dtype=host.dtype, device=host.device)
        else:
            r = raw.past_residual
            if r.ndim != 3 or r.shape[0] != batch or r.shape[-1] != horizon:
                raise ValueError("past_residual must have shape (B,W,H) aligned with host")
            shape_history = self.build_shape_history(r)
            amplitude_history = self.build_amplitude_history(r)

        return ForecastOriginBatch(
            host=host,
            valid_mask=valid,
            base_tokens=base_tokens,
            shape_history=shape_history,
            amplitude_history=amplitude_history,
        )
