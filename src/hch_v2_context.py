"""HCH v0.4 context: CoreContextEncoder + DataSignature + Optional branch.

Architecture: v0.4 universal adaptive architecture §2-§4.

Design invariant (P1-5):
    optional disabled  =>  pure core path  (h_final = h_core)

Core input (scale-free, host-relative):
    z0      : host-anchored hyperbolic coordinate
    u       : local S1 continuous rank
    time    : cyclic calendar features
    lag_sf  : scale-free lag/history channels

Data Signature (P1-7): deterministic scale-free descriptors + learned pool.
    NOT a domain classifier; NO domain-classification loss.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================ Data Signature =================================
class DataSignature(nn.Module):
    """Lightweight data-conditioned interface (P1-7).

    Combines stable human-designed geometry (deterministic descriptors) with a
    small learned pooled representation, then emits FiLM modulation (gamma, beta).
    """

    def __init__(self, d_core: int, d_det: int, d_sig: int = 32):
        super().__init__()
        self.d_core = d_core
        self.d_det = d_det
        self.learned_proj = nn.Sequential(
            nn.Linear(d_core, d_sig),
            nn.ReLU(),
        )
        self.mod_head = nn.Linear(d_det + d_sig, 2 * d_core)  # gamma + beta

    def compute_deterministic(self, z0: torch.Tensor,
                              avail_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Deterministic scale-free distribution/dynamics descriptors per day.

        Args:
            z0: [B, H] hyperbolic coordinates
        Returns:
            [B, d_det] deterministic descriptor vector
        """
        B, H = z0.shape
        # Distribution descriptors (scale-free)
        q25 = torch.quantile(z0, 0.25, dim=1)
        q50 = torch.quantile(z0, 0.50, dim=1)
        q75 = torch.quantile(z0, 0.75, dim=1)
        iqr = q75 - q25
        mean_abs = z0.abs().mean(dim=1)
        # zero-crossing rate (sign flips)
        signs = (z0 > 0).float()
        flips = (signs[:, 1:] != signs[:, :-1]).float().sum(dim=1) / max(H - 1, 1)
        # tail asymmetry proxy: mean of positive vs negative mass
        pos_mass = (z0 > 0).float().mean(dim=1)
        # dynamics: lag-1 autocorrelation
        zc = z0 - z0.mean(dim=1, keepdim=True)
        lag1 = (zc[:, :-1] * zc[:, 1:]).sum(dim=1) / (
            (zc ** 2).sum(dim=1).clamp(min=1e-8))
        det = torch.stack([q25, q50, q75, iqr, mean_abs, flips, pos_mass, lag1],
                          dim=1)  # [B, 8]
        det = torch.nan_to_num(det, nan=0.0, posinf=0.0, neginf=0.0)
        return det

    def forward(self, core_hidden: torch.Tensor, z0: torch.Tensor) -> tuple:
        """Emit FiLM modulation (gamma, beta) for the core hidden states.

        Args:
            core_hidden: [B, H, d_core] encoded core representation
            z0: [B, H] hyperbolic coordinates (for deterministic descriptors)
        Returns:
            (gamma [B,H,d_core], beta [B,H,d_core])
        """
        B, H, d = core_hidden.shape
        det = self.compute_deterministic(z0)  # [B, d_det]
        learned = self.learned_proj(core_hidden.mean(dim=1))  # [B, d_sig]
        sig = torch.cat([det, learned], dim=1)  # [B, d_det + d_sig]
        mod = self.mod_head(sig)  # [B, 2*d_core]
        gamma = mod[:, :d].unsqueeze(1)  # [B, 1, d]
        beta = mod[:, d:].unsqueeze(1)   # [B, 1, d]
        return gamma, beta


# ============================ Core Context Encoder ===========================
class CoreContextEncoder(nn.Module):
    """Consumes scale-free core input (P1-4), NOT dataset z-score prices.

    core_input convention: dimension 0 is the host hyperbolic coordinate z0.
    DataSignature deterministic descriptors are computed from this z0 channel.
    """

    def __init__(self, d_core_in: int, d_model: int = 64, d_sig: int = 32):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_core_in, d_model)
        self.signature = DataSignature(d_model, d_det=8, d_sig=d_sig)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, core_input: torch.Tensor) -> torch.Tensor:
        """core_input: [B, H, d_core_in], dimension 0 = z0.
        Returns: [B, H, d_model] FiLM-modulated core representation.
        """
        h = self.proj(core_input)          # [B, H, d_model]
        z0 = core_input[..., 0]            # extract z0 for descriptors
        gamma, beta = self.signature(h, z0)
        h = gamma * h + beta               # FiLM modulation (P1-7)
        return self.norm(h)


# ============================ Optional Covariate Encoder =====================
class OptionalCovariateEncoder(nn.Module):
    """Optional residual branch (P1-5, §4.4).

    Homogenizes covariates by role into a token space, pools/attends into
    h_optional. Zero-initialized residual so HCH-Rich ≈ HCH-Core at init.
    """

    ROLES = {"KNOWN_FUTURE": 0, "OBSERVED_PAST": 1, "STATIC": 2,
             "CALENDAR": 3, "OTHER": 4}

    def __init__(self, d_value: int, d_model: int, n_roles: int = 5):
        super().__init__()
        self.d_model = d_model
        self.value_proj = nn.Linear(d_value, d_model)
        self.role_embed = nn.Embedding(n_roles, d_model)
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # zero-init residual so optional branch starts at identity
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, values: torch.Tensor, roles: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
        """values: [B, H, N, d_value], roles: [B, H, N] role ids,
        masks: [B, H, N] 1=valid.
        Returns: [B, H, d_model] optional residual contribution.
        """
        B, H, N, _ = values.shape
        v = self.value_proj(values)  # [B,H,N,d_model]
        r = self.role_embed(roles.long())  # [B,H,N,d_model]
        e = v + r  # [B,H,N,d_model]

        mask = masks.unsqueeze(-1)  # [B,H,N,1]
        e = e * mask
        # valid-count-weighted mean pooling over covariates
        n_valid = mask.sum(dim=2).clamp(min=1)  # [B,H,1]
        pooled = e.sum(dim=2) / n_valid  # [B,H,d_model]
        return self.out(pooled)  # [B,H,d_model]
