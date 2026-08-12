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
def compute_domain_descriptors(s1_z0: np.ndarray,
                               s1_hours: np.ndarray | None = None) -> np.ndarray:
    """Estimate DOMAIN-level stable descriptors from S1 host z0 (frozen).

    These describe the stable geometry of the series (distribution/dynamics),
    computed once over the entire S1 host prediction pool — NOT per-day.
    Only S1 host predictions (pre-outcome) are used; no target, no S2/S3/S4.

    Returns [d_det] = 8: q25/q50/q75/iqr/mean_abs/flips/pos_mass/lag1.
    """
    z = np.asarray(s1_z0, dtype=np.float64).ravel()
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return np.zeros(8, dtype=np.float64)

    q25 = float(np.quantile(z, 0.25))
    q50 = float(np.quantile(z, 0.50))
    q75 = float(np.quantile(z, 0.75))
    iqr = q75 - q25
    mean_abs = float(np.mean(np.abs(z)))

    signs = (z > 0).astype(np.float64)
    flips = float(np.mean(signs[1:] != signs[:-1])) if len(z) > 1 else 0.0
    pos_mass = float(np.mean(z > 0))

    zc = z - z.mean()
    ss = float((zc ** 2).sum())
    lag1 = float((zc[:-1] * zc[1:]).sum() / ss) if (len(z) > 1 and ss > 1e-8) else 0.0

    return np.array([q25, q50, q75, iqr, mean_abs, flips, pos_mass, lag1],
                    dtype=np.float64)


class DataSignature(nn.Module):
    """Lightweight data-conditioned interface (P1-7).

    Deterministic descriptors are DOMAIN-level, estimated once on S1 and frozen
    in `domain_det` buffer. The learned part (Pool(E_core)) remains per-day.
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
        # frozen domain descriptors (S1), zero until set_domain_descriptors
        self.register_buffer("domain_det", torch.zeros(d_det))

    def set_domain_descriptors(self, det: np.ndarray):
        """Write S1-estimated domain descriptors into the frozen buffer."""
        det = np.asarray(det, dtype=np.float64).reshape(-1)
        assert det.shape[0] == self.d_det, \
            f"descriptor dim mismatch: {det.shape[0]} != {self.d_det}"
        self.domain_det.copy_(torch.tensor(det, dtype=torch.float32))

    def compute_deterministic(self, z0=None, avail_mask=None) -> torch.Tensor:
        """Return the FROZEN domain descriptors (no per-day quantile)."""
        return self.domain_det

    def forward(self, core_hidden: torch.Tensor) -> tuple:
        """Emit FiLM modulation (gamma, beta) for the core hidden states.

        sig = cat([domain_det (frozen S1), learned_proj(core_hidden.mean) (per-day)]).
        """
        B, H, d = core_hidden.shape
        det = self.domain_det.unsqueeze(0).expand(B, -1)      # [B, d_det] frozen
        learned = self.learned_proj(core_hidden.mean(dim=1))  # [B, d_sig] per-day
        sig = torch.cat([det, learned], dim=1)                # [B, d_det + d_sig]
        mod = self.mod_head(sig)                              # [B, 2*d_core]
        gamma = mod[:, :d].unsqueeze(1)                       # [B, 1, d]
        beta = mod[:, d:].unsqueeze(1)                        # [B, 1, d]
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
        gamma, beta = self.signature(h)    # domain_det (frozen) + learned (per-day)
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
