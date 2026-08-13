"""IAH Candidate Head — v0.4 universal core assembled from three layers.

Math (v0.3 §2, unchanged):
    s = mean(|host_raw|)                      per day
    z0 = asinh(host_raw / s)                  host-anchored hyperbolic coordinate
    (w-,w0,w+) = softmax([l-, 0, l+])         conditional mass (center logit=0)
    m- = ReLU(r-), m+ = ReLU(r+)              hyperbolic displacement doses
    z- = z0 - m-, z+ = z0 + m+                candidate positions
    x_a = s * sinh(z_a)                       inverse transform to raw price

Architecture (v0.4 §2-§4):
    h_core = CoreContextEncoder(core_input)       # scale-free universal core
    (FiLM modulation via DataSignature happens inside CoreContextEncoder)
    h_opt  = OptionalCovariateEncoder(optional)   # zero-init residual branch
    h      = h_core + h_opt                       # optional disabled => h_core
    (l-, l+, r-, r+) = heads(h)

The core learns correction geometry, NOT dataset identity. market_id/target_id
are audit metadata and never enter the core path.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hch_v2_context import CoreContextEncoder, OptionalCovariateEncoder


class IAHCandidateHead(nn.Module):
    """Assembled v0.4 universal core + IAH three-atom head.

    Args:
        d_core_context: dimension of scale-free context (u + time_feat + lag_sf).
            z0 is appended internally as dimension 0, so CoreContextEncoder
            receives d_core_context + 1.
        d_value: dimension of optional covariate values (0 => no optional branch).
    """

    def __init__(self, d_core_context: int, d_model: int = 64,
                 d_value: int = 0, d_sig: int = 32):
        super().__init__()
        self.d_core_context = d_core_context
        self.d_model = d_model
        # +1 for z0 (appended internally as core_input dimension 0)
        self.core_encoder = CoreContextEncoder(d_core_context + 1, d_model, d_sig)
        self.optional_encoder = (OptionalCovariateEncoder(d_value, d_model)
                                 if d_value > 0 else None)
        self.mass_head = nn.Linear(d_model, 2)   # l_minus, l_plus
        self.shift_head = nn.Linear(d_model, 2)  # r_minus, r_plus
        # Atom-collapse guard (verified on LAGO_DE S2): default PyTorch init
        # (bias=0) leaves the raw pre-activations r- / r+ in the ReLU dead
        # zone, so m-=m+=0 forever and the candidate degenerates to Identity
        # (execute_rate=0). A positive bias keeps both ReLU displacement
        # channels alive at init so the IAH-CRPS gradient (Eq 10) can move m.
        # weight*0.05 keeps the initial output bias-dominated.
        nn.init.constant_(self.shift_head.bias, 0.5)
        self.shift_head.weight.data.mul_(0.05)

    def _compute_scale(self, host_raw: torch.Tensor,
                       valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """s_d = mean(|host_raw|) over VALID AND FINITE hours (Eq 2).
        Returns s=0 for days with no valid+finite host value.
        """
        host_abs = host_raw.squeeze(-1).abs()  # [B, H]
        finite_mask = torch.isfinite(host_abs)
        if valid_mask is None:
            eff = finite_mask
        else:
            vm = valid_mask.to(host_raw.device)
            if vm.dim() == 3:
                vm = vm.squeeze(-1)
            eff = finite_mask & (vm > 0.5)
        cnt = eff.sum(dim=1)
        s_sum = torch.where(eff, host_abs, torch.zeros_like(host_abs)).sum(dim=1)
        return torch.where(cnt > 0, s_sum / cnt.clamp(min=1),
                           torch.zeros_like(s_sum))

    def forward(self, host_raw: torch.Tensor, core_context: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None,
                domain_det: Optional[torch.Tensor] = None,
                optional_values: Optional[torch.Tensor] = None,
                optional_roles: Optional[torch.Tensor] = None,
                optional_masks: Optional[torch.Tensor] = None) -> dict:
        """host_raw: [B,H,1] raw prices; core_context: [B,H,d_core_context].
        domain_det: [B,d_det] per-batch Data Signature descriptors (P0-1);
            None => single-domain frozen buffer.
        optional_*: [B,H,N,F]/[B,H,N] covariate tensors (may be None).
        """
        B, H, _ = host_raw.shape

        # ---- IAH coordinate (Eq 2-3) ----
        s = self._compute_scale(host_raw, valid_mask)
        scale_valid = (s > 0).float()
        # Implementation masking: dummy safe denominator for asinh/sinh only.
        # Cannot change math output: scale-invalid days overwritten to Identity.
        s_safe = s.clamp(min=1e-12)

        host64 = host_raw.to(torch.float64)
        s_safe64 = s_safe.to(torch.float64)
        z0_64 = torch.asinh(host64.squeeze(-1) / s_safe64.unsqueeze(-1))  # [B,H]
        z0 = z0_64.to(torch.float32)

        # ---- core input = [z0, core_context] ----
        core_input = torch.cat([z0.unsqueeze(-1), core_context], dim=-1)  # [B,H,D+1]

        # ---- core encoding + DataSignature FiLM (P0-1: domain_det context) ----
        h_core = self.core_encoder(core_input, domain_det)  # [B,H,d_model]

        # ---- optional residual (zero-init) ----
        if self.optional_encoder is not None and optional_values is not None:
            h_opt = self.optional_encoder(optional_values, optional_roles,
                                          optional_masks)
        else:
            h_opt = 0.0
        h = h_core + h_opt

        # ---- heads (Eq 6-7) ----
        l_raw = self.mass_head(h)    # [B,H,2]
        l_minus, l_plus = l_raw[..., 0], l_raw[..., 1]
        logits = torch.stack([l_minus, torch.zeros_like(l_minus), l_plus], dim=-1)
        w = F.softmax(logits, dim=-1)
        w_minus, w_zero, w_plus = w[..., 0], w[..., 1], w[..., 2]

        r_raw = self.shift_head(h)
        m_minus = F.relu(r_raw[..., 0])
        m_plus = F.relu(r_raw[..., 1])

        # ---- candidate positions + inverse transform (Eq 8) ----
        z_minus = z0 - m_minus
        z_plus = z0 + m_plus

        s_safe32 = s_safe.to(torch.float32).unsqueeze(-1).unsqueeze(-1)
        x_identity = s_safe32 * torch.sinh(z0_64).unsqueeze(-1).to(torch.float32)
        x_down = s_safe32 * torch.sinh(z_minus.to(torch.float64)).unsqueeze(-1).to(torch.float32)
        x_up = s_safe32 * torch.sinh(z_plus.to(torch.float64)).unsqueeze(-1).to(torch.float32)

        # ---- scale-invalid days: Identity (broadcast mask, autograd-safe) ----
        day_mask = scale_valid.unsqueeze(-1)  # [B,1]
        w_minus = torch.where(day_mask > 0.5, w_minus, torch.zeros_like(w_minus))
        w_zero = torch.where(day_mask > 0.5, w_zero, torch.ones_like(w_zero))
        w_plus = torch.where(day_mask > 0.5, w_plus, torch.zeros_like(w_plus))
        m_minus = torch.where(day_mask > 0.5, m_minus, torch.zeros_like(m_minus))
        m_plus = torch.where(day_mask > 0.5, m_plus, torch.zeros_like(m_plus))
        z_minus = torch.where(day_mask > 0.5, z_minus, z0)
        z_plus = torch.where(day_mask > 0.5, z_plus, z0)
        day_mask_3d = day_mask.unsqueeze(-1)
        x_identity = torch.where(day_mask_3d > 0.5, x_identity, host_raw)
        x_down = torch.where(day_mask_3d > 0.5, x_down, host_raw)
        x_up = torch.where(day_mask_3d > 0.5, x_up, host_raw)

        return {
            "s": s, "scale_valid": scale_valid,
            "z0": z0, "z_minus": z_minus, "z_plus": z_plus,
            "w_minus": w_minus, "w_zero": w_zero, "w_plus": w_plus,
            "m_minus": m_minus, "m_plus": m_plus,
            "x_identity": x_identity, "x_down": x_down, "x_up": x_up,
            "valid_mask": valid_mask if valid_mask is not None
                          else torch.ones(B, H, device=host_raw.device),
        }
