"""``HCHFinalCore`` — the four registered variants through one forward contract.

    A0 DIRECT              same shared input tensor and shared encoder/day core;
                           per-hour ``[u_h || d] -> Linear(64,32) -> LayerNorm ->
                           GELU -> near-zero Linear(32,1)``; reconstruction only.
    A1 GEOM_FLAT           exact b/B/S decoder with the geometry-derived mass
                           scalar structurally removed from the query: ``[d || 0]``.
    A2 GEOM_COUPLED        the primary final method.
    A3 GEOM_COUPLED_NOHIST exactly A2 with the deterministic W=7 prior zero-masked.

Only the registered variant switch may differ between fits.  No market ID, Host
family ID, province name or dataset branch is a model feature, and no file here
imports the v4.4 router, coordinate embeddings, history GRU, Shape-history
encoder, retrieval, safety gate or market expert.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from core.contracts import HORIZON, TOKEN_DIM
from core.decoder import decode
from core.losses import daily_mae, masked_row_mean, structured_loss
from core.scales import CoordinateScales, require_scales

from fused_encoder import (
    CALENDAR_HOUR_CHANNELS,
    CORE_HOUR_CHANNELS,
    HOST_HOUR_CHANNELS,
    WIDTH,
    FinalSharedEncoder,
)
from geometry_prior import GeometryPriorBatch
from global_local import QUERY_DIM, DayRepairCore, SignedMassQueryShapeHead

DIRECT = "DIRECT"
GEOM_FLAT = "GEOM_FLAT"
GEOM_COUPLED = "GEOM_COUPLED"
GEOM_COUPLED_NOHIST = "GEOM_COUPLED_NOHIST"
FINAL_VARIANTS = (DIRECT, GEOM_FLAT, GEOM_COUPLED, GEOM_COUPLED_NOHIST)
STRUCTURED_VARIANTS = (GEOM_FLAT, GEOM_COUPLED, GEOM_COUPLED_NOHIST)

DAY_CONTEXT_CHANNELS = 21  # 13 host day descriptors + 5 calendar day + 3 prior
DIRECT_HIDDEN = 32
DIRECT_INIT_STD = 1e-3
#: Upper bound on the DIRECT / GEOM_COUPLED trainable-parameter ratio.
DIRECT_MAX_PARAMETER_RATIO = 1.25

#: Mirrors ``core.model.FORBIDDEN_BATCH_TOKENS``.  The unit-test suite proves the
#: two lists are identical by parsing the frozen source file, so the mirror cannot
#: silently drift; the local copy exists so the active path never imports the v4.4
#: router package.
FORBIDDEN_INPUT_TOKENS = (
    "target",
    "residual",
    "y_true",
    "actual",
    "observed",
    "truth",
    "label",
)


class IllegalInputError(ValueError):
    """Raised when a final-method input carries a target or current-residual field."""


def assert_legal_input_fields(mapping: Mapping) -> Mapping:
    for key in mapping:
        lowered = str(key).lower()
        for token in FORBIDDEN_INPUT_TOKENS:
            if token in lowered:
                raise IllegalInputError(
                    f"final-method input field {key!r} matches forbidden token {token!r}; "
                    "a legal input carries no target and no current residual"
                )
    return mapping


@dataclass
class FinalInput:
    """Everything the final method may legally see at the forecast origin."""

    host_hour_channels: torch.Tensor  # [B, H, 5]
    calendar_hour: torch.Tensor  # [B, H, 7]
    calendar_day: torch.Tensor  # [B, 5]
    host_day_descriptors: torch.Tensor  # [B, 13]
    hour_valid: torch.Tensor  # [B, H]
    prior: GeometryPriorBatch
    date: Sequence[str] = ()

    def __post_init__(self) -> None:
        assert_legal_input_fields(self.__dict__)
        batch = int(self.host_hour_channels.shape[0])
        for name, width in (
            ("host_hour_channels", HOST_HOUR_CHANNELS),
            ("calendar_hour", CALENDAR_HOUR_CHANNELS),
            ("calendar_day", 5),
            ("host_day_descriptors", 13),
        ):
            value = getattr(self, name)
            if value.shape[-1] != width:
                raise IllegalInputError(f"{name} must have width {width}")
            if int(value.shape[0]) != batch:
                raise IllegalInputError(f"{name} batch size mismatch")
        if self.host_hour_channels.shape[1] != HORIZON:
            raise IllegalInputError(f"host_hour_channels must have H={HORIZON} hours")
        if self.hour_valid.shape != self.host_hour_channels.shape[:2]:
            raise IllegalInputError("hour_valid must be [B, H]")
        if self.prior.batch_size != batch:
            raise IllegalInputError("prior batch size mismatch")

    @property
    def batch_size(self) -> int:
        return int(self.host_hour_channels.shape[0])


@dataclass
class FinalOutput:
    """Structured exact-decoder output, or the matched direct per-hour output."""

    variant: str
    correction: torch.Tensor
    b_hat: torch.Tensor = None
    B_hat: torch.Tensor = None
    s_plus_hat: torch.Tensor = None
    s_minus_hat: torch.Tensor = None
    a_plus: torch.Tensor = None
    a_minus: torch.Tensor = None
    overlap_gap: torch.Tensor = None
    m_plus: torch.Tensor = None
    m_minus: torch.Tensor = None
    structured: bool = True
    diagnostics: dict = field(default_factory=dict)


class _DirectHead(nn.Module):
    """Matched direct control: one scalar per hour from ``[u_h || d]``."""

    def __init__(self, width: int = WIDTH, hidden: int = DIRECT_HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(2 * width, hidden)
        self.out = nn.Linear(hidden, 1)
        nn.init.normal_(self.out.weight, mean=0.0, std=DIRECT_INIT_STD)
        nn.init.zeros_(self.out.bias)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        state = torch.cat([u, d.unsqueeze(1).expand(-1, u.shape[1], -1)], dim=-1)
        hidden = torch.nn.functional.gelu(self.norm(self.inp(state)))
        return self.out(self.drop(hidden)).squeeze(-1)


class HCHFinalCore(nn.Module):
    """One shared code path for the four registered final-method variants."""

    def __init__(
        self,
        variant: str,
        scales: CoordinateScales,
        dropout: float = 0.1,
        width: int = WIDTH,
        direct_hidden: int = DIRECT_HIDDEN,
        in_channels: int = CORE_HOUR_CHANNELS,
        token_dim: int = TOKEN_DIM,
    ):
        super().__init__()
        if variant not in FINAL_VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {FINAL_VARIANTS}")
        scales = require_scales(scales)
        self.variant = str(variant)
        self.width = int(width)
        self.structured = variant in STRUCTURED_VARIANTS
        self.coupled = variant == GEOM_COUPLED
        self.uses_prior = variant != GEOM_COUPLED_NOHIST
        self.dropout = float(dropout)
        self.register_buffer("s_b", torch.tensor(float(scales.s_b)), persistent=True)
        self.register_buffer("s_B", torch.tensor(float(scales.s_B)), persistent=True)
        self.register_buffer("s_r", torch.tensor(float(scales.s_r)), persistent=True)

        self.encoder = FinalSharedEncoder(in_channels=in_channels, width=width, dropout=dropout)
        self.day_core = DayRepairCore(
            day_dim=DAY_CONTEXT_CHANNELS, s_B=float(scales.s_B), width=width,
            dropout=dropout, token_dim=token_dim,
        )
        if self.structured:
            self.shape_head = SignedMassQueryShapeHead(width=width)
        else:
            self.direct_head = _DirectHead(width=width, hidden=direct_hidden, dropout=dropout)

    # -- input assembly ---------------------------------------------------
    @staticmethod
    def _per_hour(value: torch.Tensor, horizon: int) -> torch.Tensor:
        """Broadcast a per-day scalar channel to one constant column per hour.

        ``avail_plus``/``avail_minus`` describe the whole history window, not a
        particular hour, so the same value is repeated across the horizon rather
        than being given a spurious hour axis of its own.
        """
        flat = torch.as_tensor(value).float().reshape(-1, 1, 1)
        return flat.expand(-1, int(horizon), 1)

    def hour_channels(self, inp: FinalInput, prior: GeometryPriorBatch) -> torch.Tensor:
        host = torch.as_tensor(inp.host_hour_channels).float()
        return torch.cat(
            [
                host,
                torch.as_tensor(inp.calendar_hour).float(),
                prior.s_plus_bar.unsqueeze(-1),
                prior.s_minus_bar.unsqueeze(-1),
                self._per_hour(prior.avail_plus, host.shape[1]),
                self._per_hour(prior.avail_minus, host.shape[1]),
            ],
            dim=-1,
        )

    def day_context(self, inp: FinalInput, prior: GeometryPriorBatch) -> torch.Tensor:
        return torch.cat(
            [
                torch.as_tensor(inp.host_day_descriptors).float(),
                torch.as_tensor(inp.calendar_day).float(),
                (prior.b_bar / self.s_b).unsqueeze(-1),
                (prior.B_bar / self.s_B).unsqueeze(-1),
                prior.avail.unsqueeze(-1),
            ],
            dim=-1,
        )

    # -- forward ----------------------------------------------------------
    def forward(self, inp: FinalInput) -> FinalOutput:
        prior = inp.prior if self.uses_prior else inp.prior.zeroed()
        x = self.hour_channels(inp, prior)
        z_day = self.day_context(inp, prior)
        u = self.encoder(x)
        d, b_hat, B_hat = self.day_core(u, inp.hour_valid, z_day)

        if not self.structured:
            correction = self.direct_head(u, d)
            return FinalOutput(
                variant=self.variant, correction=correction, structured=False,
                diagnostics={"prior_masked": not self.uses_prior},
            )

        head = self.shape_head(
            d=d, u=u, b_hat=b_hat, B_hat=B_hat, hour_valid=inp.hour_valid,
            s_b=float(self.s_b), s_B=float(self.s_B), coupled=self.coupled,
        )
        decoded = decode(b_hat, B_hat, head["s_plus_hat"], head["s_minus_hat"])
        return FinalOutput(
            variant=self.variant,
            correction=decoded.correction,
            b_hat=decoded.b_hat,
            B_hat=decoded.B_hat,
            s_plus_hat=decoded.s_plus_hat,
            s_minus_hat=decoded.s_minus_hat,
            a_plus=decoded.a_plus,
            a_minus=decoded.a_minus,
            overlap_gap=decoded.overlap_gap,
            m_plus=head["m_plus"],
            m_minus=head["m_minus"],
            structured=True,
            diagnostics={
                "s_m": head["s_m"],
                "prior_masked": not self.uses_prior,
                "coupled": bool(head["coupled"]),
            },
        )

    # -- loss -------------------------------------------------------------
    def compute_loss(
        self,
        output: FinalOutput,
        target,
        residual: torch.Tensor,
        scales: CoordinateScales,
        valid: torch.Tensor | None = None,
    ) -> dict:
        """Frozen reconstruction term; the coordinate auxiliaries are the frozen ones."""
        scales = require_scales(scales)
        if valid is None:
            valid = target.valid
        valid = valid.to(torch.bool)
        if not self.structured:
            per_day, row_mask = daily_mae(residual, output.correction, valid)
            rec = masked_row_mean(per_day, row_mask) / scales.s_r
            return {"total": rec, "L_rec": rec, "structure": "direct_reconstruction_only"}
        return structured_loss(
            correction=output.correction,
            b_hat=output.b_hat,
            B_hat=output.B_hat,
            s_plus_hat=output.s_plus_hat,
            s_minus_hat=output.s_minus_hat,
            residual=residual,
            target=target,
            scales=scales,
            valid=valid,
        )

    # -- introspection ----------------------------------------------------
    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def parameter_map(self) -> dict:
        return {
            name: int(sum(p.numel() for p in module.parameters() if p.requires_grad))
            for name, module in self.named_children()
        }

    def variant_config(self) -> dict:
        return {
            "variant": self.variant,
            "structured_decoder": bool(self.structured),
            "signed_mass_query_coupled": bool(self.coupled),
            "history_prior_enabled": bool(self.uses_prior),
            "hour_channels": int(self.encoder.in_channels),
            "day_context_channels": DAY_CONTEXT_CHANNELS,
            "width": int(self.width),
            "query_dim": int(QUERY_DIM),
            "learned_temporal_encoders": 1,
            "signed_mass_queries": 2 if self.structured else 0,
            "source_router_calls": 0,
            "parameter_count": self.parameter_count(),
            "parameter_map": self.parameter_map(),
        }


def build_variant(variant: str, scales: CoordinateScales, **kwargs) -> HCHFinalCore:
    return HCHFinalCore(variant, scales, **kwargs)


def assert_variant_parity(models: Mapping) -> dict:
    """Structural A1/A2 identity and the DIRECT ``<= 1.25x`` budget."""
    counts = {name: model.parameter_count() for name, model in models.items()}
    report = {"parameter_counts": counts, "checks": {}}
    if GEOM_FLAT in counts and GEOM_COUPLED in counts:
        report["checks"]["a1_a2_equal_parameter_count"] = (
            counts[GEOM_FLAT] == counts[GEOM_COUPLED]
        )
    if GEOM_COUPLED in counts and GEOM_COUPLED_NOHIST in counts:
        report["checks"]["a2_a3_equal_parameter_count"] = (
            counts[GEOM_COUPLED] == counts[GEOM_COUPLED_NOHIST]
        )
    if DIRECT in counts and GEOM_COUPLED in counts:
        ratio = counts[DIRECT] / max(counts[GEOM_COUPLED], 1)
        report["direct_over_a2_ratio"] = ratio
        report["checks"]["direct_within_budget"] = ratio <= DIRECT_MAX_PARAMETER_RATIO
    return report
