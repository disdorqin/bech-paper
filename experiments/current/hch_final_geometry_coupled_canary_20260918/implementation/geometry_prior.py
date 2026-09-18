"""Deterministic W=7 revealed-residual geometry prior (method freeze §4.2).

The final core keeps revealed residual history **only** as a deterministic
geometry prior.  For each of the last seven fully revealed residual days ``tau``
the same exact geometry is computed, and then

    b_bar        = median_tau b_tau
    B_bar        = median_tau B_tau
    Sbar_h^+     = sum_tau P_tau S^+_{tau,h} / (sum_tau P_tau + eps)
    Sbar_h^-     = sum_tau N_tau S^-_{tau,h} / (sum_tau N_tau + eps)

with the Shape prior exactly zero and its availability mask exactly zero where the
corresponding historical signed mass is zero.

There are **no learned parameters**, no nearest-day retrieval, no similarity
weighting and no learned reliability: the prior is a pure function of legally
revealed residuals.  Arithmetic is fp32, and the exact geometry itself comes from
``core.geometry.residual_geometry`` rather than being re-derived here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from core.contracts import EPS, HORIZON
from core.geometry import residual_geometry

HISTORY_DAYS = 7


class GeometryPriorError(ValueError):
    """Raised when a history window violates the reveal-time / shape contract."""


@dataclass(frozen=True)
class GeometryPriorBatch:
    """The deterministic W=7 geometry prior for a batch of forecast origins."""

    b_bar: torch.Tensor  # [B]     median historical signed Level
    B_bar: torch.Tensor  # [B]     median historical balanced mass
    s_plus_bar: torch.Tensor  # [B, H]  P-mass-weighted positive Shape barycenter
    s_minus_bar: torch.Tensor  # [B, H]  N-mass-weighted negative Shape barycenter
    avail_plus: torch.Tensor  # [B]  1.0 iff some historical positive mass exists
    avail_minus: torch.Tensor  # [B]  1.0 iff some historical negative mass exists
    avail: torch.Tensor  # [B]  1.0 iff at least one usable revealed day exists
    n_days_used: torch.Tensor  # [B]  int64 count of usable revealed days

    def zeroed(self) -> "GeometryPriorBatch":
        """Structurally zero-masked copy (the A3 GEOM_COUPLED_NOHIST input)."""
        z = torch.zeros_like
        return GeometryPriorBatch(
            b_bar=z(self.b_bar),
            B_bar=z(self.B_bar),
            s_plus_bar=z(self.s_plus_bar),
            s_minus_bar=z(self.s_minus_bar),
            avail_plus=z(self.avail_plus),
            avail_minus=z(self.avail_minus),
            avail=z(self.avail),
            n_days_used=torch.zeros_like(self.n_days_used),
        )

    def to(self, device) -> "GeometryPriorBatch":
        return GeometryPriorBatch(
            **{f: getattr(self, f).to(device) for f in self.__dataclass_fields__}
        )

    @property
    def batch_size(self) -> int:
        return int(self.b_bar.shape[0])

    def field_hash(self) -> str:
        import hashlib

        h = hashlib.sha256()
        for f in sorted(self.__dataclass_fields__):
            h.update(getattr(self, f).detach().cpu().numpy().tobytes())
        return h.hexdigest().upper()


def empty_prior(batch_size: int, horizon: int = HORIZON, device=None) -> GeometryPriorBatch:
    """Zero prior with zero availability (used when no legal window exists)."""
    f = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)  # noqa: E731
    return GeometryPriorBatch(
        b_bar=f(batch_size),
        B_bar=f(batch_size),
        s_plus_bar=f(batch_size, horizon),
        s_minus_bar=f(batch_size, horizon),
        avail_plus=f(batch_size),
        avail_minus=f(batch_size),
        avail=f(batch_size),
        n_days_used=torch.zeros(batch_size, dtype=torch.int64, device=device),
    )


def _check_window(days: Sequence, target_day=None, origin=None, horizon: int = HORIZON) -> None:
    if len(days) != HISTORY_DAYS:
        raise GeometryPriorError(f"history must contain exactly {HISTORY_DAYS} revealed days")
    previous = None
    for item in days:
        if item.day is None or item.reveal_time is None:
            raise GeometryPriorError("every history day must carry a day and a reveal_time")
        if previous is not None and not item.day > previous:
            raise GeometryPriorError("history days must be unique and strictly ascending")
        previous = item.day
        res = item.residual
        if res.shape[-1] != horizon:
            raise GeometryPriorError(f"history residual must have H={horizon} hours")
        if target_day is not None and not item.day < target_day:
            raise GeometryPriorError(
                f"history day {item.day} is not strictly before target {target_day}"
            )
        if origin is not None and not item.reveal_time < origin:
            raise GeometryPriorError(
                f"history day {item.day} reveals at {item.reveal_time}, not before origin {origin}"
            )


def build_geometry_prior(
    histories: Sequence[Sequence],
    *,
    target_days: Optional[Sequence] = None,
    origins: Optional[Sequence] = None,
    horizon: int = HORIZON,
    eps: float = EPS,
) -> GeometryPriorBatch:
    """Exact deterministic prior for a batch of W=7 revealed-residual windows.

    ``histories[i]`` is the ascending, oldest-first window of seven
    ``core.history.RevealedResidualDay`` objects for row ``i``.  When
    ``target_days`` / ``origins`` are supplied the reveal boundary is re-asserted
    here rather than trusted.
    """
    if not histories:
        raise GeometryPriorError("empty history list")
    batch = len(histories)
    target_days = list(target_days) if target_days is not None else [None] * batch
    origins = list(origins) if origins is not None else [None] * batch
    if len(target_days) != batch or len(origins) != batch:
        raise GeometryPriorError("target_days / origins must have one entry per row")

    res = torch.zeros(batch, HISTORY_DAYS, horizon, dtype=torch.float32)
    valid = torch.zeros(batch, HISTORY_DAYS, horizon, dtype=torch.bool)
    for i, window in enumerate(histories):
        _check_window(window, target_days[i], origins[i], horizon)
        for j, item in enumerate(window):
            r = torch.as_tensor(item.residual).float().reshape(horizon)
            v = item.valid
            if v is None:
                v = torch.ones(horizon, dtype=torch.bool)
            else:
                v = torch.as_tensor(v).reshape(horizon).to(torch.bool)
            if not bool(torch.isfinite(r[v]).all()):
                raise GeometryPriorError(f"history day {item.day} carries a non-finite residual")
            # Ineligible hours are structurally zeroed here so a non-finite value
            # on an invalid hour can never reach the geometry call.
            res[i, j] = torch.where(v, r, torch.zeros_like(r))
            valid[i, j] = v

    # One exact-geometry call on the flattened [B*7, H] stack: the geometry is
    # never re-derived in stage code.
    geom = residual_geometry(res.reshape(batch * HISTORY_DAYS, horizon), valid.reshape(batch * HISTORY_DAYS, horizon))
    b = geom.b.reshape(batch, HISTORY_DAYS).float()
    B = geom.B.reshape(batch, HISTORY_DAYS).float()
    P = geom.P.reshape(batch, HISTORY_DAYS).float()
    N = geom.N.reshape(batch, HISTORY_DAYS).float()
    s_plus = geom.s_plus.reshape(batch, HISTORY_DAYS, horizon).float()
    s_minus = geom.s_minus.reshape(batch, HISTORY_DAYS, horizon).float()

    # A day with no eligible hour carries no information at all and is dropped
    # from the median rather than contributing a structural zero.
    usable = valid.any(dim=-1)  # [B, 7]
    n_used = usable.sum(dim=-1)
    # ``keep`` must stay [B, 7] to broadcast against the per-day statistics; an
    # extra trailing axis here would silently turn the median into a [B, 7, 7]
    # reduction and leave ``b_bar``/``B_bar`` with a day axis they must not have.
    keep = usable

    def _masked_median(values: torch.Tensor) -> torch.Tensor:
        # +inf padding leaves the median of the usable entries untouched for a
        # 7-day window; a row with no usable day falls back to exact zero.
        pad = torch.where(keep, values, torch.full_like(values, float("inf")))
        med = torch.median(pad, dim=1).values
        return torch.where(n_used > 0, med, torch.zeros_like(med))

    b_bar = _masked_median(b)
    B_bar = _masked_median(B)

    p_mass = (P * usable.to(P.dtype)).sum(dim=1)  # [B]
    n_mass = (N * usable.to(N.dtype)).sum(dim=1)
    num_plus = (P.unsqueeze(-1) * s_plus).sum(dim=1)  # [B, H]
    num_minus = (N.unsqueeze(-1) * s_minus).sum(dim=1)
    s_plus_bar = num_plus / (p_mass.unsqueeze(-1) + eps)
    s_minus_bar = num_minus / (n_mass.unsqueeze(-1) + eps)

    avail_plus = (p_mass > 0.0).to(torch.float32)
    avail_minus = (n_mass > 0.0).to(torch.float32)
    avail = (n_used > 0).to(torch.float32)

    # Zero prior and zero availability wherever no valid sign mass exists.
    s_plus_bar = torch.where(avail_plus.unsqueeze(-1) > 0, s_plus_bar, torch.zeros_like(s_plus_bar))
    s_minus_bar = torch.where(avail_minus.unsqueeze(-1) > 0, s_minus_bar, torch.zeros_like(s_minus_bar))
    b_bar = torch.where(avail > 0, b_bar, torch.zeros_like(b_bar))
    B_bar = torch.where(avail > 0, B_bar, torch.zeros_like(B_bar))

    return GeometryPriorBatch(
        b_bar=b_bar.contiguous(),
        B_bar=B_bar.contiguous(),
        s_plus_bar=s_plus_bar.contiguous(),
        s_minus_bar=s_minus_bar.contiguous(),
        avail_plus=avail_plus,
        avail_minus=avail_minus,
        avail=avail,
        n_days_used=n_used.to(torch.int64),
    )


def prior_identities(prior: GeometryPriorBatch, tol: float = 1e-6) -> dict:
    """Deterministic invariants of one prior batch (never a performance metric)."""
    sp_sum = prior.s_plus_bar.sum(dim=-1)
    sm_sum = prior.s_minus_bar.sum(dim=-1)
    return {
        "positive_shape_simplex_or_zero": bool(
            (
                (sp_sum - prior.avail_plus).abs() <= tol
            ).all()
        ),
        "negative_shape_simplex_or_zero": bool(
            (
                (sm_sum - prior.avail_minus).abs() <= tol
            ).all()
        ),
        "shapes_nonnegative": bool(
            (prior.s_plus_bar >= -tol).all() and (prior.s_minus_bar >= -tol).all()
        ),
        "B_bar_nonnegative": bool((prior.B_bar >= -tol).all()),
        "zero_mass_implies_zero_availability": bool(
            ((prior.s_plus_bar.abs().sum(-1) > tol) <= (prior.avail_plus > 0)).all()
            and ((prior.s_minus_bar.abs().sum(-1) > tol) <= (prior.avail_minus > 0)).all()
        ),
        "all_finite": bool(
            torch.isfinite(prior.b_bar).all()
            and torch.isfinite(prior.B_bar).all()
            and torch.isfinite(prior.s_plus_bar).all()
            and torch.isfinite(prior.s_minus_bar).all()
        ),
    }
