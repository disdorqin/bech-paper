"""LOCAL_TRAIN target and history construction, and the Stage-2 training entry.

Two day sets live here and they are deliberately different:

* the **target support** is ``LOCAL_TRAIN = B2 u B3 u B4``.  Only these days are
  supervised, normalised or early-stopped on.  A Local target built on a warm-up
  day would be supervised against a coarse Level with no causal OOF prediction,
  and B1 would be scored on a day the protocol assigns no target role.
* the **history support** is ``B1 u B2 u B3 u B4``.  Local history is post-Level
  ``q = r - b_tilde*1``, so a history day needs its own causal OOF ``b_tilde`` —
  which B1 has.  Without it the first B2 target would look back on an all-zero
  window even though seven legal revealed days sit immediately before it.
  Including B1 here is what the protocol's "warm start" means; it changes no
  target, no loss, no scale and no early-stop day.

The q-space is the core's.  ``q = r - b_tilde*1`` and the decomposition
``q = delta*1 + A(S+ - S-)`` are computed by :mod:`core.targets`, never re-derived
here: a second implementation of the method's central identity is how the
experiment and the method drift apart.

The one decomposition that *is* built here is R2's ``q = A+ S+ - A- S-``.  That is
not the RCL identity — the core has no untied variant and the core is frozen for
this campaign — so R2's native signed masses are assembled from the core's ``q``
with the same masked, valid-count-aware conventions the core uses for the tied
decomposition.  It is supervision the parameterization can actually act on, which
is precisely what supervising R2 with R1's derived ``delta``/centred-``A`` was not.

Each variant's objective follows protocol section 10 and is assembled from the
core's loss primitives.  The variant dispatch selects *which* coordinates are
supervised; it never changes a scale, a weight or the optimizer recipe.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from core import (LocalNormalizationStats, build_base_tokens, build_q,
                  build_residual_complete_targets, centered_mass_loss,
                  delta_loss, local_reconstruction_loss, masked_mae, shape_loss)

from oof_level import seed_everything
from rcl_contract import local_train_days, oof_partition

__all__ = [
    "LOCAL_HISTORY_DAYS",
    "LOCAL_FIT_TAIL_FRACTION",
    "LOCAL_RECIPE",
    "EPS_UNTIED",
    "UNTIED_MASS_SPACE",
    "LocalTrainPanel",
    "UntiedMassTargets",
    "UntiedMassStats",
    "build_local_history",
    "build_local_train_panel",
    "history_support_days",
    "stage2_objective",
    "assert_coarse_level_matches_panel",
    "state_dict_sha256",
    "train_stage2",
    "untied_mass_targets",
]

#: Operational history window.  RCL's config deliberately has no history-window
#: field, so this is a run constant: the number of already-revealed q days the
#: Shape and magnitude GRUs look back on.  It is not per-market.
LOCAL_HISTORY_DAYS = 7

#: Inner chronological early-stop tail, as a fraction of LOCAL_TRAIN
#: (protocol section 11: early stopping uses q reconstruction MAE on an inner
#: chronological tail of LOCAL_TRAIN).
LOCAL_FIT_TAIL_FRACTION = 0.20

LOCAL_RECIPE = {
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "max_epochs": 50,
    "patience": 8,
    "grad_clip": 1.0,
}

#: Below this signed mass the corresponding R2 simplex carries no information.
EPS_UNTIED = 1e-8

#: R2's mass scales are a q-geometry quantity fitted on LOCAL_TRAIN and nothing
#: else.  The tag makes an accidental substitution of some other space visible.
UNTIED_MASS_SPACE = "local_train_q_signed_mass"


def history_support_days(n_train_days: int) -> list[int]:
    """``B1 u B2 u B3 u B4`` sorted — the days Local history may be drawn from.

    Warm-up days are excluded: they are Stage-1 support, not revealed Local days,
    and B1 is the earliest block with a causal OOF Level of its own.
    """
    partition = oof_partition(n_train_days)
    out: set[int] = set()
    for name in ("B1", "B2", "B3", "B4"):
        out.update(partition[name])
    return sorted(out)


def build_local_history(q_values, positions: Sequence[int],
                        history_days: int = LOCAL_HISTORY_DAYS):
    """``(B, W, H)`` windows of the ``W`` rows **before** each position.

    Left-padded with zeros when fewer earlier rows exist.  Padding is a missing
    day, not a zero observation: the window only ever contains rows strictly
    before the target position, so no target-day value can enter its own history.
    """
    q_values = np.asarray(q_values, dtype=np.float32)
    horizon = q_values.shape[-1]
    out = np.zeros((len(positions), history_days, horizon), dtype=np.float32)
    for row, position in enumerate(positions):
        start = max(0, int(position) - history_days)
        span = q_values[start:int(position)]
        if len(span):
            out[row, -len(span):] = span
    return out


@dataclass(frozen=True)
class UntiedMassTargets:
    """R2's native supervision: two signed masses and their simplices."""

    a_plus: torch.Tensor             # (B,) sum of the positive part of q
    a_minus: torch.Tensor            # (B,) sum of the negative part of q
    shape_plus: torch.Tensor         # (B, H) simplex over the positive support
    shape_negative: torch.Tensor     # (B, H) simplex over the negative support


def untied_mass_targets(q: torch.Tensor,
                        valid_mask: Optional[torch.Tensor] = None,
                        *, eps: float = EPS_UNTIED) -> UntiedMassTargets:
    """``A+* = sum(q_+)``, ``A-* = sum((-q)_+)`` and their q-sign simplices.

    Masked positions are replaced, never multiplied by zero, and each simplex is
    exactly zero where the horizon does not exist — the same conventions
    :func:`core.targets.build_residual_complete_targets` uses for the tied
    decomposition.  A degenerate sign falls back to a uniform simplex over the
    valid horizon; since that branch is taken only when the mass is at most
    ``eps``, it contributes at most ``eps`` to the reconstruction.
    """
    q = torch.as_tensor(q, dtype=torch.float32)
    if valid_mask is None:
        valid = torch.ones_like(q, dtype=torch.bool)
    else:
        valid = torch.as_tensor(valid_mask) > 0
        valid = valid.expand_as(q) if valid.ndim < q.ndim else valid
    keep = valid.to(q.dtype)
    masked = torch.where(valid, q, torch.zeros_like(q))

    positive = torch.clamp(masked, min=0.0)
    negative = torch.clamp(-masked, min=0.0)
    a_plus = positive.sum(dim=-1)
    a_minus = negative.sum(dim=-1)

    valid_count = keep.sum(dim=-1, keepdim=True).clamp_min(1.0)
    uniform = torch.where(valid, keep / valid_count, torch.zeros_like(q))

    def _simplex(mass: torch.Tensor, part: torch.Tensor) -> torch.Tensor:
        denom = mass.clamp_min(eps).unsqueeze(-1)
        shape = torch.where((mass > eps).unsqueeze(-1), part / denom, uniform)
        return torch.where(valid, shape, torch.zeros_like(shape))

    return UntiedMassTargets(a_plus=a_plus, a_minus=a_minus,
                             shape_plus=_simplex(a_plus, positive),
                             shape_negative=_simplex(a_minus, negative))


@dataclass(frozen=True)
class UntiedMassStats:
    """R2's mass scales, fitted from LOCAL_TRAIN ``q`` geometry only."""

    s_a_plus: float
    s_a_minus: float
    space: str = UNTIED_MASS_SPACE

    def __post_init__(self) -> None:
        if self.space != UNTIED_MASS_SPACE:
            raise ValueError(
                f"R2 mass scales must be fitted in {UNTIED_MASS_SPACE!r}, got "
                f"{self.space!r}")
        for name in ("s_a_plus", "s_a_minus"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive finite scale")

    @classmethod
    def fit(cls, q: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
            ) -> "UntiedMassStats":
        targets = untied_mass_targets(q, valid_mask)
        return cls(
            s_a_plus=float(max(1e-6, targets.a_plus.abs().median())),
            s_a_minus=float(max(1e-6, targets.a_minus.abs().median())))

    def fingerprint(self) -> str:
        payload = f"{self.space}:{self.s_a_plus:.12g}:{self.s_a_minus:.12g}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class LocalTrainPanel:
    """The LOCAL_TRAIN panel: targets, history support and the frozen Level.

    ``day_ids`` is the **target** support (B2 u B3 u B4); the arrays are indexed by
    **history-support** position (B1 u B2 u B3 u B4), so a target day's row number
    and its position in the arrays are different numbers on purpose.  Conflating
    them is how a B1 day silently becomes a supervised target.

    ``coarse_level`` is kept in physical units and is *the* registered artifact
    VAL must reuse, so an inference path that recomputes a Level instead of
    reading this one is detectable rather than merely discouraged.
    """

    market: str
    host: str
    day_ids: tuple[int, ...]
    support_ids: tuple[int, ...]
    host_forecast: "np.ndarray"
    valid_mask: "np.ndarray"
    residual: "np.ndarray"
    coarse_level: "np.ndarray"
    coarse_level_source: str
    coarse_level_sha256: str
    stats: LocalNormalizationStats
    level_condition_stats: object
    untied_stats: UntiedMassStats
    fit_ids: tuple[int, ...] = ()
    early_ids: tuple[int, ...] = ()
    _position: dict = field(default_factory=dict, repr=False)
    _support_position: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._support_position = {int(d): i
                                  for i, d in enumerate(self.support_ids)}
        if len(self._support_position) != len(self.support_ids):
            raise ValueError("history-support day indices must be unique")
        if list(self.support_ids) != sorted(self.support_ids):
            raise ValueError("history support must be chronologically ordered")
        self._position = {int(d): self._support_position[int(d)]
                          for d in self.day_ids}
        if len(self._position) != len(self.day_ids):
            raise ValueError("LOCAL_TRAIN day indices must be unique")
        for name in ("host_forecast", "valid_mask", "residual", "coarse_level"):
            array = np.asarray(getattr(self, name))
            if array.shape[0] != len(self.support_ids):
                raise ValueError(
                    f"{name} has {array.shape[0]} rows for "
                    f"{len(self.support_ids)} history-support days")
        extra = set(self.day_ids) - set(self.support_ids)
        if extra:
            raise ValueError(f"target days outside the history support: {extra}")
        warmup = set(self.day_ids) - set(self.support_ids)
        if warmup:
            raise ValueError("a warm-up day cannot be a Local target")
        fitted = set(self.fit_ids) | set(self.early_ids)
        if fitted and fitted != set(self.day_ids):
            raise ValueError(
                "fit and early-stop days must partition the target support")
        if set(self.fit_ids) - set(self.support_ids):
            raise ValueError("a fit day is outside the history support")

    def positions(self, ids: Sequence[int]) -> list[int]:
        """Array positions of target ``ids`` (which are support positions too)."""
        return [self._position[int(d)] for d in ids]

    def level_for_days(self, ids: Sequence[int]) -> np.ndarray:
        return np.asarray([self.coarse_level[i] for i in self.positions(ids)],
                          dtype=np.float32)

    def q(self) -> np.ndarray:
        """``q = r - b_tilde*1`` through the core's authority, not re-derived.

        Defined over the whole history support, so the rows a target day looks
        back on are the same q the core would build for those days.
        """
        return build_q(torch.as_tensor(self.residual, dtype=torch.float32),
                       torch.as_tensor(self.coarse_level, dtype=torch.float32),
                       torch.as_tensor(self.valid_mask, dtype=torch.float32)).numpy()

    def target_q(self) -> np.ndarray:
        """The ``q`` rows of the **target** support only, in target order."""
        return self.q()[self.positions(self.day_ids)]

    def history_windows(self, ids: Sequence[int]) -> np.ndarray:
        """Post-Level q windows for target ``ids``, from the support's q rows.

        A window ends strictly before the target's own row, so it can contain B1 —
        that is the warm start — but never the target day itself and never a
        warm-up day.
        """
        return build_local_history(self.q(), self.positions(ids))

    def targets(self):
        """The exact residual-complete targets for the target support."""
        rows = self.positions(self.day_ids)
        return build_residual_complete_targets(
            torch.as_tensor(self.residual[rows], dtype=torch.float32),
            torch.as_tensor(self.coarse_level[rows], dtype=torch.float32),
            torch.as_tensor(self.valid_mask[rows], dtype=torch.float32))

    def untied_targets(self):
        """R2's native signed-mass targets over the target support."""
        rows = self.positions(self.day_ids)
        return untied_mass_targets(
            torch.as_tensor(self.residual[rows], dtype=torch.float32)
            - torch.as_tensor(self.coarse_level[rows],
                              dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(self.valid_mask[rows], dtype=torch.float32))

    def as_dict(self) -> dict:
        return {
            "market": self.market, "host": self.host,
            "n_days": len(self.day_ids),
            "n_support_days": len(self.support_ids),
            "support_first_day": int(self.support_ids[0]) if self.support_ids else None,
            "support_last_day": int(self.support_ids[-1]) if self.support_ids else None,
            "n_fit_days": len(self.fit_ids), "n_early_days": len(self.early_ids),
            "coarse_level_source": self.coarse_level_source,
            "coarse_level_sha256": self.coarse_level_sha256,
            "history_days": LOCAL_HISTORY_DAYS,
            "history_support": "B1_u_B2_u_B3_u_B4",
            "target_support": "B2_u_B3_u_B4",
            "untied_mass_scale_fingerprint": self.untied_stats.fingerprint(),
        }


def build_local_train_panel(market: str, host: str, day_ids, host_forecast,
                            residual, valid_mask, oof_panel, n_train_days: int,
                            stats: LocalNormalizationStats,
                            level_condition_stats, untied_stats: UntiedMassStats,
                            coarse_level_source: str) -> LocalTrainPanel:
    """Assemble the LOCAL_TRAIN panel from the causal OOF Level artifact.

    The target day set is :func:`rcl_contract.local_train_days` — B2 u B3 u B4 —
    and the history support adds B1.  The Level for every one of those days comes
    from ``oof_panel``.  A day missing from the OOF panel is a hard error: falling
    back to an in-sample Level for it would quietly reintroduce the optimism the
    cross-fit exists to remove.
    """
    days = [int(d) for d in local_train_days(n_train_days)]
    support = [int(d) for d in history_support_days(n_train_days)]
    position = {int(d): i for i, d in enumerate(day_ids)}
    missing = [d for d in support if d not in position]
    if missing:
        raise KeyError(f"history-support days absent from the cell: {missing[:8]}")
    absent = [d for d in support if d not in oof_panel.values]
    if absent:
        raise KeyError(f"days without a causal OOF Level: {absent[:8]}")

    rows = [position[d] for d in support]
    level = np.asarray([oof_panel.values[d] for d in support], dtype=np.float32)
    tail = max(2, int(-(-len(days) * LOCAL_FIT_TAIL_FRACTION // 1)))
    fit_ids, early_ids = tuple(days[:-tail]), tuple(days[-tail:])

    panel = LocalTrainPanel(
        market=market, host=host, day_ids=tuple(days), support_ids=tuple(support),
        host_forecast=np.asarray(host_forecast)[rows],
        valid_mask=np.asarray(valid_mask)[rows],
        residual=np.asarray(residual)[rows],
        coarse_level=level,
        coarse_level_source=coarse_level_source,
        coarse_level_sha256=hashlib.sha256(
            np.ascontiguousarray(level, dtype=np.float32).tobytes()).hexdigest(),
        stats=LocalNormalizationStats.ensure(stats),
        level_condition_stats=level_condition_stats,
        untied_stats=untied_stats,
        fit_ids=fit_ids, early_ids=early_ids)
    if isinstance(untied_stats, UntiedMassStats) is False:
        raise TypeError("R2 mass scales must be an UntiedMassStats")
    return panel


def assert_coarse_level_matches_panel(batch_level, panel: LocalTrainPanel,
                                      ids: Sequence[int]) -> None:
    """The batch's coarse Level must be the panel's registered artifact.

    Compared per day against the panel's own values.  A batch that supplies a
    *different* Level for the same day — a refit, a re-derivation, a silently
    normalised vector — fails here instead of training Local against a target
    geometry nobody registered.
    """
    wanted = panel.level_for_days(ids)
    got = np.asarray(batch_level, dtype=np.float32).reshape(-1)
    if got.shape != wanted.shape:
        raise AssertionError(
            f"coarse Level batch has {got.shape} entries for {len(ids)} days")
    if not np.allclose(got, wanted, atol=1e-5, rtol=0.0):
        worst = int(np.argmax(np.abs(got - wanted)))
        raise AssertionError(
            "the batch coarse Level is not the registered LOCAL_TRAIN artifact "
            f"(worst day {list(ids)[worst]}: batch {got[worst]!r} vs panel "
            f"{wanted[worst]!r})")


def stage2_objective(variant: str, output, targets, stats: LocalNormalizationStats,
                     valid_mask=None, *, untied_stats: Optional[UntiedMassStats] = None
                     ) -> dict:
    """The registered objective for one variant (protocol section 10).

    ``R3`` trains on normalised q reconstruction alone: it emits an unstructured
    vector, so there is no coordinate of its own to supervise, and applying the
    coordinate terms to its *derived* decomposition would be supervision the
    parameterization cannot act on.

    ``R2`` trains on its **native** signed masses ``A+*``/``A-*`` and the q-sign
    simplices.  It gets no ``delta`` loss and no centred-``A`` loss: those are R1's
    coordinates, and supervising R2 with them would score a parameterization
    against a decomposition it does not have.
    """
    from mvp.hch_residual_complete_local_v4_1 import (R0_STACK_ZERO_SUM,
                                                      R2_UNTIED_Q, R3_DIRECT_Q)

    mask = valid_mask
    l_rec_q = local_reconstruction_loss(output.local, targets.q, stats.s_q, mask).mean()
    if variant == R3_DIRECT_Q:
        return {"loss": l_rec_q, "l_rec": l_rec_q}

    if variant == R0_STACK_ZERO_SUM:
        # R0 has no delta coordinate; its target is the centred rho only.
        l_rec = local_reconstruction_loss(output.centered_local, targets.rho,
                                          stats.s_q, mask).mean()
        l_A = centered_mass_loss(output.amplitude, targets.amplitude, stats.s_A).mean()
        l_S = shape_loss(output.shape_positive, output.shape_negative,
                         targets.shape_positive, targets.shape_negative,
                         targets.amplitude)
        total = l_rec + (l_A + l_S) / 2.0
        return {"loss": total, "l_rec": l_rec, "l_A": l_A, "l_S": l_S}

    if variant == R2_UNTIED_Q:
        if untied_stats is None:
            raise ValueError("R2 requires its native signed-mass scales")
        if not isinstance(untied_stats, UntiedMassStats):
            raise TypeError("R2 mass scales must be an UntiedMassStats")
        from mvp.hch_residual_complete_local_v4_1.comparators import native_masses

        mass_plus, mass_minus = native_masses(output)
        native = untied_mass_targets(targets.q, mask)
        l_Ap = centered_mass_loss(mass_plus, native.a_plus,
                                  untied_stats.s_a_plus).mean()
        l_Am = centered_mass_loss(mass_minus, native.a_minus,
                                  untied_stats.s_a_minus).mean()
        l_S = shape_loss(output.shape_positive, output.shape_negative,
                         native.shape_plus, native.shape_negative,
                         native.a_plus + native.a_minus)
        total = l_rec_q + (l_Ap + l_Am + l_S) / 3.0
        return {"loss": total, "l_rec": l_rec_q, "l_A_plus": l_Ap,
                "l_A_minus": l_Am, "l_S": l_S}

    # R1: the residual-complete objective over the full q.
    l_delta = delta_loss(output.delta, targets.delta, stats.s_delta).mean()
    l_A = centered_mass_loss(output.amplitude, targets.amplitude, stats.s_A).mean()
    l_S = shape_loss(output.shape_positive, output.shape_negative,
                     targets.shape_positive, targets.shape_negative,
                     targets.amplitude)
    total = l_rec_q + (l_delta + l_A + l_S) / 3.0
    return {"loss": total, "l_rec": l_rec_q, "l_delta": l_delta,
            "l_A": l_A, "l_S": l_S}


def _stage2_batch(panel: LocalTrainPanel, ids: Sequence[int], feature_builder,
                  device):
    """One Stage-2 batch assembled from the panel's post-Level q history."""
    host = torch.as_tensor(panel.host_forecast[panel.positions(ids)],
                           dtype=torch.float32, device=device)
    mask = torch.as_tensor(panel.valid_mask[panel.positions(ids)],
                           dtype=torch.float32, device=device)
    past_q = torch.as_tensor(panel.history_windows(ids), dtype=torch.float32,
                             device=device)
    past_rho = past_q - past_q.mean(dim=-1, keepdim=True)
    return feature_builder(
        host=host, valid_mask=mask, base_tokens=build_base_tokens(host, mask),
        coarse_level=torch.as_tensor(panel.level_for_days(ids),
                                     dtype=torch.float32, device=device),
        past_q=past_q, past_rho=past_rho)


def panel_targets_for(panel: LocalTrainPanel, ids: Sequence[int]):
    """Residual-complete targets for exactly ``ids``, from the panel's rows."""
    rows = panel.positions(ids)
    return build_residual_complete_targets(
        torch.as_tensor(panel.residual[rows], dtype=torch.float32),
        torch.as_tensor(panel.coarse_level[rows], dtype=torch.float32),
        torch.as_tensor(panel.valid_mask[rows], dtype=torch.float32))


def state_dict_sha256(state_dict) -> str:
    """Order-independent digest of a model's tensors.

    The initial-state hash is the evidence that ``seed_everything`` ran *before*
    ``build_candidate``: same seed must reproduce it, different seeds must not.
    A hash taken after training would prove nothing, because the data order is
    seeded independently of the initialization.
    """
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(np.ascontiguousarray(
            tensor.detach().cpu().numpy(), dtype=np.float32).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def train_stage2(variant: str, level_model, config, panel: LocalTrainPanel,
                 seed: int, device="cpu", *, authorized: bool = False) -> dict:
    """Fit one Stage-2 candidate.

    Refuses to run without ``authorized=True``.  Harness preparation implements
    this entry so the run has exactly one code path; it does not call it.

    ``seed_everything(seed)`` runs **before** ``build_candidate``.  The other order
    leaves parameter initialization entirely unseeded, so seeds 7/17/37 would
    select three data orders over one shared initialization — not the three
    independent fits the protocol registers.
    """
    if not authorized:
        raise PermissionError(
            "Stage-2 fits require explicit scientific-execution authorization; "
            "harness preparation does not run them")

    from mvp.hch_residual_complete_local_v4_1 import build_candidate

    from core import ResidualCompleteLocalFeatureBuilder
    from level_condition_stats import assert_is_level_condition_stats
    from rcl_contract import assert_stage1_frozen, stage2_optimizer

    conditioner = assert_is_level_condition_stats(panel.level_condition_stats)
    for param in level_model.parameters():
        param.requires_grad_(False)
    # Seed first: this is the initialization the registered seed selects.
    seed_everything(int(seed))
    model = build_candidate(variant, config,
                            level_center=conditioner.center,
                            level_scale=conditioner.scale,
                            horizon=panel.residual.shape[-1]).to(device)
    initial_state_sha256 = state_dict_sha256(model.state_dict())
    assert_stage1_frozen(level_model, model)
    optimizer = stage2_optimizer(model, level_model, lr=LOCAL_RECIPE["lr"],
                                 weight_decay=LOCAL_RECIPE["weight_decay"])

    feature_builder = ResidualCompleteLocalFeatureBuilder(config, panel.stats)
    torch.manual_seed(int(seed))
    best = float("inf")
    best_state, best_epoch, wait, epoch = None, 0, 0, 0
    for epoch in range(1, LOCAL_RECIPE["max_epochs"] + 1):
        model.train()
        order = np.random.default_rng(seed * 1000003 + epoch).permutation(
            list(panel.fit_ids))
        for start in range(0, len(order), LOCAL_RECIPE["batch_size"]):
            chunk = [int(d) for d in order[start:start + LOCAL_RECIPE["batch_size"]]]
            batch = _stage2_batch(panel, chunk, feature_builder, device)
            assert_coarse_level_matches_panel(batch.coarse_level, panel, chunk)
            targets = panel_targets_for(panel, chunk)
            terms = stage2_objective(variant, model(batch), targets, panel.stats,
                                     batch.valid_mask,
                                     untied_stats=panel.untied_stats)
            optimizer.zero_grad(set_to_none=True)
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           LOCAL_RECIPE["grad_clip"])
            optimizer.step()
        score = _inner_tail_q_mae(model, panel, feature_builder, device)
        if score < best - 1e-7:
            best, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= LOCAL_RECIPE["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"variant": variant, "seed": int(seed), "state_dict": best_state,
            "best_epoch": best_epoch, "epochs_run": epoch,
            "inner_tail_q_mae": best,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": state_dict_sha256(best_state),
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "level_condition_fingerprint": conditioner.fingerprint(),
            "untied_mass_scale_fingerprint": (
                panel.untied_stats.fingerprint()
                if isinstance(panel.untied_stats, UntiedMassStats) else None),
            "fit_ids": tuple(int(d) for d in panel.fit_ids),
            "early_ids": tuple(int(d) for d in panel.early_ids)}


def _inner_tail_q_mae(model, panel: LocalTrainPanel, feature_builder,
                      device) -> float:
    chunk = [int(d) for d in panel.early_ids]
    model.eval()
    with torch.no_grad():
        batch = _stage2_batch(panel, chunk, feature_builder, device)
        out = model(batch)
    truth = torch.as_tensor(panel.residual[panel.positions(chunk)],
                            dtype=torch.float32, device=device) \
        - batch.coarse_level.unsqueeze(-1)
    return float(masked_mae(out.local, truth, batch.valid_mask).mean()
                 / max(float(panel.stats.s_q), 1e-6))
