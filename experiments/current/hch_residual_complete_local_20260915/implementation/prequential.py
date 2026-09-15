"""VAL chronological CAL/EVAL prequential evaluation.

VAL splits chronologically: CAL is the first half, EVAL the second.  Both are
walked in day order, and each day follows the same steps in the same order::

    predict Level -> predict Local -> persist the emission -> reveal the target
    -> compute q -> append q to the history

The order is the whole content of "prequential".  A day's Local correction is
computed from history containing only *already revealed* days, so the correction
for day ``d`` cannot depend on ``y_d``.  :class:`PrequentialQHistory` enforces the
append half (no replay, strictly increasing); :func:`run_val_prequential` enforces
the reveal half by calling the caller's reveal callback only after the day's
emission has been recorded — the callback is the *only* way this module can learn
a target, so a reordering is not expressible rather than merely discouraged.

The primary verdict uses the raw model output.  One nonnegative scalar may be
fitted on CAL and reported as a secondary diagnostic, but it is computed after
EVAL is emitted and is returned under a separate key, so nothing downstream can
read the calibrated series where it meant to read the raw one.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from core import apply_mae_scalar, build_q, fit_mae_scalar

from rcl_contract import VAL_CAL_FRACTION, PrequentialQHistory

__all__ = [
    "ValSplit",
    "chronological_val_split",
    "ValDayEmission",
    "ValRun",
    "seed_windows_from_oof",
    "run_val_prequential",
    "secondary_calibration",
]


@dataclass(frozen=True)
class ValSplit:
    """Chronological CAL/EVAL halves of the VAL segment."""

    cal_ids: tuple[int, ...]
    eval_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.cal_ids or not self.eval_ids:
            raise ValueError("CAL and EVAL must both be non-empty")
        if max(self.cal_ids) >= min(self.eval_ids):
            raise AssertionError(
                "CAL and EVAL must be a chronological split: CAL is the earlier "
                f"half, but CAL ends at day {max(self.cal_ids)} and EVAL starts at "
                f"day {min(self.eval_ids)}")

    def as_dict(self) -> dict:
        return {"cal_ids": list(self.cal_ids), "eval_ids": list(self.eval_ids),
                "cal_fraction": VAL_CAL_FRACTION}


def chronological_val_split(val_ids: Sequence[int]) -> ValSplit:
    """First half CAL, second half EVAL, in the order VAL is delivered."""
    ordered = sorted(int(d) for d in val_ids)
    if len(ordered) < 2:
        raise ValueError("VAL needs at least two days to split into CAL and EVAL")
    cut = min(max(int(len(ordered) * VAL_CAL_FRACTION), 1), len(ordered) - 1)
    return ValSplit(cal_ids=tuple(ordered[:cut]), eval_ids=tuple(ordered[cut:]))


@dataclass(frozen=True)
class ValDayEmission:
    """What the model emitted for one VAL day, recorded **before** its reveal.

    The whole reported coordinate set is kept, not just the correction: the
    protocol's Shape and mass diagnostics compare each coordinate against the
    exact decomposition of the realised residual, so dropping the Shapes here
    would make those diagnostics uncomputable without a second model call.
    """

    day_index: int
    coarse_level: float
    delta: float
    amplitude: float
    shape_positive: "np.ndarray"
    shape_negative: "np.ndarray"
    local: "np.ndarray"
    full_correction: "np.ndarray"
    history_length: int
    #: R2's own coordinates.  ``None`` for every variant that does not emit them,
    #: which is the honest report: R1's learned mass is ``amplitude``, and filling
    #: these in from a decomposition of ``local`` would put a derived number where
    #: the verifier is checking for a native one.
    mass_plus: Optional[float] = None
    mass_minus: Optional[float] = None


@dataclass
class ValRun:
    """The VAL emissions and the reveal log, in delivery order."""

    split: ValSplit
    emissions: list[ValDayEmission] = field(default_factory=list)
    revealed: dict[int, "np.ndarray"] = field(default_factory=dict)
    q_history: dict[int, "np.ndarray"] = field(default_factory=dict)
    order: list[tuple[int, str]] = field(default_factory=list)

    def emission(self, day: int) -> ValDayEmission:
        for item in self.emissions:
            if item.day_index == int(day):
                return item
        raise KeyError(f"day {day} has no emission")

    def emissions_for(self, ids: Sequence[int]) -> list[ValDayEmission]:
        """Emissions for ``ids`` in the order given.

        One pass over the emission log rather than one :meth:`emission` call per
        day, so a per-metric sweep over the whole EVAL half stays linear.  A day
        with no emission is a hard error: silently skipping it would let a metric
        be averaged over fewer days than the split declares.
        """
        by_day = {item.day_index: item for item in self.emissions}
        missing = [int(d) for d in ids if int(d) not in by_day]
        if missing:
            raise KeyError(f"days without an emission: {missing[:8]}")
        return [by_day[int(d)] for d in ids]

    def predictions(self, ids: Sequence[int]) -> "np.ndarray":
        """Raw full corrections for ``ids``.  The primary verdict reads this."""
        return np.stack([self.emission(d).full_correction for d in ids])

    def as_dict(self) -> dict:
        return {
            "cal_ids": list(self.split.cal_ids),
            "eval_ids": list(self.split.eval_ids),
            "n_emissions": len(self.emissions),
            "n_revealed": len(self.revealed),
            "delivery_order": [f"{d}:{kind}" for d, kind in self.order],
            "emission_precedes_reveal": self.emission_precedes_reveal(),
        }

    def emission_precedes_reveal(self) -> bool:
        """Every day's emission is logged before its reveal, or the run is void."""
        emitted: set[int] = set()
        for day, kind in self.order:
            if kind == "emit":
                if day in emitted:
                    return False
                emitted.add(day)
            elif kind == "reveal":
                if day not in emitted:
                    return False
        return True


def seed_windows_from_oof(oof_panel, residual_by_day: Callable,
                          tail_days: Sequence[int]) -> list:
    """The last legal causal q windows from TRAIN/B4, in chronological order.

    Only already-revealed TRAIN days may seed it, and only in chronological order:
    the seed is the *tail of B4*, the last block whose q is causally defined.
    Seeding from a later day would hand VAL's first prediction a history its
    predecessors never had.

    ``q`` is built through the core's :func:`core.build_q` from the day's realised
    residual and its **causal OOF** Level — the same construction the Local
    training targets use, so VAL's first history entry is in the space Local was
    trained on rather than a differently derived one.

    Returns a plain list of ``(H,)`` windows; :func:`run_val_prequential` wraps it
    in the :class:`~rcl_contract.PrequentialQHistory` that enforces the append
    discipline, so the seed and the VAL days pass through the same monotonicity
    rule instead of the seed being trusted by construction.
    """
    days = sorted(int(d) for d in tail_days)
    absent = [d for d in days if d not in oof_panel.values]
    if absent:
        raise KeyError(f"seed days without a causal OOF Level: {absent[:8]}")
    windows = []
    for day in days:
        residual = np.asarray(residual_by_day(day), dtype=np.float32)
        window = build_q(
            torch.as_tensor(residual, dtype=torch.float32).reshape(1, -1),
            torch.as_tensor([oof_panel.values[day]], dtype=torch.float32),
            None).numpy()[0]
        if not np.isfinite(window).all():
            raise ValueError(f"seed day {day} produced a non-finite q window")
        windows.append(window)
    return windows


def run_val_prequential(model, build_batch: Callable, reveal: Callable,
                        val_ids: Sequence[int], seed_windows: Sequence,
                        device="cpu") -> ValRun:
    """Walk VAL in day order under the prequential discipline.

    ``build_batch(day, history_windows) -> (coarse_level_float, LocalOriginBatch)``
    is the caller's prediction step: it may read the day's legal inputs and the
    already-revealed history it is handed, and nothing else.  ``reveal(day) ->
    (residual, valid_mask)`` is the **only** way this function learns a target, and
    it is called after the day's emission has been appended.  Passing a reveal
    callback that reads a target early is therefore a visible, single-site decision
    rather than something scattered through the loop.
    """
    split = chronological_val_split(val_ids)
    run = ValRun(split=split)
    history = PrequentialQHistory(list(seed_windows))

    for day in list(split.cal_ids) + list(split.eval_ids):
        day = int(day)
        coarse_level, batch = build_batch(day, history.windows())
        with torch.no_grad():
            out = model(batch)
        native_plus = getattr(out, "mass_plus", None)
        native_minus = getattr(out, "mass_minus", None)
        run.emissions.append(ValDayEmission(
            day_index=day,
            coarse_level=float(coarse_level),
            delta=float(out.delta.detach().cpu().reshape(-1)[0]),
            amplitude=float(out.amplitude.detach().cpu().reshape(-1)[0]),
            shape_positive=out.shape_positive.detach().cpu().numpy()[0],
            shape_negative=out.shape_negative.detach().cpu().numpy()[0],
            local=out.local.detach().cpu().numpy()[0],
            full_correction=out.full_correction.detach().cpu().numpy()[0],
            history_length=len(history.windows()),
            mass_plus=(None if native_plus is None
                       else float(native_plus.detach().cpu().reshape(-1)[0])),
            mass_minus=(None if native_minus is None
                        else float(native_minus.detach().cpu().reshape(-1)[0])),
        ))
        run.order.append((day, "emit"))

        # The emission is persisted; only now does the target exist for us.
        residual, valid_mask = reveal(day)
        residual = np.asarray(residual, dtype=np.float32).reshape(1, -1)
        q = build_q(
            torch.as_tensor(residual, dtype=torch.float32),
            torch.as_tensor([coarse_level], dtype=torch.float32),
            torch.as_tensor(np.asarray(valid_mask, dtype=np.float32).reshape(1, -1)),
        ).numpy()[0]
        run.revealed[day] = residual[0]
        run.q_history[day] = q
        history.reveal_and_append(day, q)
        run.order.append((day, "reveal"))

    if not run.emission_precedes_reveal():
        raise AssertionError(
            "a VAL day was revealed before its emission was persisted")
    return run


def secondary_calibration(run: ValRun, residual_for: Callable) -> dict:
    """Fit one nonnegative scalar on CAL; report it, never apply it to raw EVAL.

    The calibrated series is returned under its own key.  :func:`core.fit_mae_scalar`
    is the pooled MAE scalar the protocol permits as a *secondary diagnostic*; the
    primary verdict reads :meth:`ValRun.predictions`, which this never modifies.
    """
    cal_ids = list(run.split.cal_ids)
    eval_ids = list(run.split.eval_ids)
    cal_correction = np.stack([run.emission(d).full_correction for d in cal_ids])
    cal_residual = np.stack([np.asarray(residual_for(d), dtype=np.float32)
                             for d in cal_ids])
    eval_correction = np.stack([run.emission(d).full_correction for d in eval_ids])
    scalar = float(fit_mae_scalar(cal_correction, cal_residual))
    return {
        "alpha": scalar,
        "cal_ids": cal_ids,
        "eval_ids": eval_ids,
        "eval_raw": eval_correction,
        "eval_calibrated": apply_mae_scalar(eval_correction, scalar),
        "calibrated_is_secondary_only": True,
    }
