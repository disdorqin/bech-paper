"""TRAIN-only conditioning statistics for the frozen coarse Level.

The Stage-2 model standardises the coarse Level before its readouts see it:

    conditioned = (b_hat - center) / scale

``center`` and ``scale`` are part of the *fitted pipeline*, not of the model, so
the run must decide where they come from and freeze that decision before any
candidate is evaluated.  This module is that decision, and it is deliberately not
``LocalNormalizationStats``:

* the Level condition is a **coarse-Level** quantity, so its stats are fitted from
  coarse-Level predictions.  ``LocalNormalizationStats`` describes the post-Level
  ``q``/``delta``/``A`` geometry; reusing its ``delta_center``/``s_delta`` here
  would scale the conditioner with the wrong distribution's spread, and
  ``LocalNormalizationStats.normalize_level_condition()`` is therefore **not** the
  Level-scaling authority anywhere in the runner.
* the Level values are **causal OOF** predictions on ``LOCAL_TRAIN`` days, never
  in-sample fitted values and never the final refit's own outputs.  An in-sample
  ``b_tilde`` is optimistic on the very days it was fitted on, so its median and
  MAD would not describe the conditioner the run actually feeds at inference.

Registered recipe (protocol section 7.1, gate report E3)::

    center = median(b_tilde)
    scale  = max(median(|b_tilde - center|), 1e-6)

The median convention is the repository's: for an even count the **lower** middle
value is returned, matching ``torch.median``/``nanmedian`` and the core's
``masked_median``.  Averaging the two middle values here would make this one
statistic disagree with every other robust scale in the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "LEVEL_CONDITION_EPS",
    "OofLevelRecord",
    "LevelConditionStats",
    "assert_is_level_condition_stats",
]

#: Registered clamp.  A degenerate OOF Level panel must not produce a zero divisor.
LEVEL_CONDITION_EPS = 1e-6

#: Provenances that may fit the conditioner.  Kept as a literal rather than
#: imported from ``core`` so this experiment-side object stays importable without
#: the active core on the path; ``test_*`` asserts it equals the core's
#: ``LOCAL_TRAIN_PROVENANCE``.
OOF_PROVENANCE = "oof"


def _lower_median(values: Sequence[float]) -> float:
    """Median with the repository's even-count convention (lower middle)."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("median of an empty sequence is undefined")
    return ordered[(len(ordered) - 1) // 2]


@dataclass(frozen=True)
class OofLevelRecord:
    """One causal OOF coarse-Level prediction on one TRAIN day."""

    day_index: int
    value: float
    provenance: str = OOF_PROVENANCE

    def __post_init__(self) -> None:
        if self.provenance != OOF_PROVENANCE:
            raise ValueError(
                f"a Level conditioner observation must be causal OOF, got "
                f"{self.provenance!r}: in-sample and final-refit Levels are not "
                "legal fitting observations")


@dataclass(frozen=True)
class LevelConditionStats:
    """Frozen ``(center, scale)`` for the Stage-2 Level conditioner.

    Fitted once, from ``LOCAL_TRAIN`` causal OOF Levels, and then reused
    unchanged for every VAL/EVAL day.  The object carries the days it was fitted
    from so a verifier can re-derive the panel instead of trusting the numbers.
    """

    center: float
    scale: float
    fitted_days: tuple[int, ...]
    n_observations: int

    def __post_init__(self) -> None:
        if not self.scale > 0.0:
            raise ValueError(f"level conditioning scale must be positive, got {self.scale}")
        if self.center != self.center or abs(self.center) == float("inf"):
            raise ValueError("level conditioning center must be finite")

    @classmethod
    def fit(cls, records: Iterable[OofLevelRecord],
            local_train_days: Sequence[int]) -> "LevelConditionStats":
        """Fit from ``LOCAL_TRAIN`` causal OOF Levels only.

        ``local_train_days`` is the registered ``B2 u B3 u B4`` day set.  Any
        record outside it is refused: warm-up and B1 days are history support, not
        Local training targets, and letting one into the conditioner would fit the
        scaling on days the Local target never uses.
        """
        allowed = set(int(d) for d in local_train_days)
        if not allowed:
            raise ValueError("LOCAL_TRAIN day set is empty; nothing to fit from")
        kept: list[OofLevelRecord] = []
        outside: list[int] = []
        for record in records:
            if not isinstance(record, OofLevelRecord):
                raise TypeError(
                    "the Level conditioner is fitted from OofLevelRecord "
                    f"observations; got {type(record).__name__}")
            if int(record.day_index) not in allowed:
                outside.append(int(record.day_index))
                continue
            kept.append(record)
        if outside:
            raise ValueError(
                "Level conditioner observations outside LOCAL_TRAIN: "
                f"{sorted(set(outside))[:8]}"
                + (" ..." if len(set(outside)) > 8 else ""))
        if not kept:
            raise ValueError("no LOCAL_TRAIN causal OOF Level observations supplied")

        values = [float(r.value) for r in kept]
        center = _lower_median(values)
        scale = max(_lower_median([abs(v - center) for v in values]),
                    LEVEL_CONDITION_EPS)
        return cls(center=center, scale=scale,
                   fitted_days=tuple(sorted(int(r.day_index) for r in kept)),
                   n_observations=len(kept))

    def normalize(self, coarse_level):
        """``(b_hat - center) / scale``.  The only legal Level scaling."""
        return (coarse_level - self.center) / self.scale

    def fingerprint(self) -> str:
        """Stable identity of the frozen conditioner, for provenance records."""
        import hashlib

        payload = f"{self.center!r}|{self.scale!r}|{self.n_observations}|" \
                  f"{','.join(str(d) for d in self.fitted_days)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_is_level_condition_stats(stats) -> LevelConditionStats:
    """Type gate: the Level conditioner is this class and nothing else.

    In particular a ``LocalNormalizationStats`` cannot be substituted.  That object
    describes the q/delta/A target geometry; using it to scale the coarse Level
    would condition Stage 2 on the wrong distribution while looking like a
    legitimate normalization step.
    """
    if not isinstance(stats, LevelConditionStats):
        raise TypeError(
            "the Stage-2 Level conditioner must be a LevelConditionStats fitted "
            "from LOCAL_TRAIN causal OOF coarse Levels; "
            f"got {type(stats).__name__}")
    return stats
