"""The seven-record honest prequential safety bank.

This module owns exactly one object: a chronological store of the last
``HISTORY_DAYS`` **completed and honest** candidate/residual pairs.  It knows
nothing about files, dates, markets, hosts or datasets -- the experiment harness
owns disk persistence and split provenance, and this module only enforces the
legality of what it is handed.  That separation is deliberate: an evidence store
that can reach the file system is a store whose honesty cannot be audited by
reading it.

``W = 7`` is a single structural constant
------------------------------------------------
There is no window length to configure, search or tune.  Seven is the minimum
complete weekly support, it is the same temporal scale the candidate generator's
own history branch consumes, and it keeps the safety bank local enough to describe
the current regime.  It is a structural default, not a learned optimum, and the
same constant is used by the warm start: the deployment controller starts with
exactly the same number of records it will maintain.  A bank with a configurable
capacity would be a window search under another name.

What makes a record honest
--------------------------
A record pairs a candidate that was **generated and persisted before its own
residual was available** with the residual that later arrived for it.  The store
enforces this from timestamps rather than trusting the caller:

* ``candidate_created_at < ordinal``      the candidate predates its own outcome;
* ``revealed_at >= ordinal``              the residual is not dated before its slot;
* ``revealed_at > candidate_created_at``  the residual strictly postdates the candidate.

The first condition is the one that matters scientifically.  If a candidate could
be created after its own residual was known, the safety bank would be scoring
foresight, every radius would be optimistic, and the controller would look safest
exactly where it had cheated.  ``InSampleCandidateError`` is raised rather than
merely logged.

Warm start
----------
``POST_TRAIN`` warm start must be built from chronological out-of-fold or
prequential predictions, never from a final model's in-sample predictions: an
in-sample candidate is fitted to the very residual it is later scored against, so
it is not evidence about deployment.  :func:`select_warm_start` therefore filters
on provenance, takes the **last** ``HISTORY_DAYS`` records in chronological order,
and refuses rather than improvising when fewer than seven exist.  Shortening the
window to whatever happens to be available would silently change the estimator;
the caller is expected to stop with a blocker instead.

Non-responsibilities
--------------------
No gradient, no parameters, no ``nn.Module``, no learned weighting, no
de-duplication heuristic, no imputation of missing residuals, and no silent
tolerance of a partially populated bank.  A record whose residual is still
pending is simply **not evidence** and is never counted, sampled around or
approximated.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .contracts import SafetyEvidence, WARM_START_PROVENANCE

__all__ = [
    "HISTORY_DAYS",
    "HistoryError",
    "DuplicateDeliveryError",
    "DishonestRecordError",
    "InSampleCandidateError",
    "IncompleteHistoryError",
    "InsufficientWarmStartError",
    "SafetyEvidenceBank",
    "select_warm_start",
]

#: Number of completed honest records the deployment safety layer consumes.
#: A single structural constant: not a configuration knob and not tunable.
HISTORY_DAYS = 7


class HistoryError(RuntimeError):
    """Base class for every refusal raised by the safety bank."""


class DuplicateDeliveryError(HistoryError):
    """A delivery identifier was seen before, including one already evicted."""


class DishonestRecordError(HistoryError):
    """A record violates the chronology contract between candidate and residual."""


class InSampleCandidateError(DishonestRecordError):
    """The candidate was created no earlier than its own residual."""


class IncompleteHistoryError(HistoryError):
    """The bank does not hold exactly ``HISTORY_DAYS`` completed records."""


class InsufficientWarmStartError(HistoryError):
    """Fewer than ``HISTORY_DAYS`` legal warm-start records were supplied."""


def _ordered_ids(records: Iterable[SafetyEvidence]) -> List[str]:
    return [r.delivery_id for r in records]


class SafetyEvidenceBank:
    """Chronological store of the last ``HISTORY_DAYS`` completed honest records.

    The bank takes no capacity argument.  Capacity is ``HISTORY_DAYS`` and nothing
    else, so there is no second place where a window length could be chosen.

    Pending candidates are held but are not evidence: only records that have
    received their residual are exposed to the safety layer, and
    :meth:`completed` refuses to hand back a short bank rather than letting the
    controller quietly run on fewer observations than the design specifies.
    """

    def __init__(self) -> None:
        self._records: List[SafetyEvidence] = []
        self._seen_delivery_ids: set = set()

    # -- introspection ------------------------------------------------------

    def __len__(self) -> int:
        """Number of completed records currently held."""
        return sum(1 for r in self._records if r.completed)

    @property
    def capacity(self) -> int:
        return HISTORY_DAYS

    @property
    def pending(self) -> Tuple[SafetyEvidence, ...]:
        """Candidates whose residual has not arrived.  Never evidence."""
        return tuple(r for r in self._records if not r.completed)

    def records(self) -> Tuple[SafetyEvidence, ...]:
        """All held records, oldest first, pending ones included."""
        return tuple(self._records)

    def completed(self) -> Tuple[SafetyEvidence, ...]:
        """Exactly ``HISTORY_DAYS`` completed records, oldest first.

        Raises :class:`IncompleteHistoryError` otherwise.  The controller has no
        legitimate use for a short history: the design fixes W = 7, so a short
        bank means the chronology was constructed wrongly upstream and must be
        reported rather than absorbed.
        """
        done = tuple(r for r in self._records if r.completed)
        if len(done) != HISTORY_DAYS:
            raise IncompleteHistoryError(
                f"safety bank holds {len(done)} completed records, "
                f"exactly {HISTORY_DAYS} are required"
            )
        return done

    def is_ready(self) -> bool:
        """Whether :meth:`completed` would succeed."""
        return len(self) == HISTORY_DAYS

    # -- mutation -----------------------------------------------------------

    def append_candidate(self, *, delivery_id: str, ordinal: int,
                         candidate_created_at: int,
                         shape_positive: torch.Tensor,
                         shape_negative: torch.Tensor,
                         correction: torch.Tensor,
                         provenance: str,
                         origin_id: str = "",
                         valid_mask: Optional[torch.Tensor] = None,
                         metadata: Optional[Mapping[str, Any]] = None) -> SafetyEvidence:
        """Register a candidate correction **before** its residual is known.

        ``ordinal`` is the chronological position of the outcome this candidate
        will be scored against; ``candidate_created_at`` is when the candidate was
        generated.  ``candidate_created_at >= ordinal`` means the residual was
        already available and the record can never be evidence, so it is refused.
        """
        if not delivery_id:
            raise DishonestRecordError("delivery_id must be a non-empty identifier")
        if delivery_id in self._seen_delivery_ids:
            raise DuplicateDeliveryError(
                f"delivery_id {delivery_id!r} was already delivered to this bank"
            )
        if provenance not in WARM_START_PROVENANCE:
            raise DishonestRecordError(
                f"provenance {provenance!r} is not legal safety evidence; a record "
                "generated with knowledge of its own outcome is never evidence"
            )
        if candidate_created_at >= ordinal:
            raise InSampleCandidateError(
                f"candidate for ordinal {ordinal} was created at "
                f"{candidate_created_at}, no earlier than its own outcome"
            )
        if self._records and ordinal <= self._records[-1].ordinal:
            raise DishonestRecordError(
                f"ordinal {ordinal} does not follow the latest ordinal "
                f"{self._records[-1].ordinal}; the bank is chronological"
            )

        evidence = SafetyEvidence(
            delivery_id=delivery_id,
            ordinal=int(ordinal),
            shape_positive=shape_positive,
            shape_negative=shape_negative,
            candidate_correction=correction,
            candidate_created_at=int(candidate_created_at),
            provenance=provenance,
            origin_id=origin_id,
            valid_mask=valid_mask,
            metadata=dict(metadata or {}),
        )
        self._seen_delivery_ids.add(delivery_id)
        self._records.append(evidence)
        self._evict_oldest_completed()
        return evidence

    def attach_residual(self, delivery_id: str, residual: torch.Tensor,
                        revealed_at: int) -> SafetyEvidence:
        """Complete a pending candidate with the residual that arrived for it.

        Refuses a residual dated at or before the candidate, and a residual dated
        before its own outcome slot.  A completed record is never rewritten.
        """
        for index, record in enumerate(self._records):
            if record.delivery_id != delivery_id:
                continue
            if record.completed:
                raise DishonestRecordError(
                    f"record {delivery_id!r} already carries a residual; "
                    "evidence is append-only"
                )
            if revealed_at < record.ordinal:
                raise DishonestRecordError(
                    f"residual for {delivery_id!r} is dated {revealed_at}, "
                    f"before its own slot {record.ordinal}"
                )
            if revealed_at <= record.candidate_created_at:
                raise DishonestRecordError(
                    f"residual for {delivery_id!r} is dated {revealed_at}, "
                    f"not after its candidate at {record.candidate_created_at}"
                )
            completed = replace(record, host_residual=residual, revealed_at=int(revealed_at))
            self._records[index] = completed
            self._evict_oldest_completed(protect=delivery_id)
            return completed
        raise HistoryError(f"no pending candidate named {delivery_id!r}")

    def _evict_oldest_completed(self, protect: Optional[str] = None) -> None:
        """Keep the most recent ``HISTORY_DAYS`` realised outcomes.

        Eviction is by completion, not by arrival: a pending candidate is not an
        outcome, and dropping one would shrink the evidence set below the design
        without anyone deciding to.  A record that has just received its residual
        is never evicted by the call that completed it.
        """
        while True:
            done = [i for i, r in enumerate(self._records) if r.completed]
            if len(done) <= HISTORY_DAYS:
                return
            deletable = [i for i in done if self._records[i].delivery_id != protect]
            if not deletable:  # pragma: no cover - latest ordinal is never the oldest
                return
            del self._records[deletable[0]]

    def clear(self) -> None:
        """Drop all held records.  The delivered-identifier ledger is kept, so a
        cleared bank still refuses to accept the same delivery twice."""
        self._records.clear()

    # -- consumption --------------------------------------------------------

    def as_arrays(self) -> Dict[str, Any]:
        """Completed evidence as stacked tensors, oldest first.

        Shapes are ``(HISTORY_DAYS, H)``; ``valid_mask`` is ``(HISTORY_DAYS, H)``
        or ``None`` when every record abstained from masking.  Raises
        :class:`IncompleteHistoryError` when the bank is not full.
        """
        done = self.completed()
        masks = [r.valid_mask for r in done]
        return {
            "delivery_ids": _ordered_ids(done),
            "shape_positive": torch.stack([r.shape_positive for r in done]),
            "shape_negative": torch.stack([r.shape_negative for r in done]),
            "correction": torch.stack([r.candidate_correction for r in done]),
            "residual": torch.stack([r.host_residual for r in done]),
            "valid_mask": torch.stack(masks) if all(m is not None for m in masks) else None,
        }

    # -- serialization ------------------------------------------------------

    def to_state(self) -> Dict[str, Any]:
        """Plain-data snapshot.  Writing it anywhere is the caller's business."""

        def _tensor(value: Optional[torch.Tensor]):
            if value is None:
                return None
            return value.detach().cpu().tolist()

        return {
            "history_days": HISTORY_DAYS,
            "delivered_ids": sorted(self._seen_delivery_ids),
            "records": [
                {
                    "delivery_id": r.delivery_id,
                    "origin_id": r.origin_id,
                    "ordinal": r.ordinal,
                    "candidate_created_at": r.candidate_created_at,
                    "provenance": r.provenance,
                    "shape_positive": _tensor(r.shape_positive),
                    "shape_negative": _tensor(r.shape_negative),
                    "candidate_correction": _tensor(r.candidate_correction),
                    "host_residual": _tensor(r.host_residual),
                    "revealed_at": r.revealed_at,
                    "valid_mask": _tensor(r.valid_mask),
                    "metadata": dict(r.metadata),
                }
                for r in self._records
            ],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "SafetyEvidenceBank":
        """Rebuild a bank from :meth:`to_state` output."""
        if int(state.get("history_days", HISTORY_DAYS)) != HISTORY_DAYS:
            raise HistoryError(
                "serialized bank was written for a different history length; "
                "the window is a structural constant and is not migrated"
            )
        bank = cls()
        bank._seen_delivery_ids = set(state.get("delivered_ids", ()))
        for entry in state.get("records", ()):
            mask = entry.get("valid_mask")
            record = SafetyEvidence(
                delivery_id=entry["delivery_id"],
                ordinal=int(entry["ordinal"]),
                shape_positive=torch.tensor(entry["shape_positive"], dtype=torch.float32),
                shape_negative=torch.tensor(entry["shape_negative"], dtype=torch.float32),
                candidate_correction=torch.tensor(entry["candidate_correction"],
                                                  dtype=torch.float32),
                candidate_created_at=int(entry["candidate_created_at"]),
                provenance=entry["provenance"],
                origin_id=entry.get("origin_id", ""),
                valid_mask=None if mask is None else torch.tensor(mask, dtype=torch.float32),
                metadata=dict(entry.get("metadata", {})),
                host_residual=(None if entry.get("host_residual") is None
                               else torch.tensor(entry["host_residual"], dtype=torch.float32)),
                revealed_at=(None if entry.get("revealed_at") is None
                             else int(entry["revealed_at"])),
            )
            bank._records.append(record)
        return bank

    @classmethod
    def from_warm_start(cls, records: Sequence[SafetyEvidence]) -> "SafetyEvidenceBank":
        """Build a full bank from legal chronological warm-start records."""
        bank = cls()
        for record in select_warm_start(records):
            if not record.completed:
                raise IncompleteHistoryError(
                    f"warm-start record {record.delivery_id!r} has no residual"
                )
            if record.candidate_created_at >= record.ordinal:
                raise InSampleCandidateError(
                    f"warm-start record {record.delivery_id!r} was created at "
                    f"{record.candidate_created_at}, no earlier than its own outcome"
                )
            bank._seen_delivery_ids.add(record.delivery_id)
            bank._records.append(record)
        if not bank.is_ready():  # pragma: no cover - select_warm_start guarantees this
            raise IncompleteHistoryError("warm start did not yield a full bank")
        return bank


def select_warm_start(records: Sequence[SafetyEvidence]) -> Tuple[SafetyEvidence, ...]:
    """The last ``HISTORY_DAYS`` legal chronological warm-start records.

    Legal provenance is out-of-fold prequential or plain prequential: both are
    generated without the outcome they are later scored against.  In-sample
    predictions are excluded, and no other provenance is accepted.

    There is no length argument.  Fewer than ``HISTORY_DAYS`` legal completed
    records is a blocker to be reported, not a shorter window to be improvised:
    W is part of the estimator, so changing it silently would change what is being
    deployed.
    """
    legal = [r for r in records if r.provenance in WARM_START_PROVENANCE]
    complete = [r for r in legal if r.completed]
    ordered = sorted(complete, key=lambda r: r.ordinal)
    if len(ordered) < HISTORY_DAYS:
        raise InsufficientWarmStartError(
            f"warm start supplied {len(ordered)} legal completed records; "
            f"{HISTORY_DAYS} are required and W is not shortened"
        )
    return tuple(ordered[-HISTORY_DAYS:])
