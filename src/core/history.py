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

The trust boundary is one validator, and one owner
--------------------------------------------------
Every record is judged by a single internal validator, whichever door it arrives
through: live insertion, warm start, or deserialization.  Restore and warm start
are deliberately **not** weaker doors than :meth:`SafetyEvidenceBank.append_candidate`
-- a persisted or precomputed record that could not have been appended online is
refused outright rather than quietly repaired into legal evidence.  A malformed
record is a blocker to report, because the alternative is a safety radius computed
from evidence that never existed.

Tensors entering the bank become owned snapshots (``detach().clone().cpu()``), and
the public accessors hand back copies.  A frozen dataclass does not make a mutable
tensor immutable, so without this a caller could deliver a candidate, mutate its
correction in place afterwards, and silently rewrite the evidence the controller
had already scored -- and the bank would also keep an autograd graph alive.

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


def _owned(value: Optional[torch.Tensor], *, field: str) -> Optional[torch.Tensor]:
    """An owned, graph-free, CPU snapshot of one evidence tensor.

    The bank must not share storage with its caller: a tensor handed in and
    mutated afterwards would otherwise rewrite evidence that has already been
    scored, and a tensor carrying ``grad_fn`` would pin an autograd graph for the
    lifetime of the bank.  ``detach().clone().cpu()`` severs both.
    """
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise DishonestRecordError(
            f"evidence field {field!r} must be a torch.Tensor or None, "
            f"got {type(value).__name__}"
        )
    return value.detach().clone().cpu()


def _own_evidence(record: SafetyEvidence) -> SafetyEvidence:
    """The same record, holding only owned snapshots and a copied metadata map.

    Applied on the way in (insertion, warm start, restore) and again on the way
    out, so neither the caller's tensors nor the bank's own storage can be reached
    by in-place writes from outside.
    """
    return replace(
        record,
        shape_positive=_owned(record.shape_positive, field="shape_positive"),
        shape_negative=_owned(record.shape_negative, field="shape_negative"),
        candidate_correction=_owned(record.candidate_correction,
                                    field="candidate_correction"),
        host_residual=_owned(record.host_residual, field="host_residual"),
        valid_mask=_owned(record.valid_mask, field="valid_mask"),
        metadata=dict(record.metadata),
    )


def _validate_evidence(record: SafetyEvidence, *, context: str) -> None:
    """The single trust boundary for one record's internal honesty.

    This answers only "could this record have been produced honestly?" -- it says
    nothing about whether the record belongs in *this* bank, which is the ledger
    and chronology concern of :func:`_validate_sequence`.  Every path that admits
    evidence (live append/attach, warm start, restore) calls it, so no door is
    weaker than another.

    A completed record must carry a reveal time that postdates both its own
    outcome slot and its own candidate.  A pending record must carry neither a
    residual nor a reveal time.  Neither branch is repaired in place: a malformed
    record is a blocker, and converting it into legal evidence would fabricate
    exactly the honesty the bank exists to guarantee.
    """
    if not record.delivery_id or not isinstance(record.delivery_id, str):
        raise DishonestRecordError(
            f"{context}: delivery_id must be a non-empty identifier, "
            f"got {record.delivery_id!r}"
        )
    if record.provenance not in WARM_START_PROVENANCE:
        raise DishonestRecordError(
            f"{context}: provenance {record.provenance!r} is not legal safety "
            "evidence; a record generated with knowledge of its own outcome is "
            "never evidence"
        )
    if record.candidate_created_at >= record.ordinal:
        raise InSampleCandidateError(
            f"{context}: candidate for ordinal {record.ordinal} was created at "
            f"{record.candidate_created_at}, no earlier than its own outcome"
        )
    if record.completed:
        if record.revealed_at is None:
            raise DishonestRecordError(
                f"{context}: record {record.delivery_id!r} carries a residual but "
                "no reveal time; a residual with no chronology is not evidence"
            )
        if record.revealed_at < record.ordinal:
            raise DishonestRecordError(
                f"{context}: residual for {record.delivery_id!r} is dated "
                f"{record.revealed_at}, before its own slot {record.ordinal}"
            )
        if record.revealed_at <= record.candidate_created_at:
            raise DishonestRecordError(
                f"{context}: residual for {record.delivery_id!r} is dated "
                f"{record.revealed_at}, not after its candidate at "
                f"{record.candidate_created_at}"
            )
    elif record.revealed_at is not None:
        raise DishonestRecordError(
            f"{context}: record {record.delivery_id!r} has a reveal time "
            f"({record.revealed_at}) but no residual; pending evidence cannot "
            "have a reveal chronology"
        )


def _validate_sequence(records: Sequence[SafetyEvidence], *, context: str) -> None:
    """The records of one bank must be unique and strictly chronological.

    Ordinals are the only chronology the safety mathematics uses, so a repeated
    ordinal means two different outcomes claim the same slot and the window is not
    a window.  A repeated delivery identifier means a day is scored twice.
    """
    seen: set = set()
    previous: Optional[int] = None
    for record in records:
        if record.delivery_id in seen:
            raise DuplicateDeliveryError(
                f"{context}: delivery_id {record.delivery_id!r} appears more than "
                "once in the same bank"
            )
        seen.add(record.delivery_id)
        if previous is not None and record.ordinal <= previous:
            raise DishonestRecordError(
                f"{context}: ordinal {record.ordinal} does not strictly follow "
                f"{previous}; the bank is chronological"
            )
        previous = record.ordinal


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
        return tuple(_own_evidence(r) for r in self._records if not r.completed)

    def records(self) -> Tuple[SafetyEvidence, ...]:
        """All held records, oldest first, pending ones included.

        Snapshots, not the bank's own storage: mutating a tensor returned here
        must not be able to rewrite evidence that has already been scored.
        """
        return tuple(_own_evidence(r) for r in self._records)

    def completed(self) -> Tuple[SafetyEvidence, ...]:
        """Exactly ``HISTORY_DAYS`` completed records, oldest first.

        Raises :class:`IncompleteHistoryError` otherwise.  The controller has no
        legitimate use for a short history: the design fixes W = 7, so a short
        bank means the chronology was constructed wrongly upstream and must be
        reported rather than absorbed.
        """
        done = [r for r in self._records if r.completed]
        if len(done) != HISTORY_DAYS:
            raise IncompleteHistoryError(
                f"safety bank holds {len(done)} completed records, "
                f"exactly {HISTORY_DAYS} are required"
            )
        return tuple(_own_evidence(r) for r in done)

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

        evidence = _own_evidence(SafetyEvidence(
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
        ))
        _validate_evidence(evidence, context="candidate insertion")
        if self._records and ordinal <= self._records[-1].ordinal:
            raise DishonestRecordError(
                f"ordinal {ordinal} does not follow the latest ordinal "
                f"{self._records[-1].ordinal}; the bank is chronological"
            )

        self._seen_delivery_ids.add(delivery_id)
        self._records.append(evidence)
        self._evict_oldest_completed()
        return _own_evidence(evidence)

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
            completed = _own_evidence(replace(record, host_residual=residual,
                                              revealed_at=int(revealed_at)))
            _validate_evidence(completed, context="residual reveal")
            self._records[index] = completed
            self._evict_oldest_completed(protect=delivery_id)
            return _own_evidence(completed)
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
        """Rebuild a bank from :meth:`to_state` output.

        Restore is a trust boundary, not a convenience: the persisted state is part
        of the deployment chronology surface, so it goes through the same validator
        as a live append.  A corrupt serialization is refused, never repaired --
        silently fixing it would make the restored bank disagree with the bank that
        produced it, and no reviewer could tell which one had been scored.
        """
        if int(state.get("history_days", HISTORY_DAYS)) != HISTORY_DAYS:
            raise HistoryError(
                "serialized bank was written for a different history length; "
                "the window is a structural constant and is not migrated"
            )
        ledger = set(state.get("delivered_ids", ()))
        bank = cls()
        bank._seen_delivery_ids = set(ledger)
        for entry in state.get("records", ()):
            mask = entry.get("valid_mask")
            try:
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
                                   else torch.tensor(entry["host_residual"],
                                                     dtype=torch.float32)),
                    revealed_at=(None if entry.get("revealed_at") is None
                                 else int(entry["revealed_at"])),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise HistoryError(
                    f"serialized bank entry {entry.get('delivery_id')!r} is not well "
                    f"formed and cannot be restored: {error}"
                ) from error
            if record.delivery_id not in ledger:
                raise DishonestRecordError(
                    f"serialized bank: active record {record.delivery_id!r} is absent "
                    "from the delivered-identifier ledger; a record that was never "
                    "delivered cannot be restored"
                )
            _validate_evidence(record, context="serialized bank")
            bank._records.append(record)

        _validate_sequence(bank._records, context="serialized bank")
        completed = sum(1 for r in bank._records if r.completed)
        if completed > HISTORY_DAYS:
            raise IncompleteHistoryError(
                f"serialized bank holds {completed} completed records, more than "
                f"the {HISTORY_DAYS}-record window"
            )
        return bank

    @classmethod
    def from_warm_start(cls, records: Sequence[SafetyEvidence]) -> "SafetyEvidenceBank":
        """Build a full bank from legal chronological warm-start records.

        Warm start is not a bypass: :func:`select_warm_start` applies the same
        per-record validator and the same uniqueness/chronology rules a live bank
        enforces, and the accepted records are copied into the bank rather than
        aliased, so the caller keeps no handle on the evidence it seeded.
        """
        bank = cls()
        selected = select_warm_start(records)
        _validate_sequence(selected, context="warm start")
        for record in selected:
            if not record.completed:  # pragma: no cover - selection is completed-only
                raise IncompleteHistoryError(
                    f"warm-start record {record.delivery_id!r} has no residual"
                )
            bank._seen_delivery_ids.add(record.delivery_id)
            bank._records.append(_own_evidence(record))
        if not bank.is_ready():  # pragma: no cover - select_warm_start guarantees this
            raise IncompleteHistoryError("warm start did not yield a full bank")
        return bank


def select_warm_start(records: Sequence[SafetyEvidence]) -> Tuple[SafetyEvidence, ...]:
    """The last ``HISTORY_DAYS`` legal chronological warm-start records.

    Legal provenance is out-of-fold prequential or plain prequential: both are
    generated without the outcome they are later scored against.  In-sample
    predictions are excluded, and no other provenance is accepted.

    Declaring a record ``oof_prequential`` or ``prequential`` is a claim, not proof,
    so every record making that claim is checked against the same honesty contract
    the live bank enforces.  A declared-legal record whose candidate/reveal
    chronology is impossible raises rather than being quietly dropped: if the
    caller's out-of-fold pipeline emitted a dishonest record, the warm start is not
    safe to run on whatever remains.

    There is no length argument.  Fewer than ``HISTORY_DAYS`` legal completed
    records is a blocker to be reported, not a shorter window to be improvised:
    W is part of the estimator, so changing it silently would change what is being
    deployed.
    """
    legal = [r for r in records if r.provenance in WARM_START_PROVENANCE]
    for record in legal:
        _validate_evidence(record, context="warm start")
    complete = [r for r in legal if r.completed]
    ordered = sorted(complete, key=lambda r: r.ordinal)
    _validate_sequence(ordered, context="warm start")
    if len(ordered) < HISTORY_DAYS:
        raise InsufficientWarmStartError(
            f"warm start supplied {len(ordered)} legal completed records; "
            f"{HISTORY_DAYS} are required and W is not shortened"
        )
    return tuple(ordered[-HISTORY_DAYS:])
