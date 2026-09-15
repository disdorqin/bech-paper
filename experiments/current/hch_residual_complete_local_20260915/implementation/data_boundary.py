"""Structural data boundary for the RCL v4.1 run.

Two things this module decides once, so no later module can decide them again:

**What may be read.**  The run is TRAIN+VAL development only.  Every loader here
takes the authorized cache path, keeps the ``TRAIN`` and ``VAL`` segments, and
drops the rest.  The V2 evidence tree also contains a ``*_joint.npz`` cache whose
rows include the sealed TEST segment; it is refused by name, so a future edit
cannot widen the run by switching one path constant.  TEST is not "filtered out
later" — the reader for it does not exist, and :class:`SealedTestReader` raises on
every method rather than returning an empty frame.  A missing reader is a
structural zero; an empty frame is a value that some later line could fill in.

**How wide the input is.**  ``d_base_tokens`` is derived from the active core's
:func:`core.base_token_dimension`, never written by hand and never a function of
the market.  There is deliberately no per-market width table anywhere in this
package: a market-keyed width is the market-specific branch the protocol forbids,
wearing a shape-only disguise.  Two cells with different forecast-feature counts
differ in *width* and in nothing else.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

from core import base_token_dimension

__all__ = [
    "V2_EVIDENCE_ROOT",
    "PROTOCOL_ID",
    "AUTHORIZED_SEGMENTS",
    "SEALED_SEGMENTS",
    "FORBIDDEN_MARKETS",
    "CALENDAR_CHANNELS",
    "FORECAST_FEATURES",
    "D_BASE_TOKENS",
    "SealedDataAccessError",
    "SealedTestReader",
    "AuditCounters",
    "AuthorizedCell",
    "load_authorized_cell",
    "cell_input_width",
    "authorized_cache_path",
]

#: The only data boundary this protocol authorizes.
V2_EVIDENCE_ROOT = Path(
    "experiments/evidence/hch_china5_common_benchmark_701020_full_v2_20260913")
PROTOCOL_ID = "COMMON_BENCHMARK_701020_FULL_V2"

#: Segments the run may materialize.
AUTHORIZED_SEGMENTS: tuple[str, ...] = ("TRAIN", "VAL")

#: Segments that exist in the boundary and are sealed for this protocol.
SEALED_SEGMENTS: tuple[str, ...] = ("TEST",)

#: QINGHAI is excluded from this mechanism stage by protocol section 3.  It is a
#: refusal here rather than an omission from a list, so a cell added downstream
#: cannot re-admit it by accident.
FORBIDDEN_MARKETS: tuple[str, ...] = ("QINGHAI_DA",)

#: Registered input width basis for the RCL v4.1 run.  These are *run* constants,
#: identical for every cell: the development panel consumes Host plus raw Host
#: residual history and no forecast-known covariates or calendar channels, so both
#: counts are zero.  Widening them is a protocol change, not a per-cell choice.
CALENDAR_CHANNELS = 0
FORECAST_FEATURES = 0

#: Derived, never written by hand.  The core is the authority on what a base token
#: contains; if the core's token layout changes, this number changes with it
#: instead of silently disagreeing.
D_BASE_TOKENS = base_token_dimension(CALENDAR_CHANNELS, FORECAST_FEATURES)


class SealedDataAccessError(RuntimeError):
    """Raised whenever a sealed part of the boundary is reached."""


class SealedTestReader:
    """A reader for a split the protocol does not authorize.

    Every method raises.  This class exists so the sealed path has a *named,
    typed* refusal at the point of use rather than an absence that a future
    refactor could quietly fill in with a working loader.  It is not a stub to be
    implemented later; implementing it is what the protocol forbids.
    """

    _MESSAGE = ("the V2 TEST split is sealed for HCH_RESIDUAL_COMPLETE_LOCAL_V4_1; "
                "test_label_read_count must stay 0")

    def __init__(self, *_, **__) -> None:
        raise SealedDataAccessError(self._MESSAGE)

    def labels(self, *_, **__):
        raise SealedDataAccessError(self._MESSAGE)

    def targets(self, *_, **__):
        raise SealedDataAccessError(self._MESSAGE)

    def any(self, *_, **__):
        raise SealedDataAccessError(self._MESSAGE)


@dataclass
class AuditCounters:
    """The access audit the protocol requires the run to report.

    ``test_label_read_count`` is initialized to the registered constant and has no
    increment path: nothing in this package can raise it.  It is reported, not
    computed, which is why the verifier re-derives it from the run's written
    artifacts rather than trusting this object.
    """

    cells_read: int = 0
    seeds_run: int = 0
    test_label_read_count: int = 0
    protected_final_read: bool = False
    foreign_read: bool = False
    host_retrained: bool = False
    baseline_rerun: bool = False
    split_modified: bool = False
    core_modified: bool = False
    markets_read: list[str] = field(default_factory=list)

    def note_cell(self, market: str) -> None:
        if market in FORBIDDEN_MARKETS:
            raise SealedDataAccessError(
                f"{market} is excluded from this mechanism stage; the run may not "
                "read it")
        self.cells_read += 1
        if market not in self.markets_read:
            self.markets_read.append(market)

    def as_dict(self) -> dict:
        return {
            "cells_read": self.cells_read,
            "seeds_run": self.seeds_run,
            "test_label_read_count": self.test_label_read_count,
            "protected_final_read": self.protected_final_read,
            "foreign_read": self.foreign_read,
            "host_retrained": self.host_retrained,
            "baseline_rerun": self.baseline_rerun,
            "split_modified": self.split_modified,
            "core_modified": self.core_modified,
            "markets_read": list(self.markets_read),
        }


def authorized_cache_path(market: str, host: str) -> Path:
    """Path of the one authorized Host prediction cache for a cell.

    The ``*_joint`` sibling holds TEST rows and is never returned.  Anything that
    wants the joint cache has to construct the name itself, which the verifier
    scans for.
    """
    if market in FORBIDDEN_MARKETS:
        raise SealedDataAccessError(f"{market} is not an authorized market")
    if "joint" in Path(host).name:
        raise SealedDataAccessError(
            "the joint Host cache carries sealed TEST rows and is not readable by "
            "this protocol")
    return V2_EVIDENCE_ROOT / "02_hosts" / market / host / "host_predictions.npz"


@dataclass(frozen=True)
class AuthorizedCell:
    """A cell's TRAIN/VAL surfaces, already restricted to the authorized split.

    ``rows`` holds the original cache row numbers of ``host_forecast`` in order, so
    the chronology of the full panel is not lost when the segments are separated,
    and :meth:`positions` converts a segment's row numbers into positions in the
    returned arrays.  The two are deliberately different: row numbers are stable
    identifiers, positions are what indexes the arrays, and conflating them is how
    a TRAIN day silently becomes a VAL day.
    """

    market: str
    host: str
    source_sha256: str
    split_hash: str
    index: dict[str, "object"]          # segment -> np.ndarray of cache row numbers
    rows: "object"                      # (n_authorized,) cache row numbers, ascending
    host_forecast: "object"             # (n_authorized, H)
    target: "object"                    # (n_authorized, H)
    valid_mask: "object"                # (n_authorized, H)
    day_ids: "object"                   # (n_authorized,) str
    declared_fit_n: int = -1
    declared_valid_n: int = -1

    def segment(self, name: str):
        """Cache row numbers of one authorized segment."""
        if name not in AUTHORIZED_SEGMENTS:
            raise SealedDataAccessError(
                f"{name!r} is not an authorized segment; only "
                f"{AUTHORIZED_SEGMENTS} may be materialized")
        return self.index[name]

    def positions(self, name: str) -> "object":
        """Positions of one authorized segment inside the returned arrays."""
        import numpy as np

        rows = np.asarray(self.rows)
        wanted = np.asarray(self.segment(name))
        pos = np.searchsorted(rows, wanted)
        if pos.size and (pos.max() >= rows.size
                         or not np.array_equal(rows[pos], wanted)):
            raise SealedDataAccessError(
                f"{name} rows are not a subset of the authorized rows for "
                f"{self.market}/{self.host}")
        return pos

    def chronological_layout(self) -> tuple[int, int]:
        """``(n_train, n_val)``, after proving the split is one contiguous run.

        The runner needs a single chronological index space in which TRAIN day 0 is
        authorized position 0 and VAL follows immediately, because the OOF partition
        and the VAL prequential walk both address days by absolute position.  Rather
        than assume that layout, this method proves it: TRAIN must be the leading
        block, VAL the trailing one, and together they must cover every authorized
        row.  A reordered or interleaved split fails here instead of producing a
        run whose "TRAIN" days are partly VAL days.
        """
        import numpy as np

        train = np.asarray(self.positions("TRAIN"))
        val = np.asarray(self.positions("VAL"))
        n = np.asarray(self.rows).size
        expected_train = np.arange(train.size)
        expected_val = np.arange(train.size, train.size + val.size)
        if not (np.array_equal(train, expected_train)
                and np.array_equal(val, expected_val)
                and train.size + val.size == n):
            raise SealedDataAccessError(
                f"{self.market}/{self.host}: the authorized split is not a "
                "contiguous chronological TRAIN-then-VAL layout; the run addresses "
                "days by absolute position and cannot proceed")
        return int(train.size), int(val.size)

    def residual(self) -> "object":
        """``r = y - host`` over every authorized row, in original order."""
        return self.target - self.host_forecast


def load_authorized_cell(market: str, host: str,
                         counters: Optional[AuditCounters] = None
                         ) -> AuthorizedCell:
    """Load one cell's TRAIN/VAL surfaces from the authorized cache.

    TEST rows are never materialized: the loader masks on the segment column and
    keeps only the authorized names, so the sealed labels do not enter the process
    at all.  The joint cache — which does carry them — is refused by
    :func:`authorized_cache_path` before any file is opened.
    """
    import numpy as np
    from utils.benchmark_cache import HostPredictionCache

    root = Path(__file__).resolve().parents[4]
    path = root / authorized_cache_path(market, host)
    cache = HostPredictionCache.load(path)
    if str(cache.metadata.get("protocol_id")) != PROTOCOL_ID:
        raise SealedDataAccessError(
            f"cache protocol {cache.metadata.get('protocol_id')!r} is not {PROTOCOL_ID!r}")
    if int(cache.metadata.get("test_target_values_read", 0)) != 0:
        raise SealedDataAccessError(
            "the source cache reports TEST target values were read; the boundary "
            "is already breached upstream")

    segment = cache.segment.astype(str)
    unseen = sorted(set(segment) - set(AUTHORIZED_SEGMENTS))
    if unseen:
        # The legal cache does contain a sealed segment; refusing here documents
        # that it was deliberately not materialized rather than never present.
        assert set(unseen) <= set(SEALED_SEGMENTS), unseen
    index = {name: np.flatnonzero(segment == name) for name in AUTHORIZED_SEGMENTS}
    keep = np.sort(np.concatenate([index[n] for n in AUTHORIZED_SEGMENTS]))
    if keep.size == 0:
        raise SealedDataAccessError(f"{market}/{host} has no authorized rows")

    # The row-count identity the protocol requires.  A cache whose TRAIN/VAL
    # selection is short is one where the split moved, and a moved split silently
    # redefines every OOF fold downstream; the metadata is the only independent
    # witness to how many rows the split was built from, so it is checked against
    # the materialized count rather than against itself.
    declared_fit_n = cache.metadata.get("fit_n")
    declared_valid_n = cache.metadata.get("valid_n")
    if declared_fit_n is None or declared_valid_n is None:
        raise SealedDataAccessError(
            f"{market}/{host}: the cache declares no fit_n/valid_n; the "
            "TRAIN+VAL row count cannot be checked")
    fit_n, valid_n = int(declared_fit_n), int(declared_valid_n)
    if int(index["TRAIN"].size) != fit_n or int(index["VAL"].size) != valid_n:
        raise SealedDataAccessError(
            f"{market}/{host}: materialized TRAIN={int(index['TRAIN'].size)} "
            f"VAL={int(index['VAL'].size)} but the metadata declares "
            f"fit_n={fit_n} valid_n={valid_n}")
    if keep.size != fit_n + valid_n:
        raise SealedDataAccessError(
            f"{market}/{host}: materialized {keep.size} rows but the metadata "
            f"declares fit_n+valid_n = {fit_n + valid_n}")
    selected = set(segment[keep].tolist())
    if selected - set(AUTHORIZED_SEGMENTS):
        raise SealedDataAccessError(
            f"{market}/{host}: a sealed segment reached the authorized selection: "
            f"{sorted(selected - set(AUTHORIZED_SEGMENTS))}")

    if counters is not None:
        counters.note_cell(market)
    return AuthorizedCell(
        market=market, host=host,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        split_hash=str(cache.metadata.get("split_hash", "")),
        index=index, rows=keep,
        host_forecast=np.asarray(cache.host_pred[keep, :, 0], dtype=np.float32),
        target=np.asarray(cache.y_true[keep, :, 0], dtype=np.float32),
        valid_mask=np.ones(cache.host_pred[keep, :, 0].shape, dtype=np.float32),
        day_ids=np.asarray(cache.timestamp[keep]).astype(str),
        declared_fit_n=fit_n, declared_valid_n=valid_n,
    )


def cell_input_width(calendar_channels: int = CALENDAR_CHANNELS,
                     forecast_features: int = FORECAST_FEATURES) -> int:
    """Input width for a run whose covariates have these counts.

    The cell is deliberately **not** a parameter.  A signature that accepted a
    market name would be the place a per-market width gets introduced, so the
    contract here is that two different cells are indistinguishable to this
    function: only the covariate counts move the width.
    """
    return base_token_dimension(calendar_channels, forecast_features)
