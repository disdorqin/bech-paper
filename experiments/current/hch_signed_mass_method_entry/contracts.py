"""Experiment-side data contracts.

These dataclasses describe *data*, not science.  They exist so that one generic
adapter can serve five markets with different legal feature counts (F = 9/8/7/4
for the CHINA5 panel markets, F = 7 for GANSU_DA) without a single province-name
branch anywhere in the model path.

Nothing here is imported by ``src/core``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import config as C


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class HarnessError(RuntimeError):
    """Base class for every refusal raised by this package."""


class ContractError(HarnessError):
    """The audited dataset contract is missing, malformed or inconsistent."""


class ProvenanceError(HarnessError):
    """A frozen artifact does not match its recorded hash."""


class LegalityError(HarnessError):
    """An operation would read a role or column the protocol forbids."""


class ReadinessError(HarnessError):
    """A cell cannot be executed as requested."""


class ProtectedFinalAccess(LegalityError):
    """Raised on any attempt to materialise ``PROTECTED_FINAL``.

    This is the last line of defence and it is expected never to fire: the
    canonical contracts declare ``PROTECTED_FINAL`` a *closed* role, so the
    segment vocabulary of the frozen artifacts does not contain it at all.
    """


class EvidenceWriteError(HarnessError):
    """A write was refused: unconfined destination, or immutable evidence.

    Every evidence write resolves its destination through
    ``raw_evidence.confine`` first.  The exception is a refusal, not a warning:
    a caller that gets one has not written anything, which is the only safe
    outcome when the alternative is overwriting frozen Host evidence or
    silently replacing the evidence a previous run produced.
    """


# --------------------------------------------------------------------------
# Read accounting
# --------------------------------------------------------------------------
@dataclass
class ReadAudit:
    """Counts every read that the protocol forbids.

    A run that finishes with ``protected_final_reads == 0`` and
    ``target_day_price_input_reads == 0`` has a machine-checkable legality
    proof, independent of any reasoning about the code path.
    """

    target_day_price_input_reads: int = 0
    realized_future_input_reads: int = 0
    protected_final_reads: int = 0
    host_prediction_reads: int = 0
    history_window_reads: int = 0

    def forbid_target_day_price_input(self, what: str) -> None:
        self.target_day_price_input_reads += 1
        raise LegalityError(f"target-day DA price read attempted: {what}")

    def forbid_realized_future_input(self, what: str) -> None:
        self.realized_future_input_reads += 1
        raise LegalityError(f"realized future exogenous read attempted: {what}")

    def forbid_protected_final(self, what: str) -> None:
        self.protected_final_reads += 1
        raise ProtectedFinalAccess(f"PROTECTED_FINAL read attempted: {what}")

    def as_dict(self) -> Dict[str, int]:
        return {
            "target_day_price_input_reads": self.target_day_price_input_reads,
            "realized_future_input_reads": self.realized_future_input_reads,
            "protected_final_reads": self.protected_final_reads,
            "host_prediction_reads": self.host_prediction_reads,
            "history_window_reads": self.history_window_reads,
        }


# --------------------------------------------------------------------------
# Market contract
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketContract:
    """The audited, controlling description of one market's legal inputs.

    ``contract_source`` records *where the adapter got this*:

    * ``CANONICAL_CHINA5``  -- an audited ``DATASET_CONTRACT.json``.
    * ``LEGACY_PREFLIGHT``  -- the frozen GANSU preflight manifests.  GANSU_DA has
      no ``DATASET_CONTRACT.json`` anywhere in this repository; its legal column
      list and its role split are published in the upstream evidence root's
      ``preflight/`` directory instead.  Adopting them is provenance adaptation,
      not a substitute dataset, and the ancestry is disclosed on every artifact.

    ``role_taxonomy`` records which role vocabulary the frozen Host artifact uses:

    * ``CANONICAL_CHINA5`` -- ``HOST_TRAIN/HOST_VAL/POST_TRAIN/DEV_EVAL`` and the
      sealed remainder is simply absent.
    * ``LEGACY_S1_DIAG``   -- ``S1/DIAG_FIT/DIAG_EVAL`` with ``S3/S4`` sealed.  A
      *disclosed, verbatim* map owned by the repository converts between them.
    """

    market: str
    legal_columns: Tuple[str, ...]
    target_column: str
    forbidden_columns: Tuple[str, ...]
    source_path: str
    source_sha256: Optional[str]
    contract_path: str
    contract_sha256: str
    contract_source: str
    role_taxonomy: str
    lead_convention: str
    role_counts: Mapping[str, int]
    closed_roles: Tuple[str, ...]
    horizon: int
    seq_len: int
    target_day_price_is_not_an_input: bool
    #: Roles the contract itself declares readable.  A role outside this tuple
    #: may never be materialised, whatever its name.
    open_roles: Tuple[str, ...] = ()
    #: Days the contract declares for the sealed role.  This is a *declared
    #: count*, not a read; it is carried so the harness can disclose what it is
    #: deliberately not touching.
    declared_sealed_days: Optional[int] = None
    #: The contract's own independent read audit, verified to be all-zero.
    contract_read_audit: Mapping[str, Any] = field(default_factory=dict)
    role_day_span: Mapping[str, Any] = field(default_factory=dict)
    excluded_features: Mapping[str, Any] = field(default_factory=dict)
    source_reader: Optional[str] = None
    source_encoding: Optional[str] = None
    feature_contract: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.legal_columns)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "market": self.market,
            "n_legal_features": self.n_features,
            "legal_columns": list(self.legal_columns),
            "target_column": self.target_column,
            "forbidden_columns": list(self.forbidden_columns),
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "contract_path": self.contract_path,
            "contract_sha256": self.contract_sha256,
            "contract_source": self.contract_source,
            "role_taxonomy": self.role_taxonomy,
            "lead_convention": self.lead_convention,
            "role_counts": dict(self.role_counts),
            "open_roles": list(self.open_roles),
            "closed_roles": list(self.closed_roles),
            "declared_sealed_days": self.declared_sealed_days,
            "contract_read_audit": dict(self.contract_read_audit),
            "horizon": self.horizon,
            "seq_len": self.seq_len,
            "target_day_price_is_not_an_input": self.target_day_price_is_not_an_input,
            "source_reader": self.source_reader,
            "source_encoding": self.source_encoding,
        }


def _require_zero_audit(audit: Mapping[str, Any], where: str) -> Dict[str, Any]:
    """Fail closed if a controlling file's own read audit is not all-zero."""
    keys = ("target_day_DA_price_input_reads", "realized_future_input_reads",
            "protected_final_reads")
    for key in keys:
        if key in audit and int(audit[key]) != 0:
            raise LegalityError(
                f"{where} reports {key}={audit[key]} != 0; refusing to build the adapter")
    return {k: audit[k] for k in audit if k in keys or k in
            ("rows_read", "columns", "publication_semantics")}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError(f"controlling file missing: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# Canonical CHINA5 contracts
# --------------------------------------------------------------------------
#: Markets served by a ``DATASET_CONTRACT.json`` panel, i.e. everything in the
#: registry except GANSU_DA, whose contract is the frozen legacy preflight pair
#: instead.  Derived from the one registry rather than restated, so the two
#: cannot drift apart.
_PANEL_MARKETS = frozenset(C.MARKETS) - {"GANSU_DA"}


def load_china5_contract(market: str) -> MarketContract:
    """Read the audited ``DATASET_CONTRACT.json`` for a CHINA5 panel market.

    The physical directory name drops the ``_DA`` suffix used in the canonical
    market vocabulary; that mapping is pure path adaptation and is the only
    place the two spellings meet.
    """
    if market not in _PANEL_MARKETS:
        raise ContractError(f"{market} has no CHINA5 panel contract")

    physical = market[: -len("_DA")]
    path = C.CHINA5_CONTRACT_DIR / physical / "DATASET_CONTRACT.json"
    raw = _load_json(path)

    feature_contract = raw.get("feature_contract", {})
    legal = tuple(feature_contract.get("legal_columns") or ())
    declared_n = feature_contract.get("n_legal_columns", len(legal))
    if declared_n != len(legal):
        raise ContractError(
            f"{market}: n_legal_columns={declared_n} != len(legal_columns)={len(legal)}")
    if not legal:
        raise ContractError(f"{market}: legal column list is empty")

    contract_audit = _require_zero_audit(
        feature_contract.get("read_audit", {}), f"{market} DATASET_CONTRACT.json")

    target_contract = raw.get("target_contract", {})
    if not target_contract.get("target_day_price_is_not_an_input", False):
        raise LegalityError(
            f"{market}: contract does not assert target_day_price_is_not_an_input")

    horizon_contract = raw.get("horizon_contract", {})
    horizon = int(horizon_contract.get("horizon", C.HORIZON))
    seq_len = int(horizon_contract.get("seq_len", C.SEQ_LEN))
    lead = str(horizon_contract.get("hour_label_convention", "hour_ending"))
    if lead not in ("hour_ending", "DAY_ORIGIN_LABEL"):
        raise ContractError(f"{market}: unsupported hour_label_convention {lead!r}")

    provenance = raw.get("provenance", {})
    split = raw.get("split", {})
    counts = {k: int(v) for k, v in (split.get("counts") or {}).items()}
    closed = tuple(split.get("closed_roles") or ())
    open_roles = tuple(split.get("open_roles") or ())

    # The contract declares the exact day counts its own frozen role boundary
    # rule produced.  Verify the rule here so a future contract edit cannot
    # silently desynchronise the harness from the frozen Host artifacts.
    boundaries = split.get("role_boundaries", {})
    if boundaries:
        mismatched = {k: (v, C.SPLIT_BOUNDARY_RATIOS[k])
                      for k, v in boundaries.items()
                      if k in C.SPLIT_BOUNDARY_RATIOS
                      and abs(float(v) - C.SPLIT_BOUNDARY_RATIOS[k]) > 1e-12}
        if mismatched:
            raise ContractError(
                f"{market}: contract role_boundaries disagree with the registered "
                f"rule: {mismatched}")

    return MarketContract(
        market=market,
        legal_columns=legal,
        target_column=str(target_contract.get("target_column", "日前电价")),
        forbidden_columns=tuple(target_contract.get("forbidden_target_day_inputs") or ()),
        source_path=str(provenance.get("source_path", "")),
        source_sha256=provenance.get("source_sha256"),
        contract_path=str(path.relative_to(C.REPO_ROOT)).replace("\\", "/"),
        contract_sha256=sha256_file(path),
        contract_source="CANONICAL_CHINA5",
        role_taxonomy="CANONICAL_CHINA5",
        lead_convention=lead,
        role_counts=counts,
        closed_roles=closed,
        open_roles=open_roles,
        declared_sealed_days=counts.get(C.ROLE_PROTECTED_FINAL),
        contract_read_audit=contract_audit,
        role_day_span=raw.get("role_day_span", {}),
        excluded_features=split.get("excluded_features", {}),
        source_reader=provenance.get("reader"),
        source_encoding=provenance.get("encoding"),
        horizon=horizon,
        seq_len=seq_len,
        target_day_price_is_not_an_input=True,
        feature_contract=feature_contract,
        extra={
            "dataset_id": raw.get("dataset_id"),
            "physical_market_group": raw.get("physical_market_group"),
            "status": raw.get("status"),
            "origin_rule": horizon_contract.get("origin_rule"),
            "day_definition": horizon_contract.get("day_definition"),
            "boundary_policy": split.get("boundary_policy"),
            "missing_duplicate_policy": raw.get("missing_duplicate_policy", {}),
            "publication_semantics": contract_audit.get("publication_semantics"),
            "declared_research_role": provenance.get("declared_research_role"),
            "raw_columns": provenance.get("raw_columns"),
        },
    )


# --------------------------------------------------------------------------
# Legacy GANSU preflight contract
# --------------------------------------------------------------------------
def load_gansu_legacy_contract() -> MarketContract:
    """Read GANSU_DA's audited provenance from the frozen preflight manifests.

    Two controlling files, both produced before any target outcome was seen:

    * ``legal_state_manifest.json`` -- the audited legal column list, the row
      count, and three zero-read assertions.
    * ``gansu_da_split_manifest.json`` -- the frozen role split, the target
      column, the forbidden inputs and the consumed-day list.

    The adapter refuses to proceed if either file is missing or if their
    assertions do not hold: this is a fail-closed legacy path, not a fallback.
    """
    legal_state = _load_json(C.GANSU_LEGAL_STATE_MANIFEST)
    split_manifest = _load_json(C.GANSU_SPLIT_MANIFEST)

    key = "GANSU_DA"
    if key not in legal_state:
        raise ContractError(f"{C.GANSU_LEGAL_STATE_MANIFEST} has no {key} entry")
    entry = legal_state[key]

    legal = tuple(entry.get("columns") or ())
    if not legal:
        raise ContractError("GANSU legal column list is empty")

    # Fail closed on the legacy manifest's own zero-read assertions.  The legacy
    # vocabulary spells the same three counters differently; they are canonicalised
    # here so every market's audit is read the same way, and the native spellings
    # are kept verbatim alongside for provenance.
    legacy_native_audit = {
        "target_day_DA_input_reads": entry.get("target_day_DA_input_reads"),
        "realized_RT_input_reads": entry.get("realized_RT_input_reads"),
        "S3_S4_reads": entry.get("S3_S4_reads"),
    }
    legacy_audit = {
        "target_day_DA_price_input_reads": legacy_native_audit["target_day_DA_input_reads"],
        "realized_future_input_reads": legacy_native_audit["realized_RT_input_reads"],
        "protected_final_reads": legacy_native_audit["S3_S4_reads"],
        "rows_read": entry.get("rows_read"),
        "columns": list(legal),
        "publication_semantics": entry.get("publication_semantics"),
        "native_spelling": legacy_native_audit,
    }
    for audit_key in ("target_day_DA_price_input_reads", "realized_future_input_reads",
                      "protected_final_reads"):
        if legacy_audit[audit_key] is None:
            raise LegalityError(
                f"GANSU legal_state_manifest omits the {audit_key} assertion; refusing "
                "to build a legacy adapter on an unattested manifest")
        if int(legacy_audit[audit_key]) != 0:
            raise LegalityError(
                f"GANSU legal_state_manifest reports {audit_key}="
                f"{legacy_audit[audit_key]} != 0; refusing to build the adapter")

    if str(split_manifest.get("schema", "")) != "gansu_da_development_split.v1":
        raise ContractError("unexpected GANSU split manifest schema")
    if not split_manifest.get("created_before_target_outcomes", False):
        raise ContractError(
            "GANSU split manifest does not attest to being created before outcomes")

    raw_counts = split_manifest.get("counts", {})
    role_counts: Dict[str, int] = {}
    for native, canonical_roles in C.GANSU_LEGACY_ROLE_MAP.items():
        if native not in raw_counts:
            continue
        n_native = int(raw_counts[native])
        if len(canonical_roles) == 1:
            role_counts[canonical_roles[0]] = role_counts.get(canonical_roles[0], 0) + n_native
            continue
        # ``S1`` feeds both ``HOST_TRAIN`` and ``HOST_VAL``.  The split inside it is
        # the registered chronological tail rule, not a second reading of the count.
        n_val = int(n_native * C.SPLIT_BOUNDARY_RATIOS["host_val_tail_of_s1"])
        role_counts[C.ROLE_HOST_TRAIN] = role_counts.get(C.ROLE_HOST_TRAIN, 0) + (n_native - n_val)
        role_counts[C.ROLE_HOST_VAL] = role_counts.get(C.ROLE_HOST_VAL, 0) + n_val

    sealed_native = tuple(split_manifest.get("closed_roles") or ())
    expected_sealed = {"S3", "S4"}
    if not expected_sealed.issubset(set(sealed_native)):
        raise ContractError(
            f"GANSU split manifest closed_roles={sealed_native} does not seal "
            f"{sorted(expected_sealed)}")

    contract_sha = sha256_bytes(
        (sha256_file(C.GANSU_LEGAL_STATE_MANIFEST)
         + "|"
         + sha256_file(C.GANSU_SPLIT_MANIFEST)).encode("ascii"))

    consumed = split_manifest.get("consumed_days")
    return MarketContract(
        market="GANSU_DA",
        legal_columns=legal,
        target_column=str(split_manifest.get("target_column", "日前电价")),
        forbidden_columns=tuple(split_manifest.get("forbidden_target_day_inputs") or ()),
        source_path=str(split_manifest.get("source", "")),
        source_sha256=entry.get("source_sha256"),
        contract_path=str(C.GANSU_SPLIT_MANIFEST.relative_to(C.REPO_ROOT)).replace("\\", "/"),
        contract_sha256=contract_sha,
        contract_source="LEGACY_PREFLIGHT",
        role_taxonomy="LEGACY_S1_DIAG",
        lead_convention="DAY_ORIGIN_LABEL",
        role_counts=role_counts,
        closed_roles=(C.ROLE_PROTECTED_FINAL,),
        open_roles=(C.ROLE_HOST_TRAIN, C.ROLE_HOST_VAL, C.ROLE_POST_TRAIN,
                    C.ROLE_DEV_EVAL),
        declared_sealed_days=sum(int(raw_counts.get(k, 0)) for k in ("S3", "S4")),
        contract_read_audit=legacy_audit,
        role_day_span={},
        excluded_features=split_manifest.get("excluded_features", {}),
        source_reader="xlsx",
        horizon=C.HORIZON,
        seq_len=C.SEQ_LEN,
        target_day_price_is_not_an_input=True,
        feature_contract={"read_audit": legacy_audit, "n_legal_columns": len(legal),
                          "legal_columns": list(legal)},
        extra={
            "legacy_role_map": {k: list(v) for k, v in C.GANSU_LEGACY_ROLE_MAP.items()},
            "native_counts": {k: int(v) for k, v in raw_counts.items()},
            "native_closed_roles": list(sealed_native),
            "rows_read": entry.get("rows_read"),
            "n_consumed_days": (len(consumed) if isinstance(consumed, list) else None),
            "boundary_policy": split_manifest.get("boundary_policy"),
            "legal_state_manifest": str(
                C.GANSU_LEGAL_STATE_MANIFEST.relative_to(C.REPO_ROOT)).replace("\\", "/"),
            "disclosure": (
                "GANSU_DA publishes no DATASET_CONTRACT.json; this contract is "
                "reconstructed from the frozen preflight manifests, and the legacy "
                "S1/DIAG_FIT/DIAG_EVAL/S3/S4 vocabulary is mapped verbatim by the "
                "repository's own breadth-stage handoff"),
        },
    )


def load_market_contract(market: str) -> MarketContract:
    """Dispatch to the right controlling source without a market-specific branch.

    The dispatch key is ``contract_source`` -- a property of the *provenance*,
    not of the province.  Adding a sixth market means adding a contract file, not
    editing model behaviour.
    """
    if market == "GANSU_DA":
        return load_gansu_legacy_contract()
    return load_china5_contract(market)


# --------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Episode:
    """One forecast episode: a target day, its origin and its Host forecast.

    ``index`` is the row's position in the frozen Host artifact, so every
    downstream artifact can point at a byte-stable address rather than at a
    timestamp string that a re-freeze might re-spell.
    """

    index: int
    market: str
    host: str
    label: np.datetime64          # the artifact's own timestamp field
    target_day: np.datetime64     # 'D' resolution
    segment: str                  # verbatim from the frozen artifact
    role: str                     # canonical role after any legacy relabelling
    run_length: int               # horizon actually present (<24 -> short episode)

    @property
    def is_full_day(self) -> bool:
        return self.run_length == C.HORIZON


@dataclass(frozen=True)
class HostPanel:
    """The frozen Host artifact for one (market, Host) cell, plus its episodes."""

    market: str
    host: str
    artifact_path: str
    artifact_sha256: str
    n_episodes: int
    episodes: Tuple[Episode, ...]
    timestamp: np.ndarray
    context: np.ndarray
    y_true: np.ndarray
    host_pred: np.ndarray
    segment: np.ndarray
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def role_indices(self, role: str) -> np.ndarray:
        """Row positions of one canonical role, in artifact order."""
        if role in C.SEALED_ROLES:
            raise ProtectedFinalAccess(f"role_indices({role})")
        return np.array([e.index for e in self.episodes if e.role == role], dtype=np.int64)

    def role_count(self, role: str) -> int:
        return int(self.role_indices(role).size)


# --------------------------------------------------------------------------
# Cell specification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CellSpec:
    """A single market x Host coordinate."""

    market: str
    host: str

    @property
    def key(self) -> str:
        return f"{self.market}::{self.host}"

    def as_dict(self) -> Dict[str, str]:
        return {"market": self.market, "host": self.host, "cell": self.key}


# --------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------
READY_FROZEN = "READY_FROZEN"
PENDING_EXTERNAL_HOST_ARTIFACT = "PENDING_EXTERNAL_HOST_ARTIFACT"
INVALID_SEMANTICS = "INVALID_SEMANTICS"
MISSING_ROLE = "MISSING_ROLE"
BLOCKED_PROVENANCE = "BLOCKED_PROVENANCE"

READINESS_FIELDS = (
    "market",
    "host",
    "status",
    "artifact_path",
    "artifact_sha256",
    "contract_path",
    "contract_sha256",
    "contract_source",
    "role_taxonomy",
    "lead_convention",
    "target_semantics",
    "origin_alignment",
    "horizon",
    "n_episodes",
    "n_host_train",
    "n_host_val",
    "n_post_train",
    "n_dev_eval",
    "n_protected_final_read",
    "contract_sealed_days",
    "n_short_episodes",
    "oof_folds",
    "blocker",
)


def readiness_row(**kwargs: Any) -> Dict[str, Any]:
    missing = [f for f in READINESS_FIELDS if f not in kwargs]
    if missing:
        raise ReadinessError(f"readiness row missing fields: {missing}")
    return {f: kwargs[f] for f in READINESS_FIELDS}


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path,
              fields: Sequence[str] = READINESS_FIELDS) -> None:
    """Write a deterministic CSV (no csv module ordering surprises)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(fields)]
    for row in rows:
        cells = []
        for name in fields:
            value = row.get(name, "")
            text = "" if value is None else str(value)
            if any(ch in text for ch in (",", '"', "\n", "\r")):
                text = '"' + text.replace('"', '""') + '"'
            cells.append(text)
        lines.append(",".join(cells))
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
