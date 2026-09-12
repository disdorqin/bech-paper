"""Frozen-Host prediction registry and loader.

One interface for all 20 ``market x Host`` coordinates.  This module is
**read-only**: it never trains, re-fits, re-freezes or substitutes a Host.  If a
cell's artifact is absent it becomes ``PENDING_EXTERNAL_HOST_ARTIFACT``; if its
semantics disagree with the market contract it becomes ``INVALID_SEMANTICS``.
Silently borrowing another Host, another market's cache or a historical
``GANSU_RT`` cache is not an available code path -- there is no fallback
argument anywhere in this file.

Frozen-array hashing note
-------------------------
``.npz`` *container* bytes are not byte-deterministic across writers, so the
container hash is recorded for provenance but the **identity** of a frozen array
is its ``array_content_sha256`` -- a digest over each member's name, dtype, shape
and canonical C-contiguous payload.  Two writers producing the same arrays agree
on this digest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C
from .contracts import (
    BLOCKED_PROVENANCE,
    INVALID_SEMANTICS,
    MISSING_ROLE,
    PENDING_EXTERNAL_HOST_ARTIFACT,
    READY_FROZEN,
    ContractError,
    Episode,
    HarnessError,
    HostPanel,
    LegalityError,
    MarketContract,
    ProtectedFinalAccess,
    ReadAudit,
    ReadinessError,
    load_market_contract,
    readiness_row,
    sha256_file,
)

NPZ_NAME = "host_predictions.npz"
FREEZE_MANIFEST_NAME = "FREEZE_MANIFEST.json"

REQUIRED_KEYS = ("timestamp", "context", "y_true", "host_pred", "segment")


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------
def array_content_sha256(arrays: Dict[str, np.ndarray]) -> str:
    """Digest of the array *contents*, independent of container bytes."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(b"\x00")
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(b"\x00")
        digest.update(arr.tobytes())
        digest.update(b"\x01")
    return digest.hexdigest().upper()


# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------
def artifact_dir(market: str, host: str) -> Path:
    try:
        return C.HOST_ARTIFACT_ROOTS[(market, host)]
    except KeyError:
        raise ReadinessError(f"no registered artifact root for {market}::{host}") from None


def artifact_path(market: str, host: str) -> Path:
    return artifact_dir(market, host) / NPZ_NAME


# --------------------------------------------------------------------------
# Role labelling
# --------------------------------------------------------------------------
def _china5_role(native_segment: str) -> str:
    """CHINA5 artifacts already speak the canonical vocabulary."""
    if native_segment in C.SEALED_ROLES:
        raise ProtectedFinalAccess(
            f"frozen artifact carries a {native_segment} segment")
    if native_segment not in C.ROLE_ORDER:
        raise ContractError(f"unrecognised CHINA5 segment {native_segment!r}")
    return native_segment


def _gansu_roles(native_segments: np.ndarray) -> np.ndarray:
    """Relabel GANSU's native roles via the disclosed, verbatim legacy map.

    ``S1`` maps to *both* ``HOST_TRAIN`` and ``HOST_VAL``; the split inside ``S1``
    is the registered ``host_val_tail_of_s1 = 0.10`` tail rule, applied
    chronologically.  The rule is verified here against a frozen, independent
    witness -- ``THRESHOLD_FREEZE.json`` records that GANSU's thresholds were fit
    on ``188`` days whose native role is ``S1``; ``208 - floor(208 * 0.10) = 188``.
    A mismatch is a hard error, because it would mean the harness is inventing a
    role boundary rather than reproducing a frozen one.
    """
    roles = np.empty(native_segments.shape[0], dtype=object)

    for native, canonical_roles in C.GANSU_LEGACY_ROLE_MAP.items():
        mask = native_segments == native
        if native in ("S3", "S4") and mask.any():
            raise ProtectedFinalAccess(f"frozen artifact carries sealed segment {native}")
        if not mask.any():
            continue
        if len(canonical_roles) == 1:
            roles[mask] = canonical_roles[0]
        else:
            # Only S1 today.  Chronological tail = HOST_VAL.
            idx = np.flatnonzero(mask)
            n_val = int(len(idx) * C.SPLIT_BOUNDARY_RATIOS["host_val_tail_of_s1"])
            roles[idx[: len(idx) - n_val]] = C.ROLE_HOST_TRAIN
            roles[idx[len(idx) - n_val:]] = C.ROLE_HOST_VAL

    unknown = np.array([r is None or r == "" for r in roles])
    if unknown.any():
        bad = sorted(set(native_segments[unknown].tolist()))
        raise ContractError(f"unmapped GANSU segments: {bad}")
    return roles


def _legacy_host_train_witness(market: str) -> Optional[int]:
    """Independent frozen day-count for ``HOST_TRAIN``, when one is published."""
    try:
        with open(C.THRESHOLD_FREEZE, "r", encoding="utf-8") as handle:
            freeze = json.load(handle)
    except (OSError, ValueError):
        return None
    record = _find_threshold_record(freeze, market)
    if not record:
        return None
    if str(record.get("native_role", "")) != "S1":
        return None
    n_days = record.get("n_days_fit")
    return int(n_days) if n_days is not None else None


def _find_threshold_record(freeze: Dict[str, Any], market: str) -> Optional[Dict[str, Any]]:
    """Locate one market record in ``THRESHOLD_FREEZE.json`` without assuming depth."""
    stack: List[Any] = [freeze]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if market in node and isinstance(node[market], dict):
                candidate = node[market]
                if "q05" in candidate or "day_spread_p90" in candidate:
                    return candidate
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _read_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as handle:
        missing = [k for k in REQUIRED_KEYS if k not in handle.files]
        if missing:
            raise ContractError(f"{path.name} missing keys {missing}")
        return {k: np.array(handle[k]) for k in REQUIRED_KEYS}


def target_day_of(labels: np.ndarray, lead_convention: str) -> np.ndarray:
    """Map an artifact timestamp field to its target day.

    Both lead conventions in this repository converge on **one** addressing rule,
    which is why the adapter needs no market branch:

    * every artifact's ``timestamp[i]`` is the label of the **first of the 24
      target rows**, so the target rows are the raw rows labelled
      ``L, L+1h, ..., L+23h``;
    * the forecast origin is the instant ``L - 1h`` -- with hour-ending labels the
      row labelled ``L`` covers the interval ``(L-1h, L]``, so its interval starts
      exactly at the origin.

    Verified against all five frozen artifacts: ``max |raw target - npz y_true|``
    <= 5.9e-05 (float32 rounding), and ``context`` equals the 168 raw hours ending
    at ``L - 1h`` to the same tolerance.

    The two conventions differ only in how the day is *named*:

    * ``hour_ending`` (CHINA5) -- ``L`` sits at ``D 01:00``, so ``L.date == D`` and
      ``(L - 1h).date == D`` agree.
    * ``DAY_ORIGIN_LABEL`` (GANSU) -- ``L`` sits at ``D 00:00``, so ``L.date == D``
      names the day exactly as the frozen ``gansu_da_split_manifest.json``
      ``consumed_days`` list does.

    ``L.date`` therefore reproduces both the CHINA5 ``role_day_span`` and the
    GANSU ``consumed_days``, and is the rule used here.
    """
    if lead_convention not in ("hour_ending", "DAY_ORIGIN_LABEL"):
        raise ContractError(f"unknown lead convention {lead_convention!r}")
    stamps = np.array(labels, dtype="datetime64[s]")
    if lead_convention == "hour_ending":
        # Structural guard: if a re-freeze moved the CHINA5 label off the first
        # target hour, the two names would silently diverge.  Catch it here.
        shifted = (stamps - np.timedelta64(1, "h")).astype("datetime64[D]")
        direct = stamps.astype("datetime64[D]")
        if not np.array_equal(shifted, direct):
            raise ContractError(
                "hour_ending labels do not sit at the first target hour; the day "
                "naming rule would be ambiguous")
    return stamps.astype("datetime64[D]")


def target_rows_of(label: np.datetime64) -> np.ndarray:
    """The 24 raw labels belonging to one episode.  Uniform for every market."""
    base = np.datetime64(label, "s")
    return np.array([base + np.timedelta64(h, "h") for h in range(C.HORIZON)])


def context_rows_of(label: np.datetime64) -> np.ndarray:
    """The 168 strictly-causal raw labels ending at the origin ``L - 1h``.

    Not a model input.  Used only as an independent legality witness.
    """
    base = np.datetime64(label, "s") - np.timedelta64(1, "h")
    return np.array([base - np.timedelta64(C.SEQ_LEN - 1 - h, "h") for h in range(C.SEQ_LEN)])


def _run_lengths(y_true: np.ndarray) -> np.ndarray:
    """Finite horizon entries per episode.  A full day is exactly ``HORIZON``."""
    flat = y_true.reshape(y_true.shape[0], -1)
    return np.isfinite(flat).sum(axis=1).astype(np.int64)


def load_host_panel(market: str, host: str,
                    contract: Optional[MarketContract] = None,
                    audit: Optional[ReadAudit] = None) -> HostPanel:
    """Load one frozen Host artifact and canonicalise its roles.

    Raises ``FileNotFoundError`` when the artifact is absent (the caller turns
    that into ``PENDING_EXTERNAL_HOST_ARTIFACT``), and a ``HarnessError`` subclass
    for every other failure mode.
    """
    contract = contract or load_market_contract(market)
    audit = audit if audit is not None else ReadAudit()

    path = artifact_path(market, host)
    arrays = _read_npz(path)

    timestamp = arrays["timestamp"]
    segment = arrays["segment"]
    context = arrays["context"]
    y_true = arrays["y_true"]
    host_pred = arrays["host_pred"]

    n = int(timestamp.shape[0])
    if not (segment.shape[0] == n and y_true.shape[0] == n
            and host_pred.shape[0] == n and context.shape[0] == n):
        raise ContractError(f"{market}::{host}: inconsistent first dimension")

    if y_true.ndim != 3 or y_true.shape[1] != C.HORIZON:
        raise ContractError(f"{market}::{host}: y_true shape {y_true.shape} != (N,24,1)")
    if host_pred.shape != y_true.shape:
        raise ContractError(
            f"{market}::{host}: host_pred {host_pred.shape} != y_true {y_true.shape}")

    declared_horizon = contract.horizon
    if declared_horizon != y_true.shape[1]:
        raise ContractError(
            f"{market}::{host}: contract horizon {declared_horizon} != "
            f"artifact horizon {y_true.shape[1]}")

    native = np.array([str(s) for s in segment])
    if contract.role_taxonomy == "LEGACY_S1_DIAG":
        roles = _gansu_roles(native)
        witness = _legacy_host_train_witness(market)
        if witness is not None:
            observed = int((roles == C.ROLE_HOST_TRAIN).sum())
            if observed != witness:
                raise ContractError(
                    f"{market}: reconstructed HOST_TRAIN={observed} disagrees with the "
                    f"frozen THRESHOLD_FREEZE witness n_days_fit={witness}; the legacy "
                    "role boundary would be invented rather than reproduced")
    else:
        roles = np.array([_china5_role(str(s)) for s in native], dtype=object)

    # The contract's own ``open_roles`` list is the authority on what may be
    # materialised.  A role outside it is refused even if its name looks benign.
    if contract.open_roles:
        stray = sorted(set(roles.tolist()) - set(contract.open_roles))
        if stray:
            raise LegalityError(
                f"{market}::{host}: artifact yields roles {stray} which the contract "
                f"does not declare open ({list(contract.open_roles)})")

    days = target_day_of(timestamp, contract.lead_convention)
    run_lengths = _run_lengths(y_true)

    order = np.lexsort((np.arange(n), days))
    if not np.array_equal(order, np.arange(n)):
        raise ContractError(
            f"{market}::{host}: artifact rows are not in chronological target-day order")

    episodes: Tuple[Episode, ...] = tuple(
        Episode(
            index=i,
            market=market,
            host=host,
            label=np.datetime64(timestamp[i]),
            target_day=np.datetime64(days[i]),
            segment=str(native[i]),
            role=str(roles[i]),
            run_length=int(run_lengths[i]),
        )
        for i in range(n)
    )

    # The sealed role must not exist in the canonicalised vocabulary.  This is
    # the structural guarantee behind the zero-read assertion.
    if any(e.role in C.SEALED_ROLES for e in episodes):
        raise ProtectedFinalAccess("canonicalised episode vocabulary contains a sealed role")

    provenance = _load_provenance(market, host)
    provenance["array_content_sha256"] = array_content_sha256(arrays)
    provenance["container_sha256"] = sha256_file(path)
    provenance["container_sha256_note"] = (
        "npz container bytes are not writer-deterministic; use array_content_sha256 "
        "as the identity of the frozen arrays")

    audit.host_prediction_reads += 1

    return HostPanel(
        market=market,
        host=host,
        artifact_path=str(path.relative_to(C.REPO_ROOT)).replace("\\", "/"),
        artifact_sha256=provenance["array_content_sha256"],
        n_episodes=n,
        episodes=episodes,
        timestamp=np.array(timestamp),
        context=context,
        y_true=y_true,
        host_pred=host_pred,
        segment=native,
        provenance=provenance,
    )


def _load_provenance(market: str, host: str) -> Dict[str, Any]:
    """Attach whatever freeze manifest the artifact's own directory publishes."""
    directory = artifact_dir(market, host)
    manifest_path = directory / FREEZE_MANIFEST_NAME
    if not manifest_path.exists():
        return {"freeze_manifest": None,
                "freeze_manifest_note": "artifact directory publishes no FREEZE_MANIFEST.json"}
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        return {"freeze_manifest": None,
                "freeze_manifest_note": f"unreadable FREEZE_MANIFEST.json: {exc}"}
    manifest = dict(manifest)
    manifest["_manifest_path"] = str(
        manifest_path.relative_to(C.REPO_ROOT)).replace("\\", "/")
    manifest["_manifest_sha256"] = sha256_file(manifest_path)
    return {"freeze_manifest": manifest, "freeze_manifest_note": None}


# --------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------
def audit_cell(market: str, host: str) -> Dict[str, Any]:
    """Resolve one coordinate to a readiness row. Never raises for a missing cell."""
    try:
        contract = load_market_contract(market)
    except HarnessError as exc:
        return readiness_row(
            market=market, host=host, status=BLOCKED_PROVENANCE,
            artifact_path="", artifact_sha256="", contract_path="", contract_sha256="",
            contract_source="", role_taxonomy="", lead_convention="",
            target_semantics="", origin_alignment="", horizon="",
            n_episodes="", n_host_train="", n_host_val="", n_post_train="",
            n_dev_eval="", n_protected_final_read="", contract_sealed_days="",
            n_short_episodes="", oof_folds="",
            blocker=f"dataset contract unusable: {exc}")

    try:
        panel = load_host_panel(market, host, contract=contract)
    except FileNotFoundError:
        return _row(market, host, contract, PENDING_EXTERNAL_HOST_ARTIFACT,
                    blocker=f"frozen Host artifact not present at "
                            f"{artifact_path(market, host)}")
    except ProtectedFinalAccess as exc:
        return _row(market, host, contract, BLOCKED_PROVENANCE,
                    blocker=f"sealed-role refusal: {exc}")
    except HarnessError as exc:
        return _row(market, host, contract, INVALID_SEMANTICS, blocker=str(exc))

    n_post = panel.role_count(C.ROLE_POST_TRAIN)
    n_dev = panel.role_count(C.ROLE_DEV_EVAL)
    n_short = sum(1 for e in panel.episodes if not e.is_full_day)

    blocker = ""
    status = READY_FROZEN

    if n_post == 0:
        return _row(market, host, contract, MISSING_ROLE, panel=panel,
                    blocker="frozen artifact exposes no POST_TRAIN episodes")
    if C.oof_fold_count(n_post) == 0:
        return _row(market, host, contract, MISSING_ROLE, panel=panel,
                    blocker=f"POST_TRAIN has {n_post} days; OOF needs at least "
                            f"{C.OOF_MIN_TRAIN_DAYS + C.OOF_MIN_HOLDOUT_DAYS}")
    if n_dev == 0:
        # POST_TRAIN-only fitting is still possible; this only blocks the future
        # evaluation stage, and is disclosed rather than hidden.
        blocker = "no DEV_EVAL episodes; fitting is possible but the future " \
                  "evaluation stage cannot cover this cell"

    return _row(market, host, contract, status, panel=panel, blocker=blocker,
                n_short=n_short)


def _row(market: str, host: str, contract: MarketContract, status: str,
         panel: Optional[HostPanel] = None, blocker: str = "",
         n_short: Optional[int] = None) -> Dict[str, Any]:
    n_episodes = panel.n_episodes if panel else ""
    counts = {r: (panel.role_count(r) if panel else "") for r in C.ROLE_ORDER[:-1]}
    if panel is not None and n_short is None:
        n_short = sum(1 for e in panel.episodes if not e.is_full_day)
    n_post = counts[C.ROLE_POST_TRAIN]
    folds = C.oof_fold_count(int(n_post)) if isinstance(n_post, int) and n_post else 0

    return readiness_row(
        market=market,
        host=host,
        status=status,
        artifact_path=(panel.artifact_path if panel else
                       str(artifact_path(market, host).relative_to(C.REPO_ROOT))
                       .replace("\\", "/")),
        artifact_sha256=(panel.artifact_sha256 if panel else ""),
        contract_path=contract.contract_path,
        contract_sha256=contract.contract_sha256,
        contract_source=contract.contract_source,
        role_taxonomy=contract.role_taxonomy,
        lead_convention=contract.lead_convention,
        target_semantics=f"DA -> next-{contract.horizon}h DA price",
        origin_alignment=("first flagged row of target day"
                          if contract.lead_convention == "hour_ending"
                          else "day-origin row of target day"),
        horizon=contract.horizon,
        n_episodes=n_episodes,
        n_host_train=counts[C.ROLE_HOST_TRAIN],
        n_host_val=counts[C.ROLE_HOST_VAL],
        n_post_train=counts[C.ROLE_POST_TRAIN],
        n_dev_eval=counts[C.ROLE_DEV_EVAL],
        # Rows of the frozen artifact carrying the sealed role -- structurally 0,
        # because the contract declares it closed and the vocabulary refuses it.
        n_protected_final_read=0,
        # Days the contract itself declares for the sealed role.  Disclosed so the
        # harness states plainly what it is not touching.  Never a read.
        contract_sealed_days=(contract.declared_sealed_days
                              if contract.declared_sealed_days is not None else ""),
        n_short_episodes=(n_short if n_short is not None else ""),
        oof_folds=folds,
        blocker=blocker,
    )


def build_readiness_table(audit: Optional[ReadAudit] = None) -> List[Dict[str, Any]]:
    """Audit all 20 registered coordinates in the fixed canonical order."""
    return [audit_cell(market, host) for market, host in C.CELLS]


def readiness_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    ready = [r for r in rows if r["status"] == READY_FROZEN]
    by_market: Dict[str, Dict[str, int]] = {}
    for row in rows:
        bucket = by_market.setdefault(row["market"], {})
        bucket[row["status"]] = bucket.get(row["status"], 0) + 1

    return {
        "n_coordinates": len(rows),
        "n_ready_frozen": len(ready),
        "status_counts": counts,
        "by_market": by_market,
        "ready_markets": sorted({r["market"] for r in ready}),
        "ready_hosts": sorted({r["host"] for r in ready}),
        "cells_with_dev_eval": sum(
            1 for r in rows
            if isinstance(r["n_dev_eval"], int) and r["n_dev_eval"] > 0),
        "total_oof_folds": sum(
            int(r["oof_folds"]) for r in rows if isinstance(r["oof_folds"], int)),
    }


def assert_all_ready(rows: Sequence[Dict[str, Any]]) -> None:
    """Refuse to execute unless every registered coordinate is READY_FROZEN."""
    bad = [r for r in rows if r["status"] != READY_FROZEN]
    if bad:
        detail = "; ".join(f"{r['market']}::{r['host']}={r['status']}" for r in bad)
        raise ReadinessError(f"not all coordinates are {READY_FROZEN}: {detail}")
