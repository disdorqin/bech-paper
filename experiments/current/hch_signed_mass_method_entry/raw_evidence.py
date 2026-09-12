"""Immutable raw prediction and branch evidence (PART A4).

Metrics alone are not evidence.  A metric block says *what* a configuration
scored; it cannot be re-derived, re-cut or independently recomputed from the
numbers it reports.  This module persists, for every evaluated
``(market, host, config, seed)``, the arrays themselves -- the frozen Host
prediction, the direct ``S_hat+``/``S_hat-``/``A_hat+``/``A_hat-`` branch outputs,
the raw correction, the pooled ``alpha``, the repaired prediction and the valid
mask -- so that the independent verifier recomputes every number from first
principles rather than re-reading a result table.

Three properties are enforced structurally rather than promised:

**Confined.**  Every write path is resolved and checked by :func:`_confine`
before anything is opened.  The destination must lie inside this experiment's
own evidence root, and must not lie inside any frozen Host/baseline root.  A
caller that passes a path pointing at the frozen breadth evidence gets an
exception, not a corrupted artifact directory.

**Immutable.**  An existing raw artifact is never overwritten: the digest of the
bytes that would be written is compared against the one already on disk, and an
identical rewrite is a no-op while a differing one raises.  Re-running a fit
therefore cannot silently replace the evidence a previous run produced.

**Self-describing.**  Every artifact carries the identity, the config switches,
the seed, and the provenance digests (frozen Host artifact, dataset contract,
harness source, core) that let a later reader tell whether two artifacts may
legitimately be compared or joined.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import config as C
from .contracts import EvidenceWriteError, sha256_file, write_csv

__all__ = ["RAW_INDEX_FIELDS", "raw_artifact_path", "confine", "write_raw_artifact",
           "read_raw_artifact", "read_raw_meta", "append_raw_index",
           "index_rows_from_artifact"]


#: The raw ``.npz`` payload.  Names are part of the evidence contract: the
#: verifier imports them by name and a rename is a breaking change.
RAW_ARRAY_KEYS = (
    "target",
    "host_prediction",
    "shape_positive",
    "shape_negative",
    "mass_positive",
    "mass_negative",
    "correction",
    "prediction",
    "valid_mask",
)

#: Everything else the artifact carries, as a JSON blob inside the same ``.npz``.
RAW_META_KEY = "meta_json"

RAW_INDEX_FIELDS = (
    "market",
    "host",
    "cell",
    "config",
    "seed",
    "partition",
    "role",
    "n_days",
    "n_entries_evaluated",
    "n_days_any_valid",
    "raw_path",
    "raw_sha256",
    "raw_bytes",
    "alpha",
    "amplitude_scale",
    "high_mass_q90_positive",
    "high_mass_q90_negative",
    "frozen_host_artifact_sha256",
    "dataset_contract_sha256",
    "source_sha256",
    "core_provenance_sha256",
    "git_commit",
    "switches",
    "generated_utc",
)


# --------------------------------------------------------------------------
# Confinement
# --------------------------------------------------------------------------
def _resolved(path: Any) -> Path:
    return Path(path).expanduser().resolve()


def evidence_root() -> Path:
    """The one directory this experiment may write inside.

    Named as a function so a test can point the harness at a temporary
    directory; production never overrides it.  Note that overriding the root
    does **not** relax the frozen-root refusal below -- those paths are absolute
    and stay absolute.
    """
    return _resolved(C.EVIDENCE_ROOT)


def confine(path: Any, what: str = "evidence write") -> Path:
    """Resolve ``path`` and refuse it unless it is legal to write there.

    Legal means: inside this experiment's own evidence root, and outside every
    frozen Host/baseline root.  Both halves matter.  The first stops a stray
    relative path from landing in the repository working tree; the second stops
    the one mistake that would be unrecoverable, writing over evidence another
    experiment already froze.
    """
    target = _resolved(path)
    root = evidence_root()

    for frozen in C.FROZEN_READ_ONLY_ROOTS:
        frozen_root = _resolved(frozen)
        if target == frozen_root or frozen_root in target.parents:
            raise EvidenceWriteError(
                f"{what} refused: {target} lies inside the frozen read-only root "
                f"{frozen_root}.  Frozen Host and baseline evidence is never "
                "rewritten by this experiment.")

    if target != root and root not in target.parents:
        raise EvidenceWriteError(
            f"{what} refused: {target} is outside this experiment's evidence "
            f"root {root}.")
    return target


def evidence_dir(*parts: str) -> Path:
    """A confined directory under the evidence root, created on demand."""
    target = confine(evidence_root().joinpath(*parts), "evidence directory")
    target.mkdir(parents=True, exist_ok=True)
    return target


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def _slug(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(text))


def raw_artifact_path(market: str, host: str, config_name: str, seed: int,
                      partition: str = "DEV_EVAL",
                      subdir: str = C.RAW_DIR_GRID) -> Path:
    """The canonical, deterministic path of one raw artifact."""
    name = f"{_slug(market)}__{_slug(host)}__{_slug(config_name)}__s{int(seed)}" \
           f"__{_slug(partition)}.npz"
    return confine(evidence_root() / subdir / name, "raw artifact path")


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def write_raw_artifact(arrays: Mapping[str, np.ndarray],
                       meta: Mapping[str, Any],
                       path: Any) -> Dict[str, Any]:
    """Persist one raw artifact, refusing to replace a differing one.

    Returns the index row's file-level fields (path, digest, size).  The write
    itself is atomic in the sense that matters here: the bytes are serialised
    into memory first, so a failure cannot leave a half-written artifact that a
    later reader would treat as evidence.

    There is deliberately no argument that permits replacement.  Re-running a fit
    whose arrays changed means something differed -- the code, the seed's
    environment, the frozen Host -- and that difference is exactly what a reader
    needs to see rather than have smoothed over.  An identical rewrite is the
    only re-run that is a no-op.
    """
    import io

    target = confine(path, "raw artifact write")

    payload: Dict[str, np.ndarray] = {}
    for key in RAW_ARRAY_KEYS:
        if key not in arrays:
            raise EvidenceWriteError(f"raw artifact is missing array {key!r}")
        payload[key] = np.ascontiguousarray(np.asarray(arrays[key]))
    payload[RAW_META_KEY] = np.frombuffer(
        json.dumps(dict(meta), ensure_ascii=False, sort_keys=True,
                   default=str).encode("utf-8"), dtype=np.uint8)

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    blob = buffer.getvalue()
    digest = _digest_bytes(blob)

    if target.exists():
        existing = target.read_bytes()
        if _digest_bytes(existing) == digest:
            return {"raw_path": str(target), "raw_sha256": digest,
                    "raw_bytes": len(existing), "written": False,
                    "note": "identical artifact already present; not rewritten"}
        raise EvidenceWriteError(
            f"raw artifact {target} already exists with different content "
            f"(on disk {sha256_file(target)}, incoming {digest}); refusing to "
            "overwrite immutable evidence")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    written = target.read_bytes()
    if _digest_bytes(written) != digest:
        raise EvidenceWriteError(
            f"raw artifact {target} did not round-trip to the bytes that were "
            "serialised; refusing to index a file whose digest is not its own")
    return {"raw_path": str(target), "raw_sha256": digest,
            "raw_bytes": len(written), "written": True}


def read_raw_artifact(path: Any) -> Dict[str, Any]:
    """Load one raw artifact back into arrays plus its metadata blob."""
    target = _resolved(path)
    if not target.exists():
        raise EvidenceWriteError(f"raw artifact is missing: {target}")
    out: Dict[str, Any] = {}
    with np.load(target, allow_pickle=False) as handle:
        for key in RAW_ARRAY_KEYS:
            out[key] = np.asarray(handle[key])
        out["meta"] = json.loads(bytes(handle[RAW_META_KEY]).decode("utf-8"))
    out["raw_path"] = str(target)
    out["raw_sha256"] = sha256_file(target)
    return out


def read_raw_meta(path: Any) -> Dict[str, Any]:
    """Just the metadata blob of one artifact, without its prediction arrays.

    ``np.load`` reads a member on access, so this does not materialise the
    arrays.  The raw-prediction index is built for hundreds of fits, and
    re-reading every prediction array to recover a JSON blob would be pure cost
    -- the arrays are already on disk for whoever needs to recompute from them.
    """
    target = _resolved(path)
    if not target.exists():
        raise EvidenceWriteError(f"raw artifact is missing: {target}")
    with np.load(target, allow_pickle=False) as handle:
        return json.loads(bytes(handle[RAW_META_KEY]).decode("utf-8"))


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------
def index_rows_from_artifact(meta: Mapping[str, Any],
                             written: Mapping[str, Any]) -> Dict[str, Any]:
    """One ``RAW_PREDICTION_INDEX.csv`` row from an artifact's metadata."""
    row = {field: meta.get(field) for field in RAW_INDEX_FIELDS}
    row.update({"raw_path": written["raw_path"],
                "raw_sha256": written["raw_sha256"],
                "raw_bytes": written["raw_bytes"]})
    return row


def append_raw_index(rows: Sequence[Mapping[str, Any]],
                     path: Optional[Any] = None) -> Path:
    """Merge rows into the raw-prediction index, keyed by identity.

    The index is *merged*, not appended blindly: a re-run of one fit replaces
    that fit's row instead of adding a duplicate, so the index stays a function
    of the artifacts rather than a log of the sessions that produced them.  The
    merge is keyed on (market, host, config, seed, partition).
    """
    if path is None:
        path = evidence_root() / "RAW_PREDICTION_INDEX.csv"
    target = confine(path, "raw index write")

    key_fields = ("market", "host", "config", "seed", "partition")
    merged: Dict[tuple, Dict[str, Any]] = {}
    if target.exists():
        existing = _read_index(target)
        for row in existing:
            merged[tuple(str(row.get(f, "")) for f in key_fields)] = row
    for row in rows:
        key = tuple(str(row.get(f, "")) for f in key_fields)
        merged[key] = {field: row.get(field) for field in RAW_INDEX_FIELDS}

    ordered: List[Dict[str, Any]] = sorted(
        merged.values(),
        key=lambda r: tuple(str(r.get(f, "")) for f in key_fields))
    for row in ordered:
        row["switches"] = _switches_text(row.get("switches"))
    write_csv(ordered, target, RAW_INDEX_FIELDS)
    return target


def _switches_text(value: Any) -> str:
    if value is None or isinstance(value, str):
        return "" if value is None else value
    try:
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _read_index(path: Path) -> List[Dict[str, str]]:
    """A minimal CSV reader, so the index round-trips without pandas."""
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln != ""]
    if not lines:
        return []
    header = _split_csv_line(lines[0])
    rows: List[Dict[str, str]] = []
    for line in lines[1:]:
        values = _split_csv_line(line)
        rows.append({name: (values[i] if i < len(values) else "")
                     for i, name in enumerate(header)})
    return rows


def _split_csv_line(line: str) -> List[str]:
    """Split one CSV line, honouring the quoting :func:`write_csv` emits."""
    out: List[str] = []
    field: List[str] = []
    quoted = False
    i = 0
    while i < len(line):
        ch = line[i]
        if quoted:
            if ch == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    field.append('"')
                    i += 1
                else:
                    quoted = False
            else:
                field.append(ch)
        elif ch == '"':
            quoted = True
        elif ch == ",":
            out.append("".join(field))
            field = []
        else:
            field.append(ch)
        i += 1
    out.append("".join(field))
    return out
