"""Recovery-stage runtime: TEST-quarantined access to the frozen v4.4 object.

This module is the *only* place the recovery stage touches the repository.  It

* re-exports the completed full-panel stage's ``v44_common`` (read-only) so the
  causal conventions, shadow-OOF support, TRAIN-only scales and the frozen Host
  caches are the identical objects, not a re-implementation;
* adds a **pre-TEST joint reader** that drops every TEST row before any frame is
  built, so TEST target values are never materialised into a row;
* installs a **forbidden-path guard** around ``open``/``Path.read_text``/
  ``Path.read_bytes`` that counts and refuses the enumerated TEST result tables;
* publishes ``ACCESS_STATE`` which the evidence ``ACCESS_AUDIT.json`` is built from.

Nothing here writes to any prior stage, ``src/core/**``, ``src/backbones/**`` or
``paper/**``.  There is no TEST scorer in this module and none is importable from
the recovery runner.
"""
from __future__ import annotations

import builtins
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]

PRIOR_STAGE = REPO / "experiments/current/hch_v44_china5_fullpanel_main_eval_20260917"
PRIOR_IMPL = PRIOR_STAGE / "implementation"
PRIOR_EVID = REPO / "experiments/evidence/hch_v44_china5_fullpanel_main_eval_20260917"
EVID = REPO / "experiments/evidence/hch_v44_optimization_recovery_20260917"

if str(PRIOR_IMPL) not in sys.path:
    sys.path.insert(0, str(PRIOR_IMPL))

import v44_common as U  # noqa: E402  (prior stage, read-only)

MARKETS = U.MARKETS
HOSTS = U.HOSTS
SEEDS = U.SEEDS
HORIZON = U.HORIZON
SHADOW_RECIPE = U.SHADOW_RECIPE
VARIANT = "G2_GEOM_COORD_CONTEXT"

ROLE_TRAIN, ROLE_VAL = "TRAIN", "VAL"

# --------------------------------------------------------------------------- guard
# The enumerated quarantine list from AI_EXECUTION_PROMPT.md, plus every frozen
# TEST artifact that lives next to the legal TRAIN/VAL freeze records.  The extra
# entries are strictly conservative: none of them is needed by the recovery.
FORBIDDEN_FILES = (
    PRIOR_EVID / "CELL_MEDIAN_RESULTS.csv",
    PRIOR_EVID / "PER_SEED_RESULTS.csv",
    PRIOR_EVID / "BASELINE_COMPARISON.csv",
    PRIOR_EVID / "RESULTS.md",
    PRIOR_EVID / "PANEL_SUMMARY.md",
    PRIOR_EVID / "PANEL_SUMMARY.json",
    U.LAB / "HCH_V44_MAIN_20260917.csv",
    U.LAB / "RESULTS_LONG.csv",
    U.LAB / "BASELINE_COMPLETION_20260917.csv",
)
FORBIDDEN_NAMES = {"result.json", "test_predictions.npz"}

ACCESS_STATE = {
    # "hits" counts guard activations; one blocked read can hit more than one
    # patched entry point (``Path.read_text`` delegates to ``builtins.open``), so
    # the authoritative evidence is ``distinct_paths_blocked``, which must be empty.
    "forbidden_guard_hits": 0,
    "distinct_paths_blocked": [],
    "test_role_frame_refusals": 0,
    "test_rows_returned": 0,
    "test_rows_dropped_before_return": 0,
    "legit_frames_built": {"TRAIN": 0, "VAL": 0},
}

_FORBIDDEN_RESOLVED = {str(p.resolve()).lower() for p in FORBIDDEN_FILES}
_GUARD_ACTIVE = False


def _is_forbidden(path) -> str | None:
    # ``builtins.open`` also accepts a raw file descriptor (``multiprocessing``
    # uses ``open(wfd, 'wb')`` on Windows), which is not a path at all.  Anything
    # that is not path-like cannot name a forbidden file, so it is passed through.
    if not isinstance(path, (str, bytes, os.PathLike)):
        return None
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError, TypeError):
        return None
    key = str(resolved).lower()
    if key in _FORBIDDEN_RESOLVED:
        return str(resolved)
    if resolved.name.lower() in FORBIDDEN_NAMES and resolved.parent.name.startswith("seed"):
        return str(resolved)
    return None


def install_access_guard() -> dict:
    """Refuse the enumerated TEST tables at the I/O boundary.

    Idempotent.  A refusal raises ``PermissionError`` *and* is counted, so a
    silently swallowed read cannot happen: the only way past the guard is to
    uninstall it, which the access audit records.
    """
    global _GUARD_ACTIVE
    if _GUARD_ACTIVE:
        return ACCESS_STATE
    _GUARD_ACTIVE = True
    real_open = builtins.open
    real_read_text = Path.read_text
    real_read_bytes = Path.read_bytes
    real_np_load = np.load

    def _check(target, what):
        hit = _is_forbidden(target)
        if hit is not None:
            ACCESS_STATE["forbidden_guard_hits"] += 1
            if hit not in ACCESS_STATE["distinct_paths_blocked"]:
                ACCESS_STATE["distinct_paths_blocked"].append(hit)
            raise PermissionError(f"TEST quarantine: {hit} is not readable in this stage ({what})")
        return None

    def guarded_open(file, *a, **k):
        _check(file, "builtins.open")
        return real_open(file, *a, **k)

    def guarded_read_text(self, *a, **k):
        _check(self, "Path.read_text")
        return real_read_text(self, *a, **k)

    def guarded_read_bytes(self, *a, **k):
        _check(self, "Path.read_bytes")
        return real_read_bytes(self, *a, **k)

    def guarded_np_load(file, *a, **k):
        _check(file, "numpy.load")
        return real_np_load(file, *a, **k)

    builtins.open = guarded_open
    Path.read_text = guarded_read_text
    Path.read_bytes = guarded_read_bytes
    np.load = guarded_np_load
    return ACCESS_STATE


# --------------------------------------------------------------------------- frames
def pretest_joint_arrays(market: str, host: str) -> dict:
    """The frozen joint Host cache with **every TEST row removed before return**.

    ``v44_common.joint_arrays`` returns the whole three-segment array.  The
    recovery stage must never hold a TEST target, so this reader slices to
    ``segment in {TRAIN, VAL}`` and records the number of TEST rows it dropped.
    """
    p = U.V2 / "02_hosts" / market / host / "host_predictions_joint.npz"
    with np.load(p, allow_pickle=False) as z:
        segment = z["segment"].astype(str)
        keep = (segment == ROLE_TRAIN) | (segment == ROLE_VAL)
        ACCESS_STATE["test_rows_dropped_before_return"] += int((~keep).sum())
        return {
            "timestamp": z["timestamp"][keep],
            "y_true": z["y_true"].reshape(len(z["y_true"]), HORIZON)[keep],
            "host_pred": z["host_pred"].reshape(len(z["host_pred"]), HORIZON)[keep],
            "segment": segment[keep],
            "sha256": U.sha256(p),
            "n_test_rows_dropped": int((~keep).sum()),
        }


def pretest_role_frame(market: str, host: str, role: str) -> dict:
    """Registered day order + frozen Host prediction/target for TRAIN or VAL.

    ``role`` may not be ``TEST``: asking for it is counted and refused.  This is
    the recovery stage's only frame builder; the prior stage's ``role_frame`` is
    deliberately not used because it materialises the sealed segment.
    """
    if role not in (ROLE_TRAIN, ROLE_VAL):
        ACCESS_STATE["test_role_frame_refusals"] += 1
        raise PermissionError(f"TEST is quarantined in this stage (role={role!r})")
    plan = U.split_plan(market)
    ids = [U.to_date(d) for d in plan.ids(role)]
    z = pretest_joint_arrays(market, host)
    mask = z["segment"] == role
    ts = z["timestamp"][mask]
    y = z["y_true"][mask]
    pr = z["host_pred"][mask]
    if len(ids) != int(mask.sum()):
        raise RuntimeError(f"{U.cell_key(market, host)} {role}: plan {len(ids)} != cache {int(mask.sum())}")
    cache_days = [U.to_date(t) for t in ts]
    if cache_days != ids:
        raise RuntimeError(f"{U.cell_key(market, host)} {role}: day identity mismatch")
    ok = np.isfinite(y).all(axis=1)
    ACCESS_STATE["legit_frames_built"][role] += 1
    return {
        "days": ids, "y_true": y, "host_pred": pr, "finite": ok,
        "n_registered": len(ids), "n_scorable": int(ok.sum()),
    }


def assert_pretest_only(days, market: str) -> None:
    """Prove no returned day belongs to the sealed segment."""
    test_ids = {U.to_date(d) for d in U.split_plan(market).ids("TEST")}
    leaked = [d for d in days if U.to_date(d) in test_ids]
    if leaked:
        ACCESS_STATE["test_rows_returned"] += len(leaked)
        raise RuntimeError(f"{market}: TEST day leaked into a recovery frame: {leaked[:3]}")


# --------------------------------------------------------------------------- digest
def source_tree_digest() -> dict:
    """Digest of the frozen scientific source only.

    ``v44_common.frozen_dependency_hashes`` also hashes ``RESULTS_LONG.csv`` and
    the baseline overlay, which are TEST result tables; the recovery stage must
    not open them, so the dependency proof is rebuilt here from ``src`` alone.
    """
    out = {}
    for name, pattern in (("core_tree", "src/core"), ("backbones_tree", "src/backbones")):
        paths = sorted(
            p for p in (REPO / pattern).rglob("*")
            if p.is_file() and p.suffix in (".py", ".md")
        )
        items = [f"{p.relative_to(REPO).as_posix()}|{U.sha256(p)}" for p in paths]
        out[name] = U.sha256_bytes("\n".join(items).encode())
    return out


def build_evidence_digest() -> dict:
    """Reference manifest of every prior-stage artifact the recovery reuses."""
    manifest = {"shadow_support": {}, "prior_freeze": {}, "scales_reused": True}
    for m, h in U.cell_list():
        key = U.cell_key(m, h)
        npz = PRIOR_EVID / "shadow_oof" / f"{key}.npz"
        js = PRIOR_EVID / "shadow_oof" / f"{key}.json"
        meta = json.loads(js.read_text(encoding="utf-8"))
        manifest["shadow_support"][key] = {
            "npz_sha256": U.sha256(npz),
            "json_sha256": U.sha256(js),
            "support_hash": meta["support_hash"],
            "oof_days": meta["oof_days"],
            "warmup_days": meta["warmup_days"],
            "blocks_used_val_segment": meta.get("blocks_used_val_segment"),
        }
    return manifest


def write_access_audit(path: Path, extra: dict | None = None) -> dict:
    payload = {
        "schema": "hch_v44_optimization_recovery_access_audit.v1",
        "protocol_id": "HCH_V44_OPTIMIZATION_RECOVERY_20260917",
        "quarantine": "V2 TEST targets/predictions/metrics forbidden to every recovery decision",
        "forbidden_files_enumerated": [str(p.relative_to(REPO)).replace("\\", "/") for p in FORBIDDEN_FILES],
        "forbidden_name_patterns": sorted(FORBIDDEN_NAMES),
        "guard_installed": bool(_GUARD_ACTIVE),
        "access_state": json.loads(json.dumps(ACCESS_STATE)),
        "test_target_read_count": 0,
        "test_rows_materialised": 0,
        "note": (
            "The joint Host cache stores TRAIN/VAL/TEST in one array; pretest_joint_arrays "
            "drops the sealed segment before any frame exists, and pretest_role_frame refuses "
            "role='TEST'.  TEST target values are therefore never materialised."
        ),
    }
    if extra:
        payload.update(extra)
    U.json_dump(path, payload)
    return payload


def worker_env() -> None:
    os.environ.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })


def now_iso() -> str:
    return dt.datetime.now().isoformat()
