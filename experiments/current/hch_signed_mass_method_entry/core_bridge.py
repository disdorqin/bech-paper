"""The single, explicit bridge from this harness to ``src/core``.

``src/core`` is the scientific object.  This package imports it and never
re-implements it, edits it, or copies its logic.  The bridge exists so that the
import is one auditable statement rather than a scattering of ``sys.path`` hacks.

``tests/test_purity.py`` asserts that no module in this package defines a model,
loss, fusion, calibration or sampling routine of its own.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any, List

from . import config as C

__all__ = ["ensure_core_importable", "core", "CORE_EXPORTS_USED"]

#: Exactly the ``src.core`` names this harness consumes.  Anything not on this
#: list is not used, which makes the surface auditable at a glance.
CORE_EXPORTS_USED = (
    # contracts
    "RawInputs",
    "ForecastOriginBatch",
    "SignedMassTargets",
    "RepairOutput",
    "ModelConfig",
    # preprocessing
    "DeterministicFeatureBuilder",
    "TrainFrozenScaler",
    "fit_residual_scale",
    "base_token_dim",
    "shape_channel_dim",
    "SHAPE_HISTORY_CHANNELS",
    "AMPLITUDE_HISTORY_CHANNELS",
    # model
    "SignedMassRepairModel",
    "default_config",
    # fusion
    "decompose_residual",
    "fuse",
    "repair",
    # losses
    "combined_loss",
    "repair_mae",
    "shape_loss",
    "amplitude_loss",
    "fit_mass_normalization",
    # calibration
    "fit_mae_scalar",
    "apply_mae_scalar",
    "mae_objective",
    "weighted_median",
    # sampling.  The numeric stratum ids are deliberately *not* on this list:
    # ``src/core`` does not re-export them, and the harness reads stratum
    # populations through
    # ``RareMassSampler.stratum_counts()``, which returns them by name.  Reading
    # a private id would couple the harness to a core internal it has no need for.
    "RareMassSampler",
    "RareMassSamplerConfig",
)

_CORE: ModuleType | None = None


def ensure_core_importable() -> str:
    """Put the repository's ``src`` directory on ``sys.path`` (idempotently)."""
    src = str(C.REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def core() -> ModuleType:
    """Import and return ``src.core``, verifying every name this harness needs."""
    global _CORE
    if _CORE is None:
        ensure_core_importable()
        import core as _core  # noqa: PLC0415  (deliberate lazy import)

        missing: List[str] = [n for n in CORE_EXPORTS_USED if not hasattr(_core, n)]
        if missing:
            raise ImportError(
                "src/core does not export the names this harness requires: "
                f"{missing}. The harness must not paper over a core API change.")
        _CORE = _core
    return _CORE


def attr(name: str) -> Any:
    """Fetch one ``src.core`` name through the bridge."""
    if name not in CORE_EXPORTS_USED:
        raise KeyError(f"{name} is not on the audited core-export list")
    return getattr(core(), name)


def core_provenance() -> dict:
    """Locate the exact ``src/core`` source tree this run is bound to."""
    from .contracts import sha256_file

    root = Path(core().__file__).resolve().parent
    digests = {}
    for path in sorted(root.glob("*.py")):
        digests[path.name] = sha256_file(path)
    return {
        "core_package": str(root.relative_to(C.REPO_ROOT)).replace("\\", "/"),
        "core_module_files": digests,
        "core_tree_digest": _tree_digest(digests),
        "exports_used": list(CORE_EXPORTS_USED),
    }


def _tree_digest(digests: dict) -> str:
    from .contracts import sha256_bytes

    payload = "|".join(f"{k}:{v}" for k, v in sorted(digests.items()))
    return sha256_bytes(payload.encode("ascii"))
