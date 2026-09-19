"""Bounded stage-local contract layer for the NORD_DK1 / NEM_SA1 confirmation.

This stage is the bounded successor authorised by
``docs/current/HCH_S1_B2_NORD_NEM_CONFIRMATION_20260919.md`` after the four-market
international stage blocked at P0.  It does not re-implement the frozen contract
layer and it does not edit it either: ``intl_shared.py`` is loaded **by path, under
its frozen digest**, and the only mutation applied to it is the rebinding of the
five stage-local globals below.

Everything else -- the two-family loader, the timestamp-projected window reader,
the manifest-gated legal-state reader, the frozen-Host identity helpers and the
protected-seal gate -- is the frozen object itself.  No frozen behaviour is
re-typed here, so none of it can drift from the layer the four-market stage already
exercised, and ``frozen_layer()`` reports the rebinding so the independent verifier
can recompute it rather than trust it.

============================  ==================================================
``MARKETS``                   ``("NORD_DK1", "NEM_SA1")`` -- the two markets whose
                              frozen Hosts re-derive bit-exactly.  LAGO_NP and
                              GEFCOM14P become structurally unreachable from this
                              stage, which is what keeps the four-market blocker
                              unrelaxed rather than merely unmentioned.
``CELLS``                     the 8 bounded cells.
``STAGE`` / ``IMPL``          this stage's own directories.
``EVID``                      this stage's own evidence root.
``P0_TOKEN``                  this stage's own seal, ``EVID/P0_PASS.json``.
============================  ==================================================

The seal rule is inherited unchanged: reading a ``PROTECTED_FINAL`` coordinate
raises until a P0 PASS token that declares zero protected reads exists on disk.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()

STAGE = ROOT / "experiments/current/hch_s1_b2_nord_nem_confirmation_20260919"
EVID = ROOT / "experiments/evidence/hch_s1_b2_nord_nem_confirmation_20260919"
IMPL = STAGE / "implementation"

#: The closed four-market stage whose contract layer this stage reuses read-only.
PRIOR_STAGE = ROOT / "experiments/current/hch_s1_b2_international_confirmation_20260919"
PRIOR_SHARED = PRIOR_STAGE / "implementation/intl_shared.py"
#: The digest that layer was published under (``P0_BLOCKED.json`` provenance).
PRIOR_SHARED_SHA256 = "617D8FDFAE228ADC0EB0AC4164983CB4755147EDA5465588477C3E2D3163356C"

MARKETS = ("NORD_DK1", "NEM_SA1")
HOSTS = ("PatchTST", "TimeMixer", "iTransformer", "LSTM")
SEEDS = (7, 17, 37)
CELLS = tuple((m, h) for m in MARKETS for h in HOSTS)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def _load_by_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_frozen_sha = sha(PRIOR_SHARED)
if _frozen_sha != PRIOR_SHARED_SHA256:
    raise RuntimeError(
        f"the reused contract layer is not the frozen one: {PRIOR_SHARED} hashes to "
        f"{_frozen_sha}, published as {PRIOR_SHARED_SHA256}")

_F = _load_by_path(PRIOR_SHARED, "nn_frozen_intl_shared")

# The whole of this stage's liberty over the frozen layer, in five assignments.
_REBOUND = {
    "STAGE": STAGE,
    "EVID": EVID,
    "IMPL": IMPL,
    "MARKETS": MARKETS,
    "CELLS": CELLS,
    "P0_TOKEN": EVID / "P0_PASS.json",
}
for _k, _v in _REBOUND.items():
    setattr(_F, _k, _v)
del _k, _v

# These must never be rebound: if a future edit widened the liberty above, the
# stage would silently stop being the bounded one.
_LOCKED = {
    "HOSTS": HOSTS,
    "SEEDS": SEEDS,
    "FIT_ROLE": "POST_TRAIN",
    "EVAL_ROLE": "PROTECTED_FINAL",
    "PIR_LEGAL_HOSTS": ("PatchTST", "TimeMixer"),
    "BLOCKED_METHODS": ("UEC-STD", "OMPB"),
    "OPEN_ROLES": ("HOST_TRAIN", "HOST_VAL", "POST_TRAIN", "DEV_EVAL"),
}
for _k, _v in _LOCKED.items():
    if getattr(_F, _k) != _v:
        raise RuntimeError(f"frozen layer {_k} is {getattr(_F, _k)!r}, expected {_v!r}")
del _k, _v


def frozen_layer() -> dict:
    """What this stage rebound, what it refused to rebound, and under what digest."""
    return {
        "reused_module": str(PRIOR_SHARED),
        "reused_module_sha256": _frozen_sha,
        "published_sha256": PRIOR_SHARED_SHA256,
        "digest_verified_before_rebinding": _frozen_sha == PRIOR_SHARED_SHA256,
        "rebound": {k: [str(x) for x in v] if isinstance(v, tuple) else str(v)
                    for k, v in _REBOUND.items()},
        "locked_unrebound": {k: list(v) if isinstance(v, tuple) else v
                             for k, v in _LOCKED.items()},
        "markets_reachable": list(_F.MARKETS),
        "excluded_markets": ["LAGO_NP", "GEFCOM14P"],
        "excluded_markets_reachable": False,
        "seal_rule": ("a PROTECTED_FINAL coordinate raises until EVID/P0_PASS.json "
                      "exists and declares zero protected reads"),
    }


def __getattr__(name: str):
    """Everything not rebound above is the frozen layer's own attribute."""
    try:
        return getattr(_F, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()) | set(dir(_F)))
