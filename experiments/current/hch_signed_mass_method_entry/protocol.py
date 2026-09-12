"""Protocol registrations frozen before the first ``DEV_EVAL`` number exists (A3).

One thing lives here today: the per-cell high-mass threshold pair ``q90(A+)`` /
``q90(A-)``, fitted on that cell's ``POST_TRAIN`` rows and never recomputed.

Why it is a module rather than three lines inside the grid runner.  The threshold
decides which ``DEV_EVAL`` days count as high-mass, so a threshold fitted *after*
seeing a ``DEV_EVAL`` metric would let the metric choose its own subset -- the
classic way a tail number becomes unfalsifiable.  Freezing it is therefore not a
bookkeeping step but the thing that makes ``high_mass_*_l1`` mean anything, and it
has to hold across processes: the grid runs in workers that each rebuild their
own dataset.

So the rule is enforced at the point of use.  :func:`high_mass_for` **refuses**
to invent a threshold when the frozen record is absent, rather than re-fitting one
from whatever data happens to be at hand.  A re-fit would usually produce the same
two numbers, which is exactly what makes it dangerous: the run would look correct
while the provenance claim ("defined once, before any ``DEV_EVAL`` metric") had
quietly become false.

This module reads and assembles; it never writes.  The single write of
``HIGH_MASS_THRESHOLDS.json`` happens in ``runner.py``, which is registered with
the pre-execution verifier as the one place allowed to create protocol files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import config as C
from . import oof
from . import raw_evidence as RE
from .contracts import HarnessError

__all__ = ["HIGH_MASS_RELATIVE", "HIGH_MASS_SCHEMA", "high_mass_path",
           "build_threshold_records", "high_mass_for", "load_thresholds",
           "unusable_records"]

HIGH_MASS_RELATIVE = "00_protocol/HIGH_MASS_THRESHOLDS.json"
HIGH_MASS_SCHEMA = "signed_mass_high_mass_thresholds.v1"


def high_mass_path() -> Path:
    """The confined path of the frozen threshold registration."""
    return RE.confine(RE.evidence_root() / HIGH_MASS_RELATIVE,
                      "high-mass threshold registration")


def build_threshold_records(cells: Sequence[Tuple[str, str]],
                            build: Callable[[str, str], Any]
                            ) -> List[Dict[str, Any]]:
    """Fit one threshold record per cell, from ``POST_TRAIN`` only.

    ``build`` is injected for the same reason the grid injects it: this function
    must not be able to reach an artifact the caller did not authorize.  The
    records are returned sorted by cell so the file is a function of the panel
    rather than of the order the cells happened to be visited.
    """
    records = [oof.fit_high_mass_thresholds(build(market, host))
               for market, host in cells]
    return sorted(records, key=lambda r: str(r["cell"]))


def load_thresholds(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The registration payload, or ``None`` when nothing has been frozen yet."""
    target = Path(path) if path is not None else high_mass_path()
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def high_mass_for(market: str, host: str,
                  path: Optional[Path] = None) -> Dict[str, Any]:
    """The frozen ``q90`` pair for one cell, or a refusal.

    Deliberately a lookup and not a computation.  The caller that wants a
    threshold for a cell it has never frozen must run the freezing step first,
    because the difference between the two is the whole of A3.
    """
    payload = load_thresholds(path)
    coordinate = f"{market}::{host}"
    if payload is None:
        raise HarnessError(
            f"no frozen high-mass threshold registration at "
            f"{path or high_mass_path()}; fit it before producing any DEV_EVAL "
            "metric rather than re-deriving it here")
    for record in payload.get("cells", []):
        if str(record.get("cell")) == coordinate:
            return dict(record)
    raise HarnessError(
        f"{coordinate} has no frozen high-mass threshold record; the cell set "
        "was frozen without it and a threshold fitted now would be chosen after "
        "the evaluation it governs")


def unusable_records(payload: Optional[Mapping[str, Any]]) -> List[str]:
    """Cells whose frozen pair cannot identify a non-empty high-mass subset.

    Reported, never repaired: a degenerate threshold is a fact about the cell's
    ``POST_TRAIN`` mass distribution, and the honest response is the
    ``NOT_APPLICABLE_N0`` status the metric layer already emits for it.
    """
    out: List[str] = []
    for record in (payload or {}).get("cells", []):
        for sign in ("positive", "negative"):
            if not record.get(f"usable_{sign}", False):
                out.append(f"{record.get('cell')}:{sign}")
    return out
