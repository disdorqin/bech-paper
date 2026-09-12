"""Test bootstrap for the signed-mass method harness.

The repository has no ``conftest.py``, no ``pytest.ini`` and no packaging metadata
at its root, so the package under test is not importable from a bare
``python -m pytest`` run.  This conftest puts the repository root on ``sys.path``
once, which is the same bootstrap the two script entry points use.

It deliberately does **not** add ``src`` -- that happens only through
``core_bridge.ensure_core_importable``, so the test suite exercises the same
single bridge the production path does.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from experiments.current.hch_signed_mass_method_entry import synthetic  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def synth():
    """A session-scoped synthetic dataset; deterministic, no market outcomes."""
    return synthetic.synthetic_dataset(n_features=6, seed=11)


@pytest.fixture(autouse=True)
def isolated_evidence_root(tmp_path, monkeypatch):
    """Never let a test write into the experiment's real evidence root.

    Autouse because the failure mode is silent and permanent.  A test that
    happens to reach a runner command writes a perfectly well-formed artifact
    into ``experiments/evidence/hch_signed_mass_method_20260912`` -- and a
    fabricated frozen-method record sitting in the evidence root is
    indistinguishable, to every later reader, from one the experiment produced.
    It is worse than a failing test: it is a false result that passes.

    The ``sandbox`` fixture redirects the same seam to its own tmp directory, so
    the two agree; this one exists so a test that forgets to ask for a sandbox
    still cannot touch the real tree.  Reads are unaffected: the readiness table
    and the frozen thresholds come from the frozen panel, not from here.
    """
    from experiments.current.hch_signed_mass_method_entry import raw_evidence

    root = (tmp_path / "evidence").resolve()
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(raw_evidence, "evidence_root", lambda: root)
    return root
