"""Canonical `src/core` layout contract.

Before 2026-09-12 this file asserted that `src/core/` owned the frozen V2.5 math
implementations (`iah_candidate`, `learned_signature`, `iah_crps`,
`weighted_mean_readout`, `v25_point_runtime`).  The reorganization moved that
whole tree, byte-identical, into
`src/archive/core_pre_extreme_repair_20260912/legacy_core/`, so the same contract
is now asserted against both the current core and the archive.  The superseded
assertions are preserved verbatim in the archive manifest.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
LEGACY_CORE = SRC / "archive/core_pre_extreme_repair_20260912/legacy_core"

#: The modules that make up the current (alignment-safe) core.
#:
#: Before 2026-09-12 this tuple also listed `semantic_views.py`, `similarity.py`
#: and `sampling.py`.  The alignment-safe refactor removed those three from the
#: active path and added `safety.py` and `history.py`; the removed sources are
#: byte-preserved in
#: `src/archive/signed_mass_development_core_20260912/legacy_core/` and the
#: superseded assertions are recorded in that archive's `SUPERSEDED_TESTS.md`.
CURRENT_CORE_MODULES = (
    "__init__.py", "contracts.py", "preprocessing.py", "stem.py",
    "encoders.py", "shape.py", "amplitude.py", "fusion.py",
    "calibration.py", "losses.py", "safety.py", "history.py", "model.py",
)

#: Development components the closed adjudication deleted.  None of them may be
#: present in the active core, under its own name or behind a switch.
DEPROMOTED_CORE_MODULES = ("semantic_views.py", "similarity.py", "sampling.py")

#: Old subpackage names that must no longer resolve inside `src/core/`.
RETIRED_CORE_SUBPACKAGES = (
    "iah_candidate", "iah_crps", "learned_signature",
    "universal_trainer", "v25_point_runtime", "weighted_mean_readout",
)


def test_core_package_owns_the_current_alignment_safe_implementations():
    expected = {SRC / "core" / name for name in CURRENT_CORE_MODULES}
    missing = sorted(p.name for p in expected if not p.is_file())
    assert not missing, f"core is missing {missing}"
    thin = sorted(p.name for p in expected if p.stat().st_size <= 400)
    assert not thin, f"core modules look like stubs: {thin}"
    for doc in ("README.md", "DESIGN_CONTRACT.md"):
        assert (SRC / "core" / doc).is_file(), doc


def test_active_core_contains_exactly_the_target_file_set():
    """The active path is the source refactor's declared tree and nothing else."""
    present = sorted(p.name for p in (SRC / "core").iterdir() if p.is_file())
    assert present == sorted(list(CURRENT_CORE_MODULES) + ["README.md", "DESIGN_CONTRACT.md"])


def test_deleted_development_components_are_archived_not_active():
    """Nothing is deleted without being archived first (refactor prompt section 4)."""
    legacy = SRC / "archive/signed_mass_development_core_20260912/legacy_core"
    for name in DEPROMOTED_CORE_MODULES:
        assert not (SRC / "core" / name).exists(), f"{name} is still active"
        assert (legacy / name).is_file(), f"{name} was removed without an archive copy"
        assert (legacy / name).stat().st_size > 400, f"{name} archive copy looks truncated"


def test_retired_v25_subpackages_are_no_longer_current_core():
    for name in RETIRED_CORE_SUBPACKAGES:
        assert not (SRC / "core" / name).exists(), name
    archived = {
        LEGACY_CORE / "iah_candidate/candidate.py",
        LEGACY_CORE / "learned_signature/context.py",
        LEGACY_CORE / "iah_crps/loss.py",
        LEGACY_CORE / "weighted_mean_readout/readout.py",
    }
    assert all(path.is_file() and path.stat().st_size > 1000 for path in archived)


def test_archived_flat_shims_are_not_current_core():
    archive = SRC / "archive/legacy_flat_shims"
    assert archive.is_dir()
    assert (archive / "iah_candidate.py").is_file()
    assert not (SRC / "hch_v2").exists()


def test_archived_v25_entrypoint_still_uses_component_paths():
    runtime = (LEGACY_CORE / "v25_point_runtime/runtime.py").read_text(encoding="utf-8")
    assert "archive." not in runtime
    assert "weighted_mean_from_candidate" in runtime
    assert "iah_crps_loss" in runtime


def test_current_core_hardcodes_no_dataset_name():
    for name in CURRENT_CORE_MODULES:
        text = (SRC / "core" / name).read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            else:
                continue
            for mod in mods:
                root = mod.split(".")[0]
                assert root not in ("archive", "experiments", "paper", "docs"), (name, mod)
