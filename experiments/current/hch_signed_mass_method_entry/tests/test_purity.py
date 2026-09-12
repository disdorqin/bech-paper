"""The harness defines no science of its own.

``core_bridge`` claims this file exists, so it does.  The claim is narrow and
checkable: every name in ``src/core`` that the protocol calls the scientific
object appears in exactly one place -- the bridge's own allow-list -- and no
module here defines a second implementation of any of them.  A harness that
quietly re-derived the weighted median, the shape loss or the fusion would make
the frozen core decorative, which is the failure mode this guards.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from experiments.current.hch_signed_mass_method_entry import config as C
from experiments.current.hch_signed_mass_method_entry.core_bridge import (
    CORE_EXPORTS_USED, attr, core)
from experiments.current.hch_signed_mass_method_entry.contracts import HarnessError

PACKAGE_DIR = Path(__file__).resolve().parents[1]
BRIDGE = "core_bridge.py"

#: Routines that belong to ``src/core``.  A definition of any of these here
#: would be a second scientific object living in the harness.
RESERVED = (
    "shape_loss", "amplitude_loss", "combined_loss", "repair_mae",
    "mae_objective", "fit_mae_scalar", "apply_mae_scalar", "weighted_median",
    "decompose_residual", "fuse", "shape_from_residual", "amplitude_from_residual",
    "TrainFrozenScaler", "DeterministicFeatureBuilder", "RareMassSampler",
    "SignedMassRepairModel", "within_day_robust_normalize",
)

#: The only modules allowed to touch ``sys.path``: the bridge, plus the two
#: script entry points, which bootstrap the repository root so they can be run
#: as plain files.  This is exactly the exemption the pre-execution verifier
#: applies; the two must not drift apart.
_SYS_PATH_ALLOWED = {BRIDGE, "runner.py", "verify_preexecution.py"}

#: Modules whose string literals are legitimately allowed to name a market or a
#: Host: the registries themselves, and the reader that maps Host artifacts.
_REGISTRY_FILES = {"config.py", "host_prediction_loader.py"}


def _package_sources():
    """The package's production modules -- the same set the verifier scans."""
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"))


def test_no_module_defines_a_core_routine():
    offenders: list = []
    for path, tree in _package_sources():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                if node.name in RESERVED:
                    offenders.append(f"{path.name}:{node.lineno}:{node.name}")
    assert not offenders, offenders


def test_no_core_module_is_imported_directly():
    """``src.core`` is reachable only through the bridge."""
    offenders: list = []
    for path, tree in _package_sources():
        if path.name == BRIDGE:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "core":
                offenders.append(f"{path.name}:{node.lineno}:from {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "core":
                        offenders.append(f"{path.name}:{node.lineno}:import {alias.name}")
    assert not offenders, offenders


def test_sys_path_is_manipulated_only_in_the_bridge():
    offenders: list = []
    for path, tree in _package_sources():
        if path.name in _SYS_PATH_ALLOWED:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "path" \
                    and isinstance(node.value, ast.Name) and node.value.id == "sys":
                offenders.append(f"{path.name}:{node.lineno}:sys.path")
    assert not offenders, offenders


def test_every_core_attribute_used_is_declared_in_the_bridge():
    """The consumption surface is exactly ``CORE_EXPORTS_USED``."""
    assert len(CORE_EXPORTS_USED) == len(set(CORE_EXPORTS_USED))
    assert attr  # the only sanctioned accessor
    for name in CORE_EXPORTS_USED:
        assert hasattr(core(), name), name
    # And nothing else is reached for: ``attr`` refuses undeclared names, so an
    # undeclared core internal cannot be borrowed by naming it.
    for name in ("not_a_registered_core_name", "__import__", "torch",
                 "_private_helper"):
        with pytest.raises(KeyError):
            attr(name)


def test_the_ban_list_is_intact_and_self_consistent():
    assert set(C.SWITCH_NAMES) == set(C.FULL_SWITCHES)
    assert len(C.SWITCH_NAMES) == 4, C.SWITCH_NAMES
    assert C.HORIZON == 24 and C.N_CALENDAR_FEATURES == 7
    # Appendix-A structural bans, as recorded in config rather than as prose.
    banned = " ".join(C.FORBIDDEN_SOURCE_TOKENS)
    for token in ("Gate", "TrustGate", "RepairabilityGate", "BenefitGate",
                  "ConfidenceGate", "MultiheadAttention", "TransformerEncoder",
                  "MixtureOfExperts"):
        assert token in banned, token
    # And the ban list is a closed, audited set -- not a growing heuristic.
    assert len(C.FORBIDDEN_SOURCE_TOKENS) == 8, C.FORBIDDEN_SOURCE_TOKENS


def test_the_harness_owns_no_numeric_constant_of_its_own():
    """No module here may carry a hard-coded loss weight or model dimension.

    Everything of that kind is either the registered protocol value in
    :mod:`config` or comes from ``src/core``'s own defaults.
    """
    suspicious = {"hidden", "dropout", "temperature"}
    for path, tree in _package_sources():
        if path.name in {"config.py", "synthetic.py", "verify_preexecution.py"}:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else "")
            # The core's own config factory may only be driven by registered
            # values or by values recovered from the fitted state.
            if name in ("default_config", "RareMassSamplerConfig"):
                for kw in node.keywords:
                    if kw.arg in suspicious:
                        raise AssertionError(
                            f"{path.name}:{node.lineno}: {name}({kw.arg}=...) "
                            "hard-codes a model dimension")


def test_config_constants_match_the_registered_protocol():
    """A few load-bearing values, restated so a silent edit fails loudly."""
    assert C.SPLIT_BOUNDARY_RATIOS == {"s1_end": 0.5, "s2_end": 0.7,
                                       "s3_end": 0.8,
                                       "host_val_tail_of_s1": 0.1,
                                       "dev_eval_tail_of_s2": 0.25}
    assert C.FITTING_ROLES == (C.ROLE_POST_TRAIN,)
    assert C.EVALUATION_ROLES == (C.ROLE_DEV_EVAL,)
    assert C.SEALED_ROLES == (C.ROLE_PROTECTED_FINAL,)
    assert C.TRAINING["batch_size"] == 32
    assert C.TRAINING["lr"] == 1e-3
    assert C.TRAINING["weight_decay"] == 1e-4
    assert C.TRAINING["max_epochs"] == 50
    assert C.TRAINING["patience"] == 8
    assert tuple(C.VARIANT_DELTA) == ("FULL", "NO_SHAPE_CONTEXT", "NO_TCN",
                                      "TIED_AMPLITUDE", "NO_RARE_MASS")


def test_no_market_or_host_name_is_branched_on():
    """No behaviour may branch on a market or a Host.

    Naming one is fine -- a CLI default, a registry entry, an evidence-root
    path.  *Branching* on one is not: it is how a per-market model, a
    per-market feature set or a per-market parameter gets smuggled in.  So the
    scan looks only at comparison and test expressions, where a literal can
    actually change what the code does.

    The single registered exception is ``contracts.py``'s ``market ==
    "GANSU_DA"``, which dispatches on *provenance* (the legacy GANSU artifact
    has no ``DATASET_CONTRACT.json`` and a different role vocabulary).  That is
    a schema choice, not a scientific one.
    """
    allowed = {("contracts.py", "GANSU_DA")}
    names = set(C.MARKETS) | set(C.HOSTS)
    offenders: list = []

    def literals(node: ast.AST):
        for inner in ast.walk(node):
            if isinstance(inner, ast.Constant) and inner.value in names:
                yield inner

    for path, tree in _package_sources():
        if path.name in _REGISTRY_FILES:
            continue
        for node in ast.walk(tree):
            tests: list = []
            if isinstance(node, ast.Compare):
                tests.append(node)
            elif isinstance(node, (ast.If, ast.IfExp, ast.While)):
                tests.append(node.test)
            elif isinstance(node, (ast.comprehension,)):
                tests.append(node.iter)
            for test in tests:
                for lit in literals(test):
                    if (path.name, lit.value) in allowed:
                        continue
                    offenders.append(f"{path.name}:{lit.lineno}:{lit.value}")
    assert not offenders, offenders

    # Exactly one market literal may appear in a comparison, anywhere.
    branches = {(path.name, lit.value)
                for path, tree in _package_sources()
                for node in ast.walk(tree) if isinstance(node, ast.Compare)
                for lit in literals(node)}
    assert branches == allowed, branches


def test_readiness_table_carries_no_scientific_result():
    """The readiness table is an input audit: it may not carry an outcome."""
    from experiments.current.hch_signed_mass_method_entry import \
        host_prediction_loader as H

    rows = H.build_readiness_table()
    assert rows
    forbidden = ("mae", "rmse", "gain", "alpha", "score", "improvement")
    for row in rows:
        for key in row:
            assert not any(token in key.lower() for token in forbidden), key
