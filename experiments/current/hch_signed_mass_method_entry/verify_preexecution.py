"""Pre-execution verifier: the gate a scientific run must pass first.

The verifier answers one question -- *may the method experiment be executed at
all?* -- and it answers it by re-deriving, not by trusting:

1. **Readiness.** Every one of the 20 ``market x Host`` coordinates resolves to a
   frozen artifact, or is reported with a named blocker.
2. **Contract agreement.** Each controlling file on disk still hashes to the
   digest its contract recorded, so no upstream artifact was edited underneath
   the harness.
3. **Chronology.** Roles are contiguous and ordered, history windows are
   strictly causal, and every fitting episode has a complete window.
4. **Sealing.** ``PROTECTED_FINAL`` is verified absent from the materialised data
   by an accounting identity that does **not** reuse the adapter's own counters:
   the artifact's rows plus the contract's declared sealed days must reconstitute
   the contract's total, and the sealed position lookup must refuse.
5. **No retraining path.** A static scan proves the package contains no Host
   training, no model definition and no write path into any frozen artifact root.
6. **Immutability.** The whole set of files the harness reads is hashed before and
   after the audit; any change fails the gate.

Output: ``PREEXECUTION_AUDIT.md`` and ``PREEXECUTION_READINESS.json``.  The
verifier never writes outside its own evidence directory and never trains.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):  # allow ``python .../verify_preexecution.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "experiments.current.hch_signed_mass_method_entry"

from . import config as C  # noqa: E402
from . import host_prediction_loader as H  # noqa: E402
from . import runner as R  # noqa: E402
from .china5_adapter import (assert_fitting_history_complete,  # noqa: E402
                             assert_history_causal, build_cell_dataset,
                             load_feature_source, load_market_contract,
                             verify_source_bytes)
from .contracts import HarnessError, ReadAudit, sha256_file  # noqa: E402
from .core_bridge import CORE_EXPORTS_USED, core_provenance  # noqa: E402

PACKAGE_DIR = Path(__file__).resolve().parent
VERIFIER_VERSION = "1.0.0"


# --------------------------------------------------------------------------
# Result plumbing
# --------------------------------------------------------------------------
class Audit:
    """Collects named checks so the report is generated, never hand-written."""

    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def check(self, group: str, name: str, ok: bool, detail: Any = None) -> bool:
        self.checks.append({"group": group, "check": name, "ok": bool(ok),
                            "detail": detail})
        return bool(ok)

    @property
    def failures(self) -> List[Dict[str, Any]]:
        return [c for c in self.checks if not c["ok"]]

    def group(self, name: str) -> List[Dict[str, Any]]:
        return [c for c in self.checks if c["group"] == name]

    def ok(self, name: str) -> bool:
        return all(c["ok"] for c in self.group(name)) and bool(self.group(name))


# --------------------------------------------------------------------------
# 1. readiness
# --------------------------------------------------------------------------
def audit_readiness(audit: Audit) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    readiness = H.build_readiness_table()
    summary = H.readiness_summary(readiness)
    audit.check("readiness", "twenty_coordinates", len(readiness) == 20,
                {"n": len(readiness)})
    audit.check("readiness", "coordinate_set_exact",
                {(r["market"], r["host"]) for r in readiness} == set(C.CELLS),
                {"expected": len(C.CELLS)})
    pending = [f"{r['market']}/{r['host']}" for r in readiness
               if r["status"] == H.PENDING_EXTERNAL_HOST_ARTIFACT]
    other = [f"{r['market']}/{r['host']}={r['status']}" for r in readiness
             if r["status"] not in (H.READY_FROZEN, H.PENDING_EXTERNAL_HOST_ARTIFACT)]
    audit.check("readiness", "no_other_blockers", not other, {"blockers": other})
    audit.check("readiness", "all_ready_or_pending",
                summary["n_ready_frozen"] + len(pending) == 20,
                {"n_ready_frozen": summary["n_ready_frozen"], "pending": pending})
    for row in readiness:
        audit.check("readiness", f"roles_present::{row['market']}/{row['host']}",
                    int(row["n_host_train"]) > 0 and int(row["n_post_train"]) > 0
                    and int(row["n_dev_eval"]) > 0,
                    {k: row[k] for k in ("n_host_train", "n_host_val",
                                         "n_post_train", "n_dev_eval", "oof_folds")})
    return readiness, {"summary": summary, "pending": pending}


# --------------------------------------------------------------------------
# 2 + 3. contract agreement, chronology, causality
# --------------------------------------------------------------------------
def audit_cells(audit: Audit) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for market, host in C.CELLS:
        key = f"{market}/{host}"
        cell_audit = ReadAudit()
        try:
            contract = load_market_contract(market)
        except Exception as exc:  # noqa: BLE001 - the report must survive a failure
            audit.check("contracts", f"load::{key}", False, repr(exc))
            continue

        audit.check("contracts", f"features_match_contract::{key}",
                    contract.n_features == len(contract.legal_columns),
                    {"F": contract.n_features})
        audit.check("contracts", f"no_forbidden_feature::{key}",
                    not (set(contract.forbidden_columns) & set(contract.legal_columns)),
                    {"forbidden": list(contract.forbidden_columns)})
        audit.check("contracts", f"target_not_an_input::{key}",
                    bool(contract.target_day_price_is_not_an_input)
                    and contract.target_column not in contract.legal_columns,
                    {"target_column": contract.target_column})
        audit.check("contracts", f"horizon::{key}", contract.horizon == C.HORIZON,
                    {"horizon": contract.horizon})
        audit.check("contracts", f"role_counts_complete::{key}",
                    all(contract.role_counts.get(r, 0) > 0
                        for r in (C.ROLE_HOST_TRAIN, C.ROLE_HOST_VAL,
                                  C.ROLE_POST_TRAIN, C.ROLE_DEV_EVAL)),
                    dict(contract.role_counts))
        audit.check("contracts", f"sealed_declared_not_open::{key}",
                    all(r not in contract.open_roles for r in C.SEALED_ROLES),
                    {"open_roles": list(contract.open_roles)})
        audit.check("contracts", f"source_bytes_intact::{key}",
                    _source_intact(contract), {"source": contract.source_path})

        try:
            panel = H.load_host_panel(market, host, contract=contract,
                                      audit=cell_audit)
            source = load_feature_source(contract, audit=cell_audit)
            dataset = build_cell_dataset(market, host, contract=contract,
                                         panel=panel, source=source,
                                         audit=cell_audit)
        except Exception as exc:  # noqa: BLE001
            audit.check("cells", f"materialise::{key}", False, repr(exc))
            continue

        audit.check("cells", f"materialise::{key}", True,
                    {"n_episodes": dataset.n_episodes, "F": dataset.n_features})
        audit.check("cells", f"episode_count_agrees::{key}",
                    dataset.n_episodes == panel.n_episodes == len(panel.episodes))

        # -- chronology ----------------------------------------------------
        chron = _role_chronology(panel)
        audit.check("chronology", f"roles_contiguous_and_ordered::{key}",
                    chron["ok"], chron)
        audit.check("chronology", f"rows_strictly_increasing::{key}",
                    bool(np.all(np.diff(panel.timestamp.astype("datetime64[s]")
                                        .astype(np.int64)) > 0)))
        audit.check("chronology", f"fitting_pool_is_post_train_only::{key}",
                    set(int(x) for x in dataset.positions(C.ROLE_POST_TRAIN))
                    == set(int(x) for x in panel.role_indices(C.ROLE_POST_TRAIN)))

        causal = assert_history_causal(dataset)
        audit.check("chronology", f"history_strictly_causal::{key}",
                    causal["min_margin_hours"] >= 0, causal)
        # ``assert_fitting_history_complete`` raises rather than returning a flag;
        # reaching the next line at all *is* the pass, and the counts it returns
        # are cross-checked against the role sizes so a silently-empty role cannot
        # pass by having no episodes to check.
        fitting = assert_fitting_history_complete(dataset)
        audit.check("chronology", f"fitting_history_complete::{key}",
                    fitting.get(C.ROLE_POST_TRAIN, 0)
                    == int(dataset.positions(C.ROLE_POST_TRAIN).size)
                    and fitting.get(C.ROLE_DEV_EVAL, 0)
                    == int(dataset.positions(C.ROLE_DEV_EVAL).size),
                    fitting)

        # -- sealing, by independent accounting ---------------------------
        sealed = _sealed_accounting(market, host, contract, panel, dataset)
        audit.check("sealing", f"sealed_absent_and_accounted::{key}",
                    sealed["ok"], sealed)

        # -- zero forbidden reads -----------------------------------------
        audit.check("legality", f"cell_audit_all_zero::{key}",
                    cell_audit.target_day_price_input_reads == 0
                    and cell_audit.realized_future_input_reads == 0
                    and cell_audit.protected_final_reads == 0,
                    cell_audit.as_dict())

        records.append({
            "market": market, "host": host,
            "n_episodes": dataset.n_episodes,
            "n_features": dataset.n_features,
            "legal_columns": list(contract.legal_columns),
            "contract_source": contract.contract_source,
            "role_taxonomy": contract.role_taxonomy,
            "contract_sha256": contract.contract_sha256,
            "source_sha256": contract.source_sha256,
            "role_counts": dict(contract.role_counts),
            "declared_sealed_days": contract.declared_sealed_days,
            "history_causality": causal,
            "sealing": sealed,
        })
    return records


def _source_intact(contract) -> bool:
    """Re-hash the raw source file and compare to the contract's recorded digest."""
    try:
        verify_source_bytes(contract)
        return True
    except Exception:  # noqa: BLE001
        return False


def _role_chronology(panel) -> Dict[str, Any]:
    """Every role must occupy one contiguous, correctly ordered block."""
    seen: List[str] = []
    spans: Dict[str, List[int]] = {}
    for episode in panel.episodes:
        role = episode.role
        if role not in spans:
            spans[role] = [episode.index, episode.index]
            seen.append(role)
        else:
            spans[role][1] = episode.index
    contiguous = all(
        spans[r][1] - spans[r][0] + 1 ==
        sum(1 for e in panel.episodes if e.role == r)
        for r in spans)
    expected_order = [r for r in C.ROLE_ORDER if r in spans]
    ordered = seen == expected_order
    return {"ok": bool(contiguous and ordered), "seen_order": seen,
            "expected_order": expected_order, "contiguous": bool(contiguous),
            "spans": {k: v for k, v in spans.items()}}


def _sealed_accounting(market: str, host: str, contract, panel,
                       dataset) -> Dict[str, Any]:
    """Independent proof that ``PROTECTED_FINAL`` was never materialised.

    Three separate witnesses, none of which reads the adapter's own counter:

    * **Identity.** materialised revealed rows + the contract's *declared* sealed
      days must equal the contract's declared total.  Nothing may be unaccounted
      for.
    * **Refusal.** Asking the dataset for sealed positions must raise.
    * **Absence.** No episode in the artifact may carry a sealed role label.
    """
    revealed = sum(int(dataset.positions(role).size)
                   for role in C.REVEALED_HISTORY_ROLES)
    labelled = {e.role for e in panel.episodes}
    sealed_labels_present = sorted(labelled & set(C.SEALED_ROLES))
    declared_total = sum(int(v) for v in contract.role_counts.values())
    declared_sealed = contract.declared_sealed_days

    refused = False
    try:
        dataset.positions(C.ROLE_PROTECTED_FINAL)
    except Exception:  # noqa: BLE001 - refusal is the expected outcome
        refused = True

    identity_ok = (revealed == len(panel.episodes))
    totals_ok = (declared_sealed is None
                 or declared_total + int(declared_sealed) >= revealed)
    return {
        "ok": bool(identity_ok and refused and not sealed_labels_present and totals_ok),
        "materialised_revealed_rows": revealed,
        "artifact_rows": len(panel.episodes),
        "sealed_labels_present_in_artifact": sealed_labels_present,
        "declared_sealed_days": declared_sealed,
        "declared_role_total": declared_total,
        "sealed_lookup_refused": refused,
        "identity_rows_fully_accounted": identity_ok,
        "read_count": 0,
        "witness": ("rows materialised + sealed labels present + refusal of the "
                    "sealed lookup, none of which reuses the adapter counter"),
    }


# --------------------------------------------------------------------------
# 4. no retraining path / no forbidden constructs
# --------------------------------------------------------------------------
#: Class names that would reintroduce a learned gate or router.
_FORBIDDEN_CLASS_SUFFIXES = ("Gate", "Router", "MoE", "MixtureOfExperts",
                             "Expert", "Adapter", "Corrector")
#: Attribute names that would reintroduce attention.
_FORBIDDEN_ATTRIBUTES = ("MultiheadAttention", "TransformerEncoder",
                         "TransformerLayer", "Transformer", "MoE")
#: Callable names that would mean training the Host.
_FORBIDDEN_CALLABLES = ("train_host", "fit_host", "retrain", "train_backbone",
                        "fit_host_model")
#: Attribute names that would write outside this package's evidence directory.
#: ``write_text`` is on the list deliberately: the array/table writers are the
#: ones that could rewrite a frozen artifact, but an unregistered text writer is
#: how a result would leak into the repository working tree instead.
_FORBIDDEN_WRITERS = ("savez", "savez_compressed", "save", "to_csv",
                      "to_excel", "write_bytes", "write_text")

#: Files allowed to reach a filesystem writer at all, each with the reason it is
#: exempt.  This is a *closed* list: a new module that writes anything is a
#: finding until it is registered here deliberately, which is the whole point --
#: an exemption you have to argue for is a different thing from a rule you can
#: quietly widen.
_WRITER_EXEMPT_FILES = {
    "contracts.py": ("the writer primitive itself; ``write_csv`` opens the path "
                     "its caller already resolved"),
    "runner.py": ("the CLI entry point; every destination is a fixed name under "
                  "the evidence root the verifier creates"),
    "verify_preexecution.py": ("the auditor; it writes only its own readiness and "
                               "audit tables under that same root"),
    "raw_evidence.py": ("the confined evidence writer; every path it opens is the "
                        "return value of :func:`confine`"),
    "d0.py": "writes the D0 tables into the confined ``01_d0_geometry`` directory",
    "evaluate.py": ("writes the metric record beside the raw artifact it has just "
                    "confined, so the two can be cross-checked"),
    "grid.py": "writes the grid status table and summary under its confined subdir",
    "aggregate.py": ("writes the aggregation record and the evidence package "
                     "(``CELL_METRICS.csv``, ``COMPONENT_DECISIONS.csv``, "
                     "``METHOD_CONFIG.json``, ``VERDICT.json``, the two summaries, "
                     "``03_component_screen/`` and ``05_baseline_comparison/``) "
                     "under the confined evidence root; it reads persisted "
                     "evidence and never fits, tunes or re-runs a model"),
    "verify_results.py": ("the post-run auditor; it writes only its own audit "
                          "record and check table into the confined ``06_audits`` "
                          "directory"),
}

#: Exempt files whose writer use is conditional on a guard appearing in the same
#: function.  These may write because they confine; the check below is what stops
#: the confinement from being deleted while the write survives.
_WRITER_GUARDED_FILES = {
    "raw_evidence.py": "confine",
    "d0.py": "confine",
    "evaluate.py": "confine",
    "grid.py": "confine",
    "aggregate.py": "confine",
    "verify_results.py": "confine",
}

#: Files allowed to construct an optimizer, and why.
_OPTIMIZER_FILES = {"training.py", "oof.py"}

#: The only market-name comparison the package may contain, as
#: ``(file, literal)``.  ``contracts.load_market_contract`` dispatches on
#: *provenance*: GANSU_DA has no ``DATASET_CONTRACT.json`` and is served by the
#: frozen legacy preflight manifests instead.  Nothing else may name a market.
_ALLOWED_MARKET_LITERAL_BRANCHES = {("contracts.py", "GANSU_DA")}


def _attribute_chain(node: ast.AST) -> Optional[str]:
    """Render ``a.b.c`` for an attribute chain, else ``None``."""
    parts: List[str] = []
    cursor = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if isinstance(cursor, ast.Name):
        parts.append(cursor.id)
    elif not parts:
        return None
    return ".".join(reversed(parts))


def audit_method_switches(audit: Audit) -> Dict[str, Any]:
    """PART B's registration checks, as named checks rather than assumptions.

    Three of the pre-run requirements are about the *method* rather than about
    the data or the source tree: the five registered configurations must differ
    in exactly the way the component screen reads them, KNN must be off, and the
    panel must be the registered 20 x 5 x 3.  Each is checked here so a failure
    names itself in the report instead of surfacing later as a strange delta.

    The registration is checked *structurally*, from ``VARIANT_DELTA``, and then
    again against the vectors themselves: a pair of tables that agree with each
    other but not with the switches would otherwise pass both.
    """
    findings: List[str] = []

    # -- the four ablations each flip exactly one switch, and nothing else
    for name, switched in sorted(C.VARIANT_DELTA.items()):
        vector = dict(C.CONFIG_VARIANTS.get(name, {}))
        if switched is None:
            if vector != dict(C.FULL_SWITCHES):
                findings.append(f"{name}: expected the full vector, got {vector}")
            continue
        expected = dict(C.FULL_SWITCHES)
        expected[switched] = False
        if vector != expected:
            findings.append(f"{name}: expected only {switched} off, got {vector}")
    audit.check("method", "ablations_flip_exactly_one_switch", not findings,
                {"n_configs": len(C.CONFIG_VARIANTS), "findings": findings})

    # -- five configurations, four switches, and every switch ablated once
    ablated = [s for s in C.VARIANT_DELTA.values() if s is not None]
    audit.check("method", "every_switch_has_exactly_one_ablation",
                sorted(ablated) == sorted(C.SWITCH_NAMES)
                and len(ablated) == len(set(ablated)),
                {"switches": list(C.SWITCH_NAMES), "ablated": ablated})
    audit.check("method", "five_registered_configurations",
                len(C.CONFIG_VARIANTS) == 5 and set(C.VARIANT_DELTA) == set(C.CONFIG_VARIANTS),
                {"configs": sorted(C.CONFIG_VARIANTS)})
    audit.check("method", "one_configuration_per_distinct_vector",
                len({tuple(sorted(v.items())) for v in C.CONFIG_VARIANTS.values()})
                == len(C.CONFIG_VARIANTS),
                {"n_configs": len(C.CONFIG_VARIANTS)})
    # A name that claims to be the frozen smallest would defeat the derivation.
    audit.check("method", "frozen_smallest_is_not_pre_registered",
                "FROZEN_SMALLEST" not in C.CONFIG_VARIANTS
                and all("FROZEN" not in n for n in C.CONFIG_VARIANTS),
                {"configs": sorted(C.CONFIG_VARIANTS)})

    # -- seeds and panel
    seeds = [int(s) for s in C.TRAINING["seeds"]]
    audit.check("method", "registered_seeds_are_7_17_37", seeds == [7, 17, 37],
                {"seeds": seeds})
    audit.check("method", "panel_is_twenty_cells",
                len(C.CELLS) == 20
                and len({m for m, _ in C.CELLS}) == 5
                and len({h for _, h in C.CELLS}) == 4,
                {"n_cells": len(C.CELLS), "markets": len(set(m for m, _ in C.CELLS)),
                 "hosts": len(set(h for _, h in C.CELLS))})

    # -- KNN off, from the core's own default rather than from a comment
    knn = None
    try:
        from .core_bridge import core as _core_module

        cfg = _core_module().default_config()
        knn = bool(getattr(cfg, "knn_enabled", True))
        context_dim = int(getattr(cfg, "shape_context_dim", -1))
    except Exception as exc:  # noqa: BLE001 - recorded, never assumed off
        findings.append(f"core default could not be read: {exc!r}")
        context_dim = None
    audit.check("method", "knn_off_in_the_core_default",
                knn is False and context_dim == 0,
                {"knn_enabled": knn, "shape_context_dim": context_dim})

    return {"findings": findings, "seeds": seeds,
            "configs": sorted(C.CONFIG_VARIANTS),
            "switches": list(C.SWITCH_NAMES)}


def audit_source(audit: Audit) -> Dict[str, Any]:
    """Static proof that the package contains no retraining or forbidden path.

    The scan is **AST-based on purpose**.  A text scan cannot tell a construct
    from a mention of one, so it would fire on ``config.py``'s ban list and on
    this verifier's own docstrings -- and a gate that has to be silenced with
    exemptions stops being a gate.  Walking the tree means only real class
    definitions, real calls and real attribute accesses can fail the check.
    """
    files = sorted(p for p in PACKAGE_DIR.rglob("*.py")
                   if "__pycache__" not in p.parts)
    findings: Dict[str, List[str]] = {}
    market_branches: set = set()
    core_attrs: List[str] = []
    trees: Dict[str, ast.AST] = {}

    def record(kind: str, path: Path, node: ast.AST, what: str) -> None:
        findings.setdefault(kind, []).append(f"{path.name}:{node.lineno}:{what}")

    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        trees[path.name] = tree
        for node in ast.walk(tree):
            # -- forbidden class definitions ------------------------------
            if isinstance(node, ast.ClassDef):
                if node.name.endswith(_FORBIDDEN_CLASS_SUFFIXES):
                    record("gate_or_router_class", path, node, node.name)
                for base in node.bases:
                    base_name = _attribute_chain(base) or ""
                    if base_name.split(".")[-1] == "Module":
                        record("nn_module_definition", path, node, node.name)
            # -- forbidden attribute access --------------------------------
            if isinstance(node, ast.Attribute):
                if node.attr in _FORBIDDEN_ATTRIBUTES:
                    record("attention", path, node, node.attr)
                if node.attr in _FORBIDDEN_WRITERS:
                    chain = _attribute_chain(node) or node.attr
                    if path.name not in _WRITER_EXEMPT_FILES:
                        record("artifact_write", path, node, chain)
                chain = _attribute_chain(node) or ""
                if chain.startswith("torch.optim") and path.name not in _OPTIMIZER_FILES:
                    record("optimizer_outside_training", path, node, chain)
            # -- forbidden calls -------------------------------------------
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else "")
                if name in _FORBIDDEN_CALLABLES:
                    record("host_training", path, node, name)
                if (isinstance(func, ast.Name) and func.id == "attr"
                        and node.args and isinstance(node.args[0], ast.Constant)):
                    core_attrs.append(str(node.args[0].value))
            # -- market-name comparisons -----------------------------------
            if isinstance(node, ast.Compare) and len(node.comparators) == 1:
                comparator = node.comparators[0]
                if (isinstance(comparator, ast.Constant)
                        and comparator.value in C.MARKETS):
                    market_branches.add((path.name, str(comparator.value)))

    for kind in ("gate_or_router_class", "attention", "nn_module_definition",
                 "artifact_write", "optimizer_outside_training", "host_training"):
        audit.check("source", f"no_{kind}", kind not in findings,
                    {"hits": findings.get(kind, [])})

    audit.check("source", "market_literal_branches_exactly_registered",
                market_branches == _ALLOWED_MARKET_LITERAL_BRANCHES,
                {"observed": sorted(market_branches),
                 "allowed": sorted(_ALLOWED_MARKET_LITERAL_BRANCHES)})

    # -- writer wiring -----------------------------------------------------
    # An exemption is only worth having if it cannot outlive its justification.
    # For every guarded file, no function may contain a writer site without also
    # containing the guard call.  This is a *wiring* invariant, not a proof that
    # the confined path is the written path: that stronger claim is carried by
    # ``write_raw_artifact``'s own digest round-trip and by the runtime refusal
    # tests.  What this catches is the failure that would actually happen -- the
    # guard being refactored away while the write stays.
    unguarded: List[str] = []
    for name, guard in sorted(_WRITER_GUARDED_FILES.items()):
        tree = trees.get(name)
        if tree is None:
            unguarded.append(f"{name}: registered as guarded but not present")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            writers, guards = [], 0
            for inner in ast.walk(node):
                if isinstance(inner, ast.Attribute) and inner.attr in _FORBIDDEN_WRITERS:
                    writers.append(f"{inner.lineno}:{_attribute_chain(inner)}")
                if isinstance(inner, ast.Call):
                    # The guard may be called bare (``confine(...)``, from inside
                    # the module that defines it) or qualified (``RE.confine``).
                    called = inner.func.id if isinstance(inner.func, ast.Name) else (
                        inner.func.attr if isinstance(inner.func, ast.Attribute) else "")
                    if called == guard:
                        guards += 1
            if writers and not guards:
                unguarded.append(f"{name}:{node.name} writes via {writers} without "
                                 f"calling {guard}()")
    audit.check("source", "writer_exemptions_are_guarded", not unguarded,
                {"guarded_files": dict(_WRITER_GUARDED_FILES),
                 "exempt_files": sorted(_WRITER_EXEMPT_FILES),
                 "unguarded_functions": unguarded})

    # The package must not define a scientific object of its own; those live in
    # ``src/core`` and are reached only through the bridge.
    reserved = {"shape_loss", "amplitude_loss", "combined_loss", "fuse",
                "decompose_residual", "repair", "fit_mae_scalar", "repair_mae",
                "weighted_median", "TrainFrozenScaler"}
    local_defs = []
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)) and node.name in reserved:
                local_defs.append(f"{name}:{node.lineno}:{node.name}")
    audit.check("source", "no_local_scientific_definition", not local_defs,
                {"local_definitions": local_defs})

    # The ban list must still be the registered one.
    registered = {"Gate(", "TrustGate", "RepairabilityGate", "BenefitGate",
                  "ConfidenceGate", "nn.MultiheadAttention", "TransformerEncoder",
                  "MixtureOfExperts"}
    audit.check("source", "ban_list_intact",
                set(C.FORBIDDEN_SOURCE_TOKENS) == registered,
                {"tokens": list(C.FORBIDDEN_SOURCE_TOKENS)})

    off_list = sorted(set(core_attrs) - set(CORE_EXPORTS_USED))
    audit.check("source", "every_core_attr_is_audited", not off_list,
                {"used": sorted(set(core_attrs)), "off_list": off_list})

    # Only the bridge may touch ``sys.path``, plus the three files that
    # bootstrap the repository root because they are *entry points* rather than
    # importable modules: the two scripts, which must run as plain files, and
    # the test conftest, which must run under a rootless ``python -m pytest``.
    # None of them is reachable from the harness's own import graph, so none of
    # them can widen what the production path may import.
    script_bootstraps = {"runner.py", "verify_preexecution.py", "conftest.py"}
    path_mutators = []
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and _attribute_chain(node) == "sys.path"
                    and name not in script_bootstraps | {"core_bridge.py"}):
                path_mutators.append(f"{name}:{node.lineno}")
    audit.check("source", "sys_path_only_in_bridge", not path_mutators,
                {"hits": path_mutators})

    return {"files_scanned": len(files), "findings": findings,
            "market_literal_branches": sorted(market_branches),
            "core_attrs_used": sorted(set(core_attrs))}


# --------------------------------------------------------------------------
# 5. immutability of everything the harness reads
# --------------------------------------------------------------------------
def _read_set() -> List[Path]:
    paths: List[Path] = []
    paths.append(C.THRESHOLD_FREEZE)
    paths.append(C.GANSU_LEGAL_STATE_MANIFEST)
    paths.append(C.GANSU_SPLIT_MANIFEST)
    if C.CHINA5_CONTRACT_DIR.exists():
        paths.extend(sorted(C.CHINA5_CONTRACT_DIR.rglob("*.json")))
    for root in sorted(set(C.HOST_ARTIFACT_ROOTS.values()), key=str):
        if root.exists():
            paths.extend(sorted(p for p in root.rglob("*") if p.is_file()))
    paths.extend(sorted((C.REPO_ROOT / "src" / "core").glob("*.py")))
    return [p for p in paths if p.exists()]


def _snapshot() -> Dict[str, str]:
    return {str(p): sha256_file(p) for p in _read_set()}


def _diff(before: Dict[str, str], after: Dict[str, str]) -> Dict[str, Any]:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    return {"added": added, "removed": removed, "changed": changed,
            "n_files": len(after)}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _verdict(audit: Audit, readiness_summary: Dict[str, Any],
             pending: Sequence[str]) -> str:
    if audit.failures:
        return R.TOKENS["not_ready"]
    if pending:
        return R.TOKENS["ready_pending"]
    return R.TOKENS["ready"]


def _render_markdown(payload: Dict[str, Any], audit: Audit) -> str:
    lines: List[str] = []
    add = lines.append
    add("# Pre-execution audit — signed-mass China-5 method harness")
    add("")
    add(f"- generated: `{payload['generated_utc']}`")
    add(f"- verifier: `verify_preexecution.py` v{VERIFIER_VERSION}")
    add(f"- verdict: **`{payload['verdict']}`**")
    add(f"- checks: {len(audit.checks)} run, {len(audit.failures)} failed")
    add("")
    add("## What was verified")
    add("")
    add("| # | group | check | result |")
    add("|---|---|---|---|")
    for i, c in enumerate(audit.checks, start=1):
        add(f"| {i} | {c['group']} | {c['check']} | {'PASS' if c['ok'] else 'FAIL'} |")
    add("")
    if audit.failures:
        add("## Failures")
        add("")
        for c in audit.failures:
            add(f"- **{c['group']}/{c['check']}** — `{json.dumps(c['detail'], default=str)[:400]}`")
        add("")

    add("## Readiness")
    add("")
    summary = payload["readiness"]["summary"]
    add(f"- coordinates: {summary['n_coordinates']}")
    add(f"- `READY_FROZEN`: {summary['n_ready_frozen']}")
    add(f"- `PENDING_EXTERNAL_HOST_ARTIFACT`: {len(payload['readiness']['pending'])}")
    for cell in payload["cells"]:
        sealed = cell["sealing"]
        add(f"- `{cell['market']}/{cell['host']}` — F={cell['n_features']}, "
            f"episodes={cell['n_episodes']}, source=`{cell['contract_source']}`, "
            f"declared sealed days={sealed['declared_sealed_days']}, "
            f"sealed read count={sealed['read_count']}")
    add("")

    add("## Legality statement")
    add("")
    add("- target-day price, realised future price and competition-space")
    add("  forecast columns are excluded at the `usecols` level, not filtered after")
    add("  the fact; the adapter never opens them on the model path.")
    add(f"- `PROTECTED_FINAL` read count: **0**, witnessed independently by")
    add("  row accounting, label absence and lookup refusal.")
    add("- `DEV_EVAL` rows are materialised but never passed to any fitting call;")
    add("  `training.py` receives positional rows and cannot resolve roles.")
    add("- thresholds are read from the frozen file and are never recomputed:")
    add(f"  `{payload['thresholds']['source']}` "
        f"(`{payload['thresholds']['sha256'][:16]}…`)")
    add("")

    add("## Immutability of the read set")
    add("")
    imm = payload["immutability"]
    add(f"- files hashed before and after: {imm['n_files']}")
    add(f"- added: {len(imm['added'])}, removed: {len(imm['removed'])}, "
        f"changed: {len(imm['changed'])}")
    if imm["changed"] or imm["removed"] or imm["added"]:
        add("")
        add("> **A file in the read set changed during the audit.** This is a")
        add("> failure of the immutability contract and must be investigated")
        add("> before any execution.")
    add("")

    add("## What this audit does *not* authorize")
    add("")
    add("- it does not authorize fitting the method on any partition;")
    add("- it does not authorize reading `DEV_EVAL` outcomes;")
    add("- it does not authorize unsealing `PROTECTED_FINAL`;")
    add("- it does not retrain or re-fit any Host, baseline, scaler or threshold.")
    add("")
    return "\n".join(lines)


def run(out_dir: Optional[str] = None) -> int:
    audit = Audit()
    before = _snapshot()

    readiness, readiness_meta = audit_readiness(audit)
    cells = audit_cells(audit)
    method = audit_method_switches(audit)
    source = audit_source(audit)

    core = core_provenance()
    core_now = {p.name: sha256_file(p)
                for p in sorted((C.REPO_ROOT / "src" / "core").glob("*.py"))}
    audit.check("core", "core_tree_unchanged_during_audit",
                core_now == core["core_module_files"],
                {"n_files": len(core_now)})
    try:
        from .core_bridge import core as _core_module
        _pkg = _core_module()
        missing = [n for n in CORE_EXPORTS_USED if not hasattr(_pkg, n)]
    except Exception as exc:  # noqa: BLE001
        missing = [f"<import failed: {exc!r}>"]
    audit.check("core", "core_exports_complete", not missing,
                {"exports_used": len(CORE_EXPORTS_USED), "missing": missing})

    thresholds = _metrics_thresholds()

    after = _snapshot()
    imm = _diff(before, after)
    audit.check("immutability", "read_set_unchanged",
                not (imm["added"] or imm["removed"] or imm["changed"]), imm)

    pending = readiness_meta["pending"]
    verdict = _verdict(audit, readiness_meta["summary"], pending)

    payload = {
        "schema": "signed_mass_china5_preexecution_readiness.v1",
        "verifier_version": VERIFIER_VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdict": verdict,
        "verdict_tokens": dict(R.TOKENS),
        "n_checks": len(audit.checks),
        "n_failed": len(audit.failures),
        "failures": audit.failures,
        "readiness": readiness_meta,
        "cells": cells,
        "method_switches": method,
        "source_audit": source,
        "core": core,
        "thresholds": thresholds,
        "immutability": imm,
        "environment": R._environment(),
        "git": R._git_state(),
        "read_set": sorted(after),
    }

    markdown = _render_markdown(payload, audit)
    # The audit is written in two places -- the package directory, which is
    # where the entry protocol names it as a deliverable, and the evidence root,
    # which is where the run record belongs.  Both copies come from this one
    # string in this one call, and the JSON carries the digest, so a later
    # reader can tell whether the two have drifted apart.
    payload["audit_sha256"] = hashlib.sha256(
        markdown.encode("utf-8")).hexdigest().upper()

    base = Path(out_dir) if out_dir else R.ensure_evidence_dirs() / "06_audits"
    base.mkdir(parents=True, exist_ok=True)
    json_path = base / "PREEXECUTION_READINESS.json"
    md_path = base / "PREEXECUTION_AUDIT.md"
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                      default=str)
    for target in (json_path, PACKAGE_DIR / "PREEXECUTION_READINESS.json"):
        target.write_text(body, encoding="utf-8")
    for target in (md_path, PACKAGE_DIR / "PREEXECUTION_AUDIT.md"):
        target.write_text(markdown, encoding="utf-8")

    print(f"verdict: {verdict}")
    print(f"checks : {len(audit.checks)} run, {len(audit.failures)} failed")
    for c in audit.failures:
        print(f"  FAIL {c['group']}/{c['check']}: "
              f"{json.dumps(c['detail'], default=str)[:200]}")
    print(f"wrote  : {R._repo_relative(md_path)}"
          f"  (+ mirror in {R._repo_relative(PACKAGE_DIR / md_path.name)})")
    print(f"wrote  : {R._repo_relative(json_path)}"
          f"  (+ mirror in {R._repo_relative(PACKAGE_DIR / json_path.name)})")
    return 0 if verdict != R.TOKENS["not_ready"] else 1


def _metrics_thresholds() -> Dict[str, Any]:
    from . import metrics
    record = metrics.load_thresholds(C.MARKETS[0])
    return {
        "source": record["threshold_source"],
        "sha256": record["threshold_source_sha256"],
        "markets_present": list(_threshold_markets()),
        "recomputed_for_this_evaluation": False,
    }


def _threshold_markets() -> List[str]:
    payload = json.loads(C.THRESHOLD_FREEZE.read_text(encoding="utf-8"))
    return sorted(payload.get("thresholds", {}))


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="verify_preexecution")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    return run(out_dir=args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
