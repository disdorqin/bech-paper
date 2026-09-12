"""P4 — freeze the proposed method before TEST exists.

Two deliverables:

1. **The active-core freeze.**  Run the active-core regression tests, then persist
   per-file SHA256 for `src/core/**`, the core tree digest, the `HISTORY_DAYS = 7`
   assertion, the candidate architecture identity, the safety formula identity, and
   a proof that the deleted development components remain absent by name and by
   construction.

2. **The TRAIN/VAL-only method-training manifest** for a future M0/M1/M2.  Every
   semantic a future run must not be free to choose is written down here, before any
   TEST result exists: how candidate/residual training pairs are produced inside
   TRAIN, the seeds, the optimizer, the epoch/early-stop rule, the loss weights, the
   train-only normalisation, the amplitude scale fitting, the `alpha_0` fit on VAL,
   the exact seven-record VAL safety warm start, the M0/M1/M2 definitions and the
   TEST prequential bank-update rule.

**The harness blocker, stated up front.**  The existing method harness
(`experiments/current/hch_signed_mass_method_entry/`) is built on the old five-role
taxonomy and cannot express the common benchmark's TRAIN/VAL/TEST split without
changing scientific semantics:

* `config.py` fixes `ROLE_ORDER = (HOST_TRAIN, HOST_VAL, POST_TRAIN, DEV_EVAL,
  PROTECTED_FINAL)`, `FITTING_ROLES = (ROLE_POST_TRAIN,)` and
  `EVALUATION_ROLES = (ROLE_DEV_EVAL,)`;
* `china5_adapter.py` raises `LegalityError` for any role outside that taxonomy and
  hard-codes `(ROLE_POST_TRAIN, ROLE_DEV_EVAL)` when counting complete episodes;
* `calibration.py` states that the pooled scalar is fitted "on `POST_TRAIN` only";
* `aggregate.py` writes `POST_TRAIN`/`DEV_EVAL` into the narrative of the rows it
  emits.

Repointing those constants at TRAIN/VAL would not be a re-run of the method — it
would be a redefinition of the fit/evaluation partition under the old role names,
which is exactly what the prompt forbids ("do not improvise using old POST_TRAIN
roles").  So PRETEST **runs no method training**: the prompt makes TRAIN/VAL fitting
optional ("may be prepared/run in PRETEST only if its complete training semantics
are already frozen").  The blocker is documented with exact evidence and the manifest
below freezes the semantics a conforming harness must implement.  No M0/M1/M2
numeric object is produced here, on any partition.

Run:  python .../cb_p4_core.py
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

IMPL = Path(__file__).resolve().parent
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))

import cb_contracts as CB                                                    # noqa: E402

ROOT = CB.ROOT
EV = CB.EV
CORE = ROOT / "src/core"
HARNESS = ROOT / "experiments/current/hch_signed_mass_method_entry"
OUT = EV / "04_core"
CORE_TESTS = ("test_core_candidate.py", "test_core_safety.py", "test_core_purity.py",
              "test_canonical_core_layout.py")

# ------------------------------------------------------- deleted-component proof
# Tier A is decisive: an actual definition/attribute/flag, not a mention.  Tier B
# (plain mentions) is reported with file:line so a reviewer can confirm each one is
# prose *about* the absence rather than the component itself.
FORBIDDEN_DEFINITIONS = {
    "tcn_or_temporal_convolution": r"^\s*(class|def)\s+\w*(TCN|TemporalConv|Conv1d)",
    "shape_semantic_context_routing": r"^\s*(class|def)\s+\w*(ShapeContext|SemanticView|SemanticContext|ContextRouter)",
    "untied_amplitude_heads": r"^\s*(class|def)\s+\w*(PositiveHead|NegativeHead|UntiedAmplitude|AmplitudeHeads)",
    "rare_mass_sampling": r"^\s*(class|def)\s+\w*(Sampler|RareMass|Stratif)",
    "knn_or_similarity_retrieval": r"^\s*(class|def)\s+\w*(KNN|Knn|Similarity|Retrieval|MemoryBank|Prototype)",
    "generic_lstm_or_bidirectional_switch": r"^\s*(class|def)\s+\w*(BiLSTM|BiGRU|Bidirectional)",
    "daily_bias_or_level_branch": r"^\s*(class|def)\s+\w*(DailyBias|BiasBranch|LevelTerm|OffsetHead)",
    "learned_gate_or_selector": r"^\s*(class|def)\s+\w*(Gate|Selector|Trust|Reliability|Uncertainty)",
}
FORBIDDEN_CONFIG_FLAGS = {
    "tcn_switch": r"use_tcn\s*[:=]",
    "shape_context_switch": r"shape_semantic_context\s*[:=]",
    "untied_amplitude_switch": r"untied_amplitude\w*\s*[:=]",
    "rare_mass_switch": r"rare_mass\w*\s*[:=]",
    "knn_switch": r"knn_enabled\s*[:=]",
    "gate_switch": r"(?:gate|trust|selector)_enabled\s*[:=]",
}
FORBIDDEN_MENTIONS = {
    "TCN": r"\bTCN\b",
    "knn": r"\bknn\b|\bKNN\b",
    "memory_bank": r"\bmemory[_ ]bank\b",
    "similarity_retrieval": r"\bsimilarity\b|\bretrieval\b",
    "generic_rnn_switch": r"nn\.LSTM\b|\bbidirectional\b|\btemporal_rnn\b",
    "untied": r"\buntied\b",
    "rare_mass": r"\brare[_ ]mass\b",
    "gate": r"\bGate\b|\bgate\b|\bTrustGate\b|\bRepairabilityGate\b|\bBenefitGate\b|\bConfidenceGate\b",
    "daily_bias": r"\bdaily[_ ]bias\b|\bbias[_ ]branch\b",
    "PROTECTED_FINAL": r"\bPROTECTED_FINAL\b",
    "baseline_names": r"\bMatchedDirectResidual\b|\bdelta-Adapter\b|\bOMPB\b|\bUEC-STD\b|\bCOSA\b|\bPIR\b",
    "market_or_host_names": r"\bGANSU_DA\b|\bSHANDONG_DA\b|\bSHAANXI_DA\b|\bNINGXIA_DA\b|\bQINGHAI_DA\b"
                             r"|\bPatchTST\b|\bTimeMixer\b|\biTransformer\b",
}
FORBIDDEN_IMPORTS = r"^\s*(?:from|import)\s+(experiments|paper|docs|archive|core\.v25|core\.iah|core\.learned_signature|core\.weighted_mean|core\.universal_trainer|core\.iah_crps)"


def core_files() -> list[Path]:
    return sorted((p for p in CORE.rglob("*.py")),
                  key=lambda p: p.relative_to(CORE).as_posix())


def tree_digest(root: Path, suffixes=(".py",)) -> tuple[str, list[dict]]:
    files = sorted((p for p in Path(root).rglob("*")
                    if p.is_file() and p.suffix in suffixes),
                   key=lambda p: p.relative_to(root).as_posix())
    h = hashlib.sha256()
    per = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        digest = hashlib.sha256(p.read_bytes()).hexdigest().upper()
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(digest.encode("ascii"))
        h.update(b"\x0a")
        per.append({"path": rel, "sha256": digest, "bytes": p.stat().st_size})
    return h.hexdigest().upper(), per


# ------------------------------------------------------------- 1. regression tests
def run_core_tests() -> dict:
    argv = [sys.executable, "-m", "pytest",
            *[str(ROOT / "experiments/foundation/tests" / t) for t in CORE_TESTS], "-q"]
    env = {**__import__("os").environ, "PYTHONPATH": "src"}
    proc = subprocess.run(argv, cwd=str(ROOT), env=env, capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    passed = failed = 0
    m = re.search(r"(\d+) passed", proc.stdout or "")
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", proc.stdout or "")
    if m:
        failed = int(m.group(1))
    return {"argv": argv[1:], "returncode": proc.returncode, "n_passed": passed,
            "n_failed": failed, "summary_line": tail[0],
            "tests": list(CORE_TESTS), "scope": "active-core regression tests",
            "passed": proc.returncode == 0 and failed == 0 and passed > 0}


# --------------------------------------------------------- 3. HISTORY_DAYS = 7
def history_assertion() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    import inspect
    from core import history as H
    from core.contracts import SafetyEvidence, WARM_START_PROVENANCE

    bank_sig = inspect.signature(H.SafetyEvidenceBank.__init__)
    warm_sig = inspect.signature(H.select_warm_start)
    src = (CORE / "history.py").read_text(encoding="utf-8")

    checks = {
        "HISTORY_DAYS_equals_7": H.HISTORY_DAYS == 7,
        "bank_constructor_takes_no_capacity": len(bank_sig.parameters) == 1,
        "warm_start_takes_no_length": len(warm_sig.parameters) == 1,
        "capacity_property_is_the_constant": (
            inspect.getsource(H.SafetyEvidenceBank.capacity.fget).find("HISTORY_DAYS") >= 0),
        "no_capacity_literal": not re.search(r"capacity\s*[:=]\s*\d", src),
        "warm_start_provenance_excludes_in_sample": (
            "in_sample" not in WARM_START_PROVENANCE),
        "warm_start_provenance_is_oof_or_prequential": (
            set(WARM_START_PROVENANCE) == {"oof_prequential", "prequential"}),
        "short_bank_raises_rather_than_shrinking": (
            "IncompleteHistoryError" in inspect.getsource(H.SafetyEvidenceBank.completed)),
        "insufficient_warm_start_raises": (
            "InsufficientWarmStartError" in inspect.getsource(H.select_warm_start)),
    }
    # behavioural: a 6-record bank must refuse, a 7-record bank must pass
    def _rec(i: int) -> SafetyEvidence:
        return SafetyEvidence(
            delivery_id=f"d{i}", ordinal=i,
            shape_positive=torch.full((24,), 1 / 24), shape_negative=torch.full((24,), 1 / 24),
            candidate_correction=torch.zeros(24), candidate_created_at=i - 1,
            provenance="prequential", host_residual=torch.zeros(24), revealed_at=i)
    try:
        H.select_warm_start([_rec(i) for i in range(1, 7)])
        checks["six_records_refused"] = False
    except H.InsufficientWarmStartError:
        checks["six_records_refused"] = True
    try:
        got = H.select_warm_start([_rec(i) for i in range(1, 9)])
        checks["seven_records_take_the_last_seven"] = (
            len(got) == 7 and [r.ordinal for r in got] == [2, 3, 4, 5, 6, 7, 8])
    except Exception:
        checks["seven_records_take_the_last_seven"] = False
    done = [r for r in (_rec(i) for i in range(1, 9))]
    bank = H.SafetyEvidenceBank.from_warm_start(done)
    checks["restored_bank_capacity_is_seven"] = bank.capacity == 7 and bank.is_ready()
    return {"HISTORY_DAYS": H.HISTORY_DAYS, "checks": checks, "passed": all(checks.values())}


# --------------------------------------------------- 4/5. architecture + formulas
def architecture_identity() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from core import (BalancedAmplitudeBranch, MinimalSignedRedistributionModel,
                      ShapeBranch, UnifiedFeatureStem, WindowGRUEncoder, default_config)
    cfg = default_config()
    model = MinimalSignedRedistributionModel(cfg)
    n_param = sum(p.numel() for p in model.parameters())
    return {
        "candidate_object": "c_h = A * (S+_h - S-_h), A >= 0, S+-, S- in Delta^{H-1}",
        "model_class": type(model).__name__,
        "fusion": "deterministic: fuse(amplitude, shape) = multiplication and subtraction only",
        "config_class": type(cfg).__name__,
        "config": _jsonable(dataclasses.asdict(cfg)),
        "config_field_names": sorted(f.name for f in dataclasses.fields(cfg)),
        "components": {
            "shared_stem": f"{UnifiedFeatureStem.__name__} "
                           f"(Linear(D,{cfg.stem_hidden}) -> GELU -> LayerNorm -> "
                           f"Linear({cfg.stem_hidden},{cfg.stem_out}))",
            "encoder": f"{WindowGRUEncoder.__name__} (GRU, one weight set per branch)",
            "shape_branch": f"{ShapeBranch.__name__} (masked softmax over horizon)",
            "amplitude_branch": f"{BalancedAmplitudeBranch.__name__} (one nonnegative scalar)",
        },
        "n_parameters_at_default_config": int(n_param),
        "module_files": sorted(p.name for p in core_files()),
        "learned_modules": sorted({type(m).__name__ for m in model.modules()
                                   if len(list(m.children())) == 0 and
                                   any(True for _ in m.parameters())}),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def safety_formula_identity() -> dict:
    """Persist the formulas *and* assert them on analytic cases."""
    sys.path.insert(0, str(ROOT / "src"))
    import inspect
    from core import safety as S

    formulas = {
        "R_lambda": "R(lambda) = sum_i q_i (|z_i - lambda| - |z_i|),  z_i = r_i/c_i, q_i = |c_i|",
        "alignment_support": "A_align = 2 Q_+ / Q - 1,  R'(0+) = Q - 2 Q_+ = -Q * A_align",
        "kappa": "kappa_{d,s} = 1 - (W1(S+_d,S+_s) + W1(S-_d,S-_s)) / (2 (H-1))",
        "wasserstein1": "W1(P,Q) = sum_{h=1}^{H-1} |F_P(h) - F_Q(h)|",
        "deployment_scale": "lambda = min(alpha_0, lambda_U, lambda_S)",
        "safe_radius": "delta = right endpoint of the no-harm interval of R, from sorted breakpoints",
    }
    # analytic case: r = [3, -1, 0, 0], c = [1, 1, 0, 0]
    r = np.array([3.0, -1.0, 0.0, 0.0])
    c = np.array([1.0, 1.0, 0.0, 0.0])
    z = np.array([3.0, -1.0])              # c_i != 0
    q = np.abs(c[:2])
    Q, Qp = q.sum(), q[z > 0].sum()
    A_align = 2 * Qp / Q - 1               # = 0
    checks = {
        "alignment_support_matches_formula": bool(
            np.isclose(S.mae_alignment_support(r, c), A_align)),
        "alignment_derivative_identity": bool(
            np.isclose(Q - 2 * Qp, -Q * A_align)),
        "excess_risk_at_zero_is_zero": bool(
            np.isclose(S.mae_excess_risk(r, c, 0.0), 0.0)),
        "excess_risk_matches_hand_computation": bool(
            np.isclose(S.mae_excess_risk(r, c, 0.5),
                       np.sum(q * (np.abs(z - 0.5) - np.abs(z))))),
        "uniform_radius_is_right_endpoint_of_no_harm": True,   # asserted below
        "deployment_scale_is_the_minimum": bool(
            S.deployment_scale(0.8, 0.5, 0.9) == 0.5 and
            S.deployment_scale(0.8, 0.9, None) == 0.8 and
            S.deployment_scale(0.3, None, None) == 0.3),
        "deployment_scale_never_exceeds_alpha0": bool(
            S.deployment_scale(0.4, 10.0, 10.0) == 0.4),
        "deployment_scale_never_negative": bool(S.deployment_scale(0.4, -3.0, None) == 0.0),
        "non_finite_radius_is_not_a_constraint": bool(
            S.deployment_scale(0.6, float("inf"), None) == 0.6),
        "nan_radius_is_refused": False,
        "kappa_is_one_for_identical_shapes": bool(
            np.isclose(S.shape_relevance(np.full((1, 4), 0.25), np.full((1, 4), 0.25),
                                         np.full((1, 4), 0.25), np.full((1, 4), 0.25)), 1.0)),
        "wasserstein1_matches_hand_computation": bool(
            np.isclose(S.wasserstein1(np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.5, 0.5])),
                       abs(0.5 - 0.0) + abs(1.0 - 0.5))),
        "wasserstein1_is_zero_for_identical_shapes": bool(
            np.isclose(S.wasserstein1(np.array([0.2, 0.3, 0.5]), np.array([0.2, 0.3, 0.5])),
                       0.0)),
    }
    try:
        S.deployment_scale(0.6, float("nan"), None)
    except ValueError:
        checks["nan_radius_is_refused"] = True
    # the radius must be the right endpoint of the no-harm interval, checked by scan
    lam = float(S.exact_mae_safe_radius(r, c))
    grid = np.linspace(0.0, max(lam, 1e-9) * 4.0, 20001)
    excess = np.array([S.mae_excess_risk(r, c, float(g)) for g in grid])
    checks["uniform_radius_is_right_endpoint_of_no_harm"] = bool(
        lam >= 0.0 and np.nanmax(excess[grid <= lam * (1 - 1e-12)]) <= 1e-9
        and (grid[-1] <= lam or np.nanmax(excess[grid > lam * (1 + 1e-12)]) > 0.0))
    return {"formulas": formulas, "checks": checks, "passed": all(checks.values()),
            "module_sha256": hashlib.sha256((CORE / "safety.py").read_bytes())
                                      .hexdigest().upper(),
            "analytic_case": {"r": r.tolist(), "c": c.tolist(),
                              "lambda_U": lam, "A_align": float(A_align)}}


def deleted_components_absence() -> dict:
    files = core_files()
    text = {p.relative_to(CORE).as_posix(): p.read_text(encoding="utf-8") for p in files}
    hits_a, hits_flag, hits_b, hits_imp = [], [], [], []
    for rel, body in text.items():
        for line_no, line in enumerate(body.splitlines(), 1):
            for name, pat in FORBIDDEN_DEFINITIONS.items():
                if re.search(pat, line):
                    hits_a.append({"check": name, "path": rel, "line": line_no,
                                   "text": line.strip()[:160]})
            for name, pat in FORBIDDEN_CONFIG_FLAGS.items():
                if re.search(pat, line):
                    hits_flag.append({"check": name, "path": rel, "line": line_no,
                                      "text": line.strip()[:160]})
            if re.search(FORBIDDEN_IMPORTS, line):
                hits_imp.append({"path": rel, "line": line_no, "text": line.strip()[:160]})
            for name, pat in FORBIDDEN_MENTIONS.items():
                if re.search(pat, line):
                    hits_b.append({"check": name, "path": rel, "line": line_no,
                                   "text": line.strip()[:200]})
    # Decisive tier: only *structural* facts -- a definition, a flag assignment, an
    # import, a quoted identifier, an actual call.  A sentence that says a component
    # is absent is prose, and prose must never be able to decide this check: it can
    # neither create a component nor hide one.
    literals_market_host = [
        {"path": rel, "line": i, "text": line.strip()[:160], "match": m}
        for rel, body in text.items() for i, line in enumerate(body.splitlines(), 1)
        for m in re.findall(r"""["'](?:GANSU_DA|SHANDONG_DA|SHAANXI_DA|NINGXIA_DA|QINGHAI_DA|"""
                            r"""PatchTST|TimeMixer|iTransformer|LSTM|HOST_TRAIN|HOST_VAL|"""
                            r"""POST_TRAIN|DEV_EVAL|PROTECTED_FINAL)["']""", line)]
    literal_baselines = [
        {"path": rel, "line": i, "text": line.strip()[:160], "match": m}
        for rel, body in text.items() for i, line in enumerate(body.splitlines(), 1)
        for m in re.findall(r"""["'](?:MatchedDirectResidual|delta-Adapter|OMPB|UEC-STD|"""
                            r"""COSA|PIR)["']""", line)]
    # Real imports, taken from the parse tree rather than from line matching: a
    # docstring line beginning with the word "from" is prose, not an import.
    import_statements = []
    for rel, body in text.items():
        for node in ast.walk(ast.parse(body)):
            if isinstance(node, ast.Import):
                import_statements += [{"path": rel, "line": node.lineno,
                                       "module": a.name} for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                import_statements.append({"path": rel, "line": node.lineno,
                                          "module": "." * node.level + (node.module or "")})
    checks = {
        "no_forbidden_definitions": not hits_a,
        "no_forbidden_config_flags": not hits_flag,
        "no_forbidden_imports": not hits_imp,
        "no_forbidden_module_imports": not [
            s for s in import_statements
            if s["module"].split(".")[0] in {"os", "pathlib", "shutil", "glob", "json", "csv",
                                             "sqlite3", "sys", "subprocess", "requests",
                                             "urllib", "socket", "experiments", "paper", "docs",
                                             "archive"}],
        "the_only_recurrent_encoder_is_gru": (
            any("nn.GRU" in body for body in text.values())
            and not any("nn.LSTM" in body for body in text.values())),
        "no_dataset_split_or_host_role_literals": not literals_market_host,
        "no_baseline_name_literals": not literal_baselines,
        "dependencies_are_stdlib_dataclasses_and_numeric_only": all(
            s["module"].split(".")[0] in {"__future__", "dataclasses", "math", "typing",
                                          "collections", "itertools", "functools", "torch",
                                          "numpy", "core", ""}
            or s["module"].startswith(".") for s in import_statements),
        "no_file_open_call": not [rel for rel, body in text.items()
                                  if re.search(r"(?<![\w.])open\s*\(", body)],
    }
    return {"checks": checks, "passed": all(checks.values()),
            "deciding_tier": "structural only (definitions, flag assignments, imports, "
                             "quoted identifiers, actual RNN class)",
            "tier_a_definition_hits": hits_a, "tier_a_flag_hits": hits_flag,
            "tier_a_import_hits": hits_imp,
            "structural_literal_hits": literals_market_host + literal_baselines,
            "imports": import_statements,
            "tier_b_mention_hits": hits_b,
            "tier_b_note": ("plain-text mentions are reported but never decisive. This package "
                            "documents its own absences in prose, so the deleted component "
                            "names legitimately appear in docstrings that state they are absent "
                            "--- e.g. 'there is no bidirectional or LSTM branch.' A prose "
                            "sentence can neither create a component nor hide one, so the "
                            "verdict rests on the structural tier alone.")}


# ------------------------------------------------------- harness role-taxonomy gate
def harness_blocker() -> dict:
    cfg = (HARNESS / "config.py").read_text(encoding="utf-8")
    checks = {
        "harness_exists": HARNESS.is_dir(),
        "declares_post_train_fitting_role": bool(re.search(r"FITTING_ROLES\s*=\s*\(ROLE_POST_TRAIN", cfg)),
        "declares_dev_eval_evaluation_role": bool(re.search(r"EVALUATION_ROLES\s*=\s*\(ROLE_DEV_EVAL", cfg)),
        "has_no_train_role": "ROLE_TRAIN" not in cfg,
        "has_no_val_role": "ROLE_VAL" not in cfg,
        "has_no_test_role": "ROLE_TEST" not in cfg,
    }
    adapter = (HARNESS / "china5_adapter.py").read_text(encoding="utf-8")
    checks["adapter_raises_on_unmaterialised_role"] = "LegalityError" in adapter
    checks["adapter_hardcodes_post_train_dev_eval"] = bool(
        re.search(r"ROLE_POST_TRAIN,\s*C\.ROLE_DEV_EVAL", adapter))
    calib = (HARNESS / "calibration.py").read_text(encoding="utf-8")
    checks["calibration_states_post_train_only"] = "POST_TRAIN" in calib
    incompatible = all(v for k, v in checks.items() if k != "harness_exists")
    return {
        "schema": "common_benchmark_harness_role_taxonomy_blocker.v1",
        "protocol_id": CB.PROTOCOL_ID,
        "harness": str(HARNESS.relative_to(ROOT)).replace("\\", "/"),
        "blocker_id": "SIGNED_MASS_HARNESS_ROLE_TAXONOMY_INCOMPATIBLE_WITH_COMMON_BENCHMARK",
        "blocker_class": "Q-HARNESS",
        "verdict": ("HARNESS_CANNOT_EXPRESS_TRAIN_VAL_TEST_WITHOUT_CHANGING_SEMANTICS"
                    if incompatible else "NO_BLOCKER_DETECTED"),
        "checks": checks,
        "evidence": {
            "config.py": "ROLE_ORDER / FITTING_ROLES=(ROLE_POST_TRAIN,) / "
                         "EVALUATION_ROLES=(ROLE_DEV_EVAL,) / SEALED_ROLES=(ROLE_PROTECTED_FINAL,)",
            "china5_adapter.py": "positions(role) raises LegalityError for an unmaterialised "
                                 "role; complete-episode counting loops over "
                                 "(ROLE_POST_TRAIN, ROLE_DEV_EVAL)",
            "calibration.py": "the pooled scalar is documented as fitted on POST_TRAIN only",
            "aggregate.py": "emitted rows name POST_TRAIN / DEV_EVAL in their narrative",
        },
        "why_not_repointed": (
            "repointing POST_TRAIN->TRAIN and DEV_EVAL->VAL would keep the old role names and "
            "change the fit/evaluation partition they denote; that is a redefinition of the "
            "method's data contract, not a re-run of it, and the prompt forbids improvising "
            "with old POST_TRAIN roles"),
        "method_training_executed_in_pretest": False,
        "method_numeric_objects_produced": 0,
        "consequence": (
            "TRAIN/VAL fitting is optional in PRETEST; it is not performed. The "
            "method-training manifest below freezes the semantics a conforming harness must "
            "implement before any TEST result can exist."),
    }


# ------------------------------------------------------------ method-training manifest
def method_training_manifest() -> dict:
    """The TRAIN/VAL-only semantics a future M0/M1/M2 run must not be free to choose."""
    splits = {}
    for m in CB.MARKETS:
        man = json.loads((EV / "01_splits" / m / "split_manifest.json").read_text(encoding="utf-8"))
        splits[m] = {"counts": man["counts"], "split_manifest_sha256": man["split_manifest_sha256"],
                     "test_target_blind_id_sha256": man.get("test_target_blind_id_sha256")}
    return {
        "schema": "china5_common_benchmark_method_training_manifest.v1",
        "protocol_id": CB.PROTOCOL_ID,
        "stage_id": "hch_china5_common_benchmark_701020",
        "status": "FROZEN_PRETEST_NO_TEST_RESULT",
        "authority": [
            "experiments/current/hch_china5_common_benchmark_701020/AI_PRETEST_EXECUTION_PROMPT.md §8",
            "src/core/DESIGN_CONTRACT.md",
            "docs/current/HCH_RESIDUAL_ALIGNMENT_SAFETY_DESIGN_20260912.md",
        ],

        # ---- 1. where every quantity comes from
        "partitions": {
            "fit": "TRAIN", "selection": "VAL", "scoring": "TEST",
            "alpha_0_fitted_on": "VAL",
            "safety_warm_start_from": "VAL",
            "normalisation_fitted_on": "TRAIN",
            "amplitude_scale_fitted_on": "TRAIN",
            "test_labels_visible_to_fitting_or_selection": False,
            "note": ("alpha_0 and the seven-record safety warm start are the method's own "
                     "calibration/selection objects, so under the common benchmark they are "
                     "fitted on VAL — the analogue of the old protocol's POST_TRAIN role for "
                     "the method. TEST is never read. This is the prompt's explicit "
                     "instruction, and it supersedes the older POST_TRAIN-only wording, which "
                     "named a role that does not exist under this protocol."),
        },

        # ---- 2. candidate/residual training pairs inside TRAIN
        "training_pairs": {
            "unit": "one forecast origin (one target day) per training example",
            "construction": (
                "for each TRAIN target day d: r_d = y_d - y_host_d over the 24-hour horizon; "
                "the example is (ForecastOriginBatch(d), SignedMassTargets derived from r_d, "
                "d). No other day contributes a target to this example."),
            "targets_from": "the frozen protocol-scoped Host cache rows of role TRAIN",
            "host_predictions_source": "experiments/evidence/"
                                       "hch_china5_common_benchmark_701020_20260912/02_hosts/"
                                       "<MARKET>/<HOST>/host_predictions.npz",
            "host_predictions_reused_from_old_stage": False,
            "feature_builder": "core.preprocessing.DeterministicFeatureBuilder (the only one)",
            "revealed_history_rule": (
                "the historical branch of a day in role R may draw only on target days that "
                "are strictly earlier in the market's chronological eligible sequence and "
                "already revealed relative to that day: TRAIN days for a TRAIN day; TRAIN and "
                "earlier VAL days for a VAL day; TRAIN, VAL and earlier TEST days for a TEST "
                "day, and only in chronological order at TEST time. No future day, no sealed "
                "role, and no cross-market history."),
            "history_windows": 7,
            "history_windows_note": (
                "W = 7 previous eligible episodes in chronological order, matching "
                "core.history.HISTORY_DAYS. Not calendar-contiguous: the frozen splits contain "
                "genuine calendar gaps and a gap is not filled."),
            "current_day_inputs_legality": (
                "C6: the frozen Host forecast, audited forecast-known numeric covariates, "
                "deterministic calendar channels and availability masks only. The target-day "
                "actual and its residual are never inputs."),
            "missing_role_policy": "C7: an explicit mask; never substituted by Host price or "
                                   "another physical variable",
            "cells": "20 = 5 markets x 4 Hosts, independently; no cross-cell pooling",
        },

        # ---- 3. optimisation
        "optimisation": {
            "seeds": [7, 17, 37],
            "seed_rule": "each cell is trained at each registered seed; reported numbers are "
                         "the seed aggregate and are never selected by seed",
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "max_epochs": 50,
            "patience": 8,
            "early_stop_rule": (
                "the chronologically last fraction of the training prefix is held out as an "
                "inner validation suffix (OOF_INNER_VAL_FRACTION = 0.2, at least "
                "OOF_MIN_INNER_VAL_DAYS = 2 days); training stops after `patience` epochs "
                "without improvement on that suffix; the suffix is always a suffix of the "
                "training prefix, so no fold statistic ever sees its own holdout block"),
            "early_stop_partition": "TRAIN-only (a suffix of the TRAIN prefix); VAL is not used "
                                    "for early stopping of the candidate generator",
            "loss_weights": {"repair": 1.0, "shape": 1.0, "amplitude": 1.0,
                             "shape_metric": "wasserstein"},
            "loss_weights_source": "hch_signed_mass_method_entry/config.py LOSS_WEIGHTS — the "
                                   "registered values, identical for every market and Host",
            "loss_weights_fidelity_caveat": (
                "DESIGN_CONTRACT.md §6 records that these weights are registered but **not "
                "scientifically frozen** (never validated as optimal). They are frozen here as "
                "the values a run must use, not as a claim that they are well chosen; changing "
                "them per market or per Host is forbidden."),
            "loss_form": {
                "repair": "L_R = H^-1 sum_h |r_h - c_h|",
                "shape": "W1(S, S_hat) = sum_{h=1}^{H-1} |F_S(h) - F_Shat(h)|",
                "amplitude": "L_A^bal = |A+ - A| + |A- - A| (identical to the archived tied-head "
                             "objective)",
                "combined": "L = L_R + lambda_S L_S + lambda_A L_A with lambda_S = lambda_A = 1.0",
            },
            "market_or_host_specific_weights": False,
        },

        # ---- 4. normalisation and scales (TRAIN only)
        "normalisation": {
            "scaler": "core.preprocessing.TrainFrozenScaler",
            "fit_partition": "TRAIN",
            "applied_at": "frozen before any VAL or TEST row is scored; never refitted",
            "amplitude_scale": {
                "symbol": "s_A",
                "rule": "s_A = median{A_d^+, A_d^- : A_d^pm > 0} over the TRAIN fitting prefix, "
                        "shared by both sign heads",
                "floor": 1.0,
                "floor_applies_only_when": "that set is empty or degenerate (no positive "
                                           "residual mass anywhere in the prefix)",
                "fit_partition": "TRAIN",
                "per_market_or_host": True,
                "cross_cell_pooling": False,
            },
            "within_day_normalisation": "core.preprocessing.within_day_robust_normalize "
                                        "(median/MAD) — deterministic, no fitted parameter",
        },

        # ---- 5. alpha_0
        "alpha_0": {
            "definition": "alpha* = argmin_{alpha >= 0} sum_i |r_i - alpha c_i| "
                          "= max(0, WeightedMedian(r_i/c_i; |c_i|))",
            "zero_handling": "coordinates with c_i = 0 are removed BEFORE forming ratios, so no "
                             "inf/nan can arise; an all-zero candidate returns 0 exactly",
            "granularity": "one pooled scalar per fitted market x Host model — never "
                           "per-instance, never per-day",
            "fit_partition": "VAL",
            "fit_rows": "the VAL days, using the frozen candidate generator's VAL predictions",
            "val_is_also_the_selection_partition": True,
            "val_scope_caveat": (
                "VAL is both the method's calibration source and its selection block. This is "
                "recorded as a scope caveat: alpha_0 is not an out-of-sample estimate of "
                "anything, it is a frozen deployment scalar."),
            "refit_at_TEST": False,
            "numeric_alpha_0_values": "NOT FROZEN IN PRETEST (no method training was executed)",
        },

        # ---- 6. the seven-record VAL safety warm start
        "safety_warm_start": {
            "required_records": 7,
            "source_partition": "VAL",
            "construction": (
                "1. take the VAL days in chronological order; "
                "2. for each VAL day d, generate the candidate c_d with the TRAIN-frozen "
                "parameters — the parameters never saw any VAL day, so each such candidate is "
                "out-of-fold with respect to its own outcome; "
                "3. build SafetyEvidence(delivery_id='VAL::<market>::<host>::<d>', "
                "ordinal=index(d), candidate_created_at=index(d)-1, provenance="
                "'oof_prequential', shape_positive=S+_d, shape_negative=S-_d, "
                "candidate_correction=c_d, host_residual=r_d, revealed_at=index(d), "
                "valid_mask=the day's availability mask); "
                "4. select_warm_start(records) takes the LAST seven in chronological order and "
                "raises InsufficientWarmStartError otherwise; "
                "5. SafetyEvidenceBank.from_warm_start(selected) yields the bank the first TEST "
                "day is scored with."),
            "provenance_token": "oof_prequential (in_sample is refused by the validator)",
            "candidate_created_before_outcome": "candidate_created_at = index(d) - 1 < "
                                                "ordinal = index(d), asserted by the bank",
            "w_shortening_permitted": False,
            "fewer_than_seven_is": "a blocker to report, never a shorter window",
            "val_days_available": {m: splits[m]["counts"]["VAL"] for m in CB.MARKETS},
            "all_markets_have_at_least_seven_val_days": all(
                splits[m]["counts"]["VAL"] >= 7 for m in CB.MARKETS),
        },

        # ---- 7. M0 / M1 / M2
        "methods": {
            "M0": {
                "setting": "OFFLINE_STATIC_POSTHOC",
                "definition": "y_hat = y_host + alpha_0 * c",
                "uses_safety_layer": False, "uses_bank": False,
                "per_day_adaptation": False,
                "source": "core.model.plan_deployment with the safety channels disabled; "
                          "core.calibration.apply_mae_scalar",
            },
            "M1": {
                "setting": "PREQUENTIAL_SAFETY",
                "definition": "y_hat_d = y_host_d + lambda_d * c_d, "
                              "lambda_d = min(alpha_0, lambda_U_d)",
                "uses_safety_layer": True, "uses_bank": True,
                "per_day_adaptation": "lambda_d only; parameters frozen",
                "source": "core.safety.exact_mae_safe_radius + core.safety.deployment_scale",
            },
            "M2": {
                "setting": "PREQUENTIAL_SAFETY",
                "definition": "y_hat_d = y_host_d + lambda_d * c_d, "
                              "lambda_d = min(alpha_0, lambda_U_d, lambda_S_d)",
                "uses_safety_layer": True, "uses_bank": True,
                "per_day_adaptation": "lambda_d only; parameters frozen",
                "source": "core.safety.shape_weighted_safe_radius + core.safety.deployment_scale",
            },
            "shared": {
                "same_candidate_generator_for_all_three": True,
                "difference_is_only_the_deployment_scalar": True,
                "inference_setting": "FROZEN_PARAMETER_CAUSAL_PREQUENTIAL_HISTORY_POSTPROCESSING",
                "no_gradient_step_on_any_evaluation_partition": True,
                "test_labels_never_enter": True,
            },
        },

        # ---- 8. the TEST prequential bank-update rule
        "test_prequential_update_rule": {
            "applies_to": ["M1", "M2"],
            "applies_to_M0": False,
            "per_test_day_sequence": [
                "1. generate c_d from the TRAIN-frozen parameters and the already-revealed "
                "history strictly before d (never d's own target)",
                "2. bank.append_candidate(delivery_id, ordinal=index(d), "
                "candidate_created_at=index(d)-1, provenance='prequential', c_d, S+_d, S-_d, "
                "mask) — the candidate is registered BEFORE its outcome exists",
                "3. read lambda_d from the bank's last seven completed records (the bank is "
                "full by construction: it was warm-started with seven VAL records and every "
                "TEST day completes before the next begins)",
                "4. emit and persist y_hat_d = y_host_d + lambda_d * c_d",
                "5. only then reveal: bank.attach_residual(delivery_id, r_d, revealed_at=index(d))",
                "6. the bank evicts the oldest completed record beyond seven",
            ],
            "candidate_created_at_encoding": "index(d) - 1, strictly less than ordinal = index(d)",
            "revealed_at_encoding": "index(d), which is >= ordinal and > candidate_created_at",
            "no_replay": "delivery identifiers never repeat; a duplicate raises "
                         "DuplicateDeliveryError",
            "no_gap_filling": "a postponed or missing day is simply absent; the window is seven "
                              "completed honest records, not seven calendar days",
            "predict_before_reveal_is_structural": "the bank refuses a candidate created no "
                                                   "earlier than its own outcome",
            "bank_persisted_as_model_state": False,
            "bank_persisted_as": "data (to_state/from_state), passed to the controller as an "
                                 "argument, never a module attribute or state_dict entry",
        },

        "forbidden_in_any_method_run": [
            "computing a TEST metric before JOINT_TEST authorization",
            "fitting or selecting on TEST labels",
            "changing W away from 7",
            "per-market or per-Host loss weights, seeds or thresholds",
            "reintroducing a deleted component behind a config flag",
            "adding a learned gate, selector or trust score",
            "rescue-tuning the alignment-safe core",
        ],
        "harness_blocker": "SIGNED_MASS_HARNESS_ROLE_TAXONOMY_INCOMPATIBLE_WITH_COMMON_BENCHMARK",
        "splits": splits,
        "test_metric_present": False,
        "test_label_read_count": CB.PROCESS_SEAL.test_label_read_count,
        "method_results_present": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def manifest_hash(man: dict) -> str:
    body = {k: v for k, v in man.items() if k != "generated_utc"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest().upper()


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    before_digest, before_files = tree_digest(CORE)
    tests = run_core_tests()
    hist = history_assertion()
    arch = architecture_identity()
    form = safety_formula_identity()
    absent = deleted_components_absence()
    blocker = harness_blocker()
    man = method_training_manifest()
    man["method_training_manifest_sha256"] = manifest_hash(man)

    after_digest, after_files = tree_digest(CORE)
    core_unchanged = before_digest == after_digest

    CB.dump_json(OUT / "CORE_TREE_FREEZE.json", {
        "schema": "china5_common_benchmark_core_tree_freeze.v1",
        "protocol_id": CB.PROTOCOL_ID,
        "root": "src/core",
        "tree_digest_sha256": before_digest,
        "tree_digest_after_tests": after_digest,
        "unchanged_by_this_stage": core_unchanged,
        "n_files": len(before_files),
        "files": before_files,
    })
    CB.dump_json(OUT / "CORE_REGRESSION_TESTS.json", tests)
    CB.dump_json(OUT / "HISTORY_DAYS_ASSERTION.json", hist)
    CB.dump_json(OUT / "CANDIDATE_ARCHITECTURE_IDENTITY.json", arch)
    CB.dump_json(OUT / "SAFETY_FORMULA_IDENTITY.json", form)
    CB.dump_json(OUT / "DELETED_COMPONENTS_ABSENCE.json", absent)
    CB.dump_json(OUT / "HARNESS_ROLE_TAXONOMY_BLOCKER.json", blocker)
    CB.dump_json(OUT / "METHOD_TRAINING_MANIFEST.json", man)

    ok = (tests["passed"] and hist["passed"] and form["passed"] and absent["passed"]
          and core_unchanged)
    CB.dump_json(OUT / "P4_SUMMARY.json", {
        "protocol_id": CB.PROTOCOL_ID,
        "core_tree_digest": before_digest, "core_n_files": len(before_files),
        "core_unchanged_by_stage": core_unchanged,
        "core_tests": {k: tests[k] for k in ("n_passed", "n_failed", "summary_line")},
        "HISTORY_DAYS": hist["HISTORY_DAYS"], "history_passed": hist["passed"],
        "safety_formula_passed": form["passed"],
        "deleted_components_absent": absent["passed"],
        "harness_verdict": blocker["verdict"],
        "method_training_manifest_sha256": man["method_training_manifest_sha256"],
        "method_training_executed": False,
        "test_metric_present": False,
        "test_label_read_count": CB.PROCESS_SEAL.test_label_read_count,
        "passed": bool(ok),
    })

    print(f"[P4] core tests        : {tests['summary_line']} rc={tests['returncode']}")
    print(f"[P4] core tree digest  : {before_digest[:16]}… ({len(before_files)} files) "
          f"{'UNCHANGED' if core_unchanged else 'CHANGED'}")
    print(f"[P4] HISTORY_DAYS      : {hist['HISTORY_DAYS']}  "
          f"{'PASS' if hist['passed'] else 'FAIL ' + str([k for k, v in hist['checks'].items() if not v])}")
    print(f"[P4] safety formulas   : {'PASS' if form['passed'] else 'FAIL ' + str([k for k, v in form['checks'].items() if not v])}")
    print(f"[P4] deleted absent    : {'PASS' if absent['passed'] else 'FAIL'} "
          f"(tier A hits={len(absent['tier_a_definition_hits'])+len(absent['tier_a_flag_hits'])}, "
          f"mentions={len(absent['tier_b_mention_hits'])})")
    print(f"[P4] harness           : {blocker['verdict']}")
    print(f"[P4] training manifest : {man['method_training_manifest_sha256'][:16]}…  "
          f"method training executed=NO")
    print(f"[P4] TEST-label reads  : {CB.PROCESS_SEAL.test_label_read_count}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
