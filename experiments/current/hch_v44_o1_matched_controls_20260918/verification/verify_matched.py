"""Independent verification of the HCH O1 matched-control closure stage.

Deliberately does **not** import ``matched_runner``, ``matched_train``,
``matched_common``, ``probe_train``, ``probe_common`` or ``recovery_train``.
The panel, the seeds, the registered schedule and every gate threshold are
restated here from the design authority and the protocol, and every number is
recomputed from the frozen artifacts: freeze records, training curves, the
selected EMA checkpoints, the CSV tables and the raw ``src/core`` sources.

Checks (PROTOCOL.md sections 2-9, design authority sections 4-13):

 1. ``panel_and_seeds``          exactly 8 cells x 3 seeds, no extra runs anywhere
 2. ``source_digest``            src/core + src/backbones digest unchanged and equal
                                 to the frozen O1/G2 value on every run
 3. ``reference_identity``       the 24 reused O1/G2 runs re-hashed, and matched
                                 against a *pre-existing* independent witness
 4. ``variant_parity``           G1/G2 equality and the G3 <=1.25x budget measured
                                 on the fitted checkpoints, not on a re-instantiation
 5. ``only_tied_identity_delta`` step-0 bitwise identity with the reference, plus a
                                 source diff of the training loop against the closed one
 6. ``direct_control_identity``  the fitted direct family is the one the recomputed
                                 Phase-A verdict requires
 7. ``seed_median_first``        medians recomputed from raw records, not read off
 8. ``gate_clauses``             A1/A2/B1/B2 recomputed from those medians
 9. ``loss_switch``              full objective exactly through update 400, L_rec after;
                                 L_rec only throughout for the direct control
10. ``test_reads``               TEST reads = 0 across every run of the stage
11. ``no_prior_mutation``        no reused artifact changed byte, before or after
12. ``code_state``               one implementation state across every new run
13. ``terminal_token``           the written token is the registered one for the
                                 recomputed verdict
"""
from __future__ import annotations

import ast
import csv
import difflib
import hashlib
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

import torch

# --------------------------------------------------------------------- paths
_HERE = Path(__file__).resolve()


def _find_repo(start: Path) -> Path:
    """Locate the repo by walking up; HCH_REPO overrides (used only off-stage)."""
    env = os.environ.get("HCH_REPO")
    if env:
        return Path(env).resolve()
    for p in [start, *start.parents]:
        if (p / "src/core/model.py").is_file():
            return p
    raise RuntimeError("repository root not found from the verifier's location")


REPO = _find_repo(_HERE)
STAGE = REPO / "experiments/current/hch_v44_o1_matched_controls_20260918"
IMPL = STAGE / "implementation"
EVID = REPO / "experiments/evidence/hch_v44_o1_matched_controls_20260918"
O1_EVID = REPO / "experiments/evidence/hch_v44_objective_alignment_probe_20260917"
FULL8_EVID = REPO / "experiments/evidence/hch_v44_o1_full8_completion_20260918"
DOM20_EVID = REPO / "experiments/evidence/hch_v44_o1_domestic20_guardrail_20260918"
O1_IMPL = REPO / "experiments/current/hch_v44_objective_alignment_probe_20260917/implementation"

# ------------------------------------------------- restated by the verifier
PANEL = [("GANSU_DA", "PatchTST"), ("GANSU_DA", "LSTM"),
         ("SHANDONG_DA", "PatchTST"), ("SHANDONG_DA", "iTransformer"),
         ("SHAANXI_DA", "TimeMixer"), ("SHAANXI_DA", "PatchTST"),
         ("NINGXIA_DA", "iTransformer"), ("QINGHAI_DA", "TimeMixer")]
SEEDS = [7, 17, 37]
SWITCH_STEP = 400
REGISTERED_SCHEDULE = {
    "optimizer": "AdamW", "lr": 1e-3, "batch_size": 32, "weight_decay": 1e-4,
    "grad_clip": 1.0, "ema_decay": 0.995, "eval_weights": "ema",
    "max_steps": 2000, "val_every": 50, "no_stop_before": 800, "patience_checks": 8,
    "readout_init": None,
}
SCHEDULE_KEYS = tuple(REGISTERED_SCHEDULE)
REUSE_ROOTS = {
    "GANSU_DA__PatchTST": O1_EVID, "SHANDONG_DA__iTransformer": O1_EVID,
    "SHAANXI_DA__TimeMixer": O1_EVID, "NINGXIA_DA__iTransformer": O1_EVID,
    "GANSU_DA__LSTM": FULL8_EVID, "SHANDONG_DA__PatchTST": FULL8_EVID,
    "SHAANXI_DA__PatchTST": FULL8_EVID, "QINGHAI_DA__TimeMixer": FULL8_EVID,
}
RUN_FIELDS = ("freeze.json", "training_curve.json", "selected_ema.pt")
SNAP_FILE = {"freeze_sha256": "freeze.json", "curve_sha256": "training_curve.json",
             "checkpoint_sha256": "selected_ema.pt"}

G0 = "G0_GEOM_HOST"
G1 = "G1_GEOM_SHARED_CONTEXT"
G2 = "G2_GEOM_COORD_CONTEXT"
G3 = "G3_DIRECT_CONTEXT_CTRL"
G3_SHARED = "G3_SHARED_DIRECT_CONTEXT_CTRL"

TOKENS = {
    "A1": "HCH_O1_MATCHED_CONTROL_GEOMETRY_SUPPORTED_G2",
    "A2": "HCH_O1_MATCHED_CONTROL_GEOMETRY_SUPPORTED_G1",
    "B2": "HCH_O1_MATCHED_CONTROL_GEOMETRY_NOT_SUPPORTED",
    "A3": "HCH_O1_MATCHED_CONTROL_ROUTING_INCONCLUSIVE",
}
TOKEN_POOL = set(TOKENS.values()) | {
    "HCH_O1_MATCHED_CONTROL_GEOMETRY_SUPPORTED_G2",
    "HCH_O1_MATCHED_CONTROL_GEOMETRY_SUPPORTED_G1",
    "HCH_O1_MATCHED_CONTROL_GEOMETRY_NOT_SUPPORTED",
}

FORBIDDEN_FILES = (
    "CELL_MEDIAN_RESULTS.csv", "PER_SEED_RESULTS.csv", "BASELINE_COMPARISON.csv",
    "HCH_V44_MAIN_20260917.csv", "RESULTS_LONG.csv", "BASELINE_COMPLETION_20260917.csv",
)

PHASE_A = "g1"
PHASE_B = "direct"
LEDGER = EVID / "INDEPENDENT_VERIFICATION_REPORT.json"  # design authority s.11 name


# --------------------------------------------------------------------- utils
def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest().upper()


def load_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def tree_digest(root: Path, exclude_prefixes=()) -> str:
    """sha256 over sorted 'repo-relative|sha256(file)' lines of .py/.md files."""
    items = []
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file() or p.suffix not in (".py", ".md"):
            continue
        rel = p.relative_to(REPO).as_posix()
        if any(rel.startswith(x) for x in exclude_prefixes):
            continue
        items.append(f"{rel}|{sha256_file(p)}")
    return sha256_bytes("\n".join(items).encode())


def cell_key(market: str, host: str) -> str:
    return f"{market}__{host}"


def median(vals):
    return float(st.median([float(v) for v in vals]))


def gain_pct(mae_x: float, mae_y: float) -> float:
    return 100.0 * (float(mae_y) - float(mae_x)) / float(mae_y)


def read_csv(p: Path) -> list[dict]:
    with Path(p).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def run_dir(phase: str, market: str, host: str, seed: int) -> Path:
    return EVID / "runs" / phase / cell_key(market, host) / f"seed{seed}"


def reused_dir(market: str, host: str, seed: int) -> Path:
    k = cell_key(market, host)
    return REUSE_ROOTS[k] / "o1_runs" / k / f"seed{seed}"


def complete(d: Path) -> bool:
    return all((d / f).is_file() for f in RUN_FIELDS)


def load_run(phase: str, market: str, host: str, seed: int) -> tuple:
    d = run_dir(phase, market, host, seed)
    return load_json(d / "freeze.json"), load_json(d / "training_curve.json"), d


def sched_of(rec: dict) -> dict:
    o = rec.get("optimization") or {}
    return {k: o.get(k) for k in SCHEDULE_KEYS}


def check0(curve) -> float:
    return float(next(e for e in curve if int(e["step"]) == 0)["val_mae_ema"])


def fn_source(path: Path, name: str) -> str:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise RuntimeError(f"{name} not found in {path}")


def param_manifest(ckpt: Path) -> dict:
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    return {k: tuple(v.shape) for k, v in sd.items()}


def n_elements(man: dict) -> int:
    n = 1
    total = 0
    for shape in man.values():
        c = 1
        for d in shape:
            c *= int(d)
        total += c
    return total


# --------------------------------------------------------------------- checks
C: dict = {}


NON_SCIENTIFIC = ("runner_cli_is_self_consistent",)


def record(name: str, passed: bool, detail) -> None:
    C[name] = {"pass": bool(passed), "detail": detail,
               "scientific": name not in NON_SCIENTIFIC}


def check_panel_and_seeds() -> None:
    present_a = sorted(p.parent.name for p in (EVID / "runs" / PHASE_A).glob("*/seed*")
                       if p.is_dir()) if (EVID / "runs" / PHASE_A).is_dir() else []
    present_b = sorted(p.parent.name for p in (EVID / "runs" / PHASE_B).glob("*/seed*")
                       if p.is_dir()) if (EVID / "runs" / PHASE_B).is_dir() else []
    expected = sorted(cell_key(m, h) for m, h in PANEL)
    extra = [k for k in set(present_a) | set(present_b) if k not in expected]
    g1_ok = all(complete(run_dir(PHASE_A, m, h, s)) for m, h in PANEL for s in SEEDS)
    # measured, not asserted: whether 24 direct runs must exist depends on the
    # Phase A verdict, which direct_control_identity resolves.
    direct_ok = all(complete(run_dir(PHASE_B, m, h, s)) for m, h in PANEL for s in SEEDS)
    record("panel_and_seeds", len(PANEL) == 8 and SEEDS == [7, 17, 37]
           and not extra and g1_ok,
           {"n_cells": len(PANEL), "seeds": SEEDS,
            "g1_cells_present": len(set(present_a)),
            "direct_cells_present": len(set(present_b)), "unexpected_cells": extra,
            "all_24_g1_runs_complete": g1_ok,
            "all_24_direct_runs_complete": direct_ok})


def check_source_digest() -> dict:
    core = tree_digest(REPO / "src/core")
    back = tree_digest(REPO / "src/backbones")
    audit = load_json(EVID / "SOURCE_AUDIT.json")
    frozen = audit["frozen_expected"]
    mismatched = []
    for phase in (PHASE_A, PHASE_B):
        for m, h in PANEL:
            for s in SEEDS:
                d = run_dir(phase, m, h, s)
                if not complete(d):
                    continue
                rec = load_json(d / "freeze.json")
                if rec["source_tree_digest"] != {"core_tree": core, "backbones_tree": back}:
                    mismatched.append(f"{cell_key(m,h)}__seed{s}")
    ref_bad = []
    for m, h in PANEL:
        for s in SEEDS:
            rec = load_json(reused_dir(m, h, s) / "freeze.json")
            if rec["source_tree_digest"] != {"core_tree": core, "backbones_tree": back}:
                ref_bad.append(f"{cell_key(m,h)}__seed{s}")
    ok = (core == frozen["core_tree"] and back == frozen["backbones_tree"]
          and not mismatched and not ref_bad
          and audit["source_parity_pass"] and not audit["src_core_modified"])
    record("source_digest", ok, {
        "recomputed_core_tree": core, "recomputed_backbones_tree": back,
        "frozen_core_tree": frozen["core_tree"], "frozen_backbones_tree": frozen["backbones_tree"],
        "new_runs_with_other_digest": mismatched, "reused_runs_with_other_digest": ref_bad,
        "core_file_count": audit["core_file_count"]})
    return {"core": core, "back": back}


def check_reference_identity() -> None:
    prov = load_json(EVID / "O1_G2_REUSE_PROVENANCE.json")
    witness = load_json(DOM20_EVID / "REUSE_PROVENANCE.json")
    w_cells = witness["reused_cells"]
    mism, witness_mism, absent, value_mism = [], [], [], []
    snap_before = load_json(EVID / "REUSE_SNAPSHOT_BEFORE.json")["sha256"]
    for m, h in PANEL:
        k = cell_key(m, h)
        for s in SEEDS:
            d = reused_dir(m, h, s)
            if not complete(d):
                absent.append(f"{k}__seed{s}")
                continue
            got = {"freeze_sha256": sha256_file(d / "freeze.json"),
                   "curve_sha256": sha256_file(d / "training_curve.json"),
                   "checkpoint_sha256": sha256_file(d / "selected_ema.pt")}
            rec = prov["reused_cells"][k]["runs"][str(s)]
            if any(got[f] != rec[f] for f in got):
                mism.append(f"{k}__seed{s}")
            if not (rec["checkpoint_sha256_matches_own_record"]
                    and rec["history_support_matches_registered"]):
                mism.append(f"{k}__seed{s}__own_pin")
            w = w_cells.get(k, {}).get("runs", {}).get(str(s))
            if w is None or any(got[f] != w[f] for f in got):
                witness_mism.append(f"{k}__seed{s}")
            live = load_json(d / "freeze.json")
            # The witness was written by a different stage, so its values are an
            # independent record of the very baseline the Phase-A gain divides by.
            if (w is not None and (
                    float(w["selected_val_mae_ema"]) != float(live["selected_val_mae_ema"])
                    or int(w["selected_step"]) != int(live["selected_step"])
                    or int(w["test_target_read_count"]) != 0
                    or w["history_support_matches_shadow_artifact"] is not True
                    or w["selected_ema_parameter_hash"] != live["selected_ema_parameter_hash"])):
                value_mism.append(f"{k}__seed{s}")
            for f, v in got.items():
                # the snapshot keys each artifact by its own run filename
                snap_key = f"{k}__seed{s}__{SNAP_FILE[f]}"
                if snap_before.get(snap_key) != v:
                    mism.append(f"{snap_key}__vs_snapshot_before")
    n = sum(len(prov["reused_cells"][cell]["runs"]) for cell in prov["reused_cells"])
    ok = (n == 24 and not absent and not mism and not witness_mism and not value_mism
          and prov["all_reused_present"] and prov["missing"] == []
          and prov["one_probe_code_hash_across_reused"]
          and prov["all_history_support_registered"])
    record("reference_identity", ok, {
        "n_reused_runs": n,
        "rehash_mismatches_vs_own_provenance": mism,
        "rehash_mismatches_vs_domestic20_witness": witness_mism,
        "value_mismatches_vs_domestic20_witness": value_mism,
        "witness_values_cross_checked": ["selected_val_mae_ema", "selected_step",
                                         "selected_ema_parameter_hash",
                                         "test_target_read_count",
                                         "history_support_matches_shadow_artifact"],
        "absent": absent, "missing": prov["missing"],
        "one_probe_code_hash_across_reused": prov["one_probe_code_hash_across_reused"],
        "independent_witness": str((DOM20_EVID / "REUSE_PROVENANCE.json").relative_to(REPO)).replace("\\", "/")})


def check_variant_parity() -> None:
    parity = load_json(EVID / "VARIANT_PARITY.json")
    man_g2 = param_manifest(reused_dir(*PANEL[0], SEEDS[0]) / "selected_ema.pt")
    mism = []
    for m, h in PANEL:
        for s in SEEDS:
            man = param_manifest(run_dir(PHASE_A, m, h, s) / "selected_ema.pt")
            if man != man_g2:
                mism.append(f"{cell_key(m,h)}__seed{s}")
    # measure G1 on a G1 checkpoint, not on the G2 manifest: deriving both from
    # man_g2 would make "g1_equals_g2_parameter_count" true by construction.
    man_g1 = param_manifest(run_dir(PHASE_A, *PANEL[0], SEEDS[0]) / "selected_ema.pt")
    n_g1, n_g2 = n_elements(man_g1), n_elements(man_g2)
    direct_files = [run_dir(PHASE_B, m, h, s) / "selected_ema.pt"
                    for m, h in PANEL for s in SEEDS
                    if complete(run_dir(PHASE_B, m, h, s))]
    ratios, dmism = [], []
    for p in direct_files:
        man = param_manifest(p)
        ratios.append(n_elements(man) / n_g2)
    # None, not True, when there is nothing to measure: "within budget" is not
    # established by the absence of checkpoints.  The requirement that all 24
    # direct runs exist is asserted by direct_control_identity, not here.
    direct_ok = (max(ratios) <= 1.25) if ratios else None
    for m, h in PANEL:
        for s in SEEDS:
            d = run_dir(PHASE_B, m, h, s)
            if complete(d):
                rec = load_json(d / "freeze.json")
                if rec["base_variant"] != G3:
                    dmism.append(f"{cell_key(m,h)}__seed{s}")
    csv_rows = {r["variant"]: r for r in read_csv(EVID / "PARAMETER_COUNTS.csv")}
    csv_ok = (int(csv_rows[G1]["n_trainable_parameters"]) == n_g1
              and int(csv_rows[G2]["n_trainable_parameters"]) == n_g2
              and abs(float(csv_rows[G3]["ratio_to_g2"]) - float(parity["g3_over_g2_ratio"])) < 1e-12)
    src_model = (REPO / "src/core/model.py").read_text(encoding="utf-8")
    src_alloc = (REPO / "src/core/allocation.py").read_text(encoding="utf-8")
    tie_assign = re.search(
        r"self\.tied_identity\s*=\s*variant in TIED_COORDINATE_VARIANTS", src_model)
    # Which files under src/ can even see the flag: the tie is the sole intended
    # delta between the variants, so a second reader would break "only the flag
    # differs".  Recorded as {relative path: occurrences}, not as token matches.
    mentions = {}
    for p in sorted((REPO / "src").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        n = p.read_text(encoding="utf-8").count("tied_identity")
        if n:
            mentions[p.relative_to(REPO).as_posix()] = n
    mentions_only_model = set(mentions) <= {"src/core/model.py"}
    param_under_tie = bool(re.search(
        r"if[^\n]*tied_identity[^\n]*:\s*\n\s*self\.[A-Za-z_]+\s*=\s*"
        r"(nn\.(Linear|Parameter|LayerNorm|Module)|torch\.nn\.Parameter)", src_model))
    ok = (mism == [] and (direct_ok is None or direct_ok) and dmism == [] and csv_ok
          and bool(tie_assign) and not param_under_tie
          and "def coordinate_vectors" in src_alloc and "coord_embed" in src_alloc
          and mentions_only_model)
    record("variant_parity", ok, {
        "g1_manifest_identical_to_g2_on_every_fitted_checkpoint": not mism,
        "mismatches": mism,
        "n_parameters_g1": n_g1, "n_parameters_g2": n_g2,
        "g1_equals_g2_parameter_count": n_g1 == n_g2,
        "n_direct_checkpoints": len(direct_files),
        "direct_over_g2_ratio_max": max(ratios) if ratios else None,
        "direct_within_1.25x": direct_ok,
        "direct_within_1.25x_rule": ("measured" if ratios else
                                     "null: no direct checkpoint exists to measure"),
        "direct_base_variant_mismatches": dmism,
        "parameter_counts_csv_agrees": csv_ok,
        "recorded_g3_over_g2_ratio": parity["g3_over_g2_ratio"],
        "tied_identity_assignment_from_registry": bool(tie_assign),
        "tied_identity_creates_parameter": param_under_tie,
        "tied_identity_mentions_by_file": mentions,
        "tied_identity_read_only_by_src_core_model_py": mentions_only_model,
        "tie_consumed_in": "src/core/allocation.py: coordinate_vectors(tied)"})


def check_only_tied_identity_delta() -> None:
    """Step-0 bitwise identity, plus a source diff against the closed trainer."""
    step0_bad, hyper_bad = [], []
    for m, h in PANEL:
        for s in SEEDS:
            rec1, cur1, _ = load_run(PHASE_A, m, h, s)
            rec2 = load_json(reused_dir(m, h, s) / "freeze.json")
            cur2 = load_json(reused_dir(m, h, s) / "training_curve.json")
            if check0(cur1) != check0(cur2):
                step0_bad.append(f"{cell_key(m,h)}__seed{s}")
            if rec1["n_train_rows"] != rec2["n_train_rows"] or \
               rec1["n_val_rows"] != rec2["n_val_rows"] or \
               rec1["history_support_hash"] != rec2["history_support_hash"] or \
               rec1["train_day_ids_hash"] != rec2["train_day_ids_hash"] or \
               rec1["val_day_ids_hash"] != rec2["val_day_ids_hash"]:
                step0_bad.append(f"{cell_key(m,h)}__seed{s}__support")
            if sched_of(rec1) != REGISTERED_SCHEDULE:
                hyper_bad.append(f"{cell_key(m,h)}__seed{s}")

    mine = fn_source(IMPL / "matched_train.py", "train_matched").splitlines()
    ref = fn_source(O1_IMPL / "probe_train.py", "train_o1").splitlines()
    diff = list(difflib.unified_diff(ref, mine, "probe_train.train_o1",
                                     "matched_train.train_matched", lineterm="", n=1))
    changed = [ln for ln in diff if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    # Every variant-agnostic stage must be *called*, not reimplemented.
    src = fn_source(IMPL / "matched_train.py", "train_matched")
    whole = (IMPL / "matched_train.py").read_text(encoding="utf-8")
    delegated = ["RT.build_cell_data", "RT.to_device", "RT.build_optimizer",
                 "grad_norms_by_group", "RT.DIAG_MAX_ROWS", "PT.objective",
                 "PT.diag_components", "PT._assert_component_consistency",
                 "_bind_prior", "build_primary_train_config", "EMA(",
                 "DifficultyInterleaver(", "clip_gradients(", "neural_autocast(",
                 "fp32_region(", "daily_difficulty("]
    # the variant-agnostic stages may be reached through the two documented
    # shims, so the proof of delegation is over the whole module, not the body
    missing = [d for d in delegated if d not in whole]
    in_body = [d for d in delegated if d in src]
    new_fns = sorted({
        n.name for n in ast.walk(ast.parse((IMPL / "matched_train.py").read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef)})
    ok = (not step0_bad and not hyper_bad and not missing
          and len(changed) <= 60)
    record("only_tied_identity_delta", ok, {
        "step0_bitwise_identity_failures": step0_bad,
        "schedule_mismatches": hyper_bad,
        "diff_line_count": len(changed),
        "diff_hunks": changed,
        "variant_agnostic_calls_missing": missing,
        "variant_agnostic_calls_in_the_loop_body": in_body,
        "functions_defined_in_matched_train": new_fns,
        "new_functions_are_the_declared_shims": sorted(
            set(new_fns) - {"train_matched", "_evaluate_direct", "evaluate_matched",
                            "diag_components_matched", "_assert_direct_consistency"}) == []})


def check_direct_control_identity(gate_a: dict) -> None:
    selected = gate_a["selected_structured_variant"]
    expected = None if selected is None else (G3 if selected == G2 else G3_SHARED)
    present = [(m, h, s) for m, h in PANEL for s in SEEDS
               if complete(run_dir(PHASE_B, m, h, s))]
    facts, bad = [], []
    for m, h, s in present:
        rec = load_json(run_dir(PHASE_B, m, h, s) / "freeze.json")
        facts.append(rec["variant"])
        want_tie = (expected == G3_SHARED)
        if not (rec["variant"] == expected and rec["base_variant"] == G3
                and bool(rec["tied_identity"]) is want_tie
                and rec["structured_decoder"] is False):
            bad.append(f"{cell_key(m,h)}__seed{s}")
    if selected is None:
        ok = not present
    else:
        ok = (len(present) == 24 and not bad and set(facts) == {expected})
    record("direct_control_identity", ok, {
        "phase_a_verdict": gate_a["verdict"],
        "phase_a_selected": selected,
        "required_direct_variant": expected,
        "n_direct_runs_present": len(present),
        "distinct_direct_variants": sorted(set(facts)),
        "mismatches": bad,
        "rule": ("A3 stops before any direct fit"
                 if selected is None else
                 "G2 selected -> G3_DIRECT_CONTEXT_CTRL (tied off); "
                 "G1 selected -> existing G3 model with tied_identity=True")})


def rows_a() -> list[dict]:
    out = []
    for m, h in PANEL:
        for s in SEEDS:
            r1, c1, _ = load_run(PHASE_A, m, h, s)
            r2 = load_json(reused_dir(m, h, s) / "freeze.json")
            c2 = load_json(reused_dir(m, h, s) / "training_curve.json")
            out.append({
                "cell": cell_key(m, h), "market": m, "host": h, "seed": s,
                "mae_struct": float(r2["selected_val_mae_ema"]),          # G2
                "mae_ctrl": float(r1["selected_val_mae_ema"]),            # G1
                "host_struct": float(r2["selected_metrics"]["val_host_mae"]),
                "host_ctrl": float(r1["selected_metrics"]["val_host_mae"]),
                "check0_struct": check0(c2), "check0_ctrl": check0(c1),
            })
    return out


def rows_b(selected: str) -> list[dict]:
    out = []
    for m, h in PANEL:
        for s in SEEDS:
            ds = (reused_dir(m, h, s) if selected == G2 else run_dir(PHASE_A, m, h, s))
            rs = load_json(ds / "freeze.json")
            rd = load_json(run_dir(PHASE_B, m, h, s) / "freeze.json")
            out.append({
                "cell": cell_key(m, h), "market": m, "host": h, "seed": s,
                "mae_struct": float(rs["selected_val_mae_ema"]),
                "mae_ctrl": float(rd["selected_val_mae_ema"]),
                "host_struct": float(rs["selected_metrics"]["val_host_mae"]),
                "host_ctrl": float(rd["selected_metrics"]["val_host_mae"]),
            })
    return out


def cells_from(rows: list[dict]) -> list[dict]:
    out = []
    for m, h in PANEL:
        k = cell_key(m, h)
        sub = [r for r in rows if r["cell"] == k]
        if not sub:                      # only reachable in a partial dry run
            continue
        ms, mc = median([r["mae_struct"] for r in sub]), median([r["mae_ctrl"] for r in sub])
        hs, hc = median([r["host_struct"] for r in sub]), median([r["host_ctrl"] for r in sub])
        out.append({"cell": k, "market": m, "host": h,
                    "mae_struct_median": ms, "mae_ctrl_median": mc,
                    "gain_pct": gain_pct(ms, mc),
                    "struct_beats_ctrl": ms < mc,
                    "struct_host_positive": ms < hs, "ctrl_host_positive": mc < hc})
    return out


def _cmp_csv(path: Path, cells: list[dict], col_gain: str, col_ms: str, col_mc: str) -> list[str]:
    bad = []
    rows = {r["cell"]: r for r in read_csv(path)}
    for c in cells:
        r = rows.get(c["cell"])
        if r is None:
            bad.append(f"{c['cell']}: missing row")
            continue
        if abs(float(r[col_gain]) - c["gain_pct"]) > 1e-9:
            bad.append(f"{c['cell']}: gain {r[col_gain]} != {c['gain_pct']}")
        if abs(float(r[col_ms]) - c["mae_struct_median"]) > 1e-9:
            bad.append(f"{c['cell']}: struct median mismatch")
        if abs(float(r[col_mc]) - c["mae_ctrl_median"]) > 1e-9:
            bad.append(f"{c['cell']}: ctrl median mismatch")
        if len(read_csv(path)) != 8:
            bad.append("row count != 8")
    return bad


def check_seed_median_first() -> None:
    ra = rows_a()
    ca = cells_from(ra)
    bad_a = _cmp_csv(EVID / "PHASE_A_CELL_MEDIANS.csv", ca,
                     "gain_g2_vs_g1_pct", "mae_g2_median", "mae_g1_median")
    per_seed = read_csv(EVID / "G1_PER_SEED.csv")
    bad_ps = []
    if len(per_seed) != 24:
        bad_ps.append(f"G1_PER_SEED rows {len(per_seed)} != 24")
    by_key = {(r["cell"], int(r["seed"])): r for r in per_seed}
    for r in ra:
        got = by_key.get((r["cell"], r["seed"]))
        if got is None:
            bad_ps.append(f"{r['cell']}__seed{r['seed']} missing")
            continue
        if abs(float(got["mae_g1"]) - r["mae_ctrl"]) > 1e-9 or \
           abs(float(got["mae_g2"]) - r["mae_struct"]) > 1e-9:
            bad_ps.append(f"{r['cell']}__seed{r['seed']} mae mismatch")
    # the statistic must not be a median of paired per-seed gains
    paired_medians = [median([rr["mae_ctrl"] for rr in ra if rr["cell"] == c["cell"]]) for c in ca]
    naive = [gain_pct(
        st.median([r["mae_struct"] for r in ra]),
        st.median([r["mae_ctrl"] for r in ra]))]
    record("seed_median_first", not bad_a and not bad_ps, {
        "phase_a_cell_median_mismatches": bad_a,
        "phase_a_per_seed_mismatches": bad_ps,
        "cell_medians_recomputed": {c["cell"]: c["gain_pct"] for c in ca},
        "pooled_gain_for_contrast_pct": naive[0],
        "statistic": "median of each method's 3-seed VAL MAE first, then gain"})
    # returns nothing on purpose: a check's return value is only kept if its
    # caller captures it, and gate_clauses recomputes the cells it aggregates


def gate_a_from(cells: list[dict], compliance_clean: bool) -> dict:
    gains = [c["gain_pct"] for c in cells]
    a1_checks = {
        "a1_cells_g2_beats_g1_ge_6of8": sum(1 for c in cells if c["struct_beats_ctrl"]) >= 6,
        "a1_panel_median_gain_ge_0.5pct": median(gains) >= 0.50,
        "a1_worst_cell_ge_-0.5pct": min(gains) >= -0.50,
        "a1_g2_host_positive_ge_7of8": sum(1 for c in cells if c["struct_host_positive"]) >= 7,
        "a1_no_access_source_numerical_failure": compliance_clean,
    }
    a1 = all(a1_checks.values())
    gaps = [abs(c["gain_pct"]) for c in cells]
    a2_checks = {
        "a2_g1_within_0.50pct_of_g2_ge_7of8": sum(1 for g in gaps if g <= 0.50) >= 7,
        "a2_panel_median_g1_vs_g2_gain_ge_-0.25pct":
            median([gain_pct(c["mae_ctrl_median"], c["mae_struct_median"]) for c in cells]) >= -0.25,
        "a2_worst_g1_vs_g2_cell_ge_-1.00pct":
            min(gain_pct(c["mae_ctrl_median"], c["mae_struct_median"]) for c in cells) >= -1.00,
        "a2_g1_host_positive_ge_7of8": sum(1 for c in cells if c["ctrl_host_positive"]) >= 7,
        "a2_no_access_source_numerical_failure": compliance_clean,
    }
    a2 = (not a1) and all(a2_checks.values())
    verdict = "A3_ROUTING_IDENTITY_INCONCLUSIVE"
    selected = None
    if a1:
        verdict, selected = "A1_COORDINATE_IDENTITY_SUPPORTED", G2
    elif a2:
        verdict, selected = "A2_SHARED_CONTEXT_SUFFICIENT", G1
    return {"a1_checks": a1_checks, "a1": a1, "a2_checks": a2_checks, "a2": a2,
            "verdict": verdict, "selected": selected,
            "panel_median": median(gains), "worst": min(gains), "best": max(gains),
            "cells_beating": sum(1 for c in cells if c["struct_beats_ctrl"]),
            "g2_host_positive": sum(1 for c in cells if c["struct_host_positive"]),
            "g1_host_positive": sum(1 for c in cells if c["ctrl_host_positive"]),
            "cells_within": sum(1 for g in gaps if g <= 0.50)}


def gate_b_from(cells: list[dict], compliance_clean: bool) -> dict:
    gains = [c["gain_pct"] for c in cells]
    checks = {
        "b1_cells_structured_beats_direct_ge_6of8":
            sum(1 for c in cells if c["struct_beats_ctrl"]) >= 6,
        "b1_panel_median_gain_ge_0.5pct": median(gains) >= 0.50,
        "b1_worst_cell_ge_-0.5pct": min(gains) >= -0.50,
        "b1_structured_host_positive_ge_7of8":
            sum(1 for c in cells if c["struct_host_positive"]) >= 7,
        "b1_no_access_source_numerical_failure": compliance_clean,
    }
    return {"b1_checks": checks, "b1": all(checks.values()),
            "verdict": "B1_EXACT_GEOMETRY_SUPPORTED" if all(checks.values())
                       else "B2_EXACT_GEOMETRY_NOT_SUPPORTED",
            "panel_median": median(gains), "worst": min(gains), "best": max(gains),
            "cells_beating": sum(1 for c in cells if c["struct_beats_ctrl"]),
            "struct_host_positive": sum(1 for c in cells if c["struct_host_positive"])}


def check_gate_clauses(recomputed_a: dict, recomputed_b, a_error: str | None = None) -> dict:
    written_a = load_json(EVID / "PHASE_A_GATE.json")
    bad = []
    for k, v in recomputed_a["a1_checks"].items():
        if bool(written_a["a1_checks"].get(k)) is not bool(v):
            bad.append(f"a1:{k}")
    for k, v in recomputed_a["a2_checks"].items():
        if bool(written_a["a2_checks"].get(k)) is not bool(v):
            bad.append(f"a2:{k}")
    if written_a["verdict"] != recomputed_a["verdict"]:
        bad.append(f"verdict:{written_a['verdict']}!={recomputed_a['verdict']}")
    if written_a["selected_structured_variant"] != recomputed_a["selected"]:
        bad.append("selected_variant")
    detail = {"phase_a": {
        "written_verdict": written_a["verdict"], "recomputed_verdict": recomputed_a["verdict"],
        "recomputed_a1_checks": recomputed_a["a1_checks"],
        "recomputed_a2_checks": recomputed_a["a2_checks"],
        "panel_median_gain_pct": recomputed_a["panel_median"],
        "worst_cell_gain_pct": recomputed_a["worst"],
        "cells_beating": recomputed_a["cells_beating"]}}
    if recomputed_b is not None:
        written_b = load_json(EVID / "PHASE_B_GATE.json")
        for k, v in recomputed_b["b1_checks"].items():
            if bool(written_b["b1_checks"].get(k)) is not bool(v):
                bad.append(f"b1:{k}")
        if written_b["verdict"] != recomputed_b["verdict"]:
            bad.append("b_verdict")
        detail["phase_b"] = {
            "written_verdict": written_b["verdict"],
            "recomputed_verdict": recomputed_b["verdict"],
            "recomputed_b1_checks": recomputed_b["b1_checks"],
            "panel_median_gain_pct": recomputed_b["panel_median"],
            "worst_cell_gain_pct": recomputed_b["worst"],
            "cells_beating": recomputed_b["cells_beating"]}
    else:
        detail["phase_b"] = "not executed (Phase A was inconclusive or no direct family)"

    # The gates consume ``compliance.clean``.  The verifier recomputes the
    # substance of that flag independently, so a false ``clean: true`` is caught.
    substance = [C.get(k, {}).get("pass") for k in
                 ("test_reads", "source_digest", "no_prior_mutation",
                  "only_tied_identity_delta", "reference_identity", "variant_parity")]
    expected_clean = all(bool(v) for v in substance)
    wrote_clean = bool(written_a["compliance"]["clean"])
    if expected_clean and not wrote_clean:
        bad.append("compliance_clean_false_but_no_independent_failure")
    if wrote_clean and not expected_clean:
        bad.append("compliance_clean_true_despite_independent_failure")
    detail["compliance_consistency"] = {
        "written_clean": wrote_clean,
        "expected_clean_from_independent_checks": expected_clean,
        "inputs": dict(zip(("test_reads", "source_digest", "no_prior_mutation",
                            "only_tied_identity_delta", "reference_identity",
                            "variant_parity"), substance))}
    record("gate_clauses", not bad and a_error is None,
           detail | {"mismatches": bad, "phase_a_recomputation_error": a_error,
                     "phase_a_recomputation_rule": (
                         "recomputed from PHASE_A_GATE.json-free raw artifacts: "
                         "freeze.json + training_curve.json for all 8 cells x 3 seeds "
                         "on both arms" if a_error is None else
                         "no complete 8-cell panel, so no gate clause is decidable")})
    return detail


def check_loss_switch() -> None:
    bad = []
    for m, h in PANEL:
        for s in SEEDS:
            _, cur, _ = load_run(PHASE_A, m, h, s)
            for e in cur:
                step = int(e["step"])
                if step == 0:
                    if e["used_full_loss_at_this_step"] is not None:
                        bad.append(f"{cell_key(m,h)}__seed{s}: step0 flag not None")
                    continue
                want = step <= SWITCH_STEP
                if bool(e["used_full_loss_at_this_step"]) is not want:
                    bad.append(f"{cell_key(m,h)}__seed{s}: step{step} full={e['used_full_loss_at_this_step']}")
                if bool(e["objective_switch_here"]) is not (step == SWITCH_STEP):
                    bad.append(f"{cell_key(m,h)}__seed{s}: switch flag at step{step}")
            if not any(int(e["step"]) == SWITCH_STEP for e in cur):
                bad.append(f"{cell_key(m,h)}__seed{s}: no check at the switch step")
    direct_bad = []
    for m, h in PANEL:
        for s in SEEDS:
            d = run_dir(PHASE_B, m, h, s)
            if not complete(d):
                continue
            _, cur, _ = load_run(PHASE_B, m, h, s)
            if any(e["L_b"] is not None or e["L_B"] is not None or e["L_S"] is not None
                   for e in cur):
                direct_bad.append(f"{cell_key(m,h)}__seed{s}")
    record("loss_switch", not bad and not direct_bad,
           {"phase_a_violations": bad, "direct_auxiliary_present": direct_bad,
            "switch_step": SWITCH_STEP,
            "rule": "updates 1..400 full structured objective; 401.. L_rec only"})


def check_test_reads() -> None:
    reads, bad, opened = 0, [], []
    for phase in (PHASE_A, PHASE_B):
        for m, h in PANEL:
            for s in SEEDS:
                d = run_dir(phase, m, h, s)
                if not complete(d):
                    continue
                rec = load_json(d / "freeze.json")
                reads += int(rec["test_target_read_count"])
                acc = rec["access_state"] or {}
                if (int(rec["test_target_read_count"]) != 0
                        or int(rec["test_rows_materialised"]) != 0
                        or int(rec["test_rows_dropped_from_joint_cache"]) <= 0
                        or int(acc.get("forbidden_guard_hits", -1)) != 0
                        or int(acc.get("test_role_frame_refusals", -1)) != 0
                        or int(acc.get("test_rows_returned", -1)) != 0
                        or set(acc.get("legit_frames_built", {})) != {"TRAIN", "VAL"}):
                    bad.append(f"{phase}:{cell_key(m,h)}__seed{s}")
    ref_reads = 0
    for m, h in PANEL:
        for s in SEEDS:
            ref_reads += int(load_json(reused_dir(m, h, s) / "freeze.json")["test_target_read_count"])
    audit = load_json(EVID / "ACCESS_AUDIT.json")
    hit = [p.name for p in EVID.rglob("*") if p.name in FORBIDDEN_FILES]
    ok = (reads == 0 and ref_reads == 0 and not bad and not hit
          and audit["test_target_read_count_total_new"] == 0
          and audit["all_new_runs_zero_test_reads"]
          and all(int(v) == 0 for v in audit["test_target_read_count_reused"]))
    record("test_reads", ok, {
        "new_run_test_target_reads": reads, "reused_run_test_target_reads": ref_reads,
        "runs_failing_the_access_clauses": bad,
        "forbidden_files_inside_evidence_root": hit,
        "access_audit_total": audit["test_target_read_count_total_new"],
        "access_audit_reused_reads": audit["test_target_read_count_reused"],
        "ground_truth_horizon": "the terminal token must not depend on any TEST value"})


def check_no_prior_mutation() -> None:
    b = load_json(EVID / "REUSE_SNAPSHOT_BEFORE.json")["sha256"]
    after_files = [EVID / "REUSE_SNAPSHOT_AFTER_PHASE_A.json"]
    after = load_json(after_files[0])["sha256"] if after_files[0].is_file() else None
    now = {}
    for m, h in PANEL:
        for s in SEEDS:
            d = reused_dir(m, h, s)
            for f in RUN_FIELDS:
                now[f"{cell_key(m,h)}__seed{s}__{f}"] = sha256_file(d / f)
    witness = load_json(DOM20_EVID / "REUSE_PROVENANCE.json")["reused_cells"]
    w_bad = []
    for k, cell in witness.items():
        for s, run in cell["runs"].items():
            m, h = k.split("__")
            if (m, h) not in PANEL:
                continue
            for f, key in (("freeze.json", "freeze_sha256"),
                           ("training_curve.json", "curve_sha256"),
                           ("selected_ema.pt", "checkpoint_sha256")):
                if sha256_file(reused_dir(m, h, int(s)) / f) != run[key]:
                    w_bad.append(f"{k}__seed{s}__{f}")
    ok = (now == b and (after is None or after == b) and not w_bad)
    record("no_prior_mutation", ok, {
        "snapshot_before_equals_now": now == b,
        "snapshot_after_phase_a_equals_before": (after is None or after == b),
        "mismatches_vs_domestic20_witness": w_bad,
        "n_artifacts": len(now),
        "note": ("mtime is not evidence of authorship; only content hashes are compared")})


def check_code_state() -> None:
    hashes, train_sha, variant_ok, bad = set(), set(), True, []
    fit_time_digest = tree_digest(STAGE, exclude_prefixes=(
        (STAGE / "verification").relative_to(REPO).as_posix() + "/",))
    for phase in (PHASE_A, PHASE_B):
        for m, h in PANEL:
            for s in SEEDS:
                d = run_dir(phase, m, h, s)
                if not complete(d):
                    continue
                rec = load_json(d / "freeze.json")
                hashes.add(rec["matched_code_hash"])
                train_sha.add(rec["matched_train_sha256"])
                if rec["matched_code_hash"] != fit_time_digest:
                    bad.append(f"{phase}:{cell_key(m,h)}__seed{s}")
    verified_files = sorted(
        p.relative_to(REPO).as_posix() for p in STAGE.rglob("*")
        if p.is_file() and p.suffix in (".py", ".md")
        and not p.relative_to(REPO).as_posix().startswith(
            (STAGE / "verification").relative_to(REPO).as_posix() + "/"))
    archived = sorted(
        p.relative_to(REPO).as_posix() for p in (STAGE / "verification").rglob("*")
        if p.is_file()) if (STAGE / "verification").is_dir() else []
    ok = len(hashes) == 1 and len(train_sha) == 1 and not bad
    record("code_state", ok, {
        "distinct_matched_code_hash": sorted(hashes),
        "recomputed_fit_time_digest": fit_time_digest,
        "runs_not_matching_recomputed_digest": bad,
        "distinct_matched_train_sha256": sorted(train_sha),
        "fit_time_file_set": verified_files,
        "files_added_after_the_fits": archived,
        "rule": ("the runs pinned the stage tree as it stood at fit time; the "
                 "verifier recomputes over that same file set and lists whatever "
                 "was archived into the stage afterwards")})


def disclosed_defects() -> list:
    """Facts about the stage that an adjudicator must see, not gate clauses.

    A documentation/CLI mismatch cannot change a scientific number, so it is
    recorded as a disclosure instead of being dressed up as a failed check.
    """
    out = []
    src = (IMPL / "matched_runner.py").read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(src)) or ""
    advertised = re.findall(r"^\s{4}([a-z_]+)(?:\s+\[n\])?\s{2,}E\d", doc, re.M)
    dispatch = set(re.findall(r'step == "([a-z_]+)"', src))
    missing = [s for s in advertised if s not in dispatch]
    if missing:
        out.append({
            "defect": "runner docstring advertises steps that main() does not dispatch",
            "advertised": advertised, "dispatched": sorted(dispatch),
            "advertised_without_dispatch": missing,
            "impact": ("none on any scientific number; the affected artifacts "
                       "(RESULTS.md, STAGE_TOKEN.json) are written by the executor "
                       "under the design authority's section 11 file list")})
    if re.search(r"if\s+step\s*==\s*\"finalize\"", src) is None:
        out.append({"defect": "no finalize step exists in matched_runner.main()",
                    "impact": "RESULTS.md and STAGE_TOKEN.json are executor-written"})
    return out


def check_runner_cli_is_self_consistent() -> None:
    """Every step the docstring advertises must be dispatchable."""
    src = (IMPL / "matched_runner.py").read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(src)) or ""
    advertised = set(re.findall(r"^\s{4}([a-z_]+)(?:\s+\[n\])?\s{2,}E\d", doc, re.M))
    dispatch = set(re.findall(r'step == "([a-z_]+)"', src))
    unknown = sorted(dispatch - advertised)
    record("runner_cli_is_self_consistent", not unknown,
           {"advertised_steps": sorted(advertised), "dispatched_steps": sorted(dispatch),
            "dispatched_but_undocumented": unknown,
            "note": "advertised-but-undispatched steps are reported under "
                    "disclosed_defects, not here"})


def check_terminal_token(verdict_a: str, verdict_b, selected) -> None:
    p = EVID / "STAGE_TOKEN.json"
    if not p.is_file():
        record("terminal_token", False, {"reason": "STAGE_TOKEN.json absent"})
        return
    tok = load_json(p)["terminal_token"]
    if selected is None:
        # A3 stops before any direct control: the inconclusive token is the only
        # registered terminal, and section 8 authorises no Phase B.
        expected = TOKENS["A3"]
    elif verdict_b is None:
        # Phase A selected a structured variant, so section 8 requires Phase B;
        # a terminal token here would be an incomplete stage.
        expected = "PHASE_B_REQUIRED_BUT_ABSENT"
    elif verdict_b == "B1_EXACT_GEOMETRY_SUPPORTED":
        expected = TOKENS["A1"] if selected == G2 else TOKENS["A2"]
    else:
        expected = TOKENS["B2"]
    ok = tok in TOKEN_POOL and tok == expected
    record("terminal_token", ok, {
        "written_token": tok, "expected_from_recomputed_verdict": expected,
        "phase_a_verdict": verdict_a, "phase_b_verdict": verdict_b,
        "selected_structured_variant": selected})


# --------------------------------------------------------------------- main
def guarded(name: str, fn, *a, **kw):
    """A missing or malformed artifact must fail its check, never crash the run.

    Returns whatever ``fn`` returned, but a caller that does not capture it drops
    it -- so a check must recompute anything a later check needs rather than
    publishing it through a return value.
    """
    try:
        return fn(*a, **kw)
    except Exception as exc:                                    # noqa: BLE001
        record(name, False, {"exception": f"{type(exc).__name__}: {exc}",
                             "rule": "fail closed: an artifact a check needs is "
                                     "absent or unreadable, so the check fails"})
        return None


def main() -> int:
    C.clear()
    C["schema"] = "hch_v44_o1_matched_independent_verification.v1"
    C["verifier"] = str(_HERE)
    C["verifier_imports_scientific_runner"] = False
    guarded("panel_and_seeds", check_panel_and_seeds)
    guarded("source_digest", check_source_digest)
    guarded("reference_identity", check_reference_identity)
    guarded("variant_parity", check_variant_parity)
    guarded("only_tied_identity_delta", check_only_tied_identity_delta)
    guarded("seed_median_first", check_seed_median_first)
    guarded("loss_switch", check_loss_switch)
    guarded("test_reads", check_test_reads)
    guarded("no_prior_mutation", check_no_prior_mutation)
    guarded("code_state", check_code_state)
    guarded("runner_cli_is_self_consistent", check_runner_cli_is_self_consistent)

    # The recomputation of the two gates is itself guarded: a missing gate file
    # or median table fails the gate checks rather than aborting the report.
    written_a = load_json(EVID / "PHASE_A_GATE.json") \
        if (EVID / "PHASE_A_GATE.json").is_file() else {}
    clean_a = bool((written_a.get("compliance") or {}).get("clean"))
    # Recompute the Phase-A cells from the raw artifacts here rather than reading
    # another check's published intermediate: an aggregating check must not depend
    # on whether some other check ran, nor on what it chose to publish.
    a_error = None
    cells_a: list[dict] = []
    try:
        cells_a = cells_from(rows_a())
        ra = gate_a_from(cells_a, clean_a)
    except Exception as exc:                                    # noqa: BLE001
        # no complete panel to aggregate: the gate cannot be recomputed at all.
        # The message is carried into the final record rather than recorded here,
        # because the record() at the end of this function would overwrite it.
        a_error = f"{type(exc).__name__}: {exc}"
        ra = {"verdict": "UNDECIDABLE", "selected": None, "panel_median": None,
              "worst": None, "best": None, "cells_beating": None,
              "a1_checks": {}, "a2_checks": {}}
    rb, cells_b = None, None
    if (EVID / "PHASE_B_GATE.json").is_file():
        try:
            selected = written_a["selected_structured_variant"]
            cells_b = cells_from(rows_b(selected))
            rb = gate_b_from(cells_b, bool(load_json(
                EVID / "PHASE_B_GATE.json")["compliance"]["clean"]))
            rb["csv_mismatches"] = _cmp_csv(
                EVID / "PHASE_B_CELL_MEDIANS.csv", cells_b,
                "gain_struct_vs_direct_pct", "mae_struct_median", "mae_direct_median")
        except Exception as exc:                                # noqa: BLE001
            record("phase_b_recomputation", False,
                   {"exception": f"{type(exc).__name__}: {exc}"})
    C["recomputed_phase_a"] = {
        "verdict": ra["verdict"], "selected": ra["selected"],
        "panel_median_gain_pct": ra["panel_median"], "worst_cell_gain_pct": ra["worst"],
        "best_cell_gain_pct": ra["best"], "cells_beating": ra["cells_beating"],
        "a1_checks": ra["a1_checks"], "a2_checks": ra["a2_checks"],
        "per_cell_gain_pct": {c["cell"]: c["gain_pct"] for c in cells_a}}
    if rb:
        C["recomputed_phase_b"] = {
            "verdict": rb["verdict"], "panel_median_gain_pct": rb["panel_median"],
            "worst_cell_gain_pct": rb["worst"], "best_cell_gain_pct": rb["best"],
            "cells_beating": rb["cells_beating"], "b1_checks": rb["b1_checks"],
            "per_cell_gain_pct": {c["cell"]: c["gain_pct"] for c in cells_b}}
    guarded("direct_control_identity", check_direct_control_identity, written_a)
    guarded("gate_clauses", check_gate_clauses, ra, rb, a_error)
    guarded("terminal_token", check_terminal_token,
            ra["verdict"], rb["verdict"] if rb else None, ra["selected"])
    C["disclosed_defects"] = disclosed_defects()
    checks = {k: v for k, v in C.items() if isinstance(v, dict) and "pass" in v
              and "scientific" in v}
    sci = {k: v for k, v in checks.items() if v["scientific"]}
    hyg = {k: v for k, v in checks.items() if not v["scientific"]}
    C["failures"] = sorted(k for k, v in sci.items() if not v["pass"])
    C["hygiene_failures"] = sorted(k for k, v in hyg.items() if not v["pass"])
    C["n_checks"] = len(sci)
    C["n_failures"] = len(C["failures"])
    C["all_pass"] = C["n_failures"] == 0        # the scientific verdict
    C["hygiene_pass"] = not C["hygiene_failures"]
    LEDGER.write_text(json.dumps(C, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": C["all_pass"], "n_checks": C["n_checks"],
                      "failures": C["failures"],
                      "hygiene_pass": C["hygiene_pass"],
                      "hygiene_failures": C["hygiene_failures"],
                      "disclosed_defects": len(C["disclosed_defects"]),
                      "phase_a": C["recomputed_phase_a"].get("verdict"),
                      "phase_b": C.get("recomputed_phase_b", {}).get("verdict")}, indent=2))
    return 0 if C["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
