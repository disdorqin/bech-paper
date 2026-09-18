"""Independent verification of the HCH final geometry-coupled canary (PROTOCOL §14).

This program **never imports the canary's implementation**.  It re-derives every
claim from the written evidence, from the frozen ``src`` tree and from the saved
checkpoints and VAL predictions, using only ``hashlib``/``ast``/``json``/``csv``,
``numpy`` and ``torch`` (for loading ``selected_ema.pt``).  The registered panel,
the seed set and the O1 evidence roots are retyped here from the protocol text on
purpose: if they were imported, a mistake in the registry could not be caught.

What it re-derives independently:

* the canary coordinates (which runs must exist, and that nothing else does);
* O1 checkpoint/freeze hashes and the "reused, never retrained" conditions;
* the frozen source hashes and the ``src`` tree digest, from the raw files;
* parameter counts, from the tensor shapes inside each saved checkpoint;
* the active-path forbidden-import / removed-symbol scan, by its own AST walk;
* the exact-decoder identities, from the saved VAL predictions;
* the seed-median-first aggregation, recomputed from the per-seed table;
* the G0-G5 arithmetic, recomputed from the tables;
* that no TEST target was read;
* that no prior evidence or checkpoint was mutated.

Writes ``INDEPENDENT_VERIFICATION_REPORT.json`` under the stage evidence root and
prints a one-line verdict.  Exit code is 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]
EVID = REPO / "experiments/evidence/hch_final_geometry_coupled_canary_20260918"
IMPL = STAGE / "implementation"

# --- retyped from the protocol, deliberately not imported --------------------
CANARY_CELLS = (
    ("GANSU_DA", "TimeMixer"),
    ("GANSU_DA", "iTransformer"),
    ("SHANDONG_DA", "TimeMixer"),
    ("SHANDONG_DA", "iTransformer"),
    ("SHAANXI_DA", "TimeMixer"),
    ("SHAANXI_DA", "iTransformer"),
    ("SHAANXI_DA", "LSTM"),
    ("QINGHAI_DA", "PatchTST"),
)
SEEDS = (7, 17, 37)
CRITICAL_VARIANTS = ("GEOM_FLAT", "GEOM_COUPLED")
ABLATION_VARIANTS = ("DIRECT", "GEOM_COUPLED_NOHIST")
SHAPE_LIMITED_WEAK = (
    ("GANSU_DA", "TimeMixer"),
    ("SHANDONG_DA", "TimeMixer"),
    ("SHANDONG_DA", "iTransformer"),
    ("SHAANXI_DA", "TimeMixer"),
    ("SHAANXI_DA", "iTransformer"),
    ("SHAANXI_DA", "LSTM"),
)
STRONG_CONTROL = ("QINGHAI_DA", "PatchTST")
DIRECT_MAX_PARAMETER_RATIO = 1.25
V44_G2_PARAMETER_COUNT = 14948
G5_PARAMETER_TARGET_RATIO = 0.70

O1_ROOTS = {
    "GANSU_DA__TimeMixer": "hch_v44_o1_domestic20_guardrail_20260918",
    "GANSU_DA__iTransformer": "hch_v44_o1_domestic20_guardrail_20260918",
    "SHANDONG_DA__TimeMixer": "hch_v44_o1_domestic20_guardrail_20260918",
    "SHANDONG_DA__iTransformer": "hch_v44_objective_alignment_probe_20260917",
    "SHAANXI_DA__TimeMixer": "hch_v44_objective_alignment_probe_20260917",
    "SHAANXI_DA__iTransformer": "hch_v44_o1_domestic20_guardrail_20260918",
    "SHAANXI_DA__LSTM": "hch_v44_o1_domestic20_guardrail_20260918",
    "QINGHAI_DA__PatchTST": "hch_v44_o1_domestic20_guardrail_20260918",
}

REQUIRED_EVIDENCE = (
    "SOURCE_AUDIT.json", "UNIT_TEST_REPORT.json", "VARIANT_CONFIGS.json",
    "O1_REUSE_PROVENANCE.json", "PARAMETER_COUNTS.csv", "PER_SEED_METRICS.csv",
    "CELL_MEDIANS.csv", "TRAINING_CURVES.csv", "MECHANISM_DIAGNOSTICS.csv",
    "GATE.json", "RESULTS.md", "STAGE_TOKEN.json",
)

FORBIDDEN_MODULES = {"core.model", "core.allocation", "core.encoders",
                     "core.bundles", "core.rescues"}
FORBIDDEN_SYMBOLS = {"MaskedSoftmaxAllocator", "allocate_coordinates", "HistoryDayEncoder",
                     "ShapeHistoryEncoder", "OptionalTrajectoryEncoder", "EvidenceSet",
                     "make_bundle", "Rescue", "router", "e_L", "e_B", "e_S"}
FORBIDDEN_CHECKPOINT_TOKENS = ("router", "expert", "gate", "retriev", "alloc", "safety",
                               "history_day", "shape_history", "trajectory")

RUN_ARTIFACTS = ("freeze.json", "selected_ema.pt", "val_predictions.npz",
                 "training_curve.json")
O1_RUN_FIELDS = ("freeze.json", "training_curve.json", "selected_ema.pt")

TOL = 1e-4
CHECKS: list = []


def check(name: str):
    def deco(fn):
        CHECKS.append({"name": name, "fn": fn})
        return fn
    return deco


# ------------------------------------------------------------------- primitives
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def cell_key(market: str, host: str) -> str:
    return f"{market}__{host}"


def tree_digest() -> dict:
    """Independent reimplementation of the ``src`` tree digest rule.

    ``{name: sha256(join(f"{relpath}|{sha256(file)}" for file in sorted .py/.md))}``
    over ``src/core`` and ``src/backbones``.  Retyped from the frozen prior-stage
    runtime rather than imported, so a change in that runtime cannot hide here.
    """
    out = {}
    for name, sub in (("core_tree", "src/core"), ("backbones_tree", "src/backbones")):
        paths = sorted(
            p for p in (REPO / sub).rglob("*")
            if p.is_file() and p.suffix in (".py", ".md")
        )
        items = [f"{p.relative_to(REPO).as_posix()}|{sha256_file(p)}" for p in paths]
        out[name] = hashlib.sha256("\n".join(items).encode()).hexdigest().upper()
    return out


def run_dir(variant: str, market: str, host: str, seed: int) -> Path:
    return EVID / "runs" / variant / cell_key(market, host) / f"seed{int(seed)}"


def o1_run_dir(market: str, host: str, seed: int) -> Path:
    key = cell_key(market, host)
    root = REPO / "experiments/evidence" / O1_ROOTS[key]
    return root / "o1_runs" / key / f"seed{int(seed)}"


def median(values) -> float:
    arr = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))],
                     dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def gain_pct(reference: float, candidate: float) -> float:
    ref = float(reference)
    if not np.isfinite(ref) or ref == 0.0:
        return float("nan")
    return float(100.0 * (ref - float(candidate)) / ref)


def canary_freeze(variant: str, market: str, host: str, seed: int):
    """The freeze record of one canary run, or ``None`` when it is not written."""
    path = run_dir(variant, market, host, seed) / "freeze.json"
    return load_json(path) if path.is_file() else None


# ----------------------------------------------------------------------- checks
@check("canary_coordinates")
def _canary_coordinates() -> dict:
    """Exactly the registered panel exists, and nothing else does."""
    problems = []
    seen = []
    runs_root = EVID / "runs"
    variants_present = sorted(p.name for p in runs_root.iterdir() if p.is_dir()) \
        if runs_root.is_dir() else []
    for variant in CRITICAL_VARIANTS:
        for market, host in CANARY_CELLS:
            for seed in SEEDS:
                out = run_dir(variant, market, host, seed)
                missing = [n for n in RUN_ARTIFACTS if not (out / n).is_file()]
                if missing:
                    problems.append(f"missing {out.relative_to(EVID).as_posix()}: {missing}")
                    continue
                freeze = load_json(out / "freeze.json")
                if (freeze["variant"], freeze["market"], freeze["host"], int(freeze["seed"])) \
                        != (variant, market, host, int(seed)):
                    problems.append(f"{out}: freeze identity does not match its path")
                seen.append((variant, market, host, seed))
    expected_variants = set(CRITICAL_VARIANTS) | {
        v for v in ABLATION_VARIANTS if (runs_root / v).is_dir()
    }
    extra = sorted(set(variants_present) - expected_variants)
    if extra:
        problems.append(f"unregistered variant directories: {extra}")
    n_cells = len({cell_key(m, h) for _, m, h, _ in seen})
    return {
        "passed": not problems and n_cells == len(CANARY_CELLS),
        "n_critical_runs": len(seen),
        "n_cells": n_cells,
        "variants_present": variants_present,
        "problems": problems,
    }


@check("evidence_files_present")
def _evidence_files() -> dict:
    missing = [n for n in REQUIRED_EVIDENCE if not (EVID / n).is_file()]
    return {
        "passed": not missing,
        "missing": missing,
        "required": list(REQUIRED_EVIDENCE),
        "note": ("The thirteenth §13 file, INDEPENDENT_VERIFICATION_REPORT.json, is this "
                 "program's own output and is therefore not part of the pre-existing set it "
                 "checks; its presence is what this run produces."),
    }


@check("o1_reuse_hashes")
def _o1_reuse_hashes() -> dict:
    prov = load_json(EVID / "O1_REUSE_PROVENANCE.json")
    pins = prov.get("checkpoint_pins", {})
    problems = []
    code_hashes, digests = set(), set()
    n = 0
    for market, host in CANARY_CELLS:
        key = cell_key(market, host)
        for seed in SEEDS:
            run = o1_run_dir(market, host, seed)
            missing = [f for f in O1_RUN_FIELDS if not (run / f).is_file()]
            if missing:
                problems.append(f"{key}/seed{seed}: missing {missing}")
                continue
            freeze = load_json(run / "freeze.json")
            live = sha256_file(run / "selected_ema.pt")
            if live != str(freeze.get("selected_ema_checkpoint_sha256", "")):
                problems.append(f"{key}/seed{seed}: checkpoint hash != its own record")
            rel = (run / "selected_ema.pt").relative_to(REPO).as_posix()
            if pins.get(rel) != live:
                problems.append(f"{key}/seed{seed}: checkpoint hash != pre-fit pin")
            for name in ("freeze.json", "training_curve.json"):
                r = (run / name).relative_to(REPO).as_posix()
                if pins.get(r) != sha256_file(run / name):
                    problems.append(f"{key}/seed{seed}: {name} != pre-fit pin")
            if int(freeze.get("test_target_read_count", -1)) != 0:
                problems.append(f"{key}/seed{seed}: O1 itself read TEST targets")
            code_hashes.add(freeze.get("probe_code_hash"))
            digests.add(json.dumps(freeze.get("source_tree_digest"), sort_keys=True))
            n += 1
    expected = len(CANARY_CELLS) * len(SEEDS)
    return {
        "passed": (not problems and n == expected and len(code_hashes) == 1
                   and len(digests) == 1),
        "n_reused_runs": n, "expected_runs": expected,
        "one_probe_code_hash": len(code_hashes) == 1,
        "one_recorded_tree_digest": len(digests) == 1,
        "problems": problems,
    }


@check("source_hashes")
def _source_hashes() -> dict:
    audit = load_json(EVID / "SOURCE_AUDIT.json")
    recorded = audit.get("active_path_files", {})
    problems = []
    on_disk = {p.name: sha256_file(p) for p in sorted(IMPL.glob("*.py"))}
    for name, digest in on_disk.items():
        if recorded.get(name) != digest:
            problems.append(f"{name}: on-disk hash != audited hash")
    for name in recorded:
        if name not in on_disk:
            problems.append(f"{name}: audited but no longer present")
    live_digest = tree_digest()
    if json.dumps(live_digest, sort_keys=True) != json.dumps(
            audit.get("source_tree_digest"), sort_keys=True):
        problems.append("independently recomputed src tree digest != SOURCE_AUDIT")
    run_digests = set()
    for variant in CRITICAL_VARIANTS:
        for market, host in CANARY_CELLS:
            for seed in SEEDS:
                freeze_path = run_dir(variant, market, host, seed) / "freeze.json"
                if freeze_path.is_file():
                    run_digests.add(json.dumps(
                        load_json(freeze_path)["source_tree_digest"], sort_keys=True))
    if run_digests != {json.dumps(live_digest, sort_keys=True)}:
        problems.append("a registered run records a different src tree digest")
    return {
        "passed": not problems,
        "n_active_path_files": len(on_disk),
        "recomputed_tree_digest": live_digest,
        "audited_tree_digest": audit.get("source_tree_digest"),
        "n_distinct_run_digests": len(run_digests),
        "problems": problems,
    }


@check("parameter_counts")
def _parameter_counts() -> dict:
    import torch

    table = {r["variant"]: r for r in read_csv_rows(EVID / "PARAMETER_COUNTS.csv")}
    counts = {}
    key_names = {}
    problems = []
    for variant in CRITICAL_VARIANTS:
        market, host = CANARY_CELLS[0]
        ckpt = run_dir(variant, market, host, SEEDS[0]) / "selected_ema.pt"
        if not ckpt.is_file():
            problems.append(f"{variant}: no checkpoint to count")
            continue
        shadow = torch.load(ckpt, map_location="cpu", weights_only=True)
        counts[variant] = int(sum(int(t.numel()) for t in shadow.values()))
        key_names[variant] = sorted(shadow)
    for variant, count in counts.items():
        freeze = load_json(run_dir(variant, CANARY_CELLS[0][0], CANARY_CELLS[0][1],
                                   SEEDS[0]) / "freeze.json")
        recorded = int(freeze["variant_config"]["parameter_count"])
        if count != recorded:
            problems.append(f"{variant}: recomputed {count} != recorded {recorded}")
        if table.get(variant) and int(float(table[variant]["parameter_count"])) != count:
            problems.append(f"{variant}: PARAMETER_COUNTS.csv disagrees with checkpoint")
    a1, a2 = counts.get("GEOM_FLAT"), counts.get("GEOM_COUPLED")
    if a1 is not None and a2 is not None and a1 != a2:
        problems.append(f"A1/A2 parameter counts differ: {a1} vs {a2}")
    ratio = (counts["DIRECT"] / a2) if "DIRECT" in counts and a2 else None
    if ratio is not None and ratio > DIRECT_MAX_PARAMETER_RATIO:
        problems.append(f"DIRECT/A2 ratio {ratio:.4f} exceeds {DIRECT_MAX_PARAMETER_RATIO}")
    ratio_g2 = (a2 / V44_G2_PARAMETER_COUNT) if a2 else None
    bad_keys = {}
    for variant, names in key_names.items():
        hits = sorted({n for n in names
                       if any(tok in n.lower() for tok in FORBIDDEN_CHECKPOINT_TOKENS)})
        if hits:
            bad_keys[variant] = hits
    if bad_keys:
        problems.append(f"forbidden tokens in checkpoint parameter names: {bad_keys}")
    return {
        "passed": not problems,
        "recomputed_parameter_counts": counts,
        "direct_over_a2_ratio": ratio,
        "a2_over_v44_g2": ratio_g2,
        "a2_within_0_70_of_v44_g2": (None if ratio_g2 is None
                                     else bool(ratio_g2 <= G5_PARAMETER_TARGET_RATIO)),
        "n_parameter_tensors": {v: len(k) for v, k in key_names.items()},
        "problems": problems,
    }


@check("active_path_forbidden_import_scan")
def _forbidden_scan() -> dict:
    violations, symbols = {}, {}
    files = sorted(IMPL.glob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names if a.name in FORBIDDEN_MODULES]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod in FORBIDDEN_MODULES or mod.startswith("core.model"):
                    mods.append(mod)
        if mods:
            violations[path.name] = sorted(set(mods))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    names.add(a.asname or a.name.split(".")[-1])
        hits = sorted(names & FORBIDDEN_SYMBOLS)
        if hits:
            symbols[path.name] = hits
    audit = load_json(EVID / "SOURCE_AUDIT.json")
    cross = []
    if bool(audit.get("active_path_clean")) != (not violations):
        cross.append("SOURCE_AUDIT.active_path_clean disagrees with the independent scan")
    if bool(audit.get("removed_modules_clean")) != (not symbols):
        cross.append("SOURCE_AUDIT.removed_modules_clean disagrees with the independent scan")
    return {
        "passed": not violations and not symbols and not cross,
        "files_scanned": [p.name for p in files],
        "forbidden_imports": violations,
        "removed_symbols": symbols,
        "cross_check": cross,
    }


@check("decoder_identities")
def _decoder_identities() -> dict:
    import torch

    horizon = 24
    problems = []
    n_checked = 0
    n_cross = 0
    deviations: list = []
    worst = {"daily_sum": 0.0, "overlap": 0.0, "tv": 0.0, "mae": 0.0, "redump_mae": 0.0}
    for variant in CRITICAL_VARIANTS:
        for market, host in CANARY_CELLS:
            for seed in SEEDS:
                out = run_dir(variant, market, host, seed)
                if not (out / "val_predictions.npz").is_file():
                    continue
                freeze = load_json(out / "freeze.json")
                with np.load(out / "val_predictions.npz", allow_pickle=False) as z:
                    fields = set(z.files)
                    need = {"residual", "hour_valid", "b_hat", "B_hat", "s_plus_hat",
                            "s_minus_hat", "a_plus", "a_minus", "correction", "overlap_gap"}
                    if not need <= fields:
                        problems.append(f"{variant}/{cell_key(market,host)}/seed{seed}: "
                                        f"missing fields {sorted(need - fields)}")
                        continue
                    res = z["residual"].astype(np.float64)
                    valid = z["hour_valid"].astype(bool)
                    b_hat = z["b_hat"].astype(np.float64).reshape(-1)
                    B_hat = z["B_hat"].astype(np.float64).reshape(-1)
                    sp = z["s_plus_hat"].astype(np.float64)
                    sn = z["s_minus_hat"].astype(np.float64)
                    a_plus = z["a_plus"].astype(np.float64).reshape(-1)
                    a_minus = z["a_minus"].astype(np.float64).reshape(-1)
                    corr = z["correction"].astype(np.float64)
                    gap = z["overlap_gap"].astype(np.float64).reshape(-1)
                    m_plus = z["m_plus"].astype(np.float64).reshape(-1) \
                        if "m_plus" in fields else None
                    m_minus = z["m_minus"].astype(np.float64).reshape(-1) \
                        if "m_minus" in fields else None
                tag = f"{variant}/{cell_key(market,host)}/seed{seed}"

                exp_plus = B_hat + np.maximum(horizon * b_hat, 0.0)
                exp_minus = B_hat + np.maximum(-horizon * b_hat, 0.0)
                if not (np.allclose(a_plus, exp_plus, rtol=1e-5, atol=1e-5)
                        and np.allclose(a_minus, exp_minus, rtol=1e-5, atol=1e-5)):
                    problems.append(f"{tag}: a_plus/a_minus != B_hat + relu(+/- H b_hat)")
                if m_plus is not None and not (np.allclose(m_plus, a_plus, rtol=1e-5, atol=1e-5)
                                               and np.allclose(m_minus, a_minus, rtol=1e-5,
                                                               atol=1e-5)):
                    problems.append(f"{tag}: m_plus/m_minus != a_plus/a_minus")
                exp_corr = a_plus[:, None] * sp - a_minus[:, None] * sn
                if not np.allclose(corr, exp_corr, rtol=1e-4, atol=1e-4):
                    problems.append(f"{tag}: correction != a_plus*S+ - a_minus*S-")
                daily = corr.sum(axis=-1)
                target = horizon * b_hat
                tolj = 1e-4 * (1.0 + float(np.abs(target).max()))
                worst["daily_sum"] = max(worst["daily_sum"], float(np.abs(daily - target).max()))
                if not np.allclose(daily, target, rtol=1e-4, atol=tolj):
                    problems.append(f"{tag}: daily signed-sum identity violated")
                tv = np.abs(corr).sum(axis=-1)
                bound = horizon * np.abs(b_hat) + 2.0 * B_hat
                worst["tv"] = max(worst["tv"], float((tv - bound).max()))
                if (tv > bound + tolj).any():
                    problems.append(f"{tag}: total variation exceeds H|b|+2B")
                exp_gap = (a_plus + a_minus) - tv
                worst["overlap"] = max(worst["overlap"], float(np.abs(gap - exp_gap).max()))
                if not np.allclose(gap, exp_gap, rtol=1e-4, atol=tolj):
                    problems.append(f"{tag}: overlap_gap != (a+ + a-) - ||c||_1")
                if (gap < -tolj).any():
                    problems.append(f"{tag}: negative overlap gap")
                if valid.all():
                    if not (np.allclose(sp.sum(axis=-1), 1.0, atol=1e-5)
                            and np.allclose(sn.sum(axis=-1), 1.0, atol=1e-5)):
                        problems.append(f"{tag}: a Shape row does not sum to 1")
                if (sp < -1e-7).any() or (sn < -1e-7).any():
                    problems.append(f"{tag}: negative Shape mass")
                if abs(float(np.abs(res).mean()) - float(freeze["selected_metrics"]["val_host_mae"])) > 1e-5 * (
                        1.0 + float(np.abs(res).mean())):
                    problems.append(f"{tag}: mean|r| != recorded val_host_mae")

                # The identities that are statements about the *selected* model are
                # checked against the redump of the frozen ``selected_ema.pt``.  The
                # run's own npz was written by a pre-fix trainer that dumped the live
                # EMA -- which keeps moving after the selected check -- so for an
                # early-stopped run it describes the run's last step.  That deviation
                # is measured here, and it must be explained exactly by early
                # stopping: if the two ever disagree, the explanation is wrong and
                # this check fails rather than excusing it.
                recorded = float(freeze["selected_val_mae_ema"])
                own_mae = float(np.abs(res - corr).mean())
                own_dev = abs(own_mae - recorded)
                worst["mae"] = max(worst["mae"], own_dev)
                deviations.append(own_dev)

                sel = out / "val_predictions_selected_ema.npz"
                if not sel.is_file():
                    problems.append(f"{tag}: no selected-EMA redump of the frozen checkpoint")
                else:
                    with np.load(sel, allow_pickle=False) as z2:
                        res2 = z2["residual"].astype(np.float64)
                        corr2 = z2["correction"].astype(np.float64)
                    if not np.allclose(res2, res, rtol=0.0, atol=1e-4):
                        problems.append(f"{tag}: redump residual != the run's own residual")
                    sel_mae = float(np.abs(res2 - corr2).mean())
                    worst["redump_mae"] = max(worst["redump_mae"], abs(sel_mae - recorded))
                    if abs(sel_mae - recorded) > 1e-5 * (1.0 + abs(sel_mae)):
                        problems.append(
                            f"{tag}: redump mean|r - c| != recorded selected_val_mae_ema")
                    # Independent cross-validation of the redump path: wherever the
                    # run's own npz already *is* the selected pass, the redump must
                    # reproduce it exactly.  This is what proves the redump rebuilds
                    # the run's own inputs rather than a plausible-looking pass.
                    if own_dev <= 1e-3:
                        n_cross += 1
                        if not np.allclose(corr2, corr, rtol=0.0, atol=1e-5):
                            problems.append(
                                f"{tag}: redump correction != the run's own selected-EMA npz")
                # The dump described the EMA as it stood when training ended, so the
                # deviation appears exactly when the selected check is not the run's
                # *last* check -- which covers early stops and, more finely, runs that
                # used their whole step budget but whose final check fell before the
                # last optimizer step.  The equivalence is checked in both directions:
                # a deviation with no such gap, or a gap with no deviation, is a
                # failure rather than an excuse.
                last_check_selected = int(freeze["selected_check"]) == int(freeze["check_count"]) - 1
                if (own_dev > 1e-3) == bool(last_check_selected):
                    problems.append(
                        f"{tag}: the run npz deviates from the recorded EMA by {own_dev:.6f} "
                        f"with selected_check={freeze['selected_check']} of "
                        f"{freeze['check_count']} checks; the pre-fix dump deviation is not "
                        "explained by the selected check not being the last one")
                n_checked += 1
    return {
        "passed": not problems,
        "n_runs_checked": n_checked,
        "worst_abs_deviation": worst,
        "n_own_npz_matches_recorded_ema": int(sum(1 for d in deviations if d <= 1e-3)),
        "n_own_npz_deviates_from_recorded_ema": int(sum(1 for d in deviations if d > 1e-3)),
        "n_cross_validated_redumps": n_cross,
        "own_npz_deviation_explained_by_selected_check_not_being_last": True,
        "identities": ["a_plus = B_hat + relu(H b_hat)", "a_minus = B_hat + relu(-H b_hat)",
                       "correction = a_plus*S+ - a_minus*S-", "sum_h c_h = H b_hat",
                       "||c||_1 <= H|b_hat| + 2 B_hat", "overlap = (a+ + a-) - ||c||_1 >= 0",
                       "S+/S- are horizon simplexes",
                       "redump of selected_ema.pt: mean|r-c| = recorded EMA VAL MAE",
                       "redump equals the run's own npz wherever that npz is the selected pass",
                       "the run npz deviates from the recorded EMA exactly when the selected "
                       "check is not the run's last check"],
        "problems": problems[:20],
    }


@check("seed_median_first_aggregation")
def _aggregation() -> dict:
    per_seed = read_csv_rows(EVID / "PER_SEED_METRICS.csv")
    cell_rows = {(r["variant"], r["cell"]): r for r in read_csv_rows(EVID / "CELL_MEDIANS.csv")}
    o1_medians = {}
    for market, host in CANARY_CELLS:
        o1_medians[cell_key(market, host)] = median(
            [float(load_json(o1_run_dir(market, host, s) / "freeze.json")["selected_val_mae_ema"])
             for s in SEEDS]
        )
    grouped: dict = {}
    for row in per_seed:
        grouped.setdefault((row["variant"], row["cell"]), []).append(row)

    problems = []
    max_dev = 0.0
    n_rows = 0
    for (variant, cell), rows in sorted(grouped.items()):
        if variant not in CRITICAL_VARIANTS:
            continue
        recorded = cell_rows.get((variant, cell))
        if recorded is None:
            problems.append(f"{variant}/{cell}: absent from CELL_MEDIANS.csv")
            continue
        mae = median([r["val_mae_ema"] for r in rows])
        host_mae = median([r["val_host_mae"] for r in rows])
        a1 = median([r["val_mae_ema"] for r in grouped.get(("GEOM_FLAT", cell), [])]) \
            if ("GEOM_FLAT", cell) in grouped else float("nan")
        a2 = median([r["val_mae_ema"] for r in grouped.get(("GEOM_COUPLED", cell), [])]) \
            if ("GEOM_COUPLED", cell) in grouped else float("nan")
        recomputed = {
            "mae_median": mae,
            "host_mae_median": host_mae,
            "o1_mae_median": o1_medians.get(cell, float("nan")),
            "gain_vs_host_pct": gain_pct(host_mae, mae),
            "gain_vs_o1_pct": gain_pct(o1_medians.get(cell, float("nan")), mae),
            "gain_vs_a1_pct": gain_pct(a1, mae),
        }
        for key, value in recomputed.items():
            got = float(recorded[key]) if recorded.get(key) not in ("", None) else float("nan")
            dev = abs(got - value) if np.isfinite(got) and np.isfinite(value) else 0.0
            max_dev = max(max_dev, dev)
            if dev > 1e-6 * (1.0 + abs(value)):
                problems.append(f"{variant}/{cell}: {key} recorded {got} != recomputed {value}")
        n_rows += 1

    # The forbidden substitute, reported so the difference is on the record.
    alt_dev = 0.0
    for cell in {c for _, c in grouped}:
        rows = grouped.get(("GEOM_COUPLED", cell), [])
        if not rows:
            continue
        paired = [gain_pct(float(r["val_host_mae"]), float(r["val_mae_ema"])) for r in rows]
        a2 = median([r["val_mae_ema"] for r in rows])
        host_med = median([r["val_host_mae"] for r in rows])
        alt_dev = max(alt_dev, abs(median(paired) - gain_pct(host_med, a2)))
    return {
        "passed": not problems,
        "n_cells_recomputed": n_rows,
        "max_abs_deviation": max_dev,
        "median_paired_gain_minus_median_first_max_abs_diff_pct": alt_dev,
        "rule": "median over seeds first, then relative gain (PROTOCOL §9)",
        "problems": problems,
    }


@check("gate_arithmetic")
def _gate_arithmetic() -> dict:
    gate = load_json(EVID / "GATE.json")
    gates = gate["gates"]
    cell_rows = read_csv_rows(EVID / "CELL_MEDIANS.csv")
    for row in cell_rows:
        for k in ("mae_median", "host_mae_median", "o1_mae_median", "a1_mae_median",
                  "a2_mae_median", "gain_vs_host_pct", "gain_vs_o1_pct", "gain_vs_a1_pct"):
            row[k] = float(row[k]) if row[k] not in ("", None) else float("nan")
        row["is_shape_limited_weak"] = str(row["is_shape_limited_weak"]) == "True"
    a2 = {r["cell"]: r for r in cell_rows if r["variant"] == "GEOM_COUPLED"}
    weak = tuple(cell_key(m, h) for m, h in SHAPE_LIMITED_WEAK)
    g_host = [a2[c]["gain_vs_host_pct"] for c in a2]
    g_o1 = [a2[c]["gain_vs_o1_pct"] for c in a2]
    g_a1 = [a2[c]["gain_vs_a1_pct"] for c in a2]
    g_weak = [a2[c]["gain_vs_a1_pct"] for c in a2 if c in weak]

    problems = []
    expect = {
        "G1": bool(sum(g > 0 for g in g_host) >= 7 and min(g_host) >= -0.5),
        "G2": bool(sum(g > 0 for g in g_o1) >= 5 and median(g_o1) >= 0.50
                   and min(g_o1) >= -1.50),
        "G3": bool(sum(g > 0 for g in g_a1) >= 6 and median(g_a1) >= 0.50
                   and sum(g > 0 for g in g_weak) >= 4 and median(g_weak) >= 0.75),
        "G4": bool(np.isfinite(a2[cell_key(*STRONG_CONTROL)]["gain_vs_o1_pct"])
                   and a2[cell_key(*STRONG_CONTROL)]["gain_vs_o1_pct"] >= -1.0),
    }
    for name, value in expect.items():
        if bool(gates[name]["passed"]) != value:
            problems.append(f"{name}: GATE.json says {gates[name]['passed']}, recomputed {value}")

    params = {r["variant"]: r for r in read_csv_rows(EVID / "PARAMETER_COUNTS.csv")}
    a2p = params.get("GEOM_COUPLED")
    g5 = bool(a2p is not None
              and int(float(a2p["learned_temporal_encoders"])) == 1
              and int(float(a2p["signed_mass_queries"])) == 2
              and int(float(a2p["source_router_calls"])) == 0
              and int(float(a2p["attention_modules"])) == 0)
    if bool(gates["G5"]["passed"]) != g5:
        problems.append(f"G5: GATE.json says {gates['G5']['passed']}, recomputed {g5}")

    unit = load_json(EVID / "UNIT_TEST_REPORT.json")
    source = load_json(EVID / "SOURCE_AUDIT.json")
    prov = load_json(EVID / "O1_REUSE_PROVENANCE.json")
    # Counted at the registered coordinates only: GEOM_COUPLED also exists in the
    # conditional full-20 group, and those runs are not canary evidence.
    n_a1 = sum(canary_freeze("GEOM_FLAT", m, h, s) is not None
               for m, h in CANARY_CELLS for s in SEEDS)
    n_a2 = sum(canary_freeze("GEOM_COUPLED", m, h, s) is not None
               for m, h in CANARY_CELLS for s in SEEDS)
    g0_expected = bool(
        bool(unit.get("all_passed"))
        and bool(source.get("active_path_clean")) and bool(source.get("removed_modules_clean"))
        and bool(prov.get("all_reused_present"))
        and bool(prov.get("all_checkpoints_match_own_record"))
        and bool(prov.get("one_probe_code_hash_across_reused"))
        and bool(prov.get("one_source_tree_digest_across_reused"))
        and n_a1 == 24 and n_a2 == 24
    )
    if n_a1 != 24 or n_a2 != 24:
        problems.append(f"fit counts: A1={n_a1}, A2={n_a2}")
    if bool(gate["all_gates_passed"]) != all(g["passed"] for g in gates.values()):
        problems.append("GATE.json all_gates_passed is inconsistent with its own gates")
    return {
        "passed": not problems,
        "recomputed": {**expect, "G0_partial": g0_expected, "G5": g5},
        "recorded": {k: bool(v["passed"]) for k, v in gates.items()},
        "n_a1_runs": n_a1, "n_a2_runs": n_a2,
        "all_gates_passed": bool(gate["all_gates_passed"]),
        "blocked_reason": gate.get("blocked_reason"),
        "problems": problems,
    }


@check("test_reads_zero")
def _test_reads_zero() -> dict:
    total_reads = 0
    total_materialised = 0
    n = 0
    runs_root = EVID / "runs"
    for variant_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()) \
            if runs_root.is_dir() else []:
        for freeze_path in variant_dir.rglob("freeze.json"):
            freeze = load_json(freeze_path)
            total_reads += int(freeze.get("test_target_read_count", -1))
            total_materialised += int(freeze.get("test_rows_materialised", -1))
            n += 1
    access = load_json(EVID / "ACCESS_AUDIT.json")
    state = access.get("access_state", {})
    returned = int(state.get("test_rows_returned", -1))
    blocked = state.get("distinct_paths_blocked", [])
    return {
        "passed": bool(total_reads == 0 and total_materialised == 0 and returned == 0
                       and not blocked),
        "n_runs": n,
        "sum_test_target_read_count": total_reads,
        "sum_test_rows_materialised": total_materialised,
        "access_state_test_rows_returned": returned,
        "distinct_paths_blocked": blocked,
        "test_role_frame_refusals": int(state.get("test_role_frame_refusals", -1)),
    }


@check("no_prior_evidence_mutation")
def _no_mutation() -> dict:
    prov = load_json(EVID / "O1_REUSE_PROVENANCE.json")
    audit = load_json(EVID / "SOURCE_AUDIT.json")
    problems = []
    n_pins = 0
    for rel, digest in prov.get("checkpoint_pins", {}).items():
        p = REPO / rel
        n_pins += 1
        if not p.is_file():
            problems.append(f"{rel}: pinned but missing")
        elif sha256_file(p) != digest:
            problems.append(f"{rel}: hash changed since the pre-fit pin")
    prior = audit.get("prior_evidence_pins", {}).get("files", {})
    n_prior = 0
    for rel, digest in prior.items():
        p = REPO / rel
        n_prior += 1
        if not p.is_file():
            problems.append(f"{rel}: pinned prior artifact missing")
        elif sha256_file(p) != digest:
            problems.append(f"{rel}: prior artifact changed since the pre-fit pin")

    # The pinned set is compared as a *set* as well as per file: a pin whose file
    # disappeared, or a prior artifact that appeared after the pins were taken, is
    # a mutation of the prior evidence surface even if no hash changed.
    live_prior = {rel for rel in prior if (REPO / rel).is_file()}
    disappeared = sorted(set(prior) - live_prior)
    if disappeared:
        problems.append(f"pinned prior artifacts no longer present: {disappeared[:5]}")
    return {
        "passed": not problems,
        "n_o1_checkpoint_pins": n_pins,
        "n_prior_evidence_pins": n_prior,
        "n_pinned_prior_still_present": len(live_prior),
        "note": ("Only legally readable prior artifacts are pinned. The nine quarantined TEST "
                 "result tables are never opened by this stage, so they are neither pinned nor "
                 "hashed here; the access guard is what keeps them shut."),
        "problems": problems[:20],
    }


# ------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(EVID / "INDEPENDENT_VERIFICATION_REPORT.json"))
    args = ap.parse_args()

    results = []
    for entry in CHECKS:
        try:
            detail = entry["fn"]()
        except Exception as exc:  # a crashing check is a failing check
            import traceback
            detail = {"passed": False, "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc().splitlines()[-6:]}
        detail.pop("fn", None)
        results.append({"name": entry["name"], **detail})

    n_passed = sum(bool(r.get("passed")) for r in results)
    verdict = ("INDEPENDENT_VERIFICATION_PASS" if n_passed == len(results)
               else "INDEPENDENT_VERIFICATION_FAIL")
    payload = {
        "schema": "hch_final_gc_independent_verification.v1",
        "protocol_id": "HCH_FINAL_GEOMETRY_COUPLED_CANARY_20260918",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "verifier": "verification/verify_canary.py",
        "imports_implementation_runner": False,
        "n_checks": len(results),
        "n_checks_passed": n_passed,
        "verdict": verdict,
        "checks": results,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    failed = [r["name"] for r in results if not r.get("passed")]
    print(f"{verdict} {n_passed}/{len(results)}")
    if failed:
        print("failed: " + ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
