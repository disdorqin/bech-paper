"""Phase driver for the HCH final geometry-coupled canary.

Sub-commands, in the order a run uses them:

    source     source/access audit of the frozen substrate       -> SOURCE_AUDIT.json
    reuse      hash-pinned snapshot of the reused O1 evidence    -> O1_REUSE_PROVENANCE.json
    tests      the PROTOCOL §8 correctness gate (subprocess)     -> UNIT_TEST_REPORT.json
    fits       the registered fits, skip-if-artifacts-exist      -> runs/**
    aggregate  seed-median-first tables                          -> PER_SEED_METRICS.csv, ...
    access     the access audit                                  -> ACCESS_AUDIT.json
    gate       the G0-G5 promotion gate                          -> GATE.json
    token      exactly one terminal token                        -> STAGE_TOKEN.json
    report     the evidence report                               -> RESULTS.md
    verify     the independent verifier (subprocess)             -> INDEPENDENT_VERIFICATION_REPORT.json
    all        the above, in that order

Two properties the driver must not lose:

* **no overwrite.**  A registered run is skipped when its artifacts are complete.
  A partially written run directory is an error, never a reason to refit: there is
  no ``--force`` and no re-run with modified settings anywhere in this file.
* **fail closed.**  Anything ambiguous (a missing artifact the gate needs, a
  parameter count that disagrees with the freeze record) raises rather than
  being defaulted.

There is no TEST reader and no TEST scorer in this module.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import gc_common as GC  # noqa: E402
import gc_data as GD  # noqa: E402

EVID = GC.EVID
RUNS = GC.RUNS
VERIFY_DIR = STAGE / "verification"
PYTHON = sys.executable

CRITICAL_FITS = len(GC.CRITICAL_VARIANTS) * len(GC.CANARY_CELLS) * len(GC.SEEDS)  # 48
ABLATION_FITS = len(GC.ABLATION_VARIANTS) * len(GC.CANARY_CELLS) * len(GC.SEEDS)  # 48
FULL20_FITS = len(GC.FULL20_MISSING_CELLS) * len(GC.SEEDS)  # 36

RUN_ARTIFACTS = ("freeze.json", "selected_ema.pt", "val_predictions.npz", "training_curve.json")

#: Frozen v4.4 G2 reference parameter count.  The final method's efficiency target
#: is descriptive (G5 is a validity/reporting gate, not a threshold).
V44_G2_PARAMETER_COUNT = 14948
G5_PARAMETER_TARGET_RATIO = 0.70

LATENCY_BATCH = 256
LATENCY_REPEATS = 30
LATENCY_WARMUP = 5

#: The Shape-limited weak cells and the strong control, as gate sets.
WEAK_CELLS = tuple(GC.cell_key(m, h) for m, h in GC.SHAPE_LIMITED_WEAK)
STRONG_CELL = GC.cell_key(*GC.STRONG_CONTROL)

DOC = {
    "PER_SEED_METRICS.csv": "one row per registered fit; VAL metrics of the selected EMA checkpoint",
    "CELL_MEDIANS.csv": "one row per (variant, cell); median-over-seeds first, then gains (PROTOCOL §9)",
    "PARAMETER_COUNTS.csv": "one row per variant: parameter count/map, structural encoder/query census, latency",
    "TRAINING_CURVES.csv": "one row per (fit, validation check) as recorded by the trainer",
    "MECHANISM_DIAGNOSTICS.csv": "one row per registered fit: exact-decoder mechanism quantities on VAL",
}


# --------------------------------------------------------------------- plumbing
def _prepare() -> None:
    GC.RC.install_access_guard()
    GC.init_worker()


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def universe(group: str) -> list:
    """The registered fit universe of one group.  No other universe may be run."""
    if group == "critical":
        return [
            (v, m, h, s)
            for v in GC.CRITICAL_VARIANTS for (m, h) in GC.CANARY_CELLS for s in GC.SEEDS
        ]
    if group == "ablation":
        return [
            (v, m, h, s)
            for v in GC.ABLATION_VARIANTS for (m, h) in GC.CANARY_CELLS for s in GC.SEEDS
        ]
    if group == "full20":
        return [
            (GC.GEOM_COUPLED, m, h, s)
            for (m, h) in GC.FULL20_MISSING_CELLS for s in GC.SEEDS
        ]
    raise KeyError(f"unknown fit group {group!r}")


def _run_state(out: Path) -> str:
    present = [n for n in RUN_ARTIFACTS if (out / n).is_file()]
    if not present:
        return "absent"
    if len(present) == len(RUN_ARTIFACTS):
        return "complete"
    return f"partial:{sorted(set(RUN_ARTIFACTS) - set(present))}"


def load_runs(group: str) -> list:
    """Freeze records of every *complete* run of a group, in registered order."""
    rows = []
    for variant, market, host, seed in universe(group):
        out = GC.run_dir(variant, market, host, seed)
        if _run_state(out) != "complete":
            continue
        rows.append({
            "group": group, "variant": variant, "market": market, "host": host,
            "cell": GC.cell_key(market, host), "seed": int(seed), "run_dir": out,
            "freeze": GC.load_json(out / "freeze.json"),
        })
    return rows


def _write(path: Path, payload) -> None:
    GC.write_json(Path(path), payload)


# ------------------------------------------------------------------ source audit
def cmd_source(args) -> dict:
    """SOURCE_AUDIT.json: what the active path imports, and what it never touches."""
    _prepare()
    import torch

    from final_model import FINAL_VARIANTS, FORBIDDEN_INPUT_TOKENS, HCHFinalCore

    impl_files = sorted(p.name for p in HERE.parent.glob("*.py"))
    active = [n for n in impl_files]
    hashes = {n: GC.sha256_file(HERE.parent / n) for n in active}

    # 1. static import scan of every active-path module
    forbidden_modules = ("core.allocation", "core.model", "core.encoders", "core.bundles")
    scan = {}
    for name in active:
        tree = ast.parse((HERE.parent / name).read_text(encoding="utf-8"))
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                hits += [a.name for a in node.names if a.name.split(".")[0] in ("allocation",)]
                hits += [a.name for a in node.names if a.name in forbidden_modules]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod in forbidden_modules or mod.startswith("core.model"):
                    hits.append(mod)
        scan[name] = sorted(set(hits))
    import_violations = {k: v for k, v in scan.items() if v}

    # 2. the imported substrate, resolved to the frozen files that define it
    scales = GD.reference_scales(*GC.CANARY_CELLS[0])
    substrate = [
        ("core.geometry", "residual_geometry, masked_softmax, masked_mean, check_identities"),
        ("core.decoder", "decode, prediction_identities, forecast_from_host"),
        ("core.losses", "structured_loss, daily_mae, masked_row_mean, component_losses, "
                        "wasserstein1_ordered, shape_auxiliary"),
        ("core.scales", "CoordinateScales, require_scales"),
        ("core.heads", "softplus_inverse"),
        ("core.contracts", "HORIZON, HISTORY_DAYS, TOKEN_DIM, EPS"),
        ("core.history", "RevealedResidualDay, select_revealed_history, build_history_tensors"),
        ("core.calendar", "hour_channels, day_channels"),
        ("core.feature_engineering", "host_hour_channels, host_day_descriptors"),
        ("core.training_support", "EMA, DifficultyInterleaver, build_primary_train_config, "
                                  "build_primary_optimizer, clip_gradients, daily_difficulty, "
                                  "fp32_region, neural_autocast, assert_primary_profile, "
                                  "assert_rescue_absent"),
    ]
    resolved = {}
    for module, symbols in substrate:
        src = sys.modules.get(module)
        path = Path(getattr(src, "__file__", "") or "")
        resolved[module] = {
            "symbols": symbols,
            "file": str(path.relative_to(REPO)).replace("\\", "/") if path.is_file() else None,
            "sha256": GC.sha256_file(path) if path.is_file() else None,
        }

    # 3. the model's own class census (no removed module may appear)
    removed = ("MaskedSoftmaxAllocator", "Allocation", "HistoryDayEncoder", "ShapeHistoryEncoder",
               "OptionalTrajectoryEncoder", "EvidenceSet", "MarketExpert", "SafetyGate")
    census = {}
    for variant in FINAL_VARIANTS:
        torch.manual_seed(0)
        model = HCHFinalCore(variant, scales)
        classes = sorted({type(m).__name__ for m in model.modules()})
        census[variant] = {
            "classes": classes,
            "removed_present": sorted(set(classes) & set(removed)),
        }
    removed_present = {v: c["removed_present"] for v, c in census.items() if c["removed_present"]}

    # 4. the disclosed package-init side effect
    legacy_in_sys_modules = sorted(
        m for m in ("core.model", "core.allocation", "core.encoders", "core.bundles")
        if m in sys.modules
    )

    # 5. prior-evidence pins, taken BEFORE any canary fit
    prior_pins = _prior_evidence_pins()

    payload = {
        "schema": "hch_final_gc_source_audit.v1",
        "protocol_id": GC.PROTOCOL_ID,
        "generated_at": _now(),
        "source_tree_digest": GC.RC.source_tree_digest(),
        "active_path_files": {n: hashes[n] for n in active},
        "active_path_forbidden_imports": import_violations,
        "active_path_clean": not import_violations,
        "reused_substrate": resolved,
        "model_class_census": census,
        "removed_modules_present_in_model": removed_present,
        "removed_modules_clean": not removed_present,
        "removed_active_path_modules": [
            "src/core/allocation.py (coordinate allocation + MaskedSoftmaxAllocator-as-router)",
            "src/core/encoders.py (HistoryDayEncoder, ShapeHistoryEncoder, calendar/host encoders)",
            "src/core/model.py (LegalEvidenceBatch, HCHV44Core, build_variants, evidence router)",
            "src/core/bundles.py (EvidenceSet, make_bundle)",
            "evidence-source routing, coordinate embeddings e_L/e_B/e_S, learned retrieval,",
            "safety/verification gates, market core/selector, market-specific experts,",
            "conditional amplitude rescue, OptionalTrajectoryEncoder",
        ],
        "forbidden_input_tokens": list(FORBIDDEN_INPUT_TOKENS),
        "disclosed_caveat": {
            "package_init_side_effect": (
                "importing core.<module> executes src/core/__init__.py, which eagerly re-exports "
                "the legacy modules, so those module objects appear in sys.modules as a package-init "
                "side effect.  No active-path module imports, names or calls them: the AST scan above "
                "is the proof, and the model class census shows no removed class is reachable."
            ),
            "legacy_modules_in_sys_modules_after_active_import": legacy_in_sys_modules,
        },
        "prior_evidence_pins": prior_pins,
        "prior_evidence_n_files": len(prior_pins["files"]),
        "test_target_read_count": 0,
        "test_rows_materialised": 0,
    }
    _write(EVID / "SOURCE_AUDIT.json", payload)
    return payload


def _prior_evidence_pins() -> dict:
    """Hash every prior artifact the canary reads, plus the exact path set.

    Only legal, non-quarantined prior files are pinned: the shadow-OOF supports the
    history index is built from, and the O1 run artifacts.  The quarantined TEST
    result tables are never opened, so they cannot be pinned.
    """
    files: dict = {}
    prior_evid = GC.RC.PRIOR_EVID
    for market, host in GC.FULL20_CELLS:
        key = GC.cell_key(market, host)
        for name in (f"{key}.npz", f"{key}.json"):
            p = prior_evid / "shadow_oof" / name
            if p.is_file():
                files[str(p.relative_to(REPO)).replace("\\", "/")] = GC.sha256_file(p)
    for market, host in GC.FULL20_CELLS:
        for seed in GC.SEEDS:
            run = GC.o1_run_dir(market, host, seed)
            for name in GC.O1_RUN_FIELDS:
                p = run / name
                files[str(p.relative_to(REPO)).replace("\\", "/")] = GC.sha256_file(p)
    return {"files": files, "n_files": len(files), "taken_at": _now()}


# ------------------------------------------------------------------ O1 provenance
def cmd_reuse(args) -> dict:
    _prepare()
    cells = GC.FULL20_CELLS if args.all_cells else GC.CANARY_CELLS
    payload = GC.o1_reuse_snapshot(cells)
    payload["checkpoint_pins"] = GC.o1_checkpoint_pins(cells)
    payload["pins_taken_at"] = _now()
    payload["o1_cell_medians"] = GC.o1_cell_medians(cells)
    payload["note"] = (
        "Reused read-only. The checkpoint pins are recorded here before the first canary fit "
        "and re-derived in GATE.json, so an unchanged map is the evidence that no O1 artifact "
        "was written, moved or re-selected."
    )
    _write(EVID / "O1_REUSE_PROVENANCE.json", payload)
    return payload


# ------------------------------------------------------------------- unit tests
def cmd_tests(args) -> dict:
    script = VERIFY_DIR / "unit_tests.py"
    cmd = [PYTHON, str(script), "--device", args.device]
    if args.no_smoke:
        cmd.append("--no-smoke")
    done = subprocess.run(cmd, cwd=str(STAGE), capture_output=True, text=True)
    report = GC.load_json(EVID / "UNIT_TEST_REPORT.json")
    report["subprocess_returncode"] = int(done.returncode)
    report["subprocess_tail"] = (done.stdout or "")[-2000:]
    _write(EVID / "UNIT_TEST_REPORT.json", report)
    return report


# ------------------------------------------------------------------------- fits
def _fit_spec(spec) -> dict:
    variant, market, host, seed = spec
    out = GC.run_dir(variant, market, host, seed)
    state = _run_state(out)
    if state == "complete":
        return {"spec": list(spec), "status": "skipped_existing"}
    if state != "absent":
        raise RuntimeError(f"{out}: {state}; refusing to overwrite a partial run")
    import trainer as TR

    device = os.environ.get("GC_DEVICE", "cuda")
    t0 = time.time()
    record = TR.train_final(variant, market, host, seed, out, device=device)
    return {
        "spec": list(spec), "status": "fit", "seconds": time.time() - t0,
        "selected_step": record["selected_step"],
        "selected_val_mae_ema": record["selected_val_mae_ema"],
        "stopped_at_step": record["optimization"]["stopped_at_step"],
    }


def cmd_fits(args) -> dict:
    _prepare()
    specs = universe(args.group)
    if args.limit:
        specs = specs[: int(args.limit)]
    os.environ["GC_DEVICE"] = args.device
    out = []
    t0 = time.time()
    if args.workers <= 1:
        GC.init_worker()
        for i, spec in enumerate(specs, 1):
            res = _fit_spec(tuple(spec))
            out.append(res)
            print(f"[fits] {i}/{len(specs)} {spec} {res['status']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers), initializer=GC.init_worker) as ex:
            futures = {ex.submit(_fit_spec, tuple(s)): tuple(s) for s in specs}
            for i, fut in enumerate(as_completed(futures), 1):
                spec = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {"spec": list(spec), "status": "error",
                           "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc().splitlines()[-8:]}
                out.append(res)
                print(f"[fits] {i}/{len(specs)} {spec} {res['status']}", flush=True)
    summary = {
        "group": args.group, "device": args.device, "workers": int(args.workers),
        "n_registered": len(specs), "n_fit": sum(r["status"] == "fit" for r in out),
        "n_skipped": sum(r["status"] == "skipped_existing" for r in out),
        "n_error": sum(r["status"] == "error" for r in out),
        "seconds": time.time() - t0, "results": out, "finished_at": _now(),
    }
    _write(EVID / f"FITS_{args.group.upper()}.json", summary)
    if summary["n_error"]:
        print(f"[fits] {summary['n_error']} errors; see FITS_{args.group.upper()}.json",
              file=sys.stderr)
    return summary


# -------------------------------------------------------------------- aggregate
def _checkpoint_fields(freeze: dict) -> dict:
    return {
        "val_mae_ema": float(freeze["selected_val_mae_ema"]),
        "val_mae_raw": float(freeze["selected_metrics"]["val_mae_raw"]),
        "val_host_mae": float(freeze["selected_metrics"]["val_host_mae"]),
        "val_gain_vs_host_pct": float(freeze["selected_metrics"]["val_gain_vs_host_pct"]),
        "val_correction_ratio": float(freeze["selected_metrics"]["val_correction_ratio"]),
        "val_mean_abs_correction": float(freeze["selected_metrics"]["val_mean_abs_correction"]),
        "train_mae_ema": float(freeze["selected_metrics"]["train_mae_ema"]),
        "train_host_mae": float(freeze["selected_metrics"]["train_host_mae"]),
        "train_correction_ratio": float(freeze["selected_metrics"]["train_correction_ratio"]),
        "diag_reconstruction_mae": freeze["selected_metrics"].get("diag_reconstruction_mae"),
        "diag_level": freeze["selected_metrics"].get("diag_level"),
        "diag_B": freeze["selected_metrics"].get("diag_B"),
        "diag_shape": freeze["selected_metrics"].get("diag_shape"),
    }


def _per_seed_rows(runs: list) -> list:
    rows = []
    for r in runs:
        f = r["freeze"]
        rows.append({
            "group": r["group"], "variant": r["variant"], "market": r["market"],
            "host": r["host"], "cell": r["cell"], "seed": r["seed"],
            **_checkpoint_fields(f),
            "selected_step": int(f["selected_step"]),
            "selected_check": int(f["selected_check"]),
            "stopped_at_step": f["optimization"]["stopped_at_step"],
            "check_count": int(f["check_count"]),
            "selected_after_switch": bool(f["selected_after_switch"]),
            "n_train_rows": int(f["n_train_rows"]), "n_val_rows": int(f["n_val_rows"]),
            "parameter_count": int(f["variant_config"]["parameter_count"]),
            "switch_step": int(f["optimization"]["switch_step"]),
            "test_target_read_count": int(f["test_target_read_count"]),
            "code_hash": f["code_hash"], "common_code_hash": f["common_code_hash"],
            "finite": bool(
                np.isfinite(f["selected_val_mae_ema"])
                and np.isfinite(f["selected_metrics"]["val_host_mae"])
            ),
        })
    return rows


def _medians(rows: list, key: str) -> float:
    return GC.median([r[key] for r in rows])


def _cell_medians(runs: list, o1_medians: dict) -> list:
    """One row per (variant, cell): median across seeds first, then the gains."""
    by_cell: dict = {}
    for r in runs:
        by_cell.setdefault((r["variant"], r["cell"]), []).append(r)

    lookup = {}
    for (variant, cell), group in by_cell.items():
        lookup[(variant, cell)] = {
            "mae_median": _medians(group, "val_mae_ema"),
            "host_mae_median": _medians(group, "val_host_mae"),
        }

    out = []
    for (variant, cell), group in sorted(by_cell.items()):
        mae = lookup[(variant, cell)]["mae_median"]
        host_mae = lookup[(variant, cell)]["host_mae_median"]
        o1 = o1_medians.get(cell, float("nan"))
        a1 = lookup.get((GC.GEOM_FLAT, cell), {}).get("mae_median", float("nan"))
        a2 = lookup.get((GC.GEOM_COUPLED, cell), {}).get("mae_median", float("nan"))
        maes = [r["val_mae_ema"] for r in group]
        market, host = group[0]["market"], group[0]["host"]
        out.append({
            "group": group[0]["group"], "variant": variant, "market": market, "host": host,
            "cell": cell,
            "is_shape_limited_weak": cell in WEAK_CELLS,
            "is_strong_control": cell == STRONG_CELL,
            "n_seeds": len(group),
            "n_finite": int(sum(bool(r["finite"]) for r in group)),
            "mae_median": mae,
            "mae_min": float(min(maes)), "mae_max": float(max(maes)),
            "mae_spread": float(max(maes) - min(maes)),
            "host_mae_median": host_mae,
            "o1_mae_median": o1,
            "a1_mae_median": a1,
            "a2_mae_median": a2,
            "gain_vs_host_pct": GC.gain_pct(host_mae, mae),
            "gain_vs_o1_pct": GC.gain_pct(o1, mae),
            "gain_vs_a1_pct": GC.gain_pct(a1, mae),
            "gain_vs_a2_pct": GC.gain_pct(a2, mae),
            "selected_step_median": _medians(group, "selected_step"),
            "stopped_at_step_median": GC.median(
                [r["stopped_at_step"] for r in group if r["stopped_at_step"] is not None]
            ),
            "correction_ratio_median": _medians(group, "val_correction_ratio"),
        })
    return out


def _training_curve_rows(runs: list) -> list:
    rows = []
    for r in runs:
        curve = GC.load_json(r["run_dir"] / "training_curve.json")
        for e in curve:
            gn = e.get("grad_norms_preclip") or {}
            rows.append({
                "group": r["group"], "variant": r["variant"], "cell": r["cell"],
                "seed": r["seed"], "check": int(e["check"]), "step": int(e["step"]),
                "is_initial": bool(e["is_initial"]),
                "is_optimization_step": bool(e["is_optimization_step"]),
                "used_full_loss_at_this_step": e["used_full_loss_at_this_step"],
                "loss_schedule": e["loss_schedule"],
                "objective_switch_here": bool(e["objective_switch_here"]),
                "epoch_equivalent": float(e["epoch_equivalent"]),
                "lr": float(e["lr"]),
                "step_loss_optimized": e.get("step_loss_optimized"),
                "val_mae_ema": float(e["val_mae_ema"]),
                "val_mae_raw": float(e["val_mae_raw"]),
                "val_host_mae": float(e["val_host_mae"]),
                "val_gain_vs_host_pct": float(e["val_gain_vs_host_pct"]),
                "val_correction_ratio": float(e["val_correction_ratio"]),
                "train_mae_ema": float(e["train_mae_ema"]),
                "diag_reconstruction_mae": e.get("diag_reconstruction_mae"),
                "diag_level": e.get("diag_level"),
                "diag_B": e.get("diag_B"),
                "diag_shape": e.get("diag_shape"),
                "grad_norm_encoder": gn.get("encoder"),
                "grad_norm_day_core": gn.get("day_core"),
                "grad_norm_shape_head": gn.get("shape_head"),
                "grad_norm_direct_head": gn.get("direct_head"),
                "selected": bool(e["selected"]),
                "best_val_mae_ema_so_far": float(e["best_val_mae_ema_so_far"]),
                "best_step_so_far": int(e["best_step_so_far"]),
            })
    return rows


def _mechanism_rows(runs: list, source: str = "selected_ema") -> list:
    """Exact-decoder mechanism quantities, re-derived from VAL predictions.

    ``source`` names the weight set the predictions describe:

    * ``"selected_ema"`` (the default, and what the table claims) reads
      ``val_predictions_selected_ema.npz``, which ``cmd_redump`` re-forwards from the
      frozen ``selected_ema.pt`` checkpoint;
    * ``"run_npz"`` reads the run's own ``val_predictions.npz``.

    The two differ for every early-stopped run: the pre-fix trainer dumped that file
    from the live EMA, which keeps moving after the selected check.  See ``cmd_redump``.
    """
    from core.geometry import residual_geometry

    name = ("val_predictions_selected_ema.npz" if source == "selected_ema"
            else "val_predictions.npz")
    rows = []
    for r in runs:
        with np.load(r["run_dir"] / name, allow_pickle=False) as z:
            residual = z["residual"].astype(np.float32)
            valid = z["hour_valid"].astype(bool)
            fields = set(z.files)
            entry = {"group": r["group"], "variant": r["variant"], "cell": r["cell"],
                     "seed": r["seed"], "weights": source,
                     "n_val_rows": int(residual.shape[0])}
            import torch

            geom = residual_geometry(torch.as_tensor(residual), torch.as_tensor(valid))
            b_true = geom.b.numpy()
            B_true = geom.B.numpy()
            entry["mean_abs_r_true"] = float(np.abs(residual).mean())
            entry["mean_abs_b_true"] = float(np.abs(b_true).mean())
            entry["mean_abs_B_true"] = float(np.abs(B_true).mean())
            if "correction" in fields:
                c = z["correction"].astype(np.float32)
                entry["mean_abs_correction"] = float(np.abs(c).mean())
                entry["correction_ratio_from_npz"] = float(
                    np.abs(c).mean() / max(np.abs(residual).mean(), 1e-12)
                )
            if "b_hat" in fields and "B_hat" in fields:
                b_hat = z["b_hat"].astype(np.float32)
                B_hat = z["B_hat"].astype(np.float32)
                entry["mean_abs_b_hat"] = float(np.abs(b_hat).mean())
                entry["mean_abs_B_hat"] = float(np.abs(B_hat).mean())
                entry["bias_b_hat"] = float(b_hat.mean() - b_true.mean())
                entry["corr_b_hat_true"] = _pearson(b_hat, b_true)
                entry["corr_B_hat_true"] = _pearson(B_hat, B_true)
                # a_plus - a_minus == H b_hat and a_plus + a_minus == ||r||_1 exactly,
                # so this is the share of the corrected L1 mass carried by the level.
                level = np.mean(np.abs(24.0 * b_hat))
                total = np.mean(np.abs(residual).sum(axis=-1))
                entry["level_share_of_l1"] = float(level / max(total, 1e-12))
            if "s_plus_hat" in fields and "s_minus_hat" in fields:
                sp = z["s_plus_hat"].astype(np.float32)
                sm = z["s_minus_hat"].astype(np.float32)
                entry["shape_entropy_plus"] = _mean_entropy(sp, valid)
                entry["shape_entropy_minus"] = _mean_entropy(sm, valid)
                entry["shape_w1_plus"] = _mean_w1(sp, geom.s_plus.numpy(), valid)
                entry["shape_w1_minus"] = _mean_w1(sm, geom.s_minus.numpy(), valid)
            if "overlap_gap" in fields:
                og = np.asarray(z["overlap_gap"], dtype=np.float64)
                entry["overlap_gap_median"] = float(np.nanmedian(og))
                entry["overlap_gap_max"] = float(np.nanmax(og))
            rows.append(entry)
    return rows


def _pearson(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 3 or a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _row_mask(valid, ndim: int):
    """Reduce an ``[N, H]`` validity mask to the leading shape of a per-row value.

    The hour axis is *reduced*, not indexed: entropy is summed over hours and W1 is
    a distance between hour distributions, so both are per-*day* quantities and a
    day counts when it has at least one valid hour.  Masked-softmax simplexes carry
    zero mass on invalid hours, so the ``1e-12`` clip adds a negligible per-hour
    term rather than a spurious one.
    """
    m = np.asarray(valid, dtype=bool)
    while m.ndim > int(ndim):
        m = m.any(axis=-1)
    return m


def _mean_entropy(simplex, valid) -> float:
    p = np.clip(np.asarray(simplex, dtype=np.float64), 1e-12, None)
    ent = -(p * np.log(p)).sum(axis=-1)
    m = _row_mask(valid, ent.ndim)
    return float(ent[m].mean()) if m.any() else float("nan")


def _mean_w1(pred, true, valid) -> float:
    from core.losses import wasserstein1_ordered
    import torch

    w1 = (
        wasserstein1_ordered(
            torch.as_tensor(np.asarray(pred)).float(),
            torch.as_tensor(np.asarray(true)).float(),
        )
        .detach().cpu().numpy()
    )
    m = _row_mask(valid, w1.ndim)
    return float(w1[m].mean()) if m.any() else float("nan")


def _structural_facts(scales, device: str) -> dict:
    """Parameter count/map, encoder/query census and inference latency per variant."""
    import torch
    import torch.nn as nn

    from final_model import DIRECT, FINAL_VARIANTS, HCHFinalCore

    counted = (nn.GRU, nn.LSTM, nn.RNN, nn.Transformer, nn.TransformerEncoder,
               nn.TransformerEncoderLayer, nn.MultiheadAttention)
    inp = GD.synthetic_input(LATENCY_BATCH, scales, seed=0, device=device)
    facts = {}
    for variant in FINAL_VARIANTS:
        torch.manual_seed(0)
        model = HCHFinalCore(variant, scales).to(device).eval()
        modules = list(model.modules())
        encoders = sorted(type(m).__name__ for m in modules if isinstance(m, counted))
        with torch.no_grad():
            for _ in range(LATENCY_WARMUP):
                model(inp)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times = []
            for _ in range(LATENCY_REPEATS):
                t0 = time.perf_counter()
                model(inp)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
        cfg = model.variant_config()
        facts[variant] = {
            "parameter_count": int(cfg["parameter_count"]),
            "parameter_map": cfg["parameter_map"],
            "learned_temporal_encoders": int(sum(
                1 for m in modules if isinstance(m, (nn.GRU, nn.LSTM, nn.RNN))
            )),
            "attention_modules": int(sum(
                1 for m in modules
                if isinstance(m, (nn.MultiheadAttention, nn.Transformer, nn.TransformerEncoder,
                                  nn.TransformerEncoderLayer))
            )),
            "encoder_class_census": encoders,
            "signed_mass_queries": int(cfg["signed_mass_queries"]),
            "source_router_calls": int(cfg["source_router_calls"]),
            "latency_batch": int(LATENCY_BATCH),
            "latency_device": device,
            "latency_ms_median": float(np.median(times)),
            "latency_ms_p95": float(np.quantile(times, 0.95)),
            "latency_ms_mean": float(np.mean(times)),
            "latency_ms_min": float(np.min(times)),
        }
    ratio = facts[DIRECT]["parameter_count"] / max(facts[GC.GEOM_COUPLED]["parameter_count"], 1)
    for variant, f in facts.items():
        f["direct_over_a2_ratio"] = ratio if variant == DIRECT else None
        f["ratio_to_v44_g2"] = f["parameter_count"] / float(V44_G2_PARAMETER_COUNT)
        f["within_0_70_of_v44_g2"] = bool(
            f["ratio_to_v44_g2"] <= G5_PARAMETER_TARGET_RATIO
        )
    return facts


# ------------------------------------------------------------------------ redump
def cmd_redump(args) -> dict:
    """Re-forward every frozen ``selected_ema.pt`` to rebuild the mechanism table.

    The E2 runs were produced by a trainer that dumped ``val_predictions.npz`` from
    the *live* EMA, which keeps moving after the selected check, so for every
    early-stopped run that file describes the run's last step rather than the model
    the freeze record describes.  Retraining is neither needed nor permitted here,
    and overwriting a run's own file would invalidate the hash its freeze record
    carries, so this writes a *new* per-run artifact instead:
    ``val_predictions_selected_ema.npz``, forwarded from the frozen checkpoint.

    It is self-validating in two ways.  Every redump must reproduce the recorded
    ``selected_val_mae_ema``, and for the runs whose own npz already came from the
    selected EMA -- the ones that ran to the step budget -- the redump must also
    reproduce that file's correction, which proves the reconstructed inputs are the
    run's own inputs.
    """
    _prepare()
    import torch

    from final_model import HCHFinalCore
    from trainer import _chunked_forward, _save_val_predictions

    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    records = []
    cache: dict = {}
    for group in ("critical", "ablation", "full20"):
        for r in load_runs(group):
            market, host = r["market"], r["host"]
            if (market, host) not in cache:
                cache[(market, host)] = GD.to_device(GD.build_cell_data(market, host), device)
            data = cache[(market, host)]
            model = HCHFinalCore(r["variant"], data["scales"], dropout=0.1).to(device)
            ckpt = torch.load(r["run_dir"] / "selected_ema.pt", map_location=device,
                              weights_only=True)
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            # The EMA shadow tracks trainable parameters only.  The registered
            # buffers -- the TRAIN-frozen coordinate scales and the B-init offset --
            # are constants of the variant, rebuilt by the constructor from the
            # frozen scales, so they are legitimately absent.  Any *other* absent
            # key, or any unexpected key, means the checkpoint and this code state
            # disagree and the redump would not be the run's own model.
            buffers = {name for name, _ in model.named_buffers()}
            if unexpected or set(missing) != buffers:
                raise RuntimeError(
                    f"{r['cell']}/seed{r['seed']}: checkpoint does not match the "
                    f"registered architecture -- missing={sorted(missing)} "
                    f"(buffers={sorted(buffers)}), unexpected={sorted(unexpected)}"
                )
            model.eval()
            outs = _chunked_forward(model, data["val_input"])
            info = _save_val_predictions(
                r["run_dir"] / "val_predictions_selected_ema.npz", outs, data
            )
            residual = data["val_res"].detach().cpu().numpy().astype(np.float64)
            correction = outs["correction"].detach().cpu().numpy().astype(np.float64)
            mae = float(np.abs(residual - correction).mean())
            recorded = float(r["freeze"]["selected_val_mae_ema"])

            own = r["run_dir"] / "val_predictions.npz"
            own_mae = own_deviation = None
            own_match = None
            if own.is_file():
                with np.load(own, allow_pickle=False) as z:
                    own_c = z["correction"].astype(np.float64)
                own_mae = float(np.abs(residual - own_c).mean())
                own_deviation = abs(own_mae - recorded)
                own_match = bool(np.allclose(own_c, correction, rtol=0.0, atol=1e-5))
            records.append({
                "group": group, "variant": r["variant"], "cell": r["cell"],
                "seed": r["seed"],
                "redump_mae": mae, "recorded_selected_val_mae_ema": recorded,
                "abs_deviation": abs(mae - recorded),
                "own_npz_mae": own_mae,
                "own_npz_abs_deviation_from_recorded": own_deviation,
                "own_npz_is_selected_ema": (None if own_deviation is None
                                            else bool(own_deviation < 1e-3)),
                "own_npz_matches_redump": own_match,
                "selected_step": int(r["freeze"]["selected_step"]),
                "stopped_at_step": r["freeze"]["optimization"]["stopped_at_step"],
                "npz_sha256": info["sha256"],
                "checkpoint_sha256": GC.sha256_file(r["run_dir"] / "selected_ema.pt"),
            })

    already = [x for x in records if x["own_npz_is_selected_ema"]]
    comparable = [x for x in records if x["own_npz_is_selected_ema"]]
    payload = {
        "schema": "hch_final_gc_redump.v1",
        "protocol_id": GC.PROTOCOL_ID,
        "generated_at": _now(),
        "device": device,
        "reason": (
            "The 48 E2 runs were fitted by a trainer that dumped val_predictions.npz from "
            "the live EMA rather than the selected EMA snapshot.  This command re-forwards "
            "the frozen selected_ema.pt to produce val_predictions_selected_ema.npz.  No run "
            "artifact was overwritten, no fit was retrained and no selection was re-run."
        ),
        "n_runs": len(records),
        "n_own_npz_already_selected_ema": len(already),
        "n_own_npz_matching_redump": sum(1 for x in comparable
                                         if x["own_npz_matches_redump"]),
        "max_abs_deviation_from_recorded": max(
            [x["abs_deviation"] for x in records], default=float("nan")
        ),
        "all_redumps_reproduce_recorded_ema": bool(
            records and all(x["abs_deviation"] < 1e-3 for x in records)
        ),
        "all_comparable_reproduce_own_npz": bool(
            comparable and all(x["own_npz_matches_redump"] for x in comparable)
        ),
        "runs": records,
    }
    _write(EVID / "REDUMP_REPORT.json", payload)
    return {
        "n_runs": len(records),
        "n_own_npz_already_selected_ema": len(already),
        "n_own_npz_matching_redump": payload["n_own_npz_matching_redump"],
        "max_abs_deviation_from_recorded": payload["max_abs_deviation_from_recorded"],
        "all_redumps_reproduce_recorded_ema": payload["all_redumps_reproduce_recorded_ema"],
    }


def cmd_aggregate(args) -> dict:
    _prepare()
    import torch

    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    o1_medians = GC.o1_cell_medians()
    all_rows = {}
    summary = {}
    for group in ("critical", "ablation", "full20"):
        runs = load_runs(group)
        if not runs:
            summary[group] = {"n_runs": 0, "complete": False}
            continue
        per_seed = _per_seed_rows(runs)
        # ``_cell_medians`` reads the *flattened* per-seed fields (``val_mae_ema``,
        # ``val_host_mae``, ``finite``, ...) that ``_per_seed_rows`` derives from the
        # freeze record; the raw ``load_runs`` records carry only the freeze itself.
        cell_medians = _cell_medians(per_seed, o1_medians)
        GC.write_csv(EVID / f"{group.upper()}_PER_SEED_METRICS.csv", per_seed)
        GC.write_csv(EVID / f"{group.upper()}_CELL_MEDIANS.csv", cell_medians)
        all_rows[group] = (per_seed, cell_medians, runs)
        expected = {"critical": CRITICAL_FITS, "ablation": ABLATION_FITS,
                    "full20": FULL20_FITS}[group]
        summary[group] = {
            "n_runs": len(runs), "expected_runs": expected,
            "complete": len(runs) == expected,
            "n_cells": len({r["cell"] for r in runs}),
        }

    # The PROTOCOL §13 names are the critical group's tables; group-specific copies
    # are kept alongside so E4 evidence is complete without overwriting anything.
    if "critical" in all_rows:
        per_seed, cell_medians, runs = all_rows["critical"]
        GC.write_csv(EVID / "PER_SEED_METRICS.csv", per_seed)
        GC.write_csv(EVID / "CELL_MEDIANS.csv", cell_medians)
        GC.write_csv(EVID / "TRAINING_CURVES.csv", _training_curve_rows(runs))
        # The registered table describes the *selected* model, so it reads the
        # checkpoint re-forward when that evidence exists.  Falling back to the run's
        # own npz would silently describe the last EMA step for early-stopped runs.
        mech_source = ("selected_ema"
                       if all((r["run_dir"] / "val_predictions_selected_ema.npz").is_file()
                              for r in runs) else "run_npz")
        GC.write_csv(EVID / "MECHANISM_DIAGNOSTICS.csv",
                     _mechanism_rows(runs, mech_source))
    if "ablation" in all_rows:
        GC.write_csv(EVID / "ABLATION_CELL_MEDIANS.csv", all_rows["ablation"][1])
    if "full20" in all_rows:
        GC.write_csv(EVID / "FULL20_PER_SEED.csv", all_rows["full20"][0])
        GC.write_csv(EVID / "FULL20_CELL_MEDIANS.csv", all_rows["full20"][1])
        GC.write_csv(EVID / "FULL20_MARKET_SUMMARY.csv", _market_summary(all_rows["full20"][1]))

    scales = GD.reference_scales(*GC.CANARY_CELLS[0])
    facts = _structural_facts(scales, device)
    # Cross-check the measured counts against what every registered run recorded.
    disagreements = []
    for group, (_, _, runs) in all_rows.items():
        for r in runs:
            recorded = int(r["freeze"]["variant_config"]["parameter_count"])
            if recorded != facts[r["variant"]]["parameter_count"]:
                disagreements.append({
                    "group": group, "variant": r["variant"], "cell": r["cell"],
                    "seed": r["seed"], "recorded": recorded,
                    "measured": facts[r["variant"]]["parameter_count"],
                })
    if disagreements:
        raise RuntimeError(f"parameter count disagrees with the freeze records: {disagreements[:3]}")
    GC.write_csv(EVID / "PARAMETER_COUNTS.csv", [
        {"variant": v, **{k: val for k, val in f.items()}} for v, f in facts.items()
    ])
    _write(EVID / "VARIANT_CONFIGS.json", {
        "schema": "hch_final_gc_variant_configs.v1",
        "protocol_id": GC.PROTOCOL_ID,
        "variants": facts,
        "g5_parameter_target_ratio_of_v44_g2": G5_PARAMETER_TARGET_RATIO,
        "v44_g2_parameter_count": V44_G2_PARAMETER_COUNT,
        "tables": {k: v for k, v in summary.items()},
        "table_documentation": DOC,
        "generated_at": _now(),
    })
    return {"tables": summary, "parameter_counts": {v: f["parameter_count"] for v, f in facts.items()}}


def _market_summary(cell_medians: list) -> list:
    """Per-market aggregation of the full-20 A2 panel (median over its cells)."""
    by_market: dict = {}
    for row in cell_medians:
        if row["variant"] != GC.GEOM_COUPLED:
            continue
        by_market.setdefault(row["market"], []).append(row)
    out = []
    for market, rows in sorted(by_market.items()):
        out.append({
            "market": market,
            "n_cells": len(rows),
            "mae_median": GC.median([r["mae_median"] for r in rows]),
            "gain_vs_host_pct_median": GC.median([r["gain_vs_host_pct"] for r in rows]),
            "gain_vs_o1_pct_median": GC.median([r["gain_vs_o1_pct"] for r in rows]),
            "n_cells_host_positive": int(sum(r["gain_vs_host_pct"] > 0 for r in rows)),
            "n_cells_o1_positive": int(sum(r["gain_vs_o1_pct"] > 0 for r in rows)),
            "worst_gain_vs_host_pct": float(min(r["gain_vs_host_pct"] for r in rows)),
            "worst_gain_vs_o1_pct": float(min(r["gain_vs_o1_pct"] for r in rows)),
        })
    return out


# ------------------------------------------------------------------------ access
def cmd_access(args) -> dict:
    _prepare()
    runs = load_runs("critical") + load_runs("ablation") + load_runs("full20")
    per_run = {}
    total_reads = 0
    for r in runs:
        key = f"{r['variant']}/{r['cell']}/seed{r['seed']}"
        n = int(r["freeze"]["test_target_read_count"])
        per_run[key] = n
        total_reads += n
    payload = GC.RC.write_access_audit(EVID / "ACCESS_AUDIT.json", extra={
        "protocol_id": GC.PROTOCOL_ID,
        "stage_protocol_id": GC.PROTOCOL_ID,
        "stage": "hch_final_geometry_coupled_canary_20260918",
        "role_scope": "TRAIN and VAL only; TEST is refused at the frame builder and counted",
        "n_registered_runs_audited": len(runs),
        "per_run_test_target_read_count": per_run,
        "sum_run_test_target_read_counts": total_reads,
        "o1_evidence_read_paths_are_legal": [
            "experiments/evidence/hch_v44_objective_alignment_probe_20260917/o1_runs/**",
            "experiments/evidence/hch_v44_o1_full8_completion_20260918/o1_runs/**",
            "experiments/evidence/hch_v44_o1_domestic20_guardrail_20260918/o1_runs/**",
        ],
        "disclosed_caveat": (
            "test_rows_dropped_before_return is a per-process counter: the writer of this file "
            "reports its own process's count, not the sum over fit workers.  The authoritative "
            "evidence that no TEST row was ever materialised is that every registered run recorded "
            "test_target_read_count == 0 and test_rows_materialised == 0, and that "
            "distinct_paths_blocked is empty."
        ),
    })
    return payload


# -------------------------------------------------------------------------- gate
def _gate_g0(runs: list, extra: dict) -> dict:
    codes = {r["freeze"]["code_hash"] for r in runs}
    commons = {r["freeze"]["common_code_hash"] for r in runs}
    # Self-reporting integrity check: are the registered runs' own recorded hashes
    # still the hashes of the files on disk?  A run's ``code_hash`` pins the trainer
    # that produced it.  This is reported and never gated -- the fits are frozen
    # evidence and a later, disclosed repair to a *post-training* code path (here:
    # the VAL-prediction dump in ``cmd_redump``'s docstring) must not retroactively
    # invalidate them -- but it must be visible rather than silently absorbed into
    # ``one_scientific_code_state``, which asserts homogeneity *between* runs.
    trainer_now = GC.sha256_file(HERE.parent / "trainer.py")
    common_now = GC.sha256_file(GC.HERE)
    # The key carries the variant: a cell/seed pair names two runs, one per arm, and
    # a variant-less key would silently dedupe the two and halve the count.
    stale = sorted({
        f"{r['variant']}/{r['cell']}/seed{r['seed']}" for r in runs
        if r["freeze"]["code_hash"] != trainer_now
        or r["freeze"]["common_code_hash"] != common_now
    })
    digests = {json.dumps(r["freeze"]["source_tree_digest"], sort_keys=True) for r in runs}
    current = json.dumps(GC.RC.source_tree_digest(), sort_keys=True)
    a1 = [r for r in runs if r["variant"] == GC.GEOM_FLAT]
    a2 = [r for r in runs if r["variant"] == GC.GEOM_COUPLED]
    reads = sum(int(r["freeze"]["test_target_read_count"]) for r in runs)
    materialised = sum(int(r["freeze"]["test_rows_materialised"]) for r in runs)
    nonfinite = [
        f"{r['variant']}/{r['cell']}/seed{r['seed']}" for r in runs
        if not (np.isfinite(r["freeze"]["selected_val_mae_ema"])
                and np.isfinite(r["freeze"]["selected_metrics"]["val_host_mae"]))
    ]
    checks = {
        "source_audit_clean": bool(
            extra["source"]["active_path_clean"]
            and extra["source"]["removed_modules_clean"]
            and not extra["source_prior_drift"]["changed"]
            and not extra["source_prior_drift"]["added"]
            and not extra["source_prior_drift"]["removed"]
        ),
        "unit_test_gate_clean": bool(extra["unit"]["all_passed"]),
        "fits_complete_a1_24_of_24": len(a1) == 24,
        "fits_complete_a2_24_of_24": len(a2) == 24,
        "o1_reused_never_retrained": bool(
            extra["o1"]["all_reused_present"]
            and extra["o1"]["all_checkpoints_match_own_record"]
            and extra["o1"]["one_probe_code_hash_across_reused"]
            and extra["o1"]["one_source_tree_digest_across_reused"]
        ),
        "o1_artifacts_unmutated": bool(extra["o1_pins_unchanged"]),
        "one_scientific_code_state": bool(
            len(codes) == 1 and len(commons) == 1 and digests == {current}
        ),
        "all_finite": not nonfinite,
        "test_reads_zero": bool(
            reads == 0 and materialised == 0
            and int(extra["access"]["access_state"]["test_rows_returned"]) == 0
            and not extra["access"]["access_state"]["distinct_paths_blocked"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "detail": {
            "n_a1": len(a1), "n_a2": len(a2),
            "distinct_code_hash": sorted(codes), "distinct_common_code_hash": sorted(commons),
            "trainer_hash_on_disk": trainer_now, "common_hash_on_disk": common_now,
            "runs_whose_recorded_hash_differs_from_disk": stale,
            "n_runs_whose_recorded_hash_differs_from_disk": len(stale),
            "n_distinct_source_tree_digests": len(digests),
            "source_tree_digest_matches_current": digests == {current},
            "sum_test_target_read_counts": reads,
            "sum_test_rows_materialised": materialised,
            "nonfinite_runs": nonfinite,
            "source_prior_drift": extra["source_prior_drift"],
        },
    }


def _gate_cell_block(cell_medians: list, variant: str) -> dict:
    return {r["cell"]: r for r in cell_medians if r["variant"] == variant}


def cmd_gate(args) -> dict:
    _prepare()
    source = GC.load_json(EVID / "SOURCE_AUDIT.json")
    unit = GC.load_json(EVID / "UNIT_TEST_REPORT.json")
    o1 = GC.load_json(EVID / "O1_REUSE_PROVENANCE.json")
    access = GC.load_json(EVID / "ACCESS_AUDIT.json")
    runs = load_runs("critical")

    # Re-derive the pre-fit pins: an unchanged map is the no-mutation evidence.
    current_pins = GC.o1_checkpoint_pins()
    recorded_pins = o1.get("checkpoint_pins", {})
    pins_changed = sorted(
        k for k in set(current_pins) | set(recorded_pins)
        if current_pins.get(k) != recorded_pins.get(k)
    )

    current_prior = _prior_evidence_pins()
    recorded_prior = source.get("prior_evidence_pins", {}).get("files", {})
    prior_changed = sorted(
        k for k in set(current_prior["files"]) & set(recorded_prior)
        if current_prior["files"][k] != recorded_prior[k]
    )
    prior_added = sorted(set(current_prior["files"]) - set(recorded_prior))
    prior_removed = sorted(set(recorded_prior) - set(current_prior["files"]))

    both = GC.load_json(EVID / "FITS_CRITICAL.json") if (EVID / "FITS_CRITICAL.json").is_file() else {}
    g0 = _gate_g0(runs, {
        "source": source, "unit": unit, "o1": o1, "access": access,
        "o1_pins_unchanged": not pins_changed,
        "source_prior_drift": {"changed": prior_changed, "added": prior_added,
                               "removed": prior_removed},
    })

    cell_medians = GC.read_csv_rows(EVID / "CELL_MEDIANS.csv") if (EVID / "CELL_MEDIANS.csv").is_file() else []
    for row in cell_medians:
        for k in ("mae_median", "host_mae_median", "o1_mae_median", "a1_mae_median",
                  "a2_mae_median", "gain_vs_host_pct", "gain_vs_o1_pct", "gain_vs_a1_pct"):
            row[k] = float(row[k]) if row[k] not in ("", None) else float("nan")
        row["is_shape_limited_weak"] = str(row["is_shape_limited_weak"]) == "True"

    params = GC.read_csv_rows(EVID / "PARAMETER_COUNTS.csv") if (EVID / "PARAMETER_COUNTS.csv").is_file() else []

    a2 = _gate_cell_block(cell_medians, GC.GEOM_COUPLED)
    canary_gains_host = [a2[c]["gain_vs_host_pct"] for c in a2]
    canary_gains_o1 = [a2[c]["gain_vs_o1_pct"] for c in a2]
    canary_gains_a1 = [a2[c]["gain_vs_a1_pct"] for c in a2]
    weak_gains = [a2[c]["gain_vs_a1_pct"] for c in a2 if a2[c]["is_shape_limited_weak"]]

    g1 = {
        "n_host_positive": int(sum(g > 0 for g in canary_gains_host)),
        "n_cells": len(canary_gains_host),
        "worst_gain_vs_host_pct": float(min(canary_gains_host)) if canary_gains_host else None,
        "thresholds": {"n_host_positive": 7, "worst_gain_vs_host_pct": -0.5},
    }
    g1["passed"] = bool(g1["n_host_positive"] >= 7
                        and g1["worst_gain_vs_host_pct"] is not None
                        and g1["worst_gain_vs_host_pct"] >= -0.5)

    g2 = {
        "n_cells_better_than_o1": int(sum(g > 0 for g in canary_gains_o1)),
        "panel_median_gain_vs_o1_pct": GC.median(canary_gains_o1),
        "worst_gain_vs_o1_pct": float(min(canary_gains_o1)) if canary_gains_o1 else None,
        "thresholds": {"n_cells_better_than_o1": 5, "panel_median_gain_vs_o1_pct": 0.50,
                       "worst_gain_vs_o1_pct": -1.50},
    }
    g2["passed"] = bool(g2["n_cells_better_than_o1"] >= 5
                        and np.isfinite(g2["panel_median_gain_vs_o1_pct"])
                        and g2["panel_median_gain_vs_o1_pct"] >= 0.50
                        and g2["worst_gain_vs_o1_pct"] >= -1.50)

    g3 = {
        "n_cells_better_than_a1": int(sum(g > 0 for g in canary_gains_a1)),
        "panel_median_gain_vs_a1_pct": GC.median(canary_gains_a1),
        "n_weak_cells": len(weak_gains),
        "n_weak_cells_improved": int(sum(g > 0 for g in weak_gains)),
        "weak_median_gain_vs_a1_pct": GC.median(weak_gains),
        "weak_cells": sorted(c for c in a2 if a2[c]["is_shape_limited_weak"]),
        "thresholds": {"n_cells_better_than_a1": 6, "panel_median_gain_vs_a1_pct": 0.50,
                       "n_weak_cells_improved": 4, "weak_median_gain_vs_a1_pct": 0.75},
    }
    g3["passed"] = bool(g3["n_cells_better_than_a1"] >= 6
                        and np.isfinite(g3["panel_median_gain_vs_a1_pct"])
                        and g3["panel_median_gain_vs_a1_pct"] >= 0.50
                        and g3["n_weak_cells_improved"] >= 4
                        and np.isfinite(g3["weak_median_gain_vs_a1_pct"])
                        and g3["weak_median_gain_vs_a1_pct"] >= 0.75)

    strong = a2.get(STRONG_CELL)
    g4 = {
        "cell": STRONG_CELL,
        "gain_vs_o1_pct": None if strong is None else strong["gain_vs_o1_pct"],
        "gain_vs_host_pct": None if strong is None else strong["gain_vs_host_pct"],
        "thresholds": {"min_gain_vs_o1_pct": -1.0},
    }
    g4["passed"] = bool(strong is not None and np.isfinite(strong["gain_vs_o1_pct"])
                        and strong["gain_vs_o1_pct"] >= -1.0)

    a2_row = next((p for p in params if p["variant"] == GC.GEOM_COUPLED), None)
    enc = int(float(a2_row["learned_temporal_encoders"])) if a2_row else None
    qry = int(float(a2_row["signed_mass_queries"])) if a2_row else None
    n_router_calls = int(float(a2_row["source_router_calls"])) if a2_row else None
    attn = int(float(a2_row["attention_modules"])) if a2_row else None
    g5 = {
        "learned_temporal_encoders": enc,
        "signed_mass_queries": qry,
        "source_router_calls": n_router_calls,
        "attention_modules": attn,
        "parameter_count": int(float(a2_row["parameter_count"])) if a2_row else None,
        "parameter_map": a2_row["parameter_map"] if a2_row else None,
        "latency_ms_median": float(a2_row["latency_ms_median"]) if a2_row else None,
        "latency_ms_p95": float(a2_row["latency_ms_p95"]) if a2_row else None,
        "latency_batch": int(float(a2_row["latency_batch"])) if a2_row else None,
        "latency_device": a2_row["latency_device"] if a2_row else None,
        "ratio_to_v44_g2": float(a2_row["ratio_to_v44_g2"]) if a2_row else None,
        "thresholds": {"learned_temporal_encoders": 1, "signed_mass_queries": 2,
                       "source_router_calls": 0},
        "is_validity_reporting_gate": True,
    }
    g5["passed"] = bool(enc == 1 and qry == 2 and n_router_calls == 0 and attn == 0)

    gates = {"G0": g0, "G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5}
    blocked_reason = None
    if not g0["checks"]["unit_test_gate_clean"]:
        blocked_reason = "UNIT_TEST_GATE"
    elif not g0["checks"]["source_audit_clean"]:
        blocked_reason = "SOURCE_GATE"
    elif not g0["checks"]["o1_reused_never_retrained"] or not g0["checks"]["o1_artifacts_unmutated"]:
        blocked_reason = "O1_REUSE"
    elif not g0["checks"]["fits_complete_a1_24_of_24"] or not g0["checks"]["fits_complete_a2_24_of_24"]:
        blocked_reason = "FITS_INCOMPLETE"
    elif not g0["checks"]["one_scientific_code_state"]:
        blocked_reason = "CODE_STATE"
    elif not g0["checks"]["all_finite"]:
        blocked_reason = "NONFINITE_METRIC"
    elif not g0["checks"]["test_reads_zero"]:
        blocked_reason = "ACCESS_GATE"

    payload = {
        "schema": "hch_final_gc_gate.v1",
        "protocol_id": GC.PROTOCOL_ID,
        "generated_at": _now(),
        "n_critical_runs": len(runs),
        "critical_fits_expected": CRITICAL_FITS,
        "gates": gates,
        "all_gates_passed": all(g["passed"] for g in gates.values()),
        "blocked_reason": blocked_reason,
        "primary_statistic": "median VAL MAE across seeds first, then relative gain (PROTOCOL §9)",
        "fits_critical_summary": {
            k: both.get(k) for k in ("n_registered", "n_fit", "n_skipped", "n_error")
        },
    }
    _write(EVID / "GATE.json", payload)

    # Conditional groups, if they were executed.
    cond = {}
    if (EVID / "ABLATION_CELL_MEDIANS.csv").is_file():
        cond["ablation"] = _ablation_gate()
    if (EVID / "FULL20_CELL_MEDIANS.csv").is_file():
        cond["full20"] = _full20_gate()
    if cond:
        _write(EVID / "CONDITIONAL_GATES.json", cond)
    return payload


def _ablation_gate() -> dict:
    rows = GC.read_csv_rows(EVID / "ABLATION_CELL_MEDIANS.csv")
    out = {}
    for variant in GC.ABLATION_VARIANTS:
        sub = [r for r in rows if r["variant"] == variant]
        if not sub:
            continue
        gains_host = [float(r["gain_vs_host_pct"]) for r in sub]
        gains_a2 = [float(r["gain_vs_a2_pct"]) for r in sub]
        out[variant] = {
            "n_cells": len(sub),
            "n_host_positive": int(sum(g > 0 for g in gains_host)),
            "median_gain_vs_host_pct": GC.median(gains_host),
            "median_gain_vs_a2_pct": GC.median(gains_a2),
            "worst_gain_vs_host_pct": float(min(gains_host)),
        }
    return {
        "role": "paper ablations; not rescue arms and not replacements for GEOM_COUPLED",
        "per_variant": out,
        "verdict_note": (
            "Ablations are reported as measured.  No ablation outcome can replace the "
            "GEOM_COUPLED canary result."
        ),
    }


def _full20_gate() -> dict:
    rows = GC.read_csv_rows(EVID / "FULL20_CELL_MEDIANS.csv")
    a2 = [r for r in rows if r["variant"] == GC.GEOM_COUPLED]
    gains_host = [float(r["gain_vs_host_pct"]) for r in a2]
    gains_o1 = [float(r["gain_vs_o1_pct"]) for r in a2]
    payload = {
        "n_cells": len(a2),
        "n_host_positive": int(sum(g > 0 for g in gains_host)),
        "panel_median_gain_vs_host_pct": GC.median(gains_host),
        "panel_median_gain_vs_o1_pct": GC.median(gains_o1),
        "worst_gain_vs_host_pct": float(min(gains_host)) if gains_host else None,
        "n_cells_o1_positive": int(sum(g > 0 for g in gains_o1)),
        "role": "full-20 TRAIN+VAL development table; TEST is not opened",
    }
    return payload


# ------------------------------------------------------------------------- token
def cmd_token(args) -> dict:
    gate = GC.load_json(EVID / "GATE.json")
    if gate["blocked_reason"]:
        token = f"HCH_FINAL_GC_CANARY_BLOCKED_{gate['blocked_reason']}"
    elif gate["all_gates_passed"]:
        token = "HCH_FINAL_GC_CANARY_SUPPORTED"
    else:
        token = "HCH_FINAL_GC_CANARY_NOT_SUPPORTED"

    conditional = []
    ablation_runs = load_runs("ablation")
    full20_runs = load_runs("full20")
    if len(ablation_runs) == ABLATION_FITS:
        conditional.append("HCH_FINAL_GC_ABLATION_COMPLETE")
    if len(full20_runs) == FULL20_FITS:
        conditional.append("HCH_FINAL_GC_FULL20_COMPLETE_FOR_ADJUDICATION")
    if conditional and not gate["all_gates_passed"]:
        conditional = []

    payload = {
        "schema": "hch_final_gc_stage_token.v1",
        "protocol_id": GC.PROTOCOL_ID,
        "terminal_token": token,
        "conditional_tokens": conditional,
        "gate_summary": {k: bool(v["passed"]) for k, v in gate["gates"].items()},
        "blocked_reason": gate["blocked_reason"],
        "n_critical_fits": len(load_runs("critical")),
        "n_ablation_fits": len(ablation_runs),
        "n_full20_fits": len(full20_runs),
        "test_opened": False,
        "evidence_root": "experiments/evidence/hch_final_geometry_coupled_canary_20260918",
        "issued_at": _now(),
    }
    _write(EVID / "STAGE_TOKEN.json", payload)
    return payload


# ------------------------------------------------------------------------ report
def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "n/a" if v != v else f"{v:.{nd}f}"


def cmd_report(args) -> dict:
    gate = GC.load_json(EVID / "GATE.json")
    token = GC.load_json(EVID / "STAGE_TOKEN.json")
    rows = GC.read_csv_rows(EVID / "CELL_MEDIANS.csv")
    params = {r["variant"]: r for r in GC.read_csv_rows(EVID / "PARAMETER_COUNTS.csv")}
    a2 = [r for r in rows if r["variant"] == GC.GEOM_COUPLED]

    lines = []
    A = lines.append
    A("# HCH Final Geometry-Coupled Repair Canary — Results")
    A("")
    A(f"- Protocol: `{GC.PROTOCOL_ID}`")
    A(f"- Evidence root: `experiments/evidence/hch_final_geometry_coupled_canary_20260918/`")
    A(f"- Generated: {_now()}")
    A(f"- **Terminal token: `{token['terminal_token']}`**")
    if token["conditional_tokens"]:
        A(f"- Conditional tokens recorded inside evidence: {', '.join('`'+t+'`' for t in token['conditional_tokens'])}")
    A("")
    A("## 1. Canary panel (seed-median VAL MAE, PROTOCOL §9)")
    A("")
    A("| cell | Host | O1 | A1 GEOM_FLAT | A2 GEOM_COUPLED | gain vs Host | gain vs O1 | gain vs A1 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in sorted(a2, key=lambda x: (x["market"], x["host"])):
        A(f"| {r['cell']} | {_fmt(r['host_mae_median'],3)} | {_fmt(r['o1_mae_median'],3)} | "
          f"{_fmt(r['a1_mae_median'],3)} | {_fmt(r['a2_mae_median'],3)} | "
          f"{_fmt(r['gain_vs_host_pct'],3)}% | {_fmt(r['gain_vs_o1_pct'],3)}% | "
          f"{_fmt(r['gain_vs_a1_pct'],3)}% |")
    A("")
    A("Panel medians are computed as the median over the eight per-cell gains, not as a")
    A("median of paired seed-level gains.")
    A("")
    A("## 2. Promotion gate G0–G5")
    A("")
    for name, g in gate["gates"].items():
        A(f"### {name}: {'PASS' if g['passed'] else 'FAIL'}")
        A("")
        A("```json")
        A(json.dumps({k: v for k, v in g.items() if k != "detail"}, indent=2, default=str))
        A("```")
        A("")
    A(f"**All gates passed: {gate['all_gates_passed']}**  ")
    if gate["blocked_reason"]:
        A(f"**Blocked reason: `{gate['blocked_reason']}`**")
    A("")
    A("## 3. Parameter count and latency")
    A("")
    A("| variant | parameters | ratio to v4.4 G2 (14948) | learned temporal encoders | attention modules | signed-mass queries | source-router calls | latency median (ms) | latency p95 (ms) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for v, p in params.items():
        A(f"| {v} | {p['parameter_count']} | {_fmt(p['ratio_to_v44_g2'],4)} | "
          f"{p['learned_temporal_encoders']} | {p['attention_modules']} | "
          f"{p['signed_mass_queries']} | {p['source_router_calls']} | "
          f"{_fmt(p['latency_ms_median'],3)} | {_fmt(p['latency_ms_p95'],3)} |")
    A("")
    A(f"Latency is a single-forward measurement at batch {LATENCY_BATCH} on "
      f"`{next(iter(params.values()))['latency_device']}`; it is reported, not gated.")
    A("")
    A("## 4. Active-path removals")
    A("")
    src = GC.load_json(EVID / "SOURCE_AUDIT.json")
    for item in src["removed_active_path_modules"]:
        A(f"- {item}")
    A("")
    A(f"Static scan of the {len(src['active_path_files'])} active-path modules found "
      f"{len(src['active_path_forbidden_imports'])} forbidden imports; the model class census "
      f"found {len(src['removed_modules_present_in_model'])} removed classes reachable.")
    A("")
    A("## 5. Access boundary")
    A("")
    A(f"- Registered runs audited: {gate['fits_critical_summary']}")
    A(f"- Sum of per-run TEST target reads: "
      f"{GC.load_json(EVID / 'ACCESS_AUDIT.json')['sum_run_test_target_read_counts']}")
    A(f"- Distinct forbidden paths blocked: "
      f"{GC.load_json(EVID / 'ACCESS_AUDIT.json')['access_state']['distinct_paths_blocked']}")
    A("- TEST role frames refused: "
      f"{GC.load_json(EVID / 'ACCESS_AUDIT.json')['access_state']['test_role_frame_refusals']}")
    A("- V2 TEST was never opened; no TEST target, prediction or metric was read for selection.")
    A("")
    A("## 6. Provenance")
    A("")
    prov = GC.load_json(EVID / "O1_REUSE_PROVENANCE.json")
    A(f"- O1 reused runs: {prov['n_reused_runs']} across {prov['n_reused_cells']} cells, "
      f"read-only; checkpoints match their own recorded hashes: "
      f"{prov['all_checkpoints_match_own_record']}")
    A(f"- One O1 code state across reused runs: {prov['one_probe_code_hash_across_reused']}")
    A(f"- One source-tree digest across reused runs: {prov['one_source_tree_digest_across_reused']}")
    A(f"- Prior-evidence pins unchanged at gate time: "
      f"{not gate['gates']['G0']['detail']['source_prior_drift']['changed']}")
    A("")
    A("## 7. Disclosed caveats")
    A("")
    A(f"- {src['disclosed_caveat']['package_init_side_effect']}")
    A(f"- {GC.load_json(EVID / 'ACCESS_AUDIT.json')['disclosed_caveat']}")
    if (EVID / "REDUMP_REPORT.json").is_file():
        rd = GC.load_json(EVID / "REDUMP_REPORT.json")
        n_stale = gate["gates"]["G0"]["detail"]["n_runs_whose_recorded_hash_differs_from_disk"]
        A(f"- VAL-prediction dump defect (found while verifying this stage, repaired in the "
          f"same change).  The trainer wrote each run's `val_predictions.npz` from the *live* "
          f"EMA rather than the selected-EMA snapshot, so for "
          f"{rd['n_runs'] - rd['n_own_npz_already_selected_ema']} of {rd['n_runs']} runs it "
          f"describes the run's last step instead of the model the freeze record describes.  "
          f"No gated quantity depends on that file: G0-G5 read the freeze records, and the "
          f"selection, `selected_ema.pt` and every recorded metric are computed on the "
          f"selected EMA and are unaffected.  The registered MECHANISM_DIAGNOSTICS.csv is "
          f"rebuilt from `val_predictions_selected_ema.npz`, a re-forward of the frozen "
          f"`selected_ema.pt` that writes a *new* file and overwrites no run artifact.  The "
          f"re-forward reproduces the recorded `selected_val_mae_ema` for all {rd['n_runs']} "
          f"runs (max |deviation| {rd['max_abs_deviation_from_recorded']:.2e}) and reproduces "
          f"the run's own npz exactly wherever that npz already was the selected pass "
          f"({rd['n_own_npz_matching_redump']}/{rd['n_own_npz_already_selected_ema']}), which "
          f"is what proves it rebuilds the runs' own inputs.  No fit was retrained and no "
          f"selection re-run.  Registry consequence: all {n_stale} registered runs carry a "
          f"`code_hash` for the pre-repair `trainer.py`, so their recorded hash no longer "
          f"matches the file on disk.  That drift is reported in GATE.json "
          f"(`runs_whose_recorded_hash_differs_from_disk`) and deliberately not gated: G0's "
          f"`one_scientific_code_state` asserts homogeneity *between* runs, which still holds "
          f"because all {n_stale} share the one pre-repair hash, and a repair to a "
          f"post-training code path cannot retroactively invalidate frozen fits.")
    if (EVID / "INDEPENDENT_VERIFICATION_REPORT.json").is_file():
        ver = GC.load_json(EVID / "INDEPENDENT_VERIFICATION_REPORT.json")
        A(f"- Independent verifier: {ver.get('verdict')} "
          f"({ver.get('n_checks_passed')}/{ver.get('n_checks')} checks)")
    A("")
    (EVID / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(EVID / "RESULTS.md"), "lines": len(lines)}


# ------------------------------------------------------------------------ verify
def cmd_verify(args) -> dict:
    script = VERIFY_DIR / "verify_canary.py"
    done = subprocess.run(
        [PYTHON, str(script), "--json", str(EVID / "INDEPENDENT_VERIFICATION_REPORT.json")],
        cwd=str(STAGE), capture_output=True, text=True,
    )
    print((done.stdout or "")[-4000:])
    if done.returncode != 0:
        print((done.stderr or "")[-4000:], file=sys.stderr)
    return {"returncode": int(done.returncode)}


# --------------------------------------------------------------------------- all
def cmd_all(args) -> dict:
    steps = []
    steps.append(("source", cmd_source(args)))
    steps.append(("reuse", cmd_reuse(args)))
    steps.append(("tests", cmd_tests(args)))
    if not steps[-1][1].get("all_passed") and not args.ignore_gate:
        print("[all] unit-test gate FAILED; no scientific fit is permitted", file=sys.stderr)
        return {"stopped_at": "tests", "steps": [s for s, _ in steps]}
    steps.append(("fits:critical", cmd_fits(_with(args, group="critical"))))
    steps.append(("aggregate", cmd_aggregate(args)))
    steps.append(("access", cmd_access(args)))
    gate = cmd_gate(args)
    steps.append(("gate", gate))
    steps.append(("token", cmd_token(args)))

    if gate["all_gates_passed"] and args.with_ablations:
        steps.append(("fits:ablation", cmd_fits(_with(args, group="ablation"))))
        steps.append(("aggregate", cmd_aggregate(args)))
        steps.append(("gate", cmd_gate(args)))
        steps.append(("token", cmd_token(args)))
    if gate["all_gates_passed"] and args.with_full20:
        steps.append(("fits:full20", cmd_fits(_with(args, group="full20"))))
        steps.append(("aggregate", cmd_aggregate(args)))
        steps.append(("gate", cmd_gate(args)))
        steps.append(("token", cmd_token(args)))
    steps.append(("report", cmd_report(args)))
    steps.append(("verify", cmd_verify(args)))
    return {"steps": [s for s, _ in steps], "token": GC.load_json(EVID / "STAGE_TOKEN.json")}


def _with(args, **overrides):
    clone = argparse.Namespace(**vars(args))
    for k, v in overrides.items():
        setattr(clone, k, v)
    return clone


# -------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=[
        "source", "reuse", "tests", "fits", "redump", "aggregate", "access", "gate",
        "token", "report", "verify", "all",
    ])
    ap.add_argument("--group", default="critical", choices=["critical", "ablation", "full20"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="fit only the first N specs (smoke only)")
    ap.add_argument("--no-smoke", action="store_true", help="skip the schedule smoke in the gate")
    ap.add_argument("--all-cells", action="store_true", help="pin all 20 O1 cells (E4)")
    ap.add_argument("--with-ablations", action="store_true")
    ap.add_argument("--with-full20", action="store_true")
    ap.add_argument("--ignore-gate", action="store_true",
                    help="run fits even if the unit gate failed (diagnosis only; never for evidence)")
    args = ap.parse_args()

    EVID.mkdir(parents=True, exist_ok=True)
    handler = {
        "source": cmd_source, "reuse": cmd_reuse, "tests": cmd_tests, "fits": cmd_fits,
        "redump": cmd_redump,
        "aggregate": cmd_aggregate, "access": cmd_access, "gate": cmd_gate,
        "token": cmd_token, "report": cmd_report, "verify": cmd_verify, "all": cmd_all,
    }[args.command]
    result = handler(args)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("terminal_token", "all_gates_passed", "status", "path",
                               "n_fit", "n_skipped", "n_error", "n_runs", "returns")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
