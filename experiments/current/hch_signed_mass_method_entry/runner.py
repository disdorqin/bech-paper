"""Command line for the signed-mass China-5 harness.

Nine commands, in increasing order of what they are allowed to touch:

``readiness``
    Resolve the 20 ``market x Host`` coordinates against their frozen artifacts
    and write ``HOST_INPUT_READINESS.csv``.  Reads no outcome.

``verify``
    Re-run the pre-execution verifier: legality, chronology, contract agreement,
    sealed-role counts, core provenance.  Writes ``PREEXECUTION_AUDIT.md``.

``smoke``
    Integration smoke on the **synthetic** substrate plus a one-cell plumbing
    check on the real data with a truncated epoch budget.  Writes structural
    invariants only -- no error metric is recorded, because a number computed
    from real outcomes during a landing stage would be a scientific result.

``run``
    The scientific execution.  It is refused unless ``--execute`` is given, and
    it refuses to touch ``DEV_EVAL`` unless the caller additionally passes
    ``--dev-eval-days N`` (N > 0) together with the execution authorization
    token.  Neither is supplied by this landing stage.

New in the scientific-execution stage:

``d0``
    PART C.  The geometry study: ``POST_TRAIN`` only, no training, all 20 cells.
    It also freezes the per-cell high-mass ``q90`` pair first, because that
    registration has to exist before any number that depends on it.

``grid``
    PART D.  The registered component grid -- 20 cells x 5 configurations x 3
    seeds, each evaluated on the complete ``DEV_EVAL`` partition of its own cell.
    It refuses to run without ``--dev-eval-all`` *and* the execution
    authorization token, and it exposes no epoch override: a shortened fit is not
    one of the 300 registered fits.

``aggregate``
    PART E-H.  Seed aggregation by cell median, the four leave-one-out component
    decisions, the frozen smallest method, the strict offline join and the six
    development gates.  Reads persisted evidence; fits nothing.

``frozen``
    PART F.  The smallest surviving method on all 20 cells.  When the derived
    vector coincides with a registered configuration the screen's persisted
    predictions are reused, after an identity check rather than on the strength
    of the name; otherwise it is fitted here through the grid path.

``verify-results``
    PART A6.  The independent read-only audit of the executed run.  It recomputes
    every reported number from the raw arrays with its own arithmetic and derives
    the verdict before reading the recorded one.

``run``'s ``--dev-eval-days`` path is now closed: it wrote no raw prediction
evidence and loaded no frozen threshold, so an evaluation taken through it would
be one the audit could never see.  Use ``grid``.

Nothing in this file decides anything scientific.  The default of every flag is
the one that reads nothing.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

if __package__ in (None, ""):  # allow ``python .../runner.py`` as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "experiments.current.hch_signed_mass_method_entry"

from . import calibration, config as C, metrics, metrics as _metrics  # noqa: E402
from . import raw_evidence as RE  # noqa: E402
from . import host_prediction_loader as H, oof, synthetic, training as T  # noqa: E402
from .china5_adapter import (assert_fitting_history_complete,  # noqa: E402
                             assert_history_causal, build_cell_dataset,
                             load_feature_source, load_market_contract,
                             verify_source_bytes)
from .contracts import (READINESS_FIELDS, HarnessError, LegalityError,  # noqa: E402
                        ReadAudit, readiness_row, write_csv)
from .core_bridge import core_provenance  # noqa: E402

EXECUTION_TOKEN = "SIGNED_MASS_METHOD_EXECUTION_AUTHORIZED"

TOKENS = {
    "ready": "SIGNED_MASS_CHINA5_METHOD_HARNESS_READY",
    "ready_pending": "SIGNED_MASS_CHINA5_METHOD_HARNESS_READY_WITH_EXTERNAL_HOSTS_PENDING",
    "not_ready": "SIGNED_MASS_CHINA5_METHOD_HARNESS_NOT_READY",
}

PACKAGE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(C.REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _git_state() -> Dict[str, Any]:
    def _run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", *args], cwd=str(C.REPO_ROOT),
                                 capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    return {
        "head": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("status", "--porcelain")),
    }


def _environment() -> Dict[str, Any]:
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in ("torch", "numpy", "pandas", "scipy"):
        try:
            env[module] = __import__(module).__version__
        except Exception:  # pragma: no cover - absence is recorded, not fatal
            env[module] = None
    return env


def ensure_evidence_dirs(root: Optional[Path] = None) -> Path:
    """Create the reserved evidence subdirectories, empty.

    The verifier creates them so the layout the entry protocol promises exists
    before any scientific run; it never populates them with results.

    The default comes from :func:`raw_evidence.evidence_root` rather than from
    ``C.EVIDENCE_ROOT`` directly, so this function and the writer guard agree on
    where the experiment's evidence lives.  Two answers to that question would let
    a run create its directories in one tree and write its artifacts into another
    -- and would make the guard's redirection seam untestable here.
    """
    base = Path(root) if root is not None else RE.evidence_root()
    base.mkdir(parents=True, exist_ok=True)
    for sub in C.EVIDENCE_SUBDIRS:
        (base / sub).mkdir(exist_ok=True)
    (base / "README.md").write_text(_EVIDENCE_README, encoding="utf-8")
    return base


_EVIDENCE_README = """# hch_signed_mass_method_20260912

Reserved by the signed-mass method entry protocol
(`docs/current/HCH_SIGNED_MASS_METHOD_EXPERIMENT_ENTRY_PROTOCOL_20260912.md`).

Subdirectories are created empty by `verify_preexecution.py`.  This landing stage
writes only into `00_protocol/` (frozen registrations) and `06_audits/`
(readiness and pre-execution audits).  `01_d0_geometry/`, `02_crossfit/`,
`03_component_screen/`, `04_frozen_method/` and `05_baseline_comparison/` stay
empty until a separately authorized scientific run fills them.

An empty directory here is a statement, not an omission: it records that no
result of that kind has been produced.
"""


def load_cell(market: str, host: str,
              audit: Optional[ReadAudit] = None):
    """Load one cell's contract, frozen panel and legal feature tensor."""
    audit = audit if audit is not None else ReadAudit()
    contract = load_market_contract(market)
    verify_source_bytes(contract)
    panel = H.load_host_panel(market, host, contract=contract, audit=audit)
    source = load_feature_source(contract, audit=audit)
    dataset = build_cell_dataset(market, host, contract=contract, panel=panel,
                                 source=source, audit=audit)
    return contract, panel, dataset, audit


# --------------------------------------------------------------------------
# readiness
# --------------------------------------------------------------------------
def cmd_readiness(args) -> int:
    readiness = H.build_readiness_table()
    fields = READINESS_FIELDS
    out_csv = (Path(args.out) if args.out
               else PACKAGE_DIR / "HOST_INPUT_READINESS.csv")
    write_csv(readiness, out_csv, fields)
    summary = H.readiness_summary(readiness)

    evidence = ensure_evidence_dirs()
    target = evidence / "06_audits" / "HOST_INPUT_READINESS.csv"
    write_csv(readiness, target, fields)
    (evidence / "06_audits" / "HOST_INPUT_READINESS_SUMMARY.json").write_text(
        json.dumps({"generated_utc": _utc_now(), **summary,
                    "csv": _repo_relative(out_csv),
                    "evidence_copy": _repo_relative(target)},
                   ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")

    print(f"readiness: {summary['n_coordinates']} coordinates, "
          f"{summary['n_ready_frozen']} READY_FROZEN")
    for row in readiness:
        if row["status"] != "READY_FROZEN":
            print(f"  {row['market']}/{row['host']}: {row['status']} "
                  f"-- {row['blocker']}")
    print(f"wrote {_repo_relative(out_csv)}")
    return 0 if summary["n_ready_frozen"] == summary["n_coordinates"] else 1


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------
def cmd_smoke(args) -> int:
    """Integration smoke: synthetic pipeline + one real cell's plumbing."""
    import time

    report: Dict[str, Any] = {
        "kind": "INTEGRATION_SMOKE",
        "not_a_scientific_result": True,
        "reason": ("this stage may run unit/integration/synthetic smoke checks "
                   "only; no error metric computed from real outcomes is "
                   "recorded here"),
        "generated_utc": _utc_now(),
        "checks": [],
    }

    def check(name: str, ok: bool, detail: Any = None) -> None:
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            raise HarnessError(f"smoke check failed: {name} ({detail})")

    # -- 1. synthetic end-to-end, all five configurations -------------------
    ds = synthetic.synthetic_dataset(n_features=6, seed=args.seed)
    plan = oof.plan_folds(ds)
    check("synthetic.fold_plan_expands", plan.n_folds >= 1,
          {"n_folds": plan.n_folds, "n_oof_rows": int(plan.oof_rows.size)})
    check("synthetic.history_causal", True, assert_history_causal(ds))
    check("synthetic.fitting_history_complete", True,
          assert_fitting_history_complete(ds))

    for name, switches in C.CONFIG_VARIANTS.items():
        started = time.time()
        method = oof.fit_method(ds, switches, args.seed, max_epochs=args.smoke_epochs)
        check(f"synthetic.fit.{name}", method.alpha >= 0.0,
              {"alpha": method.alpha, "folds": len(method.folds),
               "final_epochs": method.epoch_rule["value"],
               "seconds": round(time.time() - started, 2)})
        out = method.predict(ds, ds.positions(C.ROLE_DEV_EVAL))
        check(f"synthetic.predict.{name}",
              bool(np.isfinite(out["prediction"]).all()),
              {"shape": list(out["prediction"].shape)})

    # An ablation must differ from FULL in exactly one switch.
    for name, switches in C.CONFIG_VARIANTS.items():
        delta = C.VARIANT_DELTA[name]
        diff = {k for k in C.SWITCH_NAMES if switches[k] != C.FULL_SWITCHES[k]}
        expected = set() if delta is None else {delta}
        check(f"synthetic.variant_delta.{name}", diff == expected,
              {"differing_switches": sorted(diff)})

    # -- 2. one real cell: plumbing only, truncated epochs, no metrics ------
    market, host = args.cell_market, args.cell_host
    started = time.time()
    contract, panel, dataset, audit = load_cell(market, host)
    check("real.cell_loaded", dataset.n_features == contract.n_features,
          {"market": market, "host": host, "F": dataset.n_features,
           "n_episodes": dataset.n_episodes,
           "seconds": round(time.time() - started, 2)})
    check("real.history_causal", True, assert_history_causal(dataset))
    check("real.fitting_history_complete", True,
          assert_fitting_history_complete(dataset))

    method = oof.fit_method(dataset, C.CONFIG_VARIANTS["FULL"], args.seed,
                            max_epochs=args.smoke_epochs)
    check("real.fit_method_runs", method.alpha >= 0.0,
          {"alpha": method.alpha, "n_folds": len(method.folds),
           "oof_rows": method.oof_plan["n_oof_rows"]})
    check("real.alpha_fitted_on_oof_only",
          method.alpha_objective["fit_rows"].startswith("chronological out-of-fold"),
          method.alpha_objective["fit_rows"])
    check("real.prediction_finite",
          bool(np.isfinite(method.predict(dataset,
                                          dataset.positions(C.ROLE_DEV_EVAL)
                                          )["prediction"]).all()))
    check("real.audit_zero_forbidden_reads",
          audit.target_day_price_input_reads == 0
          and audit.realized_future_input_reads == 0
          and audit.protected_final_reads == 0, audit.as_dict())

    evidence = ensure_evidence_dirs()
    path = evidence / "06_audits" / "SMOKE_AUDIT.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8")
    print(f"smoke: {len(report['checks'])} checks passed "
          f"({len(C.CONFIG_VARIANTS)} configurations, synthetic + {market}/{host})")
    print(f"wrote {_repo_relative(path)}")
    return 0


# --------------------------------------------------------------------------
# run (scientific execution -- gated)
# --------------------------------------------------------------------------
def _execution_plan(readiness: Sequence[Dict[str, Any]], markets, hosts) -> List[Dict[str, str]]:
    market_set = set(markets or C.MARKETS)
    host_set = set(hosts or C.HOSTS)
    unknown_m = market_set - set(C.MARKETS)
    unknown_h = host_set - set(C.HOSTS)
    if unknown_m or unknown_h:
        raise HarnessError(f"unknown market(s) {sorted(unknown_m)} / "
                           f"host(s) {sorted(unknown_h)}")
    plan = [{"market": r["market"], "host": r["host"]}
            for r in readiness
            if r["market"] in market_set and r["host"] in host_set]
    blocked = [r for r in readiness
               if r["market"] in market_set and r["host"] in host_set
               and r["status"] != "READY_FROZEN"]
    if blocked:
        raise HarnessError(
            "refusing to execute over non-ready coordinates: "
            + ", ".join(f"{r['market']}/{r['host']}={r['status']}" for r in blocked))
    return plan


def cmd_run(args) -> int:
    readiness = H.build_readiness_table()
    plan = _execution_plan(readiness, args.markets, args.hosts)
    configs = list(C.CONFIG_VARIANTS) if args.configs in (None, ["ALL"]) else args.configs
    for name in configs:
        if name not in C.CONFIG_VARIANTS:
            raise HarnessError(f"unknown configuration {name!r}")

    dev_days = int(args.dev_eval_days)
    authorized = args.authorization_token == EXECUTION_TOKEN
    if dev_days:
        raise HarnessError(
            "--dev-eval-days is closed.  This path evaluated a prefix of "
            "DEV_EVAL while writing no raw prediction arrays and loading no "
            "frozen high-mass threshold, so a number produced here could not be "
            "recomputed from evidence or checked against a pre-registered "
            "subset.  Use ``grid --dev-eval-all``, which evaluates the complete "
            "partition of every cell and persists the arrays.")
    if not args.execute:
        print("plan (nothing executed; pass --execute to run):")
        print(f"  cells        : {len(plan)}")
        print(f"  configurations: {configs}")
        print(f"  seeds        : {list(C.TRAINING['seeds'])}")
        print("  dev_eval     : closed on this command; use ``grid --dev-eval-all``")
        return 0

    evidence = ensure_evidence_dirs()
    manifests: List[Dict[str, Any]] = []
    for cell in plan:
        contract, panel, dataset, audit = load_cell(cell["market"], cell["host"])
        for config_name in configs:
            switches = C.CONFIG_VARIANTS[config_name]
            for seed in C.TRAINING["seeds"]:
                method = oof.fit_method(dataset, switches, int(seed))
                record = _cell_record(method, dataset, config_name, contract)
                manifests.append(record)
                _write_run_evidence(evidence, record)

    manifest_path = evidence / "00_protocol" / "RUN_MANIFEST.json"
    manifest_path.write_text(json.dumps(
        _run_manifest(manifests, plan, configs, dev_days, authorized),
        ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"executed {len(manifests)} (cell, config, seed) fits")
    print(f"wrote {_repo_relative(manifest_path)}")
    return 0


def _cell_record(method, dataset, config_name: str, contract) -> Dict[str, Any]:
    return {
        "config": config_name,
        "market": method.market,
        "host": method.host,
        "seed": method.seed,
        "alpha": method.alpha,
        "alpha_objective": method.alpha_objective,
        "epoch_rule": method.epoch_rule,
        "oof": {k: v for k, v in method.oof_plan.items() if k != "folds"},
        "oof_correction_sha256": method.oof_correction_sha256,
        "fitted_stats": method.stats.as_dict(),
        "folds": method.folds,
        "contract_sha256": contract.contract_sha256,
        "source_sha256": contract.source_sha256,
        "n_features": dataset.n_features,
        "n_episodes": dataset.n_episodes,
    }


def _write_run_evidence(evidence: Path, record: Dict[str, Any]) -> None:
    sub = "04_frozen_method" if record["config"] == "FULL" else "03_component_screen"
    target = evidence / sub / f"{record['market']}__{record['host']}__{record['config']}__seed{record['seed']}.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True),
                      encoding="utf-8")


def _run_manifest(records, plan, configs, dev_days, authorized) -> Dict[str, Any]:
    return {
        "schema": "signed_mass_china5_run_manifest.v1",
        "generated_utc": _utc_now(),
        "cells": plan,
        "configurations": configs,
        "config_switches": {k: dict(v) for k, v in C.CONFIG_VARIANTS.items()},
        "seeds": list(C.TRAINING["seeds"]),
        "dev_eval_days": int(dev_days),
        "dev_eval_authorized": bool(authorized),
        # ``seeds`` is registered as a tuple; emit it as a list so the manifest
        # is JSON-shaped in memory and not only after ``json.dumps``.
        "training": {**C.TRAINING, "seeds": list(C.TRAINING["seeds"])},
        "loss_weights": dict(C.LOSS_WEIGHTS),
        "oof_rule": {
            "min_train_days": C.OOF_MIN_TRAIN_DAYS,
            "min_holdout_days": C.OOF_MIN_HOLDOUT_DAYS,
            "max_folds": C.OOF_MAX_FOLDS,
            "inner_val_fraction": C.OOF_INNER_VAL_FRACTION,
        },
        "thresholds_source": _repo_relative(C.THRESHOLD_FREEZE),
        "core": core_provenance(),
        "environment": _environment(),
        "git": _git_state(),
        "n_fits": len(records),
        "fits": [{k: r[k] for k in ("config", "market", "host", "seed", "alpha",
                                    "oof_correction_sha256")} for r in records],
    }


# --------------------------------------------------------------------------
# d0 / grid / aggregate / verify-results  (PART C, D, E-H, A6)
# --------------------------------------------------------------------------
def _ready_cells(args) -> List[Dict[str, str]]:
    """The readiness-gated coordinate list, refusing a partial panel quietly."""
    readiness = H.build_readiness_table()
    return _execution_plan(readiness, getattr(args, "markets", None),
                           getattr(args, "hosts", None))


def _dataset_for(market: str, host: str):
    """The cached cell *dataset*, through the grid's read-audited loader.

    ``grid.dataset_for`` hands back ``(dataset, audit)`` because a fit needs the
    one audit object that actually saw the loads.  Both callers here -- the A3
    threshold freeze and D0 -- consume a dataset and then measure it; neither
    reasons about an audit, so the tuple is unwrapped at this seam rather than
    pushed into those callers.  The unwrap does not move the guard that matters:
    the loader itself raises when ``PROTECTED_FINAL`` reads are non-zero, before
    the tuple is ever returned.
    """
    from . import grid

    dataset, _audit = grid.dataset_for(market, host)
    return dataset


def freeze_high_mass_thresholds(cells: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    """Write the A3 registration, or refuse to disturb an existing one.

    A re-freeze that produced *different* numbers would mean the registration had
    already been read by something, and silently replacing it would rewrite the
    definition of the ``high_mass_*`` subset after metrics had been computed
    against the old one.  An identical re-freeze is a no-op, which is what makes
    this safe to call from a resumed run.
    """
    from . import protocol

    records = protocol.build_threshold_records(
        [(c["market"], c["host"]) for c in cells],
        lambda market, host: _dataset_for(market, host))
    payload = {
        "schema": protocol.HIGH_MASS_SCHEMA,
        "generated_utc": _utc_now(),
        "quantile": C.HIGH_MASS_QUANTILE,
        "fit_partition": C.ROLE_POST_TRAIN,
        "rule": ("per cell, the q-quantile of the realised signed mass A^+ and "
                 "A^- over that cell's POST_TRAIN rows, computed from the frozen "
                 "Host's own residuals; never recomputed from an evaluated "
                 "partition"),
        "n_cells": len(records),
        "cells": records,
    }

    target = protocol.high_mass_path()
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("cells") != records:
            raise HarnessError(
                f"{target} already holds a different high-mass registration; "
                "refusing to replace a frozen definition of the subset the "
                "high-mass metrics are reported on")
        print(f"high-mass thresholds: unchanged ({len(records)} cells)")
        return existing

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                 sort_keys=True, default=str), encoding="utf-8")
    unusable = protocol.unusable_records(payload)
    print(f"high-mass thresholds: froze {len(records)} cells "
          f"({len(unusable)} degenerate, reported as NOT_APPLICABLE_N0)")
    return payload


def cmd_d0(args) -> int:
    """PART C: the geometry study, before any network fit exists."""
    from . import d0

    cells = _ready_cells(args)
    evidence = ensure_evidence_dirs()
    freeze_high_mass_thresholds(cells)
    result = d0.run([(c["market"], c["host"]) for c in cells],
                    build=lambda market, host: _dataset_for(market, host))
    print(f"d0: {len(result['cells'])} cells analysed, max reconstruction error "
          f"{result['max_reconstruction_abs_err']:.3e}")
    print(f"wrote {_repo_relative(evidence / '01_d0_geometry')}")
    return 0


def cmd_grid(args) -> int:
    """PART D: the 300-fit registered component grid."""
    from . import grid, protocol

    if not args.dev_eval_all:
        raise HarnessError(
            "the component grid evaluates every fit on the complete DEV_EVAL "
            "partition; pass --dev-eval-all to say so explicitly")
    if args.authorization_token != EXECUTION_TOKEN:
        raise HarnessError(
            f"the component grid requires --authorization-token {EXECUTION_TOKEN}")

    if protocol.load_thresholds() is None:
        raise HarnessError(
            "no frozen high-mass threshold registration is present; run the "
            "``d0`` command first, which freezes it before the grid can produce "
            "a high-mass metric")

    cells = _ready_cells(args)
    evidence = ensure_evidence_dirs()
    configs = (list(C.CONFIG_VARIANTS) if args.configs in (None, ["ALL"])
               else args.configs)
    result = grid.run_grid(
        [(c["market"], c["host"]) for c in cells],
        configs=configs,
        workers=int(args.workers) if args.workers else grid.default_workers(),
        resume=not args.no_resume,
        progress=lambda message: print(message, flush=True))
    summary = result["summary"]
    print(f"grid: {summary['n_records']} fits "
          f"({summary['n_resumed']} resumed, {summary['n_failed']} failed) in "
          f"{summary['elapsed_seconds']}s, workers={summary['workers']}")
    print(f"wrote {_repo_relative(Path(result['status_table']))}")
    if summary["n_failed"]:
        print("failures are recorded in the status table; the aggregation refuses "
              "to decide on an incomplete panel")
        return 1
    print(f"wrote {_repo_relative(evidence / '02_crossfit')}")
    return 0


def cmd_aggregate(args) -> int:
    """PART E-H: adjudicate, freeze, compare and gate from persisted evidence.

    The records are the component screen plus, when the frozen method was
    actually fitted, the frozen run.  Both are needed in one table: the four
    adjudications read the five registered configurations, while the comparison
    and the gates read the frozen one, and a table containing only the screen
    would silently compare a method that was never run.
    """
    from . import aggregate

    root = RE.evidence_root()
    status = root / "02_crossfit" / "GRID_STATUS.csv"
    records = list(aggregate.read_csv(status))
    if not records:
        raise HarnessError(
            f"no component-grid results at {status}; run the ``grid`` command "
            "first -- there is nothing to aggregate")

    config = args.config
    frozen_config_path = root / "04_frozen_method" / "FROZEN_METHOD_CONFIG.json"
    if config is None and frozen_config_path.exists():
        frozen = json.loads(frozen_config_path.read_text(encoding="utf-8"))
        config = frozen.get("frozen_label")
        print(f"aggregate: frozen method {config} via {frozen.get('route')}")
        if frozen.get("route") == "FITTED_DERIVED_VECTOR":
            frozen_status = root / "04_frozen_method" / "FROZEN_STATUS.csv"
            frozen_records = aggregate.read_csv(frozen_status)
            if not frozen_records:
                raise HarnessError(
                    f"the frozen-method record names route "
                    f"FITTED_DERIVED_VECTOR but {frozen_status} is missing; the "
                    "frozen method has no predictions to aggregate")
            records += frozen_records
            print(f"aggregate: +{len(frozen_records)} frozen-method fits")
        else:
            print("aggregate: the frozen vector reuses registered predictions "
                  "from the screen; nothing is added")
    elif config is None:
        print("aggregate: no frozen-method record; the frozen configuration is "
              "taken from the component decisions")

    print(f"aggregate: {len(records)} persisted fits")
    result = aggregate.run(records, config=config)
    print(f"frozen configuration: {result['frozen_config']}")
    print(f"components retained: {result['vector']['retained_components']}")
    print(f"deleted: {result['vector']['deleted_components'] or 'none'}")
    for decision in result["decisions"]:
        print(f"  {decision['component']:<26} {decision['decision']:<9} "
              f"{decision['reason']}")
    for name, gate in result["gates"]["gates"].items():
        print(f"  {'PASS' if gate['pass'] else 'FAIL'}  {name}")
    print(f"verdict: {result['verdict']}")
    return 0


def cmd_frozen(args) -> int:
    """PART F: materialise the smallest surviving method on all 20 cells.

    The vector is *derived* from the screen's own records here rather than read
    from a verdict file, so the frozen run does not depend on an artifact that
    can only be written after it.  It is the same
    :func:`aggregate.component_decisions` / :func:`aggregate.frozen_vector` pair
    the final aggregation uses, so the run and the report cannot disagree about
    which components survived.

    Reuse is allowed only for a vector that *is* one of the registered
    configurations and only after an identity check against the persisted
    artifact -- the name matching is not evidence that the recipe matched.  When
    two or more components are deleted the vector is usually not a registered
    name at all, and then it is genuinely fitted here, through the same code path
    the screen used, into ``04_frozen_method``.
    """
    from . import aggregate, grid

    if args.authorization_token != EXECUTION_TOKEN:
        raise HarnessError(
            f"the frozen-method run requires --authorization-token {EXECUTION_TOKEN}")

    root = RE.evidence_root()
    screen_path = root / "02_crossfit" / "GRID_STATUS.csv"
    screen = aggregate.read_csv(screen_path)
    if not screen:
        raise HarnessError(
            f"no component-screen results at {screen_path}; the frozen method is "
            "derived from the four adjudications rather than chosen here")
    decisions = aggregate.component_decisions(screen)
    derived = aggregate.frozen_vector(decisions)
    vector = dict(derived["switches"])
    matches = derived["matches_registered_config"]
    print("frozen vector derived from the screen: "
          + ", ".join(f"{d['component']}={d['decision']}" for d in decisions))
    print(f"  retained {derived['retained_components']} "
          f"deleted {derived['deleted_components']}")

    cells = _ready_cells(args)
    evidence = ensure_evidence_dirs()
    out_dir = evidence / "04_frozen_method"
    out_dir.mkdir(parents=True, exist_ok=True)

    if matches:
        identity = _reuse_identity(screen, matches, vector)
        _write_frozen_config(out_dir / "FROZEN_METHOD_CONFIG.json", {
            "schema": "signed_mass_frozen_method.v1",
            "generated_utc": _utc_now(),
            "route": "REUSED_REGISTERED_CONFIG",
            "frozen_label": matches,
            "registered_config": matches,
            "switches": vector,
            "seeds": [int(s) for s in C.TRAINING["seeds"]],
            "n_cells": len(cells),
            "identity_check": identity,
            "source_status_table": _repo_relative(screen_path),
            "note": ("the frozen vector coincides with an executed registered "
                     "configuration; the screen's persisted predictions are "
                     "reused, and the identity check above is what makes the "
                     "reuse legitimate"),
        })
        ok = identity["all_identities_match"]
        print(f"frozen method: reuses registered {matches} "
              f"({identity['n_checked']} artifacts checked, "
              f"identity {'confirmed' if ok else 'FAILED'})")
        if not ok:
            raise HarnessError(
                "the frozen vector names a registered configuration but the "
                f"persisted artifacts do not all carry its switch vector; "
                f"mismatches: {identity['mismatches'][:5]}")
        return 0

    result = grid.run_grid(
        [(c["market"], c["host"]) for c in cells],
        configs=[_FROZEN_NAME],
        workers=int(args.workers) if args.workers else grid.default_workers(),
        resume=not args.no_resume,
        raw_subdir=C.RAW_DIR_FROZEN,
        metric_subdir=C.METRIC_DIR_FROZEN,
        status_name="FROZEN_STATUS.csv",
        out_subdir="04_frozen_method",
        variants={_FROZEN_NAME: vector},
        progress=lambda message: print(message, flush=True))
    summary = result["summary"]
    _write_frozen_config(out_dir / "FROZEN_METHOD_CONFIG.json", {
        "schema": "signed_mass_frozen_method.v1",
        "generated_utc": _utc_now(),
        "route": "FITTED_DERIVED_VECTOR",
        "frozen_label": _FROZEN_NAME,
        "registered_config": None,
        "switches": vector,
        "seeds": [int(s) for s in C.TRAINING["seeds"]],
        "n_cells": len(cells),
        "status_table": _repo_relative(Path(result["status_table"])),
        "n_records": summary["n_records"],
        "n_failed": summary["n_failed"],
        "note": ("the frozen vector is not one of the five registered "
                 "configurations, so it was fitted here through the registered "
                 "grid path rather than reconstructed from the ablations"),
    })
    print(f"frozen method: {summary['n_records']} fits "
          f"({summary['n_resumed']} resumed, {summary['n_failed']} failed)")
    return 1 if summary["n_failed"] else 0


def _write_frozen_config(path: Path, body: Mapping[str, Any]) -> None:
    """Write the frozen-method record.

    Rewriting it is allowed and expected: a re-run of the same screen must
    reproduce the same derived vector, and the file describes which run produced
    the predictions actually on disk.  The vector it names is derived from the
    screen every time, so a rewrite cannot smuggle in a different method -- it
    can only record a different *route* to the same one.

    It is written through the same confinement guard as every other artifact in
    the package: the verifier reads the frozen label and the route out of this
    record, so it has to be as impossible to place outside the evidence root as a
    raw prediction is.
    """
    target = RE.confine(path, "frozen method record")
    target.write_text(json.dumps(body, ensure_ascii=False, indent=2,
                                 sort_keys=True, default=str), encoding="utf-8")


_FROZEN_NAME = "FROZEN_SMALLEST"


def _reuse_identity(screen: Sequence[Mapping[str, Any]], config: str,
                    vector: Mapping[str, bool]) -> Dict[str, Any]:
    """Prove the persisted screen rows are the frozen vector's own fits.

    A registered name is a label, and a label can be wrong.  The record has to
    carry the vector, and it has to carry *this* vector, before its predictions
    may stand in for a run that was never performed.
    """
    rows = [r for r in screen
            if str(r.get("config")) == config
            and str(r.get("status", "")).startswith("OK")]
    mismatches = []
    for row in rows:
        recorded = row.get("switches")
        if isinstance(recorded, str):
            try:
                recorded = json.loads(recorded)
            except ValueError:
                recorded = None
        if dict(recorded or {}) != dict(vector):
            mismatches.append(f"{row.get('cell')}::s{row.get('seed')}: "
                              f"{recorded}")
    n_seeds = len(C.TRAINING["seeds"])
    expected = len(C.CELLS) * n_seeds
    return {
        "registered_config": config,
        "switches": dict(vector),
        "n_checked": len(rows),
        "n_expected": expected,
        "all_identities_match": bool(rows) and not mismatches
        and len(rows) == expected,
        "mismatches": mismatches,
    }


def cmd_verify_results(args) -> int:
    """PART A6: the independent read-only audit."""
    from . import verify_results

    return verify_results.run(host_recheck=not args.skip_host_recheck)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hch_signed_mass_method_entry.runner",
        description="Signed-mass China-5 method harness (integration only).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ready = sub.add_parser("readiness", help="write HOST_INPUT_READINESS.csv")
    p_ready.add_argument("--out", default=None)
    p_ready.set_defaults(func=cmd_readiness)

    p_verify = sub.add_parser("verify", help="run the pre-execution verifier")
    p_verify.add_argument("--out-dir", default=None)
    p_verify.set_defaults(func=None)   # bound in verify_preexecution.main

    p_smoke = sub.add_parser("smoke", help="integration smoke (synthetic + plumbing)")
    p_smoke.add_argument("--seed", type=int, default=7)
    p_smoke.add_argument("--smoke-epochs", type=int, default=2)
    p_smoke.add_argument("--cell-market", default="GANSU_DA")
    p_smoke.add_argument("--cell-host", default="PatchTST")
    p_smoke.set_defaults(func=cmd_smoke)

    p_run = sub.add_parser("run", help="scientific execution (gated)")
    p_run.add_argument("--execute", action="store_true",
                       help="actually fit; without it the runner only prints the plan")
    p_run.add_argument("--configs", nargs="*", default=None)
    p_run.add_argument("--markets", nargs="*", default=None)
    p_run.add_argument("--hosts", nargs="*", default=None)
    p_run.add_argument("--dev-eval-days", type=int, default=0)
    p_run.add_argument("--authorization-token", default="")
    p_run.set_defaults(func=cmd_run)

    p_d0 = sub.add_parser("d0", help="PART C: POST_TRAIN-only geometry study")
    p_d0.add_argument("--markets", nargs="*", default=None)
    p_d0.add_argument("--hosts", nargs="*", default=None)
    p_d0.set_defaults(func=cmd_d0)

    p_grid = sub.add_parser("grid", help="PART D: the 20 x 5 x 3 component grid")
    p_grid.add_argument("--dev-eval-all", action="store_true",
                        help="required: every fit is evaluated on the complete "
                             "DEV_EVAL partition of its own cell")
    p_grid.add_argument("--configs", nargs="*", default=None)
    p_grid.add_argument("--markets", nargs="*", default=None)
    p_grid.add_argument("--hosts", nargs="*", default=None)
    p_grid.add_argument("--workers", type=int, default=0)
    p_grid.add_argument("--no-resume", action="store_true",
                        help="re-fit instead of reusing an identical artifact")
    p_grid.add_argument("--authorization-token", default="")
    p_grid.set_defaults(func=cmd_grid)

    p_agg = sub.add_parser("aggregate",
                           help="PART E-H: adjudicate, freeze, compare, gate")
    p_agg.add_argument("--config", default=None,
                       help="the frozen switch vector's registered name; "
                            "inferred from the decisions when omitted")
    p_agg.set_defaults(func=cmd_aggregate)

    p_frozen = sub.add_parser("frozen",
                              help="PART F: the smallest surviving method, 20 cells")
    p_frozen.add_argument("--markets", nargs="*", default=None)
    p_frozen.add_argument("--hosts", nargs="*", default=None)
    p_frozen.add_argument("--workers", type=int, default=0)
    p_frozen.add_argument("--no-resume", action="store_true")
    p_frozen.add_argument("--authorization-token", default="")
    p_frozen.set_defaults(func=cmd_frozen)

    p_vr = sub.add_parser("verify-results",
                          help="PART A6: independent audit of the executed run")
    p_vr.add_argument("--skip-host-recheck", action="store_true",
                      help="skip reloading the frozen Host panels; this is the "
                           "one check that can detect a retrained Host, so "
                           "skipping it weakens the audit")
    p_vr.set_defaults(func=cmd_verify_results)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        from . import verify_preexecution as V
        return V.run(out_dir=args.out_dir)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
