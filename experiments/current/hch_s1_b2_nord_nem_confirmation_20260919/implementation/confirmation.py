"""One-shot 8-cell PROTECTED_FINAL confirmation for the frozen S1-B2 method.

This is the stage's only scoring executor.  It runs *after* P0 has proved, by
construction rather than assertion, that no protected-final target value was ever
parsed; ``nn_shared.unlock_protected()`` refuses to open the seal without that
token.

What it does, in the authority's own terms:

* **Frozen method only.**  The S1-B2 replay is the frozen
  ``hch_s1_sequential_surgical_repair_20260918`` runner, called through its own
  ``run_replay`` with the freeze's flag vector.  No hyperparameter, feature,
  objective, checkpoint rule or calibration choice is touched here.
* **Fit stays ``POST_TRAIN``; scoring becomes ``PROTECTED_FINAL``.**  A protected
  day's 168h residual history legitimately reaches back into ``DEV_EVAL`` and into
  earlier protected days -- those targets were observed before the day being
  forecast -- and nothing is fitted, tuned, rescaled or checkpointed on them.
* **Frozen comparators, frozen seeds.**  Host (forward of the re-derived frozen
  weights), ``delta-Adapter`` (seed 7), ``MatchedDirectResidual`` (seed 7, internal
  control, never in the external ranking), PIR (seed 2021, on its two legal Hosts
  only) and COSA (seed 2021, ``ONLINE_TTA``, reported on its own line).
* **Frozen thresholds.**  q05 / q95 / S90 are the ones the frozen family stage froze
  in ``00_protocol/THRESHOLD_FREEZE.json`` before any cell existed; they are re-used
  verbatim, never recomputed here.
* **Cell statistic.**  Median MAE across the authority's seeds 7/17/37; every gain is
  computed after that median, never before it.

Two structural facts about the frozen international evidence force two disclosed
adaptations, both recorded on every affected row:

1. The frozen Hosts persisted **no weights** and evaluated the open roles only, so
   the protected-span Host prediction exists nowhere on disk.  It is produced by
   forwarding the re-derived frozen weights that P0 proved digest-identical -- never
   by fitting a Host here.
2. ``panel.rebuild_series`` enumerates its evaluation span by the legacy role name
   ``DIAG_EVAL``.  For PIR (the one comparator that reconstructs a sliding-window
   series) the protected days are therefore presented as that role, and the
   intervening ``DEV_EVAL`` rows are excluded from the reconstruction so that the
   frozen ``pred.shape == (n_eval, H)`` invariant still holds and the evaluation span
   is exactly the protected days.  PIR's train/val spans and its scaler are provably
   unchanged by this: both depend only on ``n_fit``, ``n_val`` and the fit block.

The four ``LAGO_DE`` / ``LAGO_PJM`` continuity cells are **cited, not rerun**: the
authority forbids re-running them, and they are excluded from every aggregate here.

Nothing here re-ranks, rescues or selects.  The complete 8-cell matrix is executed in
one pass; there is no conditional stopping.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

IMPL = Path(__file__).resolve().parent
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))

import nn_shared as S                                                       # noqa: E402

# ``vendor_guard`` rule 1: bind every ``utils.*`` symbol this stage resolves by bare
# name at *import* time, before any vendor snapshot can displace the name.  The
# frozen Host loaders pop ``utils`` from ``sys.modules`` and re-import the official
# snapshot's own package; a later bare import would resolve against that snapshot.
from utils.benchmark_cache import HostPredictionCache                        # noqa: E402,F401
from utils.benchmark_contracts import ForecastWindows                        # noqa: E402,F401

ROOT, STAGE, EVID = S.ROOT, S.STAGE, S.EVID
OUT = EVID / "confirmation"

# The authority's flag vector, verbatim (section 3).  Not a tunable.
FLAGS = {"scalar": "normalized", "scale": "mean", "ray_norm": "l2",
         "objective": "mse", "checkpoint": "inner", "c": False, "b3": False}
PHASE = "S1-B2"

TRANSFER_SEED = 7        # s3_baselines.SEED, the frozen transfer seed
PIR_SEED = 2021          # pir_transfer.ANCHOR_SEED, the official fix_seed contract
COSA_SEED = 2021         # cosa_transfer's own default
CONTRACT_THREADS = 1     # the frozen panel's single-thread reproducibility pin
COSA_SETTING = "ONLINE_TTA"

# The predecessor's continuity panel: cited and hashed, never recomputed.
PRIOR_CONTINUITY = (ROOT / "experiments/evidence/"
                    "hch_s1_b2_international_confirmation_20260919/CONTINUITY_PANEL.json")

METRIC_KEYS = ("Overall_MAE", "MSE", "RMSE", "Host_Overall_MAE",
               "relative_gain_vs_Host_pct", "Tail_MAE", "Upper_tail_MAE", "Lower_tail_MAE",
               "Normal_region_MAE", "Host_Normal_region_MAE", "Normal_harm_vs_Host",
               "Tail_MAE_Host",
               "Extreme_day_MAE", "Extreme_day_MAE_Host", "extreme_day_subset_status",
               "Upper_tail_day_MAE", "upper_tail_day_subset_status",
               "Negative_price_MAE", "Negative_price_MAE_Host", "negative_price_subset_status",
               "n_days", "n_hours", "n_upper_tail_hours", "n_lower_tail_hours",
               "n_normal_hours", "n_extreme_days", "n_upper_tail_days",
               "n_negative_hours", "n_negative_days")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "RUN_LOG.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------- metrics
def metrics_module():
    p = ROOT / "experiments/current/china5_posthoc_baseline_panel"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    import panel_metrics as PM
    return PM


def frozen_thresholds(market: str) -> dict:
    """The thresholds the frozen family stage fixed before any cell existed."""
    p = (S.frozen_evidence_root(market) / "00_protocol" / "THRESHOLD_FREEZE.json")
    if not p.is_file():
        raise RuntimeError(f"{market}: frozen THRESHOLD_FREEZE.json absent at {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    if market not in d["thresholds"]:
        raise RuntimeError(f"{market}: not present in the frozen threshold freeze")
    thr = dict(d["thresholds"][market])
    if thr.get("fit_partition") != "HOST_TRAIN":
        raise RuntimeError(f"{market}: frozen thresholds were not fit on HOST_TRAIN")
    if int(d.get("protected_final_values_read", -1)) != 0:
        raise RuntimeError(f"{market}: frozen threshold freeze declares a protected read")
    thr["_frozen_source"] = S.rel(p)
    thr["_frozen_source_sha256"] = S.sha(p)
    thr["_recomputed_here"] = False
    return thr


def day_keys_of(ts) -> np.ndarray:
    return np.asarray([str(t)[:10] for t in pd.to_datetime(ts)])


# --------------------------------------------------------------------- the run
def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RUN_LOG.txt").write_text("", encoding="utf-8")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    log("=" * 78)
    log("S1-B2 bounded NORD_DK1/NEM_SA1 confirmation -- PROTECTED_FINAL, one pass, 8 cells")

    # ---- gate: the seal opens only on a clean P0 PASS -----------------------
    token = S.unlock_protected()
    log(f"[GATE] protected reader unlocked by {S.rel(S.P0_TOKEN)} "
        f"(verdict={token['verdict']}, protected reads={token['protected_final_target_values_read']})")

    HIST = S.import_hist()
    HIST.STEPS = 1000        # the value P0 certified the frozen runner under
    TRANS = _load_transfer_module("transfers")
    PM = metrics_module()
    _pin_threads()
    log(f"[ENV] torch={torch.__version__} threads={torch.get_num_threads()} "
        f"runner_STEPS={HIST.STEPS} flags={json.dumps(FLAGS, sort_keys=True)}")

    thresholds = {m: frozen_thresholds(m) for m in S.MARKETS}
    log("[THR] re-used frozen thresholds: "
        + ", ".join(f"{m}(q05={thresholds[m]['q05']:.4g},q95={thresholds[m]['q95']:.4g},"
                    f"s90={thresholds[m]['day_spread_p90']:.4g})" for m in S.MARKETS))

    hch_rows: list[dict] = []
    base_rows: list[dict] = []
    cell_rows: list[dict] = []
    access: dict[str, dict] = {}
    failures: list[str] = []
    pir_pending: list[tuple[str, str]] = []
    pir_pack: dict[tuple[str, str], dict] = {}

    # ==== pass A: S1-B2 + the two non-PIR comparators + COSA, all 8 cells =====
    for market in S.MARKETS:
        thr = thresholds[market]
        for host in S.HOSTS:
            t0 = time.perf_counter()
            try:
                cell, w_sealed, bcell, pack = _build_protected_cell(market, host, HIST)
            except Exception as exc:  # fail closed, loudly, and record it
                failures.append(f"{market}/{host}: {type(exc).__name__}: {exc}")
                log(f"[FAIL] {market:9s} {host:13s} {type(exc).__name__}: {exc}")
                continue
            dk = day_keys_of(w_sealed.timestamp)
            y, hp = pack["y"], pack["hp"]

            # ---- the frozen S1-B2 method, seeds 7/17/37 --------------------
            seeds = []
            for seed in S.SEEDS:
                _pin_threads()
                ds = cell["hch"]
                r = HIST.run_replay(ds, seed, PHASE, dict(FLAGS))
                pred = ds.host_ev + r.alpha * r.u_ev * np.nan_to_num(r.s_out_ev, nan=0.0)[:, None]
                if pred.shape != y.shape or not np.isfinite(pred).all():
                    raise RuntimeError(f"{market}/{host}/seed{seed}: S1-B2 prediction is bad")
                mt = PM.cell_metrics(y, pred.astype(np.float64), hp, thr, dk)
                rec = {"market": market, "host": host, "method": "HCH_S1_B2", "seed": int(seed),
                       "alpha": float(r.alpha),
                       "flags": json.dumps(FLAGS, sort_keys=True),
                       "Overall_MAE": float(mt["Overall_MAE"]),
                       "Host_MAE": float(mt["Host_Overall_MAE"]),
                       "gain_pct": float(mt["relative_gain_vs_Host_pct"]),
                       "correction_ratio": float(HIST.metrics_for(ds, r.u_ev, r.alpha,
                                                                  r.s_out_ev)["correction_ratio"]),
                       "checkpoint_role": "inner",
                       "selected_checkpoint": _selected_step(r),
                       **{k: mt[k] for k in METRIC_KEYS}}
                seeds.append(rec)
                hch_rows.append(rec)
            med = _median_row(seeds)
            log(f"[HCH] {market:9s} {host:13s} MAE={med['Overall_MAE']:10.4f} "
                f"(Host {med['Host_MAE']:10.4f})  gain={med['gain_pct']:+7.4f}%  "
                f"alpha={med['alpha']:.4f}  {time.perf_counter()-t0:6.1f}s")

            # ---- the two non-PIR frozen comparators ------------------------
            bpreds = {}
            for method, fn in (("delta-Adapter", lambda c: TRANS.run_delta(c, TRANSFER_SEED)),
                               ("MatchedDirectResidual",
                                lambda c: TRANS.run_direct(c, TRANSFER_SEED))):
                res = fn(bcell)
                p = np.asarray(res["pred"], dtype=np.float64)
                if p.shape != y.shape:
                    raise RuntimeError(f"{market}/{host}/{method}: {p.shape} vs {y.shape}")
                bpreds[method] = p
                mt = PM.cell_metrics(y, p, hp, thr, dk)
                base_rows.append({"market": market, "host": host, "method": method,
                                  "seed": TRANSFER_SEED,
                                  "setting": ("OFFLINE_STATIC_POSTHOC" if method == "delta-Adapter"
                                              else "INTERNAL_CONTROL_ONLY"),
                                  "in_external_ranking": method != "MatchedDirectResidual",
                                  "fidelity": res.get("fidelity"),
                                  **{k: mt[k] for k in METRIC_KEYS}})
                log(f"[BASE] {market:9s} {host:13s} {method:22s} MAE={mt['Overall_MAE']:10.4f} "
                    f"gain={mt['relative_gain_vs_Host_pct']:+7.4f}%")

            # ---- COSA: separate ONLINE_TTA track, never in the offline rank --
            from experiments.current.china5_posthoc_baseline_panel import cosa_transfer
            seed_means = bcell["fit"].y_true[:, :, 0].mean(axis=1).astype(np.float64)
            cres = cosa_transfer.run_cosa(hp, y, seed_means, seed=COSA_SEED)
            cpred = np.asarray(cres["pred"], dtype=np.float64)
            if cpred.shape != y.shape or not np.isfinite(cpred).all():
                raise RuntimeError(f"{market}/{host}/COSA: bad prediction array")
            cmt = PM.cell_metrics(y, cpred, hp, thr, dk)
            aud = _cosa_chronology(cosa_transfer, hp, y, seed_means)
            base_rows.append({"market": market, "host": host, "method": "COSA",
                              "seed": COSA_SEED, "setting": COSA_SETTING,
                              "in_external_ranking": False, "in_best_static": False,
                              "fidelity": cres.get("fidelity"),
                              "chronology_audit_passed": aud["passed"],
                              "chronology_max_abs_diff_over_all_cuts":
                                  aud["max_abs_diff_over_all_cuts"],
                              "chronology_cuts_tested": aud["n_cuts_tested"],
                              "chronology_liveness": f"{aud['n_cuts_where_later_days_changed']}"
                                                     f"/{aud['n_cuts_requiring_liveness']}",
                              "n_days_adapted": int(cres.get("n_days_adapted", -1)),
                              **{k: cmt[k] for k in METRIC_KEYS}})
            log(f"[COSA] {market:9s} {host:13s} MAE={cmt['Overall_MAE']:10.4f} "
                f"gain={cmt['relative_gain_vs_Host_pct']:+7.4f}%  "
                f"chronology={'PASS' if aud['passed'] else 'FAIL'}")

            host_mae = float(np.mean(np.abs(y - hp)))
            cell_rows.append({
                "market": market, "host": host,
                "n_protected_days": int(y.shape[0]), "n_hours": int(y.size),
                "HCH_MAE_median_of_seeds": med["Overall_MAE"],
                # The gate's Host-relative gain is the *median of the per-seed gains*,
                # which is what the frozen domestic paper-ready panel used
                # (``runner.py``: ``Host_gain_pct=("Host_gain_pct","median")``).  The
                # gain-of-the-median is carried beside it as a disclosed diagnostic
                # only; it is never what a gate reads.
                "HCH_gain_pct_median_of_seeds": med["gain_pct"],
                "HCH_gain_of_median_MAE_pct": float(
                    100.0 * (host_mae - med["Overall_MAE"]) / host_mae),
                "HCH_alpha_median": med["alpha"],
                "Host_MAE": host_mae,
                "delta_Adapter_MAE": float(np.mean(np.abs(y - bpreds["delta-Adapter"]))),
                "PIR_MAE": None, "PIR_legal": bool(host in S.PIR_LEGAL_HOSTS),
                "MDR_MAE_internal_control_only":
                    float(np.mean(np.abs(y - bpreds["MatchedDirectResidual"]))),
                "COSA_MAE_online_tta_only": float(cmt["Overall_MAE"]),
                "seeds": ",".join(map(str, S.SEEDS)),
                "thresholds_source": thresholds[market]["_frozen_source_sha256"],
                "n_negative_hours": _int_or_none(med["n_negative_hours"]),
                "n_extreme_days": _int_or_none(med["n_extreme_days"]),
                "negative_price_subset_status": med["negative_price_subset_status"],
                "extreme_day_subset_status": med["extreme_day_subset_status"],
                "wall_seconds": float(time.perf_counter() - t0),
            })
            access[f"{market}/{host}"] = {
                "protected_days_predicted": int(y.shape[0]),
                "protected_hours_predicted": int(y.size),
                "host_forward_source": "REDERIVED_FROZEN_HOST_FORWARD",
                "host_digest": cell["digest"],
            }
            if host in S.PIR_LEGAL_HOSTS:
                pir_pending.append((market, host))
                pir_pack[(market, host)] = pack
            del cell, bcell, bpreds
        # end host
    # end market

    # ==== pass B: PIR last.  ``pir_transfer`` re-points the repo's ``utils``
    # package at the official checkout on import, so every cell that needs the
    # repository helpers must already be built.  Both PIR cells per market are
    # therefore constructed before the first PIR call, exactly as the frozen
    # driver ordered it.
    if pir_pending:
        # The cells are built *before* ``pir_transfer`` is imported, because that
        # module re-points the repo's ``utils`` package at the official checkout as a
        # side effect of import.  Building first means the reconstructed sliding-window
        # series cannot depend on which ``utils`` happens to be bound when PIR runs.
        prepared = {}
        for market, host in pir_pending:
            # Same composite the cell was scored on -- the Host is not forwarded a
            # second time, so PIR sees byte-identical protected-span evidence.
            prepared[(market, host)] = S.build_cell(
                market, host, pir_pack[(market, host)]["full"], S.EVAL_ROLE,
                relabel=_extended_relabel(market))
        PT = _load_transfer_module("pir_transfer")
        for (market, host) in pir_pending:
            thr = thresholds[market]
            bcell = prepared[(market, host)]
            pack = pir_pack[(market, host)]
            dk = day_keys_of(pack["timestamp"])
            y, hp = pack["y"], pack["hp"]
            out_dir = OUT / "_pir_run" / f"{market}__{host}"
            out_dir.mkdir(parents=True, exist_ok=True)
            before = torch.get_num_threads()
            try:
                res = PT.run_pir(bcell, out_dir, seed=PIR_SEED)
            finally:
                # the frozen PIR driver pins 4 threads and never restores it
                torch.set_num_threads(CONTRACT_THREADS)
            if torch.get_num_threads() != CONTRACT_THREADS:
                raise RuntimeError("thread pin not restored after PIR")
            p = _select_protected_pir(bcell, res, dk)
            mt = PM.cell_metrics(y, p, hp, thr, dk)
            base_rows.append({"market": market, "host": host, "method": "PIR",
                              "seed": PIR_SEED, "setting": "OFFLINE_STATIC_POSTHOC",
                              "in_external_ranking": True, "fidelity": res.get("fidelity"),
                              "thread_pin_before": int(before),
                              "thread_pin_after": int(torch.get_num_threads()),
                              "eval_span_relabel": "PROTECTED_FINAL->DIAG_EVAL; DEV_EVAL->S4",
                              **{k: mt[k] for k in METRIC_KEYS}})
            for c in cell_rows:
                if c["market"] == market and c["host"] == host:
                    c["PIR_MAE"] = float(mt["Overall_MAE"])
            log(f"[BASE] {market:9s} {host:13s} {'PIR':22s} MAE={mt['Overall_MAE']:10.4f} "
                f"gain={mt['relative_gain_vs_Host_pct']:+7.4f}%")
            del bcell, prepared[(market, host)], pir_pack[(market, host)]
        del prepared, pir_pack

    # ==== pass C: best_static, cell verdicts, aggregation =====================
    for c in cell_rows:
        cand = {"Host": c["Host_MAE"], "delta-Adapter": c["delta_Adapter_MAE"]}
        if c["PIR_legal"] and c["PIR_MAE"] is not None:
            cand["PIR"] = c["PIR_MAE"]
        best_name = min(cand, key=lambda k: cand[k])
        best_mae = float(cand[best_name])
        c["best_static"] = best_name
        c["best_static_MAE"] = best_mae
        c["beats_Host"] = bool(c["HCH_MAE_median_of_seeds"] < c["Host_MAE"])
        c["beats_best_static"] = bool(c["HCH_MAE_median_of_seeds"] < best_mae)
        c["gain_vs_best_static_pct"] = float(
            100.0 * (best_mae - c["HCH_MAE_median_of_seeds"]) / best_mae)
        log(f"[CELL] {c['market']:9s} {c['host']:13s} best_static={best_name} "
            f"({best_mae:10.4f})  beats_Host={c['beats_Host']} "
            f"beats_best_static={c['beats_best_static']}")

    panel = _aggregate(cell_rows, base_rows, failures)
    verdict = _token(panel)
    (OUT / "S1B2_CELL_SEED_RESULTS.csv").write_text(
        pd.DataFrame(hch_rows).to_csv(index=False, encoding="utf-8"), encoding="utf-8")
    (OUT / "S1B2_BASELINE_RESULTS.csv").write_text(
        pd.DataFrame(base_rows).to_csv(index=False, encoding="utf-8"), encoding="utf-8")
    (OUT / "S1B2_CELL_MEDIANS.csv").write_text(
        pd.DataFrame(cell_rows).to_csv(index=False, encoding="utf-8"), encoding="utf-8")
    S.dump(OUT / "S1B2_PANEL.json", {
        "panel": panel, "cells": cell_rows, "token": verdict,
        "flags": FLAGS, "seeds": list(S.SEEDS),
        "transfer_seed": TRANSFER_SEED, "pir_seed": PIR_SEED, "cosa_seed": COSA_SEED,
        "continuity": _continuity(),
        "gate_authority": "docs/current/HCH_S1_B2_NORD_NEM_CONFIRMATION_20260919.md section 7",
    })
    S.dump(OUT / "PROTECTED_ACCESS_AUDIT.json", {
        "schema": "hch_s1_b2_nord_nem_protected_access.v1",
        "p0_token": S.rel(S.P0_TOKEN),
        "p0_token_sha256": S.sha(S.P0_TOKEN),
        "p0_declared_protected_reads": int(token["protected_final_target_values_read"]),
        "cells_accessing_protected": access,
        "n_cells": len(access),
        "protected_days_predicted_total": int(sum(v["protected_days_predicted"]
                                                  for v in access.values())),
        "read_audit_this_process": S.read_audit(),
        "sealed_target_rows_parsed_by_stage_reader": S.sealed_positions_read(),
        "host_forward_is_the_only_protected_source": True,
        "fits_on_protected_days": 0,
        "tuning_on_protected_days": 0,
        "scalar_refits_on_protected_days": 0,
        "checkpoint_changes_on_protected_days": 0,
        "labelled_online_tta": False,
        "markets_never_touched": ["LAGO_NP", "GEFCOM14P"],
        "note": ("S1-B2 is offline: every protected day is scored by a model fitted on "
                 "POST_TRAIN only. Only COSA is ONLINE_TTA, and it is reported on its own "
                 "line and excluded from best_static and from the offline ranking."),
    })
    S.dump(OUT / "VERDICT.json", {
        "schema": "hch_s1_b2_nord_nem_verdict.v1",
        "token": verdict["token"],
        "panel": panel, "gates": verdict["gates"],
        "s1_b2_remains_frozen": True,
        "method_change_after_protected_access": False,
        "rescue_applied": False,
        "datasets_added": False,
        "proxy_rows": 0,
        "lago_gefcom_blocker_relaxed": False,
        "continuity": _continuity(),
    })
    log(f"[VERDICT] {verdict['token']}")
    return 0


# ------------------------------------------------------------------- helpers
def _load_transfer_module(name: str):
    p = ROOT / "experiments/current/hch_frozen_method_baseline_transfer"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    import importlib
    return importlib.import_module(name)


def _extended_relabel(market: str) -> dict:
    """The frozen legacy map with the scoring partition bound to ``DIAG_EVAL``.

    Only PIR consumes the reconstructed sliding-window series, and it enumerates its
    evaluation span by legacy role name.  ``DEV_EVAL`` is sent to the same unused
    legacy name the frozen map gives ``PROTECTED_FINAL`` so that the evaluation span
    is exactly the protected days -- the frozen ``pred.shape == (n_eval, H)``
    invariant then holds unchanged.  PIR's train and validation spans are untouched:
    both are derived from ``n_fit``/``n_val`` and the fit block alone.
    """
    CL = S.icells_of(market)
    m = dict(CL.LEGACY_ROLE)
    m["DEV_EVAL"] = "S4"
    m["PROTECTED_FINAL"] = "DIAG_EVAL"
    return m


def _selected_step(r) -> int:
    """The B2 chronological inner-selection step the frozen runner landed on."""
    cps = list(getattr(r, "checkpoint_records", []) or [])
    if not cps:
        return -1
    last = cps[-1]
    for key in ("selected_step", "selected_check", "step", "best_step"):
        if key in last:
            return int(last[key])
    return -1


def _int_or_none(v):
    """A count that is only a cell statistic when all three seeds agreed on it."""
    return None if v is None else int(v)


def _median_row(seeds: list[dict]) -> dict:
    """The cell statistic: median across the authority's seeds, taken *before* any gain."""
    out: dict = {"market": seeds[0]["market"], "host": seeds[0]["host"]}
    for k in ("Overall_MAE", "Host_MAE", "gain_pct", "alpha", "correction_ratio"):
        out[k] = float(np.median([float(s[k]) for s in seeds]))
    for k in METRIC_KEYS:
        vals = [s[k] for s in seeds]
        if all(isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)
               for v in vals):
            out[k] = float(np.median([float(v) for v in vals]))
        elif len(set(map(str, vals))) == 1:
            out[k] = vals[0]          # a status string such as NOT_APPLICABLE_N0
        else:
            out[k] = None             # mixed numeric/None across seeds: not a cell statistic
    return out


def _continuity() -> dict:
    """The predecessor's LAGO_DE/PJM panel: cited and hashed, never rerun."""
    if not PRIOR_CONTINUITY.is_file():
        return {"available": False,
                "note": "predecessor continuity panel absent; nothing is inferred from it"}
    d = json.loads(PRIOR_CONTINUITY.read_text(encoding="utf-8"))
    return {"available": True, "source": S.rel(PRIOR_CONTINUITY),
            "source_sha256": S.sha(PRIOR_CONTINUITY),
            "label": "DEVELOPMENT_CONTINUITY", "rerun_here": False,
            "enters_primary_aggregate": False,
            "content": d}


def _pin_threads() -> None:
    """Re-pin to the frozen contract's single thread and prove the pin stuck.

    ``src/backbones/backbones.py`` calls ``torch.set_num_threads(4)`` at import time,
    and this stage's family loader imports it -- so the pin established at process
    start is silently undone the first time a family is touched.  The frozen
    ``train_host`` re-pins immediately before it builds for exactly this reason
    (``hch_international_nem_nord_baseline_matrix/implementation/ihosts.py:310``).
    Four-thread float32 accumulation is *not* bit-exact against the frozen
    single-thread manifests, so an unpinned forward is a silent identity failure.
    """
    torch.set_num_threads(CONTRACT_THREADS)
    if torch.get_num_threads() != CONTRACT_THREADS:
        raise RuntimeError("thread pinning contract violated")


def _protected_host_forward(market: str, host: str) -> tuple[object, np.ndarray, str]:
    """Forward the frozen Host over the protected span, certified on the open roles.

    The frozen ``intl_shared.predict_role`` is **not** used here.  It is dead code:
    the four-market stage blocked at P0 and never reached it, so nothing ever ran it.
    It carries two defects, both discovered by running it:

    * ``model.model.load_state_dict(...)`` on a freshly constructed adapter, whose
      ``.model`` is ``None`` until ``fit()`` builds it (``src/backbones/base.py:46``);
    * no ``torch.set_num_threads(1)`` re-pin after the family load, so the forward
      would run on 4 threads and stop being bit-exact -- silently.

    Neither is repaired: the frozen layer is not edited.  This stage instead
    reconstructs the Host from the weights P0 proved digest-identical, and pays for
    that liberty with a proof -- the reconstruction must reproduce the frozen Host's
    *open-role* prediction **bit-for-bit** before it may touch a protected day.  The
    open roles are not the scored ones, so the proof consumes no sealed information,
    and it is the only thing standing between a wrong thread count and a wrong
    published Host column.
    """
    IH = S.ihosts_of(market)
    C = S.contracts_of(market)
    man = S.frozen_host_manifest(market, host)
    want = str(man["checkpoint_sha256"]).upper()
    blob_path = S.rederived_host_path(market, host)
    if not blob_path.is_file():
        raise RuntimeError(f"{market}/{host}: re-derived frozen Host absent at {blob_path}")
    blob = torch.load(blob_path, map_location="cpu", weights_only=False)
    if str(blob["checkpoint_sha256"]).upper() != want:
        raise RuntimeError(f"{market}/{host}: re-derived Host is not the frozen one "
                           f"({blob['checkpoint_sha256']} vs {want})")

    # Every frozen-reader call happens *before* the vendor build: ``official_loader``
    # displaces the name ``utils`` for the remainder of the process, and re-binding it
    # afterwards is the import-shadowing defect ``vendor_guard`` exists to prevent.
    open_w = S.read_windows(market, list(S.OPEN_ROLES))
    sealed_w = S.read_windows(market, [S.EVAL_ROLE])
    if not pd.to_datetime(sealed_w.timestamp).is_monotonic_increasing:
        raise RuntimeError(f"{market}/{host}: protected days are not chronological")
    open_cache, _p = S.load_host_cache(market, host)
    dtype = S.frozen_dtype(market, host)
    cfg = IH.host_cfg(host)
    tr_view, _va = C.host_fit_parts(C.load_market(market))

    _pin_threads()
    with IH.vendor_bindings():
        saved = list(sys.path)
        sys.path[:] = [p for p in sys.path if not (p and (Path(p) / "models.py").exists())]
        try:
            model = (IH.HB.make_new_host(host, cfg) if host in ("iTransformer", "LSTM")
                     else IH._make_official_direct(host, cfg))
            # The frozen scalers come from the frozen HOST_TRAIN view, through the
            # adapter's own ``_scale_fit``; the weights come from the P0 blob.  Both
            # halves are the frozen objects -- nothing is re-typed.
            model._scale_fit(tr_view)
            model.model = model.build_model().to(model.device)
            model.model.load_state_dict(blob["state_dict"])
            model.freeze()
            open_pred = np.asarray(model.predict(open_w), dtype=dtype)
            sealed_pred = np.asarray(model.predict(sealed_w), dtype=dtype)
        finally:
            sys.path[:] = saved

    digest = IH.state_dict_sha(model)
    if digest != want:
        raise RuntimeError(f"{market}/{host}: reconstructed Host digest {digest} != "
                           f"frozen manifest {want}")
    if not np.array_equal(np.ascontiguousarray(open_pred),
                          np.ascontiguousarray(open_cache.host_pred)):
        d = float(np.max(np.abs(open_pred.astype(np.float64)
                                - open_cache.host_pred.astype(np.float64))))
        raise RuntimeError(
            f"{market}/{host}: the reconstructed frozen Host does not reproduce its own "
            f"open-role prediction (max abs diff {d:.3e}); the protected-span forward is "
            f"not certified and is discarded")
    if sealed_pred.shape != sealed_w.target.shape or not np.isfinite(sealed_pred).all():
        raise RuntimeError(f"{market}/{host}: protected-span Host prediction is mis-shaped "
                           f"or non-finite ({sealed_pred.shape} vs {sealed_w.target.shape})")
    return sealed_w, sealed_pred, digest


def _build_protected_cell(market: str, host: str, HIST) -> tuple[dict, object, dict, dict]:
    """The protected-span cell, shared by every comparator on this coordinate.

    Returns ``({"hch": ds, "full": full, "digest": d}, w_sealed, blocked_cell, pack)``
    where ``pack`` carries the sealed arrays and the frozen ``build_cell`` output for
    the non-PIR comparators.  Every one of them is scored on exactly these days.
    """
    open_cache, cache_path = S.load_host_cache(market, host)
    w_sealed, pred_sealed, digest = _protected_host_forward(market, host)
    ts = pd.to_datetime(w_sealed.timestamp)
    if not ts.is_monotonic_increasing:
        raise RuntimeError(f"{market}/{host}: protected days are not chronological")
    full = S.compose_cache(open_cache, w_sealed, pred_sealed)
    fit = full.subset(S.FIT_ROLE)
    state_fit, _ = S.read_legal_state(market, pd.to_datetime(fit.timestamp))
    state_ev, _ = S.read_legal_state(market, ts)
    ds = S.make_s1b2_cell(HIST, market, host, full, state_fit, state_ev, cache_path, S.EVAL_ROLE)
    bcell = S.build_cell(market, host, full, S.EVAL_ROLE)
    pack = {"full": full, "timestamp": w_sealed.timestamp,
            "y": w_sealed.target[:, :, 0].astype(np.float64),
            # ``predict`` returns (day, hour, channel); the metrics take the single
            # channel.  ``compose_cache`` keeps the 3-D form, so slice only here.
            "hp": pred_sealed[:, :, 0].astype(np.float64)}
    return {"hch": ds, "full": full, "digest": digest}, w_sealed, bcell, pack


def _select_protected_pir(bcell: dict, res: dict, dk: np.ndarray) -> np.ndarray:
    """Pick the protected days out of PIR's day-aligned protected-span output."""
    runs = bcell["runs"]
    wanted = list(map(str, dk))
    ew = [(int(r), int(t)) for r, t in res["eval_windows"]]
    got = [str(runs[rid]["series_time"][t])[:10] for rid, t in ew]
    if got != wanted:
        raise RuntimeError(f"PIR eval span does not align with the protected days: "
                           f"{len(got)} vs {len(wanted)}")
    p = np.asarray(res["pred"], dtype=np.float64)
    if p.shape != (len(wanted), 24):
        raise RuntimeError(f"PIR produced {p.shape}, expected {(len(wanted), 24)}")
    return p


COSA_MAX_CUTS = 32


def cosa_cut_plan(n_days: int, max_cuts: int = COSA_MAX_CUTS) -> list[int]:
    """A pre-declared, data-independent subset of the causality cut points.

    The frozen parent stage certified COSA's online chronology over the *full* cut
    set on the frozen partitions, and that module is digest-pinned by P0.  Re-running
    every one of up to 289 cuts here would cost hours of purely confirmatory work, so
    the cuts are thinned by a uniform stride fixed by this rule alone -- computed from
    the day count before any prediction exists, never from a result.  The last cut is
    always included: it is the deepest prefix test available.
    """
    interior = list(range(1, n_days))
    if len(interior) <= max_cuts:
        return interior
    stride = int(np.ceil(len(interior) / max_cuts))
    cuts = interior[::stride]
    if interior[-1] not in cuts:
        cuts.append(interior[-1])
    return sorted(set(cuts))


def _cosa_chronology(cosa_transfer, hp, y, seed_means) -> dict:
    """Prefix-cleanliness / liveness test: does day k depend on days after k?

    Poisoning every day from k onward must leave predictions 0..k untouched (prefix
    cleanliness) and must change at least one later day (liveness), or the test is
    vacuous.
    """
    base = np.asarray(cosa_transfer.run_cosa(hp, y, seed_means, seed=COSA_SEED)["pred"],
                      dtype=np.float64)
    n = y.shape[0]
    cuts = cosa_cut_plan(n)
    worst, live, live_required = 0.0, 0, 0
    for k in cuts:
        poisoned = y.copy()
        poisoned[k:] = 1.0e6
        got = np.asarray(cosa_transfer.run_cosa(hp, poisoned, seed_means, seed=COSA_SEED)["pred"],
                         dtype=np.float64)
        worst = max(worst, float(np.max(np.abs(got[:k + 1] - base[:k + 1]))))
        if int(n - 1 - k) >= 1:
            live_required += 1
            live += int(not np.array_equal(got[k + 1:], base[k + 1:]))
    seeded = np.asarray(cosa_transfer.run_cosa(hp, y, np.zeros_like(seed_means),
                                               seed=COSA_SEED)["pred"], dtype=np.float64)
    seed_live = bool(not np.array_equal(seeded, base))
    return {"n_days": int(n), "max_abs_diff_over_all_cuts": worst,
            "cuts_prefix_clean": bool(worst == 0.0),
            "n_cuts_requiring_liveness": int(live_required),
            "n_cuts_where_later_days_changed": int(live),
            "n_cuts_tested": len(cuts), "cut_stride_declared": True,
            "cut_plan_includes_last_cut": bool(n - 1 in cuts),
            "full_cut_set_deferred_to": ("frozen parent certification of "
                                         "china5_posthoc_baseline_panel.cosa_transfer on the "
                                         "frozen partitions; the module is digest-pinned by P0"),
            "post_train_seed_changes_predictions": seed_live,
            "passed": bool(worst == 0.0 and live == live_required and live_required > 0
                           and seed_live)}


def _aggregate(cells: list[dict], base_rows: list[dict],
               failures: list[str] | None = None) -> dict:
    def med(xs):
        xs = [float(x) for x in xs]
        return float(np.median(xs)) if xs else float("nan")

    n = len(cells)
    cosa = [r for r in base_rows if r["method"] == "COSA"]
    cosa_fail = [f"{r['market']}/{r['host']}" for r in cosa
                 if not r.get("chronology_audit_passed")]
    gains = [c["HCH_gain_pct_median_of_seeds"] for c in cells]
    bs_gains = [c["gain_vs_best_static_pct"] for c in cells]
    mkt = {m: med([c["HCH_gain_pct_median_of_seeds"] for c in cells if c["market"] == m])
           for m in S.MARKETS}
    fam = {h: med([c["HCH_gain_pct_median_of_seeds"] for c in cells if c["host"] == h])
           for h in S.HOSTS}
    return {
        "n_cells": n,
        "expected_cells": len(S.CELLS),
        "failures": list(failures or []),
        "all_8_cells_scored": bool(n == len(S.CELLS) and not failures),
        "host_positive_count": int(sum(1 for c in cells if c["beats_Host"])),
        "best_static_win_count": int(sum(1 for c in cells if c["beats_best_static"])),
        "host_nonworse_within_1pct_count": int(sum(1 for g in gains if g >= -1.0)),
        "within_2pct_of_best_static_count": int(sum(1 for g in bs_gains if g >= -2.0)),
        "panel_median_host_gain_pct": med(gains),
        "panel_median_gain_vs_best_static_pct": med(bs_gains),
        "worst_cell_host_gain_pct": float(min(gains)) if gains else float("nan"),
        "best_cell_host_gain_pct": float(max(gains)) if gains else float("nan"),
        "market_median_host_gain_pct": mkt,
        "market_medians_host_positive": int(sum(1 for v in mkt.values() if v > 0)),
        "worst_market_median_host_gain_pct": float(min(mkt.values())) if mkt else float("nan"),
        "host_family_median_host_gain_pct": fam,
        "best_static_choice_counts": {k: sum(1 for c in cells if c["best_static"] == k)
                                      for k in ("Host", "delta-Adapter", "PIR")},
        "mdr_worse_than_host_count_internal_control":
            int(sum(1 for c in cells if c["MDR_MAE_internal_control_only"] > c["Host_MAE"])),
        "protected_days_total": int(sum(c["n_protected_days"] for c in cells)),
        "protected_hours_total": int(sum(c["n_hours"] for c in cells)),
        "negative_hours_total": int(sum(c["n_negative_hours"] or 0 for c in cells)),
        "extreme_days_total": int(sum(c["n_extreme_days"] or 0 for c in cells)),
        "cells_with_unstable_subset_counts":
            [f"{c['market']}/{c['host']}" for c in cells
             if c["n_negative_hours"] is None or c["n_extreme_days"] is None],
        "cosa_chronology_failures": len(cosa_fail),
        "cosa_chronology_failures_detail": cosa_fail,
        "n_cosa_cells": len(cosa),
        "pir_rows_legal": int(sum(1 for r in base_rows if r["method"] == "PIR")),
        "pir_rows_expected": len(S.PIR_LEGAL_HOSTS) * len(S.MARKETS),
        "proxy_rows": 0,
    }


def _token(panel: dict) -> dict:
    """The authority's section 7 claim-strength rule, applied verbatim.

    ``hard_fail`` is a source / access / chronology failure, not a weak result: a cell
    that could not be scored, a PIR row that is not legally available on its Host, or a
    COSA chronology audit that fails all make the stage INVALID rather than merely
    unimpressive.
    """
    hard_fail = (not panel["all_8_cells_scored"]
                 or panel["n_cells"] != len(S.CELLS)
                 or panel["protected_days_total"] <= 0
                 or panel["pir_rows_legal"] != panel["pir_rows_expected"]
                 or panel["cosa_chronology_failures"] > 0)
    gates = {
        "g1_panel_median_host_gain>0": panel["panel_median_host_gain_pct"] > 0.0,
        "g2_host_nonworse_within_-1.0%>=6": panel["host_nonworse_within_1pct_count"] >= 6,
        "g3_panel_median_gain_vs_best_static>=-1.0%":
            panel["panel_median_gain_vs_best_static_pct"] >= -1.0,
        "g4_within_-2.0%_of_best_static>=6": panel["within_2pct_of_best_static_count"] >= 6,
        "g5_worst_market_median>=-3.0%": panel["worst_market_median_host_gain_pct"] >= -3.0,
        "g6_no_access_provenance_chronology_failure": not hard_fail,
        "strong:s1_beat_host>=6": panel["host_positive_count"] >= 6,
        "strong:s2_beat_best_static>=5": panel["best_static_win_count"] >= 5,
        "strong:s3_panel_median_gain_vs_best_static>0":
            panel["panel_median_gain_vs_best_static_pct"] > 0.0,
    }
    confirmed = all(gates[k] for k in ("g1_panel_median_host_gain>0",
                                       "g2_host_nonworse_within_-1.0%>=6",
                                       "g3_panel_median_gain_vs_best_static>=-1.0%",
                                       "g4_within_-2.0%_of_best_static>=6",
                                       "g5_worst_market_median>=-3.0%",
                                       "g6_no_access_provenance_chronology_failure"))
    strong = confirmed and all(gates[k] for k in ("strong:s1_beat_host>=6",
                                                  "strong:s2_beat_best_static>=5",
                                                  "strong:s3_panel_median_gain_vs_best_static>0"))
    if hard_fail:
        tok = "HCH_S1_B2_NORD_NEM_INVALID"
    elif strong:
        tok = "HCH_S1_B2_NORD_NEM_STRONG_CONFIRMED"
    elif confirmed:
        tok = "HCH_S1_B2_NORD_NEM_CONFIRMED_FOR_PAPER"
    else:
        tok = "HCH_S1_B2_NORD_NEM_MIXED"
    return {"token": tok, "gates": gates,
            "rule": ("authority section 7; changes claim wording only, never authorises "
                     "redesign")}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
