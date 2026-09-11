"""Frozen six-cell panel for the frozen-method baseline-transfer canary.

Controlling protocol:
    docs/current/HCH_FROZEN_METHOD_BASELINE_TRANSFER_ADJUDICATION_20260911.md

Executor contract:
    .agents/skills/solar-research-executor/SKILL.md

This module is the ONLY legal entry point to the frozen panel in this stage.  It
owns:

* the frozen Host caches (international + GANSU) and their hashes;
* the frozen GANSU natural-day split contract;
* the frozen legal forecast-time state columns;
* the frozen S1 Daily-Patch GRU32 replay primitives;
* the frozen metric contract (Overall / Tail / Normal MAE, MSE, relative gain);
* the frozen day-level evaluation rows (DIAG_EVAL) and the reconstructed
  contiguous DA price series used to give the paper-native baselines their
  native sliding-window training protocol.

Every compared method consumes the same rows, the same DA target and the same
day-level MAE implementation from here, so that "same dataset rows / DA target /
Host family / final metric" is a structural property of the code rather than a
promise.

Nothing in this stage modifies HCH.  S1 is replayed, never redesigned.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.current.hch_unified_da_shape_upgrade.run_canary import (  # noqa: E402
    H,
    HOSTS,
    LEGAL,
    MARKETS,
    SEQ,
    SEEDS,
    TARGET,
    TRACK,
    load_cache,
    load_gansu_da,
    make_inputs,
    read_state,
    timestamp_contract_gansu,
)
from experiments.current.hch_anchored_compact_amplitude.run_closure import (  # noqa: E402
    eval_metrics,
    fit_shape_outputs,
    load_frozen_cache,
    pooled_alpha,
    shape_oof,
    utility,
)
from src.mvp.hch_minimal_repair.crossfit import chronological_blocks  # noqa: E402
from src.mvp.hch_minimal_repair.metrics import mae, mse  # noqa: E402
from src.mvp.hch_minimal_repair.training import unit_shape_target  # noqa: E402
from utils.benchmark_cache import stable_hash  # noqa: E402

# REPRODUCIBILITY INVARIANT.
# src/backbones/backbones.py calls torch.set_num_threads(4) at import time, so a
# single-thread pin issued BEFORE these imports is silently reverted and the
# discrete weighted-median alpha flips at the ~1e-3 level.  Pinning after the
# imports makes the frozen S1 replay reproduce the registered evidence exactly.
torch.set_num_threads(1)
if torch.get_num_threads() != 1:
    raise RuntimeError("baseline-transfer panel requires single-threaded torch for frozen S1 replay")

SHAPE_EVIDENCE = ROOT / "experiments/evidence/hch_unified_da_shape_engineering_20260910"
HRSI_EVIDENCE = ROOT / "experiments/evidence/hch_host_relative_state_interaction_20260911"
FROZEN_S1_CELLS = HRSI_EVIDENCE / "metrics_by_cell.csv"

# Frozen S1 replay tolerance.  Prior closures registered 1e-10 for the same
# replay; the observed cross-stage difference is <= 1.5e-14.  The tolerance is
# fixed here before any baseline is executed and is never relaxed.
S1_REPLAY_TOL = 1e-6

# Chronological train/validation carve-out for the paper-native baselines.
# Validation exists only because the audited baselines early-stop on their own
# official validation split; it is carved from DIAG_FIT and is never DIAG_EVAL.
VAL_FRACTION = 0.10


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree_sha(p: Path) -> str:
    h = hashlib.sha256()
    if not p.exists():
        return "ABSENT"
    for x in sorted(y for y in p.rglob("*") if y.is_file() and "__pycache__" not in y.parts):
        h.update(x.relative_to(p).as_posix().encode())
        h.update(x.read_bytes())
    return h.hexdigest()


def hash_json(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# --------------------------------------------------------------------------
# Frozen S1 replay (verbatim from the HRSI closure; S1 is never modified)
# --------------------------------------------------------------------------

def s1_artifacts(fi, ei, state_fit, state_eval, ef, seed):
    """Frozen S1 OOF directions and final DIAG_EVAL directions."""
    oof = shape_oof(fi.x, state_fit, ef, fi.repair_available, seed)
    v, u, _, info = fit_shape_outputs(fi.x, state_fit, ef, fi.repair_available, ei.x, state_eval, seed)
    info = {**info, "parameter_count": info["shape_parameter_count"],
            "inference_seconds_per_day": info["inference_seconds"] / max(len(ei.x), 1)}
    return oof, u, info


def s1_alpha(e, u, av, blocks):
    """Frozen S1 pooled-alpha OOF chronology (B1..B4 row set), unchanged."""
    d = np.full_like(u, np.nan)
    records = []
    for k in range(2, 5):
        prior = np.concatenate([blocks[f"B{j}"][av[blocks[f"B{j}"]]] for j in range(1, k)])
        current = blocks[f"B{k}"][av[blocks[f"B{k}"]]]
        alpha = pooled_alpha(e, u, prior)
        d[current] = alpha * u[current]
        records.append({"record_type": "alpha", "block": f"B{k}", "alpha": alpha,
                        "calibration_max_index": int(prior.max()),
                        "proposal_min_index": int(current.min()),
                        "self_excluded": bool(not np.intersect1d(prior, current).size)})
    allrows = np.concatenate([blocks[f"B{k}"][av[blocks[f"B{k}"]]] for k in range(1, 5)])
    return pooled_alpha(e, u, allrows), d, allrows, records


def s1_complete_rowset(blocks, av):
    return np.concatenate([blocks[f"B{k}"][av[blocks[f"B{k}"]]] for k in range(2, 5)])


# --------------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------------

def gansu_split_manifest(contract):
    """The frozen GANSU split payload, reconstructed without writing anywhere.

    ``write_pre_target_contract`` in the Unified-DA Shape stage both built this
    payload and wrote it into that stage's evidence root.  This stage may only
    write under its own roots, so the payload is rebuilt here byte-for-byte and
    the resulting ``stable_hash`` is checked against the hash recorded inside the
    frozen GANSU Host caches.  Any divergence fails closed.
    """
    used = [d for d, s in contract.segments.items() if s in {"S1", "DIAG_FIT", "DIAG_EVAL"}]
    return {"schema": "gansu_da_development_split.v1", "created_before_target_outcomes": True,
            "dataset_id": "GANSU_DA",
            "source": "data/CHINA/GANSU/甘肃24h电价数据集.xlsx",
            "target_column": "日前电价",
            "forbidden_target_day_inputs": ["日前电价", "实时电价"],
            "counts": contract.counts, "consumed_days": used,
            "closed_roles": ["S3", "S4", "protected", "final"],
            "boundary_policy": ("same deterministic natural-day proportions as prior provincial contract; "
                                "timestamp-only construction")}


def build_cells():
    """Seed-independent frozen panel: caches, split contract, state, inputs.

    Returns ``(cells, host_paths, host_hash, access)`` where ``cells`` is keyed
    by ``(market, host)``.
    """
    contract = timestamp_contract_gansu()
    split = gansu_split_manifest(contract)
    gansu_windows, target_access = load_gansu_da(contract)
    split_hash = stable_hash(split)

    caches, host_paths = {}, {}
    for m in MARKETS:
        for h in HOSTS:
            caches[m, h], host_paths[m, h] = load_cache(m, h, gansu_windows, split_hash)
    host_hash = {f"{m}/{h}": sha(host_paths[m, h]) for m in MARKETS for h in HOSTS}

    # Fail closed on any cache whose identity, target or segment roles drifted.
    for (m, h), c in caches.items():
        if c.metadata.get("target_column") != TARGET[m] or c.market_id != m:
            raise RuntimeError(f"{m}/{h}: frozen Host cache identity mismatch")
        if set(c.segment.astype(str)) - {"S1", "DIAG_FIT", "DIAG_EVAL"}:
            raise RuntimeError(f"{m}/{h}: frozen Host cache carries a forbidden segment")

    by_market = {m: caches[m, HOSTS[0]] for m in MARKETS}
    states, state_access = {}, {}
    for m in MARKETS:
        c = by_market[m]
        origins = np.r_[c.subset("DIAG_FIT").timestamp, c.subset("DIAG_EVAL").timestamp]
        z, a = read_state(m, origins)
        nf = len(c.subset("DIAG_FIT").timestamp)
        states[m] = (z[:nf], z[nf:])
        state_access[m] = a

    cells = {}
    for m in MARKETS:
        state_fit, state_eval = states[m]
        for h in HOSTS:
            cache = caches[m, h]
            fit = cache.subset("DIAG_FIT")
            ev = cache.subset("DIAG_EVAL")
            # Both Hosts of a market must share the exact row ordering, otherwise
            # the single market-level state read would be misaligned for one of
            # them and the S1 replay would silently stop matching the evidence.
            ref = by_market[m]
            if not (len(fit.timestamp) == len(state_fit) and len(ev.timestamp) == len(state_eval)
                    and np.array_equal(fit.timestamp, ref.subset("DIAG_FIT").timestamp)
                    and np.array_equal(ev.timestamp, ref.subset("DIAG_EVAL").timestamp)):
                raise RuntimeError(f"{m}/{h}: Host split manifest disagrees with the state row alignment")
            fi, floor = make_inputs(fit, cache)
            ei, _ = make_inputs(ev, cache, floor)
            ef = (fit.y_true[:, :, 0] - fit.host_pred[:, :, 0]) / np.where(np.isfinite(fi.scale), fi.scale, 1)[:, None]
            ee = (ev.y_true[:, :, 0] - ev.host_pred[:, :, 0]) / np.where(np.isfinite(ei.scale), ei.scale, 1)[:, None]
            y = ev.y_true[:, :, 0]
            hp = ev.host_pred[:, :, 0]
            q05, q95 = np.quantile(fit.y_true[:, :, 0], [.05, .95])
            tail = (y <= q05) | (y >= q95)
            normal = ~tail
            runs, span = rebuild_series(cache)
            cells[m, h] = {
                "market": m, "host": h, "track": TRACK[m],
                "cache": cache, "fit": fit, "ev": ev, "fi": fi, "ei": ei,
                "ef": ef, "ee": ee, "y": y, "hp": hp,
                "scale": np.nan_to_num(ei.scale, nan=0)[:, None],
                "avail": ei.repair_available,
                "tail": tail, "normal": normal,
                "state_fit_raw": state_fit, "state_eval_raw": state_eval,
                "n_fit": len(fi.x), "n_eval": len(ei.x),
                "q05": float(q05), "q95": float(q95),
                "runs": runs, "span": span,
                "gap_breaks": int(span["n_gap_breaks"]),
                "split_manifest": {
                    "DIAG_FIT_timestamp_hash": hash_json(list(map(str, fit.timestamp))),
                    "DIAG_EVAL_timestamp_hash": hash_json(list(map(str, ev.timestamp))),
                    "DIAG_FIT_n": int(len(fit.timestamp)), "DIAG_EVAL_n": int(len(ev.timestamp)),
                    "DIAG_FIT_first": str(fit.timestamp[0]),
                    "DIAG_FIT_last": str(fit.timestamp[-1]),
                    "DIAG_EVAL_first": str(ev.timestamp[0]),
                    "DIAG_EVAL_last": str(ev.timestamp[-1]),
                },
                "host_metadata": dict(cache.metadata),
            }
    access = {
        "target_access": target_access,
        "state_access": state_access,
        "roles_read": ["S1", "DIAG_FIT", "DIAG_EVAL"],
        "S3_reads": 0, "S4_reads": 0, "protected_reads": 0, "final_reads": 0,
        "other_China_markets_read": 0, "Shandong_reads": 0, "Shaanxi_reads": 0,
        "Ningxia_reads": 0, "Qinghai_reads": 0,
        "note": ("The only market files opened are LAGO_DE / LAGO_PJM / GANSU_DA. "
                 "SEC/Regelenergie-style tracks are untouched; only the A_DA track is read."),
    }
    return cells, host_paths, host_hash, access


def rebuild_series(cache):
    """Reconstruct the hourly DA price series as maximal *contiguous* runs.

    The frozen panel stores one window per natural day.  Not every market is
    day-contiguous: `GANSU_DA` in particular drops the days its source could not
    supply, so concatenating DIAG_FIT + DIAG_EVAL across such a hole would
    fabricate a price series that never existed and let a sliding-window baseline
    train on spliced non-adjacent hours.  The reconstruction therefore splits the
    frozen rows into maximal runs of strictly 24h-spaced natural days and every
    downstream window is required to live inside a single run.

    The reconstruction reads only DIAG_FIT / DIAG_EVAL rows plus the 72h context
    already stored in the frozen cache, so it cannot widen the authorised read
    set.
    """
    fit = cache.subset("DIAG_FIT")
    ev = cache.subset("DIAG_EVAL")
    n_fit, n_eval = len(fit.timestamp), len(ev.timestamp)
    origins = pd.to_datetime(np.r_[fit.timestamp, ev.timestamp])
    if not origins.is_monotonic_increasing:
        raise RuntimeError("DIAG_FIT/DIAG_EVAL rows are not chronological")
    if pd.to_datetime(fit.timestamp[0]) >= pd.to_datetime(ev.timestamp[0]):
        raise RuntimeError("frozen panel segment ordering is not DIAG_FIT then DIAG_EVAL")

    y_all = np.r_[fit.y_true[:, :, 0], ev.y_true[:, :, 0]].reshape(-1).astype(np.float64)
    ctx = np.concatenate([fit.context[:, :, 0], ev.context[:, :, 0]], axis=0).astype(np.float64)
    if ctx.shape[1] != SEQ:
        raise RuntimeError("frozen context width is not the registered SEQ")

    cuts = [0] + [i for i in range(1, len(origins))
                  if origins[i] - origins[i - 1] != pd.Timedelta(hours=24)] + [len(origins)]
    runs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        series = np.concatenate([ctx[a], y_all[24 * a:24 * b]])
        base = SEQ + 24 * np.arange(b - a, dtype=np.int64)
        # Every stored window must be recoverable by pure index arithmetic from
        # this run's series, otherwise the baselines would be trained/served a
        # different series than the one the frozen targets came from.
        for j, i in enumerate(range(a, b)):
            lo = int(base[j])
            if not np.allclose(series[lo:lo + 24], y_all[24 * i:24 * (i + 1)], atol=1e-9, rtol=0):
                raise RuntimeError("reconstructed run does not reproduce frozen targets")
            if not np.allclose(series[lo - SEQ:lo], ctx[i], atol=1e-9, rtol=0):
                raise RuntimeError("reconstructed run does not reproduce frozen contexts")
        seg = np.array(["DIAG_FIT" if i < n_fit else "DIAG_EVAL" for i in range(a, b)])
        t0 = origins[a] - pd.Timedelta(hours=SEQ)
        runs.append({
            "run_id": len(runs),
            "first_day": int(a), "last_day": int(b - 1), "n_days": int(b - a),
            "days": np.arange(a, b, dtype=np.int64),
            "segment": seg,
            "series": series,
            "series_time": pd.date_range(t0, periods=len(series), freq="h"),
            "day_base": base,
        })
    span = {
        "n_fit": int(n_fit), "n_eval": int(n_eval),
        "n_runs": len(runs), "n_gap_breaks": len(runs) - 1,
        "runs": [{"run_id": int(r["run_id"]), "first_day": r["first_day"], "last_day": r["last_day"],
                  "n_days": r["n_days"], "n_hours": int(len(r["series"])),
                  "segments": sorted(set(map(str, r["segment"]))),
                  "series_start": str(r["series_time"][0]), "series_end": str(r["series_time"][-1])}
                 for r in runs],
    }
    return runs, span


def role_of_day(i, n_fit, n_val):
    """The frozen chronology role of a panel day index."""
    if i < n_fit - n_val:
        return "train"
    if i < n_fit:
        return "val"
    return "test"


def role_hours(runs, n_fit, n_val):
    """Hour spans `(run_id, hour_start, hour_end)` whose day blocks hold a role.

    `hour_start:hour_end` covers whole 24h day blocks only, so a window whose
    target lives inside one of these spans is a target for exactly that role and
    never borrows a label from another role.
    """
    out = []
    for r in runs:
        labels = [role_of_day(int(d), n_fit, n_val) for d in r["days"]]
        start = 0
        for k in range(1, len(labels) + 1):
            if k == len(labels) or labels[k] != labels[start]:
                out.append((int(r["run_id"]), labels[start],
                            int(r["day_base"][start]), int(r["day_base"][k - 1]) + 24))
                start = k
    return out


def attach_s1(cells, seed):
    """Replay frozen S1 on every cell for one seed and attach the artefacts."""
    for key, c in cells.items():
        oof, u_eval, info = s1_artifacts(
            c["fi"], c["ei"], c["state_fit_raw"], c["state_eval_raw"], c["ef"], seed)
        c["blocks"] = oof["blocks"]
        c["oof_u"] = oof["u"]
        c["s1_oof_seconds"] = oof["training_seconds"]
        c["s1_info"] = info
        c["u_s1_eval"] = u_eval
        c["has_dir"] = np.isfinite(oof["u"]).all(axis=1)
        if c["has_dir"].any() and not np.array_equal(
                np.flatnonzero(c["has_dir"]), np.flatnonzero(c["fi"].repair_available & c["has_dir"])):
            raise RuntimeError(f"{key}: frozen S1 OOF directions exist outside repair-available rows")
        alpha, _, _, records = s1_alpha(c["ef"], c["oof_u"], c["fi"].repair_available, c["blocks"])
        c["alpha_s1"] = alpha
        c["alpha_records"] = records
    return cells


def frozen_s1_reference():
    """Registered frozen S1 cell medians used as the B0 replay target."""
    df = pd.read_csv(FROZEN_S1_CELLS)
    s1 = df[df.method.eq("S1_DailyPatch_GRU32")]
    return s1.set_index(["market", "host"])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def metric_row(y, hp, pred, tail, normal):
    """Frozen metric contract plus the secondary geometry metrics."""
    m = eval_metrics(y, hp, pred, tail, normal)
    m["RMSE"] = float(np.sqrt(m["MSE"]))
    m["Normal_harm_vs_host_pct"] = float(100 * (m["Normal_MAE"] / np.mean(np.abs(y[normal] - hp[normal])) - 1))
    m["Tail_harm_vs_host_pct"] = float(100 * (m["Tail_MAE"] / np.mean(np.abs(y[tail] - hp[tail])) - 1))
    return m


def day_level_mae(y, pred):
    """Per-day 24h MAE — the frozen day-level comparison unit."""
    return np.mean(np.abs(y - pred), axis=1)


def s1_row(c, method="S1_DailyPatch_GRU32"):
    """The frozen S1 proposal row for one cell (alpha fixed by the OOF rule)."""
    pred = c["hp"] + c["alpha_s1"] * c["u_s1_eval"] * c["scale"] * c["avail"][:, None]
    m = metric_row(c["y"], c["hp"], pred, c["tail"], c["normal"])
    true_u, valid = unit_shape_target(c["ee"])
    m.update({
        "shape_cosine": float(np.mean(np.sum(true_u * c["u_s1_eval"], axis=1)[valid])),
        "wrong_hemisphere_rate": float(np.mean(np.sum(c["ee"] * c["u_s1_eval"], axis=1) <= 0)),
        "alpha": float(c["alpha_s1"]),
        "oof_calibrated_utility": float(np.mean(utility(c["ef"][s1_complete_rowset(c["blocks"], c["fi"].repair_available)],
                                                        c["alpha_s1"] * c["oof_u"][s1_complete_rowset(c["blocks"], c["fi"].repair_available)]))),
        "parameter_count": int(c["s1_info"]["parameter_count"]),
        "training_seconds": float(c["s1_oof_seconds"] + c["s1_info"]["training_seconds"]),
        "inference_seconds": float(c["s1_info"]["inference_seconds"]),
        "inference_seconds_per_day": float(c["s1_info"]["inference_seconds_per_day"]),
    })
    return m, pred


def host_row(c):
    m = metric_row(c["y"], c["hp"], c["hp"], c["tail"], c["normal"])
    md = c["host_metadata"]
    m.update({
        "parameter_count": float(md.get("trainable_params_base", np.nan)),
        "training_seconds": float(md.get("T_base_train", np.nan)),
        "inference_seconds": float(md.get("T_base_infer", np.nan)),
        "inference_seconds_per_day": float(md.get("T_base_infer", np.nan)) / max(c["n_eval"], 1),
    })
    return m, c["hp"].copy()


__all__ = [
    "H", "HOSTS", "LEGAL", "MARKETS", "SEQ", "SEEDS", "TARGET", "TRACK",
    "ROOT", "SHAPE_EVIDENCE", "HRSI_EVIDENCE", "FROZEN_S1_CELLS", "S1_REPLAY_TOL", "VAL_FRACTION",
    "sha", "tree_sha", "hash_json",
    "build_cells", "rebuild_series", "role_of_day", "role_hours",
    "attach_s1", "frozen_s1_reference",
    "s1_artifacts", "s1_alpha", "s1_complete_rowset",
    "metric_row", "day_level_mae", "s1_row", "host_row",
    "chronological_blocks", "mae", "mse", "unit_shape_target", "eval_metrics",
    "load_frozen_cache", "fit_shape_outputs", "shape_oof", "pooled_alpha", "utility",
]
