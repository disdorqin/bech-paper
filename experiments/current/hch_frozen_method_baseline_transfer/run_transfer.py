"""Stage driver: frozen HCH-S1 vs the admitted offline baselines on six cells.

Controlling protocol:
    docs/current/HCH_FROZEN_METHOD_BASELINE_TRANSFER_ADJUDICATION_20260911.md

Executed comparison set (nothing else enters the main table):

    Host                          frozen, seed-independent
    Frozen HCH-S1                 replayed per seed, never modified
    MatchedDirectResidual         frozen internal same-information control
    delta-Adapter Ada-Y           audited official transfer
    PIR-paper-protocol            verified official backbone + PIR pair

Forbidden paths (``PIR_PROXY_10epoch_K10``, ``src/baselines/pir.py`` ridge/refiner
substitutes, the old 2-epoch delta wrapper, and COSA/UEC-STD/OMPB in the main
offline ranking) are never imported or executed by this module or anything it
imports.

Writes only under
    experiments/evidence/hch_frozen_method_baseline_transfer_20260911/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.current.hch_frozen_method_baseline_transfer import panel  # noqa: E402
from experiments.current.hch_frozen_method_baseline_transfer import transfers as tr  # noqa: E402

CURRENT = ROOT / "experiments/current/hch_frozen_method_baseline_transfer"
EVIDENCE = ROOT / "experiments/evidence/hch_frozen_method_baseline_transfer_20260911"
PIR_CACHE = CURRENT / "_pir_cache"
PROTOCOL = ROOT / "docs/current/HCH_FROZEN_METHOD_BASELINE_TRANSFER_ADJUDICATION_20260911.md"

BASELINES = ("MatchedDirectResidual", "delta_Adapter_AdaY", "PIR_paper_protocol")
METHOD_ORDER = ("Host", "S1_DailyPatch_GRU32", *BASELINES)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260911
BOOTSTRAP_BLOCK_DAYS = 7

GATE_B1_BASELINE_SLACK_PCT = 0.5
GATE_B1_MEDIAN_GAIN_MIN_PCT = 4.0
GATE_B2_TIGHT_PCT = 1.0
GATE_B2_LOOSE_PCT = 2.0
GATE_B2_BEST_SLACK_PCT = 0.5
GATE_NORMAL_HARM_MAX_PCT = 1.0


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------

SEEDED_METHODS = (
    ("MatchedDirectResidual", tr.run_direct, tr.DIRECT_LABEL),
    ("delta_Adapter_AdaY", tr.run_delta, tr.DELTA_LABEL),
)


def seeded_rows(cells, seeds):
    """One metric row per (cell, method, seed) for every seed-parameterised method."""
    rows = []
    for seed in seeds:
        t_seed = time.perf_counter()
        panel.attach_s1(cells, seed)
        for (m, h), c in cells.items():
            s1m, s1pred = panel.s1_row(c)
            rows.append(_row(m, h, "S1_DailyPatch_GRU32", seed, s1m,
                             {"fidelity": "FROZEN_METHOD_REPLAY",
                              "source": _source_of("S1_DailyPatch_GRU32"),
                              "pred": s1pred}))
            for name, fn, label in SEEDED_METHODS:
                res = fn(c, seed)
                mm = panel.metric_row(c["y"], c["hp"], res["pred"], c["tail"], c["normal"])
                mm["parameter_count"] = res["parameter_count"]
                mm["training_seconds"] = res["training_seconds"]
                mm["inference_seconds"] = res["inference_seconds"]
                mm["inference_seconds_per_day"] = res["inference_seconds"] / max(c["n_eval"], 1)
                rows.append(_row(m, h, name, seed, mm,
                                 {"fidelity": label, "source": _source_of(name),
                                  "pred": res["pred"]}))
        print(f"[seed {seed}] {time.perf_counter() - t_seed:.1f}s", flush=True)
    return rows


def _jsonable(x):
    """JSON needs string keys; the panel's access maps are keyed by tuples."""
    if isinstance(x, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): _jsonable(v)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def _source_of(method):
    return {
        "S1_DailyPatch_GRU32": "experiments/current/hch_frozen_method_baseline_transfer/panel.py::s1_row (frozen replay)",
        "MatchedDirectResidual": "experiments/current/hch_unified_da_shape_upgrade/run_canary.py::fit_direct",
        "delta_Adapter_AdaY": f"official delta-Adapter Adapter-X+Y @ {tr.DELTA_COMMIT} :: PostProcessingNet",
        "PIR_paper_protocol": "official PIR @ fc372bb :: models/PIR.py + exp/exp_long_term_forecasting_pir.py",
        "Host": "frozen Host cache",
    }[method]


def _row(market, host, method, seed, m, extra=None):
    row = {"market": market, "host": host, "method": method, "seed": int(seed),
           "Overall_MAE": m["Overall_MAE"], "Tail_MAE": m["Tail_MAE"],
           "Normal_MAE": m["Normal_MAE"], "MSE": m["MSE"], "RMSE": m["RMSE"],
           "relative_gain_vs_host_pct": m["relative_gain_pct"],
           "Normal_harm_vs_host_pct": m["Normal_harm_vs_host_pct"],
           "Tail_harm_vs_host_pct": m["Tail_harm_vs_host_pct"],
           "parameter_count": m.get("parameter_count", np.nan),
           "training_seconds": m.get("training_seconds", np.nan),
           "inference_seconds": m.get("inference_seconds", np.nan),
           "inference_seconds_per_day": m.get("inference_seconds_per_day", np.nan),
           "fidelity": None, "source": None}
    row.update(extra or {})
    return row


def host_rows(cells):
    out = []
    for (m, h), c in cells.items():
        m_, hp = panel.host_row(c)
        out.append(_row(m, h, "Host", -1, m_,
                        {"fidelity": "FROZEN_HOST", "source": _source_of("Host"),
                         "pred": hp}))
    return out


def pir_rows(cells, use_cache=True):
    """Official PIR at the paper seed contract; predictions cached to disk."""
    from experiments.current.hch_frozen_method_baseline_transfer import pir_transfer as pt
    PIR_CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    for (m, h), c in sorted(cells.items()):
        tag = f"{m}__{h}"
        npz, js = PIR_CACHE / f"{tag}.npz", PIR_CACHE / f"{tag}.json"
        t0 = time.perf_counter()
        if use_cache and npz.exists() and js.exists():
            pred = np.load(npz)["pred"]
            info = json.loads(js.read_text(encoding="utf-8"))
            print(f"[pir {tag}] cache hit", flush=True)
        else:
            res = pt.run_pir(c, PIR_CACHE / tag)
            pred = res.pop("pred")
            info = res
            np.savez_compressed(npz, pred=pred)
            js.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
            print(f"[pir {tag}] {time.perf_counter() - t0:.1f}s", flush=True)
        if pred.shape != c["y"].shape:
            raise RuntimeError(f"{tag}: cached PIR prediction shape {pred.shape} != {c['y'].shape}")
        if int(np.asarray(info["seed"])) != pt.ANCHOR_SEED:
            raise RuntimeError(f"{tag}: cached PIR row is not on the paper seed contract")
        m_ = panel.metric_row(c["y"], c["hp"], pred, c["tail"], c["normal"])
        m_["parameter_count"] = info["parameter_count"]
        m_["training_seconds"] = info["training_seconds"]
        m_["inference_seconds"] = info["inference_seconds"]
        m_["inference_seconds_per_day"] = info["inference_seconds"] / max(c["n_eval"], 1)
        r = _row(m, h, "PIR_paper_protocol", pt.ANCHOR_SEED, m_,
                 {"fidelity": pt.PIR_LABEL, "source": _source_of("PIR_paper_protocol"),
                  "backbone_parameter_count": info.get("backbone_parameter_count")})
        r["pred"] = pred
        out.append(r)
    return out


# --------------------------------------------------------------------------
# Aggregation and gates
# --------------------------------------------------------------------------

def aggregate(rows):
    """Cell medians over seeds for every method that has one than more seed."""
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("market", "host", "method", "fidelity", "source",
                                              "fidelity_label", "pred") and df[c].dtype != object]
    agg = (df.groupby(["market", "host", "method"])[num].median(numeric_only=True).reset_index())
    lab = df.groupby(["market", "host", "method"])["fidelity"].first().reset_index()
    src = df.groupby(["market", "host", "method"])["source"].first().reset_index()
    n = df.groupby(["market", "host", "method"]).size().reset_index(name="n_seeds")
    return agg.merge(lab, on=["market", "host", "method"]).merge(src, on=["market", "host", "method"]).merge(
        n, on=["market", "host", "method"])


def gap_table(cell_df):
    out = []
    for (m, h), g in cell_df.groupby(["market", "host"]):
        idx = g.set_index("method")
        if "S1_DailyPatch_GRU32" not in idx.index:
            continue
        base = idx.loc[list(BASELINES)]
        adm = base.dropna(subset=["Overall_MAE"])
        best_name = adm["Overall_MAE"].idxmin()
        best = float(adm.loc[best_name, "Overall_MAE"])
        s1 = float(idx.loc["S1_DailyPatch_GRU32", "Overall_MAE"])
        host = float(idx.loc["Host", "Overall_MAE"]) if "Host" in idx.index else np.nan
        out.append({
            "market": m, "host": h,
            "HCH_MAE": s1, "Host_MAE": host,
            "best_baseline": best_name, "best_baseline_MAE": best,
            "gap_to_best_baseline_pct": 100 * (s1 - best) / best,
            "hch_gain_vs_host_pct": 100 * (host - s1) / host,
            "MatchedDirectResidual_MAE": float(idx.loc["MatchedDirectResidual", "Overall_MAE"]),
            "delta_Adapter_AdaY_MAE": float(idx.loc["delta_Adapter_AdaY", "Overall_MAE"]),
            "PIR_paper_protocol_MAE": float(idx.loc["PIR_paper_protocol", "Overall_MAE"]),
            "HCH_strict_best": bool(s1 < adm["Overall_MAE"].min()),
            "HCH_within_half_pct_of_best": bool(100 * (s1 - best) / best <= GATE_B2_BEST_SLACK_PCT),
        })
    return pd.DataFrame(out).sort_values(["market", "host"]).reset_index(drop=True)


def day_mae_by_method(rows, cells):
    """Day-level MAE per (cell, method), on the frozen DIAG_EVAL day order.

    Every collected row carries its own raw prediction (Host rows at seed -1),
    so the vectors are taken from the same objects the headline metrics were
    computed from rather than recomputed from a separate code path.
    """
    out = {}
    for r in rows:
        if "pred" not in r:
            raise RuntimeError(f"{r['market']}/{r['host']}/{r['method']} carries no prediction")
        c = cells[(r["market"], r["host"])]
        out[(r["market"], r["host"], r["method"], int(r["seed"]))] = panel.day_level_mae(c["y"], r["pred"])
    return out


def block_bootstrap(diff, block=BOOTSTRAP_BLOCK_DAYS, reps=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Fixed contiguous block bootstrap over the frozen day order."""
    n = len(diff)
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        means[b] = float(np.mean(diff[idx]))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n_days": int(n), "n_blocks": len(blocks), "block_days": int(block),
            "observed_mean_diff": float(np.mean(diff)),
            "ci95_low": float(lo), "ci95_high": float(hi),
            "frac_replicates_negative": float(np.mean(means < 0)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


# The paired day-level comparison uses one pre-registered seed per method, so
# the pairing rule cannot depend on which baseline happens to win a cell.
#
#   HCH-S1        seed 17 -- the numerically middle panel seed, fixed by rule
#                 before any result existed (S1 is a frozen replay; the seed only
#                 selects the model initialisation).
#   Direct/delta  seed 17, the same pre-registered seed.
#   PIR           seed 2021, the official papers' own single-seed contract.
#
# Exactly one bootstrap row per cell.  Every per-seed day vector is still
# persisted in ``_day_mae.npz`` and every per-seed headline metric in
# ``metrics_by_seed.csv``; seed variation is reported there, not in the interval.
PRIMARY_SEED = sorted(panel.SEEDS)[len(panel.SEEDS) // 2]
PRIMARY_PIR_SEED = 2021


def baseline_seed(bname):
    return PRIMARY_PIR_SEED if bname == "PIR_paper_protocol" else PRIMARY_SEED


def save_day_mae(day):
    """Persist the day-level MAE vectors so the bootstrap can be recomputed independently."""
    flat = {f"{m}|{h}|{meth}|{seed}": v for (m, h, meth, seed), v in day.items()}
    np.savez_compressed(CURRENT / "_day_mae.npz", **flat)
    return {k: list(map(float, v)) for k, v in flat.items()}


def bootstrap_table(rows, gaps, cells, day):
    seeds = {}
    for r in rows:
        seeds.setdefault((r["market"], r["host"], r["method"]), set()).add(int(r["seed"]))
    out = []
    for _, g in gaps.iterrows():
        m, h, bname = g["market"], g["host"], g["best_baseline"]
        bseed = baseline_seed(bname)
        if bseed not in seeds[(m, h, bname)]:
            raise RuntimeError(f"{m}/{h}: {bname} has no run at its contract seed {bseed}")
        if PRIMARY_SEED not in seeds[(m, h, "S1_DailyPatch_GRU32")]:
            raise RuntimeError(f"{m}/{h}: S1 has no run at the pre-registered seed {PRIMARY_SEED}")
        s1 = day[(m, h, "S1_DailyPatch_GRU32", PRIMARY_SEED)]
        b = day[(m, h, bname, bseed)]
        if len(s1) != len(b):
            raise RuntimeError(f"{m}/{h}: day-level vectors disagree in length")
        r = block_bootstrap(s1 - b)
        r.update({"market": m, "host": h, "HCH_method": "S1_DailyPatch_GRU32",
                  "HCH_seed": int(PRIMARY_SEED), "baseline": bname,
                  "baseline_seed": int(bseed),
                  "HCH_day_MAE_mean": float(s1.mean()),
                  "baseline_day_MAE_mean": float(b.mean())})
        out.append(r)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(panel.SEEDS))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--stamp", default=None, help="run stamp for the provenance record")
    ns = ap.parse_args(argv)
    stamp = ns.stamp or time.strftime("%Y%m%d_%H%M%S")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cells, host_paths, host_hash, access = panel.build_cells()

    # The run record lives in its own file: ``_access_audit.json`` is owned
    # solely by ``audit_access.py`` so that the partition-access artifact has
    # exactly one writer and one source hash.
    audit = {**access,
             "host_cache_sha256": host_hash,
             "host_cache_paths": {k: str(Path(v).relative_to(ROOT)) for k, v in host_paths.items()},
             "seeds_run": list(map(int, ns.seeds)),
             "stamp": stamp,
             "protocol": str(PROTOCOL.relative_to(ROOT)),
             "protocol_sha256": panel.sha(PROTOCOL),
             "cell_gap_breaks": {f"{m}/{h}": int(c["gap_breaks"]) for (m, h), c in cells.items()},
             "cell_split": {f"{m}/{h}": c["split_manifest"] for (m, h), c in cells.items()}}
    (CURRENT / "_run_audit.json").write_text(
        json.dumps(_jsonable(audit), indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    rows = host_rows(cells)
    rows += seeded_rows(cells, ns.seeds)
    rows += pir_rows(cells, use_cache=not ns.no_cache)

    by_seed = pd.DataFrame([{k: v for k, v in r.items() if k != "pred"} for r in rows])
    by_seed.to_csv(CURRENT / "metrics_by_seed.csv", index=False)
    by_seed.to_csv(EVIDENCE / "metrics_by_seed.csv", index=False)

    cell_df = aggregate(rows)
    cell_df.to_csv(CURRENT / "metrics_by_cell_raw.csv", index=False)
    gaps = gap_table(cell_df)
    gaps.to_csv(CURRENT / "best_baseline_gap_by_cell.csv", index=False)
    day = day_mae_by_method(rows, cells)
    save_day_mae(day)
    boot = bootstrap_table(rows, gaps, cells, day)
    boot.to_csv(CURRENT / "paired_block_bootstrap.csv", index=False)

    print(gaps.to_string(index=False))
    print(boot.to_string(index=False))
    print(f"elapsed {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
