"""Adversarial verification of the baseline-transfer stage (P3).

Controlling protocol:
    docs/current/HCH_FROZEN_METHOD_BASELINE_TRANSFER_ADJUDICATION_20260911.md

This module is written to *disagree* with the runner.  It never imports
``run_transfer``, ``finalize`` or ``transfers``; every aggregate, gate and
bootstrap interval is recomputed here from raw inputs with independently written
arithmetic, and the two are compared numerically.  A check that cannot be
recomputed is reported as ``NOT_VERIFIABLE`` rather than assumed to hold.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CURRENT = ROOT / "experiments/current/hch_frozen_method_baseline_transfer"
EVIDENCE = ROOT / "experiments/evidence/hch_frozen_method_baseline_transfer_20260911"
HRSI_EVIDENCE = ROOT / "experiments/evidence/hch_host_relative_state_interaction_20260911"
OFFICIAL_PIR = ROOT / "experiments/foundation/reference_deps/official_repos/PIR"

BASELINES = ("MatchedDirectResidual", "delta_Adapter_AdaY", "PIR_paper_protocol")
TOL = 1e-9
TOL_REPLAY = 1e-6
REPS = 1000
BOOT_SEED = 20260911
BLOCK = 7

FORBIDDEN_MODULES = ("pir", "cosa", "uec", "ompb")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Independently written primitives
# --------------------------------------------------------------------------

def my_block_bootstrap(diff, block=BLOCK, reps=REPS, seed=BOOT_SEED):
    """The protocol's fixed block bootstrap, written independently of the runner.

    The procedure -- contiguous blocks of `block` days in the frozen day order,
    blocks resampled with replacement, `reps` replicates from seed `seed` -- is
    fully specified, so an independent implementation must reproduce the interval
    bit-for-bit.  Using a different generator API would reproduce nothing and
    would make this check vacuous.
    """
    n = len(diff)
    blocks = [list(range(s, min(s + block, n))) for s in range(0, n, block)]
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=np.float64)
    acc = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        idx = []
        for c in rng.integers(0, len(blocks), size=len(blocks)):
            idx.extend(blocks[int(c)])
        acc[b] = np.mean(diff[np.asarray(idx, dtype=np.int64)])
    lo, hi = np.percentile(acc, 2.5), np.percentile(acc, 97.5)
    return {"observed_mean_diff": float(np.mean(diff)), "ci95_low": float(lo),
            "ci95_high": float(hi), "n_blocks": len(blocks),
            "frac_replicates_negative": float(np.mean(acc < 0))}


def my_gap_table(cell_df):
    out = {}
    for (m, h), g in cell_df.groupby(["market", "host"]):
        idx = g.set_index("method")["Overall_MAE"]
        vals = {b: float(idx[b]) for b in BASELINES if b in idx.index}
        best = min(vals, key=vals.get)
        s1 = float(idx["S1_DailyPatch_GRU32"])
        host = float(idx["Host"])
        out[f"{m}/{h}"] = {
            "best_baseline": best, "best_baseline_MAE": vals[best],
            "gap_to_best_baseline_pct": 100 * (s1 - vals[best]) / vals[best],
            "hch_gain_vs_host_pct": 100 * (host - s1) / host,
            "HCH_strict_best": bool(s1 < min(vals.values())),
            "HCH_within_half_pct_of_best": bool(100 * (s1 - vals[best]) / vals[best] <= 0.5),
            "normal_harm_vs_host_pct": float(
                g[g.method.eq("S1_DailyPatch_GRU32")]["Normal_harm_vs_host_pct"].iloc[0])}
    return out


def my_gates(gaps):
    g = pd.DataFrame.from_dict(gaps, orient="index")
    g.index.name = "cell"
    g = g.reset_index()
    g["market"] = g.cell.str.split("/").str[0]
    g["host"] = g.cell.str.split("/").str[1]

    gan = g[g.market.eq("GANSU_DA")]
    intl = g[~g.market.eq("GANSU_DA")]
    b1 = bool((gan.gap_to_best_baseline_pct <= 0.5).all()
              and (gan.HCH_strict_best.sum() >= 1)
              and (float(np.median(gan.hch_gain_vs_host_pct)) >= 4.0)
              and (gan.normal_harm_vs_host_pct <= 1.0).all())
    b2 = bool((intl.gap_to_best_baseline_pct <= 1.0).sum() >= 3
              and (intl.gap_to_best_baseline_pct <= 2.0).all()
              and (intl.normal_harm_vs_host_pct <= 1.0).all()
              and (intl.HCH_strict_best | intl.HCH_within_half_pct_of_best).sum() >= 2)
    b3 = bool(b1 and b2 and (g.hch_gain_vs_host_pct >= 0).sum() >= 5
              and float(np.median(g.hch_gain_vs_host_pct)) > 0
              and g.gap_to_best_baseline_pct.max() <= 2.0)
    return {"B1": b1, "B2": b2, "B3": b3,
            "n_host_nonworse": int((g.hch_gain_vs_host_pct >= 0).sum()),
            "median_gain_vs_host_pct": float(np.median(g.hch_gain_vs_host_pct)),
            "max_gap_pct": float(g.gap_to_best_baseline_pct.max())}


def import_scan():
    """Every ``import`` statement in this stage's own modules."""
    found = {}
    for f in sorted(CURRENT.glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        found[f.name] = sorted(mods)
    return found


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def main():
    checks = {}

    by_seed = pd.read_csv(EVIDENCE / "metrics_by_seed.csv")
    cell_df = pd.read_csv(EVIDENCE / "metrics_by_cell.csv")
    gaps = pd.read_csv(EVIDENCE / "best_baseline_gap_by_cell.csv")
    boot = pd.read_csv(EVIDENCE / "paired_block_bootstrap.csv")
    verdict = json.loads((EVIDENCE / "BASELINE_TRANSFER_VERDICT.json").read_text(encoding="utf-8"))

    # V1 cell medians re-derived from seed rows
    worst, detail = 0.0, {}
    for (m, h, meth), g in by_seed.groupby(["market", "host", "method"]):
        have = cell_df[(cell_df.market == m) & (cell_df.host == h) & (cell_df.method == meth)]
        if have.empty:
            detail.setdefault("missing", []).append(f"{m}/{h}/{meth}")
            continue
        d = abs(float(g["Overall_MAE"].median()) - float(have["Overall_MAE"].iloc[0]))
        worst = max(worst, d)
        if d > TOL:
            detail[f"{m}/{h}/{meth}"] = d
    checks["V1_cell_medians_reproduced"] = {"pass": worst <= TOL, "max_abs_diff": worst,
                                            "disagreements": detail}

    # V2 gap table re-derived
    mine = my_gap_table(cell_df)
    bad = {}
    for _, r in gaps.iterrows():
        k = f"{r['market']}/{r['host']}"
        ref = mine[k]
        for fld in ("gap_to_best_baseline_pct", "hch_gain_vs_host_pct", "best_baseline_MAE"):
            if abs(float(r[fld]) - float(ref[fld])) > 1e-9:
                bad.setdefault(k, {})[fld] = [float(r[fld]), float(ref[fld])]
        if str(r["best_baseline"]) != ref["best_baseline"]:
            bad.setdefault(k, {})["best_baseline"] = [str(r["best_baseline"]), ref["best_baseline"]]
        for fld in ("HCH_strict_best", "HCH_within_half_pct_of_best"):
            if bool(r[fld]) != bool(ref[fld]):
                bad.setdefault(k, {})[fld] = [bool(r[fld]), bool(ref[fld])]
    checks["V2_gap_table_reproduced"] = {"pass": not bad, "disagreements": bad}

    # V3 gates re-derived
    mg = my_gates(mine)
    vg = verdict["gates"]
    disagree = {k: [mg[k], bool(vg[k]["pass"])] for k in ("B1", "B2", "B3") if mg[k] != bool(vg[k]["pass"])}
    checks["V3_gates_reproduced"] = {"pass": not disagree, "mine": mg,
                                     "verdict": {k: bool(vg[k]["pass"]) for k in ("B1", "B2", "B3")},
                                     "disagreements": disagree}

    # V4 token follows the protocol decision rule
    expected = ("HCH_FROZEN_BASELINE_TRANSFER_INVALID" if not vg["B0"]["pass"] else
                "HCH_BASELINE_TRANSFER_TARGET_MET_FULL_PANEL_ALLOWED"
                if (mg["B1"] and mg["B2"] and mg["B3"]) else
                "HCH_BASELINE_TRANSFER_NOT_COMPETITIVE_NEW_IDEA_REQUIRED")
    checks["V4_token_rule"] = {"pass": verdict["token"] == expected,
                               "token": verdict["token"], "expected_from_rule": expected}

    # V5 bootstrap re-derived with an independently written resampler
    day_path = CURRENT / "_day_mae.npz"
    bad_boot = {}
    if not day_path.exists():
        bad_boot["_"] = "day-level MAE cache absent; bootstrap not verifiable"
    else:
        z = np.load(day_path)
        for _, r in boot.iterrows():
            # The pairing rule is pre-registered per method: HCH at the middle
            # panel seed, the baseline at its own contract seed (PIR's paper
            # seed differs from the panel seeds, so the two are not the same).
            key = (r["market"], r["host"], r["baseline"], int(r["baseline_seed"]))
            k_s1 = f"{key[0]}|{key[1]}|S1_DailyPatch_GRU32|{int(r['HCH_seed'])}"
            k_b = f"{key[0]}|{key[1]}|{key[2]}|{key[3]}"
            if k_s1 not in z or k_b not in z:
                bad_boot[str(key)] = "day-level vector absent"
                continue
            diff = z[k_s1] - z[k_b]
            mine_b = my_block_bootstrap(diff)
            for fld in ("observed_mean_diff", "ci95_low", "ci95_high", "frac_replicates_negative"):
                if abs(float(mine_b[fld]) - float(r[fld])) > 1e-9:
                    bad_boot.setdefault(str(key), {})[fld] = [float(r[fld]), float(mine_b[fld])]
            if int(mine_b["n_blocks"]) != int(r["n_blocks"]):
                bad_boot.setdefault(str(key), {})["n_blocks"] = [int(r["n_blocks"]), mine_b["n_blocks"]]
            if abs(float(r["HCH_day_MAE_mean"]) - float(z[k_s1].mean())) > 1e-9:
                bad_boot.setdefault(str(key), {})["HCH_day_MAE_mean"] = [
                    float(r["HCH_day_MAE_mean"]), float(z[k_s1].mean())]
            if abs(float(r["baseline_day_MAE_mean"]) - float(z[k_b].mean())) > 1e-9:
                bad_boot.setdefault(str(key), {})["baseline_day_MAE_mean"] = [
                    float(r["baseline_day_MAE_mean"]), float(z[k_b].mean())]
    checks["V5_bootstrap_reproduced"] = {"pass": not bad_boot, "disagreements": bad_boot,
                                         "replicates": REPS, "block_days": BLOCK, "seed": BOOT_SEED}

    # V6 PIR metric re-derived from the stored predictions
    pir_cache = CURRENT / "_pir_cache"
    bad_pir = {}
    if pir_cache.exists():
        from experiments.current.hch_frozen_method_baseline_transfer import panel
        cells, _, _, _ = panel.build_cells()
        for (m, h), c in cells.items():
            npz = pir_cache / f"{m}__{h}.npz"
            if not npz.exists():
                bad_pir[f"{m}/{h}"] = "prediction cache absent"
                continue
            pred = np.load(npz)["pred"]
            mae = float(np.mean(np.abs(c["y"] - pred)))
            row = cell_df[(cell_df.market == m) & (cell_df.host == h)
                          & (cell_df.method == "PIR_paper_protocol")]
            if abs(mae - float(row["Overall_MAE"].iloc[0])) > 1e-9:
                bad_pir[f"{m}/{h}"] = [mae, float(row["Overall_MAE"].iloc[0])]
            if pred.shape != c["y"].shape:
                bad_pir[f"{m}/{h}"] = ["shape", list(pred.shape), list(c["y"].shape)]
    else:
        bad_pir["_"] = "PIR cache absent"
    checks["V6_pir_metric_from_raw_predictions"] = {"pass": not bad_pir, "disagreements": bad_pir}

    # V7 frozen S1 / Host replay against the registered evidence
    ref = pd.read_csv(HRSI_EVIDENCE / "metrics_by_cell.csv")
    ref = ref[ref.method.eq("S1_DailyPatch_GRU32")].set_index(["market", "host"])
    s1 = by_seed[by_seed.method.eq("S1_DailyPatch_GRU32")].groupby(["market", "host"])["Overall_MAE"].median()
    diff = {f"{k[0]}/{k[1]}": float(abs(v - ref.loc[k, "Overall_MAE"])) for k, v in s1.items()}
    checks["V7_frozen_s1_replay"] = {"pass": max(diff.values()) <= TOL_REPLAY,
                                     "max_abs_diff": max(diff.values()), "detail": diff}

    # V8 import scan
    imports = import_scan()
    suspicious = {f: [x for x in mods if any(t in x.lower() for t in FORBIDDEN_MODULES)]
                  for f, mods in imports.items()}
    suspicious = {f: v for f, v in suspicious.items() if v}
    checks["V8_no_forbidden_imports"] = {"pass": not suspicious, "hits": suspicious,
                                         "imports": imports}

    # V9 the PIR transfer really loads the official module (checked out of process,
    #    so the stage's own sys.path surgery cannot mask the answer)
    import subprocess
    probe = (
        "import sys; sys.path.insert(0, r'%s');"
        "from experiments.current.hch_frozen_method_baseline_transfer import pir_transfer as pt;"
        "import importlib;"
        "m = sys.modules[pt.Exp_Long_Term_Forecast_PIR.__module__];"
        "print(m.__file__);"
        "print(sys.modules['models.PIR'].__file__ if 'models.PIR' in sys.modules else 'models.PIR not loaded');"
        "print(pt.PanelPIRExp.__mro__[1].__module__)" % str(ROOT))
    try:
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=900)
        lines = [ln.strip() for ln in out.stdout.strip().splitlines()]
        exp_file = Path(lines[0]) if lines else None
        exp_ok = exp_file is not None and exp_file.resolve() == (
            OFFICIAL_PIR / "exp/exp_long_term_forecasting_pir.py").resolve()
        mro_ok = len(lines) > 2 and lines[2] == "exp.exp_long_term_forecasting_pir"
        checks["V9_pir_module_identity"] = {
            "pass": bool(exp_ok and mro_ok),
            "loaded_exp_file": lines[0] if lines else None,
            "base_class_module": lines[2] if len(lines) > 2 else None,
            "stderr_tail": out.stderr[-400:] if out.returncode else "",
            "official_exp_sha256": sha(OFFICIAL_PIR / "exp/exp_long_term_forecasting_pir.py"),
            "official_model_sha256": sha(OFFICIAL_PIR / "models/PIR.py")}
    except Exception as exc:  # noqa: BLE001
        checks["V9_pir_module_identity"] = {"pass": False, "error": repr(exc)}

    checks["pass"] = all(c["pass"] for c in checks.values())
    out = {"pass": checks["pass"], "checks": checks}
    (EVIDENCE / "VERIFICATION.json").write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v["pass"] for k, v in checks.items() if k != "pass"}, indent=2))
    return 0 if checks["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
