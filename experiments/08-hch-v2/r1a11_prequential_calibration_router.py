"""R1A.11 Prequential Cross-Fitted Calibration Authorization (plan v0.1).

R1A.10 used one fixed `S3M-prefix fit -> S3M-suffix validate`; for rare-fire
domains the suffix held only 1-2 action events, so Gate D's moving-block
bootstrap LCB landed exactly on 0 and the conservative router kept C0
(CALIBRATOR_ROUTING_UNRESOLVED).

R1A.11 changes ONLY the evidence-extraction protocol: rolling-origin /
prequential cross-fitting over the whole S3M. Every evaluated day satisfies

    Y_t notin (C3 fit set used for day t).

Fixed protocol (§3): initial burn-in B = 30 days; evaluation block H = 7 days.
If |S3M| < B + 14 -> INSUFFICIENT_PREQUENTIAL_EVIDENCE -> C0 (NEM stays C0).

Selector v2 (§5, default-to-C0, no learned router):
  A  initial_fit_days >= 30 and n_oos_eval_days >= 14
  B  over ALL prequential OOS rows: net_C0 <= 0 or harmful_C0 >= 0.50
  C  net_C3 > net_C0 and harm_C3 <= harm_C0
  D  moving-block bootstrap (7d block, 1000 resamples, one-sided 90% LCB,
     per-domain deterministic seed): LCB > 0  (strictly; not relaxed)

Final fit (§6): if C3 authorized -> refit final local isotonic on the FULL
S3M, freeze, S3C fits the selected estimator's own DVG q, S4 development
confirmation only.

Prohibited (§8): regularized/shrunk isotonic, C3 hyperparameter tuning,
changing B/H/alpha after the result, learned router, S3C/S4 in selection,
new market/host, IAH/CRPS/DVG changes.

Verdicts (§11): PREQUENTIAL_CALIBRATION_ROUTING_SUPPORTED (GREEN) /
RARE_EVENT_EVIDENCE_LIMITED (YELLOW-A, map stable) /
LOCAL_CALIBRATOR_VARIANCE_LIMITED (YELLOW-B, map unstable) /
LOCAL_ISOTONIC_UPPER_BOUND_NOT_ROBUST (RED).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

import r1a9_action_calibration as M  # reuse C0/C3, collection, S3C/S4 eval

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

SEED = 20260813
BURN_IN_DAYS = 30
BLOCK_DAYS = 7
GATE_A_MIN_OOS_DAYS = 14
GATE_B_HARMFUL_THRESHOLD = 0.50
N_RESAMPLE = 1000
ALPHA = 0.10
MIN_FIT_SAMPLES = 8
MAP_GRID = np.linspace(-1.0, 1.0, 21)
MAP_DRIFT_MEAN_THRESHOLD = 0.15
MAP_DRIFT_MAX_THRESHOLD = 0.40


def _f(x):
    return f"{x:.3f}" if x is not None and pd.notna(x) else "NaN"


def _g(x):
    return f"{x:+.4f}" if x is not None and pd.notna(x) else "NaN"


def _r(lo, hi):
    if lo is None or hi is None:
        return None
    return f"[{lo:.3f},{hi:.3f}]"


def domain_bootstrap_seed(domain, global_seed=SEED, stage="gate_d"):
    """Per-domain deterministic seed (§1.1): same domain -> same bootstrap
    regardless of execution order. Stable hash, not Python's randomized hash."""
    key = f"{global_seed}:{domain}:{stage}".encode()
    return int(hashlib.md5(key).hexdigest(), 16) % (2**31)


def _iso_stats(iso):
    if iso is None:
        return {"plateau_count": None, "map_range_lo": None,
                "map_range_hi": None, "max_jump": None,
                "grid_pred": None}
    xs = np.asarray(iso.X_thresholds_, dtype=np.float64)
    ys = np.asarray(iso.y_thresholds_, dtype=np.float64)
    uniq = np.unique(ys)
    jumps = np.abs(np.diff(ys))
    grid = np.clip(iso.predict(MAP_GRID), -1.0, 1.0)
    return {"plateau_count": int(len(uniq)),
            "map_range_lo": float(uniq.min()),
            "map_range_hi": float(uniq.max()),
            "max_jump": float(jumps.max()) if len(jumps) else 0.0,
            "grid_pred": grid}


def hour_rows(days_subset, direction):
    """(s, Y) hour pairs for a direction over a date-ordered day subset.
    Down: m=mm>0 & valid, s=sd, Y=gd/max(mm).  Up: mp>0, s=su, Y=gu/mp."""
    sk, gk, mk = ("sd", "gd", "mm") if direction == "d" else \
        ("su", "gu", "mp")
    ss, yy = [], []
    for day in days_subset:
        m = day["vm"] & (day[mk] > 0)
        s = np.asarray(day[sk], dtype=float)[m]
        g = np.asarray(day[gk], dtype=float)[m]
        mag = np.asarray(day[mk], dtype=float)[m]
        ss.append(s)
        yy.append(g / np.maximum(mag, 1e-12))
    if not ss:
        return np.array([]), np.array([])
    return np.concatenate(ss), np.concatenate(yy)


def fit_direction(iso_d, iso_u):
    iso_d.fit(*iso_d)
    iso_u.fit(*iso_u)


class _RollingC3(M.Calibrator):
    """C3 frozen at one rolling fold; applies a fixed (iso_d, iso_u)."""
    name = "C3_rolling"

    def __init__(self, iso_d, iso_u):
        self.iso_d = iso_d
        self.iso_u = iso_u

    def apply(self, s, direction, domain=None):
        iso = self.iso_d if direction == "d" else self.iso_u
        s = np.asarray(s, dtype=float)
        if iso is None:
            return np.clip(s, -1.0, 1.0)
        return np.clip(iso.predict(s), -1.0, 1.0)


def fit_c3_from_days(days_subset):
    """Fit per-direction isotonic maps from a date-ordered day subset."""
    iso_params = {}
    for direction in ("d", "u"):
        s, y = hour_rows(days_subset, direction)
        m2 = np.isfinite(s) & np.isfinite(y)
        s, y = s[m2], y[m2]
        if len(s) < MIN_FIT_SAMPLES:
            iso_params[direction] = None
            continue
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(s, y)
        iso_params[direction] = iso
    return iso_params


def _summarize_oos(rows):
    """Decision-level summary over OOS rows (execute iff A_hat>0)."""
    if not rows:
        return {"n": 0, "net": None, "harmful": 0.0, "gain_rel": None,
                "release": 0.0, "n_fire": 0}
    A = np.array([r["A_true"] for r in rows], dtype=np.float64)
    ex = np.array([r["A_hat"] for r in rows], dtype=np.float64) > 0
    rel = A[ex]
    return {"n": int(len(rows)),
            "net": float(rel.sum() / len(rows)) if len(rows) else 0.0,
            "harmful": float((rel < 0).mean()) if len(rel) else 0.0,
            "gain_rel": float(rel.mean()) if len(rel) else None,
            "release": float(ex.mean()), "n_fire": int(ex.sum())}


def _summarize_paired(rows, prefix):
    """Decision-level summary from paired prequential rows for one estimator."""
    if not rows:
        return {"n": 0, "net": None, "harmful": 0.0, "gain_rel": None,
                "release": 0.0, "n_fire": 0}
    A = np.array([r[f"{prefix}_a_true"] for r in rows], dtype=np.float64)
    ex = np.array([r[f"{prefix}_execute"] for r in rows], dtype=bool)
    rel = A[ex]
    return {"n": int(len(rows)),
            "net": float(rel.sum() / len(rows)) if len(rows) else 0.0,
            "harmful": float((rel < 0).mean()) if len(rel) else 0.0,
            "gain_rel": float(rel.mean()) if len(rel) else None,
            "release": float(ex.mean()), "n_fire": int(ex.sum())}


def moving_block_bootstrap_domain(deltas, domain, block=BLOCK_DAYS,
                                  n_resample=N_RESAMPLE, alpha=ALPHA,
                                  global_seed=SEED):
    """Moving-block bootstrap with per-domain deterministic seed (§1.1)."""
    d = np.asarray(deltas, dtype=np.float64)
    n = len(d)
    seed_g = domain_bootstrap_seed(domain, global_seed)
    rng = np.random.default_rng(seed_g)
    if n < 2:
        m = float(d[0]) if n else 0.0
        return m, m, float(d[0] > 0) if n else 0.0, seed_g
    n_blocks = int(np.ceil(n / block))
    max_start = max(1, n - block + 1)
    means = np.empty(n_resample)
    for i in range(n_resample):
        starts = rng.integers(0, max_start, size=n_blocks)
        sample = np.concatenate([d[s:s + block] for s in starts])[:n]
        means[i] = sample.mean()
    lcb = float(np.percentile(means, 100.0 * alpha))
    return float(means.mean()), lcb, float((means > 0).mean()), seed_g


class PrequentialCalibrationEvaluator:
    """Rolling-origin OOS evidence extraction + transparent pre-S4 selector."""

    def __init__(self, cfg):
        self.cfg = cfg

    def build_rolling_folds(self, dd):
        """Ordered S3M days (train + val blocks by date) -> folds + reason."""
        days = sorted([d for d in dd["days"]
                       if d["block"] in ("train", "val")],
                      key=lambda d: d["date"])
        n = len(days)
        if n < self.cfg["burn_in"] + self.cfg["min_oos_days"]:
            return [], "INSUFFICIENT_PREQUENTIAL_EVIDENCE", days, n
        folds = []
        start = self.cfg["burn_in"]
        block_id = 0
        while start < n:
            end = min(start + BLOCK_DAYS, n)
            folds.append({
                "block_id": block_id,
                "fit_days": days[:start],
                "eval_days": days[start:end],
                "fit_day_count": start,
                "eval_day_count": end - start,
                "fit_end_date": days[start - 1]["date"],
                "remainder": (end - start) < BLOCK_DAYS,
            })
            start = end
            block_id += 1
        return folds, "OK", days, n

    def evaluate_block(self, dom, fold, c0):
        """Evaluate C0 and (fold-frozen) C3 on one OOS block."""
        eval_days = fold["eval_days"]
        rows_c0 = M.evaluate_days(c0, {"domain": dom, "days": eval_days})
        iso_params = fit_c3_from_days(fold["fit_days"])
        c3 = _RollingC3(iso_params["d"], iso_params["u"])
        rows_c3 = M.evaluate_days(c3, {"domain": dom, "days": eval_days})
        return rows_c0, rows_c3, c3

    def collect_paired_oos_rows(self, dom, folds, c0):
        """Per-OOS-day rows: C0/C3 A_hat, execute, A_true, paired delta."""
        out = []
        for fold in folds:
            rows_c0, rows_c3, c3 = self.evaluate_block(dom, fold, c0)
            for r0, r3 in zip(rows_c0, rows_c3):
                ex0 = r0["A_hat"] > 0
                ex3 = r3["A_hat"] > 0
                v0 = r0["A_true"] * ex0
                v3 = r3["A_true"] * ex3
                out.append({
                    "date": r0["date"], "block": r0["block"],
                    "block_id": fold["block_id"],
                    "fit_day_count": fold["fit_day_count"],
                    "c0_a_hat": r0["A_hat"], "c0_execute": int(ex0),
                    "c0_a_true": r0["A_true"],
                    "c3_a_hat": r3["A_hat"], "c3_execute": int(ex3),
                    "c3_a_true": r3["A_true"],
                    "c0_harm": int(ex0 and r0["A_true"] < 0),
                    "c3_harm": int(ex3 and r3["A_true"] < 0),
                    "delta": v3 - v0,
                })
        return out

    def map_stability(self, dom, folds):
        """Per-fold map stats + grid snapshots + D_map between neighbours."""
        rows = []
        prev_grid = {"d": None, "u": None}
        for i, fold in enumerate(folds):
            iso_params = fit_c3_from_days(fold["fit_days"])
            for direction in ("d", "u"):
                st = _iso_stats(iso_params[direction])
                grid = st.pop("grid_pred")
                d_map = None
                if prev_grid[direction] is not None and grid is not None:
                    d_map = float(np.mean(np.abs(grid - prev_grid[direction])))
                rows.append({
                    "domain": dom, "block_id": i, "direction": direction,
                    "fit_day_count": fold["fit_day_count"],
                    "plateau": st["plateau_count"],
                    "range_lo": st["map_range_lo"],
                    "range_hi": st["map_range_hi"],
                    "max_jump": st["max_jump"],
                    "d_map_from_prev": d_map,
                })
                prev_grid[direction] = grid
        return rows

    def select(self, dom, oos_rows, folds):
        """Gates A/B/C/D over prequential OOS rows; default-to-C0."""
        s0 = _summarize_paired(oos_rows, "c0")
        s3 = _summarize_paired(oos_rows, "c3")
        gates = {"A_prequential": None, "B_raw_problem": None,
                 "C_value_improve": None, "D_bootstrap": None}
        boot = None
        if not folds:
            return "C0", "INSUFFICIENT_PREQUENTIAL_EVIDENCE_KEEP_C0", \
                s0, s3, gates, boot
        initial_fit = folds[0]["fit_day_count"]
        n_oos = len(oos_rows)
        if initial_fit < self.cfg["burn_in"] or \
                n_oos < self.cfg["min_oos_days"]:
            gates["A_prequential"] = False
            return "C0", "INSUFFICIENT_PREQUENTIAL_EVIDENCE_KEEP_C0", \
                s0, s3, gates, boot
        gates["A_prequential"] = True
        if not (s0["net"] is not None and
                (s0["net"] <= 0
                 or s0["harmful"] >= self.cfg["harm_threshold"])):
            gates["B_raw_problem"] = False
            return "C0", "RAW_HEALTHY_KEEP_C0", s0, s3, gates, boot
        gates["B_raw_problem"] = True
        if not (s3["net"] is not None and s3["net"] > s0["net"]
                and s3["harmful"] <= s0["harmful"]):
            gates["C_value_improve"] = False
            return "C0", "C3_VALUE_NOT_BETTER_KEEP_C0", s0, s3, gates, boot
        gates["C_value_improve"] = True
        deltas = np.array([r["delta"] for r in oos_rows], dtype=np.float64)
        mean_d, lcb, frac_pos, seed_g = moving_block_bootstrap_domain(
            deltas, dom, self.cfg["block"], self.cfg["n_resample"],
            self.cfg["alpha"])
        boot = {"n_oos_days": len(deltas), "mean_delta": mean_d, "lcb": lcb,
                "frac_gt0": frac_pos, "seed_g": seed_g}
        gates["D_bootstrap"] = bool(lcb > 0)
        if lcb > 0:
            return "C3", "PREQUENTIAL_AUTHORIZATION_C3", s0, s3, gates, boot
        return "C0", "C3_IMPROVEMENT_UNCERTAIN_KEEP_C0", s0, s3, gates, boot


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--variant", type=str, default="learned_sig")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    results_dir = HERE / "results"
    if args.artifacts:
        artifact_dir = Path(args.artifacts)
    else:
        dirs = sorted(results_dir.glob("R1A_[0-9]*"), key=lambda p: p.name)
        artifact_dir = dirs[-1] if dirs else None
        if artifact_dir is None:
            raise SystemExit("no R1A_* artifact dir")
    out_dir = Path(args.out) if args.out else \
        results_dir / f"R1A11_PREQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig = out_dir / "figures"
    out_fig.mkdir(exist_ok=True)
    print(f"[R1A.11] frozen artifacts: {artifact_dir}")
    print(f"[R1A.11] out: {out_dir}", flush=True)

    cfg = {"burn_in": BURN_IN_DAYS, "block": BLOCK_DAYS,
           "min_oos_days": GATE_A_MIN_OOS_DAYS,
           "harm_threshold": GATE_B_HARMFUL_THRESHOLD,
           "n_resample": N_RESAMPLE, "alpha": ALPHA, "seed": args.seed}
    with open(out_dir / "prequential_config.json", "w", encoding="utf-8") as f:
        json.dump({"variant": args.variant, **cfg,
                   "code_commit": M._git_head()}, f, indent=2)

    domain_data = {}
    for ds_key, bb in M.R.DOMAINS:
        dom = f"{ds_key}:{bb}"
        print(f"[R1A.11] collect {dom} ...", flush=True)
        dd = M.collect_domain(artifact_dir, ds_key, bb, args.variant)
        domain_data[dom] = dd
        for pr in dd["problems"]:
            print(f"      WARN {pr}", flush=True)

    c0 = M.RawIAH()
    evaluator = PrequentialCalibrationEvaluator(cfg)

    fold_rows, oos_rows_all, boot_rows, sel_rows = [], [], [], []
    map_rows = []
    for dom, dd in domain_data.items():
        print(f"[R1A.11] rolling {dom} ...", flush=True)
        folds, proto, _days, n_total = evaluator.build_rolling_folds(dd)
        if folds:
            oos_rows = evaluator.collect_paired_oos_rows(dom, folds, c0)
            for r in oos_rows:
                oos_rows_all.append({**{"domain": dom}, **r})
            for f in folds:
                fold_rows.append({"domain": dom, "block_id": f["block_id"],
                                  "fit_end_date": str(f["fit_end_date"]),
                                  "fit_day_count": f["fit_day_count"],
                                  "eval_day_count": f["eval_day_count"],
                                  "remainder": f["remainder"]})
            map_rows.extend(evaluator.map_stability(dom, folds))
        else:
            oos_rows = []
        sel, reason, s0, s3, gates, boot = evaluator.select(dom, oos_rows,
                                                            folds)
        boot_rows.append({"domain": dom,
                          "total_s3m_days": n_total,
                          "n_blocks": len(folds),
                          "n_oos_days": len(oos_rows),
                          **({} if boot is None else boot),
                          "reason": reason})
        sel_rows.append({"domain": dom, "selected": sel, "reason": reason,
                         "n_blocks": len(folds), "n_oos_days": len(oos_rows),
                         **{f"gate_{k}": v for k, v in gates.items()}})
        print(f"      {dom}: -> {sel} ({reason}); n_blocks={len(folds)} "
              f"n_oos={len(oos_rows)}", flush=True)

    sel_df = pd.DataFrame(sel_rows)
    sel_map = dict(zip(sel_df["domain"], sel_df["selected"]))

    # ---- final estimator per domain (C3 refit on FULL S3M if authorized) ----
    rows_final = {}
    for dom, dd in domain_data.items():
        if sel_map.get(dom) == "C3":
            s_d, y_d = hour_rows(
                [d for d in dd["days"] if d["block"] in ("train", "val")], "d")
            s_u, y_u = hour_rows(
                [d for d in dd["days"] if d["block"] in ("train", "val")], "u")
            final_c3 = M.LocalIsotonic()
            final_c3.fit([{"domain": dom, "sd": s_d, "Yd": y_d,
                           "su": s_u, "Yu": y_u}])
            rows_final[dom] = M.evaluate_days(final_c3, dd)
        else:
            rows_final[dom] = M.evaluate_days(c0, dd)

    # ---- S3C DVG + S4 metrics for R0 (C0 everywhere) and R11 (gated) ----
    s4_pol, s4_pt, dvg_rows = [], [], []
    rows_r0 = {dom: M.evaluate_days(c0, dd) for dom, dd in domain_data.items()}
    for dom, dd in domain_data.items():
        for ver, rows in (("R0", rows_r0[dom]),):
            dv = M.dvg_and_s4(rows, M.R.ALPHA)
            pm = M.point_metrics(dv, None)
            s4_pol.append({"domain": dom, "version": ver,
                           "calibrator": ("C3" if sel_map.get(dom) == "C3"
                                          else "C0_raw"),
                           "selected": sel_map.get(dom),
                           "q": dv["q"], "n_calib": dv["n_calib"],
                           "n_eval": dv["n_eval"],
                           "release": dv["release_rate"],
                           "identity": dv["identity_rate"],
                           "harmful": dv["harmful_rate"],
                           "gain_rel": dv["mean_gain_release"],
                           "net": dv["net_value"], "coverage": dv["coverage"]})
            s4_pt.append({"domain": dom, "version": ver,
                          "calibrator": ("C3" if sel_map.get(dom) == "C3"
                                         else "C0_raw"), **pm})
            dvg_rows.append({"domain": dom, "version": ver,
                             "q": dv["q"], "n_calib": dv["n_calib"]})
        dv = M.dvg_and_s4(rows_final[dom], M.R.ALPHA)
        pm = M.point_metrics(dv, None)
        s4_pol.append({"domain": dom, "version": "R11",
                       "calibrator": ("C3" if sel_map.get(dom) == "C3"
                                      else "C0_raw"),
                       "selected": sel_map.get(dom),
                       "q": dv["q"], "n_calib": dv["n_calib"],
                       "n_eval": dv["n_eval"],
                       "release": dv["release_rate"],
                       "identity": dv["identity_rate"],
                       "harmful": dv["harmful_rate"],
                       "gain_rel": dv["mean_gain_release"],
                       "net": dv["net_value"], "coverage": dv["coverage"]})
        s4_pt.append({"domain": dom, "version": "R11",
                      "calibrator": ("C3" if sel_map.get(dom) == "C3"
                                     else "C0_raw"), **pm})
        dvg_rows.append({"domain": dom, "version": "R11",
                         "q": dv["q"], "n_calib": dv["n_calib"]})

    s4_df = pd.DataFrame(s4_pol)
    pt_df = pd.DataFrame(s4_pt)
    map_df = pd.DataFrame(map_rows)

    # ---- figures ----
    def fig_cumulative():
        for dom in domain_data:
            rows = [r for r in oos_rows_all if r["domain"] == dom]
            if not rows:
                continue
            df = pd.DataFrame(rows).sort_values("date")
            cum = df["delta"].cumsum()
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.plot(range(len(df)), cum, "-o", ms=2, lw=1)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xlabel("prequential OOS day index")
            ax.set_ylabel(r"cumulative $\sum \Delta_t$")
            ax.set_title(f"cumulative paired value: {dom}", fontsize=9)
            ax.tick_params(labelsize=7)
            fig.tight_layout()
            fig.savefig(out_fig / f"cumulative_delta_{dom.replace(':', '_')}.png",
                        dpi=130)
            plt.close(fig)

    fig_cumulative()
    # rolling map plots need fold fits; redraw properly
    for dom, dd in domain_data.items():
        folds, proto, _d, _n = evaluator.build_rolling_folds(dd)
        if not folds:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
        for ax, direction in zip(axes, ("d", "u")):
            ax.plot(MAP_GRID, MAP_GRID, "--", color="gray", lw=0.8,
                    label="identity")
            for i, f in enumerate(folds):
                iso_params = fit_c3_from_days(f["fit_days"])
                iso = iso_params[direction]
                if iso is None:
                    continue
                pred = np.clip(iso.predict(MAP_GRID), -1.0, 1.0)
                ax.plot(MAP_GRID, pred, lw=0.9, alpha=0.55,
                        color=plt.cm.viridis(i / max(len(folds) - 1, 1)))
            ax.axhline(0, color="k", lw=0.5, alpha=0.4)
            ax.set_title(f"{direction}", fontsize=8)
            ax.tick_params(labelsize=6)
        fig.suptitle(f"rolling C3 maps: {dom} "
                     f"({len(folds)} folds)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_fig / f"rolling_map_{dom.replace(':', '_')}.png",
                    dpi=130)
        plt.close(fig)

    # ---- CSVs ----
    pd.DataFrame(fold_rows).to_csv(out_dir / "rolling_folds.csv", index=False)
    pd.DataFrame(oos_rows_all).to_csv(out_dir / "paired_oos_value.csv",
                                      index=False)
    oos_sum = []
    for dom in domain_data:
        rows = [r for r in oos_rows_all if r["domain"] == dom]
        s0 = _summarize_paired(rows, "c0")
        s3 = _summarize_paired(rows, "c3")
        boot = next((b for b in boot_rows if b["domain"] == dom), {})
        oos_sum.append({
            "domain": dom,
            "total_s3m_days": boot.get("total_s3m_days"),
            "initial_fit_days": cfg["burn_in"] if boot.get("n_blocks") else None,
            "n_blocks": boot.get("n_blocks"),
            "n_oos_eval_days": boot.get("n_oos_days"),
            "c0_fire_days": s0["n_fire"], "c3_fire_days": s3["n_fire"],
            "c0_harm": s0["harmful"], "c3_harm": s3["harmful"],
            "c0_net": s0["net"], "c3_net": s3["net"],
            "mean_paired_delta": boot.get("mean_delta"),
            "lcb90": boot.get("lcb"),
            "frac_bootstrap_gt0": boot.get("frac_gt0"),
            "selected": sel_map.get(dom),
            "reason": next((r["reason"] for r in sel_rows
                            if r["domain"] == dom), None),
        })
    pd.DataFrame(oos_sum).to_csv(out_dir / "oos_summary_by_domain.csv",
                                 index=False)
    pd.DataFrame(boot_rows).to_csv(out_dir / "bootstrap_by_domain.csv",
                                   index=False)
    map_df.to_csv(out_dir / "map_stability.csv", index=False)
    sel_df.to_csv(out_dir / "selected_calibrator_by_domain.csv", index=False)
    pd.DataFrame(dvg_rows).to_csv(out_dir / "s3c_dvg.csv", index=False)
    s4_df.to_csv(out_dir / "s4_policy_metrics.csv", index=False)
    pt_df.to_csv(out_dir / "s4_point_metrics.csv", index=False)
    with open(out_dir / "code_commit.txt", "w", encoding="utf-8") as f:
        f.write(f"{M._git_head()}\n")
        f.write(f"run: {datetime.now().isoformat(timespec='seconds')}\n")

    # ------------------------------------------------------------ verdict ----
    r0 = {r["domain"]: r for _, r in s4_df[s4_df["version"] == "R0"].iterrows()}
    r11 = {r["domain"]: r for _, r in
           s4_df[s4_df["version"] == "R11"].iterrows()}

    def macro(d):
        vals = [d[dom]["net"] for dom in d if d[dom]["net"] is not None]
        return float(np.mean(vals)) if vals else None

    def worst(d):
        vals = [d[dom]["net"] for dom in d if d[dom]["net"] is not None]
        return float(np.min(vals)) if vals else None

    macro_r0, macro_r11 = macro(r0), macro(r11)
    worst_r0, worst_r11 = worst(r0), worst(r11)
    sel_sel = dict(zip(sel_df["domain"], sel_df["selected"]))
    pjm = r11["LAGO_PJM:MLP"]
    pjm_r0 = r0["LAGO_PJM:MLP"]
    pjm_authorized = sel_sel.get("LAGO_PJM:MLP") == "C3"
    pjm_boot = next((b for b in boot_rows
                     if b["domain"] == "LAGO_PJM:MLP"), {})
    pjm_lcb = pjm_boot.get("lcb")
    pjm_reduced = (pjm["harmful"] is not None and pjm_r0["harmful"] is not None
                   and pjm["harmful"] <= pjm_r0["harmful"] - 0.05) or \
                  (pjm["release"] is not None and pjm["release"] <= 0.02)

    # map drift aggregate. §10 semantics: D_map judges the stability of the
    # maps we would DEPLOY (i.e. domains selected C3). A domain the gate keeps
    # on C0 has no deployed C3 map, so its drift is diagnostic only, not a
    # reason to regularize the deployed calibrators. Report both views.
    msub_all = map_df[map_df["d_map_from_prev"].notna()]
    mean_drift = float(msub_all["d_map_from_prev"].mean()) if len(msub_all) else 0.0
    max_drift = float(msub_all["d_map_from_prev"].max()) if len(msub_all) else 0.0
    deployed = [dom for dom, s in sel_sel.items() if s == "C3"]
    if deployed:
        msub_dep = msub_all[msub_all["domain"].isin(deployed)]
        mean_drift_dep = float(msub_dep["d_map_from_prev"].mean()) if len(msub_dep) else 0.0
        max_drift_dep = float(msub_dep["d_map_from_prev"].max()) if len(msub_dep) else 0.0
    else:
        mean_drift_dep = max_drift_dep = 0.0
    map_stable = (mean_drift_dep <= MAP_DRIFT_MEAN_THRESHOLD
                  and max_drift_dep <= MAP_DRIFT_MAX_THRESHOLD)

    nem_ret = {}
    for dom in ("NEM_SA1:MLP", "NEM_SA1:Linear", "LAGO_DE:MLP"):
        a = r0[dom]["net"]; b = r11[dom]["net"]
        nem_ret[dom] = (b / a) if (a is not None and a != 0) else None

    checks = {
        "NEM stays C0 (insufficient evidence)": sel_sel.get("NEM_SA1:MLP") == "C0"
        and sel_sel.get("NEM_SA1:Linear") == "C0",
        "healthy domains not miscalibrated": sel_sel.get("LAGO_DE:MLP") == "C0"
        and sel_sel.get("LAGO_PJM:Linear") == "C0",
        "PJM:MLP C3 authorized by rolling OOS": bool(pjm_authorized),
        "LCB90 > 0 (not relaxed)": (pjm_authorized and pjm_lcb is not None
                                    and pjm_lcb > 0),
        "S4 PJM harmful rare-fire reduced": bool(pjm_reduced),
        "macro net R11 >= R0": (macro_r11 is not None and macro_r0 is not None
                                and macro_r11 >= macro_r0 - 1e-9),
        "worst-domain R11 >= R0 - 0.01": (worst_r11 is not None
                                          and worst_r0 is not None
                                          and worst_r11 >= worst_r0 - 0.01),
        "NEM retention near 1": all(
            (v is None) or (v >= 0.95) for v in nem_ret.values()),
        "map stability (no severe rolling drift)": bool(map_stable),
    }
    green = all(checks.values())

    if green:
        verdict = "PREQUENTIAL_CALIBRATION_ROUTING_SUPPORTED"
        vreason = ("Rolling-origin OOS resolved the sparse-evidence ambiguity: "
                   "PJM:MLP C3 authorized via LCB90>0 from prequential evidence, "
                   "NEM + healthy domains kept C0, S4 PJM harmful reduced, "
                   "maps stable.")
    elif pjm_authorized and not pjm_reduced:
        verdict = "LOCAL_ISOTONIC_UPPER_BOUND_NOT_ROBUST"
        vreason = ("PJM:MLP got C3 authorization but S4 development does not "
                   "show reduced harmful rare-fire -> withdraw R1A.9 "
                   "deployment interpretation.")
    elif not map_stable:
        verdict = "LOCAL_CALIBRATOR_VARIANCE_LIMITED"
        vreason = ("Deployed (C3-authorized) rolling maps drift severely: "
                   f"deployed mean D_map={_f(mean_drift_dep)} "
                   f"max={_f(max_drift_dep)} "
                   f"(global mean={_f(mean_drift)} max={_f(max_drift)}) -> "
                   "high-variance calibrator; THEN regularized/shrunk isotonic "
                   "is authorized.")
    elif not pjm_authorized:
        verdict = "RARE_EVENT_EVIDENCE_LIMITED"
        vreason = ("Maps stable but prequential OOS still does not push "
                   "LCB90>0 for PJM:MLP -> evidence stays sparse; do NOT "
                   "regularize; next = longer adaptation horizon / few-shot "
                   "protocol / sequential accumulation.")
    else:
        verdict = "RARE_EVENT_EVIDENCE_LIMITED"
        vreason = ("Mixed non-GREEN outcome without a clean RED; evidence "
                   "remains sparse.")

    L = []
    L.append("# R1A.11 PREQUENTIAL VERDICT — Prequential Cross-Fitted "
             "Calibration Authorization")
    L.append("")
    L.append(f"- date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- code commit: `{M._git_head()}`")
    L.append(f"- frozen artifacts: `{artifact_dir.name}` (variant `{args.variant}`)")
    L.append(f"- protocol: B={cfg['burn_in']} burn-in, H={cfg['block']} block, "
             f"bootstrap {cfg['n_resample']} resamples alpha={cfg['alpha']} "
             f"(unchanged LCB>0 gate)")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append(f"### **{verdict}**")
    L.append("")
    L.append(f"- reason: {vreason}")
    L.append("")
    L.append("## Prequential OOS diagnostics (§9)")
    for _, r in pd.DataFrame(oos_sum).iterrows():
        L.append(f"- {r['domain']}: S3M={r['total_s3m_days']}d "
                 f"blocks={r['n_blocks']} oos_days={r['n_oos_eval_days']} "
                 f"C0 fire={r['c0_fire_days']} C3 fire={r['c3_fire_days']} "
                 f"C0 net={_g(r['c0_net'])} C3 net={_g(r['c3_net'])} "
                 f"mean_delta={_g(r['mean_paired_delta'])} "
                 f"LCB90={_g(r['lcb90'])} frac>0={_f(r['frac_bootstrap_gt0'])} "
                 f"-> {r['selected']} ({r['reason']})")
    L.append("")
    L.append("## Selection (pre-S4, pre-S3C)")
    for _, r in sel_df.iterrows():
        L.append(f"- {r['domain']}: selected={r['selected']} reason="
                 f"{r['reason']} (A={r['gate_A_prequential']} "
                 f"B={r['gate_B_raw_problem']} C={r['gate_C_value_improve']} "
                 f"D={r['gate_D_bootstrap']})")
    L.append("")
    L.append("## Map stability (rolling fits, grid s in [-1,1])")
    L.append(f"- global: mean D_map = {_f(mean_drift)}, "
             f"max D_map = {_f(max_drift)} (all 6 domains)")
    L.append(f"- deployed (C3-authorized: {', '.join(deployed) if deployed else 'none'}): "
             f"mean D_map = {_f(mean_drift_dep)}, "
             f"max D_map = {_f(max_drift_dep)}")
    L.append(f"- stable iff deployed mean<={MAP_DRIFT_MEAN_THRESHOLD} and "
             f"deployed max<={MAP_DRIFT_MAX_THRESHOLD} -> "
             f"{'stable' if map_stable else 'UNSTABLE'} "
             f"(§10 semantics: D_map judges the maps we deploy; a domain the "
             f"gate keeps on C0 has no deployed C3 map, its drift is diagnostic "
             f"only)")
    L.append("")
    L.append("## S4 policy metrics (development confirmation only)")
    for _, r in s4_df[s4_df["version"] != "R0"].iterrows():
        L.append(f"- {r['domain']} {r['version']}: release={_f(r['release'])} "
                 f"harm={_f(r['harmful'])} gain|rel={_g(r['gain_rel'])} "
                 f"net={_g(r['net'])} q={_f(r['q'])}")
    L.append("")
    L.append("## GREEN checklist (§11)")
    for k, v in checks.items():
        L.append(f"- {'PASS' if v else 'FAIL'}  {k}")
    L.append(f"- macro net R0={_g(macro_r0)} R11={_g(macro_r11)}; "
             f"worst-domain R0={_g(worst_r0)} R11={_g(worst_r11)}")
    for dom, v in nem_ret.items():
        L.append(f"- NEM/DE retention {dom}: {_f(v) if v is not None else 'n/a'}")
    L.append("")
    L.append("## Notes")
    L.append("- R10 (R1A.10 fixed-prefix selector) selected C0 everywhere -> "
             "R10 == R0 on S4. R11 is the prequential selector proposed here.")
    L.append("- Chronology: each OOS day used a C3 fit from strictly-earlier "
             "S3M days only; S3C DVG-q and S4 are development confirmation.")
    L.append("- Bootstrap uses per-domain deterministic seed (hash of "
             "global_seed, domain, stage) -> order-independent reproducibility.")
    L.append("- Protocol constants B/H/alpha and the LCB>0 rule were "
             "pre-registered and are not changed by this result.")
    L.append("- R1B starts ONLY on GREEN (R1A.11 plan §13).")
    L.append("")
    verdict_text = "\n".join(L)
    with open(out_dir / "PREQUENTIAL_VERDICT.md", "w", encoding="utf-8") as f:
        f.write(verdict_text)
    print("\n==========================================================")
    print(verdict_text)
    print(f"\n[R1A.11] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
