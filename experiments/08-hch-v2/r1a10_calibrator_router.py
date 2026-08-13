"""R1A.10 Evidence-Gated Local Action Calibration (plan v0.1).

Question: can HCH decide, BEFORE seeing S4, whether a domain's local isotonic
action calibration (C3) is trustworthy enough to deploy — with C0 (Raw IAH)
as the default everywhere?

Protocol:
  E0 = C0 everywhere (raw baseline)
  E1 = C3 everywhere (R1A.9 upper bound / negative control)
  E2 = evidence-gated C0/C3  (proposed deployment protocol, this round)

Selection chronology (plan §3, §10): S3M-prefix fits C3 only; S3M-suffix
selects C0 vs C3 (decision-level, execute iff A_hat>0); S3C fits the SELECTED
estimator's own DVG q; S4 is development confirmation only. No S3C/S4 usage in
selection; no market/host hard-coding; no learned router.

Evidence Gate v1 (plan §5), all on S3M-suffix:
  A  n_fit_days >= 30 and n_val_days >= 10            (else keep C0)
  B  raw C0 shows a real problem: val_net_C0 <= 0
     OR val_harmful_C0 >= 0.50                        (else keep C0: healthy)
  C  val_net_C3 > val_net_C0 AND val_harmful_C3
     <= val_harmful_C0                                (else keep C0)
  D  moving-block bootstrap (7d block, 1000 resamples,
     one-sided 90% lower CI): LCB_0.90(E[Delta]) > 0
     (n_val_days < 14 => no strong claim => keep C0)

Reason codes:
  RAW_HEALTHY_KEEP_C0 / INSUFFICIENT_LOCAL_EVIDENCE_KEEP_C0 /
  C3_VALUE_NOT_BETTER_KEEP_C0 / C3_IMPROVEMENT_UNCERTAIN_KEEP_C0 /
  LOCAL_MISCALIBRATION_C3_AUTHORIZED

Verdict (plan §12): EVIDENCE_GATED_LOCAL_CALIBRATION_SUPPORTED (GREEN) /
CALIBRATOR_ROUTING_UNRESOLVED (YELLOW) / LOCAL_CALIBRATION_NOT_DEPLOYABLE (RED).
"""
from __future__ import annotations

import argparse
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

import r1a9_action_calibration as M  # reuses collection/eval/calibrators

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

SEED = 20260813
RNG = np.random.default_rng(SEED)

GATE_A_MIN_FIT_DAYS = 30
GATE_A_MIN_VAL_DAYS = 10
GATE_B_HARMFUL_THRESHOLD = 0.50
GATE_D_BLOCK = 7
GATE_D_N_RESAMPLE = 1000
GATE_D_ALPHA = 0.10
GATE_D_MIN_VAL_DAYS = 14

STRESS_W = (7, 14, 30, 60)


def _f(x):
    return f"{x:.3f}" if x is not None and pd.notna(x) else "NaN"


def _g(x):
    return f"{x:+.4f}" if x is not None and pd.notna(x) else "NaN"


def _r(lo, hi):
    if lo is None or hi is None:
        return None
    return f"[{lo:.3f},{hi:.3f}]"


def _iso_stats(iso):
    """R1A.9-style isotonic-map audit (modern sklearn threshold attrs)."""
    if iso is None:
        return {"plateau_count": None, "map_range_lo": None,
                "map_range_hi": None, "max_jump": None, "n_thresholds": None}
    xs = np.asarray(iso.X_thresholds_, dtype=np.float64)
    ys = np.asarray(iso.y_thresholds_, dtype=np.float64)
    uniq = np.unique(ys)
    jumps = np.abs(np.diff(ys))
    return {"plateau_count": int(len(uniq)),
            "map_range_lo": float(uniq.min()),
            "map_range_hi": float(uniq.max()),
            "max_jump": float(jumps.max()) if len(jumps) else 0.0,
            "n_thresholds": int(len(xs))}


def moving_block_bootstrap(deltas, block=GATE_D_BLOCK,
                           n_resample=GATE_D_N_RESAMPLE, alpha=GATE_D_ALPHA):
    """Moving-block bootstrap over paired per-day deltas.

    Returns (mean_delta, one-sided lower CI bound, frac_resamples_positive).
    """
    d = np.asarray(deltas, dtype=np.float64)
    n = len(d)
    if n < 2:
        m = float(d[0]) if n else 0.0
        return m, m, float(d[0] > 0) if n else 0.0
    n_blocks = int(np.ceil(n / block))
    max_start = max(1, n - block + 1)
    means = np.empty(n_resample)
    for i in range(n_resample):
        starts = RNG.integers(0, max_start, size=n_blocks)
        sample = np.concatenate([d[s:s + block] for s in starts])[:n]
        means[i] = sample.mean()
    lcb = float(np.percentile(means, 100.0 * alpha))
    return float(means.mean()), lcb, float((means > 0).mean())


# ---------------------------------------------------------------- selector ----
class LocalCalibrationSelector:
    """Transparent deterministic pre-S4 C0-vs-C3 gate (plan §5)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def collect_support_stats(self, dd, c3, c0_rows):
        """Calibration support audit (plan §7)."""
        train_days = [d for d in dd["days"] if d["block"] == "train"]
        val_days = [d for d in dd["days"] if d["block"] == "val"]
        n_fit_days = len(train_days)
        n_val_days = len(val_days)
        t_d = M.hour_table(dd["hour_split"], "train", "d")
        t_u = M.hour_table(dd["hour_split"], "train", "u")
        n_fit_hours_d = int(len(t_d["Yd"])) if n_fit_days else 0
        n_fit_hours_u = int(len(t_u["Yu"])) if n_fit_days else 0

        down_cross = up_cross = 0
        for day in val_days:
            vm = day["vm"]
            if (day["wm"][vm] > 0.5).any():
                down_cross += 1
            if (day["wp"][vm] > 0.5).any():
                up_cross += 1

        def _fire_benefit(rows):
            fire = [r for r in rows if r["A_hat"] > 0]
            if not fire:
                return None, 0, 0
            pos = sum(1 for r in fire if r["A_true"] > 0)
            neg = sum(1 for r in fire if r["A_true"] < 0)
            return pos / len(fire), pos, neg

        fb_train, fb_pos_t, fb_neg_t = _fire_benefit(
            [r for r in c0_rows if r["block"] == "train"])
        fb_val, fb_pos_v, fb_neg_v = _fire_benefit(
            [r for r in c0_rows if r["block"] == "val"])

        c0_fire_days = [r for r in c0_rows if r["block"] == "val"
                        and r["A_hat"] > 0]
        iso_d = _iso_stats(c3.params.get(dd["domain"], {}).get("d"))
        iso_u = _iso_stats(c3.params.get(dd["domain"], {}).get("u"))
        return {
            "domain": dd["domain"],
            "n_fit_days": n_fit_days, "n_val_days": n_val_days,
            "n_fit_hours_d": n_fit_hours_d, "n_fit_hours_u": n_fit_hours_u,
            "raw_c0_fire_days": len(c0_fire_days),
            "raw_c0_fire_hours": int(sum(
                (r["pi"][r["vm"]] != 0).sum() for r in c0_fire_days)),
            "down_w_cross_days": down_cross, "up_w_cross_days": up_cross,
            "train_fire_benefit_rate": fb_train,
            "val_fire_benefit_rate": fb_val,
            "benefit_drift_prefix_to_suffix":
                (fb_val - fb_train) if (fb_train is not None
                                        and fb_val is not None) else None,
            "fire_pos_neg_val": f"{fb_pos_v}/{fb_neg_v}",
            "iso_d_plateau": iso_d["plateau_count"],
            "iso_d_range": _r(iso_d["map_range_lo"], iso_d["map_range_hi"]),
            "iso_d_max_jump": iso_d["max_jump"],
            "iso_u_plateau": iso_u["plateau_count"],
            "iso_u_range": _r(iso_u["map_range_lo"], iso_u["map_range_hi"]),
            "iso_u_max_jump": iso_u["max_jump"],
        }

    def evaluate_raw_health(self, c0_rows):
        """Gate B inputs: S3M-suffix C0 summary."""
        return _val_summary([r for r in c0_rows if r["block"] == "val"])

    def paired_value_bootstrap(self, c0_rows, c3_rows):
        """Gate D: moving-block bootstrap of Delta = V^C3 - V^C0 (paired)."""
        v0 = {r["date"]: r for r in c0_rows if r["block"] == "val"}
        v3 = {r["date"]: r for r in c3_rows if r["block"] == "val"}
        dates = sorted(set(v0) & set(v3))
        deltas = np.array([
            v3[d]["A_true"] * (v3[d]["A_hat"] > 0)
            - v0[d]["A_true"] * (v0[d]["A_hat"] > 0)
            for d in dates], dtype=np.float64)
        return dates, deltas

    def select(self, dd, c0_rows, c3_rows, c3):
        audit = self.collect_support_stats(dd, c3, c0_rows)
        h = self.evaluate_raw_health(c0_rows)
        gates = {"A_support": None, "B_raw_problem": None,
                 "C_value_improve": None, "D_bootstrap": None}
        bootstrap = None

        # Gate A
        ok_a = (audit["n_fit_days"] >= self.cfg["min_fit_days"]
                and audit["n_val_days"] >= self.cfg["min_val_days"])
        gates["A_support"] = bool(ok_a)
        if not ok_a:
            return "C0", "INSUFFICIENT_LOCAL_EVIDENCE_KEEP_C0", audit, gates, \
                bootstrap

        # Gate B
        ok_b = (h["net"] is not None
                and (h["net"] <= 0
                     or h["harmful"] >= self.cfg["harm_threshold"]))
        gates["B_raw_problem"] = bool(ok_b)
        if not ok_b:
            return "C0", "RAW_HEALTHY_KEEP_C0", audit, gates, bootstrap

        # Gate C
        s3 = _val_summary([r for r in c3_rows if r["block"] == "val"])
        ok_c = (s3["net"] is not None and s3["net"] > h["net"]
                and s3["harmful"] <= h["harmful"])
        gates["C_value_improve"] = bool(ok_c)
        if not ok_c:
            return "C0", "C3_VALUE_NOT_BETTER_KEEP_C0", audit, gates, bootstrap

        # Gate D
        if audit["n_val_days"] < self.cfg["min_val_bootstrap_days"]:
            return "C0", "C3_IMPROVEMENT_UNCERTAIN_KEEP_C0", audit, gates, \
                bootstrap
        dates, deltas = self.paired_value_bootstrap(c0_rows, c3_rows)
        mean_d, lcb, frac_pos = moving_block_bootstrap(
            deltas, self.cfg["block"], self.cfg["n_resample"],
            self.cfg["alpha"])
        bootstrap = {"n_val_days": audit["n_val_days"], "mean_delta": mean_d,
                     "lcb": lcb, "frac_resamples_positive": frac_pos,
                     "n_days_used": len(dates)}
        gates["D_bootstrap"] = bool(lcb > 0)
        if lcb > 0:
            return "C3", "LOCAL_MISCALIBRATION_C3_AUTHORIZED", audit, gates, \
                bootstrap
        return "C0", "C3_IMPROVEMENT_UNCERTAIN_KEEP_C0", audit, gates, bootstrap


def _val_summary(rows):
    """Decision-level S3M-suffix summary: execute iff A_hat>0 (no gate)."""
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


# ------------------------------------------------------------ stress test -----
def support_size_stress(dd, W_list=STRESS_W):
    """Diagnostic only (§8): fit C3 on first W S3M days, eval on fixed suffix."""
    train_days = sorted([d for d in dd["days"] if d["block"] == "train"],
                        key=lambda d: d["date"])
    val_days = [d for d in dd["days"] if d["block"] == "val"]
    if not val_days:
        return []
    out = []
    for W in W_list:
        fit_days = train_days[:W]
        if not fit_days:
            continue
        iso_params = {}
        for direction, sk, gk, mk in (("d", "sd", "gd", "mm"),
                                      ("u", "su", "gu", "mp")):
            ss, yy = [], []
            for day in fit_days:
                m = day[mk] > 0
                ss.append(day[sk][m])
                yy.append(day[gk][m] / np.maximum(day[mk][m], 1e-12))
            s = np.concatenate(ss) if ss else np.array([])
            y = np.concatenate(yy) if yy else np.array([])
            m2 = np.isfinite(s) & np.isfinite(y)
            s, y = s[m2], y[m2]
            if len(s) < 8:
                iso_params[direction] = None
                continue
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(s, y)
            iso_params[direction] = iso
        st_d = _iso_stats(iso_params["d"])
        st_u = _iso_stats(iso_params["u"])

        class _StressCal(M.Calibrator):
            name = "C3_stress"

            def apply(self, s, direction, domain=None):
                iso = iso_params[direction]
                s = np.asarray(s, dtype=float)
                if iso is None:
                    return np.clip(s, -1.0, 1.0)
                return np.clip(iso.predict(s), -1.0, 1.0)

        rows = M.evaluate_days(_StressCal(), dd)
        s = _val_summary([r for r in rows if r["block"] == "val"])
        out.append({
            "domain": dd["domain"], "W_days": W,
            "n_fit_days_used": len(fit_days),
            "iso_d_plateau": st_d["plateau_count"],
            "iso_d_range": _r(st_d["map_range_lo"], st_d["map_range_hi"]),
            "iso_d_max_jump": st_d["max_jump"],
            "iso_u_plateau": st_u["plateau_count"],
            "iso_u_range": _r(st_u["map_range_lo"], st_u["map_range_hi"]),
            "iso_u_max_jump": st_u["max_jump"],
            "val_release": s["release"], "val_harmful": s["harmful"],
            "val_gain_rel": s["gain_rel"], "val_net": s["net"],
            "val_n_fire": s["n_fire"],
        })
    return out


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--variant", type=str, default="learned_sig")
    ap.add_argument("--no-stress", action="store_true")
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
        results_dir / f"R1A10_ROUTER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig = out_dir / "figures"
    out_fig.mkdir(exist_ok=True)
    print(f"[R1A.10] frozen artifacts: {artifact_dir}")
    print(f"[R1A.10] out: {out_dir}", flush=True)

    cfg = {
        "min_fit_days": GATE_A_MIN_FIT_DAYS,
        "min_val_days": GATE_A_MIN_VAL_DAYS,
        "harm_threshold": GATE_B_HARMFUL_THRESHOLD,
        "min_val_bootstrap_days": GATE_D_MIN_VAL_DAYS,
        "block": GATE_D_BLOCK, "n_resample": GATE_D_N_RESAMPLE,
        "alpha": GATE_D_ALPHA, "seed": args.seed,
    }
    with open(out_dir / "selector_config.json", "w", encoding="utf-8") as f:
        json.dump({"variant": args.variant, **cfg,
                   "code_commit": M._git_head()}, f, indent=2)

    domain_data = {}
    for ds_key, bb in M.R.DOMAINS:
        dom = f"{ds_key}:{bb}"
        print(f"[R1A.10] collect {dom} ...", flush=True)
        dd = M.collect_domain(artifact_dir, ds_key, bb, args.variant)
        domain_data[dom] = dd
        for pr in dd["problems"]:
            print(f"      WARN {pr}", flush=True)

    # ---- fit C0 + C3 (C3 on S3M-prefix hours only) ----
    train_slices = []
    for dom, dd in domain_data.items():
        t_d = M.hour_table(dd["hour_split"], "train", "d")
        t_u = M.hour_table(dd["hour_split"], "train", "u")
        train_slices.append({"domain": dom, "sd": t_d["sd"], "Yd": t_d["Yd"],
                             "su": t_u["su"], "Yu": t_u["Yu"]})
    c0 = M.RawIAH()
    c3 = M.LocalIsotonic()
    c3.fit(train_slices)

    # ---- evaluate both on all days ----
    print("[R1A.10] evaluate C0/C3 across all blocks ...", flush=True)
    rows_c0 = {dom: M.evaluate_days(c0, dd) for dom, dd in domain_data.items()}
    rows_c3 = {dom: M.evaluate_days(c3, dd) for dom, dd in domain_data.items()}

    # ---- selection (pre-S4, pre-S3C) ----
    selector = LocalCalibrationSelector(cfg)
    support_rows, val_cmp_rows, boot_rows, sel_rows = [], [], [], []
    for dom, dd in domain_data.items():
        sel, reason, audit, gates, boot = selector.select(
            dd, rows_c0[dom], rows_c3[dom], c3)
        support_rows.append(audit)
        s0 = _val_summary([r for r in rows_c0[dom] if r["block"] == "val"])
        s3 = _val_summary([r for r in rows_c3[dom] if r["block"] == "val"])
        val_cmp_rows.append({"domain": dom,
                             "c0_net": s0["net"], "c0_harmful": s0["harmful"],
                             "c0_gain_rel": s0["gain_rel"],
                             "c0_release": s0["release"],
                             "c0_n_fire_days": s0["n_fire"],
                             "c3_net": s3["net"], "c3_harmful": s3["harmful"],
                             "c3_gain_rel": s3["gain_rel"],
                             "c3_release": s3["release"],
                             "c3_n_fire_days": s3["n_fire"]})
        sel_rows.append({"domain": dom, "selected": sel, "reason": reason,
                         **{f"gate_{k}": v for k, v in gates.items()}})
        boot_rows.append({"domain": dom,
                          **({} if boot is None else boot),
                          "reason": reason})
        print(f"      {dom}: -> {sel} ({reason})", flush=True)
    sel_df = pd.DataFrame(sel_rows)
    sel_map = dict(zip(sel_df["domain"], sel_df["selected"]))

    # ---- S3C DVG per estimator + S4 metrics for E0/E1/E2 ----
    s4_pol, s4_pt = [], []
    for dom, dd in domain_data.items():
        for ver, cal, rows in (("E0", c0, rows_c0[dom]),
                               ("E1", c3, rows_c3[dom])):
            dv = M.dvg_and_s4(rows, M.R.ALPHA)
            pm = M.point_metrics(dv, None)
            s4_pol.append({"domain": dom, "version": ver,
                           "calibrator": cal.name,
                           "selected": sel_map.get(dom),
                           "q": dv["q"], "n_calib": dv["n_calib"],
                           "n_eval": dv["n_eval"],
                           "release": dv["release_rate"],
                           "identity": dv["identity_rate"],
                           "harmful": dv["harmful_rate"],
                           "gain_rel": dv["mean_gain_release"],
                           "net": dv["net_value"], "coverage": dv["coverage"]})
            s4_pt.append({"domain": dom, "version": ver,
                          "calibrator": cal.name, **pm})
        cal = c3 if sel_map.get(dom) == "C3" else c0
        rows = rows_c3[dom] if cal is c3 else rows_c0[dom]
        dv = M.dvg_and_s4(rows, M.R.ALPHA)
        pm = M.point_metrics(dv, None)
        s4_pol.append({"domain": dom, "version": "E2",
                       "calibrator": cal.name,
                       "selected": sel_map.get(dom),
                       "q": dv["q"], "n_calib": dv["n_calib"],
                       "n_eval": dv["n_eval"],
                       "release": dv["release_rate"],
                       "identity": dv["identity_rate"],
                       "harmful": dv["harmful_rate"],
                       "gain_rel": dv["mean_gain_release"],
                       "net": dv["net_value"], "coverage": dv["coverage"]})
        s4_pt.append({"domain": dom, "version": "E2",
                      "calibrator": cal.name, **pm})

    s4_df = pd.DataFrame(s4_pol)
    pt_df = pd.DataFrame(s4_pt)

    # ---- support-size stress (LAGO only, diagnostic) ----
    stress_rows = []
    if not args.no_stress:
        for dom in domain_data:
            if dom.startswith("LAGO_"):
                print(f"[R1A.10] stress {dom} ...", flush=True)
                stress_rows.extend(support_size_stress(domain_data[dom]))
    stress_df = pd.DataFrame(stress_rows)

    # ---- figures ----
    def fig_gate():
        df = sel_df.copy()
        fig, ax = plt.subplots(figsize=(9, 4))
        y = np.arange(len(df))
        colors = ["#c0392b" if r == "C0" else "#27ae60"
                  for r in df["selected"]]
        ax.barh(y, [1] * len(df), color=colors, alpha=0.85)
        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(0.02, i, f"{row['domain']}: {row['reason']}",
                    va="center", fontsize=7)
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_title("E2 selected calibrator per domain (pre-S4 evidence)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out_fig / "gate_decisions.png", dpi=130)
        plt.close(fig)

    def fig_net():
        piv = s4_df.pivot_table(index="domain", columns="version",
                                values="net", aggfunc="first")
        piv = piv[["E0", "E1", "E2"]]
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(piv)); w = 0.26
        for j, col in enumerate(piv.columns):
            ax.bar(x + (j - 1) * w, piv[col], w, label=col)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=30,
                                             ha="right", fontsize=7)
        ax.set_ylabel("S4 net daily action value")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_fig / "s4_net_by_version.png", dpi=130)
        plt.close(fig)

    def fig_stress():
        if stress_df.empty:
            return
        for dom in stress_df["domain"].unique():
            sub = stress_df[stress_df["domain"] == dom]
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.plot(sub["W_days"], sub["val_net"], "o-", label="val net")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xlabel("W fit days")
            ax.set_title(f"support-size stress: {dom}", fontsize=9)
            ax.tick_params(labelsize=7)
            fig.tight_layout()
            fig.savefig(out_fig / f"stress_{dom.replace(':', '_')}.png",
                        dpi=130)
            plt.close(fig)

    fig_gate(); fig_net(); fig_stress()

    # ---- CSVs ----
    pd.DataFrame(support_rows).to_csv(out_dir / "support_by_domain.csv",
                                      index=False)
    pd.DataFrame(val_cmp_rows).to_csv(out_dir / "val_policy_comparison.csv",
                                      index=False)
    pd.DataFrame(boot_rows).to_csv(out_dir / "paired_bootstrap.csv",
                                   index=False)
    sel_df.to_csv(out_dir / "selected_calibrator_by_domain.csv", index=False)
    dvg_rows = []
    for dom in domain_data:
        for ver, rows in (("E0", rows_c0[dom]), ("E1", rows_c3[dom])):
            dv = M.dvg_and_s4(rows, M.R.ALPHA)
            dvg_rows.append({"domain": dom, "version": ver,
                             "q": dv["q"], "n_calib": dv["n_calib"]})
    pd.DataFrame(dvg_rows).to_csv(out_dir / "s3c_dvg.csv", index=False)
    s4_df.to_csv(out_dir / "s4_policy_metrics.csv", index=False)
    pt_df.to_csv(out_dir / "s4_point_metrics.csv", index=False)
    stress_df.to_csv(out_dir / "support_size_stress.csv", index=False)
    with open(out_dir / "code_commit.txt", "w", encoding="utf-8") as f:
        f.write(f"{M._git_head()}\n")
        f.write(f"run: {datetime.now().isoformat(timespec='seconds')}\n")

    # ------------------------------------------------------------ verdict ----
    e0 = {r["domain"]: r for _, r in
          s4_df[s4_df["version"] == "E0"].iterrows()}
    e2 = {r["domain"]: r for _, r in
          s4_df[s4_df["version"] == "E2"].iterrows()}

    def macro(d):
        vals = [d[dom]["net"] for dom in d if d[dom]["net"] is not None]
        return float(np.mean(vals)) if vals else None

    def worst(d):
        vals = [d[dom]["net"] for dom in d if d[dom]["net"] is not None]
        return float(np.min(vals)) if vals else None

    macro_e0, macro_e2 = macro(e0), macro(e2)
    worst_e0, worst_e2 = worst(e0), worst(e2)
    nem_ret = {}
    for dom in ("NEM_SA1:MLP", "NEM_SA1:Linear", "LAGO_DE:MLP"):
        n0 = e0[dom]["net"]; n2 = e2[dom]["net"]
        nem_ret[dom] = (n2 / n0) if (n0 is not None and n0 != 0) else None

    sel_sel = dict(zip(sel_df["domain"], sel_df["selected"]))
    n_c3 = int((sel_df["selected"] == "C3").sum())
    nem_c3_wrong = any(sel_sel.get(d) == "C3" for d in
                       ("NEM_SA1:MLP", "NEM_SA1:Linear"))
    pjm_row = e2["LAGO_PJM:MLP"]
    pjm_e0 = e0["LAGO_PJM:MLP"]
    pjm_handled = (pjm_row["harmful"] is not None
                   and pjm_e0["harmful"] is not None
                   and pjm_row["harmful"] <= pjm_e0["harmful"] - 0.05) or \
                  (pjm_row["release"] is not None
                   and pjm_row["release"] <= 0.02)

    checks = {
        "no_NEM_C3 (gate keeps NEM on C0)": not nem_c3_wrong,
        "at least one C3 authorized": n_c3 >= 1,
        "macro net E2 >= E0": (macro_e2 is not None and macro_e0 is not None
                               and macro_e2 >= macro_e0 - 1e-9),
        "worst-domain E2 >= E0 - 0.01": (worst_e2 is not None
                                         and worst_e0 is not None
                                         and worst_e2 >= worst_e0 - 0.01),
        "NEM retention >= 0.5": all(
            (v is None) or (v >= 0.5) for v in nem_ret.values()),
        "PJM:MLP harmful reduced or abstained": bool(pjm_handled),
    }
    green = all(checks.values())

    reached_gate_d = [d for d in boot_rows if d.get("lcb") is not None]
    yellow = (not green) and (not nem_c3_wrong) and (
        n_c3 == 0 and len(reached_gate_d) > 0
        and all(b["reason"] == "C3_IMPROVEMENT_UNCERTAIN_KEEP_C0"
                for b in reached_gate_d))
    red = nem_c3_wrong

    if green:
        verdict = "EVIDENCE_GATED_LOCAL_CALIBRATION_SUPPORTED"
        vreason = ("E2 gate authorized C3 only where pre-S4 evidence supported "
                   "it, kept NEM on C0, and preserved macro/worst-domain value "
                   "and PJM:MLP safety.")
    elif red:
        verdict = "LOCAL_CALIBRATION_NOT_DEPLOYABLE"
        vreason = ("The gate deployed C3 to an evidence-insufficient domain "
                   "(NEM) -> pre-S4 evidence cannot distinguish.")
    elif yellow:
        verdict = "CALIBRATOR_ROUTING_UNRESOLVED"
        vreason = ("Router direction is right (NEM/healthy domains protected) "
                   "but the bootstrap CI never authorizes C3 -> support/CI "
                   "still unstable; next step = regularized/shrunk isotonic.")
    else:
        verdict = "CALIBRATOR_ROUTING_UNRESOLVED"
        vreason = ("Router produced a mixed/non-GREEN outcome without a clean "
                   "RED; selection stability needs strengthening.")

    L = []
    L.append("# R1A.10 ROUTER VERDICT — Evidence-Gated Local Action Calibration")
    L.append("")
    L.append(f"- date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- code commit: `{M._git_head()}`")
    L.append(f"- frozen artifacts: `{artifact_dir.name}` (variant `{args.variant}`)")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append(f"### **{verdict}**")
    L.append("")
    L.append(f"- reason: {vreason}")
    L.append("")
    L.append("## Selection (pre-S4, pre-S3C)")
    for _, r in sel_df.iterrows():
        L.append(f"- {r['domain']}: selected={r['selected']} reason="
                 f"{r['reason']} (A={r['gate_A_support']} "
                 f"B={r['gate_B_raw_problem']} C={r['gate_C_value_improve']} "
                 f"D={r['gate_D_bootstrap']})")
    L.append("")
    L.append("## S3M-suffix paired value (V^C3 - V^C0)")
    for b in boot_rows:
        L.append(f"- {b['domain']}: mean_delta="
                 f"{_g(b.get('mean_delta'))} lcb="
                 f"{_g(b.get('lcb'))} frac_pos="
                 f"{_f(b.get('frac_resamples_positive'))} "
                 f"reason={b['reason']}")
    L.append("")
    L.append("## S4 policy metrics (development confirmation only)")
    for _, r in s4_df[s4_df["version"] != "E1"].iterrows():
        L.append(f"- {r['domain']} {r['version']}: release={_f(r['release'])} "
                 f"harm={_f(r['harmful'])} gain|rel={_g(r['gain_rel'])} "
                 f"net={_g(r['net'])} q={_f(r['q'])}")
    L.append("")
    L.append("## GREEN checklist (§12)")
    for k, v in checks.items():
        L.append(f"- {'PASS' if v else 'FAIL'}  {k}")
    L.append(f"- macro net E0={_g(macro_e0)} E2={_g(macro_e2)}; "
             f"worst-domain E0={_g(worst_e0)} E2={_g(worst_e2)}")
    for dom, v in nem_ret.items():
        L.append(f"- NEM/DE retention {dom}: {_f(v) if v is not None else 'n/a'}")
    L.append("")
    L.append("## Notes")
    L.append("- Chronology strictly enforced: S3M-prefix fit, S3M-suffix "
             "selection, S3C DVG-q for the SELECTED estimator only, S4 "
             "confirmation. No S3C/S4 feedback into the gate.")
    L.append("- Gate is deterministic + transparent (no learned router, no "
             "market/host hard-coding).")
    L.append("- default-to-C0: calibration is evidence-authorized safety "
             "correction, not a default module.")
    L.append("- R1B starts ONLY on GREEN (R1A.10 plan §16).")
    L.append("")
    verdict_text = "\n".join(L)
    with open(out_dir / "ROUTER_VERDICT.md", "w", encoding="utf-8") as f:
        f.write(verdict_text)
    print("\n==========================================================")
    print(verdict_text)
    print(f"\n[R1A.10] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
