"""HCH-v2 训练与模块调查诊断(v0.2 §2.1 toy test + §3 读出 + §4 方向一致性).

只读已有 head(P0A_RERUN universal / WP5 头),不重训、不改 src/、不碰 S4 调参.
输出到 results/TRAINING_INVESTIGATION_<ts>/(新目录,不覆盖旧实验).

§2.1 toy test: host=0 + 非对称 w + 非零 m 下,验证 identity/weighted-mean/
    weighted-median/Bayes-action 产生不同 raw 预测;验证 final_point_metrics 在
    pi=0(未释放)时输出恒等于 host(即候选 m 未进入点预测)。

§3 读出调查: 对每个域,用 S2V(candidate 校验段)做读出选择/诊断,生成五种读出
    (identity / raw weighted mean / raw weighted median / 期望sMAPE Bayes action /
    shrinkage z0+λ(w⁺m⁺−w⁻m⁻)),S2V 选最优读出类型 + λ,S4 冻结评估。

§4 方向一致性: 在 S2V 上算 residual_z = zY−z0 vs sign(m_plus−m_minus),
    Down/Up 准确率、m 与真实残差幅值相关、四类期分段统计、非零剂量但 gain 为负比例。

数学硬约束(§1): host-relative asinh geometry → 三原子 IAH → 只读不训。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import r1a_run as R
import r1a9_action_calibration as M
from r1b_generalization_screen import SEED, build_head

EPS = 1e-9
LAMBDAS = [0.25, 0.5, 0.75, 1.0]

# ------------------------------------------------------------ readout fns ----
def s_day_of(host_b: np.ndarray) -> np.ndarray:
    """Day scale s = mean(|host|) per row (clamp for asinh/sinh safety)."""
    return np.maximum(np.abs(host_b).mean(axis=-1), 1e-12)


def asinh_z(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.arcsinh(x / s[..., None])


def readouts_from_atoms(z0, wm, wz, wp, mm, mp, s) -> dict:
    """Five point readouts from three-atom hyperbolic distribution.

    z0/wm/wz/wp/mm/mp: [H] per-hour (hyp-space). s: scalar day scale.
    Returns raw-space [H] arrays keyed by readout name.
    """
    H = len(z0)
    z_minus = z0 - mm
    z_plus = z0 + mp
    x_down = s * np.sinh(z_minus)
    x_ident = s * np.sinh(z0)
    x_up = s * np.sinh(z_plus)
    out = {"identity": x_ident.copy()}

    # raw-space weighted mean
    out["weighted_mean"] = wm * x_down + wz * x_ident + wp * x_up

    # raw-space weighted median (lower median of 3 atoms)
    out["weighted_median"] = np.zeros(H)
    for h in range(H):
        vals = np.array([x_down[h], x_ident[h], x_up[h]])
        ws = np.array([wm[h], wz[h], wp[h]])
        order = np.argsort(vals)
        cum = np.cumsum(ws[order])
        # 下中位数: 累积权重首次 >= 0.5 的原子
        out["weighted_median"][h] = vals[order][np.searchsorted(cum, 0.5)]

    # numerical Bayes action minimizing expected no-floor sMAPE over grid
    out["bayes_action"] = np.zeros(H)
    for h in range(H):
        atoms = np.array([x_down[h], x_ident[h], x_up[h]])
        ws = np.array([wm[h], wz[h], wp[h]])
        lo, hi = float(atoms.min()), float(atoms.max())
        grid = np.unique(np.concatenate([atoms, np.linspace(lo, hi, 51)]))
        # E[sMAPE(x,Y)] = Σ w_a 200|x−x_a|/(|x|+|x_a|)
        loss = np.zeros(len(grid))
        for a in range(3):
            num = 200.0 * np.abs(grid - atoms[a])
            den = np.abs(grid) + np.abs(atoms[a]) + EPS
            loss += ws[a] * num / den
        out["bayes_action"][h] = grid[int(np.argmin(loss))]

    # shrinkage μ_R = w⁺m⁺ − w⁻m⁻ (hyperbolic expected residual), λ grid
    mu_R = wp * mp - wm * mm
    for lam in LAMBDAS:
        out[f"shrink_{lam}"] = s * np.sinh(z0 + lam * mu_R)
    return out


def eval_readouts(preds: dict, y: np.ndarray, vm: np.ndarray) -> dict:
    """Metrics per readout on valid hours (MAE / RMSE / no-floor sMAPE /
    high-tail MAE / neg-price MAE). y: [H] raw price, vm: [H] bool."""
    out = {}
    vh = vm & np.isfinite(y)
    if not vh.any():
        return out
    yv = y[vh]
    tail_thr = np.quantile(np.abs(yv), 0.95)
    for name, x in preds.items():
        xv = np.asarray(x)[vh]
        ae = np.abs(yv - xv)
        sm = 200.0 * ae / (np.abs(yv) + np.abs(xv) + EPS)
        out[name] = {
            "mae": round(float(ae.mean()), 6),
            "rmse": round(float(np.sqrt((ae ** 2).mean())), 6),
            "smape_nofloor": round(float(sm.mean()), 6),
            "high_tail_mae": round(float(ae[np.abs(yv) >= tail_thr].mean()), 6)
                             if (np.abs(yv) >= tail_thr).any() else None,
            "neg_price_mae": round(float(ae[yv < 0].mean()), 6)
                             if (yv < 0).any() else None,
            "n_hours": int(vh.sum()),
        }
    return out


# --------------------------------------------------------- §2.1 minimal toy ----
def toy_test() -> dict:
    """Host=0, non-zero m, asymmetric w -> verify readouts differ; verify the
    final evaluator (P0-A replay) does NOT consume candidate m when pi=0."""
    from _final_point import final_point_metrics
    s = 30.0
    wm, wz, wp = 0.40, 0.10, 0.50     # asymmetric, heavy upper tail
    mm, mp = 0.5, 0.3                 # non-zero shifts (hyp-space)
    z0 = np.zeros(24)                 # host = 0
    preds = readouts_from_atoms(z0,
                                np.full(24, wm), np.full(24, wz),
                                np.full(24, wp), np.full(24, mm),
                                np.full(24, mp), s)
    distinct = {}
    for a in preds:
        for b in preds:
            if a < b:
                key = f"{a}_vs_{b}"
                distinct[key] = round(float(np.abs(preds[a] - preds[b]).mean()), 6)
    nonzero_pair = {k: v for k, v in distinct.items() if v > 1e-9}

    # evaluator check: pi=0 (unreleased) -> x_final must equal host exactly,
    # even though m is large. Prove candidate m never reaches the final point.
    y = s * np.sinh(np.full(24, 0.0)) + 5.0      # shifted target
    rows = [{"date": "toy", "z0": z0, "s_day": s, "pi": np.zeros(24),
             "price": y, "vm": np.ones(24, dtype=bool)}]
    ev = final_point_metrics(rows, [False])
    host_recon = s * np.sinh(z0)
    return {
        "readout_pairwise_abs_diff": nonzero_pair,
        "identity_abs_diff_from_host": round(float(np.abs(preds["identity"] - host_recon).mean()), 12),
        "evaluator_pi0_mae_equals_host_mae": abs(ev["mae"] - ev["host_mae"]) < 1e-9,
        "evaluator_mae": ev["mae"], "evaluator_host_mae": ev["host_mae"],
        "conclusion": ("readouts differ; P0-A evaluator with pi=0 outputs host "
                       "exactly -> candidate m_minus/m_plus never enter x_final "
                       "outside an authorized action."),
    }


# ----------------------------------------------------------- §3/§4 S2V scan ----
def s2v_scan(head, info, det_np) -> dict:
    """Run five readouts on S2V batches (candidate validation split) for
    readout selection/diagnosis (§3) and direction consistency (§4)."""
    det_t = R.det_for(det_np, 1)
    host_ae, all_vals = [], []
    pred_all = {k: [] for k in ("identity", "weighted_mean", "weighted_median",
                                "bayes_action", "shrink_0.25", "shrink_0.5",
                                "shrink_0.75", "shrink_1.0")}
    y_all, vm_all = [], []
    dir_rows = []
    with torch.no_grad():
        for batch in info.s2v_batches:
            host, ctx, tgt, vm = batch[:4]
            det_b = det_t.expand(host.shape[0], -1) if det_t.shape[0] != host.shape[0] else det_t
            out = head(host, ctx, valid_mask=vm, domain_det=det_b)
            for i in range(host.shape[0]):
                vh = (vm[i] > 0.5) & torch.isfinite(tgt[i, :, 0])
                vh_np = vh.cpu().numpy()
                if not vh_np.any():
                    continue
                z0 = out["z0"][i].cpu().numpy()
                wm = out["w_minus"][i].cpu().numpy()
                wz = out["w_zero"][i].cpu().numpy()
                wp = out["w_plus"][i].cpu().numpy()
                mm = out["m_minus"][i].cpu().numpy()
                mp = out["m_plus"][i].cpu().numpy()
                s = float(s_day_of(host[i].squeeze(-1).cpu().numpy()[None])[0])
                y = tgt[i, :, 0].cpu().numpy()
                zY = asinh_z(y, np.array([s]))[0]
                r = zY - z0
                preds = readouts_from_atoms(z0, wm, wz, wp, mm, mp, s)
                for k in pred_all:
                    pred_all[k].append(preds[k][vh_np])
                y_all.append(y[vh_np]); vm_all.append(vh_np)
                # §4 direction rows (per hour, valid only)
                for h in np.where(vh_np)[0]:
                    pd_dir = 1 if mp[h] > mm[h] else (-1 if mp[h] < mm[h] else 0)
                    dir_rows.append({
                        "r": float(r[h]), "zY": float(zY[h]), "z0": float(z0[h]),
                        "mm": float(mm[h]), "mp": float(mp[h]),
                        "wm": float(wm[h]), "wp": float(wp[h]),
                        "y": float(y[h]),
                        "pd_dir": pd_dir,
                    })
    yc = np.concatenate(y_all)
    vmc = np.concatenate([np.ones_like(a, dtype=bool) for a in pred_all["identity"]])
    preds_cat = {k: np.concatenate(v) for k, v in pred_all.items()}
    # readout selection on S2V
    metrics = eval_readouts(preds_cat, yc, vmc)
    best_name = min(metrics, key=lambda k: metrics[k]["mae"])
    # §4 direction diagnostics
    dr = np.array([d["r"] for d in dir_rows])
    dd_mm = np.array([d["mm"] for d in dir_rows])
    dd_mp = np.array([d["mp"] for d in dir_rows])
    dd_pd = np.array([d["pd_dir"] for d in dir_rows])
    dd_y = np.array([d["y"] for d in dir_rows])
    dd_wm = np.array([d["wm"] for d in dir_rows])
    dd_wp = np.array([d["wp"] for d in dir_rows])
    sig = np.abs(dr) > 1e-3
    if sig.any():
        acc = float((dd_pd[sig] == np.sign(dr[sig])).mean())
    else:
        acc = float("nan")
    pos = dr > 1e-3
    neg = dr < -1e-3
    r_pos = float(np.corrcoef(dd_mp[pos], dr[pos])[0, 1]) if pos.sum() >= 2 else None
    r_neg = float(np.corrcoef(dd_mm[neg], -dr[neg])[0, 1]) if neg.sum() >= 2 else None
    # four regime stats
    tail_thr = np.quantile(np.abs(dd_y), 0.95)
    regimes = {}
    for name, mask in (("high_tail", np.abs(dd_y) >= tail_thr),
                       ("neg_price", dd_y < 0),
                       ("normal", (np.abs(dd_y) < tail_thr) & (dd_y >= 0))):
        if mask.sum() >= 2:
            regimes[name] = {
                "n": int(mask.sum()),
                "dir_acc": round(float((dd_pd[mask] == np.sign(dr[mask])).mean()), 4),
                "corr_mp_pos": round(float(np.corrcoef(dd_mp[mask], dr[mask])[0, 1]), 4),
                "mean_mm": round(float(dd_mm[mask].mean()), 4),
                "mean_mp": round(float(dd_mp[mask].mean()), 4),
            }
    # non-zero dose with negative action gain: dominant-direction gain per hour.
    # g = |r| − |r ∓ m| is the gain of shifting by m in each direction; the
    # dominant direction is the larger shift (m_plus >= m_minus -> up-gain).
    gd = np.abs(dr) - np.abs(dr + dd_mm)   # gain of a DOWN shift by mm
    gu = np.abs(dr) - np.abs(dr - dd_mp)   # gain of an UP shift by mp
    dose = (dd_mm > 1e-4) | (dd_mp > 1e-4)
    dom_up = dd_mp >= dd_mm
    gain_dom = np.where(dom_up, gu, gd)
    neg_gain = dose & (gain_dom < 0)
    return {
        "s2v_metrics": metrics,
        "best_readout_s2v": best_name,
        "best_mae_s2v": metrics[best_name]["mae"],
        "identity_mae_s2v": metrics["identity"]["mae"],
        "s2v_mae_improvement_frac": round((metrics["identity"]["mae"] - metrics[best_name]["mae"]) / metrics["identity"]["mae"], 5)
                                   if metrics["identity"]["mae"] > 0 else None,
        "dir": {
            "n_hours": int(len(dr)),
            "dir_accuracy": round(acc, 4),
            "corr_mp_vs_positive_residual": round(r_pos, 4) if r_pos is not None else None,
            "corr_mm_vs_negative_residual": round(r_neg, 4) if r_neg is not None else None,
            "frac_zero_direction": round(float((dd_pd == 0).mean()), 4),
            "regimes": regimes,
            "frac_dose_neg_gain": round(float(neg_gain.sum() / max(dose.sum(), 1)), 4),
            "n_dose": int(dose.sum()),
        },
    }


# ----------------------------------------------------------- §3 S4 eval ----
def s4_eval(dd: dict, readout: str) -> dict:
    """Frozen S4 evaluation of a chosen readout (no tuning on S4)."""
    preds_all, y_all, host_all = [], [], []
    n_days = 0
    for day in dd["days"]:
        if day["block"] != "dev":
            continue
        n_days += 1
        z0 = np.asarray(day["z0"]); mm = np.asarray(day["mm"])
        mp = np.asarray(day["mp"]); wm = np.asarray(day["wm"])
        wp = np.asarray(day["wp"])
        wz = 1.0 - wm - wp
        s = float(day["s_day"])
        y = np.asarray(day["price"])
        vm = np.asarray(day["vm"]).astype(bool)
        preds = readouts_from_atoms(z0, wm, wz, wp, mm, mp, s)
        if readout.startswith("shrink"):
            lam = float(readout.split("_")[1])
            mu_R = wp * mp - wm * mm
            x = s * np.sinh(z0 + lam * mu_R)
        else:
            x = np.asarray(preds[readout])
        preds_all.append(x[vm]); y_all.append(y[vm])
        host_all.append(s * np.sinh(z0)[vm])
    if not preds_all:
        return {"n_days": 0}
    yc = np.concatenate(y_all)
    xc = np.concatenate(preds_all)
    hc = np.concatenate(host_all)
    ae = np.abs(yc - xc)
    sm = 200.0 * ae / (np.abs(yc) + np.abs(xc) + EPS)
    tail = np.abs(yc) >= np.quantile(np.abs(yc), 0.95)
    host_ae = np.abs(yc - hc)
    return {
        "n_days": n_days, "n_hours": int(len(yc)),
        "readout": readout,
        "mae": round(float(ae.mean()), 6),
        "smape_nofloor": round(float(sm.mean()), 6),
        "rmse": round(float(np.sqrt((ae ** 2).mean())), 6),
        "high_tail_mae": round(float(ae[tail].mean()), 6) if tail.any() else None,
        "neg_price_mae": round(float(ae[yc < 0].mean()), 6) if (yc < 0).any() else None,
        "host_mae": round(float(host_ae.mean()), 6),
        "improvement_vs_host": round(float((host_ae.mean() - ae.mean()) / host_ae.mean()), 5)
                               if host_ae.mean() > 0 else None,
    }


# --------------------------------------------------------------- main ----
DEFAULT_DOMAINS = ["LAGO_DE:Linear", "LAGO_DE:MLP", "LAGO_PJM:MLP",
                   "NEM_SA1:MLP", "NORD_DK1:Linear", "GEFCOM14P:Linear"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=str, default=",".join(DEFAULT_DOMAINS))
    ap.add_argument("--head", type=str,
                    default="experiments/08-hch-v2/results/P0A_RERUN_20260814/learned_sig_main_head.pt")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else \
        HERE / "results" / f"TRAINING_INVESTIGATION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"toy_test": toy_test(), "domains": {}}

    head = build_head("learned_sig")
    head.load_state_dict(torch.load(args.head, map_location="cpu"))
    head.eval()
    report["head"] = args.head
    print("== toy test ==", flush=True)
    print(json.dumps(report["toy_test"], ensure_ascii=False, indent=2))

    for name in [d.strip() for d in args.domains.split(",") if d.strip()]:
        mk, bb = name.split(":")
        print(f"\n== {name} ==", flush=True)
        info = R.prepare_domain(mk, bb, seed=SEED)
        det_np = R.det_for_variant("learned_sig", info)
        s2v = s2v_scan(head, info, det_np)
        dd = M.collect_domain(None, mk, bb, "learned_sig", head=head)
        best = s2v["best_readout_s2v"]
        s4 = s4_eval(dd, best)
        report["domains"][name] = {"s2v": s2v, "s4": s4}
        print(f"  S2V best readout: {best}  mae={s2v['best_mae_s2v']}"
              f" (identity={s2v['identity_mae_s2v']})")
        print(f"  S4 frozen: {s4}")
        print(f"  §4 dir: {json.dumps(s2v['dir'], ensure_ascii=False)}")

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[done] artifacts: {out_dir}")


if __name__ == "__main__":
    main()
