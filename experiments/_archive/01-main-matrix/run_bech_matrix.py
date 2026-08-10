"""Phase 3 driver: 5 backbones x {base, +BECH} on public extreme-price data.

Protocol (four-segment rolling-origin isolation, chronological, disjoint):
    S1 50%  train the backbone            -> backbone FROZEN
    S2 20%  train the BOM-SSC corrector   -> corrector FROZEN
    S3 10%  SCARR conformal risk routing  -> lambdas FROZEN
    S4 20%  final test (never touched)

Everything reported comes from S4 only. The spike threshold is taken from S1
(training) prices, never from the test segment.

Usage
-----
  python run_bech_matrix.py                       # full matrix
  python run_bech_matrix.py --quick               # smoke test
  python run_bech_matrix.py --datasets LAGO_DE NEM_SA1 --backbones Linear GBDT
"""
from __future__ import annotations

import argparse, json, os, sys, time, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import backbones as B                                 # noqa: E402
from bech import BECH, build_corrector_features, harm_stats   # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(OUT, "results")
os.makedirs(RESDIR, exist_ok=True)
TAG = ""        # suffix for the per-dataset result json
SUMTAG = None   # suffix for the shard-level summary csv/md (defaults to TAG)

DEFAULT_DATASETS = ["LAGO_DE", "LAGO_BE", "LAGO_FR", "LAGO_PJM", "LAGO_NP",
                    "NEM_SA1", "NEM_VIC1", "NEM_NSW1", "GEFCOM14P"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def run_dataset(key: str, backbone_names: list[str], alpha: float,
                rho: float = 0.50, seed: int = 0, tau_mode: str = "bayes",
                lam_select: str = "lcb") -> dict:
    t0 = time.time()
    ds = C.load_dataset(key)
    X, y, names, valid = C.build_tabular(ds)
    C.assert_no_leakage(ds, X, y, valid, names)
    log(f"{key}: rows={len(y)} feats={X.shape[1]} "
        f"span={ds['ts'].iloc[valid[0]].date()}..{ds['ts'].iloc[valid[-1]].date()} "
        f"neg={float((y < 0).mean()):.2%}")

    need_seq = any(B.needs_seq(b) for b in backbone_names)
    seq = C.build_sequences(ds, valid) if need_seq else None

    sp = C.four_segment_split(len(y))
    S1, S2, S3, S4 = sp["S1"], sp["S2"], sp["S3"], sp["S4"]
    spike_thr = float(np.quantile(y[S1], 0.99))        # from TRAIN only
    naive4 = C.weekly_naive(ds["price"], valid, S4)

    hour = ds["ts"].dt.hour.to_numpy()[valid]
    dayid = ds["ts"].dt.floor("D").astype("int64").to_numpy()[valid]

    rec = dict(dataset=key, tier=ds["meta"].get("tier"),
               currency=ds["meta"].get("currency"), note=ds["meta"].get("note", ""),
               n_rows=int(len(y)), spike_thr_train=spike_thr,
               neg_pct_all=float((y < 0).mean() * 100),
               neg_pct_test=float((y[S4] < 0).mean() * 100),
               split={k: [int(v[0]), int(v[-1])] for k, v in sp.items()},
               test_span=[str(ds["ts"].iloc[valid[S4[0]]]),
                          str(ds["ts"].iloc[valid[S4[-1]]])],
               models={})

    for bn in backbone_names:
        tb = time.time()
        try:
            m = B.make_backbone(bn, seed)
            if B.needs_seq(bn):
                m.fit(X[S1], y[S1], seq[S1])
                yhat = m.predict(X, seq)
            else:
                m.fit(X[S1], y[S1])
                yhat = m.predict(X)

            Z, _zn = build_corrector_features(X, names, yhat, y, hour, dayid,
                                              oos_start=int(S2[0]))
            head = BECH(neg_thr=0.0, spike_thr=spike_thr, alpha=alpha,
                        harm_budget_ratio=rho, tau_mode=tau_mode,
                        lam_select=lam_select, seed=seed)
            head.fit(Z[S2], yhat[S2], y[S2])
            head.calibrate(Z[S3], yhat[S3], y[S3])
            corrected, diag = head.apply(Z[S4], yhat[S4])

            base_m = C.evaluate(y[S4], yhat[S4], naive4, 0.0, spike_thr)
            bech_m = C.evaluate(y[S4], corrected, naive4, 0.0, spike_thr)
            dm = C.dm_test(yhat[S4] - y[S4], corrected - y[S4], lag=24)
            hs = harm_stats(y[S4], yhat[S4], corrected)

            rec["models"][bn] = dict(base=base_m, bech=bech_m, dm=dm,
                                     routing=diag, harm=hs, head_info=head.info,
                                     seconds=round(time.time() - tb, 1))
            log(f"   {key}/{bn}: MAE {base_m['mae']:.2f}->{bech_m['mae']:.2f} "
                f"tailRMSE {base_m['tail_rmse']:.1f}->{bech_m['tail_rmse']:.1f} "
                f"negMiss {base_m['neg_miss_rate']}->{bech_m['neg_miss_rate']} "
                f"fire={diag['fire_rate']:.3%} lam=({diag['lam_neg']},{diag['lam_pos']}) "
                f"[{time.time()-tb:.0f}s]")
        except Exception as e:
            log(f"   ERROR {key}/{bn}: {type(e).__name__}: {e}")
            traceback.print_exc()
            rec["models"][bn] = dict(error=f"{type(e).__name__}: {e}")

    rec["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(RESDIR, f"bech_{key}{TAG}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    return rec


def _fmt(v, spec="{:.2f}", pct=False):
    if v is None:
        return "—"
    return f"{v:.2%}" if pct else spec.format(v)


def summarize(records: list[dict]) -> None:
    rows = []
    for r in records:
        for bn, mm in r["models"].items():
            if "error" in mm:
                continue
            b, c = mm["base"], mm["bech"]
            rows.append(dict(
                dataset=r["dataset"], tier=r["tier"], backbone=bn,
                neg_pct_test=r["neg_pct_test"],
                mae_base=b["mae"], mae_bech=c["mae"],
                mae_gain_pct=100 * (b["mae"] - c["mae"]) / b["mae"],
                rmae_base=b.get("rmae"), rmae_bech=c.get("rmae"),
                tail_rmse_base=b["tail_rmse"], tail_rmse_bech=c["tail_rmse"],
                tail_gain_pct=100 * (b["tail_rmse"] - c["tail_rmse"]) / b["tail_rmse"],
                neg_n=b["neg_n"],
                neg_miss_base=b["neg_miss_rate"], neg_miss_bech=c["neg_miss_rate"],
                mae_neg_base=b["mae_on_neg"], mae_neg_bech=c["mae_on_neg"],
                spike_n=b["spike_n"],
                mae_spike_base=b["mae_on_spike"], mae_spike_bech=c["mae_on_spike"],
                mae_normal_base=b["mae_on_normal"], mae_normal_bech=c["mae_on_normal"],
                fire_rate=mm["routing"]["fire_rate"],
                lam_neg=mm["routing"]["lam_neg"], lam_pos=mm["routing"]["lam_pos"],
                harm_rate=mm["harm"]["harm_rate"],
                dm_stat=mm["dm"]["dm_stat"], dm_p=mm["dm"]["p_value"],
            ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESDIR, (f"bech_matrix_summary{SUMTAG if SUMTAG is not None else TAG}.csv")), index=False)

    md = ["# Phase 3：BECH（BOM-SSC + SCARR）在公开极端电价数据上的基座无关性实验", "",
          "> 协议：四段 rolling-origin 隔离 S1(50%)基座训练 / S2(20%)校正器训练 / "
          "S3(10%)共形风险标定 / S4(20%)最终测试，**全部指标仅来自 S4**。",
          "> 尖峰阈值取自 S1 训练段 p99，绝不使用测试段信息。基座训练后**冻结**，校正头只见其输出。",
          "> DM = Diebold-Mariano 单边检验（绝对误差，HAC lag=24），H1: +BECH 更优。", ""]

    md += ["## 1. 主表：5 基座 × {base, +BECH}", "",
           "| 数据集 | 层级 | 基座 | 测试段负价% | MAE base→+BECH | MAE↓% | 尾部RMSE base→+BECH | 尾部↓% | DM p | 触发率 | λ(neg,pos) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['dataset']} | {r['tier']} | {r['backbone']} | {r['neg_pct_test']:.2f}% | "
            f"{r['mae_base']:.2f} → {r['mae_bech']:.2f} | {r['mae_gain_pct']:+.2f}% | "
            f"{r['tail_rmse_base']:.1f} → {r['tail_rmse_bech']:.1f} | {r['tail_gain_pct']:+.2f}% | "
            f"{_fmt(r['dm_p'], '{:.4f}')} | {r['fire_rate']:.2%} | "
            f"({r['lam_neg']:.2f}, {r['lam_pos']:.2f}) |")

    md += ["", "## 2. 负电价分支（核心创新点的直接证据）", "",
           "| 数据集 | 基座 | 测试段负价点数 | 漏判率 base→+BECH | 负价MAE base→+BECH |",
           "|---|---|---|---|---|"]
    for r in rows:
        if not r["neg_n"]:
            continue
        md.append(f"| {r['dataset']} | {r['backbone']} | {r['neg_n']} | "
                  f"{_fmt(r['neg_miss_base'], pct=True)} → {_fmt(r['neg_miss_bech'], pct=True)} | "
                  f"{_fmt(r['mae_neg_base'])} → {_fmt(r['mae_neg_bech'])} |")

    md += ["", "## 3. 安全性：正常时段退化预算 + 伤害率", "",
           "> BOM-SSC 在未触发点上恒等（Δ≡0），因此正常时段 MAE 理论上**逐点相同**；",
           "> SCARR 在无法给出共形保证时令 λ=0（弃权），负对照市场应观察到"
           "触发率≈0 或伤害率受控。", "",
           "| 数据集 | 基座 | 正常时段MAE base→+BECH | 触发点数 | 伤害率 | 触发点平均收益 |",
           "|---|---|---|---|---|---|"]
    for r in records:
        for bn, mm in r["models"].items():
            if "error" in mm:
                continue
            b, c, h = mm["base"], mm["bech"], mm["harm"]
            md.append(f"| {r['dataset']} | {bn} | {_fmt(b['mae_on_normal'])} → "
                      f"{_fmt(c['mae_on_normal'])} | {h['n_fired']} | "
                      f"{_fmt(h['harm_rate'], pct=True)} | "
                      f"{_fmt(h['mean_gain_on_fired'])} |")

    md += ["", "## 4. 负对照（应当「不伤害」）", ""]
    for r in records:
        if r["tier"] != "L3":
            continue
        md.append(f"- **{r['dataset']}**（{r['note']}）：")
        for bn, mm in r["models"].items():
            if "error" in mm:
                continue
            d = mm["routing"]; h = mm["harm"]
            md.append(f"  - {bn}: 触发率 {d['fire_rate']:.2%}，λ=({d['lam_neg']:.2f},"
                      f"{d['lam_pos']:.2f})，MAE {mm['base']['mae']:.2f}→{mm['bech']['mae']:.2f}"
                      f"，伤害率 {_fmt(h['harm_rate'], pct=True)}")
    md.append("")
    with open(os.path.join(RESDIR, (f"bech_matrix_evidence{SUMTAG if SUMTAG is not None else TAG}.md")), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))
    log(f"[done] -> {RESDIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--backbones", nargs="*", default=list(B.BACKBONES))
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--rho", type=float, default=0.50,
                    help="SCARR harm budget as a fraction of branch baseline MAE")
    ap.add_argument("--tau-mode", default="bayes",
                    choices=["bayes", "auto", "grid_capped"])
    ap.add_argument("--lam-select", default="lcb", choices=["lcb", "largest"])
    ap.add_argument("--tag", default="", help="suffix for per-dataset result json")
    ap.add_argument("--sum-tag", default=None,
                    help="suffix for the shard summary csv/md (defaults to --tag)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    global TAG, SUMTAG
    TAG = a.tag
    SUMTAG = a.sum_tag

    dsl = a.datasets or DEFAULT_DATASETS
    bbl = a.backbones
    if a.quick:
        dsl, bbl = ["LAGO_DE", "GEFCOM14P"], ["Linear", "GBDT"]

    log(f"datasets={dsl}")
    log(f"backbones={bbl} alpha={a.alpha} rho={a.rho} tau_mode={a.tau_mode} "
        f"lam_select={a.lam_select} device=CPU")
    recs = []
    for k in dsl:
        try:
            recs.append(run_dataset(k, bbl, a.alpha, a.rho,
                                    tau_mode=a.tau_mode, lam_select=a.lam_select))
        except Exception as e:
            log(f"FATAL {k}: {type(e).__name__}: {e}")
            traceback.print_exc()
    summarize(recs)


if __name__ == "__main__":
    main()
