"""Phase 3 peer comparison: BECH vs. the standard model-agnostic post-processors.

Why this file exists
--------------------
`run_bech_matrix.py` only answers "does BECH beat an uncorrected backbone?".
That is a necessary but very weak claim: *any* residual learner beats "do
nothing" on a biased backbone. The paper-relevant question is whether the two
things BECH actually contributes -- (a) SELECTIVITY (act only where the event
probability crosses the Bayes threshold) and (b) the CERTIFICATE (act only with
a bootstrap/conformal guarantee) -- buy anything over the post-processors a
reviewer will immediately name.

Fairness rules (deliberately tilted AGAINST our method)
-------------------------------------------------------
* identical four-segment split, identical frozen backbone, identical corrector
  feature matrix Z for every competitor;
* identical learner family and hyper-parameters (LightGBM, the ones BECH's own
  magnitude head uses) -- so any difference is the CORRECTION POLICY, not the
  function approximator;
* every competitor that can use a calibration segment is GIVEN S3, exactly like
  SCARR gets it (M3/M4/M5 below). We do not compare a calibrated method against
  uncalibrated competitors;
* all numbers come from S4 only.

Reported comparisons
--------------------
* each method vs. the uncorrected backbone (DM, one-sided);
* **BECH vs. each peer directly** (paired DM on absolute errors) -- not inferred
  indirectly through the shared baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np
import pandas as pd

import backbones as B
import common as C
from bech import BECH, build_corrector_features, harm_stats

OUT = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(OUT, "results")
os.makedirs(RESDIR, exist_ok=True)

DEFAULT_DATASETS = ["NEM_SA1", "NEM_VIC1", "NEM_NSW1", "LAGO_DE", "LAGO_BE",
                    "LAGO_FR", "LAGO_PJM", "LAGO_NP", "GEFCOM14P"]

METHOD_ORDER = [
    "base",
    "M0 retrain-on-S1+S2",
    "M1 delta-global-L2",
    "M2 delta-global-L1",
    "M3 delta-global-L1+shrink",
    "M4 quantile-postproc",
    "M5 EVT-tail-rescale",
    "M6 selective-no-cert",
    "M7 BECH v1",
    "M8 delta-global-L1 -> BECH",
    "M9 retrain-on-S1+S2 -> BECH",
]

METHOD_NOTE = {
    "M0 retrain-on-S1+S2": "对照：不做任何后处理，直接把基座在 S1∪S2 上重训——"
                           "用来隔离「数据新鲜度」与「后处理方法」两种收益来源",
    "M1 delta-global-L2": "全局残差适配器（δ-Adapter），L2 目标，逐点无条件校正",
    "M2 delta-global-L1": "同上但改条件中位数目标（把我们的 L1 发现让渡给竞品）",
    "M3 delta-global-L1+shrink": "M2 + 在 S3 上按 MAE 最优选标量收缩（竞品也拿到标定段）",
    "M4 quantile-postproc": "分位后处理：直接对 y 做 q=0.5 回归 + S3 共形中位偏移",
    "M5 EVT-tail-rescale": "尾部仿射再标定：对基座判定的尾部点拟合 a+b·ŷ，S3 上把关",
    "M6 selective-no-cert": "我们的选择性门控但 λ≡1（去掉证书层）",
    "M7 BECH v1": "本文方法：选择性门控 + 两层证书",
    "M8 delta-global-L1 -> BECH": "组合：先用 M2 修基座，再把 BECH 叠加在修好的预测上",
    "M9 retrain-on-S1+S2 -> BECH": "组合：先把基座重训（M0，最强的现实部署基线），"
                                   "再把 BECH 叠加其上",
}

LGB_KW = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
              min_child_samples=10, verbose=-1)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------- competitors --
def _lgb(objective, seed, **kw):
    import lightgbm as lgb
    p = dict(LGB_KW)
    p.update(kw)
    return lgb.LGBMRegressor(objective=objective, random_state=seed,
                             n_jobs=int(os.environ.get("BECH_LGB_JOBS", "8")), **p)


def m_delta_global(Zs2, r2, Z4, yhat4, objective, seed):
    """δ-Adapter: learn the residual on the whole corrector segment and add it
    back everywhere. No gate, no certificate -- the textbook post-processor."""
    m = _lgb(objective, seed)
    m.fit(Zs2, r2)
    return yhat4 + m.predict(Z4), m


def m_delta_shrink(model, Z3, yhat3, y3, Z4, yhat4):
    """Same adapter, but allowed to shrink itself on the calibration segment.
    Grid is the same one SCARR searches, so the competitor is not handicapped."""
    d3 = model.predict(Z3)
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 3)
    mae = [np.abs(y3 - (yhat3 + g * d3)).mean() for g in grid]
    g = float(grid[int(np.argmin(mae))])
    return yhat4 + g * model.predict(Z4), g


def m_quantile_post(Zs2, yhat2, y2, Z3, yhat3, y3, Z4, yhat4, seed):
    """Distributional post-processing: re-predict the conditional median of y
    from the corrector features (the base forecast is inside Z), then remove the
    residual median measured on the held-out calibration segment (the point
    analogue of conformalising a quantile forecast)."""
    m = _lgb("quantile", seed, alpha=0.5)
    m.fit(Zs2, y2)
    off = float(np.median(y3 - m.predict(Z3)))
    return m.predict(Z4) + off, off


def oof_delta(Zs2, r2, seed, folds=3):
    """Time-blocked out-of-fold residual correction on S2.

    Needed for the COMPOSITION experiment: if we simply added the in-sample
    adapter output to the S2 forecasts, the composed backbone would look far
    better on S2 than it will ever be at deployment, and BECH's heads would be
    trained on a fantasy residual distribution. Cross-fitting keeps the composed
    forecast out-of-sample everywhere the corrector can see it.
    """
    n = len(r2)
    d = np.zeros(n)
    b = np.linspace(0, n, folds + 1).astype(int)
    for i in range(folds):
        te = np.arange(b[i], b[i + 1])
        tr = np.setdiff1d(np.arange(n), te)
        m = _lgb("regression_l1", seed + i)
        m.fit(Zs2[tr], r2[tr])
        d[te] = m.predict(Zs2[te])
    return d


def m_evt_tail(yhat2, y2, yhat3, y3, yhat4, spike_thr):
    """Classic extreme-value style fix: only the tail region of the BASE
    forecast is re-scaled by an affine map, fitted on S2 and kept only if it
    also helps on S3. Sign-aware: negative and spike tails get separate maps."""
    out = yhat4.copy()
    used = {}
    for name, sel2, sel3, sel4 in (
        ("neg", yhat2 < 0, yhat3 < 0, yhat4 < 0),
        ("pos", yhat2 > spike_thr, yhat3 > spike_thr, yhat4 > spike_thr),
    ):
        if sel2.sum() < 30 or sel3.sum() < 10 or sel4.sum() == 0:
            used[name] = None
            continue
        A_ = np.c_[np.ones(sel2.sum()), yhat2[sel2]]
        coef, *_ = np.linalg.lstsq(A_, y2[sel2], rcond=None)
        cand3 = coef[0] + coef[1] * yhat3[sel3]
        if np.abs(y3[sel3] - cand3).mean() >= np.abs(y3[sel3] - yhat3[sel3]).mean():
            used[name] = None            # rejected on the calibration segment
            continue
        out[sel4] = coef[0] + coef[1] * yhat4[sel4]
        used[name] = [float(coef[0]), float(coef[1])]
    return out, used


# ------------------------------------------------------------------- driver --
def run_pair(key, bn, ds, X, y, names, valid, seq, sp, spike_thr, naive4,
             hour, dayid, alpha, rho, seed):
    S1, S2, S3, S4 = sp["S1"], sp["S2"], sp["S3"], sp["S4"]
    m = B.make_backbone(bn, seed)
    if B.needs_seq(bn):
        m.fit(X[S1], y[S1], seq[S1])
        yhat = m.predict(X, seq)
    else:
        m.fit(X[S1], y[S1])
        yhat = m.predict(X)

    Z, _ = build_corrector_features(X, names, yhat, y, hour, dayid,
                                    oos_start=int(S2[0]))
    r = y - yhat
    preds = {"base": yhat[S4]}
    extra = {}

    # ---- M0 recency control ---------------------------------------------
    # Every post-processor below is fitted on S2, i.e. on data STRICTLY MORE
    # RECENT than the frozen backbone's training window. In a non-stationary
    # market that alone is worth a lot. Without this control one cannot tell
    # "the post-processor is clever" from "the post-processor saw newer data".
    S12 = np.concatenate([S1, S2])
    m0 = B.make_backbone(bn, seed)
    if B.needs_seq(bn):
        m0.fit(X[S12], y[S12], seq[S12])
        preds["M0 retrain-on-S1+S2"] = m0.predict(X[S4], seq[S4])
    else:
        m0.fit(X[S12], y[S12])
        preds["M0 retrain-on-S1+S2"] = m0.predict(X[S4])

    # ---- M1 / M2 global residual adapters -------------------------------
    p1, _ = m_delta_global(Z[S2], r[S2], Z[S4], yhat[S4], "regression", seed)
    preds["M1 delta-global-L2"] = p1
    p2, mod2 = m_delta_global(Z[S2], r[S2], Z[S4], yhat[S4],
                              "regression_l1", seed)
    preds["M2 delta-global-L1"] = p2

    # ---- M3 adapter + calibrated shrink ---------------------------------
    p3, g3 = m_delta_shrink(mod2, Z[S3], yhat[S3], y[S3], Z[S4], yhat[S4])
    preds["M3 delta-global-L1+shrink"] = p3
    extra["M3_shrink"] = g3

    # ---- M4 quantile post-processing ------------------------------------
    p4, off4 = m_quantile_post(Z[S2], yhat[S2], y[S2], Z[S3], yhat[S3], y[S3],
                               Z[S4], yhat[S4], seed)
    preds["M4 quantile-postproc"] = p4
    extra["M4_offset"] = off4

    # ---- M5 EVT-style tail rescaling ------------------------------------
    p5, used5 = m_evt_tail(yhat[S2], y[S2], yhat[S3], y[S3], yhat[S4], spike_thr)
    preds["M5 EVT-tail-rescale"] = p5
    extra["M5_maps"] = used5

    # ---- M6 our gate, no certificate ------------------------------------
    h6 = BECH(neg_thr=0.0, spike_thr=spike_thr, alpha=alpha,
              harm_budget_ratio=rho, tau_mode="bayes", lam_select="lcb",
              seed=seed)
    h6.fit(Z[S2], yhat[S2], y[S2])
    # strip the safety layer: fire wherever the gate fires, lambda == 1
    h6.lam = {b: (1.0 if h6.clf.get(b) is not None else 0.0)
              for b in ("neg", "pos")}
    p6, _d6 = h6.apply(Z[S4], yhat[S4])
    preds["M6 selective-no-cert"] = p6

    # ---- M7 BECH v1 ------------------------------------------------------
    h7 = BECH(neg_thr=0.0, spike_thr=spike_thr, alpha=alpha,
              harm_budget_ratio=rho, tau_mode="bayes", lam_select="lcb",
              seed=seed)
    h7.fit(Z[S2], yhat[S2], y[S2])
    h7.calibrate(Z[S3], yhat[S3], y[S3])
    p7, d7 = h7.apply(Z[S4], yhat[S4])
    preds["M7 BECH v1"] = p7
    extra["M7_routing"] = d7

    # ---- M8 composition: repair the backbone first, then stack BECH -------
    # This is the "is our module front/back compatible with a peer module?"
    # experiment. A selective corrector is only worth publishing if it still
    # adds value AFTER a generic residual adapter has taken the easy signal.
    yhat_c = yhat.copy()
    yhat_c[S2] = yhat[S2] + oof_delta(Z[S2], r[S2], seed)     # cross-fitted
    yhat_c[S3] = yhat[S3] + mod2.predict(Z[S3])
    yhat_c[S4] = p2
    Zc, _ = build_corrector_features(X, names, yhat_c, y, hour, dayid,
                                     oos_start=int(S2[0]))
    h8 = BECH(neg_thr=0.0, spike_thr=spike_thr, alpha=alpha,
              harm_budget_ratio=rho, tau_mode="bayes", lam_select="lcb",
              seed=seed)
    h8.fit(Zc[S2], yhat_c[S2], y[S2])
    h8.calibrate(Zc[S3], yhat_c[S3], y[S3])
    p8, d8 = h8.apply(Zc[S4], yhat_c[S4])
    preds["M8 delta-global-L1 -> BECH"] = p8
    extra["M8_routing"] = d8

    # ---- M9 composition on top of a PERIODICALLY REFIT backbone ----------
    # The audit (audit_peer_gain.py) shows most of the peers' headline gain is
    # data recency, not post-processing skill. So the honest, deployment-
    # relevant question is not "does BECH beat a stale backbone" but "does BECH
    # still add anything once you simply retrain". S2 is in-sample for the
    # refit backbone, so the corrector's view of S2 is built by cross-fitting
    # (train on S1 + the other folds of S2, predict the held-out fold).
    yhat_r = yhat.copy()
    folds = 3
    bnd = np.linspace(0, len(S2), folds + 1).astype(int)
    for i in range(folds):
        te = S2[bnd[i]:bnd[i + 1]]
        tr = np.concatenate([S1, np.setdiff1d(S2, te)])
        mf = B.make_backbone(bn, seed)
        if B.needs_seq(bn):
            mf.fit(X[tr], y[tr], seq[tr])
            yhat_r[te] = mf.predict(X[te], seq[te])
        else:
            mf.fit(X[tr], y[tr])
            yhat_r[te] = mf.predict(X[te])
    if B.needs_seq(bn):
        yhat_r[S3] = m0.predict(X[S3], seq[S3])
    else:
        yhat_r[S3] = m0.predict(X[S3])
    yhat_r[S4] = preds["M0 retrain-on-S1+S2"]
    Zr, _ = build_corrector_features(X, names, yhat_r, y, hour, dayid,
                                     oos_start=int(S2[0]))
    h9 = BECH(neg_thr=0.0, spike_thr=spike_thr, alpha=alpha,
              harm_budget_ratio=rho, tau_mode="bayes", lam_select="lcb",
              seed=seed)
    h9.fit(Zr[S2], yhat_r[S2], y[S2])
    h9.calibrate(Zr[S3], yhat_r[S3], y[S3])
    p9, d9 = h9.apply(Zr[S4], yhat_r[S4])
    preds["M9 retrain-on-S1+S2 -> BECH"] = p9
    extra["M9_routing"] = d9

    # ---- scoring ---------------------------------------------------------
    y4, base4 = y[S4], yhat[S4]
    mae_base = float(np.abs(base4 - y4).mean())
    rows = []
    for name in METHOD_ORDER:
        p = preds[name]
        # stacked variants are scored against the thing they are stacked on
        if name.startswith("M8"):
            ref = preds["M2 delta-global-L1"]
        elif name.startswith("M9"):
            ref = preds["M0 retrain-on-S1+S2"]
        else:
            ref = base4
        ev = C.evaluate(y4, p, naive4, 0.0, spike_thr)
        hs = harm_stats(y4, ref, p)
        dmb = C.dm_test(base4 - y4, p - y4, lag=24)
        dmr = C.dm_test(ref - y4, p - y4, lag=24)
        # direct pairwise DM: BECH vs this peer (H1: BECH better) -- computed
        # head-to-head, never inferred through the shared baseline
        dmp = (C.dm_test(p - y4, preds["M7 BECH v1"] - y4, lag=24)
               if name != "M7 BECH v1" else dict(dm_stat=None, p_value=None))
        rows.append(dict(
            dataset=key, backbone=bn, method=name,
            mae=ev["mae"], tail_rmse=ev["tail_rmse"],
            neg_miss=ev["neg_miss_rate"], mae_neg=ev["mae_on_neg"],
            mae_normal=ev["mae_on_normal"],
            gain_vs_base=100 * (mae_base - ev["mae"]) / mae_base,
            gain_vs_ref=100 * (float(np.abs(ref - y4).mean()) - ev["mae"])
            / float(np.abs(ref - y4).mean()),
            n_touched=int((np.abs(p - ref) > 1e-9).sum()),
            touch_rate=float((np.abs(p - ref) > 1e-9).mean()),
            worst_harm=hs["worst_harm"], harm_rate=hs["harm_rate"],
            dm_p_vs_base=dmb["p_value"], dm_p_vs_ref=dmr["p_value"],
            dm_p_bech_beats=dmp["p_value"],
        ))
    return rows, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--backbones", nargs="*", default=["Linear", "GBDT"])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--rho", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    dsl = a.datasets or DEFAULT_DATASETS
    log(f"peer comparison datasets={dsl} backbones={a.backbones} "
        f"alpha={a.alpha} rho={a.rho}")

    all_rows, all_extra = [], {}
    for key in dsl:
        try:
            ds = C.load_dataset(key)
            X, y, names, valid = C.build_tabular(ds)
            C.assert_no_leakage(ds, X, y, valid, names)
            need_seq = any(B.needs_seq(b) for b in a.backbones)
            seq = C.build_sequences(ds, valid) if need_seq else None
            sp = C.four_segment_split(len(y))
            spike_thr = float(np.quantile(y[sp["S1"]], 0.99))
            naive4 = C.weekly_naive(ds["price"], valid, sp["S4"])
            hour = ds["ts"].dt.hour.to_numpy()[valid]
            dayid = ds["ts"].dt.floor("D").astype("int64").to_numpy()[valid]
            log(f"{key}: rows={len(y)} neg={float((y < 0).mean()):.2%}")
            for bn in a.backbones:
                t0 = time.time()
                try:
                    rows, extra = run_pair(key, bn, ds, X, y, names, valid, seq,
                                           sp, spike_thr, naive4, hour, dayid,
                                           a.alpha, a.rho, a.seed)
                    all_rows += rows
                    all_extra[f"{key}/{bn}"] = extra
                    tbl = {r["method"]: r["mae"] for r in rows}
                    log(f"   {key}/{bn} [{time.time()-t0:.0f}s] " + "  ".join(
                        f"{k.split()[0]}={v:.2f}" for k, v in tbl.items()))
                except Exception as e:
                    log(f"   ERROR {key}/{bn}: {type(e).__name__}: {e}")
                    traceback.print_exc()
        except Exception as e:
            log(f"FATAL {key}: {type(e).__name__}: {e}")
            traceback.print_exc()

    df = pd.DataFrame(all_rows)
    p = os.path.join(RESDIR, f"bech_peers{a.tag}.csv")
    df.to_csv(p, index=False)
    with open(os.path.join(RESDIR, f"bech_peers_extra{a.tag}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(all_extra, fh, indent=2, ensure_ascii=False, default=str)
    log(f"[done] -> {p}  rows={len(df)}")


if __name__ == "__main__":
    main()
