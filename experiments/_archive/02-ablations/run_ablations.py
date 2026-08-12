"""Phase 3 ablations: which component of BECH v1 actually earns the gain?

Cumulative ladder v0 -> v1 (each row adds ONE change to the row above):
    A0  v0            L2 magnitude, harm-rate-capped tau grid,
                      largest feasible lambda, rho = 0.10
    A1  + L1 magnitude          conditional MEDIAN instead of MEAN tail residual
    A2  + Bayes gate            tau fixed a priori at 0.5, no tuning window
    A3  + two-tier certificate  bootstrap LCB on the mean gain, rho = 0.50
    A4  + S2b reuse  (= v1)     the freed tuning slice joins the calibration set

Plus two sensitivity sweeps on the certificate parameters (rho, alpha) and a
DEGRADATION ablation that removes SCARR entirely (lambda == 1, no certificate)
to show what the safety layer is actually buying.

Fast backbones only (Linear, GBDT): the ablation question is about the
correction head, not about the backbone family, and the full backbone sweep is
covered by run_bech_matrix.py.
"""
from __future__ import annotations

import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, backbones as B                      # noqa: E402
from _legacy.bech import BECH, build_corrector_features, harm_stats   # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(OUT, "results"); os.makedirs(RESDIR, exist_ok=True)

PAIRS = [("NEM_SA1", "GBDT"), ("NEM_SA1", "Linear"),
         ("NEM_VIC1", "GBDT"), ("NEM_NSW1", "GBDT"),
         ("LAGO_DE", "GBDT"), ("LAGO_BE", "GBDT"),
         ("LAGO_PJM", "GBDT"), ("LAGO_NP", "GBDT"), ("GEFCOM14P", "GBDT")]

LADDER = {
    "A0 v0": dict(mag_objective="regression", tau_mode="grid_capped",
                  lam_select="largest", harm_budget_ratio=0.10,
                  reuse_s2b_for_calib=False),
    "A1 +L1 magnitude": dict(mag_objective="regression_l1", tau_mode="grid_capped",
                             lam_select="largest", harm_budget_ratio=0.10,
                             reuse_s2b_for_calib=False),
    "A2 +Bayes gate": dict(mag_objective="regression_l1", tau_mode="bayes",
                           lam_select="largest", harm_budget_ratio=0.10,
                           reuse_s2b_for_calib=False),
    "A3 +two-tier cert": dict(mag_objective="regression_l1", tau_mode="bayes",
                              lam_select="lcb", harm_budget_ratio=0.50,
                              reuse_s2b_for_calib=False),
    "A4 +S2b reuse (=v1)": dict(mag_objective="regression_l1", tau_mode="bayes",
                                lam_select="lcb", harm_budget_ratio=0.50,
                                reuse_s2b_for_calib=True),
}

V1 = LADDER["A4 +S2b reuse (=v1)"]
SENS_RHO = [0.0, 0.10, 0.25, 0.50, 1.00]
SENS_ALPHA = [0.05, 0.10, 0.20]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def prep(key, bb):
    ds = C.load_dataset(key)
    X, y, names, valid = C.build_tabular(ds)
    C.assert_no_leakage(ds, X, y, valid, names)
    sp = C.four_segment_split(len(y))
    S1, S2, S3, S4 = sp["S1"], sp["S2"], sp["S3"], sp["S4"]
    spike = float(np.quantile(y[S1], 0.99))
    hour = ds["ts"].dt.hour.to_numpy()[valid]
    dayid = ds["ts"].dt.floor("D").astype("int64").to_numpy()[valid]
    m = B.make_backbone(bb); m.fit(X[S1], y[S1]); yhat = m.predict(X)
    Z, _ = build_corrector_features(X, names, yhat, y, hour, dayid,
                                    oos_start=int(S2[0]))
    naive4 = C.weekly_naive(ds["price"], valid, S4)
    return dict(y=y, yhat=yhat, Z=Z, S2=S2, S3=S3, S4=S4, spike=spike,
                naive4=naive4, tier=ds["meta"].get("tier"))


def one(d, label, kw, alpha=0.10, no_scarr=False):
    y, yhat, Z = d["y"], d["yhat"], d["Z"]
    S2, S3, S4 = d["S2"], d["S3"], d["S4"]
    h = BECH(neg_thr=0.0, spike_thr=d["spike"], alpha=alpha, weight="gate", **kw)
    h.fit(Z[S2], yhat[S2], y[S2])
    if no_scarr:
        # remove the safety layer: fire wherever the gate fires, lambda == 1
        h.lam = {b: (1.0 if h.clf.get(b) is not None else 0.0)
                 for b in ("neg", "pos")}
        h.info["scarr"] = {"status": "DISABLED (ablation)"}
    else:
        h.calibrate(Z[S3], yhat[S3], y[S3])
    cor, dg = h.apply(Z[S4], yhat[S4])
    b = C.evaluate(y[S4], yhat[S4], d["naive4"], 0.0, d["spike"])
    c = C.evaluate(y[S4], cor, d["naive4"], 0.0, d["spike"])
    hs = harm_stats(y[S4], yhat[S4], cor)
    dm = C.dm_test(yhat[S4] - y[S4], cor - y[S4], lag=24)
    return dict(variant=label, mae_base=b["mae"], mae_bech=c["mae"],
                mae_gain_pct=100 * (b["mae"] - c["mae"]) / b["mae"],
                tail_base=b["tail_rmse"], tail_bech=c["tail_rmse"],
                tail_gain_pct=100 * (b["tail_rmse"] - c["tail_rmse"]) / b["tail_rmse"],
                neg_miss_base=b["neg_miss_rate"], neg_miss_bech=c["neg_miss_rate"],
                mae_neg_base=b["mae_on_neg"], mae_neg_bech=c["mae_on_neg"],
                mae_normal_base=b["mae_on_normal"], mae_normal_bech=c["mae_on_normal"],
                tau_neg=h.tau["neg"], tau_pos=h.tau["pos"],
                lam_neg=h.lam["neg"], lam_pos=h.lam["pos"],
                fire_rate=dg["fire_rate"], n_fired=hs["n_fired"],
                harm_rate=hs["harm_rate"], worst_harm=hs["worst_harm"],
                dm_p=dm["p_value"])


def main():
    rows = []
    for key, bb in PAIRS:
        t0 = time.time()
        d = prep(key, bb)
        base = dict(dataset=key, backbone=bb, tier=d["tier"])
        for label, kw in LADDER.items():
            r = one(d, label, kw); r.update(base); r["family"] = "ladder"
            rows.append(r)
            log(f"{key}/{bb} {label:22s} MAE {r['mae_base']:8.3f}->{r['mae_bech']:8.3f} "
                f"({r['mae_gain_pct']:+.2f}%) fire={r['fire_rate']:.2%} "
                f"lam=({r['lam_neg']:.2f},{r['lam_pos']:.2f})")
        for rho in SENS_RHO:
            kw = dict(V1); kw["harm_budget_ratio"] = rho
            r = one(d, f"rho={rho:.2f}", kw); r.update(base); r["family"] = "rho"
            rows.append(r)
        for al in SENS_ALPHA:
            r = one(d, f"alpha={al:.2f}", dict(V1), alpha=al)
            r.update(base); r["family"] = "alpha"
            rows.append(r)
        r = one(d, "no-SCARR (lambda=1)", dict(V1), no_scarr=True)
        r.update(base); r["family"] = "safety"
        rows.append(r)
        log(f"{key}/{bb} no-SCARR       MAE {r['mae_base']:8.3f}->{r['mae_bech']:8.3f} "
            f"({r['mae_gain_pct']:+.2f}%) worst_harm={r['worst_harm']}")
        log(f"--- {key}/{bb} done in {time.time()-t0:.0f}s ---")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESDIR, "bech_ablations.csv"), index=False)
    log(f"[done] -> {os.path.join(RESDIR, 'bech_ablations.csv')}  rows={len(df)}")


if __name__ == "__main__":
    main()
