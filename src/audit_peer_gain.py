"""Audit: where does the huge gain of the "global post-processors" come from?

On NEM-SA1/GBDT the quantile post-processor reaches MAE 71.9 against a frozen
backbone at 113.6 (-36.7%). A number that large must be explained before it can
go into a paper -- either it is leakage, or the comparison is not measuring what
its name suggests.

This script decomposes the gap into three additive sources, all measured on the
same S4:

  (a) LEAKAGE                 -- is any corrector feature a function of y_t?
  (b) DATA RECENCY            -- the post-processor is fitted on S2, which is
                                 strictly newer than the backbone's S1.
  (c) FEATURE SET             -- Z contains the backbone's own output plus the
                                 realised residual history (lag >= 24h), which
                                 the backbone itself never sees.

Reference points fitted for the decomposition (all LightGBM, same hyper-params):
  R0  backbone            : X[S1]  -> y            (the frozen baseline)
  R1  same feats, new data: X[S2]  -> y            (isolates recency)
  R2  rich feats, new data: Z[S2]  -> y            (recency + feature set) == M4 family
  R3  rich feats, old data: Z'[S1] -> y            (feature set only; residual
                                                    history unavailable on S1,
                                                    so this one is reported as
                                                    N/A and explained instead)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import backbones as B
import common as C
from bech import build_corrector_features

OUT = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(OUT, "results")


def lgbm(seed=0, objective="regression_l1"):
    import lightgbm as lgb
    return lgb.LGBMRegressor(objective=objective, n_estimators=300,
                             learning_rate=0.05, num_leaves=15,
                             min_child_samples=10, random_state=seed,
                             n_jobs=int(os.environ.get("BECH_LGB_JOBS", "8")),
                             verbose=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["NEM_SA1", "NEM_VIC1", "LAGO_DE"])
    ap.add_argument("--backbone", default="GBDT")
    a = ap.parse_args()

    rows = []
    for key in a.datasets:
        ds = C.load_dataset(key)
        X, y, names, valid = C.build_tabular(ds)
        C.assert_no_leakage(ds, X, y, valid, names)
        sp = C.four_segment_split(len(y))
        S1, S2, S3, S4 = sp["S1"], sp["S2"], sp["S3"], sp["S4"]
        hour = ds["ts"].dt.hour.to_numpy()[valid]
        dayid = ds["ts"].dt.floor("D").astype("int64").to_numpy()[valid]

        m = B.make_backbone(a.backbone, 0)
        m.fit(X[S1], y[S1])
        yhat = m.predict(X)
        Z, zn = build_corrector_features(X, names, yhat, y, hour, dayid,
                                         oos_start=int(S2[0]))

        # ---- (a) leakage audit on the CORRECTOR matrix ------------------
        y4 = y[S4]
        worst_r, worst_nm = 0.0, ""
        for j, nm in enumerate(zn):
            c = Z[S4][:, j]
            if np.std(c) < 1e-12:
                continue
            r = abs(np.corrcoef(c, y4)[0, 1])
            if r > worst_r:
                worst_r, worst_nm = r, nm
        # structural check on the residual-history feature
        j24 = zn.index("resid_lag24")
        ref = np.full(len(y), np.nan)
        ref[int(S2[0]):] = (y - yhat)[int(S2[0]):]
        ref = pd.Series(ref).shift(24).to_numpy()
        chk = np.nanmax(np.abs(np.nan_to_num(ref[S4], nan=0.0) - Z[S4][:, j24]))

        # ---- (b)(c) decomposition ---------------------------------------
        mae = lambda p: float(np.abs(p - y4).mean())
        r0 = mae(yhat[S4])

        m1 = lgbm(); m1.fit(X[S2], y[S2])
        r1 = mae(m1.predict(X[S4]))

        m2 = lgbm(); m2.fit(Z[S2], y[S2])
        r2 = mae(m2.predict(Z[S4]))

        # how much of Z's edge is the residual history alone?
        keep = [i for i, nm in enumerate(zn) if not nm.startswith("resid_")]
        m2b = lgbm(); m2b.fit(Z[S2][:, keep], y[S2])
        r2b = mae(m2b.predict(Z[S4][:, keep]))

        rows.append(dict(
            dataset=key, backbone=a.backbone,
            max_abs_corr_Z_vs_y=round(worst_r, 4), worst_col=worst_nm,
            resid_lag24_structural_err=float(chk),
            R0_frozen_backbone=round(r0, 3),
            R1_sameFeat_newData=round(r1, 3),
            R2_richFeat_newData=round(r2, 3),
            R2b_richFeat_noResidHist=round(r2b, 3),
            pct_from_recency=round(100 * (r0 - r1) / r0, 2),
            pct_from_features=round(100 * (r1 - r2) / r0, 2),
            pct_from_resid_history=round(100 * (r2b - r2) / r0, 2),
            pct_total=round(100 * (r0 - r2) / r0, 2),
        ))
        print(f"[{key}/{a.backbone}] maxcorr={worst_r:.4f} ({worst_nm})  "
              f"R0={r0:.2f} R1={r1:.2f} R2={r2:.2f} | "
              f"recency={100*(r0-r1)/r0:+.1f}% feats={100*(r1-r2)/r0:+.1f}%",
              flush=True)

    df = pd.DataFrame(rows)
    p = os.path.join(RESDIR, "bech_peer_gain_audit.csv")
    df.to_csv(p, index=False)
    print(df.to_string(index=False))
    print(f"[done] -> {p}")


if __name__ == "__main__":
    main()
