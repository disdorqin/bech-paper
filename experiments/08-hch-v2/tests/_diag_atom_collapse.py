"""Diagnostic: why do m_minus/m_plus collapse to exactly 0 in the v0.4 smoke?

Questions answered on REAL LAGO_DE S2 data:
  Q1. At m=0, does the CRPS loss push m away from 0?
      dL/dm- = w- [sign(zY-z0) - (1-w-)]
      dL/dm+ = w+ [-sign(zY-z0) - (1-w+)]
      negative gradient => loss decreases by increasing m => loss wants displacement.
  Q2. Through ReLU(r), can the network actually move m at r<0? (dead-ReLU check)
  Q3. During short training, does m ever leave 0?
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "08-hch-v2"))

import pandas as pd
from common import load_dataset, build_tabular, assert_no_leakage
from backbones import make_backbone
from eval_manifest import ExperimentManifest
from hch_v2_pipeline import HCHV2UniversalPipeline
from smoke_v4 import precompute_scale_free, build_core_context, _scale_z0, pd_date


def main():
    torch.manual_seed(0); np.random.seed(0)
    ds = load_dataset("LAGO_DE")
    y_full = ds["price"].astype(np.float32)
    ts = ds["ts"]
    X, y, names, valid = build_tabular(ds)
    assert_no_leakage(ds, X, y, valid, names)

    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id="LAGO_DE", s3m_frac=0.5)
    bb = make_backbone("Linear", seed=0)
    bb.fit(X[exp.valid_row_in_split("S1")], y[exp.valid_row_in_split("S1")])
    yhat_valid = bb.predict(X).astype(np.float32)
    yhat_full = np.full(len(y_full), np.nan, dtype=np.float32)
    yhat_full[valid] = yhat_valid

    z0_full, s_full = precompute_scale_free(yhat_full, ts)

    # ---- assemble S1 signature + S2 day list exactly like smoke_v4 ----
    s1_z0, s1_hours = [], []
    for d in exp.dates_in_split("S1"):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        _, z0d = _scale_z0(yhat_full[idxs].astype(np.float64))
        s1_z0.append(z0d); s1_hours.append(ts.iloc[idxs].dt.hour.values)
    s1_z0 = np.concatenate(s1_z0); s1_hours = np.concatenate(s1_hours)

    pipe = HCHV2UniversalPipeline(d_core_context=13, d_model=32, alpha=0.10, seed=0)
    pipe.fit_s1_reference(s1_z0, s1_hours)
    pipe.fit_s1_signature(s1_z0, s1_hours)

    days = []
    for d in sorted(exp.dates_in_split("S2")):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        host_day = yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            continue
        hours = ts.iloc[idxs].dt.hour.values
        days.append((d, idxs, host_day, hours))

    # ---------------------------------------------------------- Q1 ----------
    print("Q1: loss gradient at m=0 on real S2 (per-day mean over hours)")
    grads_minus, grads_plus = [], []
    sign_means = []
    for (d, idxs, host_day, hours) in days:
        s = float(np.mean(np.abs(host_day)))
        z0 = np.arcsinh(host_day / s)
        zY = np.arcsinh(y_full[idxs].astype(np.float64) / s)
        r = zY - z0
        sign = np.sign(r)
        sign_means.append(sign.mean())
        # use a fixed plausible w to probe the geometry (candidate w is ~trainable;
        # the sign structure is what matters)
        for w_ in (0.3, 0.5):
            pass
        # use w- = w+ = 0.3 (equal-ish, close to observed ~0.47/0.23)
        wm = np.full(24, 0.47); wp = np.full(24, 0.23)
        dg_minus = wm * (sign - (1 - wm))     # negative => wants down atom to grow
        dg_plus = wp * (-sign - (1 - wp))     # negative => wants up atom to grow
        grads_minus.append(dg_minus.mean())
        grads_plus.append(dg_plus.mean())

    print(f"  n_S2_days={len(days)}")
    print(f"  mean sign(zY-z0) per day: {np.mean(sign_means):+.4f}  "
          f"(0 = symmetric residual)")
    print(f"  E[dL/dm- at m=0] = {np.mean(grads_minus):+.4f}   "
          f"(<0 => loss pushes down atom up)")
    print(f"  E[dL/dm+ at m=0] = {np.mean(grads_plus):+.4f}   "
          f"(<0 => loss pushes up atom up)")
    print(f"  frac days with dL/dm-<0: {np.mean([g<0 for g in grads_minus]):.3f}")
    print(f"  frac days with dL/dm+<0: {np.mean([g<0 for g in grads_plus]):.3f}")

    # ---------------------------------------------------------- Q2/Q3 ---------
    print("\nQ2/Q3: short training on a handful of S2 days; track m through ReLU")
    s2_batches = []
    for (d, idxs, host_day, hours) in days[:40]:
        ctx = build_core_context(host_day, hours, pipe, z0_full, s_full, y_full, idxs)
        s2_batches.append((
            torch.tensor(host_day.reshape(1, 24, 1), dtype=torch.float32),
            torch.tensor(ctx.reshape(1, 24, -1), dtype=torch.float32),
            torch.tensor(y_full[idxs].astype(np.float64).reshape(1, 24, 1), dtype=torch.float32),
            torch.ones(1, 24),
        ))
    opt = torch.optim.AdamW(pipe.candidate_head.parameters(), lr=1e-3, weight_decay=1e-4)
    # report initial r values
    with torch.no_grad():
        h0 = s2_batches[0][0]
        c0 = s2_batches[0][1]
        out0 = pipe.candidate_head(h0, c0)
        r_raw = out0["m_minus"]  # after ReLU, so this is max(r,0)
        print(f"  init: max m-={out0['m_minus'].max().item():.5f} "
              f"max m+={out0['m_plus'].max().item():.5f} "
              f"(post-ReLU)")
        # check pre-activation via manual head
        with torch.no_grad():
            h_core = pipe.candidate_head.core_encoder(
                torch.cat([out0["z0"].unsqueeze(-1), c0], dim=-1))
            l_raw = pipe.candidate_head.mass_head(h_core)
            r_raw = pipe.candidate_head.shift_head(h_core)
            print(f"  init raw r-: mean {r_raw[...,0].mean().item():+.4f} "
                  f"min {r_raw[...,0].min().item():+.4f} max {r_raw[...,0].max().item():+.4f}")
            print(f"  init raw r+: mean {r_raw[...,1].mean().item():+.4f} "
                  f"min {r_raw[...,1].min().item():+.4f} max {r_raw[...,1].max().item():+.4f}")

    for ep in range(8):
        for host, ctx, target, vm in s2_batches:
            opt.zero_grad()
            out = pipe.candidate_head(host, ctx, valid_mask=vm)
            from iah_crps_loss import iah_crps_loss
            loss = iah_crps_loss(out, target)
            loss.backward()
            opt.step()
        with torch.no_grad():
            mx_mi = mx_pl = 0.0
            for (h, c, t, v) in s2_batches:
                o = pipe.candidate_head(h, c, valid_mask=v)
                mx_mi = max(mx_mi, float(o["m_minus"].max()))
                mx_pl = max(mx_pl, float(o["m_plus"].max()))
            print(f"  epoch {ep}: loss={loss.item():.4f}  max m-={mx_mi:.6f}  max m+={mx_pl:.6f}")


if __name__ == "__main__":
    main()
