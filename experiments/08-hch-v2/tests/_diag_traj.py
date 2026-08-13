"""Diagnostic 2: full-S2 training trajectory — when/why do m- / m+ collapse?

Tracks per epoch:
  - max / frac>0 of m- and m+ (post-ReLU)
  - fraction of hours with raw r- / r+ pre-activation > 0  (ReLU alive rate)
  - whether the best_state restore lands on a collapsed point
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "08-hch-v2"))

from common import load_dataset, build_tabular, assert_no_leakage
from backbones import make_backbone
from eval_manifest import ExperimentManifest
from hch_v2_pipeline import HCHV2UniversalPipeline
from smoke_v4 import precompute_scale_free, build_core_context, _scale_z0, pd_date


def main():
    torch.manual_seed(0); np.random.seed(0)
    init_bias = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
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

    s1_z0, s1_hours = [], []
    for d in exp.dates_in_split("S1"):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        _, z0d = _scale_z0(yhat_full[idxs].astype(np.float64))
        s1_z0.append(z0d); s1_hours.append(ts.iloc[idxs].dt.hour.values)
    s1_z0 = np.concatenate(s1_z0); s1_hours = np.concatenate(s1_hours)

    pipe = HCHV2UniversalPipeline(d_core_context=13, d_model=32, alpha=0.10, seed=0)
    if init_bias > 0:
        with torch.no_grad():
            pipe.candidate_head.shift_head.bias.data.fill_(init_bias)
            pipe.candidate_head.shift_head.weight.data.mul_(0.05)
        print(f"[init_bias={init_bias} applied to shift_head]")
    pipe.fit_s1_reference(s1_z0, s1_hours)
    pipe.fit_s1_signature(s1_z0, s1_hours)

    s2_batches = []
    for d in sorted(exp.dates_in_split("S2")):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        host_day = yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            continue
        hours = ts.iloc[idxs].dt.hour.values
        ctx = build_core_context(host_day, hours, pipe, z0_full, s_full, y_full, idxs)
        s2_batches.append((
            torch.tensor(host_day.reshape(1, 24, 1), dtype=torch.float32),
            torch.tensor(ctx.reshape(1, 24, -1), dtype=torch.float32),
            torch.tensor(y_full[idxs].astype(np.float64).reshape(1, 24, 1), dtype=torch.float32),
            torch.ones(1, 24),
        ))
    print(f"n_S2_days={len(s2_batches)}")

    from iah_crps_loss import iah_crps_loss
    opt = torch.optim.AdamW(pipe.candidate_head.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, pat = float("inf"), None, 0

    def stats():
        with torch.no_grad():
            mi = np.concatenate([pipe.candidate_head(b[0], b[1], valid_mask=b[3])["m_minus"].cpu().numpy().ravel()
                                 for b in s2_batches])
            pl = np.concatenate([pipe.candidate_head(b[0], b[1], valid_mask=b[3])["m_plus"].cpu().numpy().ravel()
                                 for b in s2_batches])
            # pre-activations
            raws = []
            for b in s2_batches:
                o = pipe.candidate_head(b[0], b[1], valid_mask=b[3])
                h_core = pipe.candidate_head.core_encoder(
                    torch.cat([o["z0"].unsqueeze(-1), b[1]], dim=-1))
                raws.append(pipe.candidate_head.shift_head(h_core).cpu().numpy().reshape(-1, 2))
            raws = np.concatenate(raws)
        return mi, pl, raws

    # init stats
    mi, pl, raws = stats()
    print(f"init: m- max={mi.max():.4f} frac>0={np.mean(mi>0):.3f} | "
          f"m+ max={pl.max():.4f} frac>0={np.mean(pl>0):.3f} | "
          f"r- alive={np.mean(raws[:,0]>0):.3f} r+ alive={np.mean(raws[:,1]>0):.3f}")

    for ep in range(8):
        for host, ctx, target, vm in s2_batches:
            opt.zero_grad()
            out = pipe.candidate_head(host, ctx, valid_mask=vm)
            loss = iah_crps_loss(out, target)
            loss.backward()
            nn.utils.clip_grad_norm_(pipe.candidate_head.parameters(), 1.0)
            opt.step()
        mi, pl, raws = stats()
        print(f"ep{ep}: loss={loss.item():.4f} | "
              f"m- max={mi.max():.4f} frac>0={np.mean(mi>0):.3f} | "
              f"m+ max={pl.max():.4f} frac>0={np.mean(pl>0):.3f} | "
              f"r- alive={np.mean(raws[:,0]>0):.3f} r+ alive={np.mean(raws[:,1]>0):.3f}")
        avg = float(loss)
        if avg < best - 1e-5:
            best, pat = avg, 0
            best_state = {k: v.clone() for k, v in pipe.candidate_head.state_dict().items()}
        else:
            pat += 1
            if pat >= 4:
                print(f"  early stop at ep{ep}; best={best:.4f}")
                break
    # restore best and re-stats
    if best_state is not None:
        pipe.candidate_head.load_state_dict(best_state)
        mi, pl, raws = stats()
        print(f"BEST restored (loss {best:.4f}): m- max={mi.max():.4f} frac>0={np.mean(mi>0):.3f} | "
              f"m+ max={pl.max():.4f} frac>0={np.mean(pl>0):.3f} | "
              f"r- alive={np.mean(raws[:,0]>0):.3f} r+ alive={np.mean(raws[:,1]>0):.3f}")


if __name__ == "__main__":
    main()
