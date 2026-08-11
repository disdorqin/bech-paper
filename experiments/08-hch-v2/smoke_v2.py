"""HCH v2 repair smoke — micro end-to-end with new API (§12.2)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "07-route-e" / "peers"))

from common import load_dataset, load_shandong, build_tabular, assert_no_leakage, \
    four_segment_split, weekly_naive
from backbones import make_backbone, needs_seq
from hch_v2_data import DailyEpisodeBatch, build_dataloaders
from hch_v2 import HCHV2, HCHV2Config, compute_action_gain, candidate_loss_fn, \
    state_loss_fn, compute_state_targets

DEV = torch.device("cpu")


def _s1_stats(price_s1):
    y = np.sort(price_s1.ravel().astype(np.float64))
    def cdf(v):
        return float(np.searchsorted(y, v) / len(y))
    return {
        "cdf": cdf,
        "median": float(np.median(y)),
        "mad": float(np.median(np.abs(y - np.median(y)))),
    }


def train_step(model, loader, cfg, s1_stats):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    best, best_state, patience = float("inf"), None, 0
    for ep in range(cfg.epochs):
        total = 0.0
        for batch in loader:
            batch = DailyEpisodeBatch(
                host_pred=batch.host_pred.to(DEV),
                target=batch.target.to(DEV),
                exog=batch.exog.to(DEV),
                exog_mask=batch.exog_mask.to(DEV),
                time_feat=batch.time_feat.to(DEV),
                date_ids=batch.date_ids,
            )
            z, state = model.encode(batch)
            cand = model.biomc(z, state)
            cl, _ = candidate_loss_fn(cand, batch.target, batch.host_pred, cfg)
            s_tgt = compute_state_targets(batch.target, s1_stats["cdf"],
                                          s1_stats["median"], s1_stats["mad"])
            s_tgt = s_tgt.to(DEV)
            sl = state_loss_fn(state, s_tgt)
            loss = cl + cfg.state_loss_weight * sl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        avg = total / max(len(loader), 1)
        if avg < best - 1e-5:
            best, patience = avg, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return best


def build_memory(model, s3_loader, s1_stats):
    model.eval()
    keys, gains, dates = [], [], []
    with torch.no_grad():
        for batch in s3_loader:
            batch = DailyEpisodeBatch(
                host_pred=batch.host_pred.to(DEV), target=batch.target.to(DEV),
                exog=batch.exog.to(DEV), exog_mask=batch.exog_mask.to(DEV),
                time_feat=batch.time_feat.to(DEV), date_ids=batch.date_ids,
            )
            z, state = model.encode(batch)
            cand = model.biomc(z, state)
            dd = cand["delta_down"].squeeze(-1)
            du = cand["delta_up"].squeeze(-1)
            yh = batch.host_pred.squeeze(-1)
            yt = batch.target.squeeze(-1)
            yd = yh + dd
            yu = yh + du
            gain = compute_action_gain(yt, yh, yd, yu)
            k = model.memory.encode_key(z, state, dd, du)
            keys.append(k.cpu())
            gains.append(gain.cpu())
            if isinstance(batch.date_ids, list):
                dates.extend(batch.date_ids)
    if keys:
        model.memory.build(torch.cat(keys), torch.cat(gains), dates)
        model.built = True


def run_one(ds_key, bb_name, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = load_dataset(ds_key) if ds_key != "shandong_DA" else \
        load_shandong(price_col="日前电价", encoding="gbk")
    y_full = ds["price"]
    X, y, names, valid = build_tabular(ds)
    seg = four_segment_split(len(valid))
    assert_no_leakage(ds, X, y, valid, names)

    bb = make_backbone(bb_name, seed=seed)
    if needs_seq(bb_name):
        from common import build_sequences
        sq = build_sequences(ds, valid)
        bb.fit(X[seg["S1"]], y[seg["S1"]], sq[seg["S1"]])
        yhat = bb.predict(X, sq)
    else:
        bb.fit(X[seg["S1"]], y[seg["S1"]])
        yhat = bb.predict(X)

    yhat_full = np.full(len(y_full), np.nan, dtype=np.float32)
    yhat_full[valid] = yhat.astype(np.float32)

    s1_stats = _s1_stats(y[seg["S1"]])
    loaders = build_dataloaders(ds_key, yhat_full, ds, batch_size=32)
    p_mean, p_std = loaders["price_mean"], loaders["price_std"]

    cfg = HCHV2Config(d_model=48, epochs=5, patience=3, lr=1e-3,
                      state_loss_weight=0.5, cara_eta=0.1, kl_tau=0.2)

    model = HCHV2(cfg).to(DEV)

    # train on S2
    train_step(model, loaders["S2"], cfg, s1_stats)

    # state gradient check (C4)
    state_grad = 0.0
    for name, p in model.named_parameters():
        if "state_head" in name and p.grad is not None:
            state_grad += p.grad.norm().item()

    # build memory on S3
    build_memory(model, loaders["S3"], s1_stats)

    # freeze
    bundle = model.freeze(s1_stats=s1_stats)
    bundle_hash = bundle.hash()

    # reload from bundle
    del model
    model2 = HCHV2.from_bundle(bundle)
    model2.built = True
    model2.eval()

    # predict S4
    all_yf, all_yt, all_yb, all_act, all_state = [], [], [], [], []
    with torch.no_grad():
        for batch in loaders["S4"]:
            batch = DailyEpisodeBatch(
                host_pred=batch.host_pred.to(DEV), target=batch.target.to(DEV),
                exog=batch.exog.to(DEV), exog_mask=batch.exog_mask.to(DEV),
                time_feat=batch.time_feat.to(DEV), date_ids=batch.date_ids,
            )
            out = model2(batch)
            all_yf.append(out["y_final"].cpu().numpy().ravel())
            all_yt.append(batch.target.cpu().numpy().ravel())
            all_yb.append(out["y_base"].cpu().numpy().ravel())
            all_state.append(out["state"].cpu().numpy())
            if out.get("chosen_action") is not None:
                all_act.append(out["chosen_action"].cpu().numpy().ravel())

    yf = np.concatenate(all_yf) * p_std + p_mean
    yt = np.concatenate(all_yt) * p_std + p_mean
    yb = np.concatenate(all_yb) * p_std + p_mean
    act = np.concatenate(all_act) if all_act else np.array([])

    mae_base = float(np.abs(yb - yt).mean())
    mae_hch = float(np.abs(yf - yt).mean())
    touch = float((np.abs(yf - yb) > 1e-6).mean())
    harm = float((np.abs(yf - yt) > np.abs(yb - yt)).mean())

    return {
        "ds": ds_key, "bb": bb_name, "bundle_hash": bundle_hash,
        "mae_base": mae_base, "mae_hch": mae_hch, "delta": mae_hch - mae_base,
        "touch": touch, "harm": harm,
        "state_grad": state_grad,
        "action_id": float((act == 0).mean()) if len(act) else 1.0,
        "action_down": float((act == 1).mean()) if len(act) else 0.0,
        "action_up": float((act == 2).mean()) if len(act) else 0.0,
        "mem_entries": len(model2.memory.m_keys) if model2.memory.m_keys is not None else 0,
    }


def main():
    print("=== HCH v2 Repair Micro Smoke ===\n")
    for ds in ["NEM_SA1", "LAGO_DE"]:
        for bb in ["Linear", "PatchTST"]:
            print(f"{ds} x {bb} ", end="", flush=True)
            try:
                r = run_one(ds, bb)
                print(f"MAE={r['mae_hch']:.2f} d={r['delta']:+.3f} "
                      f"touch={r['touch']:.1%} harm={r['harm']:.1%} "
                      f"act=({r['action_id']:.0%}I/{r['action_down']:.0%}D/{r['action_up']:.0%}U) "
                      f"sg={r['state_grad']:.4f} hash={r['bundle_hash']}")
            except Exception as e:
                print(f"FAIL: {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
