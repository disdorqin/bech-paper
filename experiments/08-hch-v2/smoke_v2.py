"""HCH v2 smoke test — 3 scenarios with all baselines.

Usage: python smoke_v2.py --dataset NEM_SA1 --backbone Linear
       python smoke_v2.py --dataset all --backbone all  (3x2=6 combos)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import load_dataset, load_shandong, build_tabular, assert_no_leakage, \
    four_segment_split, evaluate, weekly_naive
from backbones import make_backbone, needs_seq
from hch_v2_data import DailyEpisodeDataset, DailyEpisodeBatch, build_dataloaders
from hch_v2 import HCHV2, HCHV2Config
from baselines_v2 import Identity, ResidualL1, QuantileResidualLGBM
from selective_hurdle import build_corrector_features

EPS = 1e-8
DEVICE = torch.device("cpu")
SMOKE_DATASETS = ["NEM_SA1", "LAGO_DE", "shandong_DA"]
SMOKE_BACKBONES = ["Linear", "MLP"]


def _w1_cdf(p, q):
    cp = torch.cumsum(p, dim=-1)
    cq = torch.cumsum(q, dim=-1)
    return (cp - cq).abs().sum(dim=-1)


def candidate_loss_fn(cand, target, host_pred, cfg):
    resid = target - host_pred
    r_d = F.relu(-resid).squeeze(-1)
    r_u = F.relu(resid).squeeze(-1)
    o_d = (resid < 0).float().squeeze(-1)
    o_u = (resid > 0).float().squeeze(-1)

    loss_occ = F.binary_cross_entropy(cand["p_down"].squeeze(-1), o_d) \
               + F.binary_cross_entropy(cand["p_up"].squeeze(-1), o_u)

    loss_mag = torch.tensor(0.0)
    n = 0
    for om, mp, mt in [(o_d > 0.5, cand["m_down"].squeeze(-1), r_d),
                       (o_u > 0.5, cand["m_up"].squeeze(-1), r_u)]:
        if om.sum() > 0:
            loss_mag = loss_mag + F.smooth_l1_loss(mp * om.float(), mt * om.float(),
                                                    reduction="sum") / om.sum().clamp(min=1)
            n += 1
    if n:
        loss_mag = loss_mag / n

    loss_loc = torch.tensor(0.0)
    for r_raw, d_raw in [(r_d, -cand["delta_down"].squeeze(-1)),
                         (r_u, cand["delta_up"].squeeze(-1))]:
        rs = r_raw.sum(-1).clamp(min=EPS)
        mask = (rs > EPS).float()
        if mask.sum() == 0:
            continue
        ds = d_raw.abs().sum(-1).clamp(min=EPS)
        loss_loc = loss_loc + (_w1_cdf(r_raw / rs.unsqueeze(-1),
                                        d_raw.abs() / ds.unsqueeze(-1))
                               * mask).sum() / mask.sum().clamp(min=1)
    return (cfg.occurrence_loss_weight * loss_occ
            + cfg.magnitude_loss_weight * loss_mag
            + cfg.location_loss_weight * loss_loc)


def train_hchv2(model, s2_loader, config):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    best, best_state, patience = float("inf"), None, 0
    for ep in range(config.epochs):
        total = 0.0
        for batch in s2_loader:
            batch = DailyEpisodeBatch(
                host_pred=batch.host_pred.to(DEVICE),
                target=batch.target.to(DEVICE),
                exog=batch.exog.to(DEVICE), exog_mask=batch.exog_mask.to(DEVICE),
                time_feat=batch.time_feat.to(DEVICE), date_ids=batch.date_ids,
            )
            z, sr, sz = model.encode(batch)
            cand = model.biomc(z)
            loss = candidate_loss_fn(cand, batch.target, batch.host_pred, config)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        avg = total / max(len(s2_loader), 1)
        if avg < best - 1e-5:
            best, patience = avg, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return best


def build_hchv2_memory(model, s3_loader):
    model.eval()
    keys, gains_list, dates_list = [], [], []
    with torch.no_grad():
        for batch in s3_loader:
            batch = DailyEpisodeBatch(
                host_pred=batch.host_pred.to(DEVICE),
                target=batch.target.to(DEVICE),
                exog=batch.exog.to(DEVICE), exog_mask=batch.exog_mask.to(DEVICE),
                time_feat=batch.time_feat.to(DEVICE), date_ids=batch.date_ids,
            )
            z, sr, sz = model.encode(batch)
            cand = model.biomc(z)
            k = model.memory.encode_key(z, cand["delta_down"].squeeze(-1),
                                        cand["delta_up"].squeeze(-1))
            base_ae = (batch.target.squeeze(-1) - batch.host_pred.squeeze(-1)).abs()
            down_ae = (batch.target.squeeze(-1) - cand["y_down"].squeeze(-1)).abs()
            up_ae = (batch.target.squeeze(-1) - cand["y_up"].squeeze(-1)).abs()
            g = torch.stack([torch.zeros_like(base_ae), base_ae - down_ae,
                             base_ae - up_ae], dim=-1)
            keys.append(k.cpu())
            gains_list.append(g.cpu())
            if isinstance(batch.date_ids, list):
                dates_list.extend(batch.date_ids)
    if keys:
        model.memory.build(torch.cat(keys), torch.cat(gains_list), dates_list)
        model.built = True


def eval_method(name, pred, y_true, naive, yhat_base, neg_thr, spike_thr):
    ae = np.abs(pred - y_true)
    out = {
        "method": name,
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt((ae ** 2).mean())),
        "rmae": float(ae.mean() / np.abs(naive - y_true).mean())
        if naive is not None else None,
        "neg_n": int((y_true < neg_thr).sum()),
        "spike_n": int((y_true > spike_thr).sum()) if spike_thr else 0,
        "touch_rate": float((np.abs(pred - yhat_base) > 1e-9).mean()),
    }
    neg = y_true < neg_thr
    out["mae_on_neg"] = float(ae[neg].mean()) if neg.sum() else None
    out["neg_miss_rate"] = float((pred[neg] >= neg_thr).mean()) if neg.sum() else None
    if spike_thr:
        sp = y_true > spike_thr
        out["mae_on_spike"] = float(ae[sp].mean()) if sp.sum() else None
        out["spike_miss_rate"] = float((pred[sp] <= spike_thr).mean()) if sp.sum() else None
        normal = (~neg) & (~sp)
    else:
        out["mae_on_spike"] = out["spike_miss_rate"] = None
        normal = ~neg
    out["mae_on_normal"] = float(ae[normal].mean()) if normal.sum() else None
    out["harm_rate"] = float((ae > np.abs(yhat_base - y_true)).mean())
    return out


def run_one_scenario(ds_key, bb_name, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = load_dataset(ds_key) if ds_key != "shandong_DA" else \
        load_shandong(price_col="日前电价", encoding="gbk")
    y_full = ds["price"]
    X, y, names, valid = build_tabular(ds)
    n = len(valid)
    seg = four_segment_split(n)
    assert_no_leakage(ds, X, y, valid, names)

    bb = make_backbone(bb_name, seed=seed)
    if needs_seq(bb_name):
        sq = build_sequences(ds, valid)
        bb.fit(X[seg["S1"]], y[seg["S1"]], sq[seg["S1"]])
        yhat = bb.predict(X, sq)
    else:
        bb.fit(X[seg["S1"]], y[seg["S1"]])
        yhat = bb.predict(X)
    yhat_full = np.full(len(y_full), np.nan, dtype=np.float32)
    yhat_full[valid] = yhat.astype(np.float32)

    spike_thr = float(np.quantile(y[seg["S1"]], 0.99))
    neg_thr = 0.0

    loaders = build_dataloaders(ds_key, yhat_full, ds, batch_size=32)

    # build corrector features Z from tabular X, aligned with yhat (len=valid)
    hour = ds["ts"].dt.hour.to_numpy()
    dayid = (ds["ts"] - ds["ts"].min()).dt.days.to_numpy()
    oos = seg["S1"][-1] + 1
    Z_all, z_names = build_corrector_features(
        X, names, yhat, y, hour[valid], dayid[valid], oos,
    )
    Z_s2 = Z_all[seg["S2"]]
    Z_s3 = Z_all[seg["S3"]]
    Z_s4 = Z_all[seg["S4"]]

    yhat_s2 = yhat[seg["S2"]]
    y_s2 = y[seg["S2"]]
    yhat_s3 = yhat[seg["S3"]]
    y_s3 = y[seg["S3"]]
    yhat_s4 = yhat[seg["S4"]]
    y_true_flat = y[seg["S4"]]

    naive = weekly_naive(y_full, valid, seg["S4"])

    # --- run all methods ---
    results = []
    cfg = HCHV2Config(d_model=64, epochs=40, patience=10, lr=1e-3,
                      cara_eta=0.1, kl_tau=0.2, memory_temperature=0.05,
                      occurrence_loss_weight=1.5, magnitude_loss_weight=1.5,
                      location_loss_weight=0.5)

    # Identity
    id_pred = yhat_s4.copy()
    results.append(eval_method("Identity", id_pred, y_true_flat, naive,
                               yhat_s4, neg_thr, spike_thr))

    # ResidualL1
    rl1 = ResidualL1(seed=seed)
    rl1.fit(Z_s2, yhat_s2, y_s2)
    rl1_pred = rl1.predict(Z_s4, yhat_s4)
    results.append(eval_method("ResidualL1", rl1_pred, y_true_flat, naive,
                               yhat_s4, neg_thr, spike_thr))

    # QuantileResidualLGBM
    qr = QuantileResidualLGBM(seed=seed)
    qr.fit(Z_s2, yhat_s2, y_s2)
    qr.calibrate(Z_s3, yhat_s3, y_s3)
    qr_pred = qr.predict(Z_s4, yhat_s4)
    results.append(eval_method("QuantileResidualLGBM", qr_pred, y_true_flat,
                               naive, yhat_s4, neg_thr, spike_thr))

    # --- HCH v2 ---
    p_mean = loaders["price_mean"]
    p_std = loaders["price_std"]

    model = HCHV2(cfg).to(DEVICE)
    train_hchv2(model, loaders["S2"], cfg)
    build_hchv2_memory(model, loaders["S3"])

    model.eval()
    hch_all_v2, hch_all_tgt, hch_all_base, hch_all_act = [], [], [], []
    with torch.no_grad():
        for batch in loaders["S4"]:
            batch = DailyEpisodeBatch(
                host_pred=batch.host_pred.to(DEVICE),
                target=batch.target.to(DEVICE),
                exog=batch.exog.to(DEVICE), exog_mask=batch.exog_mask.to(DEVICE),
                time_feat=batch.time_feat.to(DEVICE), date_ids=batch.date_ids,
            )
            out = model(batch)
            hch_all_v2.append(out["y_final"].cpu().numpy().ravel())
            hch_all_tgt.append(batch.target.cpu().numpy().ravel())
            hch_all_base.append(out["y_base"].cpu().numpy().ravel())
            if "chosen_action" in out and out["chosen_action"] is not None:
                hch_all_act.append(out["chosen_action"].cpu().numpy().ravel())

    hch_pred_z = np.concatenate(hch_all_v2)
    hch_tgt_z = np.concatenate(hch_all_tgt)
    hch_base_z = np.concatenate(hch_all_base)

    hch_pred_raw = hch_pred_z * p_std + p_mean
    hch_tgt_raw = hch_tgt_z * p_std + p_mean
    hch_base_raw = hch_base_z * p_std + p_mean

    hch_mae_base = float(np.abs(hch_base_raw - hch_tgt_raw).mean())
    hch_mae = float(np.abs(hch_pred_raw - hch_tgt_raw).mean())
    hch_touch = float((np.abs(hch_pred_raw - hch_base_raw) > 1e-6).mean())
    hch_harm = float((np.abs(hch_pred_raw - hch_tgt_raw) > np.abs(hch_base_raw - hch_tgt_raw)).mean())

    hch_info = {
        "method": "HCHv2",
        "mae": hch_mae,
        "delta_mae": round(hch_mae - hch_mae_base, 6),
        "touch_rate": hch_touch,
        "harm_rate": hch_harm,
        "memory_entries": len(model.memory.m_keys) if model.memory.m_keys is not None else 0,
    }
    if hch_all_act:
        acts = np.concatenate(hch_all_act)
        hch_info["action_identity"] = float((acts == 0).mean())
        hch_info["action_down"] = float((acts == 1).mean())
        hch_info["action_up"] = float((acts == 2).mean())
    results.append(hch_info)

    # oracle gap — baselines vs Identity
    id_mae = results[0]["mae"]
    for r in results:
        r["delta_mae"] = round(r["mae"] - id_mae, 6) if id_mae else None

    return {
        "dataset": ds_key, "backbone": bb_name, "seed": seed,
        "spike_thr": spike_thr, "neg_thr": neg_thr,
        "n_S1": loaders["S1_n_days"], "n_S2": loaders["S2_n_days"],
        "n_S3": loaders["S3_n_days"], "n_S4": loaders["S4_n_days"],
        "methods": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NEM_SA1")
    ap.add_argument("--backbone", default="Linear")
    args = ap.parse_args()

    ds_list = SMOKE_DATASETS if args.dataset == "all" else [args.dataset]
    bb_list = SMOKE_BACKBONES if args.backbone == "all" else [args.backbone]

    all_results = []
    for ds in ds_list:
        for bb in bb_list:
            print(f"\n=== {ds} x {bb} ===")
            r = run_one_scenario(ds, bb)
            all_results.append(r)
            for m in r["methods"]:
                act = ""
                if "action_identity" in m:
                    act = f" act=({m['action_identity']:.0%}I/{m.get('action_down',0):.0%}D/{m.get('action_up',0):.0%}U)"
                if m["method"] == "HCHv2":
                    print(f"  {m['method']:22s} MAE={m['mae']:.4f} Δ={m.get('delta_mae',0):+.4f} "
                          f"touch={m['touch_rate']:.1%} harm={m['harm_rate']:.1%} "
                          f"mem={m.get('memory_entries',0)}{act}")
                else:
                    print(f"  {m['method']:22s} MAE={m['mae']:.4f} Δ={m.get('delta_mae',0):+.4f} "
                          f"neg_n={m['neg_n']} neg_mae={m.get('mae_on_neg','?'):.4f} "
                          f"touch={m['touch_rate']:.1%} harm={m['harm_rate']:.1%}")

    n_ok = sum(1 for r in all_results for m in r["methods"])
    print(f"\nTotal: {len(all_results)} combos, {n_ok} method evals")


if __name__ == "__main__":
    main()
