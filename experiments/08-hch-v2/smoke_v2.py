"""HCH v2 repair smoke — S3 calibration + unified S4 manifold."""
from __future__ import annotations

import sys, json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "07-route-e" / "peers"))

from common import load_dataset, load_shandong, build_tabular, assert_no_leakage, \
    weekly_naive
from backbones import make_backbone, needs_seq
from hch_v2_data import DailyEpisodeBatch, build_dataloaders
from hch_v2 import (
    HCHV2, HCHV2Config, compute_action_gain,
    candidate_loss_fn, state_loss_fn, compute_state_targets,
)
from eval_manifest import ExperimentManifest, evaluate_on_manifest
from baselines_v2 import Identity, ResidualL1, QuantileResidualLGBM
from official_adapters import DeltaAdapterLimited, PIRLimited

DEV = torch.device("cpu")


def _to_device(batch):
    return DailyEpisodeBatch(
        host_raw=batch.host_raw.to(DEV),
        host_model=batch.host_model.to(DEV),
        target_raw=batch.target_raw.to(DEV) if batch.target_raw is not None else None,
        target_model=batch.target_model.to(DEV) if batch.target_model is not None else None,
        exog_value=batch.exog_value.to(DEV),
        exog_type=batch.exog_type.to(DEV),
        exog_mask=batch.exog_mask.to(DEV),
        lag_context=batch.lag_context.to(DEV),
        time_feat=batch.time_feat.to(DEV),
        market_id=batch.market_id.to(DEV),
        target_id=batch.target_id.to(DEV),
        timestamps=batch.timestamps,
        date_ids=batch.date_ids,
    )


def _s1_stats(price_s1):
    y = np.sort(price_s1.ravel().astype(np.float64))
    def cdf(v):
        return float(np.searchsorted(y, v) / len(y))
    return {"cdf": cdf, "median": float(np.median(y)),
            "mad": float(np.median(np.abs(y - np.median(y))))}


def train_step(model, loader, cfg, s1_stats):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    best, best_state, patience = float("inf"), None, 0
    for ep in range(cfg.epochs):
        total = 0.0
        for batch in loader:
            batch = _to_device(batch)
            z, state = model.encode(batch)
            cand = model.biomc(z, state)
            cl, _ = candidate_loss_fn(cand, batch.target_model, batch.host_model, cfg)
            s_tgt = compute_state_targets(batch.target_model, s1_stats["cdf"],
                                          s1_stats["median"], s1_stats["mad"]).to(DEV)
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


def build_memory_s3(model, s3_loader, s1_stats):
    model.eval()
    keys, gains, dates = [], [], []
    with torch.no_grad():
        for batch in s3_loader:
            batch = _to_device(batch)
            z, state = model.encode(batch)
            cand = model.biomc(z, state)
            dd = cand["delta_down"].squeeze(-1)
            du = cand["delta_up"].squeeze(-1)
            yh = batch.host_model.squeeze(-1)
            yt = batch.target_model.squeeze(-1)
            gain = compute_action_gain(yt, yh, yh + dd, yh + du)
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
    assert_no_leakage(ds, X, y, valid, names)

    # -- unified date-first manifest (§3.2 addendum) --
    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id=ds_key)
    s1_indices = exp.valid_indices_in_split("S1")

    bb = make_backbone(bb_name, seed=seed)
    if needs_seq(bb_name):
        from common import build_sequences
        sq = build_sequences(ds, valid)
        bb.fit(X[s1_indices], y[s1_indices], sq[s1_indices])
        yhat = bb.predict(X, sq)
    else:
        bb.fit(X[s1_indices], y[s1_indices])
        yhat = bb.predict(X)

    yhat_full = np.full(len(y_full), np.nan, dtype=np.float32)
    yhat_full[valid] = yhat.astype(np.float32)

    spike_thr = float(np.quantile(y[s1_indices], 0.99))
    neg_thr = 0.0
    s1_stats = _s1_stats(y[s1_indices])
    loaders = build_dataloaders(ds_key, yhat_full, ds, batch_size=32,
                                exp_manifest=exp)
    p_mean, p_std = loaders["price_mean"], loaders["price_std"]

    # -- unified S4 manifest from same ExperimentManifest --
    manifest = exp.build_s4_eval_manifest(yhat_full)

    # subset y_full and yhat to manifest valid_indices
    y_s4_truth = y_full[manifest.valid_indices]
    y_s4_host = yhat_full[manifest.valid_indices]
    naive_s4 = y_full[np.clip(manifest.valid_indices - 168, 0, len(y_full) - 1)]

    results = []

    # ---- Identity ----
    results.append({"method": "Identity",
                    **evaluate_on_manifest(y_s4_truth, y_s4_host, manifest, neg_thr, spike_thr),
                    "touch_rate": 0.0, "harm_rate": 0.0})

    # ---- HCH v2 with calibration ----
    cfg = HCHV2Config(d_model=48, epochs=5, patience=3, lr=1e-3,
                      state_loss_weight=0.5)
    model = HCHV2(cfg).to(DEV)
    train_step(model, loaders["S2"], cfg, s1_stats)

    # S3: build temp memory + calibrate k/eta/tau (§F2)
    build_memory_s3(model, loaders["S3"], s1_stats)
    best_cfg = model.calibrate_s3(loaders["S3"], s1_stats)
    model.built = True

    bundle = model.freeze(s1_stats=s1_stats)
    del model

    mdl = HCHV2.from_bundle(bundle)
    mdl.built = True
    mdl.eval()

    all_yf, all_act = [], []
    with torch.no_grad():
        for batch in loaders["S4"]:
            batch = _to_device(batch)
            out = mdl(batch)
            all_yf.append(out["y_final"].cpu().numpy().ravel())
            if out.get("chosen_action") is not None:
                all_act.append(out["chosen_action"].cpu().numpy().ravel())

    hch_pred_z = np.concatenate(all_yf)
    hch_pred = hch_pred_z * p_std + p_mean

    n_manifest = manifest.n_hours
    if len(hch_pred) != n_manifest:
        raise RuntimeError(
            f"S4 prediction count {len(hch_pred)} != manifest {n_manifest}. "
            f"Method output must be keyed, not trim-padded."
        )

    touch = float((np.abs(hch_pred - y_s4_host) > 1e-6).mean())
    harm = float((np.abs(hch_pred - y_s4_truth) > np.abs(y_s4_host - y_s4_truth)).mean())

    hch_r = {"method": "HCHv2",
             **evaluate_on_manifest(y_s4_truth, hch_pred, manifest, neg_thr, spike_thr),
             "touch_rate": touch, "harm_rate": harm,
             "bundle_hash": bundle.hash(),
             "mem_entries": len(mdl.memory.m_keys) if mdl.memory.m_keys is not None else 0}

    if all_act:
        acts = np.concatenate(all_act)
        hch_r["action_id"] = float((acts == 0).mean())
        hch_r["action_down"] = float((acts == 1).mean())
        hch_r["action_up"] = float((acts == 2).mean())
    hch_r["calibration"] = {"k": best_cfg[0], "eta": best_cfg[1],
                             "tau": best_cfg[2]} if best_cfg else None
    results.append(hch_r)

    id_mae = results[0]["mae"]
    return {"dataset": ds_key, "backbone": bb_name, "seed": seed,
            "n_S4_manifest": n_manifest,
            "spike_thr": spike_thr, "neg_thr": neg_thr,
            "methods": [{**r, "delta_mae": round(r["mae"] - id_mae, 4)}
                        for r in results]}


def main():
    print("=== HCH v2 Calibrated Smoke ===\n")
    all_r = []
    for ds in ["NEM_SA1", "LAGO_DE"]:
        for bb in ["Linear", "PatchTST"]:
            print(f"{ds} x {bb} ", end="", flush=True)
            try:
                r = run_one(ds, bb)
                all_r.append(r)
                for m in r["methods"]:
                    extra = ""
                    if "calibration" in m and m["calibration"]:
                        extra = f' cal=({m["calibration"]["k"]},{m["calibration"]["eta"]},{m["calibration"]["tau"]})'
                    if "bundle_hash" in m:
                        extra += f' hash={m["bundle_hash"]}'
                    act = ""
                    if "action_id" in m:
                        act = f" act=({m['action_id']:.0%}I/{m.get('action_down',0):.0%}D/{m.get('action_up',0):.0%}U)"
                    print(f"  {m['method']:10s} MAE={m['mae']:.2f} d={m['delta_mae']:+.3f} "
                          f"n={m['n']} touch={m.get('touch_rate',0):.1%}{act}{extra}")
            except Exception as e:
                import traceback
                print(f"FAIL: {e}\n{traceback.format_exc()}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).resolve().parent / "results" / f"v2_smoke_calibrated_{ts}.json"
    with open(out, "w") as f:
        json.dump(all_r, f, indent=2, default=str, ensure_ascii=False)
    n_methods = sum(len(r["methods"]) for r in all_r)
    print(f"\n{len(all_r)} combos, {n_methods} evals -> {out}")


if __name__ == "__main__":
    main()
