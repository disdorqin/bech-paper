"""Formal v0.4 real-data smoke — public price-only market + Linear backbone.

Per code-correction §7 + v0.4 architecture:
    S1 rank -> IAH core (S2) -> S3-M memory + k selection -> S3-C DVG
    -> freeze/from_bundle round-trip -> S4 target-free predict -> evidence JSON.

The decision chain (retrieval/replay/proposal/LCB) lives ONLY inside
HCHV2UniversalPipeline (predict_s4 / calibrate_s3c / select_s3m_k). This smoke
only prepares data and serializes the returned evidence — it never reimplements
the chain inline.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import load_dataset, build_tabular, assert_no_leakage
from backbones import make_backbone
from eval_manifest import ExperimentManifest
from hch_v2_pipeline import HCHV2UniversalPipeline

OUT = Path(__file__).resolve().parent / "results" / "v0.3"
OUT.mkdir(parents=True, exist_ok=True)


def pd_date(d):
    import pandas as pd
    return pd.Timestamp(d).date()


def _cyclic_time_features(hours: np.ndarray) -> np.ndarray:
    h = hours.astype(np.float32)
    return np.column_stack([
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * h / 168), np.cos(2 * np.pi * h / 168),
        np.sin(2 * np.pi * (h // 24) / 7), np.cos(2 * np.pi * (h // 24) / 7),
        (h % 24 < 6).astype(np.float32),
    ])


def _scale_z0(host_day):
    fin = np.isfinite(host_day)
    if fin.sum() == 0:
        return 0.0, np.zeros_like(host_day)
    s = float(np.mean(np.abs(host_day[fin])))
    if s <= 0:
        return 0.0, np.zeros_like(host_day)
    z0 = np.arcsinh(host_day / s)
    z0[~fin] = 0.0
    return s, z0


def precompute_scale_free(yhat_full, ts):
    """Per-hour z0 and day scale for the whole series (scale-free)."""
    import pandas as pd
    n = len(yhat_full)
    z0_full = np.full(n, np.nan, dtype=np.float64)
    s_full = np.full(n, np.nan, dtype=np.float64)
    for d in pd.unique(ts.dt.date):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        hd = yhat_full[idxs].astype(np.float64)
        fin = np.isfinite(hd)
        if fin.sum() == 0:
            continue
        s = float(np.mean(np.abs(hd[fin])))
        if s <= 0:
            continue
        s_full[idxs] = s
        z0_full[idxs] = np.arcsinh(hd / s)
    return z0_full, s_full


def build_core_context(host_day, hours, pipe, z0_full, s_full, y_full, raw_idxs):
    """Scale-free core context [u, time_feat(7), lag_sf(5)] = 13 dims (no z0)."""
    _, z0_day = _scale_z0(host_day)
    u = pipe.s1_rank_ref(z0_day, hours)
    time_feat = _cyclic_time_features(hours)

    H = len(hours)
    lag_sf = np.zeros((H, 5), dtype=np.float32)
    for h in range(H):
        ri = raw_idxs[h]
        lag24, lag48, lag168 = ri - 24, ri - 48, ri - 168
        if lag24 >= 0 and np.isfinite(z0_full[lag24]):
            lag_sf[h, 0] = z0_full[lag24]
            lag_sf[h, 4] = 1.0
        if lag48 >= 0 and np.isfinite(z0_full[lag48]):
            lag_sf[h, 1] = z0_full[lag48]
        if lag168 >= 0 and np.isfinite(z0_full[lag168]):
            lag_sf[h, 2] = z0_full[lag168]
        if lag24 >= 0 and np.isfinite(z0_full[lag24]) and np.isfinite(y_full[lag24]):
            s_prev = s_full[lag24]
            if s_prev > 0:
                zY_prev = np.arcsinh(y_full[lag24] / s_prev)
                lag_sf[h, 3] = zY_prev - z0_full[lag24]

    return np.concatenate([u.reshape(-1, 1), time_feat, lag_sf], axis=1)


def _run_candidate(pipe, host_day, hours, z0_full, s_full, y_full, idxs):
    """Run candidate; return None if SCALE_UNIDENTIFIED (HIGH-7)."""
    ctx = build_core_context(host_day, hours, pipe, z0_full, s_full, y_full, idxs)
    with torch.no_grad():
        out = pipe.candidate_head(
            torch.tensor(host_day.reshape(1, 24, 1), dtype=torch.float32),
            torch.tensor(ctx.reshape(1, 24, -1), dtype=torch.float32))
    if float(out["scale_valid"][0]) < 0.5:
        return None, None
    s = float(out["s"][0])
    zY = np.arcsinh(y_full[idxs].astype(np.float64) / s)  # no epsilon clamp
    return out, zY


def run(dataset_key="LAGO_DE", backbone="Linear", seed=0,
        s3m_frac=0.5, k_validation_frac=0.25, k_candidates=(5, 10, 20),
        alpha=0.10):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = load_dataset(dataset_key)
    y_full = ds["price"].astype(np.float32)
    ts = ds["ts"]
    X, y, names, valid = build_tabular(ds)
    assert_no_leakage(ds, X, y, valid, names)

    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id=dataset_key,
                                          s3m_frac=s3m_frac)
    assert exp.assert_s3_disjoint()

    # frozen Linear backbone on S1
    bb = make_backbone(backbone, seed=seed)
    bb.fit(X[exp.valid_row_in_split("S1")], y[exp.valid_row_in_split("S1")])
    yhat_valid = bb.predict(X).astype(np.float32)
    yhat_full = np.full(len(y_full), np.nan, dtype=np.float32)
    yhat_full[valid] = yhat_valid

    # scale-free precompute + S1 rank reference
    z0_full, s_full = precompute_scale_free(yhat_full, ts)
    s1_z0, s1_hours = [], []
    for d in exp.dates_in_split("S1"):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        _, z0d = _scale_z0(yhat_full[idxs].astype(np.float64))
        s1_z0.append(z0d)
        s1_hours.append(ts.iloc[idxs].dt.hour.values)
    s1_z0 = np.concatenate(s1_z0) if s1_z0 else np.zeros(1)
    s1_hours = np.concatenate(s1_hours) if s1_hours else None

    pipe = HCHV2UniversalPipeline(d_core_context=13, d_model=32,
                                  alpha=alpha, k=None, seed=seed)
    pipe.fit_s1_reference(s1_z0, s1_hours)
    pipe.fit_s1_signature(s1_z0, s1_hours)

    # ---- S2 training ----
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
            torch.tensor(y_full[idxs].astype(np.float64).reshape(1, 24, 1),
                         dtype=torch.float32),
            torch.ones(1, 24),
        ))
    loss = pipe.train_candidate_s2(s2_batches, epochs=8, lr=1e-3, patience=4)
    print(f"S2 training loss: {loss:.4f}")

    # ---- S3-M: memory prefix + k-validation suffix ----
    s3m_all = sorted(exp.dates_in_s3m())
    n_mem = int(len(s3m_all) * (1.0 - k_validation_frac))
    mem_dates, val_dates = s3m_all[:n_mem], s3m_all[n_mem:]

    def make_day(d):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            return None
        host_day = yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            return None
        hours = ts.iloc[idxs].dt.hour.values
        out, zY = _run_candidate(pipe, host_day, hours, z0_full, s_full, y_full, idxs)
        if out is None:
            return None
        return {"date": d, "candidate": out, "target_zY": zY}

    mem_days = [md for md in (make_day(d) for d in mem_dates) if md is not None]
    pipe.fit_s3_memory(mem_days)

    val_days = [vd for vd in (make_day(d) for d in val_dates) if vd is not None]
    k = pipe.select_s3m_k(list(k_candidates), val_days)
    print(f"S3-M memory: {len(mem_days)} days, validation: {len(val_days)}, "
          f"selected k={k}")

    # ---- S3-C calibration ----
    s3c_days = [sd for sd in (make_day(d) for d in sorted(exp.dates_in_s3c()))
                if sd is not None]
    q_info = pipe.calibrate_s3c(s3c_days)
    print(f"S3-C calibration: n={q_info['n']}, q={q_info['q']:.4f}")

    # ---- freeze + reload round-trip ----
    bundle = pipe.freeze_bundle(dataset_id=dataset_key, split_hash=exp.split_hash)
    pipe2 = HCHV2UniversalPipeline.from_bundle(bundle)

    # ---- S4: batch through predict_s4 (single authority) ----
    s4_hosts, s4_ctxs, s4_dates = [], [], []
    for d in sorted(exp.dates_in_split("S4")):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        host_day = yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            continue
        hours = ts.iloc[idxs].dt.hour.values
        ctx = build_core_context(host_day, hours, pipe2, z0_full, s_full, y_full, idxs)
        s4_hosts.append(host_day.reshape(24, 1))
        s4_ctxs.append(ctx)
        s4_dates.append(d)

    if s4_hosts:
        batch_host = torch.tensor(np.stack(s4_hosts), dtype=torch.float32)
        batch_ctx = torch.tensor(np.stack(s4_ctxs), dtype=torch.float32)
        ev = pipe2.predict_s4(batch_host, batch_ctx)
    else:
        ev = {"final_action": [], "candidate": None}

    evidence = []
    n_execute = 0
    for i, d in enumerate(s4_dates):
        cand = ev["candidate"]
        scale_v = float(cand["scale_valid"][i]) if cand is not None else 0.0
        if scale_v < 0.5:
            # SCALE_UNIDENTIFIED -> Identity, no chain evidence (HIGH-7)
            evidence.append({"date": d, "action": "identity",
                             "fallback": "scale_unidentified"})
            continue
        action = ev["final_action"][i]
        if action == "execute":
            n_execute += 1
        prop = ev["proposals"][i]
        evidence.append({
            "date": d,
            "scale": float(cand["s"][i]),
            "w_minus": [float(v) for v in cand["w_minus"][i].detach().cpu().numpy()],
            "w_plus": [float(v) for v in cand["w_plus"][i].detach().cpu().numpy()],
            "m_minus": [float(v) for v in cand["m_minus"][i].detach().cpu().numpy()],
            "m_plus": [float(v) for v in cand["m_plus"][i].detach().cpu().numpy()],
            "pi_q": [float(v) for v in ev["pi"][i]],
            "A_hat": float(ev["A_hat"][i]),
            "q": ev["q"],
            "LCB": float(ev["lcb"][i]),
            "action": action,
            "proposal_down": prop["I_down"],
            "proposal_up": prop["I_up"],
            "neighbor_dates": [pipe2.memory.dates[j] for j in ev["neighbors"][i]],
            "neighbor_distances": ev["neighbor_distances"][i],
            "x_final": [float(v) for v in ev["x_final"][i].detach().cpu().numpy().ravel()],
        })

    result = {
        "dataset": dataset_key, "backbone": backbone, "seed": seed,
        "S2_loss": float(loss),
        "n_S3M": len(mem_days), "n_S3C": len(s3c_days),
        "selected_k": k, "q": q_info["q"],
        "n_S4_days": len(evidence),
        "n_execute": n_execute,
        "execute_rate": n_execute / max(len(evidence), 1),
        "bundle_hash": bundle.hash(),
        "roundtrip_hash_match": bundle.hash() == pipe2.freeze_bundle(
            dataset_id=dataset_key, split_hash=exp.split_hash).hash(),
        "evidence": evidence,
    }
    return result


if __name__ == "__main__":
    result = run()
    out_path = OUT / "smoke_v4_lago_de_linear.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved evidence: {out_path}")
    print(f"selected_k={result['selected_k']}, "
          f"execute_rate={result['execute_rate']:.3f} "
          f"({result['n_execute']}/{result['n_S4_days']} days)")
    print(f"roundtrip_hash_match={result['roundtrip_hash_match']}")
