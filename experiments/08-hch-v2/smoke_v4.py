"""Formal v0.4 real-data smoke — public price-only market + Linear backbone.

Per code-correction §7: S1 rank -> IAH candidate (S2) -> S3-M memory/k ->
S3-C DVG -> S4 target-free predict -> evidence JSON.

Uses ONLY the new v0.3 IAH path. Never imports legacy HCH.
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
from s1_rank import S1RankReference

OUT = Path(__file__).resolve().parent / "results" / "v0.3"
OUT.mkdir(parents=True, exist_ok=True)


def _cyclic_time_features(hours: np.ndarray) -> np.ndarray:
    """Cyclic calendar features (7-dim), no raw hour as price channel."""
    h = hours.astype(np.float32)
    return np.column_stack([
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * h / 168), np.cos(2 * np.pi * h / 168),
        np.sin(2 * np.pi * (h // 24) / 7), np.cos(2 * np.pi * (h // 24) / 7),
        (h % 24 < 6).astype(np.float32),  # night flag
    ])


def compute_scale_z0(host_day: np.ndarray) -> tuple:
    """Per-day scale + z0 (Eq 2-3). host_day [H]."""
    finite = np.isfinite(host_day)
    if finite.sum() == 0:
        return 0.0, np.zeros_like(host_day)
    s = float(np.mean(np.abs(host_day[finite])))
    if s <= 0:
        return 0.0, np.zeros_like(host_day)
    z0 = np.arcsinh(host_day / s)
    z0[~finite] = 0.0
    return s, z0


def run(dataset_key="LAGO_DE", backbone="Linear", seed=0,
        s3m_frac=0.5, k_candidates=(5, 10, 20), alpha=0.10):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = load_dataset(dataset_key)
    y_full = ds["price"].astype(np.float32)
    ts = ds["ts"]
    X, y, names, valid = build_tabular(ds)
    assert_no_leakage(ds, X, y, valid, names)

    # ---- unified date-first manifest ----
    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id=dataset_key,
                                          s3m_frac=s3m_frac)
    assert exp.assert_s3_disjoint(), "S1/S2/S3-M/S3-C/S4 must be disjoint"

    # ---- train frozen Linear backbone on S1 ----
    bb = make_backbone(backbone, seed=seed)
    s1_row = exp.valid_row_in_split("S1")  # valid-row positions for X/y
    bb.fit(X[s1_row], y[s1_row])
    yhat_valid = bb.predict(X).astype(np.float32)

    # full-length host prediction (NaN outside valid); valid IS raw indices
    yhat_full = np.full(len(y_full), np.nan, dtype=np.float32)
    yhat_full[valid] = yhat_valid

    # ---- build S1 rank reference from S1 host z0 ----
    s1_host_z0 = []
    s1_hours = []
    for d in exp.dates_in_split("S1"):
        mask = ts.dt.date == pd_date(d)
        idxs = np.where(mask.values)[0]
        if len(idxs) != 24:
            continue
        _, z0 = compute_scale_z0(yhat_full[idxs].astype(np.float64))
        s1_host_z0.append(z0)
        s1_hours.append(ts.iloc[idxs].dt.hour.values)
    s1_host_z0 = np.concatenate(s1_host_z0) if s1_host_z0 else np.zeros(1)
    s1_hours = np.concatenate(s1_hours) if s1_hours else None

    # ---- build pipeline ----
    d_context = 1 + 1 + 7  # z0 + u + time_feat (core path; lag optional)
    pipe = HCHV2UniversalPipeline(d_context=d_context, d_hidden=32,
                                  alpha=alpha, k=None, seed=seed)
    pipe.fit_s1_reference(s1_host_z0, s1_hours)

    # ---- context builder (per day) ----
    def build_context(host_day: np.ndarray, hours: np.ndarray) -> np.ndarray:
        s, z0 = compute_scale_z0(host_day.astype(np.float64))
        u = pipe.s1_rank_ref(z0, hours)
        time_feat = _cyclic_time_features(hours)
        return np.concatenate([z0.reshape(-1, 1), u.reshape(-1, 1), time_feat],
                              axis=1), s, z0

    # ---- S2 training batches ----
    s2_dates = sorted(exp.dates_in_split("S2"))
    s2_batches = []
    for d in s2_dates:
        mask = ts.dt.date == pd_date(d)
        idxs = np.where(mask.values)[0]
        if len(idxs) != 24:
            continue
        host_day = yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            continue
        hours = ts.iloc[idxs].dt.hour.values
        ctx, s, z0 = build_context(host_day, hours)
        tgt = y_full[idxs].astype(np.float64)
        s2_batches.append((
            torch.tensor(host_day.reshape(1, 24, 1), dtype=torch.float32),
            torch.tensor(ctx.reshape(1, 24, -1), dtype=torch.float32),
            torch.tensor(tgt.reshape(1, 24, 1), dtype=torch.float32),
            torch.ones(1, 24),
        ))

    loss = pipe.train_candidate_s2(s2_batches, epochs=8, lr=1e-3, patience=4)
    print(f"S2 training loss: {loss:.4f}")

    # ---- S3-M memory ----
    s3m_dates = sorted(exp.dates_in_s3m())
    s3m_days = []
    for d in s3m_dates:
        mask = ts.dt.date == pd_date(d)
        idxs = np.where(mask.values)[0]
        if len(idxs) != 24:
            continue
        host_day = yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            continue
        hours = ts.iloc[idxs].dt.hour.values
        ctx, s, z0 = build_context(host_day, hours)
        with torch.no_grad():
            out = pipe.candidate_head(
                torch.tensor(host_day.reshape(1, 24, 1), dtype=torch.float32),
                torch.tensor(ctx.reshape(1, 24, -1), dtype=torch.float32))
        s_prev = max(float(out["s"][0]), 1e-12)
        zY = np.arcsinh(y_full[idxs].astype(np.float64) / s_prev)
        s3m_days.append({"date": d, "candidate": out, "target_zY": zY})
    pipe.fit_s3_memory(s3m_days)
    pipe.k = k_candidates[len(k_candidates) // 2]  # frozen k (validated below)
    print(f"S3-M memory: {len(s3m_days)} days, k={pipe.k}")

    # ---- S3-C calibration ----
    s3c_dates = sorted(exp.dates_in_s3c())
    s3c_days = []
    for d in s3c_dates:
        mask = ts.dt.date == pd_date(d)
        idxs = np.where(mask.values)[0]
        if len(idxs) != 24:
            continue
        host_day = yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            continue
        hours = ts.iloc[idxs].dt.hour.values
        ctx, s, z0 = build_context(host_day, hours)
        with torch.no_grad():
            out = pipe.candidate_head(
                torch.tensor(host_day.reshape(1, 24, 1), dtype=torch.float32),
                torch.tensor(ctx.reshape(1, 24, -1), dtype=torch.float32))
        s_prev = max(float(out["s"][0]), 1e-12)
        zY = np.arcsinh(y_full[idxs].astype(np.float64) / s_prev)
        # whole-day action value: replay final pi on memory neighbors
        from query_replay import full_replay_chain
        from double_event import double_event_proposal
        m_minus = out["m_minus"].detach().cpu().numpy().reshape(-1)
        m_plus = out["m_plus"].detach().cpu().numpy().reshape(-1)
        dists = pipe.memory.build_retrieval_index(out)
        nbr = pipe.memory.get_neighbors(dists, pipe.k)
        chain = full_replay_chain(pipe.memory, nbr, m_minus, m_plus,
                                  double_event_proposal)
        A_hat = chain["A_hat"]
        # realized whole-day A for the proposed pi_q
        A_true = estimate_realized_A(z0, zY, chain["pi_q"])
        s3c_days.append({"date": d, "A_hat": A_hat, "A_true": A_true})
    q_info = pipe.calibrate_s3c(s3c_days)
    print(f"S3-C calibration: n={q_info['n']}, q={q_info['q']:.4f}")

    # ---- S4 target-free predict ----
    s4_dates = sorted(exp.dates_in_split("S4"))
    evidence = []
    n_execute = 0
    for d in s4_dates:
        mask = ts.dt.date == pd_date(d)
        idxs = np.where(mask.values)[0]
        if len(idxs) != 24:
            continue
        host_day = yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            continue
        hours = ts.iloc[idxs].dt.hour.values
        ctx, s, z0 = build_context(host_day, hours)
        with torch.no_grad():
            out = pipe.candidate_head(
                torch.tensor(host_day.reshape(1, 24, 1), dtype=torch.float32),
                torch.tensor(ctx.reshape(1, 24, -1), dtype=torch.float32))
        m_minus = out["m_minus"].detach().cpu().numpy().reshape(-1)
        m_plus = out["m_plus"].detach().cpu().numpy().reshape(-1)
        dists = pipe.memory.build_retrieval_index(out)
        nbr = pipe.memory.get_neighbors(dists, pipe.k)
        chain = full_replay_chain(pipe.memory, nbr, m_minus, m_plus,
                                  double_event_proposal)
        lcb_info = pipe.dvg.lcb(chain["A_hat"])
        action = "execute" if lcb_info["execute"] else "identity"
        if lcb_info["execute"]:
            n_execute += 1
        # final raw prediction: apply pi_q to identity in raw coordinates
        x_final = host_day.copy()
        for h in range(24):
            if chain["pi_q"][h] < 0:
                x_final[h] = float(out["x_down"][0, h, 0])
            elif chain["pi_q"][h] > 0:
                x_final[h] = float(out["x_up"][0, h, 0])
        evidence.append({
            "date": d,
            "scale": float(out["s"][0]),
            "rank_u": [float(v) for v in build_context(host_day, hours)[0][:, 1]],
            "w_minus": [float(v) for v in out["w_minus"][0].detach().cpu().numpy()],
            "w_zero": [float(v) for v in out["w_zero"][0].detach().cpu().numpy()],
            "w_plus": [float(v) for v in out["w_plus"][0].detach().cpu().numpy()],
            "m_minus": [float(v) for v in m_minus],
            "m_plus": [float(v) for v in m_plus],
            "pi_q": [float(v) for v in chain["pi_q"]],
            "A_hat": float(chain["A_hat"]),
            "q": pipe.dvg.q,
            "LCB": float(lcb_info["lcb"]),
            "action": action,
            "proposal_down": chain["proposal"]["I_down"],
            "proposal_up": chain["proposal"]["I_up"],
            "neighbor_dates": [pipe.memory.dates[i] for i in nbr],
            "neighbor_distances": [float(dists[i]) for i in nbr],
            "x_final": [float(v) for v in x_final],
        })

    # ---- freeze bundle ----
    bundle = pipe.freeze_bundle(dataset_id=dataset_key, split_hash=exp.split_hash)

    result = {
        "dataset": dataset_key, "backbone": backbone, "seed": seed,
        "S2_loss": float(loss),
        "n_S3M": len(s3m_days), "n_S3C": len(s3c_days),
        "q": q_info["q"], "k": pipe.k,
        "n_S4_days": len(evidence),
        "n_execute": n_execute,
        "execute_rate": n_execute / max(len(evidence), 1),
        "bundle_hash": bundle.hash(),
        "evidence": evidence,
    }
    return result


def estimate_realized_A(z0, zY, pi_q):
    """Whole-day realized action value A = mean_h(|r_z| - |r_z - pi|)."""
    r_z = zY - z0
    g = np.abs(r_z) - np.abs(r_z - pi_q)
    return float(np.mean(g))


def pd_date(d):
    import pandas as pd
    return pd.Timestamp(d).date()


if __name__ == "__main__":
    result = run()
    out_path = OUT / "smoke_v4_lago_de_linear.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved evidence: {out_path}")
    print(f"execute_rate: {result['execute_rate']:.3f} "
          f"({result['n_execute']}/{result['n_S4_days']} days)")
