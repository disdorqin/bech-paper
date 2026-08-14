"""Phase4 P2 — CAVM composite-retrieval experiment (E0-E3).

One frozen chain (S1->S2->S3-M->S3-C) is run once, then the same S4 batch is
predicted under four retrieval regimes:
    E0  w1           : legacy CAGM W1-only (control, must equal smoke_v4)
    E1  cavm (1,0)   : CAVM ledger, W1-equivalent retrieval (must reproduce E0
                       neighbor IDs day-by-day)
    E2  cavm (0,1)   : context-only retrieval
    E3  cavm (1,1)   : composite retrieval

Prediction path is target-free (predict_s4). A_true is computed offline after
labels are revealed, using the SAME estimate_realized_A the pipeline uses.
Per experiment design §6/§9, λ weights are fixed at S3-M (k selected by W1
forward validation; λ NOT tuned on S4).
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
from query_replay import estimate_realized_A

OUT = Path(__file__).resolve().parent / "results" / "phase4"
OUT.mkdir(parents=True, exist_ok=True)


def pd_date(d):
    import pandas as pd
    return pd.Timestamp(d).date()


def _cyclic_time_features(hours):
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
    ctx = build_core_context(host_day, hours, pipe, z0_full, s_full, y_full, idxs)
    with torch.no_grad():
        out = pipe.candidate_head(
            torch.tensor(host_day.reshape(1, 24, 1), dtype=torch.float32),
            torch.tensor(ctx.reshape(1, 24, -1), dtype=torch.float32))
    if float(out["scale_valid"][0]) < 0.5:
        return None, None
    s = float(out["s"][0])
    zY = np.arcsinh(y_full[idxs].astype(np.float64) / s)
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

    bb = make_backbone(backbone, seed=seed)
    bb.fit(X[exp.valid_row_in_split("H0")], y[exp.valid_row_in_split("H0")])
    yhat_valid = bb.predict(X).astype(np.float32)
    yhat_full = np.full(len(y_full), np.nan, dtype=np.float32)
    yhat_full[valid] = yhat_valid

    z0_full, s_full = precompute_scale_free(yhat_full, ts)
    s1_z0, s1_hours = [], []
    for d in exp.dates_in_split("S1R"):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        _, z0d = _scale_z0(yhat_full[idxs].astype(np.float64))
        s1_z0.append(z0d)
        s1_hours.append(ts.iloc[idxs].dt.hour.values)
    s1_z0 = np.concatenate(s1_z0) if s1_z0 else np.zeros(1)
    s1_hours = np.concatenate(s1_hours) if s1_hours else None

    pipe = HCHV2UniversalPipeline(d_core_context=13, d_model=32,
                                  alpha=alpha, k=None, seed=seed,
                                  memory_mode="cavm")
    pipe.fit_s1_reference(s1_z0, s1_hours)
    pipe.fit_s1_signature(s1_z0, s1_hours)

    det_broadcast = None
    if pipe._domain_det is not None:
        det_b = torch.tensor(np.asarray(pipe._domain_det, dtype=np.float32),
                             dtype=torch.float32).unsqueeze(0)
        det_broadcast = det_b

    def _s2_batches_for(split):
        batches = []
        for d in sorted(exp.dates_in_split(split)):
            idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
            if len(idxs) != 24:
                continue
            host_day = yhat_full[idxs].astype(np.float64)
            if not np.isfinite(host_day).all():
                continue
            hours = ts.iloc[idxs].dt.hour.values
            ctx = build_core_context(host_day, hours, pipe, z0_full, s_full,
                                     y_full, idxs)
            batches.append((
                torch.tensor(host_day.reshape(1, 24, 1), dtype=torch.float32),
                torch.tensor(ctx.reshape(1, 24, -1), dtype=torch.float32),
                torch.tensor(y_full[idxs].astype(np.float64).reshape(1, 24, 1),
                             dtype=torch.float32),
                torch.ones(1, 24),
                det_broadcast.clone(),
            ))
        return batches

    s2_batches = _s2_batches_for("S2T")
    s2v_batches = _s2_batches_for("S2V")
    s2_loss = pipe.train_candidate_s2(s2_batches, s2v_batches=s2v_batches,
                                      epochs=8, lr=1e-3, patience=4)
    print(f"S2 checkpoint (S2V-selected): {s2_loss:.4f} "
          f"({len(s2v_batches)} validation days)")

    # ---- S3-M: memory prefix + k-validation suffix ----
    s3m_all = sorted(exp.dates_in_s3m())
    n_mem = int(len(s3m_all) * (1.0 - k_validation_frac))
    mem_dates, val_dates = s3m_all[:n_mem], s3m_all[n_mem:]

    det_day = pipe._domain_det  # single-domain frozen descriptor

    def make_day(d):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            return None
        host_day = yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            return None
        hours = ts.iloc[idxs].dt.hour.values
        out, zY = _run_candidate(pipe, host_day, hours, z0_full, s_full,
                                 y_full, idxs)
        if out is None:
            return None
        ctx = build_core_context(host_day, hours, pipe, z0_full, s_full,
                                 y_full, idxs)
        return {"date": d, "candidate": out, "target_zY": zY,
                "core_context": ctx, "domain_det": det_day}

    mem_days = [md for md in (make_day(d) for d in mem_dates) if md is not None]
    pipe.fit_s3_memory(mem_days)

    val_days = [vd for vd in (make_day(d) for d in val_dates) if vd is not None]
    k = pipe.select_s3m_k(list(k_candidates), val_days)
    print(f"S3-M memory: {len(mem_days)} days, validation: {len(val_days)}, "
          f"selected k={k}")

    s3c_days = [sd for sd in (make_day(d) for d in sorted(exp.dates_in_s3c()))
                if sd is not None]
    q_info = pipe.calibrate_s3c(s3c_days)
    print(f"S3-C calibration: n={q_info['n']}, q={q_info['q']:.4f}")

    # CAVM global ledger from the SAME revealed S3-M days (with context).
    pipe.fit_cavm_memory(mem_days)
    print(f"CAVM global ledger: {len(pipe.cavm_global)} days, "
          f"key_dim={pipe.cavm_key_builder.dim}")

    # ---- S4 batch (shared across regimes; predict_s4 is target-free) ----
    s4_hosts, s4_ctxs, s4_dates, s4_y = [], [], [], []
    for d in sorted(exp.dates_in_split("S4")):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        host_day = yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            continue
        hours = ts.iloc[idxs].dt.hour.values
        ctx = build_core_context(host_day, hours, pipe, z0_full, s_full,
                                 y_full, idxs)
        s4_hosts.append(host_day.reshape(24, 1))
        s4_ctxs.append(ctx)
        s4_dates.append(d)
        s4_y.append(y_full[idxs].astype(np.float64))

    n_s4 = len(s4_hosts)
    batch_host = torch.tensor(np.stack(s4_hosts), dtype=torch.float32)
    batch_ctx = torch.tensor(np.stack(s4_ctxs), dtype=torch.float32)
    domain_det = None
    if pipe._domain_det is not None:
        det_t = torch.tensor(np.asarray(pipe._domain_det, dtype=np.float32),
                             dtype=torch.float32)
        domain_det = det_t.unsqueeze(0).expand(n_s4, -1)

    def predict(memory_mode, lam):
        pipe.memory_mode = memory_mode
        pipe.set_cavm_retrieval(lam[0], lam[1])
        ev = pipe.predict_s4(batch_host, batch_ctx, domain_det=domain_det)
        return summarize(ev)

    def summarize(ev):
        n = len(s4_dates)
        acts = [ev["final_action"][i] for i in range(n)]
        n_exec = sum(a == "execute" for a in acts)
        # Point readout (weighted_mean here == x_final identity/executed mix).
        y_hat = np.stack([ev["x_final"][i].detach().cpu().numpy().ravel()
                          for i in range(n)]) if n else np.zeros((0, 24))
        y_true = np.stack(s4_y) if n else np.zeros((0, 24))
        mae = float(np.mean(np.abs(y_hat - y_true)))
        rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))

        # A_true offline (labels revealed AFTER prediction): same estimator.
        a_hats = [float(ev["A_hat"][i]) for i in range(n)]
        a_trues = []
        for i in range(n):
            cand = ev["candidate"]
            if float(cand["scale_valid"][i]) < 0.5:
                a_trues.append(np.nan)
                continue
            s = float(cand["s"][i])
            z0 = cand["z0"][i].detach().cpu().numpy().ravel()
            zY = np.arcsinh(s4_y[i] / s)
            vm = cand["valid_mask"][i].detach().cpu().numpy().ravel().astype(bool)
            pi = np.asarray(ev["pi"][i], dtype=np.float64).ravel()
            a_trues.append(float(estimate_realized_A(z0, zY, pi, vm)))

        return {
            "n_days": n, "n_execute": n_exec, "execute_rate": n_exec / max(n, 1),
            "n_identity": n - n_exec, "identity_rate": (n - n_exec) / max(n, 1),
            "MAE": mae, "RMSE": rmse,
            "mean_A_hat": float(np.mean(a_hats)) if a_hats else 0.0,
            "mean_A_true": float(np.nanmean(a_trues)) if a_trues else 0.0,
            "mean_lcb": float(np.mean([float(ev["lcb"][i]) for i in range(n)]))
                if n else 0.0,
            "actions": acts,
            "A_hat": a_hats, "A_true": a_trues,
            "neighbors": [list(ev["neighbors"][i]) for i in range(n)],
            "fallback_reasons": ev.get("fallback_reasons", []),
        }

    modes = [
        ("E0_w1", "w1", (1.0, 0.0)),
        ("E1_cavm_10", "cavm", (1.0, 0.0)),
        ("E2_cavm_01", "cavm", (0.0, 1.0)),
        ("E3_cavm_11", "cavm", (1.0, 1.0)),
    ]
    results = {}
    for name, mm, lam in modes:
        results[name] = predict(mm, lam)
        print(f"  {name}: exec_rate={results[name]['execute_rate']:.3f} "
              f"MAE={results[name]['MAE']:.3f} RMSE={results[name]['RMSE']:.3f} "
              f"mean_A_hat={results[name]['mean_A_hat']:.4f} "
              f"mean_A_true={results[name]['mean_A_true']:.4f}")

    # ---- day-by-day deltas vs E0 (w1 control) ----
    n = n_s4
    delta = {}
    e0 = results["E0_w1"]
    for name in ("E1_cavm_10", "E2_cavm_01", "E3_cavm_11"):
        r = results[name]
        nbr_changed = sum(r["neighbors"][i] != e0["neighbors"][i]
                          for i in range(n))
        act_changed = sum(r["actions"][i] != e0["actions"][i]
                          for i in range(n))
        ahat_changed = sum(abs(r["A_hat"][i] - e0["A_hat"][i]) > 1e-12
                           for i in range(n))
        delta[name] = {
            "neighbor_changed_days": nbr_changed,
            "action_changed_days": act_changed,
            "A_hat_changed_days": ahat_changed,
            "MAE_delta": r["MAE"] - e0["MAE"],
            "RMSE_delta": r["RMSE"] - e0["RMSE"],
        }
        print(f"  {name} vs E0: neighbor-change {nbr_changed}/{n}, "
              f"action-change {act_changed}/{n}, "
              f"MAE delta {delta[name]['MAE_delta']:+.4f}")

    result = {
        "dataset": dataset_key, "backbone": backbone, "seed": seed,
        "S2_loss": float(s2_loss), "n_S3M": len(mem_days),
        "selected_k": k, "q": q_info["q"],
        "n_S4_days": n,
        "cavm_key_version": pipe.cavm_key_builder.version,
        "cavm_key_dim": pipe.cavm_key_builder.dim,
        "cavm_global_days": len(pipe.cavm_global),
        "modes": results,
        "delta_vs_E0": delta,
    }
    return result


if __name__ == "__main__":
    out_path = OUT / "p2_cavm_experiment_lago_de_linear.json"
    result = run()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
