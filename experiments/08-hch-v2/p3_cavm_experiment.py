"""Phase4 P3 — observe_outcome() streaming experiment (build spec §5.2, P3).

Two S4 regimes on the SAME frozen chain (universal/k/lambda/q untouched):

  FROZEN  observe OFF : full-batch predict_s4, no local ledger recorded.
  STREAM  observe ON  : day-by-day predict (single-day batch) -> reveal ->
                        observe_outcome() appends to local ledger.

Invariance (spec §5.2, §8.3): predictions in STREAM must be byte-identical to
FROZEN day-by-day — local memory is NEVER consumed by predict_s4.

Cold-start / steady-state curve (spec P3.4): running mean of realized A_true
and action_error as the local ledger grows — cold-start = first window,
steady-state = last window. Local is an append-only audit ledger (P4 feedstock),
so the "curve" is the ledger's recorded value statistics converging.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from p2_cavm_experiment import (
    run as _p2_run_guard,  # noqa: F401 (module import triggers no execution)
    pd_date, precompute_scale_free, build_core_context, _scale_z0,
    _run_candidate,
)

OUT = Path(__file__).resolve().parent / "results" / "phase4"
OUT.mkdir(parents=True, exist_ok=True)


def _build_chain(pipe, dataset_key, backbone, seed, s3m_frac,
                 k_validation_frac, k_candidates, alpha):
    """Reconstruct the P2 chain (S1->S2->S3-M->S3-C + CAVM global)."""
    import numpy as np
    import torch
    from common import load_dataset, build_tabular, assert_no_leakage
    from backbones import make_backbone
    from eval_manifest import ExperimentManifest

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

    s2_loss = pipe.train_candidate_s2(_s2_batches_for("S2T"),
                                      s2v_batches=_s2_batches_for("S2V"),
                                      epochs=8, lr=1e-3, patience=4)

    s3m_all = sorted(exp.dates_in_s3m())
    n_mem = int(len(s3m_all) * (1.0 - k_validation_frac))
    mem_dates, val_dates = s3m_all[:n_mem], s3m_all[n_mem:]
    det_day = pipe._domain_det

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
    s3c_days = [sd for sd in (make_day(d) for d in sorted(exp.dates_in_s3c()))
                if sd is not None]
    q_info = pipe.calibrate_s3c(s3c_days)
    pipe.fit_cavm_memory(mem_days)

    s4 = []
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
        s4.append({"date": d, "host": host_day, "ctx": ctx, "y": y_full[idxs]})
    return pipe, s4, s2_loss, k, q_info["q"]


def _day_batch(day):
    host = torch.tensor(day["host"].reshape(1, 24, 1), dtype=torch.float32)
    ctx = torch.tensor(day["ctx"].reshape(1, 24, -1), dtype=torch.float32)
    return host, ctx


def main(dataset_key="LAGO_DE", backbone="Linear", seed=0,
         s3m_frac=0.5, k_validation_frac=0.25, k_candidates=(5, 10, 20),
         alpha=0.10, cold_window=50):
    from hch_v2_pipeline import HCHV2UniversalPipeline

    pipe = HCHV2UniversalPipeline(d_core_context=13, d_model=32, alpha=alpha,
                                  k=None, seed=seed, memory_mode="cavm")
    pipe, s4, s2_loss, k, q = _build_chain(
        pipe, dataset_key, backbone, seed, s3m_frac, k_validation_frac,
        k_candidates, alpha)
    print(f"chain: S2={s2_loss:.4f} k={k} q={q:.4f} "
          f"cavm_global={len(pipe.cavm_global)} days, S4={len(s4)} days")

    # ---- FROZEN: observe off, full-batch predict ----
    host = torch.tensor(np.stack([d["host"] for d in s4]).reshape(-1, 24, 1),
                        dtype=torch.float32)
    ctx = torch.tensor(np.stack([d["ctx"] for d in s4]),
                       dtype=torch.float32)  # [N,24,13]
    det = None
    if pipe._domain_det is not None:
        det = torch.tensor(np.asarray(pipe._domain_det, dtype=np.float32),
                           dtype=torch.float32)
        det = det.unsqueeze(0).expand(len(s4), -1)
    ev_frozen = pipe.predict_s4(host, ctx, domain_det=det)
    frozen_A_hat = [float(x) for x in ev_frozen["A_hat"]]
    frozen_act = list(ev_frozen["final_action"])
    print(f"FROZEN: predictions recorded, local ledger = "
          f"{len(pipe.cavm_local) if pipe.cavm_local else 0}")

    # ---- STREAM: observe on, day-by-day predict -> reveal -> observe ----
    pipe.set_cavm_update_policy(observe=True)
    stream = []
    max_ahat_delta, max_x_delta = 0.0, 0.0
    for i, day in enumerate(s4):
        h, c = _day_batch(day)
        ev = pipe.predict_s4(h, c, domain_det=det[i:i + 1])
        # Invariance vs FROZEN (universal never touched by observe).
        max_ahat_delta = max(max_ahat_delta,
                             abs(float(ev["A_hat"][0]) - frozen_A_hat[i]))
        if ev["x_final"].shape == ev_frozen["x_final"][i:i + 1].shape:
            max_x_delta = max(max_x_delta, float(
                (ev["x_final"] - ev_frozen["x_final"][i:i + 1]).abs().max()))
        ev["query_ids"] = [str(day["date"])]
        # Revealed SCALE-FREE target (pipeline convention, same space as z0).
        s = float(ev["candidate"]["s"][0])
        zY = np.arcsinh(np.asarray(day["y"], dtype=np.float64) / s)
        r = pipe.observe_outcome(str(day["date"]), zY, ev)
        assert r["applied"], r
        stream.append({
            "date": str(day["date"]),
            "n_local": r["n_local"],
            "A_hat": r["A_hat"], "A_true": r["A_true"],
            "action_error": r["action_error"],
            "action": ev["final_action"][0],
        })
    n = len(stream)
    print(f"STREAM: {n} days observed into local ledger "
          f"(final n_local={pipe.cavm_local and len(pipe.cavm_local)})")

    # ---- cold-start vs steady-state curve ----
    cw = min(cold_window, n)
    def mean_err(rows):
        e = [x["action_error"] for x in rows if x["action_error"] == x["action_error"]]
        return float(np.mean(e)) if e else None
    def mean_true(rows):
        t = [x["A_true"] for x in rows if x["A_true"] == x["A_true"]]
        return float(np.mean(t)) if t else None
    def pos_rate(rows):
        t = [x["A_true"] for x in rows if x["A_true"] == x["A_true"]]
        return float(np.mean([1.0 if x > 0 else 0.0 for x in t])) if t else None
    curve = {
        "window": cw,
        "cold_start": {
            "mean_action_error": mean_err(stream[:cw]),
            "mean_A_true": mean_true(stream[:cw]),
            "pos_rate": pos_rate(stream[:cw]),
        },
        "steady_state": {
            "mean_action_error": mean_err(stream[-cw:]),
            "mean_A_true": mean_true(stream[-cw:]),
            "pos_rate": pos_rate(stream[-cw:]),
        },
        "full": {
            "mean_action_error": mean_err(stream),
            "mean_A_true": mean_true(stream),
            "pos_rate": pos_rate(stream),
        },
    }
    exec_days = [x for x in stream if x["action"] == "execute"]
    curve["execute_days"] = {
        "n": len(exec_days),
        "mean_A_true": mean_true(exec_days),
        "pos_rate": pos_rate(exec_days),
    }

    result = {
        "dataset": dataset_key, "backbone": backbone, "seed": seed,
        "S2_loss": float(s2_loss), "selected_k": k, "q": q,
        "n_S4": n,
        "cavm_key_version": pipe.cavm_key_builder.version,
        "cavm_global_days": len(pipe.cavm_global),
        "invariance": {
            "max_A_hat_delta": max_ahat_delta,
            "max_x_final_delta": max_x_delta,
            "action_switch_days": sum(
                1 for i in range(n)
                if stream[i]["action"] != frozen_act[i]),
        },
        "curve": curve,
        "stream": stream,
    }
    return result


if __name__ == "__main__":
    out_path = OUT / "p3_cavm_experiment_lago_de_linear.json"
    result = main()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
