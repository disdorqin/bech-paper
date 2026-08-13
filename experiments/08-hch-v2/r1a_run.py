"""R1A — first-round universal HCH-v2 experiment runner (protocol v0.1).

Authoritative protocol: docs/paper_prep/v2_final/hch_v2_first_round_training_protocol_v0.1_2026-08-13.md

Runs exactly 6 domains (LAGO_DE, LAGO_PJM, NEM_SA1) x (Linear, MLP) with two
variants sharing the SAME architecture and every hyperparameter.
R1B naming (audit §2.1 of hch_v2_r1a_review_r1a5_diagnostics_r1b_revision_v0.1):

    learned_sig (LearnedSig)      — old "Universal-NoSig". Deterministic Data
                                    Signature ZEROED (np.zeros(8)); the learned
                                    per-day pool + identity-init FiLM stay
                                    identical to the other variant. The ONLY
                                    difference is whether the frozen S1R domain
                                    descriptor reaches the model. Tentative main.
    learned_det_sig (Learned+DetSig) — old "Universal-Sig". Frozen S1R
                                    deterministic descriptor [8] as forward
                                    context (protocol §8 / P1-2 descriptor v1).
                                    Demoted to an ablation by R1A evidence.

PlainCore (h'=h, no DataSignature FiLM) is an R1B addition, NOT part of R1A.5.

Shared candidate trained by UniversalCoreTrainer: equal-domain sampling with
K = median_g N_g updates/domain/epoch (P0-C), macro S2V IAH-CRPS checkpoint
selection (P0-3/G0-3), host baseline L^host_g = E|zY-z0| (protocol §12).

Then per-domain local chain: S3-M memory + k selection ({5,10,20}, §16 drops
k>memory), S3-C DVG split-conformal calibration (alpha=0.10), freeze/from_bundle
round-trip, S4 target-free predict.

Artifacts per protocol §22 -> R1A_<timestamp>/ (run_config.yaml, git_commit.txt,
domain_manifest.csv, split_manifest.json, host_cache_manifest.csv,
signature_by_domain.csv, train_curve_*.csv, val_crps_by_domain_*.csv,
mass_shift_stats_*.csv, checkpoint_*.pt, checkpoint_hashes.json,
candidate_s4.parquet, final_s4.parquet, gate_evidence.jsonl,
metrics_by_domain_host.csv, dm_tests.csv, VERDICT.md).

Variant-disable note (P0-1/G0-1): domain_det=None falls back to a MUTABLE buffer
in DataSignature, so NoSig must pass an explicit np.zeros(8) as the domain's
descriptor — that is the ONLY legitimate way to "zero" the signature without
changing the shared architecture.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent

from common import load_dataset, build_tabular, assert_no_leakage, dm_test
from host_cache import cache_one
from eval_manifest import ExperimentManifest, SPLIT_7
from hch_v2_context import compute_domain_descriptors
from hch_v2_pipeline import HCHV2UniversalPipeline
from iah_candidate import IAHCandidateHead
from iah_crps_loss import iah_crps_loss
from universal_trainer import UniversalCoreTrainer, DomainBatch, _eval_batch_losses
from s1_rank import S1RankReference
from query_replay import estimate_realized_A

# ---------------------------------------------------------------- config ----
D_CORE_CONTEXT = 13      # u + 7 time feats + 5 lag_sf (protocol §9)
D_MODEL = 64             # protocol §9
D_SIG = 32               # protocol §9
D_VALUE = 0              # optional branch disabled (protocol §9)
ALPHA = 0.10             # DVG split-conformal level (protocol §16)
K_CANDIDATES = (5, 10, 20)
S3M_MEM_FRAC = 0.75      # first 75% of S3M = memory, last 25% = forward-validation
LR = 3e-4                # protocol §11
WD = 1e-4                # protocol §11
CLIP = 1.0               # protocol §11
BATCH_DAYS = 16          # protocol §10: 16-32 day-homogeneous minibatches
SEED = 0

DOMAINS = [("LAGO_DE", "Linear"), ("LAGO_DE", "MLP"),
           ("LAGO_PJM", "Linear"), ("LAGO_PJM", "MLP"),
           ("NEM_SA1", "Linear"), ("NEM_SA1", "MLP")]

# R1B naming (review §2.1 audit fix). Old R1A keys were "nosig"/"sig".
VARIANTS = ("learned_sig", "learned_det_sig")
VARIANT_LABELS = {
    "learned_sig": "LearnedSig",            # = old Universal-NoSig, det=0
    "learned_det_sig": "Learned+DetSig",    # = old Universal-Sig
}


def det_for_variant(variant: str, info: "DomainInfo") -> np.ndarray:
    """Deterministic descriptor for a variant: zeros (LearnedSig) or S1R det."""
    return np.zeros(8) if variant == "learned_sig" else info.det_real


# ------------------------------------------------------------ helpers ------
def pd_date(d):
    import pandas as _pd
    return _pd.Timestamp(d).date()


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
    """Per-hour z0 and day scale for the whole series (scale-free, smoke-identical)."""
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


def build_core_context(host_day, hours, rank_ref, z0_full, s_full, y_full, raw_idxs):
    """Scale-free core context [u, time_feat(7), lag_sf(5)] = 13 dims (no z0)."""
    _, z0_day = _scale_z0(host_day)
    u = rank_ref(z0_day, hours)
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


def det_for(det_np: np.ndarray, n: int) -> torch.Tensor:
    """[n, d_det] descriptor tensor (real or zeros) broadcast to n rows."""
    det = torch.tensor(np.asarray(det_np, dtype=np.float32), dtype=torch.float32)
    return det.unsqueeze(0).expand(n, -1)


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, default=str, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------ domain preparation ----
@dataclass
class DomainInfo:
    ds_key: str
    bb: str
    ds: dict
    exp: ExperimentManifest
    yhat_full: np.ndarray
    z0_full: np.ndarray
    s_full: np.ndarray
    s1_z0: np.ndarray
    s1_hours: np.ndarray
    rank_ref: S1RankReference
    det_real: np.ndarray
    s1_prices: np.ndarray = field(default_factory=lambda: np.zeros(0))
    s2t_batches: list = field(default_factory=list)
    s2v_batches: list = field(default_factory=list)


def prepare_domain(ds_key: str, bb: str, seed: int = 0) -> DomainInfo:
    """Load cache (host fitted on H0 only, P0-2), build S1R ref/signature + S2 batches."""
    ds = load_dataset(ds_key)
    y_full = ds["price"].astype(np.float32)
    ts = ds["ts"]
    X, y, names, valid = build_tabular(ds)
    assert_no_leakage(ds, X, y, valid, names)
    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id=ds_key)
    assert exp.assert_7seg_disjoint(), f"{ds_key}: 7 segments not disjoint/exhaustive"

    cache_dir = HERE / "results" / "cache" / ds_key / bb
    seg = json.load(open(cache_dir / "seg.json"))
    assert seg["split_hash"] == exp.split_hash, \
        f"{ds_key} x {bb}: cache split_hash {seg['split_hash']} != manifest {exp.split_hash}"
    yhat_full = np.load(cache_dir / "pred_full.npy")          # full-array, NaN elsewhere
    yhat_valid = np.load(cache_dir / "pred.npy")
    valid_c = np.load(cache_dir / "valid.npy")
    assert np.allclose(yhat_full[valid_c], yhat_valid), "cache pred_full/pred mismatch"

    # scale-free precompute over all OOS host predictions
    z0_full, s_full = precompute_scale_free(yhat_full, ts)

    # S1R only: OOS host predictions build rank reference + deterministic signature
    s1_z0, s1_hours, s1_prices = [], [], []
    for d in sorted(exp.dates_in_split("S1R")):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        _, z0d = _scale_z0(yhat_full[idxs].astype(np.float64))
        s1_z0.append(z0d)
        s1_hours.append(ts.iloc[idxs].dt.hour.values)
        s1_prices.append(y_full[idxs])
    s1_z0 = np.concatenate(s1_z0) if s1_z0 else np.zeros(1)
    s1_hours = np.concatenate(s1_hours) if s1_hours else None
    s1_prices = np.concatenate(s1_prices) if s1_prices else np.zeros(0)

    rank_ref = S1RankReference(s1_z0, s1_hours)
    det_real = compute_domain_descriptors(s1_z0, s1_hours)

    info = DomainInfo(ds_key=ds_key, bb=bb, ds=ds, exp=exp,
                      yhat_full=yhat_full, z0_full=z0_full, s_full=s_full,
                      s1_z0=s1_z0, s1_hours=s1_hours, rank_ref=rank_ref,
                      det_real=det_real, s1_prices=s1_prices)

    # ---- S2T / S2V day-batches (chronological, domain-homogeneous) ----
    def _day_entries(split):
        entries = []
        for d in sorted(exp.dates_in_split(split)):
            idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
            if len(idxs) != 24:
                continue
            host_day = yhat_full[idxs].astype(np.float64)
            if not np.isfinite(host_day).all():
                continue
            hours = ts.iloc[idxs].dt.hour.values
            ctx = build_core_context(host_day, hours, rank_ref, z0_full, s_full,
                                     y_full, idxs)
            entries.append((host_day.reshape(24, 1), ctx, y_full[idxs]))
        return entries

    def _pack(entries, B):
        batches = []
        for i in range(0, len(entries), B):
            chunk = entries[i:i + B]
            host = np.stack([e[0] for e in chunk]).astype(np.float32)
            ctx = np.stack([e[1] for e in chunk]).astype(np.float32)
            tgt = np.stack([e[2] for e in chunk]).astype(np.float32)
            Bc = host.shape[0]
            batches.append((
                torch.tensor(host, dtype=torch.float32),          # [B,H,1]
                torch.tensor(ctx, dtype=torch.float32),           # [B,H,13]
                torch.tensor(tgt, dtype=torch.float32).reshape(Bc, 24, 1),
                torch.ones(Bc, 24),
            ))
        return batches

    info.s2t_batches = _pack(_day_entries("S2T"), BATCH_DAYS)
    info.s2v_batches = _pack(_day_entries("S2V"), BATCH_DAYS)
    return info


# ----------------------------------------------------------- training --------
def train_variant(variant: str, infos: list[DomainInfo],
                  epochs: int, patience: int) -> tuple[IAHCandidateHead, dict]:
    """Train shared candidate on all 6 domains for one variant (B2/B3)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    head = IAHCandidateHead(D_CORE_CONTEXT, D_MODEL, d_value=D_VALUE, d_sig=D_SIG)
    domains = []
    for info in infos:
        det = det_for_variant(variant, info)
        domains.append(DomainBatch(
            name=f"{info.ds_key}:{info.bb}",
            s2t_batches=info.s2t_batches, s2v_batches=info.s2v_batches,
            domain_det=det))
    trainer = UniversalCoreTrainer(head, seed=SEED)
    report = trainer.train(domains, epochs=epochs, lr=LR, weight_decay=WD,
                           clip=CLIP, patience=patience)
    return head, report


def _best_epoch_index(report: dict) -> int:
    best = report["best_macro_s2v"]
    for i, h in enumerate(report["history"]):
        if abs(h["macro_s2v"] - best) < 1e-6:
            return i
    return int(np.argmin([h["macro_s2v"] for h in report["history"]]))


def reproducibility_check(infos: list[DomainInfo], variant: str,
                          head: IAHCandidateHead, report: dict) -> dict:
    """Protocol §14 GREEN #6: same checkpoint/reference objects -> identical eval.

    Re-evaluates macro S2V with a FRESH head loaded from the best state and a
    RE-SHUFFLED DomainBatch order; must reproduce best_macro_s2v exactly.
    """
    best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head2 = IAHCandidateHead(D_CORE_CONTEXT, D_MODEL, d_value=D_VALUE, d_sig=D_SIG)
    head2.load_state_dict(best_state)
    head2.eval()

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(infos))
    losses = []
    for i in order:
        info = infos[int(i)]
        det_np = det_for_variant(variant, info)
        l, _ = _eval_batch_losses(head2, info.s2v_batches, det_for(det_np, 1))
        losses.append(l)
    macro = float(np.mean([l for l in losses if np.isfinite(l)]))
    ok = abs(macro - report["best_macro_s2v"]) < 1e-4
    return {"ok": bool(ok), "re_eval_macro_s2v": macro,
            "train_best_macro_s2v": report["best_macro_s2v"]}


# ------------------------------------------------- S3-M / S3-C / S4 ----------
def _run_candidate_day(pipe, host_day, hours, z0_full, s_full, y_full, idxs,
                       det_np):
    ctx = build_core_context(host_day, hours, pipe.s1_rank_ref, z0_full, s_full,
                             y_full, idxs)
    det = det_for(det_np, 1)
    with torch.no_grad():
        out = pipe.candidate_head(
            torch.tensor(host_day.reshape(1, 24, 1), dtype=torch.float32),
            torch.tensor(ctx.reshape(1, 24, -1), dtype=torch.float32),
            valid_mask=torch.ones(1, 24), domain_det=det)
    if float(out["scale_valid"][0]) < 0.5:
        return None, None
    s = float(out["s"][0])
    zY = np.arcsinh(y_full[idxs].astype(np.float64) / s)  # no epsilon clamp
    return out, zY


def run_domain_chain(info: DomainInfo, variant: str, best_state: dict,
                     spike_thr: float) -> dict:
    """Per-domain local S3-M/S3-C/S4 on frozen universal candidate (B2/B3)."""
    ds_key, bb = info.ds_key, info.bb
    ts = info.ds["ts"]
    y_full = info.ds["price"].astype(np.float32)
    det_np = det_for_variant(variant, info)

    pipe = HCHV2UniversalPipeline(d_core_context=D_CORE_CONTEXT, d_model=D_MODEL,
                                  alpha=ALPHA, k=None, seed=SEED)
    pipe.candidate_head.load_state_dict(best_state)
    pipe.candidate_head.eval()
    pipe.fit_s1_reference(info.s1_z0, info.s1_hours)
    if variant == "learned_sig":
        # bundle self-describes the zeroed signature (forward still gets zeros)
        pipe._domain_det = np.zeros(8)
        pipe.candidate_head.core_encoder.signature.set_domain_descriptors(np.zeros(8))
    else:
        pipe.fit_s1_signature(info.s1_z0, info.s1_hours)

    # ---- S3-M: memory prefix + forward-validation suffix ----
    s3m_all = sorted(info.exp.dates_in_split("S3M"))
    n_mem = int(len(s3m_all) * S3M_MEM_FRAC)
    mem_dates, val_dates = s3m_all[:n_mem], s3m_all[n_mem:]

    def make_day(d):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            return None
        host_day = info.yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            return None
        hours = ts.iloc[idxs].dt.hour.values
        out, zY = _run_candidate_day(pipe, host_day, hours, info.z0_full,
                                     info.s_full, y_full, idxs, det_np)
        if out is None:
            return None
        return {"date": d, "candidate": out, "target_zY": zY}

    mem_days = [md for md in (make_day(d) for d in mem_dates) if md is not None]
    if not mem_days:
        raise ValueError(f"{ds_key} x {bb} {variant}: no S3M memory days")
    pipe.fit_s3_memory(mem_days)

    val_days = [vd for vd in (make_day(d) for d in val_dates) if vd is not None]
    k = pipe.select_s3m_k(list(K_CANDIDATES), val_days)

    # ---- S3-C: split-conformal calibration ----
    s3c_days = [sd for sd in (make_day(d) for d in sorted(info.exp.dates_in_split("S3C")))
                if sd is not None]
    if not s3c_days:
        raise ValueError(f"{ds_key} x {bb} {variant}: no S3C days")
    q_info = pipe.calibrate_s3c(s3c_days)

    # ---- freeze + round-trip ----
    bundle = pipe.freeze_bundle(dataset_id=ds_key, split_hash=info.exp.split_hash)
    pipe2 = HCHV2UniversalPipeline.from_bundle(bundle)
    q_restored = pipe2.dvg.q if pipe2.dvg is not None else None

    # ---- S4 ----
    s4_days = []
    for d in sorted(info.exp.dates_in_split("S4")):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        host_day = info.yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            continue
        hours = ts.iloc[idxs].dt.hour.values
        ctx = build_core_context(host_day, hours, pipe2.s1_rank_ref,
                                 info.z0_full, info.s_full, y_full, idxs)
        s4_days.append({"date": d, "idxs": idxs, "host_day": host_day,
                        "hours": hours, "ctx": ctx})

    if not s4_days:
        raise ValueError(f"{ds_key} x {bb} {variant}: no S4 days")
    batch_host = torch.tensor(np.stack([d["host_day"].reshape(24, 1) for d in s4_days]),
                              dtype=torch.float32)
    batch_ctx = torch.tensor(np.stack([d["ctx"] for d in s4_days]),
                             dtype=torch.float32)
    domain_det = det_for(det_np, len(s4_days))
    ev = pipe2.predict_s4(batch_host, batch_ctx,
                          valid_mask=torch.ones(len(s4_days), 24),
                          domain_det=domain_det)
    cand_out = ev["candidate"]

    # ---- §2.2 audit fix: full-chain before-freeze vs after-reload ----
    # Same ≥3 fixed S4 queries through pipe (pre-freeze) and pipe2 (from_bundle);
    # compares scale/rank/atoms/shifts/W1/neighbors/intervals/pi/A_hat/q/LCB/
    # action/x_final. Audit only, no math change.
    roundtrip = _roundtrip_check(pipe, pipe2, info, s4_days, det_np, n_check=3)

    # ---- per-day + per-hour evidence ----
    day_rows, hour_rows, final_hour_rows = [], [], []
    gate = {"n_days": 0, "n_execute": 0, "n_harmful": 0, "realized_gain_exec": [],
            "A_true_all": [], "lcb_all": []}
    cand_crps, host_crps = [], []
    ts_all, y_arr, host_arr, final_arr = [], [], [], []
    for i, day in enumerate(s4_days):
        d = day["date"]
        idxs = day["idxs"]
        target_day = y_full[idxs].astype(np.float64)
        scale_v = float(cand_out["scale_valid"][i])
        if scale_v < 0.5:
            day_rows.append({"date": d, "action": "identity",
                             "fallback": "scale_unidentified"})
            continue
        z0 = cand_out["z0"][i].detach().cpu().numpy().ravel()
        s_d = float(cand_out["s"][i])
        zY = np.arcsinh(target_day / max(s_d, 1e-12))
        vm = cand_out["valid_mask"][i].detach().cpu().numpy()
        vm = vm.squeeze(-1) if vm.ndim == 3 else vm
        vm_bool = vm.astype(bool)

        # candidate / host CRPS (protocol §18.1)
        out_i = {kk: (vv[i:i + 1] if isinstance(vv, torch.Tensor) else vv)
                 for kk, vv in cand_out.items()}
        cc = float(iah_crps_loss(out_i, torch.tensor(target_day.reshape(1, 24, 1),
                                                     dtype=torch.float32)))
        hc = float((np.abs(zY - z0)[vm_bool]).mean())
        cand_crps.append(cc)
        host_crps.append(hc)

        action = ev["final_action"][i]
        lcb = float(ev["lcb"][i])
        A_hat = float(ev["A_hat"][i])
        pi_q = ev["pi"][i]
        A_true = estimate_realized_A(z0, zY, pi_q, vm_bool)
        gate["n_days"] += 1
        gate["A_true_all"].append(A_true)
        gate["lcb_all"].append(lcb)
        if action == "execute":
            gate["n_execute"] += 1
            gate["realized_gain_exec"].append(A_true)
            if A_true < 0:
                gate["n_harmful"] += 1

        day_rows.append({
            "date": d, "variant": variant, "domain": f"{ds_key}:{bb}",
            "action": action, "A_hat": A_hat, "q": float(ev["q"]),
            "lcb": lcb, "A_true": A_true,
            "cand_crps": cc, "host_crps": hc, "delta_crps": cc - hc,
            "n_neighbors": k,
        })

        # hourly rows
        xf = ev["x_final"][i].detach().cpu().numpy().ravel()
        w_m = cand_out["w_minus"][i].detach().cpu().numpy().ravel()
        w_z = cand_out["w_zero"][i].detach().cpu().numpy().ravel()
        w_p = cand_out["w_plus"][i].detach().cpu().numpy().ravel()
        mm = cand_out["m_minus"][i].detach().cpu().numpy().ravel()
        mp = cand_out["m_plus"][i].detach().cpu().numpy().ravel()
        host_day = day["host_day"].ravel()
        for h in range(24):
            base = {"variant": variant, "domain": f"{ds_key}:{bb}",
                    "date": d, "hour": h, "timestamp": str(ts.iloc[idxs[h]]),
                    "target": float(target_day[h]),
                    "host_pred": float(host_day[h]),
                    "final_pred": float(xf[h]), "action": action,
                    "cand_crps": cc, "host_crps": hc}
            hour_rows.append({**base, "w_minus": float(w_m[h]), "w_zero": float(w_z[h]),
                              "w_plus": float(w_p[h]), "m_minus": float(mm[h]),
                              "m_plus": float(mp[h])})
            final_hour_rows.append(base)
        ts_all.extend([str(ts.iloc[j]) for j in idxs])
        y_arr.extend(target_day.tolist())
        host_arr.extend(host_day.tolist())
        final_arr.extend(xf.tolist())

    n = max(gate["n_days"], 1)
    gate_metrics = {
        "execute_rate": gate["n_execute"] / n,
        "identity_rate": 1.0 - gate["n_execute"] / n,
        "harmful_rate": (gate["n_harmful"] / gate["n_execute"])
                        if gate["n_execute"] else None,
        "mean_realized_gain_exec": float(np.mean(gate["realized_gain_exec"]))
                                   if gate["realized_gain_exec"] else None,
        "empirical_lcb_coverage": float(np.mean(
            [a >= l for a, l in zip(gate["A_true_all"], gate["lcb_all"])]))
            if gate["A_true_all"] else None,
        "A_true_min/med/max": [float(np.min(gate["A_true_all"])) if gate["A_true_all"] else None,
                               float(np.median(gate["A_true_all"])) if gate["A_true_all"] else None,
                               float(np.max(gate["A_true_all"])) if gate["A_true_all"] else None],
        "lcb_min/med/max": [float(np.min(gate["lcb_all"])) if gate["lcb_all"] else None,
                            float(np.median(gate["lcb_all"])) if gate["lcb_all"] else None,
                            float(np.max(gate["lcb_all"])) if gate["lcb_all"] else None],
    }

    # ---- point metrics ----
    y_arr = np.array(y_arr, dtype=np.float64)
    host_arr = np.array(host_arr, dtype=np.float64)
    final_arr = np.array(final_arr, dtype=np.float64)
    ae_host = np.abs(host_arr - y_arr)
    ae_final = np.abs(final_arr - y_arr)
    smape = float(np.mean(2 * ae_final / (np.abs(y_arr) + np.abs(final_arr) + 1e-12)))
    t_idx = np.concatenate([s4_days[i]["idxs"] for i in range(len(s4_days))
                            if float(cand_out["scale_valid"][i]) >= 0.5])
    prev = t_idx - 168
    ok_naive = prev >= 0
    rmae_final = (float(ae_final[ok_naive].mean() /
                        np.abs(y_full[prev[ok_naive]] - y_arr[ok_naive]).mean())
                  if ok_naive.any() else None)

    # ---- tail metrics (thresholds from S1R reference split, protocol §18.3) ----
    thr = spike_thr if spike_thr is not None else float(np.inf)
    neg = y_arr < 0.0
    spk = y_arr > thr
    tail = {
        "neg_n": int(neg.sum()), "spike_n": int(spk.sum()),
        "mae_on_neg_host": float(ae_host[neg].mean()) if neg.sum() else None,
        "mae_on_neg_final": float(ae_final[neg].mean()) if neg.sum() else None,
        "neg_miss_host": float((host_arr[neg] >= 0).mean()) if neg.sum() else None,
        "neg_miss_final": float((final_arr[neg] >= 0).mean()) if neg.sum() else None,
        "neg_bias_host": float((host_arr[neg] - y_arr[neg]).mean()) if neg.sum() else None,
        "neg_bias_final": float((final_arr[neg] - y_arr[neg]).mean()) if neg.sum() else None,
        "mae_on_spike_host": float(ae_host[spk].mean()) if spk.sum() else None,
        "mae_on_spike_final": float(ae_final[spk].mean()) if spk.sum() else None,
        "high_tail_underbias_host": float(np.maximum(0, y_arr[spk] - host_arr[spk]).mean()) if spk.sum() else None,
        "high_tail_underbias_final": float(np.maximum(0, y_arr[spk] - final_arr[spk]).mean()) if spk.sum() else None,
    }

    dm = dm_test(ae_host, ae_final, lag=24)

    return {
        "domain": f"{ds_key}:{bb}", "ds_key": ds_key, "bb": bb, "variant": variant,
        "selected_k": k, "k_dropped": [kk for kk in K_CANDIDATES if kk > len(mem_days)],
        "n_s3m_mem": len(mem_days), "n_s3m_val": len(val_days), "n_s3c": len(s3c_days),
        "q": float(q_info["q"]), "n_cal": q_info["n"],
        "n_s4_days": len(day_rows), "gate": gate_metrics,
        "point": {"mae_host": float(ae_host.mean()), "mae_final": float(ae_final.mean()),
                  "rmse_host": float(np.sqrt(np.mean(ae_host ** 2))),
                  "rmse_final": float(np.sqrt(np.mean(ae_final ** 2))),
                  "smape_final": smape, "rmae_final": rmae_final,
                  "n_hours": int(len(y_arr))},
        "tail": tail, "dm": dm,
        "cand_crps_mean": float(np.mean(cand_crps)) if cand_crps else None,
        "host_crps_mean": float(np.mean(host_crps)) if host_crps else None,
        "delta_crps_mean": float(np.mean([c - h for c, h in zip(cand_crps, host_crps)]))
                           if cand_crps else None,
        "spike_thr": float(spike_thr),
        "bundle": bundle, "roundtrip_q_restored": q_restored,
        "roundtrip_ok": roundtrip["ok"], "roundtrip_detail": roundtrip,
        "days": day_rows, "hours": hour_rows, "final_hours": final_hour_rows,
    }


def _roundtrip_check(pipe_before, pipe_after, info, s4_days, det_np,
                     n_check: int = 3) -> dict:
    """§2.2 audit fix: full-chain before-freeze vs after-reload on ≥3 fixed queries.

    Runs the SAME fixed S4 queries through pipe_before (pre-freeze) and
    pipe_after (from_bundle) and compares the ENTIRE contract:

        scale -> rank(u) -> atom masses -> shifts -> W1 distances ->
        neighbor IDs -> Down/Up intervals -> final pi -> A_hat -> q -> LCB ->
        execute/Identity -> final raw prediction.

    Rank is checked by building each pipe's context with its OWN s1_rank_ref
    and comparing the u column (context dim 0). Audit only; no math change.
    Returns {"ok": bool, "n_checked": int, "max_abs_diffs": {field: max|Δ|}}.
    """
    checked = 0
    max_diff: dict[str, float] = {}

    def _cmp(name, b, a, atol=1e-5, exact=False):
        if exact:
            eq = (list(b) if not isinstance(b, str) else [b]) == \
                 (list(a) if not isinstance(a, str) else [a])
            max_diff[name] = max(max_diff.get(name, 0.0), 0.0 if eq else 1.0)
            return eq
        b_arr = np.asarray(b, dtype=np.float64).ravel()
        a_arr = np.asarray(a, dtype=np.float64).ravel()
        md = float(np.max(np.abs(b_arr - a_arr))) if b_arr.size else 0.0
        max_diff[name] = max(max_diff.get(name, 0.0), md)
        return md < atol

    for i, day in enumerate(s4_days):
        if checked >= n_check:
            break
        if not np.isfinite(day["host_day"]).all():
            continue
        host_t = torch.tensor(day["host_day"].reshape(1, 24, 1), dtype=torch.float32)
        vm = torch.ones(1, 24)
        det = det_for(det_np, 1)
        y_full = info.ds["price"].astype(np.float32)
        ctx_b = build_core_context(day["host_day"], day["hours"],
                                   pipe_before.s1_rank_ref, info.z0_full,
                                   info.s_full, y_full, day["idxs"])
        ctx_a = build_core_context(day["host_day"], day["hours"],
                                   pipe_after.s1_rank_ref, info.z0_full,
                                   info.s_full, y_full, day["idxs"])
        if not _cmp("rank_u", ctx_b[:, 0], ctx_a[:, 0], atol=1e-6):
            return {"ok": False, "n_checked": checked, "max_abs_diffs": max_diff}
        with torch.no_grad():
            ev_b = pipe_before.predict_s4(
                host_t, torch.tensor(ctx_b.reshape(1, 24, -1), dtype=torch.float32),
                valid_mask=vm, domain_det=det)
            ev_a = pipe_after.predict_s4(
                host_t, torch.tensor(ctx_a.reshape(1, 24, -1), dtype=torch.float32),
                valid_mask=vm, domain_det=det)

        ob, oa = ev_b["candidate"], ev_a["candidate"]
        ok = True
        ok &= _cmp("scale_s", ob["s"][0], oa["s"][0])
        ok &= _cmp("scale_valid", ob["scale_valid"][0], oa["scale_valid"][0])
        for f in ("z0", "w_minus", "w_zero", "w_plus", "m_minus", "m_plus",
                  "x_identity"):
            ok &= _cmp(f"atom_{f}", ob[f][0], oa[f][0])
        ok &= _cmp("neighbor_ids", ev_b["neighbors"][0], ev_a["neighbors"][0],
                   exact=True)
        ok &= _cmp("w1_dists", ev_b["neighbor_distances"][0],
                   ev_a["neighbor_distances"][0])
        pb, pa = ev_b["proposals"][0], ev_a["proposals"][0]
        ok &= _cmp("I_down", pb["I_down"] or (-1,), pa["I_down"] or (-1,),
                   exact=True)
        ok &= _cmp("I_up", pb["I_up"] or (-1,), pa["I_up"] or (-1,), exact=True)
        ok &= _cmp("final_pi", ev_b["pi"][0], ev_a["pi"][0])
        ok &= _cmp("A_hat", ev_b["A_hat"][0], ev_a["A_hat"][0])
        ok &= _cmp("q", ev_b["q"], ev_a["q"])
        ok &= _cmp("lcb", ev_b["lcb"][0], ev_a["lcb"][0])
        ok &= _cmp("action", ev_b["final_action"][0], ev_a["final_action"][0],
                   exact=True)
        ok &= _cmp("x_final", ev_b["x_final"][0], ev_a["x_final"][0])
        if not ok:
            return {"ok": False, "n_checked": checked, "max_abs_diffs": max_diff}
        checked += 1

    return {"ok": checked >= n_check, "n_checked": checked,
            "max_abs_diffs": max_diff}


# -------------------------------------------------------------- artifacts ----
def write_artifacts(out_dir: Path, infos, reports, results, run_config, extra):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False, allow_unicode=True)
    with open(out_dir / "git_commit.txt", "w") as f:
        f.write(_git_head() + "\n")

    # domain_manifest.csv
    dom_rows = []
    for info in infos:
        exp = info.exp
        row = {"domain": f"{info.ds_key}:{info.bb}", "dataset": info.ds_key,
               "backbone": info.bb}
        for s in SPLIT_7:
            row[f"n_dates_{s}"] = len(exp.dates_in_split(s))
            row[f"n_hours_{s}"] = len(exp.valid_indices_in_split(s))
        row["n_excluded_dates"] = exp.n_excluded_dates
        row["n_original_hours"] = exp.n_original_hours
        row["split_hash"] = exp.split_hash
        dom_rows.append(row)
    pd.DataFrame(dom_rows).to_csv(out_dir / "domain_manifest.csv", index=False)

    # split_manifest.json
    split_man = {}
    for info in infos:
        exp = info.exp
        bounds = {}
        for s in SPLIT_7:
            dates = sorted(exp.dates_in_split(s))
            bounds[s] = {"start": dates[0] if dates else None,
                         "end": dates[-1] if dates else None,
                         "n_dates": len(dates),
                         "n_hours": len(exp.valid_indices_in_split(s))}
        split_man[f"{info.ds_key}:{info.bb}"] = {
            "dataset": info.ds_key, "backbone": info.bb, "bounds": bounds,
            "excluded_dates": exp.excluded_dates,
            "n_original_hours": exp.n_original_hours,
            "split_hash": exp.split_hash,
        }
    with open(out_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_man, f, indent=2, default=str)

    # host_cache_manifest.csv
    cache_rows = [extra["cache_records"][f"{i.ds_key}:{i.bb}"] for i in infos]
    with open(out_dir / "host_cache_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cache_rows[0].keys()))
        w.writeheader()
        w.writerows(cache_rows)

    # signature_by_domain.csv
    sig_rows = []
    for info in infos:
        d = info.det_real
        sig_rows.append({"domain": f"{info.ds_key}:{info.bb}",
                         "q05": d[0], "q25": d[1], "q50": d[2], "q75": d[3],
                         "q95": d[4], "iqr": d[5], "E|z0|": d[6], "P(z0<0)": d[7]})
    pd.DataFrame(sig_rows).to_csv(out_dir / "signature_by_domain.csv", index=False)

    # train curves + val crps + mass/shift stats (per variant)
    for variant, report in reports.items():
        curve, val, ms = [], [], []
        for h in report["history"]:
            curve.append({
                "epoch": h["epoch"], "macro_s2v": h["macro_s2v"],
                "worst_s2v": h["worst_s2v"], "train_loss": h["train_loss"],
                "mean_grad_norm": h["grad_health"]["mean_grad_norm"],
                "nan_inf_batches": h["grad_health"]["nan_inf_batches"],
                "scale_unidentified_days": h["grad_health"]["scale_unidentified_days"],
                "updates_per_domain": h["updates_per_domain"],
                **{k: h["health"].get(k) for k in
                   ("mean_w_minus", "mean_w_zero", "mean_w_plus", "mass_entropy",
                    "frac_m_minus_alive", "frac_m_plus_alive", "med_m_minus",
                    "p95_m_minus", "med_m_plus", "p95_m_plus",
                    "mean_abs_delta_gamma", "mean_abs_beta")},
            })
            for dom in h["per_domain"]:
                val.append({"epoch": h["epoch"], "domain": dom,
                            "L_g": h["per_domain"][dom],
                            "host_g": h["host_baseline"][dom],
                            "delta": h["delta"].get(dom)})
            ms.append({"epoch": h["epoch"],
                       **{k: h["health"].get(k) for k in
                          ("mean_w_minus", "mean_w_zero", "mean_w_plus", "mass_entropy",
                           "frac_m_minus_alive", "frac_m_plus_alive", "med_m_minus",
                           "p95_m_minus", "med_m_plus", "p95_m_plus",
                           "mean_abs_delta_gamma", "mean_abs_beta")}})
        pd.DataFrame(curve).to_csv(out_dir / f"train_curve_{variant}.csv", index=False)
        pd.DataFrame(val).to_csv(out_dir / f"val_crps_by_domain_{variant}.csv", index=False)
        pd.DataFrame(ms).to_csv(out_dir / f"mass_shift_stats_{variant}.csv", index=False)

    # checkpoints
    ckpt = {}
    for variant in reports:
        head = extra[f"head_{variant}"]
        state = {k: v.clone() for k, v in head.state_dict().items()}
        torch.save({
            "variant": variant,
            "core_config": {"d_core_context": D_CORE_CONTEXT, "d_model": D_MODEL,
                            "d_value": D_VALUE, "d_sig": D_SIG, "seed": SEED},
            "model_state": state,
            "best_macro_s2v": reports[variant]["best_macro_s2v"],
            "epochs_run": reports[variant]["epochs_run"],
            "code_commit": _git_head(),
        }, out_dir / f"checkpoint_{variant}.pt")
        ckpt[f"checkpoint_{variant}.pt"] = {
            "param_hash": sha256_obj({k: v.detach().cpu().numpy().tobytes()
                                      for k, v in state.items()}),
            "best_macro_s2v": reports[variant]["best_macro_s2v"],
        }
    for variant, dom_results in results.items():
        for r in dom_results:
            bpath = out_dir / f"checkpoint_{variant}_{r['ds_key']}_{r['bb']}.pt"
            r["bundle"].save(str(bpath))
            ckpt[bpath.name] = {"bundle_hash": r["bundle"].hash()}
    with open(out_dir / "checkpoint_hashes.json", "w") as f:
        json.dump(ckpt, f, indent=2)

    # candidate_s4.parquet + final_s4.parquet + gate_evidence.jsonl
    cand_rows, final_rows, gate_lines = [], [], []
    for variant, dom_results in results.items():
        for r in dom_results:
            cand_rows.extend(r["hours"])
            final_rows.extend(r["final_hours"])
            gate_lines.extend(r["days"])
    pd.DataFrame(cand_rows).to_parquet(out_dir / "candidate_s4.parquet")
    pd.DataFrame(final_rows).to_parquet(out_dir / "final_s4.parquet")
    with open(out_dir / "gate_evidence.jsonl", "w", encoding="utf-8") as f:
        for line in gate_lines:
            f.write(json.dumps(line, default=str) + "\n")

    # metrics_by_domain_host.csv
    m_rows = []
    for variant, dom_results in results.items():
        for r in dom_results:
            g = r["gate"]
            m_rows.append({
                "variant": variant, "domain": r["domain"],
                "mae_host": r["point"]["mae_host"], "mae_final": r["point"]["mae_final"],
                "rmse_host": r["point"]["rmse_host"], "rmse_final": r["point"]["rmse_final"],
                "smape_final": r["point"]["smape_final"],
                "rmae_final": r["point"]["rmae_final"],
                "cand_crps": r["cand_crps_mean"], "host_crps": r["host_crps_mean"],
                "delta_crps": r["delta_crps_mean"],
                "execute_rate": g["execute_rate"], "harmful_rate": g["harmful_rate"],
                "mean_gain_exec": g["mean_realized_gain_exec"],
                "lcb_coverage": g["empirical_lcb_coverage"],
                "selected_k": r["selected_k"], "q": r["q"],
                "roundtrip_ok": r["roundtrip_ok"],
            })
    pd.DataFrame(m_rows).to_csv(out_dir / "metrics_by_domain_host.csv", index=False)

    # dm_tests.csv
    dm_rows = []
    for variant, dom_results in results.items():
        for r in dom_results:
            dm_rows.append({"variant": variant, "domain": r["domain"],
                            "dm_stat": r["dm"]["dm_stat"], "p_value": r["dm"]["p_value"],
                            "mean_gain": r["dm"]["mean_gain"]})
    pd.DataFrame(dm_rows).to_csv(out_dir / "dm_tests.csv", index=False)


# ----------------------------------------------------------------- verdict ----
def compute_verdict(reports, results, repro) -> str:
    """Protocol §14 go/no-go classification (GREEN/YELLOW/RED)."""
    lines = ["# R1A VERDICT — first-round universal HCH-v2 (protocol v0.1)", ""]
    lines.append(f"- generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- git commit: `{_git_head()}`")
    lines.append("")

    for variant in VARIANTS:
        rep = reports[variant]
        bi = _best_epoch_index(rep)
        h = rep["history"][bi]
        host_macro = float(np.mean(list(h["host_baseline"].values())))
        n_improved = sum(1 for v in h["delta"].values() if v < 0)
        lines.append(f"## {VARIANT_LABELS[variant]} = `{variant}` (best epoch {bi})")
        lines.append("")
        lines.append(f"- best macro S2V IAH-CRPS = `{rep['best_macro_s2v']:.5f}`")
        lines.append(f"- worst domain L_worst at best = `{rep['worst_s2v_at_best']:.5f}`")
        lines.append(f"- frozen-host macro baseline = `{host_macro:.5f}` "
                     f"(delta = {rep['best_macro_s2v'] - host_macro:+.5f})")
        lines.append(f"- domains with Δg<0 at best = `{n_improved}/6`")
        for dom, d in h["delta"].items():
            lines.append(f"    Δ({dom}) = {d:+.5f}")
        hl = h["health"]
        lines.append(f"- health: mean_w0={hl['mean_w_zero']:.4f} "
                     f"entropy={hl['mass_entropy']:.4f} "
                     f"frac_m-_alive={hl['frac_m_minus_alive']:.4f} "
                     f"frac_m+_alive={hl['frac_m_plus_alive']:.4f} "
                     f"|Δγ|={hl['mean_abs_delta_gamma']:.5f} |β|={hl['mean_abs_beta']:.5f}")
        lines.append(f"- reproducibility (shuffled re-eval) = `{repro[variant]}`")
        lines.append("")

    # verdict on Learned+DetSig (old Universal-Sig); R1A ran on both variants
    rep = reports["learned_det_sig"]
    bi = _best_epoch_index(rep)
    h = rep["history"][bi]
    host_macro = float(np.mean(list(h["host_baseline"].values())))
    n_improved = sum(1 for v in h["delta"].values() if v < 0)
    health = h["health"]

    red = []
    if not (np.isfinite(rep["best_macro_s2v"])
            and all(np.isfinite(x["train_loss"]) for x in rep["history"])):
        red.append("macro S2V / train losses not finite")
    if rep["best_macro_s2v"] >= host_macro:
        red.append(f"macro S2V {rep['best_macro_s2v']:.5f} does not beat host "
                   f"baseline {host_macro:.5f}")
    tr = [x["train_loss"] for x in rep["history"]]
    va = [x["macro_s2v"] for x in rep["history"]]
    if len(tr) >= 3 and tr[-1] < tr[0] and va[-1] > va[0] + 1e-4:
        red.append("validation worsens while train improves (overfitting pattern)")
    if not repro["learned_det_sig"]["ok"]:
        red.append("S2V eval not reproducible after reload (order effect)")
    for r in results["learned_det_sig"]:
        if not r["roundtrip_ok"]:
            red.append(f"{r['domain']}: S4 chain NOT reproduced after freeze/reload")
        if r["roundtrip_q_restored"] != r["q"]:
            red.append(f"{r['domain']}: DVG q not restored ({r['q']} vs "
                       f"{r['roundtrip_q_restored']})")

    yellow = []
    deltas = h["delta"]
    if n_improved > 0 and n_improved < 6 and any(v > 0.05 for v in deltas.values()):
        yellow.append("some domain severely sacrificed (Δg large positive while others improve)")
    nem = {k: v for k, v in deltas.items() if "NEM_SA1" in k}
    if any(v > 0 for v in nem.values()) and n_improved > 0:
        yellow.append("NEM_SA1 sacrificed despite overall improvement")
    s_tr = [x["macro_s2v"] for x in reports["learned_det_sig"]["history"]]
    n_tr = [x["macro_s2v"] for x in reports["learned_sig"]["history"]]
    if len(s_tr) >= 3 and np.std(s_tr) > 2 * np.std(n_tr):
        yellow.append("Learned+DetSig much less stable than LearnedSig")

    collapsed = (health["mean_w_zero"] > 0.999
                 or (health["frac_m_minus_alive"] < 0.01
                     and health["frac_m_plus_alive"] < 0.01))

    if red:
        verdict, reasons = "RED", red
    elif yellow and n_improved == 0:
        verdict, reasons = "RED", ["no domain improved"] + yellow
    elif yellow:
        verdict, reasons = "YELLOW", yellow
    else:
        green_ok = (n_improved >= 2 and not collapsed
                    and rep["best_macro_s2v"] < host_macro
                    and all(np.isfinite(x["train_loss"]) for x in rep["history"]))
        if green_ok:
            verdict, reasons = "GREEN", ["all GREEN criteria satisfied"]
        else:
            verdict = "YELLOW"
            reasons = (["n_improved < 2"] if n_improved < 2 else []) + \
                      (["mass collapsed"] if collapsed else [])

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"### **{verdict}**")
    lines.append("")
    lines.append("Why:")
    for r in reasons:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("Hypotheses tested (protocol §2):")
    lines.append("- Q1 (one candidate across heterogeneous markets): "
                 f"{'supports' if n_improved >= 2 else 'not yet supported'}")
    lines.append("- Q2 (8-dim frozen S1R descriptor beyond the learned daily "
                 "signature): "
                 f"best_macro LearnedSig={reports['learned_sig']['best_macro_s2v']:.5f} "
                 f"vs Learned+DetSig="
                 f"{reports['learned_det_sig']['best_macro_s2v']:.5f} — the "
                 "frozen descriptor adds no measurable gain. Audit §2.1: this "
                 "is \"simplifiable\", NOT \"Data Signature has no gain\".")
    lines.append("- Q3/Q4 (Local-Core / unseen DK1): OUT OF SCOPE for R1A (R1B / §17)")
    lines.append("")
    lines.append("Server U0 authorization: "
                 f"{'AUTHORIZED' if verdict == 'GREEN' else 'NOT authorized'}")
    return "\n".join(lines)


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None,
                    help="artifact dir; default R1A_<timestamp>")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--phase", choices=["cache", "all"], default="all")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else \
        HERE / "results" / f"R1A_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[R1A] out dir: {out_dir}")

    # ---- cache phase (protocol §6) ----
    cache_records = {}
    for ds_key, bb in DOMAINS:
        print(f"[R1A/cache] {ds_key} x {bb} ...", flush=True)
        rec = cache_one(ds_key, bb, seed=SEED)
        cache_records[f"{ds_key}:{bb}"] = rec
        print(f"  OK split_hash={rec['split_hash']} n_S4={rec['n_S4']}")
    print(f"[R1A/cache] {len(cache_records)}/6 caches OK")

    if args.phase == "cache":
        print("[R1A] phase=cache done. Run without --phase for full R1A.")
        return

    # ---- domain preparation ----
    infos = [prepare_domain(ds_key, bb) for ds_key, bb in DOMAINS]
    for info in infos:
        print(f"[R1A/prep] {info.ds_key} x {info.bb}: S2T={len(info.s2t_batches)} "
              f"S2V={len(info.s2v_batches)} det={np.round(info.det_real, 4).tolist()}")

    # ---- train both variants ----
    heads, reports, repro = {}, {}, {}
    for variant in VARIANTS:
        print(f"\n[R1A/train] variant={variant} ...")
        head, report = train_variant(variant, infos, args.epochs, args.patience)
        heads[variant] = head
        reports[variant] = report
        rp = reproducibility_check(infos, variant, head, report)
        repro[variant] = rp
        print(f"[R1A/train] {variant}: best_macro_s2v={report['best_macro_s2v']:.5f} "
              f"worst={report['worst_s2v_at_best']:.5f} epochs_run={report['epochs_run']} "
              f"| repro_ok={rp['ok']}")

    # ---- per-domain S3/S4 ----
    results = {}
    spike_thr = {}
    for info in infos:
        s1_p = info.s1_prices
        spike_thr[f"{info.ds_key}:{info.bb}"] = float(
            np.quantile(s1_p, 0.99)) if len(s1_p) else None
    for variant in VARIANTS:
        results[variant] = []
        best_state = {k: v.clone() for k, v in heads[variant].state_dict().items()}
        for info in infos:
            dom_key = f"{info.ds_key}:{info.bb}"
            print(f"[R1A/s4] {variant} {dom_key} ...", flush=True)
            res = run_domain_chain(info, variant, best_state, spike_thr[dom_key])
            results[variant].append(res)
            g = res["gate"]
            print(f"   k={res['selected_k']} q={res['q']:.4f} "
                  f"execute={g['execute_rate']:.3f} harmful={g['harmful_rate']} "
                  f"mae host={res['point']['mae_host']:.3f} "
                  f"final={res['point']['mae_final']:.3f} "
                  f"delta_crps={res['delta_crps_mean']:+.4f} "
                  f"roundtrip={res['roundtrip_ok']}")

    # ---- write artifacts ----
    run_config = {
        "protocol": "hch_v2_first_round_training_protocol_v0.1_2026-08-13.md",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "domains": DOMAINS,
        "variants": {
            "learned_sig": "LearnedSig (=old Universal-NoSig): deterministic "
                           "Data Signature zeroed (np.zeros(8)); learned per-day "
                           "pool + FiLM identical to the other variant. "
                           "Tentative main (review §2.1).",
            "learned_det_sig": "Learned+DetSig (=old Universal-Sig): frozen S1R "
                               "deterministic descriptor [8] as forward context. "
                               "Demoted to ablation.",
        },
        "variant_rename_note": "R1B naming audit §2.1: old nosig->learned_sig, "
                               "old sig->learned_det_sig; PlainCore is R1B-only",
        "d_core_context": D_CORE_CONTEXT, "d_model": D_MODEL, "d_sig": D_SIG,
        "d_value": D_VALUE,
        "optimizer": "AdamW", "lr": LR, "weight_decay": WD, "clip": CLIP,
        "epochs": args.epochs, "patience": args.patience,
        "batch_days": BATCH_DAYS, "host_seed": SEED, "hch_seed": SEED,
        "equal_domain": "K=median(N_g) updates/domain/epoch (P0-C)",
        "checkpoint_selection": "macro S2V IAH-CRPS (P0-3/G0-3)",
        "s3m_mem_frac": S3M_MEM_FRAC, "k_candidates": list(K_CANDIDATES),
        "dvg_alpha": ALPHA,
        "spike_threshold_source": "S1R reference split p99",
        "reproducibility": repro,
    }
    write_artifacts(out_dir, infos, reports, results, run_config, {
        "cache_records": cache_records,
        "head_learned_sig": heads["learned_sig"],
        "head_learned_det_sig": heads["learned_det_sig"],
    })

    verdict = compute_verdict(reports, results, repro)
    with open(out_dir / "VERDICT.md", "w", encoding="utf-8") as f:
        f.write(verdict)
    print(f"\n[R1A] artifacts written to {out_dir}")
    print(verdict)


if __name__ == "__main__":
    main()
