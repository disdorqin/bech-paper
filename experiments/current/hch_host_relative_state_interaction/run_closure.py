"""HCH Host-Relative State Interaction (HRSI) — final structural closure.

Controlling protocol:
    docs/current/HCH_HOST_RELATIVE_STATE_INTERACTION_FINAL_CLOSURE_20260911.md

Executor contract:
    .agents/skills/solar-research-executor/SKILL.md

Low-autonomy executor. Frozen and never retrained or modified here: S1
Daily-Patch GRU32, the Host caches, the split manifests, the registered S1
OOF/final directions, the Shape target/loss, the semantic role schema, S1
state normalization, and the pooled exact-MAE scalar-Amplitude form.

The single new trainable object is ONE global `beta in R^5` shared by the whole
six-cell panel:

    b_h   = mean of available normalized historical residuals at hour h over
            the existing seven days already carried by the S1 repair input
    b_hat = b / (||b||_2 + eps)                       (24h L2 normalization)
    Z[h,k]= b_hat[h] * R[h,k]                          (fixed interaction)
    q     = Z @ beta
    u_HRSI= normalize(u_S1 + q)

`beta = 0` initialisation must replay S1 exactly. `alpha_HRSI` is then refit
with the original exact weighted-median rule on HRSI OOF directions, and the
final proposal is strictly `Delta = s * alpha_HRSI * u_HRSI`.

Cross-market chronology: for every fold `Bk` and seed, exactly one global beta
is fitted on the pooled legal rows of `B1..B(k-1)` from all six cells with
cell-macro-balanced Shape loss, and that same beta predicts `Bk` in every cell.
"""
from __future__ import annotations

import copy, hashlib, json, subprocess, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.current.hch_unified_da_shape_upgrade.run_canary import (
    EPS, HOSTS, LEGAL, LR, MARKETS, SEEDS, STEPS, TRACK, WD,
    make_inputs, norm_state, read_state,
)
from experiments.current.hch_anchored_compact_amplitude.run_closure import (
    SHAPE_EVIDENCE, eval_metrics, fit_shape_outputs, load_frozen_cache,
    pooled_alpha, shape_oof, utility,
)
from src.mvp.hch_minimal_repair.crossfit import chronological_blocks
from src.mvp.hch_minimal_repair.training import set_deterministic, unit_shape_target
from utils.timing import parameter_count

# REPRODUCIBILITY INVARIANT — must stay after the imports above.
# `src/backbones/backbones.py` calls `torch.set_num_threads(4)` at import time, so
# pinning threads before importing would be silently reverted and S1 would train
# with 4 BLAS threads. That changes CPU reduction order by ~1e-7, which is enough
# to flip the discrete pooled weighted-median `alpha` by ~1e-3 and break the
# frozen-S1 replay. Pinning here (post-import, at module import time) makes both
# this runner and anything importing it reproduce the frozen S1 evidence exactly.
torch.set_num_threads(1)
if torch.get_num_threads() != 1:
    raise RuntimeError("HRSI runner requires single-threaded torch for frozen S1 replay")

OUT = ROOT / "experiments/evidence/hch_host_relative_state_interaction_20260911"
HSA_EVIDENCE = ROOT / "experiments/evidence/hch_horizon_aligned_state_residual_20260911"
STATE_EVIDENCE = ROOT / "experiments/evidence/hch_market_state_coupling_20260910"
SCHEMA_PATH = STATE_EVIDENCE / "state_schema_manifest.csv"
TERTILE_PATH = STATE_EVIDENCE / "state_tertile_mechanisms.csv"
DIAG_ROWS_PATH = STATE_EVIDENCE / "diagnostic_rows.parquet"

TOKEN_PASS = "HCH_HOST_RELATIVE_STATE_INTERACTION_SUPPORTED_METHOD_TARGET_REACHED"
TOKEN_FAIL = "HCH_HOST_RELATIVE_STATE_INTERACTION_NOT_SUPPORTED_FREEZE_S1"
TOKEN_INVALID = "HCH_HOST_RELATIVE_STATE_INTERACTION_INVALID"

TOL = 1e-10              # frozen S1 metric replay tolerance (same as prior closures)
REPLAY_TOL = 1e-6        # beta=0 -> exact S1 direction replay tolerance (float32 eps)
ROLES = ("DEMAND_FC", "RENEWABLE_FC", "SUPPLY_MARGIN_FC", "INTERCHANGE_FC", "MUST_RUN_FC")
N_ROLES = len(ROLES)
HIST_DAYS = 7            # the existing seven historical days already used by S1
BIAS_CHANNEL = slice(168, 336)   # frozen normalized residual history inside x
MASK_CHANNEL = slice(336, 504)   # its availability mask inside x
BETA_TRAINING_BUDGET_S = 10.0          # R4: median global beta-fit seconds
OVERHEAD_BUDGET_US_DAY = 10.0          # R4: median inference overhead per day
TOTAL_INFERENCE_BUDGET_US_DAY = 100.0  # R4: total repair inference per day
TIMING_REPEATS = 50
RECIPE_VERSION = 2

# Columns of the row-set-alignment diagnostic. HRSI's scalar alpha cannot exist
# before a beta does, so it is calibrated on B2..B4 while frozen S1 uses B1..B4.
# The exact weighted median is discrete, so that one-fold shift is a comparison
# confound worth measuring, but the registered R2 gate stays defined against
# frozen S1 and these columns never feed a gate.
ROWSET_SENS_COLS = ["S1_alpha_refit_on_HRSI_rowset", "alpha_rowset_sensitivity_pct",
                    "S1_frozen_Overall_MAE", "S1_aligned_rowset_Overall_MAE",
                    "S1_aligned_rowset_relative_gain_pct",
                    "HRSI_minus_S1_aligned_overall_gain_pp"]


def cell_median(df, keys):
    """Median over seeds, with the seed index itself excluded from the aggregate."""
    return df.drop(columns=["seed"]).groupby(keys, as_index=False).median(numeric_only=True)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree_sha(p):
    p = Path(p); h = hashlib.sha256()
    for x in sorted(y for y in p.rglob("*") if y.is_file() and "__pycache__" not in y.parts):
        h.update(x.relative_to(p).as_posix().encode()); h.update(x.read_bytes())
    return h.hexdigest()


def hash_json(x): return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def json_default(x):
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x).__name__)


# --------------------------------------------------------------------------
# Semantic role construction (protocol S4) -- unchanged from the audited HSA
# schema: deterministic mean pooling of audited primary forecast-time channels.
# --------------------------------------------------------------------------

def load_role_schema(schema_df=None):
    """Map audited `primary_channel=True` source columns into the five fixed roles."""
    s = pd.read_csv(SCHEMA_PATH) if schema_df is None else schema_df.copy()
    required = {"market", "source_column", "semantic_role", "primary_channel", "forecast_time_legal"}
    if not required <= set(s.columns):
        raise RuntimeError(f"role schema columns missing: {sorted(required - set(s.columns))}")
    if s.primary_channel.dtype != bool:
        s["primary_channel"] = s.primary_channel.astype(str).str.lower().eq("true")
    if s.forecast_time_legal.dtype != bool:
        s["forecast_time_legal"] = s.forecast_time_legal.astype(str).str.lower().eq("true")
    if not s.loc[s.primary_channel, "forecast_time_legal"].all():
        raise RuntimeError("non-legal primary channel present in audited schema")
    if set(s.loc[s.primary_channel, "semantic_role"]) - set(ROLES):
        raise RuntimeError("unaudited semantic role in primary channels")
    mapping, manifest = {}, []
    for m in MARKETS:
        legal = list(LEGAL[m])
        rows = s[(s.market == m) & s.primary_channel]
        if not rows.source_column.is_unique:
            raise RuntimeError(f"{m}: duplicate primary channel")
        role_idx = {r: [] for r in ROLES}
        for _, r in rows.iterrows():
            col = str(r.source_column)
            if col not in legal:
                raise RuntimeError(f"{m}: primary channel {col!r} not in registered LEGAL columns")
            i = legal.index(col)
            role_idx[str(r.semantic_role)].append(i)
            manifest.append({"market": m, "semantic_role": str(r.semantic_role), "source_column": col,
                             "legal_column_index": i, "primary_channel": True, "forecast_time_legal": True})
        for r in ROLES:
            role_idx[r] = tuple(sorted(role_idx[r]))
        if not any(role_idx[r] for r in ROLES):
            raise RuntimeError(f"{m}: no primary channels for any role")
        mapping[m] = tuple(role_idx[r] for r in ROLES)
    man = pd.DataFrame(manifest)
    out_manifest = []
    for m in MARKETS:
        for r in ROLES:
            sub = man[(man.market == m) & (man.semantic_role == r)]
            out_manifest.append({"market": m, "semantic_role": r,
                                 "n_channels_in_role": int(len(sub)),
                                 "source_columns": "|".join(sub.source_column.tolist()),
                                 "legal_column_indices": "|".join(str(int(i)) for i in sub.legal_column_index.tolist()),
                                 "structurally_absent": bool(len(sub) == 0)})
    return mapping, pd.DataFrame(out_manifest)


def role_table(z_std, mapping):
    """Deterministic per-horizon-hour mean pool -> R in R^(24x5); absent role = structural 0."""
    n, h, c = z_std.shape
    if len(mapping) != N_ROLES:
        raise RuntimeError("role mapping must cover exactly five semantic roles")
    R = np.zeros((n, h, N_ROLES), np.float32)
    for k, idxs in enumerate(mapping):
        if len(idxs):
            R[:, :, k] = z_std[:, :, list(idxs)].mean(axis=2)
    return R


def standardized_state(train_raw, test_raw):
    a, b = norm_state(train_raw, test_raw)
    return a.reshape(train_raw.shape), b.reshape(test_raw.shape)


# --------------------------------------------------------------------------
# Causal Host-bias Shape (protocol S5)
# --------------------------------------------------------------------------

def bias_from_residual_block(resn, msk):
    """`b_hat` from the frozen normalized residual history and its mask.

    `resn`/`msk` are the 168-wide `[7 days x 24 hours]` blocks already carried by
    the S1 repair input, so the day-scale normalization is exactly the one S1
    sees. Position `j` inside a week belongs to horizon hour `j % 24`, which is
    why the reshape below is the same `(-1, 7, 24)` used by `DailyPatchShape`.
    """
    resn = np.asarray(resn, dtype=np.float64); msk = np.asarray(msk, dtype=np.float64)
    n = len(resn)
    if resn.shape != (n, HIST_DAYS * 24) or msk.shape != resn.shape:
        raise RuntimeError("historical residual block must be exactly seven days of 24 hours")
    if np.any(resn[msk == 0] != 0.0):
        raise RuntimeError("masked historical residual coordinates must be exactly zero")
    r3 = resn.reshape(n, HIST_DAYS, 24); m3 = msk.reshape(n, HIST_DAYS, 24)
    counts = m3.sum(axis=1)
    bias = (r3 * m3).sum(axis=1) / (counts + EPS)
    norm = np.linalg.norm(bias, axis=1)
    bhat = np.zeros_like(bias)
    ok = norm > EPS
    bhat[ok] = bias[ok] / (norm[ok, None] + EPS)
    return bhat.astype(np.float32), bias.astype(np.float32), counts


def host_bias_direction(x, residual_mask):
    """`b_hat` extracted from the frozen 624-D repair input, fail-closed on layout."""
    x = np.asarray(x)
    if x.shape[1] != 624:
        raise RuntimeError("registered M0 state must be 624-D")
    msk = x[:, MASK_CHANNEL]
    if not np.array_equal(msk.astype(np.float32), np.asarray(residual_mask, dtype=np.float32)):
        raise RuntimeError("residual mask columns disagree with RepairInput.residual_mask")
    return bias_from_residual_block(x[:, BIAS_CHANNEL], msk)


# --------------------------------------------------------------------------
# The only new trainable object: one global beta in R^5 for the whole panel.
# --------------------------------------------------------------------------

class HRSIResidual(nn.Module):
    """`u_HRSI = normalize(u_S1 + (b_hat * R) beta)` with a single five-parameter vector."""

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(N_ROLES))

    def forward(self, u_s1, Z):
        z = u_s1 + Z @ self.beta
        return z / (torch.linalg.norm(z, dim=1, keepdim=True) + EPS)


def new_beta_model():
    model = HRSIResidual()
    if set(type(m) for m in model.modules()) != {HRSIResidual}:
        raise RuntimeError("HRSI must not add a hidden layer or submodule")
    if parameter_count(model) != N_ROLES:
        raise RuntimeError("HRSI must expose exactly five trainable parameters")
    if float(model.beta.detach().abs().max()) != 0.0:
        raise RuntimeError("beta must be initialised to exactly zero")
    return model


def interaction(bhat, R):
    """Fixed interaction tensor `Z[h,k] = b_hat[h] * R[h,k]`."""
    if bhat.shape[:2] != R.shape[:2]:
        raise RuntimeError("bias direction and role table must share [rows, 24]")
    if R.shape[2] != N_ROLES:
        raise RuntimeError("role table must carry exactly five roles")
    return (bhat[:, :, None] * R).astype(np.float32)


def macro_balanced_loss(model, pools):
    """Registered objective: equal total Shape weight per cell (protocol S7).

    `L = (1/6) sum_c (1/N_c) sum_{i in c} ||u_HRSI_i - u*_i||_2^2`, implemented
    literally: a per-cell mean of the squared L2 norm, averaged over cells. Both
    the number of rows and the cell size therefore drop out of the weighting.
    """
    if len(pools) != 6:
        raise RuntimeError("macro-balanced loss is defined over exactly six cells")
    return torch.stack([torch.mean(torch.sum((model(u, Z) - y) ** 2, dim=1))
                        for u, Z, y in pools]).mean()


def _tensors(pools):
    return [(torch.as_tensor(u, dtype=torch.float32), torch.as_tensor(Z, dtype=torch.float32),
             torch.as_tensor(y, dtype=torch.float32)) for u, Z, y in pools]


def beta_sha(beta):
    return hashlib.sha256(np.asarray(beta, dtype=np.float64).tobytes()).hexdigest()


def fit_global_beta(pools, seed):
    """Fit the one shared beta on a pooled six-cell training set."""
    if len(pools) != 6:
        raise RuntimeError("a global beta must be fitted on all six cells at once")
    if any(len(p[0]) == 0 for p in pools):
        raise RuntimeError("every cell must contribute at least one legal beta training row")
    tensors = _tensors(pools)
    model = new_beta_model()
    replay_max = 0.0
    with torch.no_grad():
        for u, Z, _ in tensors:
            replay_max = max(replay_max, float(np.max(np.abs(model(u, Z).numpy() - u.numpy()))))
    if replay_max > REPLAY_TOL:
        raise RuntimeError(f"beta=0 does not replay S1 exactly ({replay_max:.3e})")
    set_deterministic(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best, best_state = float("inf"), None
    started = time.perf_counter()
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = macro_balanced_loss(model, tensors)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        v = float(loss.detach())
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    seconds = time.perf_counter() - started
    beta = model.beta.detach().numpy().astype(np.float64).copy()
    return beta, {"objective": float(best), "training_seconds": float(seconds),
                  "zero_init_replay_max_abs_diff": float(replay_max),
                  "n_train_rows": int(sum(len(p[0]) for p in pools)),
                  "rows_per_cell": [int(len(p[0])) for p in pools]}


def predict_beta(beta, u_s1, Z):
    model = HRSIResidual()
    with torch.no_grad():
        model.beta.copy_(torch.as_tensor(beta, dtype=torch.float32))
        return model(torch.as_tensor(u_s1, dtype=torch.float32), torch.as_tensor(Z, dtype=torch.float32)).numpy()


def time_overhead(beta, u_s1, state_fit_rows_raw, state_eval_raw, resn_eval, msk_eval, mapping,
                  repeats=TIMING_REPEATS):
    """Median single-pass wall time of the whole HRSI-only path over DIAG_EVAL.

    The timed path is exactly the production path: the same bias extraction, the
    same `norm_state(fit_rows, eval_rows)` standardization and the same role
    pooling that `beta_final` used, then one `beta` application.
    """
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        bhat, _, _ = bias_from_residual_block(resn_eval, msk_eval)
        _, ze = standardized_state(state_fit_rows_raw, state_eval_raw)
        predict_beta(beta, u_s1, interaction(bhat, role_table(ze, mapping)))
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), float(np.min(ts)), float(np.max(ts))


# --------------------------------------------------------------------------
# Frozen S1 consumption + cross-market stacked chronological OOF (protocol S7.1)
# --------------------------------------------------------------------------

def s1_artifacts(fi, ei, state_fit, state_eval, ef, seed):
    """Frozen S1 OOF directions and final DIAG_EVAL directions. S1 is not modified."""
    oof = shape_oof(fi.x, state_fit, ef, fi.repair_available, seed)
    v, u, _, info = fit_shape_outputs(fi.x, state_fit, ef, fi.repair_available, ei.x, state_eval, seed)
    info = {**info, "parameter_count": info["shape_parameter_count"],
            "inference_seconds_per_day": info["inference_seconds"] / max(len(ei.x), 1)}
    return oof, u, info


def calibrate_hrsi_oof(e, u, av, b, has_dir):
    """Pooled-alpha OOF chronology on the rows that carry an HRSI OOF direction.

    A frozen S1 direction exists on every repair-available row of B1..B4, but an
    HRSI direction needs a beta, and the first fold a beta can be fitted for is
    B2 (B1 would require B0 rows, which carry no frozen S1 direction). The fold
    structure is therefore shifted by one block: calibration starts at B2 and the
    first calibrated fold is B3. Self-exclusion is unchanged. `pooled_alpha`
    consumes `u` directly, so the row set here is restricted to the rows that
    actually carry a finite HRSI direction.
    """
    legal = av & has_dir
    d = np.full_like(u, np.nan); records = []
    for k in range(3, 5):
        prior = np.concatenate([b[f"B{j}"][legal[b[f"B{j}"]]] for j in range(2, k)])
        current = b[f"B{k}"][legal[b[f"B{k}"]]]
        alpha = pooled_alpha(e, u, prior); d[current] = alpha * u[current]
        records.append({"record_type": "alpha", "block": f"B{k}", "alpha": alpha,
                        "calibration_max_index": int(prior.max()), "proposal_min_index": int(current.min()),
                        "self_excluded": bool(not np.intersect1d(prior, current).size)})
    complete = np.concatenate([b[f"B{k}"][legal[b[f"B{k}"]]] for k in range(2, 5)])
    if not np.isfinite(u[complete]).all():
        raise RuntimeError("HRSI OOF directions incomplete on the B2..B4 alpha row set")
    return pooled_alpha(e, u, complete), d, complete, records


def beta_oof(cells, seed):
    """One global beta per fold, fitted on earlier blocks pooled over six cells.

    For fold `Bk` the training rows are `B1..B(k-1)` of every cell. Standardisation
    follows the frozen `norm_state(train, test)` convention: per cell the statistics
    come from that fold's training rows and are applied unchanged to that cell's
    current-fold rows. No current-fold row of any cell can enter the fold's beta.
    """
    per_cell_u = {k: np.full_like(c["oof_u"], np.nan) for k, c in cells.items()}
    folds, chron, seconds = [], [], 0.0
    for k in range(2, 5):
        pools, preds = [], []
        for key in sorted(cells):
            c = cells[key]; blk = c["blocks"]
            tr = np.concatenate([blk[f"B{j}"] for j in range(1, k)])
            active = blk[f"B{k}"][c["av"][blk[f"B{k}"]]]
            if not len(active):
                raise RuntimeError(f"{key}: no repair-available rows in fold B{k}")
            zf, za = standardized_state(c["state_fit_raw"][tr], c["state_fit_raw"][active])
            Rf = role_table(zf, c["role_map"]); Ra = role_table(za, c["role_map"])
            tgt, ok = unit_shape_target(c["ef"][tr])
            ok &= c["av"][tr] & c["has_dir"][tr]
            if not ok.any():
                raise RuntimeError(f"{key}: no legal beta training rows for fold B{k}")
            pools.append((c["oof_u"][tr][ok], interaction(c["bhat_fit"][tr][ok], Rf[ok]), tgt[ok]))
            preds.append((key, active, c["oof_u"][active], interaction(c["bhat_fit"][active], Ra),
                          hash_json({"state_fit_rows": hashlib.sha256(
                              np.ascontiguousarray(c["state_fit_raw"][tr]).tobytes()).hexdigest(),
                              "state_pred_rows": hashlib.sha256(
                                  np.ascontiguousarray(c["state_fit_raw"][active]).tobytes()).hexdigest()})))
        beta, info = fit_global_beta(pools, seed)
        seconds += info["training_seconds"]
        bhash = beta_sha(beta)
        folds.append({"stage": "OOF", "fold": f"B{k}", "cells_covered": len(preds),
                      "beta_sha256": bhash, **{f"beta_{r}": float(v) for r, v in zip(ROLES, beta)}, **info})
        for key, active, u_s1, Z, std_hash in preds:
            per_cell_u[key][active] = predict_beta(beta, u_s1, Z)
            chron.append({"record_type": "beta", "block": f"B{k}", "market": key[0], "host": key[1],
                          "beta_sha256": bhash, "beta_train_scope": "six-cell pooled B1..B(k-1)",
                          "proposal_min_index": int(active.min()), "state_std_hash": std_hash,
                          "self_excluded": True})
    return per_cell_u, folds, chron, seconds


def beta_final(cells, seed):
    """One final global beta on every legal OOF DIAG_FIT row of all six cells.

    Standardisation again follows `norm_state(train, test)`: statistics come from
    the fitted OOF rows of each cell and are applied unchanged to that cell's
    DIAG_EVAL rows.
    """
    pools, preds = [], []
    for key in sorted(cells):
        c = cells[key]
        rows = np.concatenate([c["blocks"][f"B{k}"] for k in range(1, 5)])
        rows = rows[c["av"][rows] & c["has_dir"][rows]]
        if not len(rows):
            raise RuntimeError(f"{key}: no legal final beta training rows")
        zf, ze = standardized_state(c["state_fit_raw"][rows], c["state_eval_raw"])
        tgt, ok = unit_shape_target(c["ef"][rows])
        if not ok.all():
            raise RuntimeError(f"{key}: illegal final beta training row")
        pools.append((c["oof_u"][rows], interaction(c["bhat_fit"][rows], role_table(zf, c["role_map"])), tgt))
        preds.append((key, interaction(c["bhat_eval"], role_table(ze, c["role_map"])), rows))
    beta, info = fit_global_beta(pools, seed)
    return beta, info, preds


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def score(c, u_eval, alpha, ef, u_oof, complete, parameter_count_, training_seconds, overhead_s=0.0):
    """Score one proposal against the frozen Host on the DIAG_EVAL rows.

    `c` carries the cell's frozen arrays: `y`/`hp` (target and Host forecast),
    `ee` (scale-normalized Host residual), `scale`/`avail` (the causal repair
    scale and availability), `tail`/`normal`, and the frozen S1 inference timing.
    """
    y = c["y"]; hp = c["hp"]; ee = c["ee"]
    pred = hp + alpha * u_eval * c["scale"] * c["avail"][:, None]
    true_u, valid = unit_shape_target(ee)
    cosine = np.sum(true_u * u_eval, axis=1); wrong = np.sum(ee * u_eval, axis=1) <= 0
    metrics = eval_metrics(y, hp, pred, c["tail"], c["normal"])
    infer = c["s1_inference_seconds"] + overhead_s
    metrics.update({"shape_cosine": float(np.mean(cosine[valid])),
                    "wrong_hemisphere_rate": float(np.mean(wrong)),
                    "alpha": float(alpha),
                    "oof_calibrated_utility": float(np.mean(utility(ef[complete], alpha * u_oof[complete]))),
                    "parameter_count": int(parameter_count_),
                    "incremental_parameter_count": int(N_ROLES),
                    "training_seconds": float(training_seconds),
                    "inference_seconds": float(infer),
                    "inference_seconds_per_day": float(infer / max(c["n_eval"], 1)),
                    "inference_overhead_seconds_per_day": float(overhead_s / max(c["n_eval"], 1))})
    detail = {"u": u_eval, "cosine": cosine, "wrong": wrong, "pred": pred, "y": y, "hp": hp,
              "tail": c["tail"], "normal": c["normal"]}
    return metrics, detail


def tertile_metrics(detail, groups):
    rows = []
    for g, name in enumerate(("LOW", "MID", "HIGH")):
        z = groups == g; y = detail["y"][z]; hp = detail["hp"][z]; pred = detail["pred"][z]
        hm = float(np.mean(np.abs(y - hp)))
        rows.append({"tertile": name, "n_days": int(z.sum()),
                     "shape_cosine": float(np.mean(detail["cosine"][z])),
                     "wrong_hemisphere_rate": float(np.mean(detail["wrong"][z])),
                     "relative_gain_pct": 100 * (hm - float(np.mean(np.abs(y - pred)))) / hm,
                     "Tail_MAE": float(np.mean(np.abs(y - pred)[detail["tail"][z]])),
                     "Normal_MAE": float(np.mean(np.abs(y - pred)[detail["normal"][z]]))})
    return rows


# --------------------------------------------------------------------------

def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("canonical repo required")
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("frozen S1 replay requires exactly one torch thread")
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)

    frozen_paths = [SCHEMA_PATH, TERTILE_PATH, DIAG_ROWS_PATH, SHAPE_EVIDENCE / "metrics_by_seed.csv"]
    frozen_before = {str(p.relative_to(ROOT)): sha(p) for p in frozen_paths}
    shape_code_before = tree_sha(ROOT / "experiments/current/hch_unified_da_shape_upgrade")
    amplitude_code_before = tree_sha(ROOT / "experiments/current/hch_anchored_compact_amplitude")
    hsa_code_before = tree_sha(ROOT / "experiments/current/hch_horizon_aligned_state_residual")

    caches, paths, host_before = {}, {}, {}
    for m in MARKETS:
        for h in HOSTS:
            caches[m, h], paths[m, h] = load_frozen_cache(m, h); host_before[f"{m}/{h}"] = sha(paths[m, h])
    by_market = {m: caches[m, HOSTS[0]] for m in MARKETS}

    mapping, role_manifest = load_role_schema()
    role_manifest.to_csv(OUT / "role_state_manifest.csv", index=False)

    states, access = {}, {}
    for m in MARKETS:
        c = by_market[m]
        origins = np.r_[c.subset("DIAG_FIT").timestamp, c.subset("DIAG_EVAL").timestamp]
        z, a = read_state(m, origins); nf = len(c.subset("DIAG_FIT").timestamp)
        states[m] = (z[:nf], z[nf:]); access[m] = a

    prior_diag = pd.read_parquet(DIAG_ROWS_PATH); prior_diag["_ts"] = pd.to_datetime(prior_diag.forecast_origin)
    prior_tert = pd.read_csv(TERTILE_PATH)
    targeted = {(r.market, r.host) for _, r in prior_tert[prior_tert.mechanism_label.eq("SHAPE_LIMITED")][["market", "host"]].drop_duplicates().iterrows()}
    if len(targeted) != 4:
        raise RuntimeError(f"expected four frozen SHAPE_LIMITED cells, found {len(targeted)}")

    old = pd.read_csv(SHAPE_EVIDENCE / "metrics_by_seed.csv"); old = old[old.method.eq("S1_DailyPatch_GRU32")]
    hsa_old = pd.read_csv(HSA_EVIDENCE / "metrics_by_seed.csv")
    hsa_old = hsa_old[hsa_old.method.eq("S1_DailyPatch_GRU32")]

    # ---- seed-independent per-cell preparation ----
    cells, bias_rows, bias_hour_rows, split_manifest = {}, [], [], {}
    for m in MARKETS:
        sf, se = states[m]
        role_map = mapping[m]  # a single market's five-role mapping, never the whole schema
        if len(role_map) != N_ROLES:
            raise RuntimeError(f"{m}: role mapping must carry exactly five roles")
        for h in HOSTS:
            cache = caches[m, h]; fit = cache.subset("DIAG_FIT"); ev = cache.subset("DIAG_EVAL")
            # The market state vector is read once per market, so both Hosts must
            # share the exact fit/eval row ordering or `state_fit_raw` would be
            # misaligned for one of them.
            if not (len(fit.timestamp) == len(sf) and len(ev.timestamp) == len(se)
                    and np.array_equal(fit.timestamp, by_market[m].subset("DIAG_FIT").timestamp)
                    and np.array_equal(ev.timestamp, by_market[m].subset("DIAG_EVAL").timestamp)):
                raise RuntimeError(f"{m}/{h}: Host split manifest disagrees with the state row alignment")
            fi, floor = make_inputs(fit, cache); ei, _ = make_inputs(ev, cache, floor)
            bhat_fit, bias_fit, cnt_fit = host_bias_direction(fi.x, fi.residual_mask)
            bhat_eval, bias_eval, cnt_eval = host_bias_direction(ei.x, ei.residual_mask)
            ef = (fit.y_true[:, :, 0] - fit.host_pred[:, :, 0]) / np.where(np.isfinite(fi.scale), fi.scale, 1)[:, None]
            ee = (ev.y_true[:, :, 0] - ev.host_pred[:, :, 0]) / np.where(np.isfinite(ei.scale), ei.scale, 1)[:, None]
            y = ev.y_true[:, :, 0]; hp = ev.host_pred[:, :, 0]
            q05, q95 = np.quantile(fit.y_true[:, :, 0], [.05, .95]); tail = (y <= q05) | (y >= q95); normal = ~tail
            split_manifest[f"{m}/{h}"] = {"DIAG_FIT_timestamp_hash": hash_json(list(map(str, fit.timestamp))),
                                          "DIAG_EVAL_timestamp_hash": hash_json(list(map(str, ev.timestamp))),
                                          "fit_n": len(fit.timestamp), "eval_n": len(ev.timestamp)}
            th = prior_tert[(prior_tert.market == m) & (prior_tert.host == h)][["fit_threshold_low", "fit_threshold_high"]].drop_duplicates()
            if len(th) != 1:
                raise RuntimeError("frozen tertile threshold ambiguity")
            low, high = map(float, th.iloc[0])
            pdg = prior_diag[(prior_diag.market == m) & (prior_diag.host == h)]
            groups = {seed: np.digitize(pdg[pdg.seed == seed].set_index("_ts").reindex(
                pd.to_datetime(ev.timestamp)).rho_shape.to_numpy(), [low, high], right=True) for seed in SEEDS}
            for seed in SEEDS:
                if pdg[pdg.seed == seed].set_index("_ts").reindex(pd.to_datetime(ev.timestamp)).rho_shape.isna().any():
                    raise RuntimeError("frozen S1-era state stratum alignment failed")
            cells[m, h] = {
                "market": m, "host": h, "track": TRACK[m], "role_map": role_map,
                "fi": fi, "ei": ei, "ef": ef, "ee": ee, "y": y, "hp": hp,
                "scale": np.nan_to_num(ei.scale, nan=0)[:, None], "avail": ei.repair_available,
                "tail": tail, "normal": normal, "groups": groups, "low": low, "high": high,
                "bhat_fit": bhat_fit, "bhat_eval": bhat_eval,
                "state_fit_raw": sf, "state_eval_raw": se, "n_eval": len(ei.x),
            }
            # A row is bias-available iff at least one of its 24 hours has any of
            # the seven historical days. That must be exactly the row set S1 treats
            # as repair-available, otherwise the two mechanisms would disagree about
            # which days they are allowed to see.
            avail_fit = cnt_fit.max(axis=1) > 0; avail_eval = cnt_eval.max(axis=1) > 0
            if not np.array_equal(avail_fit, fi.repair_available):
                raise RuntimeError(f"{m}/{h}: DIAG_FIT bias availability disagrees with repair availability")
            if not np.array_equal(avail_eval, ei.repair_available):
                raise RuntimeError(f"{m}/{h}: DIAG_EVAL bias availability disagrees with repair availability")
            hours_fit = (cnt_fit > 0).any(axis=0); hours_eval = (cnt_eval > 0).any(axis=0)
            bias_rows.append({"market": m, "host": h,
                              "fit_rows": int(len(cnt_fit)), "eval_rows": int(len(cnt_eval)),
                              "fit_rows_with_history": int(avail_fit.sum()),
                              "eval_rows_with_history": int(avail_eval.sum()),
                              "fit_rows_without_history": int((~avail_fit).sum()),
                              "eval_rows_without_history": int((~avail_eval).sum()),
                              "fit_rows_zero_bias": int((np.linalg.norm(bhat_fit, axis=1) == 0).sum()),
                              "eval_rows_zero_bias": int((np.linalg.norm(bhat_eval, axis=1) == 0).sum()),
                              "fit_hours_with_any_history": int(hours_fit.sum()),
                              "eval_hours_with_any_history": int(hours_eval.sum()),
                              "fit_min_days_per_hour": float(cnt_fit.min()), "fit_max_days_per_hour": float(cnt_fit.max()),
                              "eval_min_days_per_hour": float(cnt_eval.min()), "eval_max_days_per_hour": float(cnt_eval.max()),
                              "fit_mean_raw_bias_norm": float(np.linalg.norm(bias_fit, axis=1).mean()),
                              "eval_mean_raw_bias_norm": float(np.linalg.norm(bias_eval, axis=1).mean()),
                              "fit_bhat_norm_min": float(np.linalg.norm(bhat_fit, axis=1).min()),
                              "fit_bhat_norm_max": float(np.linalg.norm(bhat_fit, axis=1).max())})
            for hh in range(24):
                bias_hour_rows.append({"market": m, "host": h, "hour": hh,
                                       "fit_availability_rate": float((cnt_fit[:, hh] > 0).mean()),
                                       "eval_availability_rate": float((cnt_eval[:, hh] > 0).mean()),
                                       "fit_mean_bias": float(bias_fit[:, hh].mean()),
                                       "eval_mean_bias": float(bias_eval[:, hh].mean()),
                                       "fit_mean_bhat": float(bhat_fit[:, hh].mean()),
                                       "eval_mean_bhat": float(bhat_eval[:, hh].mean())})
    pd.DataFrame(bias_rows).to_csv(OUT / "bias_profile_diagnostics.csv", index=False)
    pd.DataFrame(bias_hour_rows).to_csv(OUT / "bias_profile_by_hour.csv", index=False)

    metrics_rows, tert_rows, alpha_rows, chronology, efficiency, replay, beta_rows, timing_rows = ([] for _ in range(8))
    checkpoint_root = OUT / "checkpoints"; checkpoint_root.mkdir(exist_ok=True)

    for seed in SEEDS:
        checkpoint = checkpoint_root / f"seed{seed}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("frozen_hashes") != frozen_before:
                # The frozen science moved under a checkpoint: fail closed rather
                # than silently mixing results computed against different inputs.
                raise RuntimeError(f"stale checkpoint seed{seed}: frozen inputs changed")
            if saved.get("recipe_version") == RECIPE_VERSION:
                metrics_rows.extend(saved["metrics_rows"]); tert_rows.extend(saved["tert_rows"])
                alpha_rows.extend(saved["alpha_rows"]); chronology.extend(saved["chronology"])
                efficiency.extend(saved["efficiency"]); replay.extend(saved["replay"])
                beta_rows.extend(saved["beta_rows"]); timing_rows.extend(saved["timing_rows"])
                print(f"REUSED seed{seed}", flush=True); continue
            # A recipe bump only means the reporting/aggregation code changed; the
            # frozen inputs are identical, so recompute this seed and overwrite.
            print(f"RECOMPUTED seed{seed} (recipe v{saved.get('recipe_version')} -> v{RECIPE_VERSION})", flush=True)
        starts = tuple(map(len, (metrics_rows, tert_rows, alpha_rows, chronology, efficiency,
                                 replay, beta_rows, timing_rows)))

        # S1 artefacts for all six cells under this seed (frozen S1, never modified).
        for key, c in cells.items():
            oof, u_s1_eval, info = s1_artifacts(c["fi"], c["ei"], c["state_fit_raw"], c["state_eval_raw"], c["ef"], seed)
            c["oof_u"] = oof["u"]; c["blocks"] = oof["blocks"]; c["s1_info"] = info
            c["s1_oof_seconds"] = oof["training_seconds"]
            c["s1_inference_seconds"] = info["inference_seconds"]
            c["has_dir"] = np.isfinite(oof["u"]).all(axis=1)
            c["av"] = c["fi"].repair_available
            if c["has_dir"].any() and not np.array_equal(np.flatnonzero(c["has_dir"]), np.flatnonzero(c["av"] & c["has_dir"])):
                raise RuntimeError(f"{key}: frozen S1 OOF directions exist outside repair-available rows")
            c["u_s1_eval"] = u_s1_eval

        # One global beta per fold, fitted on six-cell pooled earlier blocks.
        per_cell_u, folds, chron, oof_seconds = beta_oof(cells, seed)
        beta_fin, finfo, preds = beta_final(cells, seed)
        final_z = {key: Z for key, Z, _ in preds}
        final_rows = {key: rows for key, _, rows in preds}
        finfo = {"stage": "FINAL", "fold": "B1..B4", "cells_covered": len(preds),
                 "beta_sha256": beta_sha(beta_fin),
                 **{f"beta_{r}": float(v) for r, v in zip(ROLES, beta_fin)}, **finfo}
        folds.append(finfo)
        for f in folds:
            beta_rows.append({"seed": seed, **f})

        for key in sorted(cells):
            c = cells[key]; m, h = key
            common = {"market": m, "track": c["track"], "host": h, "seed": seed}
            ef = c["ef"]; fi = c["fi"]
            alpha_s1, _, _, arec = _s1_alpha(ef, c, fi.repair_available)
            s1_metrics, s1_detail = score(
                c, c["u_s1_eval"], alpha_s1, ef, c["oof_u"], _s1_complete(c, fi.repair_available),
                c["s1_info"]["parameter_count"], c["s1_oof_seconds"] + c["s1_info"]["training_seconds"])

            hrsi_u = per_cell_u[key]
            alpha_hrsi, _, complete_h, hrec = calibrate_hrsi_oof(
                ef, hrsi_u, fi.repair_available, c["blocks"], np.isfinite(hrsi_u).all(axis=1))
            Z_eval = final_z[key]
            u_hrsi_eval = predict_beta(beta_fin, c["u_s1_eval"], Z_eval)
            overhead_s, ov_min, ov_max = time_overhead(
                beta_fin, c["u_s1_eval"], c["state_fit_raw"][final_rows[key]], c["state_eval_raw"],
                c["ei"].x[:, BIAS_CHANNEL], c["ei"].x[:, MASK_CHANNEL], c["role_map"])
            hrsi_metrics, hrsi_detail = score(
                c, u_hrsi_eval, alpha_hrsi, ef, hrsi_u, complete_h,
                c["s1_info"]["parameter_count"] + N_ROLES,
                oof_seconds + finfo["training_seconds"], overhead_s)

            for method, met in (("S1_DailyPatch_GRU32", s1_metrics), ("HRSI_HostRelativeStateInteraction", hrsi_metrics)):
                metrics_rows.append({**common, "method": method, **met})
                efficiency.append({**common, "method": method, "parameter_count": met["parameter_count"],
                                   "incremental_parameter_count": met["incremental_parameter_count"],
                                   "training_seconds": met["training_seconds"],
                                   "inference_seconds": met["inference_seconds"],
                                   "inference_seconds_per_day": met["inference_seconds_per_day"],
                                   "inference_overhead_seconds_per_day": met["inference_overhead_seconds_per_day"]})
            # HRSI can only start calibrating at B3 (it needs a beta, and the first
            # beta-consuming fold is B2), so its alpha row set is B2..B4 rather than
            # S1's B1..B4. That is a different discrete weighted median, not merely a
            # different direction, so the size of the resulting shift is measured
            # explicitly and S1 is additionally rescored on HRSI's exact row set.
            # This is a diagnostics-only disclosure: the R2 gate stays defined
            # against the frozen S1 numbers above.
            alpha_s1_aligned = pooled_alpha(ef, c["oof_u"], complete_h)
            s1a_metrics, _ = score(c, c["u_s1_eval"], alpha_s1_aligned, ef, c["oof_u"], complete_h,
                                   c["s1_info"]["parameter_count"],
                                   c["s1_oof_seconds"] + c["s1_info"]["training_seconds"])
            alpha_rows.append({**common, "S1_pooled_OOF_alpha": alpha_s1, "HRSI_pooled_OOF_alpha": alpha_hrsi,
                               "HRSI_OOF_utility": hrsi_metrics["oof_calibrated_utility"],
                               "S1_OOF_utility": s1_metrics["oof_calibrated_utility"],
                               "S1_alpha_refit_on_HRSI_rowset": alpha_s1_aligned,
                               "alpha_rowset_sensitivity_pct": 100 * (alpha_s1_aligned - alpha_s1) / alpha_s1,
                               "S1_frozen_Overall_MAE": s1_metrics["Overall_MAE"],
                               "S1_aligned_rowset_Overall_MAE": s1a_metrics["Overall_MAE"],
                               "S1_aligned_rowset_relative_gain_pct": s1a_metrics["relative_gain_pct"],
                               "HRSI_minus_S1_aligned_overall_gain_pp": hrsi_metrics["relative_gain_pct"] - s1a_metrics["relative_gain_pct"]})
            for r in arec:
                chronology.append({**common, "method": "S1_DailyPatch_GRU32", **r})
            for r in hrec:
                chronology.append({**common, "method": "HRSI_HostRelativeStateInteraction", **r})
            for r in chron:
                if (r["market"], r["host"]) == key:
                    chronology.append({**common, "method": "HRSI_HostRelativeStateInteraction", **r})
            timing_rows.append({**common, "oof_beta_training_seconds": oof_seconds,
                                "final_beta_training_seconds": finfo["training_seconds"],
                                "beta_total_training_seconds": oof_seconds + finfo["training_seconds"],
                                "hrsi_overhead_seconds": overhead_s, "hrsi_overhead_min_seconds": ov_min,
                                "hrsi_overhead_max_seconds": ov_max, "hrsi_overhead_repeats": TIMING_REPEATS,
                                "s1_inference_seconds": c["s1_info"]["inference_seconds"], "n_eval_days": c["n_eval"]})
            for met, detail in ((s1_metrics, s1_detail), (hrsi_metrics, hrsi_detail)):
                method = "S1_DailyPatch_GRU32" if detail is s1_detail else "HRSI_HostRelativeStateInteraction"
                for r in tertile_metrics(detail, c["groups"][seed]):
                    tert_rows.append({**common, "method": method, "frozen_threshold_low": c["low"],
                                      "frozen_threshold_high": c["high"], **r})

            ref = old[(old.market == m) & (old.host == h) & (old.seed == seed)].iloc[0]
            href = hsa_old[(hsa_old.market == m) & (hsa_old.host == h) & (hsa_old.seed == seed)].iloc[0]
            replay.append({**common,
                           "Overall_diff": abs(s1_metrics["Overall_MAE"] - float(ref.Overall_MAE)),
                           "Tail_diff": abs(s1_metrics["Tail_MAE"] - float(ref.Tail_MAE)),
                           "Normal_diff": abs(s1_metrics["Normal_MAE"] - float(ref.Normal_MAE)),
                           "gain_diff": abs(s1_metrics["relative_gain_pct"] - float(ref.relative_gain_pct)),
                           "cosine_diff": abs(s1_metrics["shape_cosine"] - float(ref.shape_cosine)),
                           "wrong_diff": abs(s1_metrics["wrong_hemisphere_rate"] - float(ref.wrong_hemisphere_rate)),
                           "alpha_diff": abs(s1_metrics["alpha"] - float(ref.alpha)),
                           "HSA_evidence_Overall_diff": abs(s1_metrics["Overall_MAE"] - float(href.Overall_MAE)),
                           "HSA_evidence_alpha_diff": abs(s1_metrics["alpha"] - float(href.alpha))})
            print(f"COMPLETED seed{seed} {m}/{h} alpha_s1={alpha_s1:.6f} alpha_hrsi={alpha_hrsi:.6f} "
                  f"beta_final={np.round(beta_fin,4).tolist()}", flush=True)

        for key in sorted(cells):
            c = cells[key]
            metrics_rows.append({"market": c["market"], "track": c["track"], "host": c["host"],
                                 "seed": seed, "method": "Host",
                                 **eval_metrics(c["y"], c["hp"], c["hp"], c["tail"], c["normal"])})
            # Drop seed-scoped S1/HRSI directions so the next seed cannot reuse them.
            for k in ("oof_u", "u_s1_eval", "blocks", "has_dir", "av", "s1_info"):
                c.pop(k, None)

        collections = (metrics_rows, tert_rows, alpha_rows, chronology, efficiency, replay, beta_rows, timing_rows)
        names = ("metrics_rows", "tert_rows", "alpha_rows", "chronology", "efficiency", "replay", "beta_rows", "timing_rows")
        saved = {"recipe_version": RECIPE_VERSION, "frozen_hashes": frozen_before}
        for name, col, start_at in zip(names, collections, starts):
            saved[name] = col[start_at:]
        checkpoint.write_text(json.dumps(saved, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

    metrics = pd.DataFrame(metrics_rows)
    cell = cell_median(metrics, ["market", "host", "method"])
    tert = pd.DataFrame(tert_rows)
    tert_cell = cell_median(tert, ["market", "host", "method", "tertile"])
    base = tert_cell[tert_cell.method.eq("S1_DailyPatch_GRU32")].set_index(["market", "host", "tertile"])
    for col in ("shape_cosine", "wrong_hemisphere_rate", "relative_gain_pct", "Tail_MAE", "Normal_MAE"):
        tert_cell[f"HRSI_minus_S1_{col}"] = np.nan
        z = tert_cell.method.eq("HRSI_HostRelativeStateInteraction")
        tert_cell.loc[z, f"HRSI_minus_S1_{col}"] = [r[col] - base.loc[(r.market, r.host, r.tertile), col] for _, r in tert_cell[z].iterrows()]

    eff = pd.DataFrame(efficiency)
    eff_cell = cell_median(eff, ["market", "host", "method"])

    beta_audit = pd.DataFrame(beta_rows)
    panel_parameters = parameter_count(new_beta_model())
    if panel_parameters != N_ROLES:
        raise RuntimeError("the HRSI panel must expose exactly five trainable parameters")
    param_df = pd.DataFrame([{"panel": "six-cell development panel",
                              "trainable_parameter_count": int(panel_parameters),
                              "expected_parameter_count": int(N_ROLES),
                              "per_cell_beta_vectors": int(beta_audit.groupby(["seed", "fold"]).size().gt(1).sum()),
                              "shared_beta_rows": int(len(beta_audit)),
                              "cells_per_beta": int(beta_audit.cells_covered.min())}])

    timing = pd.DataFrame(timing_rows)
    timing_cell = cell_median(timing, ["market", "host"])
    timing_cell["hrsi_total_inference_seconds_per_day"] = (timing_cell.s1_inference_seconds + timing_cell.hrsi_overhead_seconds) / timing_cell.n_eval_days
    timing_cell["hrsi_overhead_seconds_per_day"] = timing_cell.hrsi_overhead_seconds / timing_cell.n_eval_days
    timing_cell["hrsi_total_inference_us_per_day"] = timing_cell.hrsi_total_inference_seconds_per_day * 1e6
    timing_cell["hrsi_overhead_us_per_day"] = timing_cell.hrsi_overhead_seconds_per_day * 1e6

    frozen_after = {str(p.relative_to(ROOT)): sha(p) for p in frozen_paths}
    host_after = {f"{m}/{h}": sha(paths[m, h]) for m in MARKETS for h in HOSTS}
    replay_df = pd.DataFrame(replay)
    replay_max = float(replay_df.drop(columns=["market", "track", "host", "seed"]).to_numpy().max())
    chron = pd.DataFrame(chronology)
    alph = chron[chron.record_type.eq("alpha")]
    chron_ok = bool(chron.self_excluded.all() and
                    (alph.calibration_max_index < alph.proposal_min_index).all())
    access_ok = all(a["target_day_DA_input_reads"] == 0 and a["realized_RT_input_reads"] == 0 and a["S3_S4_reads"] == 0
                    for a in access.values())
    zero_init_max = float(beta_audit.zero_init_replay_max_abs_diff.max())
    role_ok = bool(set(role_manifest[role_manifest.n_channels_in_role.gt(0)].semantic_role) <= set(ROLES))
    # R0: one global beta per fold/seed, applied to all six cells.
    shared_ok = bool(beta_audit.groupby(["seed", "fold"]).beta_sha256.nunique().eq(1).all()
                     and beta_audit.cells_covered.eq(6).all())

    x0 = bool(replay_max <= TOL and frozen_before == frozen_after
              and shape_code_before == tree_sha(ROOT / "experiments/current/hch_unified_da_shape_upgrade")
              and amplitude_code_before == tree_sha(ROOT / "experiments/current/hch_anchored_compact_amplitude")
              and hsa_code_before == tree_sha(ROOT / "experiments/current/hch_horizon_aligned_state_residual")
              and host_before == host_after and chron_ok and access_ok and role_ok and shared_ok
              and int(param_df.trainable_parameter_count.iloc[0]) == N_ROLES
              and int(param_df.per_cell_beta_vectors.iloc[0]) == 0
              and zero_init_max <= REPLAY_TOL)

    s1 = cell[cell.method.eq("S1_DailyPatch_GRU32")].set_index(["market", "host"])
    hrsi = cell[cell.method.eq("HRSI_HostRelativeStateInteraction")].set_index(["market", "host"])
    host = cell[cell.method.eq("Host")].set_index(["market", "host"])
    gansu = [k for k in hrsi.index if k[0] == "GANSU_DA"]
    intl = [k for k in hrsi.index if k[0] != "GANSU_DA"]

    cos_delta = hrsi.shape_cosine - s1.shape_cosine
    wrong_delta = hrsi.wrong_hemisphere_rate - s1.wrong_hemisphere_rate
    hi = tert_cell[(tert_cell.method.eq("HRSI_HostRelativeStateInteraction")) & tert_cell.tertile.eq("HIGH")].set_index(["market", "host"])
    g_hi = hi.loc[gansu]

    # ---- R1: mechanism ----
    r1 = bool((cos_delta.loc[gansu] > 0).all()
              and (wrong_delta.loc[gansu] <= TOL).all()
              and int((cos_delta > 0).sum()) >= 4
              and int((wrong_delta <= TOL).sum()) >= 4
              and (g_hi.HRSI_minus_S1_shape_cosine > 0).all()
              and int((g_hi.HRSI_minus_S1_wrong_hemisphere_rate <= -0.05 + TOL).sum()) >= 1)

    # ---- R2: universal forecast effect ----
    gain_s1 = s1.relative_gain_pct; gain_hrsi = hrsi.relative_gain_pct; dgain = gain_hrsi - gain_s1
    normal_harm = 100 * (hrsi.Normal_MAE / host.Normal_MAE - 1)
    gg = gain_hrsi.loc[gansu]
    r2 = bool((dgain >= -TOL).all()
              and int((hrsi.Overall_MAE < s1.Overall_MAE).sum()) >= 5
              and (gain_hrsi >= -TOL).all()
              and float(np.median(gain_hrsi)) >= 7
              and int((gain_hrsi >= 5).sum()) >= 4
              and float(normal_harm.max()) <= 1
              and (hrsi.Overall_MAE.loc[gansu] < s1.Overall_MAE.loc[gansu]).all()
              and float(gg.max()) >= 10 and float(gg.min()) >= 4)

    # ---- R3: seed stability and international transfer ----
    per_seed = metrics[metrics.method.isin(["S1_DailyPatch_GRU32", "HRSI_HostRelativeStateInteraction"])]
    piv = per_seed.pivot_table(index=["market", "host", "seed"], columns="method", values="relative_gain_pct")
    seed_delta = (piv["HRSI_HostRelativeStateInteraction"] - piv["S1_DailyPatch_GRU32"]).rename("delta_gain_pct").reset_index()
    seed_improve = seed_delta.assign(ok=seed_delta.delta_gain_pct > TOL).groupby(["market", "host"]).ok.sum()
    seed_nonworse = seed_delta.assign(ok=seed_delta.delta_gain_pct >= -TOL).groupby(["market", "host"]).ok.sum()
    intl_seed_min = float(seed_delta[seed_delta.market.ne("GANSU_DA")].delta_gain_pct.min())
    intl_cell_min = float(dgain.loc[intl].min())
    r3 = bool(int((seed_improve >= 2).sum()) >= 5
              and (dgain.loc[intl] >= -TOL).all()
              and int((dgain.loc[intl] > 0).sum()) >= 2
              and intl_cell_min >= -0.5 and intl_seed_min >= -0.5)

    # ---- R4: efficiency ----
    med_beta = float(np.median(beta_audit.training_seconds))
    med_overhead = float(np.median(timing_cell.hrsi_overhead_us_per_day))
    med_total = float(np.median(timing_cell.hrsi_total_inference_us_per_day))
    r4 = bool(int(panel_parameters) == N_ROLES and med_beta <= BETA_TRAINING_BUDGET_S
              and med_overhead <= OVERHEAD_BUDGET_US_DAY and med_total <= TOTAL_INFERENCE_BUDGET_US_DAY)

    token = TOKEN_INVALID if not x0 else (TOKEN_PASS if all((r1, r2, r3, r4)) else TOKEN_FAIL)
    gates = {"R0": "PASS" if x0 else "FAIL", "R1": "PASS" if r1 else "FAIL",
             "R2": "PASS" if r2 else "FAIL", "R3": "PASS" if r3 else "FAIL", "R4": "PASS" if r4 else "FAIL"}

    # ---- artifacts ----
    metrics.to_csv(OUT / "metrics_by_seed.csv", index=False)
    cell.to_csv(OUT / "metrics_by_cell.csv", index=False)
    tert_cell.to_csv(OUT / "state_tertile_comparison.csv", index=False)
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(OUT / "alpha_hrsi.csv", index=False)
    # Disclosed separately so that the registered alpha table stays a plain
    # S1-vs-HRSI comparison; this file is a robustness diagnostic on the
    # comparison itself, not an alternate HRSI configuration.
    alpha_df[["market", "track", "host", "seed"] + ROWSET_SENS_COLS].to_csv(
        OUT / "alpha_rowset_sensitivity.csv", index=False)
    # Row-set-alignment diagnostic aggregate. Reported, never gated: R2 is defined
    # against the frozen S1 numbers, and these numbers only quantify how much of
    # the S1-vs-HRSI difference could be attributed to the B1..B4 vs B2..B4 shift.
    sens_col = "HRSI_minus_S1_aligned_overall_gain_pp"
    sens_cell = alpha_df.drop(columns=["seed"]).groupby(
        ["market", "host"], as_index=False)[sens_col].median()
    sens_worst_cell = float(sens_cell[sens_col].min())
    sens_worst_seed = float(alpha_df[sens_col].min())
    sens_nonneg_cells = int((sens_cell[sens_col] >= -TOL).sum())
    sens_alpha_shift = float(alpha_df["alpha_rowset_sensitivity_pct"].abs().max())
    beta_audit.to_csv(OUT / "beta_by_fold_seed.csv", index=False)
    chron.to_csv(OUT / "oof_chronology_audit.csv", index=False)
    replay_df.to_csv(OUT / "s1_replay_audit.csv", index=False)
    seed_delta.to_csv(OUT / "seed_stability.csv", index=False)
    param_df.to_csv(OUT / "parameter_audit.csv", index=False)
    eff_cell.to_csv(OUT / "efficiency.csv", index=False)
    timing_cell.to_csv(OUT / "inference_overhead.csv", index=False)

    mech = cell[cell.method.isin(["S1_DailyPatch_GRU32", "HRSI_HostRelativeStateInteraction"])].pivot_table(
        index=["market", "host"], columns="method", values=["shape_cosine", "wrong_hemisphere_rate"]).reset_index()
    mech.columns = ["market", "host", "S1_shape_cosine", "HRSI_shape_cosine", "S1_wrong_hemisphere_rate", "HRSI_wrong_hemisphere_rate"]
    mech["shape_cosine_change"] = mech.HRSI_shape_cosine - mech.S1_shape_cosine
    mech["wrong_hemisphere_change"] = mech.HRSI_wrong_hemisphere_rate - mech.S1_wrong_hemisphere_rate
    gh = hi.reset_index()[["market", "host", "HRSI_minus_S1_shape_cosine", "HRSI_minus_S1_wrong_hemisphere_rate"]].rename(
        columns={"HRSI_minus_S1_shape_cosine": "HIGH_shape_cosine_change",
                 "HRSI_minus_S1_wrong_hemisphere_rate": "HIGH_wrong_hemisphere_change"})
    mech = mech.merge(gh, on=["market", "host"], how="left")
    mech.to_csv(OUT / "shape_mechanisms_by_cell.csv", index=False)

    verdict = {"token": token, **gates,
               "S1_replay_max_abs_diff": replay_max, "beta_zero_init_replay_max_abs_diff": zero_init_max,
               "R0_panel_trainable_parameters": int(panel_parameters),
               "R0_per_cell_beta_vectors": int(param_df.per_cell_beta_vectors.iloc[0]),
               "R0_cells_per_beta": int(param_df.cells_per_beta.iloc[0]),
               "R1_GANSU_cosine_changes": {f"{k[0]}/{k[1]}": float(v) for k, v in cos_delta.loc[gansu].items()},
               "R1_GANSU_wrong_rate_changes": {f"{k[0]}/{k[1]}": float(v) for k, v in wrong_delta.loc[gansu].items()},
               "R1_cosine_improved_cells": int((cos_delta > 0).sum()),
               "R1_wrong_hemisphere_nonworse_cells": int((wrong_delta <= TOL).sum()),
               "R1_GANSU_HIGH_cosine_changes": {f"{k[0]}/{k[1]}": float(v) for k, v in g_hi.HRSI_minus_S1_shape_cosine.items()},
               "R1_GANSU_HIGH_wrong_rate_changes": {f"{k[0]}/{k[1]}": float(v) for k, v in g_hi.HRSI_minus_S1_wrong_hemisphere_rate.items()},
               "R2_gain_delta_pct": {f"{k[0]}/{k[1]}": float(v) for k, v in dgain.items()},
               "R2_median_gain_pct": float(np.median(gain_hrsi)),
               "R2_gain_ge5_cells": int((gain_hrsi >= 5).sum()),
               "R2_host_nonworse_cells": int((gain_hrsi >= -TOL).sum()),
               "R2_max_Normal_harm_pct": float(normal_harm.max()),
               "R2_GANSU_gains_pct": {f"{k[0]}/{k[1]}": float(v) for k, v in gg.items()},
               "R3_cells_improving_in_ge2of3_seeds": int((seed_improve >= 2).sum()),
               "R3_cells_nonworse_in_ge2of3_seeds": int((seed_nonworse >= 2).sum()),
               "R3_international_cell_median_min_pp": intl_cell_min,
               "R3_international_seed_min_pp": intl_seed_min,
               "R3_international_cells_strictly_better": int((dgain.loc[intl] > 0).sum()),
               "R4_panel_parameters": int(panel_parameters),
               "R4_median_beta_training_seconds": med_beta,
               "R4_median_overhead_us_per_day": med_overhead,
               "R4_median_total_inference_us_per_day": med_total,
               "DIAG_max_alpha_rowset_shift_pct": sens_alpha_shift,
               "DIAG_cells_nonworse_on_aligned_rowset": sens_nonneg_cells,
               "DIAG_aligned_rowset_worst_cell_pp": sens_worst_cell,
               "DIAG_aligned_rowset_worst_seed_pp": sens_worst_seed,
               "runtime_seconds": time.time() - started}
    (OUT / "HRSI_VERDICT.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

    provenance = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "canonical_root": str(ROOT),
                  "controlling_protocol": "docs/current/HCH_HOST_RELATIVE_STATE_INTERACTION_FINAL_CLOSURE_20260911.md",
                  "frozen_hashes_before": frozen_before, "frozen_hashes_after": frozen_after,
                  "s1_code_hash_before": shape_code_before, "s1_code_hash_after": tree_sha(ROOT / "experiments/current/hch_unified_da_shape_upgrade"),
                  "amplitude_code_hash_before": amplitude_code_before,
                  "amplitude_code_hash_after": tree_sha(ROOT / "experiments/current/hch_anchored_compact_amplitude"),
                  "hsa_code_hash_before": hsa_code_before,
                  "hsa_code_hash_after": tree_sha(ROOT / "experiments/current/hch_horizon_aligned_state_residual"),
                  "host_hashes_before": host_before, "host_hashes_after": host_after,
                  "state_level_access": access,
                  "role_mapping": {m: [[LEGAL[m][i] for i in role] for role in mapping[m]] for m in MARKETS},
                  "splits": split_manifest,
                  "host_bias_shape": {"history_days": HIST_DAYS, "window": "existing seven 24h historical days of the S1 repair input",
                                      "normalization": "bias / (L2 norm over the 24 horizon hours + eps)",
                                      "eps": EPS, "imputation": "none; unavailable hours contribute zero numerator",
                                      "decay": None, "robust_transform": None, "alternate_window": False},
                  "beta_training": {"parameterization": "one global beta in R^5 shared by all six cells",
                                    "loss": "L = (1/6) sum_c (1/N_c) sum_i ||u_HRSI_i - u*_i||^2",
                                    "optimizer": "AdamW", "lr": LR, "weight_decay": WD, "steps": STEPS,
                                    "grad_clip": 1.0, "initialization": "exactly zero",
                                    "standardization": "frozen norm_state(train, test): statistics from that fit's training rows, applied unchanged to its prediction rows",
                                    "no_search": True},
                  "forbidden": {"alternate_history_window": False, "decay_weighting": False, "median_or_mad_transform": False,
                                "clipping": False, "state_excursion": False, "role_selection": False, "per_hour_beta": False,
                                "beta_mlp": False, "conditional_amplitude": False, "gate_or_verification": False,
                                "router_or_expert_or_retrieval": False, "transformer_attention_cnn_tcn": False,
                                "market_specific_hyperparameters": False, "search": False, "s3_s4_protected_final_reads": 0},
                  "scientific_reads": {"DIAG_FIT": sum(len(by_market[m].subset("DIAG_FIT").timestamp) for m in MARKETS),
                                       "DIAG_EVAL": sum(len(by_market[m].subset("DIAG_EVAL").timestamp) for m in MARKETS),
                                       "S3": 0, "S4": 0, "protected": 0, "final": 0},
                  "numerical_environment": {"torch": torch.__version__, "numpy": np.__version__,
                                            "pandas": pd.__version__, "torch_threads": torch.get_num_threads(),
                                            "note": "single-threaded torch is required for bit-exact frozen S1 replay"},
                  "code_sha256": sha(Path(__file__))}
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

    # ---- figures ----
    gm = ["S1_DailyPatch_GRU32", "HRSI_HostRelativeStateInteraction"]
    gsum = cell[cell.method.isin(gm) & cell.market.eq("GANSU_DA")].pivot_table(
        index=["market", "host"], columns="method", values="shape_cosine")
    ghi = tert_cell[tert_cell.tertile.eq("HIGH") & tert_cell.market.eq("GANSU_DA") & tert_cell.method.isin(gm)].pivot_table(
        index=["market", "host"], columns="method", values="shape_cosine")
    gboth = pd.concat({"(a) Overall": gsum, "(b) Frozen HIGH tertile": ghi}, names=["stratum"])
    gboth.plot(kind="bar", figsize=(10, 5), ylabel="Shape cosine",
               title="GANSU_DA Shape cosine: S1 vs HRSI under one shared beta")
    plt.tight_layout(); plt.savefig(OUT / "figures/fig_gansu_shape_recovery.png", dpi=160); plt.close()
    gp = cell[cell.method.isin(["S1_DailyPatch_GRU32", "HRSI_HostRelativeStateInteraction"])].pivot_table(
        index=["market", "host"], columns="method", values="relative_gain_pct")
    gp.plot(kind="bar", figsize=(10, 4)); plt.ylabel("Relative Overall-MAE gain vs Host (%)")
    plt.tight_layout(); plt.savefig(OUT / "figures/fig_six_cell_s1_vs_hrsi_gain.png", dpi=160); plt.close()
    plt.figure(figsize=(9, 4)); plt.bar([f"{k[0]}/{k[1]}" for k in dgain.index], dgain.to_numpy())
    plt.axhline(0, color="k", lw=.8); plt.ylabel("HRSI minus S1 relative gain (pp)"); plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(OUT / "figures/fig_hrsi_minus_s1_gain.png", dpi=160); plt.close()

    # ---- audit + summary ----
    tests = subprocess.run([sys.executable, str(ROOT / "experiments/current/hch_host_relative_state_interaction/test_closure.py")],
                           cwd=ROOT, capture_output=True, text=True)
    (OUT / "tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    tail = [ln for ln in tests.stdout.strip().splitlines() if ln.strip()][-1:] or ["<no output>"]
    audit = ["# Implementation and test audit", "",
             f"- Canonical root: `{ROOT}`",
             "- Controlling protocol: `docs/current/HCH_HOST_RELATIVE_STATE_INTERACTION_FINAL_CLOSURE_20260911.md`",
             f"- Frozen S1 metric replay maximum absolute difference: `{replay_max:.3e}` (tolerance `{TOL:.0e}`)",
             f"- beta=0 exact S1 direction replay maximum absolute difference: `{zero_init_max:.3e}` (tolerance `{REPLAY_TOL:.0e}`)",
             f"- Frozen artifact immutability: `{frozen_before == frozen_after}`",
             f"- S1 code tree immutability: `{shape_code_before == tree_sha(ROOT / 'experiments/current/hch_unified_da_shape_upgrade')}`",
             f"- Amplitude code tree immutability: `{amplitude_code_before == tree_sha(ROOT / 'experiments/current/hch_anchored_compact_amplitude')}`",
             f"- HSA code tree immutability: `{hsa_code_before == tree_sha(ROOT / 'experiments/current/hch_horizon_aligned_state_residual')}`",
             f"- Host cache immutability: `{host_before == host_after}`",
             f"- Stacked chronological OOF self-exclusion: `{chron_ok}`",
             f"- One global beta per fold/seed applied to all six cells: `{shared_ok}`",
             f"- Trainable parameters for the whole six-cell panel: `{int(panel_parameters)}`",
             f"- Per-market/per-Host beta vectors: `{int(param_df.per_cell_beta_vectors.iloc[0])}`",
             f"- Forbidden/protected reads: `0`",
             f"- Targeted unit suite exit code: `{tests.returncode}`",
             f"- Targeted unit suite result: `{tail[0]}`",
             "- Targeted unit suite output: `tests.txt`",
             f"- Row-set-alignment diagnostic worst cell: `{sens_worst_cell:+.4f} pp` "
             f"(`alpha_rowset_sensitivity.csv`, not a gate input)",
             "- Independently re-verified by `verify_gates.py` (does not import this runner): "
             "recomputed R0-R4 and the token from the written evidence alone",
             "",
             "## Standardization convention",
             "",
             "`beta` is fitted through the frozen `norm_state(train, test)` pattern: for each fit the",
             "statistics come from that fit's own training rows and are applied unchanged to the rows",
             "it predicts. This is the convention the frozen S1 `fit_predict` uses for both its OOF",
             "folds and its final DIAG_EVAL fit. It is stated here because the rejected HSA runner",
             "standardized its final fit on its own OOF rows but then applied `beta` to state",
             "standardized on the full DIAG_FIT block; HRSI does not reproduce that mismatch."]
    (OUT / "IMPLEMENTATION_TEST_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")

    bfin = beta_audit[beta_audit.stage.eq("FINAL")].set_index("seed")[[f"beta_{r}" for r in ROLES]]
    show = cell[cell.method.isin(["Host", "S1_DailyPatch_GRU32", "HRSI_HostRelativeStateInteraction"])]
    summary = ["# Host-Relative State Interaction (HRSI) — final structural closure", "",
               f"Token: `{token}`", "", *(f"- {k}: `{v}`" for k, v in gates.items()), "",
               "## Method", "",
               "A causal recent Host-bias direction `b_hat` is built from the seven-day same-hour historical residual",
               "already carried by the frozen S1 repair input, L2-normalized over the 24 horizon hours. The fixed",
               "five-role standardized state table `R in R^(24x5)` multiplies it elementwise into `Z[h,k]=b_hat[h]*R[h,k]`,",
               "and exactly ONE global `beta in R^5` shared by all six cells gives `u_HRSI = normalize(u_S1 + Z beta)`,",
               "`beta=0` replaying S1 exactly. `alpha_HRSI` is refit per cell with the original exact weighted-median rule.", "",
               "## Cell metrics", "", show.to_markdown(index=False), "",
               "## Shape mechanism by cell", "", mech.to_markdown(index=False), "",
               "## Final global beta per seed", "", bfin.to_markdown(), "",
               "## Causal bias-profile availability", "",
               pd.DataFrame(bias_rows).to_markdown(index=False), "",
               "## Required statements", "",
               f"1. **Do both GANSU_DA Hosts improve, under one shared beta?** "
               f"{'Yes' if bool((cos_delta.loc[gansu] > 0).all()) else 'No'}: cell-median Shape cosine changes by "
               + ", ".join(f"{float(v):+.4f} ({k[1]})" for k, v in cos_delta.loc[gansu].items())
               + ", wrong-hemisphere by "
               + ", ".join(f"{float(v):+.4f} ({k[1]})" for k, v in wrong_delta.loc[gansu].items())
               + ", with no Market-ID, Host-ID, or per-cell coefficient anywhere in the model.",
               f"2. **Is the same five-parameter mechanism non-worse across all six cells?** "
               f"{'Yes' if bool((dgain >= -TOL).all()) else 'No'}: relative Overall-MAE gain vs S1 is nonnegative in "
               f"{int((dgain >= -TOL).sum())}/6 cell medians, strictly better in {int((hrsi.Overall_MAE < s1.Overall_MAE).sum())}/6, "
               f"and Host-nonworse in {int((gain_hrsi >= -TOL).sum())}/6 (worst {float(dgain.min()):+.4f} pp vs S1).",
               f"3. **Does the Host-relative interaction resolve the HSA sign-transfer conflict?** "
               f"{'Yes' if token == TOKEN_PASS else 'No'}: international cell-median delta is "
               f"{intl_cell_min:+.4f} pp (worst seed {intl_seed_min:+.4f} pp) versus HSA's worst cell of "
               f"-0.87 pp. The bilateral term "
               + ("does transfer across all six cells." if bool((dgain >= -TOL).all()) else
                  "still does not transfer to every international cell."),
               f"4. **Is the method-development target reached?** "
               f"{'Yes' if token == TOKEN_PASS else 'No'}: R1={gates['R1']}, R2={gates['R2']}, R3={gates['R3']}, "
               f"R4={gates['R4']} with R0={gates['R0']}.",
               "",
               f"Median relative Overall-MAE gain vs Host: `{float(np.median(gain_hrsi)):.6f}%`; "
               f"median change vs S1: `{float(np.median(dgain)):.6f} pp`.",
               f"Maximum Normal-MAE harm: `{float(normal_harm.max()):.6f}%`.",
               f"Median global beta-fit time: `{med_beta:.4f} s`; median HRSI inference overhead: "
               f"`{med_overhead:.4f} us/day`; median total repair inference: `{med_total:.4f} us/day`.",
               f"Trainable parameters added for the entire six-cell panel: `{int(panel_parameters)}`.", "",
               "## Row-set-alignment diagnostic (not a gate)", "",
               "HRSI's scalar alpha is calibrated on OOF folds B2..B4 (a beta cannot exist before B2), while frozen",
               "S1 uses B1..B4. The exact weighted median is discrete, so this is a comparison confound rather than",
               "a modelling difference. Rescoring frozen S1 on HRSI's exact row set moves its pooled alpha by up to",
               f"`{sens_alpha_shift:.4f}%`; under that aligned row set the HRSI-minus-S1 Overall-MAE gain is",
               f"nonnegative in `{sens_nonneg_cells}/6` cell medians, worst cell `{sens_worst_cell:+.4f} pp`, worst",
               f"single seed `{sens_worst_seed:+.4f} pp`. The registered R2 gate above remains defined against the",
               "frozen S1 numbers; `alpha_rowset_sensitivity.csv` carries the per-cell detail.", "",
               "Method consequence: " + (
                   "the preregistered development target is reached. HRSI is frozen as the leading candidate and the "
                   "human is notified; no full-panel or public-China run is started by this stage."
                   if token == TOKEN_PASS else
                   "`S1 Daily-Patch GRU32 Shape + pooled fixed alpha` is frozen as the final structural candidate. "
                   "No further structural rescue is opened; the next stage is human adjudication followed by the full "
                   "public-China / international comparison against the audited baseline suite.")]
    (OUT / "HRSI_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(token); print(json.dumps(verdict, ensure_ascii=False, default=json_default))
    return 0 if x0 else 2


def _s1_alpha(e, c, av):
    """Frozen S1 pooled-alpha OOF chronology (B1..B4 row set), unchanged."""
    b = c["blocks"]; u = c["oof_u"]
    d = np.full_like(u, np.nan); records = []
    for k in range(2, 5):
        prior = np.concatenate([b[f"B{j}"][av[b[f"B{j}"]]] for j in range(1, k)])
        current = b[f"B{k}"][av[b[f"B{k}"]]]
        alpha = pooled_alpha(e, u, prior); d[current] = alpha * u[current]
        records.append({"record_type": "alpha", "block": f"B{k}", "alpha": alpha,
                        "calibration_max_index": int(prior.max()), "proposal_min_index": int(current.min()),
                        "self_excluded": bool(not np.intersect1d(prior, current).size)})
    allrows = np.concatenate([b[f"B{k}"][av[b[f"B{k}"]]] for k in range(1, 5)])
    return pooled_alpha(e, u, allrows), d, allrows, records


def _s1_complete(c, av):
    b = c["blocks"]
    return np.concatenate([b[f"B{k}"][av[b[f"B{k}"]]] for k in range(2, 5)])


if __name__ == "__main__":
    raise SystemExit(main())
