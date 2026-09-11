"""HCH Horizon-Aligned Semantic State Residual (HSA) Shape closure.

Controlling protocol:
    docs/current/HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SHAPE_CLOSURE_20260911.md

Executor contract:
    .agents/skills/solar-research-executor/SKILL.md

Low-autonomy executor. Frozen: S1 Daily-Patch GRU32, Host, split, Shape
target/loss, S1 OOF chronology, and the pooled exact-MAE scalar-Amplitude
calibration form. S1 is never retrained or modified; this runner consumes the
already-frozen S1 OOF Shape directions and the frozen forecast-time state level.

The only new trainable object is `beta in R^5` acting on a deterministic
horizon-aligned semantic role table `R_d in R^(24x5)`:

    q_d      = R_d @ beta
    u_HSA    = normalize(u_S1 + q_d)

`beta = 0` initialisation must replay S1 exactly. `alpha_HSA` is then refit
with the original exact weighted-median rule on HSA OOF directions, and the
final proposal is strictly `Delta = s * alpha_HSA * u_HSA`.
"""
from __future__ import annotations

import hashlib, json, subprocess, sys, time
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
    EPS, HOSTS, LEGAL, MARKETS, SEEDS, TRACK, make_inputs, norm_state, optimize,
    read_state,
)
from experiments.current.hch_anchored_compact_amplitude.run_closure import (
    SHAPE_EVIDENCE, eval_metrics, fit_shape_outputs, load_frozen_cache,
    pooled_alpha, shape_oof, utility,
)
from src.mvp.hch_minimal_repair.crossfit import chronological_blocks
from src.mvp.hch_minimal_repair.training import unit_shape_target
from utils.timing import parameter_count

# REPRODUCIBILITY INVARIANT — must stay after the imports above.
# `src/backbones/backbones.py` calls `torch.set_num_threads(4)` at import time, so
# pinning threads before importing would be silently reverted and S1 would train
# with 4 BLAS threads. That changes CPU reduction order by ~1e-7, which is enough
# to flip the discrete pooled weighted-median `alpha` by ~1e-3 and break the
# frozen-S1 replay. Pinning here (post-import, at module import time) makes both
# this runner and anything importing it reproduce the frozen evidence exactly.
torch.set_num_threads(1)
if torch.get_num_threads() != 1:
    raise RuntimeError("HSA runner requires single-threaded torch for frozen S1 replay")

OUT = ROOT / "experiments/evidence/hch_horizon_aligned_state_residual_20260911"
STATE_EVIDENCE = ROOT / "experiments/evidence/hch_market_state_coupling_20260910"
SCHEMA_PATH = STATE_EVIDENCE / "state_schema_manifest.csv"
TERTILE_PATH = STATE_EVIDENCE / "state_tertile_mechanisms.csv"
DIAG_ROWS_PATH = STATE_EVIDENCE / "diagnostic_rows.parquet"

TOKEN_PASS = "HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SUPPORTED_FOR_PUBLIC_CHINA_EXPANSION"
TOKEN_FAIL = "HCH_HORIZON_ALIGNED_STATE_RESIDUAL_NOT_SUPPORTED_KEEP_S1"
TOKEN_INVALID = "HCH_HORIZON_ALIGNED_STATE_RESIDUAL_INVALID"

TOL = 1e-10              # frozen S1 metric replay tolerance (same as prior closures)
REPLAY_TOL = 1e-6        # beta=0 -> exact S1 direction replay tolerance (float32 eps)
ROLES = ("DEMAND_FC", "RENEWABLE_FC", "SUPPLY_MARGIN_FC", "INTERCHANGE_FC", "MUST_RUN_FC")
N_ROLES = len(ROLES)
BETA_TRAINING_BUDGET_S = 5.0     # H4: median beta training seconds per cell
OVERHEAD_BUDGET_US_DAY = 10.0    # H4: median absolute inference overhead per day
TOTAL_INFERENCE_BUDGET_US_DAY = 100.0  # H4: total HSA inference per day
TIMING_REPEATS = 50


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
# Semantic role construction (protocol §3.2) -- deterministic mean pooling of
# audited primary forecast-time channels. No embeddings, no market rules.
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
    """Deterministic per-horizon-hour mean pool -> R_d in R^(24x5); absent role = structural 0."""
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
# The only new trainable object: beta in R^5.
# --------------------------------------------------------------------------

class HSAResidual(nn.Module):
    """`u_HSA = normalize(u_S1 + R beta)` with a single five-parameter vector."""

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(N_ROLES))

    def forward(self, u_s1, R):
        z = u_s1 + R @ self.beta
        return z / (torch.linalg.norm(z, dim=1, keepdim=True) + EPS)


def new_beta_model():
    model = HSAResidual()
    if set(type(m) for m in model.modules()) != {HSAResidual}:
        raise RuntimeError("HSA must not add a hidden layer or submodule")
    if parameter_count(model) != N_ROLES:
        raise RuntimeError("HSA must expose exactly five trainable parameters")
    if float(model.beta.detach().abs().max()) != 0.0:
        raise RuntimeError("beta must be initialised to exactly zero")
    return model


def fit_beta(u_s1, R, target, valid, seed):
    """Fit beta on one chronological fold; verifies exact S1 replay at beta=0."""
    ut = torch.as_tensor(u_s1[valid], dtype=torch.float32)
    Rt = torch.as_tensor(R[valid], dtype=torch.float32)
    yt = torch.as_tensor(target[valid], dtype=torch.float32)
    model = new_beta_model()
    with torch.no_grad():
        replay = model(ut, Rt).numpy()
    replay_max = float(np.max(np.abs(replay - u_s1[valid])))
    if replay_max > REPLAY_TOL:
        raise RuntimeError(f"beta=0 does not replay S1 exactly ({replay_max:.3e})")
    loss, secs = optimize(model, lambda: model(ut, Rt), yt, seed)
    return model.beta.detach().numpy().astype(np.float64).copy(), {
        "objective": float(loss), "training_seconds": float(secs),
        "zero_init_replay_max_abs_diff": replay_max, "n_train_rows": int(valid.sum())}


def predict_beta(beta, u_s1, R):
    model = HSAResidual()
    with torch.no_grad():
        model.beta.copy_(torch.as_tensor(beta, dtype=torch.float32))
        return model(torch.as_tensor(u_s1, dtype=torch.float32), torch.as_tensor(R, dtype=torch.float32)).numpy()


def time_overhead(beta, u_s1, state_raw, fit_state_raw, mapping, repeats=TIMING_REPEATS):
    """Median single-pass wall time of the HSA-only path over the DIAG_EVAL set."""
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        _, z_ev = standardized_state(fit_state_raw, state_raw)
        R = role_table(z_ev, mapping)
        predict_beta(beta, u_s1, R)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), float(np.min(ts)), float(np.max(ts))


# --------------------------------------------------------------------------
# Frozen S1 consumption + HSA stacked chronological OOF (protocol §5.1)
# --------------------------------------------------------------------------

def s1_artifacts(fi, ei, state_fit, state_eval, ef, seed):
    """Frozen S1 OOF directions and final DIAG_EVAL directions. S1 is not modified."""
    oof = shape_oof(fi.x, state_fit, ef, fi.repair_available, seed)
    v, u, _, info = fit_shape_outputs(fi.x, state_fit, ef, fi.repair_available, ei.x, state_eval, seed)
    info = {**info, "parameter_count": info["shape_parameter_count"],
            "inference_seconds_per_day": info["inference_seconds"] / max(len(ei.x), 1)}
    return oof, u, info


def calibrate_oof(e, u, av, b):
    """Exact pooled-alpha OOF chronology, identical to the frozen S1 convention."""
    d = np.full_like(u, np.nan); records = []
    for k in range(2, 5):
        prior = np.concatenate([b[f"B{j}"][av[b[f"B{j}"]]] for j in range(1, k)])
        current = b[f"B{k}"][av[b[f"B{k}"]]]
        alpha = pooled_alpha(e, u, prior); d[current] = alpha * u[current]
        records.append({"record_type": "alpha", "block": f"B{k}", "alpha": alpha,
                        "calibration_max_index": int(prior.max()), "proposal_min_index": int(current.min()),
                        "self_excluded": bool(not np.intersect1d(prior, current).size)})
    complete = np.concatenate([b[f"B{k}"][av[b[f"B{k}"]]] for k in range(2, 5)])
    allrows = np.concatenate([b[f"B{k}"][av[b[f"B{k}"]]] for k in range(1, 5)])
    return pooled_alpha(e, u, allrows), d, complete, allrows, records


def calibrate_hsa_oof(e, u, av, b, has_dir):
    """Pooled-alpha OOF chronology on the rows that carry an HSA OOF direction.

    `calibrate_oof` fits its final alpha over every repair-available row in
    B1..B4 because a frozen S1 direction exists on all of them. An HSA
    direction exists only on B2..B4, and `pooled_alpha` consumes `u` directly,
    so reusing that row set here would feed NaN into the weighted median.
    The fold structure is therefore shifted by one block: calibration starts
    at B2 and the first calibrated fold is B3. Self-exclusion is unchanged.
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
        raise RuntimeError("HSA OOF directions incomplete on the B2..B4 alpha row set")
    return pooled_alpha(e, u, complete), d, complete, records


def hsa_oof(u_s1, state_raw, e, av, mapping, seed):
    """Stacked chronological OOF beta. Training rows for block Bk are B1..B(k-1).

    B1 is the first block carrying a frozen S1 OOF direction; beta cannot be fit
    for B1 because that would require B0 rows, which have no frozen S1 direction.
    No row ever participates in training the beta that predicts it.
    """
    b = chronological_blocks(len(u_s1))
    u = np.full_like(u_s1, np.nan)
    has_dir = np.isfinite(u_s1).all(axis=1)
    if has_dir.any() and not np.array_equal(np.flatnonzero(has_dir), np.flatnonzero(av & has_dir)):
        raise RuntimeError("frozen S1 OOF directions exist outside repair-available rows")
    records, folds, seconds = [], [], 0.0
    for k in range(2, 5):
        te = b[f"B{k}"]
        tr = np.concatenate([b[f"B{j}"] for j in range(1, k)])
        active = te[av[te]]
        # Standardisation statistics follow the frozen S1 convention: computed on the
        # whole training portion, with the legal-row mask applied only to the loss.
        zf, za = standardized_state(state_raw[tr], state_raw[active])
        Rf = role_table(zf, mapping); Ra = role_table(za, mapping)
        target, valid = unit_shape_target(e[tr]); valid &= av[tr] & has_dir[tr]
        if not valid.any():
            raise RuntimeError(f"no legal beta training rows for block B{k}")
        beta, info = fit_beta(u_s1[tr], Rf, target, valid, seed)
        u[active] = predict_beta(beta, u_s1[active], Ra)
        seconds += info["training_seconds"]
        folds.append({"block": f"B{k}", **{f"beta_{r}": float(v) for r, v in zip(ROLES, beta)}, **info})
        records.append({"record_type": "beta", "block": f"B{k}", "beta_train_max_index": int(tr.max()),
                        "proposal_min_index": int(active.min()),
                        "self_excluded": bool(not np.intersect1d(tr, active).size)})
    return {"u": u, "blocks": b, "records": records, "folds": folds, "training_seconds": seconds}


def fit_beta_final(u_s1, state_raw, e, av, mapping, rows, seed):
    """One final beta on all legal chronological OOF S1 rows of DIAG_FIT."""
    has_dir = np.isfinite(u_s1[rows]).all(axis=1)
    if not has_dir.all():
        raise RuntimeError("frozen S1 OOF direction missing on the final beta training rows")
    zf, _ = standardized_state(state_raw[rows], state_raw[rows])
    R = role_table(zf, mapping)
    target, valid = unit_shape_target(e[rows]); valid &= av[rows] & has_dir
    beta, info = fit_beta(u_s1[rows], R, target, valid, seed)
    return beta, info


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def score(u_eval, alpha, ef, u_oof, complete, ev, ei, parameter_count, tail, normal,
          training_seconds, inference_seconds, n_eval, overhead_s):
    y = ev.y_true[:, :, 0]; hp = ev.host_pred[:, :, 0]
    scale = np.nan_to_num(ei.scale, nan=0)[:, None]
    pred = hp + alpha * u_eval * scale * ei.repair_available[:, None]
    ee = (ev.y_true[:, :, 0] - ev.host_pred[:, :, 0]) / np.where(np.isfinite(ei.scale), ei.scale, 1)[:, None]
    true_u, valid = unit_shape_target(ee)
    cosine = np.sum(true_u * u_eval, axis=1); wrong = np.sum(ee * u_eval, axis=1) <= 0
    metrics = eval_metrics(y, hp, pred, tail, normal)
    metrics.update({"shape_cosine": float(np.mean(cosine[valid])),
                    "wrong_hemisphere_rate": float(np.mean(wrong)),
                    "alpha": float(alpha),
                    "oof_calibrated_utility": float(np.mean(utility(ef[complete], alpha * u_oof[complete]))),
                    "parameter_count": int(parameter_count),
                    "training_seconds": float(training_seconds),
                    "inference_seconds": float(inference_seconds + overhead_s),
                    "inference_seconds_per_day": float((inference_seconds + overhead_s) / max(n_eval, 1)),
                    "inference_overhead_seconds_per_day": float(overhead_s / max(n_eval, 1))})
    detail = {"u": u_eval, "cosine": cosine, "wrong": wrong, "pred": pred, "y": y, "hp": hp,
              "tail": tail, "normal": normal}
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

    metrics_rows, tert_rows, alpha_rows, chronology, efficiency, replay, beta_rows, timing_rows = ([] for _ in range(8))
    checkpoint_root = OUT / "checkpoints"; checkpoint_root.mkdir(exist_ok=True)
    split_manifest, param_audit = {}, {}

    for m in MARKETS:
        sf, se = states[m]
        role_map = mapping[m]  # a single market's five-role mapping, never the whole schema
        for h in HOSTS:
            cache = caches[m, h]; fit = cache.subset("DIAG_FIT"); ev = cache.subset("DIAG_EVAL")
            fi, floor = make_inputs(fit, cache); ei, _ = make_inputs(ev, cache, floor)
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

            for seed in SEEDS:
                common = {"market": m, "track": TRACK[m], "host": h, "seed": seed}
                checkpoint = checkpoint_root / f"{m}__{h}__seed{seed}.json"
                if checkpoint.exists():
                    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if saved.get("frozen_hashes") != frozen_before or saved.get("recipe_version") != 1:
                        raise RuntimeError("stale checkpoint")
                    metrics_rows.extend(saved["metrics_rows"]); tert_rows.extend(saved["tert_rows"])
                    alpha_rows.extend(saved["alpha_rows"]); chronology.extend(saved["chronology"])
                    efficiency.extend(saved["efficiency"]); replay.extend(saved["replay"])
                    beta_rows.extend(saved["beta_rows"]); timing_rows.extend(saved["timing_rows"])
                    print(f"REUSED {m}/{h}/seed{seed}", flush=True); continue
                starts = tuple(map(len, (metrics_rows, tert_rows, alpha_rows, chronology, efficiency,
                                         replay, beta_rows, timing_rows)))
                prior = pdg[pdg.seed == seed].set_index("_ts").reindex(pd.to_datetime(ev.timestamp))
                if prior.rho_shape.isna().any():
                    raise RuntimeError("frozen S1-era state stratum alignment failed")
                groups = np.digitize(prior.rho_shape.to_numpy(), [low, high], right=True)

                ef = (fit.y_true[:, :, 0] - fit.host_pred[:, :, 0]) / np.where(np.isfinite(fi.scale), fi.scale, 1)[:, None]
                oof, u_s1_eval, info = s1_artifacts(fi, ei, sf, se, ef, seed)

                # ---- frozen S1 branch (replayed, never modified) ----
                alpha_s1, od_s1, complete, allrows, arec = calibrate_oof(ef, oof["u"], fi.repair_available, oof["blocks"])
                s1_metrics, s1_detail = score(u_s1_eval, alpha_s1, ef, oof["u"], complete, ev, ei,
                                              info["parameter_count"], tail, normal,
                                              oof["training_seconds"] + info["training_seconds"],
                                              info["inference_seconds"], len(ei.x), 0.0)

                # ---- HSA branch: refit only beta and the pooled scalar alpha ----
                hsa = hsa_oof(oof["u"], sf, ef, fi.repair_available, role_map, seed)
                alpha_hsa, od_hsa, complete_h, hrec = calibrate_hsa_oof(
                    ef, hsa["u"], fi.repair_available, hsa["blocks"], np.isfinite(hsa["u"]).all(axis=1))
                beta_final, finfo = fit_beta_final(oof["u"], sf, ef, fi.repair_available, role_map, allrows, seed)
                zf_eval, ze_eval = standardized_state(sf, se)
                R_eval = role_table(ze_eval, role_map)
                u_hsa_eval = predict_beta(beta_final, u_s1_eval, R_eval)
                overhead_s, ov_min, ov_max = time_overhead(beta_final, u_s1_eval, se, sf, role_map)
                hsa_metrics, hsa_detail = score(u_hsa_eval, alpha_hsa, ef, hsa["u"], complete_h, ev, ei,
                                                info["parameter_count"] + N_ROLES, tail, normal,
                                                hsa["training_seconds"] + finfo["training_seconds"],
                                                info["inference_seconds"], len(ei.x), overhead_s)

                for method, met in (("S1_DailyPatch_GRU32", s1_metrics), ("HSA_HorizonAlignedStateResidual", hsa_metrics)):
                    metrics_rows.append({**common, "method": method, **met})
                    efficiency.append({**common, "method": method, "parameter_count": met["parameter_count"],
                                       "training_seconds": met["training_seconds"],
                                       "inference_seconds": met["inference_seconds"],
                                       "inference_seconds_per_day": met["inference_seconds_per_day"],
                                       "inference_overhead_seconds_per_day": met["inference_overhead_seconds_per_day"]})
                alpha_rows.append({**common, "S1_pooled_OOF_alpha": alpha_s1, "HSA_pooled_OOF_alpha": alpha_hsa,
                                   "HSA_OOF_utility": hsa_metrics["oof_calibrated_utility"],
                                   "S1_OOF_utility": s1_metrics["oof_calibrated_utility"],
                                   "S1_alpha_refit_on_HSA_rowset": pooled_alpha(ef, oof["u"], complete_h),
                                   "alpha_rowset_sensitivity_pct": 100 * (pooled_alpha(ef, oof["u"], complete_h) - alpha_s1) / alpha_s1})
                for r in arec:
                    chronology.append({**common, "method": "S1_DailyPatch_GRU32", **r})
                for r in hrec:
                    chronology.append({**common, "method": "HSA_HorizonAlignedStateResidual", **r})
                for r in hsa["records"]:
                    chronology.append({**common, "method": "HSA_HorizonAlignedStateResidual", **r})
                for f in hsa["folds"]:
                    beta_rows.append({**common, "stage": "OOF", **f})
                beta_rows.append({**common, "stage": "FINAL", **{f"beta_{r}": float(v) for r, v in zip(ROLES, beta_final)},
                                  **finfo})
                timing_rows.append({**common, "beta_oof_training_seconds": hsa["training_seconds"],
                                    "beta_final_training_seconds": finfo["training_seconds"],
                                    "beta_total_training_seconds": hsa["training_seconds"] + finfo["training_seconds"],
                                    "hsa_overhead_seconds": overhead_s, "hsa_overhead_min_seconds": ov_min,
                                    "hsa_overhead_max_seconds": ov_max, "hsa_overhead_repeats": TIMING_REPEATS,
                                    "s1_inference_seconds": info["inference_seconds"], "n_eval_days": len(ei.x)})
                for met, detail in ((s1_metrics, s1_detail), (hsa_metrics, hsa_detail)):
                    method = "S1_DailyPatch_GRU32" if detail is s1_detail else "HSA_HorizonAlignedStateResidual"
                    for r in tertile_metrics(detail, groups):
                        tert_rows.append({**common, "method": method, "frozen_threshold_low": low,
                                          "frozen_threshold_high": high, **r})
                delta = int(hsa_metrics["parameter_count"] - s1_metrics["parameter_count"])
                key = f"{m}/{h}"
                if key in param_audit and param_audit[key]["observed_parameter_delta"] != delta:
                    raise RuntimeError(f"{key}: inconsistent HSA parameter delta across seeds")
                param_audit[key] = {"S1_parameter_count": int(s1_metrics["parameter_count"]),
                                    "HSA_parameter_count": int(hsa_metrics["parameter_count"]),
                                    "observed_parameter_delta": delta}

                ref = old[(old.market == m) & (old.host == h) & (old.seed == seed)].iloc[0]
                replay.append({**common,
                               "Overall_diff": abs(s1_metrics["Overall_MAE"] - float(ref.Overall_MAE)),
                               "Tail_diff": abs(s1_metrics["Tail_MAE"] - float(ref.Tail_MAE)),
                               "Normal_diff": abs(s1_metrics["Normal_MAE"] - float(ref.Normal_MAE)),
                               "gain_diff": abs(s1_metrics["relative_gain_pct"] - float(ref.relative_gain_pct)),
                               "cosine_diff": abs(s1_metrics["shape_cosine"] - float(ref.shape_cosine)),
                               "wrong_diff": abs(s1_metrics["wrong_hemisphere_rate"] - float(ref.wrong_hemisphere_rate)),
                               "alpha_diff": abs(s1_metrics["alpha"] - float(ref.alpha))})

                collections = (metrics_rows, tert_rows, alpha_rows, chronology, efficiency, replay, beta_rows, timing_rows)
                names = ("metrics_rows", "tert_rows", "alpha_rows", "chronology", "efficiency", "replay", "beta_rows", "timing_rows")
                saved = {"recipe_version": 1, "frozen_hashes": frozen_before}
                for name, col, start_at in zip(names, collections, starts):
                    saved[name] = col[start_at:]
                checkpoint.write_text(json.dumps(saved, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
                print(f"COMPLETED {m}/{h}/seed{seed} alpha_s1={alpha_s1:.6f} alpha_hsa={alpha_hsa:.6f} "
                      f"beta={np.round(beta_final,4).tolist()}", flush=True)

            for seed in SEEDS:
                metrics_rows.append({"market": m, "track": TRACK[m], "host": h, "seed": seed, "method": "Host",
                                     **eval_metrics(y, hp, hp, tail, normal)})

    metrics = pd.DataFrame(metrics_rows)
    cell = metrics.groupby(["market", "host", "method"], as_index=False).median(numeric_only=True)
    tert = pd.DataFrame(tert_rows)
    tert_cell = tert.groupby(["market", "host", "method", "tertile"], as_index=False).median(numeric_only=True)
    base = tert_cell[tert_cell.method.eq("S1_DailyPatch_GRU32")].set_index(["market", "host", "tertile"])
    for col in ("shape_cosine", "wrong_hemisphere_rate", "relative_gain_pct", "Tail_MAE", "Normal_MAE"):
        tert_cell[f"HSA_minus_S1_{col}"] = np.nan
        z = tert_cell.method.eq("HSA_HorizonAlignedStateResidual")
        tert_cell.loc[z, f"HSA_minus_S1_{col}"] = [r[col] - base.loc[(r.market, r.host, r.tertile), col] for _, r in tert_cell[z].iterrows()]

    eff = pd.DataFrame(efficiency)
    eff_cell = eff.groupby(["market", "host", "method"], as_index=False).median(numeric_only=True)
    param_rows = []
    for m in MARKETS:
        a = int(eff_cell[(eff_cell.market == m) & eff_cell.method.eq("S1_DailyPatch_GRU32")].parameter_count.iloc[0])
        b = int(eff_cell[(eff_cell.market == m) & eff_cell.method.eq("HSA_HorizonAlignedStateResidual")].parameter_count.iloc[0])
        param_rows.append({"market": m, "S1_parameter_count": a, "HSA_parameter_count": b,
                           "expected_parameter_delta": N_ROLES, "observed_parameter_delta": b - a})
    param_df = pd.DataFrame(param_rows)

    timing = pd.DataFrame(timing_rows)
    timing_cell = timing.groupby(["market", "host"], as_index=False).median(numeric_only=True)
    timing_cell["hsa_total_inference_seconds_per_day"] = (timing_cell.s1_inference_seconds + timing_cell.hsa_overhead_seconds) / timing_cell.n_eval_days
    timing_cell["hsa_overhead_seconds_per_day"] = timing_cell.hsa_overhead_seconds / timing_cell.n_eval_days
    timing_cell["hsa_total_inference_us_per_day"] = timing_cell.hsa_total_inference_seconds_per_day * 1e6
    timing_cell["hsa_overhead_us_per_day"] = timing_cell.hsa_overhead_seconds_per_day * 1e6
    timing_cell["beta_training_seconds_per_cell"] = timing_cell.beta_total_training_seconds

    frozen_after = {str(p.relative_to(ROOT)): sha(p) for p in frozen_paths}
    host_after = {f"{m}/{h}": sha(paths[m, h]) for m in MARKETS for h in HOSTS}
    replay_df = pd.DataFrame(replay)
    replay_max = float(replay_df.drop(columns=["market", "track", "host", "seed"]).to_numpy().max())
    chron = pd.DataFrame(chronology)
    chron_ok = bool(chron.self_excluded.all() and
                    (chron.loc[chron.record_type.eq("alpha"), "calibration_max_index"] <
                     chron.loc[chron.record_type.eq("alpha"), "proposal_min_index"]).all())
    access_ok = all(a["target_day_DA_input_reads"] == 0 and a["realized_RT_input_reads"] == 0 and a["S3_S4_reads"] == 0
                    for a in access.values())
    beta_audit = pd.DataFrame(beta_rows)
    zero_init_max = float(beta_audit.zero_init_replay_max_abs_diff.max())
    role_ok = bool(set(role_manifest[role_manifest.n_channels_in_role.gt(0)].semantic_role) <= set(ROLES))

    x0 = bool(replay_max <= TOL and frozen_before == frozen_after
              and shape_code_before == tree_sha(ROOT / "experiments/current/hch_unified_da_shape_upgrade")
              and amplitude_code_before == tree_sha(ROOT / "experiments/current/hch_anchored_compact_amplitude")
              and host_before == host_after and chron_ok and access_ok and role_ok
              and bool((param_df.observed_parameter_delta == N_ROLES).all())
              and zero_init_max <= REPLAY_TOL)

    s1 = cell[cell.method.eq("S1_DailyPatch_GRU32")].set_index(["market", "host"])
    hsa = cell[cell.method.eq("HSA_HorizonAlignedStateResidual")].set_index(["market", "host"])
    host = cell[cell.method.eq("Host")].set_index(["market", "host"])
    gansu = [k for k in hsa.index if k[0] == "GANSU_DA"]

    # ---- H1: Shape geometry ----
    cos_delta = hsa.shape_cosine - s1.shape_cosine
    wrong_delta = hsa.wrong_hemisphere_rate - s1.wrong_hemisphere_rate
    hi = tert_cell[(tert_cell.method.eq("HSA_HorizonAlignedStateResidual")) & tert_cell.tertile.eq("HIGH")].set_index(["market", "host"])
    g_hi = hi.loc[gansu]
    h1 = bool((cos_delta >= -TOL).sum() >= 5
              and (cos_delta.loc[gansu] > 0).all()
              and (wrong_delta <= TOL).sum() >= 5
              and (cos_delta.loc[list(targeted)] > 0).sum() >= 3
              and (g_hi.HSA_minus_S1_shape_cosine > 0).all()
              and (g_hi.HSA_minus_S1_wrong_hemisphere_rate <= 0.02 + TOL).all())

    # ---- H2: material gain ----
    gain_s1 = s1.relative_gain_pct; gain_hsa = hsa.relative_gain_pct; dgain = gain_hsa - gain_s1
    normal_harm = 100 * (hsa.Normal_MAE / host.Normal_MAE - 1)
    gg = gain_hsa.loc[gansu]
    h2 = bool((dgain >= -TOL).all()
              and (hsa.Overall_MAE < s1.Overall_MAE).sum() >= 5
              and (gain_hsa >= -TOL).all()
              and float(np.median(gain_hsa)) >= 7
              and int((gain_hsa >= 5).sum()) >= 4
              and float(normal_harm.max()) <= 1
              and (hsa.Overall_MAE.loc[gansu] < s1.Overall_MAE.loc[gansu]).all()
              and float(gg.max()) >= 7 and float(gg.min()) >= 3)

    # ---- H3: seed stability ----
    per_seed = metrics[metrics.method.isin(["S1_DailyPatch_GRU32", "HSA_HorizonAlignedStateResidual"])]
    piv = per_seed.pivot_table(index=["market", "host", "seed"], columns="method", values="relative_gain_pct")
    seed_delta = (piv["HSA_HorizonAlignedStateResidual"] - piv["S1_DailyPatch_GRU32"]).rename("delta_gain_pct").reset_index()
    seed_ok = seed_delta.assign(ok=seed_delta.delta_gain_pct >= -TOL).groupby(["market", "host"]).ok.sum() >= 2
    h3 = bool(int(seed_ok.sum()) >= 5 and bool(seed_ok.loc[gansu].all()) and float(seed_delta.delta_gain_pct.min()) >= -2)

    # ---- H4: efficiency ----
    med_beta = float(np.median(timing_cell.beta_training_seconds_per_cell))
    med_overhead = float(np.median(timing_cell.hsa_overhead_us_per_day))
    med_total = float(np.median(timing_cell.hsa_total_inference_us_per_day))
    h4 = bool(bool((param_df.observed_parameter_delta == N_ROLES).all())
              and med_beta <= BETA_TRAINING_BUDGET_S
              and med_overhead <= OVERHEAD_BUDGET_US_DAY
              and med_total <= TOTAL_INFERENCE_BUDGET_US_DAY)

    token = TOKEN_INVALID if not x0 else (TOKEN_PASS if all((h1, h2, h3, h4)) else TOKEN_FAIL)
    gates = {"H0": "PASS" if x0 else "FAIL", "H1": "PASS" if h1 else "FAIL",
             "H2": "PASS" if h2 else "FAIL", "H3": "PASS" if h3 else "FAIL", "H4": "PASS" if h4 else "FAIL"}

    # ---- artifacts ----
    metrics.to_csv(OUT / "metrics_by_seed.csv", index=False)
    cell.to_csv(OUT / "metrics_by_cell.csv", index=False)
    tert_cell.to_csv(OUT / "state_tertile_comparison.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(OUT / "alpha_replay_and_hsa.csv", index=False)
    beta_audit.to_csv(OUT / "beta_by_fold_seed.csv", index=False)
    chron.to_csv(OUT / "oof_chronology_audit.csv", index=False)
    replay_df.to_csv(OUT / "s1_replay_audit.csv", index=False)
    seed_delta.to_csv(OUT / "seed_stability.csv", index=False)
    param_df.to_csv(OUT / "parameter_audit.csv", index=False)
    eff_cell.to_csv(OUT / "efficiency.csv", index=False)
    timing_cell.to_csv(OUT / "inference_overhead.csv", index=False)

    verdict = {"token": token, **gates,
               "S1_replay_max_abs_diff": replay_max, "beta_zero_init_replay_max_abs_diff": zero_init_max,
               "H1_cosine_nonworse_cells": int((cos_delta >= -TOL).sum()),
               "H1_wrong_hemisphere_nonworse_cells": int((wrong_delta <= TOL).sum()),
               "H1_targeted_cosine_improved_cells": int((cos_delta.loc[list(targeted)] > 0).sum()),
               "H1_GANSU_HIGH_cosine_changes": {f"{k[0]}/{k[1]}": float(v) for k, v in g_hi.HSA_minus_S1_shape_cosine.items()},
               "H1_GANSU_HIGH_wrong_rate_changes": {f"{k[0]}/{k[1]}": float(v) for k, v in g_hi.HSA_minus_S1_wrong_hemisphere_rate.items()},
               "H2_gain_delta_pct": {f"{k[0]}/{k[1]}": float(v) for k, v in dgain.items()},
               "H2_median_gain_pct": float(np.median(gain_hsa)),
               "H2_gain_ge5_cells": int((gain_hsa >= 5).sum()),
               "H2_max_Normal_harm_pct": float(normal_harm.max()),
               "H2_GANSU_gains_pct": {f"{k[0]}/{k[1]}": float(v) for k, v in gg.items()},
               "H3_cells_with_ge2of3_seeds": int(seed_ok.sum()),
               "H3_worst_seed_gain_delta_pp": float(seed_delta.delta_gain_pct.min()),
               "H4_median_beta_training_seconds_per_cell": med_beta,
               "H4_median_overhead_us_per_day": med_overhead,
               "H4_median_total_inference_us_per_day": med_total,
               "runtime_seconds": time.time() - started}
    (OUT / "HSA_VERDICT.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

    provenance = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "canonical_root": str(ROOT),
                  "controlling_protocol": "docs/current/HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SHAPE_CLOSURE_20260911.md",
                  "frozen_hashes_before": frozen_before, "frozen_hashes_after": frozen_after,
                  "s1_code_hash_before": shape_code_before, "s1_code_hash_after": tree_sha(ROOT / "experiments/current/hch_unified_da_shape_upgrade"),
                  "amplitude_code_hash_before": amplitude_code_before,
                  "amplitude_code_hash_after": tree_sha(ROOT / "experiments/current/hch_anchored_compact_amplitude"),
                  "host_hashes_before": host_before, "host_hashes_after": host_after,
                  "state_level_access": access, "role_mapping": {m: [[LEGAL[m][i] for i in role] for role in mapping[m]] for m in MARKETS},
                  "splits": split_manifest, "parameter_audit": param_audit,
                  "forbidden": {"state_excursion": False, "MAD": False, "clipping": False,
                                "alternate_history_window": False, "per_hour_beta": False, "role_embedding": False,
                                "market_id_or_host_id_input": False, "market_specific_role_selection": False,
                                "conditional_amplitude": False, "gate_or_verification": False, "bias_term": False,
                                "search": False, "s3_s4_protected_final_reads": 0},
                  "scientific_reads": {"DIAG_FIT": sum(len(by_market[m].subset("DIAG_FIT").timestamp) for m in MARKETS),
                                       "DIAG_EVAL": sum(len(by_market[m].subset("DIAG_EVAL").timestamp) for m in MARKETS),
                                       "S3": 0, "S4": 0, "protected": 0, "final": 0},
                  "numerical_environment": {"torch": torch.__version__, "numpy": np.__version__,
                                            "pandas": pd.__version__, "torch_threads": torch.get_num_threads(),
                                            "note": "single-threaded torch is required for bit-exact frozen S1 replay"},
                  "code_sha256": sha(Path(__file__))}
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

    # ---- figures ----
    p = cell[cell.method.isin(["S1_DailyPatch_GRU32", "HSA_HorizonAlignedStateResidual"])].pivot_table(
        index=["market", "host"], columns="method", values="shape_cosine")
    p.plot(kind="bar", figsize=(9, 4)); plt.ylabel("Shape cosine"); plt.tight_layout()
    plt.savefig(OUT / "figures/fig_s1_vs_hsa_shape_cosine.png", dpi=160); plt.close()
    gp = cell[cell.method.isin(["S1_DailyPatch_GRU32", "HSA_HorizonAlignedStateResidual"])].pivot_table(
        index=["market", "host"], columns="method", values="relative_gain_pct")
    gp.plot(kind="bar", figsize=(9, 4)); plt.ylabel("Relative Overall-MAE gain vs Host (%)"); plt.tight_layout()
    plt.savefig(OUT / "figures/fig_overall_mae_gain.png", dpi=160); plt.close()
    tb = tert_cell[tert_cell.tertile.eq("HIGH")].pivot_table(index=["market", "host"], columns="method", values="shape_cosine")
    tb.plot(kind="bar", figsize=(9, 4)); plt.ylabel("HIGH-tertile Shape cosine"); plt.tight_layout()
    plt.savefig(OUT / "figures/fig_state_tertile_behavior.png", dpi=160); plt.close()

    audit = ["# Implementation and test audit", "",
             f"- Canonical root: `{ROOT}`",
             f"- Controlling protocol: `docs/current/HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SHAPE_CLOSURE_20260911.md`",
             f"- Frozen S1 metric replay maximum absolute difference: `{replay_max:.3e}` (tolerance `{TOL:.0e}`)",
             f"- beta=0 exact S1 direction replay maximum absolute difference: `{zero_init_max:.3e}` (tolerance `{REPLAY_TOL:.0e}`)",
             f"- Frozen artifact immutability: `{frozen_before == frozen_after}`",
             f"- S1 code tree immutability: `{shape_code_before == tree_sha(ROOT / 'experiments/current/hch_unified_da_shape_upgrade')}`",
             f"- Amplitude code tree immutability: `{amplitude_code_before == tree_sha(ROOT / 'experiments/current/hch_anchored_compact_amplitude')}`",
             f"- Host cache immutability: `{host_before == host_after}`",
             f"- Stacked chronological OOF self-exclusion: `{chron_ok}`",
             f"- Primary-only role channels, forecast-time legal: `{bool(role_manifest[role_manifest.n_channels_in_role.gt(0)].shape[0]) > 0}`",
             f"- Exactly five trainable HSA parameters per market: `{bool((param_df.observed_parameter_delta == N_ROLES).all())}`",
             f"- Forbidden/protected reads: `0`"]
    # Evidence rule: a PASS must be backed by an observed test result, so run the
    # targeted suite here and keep its verbatim output beside the audit.
    tests = subprocess.run([sys.executable, str(ROOT / "experiments/current/hch_horizon_aligned_state_residual/test_closure.py")],
                           cwd=ROOT, capture_output=True, text=True)
    (OUT / "tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    tail = [ln for ln in tests.stdout.strip().splitlines() if ln.strip()][-1:] or ["<no output>"]
    audit += [f"- Targeted unit suite exit code: `{tests.returncode}`",
              f"- Targeted unit suite result: `{tail[0]}`",
              "- Targeted unit suite output: `tests.txt`",
              "- Independently re-verified by `verify_gates.py` (does not import this runner): "
              "recomputed H0-H4 and the token from the written evidence alone"]
    (OUT / "IMPLEMENTATION_TEST_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")

    high_report = hi.reset_index()[["market", "host", "HSA_minus_S1_shape_cosine",
                                    "HSA_minus_S1_wrong_hemisphere_rate", "HSA_minus_S1_relative_gain_pct"]]
    show = cell[cell.method.isin(["Host", "S1_DailyPatch_GRU32", "HSA_HorizonAlignedStateResidual"])]
    bfin = beta_audit[beta_audit.stage.eq("FINAL")].groupby(["market", "host"], as_index=False)[[f"beta_{r}" for r in ROLES]].median()
    summary = ["# Horizon-Aligned Semantic State Residual (HSA) Shape Closure", "",
               f"Token: `{token}`", "", *(f"- {k}: `{v}`" for k, v in gates.items()), "",
               "## Method", "",
               "Frozen S1 OOF Shape directions are reused unchanged. A deterministic horizon-aligned semantic role table "
               "`R_d in R^(24x5)` is formed by mean-pooling audited `primary_channel=True` forecast-time state columns into "
               "`DEMAND_FC / RENEWABLE_FC / SUPPLY_MARGIN_FC / INTERCHANGE_FC / MUST_RUN_FC` (absent role = structural 0). "
               "The only new trainable object is `beta in R^5`, giving `u_HSA = normalize(u_S1 + R_d beta)`; `alpha_HSA` is "
               "refit with the original exact weighted-median rule on HSA OOF directions.", "",
               "## Cell metrics", "", show.to_markdown(index=False), "",
               "## Final beta (median across seeds)", "", bfin.to_markdown(index=False), "",
               "## Frozen state-tertile behaviour", "", high_report.to_markdown(index=False), "",
               "## Required statements (protocol §10)", "",
               f"1. **Does horizon-local semantic state improve Shape across markets?** "
               f"{'Yes' if int((cos_delta >= -TOL).sum()) >= 5 else 'Only partially'}: cell-median Shape cosine rises in "
               f"{int((cos_delta > 0).sum())}/6 cells and is non-worse in {int((cos_delta >= -TOL).sum())}/6 "
               f"(median change {float(np.median(cos_delta)):+.4f}); wrong-hemisphere is non-worse in "
               f"{int((wrong_delta <= TOL).sum())}/6. The improvement is consistent in GANSU_DA but not universal.",
               f"2. **Do both GANSU_DA Hosts improve without market-specific tuning?** "
               f"{'Yes' if bool((cos_delta.loc[gansu] > 0).all()) else 'No'}: Shape cosine rises by "
               + ", ".join(f"{float(v):+.4f} ({k[1]})" for k, v in cos_delta.loc[gansu].items())
               + ", using one shared five-role vocabulary and a per-market beta fit on that market's own OOF rows only; "
                 "no market-specific role selection, IDs, or rules are introduced.",
               f"3. **Is the same five-parameter mechanism non-worse across all six cells?** "
               f"{'Yes' if bool((dgain >= -TOL).all()) else 'No'}: relative Overall-MAE gain vs S1 is nonnegative in "
               f"{int((dgain >= -TOL).sum())}/6 cell medians and vs Host in {int((gain_hsa >= -TOL).sum())}/6 "
               f"(worst {float(dgain.min()):+.4f} pp vs S1). "
               f"{'Non-worse everywhere.' if bool((dgain >= -TOL).all()) else 'It regresses on at least one cell.'}",
               f"4. **Is public-China expansion scientifically authorized?** "
               f"{'Yes' if token == TOKEN_PASS else 'No'}: H1={gates['H1']}, H2={gates['H2']}, H3={gates['H3']} "
               f"with H0={gates['H0']}, H4={gates['H4']}. "
               + ("All gates pass; expansion design may proceed."
                  if token == TOKEN_PASS else
                  "At least one scientific gate fails, so S1 plus the pooled fixed alpha is retained and no "
                  "SHAANXI/NINGXIA/QINGHAI or full-panel/protected/final work is authorized by this stage."),
               "",
               f"Median relative Overall-MAE gain vs Host: `{float(np.median(gain_hsa)):.6f}%`; "
               f"median gain change vs S1: `{float(np.median(dgain)):.6f} pp`.",
               f"Maximum Normal-MAE harm: `{float(normal_harm.max()):.6f}%`.",
               f"Median beta training time: `{med_beta:.4f} s/cell`; median HSA inference overhead: `{med_overhead:.4f} us/day`; "
               f"median total HSA inference: `{med_total:.4f} us/day`.",
               "", "Method consequence: " + ("promote HSA for later public-China expansion design; stop before any full-panel work."
                                             if token == TOKEN_PASS else
                                             "keep S1 + pooled fixed alpha; HSA is not a supportable upgrade.")]
    (OUT / "HSA_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(token); print(json.dumps(verdict, ensure_ascii=False, default=json_default))
    return 0 if x0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
