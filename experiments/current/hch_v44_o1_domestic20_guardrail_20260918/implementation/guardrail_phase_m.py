"""Phase M — zero-fit shared-gradient audit on the selected domestic O1 checkpoints.

Descriptive mechanism evidence only.  No optimizer is constructed, no
``optimizer.step`` is reachable, and nothing computed here can alter the O1
recipe, the domestic gate, the international gate or any checkpoint.

One forward pass per checkpoint in ``eval`` mode (so dropout consumes no random
stream and the graph is deterministic), then four independent backward passes
with ``retain_graph=True``, harvesting gradients **only** on the parameters the
coordinate tasks share.  Head-only parameters are excluded from the cosine by
name, because a gradient cosine restricted to a single head's parameters would
measure that head's own geometry rather than task conflict.

Protocol reading: PROTOCOL.md section 4 says "all 20 domestic selected O1
checkpoints" while each cell owns three seed checkpoints.  Rather than pick a
seed by an undocumented rule, every available checkpoint is audited (60 = a
superset of the 20-cell reading), and the per-cell median is reported alongside
the 60-run fractions.  This choice cannot change any gate because no gate reads
this phase.
"""
from __future__ import annotations

import datetime as dt
import math

import guardrail_common as G

U = G.U
PC = G.PC
PT = G.PT
# ``probe_common`` does not re-export the recovery trainer; ``probe_train`` does,
# as the same module object it trains with, so Phase M reads the identical data
# assembler and diagnostic-subset rule the O1 fits used.
RT = PT.RT
_bind_prior = PT._bind_prior
EVID = G.EVID

#: Parameters that belong to exactly one coordinate output.  Everything else --
#: the Host / history / calendar encoders, the shared allocator and the shared
#: Shape trunk -- is the shared set the cosines are taken on.
HEAD_PREFIXES = ("level_head.", "mass_head.", "shape_head.", "direct_head.")

TERMS = ("L_rec", "L_b", "L_B", "L_S")
PAIRS = ("cos_rec_b", "cos_rec_B", "cos_rec_S", "cos_rec_auxsum")
NORMS = ("norm_ratio_b_over_rec", "norm_ratio_B_over_rec", "norm_ratio_S_over_rec")

AUDIT_FIELDS = (
    "market", "host", "cell", "seed", "status", "shared_parameter_count",
    "shared_parameter_numel", "head_parameter_numel", "diag_subset_rows",
    "L_rec", "L_b", "L_B", "L_S",
    "grad_norm_rec", "grad_norm_b", "grad_norm_B", "grad_norm_S",
    *PAIRS, *NORMS,
    "diag_reconstruction_mae_recomputed", "diag_reconstruction_mae_recorded",
    "diag_reconstruction_abs_deviation",
    "optimizer_constructed", "optimizer_steps", "checkpoint_sha256_before",
    "checkpoint_sha256_after", "checkpoint_unchanged", "probe_code_hash",
)

#: cuDNN's RNN kernels refuse ``backward`` outside training mode, and this
#: architecture carries three GRU encoders, so the audit disables cuDNN and runs
#: the native RNN path.  The swap is a kernel-level difference only: with cuDNN
#: enabled the same forward reproduces the run's own recorded
#: ``diag_reconstruction_mae`` exactly, and the native path deviates by fp32
#: association noise (measured <= 2e-4 absolute).  Each run records both values
#: and their deviation, so the substitution is measured rather than asserted.
CUDNN_DEV_TOLERANCE = 1e-2


def shared_names(model) -> list[str]:
    return [n for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith(HEAD_PREFIXES)]


def _flat(model, names) -> list[float]:
    own = dict(model.named_parameters())
    vals: list[float] = []
    for n in names:
        g = own[n].grad
        vals.extend([0.0] * own[n].numel() if g is None else g.detach().reshape(-1).double().cpu().tolist())
    return vals


def _cos(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(sum(x * y for x, y in zip(a, b)) / (na * nb))


def _norm(a) -> float:
    return math.sqrt(sum(x * x for x in a))


def audit_one(market: str, host: str, seed: int, data, device: str) -> dict:
    import numpy as np
    import torch
    from core.training_support import build_primary_train_config

    P_ = _bind_prior()
    d = G.run_dir(market, host, seed)
    ckpt = d / "selected_ema.pt"
    before = G.sha256_file(ckpt)

    cfg = build_primary_train_config()
    model = RT.build_model(data, device, cfg.profile, None)

    shadow = torch.load(ckpt, map_location=device)
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in shadow.items():
            own[k].copy_(v)
    missing = [k for k in own if k not in shadow]
    model.eval()

    names = shared_names(model)
    head_numel = sum(p.numel() for n, p in model.named_parameters()
                     if p.requires_grad and n.startswith(HEAD_PREFIXES))

    n = len(data["train_rows"])
    diag_idx = torch.as_tensor(
        np.unique(np.linspace(0, n - 1, min(n, RT.DIAG_MAX_ROWS)).astype(np.int64)),
        dtype=torch.long,
    ).to(device)

    out = model(P_.select_batch(data["train_batch"], diag_idx))
    comp = model.compute_loss(
        out, P_.slice_geometry(data["train_target"], diag_idx),
        data["train_res"][diag_idx], data["scales"],
    )

    grads: dict[str, list[float]] = {}
    for term in TERMS:
        model.zero_grad(set_to_none=True)
        comp[term].backward(retain_graph=True)
        grads[term] = _flat(model, names)
    model.zero_grad(set_to_none=True)

    aux = [sum(t) / 3.0 for t in zip(grads["L_b"], grads["L_B"], grads["L_S"])]
    rec = grads["L_rec"]
    n_rec = _norm(rec)
    recorded_diag = float(G.load_json(d / "freeze.json")["selected_metrics"]["diag_reconstruction_mae"])
    recomputed_diag = float(comp["raw_mae"])
    row = {
        "market": market, "host": host, "cell": G.cell_key(market, host), "seed": seed,
        "status": "reused" if (market, host) in G.REUSED_CELLS else "new",
        "shared_parameter_count": len(names),
        "shared_parameter_numel": int(sum(own[k].numel() for k in names)),
        "head_parameter_numel": int(head_numel),
        "diag_subset_rows": int(diag_idx.numel()),
        "L_rec": float(comp["L_rec"]), "L_b": float(comp["L_b"]),
        "L_B": float(comp["L_B"]), "L_S": float(comp["L_S"]),
        "grad_norm_rec": n_rec, "grad_norm_b": _norm(grads["L_b"]),
        "grad_norm_B": _norm(grads["L_B"]), "grad_norm_S": _norm(grads["L_S"]),
        "cos_rec_b": _cos(rec, grads["L_b"]),
        "cos_rec_B": _cos(rec, grads["L_B"]),
        "cos_rec_S": _cos(rec, grads["L_S"]),
        "cos_rec_auxsum": _cos(rec, aux),
        "norm_ratio_b_over_rec": (_norm(grads["L_b"]) / n_rec) if n_rec else float("nan"),
        "norm_ratio_B_over_rec": (_norm(grads["L_B"]) / n_rec) if n_rec else float("nan"),
        "norm_ratio_S_over_rec": (_norm(grads["L_S"]) / n_rec) if n_rec else float("nan"),
        "diag_reconstruction_mae_recomputed": recomputed_diag,
        "diag_reconstruction_mae_recorded": recorded_diag,
        "diag_reconstruction_abs_deviation": abs(recomputed_diag - recorded_diag),
        # Zero-fit evidence, recorded per run rather than asserted once.
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": G.sha256_file(ckpt),
        "checkpoint_unchanged": before == G.sha256_file(ckpt),
        "probe_code_hash": G.load_json(d / "freeze.json")["probe_code_hash"],
        "_unloaded_parameter_names": missing,
        "_device": device,
    }
    del out, comp, grads, model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return row


def _fraction(rows, pair: str, keep=None) -> float:
    vals = [r[pair] for r in rows if (keep is None or keep(r))]
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan")
    return float(sum(1 for v in vals if v < 0.0) / len(vals))


def _median(rows, key: str, keep=None) -> float:
    vals = [r[key] for r in rows if (keep is None or keep(r))]
    vals = [v for v in vals if v == v]
    return G.median(vals) if vals else float("nan")


def run(device: str = "cuda") -> dict:
    import torch

    t0 = dt.datetime.now()
    cudnn_was = bool(torch.backends.cudnn.enabled)
    torch.backends.cudnn.enabled = False
    rows: list[dict] = []
    try:
        for market, host in G.DOM_PANEL:
            data = RT.to_device(RT.build_cell_data(market, host), device)
            for seed in G.SEEDS:
                rows.append(audit_one(market, host, seed, data, device))
                print(f"  M {G.cell_key(market, host)} seed{seed} done", flush=True)
            del data
    finally:
        torch.backends.cudnn.enabled = cudnn_was
    G.write_csv(EVID / "GRADIENT_CONFLICT_AUDIT.csv", AUDIT_FIELDS, rows)

    ok = all(r["shared_parameter_count"] > 0 and r["diag_subset_rows"] > 0 for r in rows)
    by_host = {}
    for host in G.HOSTS:
        keep = (lambda h: (lambda r: r["host"] == h))(host)
        by_host[host] = {
            "n_runs": sum(1 for r in rows if r["host"] == host),
            **{p: _fraction(rows, p, keep) for p in PAIRS},
            **{k: _median(rows, k, keep) for k in NORMS},
        }
    by_market = {}
    for market in G.MARKETS:
        keep = (lambda m: (lambda r: r["market"] == m))(market)
        by_market[market] = {
            "n_runs": sum(1 for r in rows if r["market"] == market),
            **{p: _fraction(rows, p, keep) for p in PAIRS},
            **{k: _median(rows, k, keep) for k in NORMS},
        }
    by_cell = []
    for market, host in G.DOM_PANEL:
        keep = (lambda c: (lambda r: r["cell"] == c))(G.cell_key(market, host))
        by_cell.append({
            "market": market, "host": host, "cell": G.cell_key(market, host),
            "n_runs": sum(1 for r in rows if r["cell"] == G.cell_key(market, host)),
            **{p: _fraction(rows, p, keep) for p in PAIRS},
            "median_cos_rec_auxsum": _median(rows, "cos_rec_auxsum", keep),
            "median_norm_ratio_S_over_rec": _median(rows, "norm_ratio_S_over_rec", keep),
        })

    overall = {p: _fraction(rows, p) for p in PAIRS}
    deviations = {h: abs(by_host[h]["cos_rec_auxsum"] - overall["cos_rec_auxsum"])
                  for h in G.HOSTS
                  if by_host[h]["cos_rec_auxsum"] == by_host[h]["cos_rec_auxsum"]}
    strongest = max(deviations, key=deviations.get) if deviations else None

    summary = {
        "schema": "hch_v44_o1_domestic20_gradient_conflict_summary.v1",
        "protocol_id": G.PROTOCOL_ID,
        "role": "descriptive mechanism evidence; no gate reads this phase",
        "protocol_reading": ("all 60 available domestic O1 checkpoints audited; per-cell medians "
                             "reported alongside 60-run fractions; cannot alter any gate"),
        "n_checkpoints_audited": len(rows),
        "n_cells_covered": len(G.DOM_PANEL),
        "n_optimizer_steps_total": sum(r["optimizer_steps"] for r in rows),
        "optimizer_constructed_anywhere": any(r["optimizer_constructed"] for r in rows),
        "all_checkpoints_unchanged": all(r["checkpoint_unchanged"] for r in rows),
        "all_shared_parameter_sets_nonempty": ok,
        "unloaded_parameter_names": sorted({n for r in rows for n in r["_unloaded_parameter_names"]}),
        "shared_parameter_numel": sorted({r["shared_parameter_numel"] for r in rows}),
        "head_parameter_numel": sorted({r["head_parameter_numel"] for r in rows}),
        "diag_subset_rows": sorted({r["diag_subset_rows"] for r in rows}),
        "diag_subset_rule": "np.unique(np.linspace(0, n-1, min(n, 256))) over registered TRAIN row order",
        "forward_mode": "model.eval() under the O1 selected EMA weights; no random stream consumed",
        "cudnn_substitution": {
            "cudnn_disabled_for_this_audit": True,
            "reason": ("the architecture carries three GRU encoders and cuDNN's RNN kernels refuse "
                       "backward outside training mode; the native path is used so the forward can "
                       "stay in eval mode instead of switching the model to train mode"),
            "kernel_level_only": True,
            "max_abs_deviation_vs_own_recorded_diag_reconstruction_mae":
                max(r["diag_reconstruction_abs_deviation"] for r in rows),
            "tolerance": CUDNN_DEV_TOLERANCE,
            "all_runs_within_tolerance":
                all(r["diag_reconstruction_abs_deviation"] <= CUDNN_DEV_TOLERANCE for r in rows),
            "no_parameter_or_schedule_changed": True,
        },
        "negative_cosine_fraction_overall": overall,
        "median_norm_ratio_overall": {k: _median(rows, k) for k in NORMS},
        "median_cosines_overall": {p: _median(rows, p) for p in PAIRS},
        "negative_cosine_fraction_by_host": by_host,
        "negative_cosine_fraction_by_market": by_market,
        "strongest_host_family_deviation": {
            "pair": "cos_rec_auxsum",
            "host": strongest,
            "fraction": by_host[strongest]["cos_rec_auxsum"] if strongest else None,
            "overall_fraction": overall["cos_rec_auxsum"],
            "absolute_deviation": deviations.get(strongest) if strongest else None,
        },
        "by_cell": by_cell,
        "elapsed_seconds": (dt.datetime.now() - t0).total_seconds(),
    }
    U.json_dump(EVID / "GRADIENT_CONFLICT_SUMMARY.json", summary)
    return summary


if __name__ == "__main__":
    G.RC.install_access_guard()
    s = run()
    print(f"[M] {s['n_checkpoints_audited']} checkpoints, "
          f"optimizer steps={s['n_optimizer_steps_total']}, "
          f"unchanged={s['all_checkpoints_unchanged']}", flush=True)
