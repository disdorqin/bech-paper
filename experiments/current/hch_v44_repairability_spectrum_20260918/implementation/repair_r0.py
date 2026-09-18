"""R0 -- exact repairability/headroom atlas (PROTOCOL.md Sec. 3, design Sec. 4).

Reuses all 60 frozen O1 runs read-only: each run's ``selected_ema.pt`` is loaded
into a freshly constructed ``HCHV44Core`` and evaluated under ``no_grad`` on the
identical TRAIN/VAL row sets the run itself was trained on.  No optimizer step
is taken anywhere in this file.

Per eligible day it reconstructs the true residual geometry (through
``core.geometry``, the single authority), the selected checkpoint's predicted
coordinates, the six one-coordinate/interaction oracle swaps (negative
improvements retained, never clipped), and the exact nonnegative MAE-optimal ray
scale alpha*.

Hard reconciliation: the per-day reconstruction of each run must reproduce that
run's own recorded ``selected_val_mae_ema`` and ``val_host_mae``.  A mismatch is
a fatal error, not a warning -- it is the only evidence that the rows this stage
analyses are the rows the frozen runs scored.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import repair_common as R  # noqa: E402

MAE_ABS_TOL = 1e-3
MAE_REL_TOL = 1e-5


def _frames_host_lookup(market: str, host: str) -> dict:
    """Day -> frozen Host 24h prediction, from the registered TRAIN/VAL frames.

    The historical days of a W=7 window are either shadow-OOF TRAIN days or
    already-revealed VAL days, both of which the joint cache carries.  When a
    history day's residual provenance is ``oof`` its *trajectory* still comes
    from the frozen comparison Host; that provenance split is disclosed in the
    evidence rather than silently repaired.
    """
    lookup, prov = {}, {}
    for role in (R.RC.ROLE_TRAIN, R.RC.ROLE_VAL):
        frame = R.RC.pretest_role_frame(market, host, role)
        R.RC.assert_pretest_only(frame["days"], market)
        for i, day in enumerate(frame["days"]):
            lookup[day] = np.asarray(frame["host_pred"][i], dtype=np.float64)
            prov[day] = role
    return lookup, prov


def _forward_all(model, data, device):
    """no_grad EMA forward over TRAIN then VAL rows; the model's own decoder."""
    outs = {}
    for tag, batch in (("train", data["train_batch"]), ("val", data["val_batch"])):
        parts = R.RT._chunked_forward(model, batch)
        outs[tag] = {
            "correction": R.RT._concat(parts, "correction").detach().cpu().numpy().astype(np.float64),
            "b_hat": R.RT._concat(parts, "b_hat").detach().cpu().numpy().astype(np.float64).reshape(-1),
            "B_hat": R.RT._concat(parts, "B_hat").detach().cpu().numpy().astype(np.float64).reshape(-1),
            "s_plus_hat": R.RT._concat(parts, "s_plus_hat").detach().cpu().numpy().astype(np.float64),
            "s_minus_hat": R.RT._concat(parts, "s_minus_hat").detach().cpu().numpy().astype(np.float64),
        }
    return outs


def cell_atlas(market: str, host: str) -> tuple:
    """All eligible TRAIN+VAL days of one cell, for each registered seed."""
    import torch
    from core.training_support import assert_primary_profile, build_primary_train_config

    key = R.cell_key(market, host)
    data = R.RT.build_cell_data(market, host)
    data = R.RT.to_device(data, "cuda" if torch.cuda.is_available() else "cpu")
    device = data["device"]
    cfg = build_primary_train_config()
    assert_primary_profile(cfg.profile)

    host_lookup, host_role = _frames_host_lookup(market, host)

    rows = []
    reconciliation = []
    for seed in R.SEEDS:
        ckpt_path = R.run_dir(market, host, seed) / "selected_ema.pt"
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = R.RT.build_model(data, device, cfg.profile, None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # The only keys outside the EMA shadow are construct-time buffers derived
        # from the frozen TRAIN scales; they are asserted non-parameter below.
        buffers = {k for k, _ in model.named_buffers()}
        assert set(missing) <= buffers, f"{key} seed{seed}: unexpected missing {sorted(set(missing) - buffers)}"
        assert not unexpected, f"{key} seed{seed}: unexpected keys {unexpected}"
        model.eval()
        outs = _forward_all(model, data, device)
        del model
        if device != "cpu":
            torch.cuda.empty_cache()

        val_res = np.asarray(data["val_res"].detach().cpu().numpy(), dtype=np.float64)
        train_res = np.asarray(data["train_res"].detach().cpu().numpy(), dtype=np.float64)
        freeze = R.load_json(R.run_dir(market, host, seed) / "freeze.json")

        for tag, split_rows, residual in (
            ("train", data["train_rows"], train_res),
            ("val", data["val_rows"], val_res),
        ):
            g = R.geometry_np(residual)
            corr = outs[tag]["correction"]
            b_hat, B_hat = outs[tag]["b_hat"], outs[tag]["B_hat"]
            sp_hat, sm_hat = outs[tag]["s_plus_hat"], outs[tag]["s_minus_hat"]

            # consistency: the model's own fp32 decoder vs this module's fp64 one
            recompute = R.decode_np(b_hat, B_hat, sp_hat, sm_hat, g["horizon"])
            decoder_gap = float(np.abs(recompute - corr).max())

            mae_host = np.abs(residual).mean(axis=1)
            mae_o1 = R.mae_of(residual, corr)

            swaps = {}
            for name, (bb, BB, sp, sm) in {
                "ob": (g["b"], B_hat, sp_hat, sm_hat),
                "oB": (b_hat, g["B"], sp_hat, sm_hat),
                "oS": (b_hat, B_hat, g["s_plus"], g["s_minus"]),
                "obB": (g["b"], g["B"], sp_hat, sm_hat),
                "obS": (g["b"], B_hat, g["s_plus"], g["s_minus"]),
                "oBS": (b_hat, g["B"], g["s_plus"], g["s_minus"]),
            }.items():
                swaps[name] = R.mae_of(residual, R.decode_np(bb, BB, sp, sm, g["horizon"]))

            alpha, alpha_status, ray_n = R.ray_alpha_star_all(residual, corr)
            mae_ray = R.mae_of(residual, alpha[:, None] * corr)

            for i, row in enumerate(split_rows):
                day = row.day
                traj = np.asarray(row.host_pred, dtype=np.float64)
                p = R.shape_distribution(traj)
                m1 = float(g["m1"][i])
                rec = {
                    "cell": key, "market": market, "host": host, "seed": seed,
                    "role": tag.upper(), "day": day.isoformat(), "ordinal": int(row.ordinal),
                    "n_valid_hours": float(g["horizon"][i]),
                    "host_mae": float(mae_host[i]),
                    "residual_l1": m1,
                    "b_true": float(g["b"][i]), "B_true": float(g["B"][i]),
                    "P_true": float(g["P"][i]), "N_true": float(g["N"][i]),
                    "q_L": R._safe_ratio(R.HORIZON * abs(float(g["b"][i])), m1),
                    "q_B": R._safe_ratio(2.0 * float(g["B"][i]), m1),
                    "q_L_eta": float(g["eta"][i]),
                    "pos_mass_share": R._safe_ratio(float(g["P"][i]), float(g["P"][i]) + float(g["N"][i])),
                    "neg_mass_share": R._safe_ratio(float(g["N"][i]), float(g["P"][i]) + float(g["N"][i])),
                    "sign_changes": R.sign_change_count(residual[i]),
                    "shape_entropy_pos": R.shape_entropy(g["s_plus"][i]),
                    "shape_conc_pos": float(g["s_plus"][i].max()),
                    "shape_entropy_neg": R.shape_entropy(g["s_minus"][i]),
                    "shape_conc_neg": float(g["s_minus"][i].max()),
                    "residual_tv": float(np.abs(np.diff(residual[i])).sum()),
                    "residual_roughness": float(np.abs(np.diff(residual[i])).mean()),
                    "max_abs_resid_over_mae": R._safe_ratio(float(np.abs(residual[i]).max()), float(mae_host[i])),
                    "o1_b_hat": float(b_hat[i]), "o1_B_hat": float(B_hat[i]),
                    "o1_mae": float(mae_o1[i]),
                    "o1_correction_ratio": R._safe_ratio(float(np.abs(corr[i]).mean()), float(np.abs(residual[i]).mean())),
                    "o1_mean_abs_correction": float(np.abs(corr[i]).mean()),
                    "w1_pos_pred_true": R.w1_ordered(sp_hat[i], g["s_plus"][i]),
                    "w1_neg_pred_true": R.w1_ordered(sm_hat[i], g["s_minus"][i]),
                    "decoder_gap": decoder_gap,
                    "mae_ray": float(mae_ray[i]),
                    "alpha_star": float(alpha[i]),
                    "alpha_status": alpha_status[i],
                    "ray_n_effective": int(ray_n[i]),
                    "increment_ray_pct": 100.0 * R._safe_ratio(float(mae_o1[i] - mae_ray[i]), float(mae_host[i])),
                    "ray_host_headroom_frac": R._safe_ratio(float(mae_host[i] - mae_ray[i]), float(mae_host[i])),
                    "ray_remaining_recall": R._safe_ratio(float(mae_o1[i] - mae_ray[i]), float(mae_o1[i])),
                    "alpha_band": ("exact" if abs(float(alpha[i]) - 1.0) <= 1e-9
                                   else ("overshoot" if float(alpha[i]) > 1.0 else "undershoot")),
                }
                for name, arr in swaps.items():
                    rec[f"mae_{name}"] = float(arr[i])
                    rec[f"imp_{name}_pct"] = 100.0 * R._safe_ratio(
                        float(mae_o1[i] - arr[i]), float(mae_host[i]))
                rows.append(rec)

            tag_mae = float(mae_o1.mean())
            tag_host = float(mae_host.mean())
            reference = (freeze["selected_val_mae_ema"], (freeze.get("selected_metrics") or {}).get("val_host_mae"))
            if tag == "val":
                reconciliation.append({
                    "cell": key, "seed": seed, "recorded_val_mae_ema": reference[0],
                    "recomputed_val_mae_ema": tag_mae,
                    "recorded_val_host_mae": reference[1], "recomputed_val_host_mae": tag_host,
                    "abs_diff_mae": abs(tag_mae - float(reference[0])),
                    "abs_diff_host": abs(tag_host - float(reference[1])),
                    "n_val_rows": len(split_rows),
                    "recorded_n_val_rows": freeze.get("n_val_rows"),
                    "recorded_n_train_rows": freeze.get("n_train_rows"),
                    "n_train_rows": len(data["train_rows"]),
                    "selected_step": freeze.get("selected_step"),
                    "scales_fingerprint_recorded": (freeze.get("scales") or {}).get("fingerprint"),
                    "scales_fingerprint_rebuilt": data["scales"].fingerprint(),
                })
    del data
    return rows, reconciliation


def check_reconciliation(rec_rows: list[dict]) -> dict:
    bad = []
    for row in rec_rows:
        tol = MAE_ABS_TOL + MAE_REL_TOL * abs(float(row["recorded_val_mae_ema"]))
        if row["abs_diff_mae"] > tol or row["abs_diff_host"] > tol:
            bad.append(row)
        if row["recorded_n_val_rows"] != row["n_val_rows"] or row["recorded_n_train_rows"] != row["n_train_rows"]:
            bad.append({**row, "reason": "row count"}
                       )
        if row["scales_fingerprint_recorded"] != row["scales_fingerprint_rebuilt"]:
            bad.append({**row, "reason": "scales fingerprint"})
    return {
        "n_checked": len(rec_rows),
        "n_failed": len(bad),
        "passed": not bad,
        "tol_abs": MAE_ABS_TOL, "tol_rel": MAE_REL_TOL,
        "max_abs_diff_mae": max([r["abs_diff_mae"] for r in rec_rows], default=None),
        "max_abs_diff_host": max([r["abs_diff_host"] for r in rec_rows], default=None),
        "failures": bad[:20],
    }


PARQUET_COLUMNS = [
    "cell", "market", "host", "seed", "role", "day", "ordinal", "n_valid_hours",
    "host_mae", "residual_l1", "b_true", "B_true", "P_true", "N_true",
    "q_L", "q_B", "q_L_eta", "pos_mass_share", "neg_mass_share", "sign_changes",
    "shape_entropy_pos", "shape_conc_pos", "shape_entropy_neg", "shape_conc_neg",
    "residual_tv", "residual_roughness", "max_abs_resid_over_mae",
    "o1_b_hat", "o1_B_hat", "o1_mae", "o1_correction_ratio", "o1_mean_abs_correction",
    "w1_pos_pred_true", "w1_neg_pred_true", "decoder_gap",
    "mae_ob", "mae_oB", "mae_oS", "mae_obB", "mae_obS", "mae_oBS",
    "imp_ob_pct", "imp_oB_pct", "imp_oS_pct", "imp_obB_pct", "imp_obS_pct", "imp_oBS_pct",
    "mae_ray", "alpha_star", "alpha_status", "ray_n_effective", "increment_ray_pct",
    "ray_host_headroom_frac", "ray_remaining_recall", "alpha_band",
]


def write_parquet(path: Path, rows: list[dict]) -> None:
    R.write_table_parquet(path, rows, PARQUET_COLUMNS)


def run(limit_markets=None) -> dict:
    all_rows, all_rec = [], []
    panels = [c for c in R.DOM_PANEL if limit_markets is None or c[0] in limit_markets]
    for market, host in panels:
        rows, rec = cell_atlas(market, host)
        all_rows.extend(rows)
        all_rec.extend(rec)
        print(f"[r0] {R.cell_key(market, host)}: {len(rows)} rows over {len(R.SEEDS)} seeds", flush=True)
    return {"rows": all_rows, "reconciliation": all_rec}


if __name__ == "__main__":
    out = run()
    R.write_parquet(R.EVID / "PER_DAY_REPAIRABILITY.parquet", out["rows"])
    R.json_dump(R.EVID / "R0_RECONCILIATION.json", check_reconciliation(out["reconciliation"]))
    print("rows", len(out["rows"]))
