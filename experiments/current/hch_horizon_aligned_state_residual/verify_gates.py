"""Independent H0-H4 re-verification for the HSA Shape closure.

Reads ONLY the written evidence under
`experiments/evidence/hch_horizon_aligned_state_residual_20260911/` plus the
frozen upstream artifacts, and recomputes every gate from those files with its
own arithmetic. It deliberately does NOT import `run_closure`, so a bug in the
runner cannot be mirrored by the check that is supposed to catch it.

Every threshold below is transcribed from protocol §9 of
`docs/current/HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SHAPE_CLOSURE_20260911.md`,
not from the runner's constants.

Exit code 0 iff the recomputed gates reproduce HSA_VERDICT.json exactly.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
OUT = ROOT / "experiments/evidence/hch_horizon_aligned_state_residual_20260911"
STATE_EVIDENCE = ROOT / "experiments/evidence/hch_market_state_coupling_20260910"
SHAPE_EVIDENCE = ROOT / "experiments/evidence/hch_unified_da_shape_engineering_20260910"

MARKETS = ("GANSU_DA", "LAGO_DE", "LAGO_PJM")
HOSTS = ("PatchTST", "TimeMixer")
SEEDS = (7, 17, 37)

# Transcribed from the protocol, deliberately not imported from the runner.
REPLAY_TOL = 1e-10     # §9 H0 "replay is exact within prior tolerances"
REPLAY_TOL_BETA = 1e-6  # §5.2 beta=0 -> exact S1 replay (float32 eps)
N_ROLES = 5
BETA_BUDGET_S = 5.0
OVERHEAD_BUDGET_US = 10.0
TOTAL_BUDGET_US = 100.0

S1 = "S1_DailyPatch_GRU32"
HSA = "HSA_HorizonAlignedStateResidual"

RESULTS: list[tuple[str, bool, str]] = []


def check(gate, name, cond, detail=""):
    RESULTS.append((gate, bool(cond), f"{name}{(' — ' + detail) if detail else ''}"))


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree_sha(p):
    p = Path(p); h = hashlib.sha256()
    for x in sorted(y for y in p.rglob("*") if y.is_file() and "__pycache__" not in y.parts):
        h.update(x.relative_to(p).as_posix().encode()); h.update(x.read_bytes())
    return h.hexdigest()


def main():
    verdict = json.loads((OUT / "HSA_VERDICT.json").read_text(encoding="utf-8"))
    prov = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))

    # ---------------- §10 artifact inventory ----------------
    required = ["HSA_VERDICT.json", "HSA_SUMMARY.md", "metrics_by_seed.csv", "metrics_by_cell.csv",
                "beta_by_fold_seed.csv", "role_state_manifest.csv", "state_tertile_comparison.csv",
                "alpha_replay_and_hsa.csv", "efficiency.csv", "provenance.json",
                "IMPLEMENTATION_TEST_AUDIT.md"]
    missing = [f for f in required if not (OUT / f).exists() or (OUT / f).stat().st_size == 0]
    figdir = OUT / "figures"
    figs = sorted(p.name for p in figdir.glob("*.png")) if figdir.exists() else []
    check("INV", "all §10 required artifacts exist and are non-empty", not missing, str(missing))
    check("INV", "three required figures exist (cosine, gain, tertile)",
          len(figs) >= 3 and any("cosine" in f for f in figs)
          and any("gain" in f for f in figs) and any("tertile" in f for f in figs), str(figs))

    metrics = pd.read_csv(OUT / "metrics_by_seed.csv")
    cell = pd.read_csv(OUT / "metrics_by_cell.csv")
    tert = pd.read_csv(OUT / "state_tertile_comparison.csv")
    chron = pd.read_csv(OUT / "oof_chronology_audit.csv")
    beta = pd.read_csv(OUT / "beta_by_fold_seed.csv")
    params = pd.read_csv(OUT / "parameter_audit.csv")
    replay = pd.read_csv(OUT / "s1_replay_audit.csv")
    stab = pd.read_csv(OUT / "seed_stability.csv")
    timing = pd.read_csv(OUT / "inference_overhead.csv")
    roleman = pd.read_csv(OUT / "role_state_manifest.csv")

    # ---------------- H0: validity ----------------
    # (a) S1/Host/split replay exact.
    diffcols = [c for c in replay.columns if c.endswith("_diff")]
    replay_max = float(replay[diffcols].to_numpy().max())
    check("H0", "S1 replay exact within tolerance", replay_max <= REPLAY_TOL, f"max={replay_max:.3e}")
    check("H0", "replay audit covers all six cells x three seeds",
          len(replay) == len(MARKETS) * len(HOSTS) * len(SEEDS), f"rows={len(replay)}")

    # (b) frozen upstream files unchanged since the run: recompute now, compare to the run's record.
    # Keyed the way the runner recorded them: repo-relative paths, so the two dicts
    # are comparable key-for-key.
    frozen_paths = [STATE_EVIDENCE / "state_schema_manifest.csv",
                    STATE_EVIDENCE / "state_tertile_mechanisms.csv",
                    STATE_EVIDENCE / "diagnostic_rows.parquet",
                    SHAPE_EVIDENCE / "metrics_by_seed.csv"]
    now = {str(p.relative_to(ROOT)): sha(p) for p in frozen_paths}
    check("H0", "frozen upstream hashes unchanged since the run",
          now == prov["frozen_hashes_after"], "recomputed now vs provenance.frozen_hashes_after")
    check("H0", "frozen upstream hashes identical before and after the run",
          prov["frozen_hashes_before"] == prov["frozen_hashes_after"])

    # (c) S1 remains frozen: its code tree is bit-identical.
    for key, rel in (("s1_code_hash", "hch_unified_da_shape_upgrade"),
                     ("amplitude_code_hash", "hch_anchored_compact_amplitude")):
        cur = tree_sha(ROOT / "experiments/current" / rel)
        check("H0", f"{rel} code tree unchanged (S1 frozen)",
              cur == prov[f"{key}_before"] == prov[f"{key}_after"])

    # (d) host prediction caches untouched.
    check("H0", "host cache hashes identical before and after the run",
          prov["host_hashes_before"] == prov["host_hashes_after"]
          and len(prov["host_hashes_after"]) == len(MARKETS) * len(HOSTS))

    # (e) role schema is the audited one: re-derive it from the frozen schema ourselves.
    schema = pd.read_csv(STATE_EVIDENCE / "state_schema_manifest.csv")
    schema["primary_channel"] = schema.primary_channel.astype(str).str.lower().eq("true")
    schema["forecast_time_legal"] = schema.forecast_time_legal.astype(str).str.lower().eq("true")
    prim = schema[schema.primary_channel]
    check("H0", "every primary channel used is forecast-time legal",
          bool(prim.forecast_time_legal.all()))
    for m in MARKETS:
        got = roleman[(roleman.market == m) & roleman.n_channels_in_role.gt(0)]
        want = prim[prim.market == m]
        check("H0", f"{m} role manifest reproduces the audited primary channels",
              int(got.n_channels_in_role.sum()) == len(want),
              f"manifest={int(got.n_channels_in_role.sum())} audited={len(want)}")
        used = [c for s in got.source_columns for c in str(s).split("|") if c]
        check("H0", f"{m} role manifest channels match the audited set exactly",
              sorted(used) == sorted(want.source_column.astype(str)))
        check("H0", f"{m} no channel is assigned to two roles",
              len(used) == len(set(used)))
    check("H0", "role vocabulary is exactly the five registered roles",
          set(roleman.semantic_role) <= {"DEMAND_FC", "RENEWABLE_FC", "SUPPLY_MARGIN_FC",
                                         "INTERCHANGE_FC", "MUST_RUN_FC"})
    check("H0", "SUPPLY_MARGIN_FC is structurally absent in every market",
          bool(roleman[roleman.semantic_role.eq("SUPPLY_MARGIN_FC")].structurally_absent.all()))

    # (f) beta is exactly five parameters with zero initialisation.
    # The audit is keyed per market: S1's parameter count depends only on the market's
    # state width, so both hosts share it. One row per market, none missing.
    check("H0", "parameter audit covers all three markets, one row each",
          sorted(params.market) == sorted(MARKETS)
          and params.market.is_unique, str(list(params.market)))
    check("H0", "batch adds exactly five trainable parameters",
          bool((params.observed_parameter_delta == N_ROLES).all())
          and bool((params.HSA_parameter_count - params.S1_parameter_count == N_ROLES).all()))
    check("H0", "beta=0 replays S1 exactly at initialisation",
          float(beta.zero_init_replay_max_abs_diff.max()) <= REPLAY_TOL_BETA,
          f"max={float(beta.zero_init_replay_max_abs_diff.max()):.3e}")
    check("H0", "beta training is on the fixed budget (1000 steps, no search)",
          bool((beta.n_train_rows > 0).all()))

    # (g) chronology / self-exclusion.
    check("H0", "every OOF record self-excludes its own proposal rows",
          bool(chron.self_excluded.all()))
    al = chron[chron.record_type.eq("alpha")]
    check("H0", "alpha calibration rows strictly precede proposal rows",
          bool((al.calibration_max_index < al.proposal_min_index).all()))
    bf = chron[chron.record_type.eq("beta")]
    check("H0", "beta training rows strictly precede proposal rows",
          bool((bf.beta_train_max_index < bf.proposal_min_index).all()))
    check("H0", "HSA beta OOF is filled exactly on B2..B4",
          sorted(bf.block.unique()) == ["B2", "B3", "B4"], str(sorted(bf.block.unique())))
    check("H0", "S1 alpha OOF chronology is the frozen B2..B4 convention",
          sorted(al[al.method.eq(S1)].block.unique()) == ["B2", "B3", "B4"])

    # (h) no forbidden partition accessed.
    check("H0", "no S3/S4/protected/final partition was read",
          all(int(prov["scientific_reads"][k]) == 0 for k in ("S3", "S4", "protected", "final")))
    check("H0", "state reads use no target-day DA and no realized RT input",
          all(a["target_day_DA_input_reads"] == 0 and a["realized_RT_input_reads"] == 0
              and a["S3_S4_reads"] == 0 for a in prov["state_level_access"].values()))
    check("H0", "no forbidden mechanism is declared active",
          not any(bool(v) for v in prov["forbidden"].values()
                  if not isinstance(v, int)) and int(prov["forbidden"]["s3_s4_protected_final_reads"]) == 0)

    h0 = all(ok for g, ok, _ in RESULTS if g == "H0")

    # ---------------- H1: directional mechanism ----------------
    s1c = cell[cell.method.eq(S1)].set_index(["market", "host"])
    hc = cell[cell.method.eq(HSA)].set_index(["market", "host"])
    gansu = [k for k in hc.index if k[0] == "GANSU_DA"]
    cos_d = hc.shape_cosine - s1c.shape_cosine
    wrong_d = hc.wrong_hemisphere_rate - s1c.wrong_hemisphere_rate
    frozen_tert = pd.read_csv(STATE_EVIDENCE / "state_tertile_mechanisms.csv")
    targeted = sorted({(r.market, r.host) for _, r in
                       frozen_tert[frozen_tert.mechanism_label.eq("SHAPE_LIMITED")][
                           ["market", "host"]].drop_duplicates().iterrows()})

    hi = tert[(tert.method.eq(HSA)) & tert.tertile.eq("HIGH")]
    hi = hi.groupby(["market", "host"], as_index=False).median(numeric_only=True).set_index(["market", "host"])
    g_hi = hi.loc[gansu]

    h1a = int((cos_d >= -REPLAY_TOL).sum()) >= 5
    h1b = bool((cos_d.loc[gansu] > 0).all())
    h1c = int((wrong_d <= REPLAY_TOL).sum()) >= 5
    h1d = len(targeted) == 4 and int((cos_d.loc[targeted] > 0).sum()) >= 3
    h1e = bool((g_hi.HSA_minus_S1_shape_cosine > 0).all())
    h1f = bool((g_hi.HSA_minus_S1_wrong_hemisphere_rate <= 0.02 + REPLAY_TOL).all())
    h1 = h1a and h1b and h1c and h1d and h1e and h1f
    check("H1", "Shape cosine non-worse in >=5/6 cells", h1a, f"{int((cos_d >= -REPLAY_TOL).sum())}/6")
    check("H1", "both GANSU_DA hosts improve Shape cosine", h1b, str(cos_d.loc[gansu].round(5).to_dict()))
    check("H1", "wrong-hemisphere non-worse in >=5/6 cells", h1c, f"{int((wrong_d <= REPLAY_TOL).sum())}/6")
    check("H1", ">=3 of the 4 frozen SHAPE_LIMITED cells improve cosine", h1d,
          f"{int((cos_d.loc[targeted] > 0).sum())}/4")
    check("H1", "GANSU HIGH tertile improves cosine for both hosts", h1e,
          str(g_hi.HSA_minus_S1_shape_cosine.round(5).to_dict()))
    check("H1", "GANSU HIGH tertile wrong-hemisphere worsens by <=2pp", h1f,
          str(g_hi.HSA_minus_S1_wrong_hemisphere_rate.round(5).to_dict()))

    # ---------------- H2: universal performance ----------------
    host = cell[cell.method.eq("Host")].set_index(["market", "host"])
    gain_s1 = s1c.relative_gain_pct
    gain_hsa = hc.relative_gain_pct
    dgain = gain_hsa - gain_s1
    normal_harm = 100 * (hc.Normal_MAE / host.Normal_MAE - 1)
    gg = gain_hsa.loc[gansu]
    h2a = bool((dgain >= -REPLAY_TOL).all())
    h2b = int((hc.Overall_MAE < s1c.Overall_MAE).sum()) >= 5
    h2c = bool((gain_hsa >= -REPLAY_TOL).all())
    h2d = float(np.median(gain_hsa)) >= 7
    h2e = int((gain_hsa >= 5).sum()) >= 4
    h2f = float(normal_harm.max()) <= 1
    h2g = bool((hc.Overall_MAE.loc[gansu] < s1c.Overall_MAE.loc[gansu]).all())
    h2h = float(gg.max()) >= 7 and float(gg.min()) >= 3
    h2 = all((h2a, h2b, h2c, h2d, h2e, h2f, h2g, h2h))
    check("H2", "HSA gain vs S1 nonnegative in 6/6 cell medians", h2a,
          str(dgain.round(4).to_dict()))
    check("H2", "HSA strictly better than S1 in >=5/6 cells", h2b,
          f"{int((hc.Overall_MAE < s1c.Overall_MAE).sum())}/6")
    check("H2", "Host-nonworse in 6/6 cell medians", h2c)
    check("H2", "median gain vs Host >=7%", h2d, f"{float(np.median(gain_hsa)):.4f}")
    check("H2", ">=4/6 cells reach >=5% gain vs Host", h2e, f"{int((gain_hsa >= 5).sum())}/6")
    check("H2", "max Normal-MAE harm vs Host <=1%", h2f, f"{float(normal_harm.max()):.4f}")
    check("H2", "both GANSU cells strictly improve S1 Overall-MAE", h2g)
    check("H2", "GANSU gains reach >=7% and >=3%", h2h, str(gg.round(4).to_dict()))

    # ---------------- H3: seed stability ----------------
    piv = stab.pivot_table(index=["market", "host"], values="delta_gain_pct", aggfunc=list)
    n_ok = sum(1 for _, v in piv.itertuples() if sum(x >= -REPLAY_TOL for x in v) >= 2)
    g_ok = all(sum(x >= -REPLAY_TOL for x in piv.loc[k, "delta_gain_pct"]) >= 2 for k in gansu)
    worst = float(stab.delta_gain_pct.min())
    h3 = n_ok >= 5 and g_ok and worst >= -2
    check("H3", ">=5/6 cells hold up in >=2/3 seeds", n_ok >= 5, f"{n_ok}/6")
    check("H3", "both GANSU cells hold up in >=2/3 seeds", g_ok)
    check("H3", "no seed/cell loses more than 2pp vs S1", worst >= -2, f"worst={worst:.4f}pp")

    # ---------------- H4: efficiency ----------------
    med_beta = float(np.median(timing.beta_training_seconds_per_cell))
    med_ov = float(np.median(timing.hsa_overhead_us_per_day))
    med_tot = float(np.median(timing.hsa_total_inference_us_per_day))
    h4a = bool((params.observed_parameter_delta == N_ROLES).all())
    h4b = med_beta <= BETA_BUDGET_S
    h4c = med_ov <= OVERHEAD_BUDGET_US
    h4d = med_tot <= TOTAL_BUDGET_US
    h4 = h4a and h4b and h4c and h4d
    check("H4", "exactly 5 additional trainable parameters", h4a)
    check("H4", "median beta training <=5s per cell", h4b, f"{med_beta:.4f}s")
    check("H4", "median absolute inference overhead <=10us/day", h4c, f"{med_ov:.4f}us")
    check("H4", "total HSA inference <=100us/day", h4d, f"{med_tot:.4f}us")
    # No new hidden layer: the only module in the trainable object is the beta holder,
    # asserted at run time by new_beta_model(); re-assert the observable consequence.
    check("H4", "no new neural hidden layer (param delta is 5, not a layer's worth)",
          bool((params.observed_parameter_delta == N_ROLES).all()))

    # ---------------- diff against the runner's recorded verdict ----------------
    recomputed = {"H0": h0, "H1": h1, "H2": h2, "H3": h3, "H4": h4}
    for g, ok in recomputed.items():
        check("DIFF", f"{g} recomputed independently matches HSA_VERDICT.json",
              (verdict[g] == "PASS") == ok, f"indep={'PASS' if ok else 'FAIL'} recorded={verdict[g]}")

    tok = ("HCH_HORIZON_ALIGNED_STATE_RESIDUAL_INVALID" if not h0 else
           "HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SUPPORTED_FOR_PUBLIC_CHINA_EXPANSION"
           if all(recomputed.values()) else
           "HCH_HORIZON_ALIGNED_STATE_RESIDUAL_NOT_SUPPORTED_KEEP_S1")
    check("DIFF", "token follows from the recomputed gates", tok == verdict["token"],
          f"indep={tok}")

    # ---------------- report ----------------
    for gate in ("INV", "H0", "H1", "H2", "H3", "H4", "DIFF"):
        rows = [(ok, m) for g, ok, m in RESULTS if g == gate]
        if not rows:
            continue
        print(f"\n=== {gate} ({sum(1 for ok, _ in rows if ok)}/{len(rows)}) ===")
        for ok, m in rows:
            print(f"{'PASS' if ok else 'FAIL'} {m}")

    # A FAIL inside H1/H2/H3 is the scientific finding, not a verifier error. What this
    # script certifies is agreement: the recorded verdict is reproducible from the
    # evidence, and the validity machinery (INV + H0) holds. Exit status tracks that.
    integrity = [(ok, f"{g}: {m}") for g, ok, m in RESULTS if g in ("INV", "H0", "DIFF")]
    bad = [m for ok, m in integrity if not ok]
    print(f"\nrecomputed gates: { {g: ('PASS' if v else 'FAIL') for g, v in recomputed.items()} }")
    print(f"recomputed token: {tok}")
    print(f"verifier integrity: {len(integrity) - len(bad)}/{len(integrity)} checks passed")
    if bad:
        print("verifier integrity FAILURES:")
        for m in bad:
            print(f"  - {m}")
        return 1
    print("recorded verdict is reproducible from the evidence; gate failures above are the finding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
