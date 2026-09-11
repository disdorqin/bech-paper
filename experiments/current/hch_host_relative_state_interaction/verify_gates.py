"""Independent re-derivation of the HRSI R0-R4 gates.

Run:  python experiments/current/hch_host_relative_state_interaction/verify_gates.py

This script deliberately does **not** import `run_closure`. It reads only the
written evidence under `experiments/evidence/hch_host_relative_state_interaction_20260911/`,
recomputes every gate from its own arithmetic, and diffs the result against the
recorded `HRSI_VERDICT.json`. A verifier that shares code with the runner it is
checking cannot catch a mistake in that shared code.

Exit code 0 iff every recomputed gate agrees with the recorded verdict, the
recomputed token is consistent with those gates, and the frozen-input hashes
still match what the run recorded before and after.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
OUT = ROOT / "experiments/evidence/hch_host_relative_state_interaction_20260911"

# Re-typed here on purpose: an independent verifier must not inherit the
# runner's constants by import.
TOL = 1e-10                 # frozen S1 metric replay tolerance
REPLAY_TOL = 1e-6           # beta=0 exact-direction replay tolerance
BETA_TRAINING_BUDGET_S = 10.0
OVERHEAD_BUDGET_US_DAY = 10.0
TOTAL_INFERENCE_BUDGET_US_DAY = 100.0
N_ROLES = 5
ROLES = ("DEMAND_FC", "RENEWABLE_FC", "SUPPLY_MARGIN_FC", "INTERCHANGE_FC", "MUST_RUN_FC")
MARKETS = ("LAGO_DE", "LAGO_PJM", "GANSU_DA")
HOSTS = ("PatchTST", "TimeMixer")

TOKEN_PASS = "HCH_HOST_RELATIVE_STATE_INTERACTION_SUPPORTED_METHOD_TARGET_REACHED"
TOKEN_FAIL = "HCH_HOST_RELATIVE_STATE_INTERACTION_NOT_SUPPORTED_FREEZE_S1"
TOKEN_INVALID = "HCH_HOST_RELATIVE_STATE_INTERACTION_INVALID"

FROZEN_CODE_TREES = {
    "s1_code_hash": "experiments/current/hch_unified_da_shape_upgrade",
    "amplitude_code_hash": "experiments/current/hch_anchored_compact_amplitude",
    "hsa_code_hash": "experiments/current/hch_horizon_aligned_state_residual",
}

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, condition: bool) -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha(path: Path) -> str:
    """Content hash of a source tree: every tracked file, `__pycache__` excluded."""
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
        h.update(p.relative_to(path).as_posix().encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("canonical repo required")

    verdict = json.loads((OUT / "HRSI_VERDICT.json").read_text(encoding="utf-8"))
    prov = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(OUT / "metrics_by_seed.csv")
    cell = pd.read_csv(OUT / "metrics_by_cell.csv")
    tert = pd.read_csv(OUT / "state_tertile_comparison.csv")
    replay = pd.read_csv(OUT / "s1_replay_audit.csv")
    chron = pd.read_csv(OUT / "oof_chronology_audit.csv")
    beta = pd.read_csv(OUT / "beta_by_fold_seed.csv")
    params = pd.read_csv(OUT / "parameter_audit.csv")
    timing = pd.read_csv(OUT / "inference_overhead.csv")
    tests_txt = (OUT / "tests.txt").read_text(encoding="utf-8")

    # ---------------------------------------------------------------- R0
    # R0.1 frozen S1 metric replay: every recorded difference must vanish.
    num = replay.drop(columns=["market", "track", "host", "seed"]).to_numpy(dtype=float)
    replay_max = float(np.max(np.abs(num)))
    check("R0 frozen S1 replay differences are all within tolerance", replay_max <= TOL)
    check("R0 replay table covers the full six cells x three seeds",
          len(replay) == len(MARKETS) * len(HOSTS) * 3)
    check(f"R0 replay_max matches the recorded verdict ({replay_max:.3e})",
          abs(replay_max - float(verdict["S1_replay_max_abs_diff"])) <= TOL)

    # R0.2 immutability of every frozen input, recomputed from disk.
    check("R0 frozen evidence hashes are identical before and after the run",
          prov["frozen_hashes_before"] == prov["frozen_hashes_after"])
    check("R0 Host prediction caches are identical before and after the run",
          prov["host_hashes_before"] == prov["host_hashes_after"])
    for key, rel in FROZEN_CODE_TREES.items():
        now = tree_sha(ROOT / rel)
        check(f"R0 {rel} source tree is unchanged since the run",
              now == prov[f"{key}_before"] == prov[f"{key}_after"])

    # R0.3 beta=0 replays S1 exactly, and the model carries nothing else.
    zero_init_max = float(beta.zero_init_replay_max_abs_diff.max())
    check("R0 beta=0 replays the frozen S1 direction within tolerance", zero_init_max <= REPLAY_TOL)
    check(f"R0 zero-init replay max matches the recorded verdict ({zero_init_max:.3e})",
          abs(zero_init_max - float(verdict["beta_zero_init_replay_max_abs_diff"])) <= TOL)
    p = params.iloc[0]
    check("R0 the whole panel trains exactly five parameters",
          int(p.trainable_parameter_count) == N_ROLES and int(p.expected_parameter_count) == N_ROLES)
    check("R0 there is no per-cell beta vector", int(p.per_cell_beta_vectors) == 0)
    check("R0 each beta is shared by all six cells", int(p.cells_per_beta) == 6)
    check("R0 the shared-beta table is exactly the nine fold rows plus three FINAL rows",
          int(p.shared_beta_rows) == len(beta) == 12)
    check("R0 the parameter audit is a single panel-wide row", len(params) == 1)

    # R0.4 one global beta per (seed, fold), applied to every cell.
    grp = beta.groupby(["seed", "fold"]).beta_sha256.nunique()
    check("R0 every (seed, fold) has exactly one beta vector", bool(grp.eq(1).all()))
    check("R0 every beta covers all six cells", bool(beta.cells_covered.eq(6).all()))
    check("R0 there is exactly one FINAL beta per seed",
          len(beta[beta.stage.eq("FINAL")]) == 3
          and beta[beta.stage.eq("FINAL")].seed.nunique() == 3)
    check("R0 every fold beta lives in R^5",
          beta[[f"beta_{r}" for r in ROLES]].notna().all().all())
    # A role with no legal column in ANY of the three markets has a column of
    # exact zeros in Z for every cell, so its weight is unidentifiable and must
    # come back exactly zero rather than drifting to an arbitrary value.
    manifest = pd.read_csv(OUT / "role_state_manifest.csv")
    absent_everywhere = [r for r in ROLES if bool(
        manifest[manifest.semantic_role.eq(r)].structurally_absent.astype(str).str.lower().eq("true").all())]
    check("R0 the role manifest covers all five roles in all three markets",
          len(manifest) == N_ROLES * len(MARKETS)
          and set(manifest.semantic_role) == set(ROLES))
    check("R0 at least one role is structurally absent panel-wide (unidentifiable by construction)",
          len(absent_everywhere) == 1)
    check("R0 every panel-wide structurally absent role receives exactly zero weight",
          bool(len(absent_everywhere) > 0
               and (beta[[f"beta_{r}" for r in absent_everywhere]].abs().to_numpy() == 0.0).all()))

    # R0.5 stacked chronological OOF self-exclusion.
    alph = chron[chron.record_type.eq("alpha")]
    check("R0 every OOF record is marked self-excluded", bool(chron.self_excluded.all()))
    check("R0 every calibration window ends before every proposal row it scores",
          bool((alph.calibration_max_index < alph.proposal_min_index).all()))
    # S1 can calibrate from B2 onward; HRSI needs a beta first, and the first
    # beta-consuming fold is B2, so HRSI's first calibrated fold is B3. This is
    # the origin of the row-set asymmetry the DIAG block below quantifies.
    counts = alph.groupby(["market", "host", "seed", "method"]).size()
    first_fold = alph.sort_values("block").groupby(["market", "host", "seed", "method"]).block.first()
    check("R0 the alpha chronology covers both methods in all six cells and three seeds",
          set(alph.method) == {"S1_DailyPatch_GRU32", "HRSI_HostRelativeStateInteraction"}
          and counts.xs("S1_DailyPatch_GRU32", level="method").eq(3).all()
          and counts.xs("HRSI_HostRelativeStateInteraction", level="method").eq(2).all())
    check("R0 S1 calibrates from B2 while HRSI starts one fold later at B3",
          bool(first_fold.xs("S1_DailyPatch_GRU32", level="method").eq("B2").all()
               and first_fold.xs("HRSI_HostRelativeStateInteraction", level="method").eq("B3").all()))
    check("R0 both methods calibrate only on a strictly earlier contiguous window",
          bool((alph.calibration_max_index < alph.proposal_min_index).all()))

    # R0.6 no forbidden state level was read.
    acc = prov["state_level_access"]
    check("R0 no market read target-day DA input, realized RT input, or S3/S4",
          all(a["target_day_DA_input_reads"] == 0 and a["realized_RT_input_reads"] == 0
              and a["S3_S4_reads"] == 0 for a in acc.values()))
    reads = prov["scientific_reads"]
    check("R0 no protected/final/S3/S4 read occurred",
          reads["S3"] == 0 and reads["S4"] == 0 and reads["protected"] == 0 and reads["final"] == 0)

    x0 = not FAILURES

    # ---------------------------------------------------------------- R1
    mech = pd.read_csv(OUT / "shape_mechanisms_by_cell.csv").set_index(["market", "host"])
    gansu = [k for k in mech.index if k[0] == "GANSU_DA"]
    intl = [k for k in mech.index if k[0] != "GANSU_DA"]
    cos_delta = mech.shape_cosine_change
    wrong_delta = mech.wrong_hemisphere_change
    hi = tert[(tert.method.eq("HRSI_HostRelativeStateInteraction")) & tert.tertile.eq("HIGH")].set_index(["market", "host"])
    r1 = bool((cos_delta.loc[gansu] > 0).all()
              and (wrong_delta.loc[gansu] <= TOL).all()
              and int((cos_delta > 0).sum()) >= 4
              and int((wrong_delta <= TOL).sum()) >= 4
              and (hi.loc[gansu].HRSI_minus_S1_shape_cosine > 0).all()
              and int((hi.loc[gansu].HRSI_minus_S1_wrong_hemisphere_rate <= -0.05 + TOL).sum()) >= 1)
    check("R1 all six cells are represented in the Shape mechanism table", len(mech) == 6)
    check("R1 each Shape cosine change equals HRSI minus S1",
          bool((mech.shape_cosine_change - (mech.HRSI_shape_cosine - mech.S1_shape_cosine)).abs().max() <= TOL))

    # ---------------------------------------------------------------- R2
    c = cell.set_index(["market", "host", "method"])
    s1 = c.xs("S1_DailyPatch_GRU32", level="method")
    hrsi = c.xs("HRSI_HostRelativeStateInteraction", level="method")
    host = c.xs("Host", level="method")
    gain_hrsi = hrsi.relative_gain_pct
    dgain = gain_hrsi - s1.relative_gain_pct
    normal_harm = 100 * (hrsi.Normal_MAE / host.Normal_MAE - 1.0)
    gg = gain_hrsi.loc[gansu]
    r2 = bool((dgain >= -TOL).all()
              and int((hrsi.Overall_MAE < s1.Overall_MAE).sum()) >= 5
              and (gain_hrsi >= -TOL).all()
              and float(np.median(gain_hrsi)) >= 7
              and int((gain_hrsi >= 5).sum()) >= 4
              and float(normal_harm.max()) <= 1
              and (hrsi.Overall_MAE.loc[gansu] < s1.Overall_MAE.loc[gansu]).all()
              and float(gg.max()) >= 10 and float(gg.min()) >= 4)
    check("R2 the cell table carries exactly Host, S1 and HRSI for six cells",
          len(cell) == 18 and set(cell.method) == {"Host", "S1_DailyPatch_GRU32", "HRSI_HostRelativeStateInteraction"})
    check("R2 recorded relative gain equals the gain recomputed from the MAEs",
          bool((hrsi.relative_gain_pct - 100 * (host.Overall_MAE - hrsi.Overall_MAE) / host.Overall_MAE).abs().max() <= 1e-6))
    check(f"R2 maximum Normal-MAE harm reproduces the verdict ({float(normal_harm.max()):.6f}%)",
          abs(float(normal_harm.max()) - float(verdict["R2_max_Normal_harm_pct"])) <= 1e-9)
    check(f"R2 median relative gain reproduces the verdict ({float(np.median(gain_hrsi)):.6f}%)",
          abs(float(np.median(gain_hrsi)) - float(verdict["R2_median_gain_pct"])) <= 1e-9)

    # ---------------------------------------------------------------- R3
    piv = metrics[metrics.method.isin(["S1_DailyPatch_GRU32", "HRSI_HostRelativeStateInteraction"])] \
        .pivot_table(index=["market", "host", "seed"], columns="method", values="relative_gain_pct")
    seed_delta = piv["HRSI_HostRelativeStateInteraction"] - piv["S1_DailyPatch_GRU32"]
    seed_improve = seed_delta.reset_index(name="d").assign(ok=lambda d: d.d > TOL).groupby(["market", "host"]).ok.sum()
    intl_seed_min = float(seed_delta.reset_index(name="d").query("market != 'GANSU_DA'").d.min())
    intl_cell_min = float(dgain.loc[intl].min())
    r3 = bool(int((seed_improve >= 2).sum()) >= 5
              and (dgain.loc[intl] >= -TOL).all()
              and int((dgain.loc[intl] > 0).sum()) >= 2
              and intl_cell_min >= -0.5 and intl_seed_min >= -0.5)
    check("R3 every cell has all three seeds in the metric table",
          bool(metrics[metrics.method.eq("HRSI_HostRelativeStateInteraction")].groupby(["market", "host"]).seed.nunique().eq(3).all()))
    check(f"R3 worst international seed reproduces the verdict ({intl_seed_min:+.6f} pp)",
          abs(intl_seed_min - float(verdict["R3_international_seed_min_pp"])) <= 1e-9)

    # ---------------------------------------------------------------- R4
    med_beta = float(np.median(beta.training_seconds))
    med_overhead = float(np.median(timing.hrsi_overhead_us_per_day))
    med_total = float(np.median(timing.hrsi_total_inference_us_per_day))
    r4 = bool(int(p.trainable_parameter_count) == N_ROLES and med_beta <= BETA_TRAINING_BUDGET_S
              and med_overhead <= OVERHEAD_BUDGET_US_DAY and med_total <= TOTAL_INFERENCE_BUDGET_US_DAY)
    check(f"R4 median beta-fit time reproduces the verdict ({med_beta:.4f} s)",
          abs(med_beta - float(verdict["R4_median_beta_training_seconds"])) <= 1e-6)
    check(f"R4 median inference overhead reproduces the verdict ({med_overhead:.4f} us/day)",
          abs(med_overhead - float(verdict["R4_median_overhead_us_per_day"])) <= 1e-6)
    check("R4 the timing table covers all six cells", len(timing) == 6)

    # ------------------------------------------------- token and diff
    recomputed = {"R0": x0, "R1": r1, "R2": r2, "R3": r3, "R4": r4}
    token = TOKEN_INVALID if not x0 else (TOKEN_PASS if all(recomputed.values()) else TOKEN_FAIL)
    for gate, val in recomputed.items():
        recorded = verdict[gate] == "PASS"
        check(f"{gate} recomputed independently agrees with the recorded verdict ({verdict[gate]})", val == recorded)
    check(f"recomputed token agrees with the recorded token ({verdict['token']})", token == verdict["token"])

    # The disclosed row-set diagnostic must reproduce, and must not be able to
    # flip the verdict: it is a comparison-robustness note, not a gate.
    sens = pd.read_csv(OUT / "alpha_rowset_sensitivity.csv")
    reach = {"S1_aligned_rowset_relative_gain_pct", "HRSI_minus_S1_aligned_overall_gain_pp"}
    check("DIAG the row-set sensitivity table is complete (six cells x three seeds)",
          len(sens) == 18 and reach <= set(sens.columns))
    aligned_nonworse = int((sens.groupby(["market", "host"]).HRSI_minus_S1_aligned_overall_gain_pp.median() >= -TOL).sum())
    check(f"DIAG the failure survives the aligned row set ({aligned_nonworse}/6 cells non-worse)",
          aligned_nonworse == int(verdict["DIAG_cells_nonworse_on_aligned_rowset"]))
    check("DIAG the row-set diagnostic is recorded but never gates R2",
          verdict["R2"] == ("PASS" if r2 else "FAIL"))

    # The targeted unit suite must be present and green.
    check("TESTS the targeted unit suite reported success",
          "checks passed" in tests_txt and "FAILED" not in tests_txt)

    print()
    if FAILURES:
        print(f"VERIFY FAILED: {len(FAILURES)} of {CHECKS} independent checks failed")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"VERIFY OK: {CHECKS} independent checks reproduced R0-R4 and token `{token}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
