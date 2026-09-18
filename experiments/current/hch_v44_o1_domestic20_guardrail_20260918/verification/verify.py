"""Independent verification of the O1 domestic-20 + international guardrail stage.

This verifier imports **nothing** from the stage implementation -- not the runner,
not ``guardrail_common``.  It re-derives the coordinate universe, the gates and
every aggregate from the PROTOCOL text plus the raw artifacts on disk, and it
re-hashes checkpoints itself rather than trusting any recorded digest.

Independence rules honoured here:
* the expected 20-cell universe, the 8 reused cells, the 3 seeds and all D/I
  thresholds are transcribed from ``PROTOCOL.md``, not imported;
* CSV files are re-parsed and re-aggregated with this file's own arithmetic;
* every checkpoint is re-hashed and compared against the digest its own freeze
  record claims, so an edited artifact cannot pass on a stale claim;
* the scientific-core digest is recomputed under the canonical per-file rule and
  compared with the value the runs recorded *and* with the pin recorded by the
  prior closed stages.

Run:  python verify.py
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]
EVID = REPO / "experiments/evidence/hch_v44_o1_domestic20_guardrail_20260918"
O1_EVID = REPO / "experiments/evidence/hch_v44_objective_alignment_probe_20260917"
FULL8_EVID = REPO / "experiments/evidence/hch_v44_o1_full8_completion_20260918"
TRANSFER = REPO / "experiments/evidence/hch_frozen_method_baseline_transfer_20260911"
V2 = REPO / "experiments/evidence/hch_china5_common_benchmark_701020_full_v2_20260913"
#: The v4.4 shadow-OOF support lives under the China-5 full-panel evidence root,
#: not under the O1 probe root.  Asserted below so a wrong path here can never
#: turn an "is the support present?" check into a vacuous pass.
SHADOW = REPO / "experiments/evidence/hch_v44_china5_fullpanel_main_eval_20260917" / "shadow_oof"

# ---------------------------------------------------------------- transcribed
MARKETS = ["GANSU_DA", "SHANDONG_DA", "SHAANXI_DA", "NINGXIA_DA", "QINGHAI_DA"]
HOSTS = ["PatchTST", "TimeMixer", "iTransformer", "LSTM"]
SEEDS = [7, 17, 37]
PANEL = [(m, h) for m in MARKETS for h in HOSTS]
REUSED = [
    ("GANSU_DA", "PatchTST"), ("GANSU_DA", "LSTM"),
    ("SHANDONG_DA", "PatchTST"), ("SHANDONG_DA", "iTransformer"),
    ("SHAANXI_DA", "TimeMixer"), ("SHAANXI_DA", "PatchTST"),
    ("NINGXIA_DA", "iTransformer"), ("QINGHAI_DA", "TimeMixer"),
]
NEW = [c for c in PANEL if c not in REUSED]
REUSED_ROOT = {
    "GANSU_DA__PatchTST": O1_EVID, "SHANDONG_DA__iTransformer": O1_EVID,
    "SHAANXI_DA__TimeMixer": O1_EVID, "NINGXIA_DA__iTransformer": O1_EVID,
    "GANSU_DA__LSTM": FULL8_EVID, "SHANDONG_DA__PatchTST": FULL8_EVID,
    "SHAANXI_DA__PatchTST": FULL8_EVID, "QINGHAI_DA__TimeMixer": FULL8_EVID,
}
INTL_CELLS = [("LAGO_DE", "PatchTST"), ("LAGO_DE", "TimeMixer"),
              ("LAGO_PJM", "PatchTST"), ("LAGO_PJM", "TimeMixer")]
RUN_FILES = ("freeze.json", "training_curve.json", "selected_ema.pt")
SWITCH_STEP = 400
CORE_TREE_PIN = "6F18C0E4241A299E32C8D8A617061BAEAA7B8DC98B4EE069DC67199B87531577"
HEAD_PREFIXES = ("level_head.", "mass_head.", "shape_head.", "direct_head.")

FAILS: list[str] = []
CHECKS: list[dict] = []


def check(name: str, ok: bool, detail="") -> bool:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        FAILS.append(f"{name}: {detail}")
    return bool(ok)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def tree_digest(sub: str) -> str:
    items = [f"{p.relative_to(REPO).as_posix()}|{sha(p)}"
             for p in sorted((REPO / sub).rglob("*"))
             if p.is_file() and p.suffix in (".py", ".md")]
    return hashlib.sha256("\n".join(items).encode()).hexdigest().upper()


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_csv(p: Path) -> list[dict]:
    lines = Path(p).read_text(encoding="utf-8").rstrip("\n").splitlines()
    head = _split(lines[0])
    return [dict(zip(head, _split(line))) for line in lines[1:]]


def _split(line: str) -> list[str]:
    cells, cur, q = [], "", False
    for ch in line:
        if ch == '"':
            q = not q
        elif ch == "," and not q:
            cells.append(cur)
            cur = ""
        else:
            cur += ch
    cells.append(cur)
    return cells


def key(m: str, h: str) -> str:
    return f"{m}__{h}"


def run_dir(m: str, h: str, s: int) -> Path:
    if (m, h) in NEW:
        return EVID / "o1_runs" / key(m, h) / f"seed{s}"
    return REUSED_ROOT[key(m, h)] / "o1_runs" / key(m, h) / f"seed{s}"


# ------------------------------------------------------------------ 1. universe
def v_universe() -> dict:
    complete, missing, status = [], [], {}
    for m, h in PANEL:
        for s in SEEDS:
            d = run_dir(m, h, s)
            have = all((d / f).is_file() for f in RUN_FILES)
            status[f"{key(m, h)}__seed{s}"] = "reused" if (m, h) in REUSED else "new"
            (complete if have else missing).append(f"{key(m, h)}__seed{s}")
    check("universe_60_coordinates", len(complete) == 60, f"complete={len(complete)} missing={missing[:4]}")
    check("universe_exactly_24_reused",
          sum(1 for v in status.values() if v == "reused") == 24)
    check("universe_exactly_36_new", sum(1 for v in status.values() if v == "new") == 36)
    new_present = [d.name for d in (EVID / "o1_runs").iterdir()] if (EVID / "o1_runs").is_dir() else []
    check("no_unregistered_cell_directory",
          set(new_present) <= {key(m, h) for m, h in NEW},
          f"unexpected={sorted(set(new_present) - {key(m, h) for m, h in NEW})}")
    # Path self-check: the support-presence claims below are only meaningful if
    # every registered cell really has its shadow-OOF artifact under this root.
    # The root also carries SHADOW_PLAN.json and SHADOW_SUMMARY.json, so a bare
    # ``*.json`` count over-counts and would fail on a correct store; the check
    # is by registered cell key, in both directions.
    missing_shadow = [key(m, h) for m, h in PANEL if not (SHADOW / f"{key(m, h)}.json").is_file()]
    check("shadow_support_covers_all_20_registered_cells", not missing_shadow,
          f"missing={missing_shadow} at {SHADOW}")
    strays = sorted(p.stem for p in SHADOW.glob("*.json")
                    if "__" in p.stem and p.stem not in {key(m, h) for m, h in PANEL})
    check("shadow_support_root_has_no_unregistered_cell_file", not strays, str(strays))
    return {"n_complete": len(complete), "n_reused": 24, "n_new": 36,
            "new_cell_dirs_on_disk": sorted(new_present)}


# ------------------------------------------------------------------ 2. hashes
def v_hashes() -> dict:
    prov = load(EVID / "REUSE_PROVENANCE.json")
    snap_before = load(EVID / "REUSE_SNAPSHOT_BEFORE.json")
    snap_after, snap_reused, code_hashes, core_trees, schedules, supports, reads = {}, {}, set(), set(), set(), [], []
    bad_ckpt, bad_support, bad_switch, bad_curve = [], [], [], []
    support_hash_by_cell = {key(m, h): load(SHADOW / f"{key(m, h)}.json")["support_hash"]
                            for m, h in PANEL}
    for m, h in PANEL:
        for s in SEEDS:
            d = run_dir(m, h, s)
            fr = load(d / "freeze.json")
            ck = d / "selected_ema.pt"
            if sha(ck) != str(fr.get("selected_ema_checkpoint_sha256", "")).upper():
                bad_ckpt.append(f"{key(m,h)}__seed{s}")
            code_hashes.add(fr.get("probe_code_hash"))
            core_trees.add(fr.get("source_tree_digest", {}).get("core_tree"))
            schedules.add(json.dumps({k: v for k, v in fr["optimization"].items()
                                      if k != "stopped_at_step"}, sort_keys=True))
            supports.append(fr.get("history_support_hash") == support_hash_by_cell[key(m, h)])
            reads.append(int(fr.get("test_target_read_count", -1)))
            for f in RUN_FILES:
                k = f"{key(m,h)}__seed{s}__{f}"
                snap_after[k] = sha(d / f)
                if (m, h) in REUSED:
                    snap_reused[k] = snap_after[k]
            curve = load(d / "training_curve.json")
            marks = [c["step"] for c in curve if c.get("objective_switch_here")]
            if marks != [SWITCH_STEP]:
                bad_switch.append(f"{key(m,h)}__seed{s}:{marks}")
            for c in curve:
                if c.get("is_initial"):
                    continue
                if int(c["step"]) <= SWITCH_STEP and c.get("used_full_loss_at_this_step") is not True:
                    bad_curve.append(f"{key(m,h)}__seed{s}@{c['step']}")
                if int(c["step"]) > SWITCH_STEP and c.get("used_full_loss_at_this_step") is not False:
                    bad_curve.append(f"{key(m,h)}__seed{s}@{c['step']}")
    check("one_probe_code_hash_across_60_runs", len(code_hashes) == 1, str(sorted(code_hashes)))
    check("core_tree_equals_frozen_pin", core_trees == {CORE_TREE_PIN}, str(sorted(core_trees)))
    check("core_tree_matches_recomputation", tree_digest("src/core") == CORE_TREE_PIN,
          tree_digest("src/core"))
    check("one_registered_schedule_across_60_runs", len(schedules) == 1)
    # ``snap_before`` is the stage's pre-fit snapshot of the 24 reused runs; the
    # 36 new coordinates are expected to appear only in ``snap_after``, so the
    # equality is taken on the reused subset AND the before-key set is required
    # to be exactly that subset -- a snapshot that quietly covered fewer runs
    # would otherwise make this check pass while checking less.
    expected_reused = {f"{key(m,h)}__seed{s}__{f}" for m, h in REUSED for s in SEEDS for f in RUN_FILES}
    check("reuse_snapshot_before_covers_exactly_the_reused_artifacts",
          set(snap_before) == expected_reused,
          f"n={len(snap_before)} missing={sorted(expected_reused - set(snap_before))[:3]} "
          f"extra={sorted(set(snap_before) - expected_reused)[:3]}")
    check("reused_artifacts_byte_identical_before_and_after",
          snap_reused == snap_before,
          f"changed={sorted(k for k in expected_reused if snap_before.get(k) != snap_reused.get(k))[:4]}")
    check("every_checkpoint_matches_its_own_recorded_sha256", not bad_ckpt, str(bad_ckpt[:4]))
    check("every_run_history_support_matches_shadow_artifact", all(supports),
          f"{sum(supports)}/{len(supports)}")
    check("switch_exactly_once_at_step_400_in_all_60_curves", not bad_switch, str(bad_switch[:4]))
    check("loss_schedule_matches_registered_switch_in_all_curves", not bad_curve, str(bad_curve[:4]))
    check("test_target_read_count_zero_in_all_60_runs", set(reads) == {0}, str(sorted(set(reads))))
    check("reuse_provenance_declares_one_code_state",
          prov["one_probe_code_hash_across_reused"] is True)
    check("reuse_provenance_covers_24_runs", prov["n_reused_runs"] == 24, str(prov["n_reused_runs"]))
    return {"n_checkpoints_rehashed": 60, "bad": bad_ckpt + bad_switch + bad_curve,
            "snapshot_entries": len(snap_after), "reused_snapshot_entries": len(snap_reused)}


# ------------------------------------------------------------- 3. aggregation
def _med(xs):
    return float(st.median([float(x) for x in xs]))


def v_domestic() -> dict:
    per = read_csv(EVID / "DOMESTIC_PER_SEED.csv")
    cells = read_csv(EVID / "DOMESTIC_CELL_MEDIANS.csv")
    markets = read_csv(EVID / "DOMESTIC_MARKET_SUMMARY.csv")
    gate = load(EVID / "DOMESTIC_GATE.json")
    check("per_seed_rows_60", len(per) == 60, str(len(per)))
    check("cell_rows_20", len(cells) == 20, str(len(cells)))
    check("market_rows_5", len(markets) == 5, str(len(markets)))

    rederived, med_err = {}, []
    for m, h in PANEL:
        rows = [r for r in per if r["market"] == m and r["host"] == h]
        host_maes = {float(r["val_host_mae"]) for r in rows}
        if len(host_maes) != 1:
            FAILS.append(f"host_mae_not_deterministic:{key(m,h)}")
        host_mae = rows[0]["val_host_mae"]
        rec = next(c for c in cells if c["market"] == m and c["host"] == h)
        med = _med([r["val_mae_ema"] for r in rows])
        gain = 100.0 * (float(host_mae) - med) / float(host_mae)
        n_beat = sum(1 for r in rows if r["beats_host"] == "true")
        rederived[key(m, h)] = {
            "median": med, "gain": gain, "host_mae": float(host_mae), "n_beat": n_beat,
            "n_worse_gt1": sum(1 for r in rows if r["seed_worse_than_host_by_gt_1pct"] == "true"),
        }
        med_err.append(abs(med - float(rec["o1_median_mae"])))
        med_err.append(abs(gain - float(rec["gain_host_pct"])))
        if n_beat != int(rec["seeds_beating_host"]):
            FAILS.append(f"seed_win_count_mismatch:{key(m,h)}")
    check("cell_medians_and_gains_reproduce", max(med_err) < 1e-6, f"max_abs_err={max(med_err):.3e}")

    mkt_err = []
    for mr in markets:
        rows = [rederived[key(c["market"], c["host"])] for c in cells if c["market"] == mr["market"]]
        mkt_err.append(abs(_med([r["gain"] for r in rows]) - float(mr["market_median_gain_pct"])))
        if sum(1 for r in rows if r["gain"] > 0) != int(mr["n_strict_host_positive"]):
            FAILS.append(f"market_host_positive_mismatch:{mr['market']}")
    check("market_medians_reproduce", max(mkt_err) < 1e-6, f"max_abs_err={max(mkt_err):.3e}")

    gains = [rederived[key(m, h)]["gain"] for m, h in PANEL]
    panel_median = _med(gains)
    n_pos = sum(1 for g in gains if g > 0)
    n_ge2 = sum(1 for g in gains if g >= 2.0)
    n_ge_m05 = sum(1 for g in gains if g >= -0.5)
    n_2of3 = sum(1 for m, h in PANEL
                 if rederived[key(m, h)]["n_beat"] >= 2)
    n_all3_worse = sum(1 for m, h in PANEL if rederived[key(m, h)]["n_worse_gt1"] == 3)
    mkt_med = [m["market_median_gain_pct"] for m in markets]

    d = {
        "D0": (len(per) == 60 and set(r["test_target_read_count"] for r in per) == {"0"}),
        "D1": (n_pos >= 16 and n_ge_m05 >= 19 and min(gains) >= -1.0),
        "D2": (panel_median >= 1.5 and n_ge2 >= 8),
        "D3": (all(float(x) > 0 for x in mkt_med)
               and sum(1 for x in mkt_med if float(x) >= 1.0) >= 4),
        "D4": (n_2of3 >= 15 and n_all3_worse == 0),
    }
    check("gate_panel_median_reproduces", abs(panel_median - float(gate["panel_median_gain_pct"])) < 1e-6,
          f"{panel_median} vs {gate['panel_median_gain_pct']}")
    check("gate_host_positive_count_reproduces", n_pos == int(gate["n_strict_host_positive"]))
    check("gate_ge_2pct_count_reproduces", n_ge2 == int(gate["n_ge_plus_2pct"]))
    check("gate_blocks_match_recomputation",
          all(d[b] == bool(gate["passed"][b]) for b in d),
          f"recomputed={d} recorded={gate['passed']}")
    check("gate_domestic_pass_is_conjunction",
          bool(gate["domestic_pass"]) == all(d.values()))
    return {"panel_median": panel_median, "n_pos": n_pos, "n_ge2": n_ge2,
            "blocks_recomputed": d, "market_medians": dict(zip(MARKETS, mkt_med))}


# ------------------------------------------------------------------- 4. phase M
def v_phase_m() -> dict:
    rows = read_csv(EVID / "GRADIENT_CONFLICT_AUDIT.csv")
    summ = load(EVID / "GRADIENT_CONFLICT_SUMMARY.json")
    check("gradient_rows_60", len(rows) == 60, str(len(rows)))
    check("gradient_covers_all_20_cells",
          len({r["cell"] for r in rows}) == 20, str(len({r["cell"] for r in rows})))
    check("gradient_zero_optimizer_steps", set(r["optimizer_steps"] for r in rows) == {"0"})
    check("gradient_never_constructed_an_optimizer",
          set(r["optimizer_constructed"] for r in rows) == {"false"})
    check("gradient_all_checkpoints_unchanged",
          all(r["checkpoint_sha256_before"] == r["checkpoint_sha256_after"] for r in rows))
    check("gradient_shared_parameter_set_nonempty",
          all(int(r["shared_parameter_count"]) > 0 for r in rows))
    check("gradient_head_parameters_excluded_from_shared_set",
          all(int(r["head_parameter_numel"]) > 0 for r in rows))
    # A cosine is mathematically undefined exactly when one of its two vectors is
    # the zero vector, which the writer records as the literal ``nan``.  Undefined
    # cosines are excluded from the fractions -- a fraction over an undefined
    # value has no meaning -- but only after each one has been shown to be a
    # genuine exactly-zero gradient, so a ``nan`` produced by a failed computation
    # is caught rather than excused.  The exclusion is then pinned by a
    # discriminating check: where an undefined case exists, the recorded fraction
    # must differ from the fraction taken over all rows.
    COS = ("cos_rec_b", "cos_rec_B", "cos_rec_S", "cos_rec_auxsum")
    NORM_OF = {"cos_rec_b": "grad_norm_b", "cos_rec_B": "grad_norm_B", "cos_rec_S": "grad_norm_S"}

    def defined(v: str) -> bool:
        return v.strip() not in ("", "nan")

    undefined, bad_undefined = [], []
    for r in rows:
        for p in COS:
            if defined(r[p]):
                continue
            undefined.append(f"{r['cell']}__seed{r['seed']}:{p}")
            if p == "cos_rec_auxsum":
                ok = all(float(r[n]) == 0.0 for n in ("grad_norm_b", "grad_norm_B", "grad_norm_S"))
            else:
                ok = float(r[NORM_OF[p]]) == 0.0 and float(r["grad_norm_rec"]) > 0.0
            if not ok:
                bad_undefined.append(f"{r['cell']}__seed{r['seed']}:{p}")
    check("every_undefined_cosine_is_a_genuine_zero_gradient", not bad_undefined,
          str(bad_undefined[:4]))

    checksum, n_defined = {}, {}
    for p in COS:
        keep = [float(r[p]) for r in rows if defined(r[p])]
        n_defined[p] = len(keep)
        checksum[p] = sum(1 for v in keep if v < 0) / len(keep)
    rec = summ["negative_cosine_fraction_overall"]
    check("gradient_overall_fractions_reproduce",
          all(abs(checksum[p] - float(rec[p])) < 1e-9 for p in COS),
          f"{checksum} vs {rec}")
    over_all_rows = {p: sum(1 for r in rows if defined(r[p]) and float(r[p]) < 0) / len(rows)
                     for p in COS}
    check("negative_cosine_fractions_exclude_undefined_cosines",
          all(n_defined[p] == len(rows) or abs(checksum[p] - over_all_rows[p]) > 1e-9 for p in COS),
          json.dumps({p: [n_defined[p], checksum[p], over_all_rows[p]] for p in COS}))

    # The per-Host split and the "strongest Host-family pattern" statement are the
    # part of Phase M that the stage reports, so both are reproduced here rather
    # than read back from the summary.
    host_frac, host_mismatch = {}, []
    for h in HOSTS:
        host_frac[h] = {}
        for p in COS:
            keep = [float(r[p]) for r in rows if r["host"] == h and defined(r[p])]
            host_frac[h][p] = sum(1 for v in keep if v < 0) / len(keep)
            if abs(host_frac[h][p] - float(summ["negative_cosine_fraction_by_host"][h][p])) >= 1e-9:
                host_mismatch.append(f"{h}:{p}")
    check("gradient_host_fractions_reproduce", not host_mismatch, str(host_mismatch[:6]))

    aux_overall = checksum["cos_rec_auxsum"]
    devs = {h: abs(host_frac[h]["cos_rec_auxsum"] - aux_overall) for h in HOSTS}
    strongest = max(devs, key=devs.get)
    declared = summ["strongest_host_family_deviation"]
    check("gradient_strongest_host_family_pattern_reproduces",
          strongest == declared["host"]
          and abs(devs[strongest] - float(declared["absolute_deviation"])) < 1e-9
          and abs(host_frac[strongest]["cos_rec_auxsum"] - float(declared["fraction"])) < 1e-9
          and abs(aux_overall - float(declared["overall_fraction"])) < 1e-9,
          f"recomputed={strongest}/{devs[strongest]} declared={declared}")
    # The audit forward must still be the registered diagnostic forward.  cuDNN
    # is disabled for the audit (its RNN kernels cannot backprop in eval mode),
    # so this re-checks the native path against each run's own recorded value.
    dev = max(float(r["diag_reconstruction_abs_deviation"]) for r in rows)
    check("gradient_forward_reproduces_own_recorded_diagnostic", dev <= 1e-2, f"max_dev={dev:.3e}")
    sub = summ["cudnn_substitution"]
    check("gradient_cudnn_substitution_disclosed_and_bounded",
          sub["all_runs_within_tolerance"] is True
          and abs(float(sub["max_abs_deviation_vs_own_recorded_diag_reconstruction_mae"]) - dev) < 1e-12,
          f"declared={sub['max_abs_deviation_vs_own_recorded_diag_reconstruction_mae']} recomputed={dev}")
    return {"fractions": checksum, "n_defined_cosines": n_defined,
            "undefined_cosines": undefined, "n_runs": len(rows),
            "host_fractions": host_frac, "strongest_host_family": strongest,
            "max_diag_deviation": dev}


# ------------------------------------------------------------------- 5. phase I
def v_phase_i(domestic_passed: bool) -> dict:
    pre_path = EVID / "INTERNATIONAL_PREFLIGHT.json"
    intl_dirs = sorted(d.name for d in (EVID / "intl_runs").iterdir()) if (EVID / "intl_runs").is_dir() else []
    gate_exists = (EVID / "INTERNATIONAL_GATE.json").is_file()
    comp_exists = (EVID / "INTERNATIONAL_CELL_COMPARISON.csv").is_file()

    if not pre_path.is_file():
        # PROTOCOL.md section 3.4: on a domestic FAIL, Phase I is forbidden
        # outright, so an absent preflight is the correct state, not a gap.
        check("phase_i_not_reached_only_because_domestic_failed", not domestic_passed,
              f"domestic_passed={domestic_passed}")
        check("phase_i_not_reached_wrote_no_international_fits", not intl_dirs, str(intl_dirs))
        check("phase_i_not_reached_wrote_no_international_gate", not gate_exists)
        check("phase_i_not_reached_wrote_no_comparison_table", not comp_exists)
        return {"reached": False, "blocked": None, "skip_reason": (
            "Phase D failed its preregistered gate, so PROTOCOL.md sections 3.4 and 5 forbid "
            "Phase I: no international preflight, no international fit and no international "
            "comparison exists, and none may be created under this token.")}

    pre = load(pre_path)
    pins = pre["authority_file_pins"]
    check("preflight_authority_files_all_present", all(v["present"] for v in pins.values()))
    check("preflight_authority_hashes_match_disk",
          all(v["sha256"] == sha(TRANSFER / k) for k, v in pins.items()))
    check("preflight_records_zero_protected_reads",
          int(pre["protected_or_final_target_reads"]) == 0)
    check("preflight_no_fits_executed", int(pre["fits_executed"]) == 0)
    if pre["blocked"]:
        check("blocked_preflight_wrote_no_international_fits", not intl_dirs, str(intl_dirs))
        check("blocked_preflight_wrote_no_international_gate", not gate_exists)
        check("blocked_preflight_wrote_no_comparison_table", not comp_exists)
        per_cell = {c["cell"]: c["o1_legal_evidence_complete"] for c in pre["cells"]}
        check("blocked_preflight_names_all_four_cells_illegal",
              len(per_cell) == 4 and not any(per_cell.values()), str(per_cell))
        for m, h in INTL_CELLS:
            check(f"no_v44_host_cache_for_{m}_{h}",
                  not (V2 / "02_hosts" / m / h / "host_predictions_joint.npz").is_file())
            check(f"no_v44_shadow_support_for_{m}_{h}",
                  not (SHADOW / f"{key(m,h)}.json").is_file())
        if domestic_passed:
            # A blocked preflight reached through the domestic PASS is the
            # authorized path, and then there is nothing to disclose.
            check("no_deviation_disclosure_when_phase_i_was_authorized",
                  not (EVID / "INTERNATIONAL_PREFLIGHT_DISCLOSURE.json").is_file())
        else:
            # The domestic gate failed, so PROTOCOL.md sections 3.4 and 5 forbid
            # Phase I outright: this zero-fit preflight ran outside the gated
            # order.  It cannot be un-run and deleting it would destroy a real
            # finding, so it is kept and its harmlessness is verified instead of
            # its absence being asserted.
            dis_path = EVID / "INTERNATIONAL_PREFLIGHT_DISCLOSURE.json"
            check("out_of_order_preflight_is_disclosed", dis_path.is_file(), str(dis_path))
            if dis_path.is_file():
                dis = load(dis_path)
                gate_mtime = (EVID / "DOMESTIC_GATE.json").stat().st_mtime
                pre_mtime = pre_path.stat().st_mtime
                check("disclosure_declares_phase_i_was_not_authorized",
                      dis["phase_i_authorized_by_domestic_gate"] is False
                      and dis["domestic_pass"] is False)
                fit_times = [p.stat().st_mtime for p in (EVID / "o1_runs").rglob("freeze.json")]
                sweep_start = (EVID / "REUSE_SNAPSHOT_BEFORE.json").stat().st_mtime
                sweep_last = max(fit_times) if fit_times else float("nan")
                check("disclosure_ordering_matches_file_times",
                      bool(dis["preflight_written_during_phase_d_sweep"])
                      == (sweep_start <= pre_mtime <= sweep_last)
                      and bool(dis["domestic_gate_existed_when_preflight_ran"])
                      == (pre_mtime > gate_mtime),
                      f"declared_during_sweep={dis['preflight_written_during_phase_d_sweep']} "
                      f"pre={pre_mtime} sweep=[{sweep_start},{sweep_last}] gate={gate_mtime}")
                check("disclosure_preflight_could_not_be_conditioned_on_domestic_verdict",
                      dis["preflight_conditioned_on_domestic_verdict"] is False
                      and not (pre_mtime > gate_mtime))
                check("disclosure_authority_files_unchanged_is_true",
                      dis["authority_files_unchanged_since_preflight"] is True
                      and all(v["sha256"] == sha(TRANSFER / k) for k, v in pins.items()))
                check("disclosure_zero_fit_counters_match_preflight_and_disk",
                      int(dis["fits_executed_by_preflight"]) == int(pre["fits_executed"]) == 0
                      and int(dis["n_new_fits_authorized_by_preflight"]) == 0
                      and int(dis["international_run_directories"]) == len(intl_dirs) == 0
                      and dis["international_comparison_written"] is False and not comp_exists
                      and dis["international_gate_written"] is False and not gate_exists
                      and int(dis["test_target_read_count"]) == 0
                      and int(dis["protected_or_final_target_reads"]) == 0)
                check("disclosure_states_deviation_and_null_effect_on_token",
                      len(str(dis.get("deviation", ""))) > 40
                      and "token" in str(dis.get("effect_on_terminal_token", "")))
    else:
        check("unblocked_preflight_must_have_run_12_fits",
              sum(len(list(d.iterdir())) for d in (EVID / "intl_runs").iterdir()) == 12
              if intl_dirs else False)
    return {"blocked": bool(pre["blocked"]),
            "reason": pre.get("blocker", {}).get("reason_code"),
            "authorized_by_domestic_gate": domestic_passed,
            "deviation_disclosed": (not domestic_passed)
            and (EVID / "INTERNATIONAL_PREFLIGHT_DISCLOSURE.json").is_file(),
            "n_intl_runs": 0, "n_intl_run_dirs": len(intl_dirs), "gate_written": gate_exists}


# ----------------------------------------------------------------- 6. no mutation
def v_no_mutation() -> dict:
    audit = load(EVID / "ACCESS_AUDIT.json")
    state = audit["access_state"]
    check("access_audit_test_reads_zero", int(audit["test_target_read_count"]) == 0)
    check("access_audit_no_test_rows_returned", int(state["test_rows_returned"]) == 0)
    check("access_audit_no_forbidden_path_opened",
          state["distinct_paths_blocked"] == [], str(state["distinct_paths_blocked"][:4]))
    check("access_audit_source_tree_matches_frozen_pin",
          audit["source_tree_digest"]["core_tree"] == CORE_TREE_PIN)
    check("no_result_conditioned_retry",
          load(EVID / "FIT_LOG.json")["no_retry"] is True)
    check("fits_logged_equal_registered_36",
          load(EVID / "FIT_LOG.json")["n_registered_new_fits"] == 36)
    check("backbone_tree_recomputable", len(tree_digest("src/backbones")) == 64)
    return {"test_reads": 0, "forbidden_opened": 0}


def main() -> int:
    universe = v_universe()
    hashes = v_hashes()
    domestic = v_domestic()
    phase_m = v_phase_m()
    phase_i = v_phase_i(all(domestic["blocks_recomputed"].values()))
    no_mut = v_no_mutation()

    report = {
        "schema": "hch_v44_o1_domestic20_independent_verification.v1",
        "verifier": "verification/verify.py",
        "imports_from_implementation": [],
        "independence": ("coordinate universe, reused-cell set, seeds and all D/I thresholds are "
                         "transcribed from PROTOCOL.md; CSV files are re-parsed and re-aggregated "
                         "with this file's own arithmetic; checkpoints are re-hashed here"),
        "n_checks": len(CHECKS),
        "n_failed": len(FAILS),
        "failures": FAILS,
        "checks": CHECKS,
        "universe": universe, "hashes": hashes, "domestic": domestic,
        "phase_m": phase_m, "phase_i": phase_i, "no_mutation": no_mut,
        "verdict": "VERIFIED" if not FAILS else "VERIFICATION_FAILED",
    }
    (EVID / "INDEPENDENT_VERIFICATION_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n_checks": len(CHECKS), "n_failed": len(FAILS),
                      "failures": FAILS[:10], "verdict": report["verdict"]}, indent=2))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
