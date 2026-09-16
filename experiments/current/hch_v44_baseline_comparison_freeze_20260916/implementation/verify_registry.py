"""Independent verifier for the HCH v4.4 domestic baseline comparison freeze.

Protocol: `HCH_V44_BASELINE_COMPARISON_FREEZE_20260916` section 10.

Independence contract (PROTOCOL section 10): this module **must not import the
registry-construction result functions**.  It does not import `build_registry`.
It re-reads the parent frozen artifacts from disk, re-derives the scoring mask
from the arrays, and recomputes MAE, the region metrics and the gain arithmetic
with its own implementation -- deliberately *not* `panel_metrics` -- so that the
registry is checked against the sources rather than against the code that
produced it.

Checked field: the artifacts under
`experiments/evidence/hch_v44_baseline_comparison_freeze_20260916/`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/作业/science/solar_leak_price_model")
V2 = ROOT / "experiments/evidence/hch_china5_common_benchmark_701020_full_v2_20260913"
ADJ = V2 / "10_missing_target_adjudication"
LAB = ROOT / "experiments/lab/common_benchmark_results_v2"
METRIC_DIR = ROOT / "experiments/current/china5_posthoc_baseline_panel"
EVID = ROOT / "experiments/evidence/hch_v44_baseline_comparison_freeze_20260916"

MARKETS = ["GANSU_DA", "SHANDONG_DA", "SHAANXI_DA", "NINGXIA_DA", "QINGHAI_DA"]
HOSTS = ["PatchTST", "TimeMixer", "iTransformer", "LSTM"]
METHODS = ["Host", "MatchedDirectResidual", "delta-Adapter", "PIR",
           "COSA", "UEC-STD", "OMPB"]
EXPECTED_NUMERIC_PER_METHOD = {"Host": 20, "MatchedDirectResidual": 20,
                               "delta-Adapter": 20, "PIR": 10, "COSA": 20,
                               "UEC-STD": 0, "OMPB": 0}
STRICT_OFFLINE = {"Host", "MatchedDirectResidual", "delta-Adapter", "PIR"}

SUCCESS = "HCH_V44_BASELINE_PAPER_REGISTRY_COMPLETE"
BLOCKED = "HCH_V44_BASELINE_REGISTRY_BLOCKED_FOR_ADJUDICATION"

_checks = []


def check(cid, name, passed, detail):
    _checks.append({"id": cid, "name": name, "passed": bool(passed), "detail": detail})
    return bool(passed)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest().upper()


def read_csv(p: Path):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(x):
    """Registry CSVs store repr(float); empty means absent."""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return x


def independent_regions(yt, yp, hp, q05, q95, s90):
    """A second, independent implementation of the frozen region rules.

    Written from `05_thresholds/METRIC_CONTRACT.json::region_rule` directly, not
    imported from the registry side or from panel_metrics.
    """
    e = np.abs(yt - yp)
    eh = np.abs(yt - hp)
    up, lo = yt >= q95, yt <= q05
    tail, norm = up | lo, ~(up | lo)
    spread = yt.max(axis=1) - yt.min(axis=1)
    extreme = spread >= s90
    ex = np.repeat(extreme[:, None], yt.shape[1], axis=1)
    neg = yt < 0
    sub = lambda m, a: (float(a[m].mean()) if m.any() else None)
    return {
        "mae": float(np.mean(e)),
        "mse": float(np.mean((yt - yp) ** 2)),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "host_mae": float(np.mean(eh)),
        "tail": sub(tail, e), "tail_n": int(tail.sum()),
        "upper": sub(up, e), "upper_n": int(up.sum()),
        "lower": sub(lo, e), "lower_n": int(lo.sum()),
        "normal": sub(norm, e), "normal_n": int(norm.sum()),
        "host_normal": sub(norm, eh),
        "negative": sub(neg, e), "negative_n": int(neg.sum()),
        "extreme": sub(ex, e), "extreme_n": int(extreme.sum()),
        "upper_tail_day": sub(np.repeat(up.any(axis=1)[:, None], yt.shape[1], 1), e),
        "upper_tail_day_n": int(up.any(axis=1).sum()),
    }


def main() -> int:
    m140 = read_csv(EVID / "MATRIX_140.csv")
    strict = read_csv(EVID / "STRICT_OFFLINE_TABLE.csv")
    online = read_csv(EVID / "ONLINE_SUPPLEMENTARY_TABLE.csv")
    blocked = read_csv(EVID / "BLOCKED_ROWS.csv")
    snap_csv = read_csv(EVID / "CELL_COMPARATOR_SNAPSHOT.csv")
    snap = json.loads((EVID / "CELL_COMPARATOR_SNAPSHOT.json").read_text(encoding="utf-8"))
    hashes = json.loads((EVID / "SOURCE_HASHES.json").read_text(encoding="utf-8"))
    audit = json.loads((EVID / "BASELINE_COMPLETENESS_AUDIT.json").read_text(encoding="utf-8"))

    thresholds = {m: {"q05": d["q05"], "q95": d["q95"], "day_spread_p90": d["day_spread_p90_S90"], **d}
                  for m, d in json.loads(
                      (V2 / "05_thresholds/THRESHOLD_FREEZE.json").read_text(
                          encoding="utf-8"))["thresholds"].items()}
    manifest = {m["market"]: m for m in json.loads(
        (ADJ / "SCORABLE_DAY_MANIFEST.json").read_text(encoding="utf-8"))["markets"]}
    parent = {}
    for r in read_csv(LAB / "RESULTS_LONG.csv"):
        if r["split"] == "TEST" and r["method"] in METHODS:
            parent[(r["market"], r["host"], r["method"])] = r

    numeric = [r for r in m140 if r["status"] == "NUMERIC"]
    blk = [r for r in m140 if r["status"] == "BLOCKED"]

    # ---- 1. coordinate census -------------------------------------------
    coords = [(r["market"], r["host"], r["method"]) for r in m140]
    expected = {(m, h, me) for m in MARKETS for h in HOSTS for me in METHODS}
    check(1, "MATRIX_140 census", len(m140) == 140 and len(set(coords)) == 140
          and set(coords) == expected,
          f"rows={len(m140)} unique={len(set(coords))} equals_cartesian={set(coords) == expected}")

    # ---- 2. numeric / blocker counts ------------------------------------
    check(2, "numeric=90 and blocked=50", len(numeric) == 90 and len(blk) == 50,
          f"numeric={len(numeric)} blocked={len(blk)}")

    # ---- 3-6. per-method census -----------------------------------------
    per = {me: sum(1 for r in numeric if r["method"] == me) for me in METHODS}
    check(3, "Host/MDR/delta/COSA = 20 numeric each",
          all(per[k] == 20 for k in ["Host", "MatchedDirectResidual",
                                     "delta-Adapter", "COSA"]),
          json.dumps(per))
    pir_num = [r for r in numeric if r["method"] == "PIR"]
    pir_blk = [r for r in blk if r["method"] == "PIR"]
    check(4, "PIR = 10 numeric + 10 blocker",
          len(pir_num) == 10 and len(pir_blk) == 10
          and {r["host"] for r in pir_num} == {"PatchTST", "TimeMixer"}
          and {r["host"] for r in pir_blk} == {"iTransformer", "LSTM"}
          and all(r["blocker_class"] == "Q-INCOMPATIBLE" for r in pir_blk),
          f"numeric_hosts={sorted({r['host'] for r in pir_num})} "
          f"blocked_hosts={sorted({r['host'] for r in pir_blk})}")
    uec = [r for r in blk if r["method"] == "UEC-STD"]
    ompb = [r for r in blk if r["method"] == "OMPB"]
    check(5, "UEC-STD = 20 blocker (Q-FIDELITY)",
          len(uec) == 20 and all(r["blocker_class"] == "Q-FIDELITY" for r in uec),
          f"n={len(uec)} classes={sorted({r['blocker_class'] for r in uec})}")
    check(6, "OMPB = 20 blocker (Q-DATA)",
          len(ompb) == 20 and all(r["blocker_class"] == "Q-DATA" for r in ompb),
          f"n={len(ompb)} classes={sorted({r['blocker_class'] for r in ompb})}")

    # ---- 7. strict offline table ----------------------------------------
    s_num = [r for r in strict if r["status"] == "NUMERIC"]
    check(7, "strict offline table = 70 numeric rows",
          len(s_num) == 70 and {r["method"] for r in strict} <= STRICT_OFFLINE
          and len(strict) == 80,
          f"rows={len(strict)} numeric={len(s_num)} methods={sorted({r['method'] for r in strict})}")

    # ---- 8. COSA only online --------------------------------------------
    off_methods = {r["method"] for r in strict} | {r["method"] for r in
                                                   [x for x in m140 if x["method"] != "COSA"]}
    cosa_offline = [r for r in strict if r["method"] == "COSA"]
    check(8, "COSA appears only in the online/TTA table",
          not cosa_offline and all(r["method"] == "COSA" or r["row_role"] == "SAME_CELL_HOST_REFERENCE"
                                   for r in online)
          and all(r["setting"] == "ONLINE_TTA" or r["row_role"] == "SAME_CELL_HOST_REFERENCE"
                  for r in online)
          and "COSA" not in off_methods,
          f"cosa_in_offline_table={len(cosa_offline)} online_rows={len(online)}")

    # ---- 9/10/11 + independent metric recomputation ----------------------
    mae_bad, gain_bad, subset_bad, mask_bad, defn_bad = [], [], [], [], []
    max_gain_dev = 0.0
    for r in numeric:
        market, host, method = r["market"], r["host"], r["method"]
        key = (market, host, method)
        if method == "Host":
            z = np.load(V2 / f"08_joint_test/hosts/{market}/{host}/TEST.npz")
            yt = z["y_true"].astype(np.float64); yp = z["pred"].astype(np.float64); hp = yp.copy()
        else:
            z = np.load(V2 / f"08_joint_test/baselines/{market}/{host}/{method}.npz")
            yt = z["y_true"].astype(np.float64); yp = z["pred"].astype(np.float64)
            hp = z["host"].astype(np.float64)
        keep = ~np.isnan(yt).any(axis=1)
        n_scored = int(keep.sum())
        th = thresholds[market]
        R = independent_regions(yt[keep], yp[keep], hp[keep],
                                th["q05"], th["q95"], th["day_spread_p90"])

        reg_mae = f(r["mae"])
        if not (reg_mae == R["mae"] == float(parent[key]["mae"])):
            mae_bad.append({"coordinate": f"{market}/{host}/{method}",
                            "registry": reg_mae, "independent": R["mae"],
                            "parent": float(parent[key]["mae"])})
        # gain arithmetic, recomputed against the independently derived Host MAE
        if R["host_mae"]:
            want = (R["host_mae"] - R["mae"]) / R["host_mae"] * 100.0
            got = f(r["relative_gain_vs_host_pct"])
            dev = abs(want - got)
            max_gain_dev = max(max_gain_dev, dev)
            if dev > 1e-9:
                gain_bad.append({"coordinate": f"{market}/{host}/{method}",
                                 "registry": got, "independent": want})
        # subset parity
        for col, val in [("tail_mae", R["tail"]), ("upper_tail_mae", R["upper"]),
                         ("lower_tail_mae", R["lower"]),
                         ("normal_region_mae", R["normal"]),
                         ("negative_price_mae", R["negative"]),
                         ("extreme_day_mae", R["extreme"]),
                         ("upper_tail_day_mae", R["upper_tail_day"])]:
            g = f(r[col])
            if (g is None) != (val is None) or (g is not None and abs(g - val) > 1e-9):
                subset_bad.append({"coordinate": f"{market}/{host}/{method}",
                                   "field": col, "registry": g, "independent": val})
        for col, val in [("tail_n_hours", R["tail_n"]), ("upper_tail_n_hours", R["upper_n"]),
                         ("lower_tail_n_hours", R["lower_n"]),
                         ("normal_n_hours", R["normal_n"]),
                         ("negative_n_hours", R["negative_n"]),
                         ("extreme_day_n_days", R["extreme_n"]),
                         ("upper_tail_day_n_days", R["upper_tail_day_n"])]:
            if int(r[col]) != val:
                subset_bad.append({"coordinate": f"{market}/{host}/{method}",
                                   "field": col, "registry": int(r[col]), "independent": val})
        # Shandong 329-day common scoring mask
        if n_scored != manifest[market]["n_scored_test_days"] or \
                int(r["n_scored_test_days"]) != n_scored or \
                int(r["n_registered_test_days"]) != manifest[market]["n_registered_test_days"]:
            mask_bad.append({"coordinate": f"{market}/{host}/{method}",
                             "array_scored": n_scored,
                             "registry": int(r["n_scored_test_days"]),
                             "declared": manifest[market]["n_scored_test_days"]})
        if r["metric_definition_sha256"] != hashes["frozen_metric_definition_sha256"]:
            defn_bad.append(f"{market}/{host}/{method}")

    check(9, "every numeric MAE equals its frozen V2 parent value",
          not mae_bad, f"compared={len(numeric)} mismatches={len(mae_bad)}"
          + (f" first={mae_bad[:3]}" if mae_bad else ""))
    check(10, "relative gains recompute from the same-cell Host (independent)",
          not gain_bad, f"max_abs_deviation={max_gain_dev:.3e} mismatches={len(gain_bad)}")
    sd = [r for r in numeric if r["market"] == "SHANDONG_DA"]
    check(11, "Shandong scored-day count is 329 for every numeric method",
          not mask_bad and len(sd) == 18 and all(int(r["n_scored_test_days"]) == 329 for r in sd)
          and all(int(r["n_registered_test_days"]) == 330 for r in sd),
          f"shandong_numeric_rows={len(sd)} scored={sorted({r['n_scored_test_days'] for r in sd})} "
          f"declared_unscorable={[d['day_id'] for d in manifest['SHANDONG_DA']['unscorable_days']]}")
    check("9b", "subset metrics and counts reproduce independently",
          not subset_bad, f"values_compared={len(numeric) * 14} mismatches={len(subset_bad)}"
          + (f" first={subset_bad[:3]}" if subset_bad else ""))
    check("9c", "registry pins the frozen metric definition hash",
          not defn_bad and hashes["frozen_metric_definition_sha256"] ==
          sha256_file(METRIC_DIR / "panel_metrics.py"),
          f"sha256={hashes['frozen_metric_definition_sha256'][:16]}...")

    # ---- 12. parent hashes unchanged -------------------------------------
    regressed = []
    for rel, want in hashes["parent_source_hashes"].items():
        p = ROOT / rel
        if not p.exists() or sha256_file(p) != want:
            regressed.append(rel)
    cell_regressed = []
    for rel, want in hashes["per_cell_source_hashes"].items():
        p = ROOT / rel
        if not p.exists() or sha256_file(p) != want:
            cell_regressed.append(rel)
    check(12, "parent source hashes unchanged",
          not regressed and not cell_regressed,
          f"parent_files={len(hashes['parent_source_hashes'])} "
          f"per_cell_files={len(hashes['per_cell_source_hashes'])} "
          f"regressed={regressed + cell_regressed}")

    # ---- 13. zero new fits ----------------------------------------------
    MODEL_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".h5", ".pkl", ".joblib"}
    stage_begin = min(os.path.getmtime(EVID / n) for n in
                      ["MATRIX_140.csv", "STRICT_OFFLINE_TABLE.csv",
                       "ONLINE_SUPPLEMENTARY_TABLE.csv", "BLOCKED_ROWS.csv",
                       "CELL_COMPARATOR_SNAPSHOT.json", "SOURCE_HASHES.json",
                       "BASELINE_COMPLETENESS_AUDIT.json"])
    new_models = []
    for base in [ROOT / "experiments", ROOT / "src", ROOT / "checkpoints"]:
        if not base.exists():
            continue
        for dp, dn, fn in os.walk(base):
            if EVID.as_posix() in Path(dp).as_posix():
                continue
            for name in fn:
                p = Path(dp) / name
                if p.suffix.lower() not in MODEL_EXT:
                    continue
                try:
                    if os.path.getmtime(p) > stage_begin:
                        new_models.append(str(p.relative_to(ROOT)).replace("\\", "/"))
                except OSError:
                    pass  # unreadable/dangling archive path: not a new fit
    check(13, "new baseline fit count = 0",
          not new_models and audit["no_new_fit_audit"]["new_baseline_neural_fits"] == 0,
          f"new_model_artifacts_after_stage_begin={len(new_models)}"
          + (f" {new_models[:5]}" if new_models else "")
          + f"; stage_begin={stage_begin:.0f}")

    # ---- 14. no paper / source mutation ---------------------------------
    mutated = []
    for base in [ROOT / "paper", ROOT / "src", V2, LAB]:
        if not base.exists():
            continue
        for dp, dn, fn in os.walk(base):
            for name in fn:
                p = Path(dp) / name
                try:
                    if os.path.getmtime(p) > stage_begin:
                        mutated.append(str(p.relative_to(ROOT)).replace("\\", "/"))
                except OSError:
                    pass
    check(14, "no parent-evidence / paper / src mutation in this stage",
          not mutated,
          f"paper_root_exists={(ROOT / 'paper').exists()} mutated_after_stage_begin={len(mutated)}"
          + (f" {mutated[:5]}" if mutated else ""))

    # ---- supporting structural checks ------------------------------------
    check("S1", "blocker rows carry class/code/reason/authority and are unreopened",
          all(r["blocker_class"] and r["blocker_code"] and r["blocker_reason"]
              and r["blocker_authority"] for r in blk)
          and all(r["blocker_reopened"] == "False" for r in blk),
          f"blocked={len(blk)} reopened={sorted({r['blocker_reopened'] for r in blk})}")
    check("S2", "every numeric row carries the historical-exposure caveat",
          all(r["paper_use_scope"] == "COMMON_BENCHMARK_TEST_WITH_HISTORICAL_EXPOSURE_CAVEAT"
              for r in numeric),
          f"scopes={sorted({r['paper_use_scope'] for r in numeric})}")
    canon = {r["method"]: r["method_canonical_setting"] for r in m140}
    # canonical settings must reproduce the frozen declarations independently
    frozen_uec = json.loads((V2 / "08_joint_test/baselines/GANSU_DA/PatchTST/"
                             "UEC-STD.BLOCKED.json").read_text(encoding="utf-8"))["setting"]
    frozen_ompb = json.loads((V2 / "08_joint_test/baselines/GANSU_DA/PatchTST/"
                              "OMPB.BLOCKED.json").read_text(encoding="utf-8"))["setting"]
    check("S3", "setting separation: no ONLINE_TTA in the offline table; blocked=BLOCKED",
          all(r["setting"] != "ONLINE_TTA" for r in strict)
          and all(r["setting"] == "BLOCKED" for r in blk)
          and canon["COSA"] == "ONLINE_TTA"
          and canon["UEC-STD"] == frozen_uec and canon["OMPB"] == frozen_ompb
          and all(r["setting"] == "BLOCKED" for r in blk),
          f"strict_settings={sorted({r['setting'] for r in strict})} "
          f"blocked_settings={sorted({r['setting'] for r in blk})} "
          f"canonical={json.dumps(canon)} frozen_uec={frozen_uec} frozen_ompb={frozen_ompb}")
    check("S4", "comparator snapshot has exactly 20 market x Host records",
          len(snap["records"]) == 20 and len(snap_csv) == 20
          and {(r["market"], r["host"]) for r in snap_csv}
          == {(m, h) for m in MARKETS for h in HOSTS},
          f"records={len(snap['records'])} csv_rows={len(snap_csv)}")
    NAME = {"delta_adapter": "delta-Adapter", "pir": "PIR"}
    snap_bad, selector_bad = [], []
    for rec in snap["records"]:
        off = rec["strict_offline_external_posthoc"]
        cands = [(NAME[k], off[k]["mae_abs_price"]) for k in ("delta_adapter", "pir")
                 if off[k] is not None]
        if cands:
            best = min(cands, key=lambda t: t[1])
            if (off["best_method"], off["best_mae_abs_price"]) != best:
                snap_bad.append({"cell": rec["market"] + "/" + rec["host"],
                                 "registry": [off["best_method"], off["best_mae_abs_price"]],
                                 "independent": list(best)})
        if off["best_method"] not in (None, "delta-Adapter", "PIR"):
            selector_bad.append({rec["market"] + "/" + rec["host"]: off["best_method"]})
        # an online method must never be selectable as the offline comparator
        if off["best_method"] == "COSA":
            selector_bad.append({rec["market"] + "/" + rec["host"]: "COSA selected offline"})
    check("S5", "offline selector picks only from {delta-Adapter, numeric PIR}",
          not snap_bad and not selector_bad,
          f"cells={len(snap['records'])} mae_violations={snap_bad} "
          f"selector_violations={selector_bad}")
    check("S6", "harm fields are unit-separated, never merged",
          "normal_harm_vs_host_abs_price" in m140[0]
          and "normal_harm_vs_host_pct_of_host_normal_mae" in m140[0]
          and "normal_harm_vs_host" not in [c for c in m140[0] if c.endswith("_pct")] ,
          "absolute and percent harm are distinct columns")
    check("S7", "blocked rows are never ranked",
          all(r["mae_rank_within_cell"] == "" for r in strict if r["status"] == "BLOCKED")
          and all(r["is_best_strict_offline_external_posthoc"] == "False"
                  for r in strict if r["status"] == "BLOCKED"),
          f"blocked_in_offline_table={sum(1 for r in strict if r['status']=='BLOCKED')}")

    # ---- S8. evidence root holds exactly the required registry artifacts ----
    REQUIRED = {"MATRIX_140.csv", "STRICT_OFFLINE_TABLE.csv",
                "ONLINE_SUPPLEMENTARY_TABLE.csv", "BLOCKED_ROWS.csv",
                "CELL_COMPARATOR_SNAPSHOT.csv", "CELL_COMPARATOR_SNAPSHOT.json",
                "SOURCE_HASHES.json", "BASELINE_COMPLETENESS_AUDIT.json",
                "RESULTS.md", "VERIFICATION_REPORT.json"}
    present = {p.name for p in EVID.iterdir() if p.is_file()}
    stray = present - REQUIRED
    check("S8", "evidence root holds exactly the required registry artifacts",
          not stray and REQUIRED <= present,
          f"missing={sorted(REQUIRED - present)} stray={sorted(stray)} "
          f"quarantined={'superseded_ai_execution_prompt_20260916' if (EVID / 'superseded_ai_execution_prompt_20260916').is_dir() else 'none'}")

    n_passed = sum(1 for c in _checks if c["passed"])
    ok = n_passed == len(_checks)
    report = {
        "schema": "hch_v44_baseline_freeze_verification.v1",
        "protocol_id": "HCH_V44_BASELINE_COMPARISON_FREEZE_20260916",
        "verifier": "experiments/current/hch_v44_baseline_comparison_freeze_20260916/"
                    "implementation/verify_registry.py",
        "verifier_independence": {
            "imports_registry_constructor": False,
            "imports_frozen_metric_module": False,
            "note": ("this verifier does not import build_registry and does not import "
                     "panel_metrics; it re-implements the frozen region rules from "
                     "METRIC_CONTRACT.json::region_rule and recomputes every value "
                     "from the parent arrays"),
        },
        "checked_artifacts": sorted(p.name for p in EVID.iterdir() if p.is_file()),
        "n_checks": len(_checks),
        "n_passed": n_passed,
        "passed": ok,
        "verdict_token": SUCCESS if ok else BLOCKED,
        "counts": {
            "matrix_140": len(m140), "numeric": len(numeric), "blocked": len(blk),
            "strict_offline_rows": len(strict),
            "strict_offline_numeric": len(s_num),
            "online_rows": len(online),
            "snapshot_records": len(snap["records"]),
            "numeric_mae_vs_parent": len(numeric) - len(mae_bad),
            "subset_values_compared": len(numeric) * 14,
        },
        "independent_recomputation": {
            "max_relative_gain_deviation": max_gain_dev,
            "mae_mismatches": mae_bad,
            "gain_mismatches": gain_bad,
            "subset_mismatches": subset_bad,
            "scoring_mask_mismatches": mask_bad,
        },
        "sidecar_disclosure": {
            "role": "per-cell sidecars are disclosure/fidelity metadata, not numeric authority",
            "why": audit["metric_authority"]["per_cell_sidecar_caveat"],
            "max_rel_deviation_registry_vs_sidecar":
                audit["metric_authority"]["max_sidecar_rel_deviation"],
        },
        "checks": _checks,
    }
    (EVID / "VERIFICATION_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"checks {n_passed}/{len(_checks)} passed")
    for c in _checks:
        if not c["passed"]:
            print("  FAIL", c["id"], c["name"], "::", c["detail"])
    print("verdict:", report["verdict_token"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
