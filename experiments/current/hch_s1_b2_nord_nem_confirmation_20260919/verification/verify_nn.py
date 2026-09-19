"""Independent verifier for the bounded NORD_DK1 / NEM_SA1 confirmation.

This module imports **nothing** from the executor or from the stage's contract layer.
It re-derives every path, every hash and every number it checks from the frozen
artefacts and from the executor's written outputs, so that a defect in
``confirmation.py`` (or in the reused ``intl_shared.py``) cannot be mirrored by a
shared helper here.

What it recomputes:

* the frozen contract layer's digest, against the digest the blocked predecessor
  published -- the bounded successor's whole licence rests on that layer being
  unchanged;
* the P0 chain: token -> preflight -> per-Host re-derivation blobs -> frozen Host
  manifests, each link by content, never by timestamp;
* the protected-day inventory from the frozen ``ELIGIBLE_DAYS.csv``, so the day
  counts in the results are checked against the freeze rather than against
  themselves;
* the per-seed metric arithmetic, the cell medians, the ``best_static`` ranking and
  the two comparison booleans;
* the authority's section 7 gates and the terminal token, from the raw numbers;
* the no-proxy surface, the COSA chronology results, and the protected-read recount.

``mtime`` is never used as evidence of anything (repo ``AGENTS.md`` section 8).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
STAGE = ROOT / "experiments/current/hch_s1_b2_nord_nem_confirmation_20260919"
EVID = ROOT / "experiments/evidence/hch_s1_b2_nord_nem_confirmation_20260919"
CONF = EVID / "confirmation"
VERIFY = EVID / "verification"

MARKETS = ("NORD_DK1", "NEM_SA1")
HOSTS = ("PatchTST", "TimeMixer", "iTransformer", "LSTM")
SEEDS = (7, 17, 37)
CELLS = tuple((m, h) for m in MARKETS for h in HOSTS)
PIR_LEGAL_HOSTS = ("PatchTST", "TimeMixer")
H = 24

PRIOR_SHARED = (ROOT / "experiments/current/hch_s1_b2_international_confirmation_20260919"
                / "implementation/intl_shared.py")
PRIOR_SHARED_SHA256 = "617D8FDFAE228ADC0EB0AC4164983CB4755147EDA5465588477C3E2D3163356C"
PRIOR_BLOCK = (ROOT / "experiments/evidence/"
               "hch_s1_b2_international_confirmation_20260919/P0_BLOCKED.json")

FAMILY_EVID = {"nem_nord": "hch_international_nem_nord_baseline_matrix_20260912"}
FAMILY_OF = {"NORD_DK1": "nem_nord", "NEM_SA1": "nem_nord"}
FORBIDDEN_MARKETS = ("LAGO_NP", "GEFCOM14P")

TOL = 1e-9
FAILURES: list[str] = []
CHECKS: list[dict] = []


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def tree_sha(path: Path) -> str:
    h = hashlib.sha256()
    if not Path(path).exists():
        return "ABSENT"
    for p in sorted(x for x in Path(path).rglob("*")
                    if x.is_file() and "__pycache__" not in x.parts):
        h.update(p.relative_to(path).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest().upper()


def check(name: str, ok: bool, detail=None) -> bool:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    print(f"[{'ok ' if ok else 'FAIL'}] {name}", flush=True)
    return bool(ok)


def close(a, b) -> bool:
    a, b = float(a), float(b)
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def frozen_root(market: str) -> Path:
    return ROOT / f"experiments/evidence/{FAMILY_EVID[FAMILY_OF[market]]}"


def protected_days(market: str) -> list[str]:
    p = frozen_root(market) / "01_dataset_contracts" / market / "ELIGIBLE_DAYS.csv"
    df = pd.read_csv(p)
    if not {"role", "date"} <= set(df.columns):
        raise RuntimeError(f"{market}: frozen ELIGIBLE_DAYS.csv lacks role/date")
    sel = df[df["role"] == "PROTECTED_FINAL"]
    days = [str(d)[:10] for d in sel["date"]]
    if len(days) != len(set(days)):
        raise RuntimeError(f"{market}: duplicate protected days in the frozen day list")
    if days != sorted(days):
        raise RuntimeError(f"{market}: frozen protected days are not chronological")
    return days


def main() -> int:
    VERIFY.mkdir(parents=True, exist_ok=True)
    for name in ("S1B2_CELL_MEDIANS.csv", "S1B2_CELL_SEED_RESULTS.csv",
                 "S1B2_BASELINE_RESULTS.csv", "S1B2_PANEL.json",
                 "PROTECTED_ACCESS_AUDIT.json", "VERDICT.json"):
        if not (CONF / name).is_file():
            raise RuntimeError(f"confirmation output absent: {name}; nothing to verify")
    if not (EVID / "P0_SUPPORT_PREFLIGHT.json").is_file():
        raise RuntimeError("P0 preflight absent")

    cells = pd.read_csv(CONF / "S1B2_CELL_MEDIANS.csv")
    seeds = pd.read_csv(CONF / "S1B2_CELL_SEED_RESULTS.csv")
    base = pd.read_csv(CONF / "S1B2_BASELINE_RESULTS.csv")
    panel = load_json(CONF / "S1B2_PANEL.json")
    access = load_json(CONF / "PROTECTED_ACCESS_AUDIT.json")
    verdict = load_json(CONF / "VERDICT.json")
    preflight = load_json(EVID / "P0_SUPPORT_PREFLIGHT.json")
    token = load_json(EVID / "P0_PASS.json")

    # ---- V1  the reused contract layer must be the published one -------------
    shared_sha = sha(PRIOR_SHARED)
    check("V1.reused_contract_layer_digest",
          shared_sha == PRIOR_SHARED_SHA256,
          f"{shared_sha} vs published {PRIOR_SHARED_SHA256}")
    fl = preflight["frozen_modules"]["reused_contract_layer"]
    check("V1b.p0_recorded_the_same_digest",
          str(fl["reused_module_sha256"]).upper() == PRIOR_SHARED_SHA256
          and bool(fl["digest_verified_before_rebinding"]),
          fl["reused_module_sha256"])
    check("V1c.bounded_scope_is_two_markets",
          list(fl["markets_reachable"]) == list(MARKETS)
          and list(fl["excluded_markets"]) == list(FORBIDDEN_MARKETS)
          and fl["excluded_markets_reachable"] is False,
          fl["markets_reachable"])

    # ---- V2  the P0 token ----------------------------------------------------
    check("V2.p0_token_is_a_clean_pass",
          token.get("verdict") == "PASS"
          and int(token.get("protected_final_target_values_read", -1)) == 0
          and int(token.get("host_cells_identity_verified", -1)) == len(CELLS)
          and list(token.get("markets", [])) == list(MARKETS),
          token.get("verdict"))
    check("V2b.token_binds_the_preflight_by_content",
          str(token["preflight_sha256"]).upper() == sha(EVID / "P0_SUPPORT_PREFLIGHT.json"),
          token["preflight_sha256"])
    check("V2c.executor_cites_the_token_on_disk",
          str(access["p0_token_sha256"]).upper() == sha(EVID / "P0_PASS.json")
          and int(access["p0_declared_protected_reads"]) == 0,
          access["p0_token_sha256"])

    # ---- V3  P0 re-derivation, re-checked against the frozen manifests -------
    hosts = preflight["host_rederivation"]
    check("V3.p0_covered_all_8_hosts", len(hosts) == len(CELLS), len(hosts))
    digests = {}
    for rec in hosts:
        m, h = rec["market"], rec["host"]
        man = load_json(frozen_root(m) / "02_hosts" / m / h / "FREEZE_MANIFEST.json")
        declared = str(man["checkpoint_sha256"]).upper()
        blob = EVID / "host_rederivation" / m / f"{h}.pt"
        ok = (
            str(rec["frozen_checkpoint_sha256"]).upper() == declared
            and str(rec["rederived_checkpoint_sha256"]).upper() == declared
            and bool(rec["checkpoint_identity_exact"]) and bool(rec["prediction_bit_exact"])
            and float(rec["max_abs_open_role_prediction_delta"]) == 0.0
            and int(rec["protected_final_windows_read"]) == 0
            and blob.is_file()
            and sha(blob) == str(rec["rederived_blob_sha256"]).upper()
        )
        check(f"V3.{m}/{h}.host_identity_exact", ok,
              f"manifest {declared} vs {rec['rederived_checkpoint_sha256']}")
        digests[(m, h)] = declared
    check("V3b.p0_build_and_reader_fidelity",
          bool(preflight["environment"]["frozen_build_exact"])
          and all(v["bit_exact_vs_frozen_windows"] for v in preflight["reader_fidelity"].values())
          and all(v["inner_split_admits_fit_block"] for v in preflight["support"].values()),
          sorted(preflight["reader_fidelity"]))
    check("V3c.p0_read_zero_protected_values",
          all(int(v["protected_final_target_values_read"]) == 0
              for v in preflight["support"].values())
          and int(preflight["protected_final_access_audit"]
                  ["protected_final_target_values_read"]) == 0,
          preflight["verdict"])

    # ---- V4  the executor scored the same Host weights -----------------------
    check("V4.every_cell_forwarded_the_frozen_digest",
          all(access["cells_accessing_protected"][f"{m}/{h}"]["host_digest"] == digests[(m, h)]
              for m, h in CELLS)
          and len(access["cells_accessing_protected"]) == len(CELLS),
          sorted(access["cells_accessing_protected"]))

    # ---- V5  protected-day inventory, checked against the frozen freeze -------
    inv = {m: protected_days(m) for m in MARKETS}
    for m in MARKETS:
        rows = cells[cells["market"] == m]
        check(f"V5.{m}.day_count_matches_the_frozen_freeze",
              len(rows) == len(HOSTS)
              and all(int(r["n_protected_days"]) == len(inv[m]) for _, r in rows.iterrows())
              and all(int(r["n_hours"]) == H * len(inv[m]) for _, r in rows.iterrows()),
              f"frozen {len(inv[m])} days")
        check(f"V5b.{m}.access_audit_days_match",
              all(int(access["cells_accessing_protected"][f"{m}/{h}"]["protected_days_predicted"])
                  == len(inv[m]) for h in HOSTS))
    check("V5c.protected_totals_are_consistent",
          int(access["protected_days_predicted_total"])
          == sum(len(inv[m]) * len(HOSTS) for m in MARKETS)
          and int(panel["panel"]["protected_days_total"])
          == sum(int(r["n_protected_days"]) for _, r in cells.iterrows()),
          access["protected_days_predicted_total"])

    # ---- V6  per-seed arithmetic -------------------------------------------
    seed_bad = []
    for _, r in seeds.iterrows():
        want = 100.0 * (float(r["Host_MAE"]) - float(r["Overall_MAE"])) / float(r["Host_MAE"])
        if not close(want, r["gain_pct"]):
            seed_bad.append(f"{r['market']}/{r['host']}/{r['seed']}")
    check("V6.per_seed_gain_arithmetic", not seed_bad, seed_bad)
    check("V6b.seed_contract",
          sorted(seeds["seed"].unique().tolist()) == sorted(SEEDS)
          and len(seeds) == len(CELLS) * len(SEEDS)
          and sorted(seeds["method"].unique().tolist()) == ["HCH_S1_B2"],
          f"{len(seeds)} rows")

    # ---- V7  the cell statistic --------------------------------------------
    med_bad, host_bad, gain_bad = [], [], []
    for _, c in cells.iterrows():
        sl = seeds[(seeds["market"] == c["market"]) & (seeds["host"] == c["host"])]
        if len(sl) != len(SEEDS):
            med_bad.append(f"{c['market']}/{c['host']}:{len(sl)}")
            continue
        if not close(np.median(sl["Overall_MAE"]), c["HCH_MAE_median_of_seeds"]):
            med_bad.append(f"{c['market']}/{c['host']}")
        if not close(np.median(sl["Host_MAE"]), c["Host_MAE"]):
            host_bad.append(f"{c['market']}/{c['host']}")
        if not close(np.median(sl["gain_pct"]), c["HCH_gain_pct_median_of_seeds"]):
            gain_bad.append(f"{c['market']}/{c['host']}")
        want_secondary = (100.0 * (float(c["Host_MAE"]) - float(c["HCH_MAE_median_of_seeds"]))
                          / float(c["Host_MAE"]))
        if not close(want_secondary, c["HCH_gain_of_median_MAE_pct"]):
            gain_bad.append(f"{c['market']}/{c['host']}:secondary")
    check("V7.cell_median_MAE_recomputes", not med_bad, med_bad)
    check("V7b.cell_Host_MAE_recomputes", not host_bad, host_bad)
    check("V7c.cell_gain_definitions_recompute", not gain_bad, gain_bad)

    # ---- V8  the Host array is the same one the baselines were scored on -----
    host_x = []
    for _, c in cells.iterrows():
        bl = base[(base["market"] == c["market"]) & (base["host"] == c["host"])
                  & (base["method"] == "delta-Adapter")]
        if len(bl) != 1 or not close(bl.iloc[0]["Host_Overall_MAE"], c["Host_MAE"]):
            host_x.append(f"{c['market']}/{c['host']}")
    check("V8.baselines_share_the_host_array", not host_x, host_x)

    # ---- V9  ranking -------------------------------------------------------
    rank_bad = []
    for _, c in cells.iterrows():
        cand = {"Host": float(c["Host_MAE"]), "delta-Adapter": float(c["delta_Adapter_MAE"])}
        if bool(c["PIR_legal"]) and not pd.isna(c["PIR_MAE"]):
            cand["PIR"] = float(c["PIR_MAE"])
        want_name = min(cand, key=lambda k: cand[k])
        if want_name != c["best_static"] or not close(cand[want_name], c["best_static_MAE"]):
            rank_bad.append(f"{c['market']}/{c['host']}:{want_name}")
        hch = float(c["HCH_MAE_median_of_seeds"])
        if bool(c["beats_Host"]) != (hch < float(c["Host_MAE"])):
            rank_bad.append(f"{c['market']}/{c['host']}:beats_Host")
        if bool(c["beats_best_static"]) != (hch < cand[want_name]):
            rank_bad.append(f"{c['market']}/{c['host']}:beats_best_static")
        want_gain = 100.0 * (cand[want_name] - hch) / cand[want_name]
        if not close(want_gain, c["gain_vs_best_static_pct"]):
            rank_bad.append(f"{c['market']}/{c['host']}:gain_vs_best_static")
    check("V9.best_static_ranking_recomputes", not rank_bad, rank_bad)
    check("V9b.cosa_and_mdr_are_never_best_static",
          set(cells["best_static"].unique()) <= {"Host", "delta-Adapter", "PIR"},
          sorted(cells["best_static"].unique().tolist()))

    # ---- V10  no-proxy surface --------------------------------------------
    blob = (CONF / "S1B2_CELL_MEDIANS.csv").read_text(encoding="utf-8") \
        + (CONF / "S1B2_BASELINE_RESULTS.csv").read_text(encoding="utf-8") \
        + (CONF / "S1B2_CELL_SEED_RESULTS.csv").read_text(encoding="utf-8")
    leaked = [m for m in FORBIDDEN_MARKETS if m in blob]
    check("V10.forbidden_markets_absent_from_every_result_row", not leaked, leaked)
    pir = base[base["method"] == "PIR"]
    check("V10b.PIR_only_on_its_legal_hosts",
          len(pir) == len(PIR_LEGAL_HOSTS) * len(MARKETS)
          and set(pir["host"].unique()) <= set(PIR_LEGAL_HOSTS),
          f"{len(pir)} rows")
    check("V10c.blockers_unrelaxed",
          int(verdict["proxy_rows"]) == 0
          and verdict["datasets_added"] is False
          and verdict["lago_gefcom_blocker_relaxed"] is False
          and verdict["method_change_after_protected_access"] is False
          and verdict["rescue_applied"] is False
          and int(panel["panel"]["proxy_rows"]) == 0,
          verdict["token"])
    check("V10d.PIR_rows_are_scored_once_each",
          len(base[(base["method"] == "delta-Adapter")]) == len(CELLS)
          and len(base[(base["method"] == "MatchedDirectResidual")]) == len(CELLS)
          and len(base[base["method"] == "COSA"]) == len(CELLS),
          f"{len(base)} baseline rows")

    # ---- V11  COSA chronology ---------------------------------------------
    cosa = base[base["method"] == "COSA"]
    check("V11.cosa_is_online_tta_and_separate",
          len(cosa) == len(CELLS)
          and set(cosa["setting"].unique()) == {"ONLINE_TTA"}
          and not cosa["in_external_ranking"].any(),
          f"{len(cosa)} rows")
    check("V11b.cosa_chronology_passed_everywhere",
          bool(cosa["chronology_audit_passed"].all())
          and int(panel["panel"]["cosa_chronology_failures"]) == 0,
          cosa["chronology_max_abs_diff_over_all_cuts"].tolist())

    # ---- V12  protected read recount ---------------------------------------
    ra = access["read_audit_this_process"]
    recount = {m: int(ra.get(m, {}).get("target_rows_by_role", {})
                      .get("PROTECTED_FINAL", 0)) for m in MARKETS}
    check("V12.protected_rows_parsed_match_the_inventory",
          all(recount[m] == H * len(inv[m]) for m in MARKETS)
          and {k: int(v) for k, v in access["sealed_target_rows_parsed_by_stage_reader"].items()}
          == recount,
          recount)
    check("V12b.no_fit_tuning_or_reselection_on_protected_days",
          int(access["fits_on_protected_days"]) == 0
          and int(access["tuning_on_protected_days"]) == 0
          and int(access["scalar_refits_on_protected_days"]) == 0
          and int(access["checkpoint_changes_on_protected_days"]) == 0
          and access["labelled_online_tta"] is False
          and access["host_forward_is_the_only_protected_source"] is True,
          access["note"][:60])

    # ---- V13  the frozen modules the executor ran are unchanged -------------
    changed = []
    for key, rec in preflight["frozen_modules"].items():
        if key == "reused_contract_layer":
            continue
        p = ROOT / rec["path"]
        cur = sha(p) if p.is_file() else "ABSENT"
        if cur != rec["sha256"]:
            changed.append(key)
    check("V13.frozen_modules_unchanged_since_p0", not changed, changed)
    check("V13b.executor_and_verifier_sources_accounted",
          (STAGE / "implementation/confirmation.py").is_file()
          and (STAGE / "verification/verify_nn.py").is_file(),
          tree_sha(STAGE / "implementation"))

    # ---- V14  continuity is cited, not rerun -------------------------------
    cont = panel["continuity"]
    check("V14.continuity_is_cited_not_rerun",
          cont["rerun_here"] is False and cont["enters_primary_aggregate"] is False
          and (not cont["available"]
               or str(cont["source_sha256"]).upper() == sha(ROOT / cont["source"])),
          cont.get("source", "unavailable"))

    # ---- V15  the authority's section 7 gate, recomputed from the raw numbers -
    gains = cells["HCH_gain_pct_median_of_seeds"].to_numpy(float)
    bs_gains = cells["gain_vs_best_static_pct"].to_numpy(float)
    mkt = {m: float(np.median(gains[cells["market"].to_numpy() == m])) for m in MARKETS}
    hard_fail = (len(cells) != len(CELLS)
                 or int(panel["panel"]["protected_days_total"]) <= 0
                 or len(pir) != len(PIR_LEGAL_HOSTS) * len(MARKETS)
                 or int(panel["panel"]["cosa_chronology_failures"]) > 0
                 or bool(panel["panel"]["failures"]))
    gates = {
        "g1": float(np.median(gains)) > 0.0,
        "g2": int((gains >= -1.0).sum()) >= 6,
        "g3": float(np.median(bs_gains)) >= -1.0,
        "g4": int((bs_gains >= -2.0).sum()) >= 6,
        "g5": min(mkt.values()) >= -3.0,
        "g6": not hard_fail,
        "s1": int((gains > 0).sum()) >= 6,
        "s2": int((bs_gains > 0).sum()) >= 5,
        "s3": float(np.median(bs_gains)) > 0.0,
    }
    if hard_fail:
        want = "HCH_S1_B2_NORD_NEM_INVALID"
    elif gates["g1"] and gates["g2"] and gates["g3"] and gates["g4"] and gates["g5"] and gates["g6"]:
        want = ("HCH_S1_B2_NORD_NEM_STRONG_CONFIRMED"
                if gates["s1"] and gates["s2"] and gates["s3"]
                else "HCH_S1_B2_NORD_NEM_CONFIRMED_FOR_PAPER")
    else:
        want = "HCH_S1_B2_NORD_NEM_MIXED"
    executor_key = {
        "g1": "g1_panel_median_host_gain>0",
        "g2": "g2_host_nonworse_within_-1.0%>=6",
        "g3": "g3_panel_median_gain_vs_best_static>=-1.0%",
        "g4": "g4_within_-2.0%_of_best_static>=6",
        "g5": "g5_worst_market_median>=-3.0%",
        "g6": "g6_no_access_provenance_chronology_failure",
        "s1": "strong:s1_beat_host>=6",
        "s2": "strong:s2_beat_best_static>=5",
        "s3": "strong:s3_panel_median_gain_vs_best_static>0",
    }
    disagree = [k for k, ek in executor_key.items()
                if bool(verdict["gates"].get(ek)) != bool(gates[k])]
    check("V15.gates_recompute", not disagree, disagree or gates)
    check("V15b.terminal_token_recomputes", want == verdict["token"], f"{want} vs {verdict['token']}")
    check("V15c.panel_aggregates_recompute",
          close(np.median(gains), panel["panel"]["panel_median_host_gain_pct"])
          and close(np.median(bs_gains), panel["panel"]["panel_median_gain_vs_best_static_pct"])
          and all(close(mkt[m], panel["panel"]["market_median_host_gain_pct"][m]) for m in MARKETS),
          {"market": mkt})

    # ---- write -------------------------------------------------------------
    doc = {
        "schema": "hch_s1_b2_nord_nem_verification.v1",
        "verifier": "verification/verify_nn.py",
        "verifier_sha256": sha(Path(__file__)),
        "imports_from_executor": False,
        "checks": CHECKS,
        "n_checks": len(CHECKS),
        "n_failed": len(FAILURES),
        "failures": FAILURES,
        "recomputed": {
            "protected_days_per_market": {m: len(inv[m]) for m in MARKETS},
            "protected_days_total": int(access["protected_days_predicted_total"]),
            "panel_median_host_gain_pct": float(np.median(gains)),
            "panel_median_gain_vs_best_static_pct": float(np.median(bs_gains)),
            "market_median_host_gain_pct": mkt,
            "gates": gates,
            "terminal_token": want,
        },
        "verified_artifacts": {
            name: sha(CONF / name) for name in
            ("S1B2_CELL_MEDIANS.csv", "S1B2_CELL_SEED_RESULTS.csv", "S1B2_BASELINE_RESULTS.csv",
             "S1B2_PANEL.json", "PROTECTED_ACCESS_AUDIT.json", "VERDICT.json")
        },
        "p0": {"token_sha256": sha(EVID / "P0_PASS.json"),
               "preflight_sha256": sha(EVID / "P0_SUPPORT_PREFLIGHT.json")},
        "reused_contract_layer_sha256": shared_sha,
        "blocked_predecessor_sha256": sha(PRIOR_BLOCK),
        "evidence_root": str(EVID.relative_to(ROOT)).replace("\\", "/"),
        "mtime_used_as_evidence": False,
    }
    (VERIFY / "VERIFY_NN.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nchecks={len(CHECKS)} failed={len(FAILURES)}")
    print("HCH_S1_B2_NORD_NEM_VERIFY_FAIL" if FAILURES else "HCH_S1_B2_NORD_NEM_VERIFY_PASS")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
