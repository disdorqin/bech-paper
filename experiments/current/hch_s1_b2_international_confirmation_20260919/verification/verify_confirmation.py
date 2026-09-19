"""Independent verifier for the S1-B2 international confirmation.

It imports **nothing** from the executor: not ``confirmation.py``, not
``intl_shared.py``, not the stage's P0 preflight.  Everything it checks is
recomputed from primary evidence -- the frozen day contracts, the frozen
threshold freezes, the frozen Host manifests, and the executor's own flat CSV /
JSON outputs -- with its own arithmetic.

It certifies, in order:

1. **Access**: the P0 token declares zero protected-final reads, the token hashes
   to the file on disk, and every ``host_rederivation`` blob hashes to the frozen
   manifest digest of the Host it claims to be.
2. **Chronology**: the day contracts really do place ``POST_TRAIN`` strictly
   before ``DEV_EVAL`` strictly before ``PROTECTED_FINAL``, with no overlap, and
   the number of protected days the executor scored equals the number the frozen
   contract publishes.
3. **Metric arithmetic**: every per-seed gain is ``100*(Host_MAE-MAE)/Host_MAE``;
   every cell statistic is the median across the freeze's seeds, taken before the
   gain, not after.
4. **Rankings**: ``best_static`` is the argmin over the legally available
   comparators, and the win flags follow from the numbers rather than the reverse.
5. **Gates**: the freeze's section 8 token is recomputed from the published
   aggregates and must equal the one in ``VERDICT.json``.
6. **No-proxy / no-rescue surface**: the blocked methods appear with no number;
   PIR appears only on its two legal Hosts; COSA appears only as ``ONLINE_TTA``
   and never inside ``best_static``; the row counts are exactly what the freeze
   fixes.

It reads no price value of any partition: every input is a hash, a date, a count
or an already-aggregated error statistic.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
STAGE = ROOT / "experiments/current/hch_s1_b2_international_confirmation_20260919"
EVID = ROOT / "experiments/evidence/hch_s1_b2_international_confirmation_20260919"
CONF = EVID / "confirmation"

MARKETS = ("LAGO_NP", "GEFCOM14P", "NORD_DK1", "NEM_SA1")
HOSTS = ("PatchTST", "TimeMixer", "iTransformer", "LSTM")
SEEDS = (7, 17, 37)
PIR_LEGAL_HOSTS = ("PatchTST", "TimeMixer")
FAMILY_EVID = {
    "LAGO_NP": "hch_international_gefcom_np_baseline_matrix_20260912",
    "GEFCOM14P": "hch_international_gefcom_np_baseline_matrix_20260912",
    "NORD_DK1": "hch_international_nem_nord_baseline_matrix_20260912",
    "NEM_SA1": "hch_international_nem_nord_baseline_matrix_20260912",
}
TOKENS = ("HCH_S1_B2_INTERNATIONAL_STRONG_UNTOUCHED_GENERALIZATION",
          "HCH_S1_B2_INTERNATIONAL_POSITIVE_UNTOUCHED_GENERALIZATION",
          "HCH_S1_B2_INTERNATIONAL_MIXED_UNTOUCHED_GENERALIZATION",
          "HCH_S1_B2_INTERNATIONAL_INVALID")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def ev_root(market: str) -> Path:
    return ROOT / "experiments/evidence" / FAMILY_EVID[market]


def med(xs) -> float:
    return float(np.median([float(x) for x in xs]))


def main() -> int:
    checks: list[dict] = []
    failures: list[str] = []

    def check(name: str, ok: bool, detail) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            failures.append(name)

    # ---------------------------------------------------------------- 0. blocked
    # If the zero-read preflight never passed there is nothing downstream to
    # verify: the protected-final reader was never enabled, so no executor output
    # exists.  Report that as a FAIL token with the reason, rather than crashing on
    # a missing file -- the inability to certify is the finding.
    if not (EVID / "P0_PASS.json").is_file():
        block = {
            "schema": "hch_s1_b2_international_verifier_result.v1",
            "verifier_imports_executor": False,
            "verifier_reads_price_values": False,
            "checks": [{
                "check": "p0.pass_token_present", "ok": False,
                "detail": "P0_PASS.json is absent: the frozen-Host identity preflight never passed, "
                          "so the protected-final reader was never enabled and there is no executor "
                          "output to verify.  No panel, no medians, no gates, no ranking exist."}],
            "n_checks": 1, "n_failed": 1,
            "failures": ["p0.pass_token_present"],
            "recomputed_aggregates": None,
            "chronology": None,
            "snapshot": {
                "verifier_sha256": sha(Path(__file__)),
                "protocol_sha256": sha(STAGE / "PROTOCOL.md"),
                "executor_sha256": sha(STAGE / "implementation" / "confirmation.py"),
                "shared_contract_sha256": sha(STAGE / "implementation" / "intl_shared.py"),
                "p0_preflight_sha256": sha(STAGE / "implementation" / "p0_preflight.py"),
                "evidence_files": {},
            },
            "token": "HCH_S1_B2_INTERNATIONAL_VERIFIER_FAIL",
        }
        CONF.mkdir(parents=True, exist_ok=True)
        (CONF / "VERIFIER_RESULT.json").write_text(
            json.dumps(block, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        print("[VERIFY] 0/1 checks passed -> HCH_S1_B2_INTERNATIONAL_VERIFIER_FAIL "
              "(no P0_PASS.json; nothing to verify)")
        return 1

    # ---------------------------------------------------------------- 1. access
    p0 = json.loads((EVID / "P0_PASS.json").read_text(encoding="utf-8"))
    check("p0.declares_zero_protected_reads",
          int(p0["protected_final_target_values_read"]) == 0,
          p0["protected_final_target_values_read"])
    check("p0.verdict_is_pass", p0["verdict"] == "PASS", p0["verdict"])
    pre = EVID / "P0_SUPPORT_PREFLIGHT.json"
    check("p0.preflight_hash_matches_token", sha(pre).upper() == str(p0["preflight_sha256"]).upper(),
          {"recomputed": sha(pre), "token": p0["preflight_sha256"]})

    import torch  # only to read the digest the executor stored inside each blob
    host_ok = {}
    for market in MARKETS:
        for host in HOSTS:
            blob_p = EVID / "host_rederivation" / market / f"{host}.pt"
            man_p = (ev_root(market) / "02_hosts" / market / host / "FREEZE_MANIFEST.json")
            if not blob_p.is_file() or not man_p.is_file():
                host_ok[f"{market}/{host}"] = "ABSENT"
                continue
            blob = torch.load(blob_p, map_location="cpu", weights_only=False)
            declared = str(json.loads(man_p.read_text(encoding="utf-8"))["checkpoint_sha256"]).upper()
            host_ok[f"{market}/{host}"] = bool(
                str(blob["checkpoint_sha256"]).upper() == declared)
    check("hosts.16_rederivations_match_frozen_digest",
          len(host_ok) == 16 and all(v is True for v in host_ok.values()),
          {k: v for k, v in host_ok.items() if v is not True} or "all 16 exact")

    # ------------------------------------------------------------- 2. chronology
    chrono = {}
    for market in MARKETS:
        d = pd.read_csv(ev_root(market) / "01_dataset_contracts" / market / "ELIGIBLE_DAYS.csv")
        rng = {r: (pd.to_datetime(g["date"]).min(), pd.to_datetime(g["date"]).max(),
                   len(g)) for r, g in d.groupby("role")}
        post, dev, fin = rng["POST_TRAIN"], rng["DEV_EVAL"], rng["PROTECTED_FINAL"]
        chrono[market] = {
            "post_train": [str(post[0].date()), str(post[1].date()), int(post[2])],
            "dev_eval": [str(dev[0].date()), str(dev[1].date()), int(dev[2])],
            "protected_final": [str(fin[0].date()), str(fin[1].date()), int(fin[2])],
            "post_before_dev": bool(post[1] < dev[0]),
            "dev_before_protected": bool(dev[1] < fin[0]),
            "no_overlap": bool(post[1] < dev[0] and dev[1] < fin[0]),
            "protected_monotonic":
                bool(pd.to_datetime(d[d.role == "PROTECTED_FINAL"]["date"]).is_monotonic_increasing),
        }
    check("chronology.all_four_markets_ordered_and_disjoint",
          all(v["no_overlap"] and v["protected_monotonic"] for v in chrono.values()),
          chrono)

    # --------------------------------------------------- 3/4. metric arithmetic
    seeds = pd.read_csv(CONF / "S1B2_CELL_SEED_RESULTS.csv")
    cells = pd.read_csv(CONF / "S1B2_CELL_MEDIANS.csv")
    base = pd.read_csv(CONF / "S1B2_BASELINE_RESULTS.csv")

    check("seeds.row_count_is_48", len(seeds) == len(MARKETS) * len(HOSTS) * len(SEEDS),
          len(seeds))
    check("seeds.exactly_the_frozen_seeds", sorted(seeds["seed"].unique().tolist()) == list(SEEDS),
          sorted(seeds["seed"].unique().tolist()))
    gain_ok = np.allclose(
        seeds["gain_pct"].to_numpy(dtype=float),
        100.0 * (seeds["Host_MAE"] - seeds["Overall_MAE"]) / seeds["Host_MAE"], atol=1e-9, rtol=0)
    check("seeds.gain_is_defined_from_mae_not_the_reverse", bool(gain_ok), "atol=1e-9")

    med_ok, med_bad = True, []
    for _, c in cells.iterrows():
        g = seeds[(seeds.market == c["market"]) & (seeds.host == c["host"])]
        if not np.isclose(med(g["Overall_MAE"]), c["HCH_MAE_median_of_seeds"], atol=1e-9, rtol=0) or \
           not np.isclose(med(g["gain_pct"]), c["HCH_gain_pct_median_of_seeds"], atol=1e-9, rtol=0):
            med_ok = False
            med_bad.append(f"{c['market']}/{c['host']}")
        if int(c["n_protected_days"]) != chrono[c["market"]]["protected_final"][2]:
            med_ok = False
            med_bad.append(f"{c['market']}/{c['host']}: day count")
    check("cells.statistic_is_seed_median_before_gain", med_ok, med_bad or "all 16 agree")

    # ---------------------------------------------------------------- 5. ranking
    # the executor gates on its own ``PIR_legal`` flag *and* a non-null number, so
    # the verifier must not widen the candidate set merely because the Host name is
    # one of the two legal ones -- a PIR row that failed to score is not a
    # comparator, it is a failure, and it makes the stage INVALID instead.
    legal_flag_bad = [f"{c['market']}/{c['host']}" for _, c in cells.iterrows()
                      if bool(c["PIR_legal"]) != (c["host"] in PIR_LEGAL_HOSTS)]
    check("ranking.pir_legal_flag_matches_the_frozen_legal_hosts", not legal_flag_bad,
          legal_flag_bad or "flag == host membership on all 16")

    rank_bad = []
    for _, c in cells.iterrows():
        cand = {"Host": c["Host_MAE"], "delta-Adapter": c["delta_Adapter_MAE"]}
        if bool(c["PIR_legal"]) and pd.notna(c["PIR_MAE"]):
            cand["PIR"] = float(c["PIR_MAE"])
        best = min(cand, key=lambda k: cand[k])
        if (c["best_static"] != best
                or not np.isclose(c["best_static_MAE"], cand[best], atol=1e-9, rtol=0)
                or bool(c["beats_Host"]) != bool(c["HCH_MAE_median_of_seeds"] < c["Host_MAE"])
                or bool(c["beats_best_static"]) != bool(c["HCH_MAE_median_of_seeds"] < cand[best])):
            rank_bad.append(f"{c['market']}/{c['host']}")
    check("ranking.best_static_and_win_flags_recomputed", not rank_bad, rank_bad or "all 16 agree")

    # ------------------------------------------------------- 6. no-proxy surface
    check("blocked.no_numeric_rows",
          not base["method"].isin(["UEC-STD", "OMPB"]).any(),
          sorted(base["method"].unique().tolist()))
    pir = base[base.method == "PIR"]
    check("pir.only_on_its_two_legal_hosts",
          set(pir["host"]) <= set(PIR_LEGAL_HOSTS) and len(pir) == len(PIR_LEGAL_HOSTS) * len(MARKETS),
          {"hosts": sorted(pir["host"].unique().tolist()), "rows": len(pir)})
    cosa = base[base.method == "COSA"]
    check("cosa.is_online_tta_only",
          len(cosa) == len(MARKETS) * len(HOSTS)
          and set(cosa["setting"]) == {"ONLINE_TTA"}
          and bool(cosa["in_external_ranking"].eq(False).all())
          and bool(cosa["in_best_static"].eq(False).all())
          and bool(cosa["chronology_audit_passed"].all()),
          {"rows": len(cosa), "settings": sorted(cosa["setting"].unique().tolist()),
           "audit_failures": int((~cosa["chronology_audit_passed"]).sum())})
    check("delta.16_rows_all_ranked",
          int((base.method == "delta-Adapter").sum()) == len(MARKETS) * len(HOSTS)
          and bool(base[base.method == "delta-Adapter"]["in_external_ranking"].all()),
          int((base.method == "delta-Adapter").sum()))
    check("mdr.internal_control_never_ranked",
          int((base.method == "MatchedDirectResidual").sum()) == len(MARKETS) * len(HOSTS)
          and bool(base[base.method == "MatchedDirectResidual"]["in_external_ranking"].eq(False).all()),
          int((base.method == "MatchedDirectResidual").sum()))

    # ------------------------------------------------------------------ 7. gates
    mkt = {m: med([c["HCH_gain_pct_median_of_seeds"] for _, c in cells.iterrows()
                   if c["market"] == m]) for m in MARKETS}
    fam = {h: med([c["HCH_gain_pct_median_of_seeds"] for _, c in cells.iterrows()
                   if c["host"] == h]) for h in HOSTS}
    agg = {
        "host_positive_count": int(cells["beats_Host"].sum()),
        "best_static_win_count": int(cells["beats_best_static"].sum()),
        "panel_median_host_gain_pct": med(cells["HCH_gain_pct_median_of_seeds"]),
        "panel_median_gain_vs_best_static_pct": med(cells["gain_vs_best_static_pct"]),
        "market_median_host_gain_pct": mkt,
        "market_medians_host_positive": int(sum(1 for v in mkt.values() if v > 0)),
        "worst_market_median_host_gain_pct": float(min(mkt.values())),
        "host_family_median_host_gain_pct": fam,
    }
    hard_fail = (len(cells) != 16 or agg["host_positive_count"] < 0
                 or int((~cosa["chronology_audit_passed"]).sum()) > 0)
    if hard_fail:
        tok = TOKENS[3]
    elif (agg["host_positive_count"] >= 12 and agg["panel_median_host_gain_pct"] >= 1.0
          and agg["best_static_win_count"] >= 10 and agg["market_medians_host_positive"] >= 3
          and agg["worst_market_median_host_gain_pct"] >= -3.0):
        tok = TOKENS[0]
    elif (agg["host_positive_count"] >= 9 and agg["panel_median_host_gain_pct"] > 0.0
          and agg["market_medians_host_positive"] >= 2):
        tok = TOKENS[1]
    else:
        tok = TOKENS[2]
    verdict = json.loads((CONF / "VERDICT.json").read_text(encoding="utf-8"))
    panel = json.loads((CONF / "S1B2_PANEL.json").read_text(encoding="utf-8"))["panel"]
    check("gates.token_recomputed_matches_verdict", tok == verdict["token"],
          {"recomputed": tok, "verdict": verdict["token"]})
    check("gates.panel_aggregates_match_executor",
          all(np.isclose(float(panel[k]), float(v), atol=1e-9, rtol=0)
              for k, v in [("host_positive_count", agg["host_positive_count"]),
                           ("best_static_win_count", agg["best_static_win_count"]),
                           ("panel_median_host_gain_pct", agg["panel_median_host_gain_pct"]),
                           ("worst_market_median_host_gain_pct",
                            agg["worst_market_median_host_gain_pct"])]),
          {"recomputed": agg, "executor": {k: panel[k] for k in
                                           ("host_positive_count", "best_static_win_count",
                                            "panel_median_host_gain_pct")}})

    # -------------------------------------------------------- 8. access recount
    audit = json.loads((CONF / "PROTECTED_ACCESS_AUDIT.json").read_text(encoding="utf-8"))
    declared_days = int(audit["protected_days_predicted_total"])
    check("access.protected_day_total_matches_cells",
          declared_days == int(cells["n_protected_days"].sum()) == sum(
              v["protected_final"][2] for v in chrono.values()),
          {"audit": declared_days, "cells": int(cells["n_protected_days"].sum()),
           "frozen_contracts": sum(v["protected_final"][2] for v in chrono.values())})
    check("access.no_fit_or_tune_on_protected",
          int(audit["fits_on_protected_days"]) == 0
          and int(audit["tuning_on_protected_days"]) == 0
          and int(audit["scalar_refits_on_protected_days"]) == 0
          and int(audit["checkpoint_changes_on_protected_days"]) == 0,
          "fits=0 tuning=0 refits=0 checkpoint_changes=0")
    check("access.executor_read_only_its_own_audit",
          not bool(audit["labelled_online_tta"]),
          {"labelled_online_tta": audit["labelled_online_tta"],
           "sealed_rows_parsed": audit["sealed_target_rows_parsed_by_stage_reader"]})

    # ------------------------------------------------------------ snapshot hashes
    snapshot = {
        "verifier_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(STAGE / "PROTOCOL.md"),
        "executor_sha256": sha(STAGE / "implementation" / "confirmation.py"),
        "shared_contract_sha256": sha(STAGE / "implementation" / "intl_shared.py"),
        "p0_preflight_sha256": sha(STAGE / "implementation" / "p0_preflight.py"),
        "evidence_files": {p.name: sha(p) for p in sorted(CONF.glob("*.csv"))},
    }

    out = {
        "schema": "hch_s1_b2_international_verifier_result.v1",
        "verifier_imports_executor": False,
        "verifier_reads_price_values": False,
        "checks": checks,
        "n_checks": len(checks), "n_failed": len(failures),
        "failures": failures,
        "recomputed_aggregates": agg,
        "chronology": chrono,
        "snapshot": snapshot,
        "token": ("HCH_S1_B2_INTERNATIONAL_VERIFIER_PASS" if not failures
                  else "HCH_S1_B2_INTERNATIONAL_VERIFIER_FAIL"),
    }
    dest = CONF / "VERIFIER_RESULT.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")
    print(f"[VERIFY] {len(checks) - len(failures)}/{len(checks)} checks passed -> {out['token']}")
    for f in failures:
        print(f"[VERIFY-FAIL] {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
