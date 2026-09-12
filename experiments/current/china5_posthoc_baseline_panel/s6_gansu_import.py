"""Stage 6 — integrity/provenance audit and import of the EXISTING strict GANSU_DA evidence.

No retraining.  The frozen GANSU_DA Host caches and the frozen PIR outputs are
re-hashed, re-checked against their own recorded manifests, and cross-checked
against the published `metrics_by_cell.csv` of the frozen transfer.  Only what
survives the audit enters the raw prediction registry.

Emits 05_gansu_import/GANSU_IMPORT_AUDIT.json and rows for
RAW_PREDICTION_INDEX.csv / EVIDENCE_PROVENANCE_MAP.csv.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
for p in (HERE, ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import panel_contract as pc                                    # noqa: E402
from src.utils.benchmark_cache import HostPredictionCache       # noqa: E402

OUT = ROOT / "experiments/evidence/china5_posthoc_baseline_panel_20260912"
GANSU_HOSTS = ROOT / "experiments/evidence/hch_unified_da_shape_engineering_20260910/host_foundation/GANSU_DA"
TRANSFER = ROOT / "experiments/current/hch_frozen_method_baseline_transfer"
TRANSFER_EVIDENCE = ROOT / "experiments/evidence/hch_frozen_method_baseline_transfer_20260911"
HOSTS = ("PatchTST", "TimeMixer")


def main() -> int:
    audit = {"schema": "gansu_da_import_audit.v1",
             "policy": "integrity/provenance audit only; no retraining performed",
             "hosts": {}, "pir": {}, "cross_check": {}, "verdict": "PENDING"}
    raw_rows, prov_rows = [], []

    published = {}
    pub_csv = TRANSFER_EVIDENCE / "metrics_by_cell.csv"
    with pub_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            published[(row["market"], row["host"], row["method"])] = row

    ok = True
    for name in HOSTS:
        d = GANSU_HOSTS / name
        npz, js = d / "host_predictions.npz", d / "host_predictions.json"
        meta = json.loads(js.read_text(encoding="utf-8"))
        rec = {"array_sha256_recorded": str(meta.get("array_sha256", "")).upper(),
               "array_sha256_recomputed": pc.sha256_file(npz)}
        rec["hash_match"] = rec["array_sha256_recorded"].lower() == rec["array_sha256_recomputed"].lower()
        rec["target_column"] = meta.get("target_column")
        rec["target_compatible"] = meta.get("target_column") == "日前电价"
        rec["old_GANSU_RT_cache_reused"] = meta.get("old_GANSU_RT_cache_reused")
        rec["s3_s4_materialized"] = meta.get("s3_s4_materialized")
        rec["frozen"] = meta.get("frozen")
        c = HostPredictionCache.load(npz)
        segs = sorted(set(str(s) for s in c.segment))
        rec["segments_present"] = segs
        rec["sealed_roles_absent"] = not ({"S3", "S4"} & set(segs))
        rec["shapes"] = {"context": list(c.context.shape), "y_true": list(c.y_true.shape),
                         "host_pred": list(c.host_pred.shape)}
        rec["n_eval_days"] = int((c.segment.astype(str) == "DIAG_EVAL").sum())
        rec["finite"] = bool(np.isfinite(c.host_pred).all())
        rec["passed"] = bool(rec["hash_match"] and rec["target_compatible"]
                             and rec["sealed_roles_absent"] and rec["frozen"]
                             and rec["s3_s4_materialized"] is False and rec["finite"])
        audit["hosts"][name] = rec
        ok &= rec["passed"]
        raw_rows.append({
            "evidence_origin": "GANSU_DA_STRICT_FROZEN_IMPORT",
            "dataset_id": "GANSU_DA", "physical_market_group": "GANSU",
            "backbone": name, "method": "Host", "partition": "DIAG_EVAL",
            "n_days": rec["n_eval_days"], "horizon": 24,
            "array_path": str(npz.relative_to(ROOT)).replace("\\", "/"),
            "array_sha256": rec["array_sha256_recomputed"],
            "recorded_sha256": rec["array_sha256_recorded"],
            "contains_y_true": True, "contains_host_pred": True,
            "imported_by_audit_only": True, "retrained": False})

    for name in HOSTS:
        p = TRANSFER / "_pir_cache" / f"GANSU_DA__{name}"
        z = np.load(p.with_suffix(".npz"), allow_pickle=False)
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        rec = {"fidelity": meta.get("fidelity"),
               "fidelity_ok": meta.get("fidelity") == "PAPER_FAITHFUL_EXACT_ACCEPTED",
               "source": meta.get("source"), "seed": meta.get("seed"),
               "pred_shape": list(z["pred"].shape),
               "n_eval_days": int(z["pred"].shape[0]),
               "finite": bool(np.isfinite(z["pred"]).all()),
               "parameter_count": meta.get("parameter_count")}
        rec["passed"] = bool(rec["fidelity_ok"] and rec["finite"]
                             and rec["n_eval_days"] == audit["hosts"][name]["n_eval_days"])
        audit["pir"][name] = rec
        ok &= rec["passed"]
        raw_rows.append({
            "evidence_origin": "GANSU_DA_STRICT_FROZEN_IMPORT",
            "dataset_id": "GANSU_DA", "physical_market_group": "GANSU",
            "backbone": name, "method": "PIR_paper_protocol", "partition": "DIAG_EVAL",
            "n_days": rec["n_eval_days"], "horizon": 24,
            "array_path": str(p.with_suffix(".npz").relative_to(ROOT)).replace("\\", "/"),
            "array_sha256": pc.sha256_file(p.with_suffix(".npz")),
            "recorded_sha256": "", "contains_y_true": False, "contains_host_pred": False,
            "imported_by_audit_only": True, "retrained": False})

    # Cross-check: re-derive Host and PIR MAE from the frozen arrays and compare
    # with the published frozen metrics.  A mismatch fails the import.
    for name in HOSTS:
        c = HostPredictionCache.load(GANSU_HOSTS / name / "host_predictions.npz")
        ev = c.subset("DIAG_EVAL")
        mae_host = float(np.mean(np.abs(ev.y_true - ev.host_pred)))
        pir = np.load(TRANSFER / "_pir_cache" / f"GANSU_DA__{name}.npz")["pred"]
        mae_pir = float(np.mean(np.abs(ev.y_true[:, :, 0] - pir)))
        got = {}
        for meth, val in (("Host", mae_host), ("PIR_paper_protocol", mae_pir)):
            pub = published.get(("GANSU_DA", name, meth))
            got[meth] = {"rederived": val,
                         "published": float(pub["Overall_MAE"]) if pub and pub.get("Overall_MAE") else None}
            if got[meth]["published"] is not None:
                got[meth]["abs_diff"] = abs(got[meth]["rederived"] - got[meth]["published"])
                got[meth]["match"] = got[meth]["abs_diff"] < 5e-3
                ok &= got[meth]["match"]
        audit["cross_check"][name] = got

    audit["verdict"] = "GANSU_DA_IMPORT_OK_NO_RETRAINING" if ok else "GANSU_DA_IMPORT_FAILED"
    pc.dump_json(OUT / "05_gansu_import" / "GANSU_IMPORT_AUDIT.json", audit)

    prov_rows.append({
        "evidence_origin": "GANSU_DA_STRICT_FROZEN_IMPORT",
        "source_evidence_root": "experiments/evidence/hch_frozen_method_baseline_transfer_20260911",
        "host_cache_root": str(GANSU_HOSTS.relative_to(ROOT)).replace("\\", "/"),
        "role_vocabulary_map": "S1|HOST_TRAIN+HOST_VAL ; DIAG_FIT|POST_TRAIN ; DIAG_EVAL|DEV_EVAL ; S3+S4|PROTECTED_FINAL",
        "import_mode": "integrity/provenance audit only",
        "retrained": False, "fidelity_status_altered": False,
        "verdict": audit["verdict"],
        "note": ("delta_Adapter_AdaY and MatchedDirectResidual raw arrays were never persisted by the "
                 "frozen transfer run; only their published metrics are importable. Reported as "
                 "metrics-only, not silently reconstructed."),
    })
    pc.dump_json(OUT / "05_gansu_import" / "_rows.json", {"raw": raw_rows, "prov": prov_rows})
    print(f"[S6] GANSU_DA import verdict={audit['verdict']}  hosts_ok="
          f"{ {k: v['passed'] for k, v in audit['hosts'].items()} }  "
          f"pir_ok={ {k: v['passed'] for k, v in audit['pir'].items()} }")
    for name in HOSTS:
        print(f"[S6]   cross-check {name}: {audit['cross_check'][name]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
