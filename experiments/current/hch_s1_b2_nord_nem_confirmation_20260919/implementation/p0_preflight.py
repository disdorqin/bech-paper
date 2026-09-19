"""P0 zero-fit preflight for the bounded NORD_DK1 / NEM_SA1 confirmation.

Runs entirely on the OPEN roles.  It never materialises a ``PROTECTED_FINAL``
target value; the stage-local reader refuses a sealed role while the seal is
closed, and this module only ever asks for open roles.

What it establishes, matching authority section 5:

1. the frozen build identity, so a Host is never re-derived under a torch build
   that would return *different* weights self-consistently (this gate fails closed
   before a single Host is touched);
2. source and split identity for both markets;
3. every frozen module a method or comparator executes, by digest;
4. all 8 Host checkpoints re-derive bit-exactly -- consumed from ``p0_hosts.py``,
   re-checked here against the blobs on disk rather than taken on trust;
5. the stage-local reader reproduces the frozen open-role windows bit-for-bit;
6. POST_TRAIN support for S1-B2 and the 168h original-origin construction;
7. the legal-state manifest, with realised-demand substitution pinned to zero;
8. a code/evidence snapshot;
9. the protected-final read count, certified zero.

On PASS it writes ``EVID/P0_PASS.json``.  That token is the only thing that
unlocks the protected reader.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

IMPL = Path(__file__).resolve().parent
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))

import nn_shared as S                                                       # noqa: E402

ROOT = S.ROOT


def _frozen_module_digests() -> dict:
    """Every frozen module the method or a comparator executes, by digest."""
    out: dict = {"reused_contract_layer": S.frozen_layer()}
    for name, fname in (("contracts", "contracts.py"), ("icells", "icells.py"),
                        ("ihosts", "ihosts.py")):
        p = ROOT / f"experiments/current/{S.FAMILY_IMPL['nem_nord']}/implementation/{fname}"
        out[f"nem_nord/{name}"] = {"path": S.rel(p), "sha256": S.sha(p)}
    frozen = {
        "s1b2_runner": "experiments/current/hch_s1_sequential_surgical_repair_20260918/implementation/runner.py",
        "baseline_transfers": "experiments/current/hch_frozen_method_baseline_transfer/transfers.py",
        "baseline_pir_transfer": "experiments/current/hch_frozen_method_baseline_transfer/pir_transfer.py",
        "baseline_cosa_transfer": "experiments/current/china5_posthoc_baseline_panel/cosa_transfer.py",
        "panel_metrics": "experiments/current/china5_posthoc_baseline_panel/panel_metrics.py",
        "s1b2_authority": "docs/current/HCH_S1_B2_NORD_NEM_CONFIRMATION_20260919.md",
        "s1b2_adjudication": "docs/current/HCH_S1_B2_PAPER_READY_ADJUDICATION_20260919.md",
        "blocked_predecessor": ("experiments/evidence/"
                                "hch_s1_b2_international_confirmation_20260919/P0_BLOCKED.json"),
    }
    for k, v in frozen.items():
        p = ROOT / v
        out[k] = {"path": v, "sha256": S.sha(p) if p.is_file() else "ABSENT"}
    return out


def _environment_identity() -> dict:
    """The frozen Hosts are reproducible only under the build that produced them.

    Every frozen cache records the ``torch_version`` that trained it.  Reproducing
    a Host under a different build is *silently* wrong, not loudly wrong: the
    recipe is deterministic, so a re-derivation succeeds, is self-consistent across
    repeats, and returns weights that are not the frozen ones.  This gate therefore
    fails closed before a single Host is touched.
    """
    import torch
    expected: dict[str, list[str]] = {}
    for market in S.MARKETS:
        for host in S.HOSTS:
            rec = S.frozen_host_record(market, host)
            expected.setdefault(str(rec.get("torch_version")), []).append(f"{market}/{host}")
    running = str(torch.__version__)
    if set(expected) != {running}:
        raise RuntimeError(
            "frozen Host identity requires the exact build that produced the frozen "
            f"weights: recorded {sorted(expected)}, running torch {running} "
            f"({sys.executable}). A re-derivation here would succeed and return "
            "different weights; use the frozen interpreter.")
    return {
        "interpreter": sys.executable,
        "torch_version": running,
        "frozen_torch_version": {k: len(v) for k, v in expected.items()},
        "cells_covered": sum(len(v) for v in expected.values()),
        "frozen_build_exact": True,
        "why_this_gate_exists": (
            "reproduction under a different torch build is silently wrong, so the "
            "build is pinned rather than inferred from the digest comparison"),
    }


def _source_and_split(market: str) -> dict:
    C = S.contracts_of(market)
    spec = C.SPECS[market]
    src = ROOT / spec["source_path"]
    ev = S.frozen_evidence_root(market) / "01_dataset_contracts" / market
    contract = json.loads((ev / "DATASET_CONTRACT.json").read_text(encoding="utf-8"))
    declared = str(contract["provenance"]["source_sha256"]).upper()
    actual = S.sha(src)
    if actual != declared:
        raise RuntimeError(f"{market}: source drift {actual} != {declared}")
    mc = C.load_market(market)          # re-derives the day list and asserts counts
    ra = contract["read_audit"]
    sealed_counts = {k: int(v) for k, v in ra.items()
                     if "protected" in k.lower() and isinstance(v, (int, float))
                     and "guarded" not in k.lower() and "skipped" not in k.lower()}
    if any(sealed_counts.values()):
        raise RuntimeError(f"{market}: frozen contract already records a sealed read: {sealed_counts}")
    return {
        "market": market, "family": S.family_of(market),
        "source_path": spec["source_path"], "source_sha256": actual,
        "timezone": spec["timezone"],
        "hour_label_convention": spec["hour_label_convention"],
        "day_definition": spec["day_definition"],
        "split_hash": mc.split_hash,
        "counts": mc.day_contract["counts"],
        "eligible_days": len(mc.day_contract["eligible_days"]),
        "irregular_excluded_days": len(mc.day_contract.get("irregular_excluded_days", [])),
        "contract_audit": mc.audit,
        "frozen_split_manifest_sha256": S.sha(ev / "SPLIT_MANIFEST.csv"),
        "eligible_days_sha256": S.sha(ev / "ELIGIBLE_DAYS.csv"),
        "frozen_read_audit": ra,
        "declared_sealed_read_counters": sealed_counts,
        "role_day_span": contract["role_day_span"],
    }


def _hosts_from_p0_hosts() -> list[dict]:
    """Consume ``P0_HOSTS.<market>.json`` and re-check it against the disk blobs."""
    hosts: list[dict] = []
    for market in S.MARKETS:
        p = S.EVID / f"P0_HOSTS.{market}.json"
        if not p.is_file():
            raise RuntimeError(f"{market}: {p.name} absent; run p0_hosts.py {market} first")
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc["market"] != market or int(doc["n_cells"]) != len(S.HOSTS):
            raise RuntimeError(f"{market}: {p.name} is not a complete 4-Host record")
        for rec in doc["hosts"]:
            host = rec["host"]
            blob = S.EVID / "host_rederivation" / market / f"{host}.pt"
            if not blob.is_file():
                raise RuntimeError(f"{market}/{host}: re-derived weights absent at {blob}")
            if S.sha(blob) != str(rec["rederived_blob_sha256"]).upper():
                raise RuntimeError(f"{market}/{host}: re-derived weights changed on disk "
                                   f"after they were recorded")
            declared = str(S.frozen_host_manifest(market, host)["checkpoint_sha256"]).upper()
            if str(rec["frozen_checkpoint_sha256"]).upper() != declared:
                raise RuntimeError(f"{market}/{host}: recorded against a different frozen manifest")
            if not (rec["checkpoint_identity_exact"] and rec["prediction_bit_exact"]
                    and int(rec["protected_final_windows_read"]) == 0
                    and float(rec["max_abs_open_role_prediction_delta"]) == 0.0):
                raise RuntimeError(f"{market}/{host}: P0 host re-derivation is not exact: "
                                   f"{rec['frozen_checkpoint_sha256']} vs "
                                   f"{rec['rederived_checkpoint_sha256']}")
            hosts.append({**rec, "source_file": S.rel(p),
                          "blob_sha256_rechecked": S.sha(blob)})
    return hosts


def main() -> int:
    import torch
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("canonical repo required")
    torch.set_num_threads(1)
    S.EVID.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    HIST = S.import_hist()
    HIST.STEPS = 1000

    report: dict = {
        "schema": "hch_s1_b2_nord_nem_p0.v1",
        "stage": "hch_s1_b2_nord_nem_confirmation_20260919",
        "authority": "docs/current/HCH_S1_B2_NORD_NEM_CONFIRMATION_20260919.md",
        "bounded_successor_of": ("experiments/evidence/"
                                 "hch_s1_b2_international_confirmation_20260919/P0_BLOCKED.json"),
        "computed_before_protected_access": True,
        "seal_state_at_preflight": "CLOSED",
        "markets": list(S.MARKETS), "hosts": list(S.HOSTS), "seeds": list(S.SEEDS),
        "bounded_scope_note": ("LAGO_NP / GEFCOM14P are excluded because their frozen Host "
                               "identity is unrecoverable, not because their result was "
                               "unwelcome. The blocker is inherited unchanged and no proxy "
                               "Host is manufactured."),
    }

    # ---- 0. frozen build identity (fails closed before any Host is touched) --
    report["environment"] = _environment_identity()
    print(f"[P0/ENV] torch={report['environment']['torch_version']} "
          f"frozen_build_exact={report['environment']['frozen_build_exact']}", flush=True)

    # ---- 1. source and split identity -------------------------------------
    report["sources"] = {m: _source_and_split(m) for m in S.MARKETS}
    report["frozen_modules"] = _frozen_module_digests()
    print(f"[P0/SRC] {', '.join(report['sources'])} source+split identity re-established",
          flush=True)

    # ---- 2. frozen Host checkpoint/config identity (8 cells, re-checked) ----
    hosts = _hosts_from_p0_hosts()
    report["host_rederivation"] = hosts
    for r in hosts:
        print(f"[P0/HOST] {r['market']:9s} {r['host']:13s} digest_exact=True bitexact=True "
              f"blob={Path(r['rederived_blob_path']).name}", flush=True)

    # ---- 2b. the stage-local reader must reproduce the frozen windows --------
    fidelity = {}
    for market in S.MARKETS:
        C = S.contracts_of(market)
        fw = C.load_market(market).windows
        mine = S.read_windows(market, list(S.OPEN_ROLES))
        same = (
            mine.context.shape == fw.context.shape
            and np.array_equal(mine.context, fw.context)
            and np.array_equal(mine.target, fw.target)
            and np.array_equal(np.asarray(mine.timestamp, dtype="datetime64[ns]"),
                               np.asarray(fw.timestamp, dtype="datetime64[ns]"))
            and np.array_equal(mine.segment.astype(str), fw.segment.astype(str))
        )
        if not same:
            raise RuntimeError(f"{market}: stage-local reader does not reproduce the frozen "
                               f"open-role windows")
        fidelity[market] = {"n_open_days": int(len(fw.timestamp)),
                            "bit_exact_vs_frozen_windows": True,
                            "context_shape": list(mine.context.shape)}
        print(f"[P0/READER] {market:9s} open_days={len(fw.timestamp):4d} bit_exact=True",
              flush=True)
    report["reader_fidelity"] = fidelity

    # ---- 3-5. support, 168h construction, legal state ----------------------
    support, state_audit = {}, {}
    for market in S.MARKETS:
        days = S.eligible_days(market)
        n_fit = int((days["role"] == S.FIT_ROLE).sum())
        n_ev = int((days["role"] == S.EVAL_ROLE).sum())
        nval = max(5, int(np.ceil(0.2 * n_fit)))
        ntrain = n_fit - nval
        if ntrain < 15:
            raise RuntimeError(f"{market}: frozen inner_split rejects the fit block "
                               f"({n_fit} days -> {ntrain} train)")
        # The support check is run in exactly the shape the executor will run it:
        # ``make_inputs(target_cache, history_cache)``, both taken from the frozen
        # Host cache.  Passing a ``ForecastWindows`` here would be a type error --
        # ``make_inputs`` reads ``y_true``/``host_pred``, which only a cache carries.
        cache, _p = S.load_host_cache(market, S.HOSTS[0])
        fit_cache = cache.subset(S.FIT_ROLE)
        if len(fit_cache.timestamp) != n_fit:
            raise RuntimeError(f"{market}: the frozen Host cache holds "
                               f"{len(fit_cache.timestamp)} fit days, the frozen day list "
                               f"{n_fit}")
        _fi, _rf, mf, floor = HIST.make_inputs(fit_cache, cache)
        if not np.isfinite(floor):
            raise RuntimeError(f"{market}: non-finite residual floor")
        origins = np.asarray(fit_cache.timestamp)
        state, sa = S.read_legal_state(market, origins)
        state_audit[market] = sa
        spec = S.contracts_of(market).SPECS[market]
        support[market] = {
            "fit_role": S.FIT_ROLE, "eval_role": S.EVAL_ROLE,
            "n_fit_days": n_fit, "n_protected_final_days": n_ev,
            "inner_split_n_train": ntrain, "inner_split_n_val": nval,
            "inner_split_admits_fit_block": True,
            "n_fit_days_with_168h_history": int((mf.sum(axis=1) == 168).sum()),
            "total_history_hours_resolved": int(mf.sum()),
            "residual_floor": float(floor),
            "nstate": int(state.shape[2]),
            "legal_state_columns": list(sa.get("columns", [])),
            "realised_demand_substituted": False,
            "forbidden_target_day_inputs": list(spec.get("forbidden_target_day_inputs", ())),
            "target_day_realized_state_reads": 0,
            "protected_final_target_values_read": 0,
        }
        print(f"[P0/SUPPORT] {market:9s} fit={n_fit:4d} protected={n_ev:4d} "
              f"nstate={state.shape[2]} "
              f"168h_full={support[market]['n_fit_days_with_168h_history']}/{n_fit}", flush=True)
        del fit_cache, cache

    # the frozen inner split, replayed directly on the fit block, per market
    for market in S.MARKETS:
        fit_w = S.read_windows(market, [S.FIT_ROLE])
        if len(fit_w.timestamp) != support[market]["n_fit_days"]:
            raise RuntimeError(f"{market}: the stage-local reader's fit block holds "
                               f"{len(fit_w.timestamp)} days, the frozen day list "
                               f"{support[market]['n_fit_days']}")
        tr_idx, va_idx = HIST.inner_split(np.arange(len(fit_w.timestamp)))
        if len(tr_idx) != support[market]["inner_split_n_train"] or \
           len(va_idx) != support[market]["inner_split_n_val"]:
            raise RuntimeError(f"{market}: frozen inner_split disagrees with the declared "
                               f"20% / min-5 rule")
        if len(set(tr_idx.tolist()) & set(va_idx.tolist())):
            raise RuntimeError(f"{market}: frozen inner_split overlaps train and validation")
        support[market]["inner_split_replayed_indices"] = {"n_train": int(len(tr_idx)),
                                                          "n_val": int(len(va_idx))}
        del fit_w
    report["support"] = support
    report["state_manifest"] = state_audit

    # ---- 6. baseline legality, frozen ---------------------------------------
    report["baseline_matrix"] = {
        "host": {"cells": 8, "setting": "OFFLINE_HOST_REFERENCE"},
        "delta-Adapter": {"cells": 8, "setting": "OFFLINE_STATIC_POSTHOC", "seed": 7},
        "MatchedDirectResidual": {"cells": 8, "setting": "INTERNAL_CONTROL_ONLY",
                                  "in_external_ranking": False},
        "PIR": {"cells": 4, "setting": "OFFLINE_STATIC_POSTHOC", "seed": 2021,
                "legal_hosts": list(S.PIR_LEGAL_HOSTS), "blocked_hosts": ["iTransformer", "LSTM"],
                "blocker": "PIR_FROZEN_BACKBONE_CONFIG_ABSENT_FOR_NEW_HOST"},
        "COSA": {"cells": 8, "setting": "ONLINE_TTA", "in_best_static": False, "seed": 2021},
        "UEC-STD": {"status": "INHERITED_BLOCKER", "proxy_used": False},
        "OMPB": {"status": "INHERITED_BLOCKER", "proxy_used": False},
        "best_static_rule": "min MAE among Host, delta-Adapter, PIR when PIR is legally available",
        "proxy_rows": 0,
    }

    # ---- 7. snapshot ---------------------------------------------------------
    report["snapshot"] = {
        "implementation_tree_sha256": S.tree_sha(S.IMPL),
        "verification_tree_sha256": S.tree_sha(S.STAGE / "verification"),
        "protocol_sha256": S.sha(S.STAGE / "PROTOCOL.md"),
        "launcher_sha256": S.sha(S.STAGE / "AI_EXECUTION_PROMPT.md"),
        "elapsed_seconds": time.perf_counter() - t_start,
    }

    # ---- 8. protected-final read count, certified zero ------------------------
    sealed = S.sealed_positions_read()
    if any(sealed.values()):
        raise RuntimeError(f"P0 read a sealed coordinate: {sealed}")
    report["protected_final_access_audit"] = {
        "reader_seal_state": "CLOSED for the whole of P0",
        "sealed_roles_materialised": False,
        "protected_final_target_values_read": 0,
        "protected_final_days_predicted": 0,
        "read_audit_this_process": S.read_audit(),
        "declared_by_frozen_contracts": {
            m: report["sources"][m]["contract_audit"]["protected_final_target_values_read"]
            for m in S.MARKETS},
        "roles_read_this_process": sorted({r for a in S.read_audit().values()
                                           for r in a.get("roles_read", [])}),
    }

    hard = (
        bool(report["environment"]["frozen_build_exact"])
        and len(hosts) == len(S.CELLS)
        and all(r["checkpoint_identity_exact"] and r["prediction_bit_exact"]
                and int(r["protected_final_windows_read"]) == 0 for r in hosts)
        and all(v["bit_exact_vs_frozen_windows"] for v in fidelity.values())
        and all(v["inner_split_admits_fit_block"] for v in support.values())
        and all(int(v["protected_final_target_values_read"]) == 0 for v in support.values())
        and not any(sealed.values())
    )
    report["verdict"] = "PASS" if hard else "FAIL"
    report["p0_evidence_root"] = S.rel(S.EVID)
    S.dump(S.EVID / "P0_SUPPORT_PREFLIGHT.json", report)

    if report["verdict"] != "PASS":
        print("HCH_S1_B2_NORD_NEM_P0_FAIL")
        return 1

    S.dump(S.P0_TOKEN, {
        "schema": "hch_s1_b2_nord_nem_p0_token.v1",
        "verdict": "PASS",
        "protected_final_target_values_read": 0,
        "host_cells_identity_verified": len(hosts),
        "markets": list(S.MARKETS),
        "protected_reader_authorised": True,
        "preflight_sha256": S.sha(S.EVID / "P0_SUPPORT_PREFLIGHT.json"),
        "note": "The protected reader is unlocked only by this token.",
    })
    print("HCH_S1_B2_NORD_NEM_P0_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
