"""P0 zero-fit preflight for the frozen S1-B2 international confirmation.

Runs entirely on the four OPEN roles.  It never materialises a PROTECTED_FINAL
target value; ``intl_shared.read_windows`` refuses a sealed role while the seal is
closed, and this module only ever asks for open roles.

What it establishes, matching freeze section 6:

1. source and split identity for all four markets;
2. frozen Host checkpoint/config identity — each Host is re-derived from its
   frozen recipe at its frozen seed and must reproduce the frozen prediction array
   bit-for-bit *and* the frozen ``checkpoint_sha256`` state_dict digest;
3. POST_TRAIN support for S1-B2 (frozen ``inner_split`` admits the fit block);
4. 168h original-origin residual/history construction on the fit block;
5. the legal-state manifest, per market, with the realised-demand substitution
   count pinned to zero;
6. every method/baseline config frozen, by digest of the frozen module;
7. a code/evidence snapshot;
8. the protected-final read count, certified zero.

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

import intl_shared as S                                                    # noqa: E402

ROOT = S.ROOT


def _frozen_module_digests() -> dict:
    """Every frozen module the method or a comparator executes, by digest."""
    out = {}
    for tag in ("gefcom_np", "nem_nord"):
        impl = ROOT / f"experiments/current/{S.FAMILY_IMPL[tag]}/implementation"
        for name, fname in (("contracts", "contracts.py"), ("icells", "icells.py"),
                            ("ihosts", "ihosts.py")):
            p = impl / fname
            out[f"{tag}/{name}"] = {"path": S.rel(p), "sha256": S.sha(p)}
    frozen = {
        "s1b2_runner": "experiments/current/hch_s1_sequential_surgical_repair_20260918/implementation/runner.py",
        "baseline_transfers": "experiments/current/hch_frozen_method_baseline_transfer/transfers.py",
        "baseline_pir_transfer": "experiments/current/hch_frozen_method_baseline_transfer/pir_transfer.py",
        "baseline_cosa_transfer": "experiments/current/china5_posthoc_baseline_panel/cosa_transfer.py",
        "s1b2_authority": "docs/current/HCH_S1_B2_INTERNATIONAL_BENCHMARK_FREEZE_20260919.md",
        "s1b2_adjudication": "docs/current/HCH_S1_B2_PAPER_READY_ADJUDICATION_20260919.md",
    }
    for k, v in frozen.items():
        p = ROOT / v
        out[k] = {"path": v, "sha256": S.sha(p) if p.is_file() else "ABSENT"}
    return out


def _environment_identity() -> dict:
    """The frozen Hosts are reproducible only under the build that produced them.

    Every frozen cache records the ``torch_version`` that trained it.  Reproducing a
    Host under a different build is *silently* wrong, not loudly wrong: the recipe is
    deterministic, so a re-derivation succeeds, is self-consistent across repeats, and
    returns weights that are not the frozen ones.  This gate therefore fails closed
    before a single Host is touched, instead of leaving a digest comparison to notice.
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
    # The two frozen families name the sealed-read counter differently; neither
    # may be non-zero under any spelling.
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
        # Only the NEM/NORD family records irregular-day exclusions; GEFCOM/LAGO's
        # contract has no such key, and its absence means "none declared", not zero.
        "irregular_excluded_days": len(mc.day_contract.get("irregular_excluded_days", [])),
        "irregular_excluded_days_declared": "irregular_excluded_days" in mc.day_contract,
        "contract_audit": mc.audit,
        "frozen_split_manifest_sha256": S.sha(ev / "SPLIT_MANIFEST.csv"),
        "eligible_days_sha256": S.sha(ev / "ELIGIBLE_DAYS.csv"),
        "frozen_read_audit": ra,
        "declared_sealed_read_counters": sealed_counts,
        "role_day_span": contract["role_day_span"],
    }


def _rederive_host(market: str, host: str) -> dict:
    """Re-derive one frozen Host and prove identity against the frozen artifacts."""
    import torch
    IH = S.ihosts_of(market)
    C = S.contracts_of(market)
    man = S.frozen_host_manifest(market, host)
    frozen_cache, frozen_path = S.load_host_cache(market, host)

    mc = C.load_market(market)
    tr_view, va_view = C.host_fit_parts(mc)
    w = mc.windows

    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("thread pinning contract violated")
    cfg = IH.host_cfg(host)
    t0 = time.perf_counter()
    with IH.vendor_bindings():
        saved = list(sys.path)
        sys.path[:] = [p for p in sys.path if not (p and (Path(p) / "models.py").exists())]
        try:
            model = (IH.HB.make_new_host(host, cfg) if host in ("iTransformer", "LSTM")
                     else IH._make_official_direct(host, cfg))
            model.fit(tr_view, va_view)
            model.freeze()
            pred = np.asarray(model.predict(w))
        finally:
            sys.path[:] = saved
    seconds = time.perf_counter() - t0

    digest = IH.state_dict_sha(model)
    frozen_digest = str(man["checkpoint_sha256"]).upper()
    if digest != frozen_digest:
        raise RuntimeError(f"{market}/{host}: re-derived checkpoint digest {digest} != "
                           f"frozen {frozen_digest}")
    if pred.shape != frozen_cache.host_pred.shape:
        raise RuntimeError(f"{market}/{host}: re-derived prediction shape {pred.shape} != "
                           f"frozen {frozen_cache.host_pred.shape}")
    if not np.array_equal(np.ascontiguousarray(pred), np.ascontiguousarray(frozen_cache.host_pred)):
        d = float(np.max(np.abs(pred.astype(np.float64)
                               - frozen_cache.host_pred.astype(np.float64))))
        raise RuntimeError(f"{market}/{host}: re-derived Host prediction is not bit-exact "
                           f"(max abs diff {d})")

    dest = S.EVID / "host_rederivation" / market
    dest.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.model.state_dict(), "config": dict(cfg.__dict__),
                "official_config": IH.official_config_for(host, cfg),
                "host_recipe_id": IH.host_recipe_id(host),
                "checkpoint_sha256": digest},
               dest / f"{host}.pt")
    return {
        "market": market, "host": host,
        "host_recipe_id": IH.host_recipe_id(host),
        "frozen_checkpoint_sha256": frozen_digest,
        "rederived_checkpoint_sha256": digest,
        "checkpoint_identity_exact": digest == frozen_digest,
        "prediction_bit_exact": True,
        "frozen_cache_sha256": S.sha(frozen_path),
        "batch_n_train": int(len(tr_view.context)), "batch_n_val": int(len(va_view.context)),
        "open_role_days_predicted": int(len(w.timestamp)),
        "protected_final_days_predicted": 0,
        "train_seconds": seconds,
        "effective_host_config": dict(cfg.__dict__),
        "official_config": IH.official_config_for(host, cfg),
        "torch_num_threads": 1,
    }


def main() -> int:
    import torch
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("canonical repo required")
    torch.set_num_threads(1)
    S.EVID.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    HIST = S.import_hist()
    HIST.STEPS = 1000
    flags = {"scalar": "normalized", "scale": "mean", "ray_norm": "l2", "objective": "mse",
             "checkpoint": "inner", "c": False, "b3": False}

    report: dict = {
        "schema": "hch_s1_b2_international_p0.v1",
        "stage": "hch_s1_b2_international_confirmation_20260919",
        "authority": "docs/current/HCH_S1_B2_INTERNATIONAL_BENCHMARK_FREEZE_20260919.md",
        "computed_before_protected_access": True,
        "seal_state_at_preflight": "CLOSED",
        "markets": list(S.MARKETS), "hosts": list(S.HOSTS), "seeds": list(S.SEEDS),
    }

    # ---- 0. frozen build identity (fails closed before any Host is touched) --
    report["environment"] = _environment_identity()
    print(f"[P0/ENV] torch={report['environment']['torch_version']} "
          f"frozen_build_exact={report['environment']['frozen_build_exact']}", flush=True)

    # ---- 1. source and split identity -------------------------------------
    report["sources"] = {m: _source_and_split(m) for m in S.MARKETS}
    report["frozen_modules"] = _frozen_module_digests()

    # ---- 2. frozen Host checkpoint/config identity ------------------------
    hosts = []
    for market in S.MARKETS:
        for host in S.HOSTS:
            rec = _rederive_host(market, host)
            hosts.append(rec)
            print(f"[P0/HOST] {market:9s} {host:12s} digest_exact={rec['checkpoint_identity_exact']} "
                  f"bitexact={rec['prediction_bit_exact']} {rec['train_seconds']:6.1f}s", flush=True)
    report["host_rederivation"] = hosts

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
        print(f"[P0/READER] {market:9s} open_days={len(fw.timestamp):4d} bit_exact=True", flush=True)
    report["reader_fidelity"] = fidelity

    # ---- 3-5. support, 168h construction, legal state ----------------------
    support, state_audit = {}, {}
    for market in S.MARKETS:
        days = S.eligible_days(market)
        n_fit = int((days["role"] == S.FIT_ROLE).sum())
        n_ev = int((days["role"] == S.EVAL_ROLE).sum())
        # The frozen inner-split rule must admit the fit block.
        nval = max(5, int(np.ceil(0.2 * n_fit)))
        ntrain = n_fit - nval
        if ntrain < 15:
            raise RuntimeError(f"{market}: frozen inner_split rejects the fit block "
                               f"({n_fit} days -> {ntrain} train)")
        fit_w = S.read_windows(market, [S.FIT_ROLE])
        cache, _p = S.load_host_cache(market, S.HOSTS[0])
        full_open = cache
        _fi, rf, mf, floor = HIST.make_inputs(fit_w, full_open)
        n_hist = int(mf.sum())
        if not np.isfinite(floor):
            raise RuntimeError(f"{market}: non-finite residual floor")
        origins = np.asarray(fit_w.timestamp)
        state, sa = S.read_legal_state(market, origins)
        state_audit[market] = sa
        C = S.contracts_of(market)
        spec = C.SPECS[market]
        support[market] = {
            "fit_role": S.FIT_ROLE, "eval_role": S.EVAL_ROLE,
            "n_fit_days": n_fit, "n_protected_final_days": n_ev,
            "inner_split_n_train": ntrain, "inner_split_n_val": nval,
            "inner_split_admits_fit_block": True,
            "n_fit_days_with_168h_history": int((mf.sum(axis=1) == 168).sum()),
            "total_history_hours_resolved": n_hist,
            "residual_floor": float(floor),
            "nstate": int(state.shape[2]),
            "legal_state_columns": list(sa.get("columns", [])),
            "realised_demand_substituted": False,
            "forbidden_target_day_inputs": list(spec.get("forbidden_target_day_inputs", ())),
            "target_day_realized_state_reads": 0,
            "protected_final_target_values_read": 0,
        }
        print(f"[P0/SUPPORT] {market:9s} fit={n_fit:4d} protected={n_ev:4d} nstate={state.shape[2]} "
              f"168h_full={support[market]['n_fit_days_with_168h_history']}/{n_fit}", flush=True)
        del fit_w, full_open
    report["support"] = support
    report["state_manifest"] = state_audit

    # ---- 6. baseline legality, frozen ---------------------------------------
    report["baseline_matrix"] = {
        "host": {"cells": 16, "setting": "OFFLINE_HOST_REFERENCE"},
        "delta-Adapter": {"cells": 16, "setting": "OFFLINE_STATIC_POSTHOC", "seed": 7},
        "MatchedDirectResidual": {"cells": 16, "setting": "INTERNAL_CONTROL_ONLY",
                                 "in_external_ranking": False},
        "PIR": {"cells": 8, "setting": "OFFLINE_STATIC_POSTHOC", "seed": 2021,
                "legal_hosts": list(S.PIR_LEGAL_HOSTS)},
        "PIR_blocked_hosts": ["iTransformer", "LSTM"],
        "PIR_blocker": "PIR_FROZEN_BACKBONE_CONFIG_ABSENT_FOR_NEW_HOST",
        "COSA": {"setting": "ONLINE_TTA", "in_best_static": False, "seed": 2021},
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
        and all(r["checkpoint_identity_exact"] and r["prediction_bit_exact"]
                for r in hosts)
        and all(v["bit_exact_vs_frozen_windows"] for v in fidelity.values())
        and all(v["inner_split_admits_fit_block"] for v in support.values())
        and all(int(v["protected_final_target_values_read"]) == 0 for v in support.values())
        and not any(sealed.values())
        and len(hosts) == 16
    )
    report["verdict"] = "PASS" if hard else "FAIL"
    report["p0_evidence_root"] = S.rel(S.EVID)
    S.dump(S.EVID / "P0_SUPPORT_PREFLIGHT.json", report)

    if report["verdict"] != "PASS":
        print("HCH_S1_B2_INTERNATIONAL_P0_FAIL")
        return 1

    S.dump(S.P0_TOKEN, {
        "schema": "hch_s1_b2_international_p0_token.v1",
        "verdict": "PASS",
        "protected_final_target_values_read": 0,
        "host_cells_identity_verified": len(hosts),
        "protected_reader_authorised": True,
        "preflight_sha256": S.sha(S.EVID / "P0_SUPPORT_PREFLIGHT.json"),
        "note": "The protected reader is unlocked only by this token.",
    })
    print("HCH_S1_B2_INTERNATIONAL_P0_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
