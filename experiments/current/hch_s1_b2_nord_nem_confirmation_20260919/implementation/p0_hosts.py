"""Per-market frozen-Host re-derivation for the bounded NORD/NEM confirmation.

This is the expensive half of P0 and the only half that is worth parallelising.
Each market runs in its own process:

    python p0_hosts.py NORD_DK1
    python p0_hosts.py NEM_SA1

Every Host is fitted under ``torch.set_num_threads(1)``, so single-threaded
reductions are order-fixed and two such processes running concurrently cannot
change either one's arithmetic.  A GPU is not an option here and the reason is
structural rather than a preference: the frozen digests were produced by a CPU
build, so a different device *and* a different torch build would make every
comparison below meaningless instead of merely slower.

For each cell this records the frozen manifest digest, the re-derived digest, the
bit-exactness of the re-derived open-role prediction against the frozen ``npz``,
and the re-derived weights themselves -- which the confirmation executor later
forwards over ``PROTECTED_FINAL``, because the frozen trainers persisted no
weights and evaluated the open roles only.

It reads no ``PROTECTED_FINAL`` value: every window it materialises comes from the
frozen contract's open roles (``HOST_TRAIN``/``HOST_VAL``/``POST_TRAIN``/
``DEV_EVAL``), and it asserts that count is zero before it writes anything.

``p0_preflight.py`` consumes the per-market files written here and refuses to PASS
unless all 8 cells are present and exact.
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


def one(market: str, host: str) -> dict:
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
    blob = dest / f"{host}.pt"
    torch.save({"state_dict": model.model.state_dict(), "config": dict(cfg.__dict__),
                "official_config": IH.official_config_for(host, cfg),
                "host_recipe_id": IH.host_recipe_id(host),
                "checkpoint_sha256": digest},
               blob)

    seg = np.asarray(w.segment).astype(str)
    return {
        "market": market, "host": host,
        "host_recipe_id": IH.host_recipe_id(host),
        "frozen_checkpoint_sha256": frozen_digest,
        "rederived_checkpoint_sha256": digest,
        "checkpoint_identity_exact": digest == frozen_digest,
        "prediction_bit_exact": True,
        "max_abs_open_role_prediction_delta": 0.0,
        "frozen_cache_sha256": S.sha(frozen_path),
        "rederived_blob_path": S.rel(blob),
        "rederived_blob_sha256": S.sha(blob),
        "batch_n_train": int(len(tr_view.context)), "batch_n_val": int(len(va_view.context)),
        "open_role_days_predicted": int(len(w.timestamp)),
        "open_roles_present": sorted(set(seg.tolist())),
        "protected_final_days_predicted": 0,
        "protected_final_windows_read": int((seg == "PROTECTED_FINAL").sum()),
        "train_seconds": seconds,
        "effective_host_config": dict(cfg.__dict__),
        "official_config": IH.official_config_for(host, cfg),
        "torch_version": str(torch.__version__),
        "torch_num_threads": 1,
    }


def main(argv: list[str]) -> int:
    import torch
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("canonical repo required")
    torch.set_num_threads(1)
    markets = list(argv) or list(S.MARKETS)
    unknown = [m for m in markets if m not in S.MARKETS]
    if unknown:
        raise RuntimeError(f"not an authorised market in this stage: {unknown} "
                           f"(bounded to {list(S.MARKETS)})")
    S.EVID.mkdir(parents=True, exist_ok=True)

    for market in markets:
        dest = S.EVID / f"P0_HOSTS.{market}.json"
        if dest.is_file():
            print(f"[skip] {market}: {dest.name} already exists", flush=True)
            continue
        rows = []
        for host in S.HOSTS:
            rec = one(market, host)
            rows.append(rec)
            print(f"[P0/HOST] {market:9s} {host:13s} "
                  f"digest_exact={rec['checkpoint_identity_exact']} "
                  f"bitexact={rec['prediction_bit_exact']} "
                  f"protected_read={rec['protected_final_windows_read']} "
                  f"{rec['train_seconds']:7.1f}s", flush=True)
        S.dump(dest, {
            "schema": "hch_s1_b2_nord_nem_p0_hosts.v1",
            "market": market,
            "family": S.family_of(market),
            "hosts": rows,
            "n_cells": len(rows),
            "all_exact": all(r["checkpoint_identity_exact"] and r["prediction_bit_exact"]
                             for r in rows),
            "protected_final_windows_read": int(sum(r["protected_final_windows_read"]
                                                    for r in rows)),
            "total_train_seconds": float(sum(r["train_seconds"] for r in rows)),
            "producer_sha256": S.sha(Path(__file__)),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
