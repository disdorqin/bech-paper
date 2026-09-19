"""P0 block diagnostic -- complete frozen-Host re-derivation identity table.

This is **diagnostic only**: it is not the preflight, it gates nothing, and it
never enables the protected-final reader.  ``p0_preflight.py`` raises at the
first non-identical Host, so it can only ever report the first failure; this
script re-derives all 16 Hosts and records, per cell, the frozen digest, the
re-derived digest, and the numerical distance between the re-derived and the
frozen *open-role* predictions.

It reads no price value of ``PROTECTED_FINAL``: every window it materialises and
every prediction it compares comes from the open roles the frozen family
evidence publishes (``HOST_TRAIN``/``HOST_VAL``/``POST_TRAIN``/``DEV_EVAL``).

Results are appended to ``P0_BLOCK_DIAGNOSTIC.jsonl`` one cell at a time so a
partial run is still usable evidence.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
STAGE = ROOT / "experiments/current/hch_s1_b2_international_confirmation_20260919"
EVID = ROOT / "experiments/evidence/hch_s1_b2_international_confirmation_20260919"
sys.path.insert(0, str(STAGE / "implementation"))

import intl_shared as S  # noqa: E402

MARKETS = ("LAGO_NP", "GEFCOM14P", "NORD_DK1", "NEM_SA1")
HOSTS = ("PatchTST", "TimeMixer", "iTransformer", "LSTM")


def one(market: str, host: str) -> dict:
    IH = S.ihosts_of(market)
    C = S.contracts_of(market)
    man = S.frozen_host_manifest(market, host)
    cache, _ = S.load_host_cache(market, host)
    mc = C.load_market(market)
    tr, va = C.host_fit_parts(mc)
    w = mc.windows
    frozen_pred = np.asarray(cache.host_pred)

    torch.set_num_threads(1)
    cfg = IH.host_cfg(host)
    t0 = time.perf_counter()
    with IH.vendor_bindings():
        saved = list(sys.path)
        sys.path[:] = [p for p in sys.path if not (p and (Path(p) / "models.py").exists())]
        try:
            model = (IH.HB.make_new_host(host, cfg) if host in ("iTransformer", "LSTM")
                     else IH._make_official_direct(host, cfg))
            model.fit(tr, va)
            model.freeze()
            pred = np.asarray(model.predict(w))
        finally:
            sys.path[:] = saved
    digest = IH.state_dict_sha(model)
    frozen_digest = str(man["checkpoint_sha256"]).upper()
    seg = np.asarray(getattr(w, "segment", getattr(w, "role", np.array(["?"] * len(pred))))).astype(str)
    rec = {
        "market": market, "host": host, "family": S.family_of(market),
        "frozen_checkpoint_sha256": frozen_digest,
        "rederived_checkpoint_sha256": digest,
        "checkpoint_identity_exact": digest == frozen_digest,
        "n_windows": int(pred.shape[0]),
        "roles_present": sorted(set(seg.tolist())),
        "protected_final_windows_read": int((seg == "PROTECTED_FINAL").sum()),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    if pred.shape == frozen_pred.shape:
        d = np.abs(pred.astype(np.float64) - frozen_pred.astype(np.float64))
        rec["prediction_bit_exact"] = bool(np.array_equal(pred, frozen_pred))
        rec["max_abs_prediction_delta"] = float(d.max())
        rec["median_abs_prediction_delta"] = float(np.median(d))
        rec["frozen_prediction_scale_mean_abs"] = float(np.abs(frozen_pred.astype(np.float64)).mean())
    else:
        rec["prediction_bit_exact"] = False
        rec["max_abs_prediction_delta"] = None
        rec["shape_mismatch"] = [list(pred.shape), list(frozen_pred.shape)]
    return rec


def main(argv: list[str]) -> int:
    """With no argument, walk all 16 cells into one JSONL.

    With ``MARKET/HOST`` arguments, run only those cells and write a per-market
    JSONL instead.  Each cell pins ``torch.set_num_threads(1)``, so several such
    processes may run concurrently without changing any cell's bits -- parallel
    processes are the only legitimate way to speed this up.  (A GPU would change
    both the device and the torch build, so every digest comparison against the
    CPU-frozen manifests would be meaningless rather than merely slower.)
    """
    if argv:
        specs = [tuple(a.split("/", 1)) for a in argv]
        # name the file after the first cell too, so two processes covering
        # different cells of the same market never append to one file
        dest = EVID / f"P0_BLOCK_DIAGNOSTIC.{specs[0][0]}.{specs[0][1]}.jsonl"
    else:
        specs = [(m, h) for m in MARKETS for h in HOSTS]
        dest = EVID / "P0_BLOCK_DIAGNOSTIC.jsonl"
    done = set()
    if dest.is_file():
        for line in dest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["market"], r["host"]))
    for market, host in specs:
        if (market, host) in done:
            print(f"[skip] {market}/{host}", flush=True)
            continue
        try:
            rec = one(market, host)
        except Exception as exc:  # a failure is itself the evidence
            rec = {"market": market, "host": host, "family": S.family_of(market),
                   "error": f"{type(exc).__name__}: {exc}"}
        with dest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"[DIAG] {market}/{host} exact={rec.get('checkpoint_identity_exact')} "
              f"maxabs={rec.get('max_abs_prediction_delta')} {rec.get('seconds','-')}s",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
