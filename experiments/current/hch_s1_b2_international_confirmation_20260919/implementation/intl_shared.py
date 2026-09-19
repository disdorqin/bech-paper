"""Stage-local contract layer for the frozen S1-B2 international confirmation.

This module owns three things and nothing else:

1. **Family loading.**  The four confirmation markets come from two frozen
   baseline stages, each with its own ``contracts.py`` / ``icells.py`` /
   ``ihosts.py`` under the module name ``contracts``/``icells``/``ihosts``.  They
   are loaded one family at a time with ``sys.modules`` isolated, so each family's
   globals bind its *own* ``contracts`` and neither can silently serve the other.

2. **The stage-local reader.**  ``read_windows`` materialises forecast windows for
   a requested role set.  It reproduces the frozen stage's projection discipline
   exactly (timestamp-projected row selection, no imputation, no value-based
   repair) but is *not* restricted to the open roles.  Reading a sealed role
   requires ``unlock_protected()``, which refuses unless a P0 PASS token exists
   on disk.  Before that token exists every protected coordinate raises.

3. **The legal-state manifest.**  ``read_legal_state`` returns the target-day
   state the freeze admits, and only that: the two ``legal_forecast_known_columns``
   of GEFCOM14P / LAGO_NP, and nothing for NORD_DK1 / NEM_SA1.  It never
   substitutes realised demand and never reads a forbidden column.

Nothing here fits, tunes, selects or rescales anything.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
STAGE = ROOT / "experiments/current/hch_s1_b2_international_confirmation_20260919"
EVID = ROOT / "experiments/evidence/hch_s1_b2_international_confirmation_20260919"
IMPL = STAGE / "implementation"

# The frozen families import ``utils.benchmark_cache`` by bare name, so the repo's
# ``src`` must be importable before any family module is touched.  Several family
# ``load_family`` windows restore ``sys.path`` verbatim on exit, so this has to be
# established once, at import, and never be assumed to survive a family load.
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

H = 24
SEQ = 168
SEEDS = (7, 17, 37)

MARKETS = ("LAGO_NP", "GEFCOM14P", "NORD_DK1", "NEM_SA1")
HOSTS = ("PatchTST", "TimeMixer", "iTransformer", "LSTM")
CELLS = tuple((m, h) for m in MARKETS for h in HOSTS)

# PIR is legal only where the frozen ``pir_transfer.BACKBONE_CONFIG`` has an entry.
PIR_LEGAL_HOSTS = ("PatchTST", "TimeMixer")
# Inherited blockers.  Never re-probed, never proxied.
BLOCKED_METHODS = ("UEC-STD", "OMPB")

FIT_ROLE = "POST_TRAIN"
EVAL_ROLE = "PROTECTED_FINAL"
OPEN_ROLES = ("HOST_TRAIN", "HOST_VAL", "POST_TRAIN", "DEV_EVAL")
CLOSED_ROLES = ("PROTECTED_FINAL",)
STAGE_ROLES = OPEN_ROLES + CLOSED_ROLES

# Which frozen parent stage owns each market's contract and Host evidence.
FAMILIES = {
    "LAGO_NP": ("hch_international_gefcom_np_baseline_matrix", "gefcom_np"),
    "GEFCOM14P": ("hch_international_gefcom_np_baseline_matrix", "gefcom_np"),
    "NORD_DK1": ("hch_international_nem_nord_baseline_matrix", "nem_nord"),
    "NEM_SA1": ("hch_international_nem_nord_baseline_matrix", "nem_nord"),
}
# The current/ implementation dirs carry no date suffix; the evidence roots do.
FAMILY_IMPL = {
    "gefcom_np": "hch_international_gefcom_np_baseline_matrix",
    "nem_nord": "hch_international_nem_nord_baseline_matrix",
}
FAMILY_EVID = {
    "gefcom_np": "hch_international_gefcom_np_baseline_matrix_20260912",
    "nem_nord": "hch_international_nem_nord_baseline_matrix_20260912",
}

P0_TOKEN = EVID / "P0_PASS.json"


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


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json) + "\n",
                    encoding="utf-8")


def _json(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def ts_column(spec: dict) -> str:
    """The gefcom family declares ``timestamp_column``; nem_nord hardcodes it."""
    return str(spec.get("timestamp_column", "timestamp"))


def read_kwargs(C, market: str) -> dict:
    """The family's own CSV kwargs, so neither family's reader is re-typed here."""
    fn = getattr(C, "_read_kwargs", None)
    return dict(fn(market)) if callable(fn) else {}


# --------------------------------------------------------------------- families
def _load_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_FAMILY_CACHE: dict[str, tuple] = {}
_COLLIDING = ("contracts", "icells", "ihosts")


def load_family(tag: str) -> tuple:
    """Load one family's frozen modules under an isolated ``sys.modules`` window."""
    if tag in _FAMILY_CACHE:
        return _FAMILY_CACHE[tag]
    impl = ROOT / f"experiments/current/{FAMILY_IMPL[tag]}/implementation"
    if not impl.is_dir():
        raise RuntimeError(f"{tag}: frozen implementation dir absent at {impl}")
    saved_path = list(sys.path)
    saved_mods = {k: sys.modules.get(k) for k in _COLLIDING}
    for k in _COLLIDING:
        sys.modules.pop(k, None)
    sys.path.insert(0, str(impl))
    try:
        mods = {}
        for name in _COLLIDING:
            mods[name] = _load_file(impl / f"{name}.py", f"intl_{tag}_{name}")
            sys.modules[name] = mods[name]
        # ``icells``/``ihosts`` are loaded after their ``contracts`` is registered,
        # so each binds its own family's contract module.
        mods["contracts"] = sys.modules["contracts"]
        mods["icells"] = sys.modules["icells"]
        mods["ihosts"] = sys.modules["ihosts"]
    finally:
        sys.path[:] = saved_path
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    _FAMILY_CACHE[tag] = (mods["contracts"], mods["icells"], mods["ihosts"])
    return _FAMILY_CACHE[tag]


def family_of(market: str) -> str:
    return FAMILIES[market][1]


def contracts_of(market: str):
    return load_family(family_of(market))[0]


def icells_of(market: str):
    return load_family(family_of(market))[1]


def ihosts_of(market: str):
    return load_family(family_of(market))[2]


def frozen_evidence_root(market: str) -> Path:
    return ROOT / f"experiments/evidence/{FAMILY_EVID[family_of(market)]}"


# ---------------------------------------------------------------- P0 unlock gate
_UNLOCKED = False


def protected_unlocked() -> bool:
    return _UNLOCKED


def unlock_protected() -> dict:
    """Enable sealed-role reads.  Refuses without a P0 PASS token on disk."""
    global _UNLOCKED
    if not P0_TOKEN.is_file():
        raise RuntimeError("protected reader: P0 PASS token absent; the seal stays closed")
    tok = json.loads(P0_TOKEN.read_text(encoding="utf-8"))
    if tok.get("verdict") != "PASS" or int(tok.get("protected_final_target_values_read", -1)) != 0:
        raise RuntimeError(f"protected reader: P0 token is not a clean PASS: {tok.get('verdict')!r}")
    _UNLOCKED = True
    return tok


def assert_locked_reading_open_only(roles) -> None:
    if set(roles) & set(CLOSED_ROLES) and not _UNLOCKED:
        raise RuntimeError("PROTECTED_FINAL coordinate requested while the seal is closed")


# ------------------------------------------------------------------ day inventory
def eligible_days(market: str) -> pd.DataFrame:
    """The frozen eligible-day list — dates, roles and row positions only.

    Day *dates* and row *positions* are structural: the frozen contract already
    publishes them.  No target value is read here.
    """
    p = frozen_evidence_root(market) / "01_dataset_contracts" / market / "ELIGIBLE_DAYS.csv"
    df = pd.read_csv(p)
    need = {"market", "role", "date", "start_row", "context_start_row"}
    if not need <= set(df.columns):
        raise RuntimeError(f"{market}: ELIGIBLE_DAYS.csv lacks {sorted(need - set(df.columns))}")
    if set(df["role"]) - set(STAGE_ROLES):
        raise RuntimeError(f"{market}: unknown role in the frozen day list")
    return df


# ------------------------------------------------------------- stage-local reader
_READ_AUDIT: dict[str, dict] = {}


def read_windows(market: str, roles) -> object:
    """Materialise forecast windows for `roles` straight from the frozen source.

    Reproduces the frozen stage's construction: context is the 168 rows strictly
    before the day's first row, target is the day's own 24 rows, both read through
    a row projection.  No imputation, no gap filling, no value-based repair.
    """
    roles = tuple(roles)
    assert_locked_reading_open_only(roles)
    C = contracts_of(market)
    spec = C.SPECS[market]
    days = eligible_days(market)
    sel = days[days["role"].isin(roles)]
    if not len(sel):
        raise RuntimeError(f"{market}: no days for roles {roles}")

    target_pos: set[int] = set()
    context_pos: set[int] = set()
    for _, r in sel.iterrows():
        s = int(r["start_row"])
        target_pos.update(range(s, s + H))
        context_pos.update(range(s - SEQ, s))
    # A later day's context legitimately contains an earlier day's target rows —
    # that is observed history.  What must never happen is a day's own target
    # entering its own context.  Same rule, and same per-day form, as the frozen
    # ``load_market``.
    for _, r in sel.iterrows():
        s = int(r["start_row"])
        if set(range(s, s + H)) & set(range(s - SEQ, s)):
            raise RuntimeError(f"{market}: day {r['date']} target rows enter its own context")

    # The full timestamp column is read (one column) so window stamps come from
    # global row positions, exactly as the frozen reader derives them.  The two
    # frozen families do not share a private projection helper signature, so the
    # projection is written here rather than borrowed.
    src = ROOT / spec["source_path"]
    tcol, pcol = ts_column(spec), spec["price_column"]
    kw = read_kwargs(C, market)
    all_ts = pd.read_csv(src, usecols=[tcol], **kw)[tcol]
    all_ts = pd.to_datetime(all_ts, errors="coerce")
    keep = set(target_pos | context_pos)
    if max(keep) >= len(all_ts):
        raise RuntimeError(f"{market}: row position beyond the source")
    skip = [i + 1 for i in range(len(all_ts)) if i not in keep]
    df = pd.read_csv(src, usecols=[tcol, pcol], skiprows=skip, **kw)
    if len(df) != len(keep):
        raise RuntimeError(f"{market}: projected read returned {len(df)} rows, "
                           f"expected {len(keep)}")
    rts = pd.to_datetime(df[tcol], errors="raise")
    rpx = pd.to_numeric(df[pcol], errors="raise")
    if not np.isfinite(rpx.to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"{market}: non-finite price in the stage-local read")
    mp = {pd.Timestamp(t): float(v) for t, v in zip(rts, rpx)}
    if len(mp) != len(rts):
        raise RuntimeError(f"{market}: projected read collapsed")

    ctx, y, stamps, segs = [], [], [], []
    for _, r in sel.iterrows():
        s = int(r["start_row"])
        hts = [pd.Timestamp(all_ts.iloc[s]) - pd.Timedelta(hours=k) for k in range(SEQ, 0, -1)]
        tts = [pd.Timestamp(all_ts.iloc[s]) + pd.Timedelta(hours=k) for k in range(H)]
        if any(t not in mp for t in hts + tts):
            raise RuntimeError(f"{market}: projected read incomplete for {r['date']}")
        ctx.append(np.asarray([mp[t] for t in hts], np.float32)[:, None])
        y.append(np.asarray([mp[t] for t in tts], np.float32)[:, None])
        stamps.append(str(all_ts.iloc[s]))
        segs.append(str(r["role"]))

    from utils.benchmark_contracts import ForecastWindows
    w = ForecastWindows(np.stack(ctx), np.stack(y), np.asarray(stamps), np.asarray(segs),
                        spec["dataset_id"], spec["physical_market_group"])

    aud = _READ_AUDIT.setdefault(market, {"roles_read": set(), "target_rows": 0, "context_rows": 0})
    aud["roles_read"] |= set(roles)
    aud["target_rows"] += len(target_pos)
    aud["context_rows"] += len(context_pos)
    aud["target_rows_by_role"] = {
        **aud.get("target_rows_by_role", {}),
        **{r: int((sel["role"] == r).sum()) * H for r in roles},
    }
    aud["_positions"] = sorted(set(aud.get("_positions", [])) | set(keep))
    return w


def read_audit() -> dict:
    out = {}
    for m, a in _READ_AUDIT.items():
        out[m] = {k: (sorted(v) if isinstance(v, set) else v) for k, v in a.items()
                  if k != "_positions"}
    return out


def sealed_positions_read() -> dict[str, int]:
    """How many PROTECTED_FINAL target rows this process has actually parsed."""
    out = {}
    for m in MARKETS:
        a = _READ_AUDIT.get(m, {})
        out[m] = int(a.get("target_rows_by_role", {}).get("PROTECTED_FINAL", 0))
    return out


# ------------------------------------------------------------------- legal state
def read_legal_state(market: str, origins) -> tuple[np.ndarray, dict]:
    """The manifest-gated target-day state.  Empty where the manifest admits none."""
    C = contracts_of(market)
    spec = C.SPECS[market]
    cols = list(spec.get("legal_forecast_known_columns", ()) or ())
    origins = pd.to_datetime(pd.Series(list(origins)))
    base = {"market": market, "manifest_gated": True, "nstate": len(cols), "columns": cols,
            "realised_demand_substituted": False, "forbidden_column_reads": 0,
            "target_day_price_reads": 0}
    if not cols:
        z = np.zeros((len(origins), H, 0), dtype=np.float32)
        return z, {**base, "rows_read": 0,
                   "reader": "no legally forecast-known target-day state (freeze section 3)"}

    src = ROOT / spec["source_path"]
    tcol = ts_column(spec)
    forbidden = set(spec.get("forbidden_target_day_inputs", ())) - set(cols)
    if set(cols) & forbidden:
        raise RuntimeError(f"{market}: legal state column overlaps forbidden inputs: "
                           f"{sorted(set(cols) & forbidden)}")
    wanted = {pd.Timestamp(o) + pd.Timedelta(hours=h) for o in origins for h in range(H)}
    all_ts = pd.read_csv(src, usecols=[tcol], **read_kwargs(C, market))[tcol]
    all_ts = pd.to_datetime(all_ts, errors="coerce")
    pos = np.flatnonzero(all_ts.isin(wanted).to_numpy())
    if len(pos) != len(wanted):
        raise RuntimeError(f"{market}: legal-state projection incomplete "
                           f"({len(pos)} of {len(wanted)})")
    skip = [i + 1 for i in range(len(all_ts)) if i not in set(pos.tolist())]
    df = pd.read_csv(src, usecols=[tcol] + cols, skiprows=skip, **read_kwargs(C, market))
    df[tcol] = pd.to_datetime(df[tcol], errors="raise")
    df = df.set_index(tcol).sort_index()
    if set(df.index) != wanted:
        raise RuntimeError(f"{market}: legal-state reader escaped its projection")
    arr = df[cols].to_numpy(np.float64)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{market}: non-finite legal state")
    z = np.stack([[df.loc[pd.Timestamp(o) + pd.Timedelta(hours=h), cols].to_numpy(np.float32)
                   for h in range(H)] for o in origins])
    return z, {**base, "rows_read": int(len(df)), "source_path": spec["source_path"],
               "source_sha256": sha(src), "reader": "stage-local projected legal-state reader"}


# ---------------------------------------------------------------- host identities
def frozen_host_manifest(market: str, host: str) -> dict:
    p = frozen_evidence_root(market) / "02_hosts" / market / host / "FREEZE_MANIFEST.json"
    if not p.is_file():
        raise RuntimeError(f"{market}/{host}: frozen Host manifest absent at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def frozen_host_record(market: str, host: str) -> dict:
    """The frozen Host's full provenance sidecar (``host_predictions.json``).

    ``FREEZE_MANIFEST.json`` carries the identity hashes; this sidecar additionally
    carries the build that produced them (``torch_version``), which is what makes a
    frozen Host reproducible or not.
    """
    p = frozen_evidence_root(market) / "02_hosts" / market / host / "host_predictions.json"
    if not p.is_file():
        raise RuntimeError(f"{market}/{host}: frozen Host provenance sidecar absent at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def host_cache_path(market: str, host: str) -> Path:
    return frozen_evidence_root(market) / "02_hosts" / market / host / "host_predictions.npz"


def load_host_cache(market: str, host: str):
    from utils.benchmark_cache import HostPredictionCache
    p = host_cache_path(market, host)
    cache = HostPredictionCache.load(p)
    if set(cache.segment.astype(str)) - set(OPEN_ROLES):
        raise RuntimeError(f"{market}/{host}: frozen cache exposes a non-open role")
    return cache, p


def frozen_contract(market: str):
    """The frozen ``MarketContract`` — open roles only, by construction."""
    return contracts_of(market).load_market(market)


# ------------------------------------------------------------------- cell builder
def relabelled_with(cache, mapping: dict):
    """``icells.relabelled`` under a caller-supplied role map.

    The frozen map sends ``PROTECTED_FINAL`` to the legacy name ``S4``, which the
    frozen ``panel.rebuild_series`` does not select.  The sliding-window baselines
    reconstruct their series from ``DIAG_FIT``/``DIAG_EVAL`` only, so extending
    their evaluation span onto the protected days requires saying so explicitly
    here rather than by editing the frozen map.
    """
    from utils.benchmark_cache import HostPredictionCache
    seg = np.array([mapping[str(s)] for s in cache.segment], dtype=object)
    return HostPredictionCache(
        cache.market_id, cache.physical_market_group, cache.backbone,
        cache.timestamp, cache.context, cache.y_true, cache.host_pred, seg,
        dict(cache.metadata),
    )


def build_cell(market: str, host: str, cache, ev_role: str, relabel: dict | None = None) -> dict:
    """The frozen cell contract of ``icells.build_cell``, with a chosen scoring role.

    Body reproduced from the frozen builder; the fit role stays ``POST_TRAIN`` and
    only the scoring role changes.  ``relabel`` defaults to the frozen
    ``icells.LEGACY_ROLE`` and is only ever overridden where a downstream frozen
    consumer enumerates its series by legacy role name.
    """
    C = contracts_of(market)
    CL = icells_of(market)
    if market not in MARKETS:
        raise KeyError(market)
    if host not in HOSTS:
        raise KeyError(host)
    mc = C.load_market(market)
    if str(cache.metadata.get("split_hash")) != mc.split_hash:
        raise RuntimeError(f"{market}/{host}: cache built under a different split")
    if str(cache.metadata.get("source_sha256", "")).upper() != mc.source_sha256:
        raise RuntimeError(f"{market}/{host}: cache built from a different source")

    fit, ev = cache.subset(FIT_ROLE), cache.subset(ev_role)
    if len(fit.timestamp) == 0 or len(ev.timestamp) == 0:
        raise RuntimeError(f"{market}/{host}: empty fit or scoring block")
    fi, floor = CL._make_inputs()(fit, cache)
    ei, _ = CL._make_inputs()(ev, cache, floor)
    scale = np.asarray(fi.scale, dtype=np.float64)
    ef = ((fit.y_true[:, :, 0] - fit.host_pred[:, :, 0])
          / np.where(np.isfinite(scale), scale, 1.0)[:, None])
    role_map = dict(CL.LEGACY_ROLE) if relabel is None else dict(relabel)
    runs, span = CL._panel().rebuild_series(relabelled_with(cache, role_map))
    return {
        "market": mc.dataset_id, "market_key": market, "host": host, "family": mc.family,
        "cache": cache, "cache_origin": "STAGE_LOCAL_PROTECTED_EXTENSION",
        "fit": fit, "ev": ev, "fi": fi, "ei": ei, "ef": ef,
        "hp": ev.host_pred[:, :, 0].astype(np.float64),
        "scale": np.nan_to_num(np.asarray(ei.scale, dtype=np.float64), nan=0.0)[:, None],
        "avail": np.asarray(ei.repair_available),
        "n_fit": int(len(fi.x)), "n_eval": int(len(ei.x)),
        "runs": runs, "span": span, "gap_breaks": int(span["n_gap_breaks"]),
        "host_metadata": dict(cache.metadata),
        "fit_role": FIT_ROLE, "eval_role": ev_role,
        "role_relabelling": role_map,
        "role_relabelling_is_frozen_map": relabel is None,
        "day_convention": C.SPECS[market]["hour_label_convention"],
        "day_definition": C.SPECS[market]["day_definition"],
        "split_hash": mc.split_hash, "source_sha256": mc.source_sha256,
        "market_family": mc.family,
    }


# ------------------------------------------------- protected-span Host forward pass
def rederived_host_path(market: str, host: str) -> Path:
    return EVID / "host_rederivation" / market / f"{host}.pt"


def load_rederived_host(market: str, host: str):
    """Rebuild one frozen Host from the weights P0 proved digest-identical.

    The frozen trainers persisted no weights and evaluated only the open roles, so
    the protected-span Host prediction exists nowhere on disk.  It is produced from
    *these* weights -- never from a freshly fitted model -- because the Host is a
    frozen comparator, not something this stage is allowed to re-fit.
    """
    import torch
    IH = ihosts_of(market)
    p = rederived_host_path(market, host)
    if not p.is_file():
        raise RuntimeError(f"{market}/{host}: re-derived frozen Host absent at {p}; "
                           f"P0 must pass before any protected-span Host exists")
    blob = torch.load(p, map_location="cpu", weights_only=False)
    frozen_digest = str(blob["checkpoint_sha256"]).upper()
    declared = str(frozen_host_manifest(market, host)["checkpoint_sha256"]).upper()
    if frozen_digest != declared:
        raise RuntimeError(f"{market}/{host}: re-derived Host digest {frozen_digest} != "
                           f"frozen manifest {declared}")
    cfg = IH.host_cfg(host)
    saved_path = list(sys.path)
    with IH.vendor_bindings():
        sys.path[:] = [x for x in sys.path if not (x and (Path(x) / "models.py").exists())]
        try:
            model = (IH.HB.make_new_host(host, cfg) if host in ("iTransformer", "LSTM")
                     else IH._make_official_direct(host, cfg))
            model.model.load_state_dict(blob["state_dict"])
            model.freeze()
        finally:
            sys.path[:] = saved_path
    if IH.state_dict_sha(model) != frozen_digest:
        raise RuntimeError(f"{market}/{host}: loaded weights do not hash to the frozen digest")
    return model, IH


def predict_role(market: str, host: str, role: str) -> tuple[object, np.ndarray, str]:
    """Forward the frozen Host over ``role`` and return (windows, pred, digest)."""
    model, IH = load_rederived_host(market, host)
    w = read_windows(market, [role])
    saved_path = list(sys.path)
    with IH.vendor_bindings():
        sys.path[:] = [x for x in sys.path if not (x and (Path(x) / "models.py").exists())]
        try:
            pred = np.asarray(model.predict(w), dtype=frozen_dtype(market, host))
        finally:
            sys.path[:] = saved_path
    if pred.shape != w.target.shape or not np.isfinite(pred).all():
        raise RuntimeError(f"{market}/{host}: protected-span Host prediction is mis-shaped "
                           f"or non-finite ({pred.shape} vs {w.target.shape})")
    return w, pred, IH.state_dict_sha(model)


def frozen_dtype(market: str, host: str) -> str:
    """The frozen cache's own prediction dtype, so the composite never re-casts."""
    _cache, p = load_host_cache(market, host)
    with np.load(p, allow_pickle=False) as z:
        return str(z["host_pred"].dtype)


def compose_cache(open_cache, sealed_windows, sealed_pred):
    """Frozen open-role rows followed by the protected-span rows, in date order.

    ``make_inputs`` resolves a protected day's 168h residual history from whatever
    rows the cache holds, so the sealed days must sit in the same series as the open
    ones: a protected day's history legitimately reaches back into DEV_EVAL, and a
    later protected day's history reaches back into earlier protected days (freeze
    section 4).  Nothing here is fitted on.
    """
    from utils.benchmark_cache import HostPredictionCache
    seg = np.asarray(sealed_windows.segment).astype(str)
    if set(seg) != {EVAL_ROLE}:
        raise RuntimeError(f"sealed block carries unexpected roles: {sorted(set(seg))}")
    ts = np.concatenate([np.asarray(open_cache.timestamp), np.asarray(sealed_windows.timestamp)])
    if not pd.to_datetime(ts).is_monotonic_increasing:
        raise RuntimeError("composed cache is not chronological")
    meta = dict(open_cache.metadata)
    meta["sealed_rows_source"] = "REDERIVED_FROZEN_HOST_FORWARD"
    meta["protected_final_target_values_read"] = int(sealed_windows.target.size)
    return HostPredictionCache(
        open_cache.market_id, open_cache.physical_market_group, open_cache.backbone,
        ts,
        np.concatenate([np.asarray(open_cache.context), np.asarray(sealed_windows.context)]),
        np.concatenate([np.asarray(open_cache.y_true), np.asarray(sealed_windows.target)]),
        np.concatenate([np.asarray(open_cache.host_pred), np.asarray(sealed_pred)]),
        np.concatenate([np.asarray(open_cache.segment).astype(str), seg]).astype(object),
        meta,
    )


# ------------------------------------------------------------------ S1-B2 binding
def import_hist():
    """The frozen S1-B2 runner, imported by path and never re-typed."""
    p = ROOT / "experiments/current/hch_s1_sequential_surgical_repair_20260918/implementation/runner.py"
    mod = _load_file(p, "intl_hist_s1_runner")
    mod.EVID = EVID
    return mod


def make_s1b2_cell(HIST, market: str, host: str, full_cache, state_fit, state_ev,
                   cache_path: Path, ev_role: str = EVAL_ROLE):
    """Bind one market x Host to the frozen ``CellData`` contract.

    ``full_cache`` carries every role including PROTECTED_FINAL, so the 168h
    residual history of a protected day may draw on already-revealed earlier
    protected days (freeze section 4) while nothing is fitted on them.  The fit role
    stays the frozen ``POST_TRAIN``; only ``ev_role`` moves.
    """
    from types import SimpleNamespace
    fit, ev = full_cache.subset(FIT_ROLE), full_cache.subset(ev_role)
    fi, _rf, _mf, floor = HIST.make_inputs(fit, full_cache)
    ei, _re, _me, _ = HIST.make_inputs(ev, full_cache, floor)
    repair_fit = fit.y_true[:, :, 0] - fit.host_pred[:, :, 0]
    repair_ev = ev.y_true[:, :, 0] - ev.host_pred[:, :, 0]
    fit_small = SimpleNamespace(y_true=fit.y_true, host_pred=fit.host_pred, timestamp=fit.timestamp)
    ev_small = SimpleNamespace(y_true=ev.y_true, host_pred=ev.host_pred, timestamp=ev.timestamp)
    return HIST.CellData(
        market, host, cache_path, full_cache, fit_small, ev_small, fi.x, ei.x,
        np.asarray(state_fit, np.float32), np.asarray(state_ev, np.float32),
        None, None,
        np.asarray(repair_fit, np.float32), np.asarray(repair_ev, np.float32),
        None, None,
        np.asarray(fi.scale, np.float32), np.asarray(ei.scale, np.float32), floor)
