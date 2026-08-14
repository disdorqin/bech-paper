"""Context-Action Memory (CAVM) — optional v0.4 Local-Evidence enhancement.

Docs: hch_v2_phase4_cavm_technical_architecture / _code_build_spec /
_experiment_design v0.1 (2026-08-14).

What this module adds, and nothing else:
  * A fixed-dimension, audit-friendly continuous context key c_t built from
    pre-forecast information ONLY (host shape / dynamics / time / Data
    Signature / atom summary / optional covariates + availability). It never
    reads target, residual, or action gain.
  * A global/local experience ledger over revealed-day samples.
  * Composite retrieval
        d = lambda_atom * norm(W1) + lambda_ctx * norm(context)
    with lambda_atom=1, lambda_ctx=0 required to reproduce W1-only exactly
    (both normalisations are monotone, so top-k neighbour ORDER is preserved;
    only the candidate SET is selected, replay/proposal/DVG are untouched).
  * ContextActionMemory exposes the SAME per-day atom lists as CAGMAtomMemory
    (dates/z0/w_*/m_*/target_zY/valid_mask) so full_replay_chain /
    estimate_realized_A / double_event_proposal are reused unchanged.

The default pipeline memory_mode stays "w1"; CAVM is opt-in and experimental.
This module is standalone and does not import the legacy path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from w1_retrieval import day_w1_distance

# Context key schema version. Written into the bundle for auditability.
KEY_VERSION = "cavm-key-v1"
# Signature descriptor dim (v0.4 compute_domain_descriptors).
D_SIG = 8


# ------------------------------------------------------------------ utils ---
def _np(x) -> np.ndarray:
    """Tensor -> numpy float64, else asarray(float)."""
    if torch.is_tensor(x):
        a = x.detach().cpu().numpy()
    else:
        a = np.asarray(x)
    return np.asarray(a, dtype=np.float64)


def _hours(x) -> np.ndarray:
    """Flatten to a per-hour [H] float64 array (handles [H], [1,H], [1,H,1])."""
    return _np(x).reshape(-1)


def _hours_bool(x) -> np.ndarray:
    a = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return a.reshape(-1).astype(bool)


def _core_2d(x, d_core: int) -> np.ndarray:
    """Core context to [H, d_core]. Accepts [H,D], [1,H,D]."""
    a = _np(x)
    if a.ndim == 1:
        a = a[:, None]
    elif a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    if a.shape[1] != d_core:
        raise ValueError(f"core_context width {a.shape[1]} != d_core_context "
                         f"{d_core}")
    return a


def _mean(v: np.ndarray) -> float:
    return float(v.mean()) if v.size else 0.0


def _std(v: np.ndarray) -> float:
    return float(np.std(v)) if v.size else 0.0


def _max(v: np.ndarray) -> float:
    return float(np.max(v)) if v.size else 0.0


def _q(v: np.ndarray, p: float) -> float:
    return float(np.quantile(v, p)) if v.size else 0.0


def _clean(v: np.ndarray) -> np.ndarray:
    return v[np.isfinite(v)]


def _nan0(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------------------ data ----
@dataclass
class CAVMExperience:
    """One fully-revealed day sample in the CAVM ledger.

    context_key MUST be built before the label is revealed. target_zY exists
    only on revealed days and is used for historical replay. A_hat / A_true /
    action_error are optional audit fields (computed after reveal).
    audit_domain is for reporting/splitting only — never part of a distance.
    """
    date: str
    context_key: np.ndarray
    z0: np.ndarray
    w_minus: np.ndarray
    w_zero: np.ndarray
    w_plus: np.ndarray
    m_minus: np.ndarray
    m_plus: np.ndarray
    target_zY: np.ndarray
    valid_mask: np.ndarray
    A_hat: float | None = None
    A_true: float | None = None
    action_error: float | None = None
    timestamp: str = ""
    audit_domain: str = ""


# -------------------------------------------------------- key builder --------
class ContextKeyBuilder:
    """Fixed-dimension, pre-forecast-only continuous context key.

    Key layout (D_SIG=8, D=d_core_context, R=n_optional_roles):
        [ c_shape(8) | c_dyn(5) | c_time(2D) | c_sig(D_SIG) | c_atom(8)
          | c_opt(3R) ]
    All statistics are computed on valid hours only. Any NaN/Inf in the inputs
    is carried through the per-field stats and finally zeroed by _nan0 — a key
    never contains NaN/Inf. Absent optional covariates contribute zero
    dimensions (R=0), never fabricated numbers.
    """

    version = KEY_VERSION

    def __init__(self, d_core_context: int = 13, d_sig: int = D_SIG,
                 n_optional_roles: int = 0):
        self.d_core_context = int(d_core_context)
        self.d_sig = int(d_sig)
        self.n_optional_roles = int(n_optional_roles)

    @property
    def dim(self) -> int:
        return (8 + 5 + 2 * self.d_core_context + self.d_sig + 8
                + 3 * self.n_optional_roles)

    def build(self, candidate: dict, core_context, valid_mask, domain_det=None,
              optional_values=None, optional_roles=None,
              optional_masks=None) -> np.ndarray:
        """Return the fixed-dim key vector for one query/revealed day.

        Raises on a target/residual/action-gain field being smuggled in via the
        candidate dict — the builder must never consume the label.
        """
        # ---- hard info-isolation guard (build spec §8.1) ----
        for banned in ("target_raw", "target_zY", "residual", "action_gain",
                       "A_hat", "A_true", "action_error"):
            if banned in candidate:
                raise ValueError(
                    f"ContextKeyBuilder: {banned!r} is label/action-derived and "
                    f"must not enter the context key")

        vm = _hours_bool(valid_mask)
        z0 = _hours(candidate["z0"]) if "z0" in candidate else None

        # ---- c_shape (8): host price shape / scale ----
        if z0 is None:
            shape = np.zeros(8)
        else:
            z = _clean(z0[vm])
            shape = np.array([
                _mean(z), _std(z), _q(z, 0.10), _q(z, 0.50), _q(z, 0.90),
                _q(z, 0.90) - _q(z, 0.10), _mean(np.abs(z)),
                _mean(z > 0) if z.size else 0.0,
            ])

        # ---- c_dyn (5): adjacent-change dynamics ----
        dyn = np.zeros(5)
        if z0 is not None and len(z0) > 1:
            dz = np.diff(z0)
            keep = vm[:-1] & vm[1:]
            if keep.any():
                d = dz[keep]
                dyn = np.array([
                    _mean(np.abs(d)), _std(d), _max(np.abs(d)),
                    _mean(d > 0), _mean(d < 0),
                ])

        # ---- c_time (2D): per-channel mean/std of the core context ----
        cc = _core_2d(core_context, self.d_core_context)
        rows = vm & np.isfinite(cc).all(axis=1)
        time_part = np.zeros(2 * self.d_core_context)
        if rows.any():
            sub = cc[rows]
            for j in range(self.d_core_context):
                c = sub[:, j]
                time_part[2 * j] = _mean(c)
                time_part[2 * j + 1] = _std(c)

        # ---- c_sig (D_SIG): Data Signature descriptors ----
        if domain_det is None:
            sig = np.zeros(self.d_sig)
        else:
            det = _np(domain_det).reshape(-1)
            if det.size == self.d_sig:
                sig = det
            else:
                sig = np.zeros(self.d_sig)

        # ---- c_atom (8): mean/max of the four dose/mass channels ----
        def _stat(name: str, fn) -> float:
            v = _hours(candidate[name])
            return fn(_clean(v[vm]))

        atom = np.array([
            _stat("w_minus", _mean), _stat("w_minus", _max),
            _stat("w_plus", _mean), _stat("w_plus", _max),
            _stat("m_minus", _mean), _stat("m_minus", _max),
            _stat("m_plus", _mean), _stat("m_plus", _max),
        ])

        # ---- c_opt (3R): per-role mean/std/missing-ratio ----
        opt = self._optional_summary(optional_values, optional_roles,
                                     optional_masks, vm)

        key = _nan0(np.concatenate([shape, dyn, time_part, sig, atom, opt]))
        return key

    def _optional_summary(self, values, roles, masks, vm) -> np.ndarray:
        if values is None or not roles:
            return np.zeros(0)
        vals = _np(values)
        if vals.ndim > 2:
            vals = vals.reshape(-1, vals.shape[-1])
        m = (_np(masks) if masks is not None
             else np.ones_like(vals))
        if m.ndim > 2:
            m = m.reshape(-1, m.shape[-1])
        if vals.ndim == 1:
            vals = vals[:, None]
            m = m[:, None]
        n_roles = len(dict.fromkeys(roles))
        if n_roles != self.n_optional_roles:
            raise ValueError(
                f"n_optional_roles {n_roles} != builder {self.n_optional_roles}")
        parts = []
        for role in dict.fromkeys(roles):
            idx = [i for i, r in enumerate(roles) if r == role]
            sub_v = vals[vm][:, idx]
            sub_m = m[vm][:, idx]
            present = sub_m > 0.5
            av = sub_v[present]
            parts.append(_mean(av) if present.any() else 0.0)
            parts.append(_std(av) if present.any() else 0.0)
            parts.append(1.0 - float(sub_m.mean()) if sub_m.size else 1.0)
        return _nan0(np.asarray(parts, dtype=np.float64))


# -------------------------------------------------------------- distances ---
def context_distance(query_key: np.ndarray, memory_keys: np.ndarray) -> np.ndarray:
    """Euclidean distance from the query key to each memory key. Returns [M]."""
    memory_keys = np.asarray(memory_keys, dtype=np.float64)
    if memory_keys.size == 0:
        return np.zeros(0)
    return np.sqrt(((memory_keys - np.asarray(query_key, dtype=np.float64))
                    ** 2).sum(axis=1))


def atom_w1_distance(query_candidate: dict, memory) -> np.ndarray:
    """W1 residual-atom distance (Eq 16) from query to every memory day. [M].

    Equivalent to CAGMAtomMemory.build_retrieval_index. memory may be a
    ContextActionMemory or a CAGMAtomMemory (both expose w_minus/... lists).
    """
    q_w = np.stack([
        _hours(query_candidate["w_minus"]),
        _hours(query_candidate["w_zero"]),
        _hours(query_candidate["w_plus"]),
    ], axis=-1)
    q_m = np.stack([
        _hours(query_candidate["m_minus"]),
        _hours(query_candidate["m_plus"]),
    ], axis=-1)
    q_valid = _hours_bool(query_candidate["valid_mask"])

    M = len(memory.dates)
    out = np.full(M, float("inf"))
    for i in range(M):
        d_w = np.stack([memory.w_minus[i], memory.w_zero[i],
                        memory.w_plus[i]], axis=-1)
        d_m = np.stack([memory.m_minus[i], memory.m_plus[i]], axis=-1)
        out[i] = day_w1_distance(q_w, q_m, d_w, d_m, q_valid,
                                 np.asarray(memory.valid_mask[i]).astype(bool))
    return out


def _norm01(a: np.ndarray) -> np.ndarray:
    """Monotone /max normalisation (max<=0 or non-finite -> zeros)."""
    a = np.asarray(a, dtype=np.float64)
    m = float(a.max()) if a.size else 0.0
    if not np.isfinite(m) or m <= 0:
        return np.zeros_like(a)
    return a / m


def composite_distance(w1: np.ndarray, ctx: np.ndarray,
                       lambda_atom: float = 1.0,
                       lambda_ctx: float = 1.0) -> np.ndarray:
    """d = lambda_atom * norm(w1) + lambda_ctx * norm(ctx).

    lambda_atom=1, lambda_ctx=0 must reproduce W1-only top-k ordering exactly
    (both normalisations are monotone).
    """
    return (lambda_atom * _norm01(w1) + lambda_ctx * _norm01(ctx))


# ------------------------------------------------------------ CAVM store ----
class ContextActionMemory:
    """Global or local experience ledger with composite retrieval.

    Exposes the same per-day atom lists as CAGMAtomMemory so the existing
    full_replay_chain / estimate_realized_A / double_event_proposal pipeline is
    reused verbatim. query() returns neighbour IDs plus component distances.
    """

    def __init__(self, scope: str, key_builder: ContextKeyBuilder):
        self.scope = str(scope)           # "global" | "local"
        self.key_builder = key_builder
        self.experiences: list[CAVMExperience] = []

        # CAGMAtomMemory-compatible per-day lists (for replay reuse).
        self.dates: list[str] = []
        self.z0: list[np.ndarray] = []
        self.w_minus: list[np.ndarray] = []
        self.w_zero: list[np.ndarray] = []
        self.w_plus: list[np.ndarray] = []
        self.m_minus: list[np.ndarray] = []
        self.m_plus: list[np.ndarray] = []
        self.target_zY: list[np.ndarray] = []
        self.valid_mask: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.dates)

    # ------------------------------------------------------------------
    def add_revealed_day(self, experience: CAVMExperience) -> None:
        """Append one revealed-day sample. Rejects unrevealed targets."""
        if experience.target_zY is None:
            raise ValueError(
                "add_revealed_day: target_zY required (label must be revealed)")
        if experience.context_key is None:
            raise ValueError("add_revealed_day: context_key required "
                             "(built pre-reveal)")
        if np.any(np.isnan(experience.context_key)):
            raise ValueError("add_revealed_day: context_key contains NaN")

        self.experiences.append(experience)
        self.dates.append(str(experience.date))
        self.z0.append(np.asarray(experience.z0, dtype=np.float64).reshape(-1))
        self.w_minus.append(np.asarray(experience.w_minus,
                                       dtype=np.float64).reshape(-1))
        self.w_zero.append(np.asarray(experience.w_zero,
                                      dtype=np.float64).reshape(-1))
        self.w_plus.append(np.asarray(experience.w_plus,
                                      dtype=np.float64).reshape(-1))
        self.m_minus.append(np.asarray(experience.m_minus,
                                       dtype=np.float64).reshape(-1))
        self.m_plus.append(np.asarray(experience.m_plus,
                                      dtype=np.float64).reshape(-1))
        self.target_zY.append(np.asarray(experience.target_zY,
                                         dtype=np.float64).reshape(-1))
        self.valid_mask.append(np.asarray(experience.valid_mask,
                                          dtype=bool).reshape(-1))

    # ------------------------------------------------------------- query --
    def atom_w1_distances(self, query_candidate: dict) -> np.ndarray:
        return atom_w1_distance(query_candidate, self)

    def context_distances(self, query_key: np.ndarray) -> np.ndarray:
        if not self.experiences:
            return np.zeros(0)
        keys = np.stack([e.context_key for e in self.experiences])
        return context_distance(query_key, keys)

    def query(self, query_key: np.ndarray, query_candidate: dict, k: int,
              exclude_date: str | None = None,
              lambda_atom: float = 1.0, lambda_ctx: float = 1.0) -> dict:
        """Top-k composite retrieval.

        Returns dict with neighbor_ids, distance_total / distance_w1 /
        distance_context (per selected neighbour), source_scope,
        effective_neighbor_count and self_excluded flag. On empty memory or no
        finite neighbours, effective_neighbor_count=0 (caller must fall back).
        """
        M = len(self)
        if M == 0:
            return {"neighbor_ids": [], "distance_total": [],
                    "distance_w1": [], "distance_context": [],
                    "source_scope": self.scope,
                    "effective_neighbor_count": 0,
                    "self_excluded": exclude_date is not None}

        w1 = self.atom_w1_distances(query_candidate)
        ctx = self.context_distances(query_key)
        d = composite_distance(w1, ctx, lambda_atom, lambda_ctx)

        order = np.argsort(d)
        ids = []
        for idx in order:
            if not np.isfinite(d[idx]):
                continue
            if exclude_date is not None and self.dates[idx] == exclude_date:
                continue
            ids.append(int(idx))
            if len(ids) >= k:
                break

        return {
            "neighbor_ids": ids,
            "distance_total": [float(d[i]) for i in ids],
            "distance_w1": [float(_norm01(w1)[i]) for i in ids],
            "distance_context": [float(_norm01(ctx)[i]) for i in ids],
            "source_scope": self.scope,
            "effective_neighbor_count": len(ids),
            "self_excluded": exclude_date is not None,
        }

    # ------------------------------------------------------------ freeze --
    def freeze(self) -> dict:
        """Serialize the ledger to a plain-dict state (for the bundle)."""
        exp = []
        for e in self.experiences:
            exp.append({
                "date": e.date,
                "context_key": np.asarray(e.context_key, dtype=np.float64),
                "z0": np.asarray(e.z0, dtype=np.float64),
                "w_minus": np.asarray(e.w_minus, dtype=np.float64),
                "w_zero": np.asarray(e.w_zero, dtype=np.float64),
                "w_plus": np.asarray(e.w_plus, dtype=np.float64),
                "m_minus": np.asarray(e.m_minus, dtype=np.float64),
                "m_plus": np.asarray(e.m_plus, dtype=np.float64),
                "target_zY": np.asarray(e.target_zY, dtype=np.float64),
                "valid_mask": np.asarray(e.valid_mask, dtype=bool),
                "A_hat": e.A_hat, "A_true": e.A_true,
                "action_error": e.action_error,
                "timestamp": e.timestamp, "audit_domain": e.audit_domain,
            })
        return {
            "scope": self.scope,
            "key_version": self.key_builder.version,
            "d_core_context": self.key_builder.d_core_context,
            "d_sig": self.key_builder.d_sig,
            "n_optional_roles": self.key_builder.n_optional_roles,
            "key_dim": self.key_builder.dim,
            "experiences": exp,
        }

    @staticmethod
    def from_frozen(state: dict,
                    key_builder: ContextKeyBuilder | None = None) \
            -> "ContextActionMemory":
        """Rebuild a ledger from freeze() output."""
        kb = key_builder or ContextKeyBuilder(
            d_core_context=state.get("d_core_context", 13),
            d_sig=state.get("d_sig", D_SIG),
            n_optional_roles=state.get("n_optional_roles", 0))
        mem = ContextActionMemory(state.get("scope", ""), kb)
        for e in state.get("experiences", []):
            mem.add_revealed_day(CAVMExperience(
                date=e["date"],
                context_key=np.asarray(e["context_key"], dtype=np.float64),
                z0=np.asarray(e["z0"], dtype=np.float64),
                w_minus=np.asarray(e["w_minus"], dtype=np.float64),
                w_zero=np.asarray(e["w_zero"], dtype=np.float64),
                w_plus=np.asarray(e["w_plus"], dtype=np.float64),
                m_minus=np.asarray(e["m_minus"], dtype=np.float64),
                m_plus=np.asarray(e["m_plus"], dtype=np.float64),
                target_zY=np.asarray(e["target_zY"], dtype=np.float64),
                valid_mask=np.asarray(e["valid_mask"], dtype=bool),
                A_hat=e.get("A_hat"), A_true=e.get("A_true"),
                action_error=e.get("action_error"),
                timestamp=e.get("timestamp", ""),
                audit_domain=e.get("audit_domain", ""),
            ))
        return mem
