"""HCH v0.4 frozen bundle — universal/local package separation.

Math: v0.4 architecture design §8.

Universal package: transferable correction knowledge (shared across domains).
Local package: target-domain evidence/calibration state (non-transferable).
"""
from __future__ import annotations

import hashlib
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HCHV2Bundle:
    """Complete frozen HCH package = universal + local.

    Reload must reproduce, for a fixed query: scale, u, atoms, W1 distances,
    neighbor IDs, final pi, A_hat, q, LCB, and final raw prediction.
    A parameter-hash match alone is insufficient.
    """

    # ---- Universal package (transferable, P1-3) ----
    # Shared correction knowledge: core weights, architecture, config,
    # optimizer/config hash, training code commit, S2T/S2V definition, seed.
    architecture_version: str = "v0.4"
    core_model_state: Optional[dict] = None
    core_config: Optional[dict] = None
    iah_coord_version: str = "asinh-v1"
    optional_params: Optional[dict] = None
    training_provenance: dict = field(default_factory=dict)  # seed, opt hash,
    #                                                          code commit,
    #                                                          S2T/S2V def
    source_datasets: list = field(default_factory=list)      # market manifest
    source_hosts: list = field(default_factory=list)         # host manifest
    universal_hash: str = ""

    # ---- Local package (target-domain profile, P1-3) ----
    # Non-transferable per-domain state: deterministic signature, S1 rank
    # reference, CAGM memory, k, DVG, target/market metadata, split hash.
    data_signature_spec: Optional[dict] = None
    s1_rank_ref: Optional[dict] = None
    atom_memory: Optional[dict] = None
    memory_dates: list = field(default_factory=list)
    memory_timestamps: list = field(default_factory=list)
    w1_version: str = "exact-cdf-v1"
    frozen_k: Optional[int] = None
    proposal_version: str = "double-event-v1"
    dvg_alpha: Optional[float] = None
    dvg_errors: list = field(default_factory=list)
    dvg_q: Optional[float] = None
    local_hashes: dict = field(default_factory=dict)  # split hash, target meta
    fallback_codes: dict = field(default_factory=dict)
    local_hash: str = ""

    # ---- Optional CAVM state (Phase4; empty == v0.4 w1 default) ----
    # memory_mode is pipeline-level config (universal); CAVM ledgers are
    # experience state (global -> universal-hash, local -> local-hash).
    # Old v0.4 bundles load with defaults and fall back to memory_mode="w1".
    memory_mode: str = "w1"
    cavm_key_version: str = ""
    cavm_global_state: Optional[dict] = None
    cavm_local_state: Optional[dict] = None
    cavm_update_policy: dict = field(default_factory=dict)
    cavm_global_hash: str = ""
    cavm_local_hash: str = ""
    cavm_retrieval_lambda: dict = field(default_factory=dict)  # P2: {"atom","context"}

    # ---- Whole-bundle integrity (round-trip verification only) ----
    bundle_hash: str = ""

    # ------------------------------------------------------------------
    def _hash_tensor(self, t) -> None:
        if isinstance(t, dict):
            for k in sorted(t.keys()):
                self._hash_tensor(t[k])
        elif isinstance(t, (list, tuple)):
            for v in t:
                self._hash_tensor(v)
        elif isinstance(t, torch.Tensor):
            self._h.update(t.detach().cpu().numpy().tobytes())
        else:
            self._h.update(repr(t).encode())

    def _hash_of(self, *fields) -> str:
        """Hash exactly the given fields in order (P1-3 package separation)."""
        h = hashlib.sha256()
        self._h = h
        for f in fields:
            self._hash_tensor(getattr(self, f))
        return h.hexdigest()[:16]

    @staticmethod
    def _cavm_state_hash(state: Optional[dict]) -> str:
        """Deterministic deep hash of a freeze() ledger dict (numpy-safe)."""
        if state is None:
            return ""
        h = hashlib.sha256()

        def upd(v):
            if isinstance(v, dict):
                for k in sorted(v.keys()):
                    upd(v[k])
            elif isinstance(v, (list, tuple)):
                for x in v:
                    upd(x)
            elif isinstance(v, type(None)):
                h.update(b"None")
            else:
                h.update(repr(v).encode())
        upd(state)
        return h.hexdigest()[:16]

    def compute_hash(self) -> str:
        """Deep hash per package (P1-3) + whole-bundle round-trip hash."""
        # Universal package: must be stable across local profiles.
        # CAVM: memory_mode + key version + global ledger are pipeline-level
        # evidence; local ledger stays in the local package (§6).
        self.universal_hash = self._hash_of(
            "architecture_version", "core_model_state", "core_config",
            "iah_coord_version", "optional_params", "training_provenance",
            "source_datasets", "source_hosts",
            "memory_mode", "cavm_key_version", "cavm_global_state",
            "cavm_global_hash", "cavm_retrieval_lambda")
        # Local package: per-domain state + calibration + split.
        self.local_hash = self._hash_of(
            "data_signature_spec", "s1_rank_ref", "atom_memory",
            "memory_dates", "memory_timestamps", "w1_version", "frozen_k",
            "proposal_version", "dvg_alpha", "dvg_errors", "dvg_q",
            "local_hashes", "fallback_codes",
            "cavm_local_state", "cavm_update_policy", "cavm_local_hash")
        # Whole bundle = universal + local (round-trip integrity).
        h = hashlib.sha256()
        h.update(self.universal_hash.encode())
        h.update(self.local_hash.encode())
        self.bundle_hash = h.hexdigest()[:16]
        return self.bundle_hash

    def hash(self) -> str:
        return self.compute_hash()

    # ------------------------------------------------------------------
    def save(self, path: str):
        self.compute_hash()
        torch.save({
            "architecture_version": self.architecture_version,
            "core_model_state": self.core_model_state,
            "core_config": self.core_config,
            "iah_coord_version": self.iah_coord_version,
            "optional_params": self.optional_params,
            "training_provenance": self.training_provenance,
            "source_datasets": self.source_datasets,
            "source_hosts": self.source_hosts,
            "universal_hash": self.universal_hash,
            "data_signature_spec": self.data_signature_spec,
            "s1_rank_ref": self.s1_rank_ref,
            "atom_memory": self.atom_memory,
            "memory_dates": self.memory_dates,
            "memory_timestamps": self.memory_timestamps,
            "w1_version": self.w1_version,
            "frozen_k": self.frozen_k,
            "proposal_version": self.proposal_version,
            "dvg_alpha": self.dvg_alpha,
            "dvg_errors": self.dvg_errors,
            "dvg_q": self.dvg_q,
            "local_hashes": self.local_hashes,
            "fallback_codes": self.fallback_codes,
            "local_hash": self.local_hash,
            "memory_mode": self.memory_mode,
            "cavm_key_version": self.cavm_key_version,
            "cavm_global_state": self.cavm_global_state,
            "cavm_local_state": self.cavm_local_state,
            "cavm_update_policy": self.cavm_update_policy,
            "cavm_global_hash": self.cavm_global_hash,
            "cavm_local_hash": self.cavm_local_hash,
            "cavm_retrieval_lambda": self.cavm_retrieval_lambda,
            "bundle_hash": self.bundle_hash,
        }, path)

    @staticmethod
    def load(path: str) -> "HCHV2Bundle":
        data = torch.load(path, map_location="cpu", weights_only=False)
        b = HCHV2Bundle()
        b.architecture_version = data.get("architecture_version", "v0.4")
        b.core_model_state = data.get("core_model_state")
        b.core_config = data.get("core_config")
        b.iah_coord_version = data.get("iah_coord_version", "asinh-v1")
        b.optional_params = data.get("optional_params")
        b.training_provenance = data.get("training_provenance", {})
        b.source_datasets = data.get("source_datasets", [])
        b.source_hosts = data.get("source_hosts", [])
        b.universal_hash = data.get("universal_hash", "")
        b.data_signature_spec = data.get("data_signature_spec")
        b.s1_rank_ref = data.get("s1_rank_ref")
        b.atom_memory = data.get("atom_memory")
        b.memory_dates = data.get("memory_dates", [])
        b.memory_timestamps = data.get("memory_timestamps", [])
        b.w1_version = data.get("w1_version", "exact-cdf-v1")
        b.frozen_k = data.get("frozen_k")
        b.proposal_version = data.get("proposal_version", "double-event-v1")
        b.dvg_alpha = data.get("dvg_alpha")
        b.dvg_errors = data.get("dvg_errors", [])
        b.dvg_q = data.get("dvg_q")
        b.local_hashes = data.get("local_hashes", {})
        b.fallback_codes = data.get("fallback_codes", {})
        b.local_hash = data.get("local_hash", "")
        b.memory_mode = data.get("memory_mode", "w1")
        b.cavm_key_version = data.get("cavm_key_version", "")
        b.cavm_global_state = data.get("cavm_global_state")
        b.cavm_local_state = data.get("cavm_local_state")
        b.cavm_update_policy = data.get("cavm_update_policy", {})
        b.cavm_global_hash = data.get("cavm_global_hash", "")
        b.cavm_local_hash = data.get("cavm_local_hash", "")
        b.cavm_retrieval_lambda = data.get("cavm_retrieval_lambda", {})
        b.bundle_hash = data.get("bundle_hash", "")
        return b

    def extract_universal(self) -> dict:
        """Return the transferable universal sub-package (P1-3).

        Does NOT include the deterministic signature — that lives in the local
        profile. A universal checkpoint must be shareable across domains.
        """
        return {
            "architecture_version": self.architecture_version,
            "core_model_state": self.core_model_state,
            "core_config": self.core_config,
            "iah_coord_version": self.iah_coord_version,
            "optional_params": self.optional_params,
            "training_provenance": self.training_provenance,
            "source_datasets": self.source_datasets,
            "source_hosts": self.source_hosts,
            "universal_hash": self.universal_hash,
            "memory_mode": self.memory_mode,
            "cavm_key_version": self.cavm_key_version,
            "cavm_global_state": self.cavm_global_state,
            "cavm_global_hash": self.cavm_global_hash,
            "cavm_retrieval_lambda": self.cavm_retrieval_lambda,
        }

    def extract_local(self) -> dict:
        """Return the target-domain local sub-package (P1-3)."""
        return {
            "data_signature_spec": self.data_signature_spec,
            "s1_rank_ref": self.s1_rank_ref,
            "atom_memory": self.atom_memory,
            "memory_dates": self.memory_dates,
            "memory_timestamps": self.memory_timestamps,
            "w1_version": self.w1_version,
            "frozen_k": self.frozen_k,
            "proposal_version": self.proposal_version,
            "dvg_alpha": self.dvg_alpha,
            "dvg_errors": self.dvg_errors,
            "dvg_q": self.dvg_q,
            "local_hashes": self.local_hashes,
            "fallback_codes": self.fallback_codes,
            "local_hash": self.local_hash,
            "cavm_local_state": self.cavm_local_state,
            "cavm_update_policy": self.cavm_update_policy,
            "cavm_local_hash": self.cavm_local_hash,
        }
