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

    # ---- Universal package (transferable) ----
    architecture_version: str = "v0.4"
    core_model_state: Optional[dict] = None
    core_config: Optional[dict] = None
    data_signature_spec: Optional[dict] = None
    iah_coord_version: str = "asinh-v1"
    optional_params: Optional[dict] = None
    training_provenance: dict = field(default_factory=dict)
    source_datasets: list = field(default_factory=list)
    source_hosts: list = field(default_factory=list)

    # ---- Local package (target-domain) ----
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
    local_hashes: dict = field(default_factory=dict)
    fallback_codes: dict = field(default_factory=dict)

    # ---- Whole-bundle integrity ----
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

    def compute_hash(self) -> str:
        """Deep hash covering universal + local + calibration + split."""
        self._h = hashlib.sha256()
        self._hash_tensor(self.architecture_version)
        self._hash_tensor(self.core_model_state)
        self._hash_tensor(self.core_config)
        self._hash_tensor(self.data_signature_spec)
        self._hash_tensor(self.iah_coord_version)
        self._hash_tensor(self.optional_params)
        self._hash_tensor(self.training_provenance)
        self._hash_tensor(self.source_datasets)
        self._hash_tensor(self.source_hosts)
        self._hash_tensor(self.s1_rank_ref)
        self._hash_tensor(self.atom_memory)
        self._hash_tensor(self.memory_dates)
        self._hash_tensor(self.memory_timestamps)
        self._hash_tensor(self.w1_version)
        self._hash_tensor(self.frozen_k)
        self._hash_tensor(self.proposal_version)
        self._hash_tensor(self.dvg_alpha)
        self._hash_tensor(self.dvg_errors)
        self._hash_tensor(self.dvg_q)
        self._hash_tensor(self.local_hashes)
        self._hash_tensor(self.fallback_codes)
        self.bundle_hash = self._h.hexdigest()[:16]
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
            "data_signature_spec": self.data_signature_spec,
            "iah_coord_version": self.iah_coord_version,
            "optional_params": self.optional_params,
            "training_provenance": self.training_provenance,
            "source_datasets": self.source_datasets,
            "source_hosts": self.source_hosts,
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
            "bundle_hash": self.bundle_hash,
        }, path)

    @staticmethod
    def load(path: str) -> "HCHV2Bundle":
        data = torch.load(path, map_location="cpu", weights_only=False)
        b = HCHV2Bundle()
        b.architecture_version = data.get("architecture_version", "v0.4")
        b.core_model_state = data.get("core_model_state")
        b.core_config = data.get("core_config")
        b.data_signature_spec = data.get("data_signature_spec")
        b.iah_coord_version = data.get("iah_coord_version", "asinh-v1")
        b.optional_params = data.get("optional_params")
        b.training_provenance = data.get("training_provenance", {})
        b.source_datasets = data.get("source_datasets", [])
        b.source_hosts = data.get("source_hosts", [])
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
        b.bundle_hash = data.get("bundle_hash", "")
        return b

    def extract_universal(self) -> dict:
        """Return the transferable universal sub-package."""
        return {
            "architecture_version": self.architecture_version,
            "core_model_state": self.core_model_state,
            "core_config": self.core_config,
            "data_signature_spec": self.data_signature_spec,
            "iah_coord_version": self.iah_coord_version,
            "optional_params": self.optional_params,
            "training_provenance": self.training_provenance,
            "source_datasets": self.source_datasets,
            "source_hosts": self.source_hosts,
        }

    def extract_local(self) -> dict:
        """Return the target-domain local sub-package."""
        return {
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
        }
