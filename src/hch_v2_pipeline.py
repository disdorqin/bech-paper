"""HCH v0.4 Universal Pipeline — authoritative end-to-end orchestrator.

Owns the full v0.3 IAH decision chain in stage order:
    S1 reference -> IAH candidate -> S3-M memory/k -> S3-C DVG -> freeze -> S4 predict

This is the ONLY formal entry point. It must never import or invoke the
legacy HCH path (BiOMC, candidate_loss_fn, state_loss_fn, CARA/KL calibration).
"""
from __future__ import annotations

import datetime
import hashlib
import subprocess
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from iah_candidate import IAHCandidateHead


def _git_head() -> str:
    """Short git HEAD for training provenance (P1-3). Best-effort."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


_GIT_HEAD = _git_head()
from iah_crps_loss import iah_crps_loss
from s1_rank import S1RankReference
from hch_v2_context import compute_domain_descriptors
from w1_retrieval import CAGMAtomMemory
from query_replay import full_replay_chain, estimate_realized_A
from double_event import double_event_proposal
from dvg_calibrate import DGVSplitConformal
from hch_v2_bundle import HCHV2Bundle
from context_action_memory import (
    ContextKeyBuilder, ContextActionMemory, CAVMExperience)


class HCHV2UniversalPipeline:
    """Authoritative v0.4 orchestrator.

    Stage ownership is explicit; each stage mutates only its own state.
    S3-C can never mutate candidate / memory / k / proposal.
    """

    def __init__(self, d_core_context: int = 13, d_model: int = 64,
                 d_value: int = 0, alpha: float = 0.10,
                 k: Optional[int] = None, seed: int = 0,
                 memory_mode: str = "w1"):
        self.d_core_context = d_core_context
        self.d_model = d_model
        self.d_value = d_value
        self.alpha = alpha
        self.k = k
        self.seed = seed
        # Phase4: "w1" (default, unchanged) | "cavm". memory_mode only switches
        # the CAVM branch ON; the W1 path is always the v0.4 control.
        self.memory_mode = memory_mode

        torch.manual_seed(seed)
        self.candidate_head = IAHCandidateHead(d_core_context, d_model,
                                               d_value=d_value)
        self.s1_rank_ref: Optional[S1RankReference] = None
        self.memory: Optional[CAGMAtomMemory] = None
        self.dvg: Optional[DGVSplitConformal] = None
        self._domain_det: Optional[np.ndarray] = None
        # CAVM read-only state (P1). Predictions never read cavm_local.
        self.cavm_key_builder: Optional[ContextKeyBuilder] = None
        self.cavm_global: Optional[ContextActionMemory] = None
        self.cavm_local: Optional[ContextActionMemory] = None
        self.cavm_update_policy: dict = {}  # P3: {"observe": bool, ...}
        # P2 composite retrieval weights; (1,0) reproduces W1-only exactly.
        # Only selected on S2V/S3-M, frozen at S4 (never tuned on S4 labels).
        self.cavm_lambda: tuple = (1.0, 0.0)

    def set_cavm_retrieval(self, lambda_atom: float = 1.0,
                           lambda_ctx: float = 0.0) -> None:
        """Set composite retrieval weights d = λ_atom·norm(W1)+λ_ctx·norm(ctx).

        lambda_atom=1, lambda_ctx=0 reproduces W1-only neighbor selection.
        Select only on S2V/S3-M evidence; S4 weights are frozen in the bundle.
        """
        self.cavm_lambda = (float(lambda_atom), float(lambda_ctx))

    # ------------------------------------------------------------- S1 ----
    def fit_s1_reference(self, s1_host_z0: np.ndarray,
                         s1_hours: Optional[np.ndarray] = None) -> S1RankReference:
        """Build local S1 rank reference from S1 host hyperbolic coordinates."""
        self.s1_rank_ref = S1RankReference(s1_host_z0, s1_hours)
        return self.s1_rank_ref

    def fit_s1_signature(self, s1_z0: np.ndarray,
                         s1_hours: Optional[np.ndarray] = None) -> np.ndarray:
        """Estimate and freeze the DataSignature domain descriptors from S1.

        Descriptors are domain-level stable geometry, computed once over the S1
        host z0 pool and frozen into the candidate head's signature buffer.
        """
        det = compute_domain_descriptors(s1_z0, s1_hours)
        self.candidate_head.core_encoder.signature.set_domain_descriptors(det)
        self._domain_det = det
        return det

    # ------------------------------------------------------------- S2 ----
    def _eval_s2_batches(self, batches) -> Optional[float]:
        """Mean IAH-CRPS over a list of S2 batches (eval mode). None if empty."""
        if not batches:
            return None
        total, nb = 0.0, 0
        with torch.no_grad():
            for batch in batches:
                if len(batch) >= 5:
                    host, ctx, target, vm, domain_det = batch[:5]
                else:
                    host, ctx, target, vm = batch
                    domain_det = None
                out = self.candidate_head(host, ctx, valid_mask=vm,
                                          domain_det=domain_det)
                loss = iah_crps_loss(out, target)
                total += float(loss)
                nb += 1
        return total / max(nb, 1)

    def train_candidate_s2(self, s2_batches, s2v_batches=None,
                           epochs: int = 30, lr: float = 1e-3,
                           patience: int = 8) -> float:
        """Train the assembled v0.4 core on S2 with the single IAH-CRPS objective.

        s2_batches: iterable of
            (host_raw [B,H,1], core_context [B,H,D],
             target_raw [B,H,1], valid_mask [B,H])
            or optionally 5-tuples with
             (..., domain_det [B,d_det])  # per-batch signature context (P0-1)
        s2v_batches: optional same-shaped S2V batches. If provided (G0-3),
            the S2V IAH-CRPS selects the restored checkpoint and drives
            early stopping; the pooled S2T loss NEVER selects. If None,
            falls back to S2T train-loss patience (single-domain legacy).
        core_context excludes z0 (appended inside the head).
        """
        opt = torch.optim.AdamW(self.candidate_head.parameters(), lr=lr,
                                weight_decay=1e-4)
        best, best_state, pat = float("inf"), None, 0
        best_source = "s2v" if s2v_batches else "s2t"

        for ep in range(epochs):
            total, nb = 0.0, 0
            for batch in s2_batches:
                if len(batch) >= 5:
                    host, ctx, target, vm, domain_det = batch[:5]
                else:
                    host, ctx, target, vm = batch
                    domain_det = None
                opt.zero_grad()
                out = self.candidate_head(host, ctx, valid_mask=vm,
                                          domain_det=domain_det)
                loss = iah_crps_loss(out, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.candidate_head.parameters(), 1.0)
                opt.step()
                total += loss.item()
                nb += 1
            train_avg = total / max(nb, 1)
            sel = self._eval_s2_batches(s2v_batches) if s2v_batches else train_avg
            if sel is not None and sel < best - 1e-5:
                best, pat = sel, 0
                best_state = {k: v.clone() for k, v in
                              self.candidate_head.state_dict().items()}
            else:
                pat += 1
                if pat >= patience:
                    break
        if best_state is not None:
            self.candidate_head.load_state_dict(best_state)
        if s2v_batches:
            print(f"S2 checkpoint selected by S2V (G0-3): {best:.4f}")
        return float(best)

    # ---------------------------------------------------------- S3-M ----
    def fit_s3_memory(self, s3m_days: list) -> CAGMAtomMemory:
        """Build CAGM atom memory from S3-M days."""
        mem = CAGMAtomMemory()
        for day in s3m_days:
            mem.add_day(day["date"], day["candidate"], day["target_zY"])
        self.memory = mem
        return mem

    def fit_cavm_memory(self, global_days: list, local_days: list | None = None,
                        key_builder: ContextKeyBuilder | None = None) -> dict:
        """Build read-only CAVM ledgers (Phase4 P1). Predictions unchanged.

        global_days/local_days: list of day dicts, each with
            {"date", "candidate" (incl. z0), "target_zY",
             "core_context": optional [H,D] or [1,H,D] (per-day),
             "domain_det": optional [d_det],
             "audit_domain": optional str}
        Only revealed days (target_zY present) may enter. No S4 target is
        accepted here; predictions never consume cavm_local.
        Returns {"global": n_days, "local": n_days}.
        """
        kb = key_builder or ContextKeyBuilder(d_core_context=self.d_core_context)
        self.cavm_key_builder = kb

        g = ContextActionMemory("global", kb)
        for day in global_days:
            g.add_revealed_day(_cavm_experience(day, kb))
        self.cavm_global = g

        if local_days:
            loc = ContextActionMemory("local", kb)
            for day in local_days:
                loc.add_revealed_day(_cavm_experience(day, kb))
            self.cavm_local = loc
        else:
            self.cavm_local = None
        return {"global": len(g),
                "local": len(self.cavm_local) if self.cavm_local else 0}

    # -------------------------------------------------- P3: local observe ----
    def set_cavm_update_policy(self, observe: bool = False, **extra) -> None:
        """Configure whether observe_outcome() may append to local memory.

        Default is OFF: a strictly frozen S4 never records experience. Setting
        observe=True only enables the local ledger append — it never updates
        universal parameters, k, lambda, or q (P3 §5.2).
        """
        policy = {"observe": bool(observe)}
        policy.update({k: v for k, v in extra.items()})
        self.cavm_update_policy = policy

    def observe_outcome(self, query_id, target_zY: np.ndarray,
                        evidence: dict) -> dict:
        """Append one revealed day to LOCAL memory (Phase4 P3). Default OFF.

        Callable only AFTER: (1) the query was predicted, (2) the prediction
        time passed, (3) the real target fully revealed. The context_key is
        NEVER rebuilt here — it is read verbatim from the prediction-time
        evidence["context_keys"], so a key can never depend on the label.

        Effects are strictly scoped to local memory:
          - universal model parameters:   untouched
          - k / lambda / q:               untouched (never re-selected)
          - global ledger:                untouched
          - predictions:                  unchanged (predictions never consume
                                          cavm_local, build spec §5.2)

        query_id: int batch index into evidence, or str matched against
            evidence["query_ids"].
        target_zY: [H] revealed SCALE-FREE target, zY = arcsinh(y / s) with the
            query day's own scale s (pipeline convention — the same space as
            z0; raw price in EUR/MWh is NOT accepted and silently corrupts
            A_true). evidence: the batch dict returned by predict_s4 (must
            carry context_keys).
        """
        if not self.cavm_update_policy.get("observe", False):
            return {"applied": False, "reason": "observe_disabled",
                    "scope": "local"}
        if self.memory_mode != "cavm" or self.cavm_key_builder is None:
            return {"applied": False,
                    "reason": "cavm_not_active_at_query_time"}
        if "context_keys" not in evidence or not evidence["context_keys"]:
            raise ValueError(
                "observe_outcome: evidence has no pre-forecast context_keys "
                "(predict_s4 must run with memory_mode='cavm' first)")
        if target_zY is None:
            raise ValueError(
                "observe_outcome: target_zY required (label must be revealed)")

        idx = _resolve_query_id(query_id, evidence)
        if idx is None:
            raise ValueError(
                f"observe_outcome: cannot resolve query_id={query_id!r} "
                "(int index or str in evidence['query_ids'])")

        # Local ledger is created lazily on first observe; never touches global.
        if self.cavm_local is None:
            self.cavm_local = ContextActionMemory(
                "local", self.cavm_key_builder)

        cand = evidence["candidate"]
        qvc = _cavm_day_view(cand, idx)
        z0 = qvc["z0"].detach().cpu().numpy().reshape(-1)
        vm = qvc["valid_mask"].detach().cpu().numpy().reshape(-1).astype(bool)
        zY = np.asarray(target_zY, dtype=np.float64).reshape(-1)
        pi_q = np.asarray(evidence["pi"][idx], dtype=np.float64).reshape(-1)
        A_hat = float(evidence["A_hat"][idx])
        A_true = float(estimate_realized_A(z0, zY, pi_q, vm))
        action_error = A_hat - A_true

        exp = CAVMExperience(
            date=str(query_id),
            context_key=np.asarray(evidence["context_keys"][idx],
                                   dtype=np.float64),
            z0=qvc["z0"], w_minus=qvc["w_minus"], w_zero=qvc["w_zero"],
            w_plus=qvc["w_plus"], m_minus=qvc["m_minus"],
            m_plus=qvc["m_plus"],
            target_zY=zY, valid_mask=qvc["valid_mask"],
            A_hat=A_hat, A_true=A_true, action_error=action_error,
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            audit_domain=str(evidence.get("cavm", {}).get("neighbor_scopes",
                                                          [""])[idx])
            if evidence.get("cavm") else "",
        )
        self.cavm_local.add_revealed_day(exp)
        return {
            "applied": True, "scope": "local", "date": str(query_id),
            "n_local": len(self.cavm_local),
            "A_hat": A_hat, "A_true": A_true,
            "action_error": action_error,
            "timestamp": exp.timestamp,
            "reason": "post-reveal local observe (P3)",
        }

    def _replay_value(self, day: dict, k: int) -> tuple:
        """Single authority for the retrieval->replay->proposal->(A_hat, A_true).

        Both select_s3m_k and calibrate_s3c route through this; the chain is
        implemented exactly once. Returns (A_hat, A_true).
        """
        q_out = day["candidate"]
        z0 = q_out["z0"].detach().cpu().numpy().reshape(-1)
        zY = np.asarray(day["target_zY"]).reshape(-1)
        m_minus = q_out["m_minus"].detach().cpu().numpy().reshape(-1)
        m_plus = q_out["m_plus"].detach().cpu().numpy().reshape(-1)
        q_valid = q_out["valid_mask"].detach().cpu().numpy()
        if q_valid.ndim == 3:
            q_valid = q_valid.squeeze(-1)
        vm = q_valid.reshape(-1).astype(bool)

        dists = self.memory.build_retrieval_index(q_out)
        nbr = self.memory.get_neighbors(dists, k)
        chain = full_replay_chain(self.memory, nbr, m_minus, m_plus,
                                  double_event_proposal, query_valid=vm)
        A_hat = chain["A_hat"]
        A_true = estimate_realized_A(z0, zY, chain["pi_q"], vm)
        return A_hat, A_true

    def select_s3m_k(self, candidate_k: list[int], validation_days: list) -> int:
        """Select k by forward validation (REAL implementation, CRITICAL-1).

        For each k: replay the query dose on k neighbors to get A_hat(k), form
        the proposal pi(k), compute the realized action value on the validation
        day itself, and score by -mean|A_hat(k) - realized_A(k)|. The k whose
        retrieval estimate best predicts realized value wins. All terms depend
        on k, so this is a genuine selection.
        """
        n_mem = len(self.memory.dates) if self.memory is not None else 0
        valid_k = [k for k in candidate_k if k <= n_mem]
        dropped = [k for k in candidate_k if k > n_mem]
        if dropped:
            # protocol §16: k > available memory is INVALID and must be
            # removed, never silently clipped (get_neighbors would return
            # fewer than k neighbors and distort the selection score).
            print(f"S3-M: k candidates {dropped} exceed memory size {n_mem} "
                  f"— dropped (protocol §16)")
        if not valid_k:
            raise ValueError(
                f"S3-M: no valid k <= memory size {n_mem} in {candidate_k}")

        best_k, best_score = None, -float("inf")
        for k in valid_k:
            errors = []
            for vd in validation_days:
                A_hat, realized_A = self._replay_value(vd, k)
                errors.append(abs(A_hat - realized_A))
            score = -float(np.mean(errors)) if errors else -float("inf")
            if score > best_score:
                best_score, best_k = score, k
        self.k = best_k if best_k is not None else valid_k[0]
        return self.k

    # ---------------------------------------------------------- S3-C ----
    def _compute_A_hat_and_true(self, day: dict) -> tuple:
        """Compute A_hat (final replay on frozen memory/k) and A_true (realized)."""
        return self._replay_value(day, self.k)

    def calibrate_s3c(self, s3c_days: list) -> dict:
        """S3-C: split-conformal calibration of the whole-day action value.

        s3c_days: list of {"candidate", "target_zY"}. A_hat/A_true are computed
        internally via the frozen memory/k and realized target.
        """
        self.dvg = DGVSplitConformal(alpha=self.alpha)
        for day in s3c_days:
            A_hat, A_true = self._compute_A_hat_and_true(day)
            self.dvg.record_error(A_hat, A_true)
        return self.dvg.compute_quantile()

    # ---------------------------------------------------------- freeze ----
    def freeze_bundle(self, dataset_id: str = "", split_hash: str = "") -> HCHV2Bundle:
        """Freeze the full pipeline into a universal+local bundle (P1-3)."""
        b = HCHV2Bundle()
        b.core_model_state = {k: v.clone() for k, v in
                              self.candidate_head.state_dict().items()}
        b.core_config = {"d_core_context": self.d_core_context,
                         "d_model": self.d_model,
                         "d_value": self.d_value,
                         "seed": self.seed}
        # Universal provenance: optimizer/config hash, code commit, S2T/S2V def.
        b.training_provenance = {
            "seed": self.seed,
            "s2_split": "S2T/S2V (v0.3 protocol §5)",
            "config_hash": b.core_config and hashlib.sha256(
                repr(sorted(b.core_config.items())).encode()).hexdigest()[:16],
            "code_commit": _GIT_HEAD,
        }
        b.source_datasets = [dataset_id] if dataset_id else []
        b.source_hosts = [dataset_id or ""] if dataset_id else []
        # Local profile: per-domain deterministic signature + target metadata.
        if self._domain_det is not None:
            b.data_signature_spec = {
                "det": self._domain_det.tolist(),
                "version": "domain-det-v1",
                "source": dataset_id,
            }
        if self.s1_rank_ref is not None:
            b.s1_rank_ref = self.s1_rank_ref.freeze()
        if self.memory is not None:
            b.atom_memory = {
                "dates": self.memory.dates,
                "z0": self.memory.z0,
                "w_minus": self.memory.w_minus,
                "w_zero": self.memory.w_zero,
                "w_plus": self.memory.w_plus,
                "m_minus": self.memory.m_minus,
                "m_plus": self.memory.m_plus,
                "target_zY": self.memory.target_zY,
                "valid_mask": self.memory.valid_mask,
            }
            b.memory_dates = list(self.memory.dates)
        b.frozen_k = self.k
        if self.dvg is not None:
            frozen = self.dvg.freeze()
            b.dvg_alpha = frozen["alpha"]
            b.dvg_errors = frozen["errors"]
            b.dvg_q = frozen["q"]
        # Local metadata: target/market + split hash (P1-3).
        b.local_hashes = {"split_hash": split_hash,
                          "target_market": dataset_id,
                          "target_host": dataset_id or ""}
        # Optional CAVM state (Phase4; empty for memory_mode="w1").
        b.memory_mode = self.memory_mode
        if self.cavm_key_builder is not None:
            b.cavm_key_version = self.cavm_key_builder.version
        if self.cavm_global is not None:
            g_state = self.cavm_global.freeze()
            b.cavm_global_state = g_state
            b.cavm_global_hash = HCHV2Bundle._cavm_state_hash(g_state)
        if self.cavm_local is not None:
            l_state = self.cavm_local.freeze()
            b.cavm_local_state = l_state
            b.cavm_local_hash = HCHV2Bundle._cavm_state_hash(l_state)
        b.cavm_update_policy = dict(self.cavm_update_policy)
        b.cavm_retrieval_lambda = {"atom": self.cavm_lambda[0],
                                   "context": self.cavm_lambda[1]}
        b.compute_hash()
        return b

    @staticmethod
    def from_bundle(bundle: HCHV2Bundle) -> "HCHV2UniversalPipeline":
        """Rebuild a full pipeline from a frozen bundle (HIGH-3).

        Restores candidate head, S1 rank reference, atom memory, and DVG, so a
        reloaded pipeline reproduces candidate/neighbors/pi/A_hat/q/LCB.
        """
        cfg = bundle.core_config or {}
        pipe = HCHV2UniversalPipeline(
            d_core_context=cfg.get("d_core_context", 13),
            d_model=cfg.get("d_model", 64),
            d_value=cfg.get("d_value", 0),
            alpha=bundle.dvg_alpha if bundle.dvg_alpha is not None else 0.10,
            k=bundle.frozen_k,
            seed=cfg.get("seed", 0),
        )
        if bundle.core_model_state is not None:
            pipe.candidate_head.load_state_dict(bundle.core_model_state)
            pipe.candidate_head.eval()

        if bundle.s1_rank_ref is not None:
            pipe.s1_rank_ref = S1RankReference.from_frozen(bundle.s1_rank_ref)

        if bundle.data_signature_spec:
            det = np.asarray(bundle.data_signature_spec["det"], dtype=np.float64)
            pipe.candidate_head.core_encoder.signature.set_domain_descriptors(det)
            pipe._domain_det = det

        if bundle.atom_memory is not None:
            mem = CAGMAtomMemory()
            am = bundle.atom_memory
            mem.dates = list(am["dates"])
            mem.z0 = [np.asarray(a) for a in am["z0"]]
            mem.w_minus = [np.asarray(a) for a in am["w_minus"]]
            mem.w_zero = [np.asarray(a) for a in am["w_zero"]]
            mem.w_plus = [np.asarray(a) for a in am["w_plus"]]
            mem.m_minus = [np.asarray(a) for a in am["m_minus"]]
            mem.m_plus = [np.asarray(a) for a in am["m_plus"]]
            mem.target_zY = [np.asarray(a) for a in am["target_zY"]]
            mem.valid_mask = [np.asarray(a).astype(bool) for a in am["valid_mask"]]
            pipe.memory = mem

        if bundle.dvg_q is not None:
            pipe.dvg = DGVSplitConformal.from_frozen({
                "alpha": bundle.dvg_alpha,
                "errors": list(bundle.dvg_errors),
                "q": bundle.dvg_q,
                "n_calibration": len(bundle.dvg_errors),
            })

        # Optional CAVM state (Phase4; old bundles load with "w1" defaults).
        pipe.memory_mode = bundle.memory_mode
        if bundle.cavm_global_state is not None:
            pipe.cavm_global = ContextActionMemory.from_frozen(
                bundle.cavm_global_state)
            pipe.cavm_key_builder = pipe.cavm_global.key_builder
        if bundle.cavm_local_state is not None:
            pipe.cavm_local = ContextActionMemory.from_frozen(
                bundle.cavm_local_state)
        pipe.cavm_update_policy = dict(bundle.cavm_update_policy or {})
        if bundle.cavm_retrieval_lambda:
            pipe.cavm_lambda = (
                float(bundle.cavm_retrieval_lambda.get("atom", 1.0)),
                float(bundle.cavm_retrieval_lambda.get("context", 0.0)))

        return pipe

    # ---------------------------------------------------------- S4 ----
    def predict_s4(self, host_raw: torch.Tensor, core_context: torch.Tensor,
                   valid_mask: Optional[torch.Tensor] = None,
                   domain_det: Optional[torch.Tensor] = None) -> dict:
        """Target-free S4 inference.

        host_raw [B,H,1], core_context [B,H,D], domain_det [B,d_det] (P0-1).
        No target accepted. Returns full per-day evidence: candidate atoms,
        neighbors, proposals, pi, A_hat, q, LCB, final raw action — enough to
        serialize evidence JSON.
        """
        with torch.no_grad():
            out = self.candidate_head(host_raw, core_context,
                                      valid_mask=valid_mask,
                                      domain_det=domain_det)

        evidence = {"candidate": out, "memory_mode": self.memory_mode}

        if self.memory is None or self.dvg is None or self.dvg.q is None:
            B = host_raw.shape[0]
            evidence["final_action"] = ["identity"] * B
            evidence["fallback"] = "no_memory_or_calibration"
            evidence["x_final"] = out["x_identity"]
            if self.memory_mode == "cavm" and self.cavm_global is not None:
                evidence["context_key_version"] = self.cavm_key_builder.version
                evidence["cavm"] = {"neighbor_scopes": ["identity"] * B,
                                    "neighbor_ids": [[] for _ in range(B)],
                                    "distance_context": [[] for _ in range(B)],
                                    "distance_w1": [[] for _ in range(B)],
                                    "effective_neighbor_count": [0] * B}
            return evidence

        B, H, _ = host_raw.shape
        m_minus = out["m_minus"].detach().cpu().numpy()
        m_plus = out["m_plus"].detach().cpu().numpy()
        q_valid = out["valid_mask"].detach().cpu().numpy()
        if q_valid.ndim == 3:
            q_valid = q_valid.squeeze(-1)
        final_x = out["x_identity"].clone()
        actions, pi_all, A_hats, lcb_all = [], [], [], []
        neighbors_all, dists_all, proposals_all = [], [], []

        # CAVM composite retrieval (Phase4 P2): when memory_mode="cavm" and a
        # global ledger is present, neighbor selection AND replay run on the
        # ContextActionMemory ledger with d = λ_atom·norm(W1)+λ_ctx·norm(ctx).
        # Default λ=(1,0) reproduces W1-only exactly (verified by tests).
        # memory_mode="w1" (default) keeps the original CAGM path byte-for-byte.
        cavm_active = (self.memory_mode == "cavm" and self.cavm_global
                       is not None and self.cavm_key_builder is not None)
        cavm_scope, cavm_nbr, cavm_ctx, cavm_w1, cavm_eff = [], [], [], [], []
        cavm_keys = []  # P3: pre-forecast context keys, recorded at query time
        fallback_reasons = []

        for b in range(B):
            if cavm_active:
                qvc = _cavm_day_view(out, b)
                vm_b = out["valid_mask"][b:b + 1]
                det_b = (domain_det[b:b + 1] if domain_det is not None
                         else None)
                qk = self.cavm_key_builder.build(
                    qvc, core_context[b:b + 1], vm_b, det_b)
                cavm_keys.append(np.asarray(qk, dtype=np.float64))
                lam_a, lam_c = self.cavm_lambda
                res = self.cavm_global.query(
                    qk, qvc, k=self.k, lambda_atom=lam_a, lambda_ctx=lam_c)
                nbr = res["neighbor_ids"]
                replay_mem = self.cavm_global
                cavm_scope.append(res["source_scope"])
                cavm_nbr.append(nbr)
                cavm_ctx.append([float(x) for x in res["distance_context"]])
                cavm_w1.append([float(x) for x in res["distance_w1"]])
                cavm_eff.append(res["effective_neighbor_count"])
                neighbors_all.append(nbr)
                dists_all.append([float(x) for x in res["distance_total"]])
                if not nbr:
                    # No finite composite neighbors -> safe Identity fallback
                    # (build spec §8.6: explicit, never silent truncation).
                    actions.append("identity")
                    pi_all.append(np.zeros(H))
                    A_hats.append(0.0)
                    proposals_all.append({"I_down": [], "I_up": []})
                    lcb_all.append(float(self.dvg.lcb(0.0)["lcb"]))
                    fallback_reasons.append("cavm_no_neighbors")
                    continue
            else:
                dists = self.memory.build_retrieval_index(_day_view(out, b))
                nbr = self.memory.get_neighbors(dists, self.k)
                replay_mem = self.memory
                neighbors_all.append(nbr)
                dists_all.append([float(dists[i]) for i in nbr])

            chain = full_replay_chain(
                replay_mem, nbr,
                m_minus[b].squeeze(-1) if m_minus.ndim == 3 else m_minus[b],
                m_plus[b].squeeze(-1) if m_plus.ndim == 3 else m_plus[b],
                double_event_proposal,
                query_valid=q_valid[b].astype(bool),
            )
            pi_all.append(chain["pi_q"])
            A_hats.append(chain["A_hat"])
            proposals_all.append(chain["proposal"])

            lcb_info = self.dvg.lcb(chain["A_hat"])
            lcb_all.append(float(lcb_info["lcb"]))
            if lcb_info["execute"]:
                actions.append("execute")
                for h in range(H):
                    if chain["pi_q"][h] < 0:
                        final_x[b, h, 0] = out["x_down"][b, h, 0]
                    elif chain["pi_q"][h] > 0:
                        final_x[b, h, 0] = out["x_up"][b, h, 0]
            else:
                actions.append("identity")

        evidence["neighbors"] = neighbors_all
        evidence["neighbor_distances"] = dists_all
        evidence["proposals"] = proposals_all
        evidence["pi"] = pi_all
        evidence["A_hat"] = A_hats
        evidence["q"] = self.dvg.q
        evidence["lcb"] = lcb_all
        evidence["final_action"] = actions
        evidence["x_final"] = final_x
        if fallback_reasons:
            evidence["fallback_reasons"] = fallback_reasons
        if cavm_active:
            evidence["context_key_version"] = self.cavm_key_builder.version
            evidence["cavm_lambda"] = {"atom": self.cavm_lambda[0],
                                       "context": self.cavm_lambda[1]}
            evidence["cavm"] = {
                "neighbor_scopes": cavm_scope,
                "neighbor_ids": cavm_nbr,
                "distance_context": cavm_ctx,
                "distance_w1": cavm_w1,
                "effective_neighbor_count": cavm_eff,
            }
            # P3: pre-forecast keys recorded at query time. observe_outcome()
            # consumes these verbatim — never rebuilt from revealed labels.
            evidence["context_keys"] = cavm_keys
        return evidence


def _day_view(candidate: dict, b: int) -> dict:
    """Extract a single-day view of candidate output for retrieval."""
    return {
        "w_minus": candidate["w_minus"][b:b + 1],
        "w_zero": candidate["w_zero"][b:b + 1],
        "w_plus": candidate["w_plus"][b:b + 1],
        "m_minus": candidate["m_minus"][b:b + 1],
        "m_plus": candidate["m_plus"][b:b + 1],
        "valid_mask": candidate["valid_mask"][b:b + 1],
    }


def _cavm_day_view(candidate: dict, b: int) -> dict:
    """Single-day view INCLUDING z0 (required by the CAVM context key)."""
    view = _day_view(candidate, b)
    view["z0"] = candidate["z0"][b:b + 1]
    return view


def _resolve_query_id(query_id, evidence: dict) -> Optional[int]:
    """Map a query_id to a batch index. int -> itself; str -> dates lookup."""
    if isinstance(query_id, (int, np.integer)):
        return int(query_id)
    dates = evidence.get("query_ids")
    if dates is not None and str(query_id) in dates:
        return int(dates.index(str(query_id)))
    return None


def _cavm_experience(day: dict, kb: ContextKeyBuilder) -> CAVMExperience:
    """Build one revealed-day CAVMExperience from an offline day dict.

    Pre-forecast only: context_key is built from candidate/core_context/
    domain_det BEFORE the label enters. core_context/domain_det are optional;
    when absent, the time/sig key segments fall back to zero (explicit
    degradation, never fabricated values). target_zY is required (revealed).
    """
    cand = day["candidate"]
    vm = cand["valid_mask"]
    core = day.get("core_context")
    if core is None:
        n_hours = int(np.asarray(vm).reshape(-1).shape[0])
        core = np.zeros((n_hours, kb.d_core_context))
    det = day.get("domain_det")
    key = kb.build(cand, core, vm, det)
    return CAVMExperience(
        date=str(day["date"]),
        context_key=key,
        z0=cand["z0"],
        w_minus=cand["w_minus"],
        w_zero=cand["w_zero"],
        w_plus=cand["w_plus"],
        m_minus=cand["m_minus"],
        m_plus=cand["m_plus"],
        target_zY=day["target_zY"],
        valid_mask=vm,
        audit_domain=day.get("audit_domain", ""),
    )
