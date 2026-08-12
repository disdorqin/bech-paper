"""HCH v0.4 Universal Pipeline — authoritative end-to-end orchestrator.

Owns the full v0.3 IAH decision chain in stage order:
    S1 reference -> IAH candidate -> S3-M memory/k -> S3-C DVG -> freeze -> S4 predict

This is the ONLY formal entry point. It must never import or invoke the
legacy HCH path (BiOMC, candidate_loss_fn, state_loss_fn, CARA/KL calibration).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from iah_candidate import IAHCandidateHead
from iah_crps_loss import iah_crps_loss
from s1_rank import S1RankReference
from hch_v2_context import compute_domain_descriptors
from w1_retrieval import CAGMAtomMemory
from query_replay import full_replay_chain, estimate_realized_A
from double_event import double_event_proposal
from dvg_calibrate import DGVSplitConformal
from hch_v2_bundle import HCHV2Bundle


class HCHV2UniversalPipeline:
    """Authoritative v0.4 orchestrator.

    Stage ownership is explicit; each stage mutates only its own state.
    S3-C can never mutate candidate / memory / k / proposal.
    """

    def __init__(self, d_core_context: int = 13, d_model: int = 64,
                 d_value: int = 0, alpha: float = 0.10,
                 k: Optional[int] = None, seed: int = 0):
        self.d_core_context = d_core_context
        self.d_model = d_model
        self.d_value = d_value
        self.alpha = alpha
        self.k = k
        self.seed = seed

        torch.manual_seed(seed)
        self.candidate_head = IAHCandidateHead(d_core_context, d_model,
                                               d_value=d_value)
        self.s1_rank_ref: Optional[S1RankReference] = None
        self.memory: Optional[CAGMAtomMemory] = None
        self.dvg: Optional[DGVSplitConformal] = None
        self._domain_det: Optional[np.ndarray] = None

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
    def train_candidate_s2(self, s2_batches, epochs: int = 30,
                           lr: float = 1e-3, patience: int = 8) -> float:
        """Train the assembled v0.4 core on S2 with the single IAH-CRPS objective.

        s2_batches: iterable of (host_raw [B,H,1], core_context [B,H,D],
                                 target_raw [B,H,1], valid_mask [B,H]).
        core_context excludes z0 (appended inside the head).
        """
        opt = torch.optim.AdamW(self.candidate_head.parameters(), lr=lr,
                                weight_decay=1e-4)
        best, best_state, pat = float("inf"), None, 0

        for ep in range(epochs):
            total, nb = 0.0, 0
            for host, ctx, target, vm in s2_batches:
                opt.zero_grad()
                out = self.candidate_head(host, ctx, valid_mask=vm)
                loss = iah_crps_loss(out, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.candidate_head.parameters(), 1.0)
                opt.step()
                total += loss.item()
                nb += 1
            avg = total / max(nb, 1)
            if avg < best - 1e-5:
                best, pat = avg, 0
                best_state = {k: v.clone() for k, v in
                              self.candidate_head.state_dict().items()}
            else:
                pat += 1
                if pat >= patience:
                    break
        if best_state is not None:
            self.candidate_head.load_state_dict(best_state)
        return float(best)

    # ---------------------------------------------------------- S3-M ----
    def fit_s3_memory(self, s3m_days: list) -> CAGMAtomMemory:
        """Build CAGM atom memory from S3-M days."""
        mem = CAGMAtomMemory()
        for day in s3m_days:
            mem.add_day(day["date"], day["candidate"], day["target_zY"])
        self.memory = mem
        return mem

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
        best_k, best_score = None, -float("inf")
        for k in candidate_k:
            errors = []
            for vd in validation_days:
                A_hat, realized_A = self._replay_value(vd, k)
                errors.append(abs(A_hat - realized_A))
            score = -float(np.mean(errors)) if errors else -float("inf")
            if score > best_score:
                best_score, best_k = score, k
        self.k = best_k if best_k is not None else candidate_k[0]
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
        """Freeze the full pipeline into a universal+local bundle."""
        b = HCHV2Bundle()
        b.core_model_state = {k: v.clone() for k, v in
                              self.candidate_head.state_dict().items()}
        b.core_config = {"d_core_context": self.d_core_context,
                         "d_model": self.d_model,
                         "d_value": self.d_value,
                         "seed": self.seed}
        b.source_datasets = [dataset_id] if dataset_id else []
        if self._domain_det is not None:
            b.data_signature_spec = {
                "det": self._domain_det.tolist(),
                "version": "domain-det-v1",
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
        b.local_hashes = {"split_hash": split_hash}
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

        return pipe

    # ---------------------------------------------------------- S4 ----
    def predict_s4(self, host_raw: torch.Tensor, core_context: torch.Tensor,
                   valid_mask: Optional[torch.Tensor] = None) -> dict:
        """Target-free S4 inference.

        host_raw [B,H,1], core_context [B,H,D]. No target accepted.
        Returns full per-day evidence: candidate atoms, neighbors, proposals,
        pi, A_hat, q, LCB, final raw action — enough to serialize evidence JSON.
        """
        with torch.no_grad():
            out = self.candidate_head(host_raw, core_context, valid_mask=valid_mask)

        evidence = {"candidate": out}

        if self.memory is None or self.dvg is None or self.dvg.q is None:
            B = host_raw.shape[0]
            evidence["final_action"] = ["identity"] * B
            evidence["fallback"] = "no_memory_or_calibration"
            evidence["x_final"] = out["x_identity"]
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

        for b in range(B):
            dists = self.memory.build_retrieval_index(_day_view(out, b))
            nbr = self.memory.get_neighbors(dists, self.k)
            neighbors_all.append(nbr)
            dists_all.append([float(dists[i]) for i in nbr])

            chain = full_replay_chain(
                self.memory, nbr,
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
