"""HCH v2 — Hurdle Correction Head v2.

Core: Bi-OMC + CAGM + DVG with unified semantics.
  - Candidates: y_down = host_pred + delta_down (§B1-B2)
  - State: continuous rank/scale shared context (§C1-C2)
  - Gain: relative to Identity, computed from saved candidates (§B3)
  - DVG: S3 leave-one-day-out calibration (§F2)
  - Keys: single encoder, single metric projection (§E1-E2)
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

EPS = 1e-8
DEV = torch.device("cpu")


@dataclass
class HCHV2Config:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    memory_k: int = 10
    memory_temperature: float = 0.1
    kl_tau: float = 1.0
    cara_eta: float = 1.0
    state_loss_weight: float = 0.5
    occurrence_loss_weight: float = 1.0
    magnitude_loss_weight: float = 1.0
    location_loss_weight: float = 0.3
    dropout: float = 0.1
    gate_mode: str = "soft_hard"
    seed: int = 0
    lr: float = 1e-3
    epochs: int = 30
    patience: int = 8
    n_exog_features: int = 3


# ======================== §6.1 shared context encoder ========================
class HourTokenEncoder(nn.Module):
    def __init__(self, d_model=64, d_time=7, d_feat=3, n_heads=4, dropout=0.1):
        super().__init__()
        self.y_proj = nn.Linear(1, d_model)
        self.t_proj = nn.Linear(d_time, d_model)
        self.exog_embed = nn.Linear(d_feat, d_model)
        self.var_type = nn.Embedding(32, d_model)  # variable type identity
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.null_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, host_pred, time_feat, exog, exog_mask, exog_type=None):
        B, H = host_pred.shape[:2]
        h = self.y_proj(host_pred) + self.t_proj(time_feat)
        N_exog = exog.shape[2]
        if N_exog > 0 and exog_mask.sum() > 0:
            e_flat = exog.reshape(B * H, N_exog, -1)
            e_emb = self.exog_embed(e_flat)
            if exog_type is not None:
                et = exog_type.reshape(B * H, N_exog).long()
                e_emb = e_emb + self.var_type(et)
            q_flat = h.reshape(B * H, 1, h.shape[-1])
            key_mask = (1.0 - exog_mask.reshape(B * H, N_exog)).bool()
            a_out, _ = self.cross_attn(q_flat, e_emb, e_emb, key_padding_mask=key_mask)
            h = h + a_out.reshape(B, H, -1)
        else:
            null = self.null_token.expand(B * H, 1, -1)
            q_flat = h.reshape(B * H, 1, h.shape[-1])
            a_out, _ = self.cross_attn(q_flat, null, null)
            h = h + a_out.reshape(B, H, -1)
        return self.norm(self.dropout(h))


class DayEncoder(nn.Module):
    def __init__(self, d_model=64, n_layers=2, dropout=0.1):
        super().__init__()
        el = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model * 2,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, n_layers)

    def forward(self, h):
        return self.enc(h)


# ======================== §6.2-6.3 continuous state =========================
class ContinuousStateHead(nn.Module):
    def __init__(self, d_model=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
        )
        self.rank_head = nn.Sequential(nn.Linear(d_model // 2, 1), nn.Tanh())
        self.scale_head = nn.Sequential(nn.Linear(d_model // 2, 1), nn.Softplus())
        self.state_dim = 2

    def forward(self, z):
        f = self.net(z)
        s_rank = self.rank_head(f)       # [-1, 1]
        s_scale = self.scale_head(f)      # [0, inf)
        return torch.cat([s_rank, s_scale], dim=-1)  # [B, H, 2]


# ======================== §7 Bi-OMC with host-based candidates ===============
class BiOMC(nn.Module):
    def __init__(self, d_model=64, state_dim=2, dropout=0.1):
        super().__init__()
        inp_dim = d_model + state_dim + d_model  # z + state + direction_token
        self.down_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.up_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.body = nn.Sequential(
            nn.Linear(inp_dim, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
        )
        self.occ_d = nn.Sequential(nn.Linear(d_model // 2, 1), nn.Sigmoid())
        self.mag_d = nn.Sequential(nn.Linear(d_model // 2, 1), nn.Softplus())
        self.occ_u = nn.Sequential(nn.Linear(d_model // 2, 1), nn.Sigmoid())
        self.mag_u = nn.Sequential(nn.Linear(d_model // 2, 1), nn.Softplus())

    def forward(self, z, state):
        B, H, _ = z.shape
        ctx = torch.cat([z, state], dim=-1)
        zd = torch.cat([ctx, self.down_tok.expand(B, H, -1)], -1)
        zu = torch.cat([ctx, self.up_tok.expand(B, H, -1)], -1)
        fd, fu = self.body(zd), self.body(zu)
        pd, md = self.occ_d(fd), self.mag_d(fd)
        pu, mu = self.occ_u(fu), self.mag_u(fu)
        dd = -pd * md
        du = pu * mu
        return {"p_down": pd, "m_down": md, "p_up": pu, "m_up": mu,
                "delta_down": dd, "delta_up": du}


def build_candidates(host_pred, delta_down, delta_up):
    """§B2: single source of truth for all three actions."""
    return {
        "identity": host_pred,
        "down": host_pred + delta_down,
        "up": host_pred + delta_up,
    }


def compute_action_gain(target, host_pred, y_down, y_up):
    """§B3: gain = MAE_reduction relative to Identity."""
    base = (target - host_pred).abs()
    g_down = base - (target - y_down).abs()
    g_up = base - (target - y_up).abs()
    return torch.stack([torch.zeros_like(base), g_down, g_up], dim=-1)


# ======================== §8 CAGM with state-aware key ======================
class CAGMMemory(nn.Module):
    def __init__(self, d_model=64, state_dim=2, memory_k=10, temperature=0.1):
        super().__init__()
        self.memory_k = memory_k
        self.temperature = temperature
        in_dim = d_model + state_dim
        self.key_net = nn.Sequential(nn.Linear(in_dim, d_model), nn.ReLU(),
                                      nn.Linear(d_model, d_model))
        self.cand_proj = nn.Sequential(nn.Linear(4, d_model // 2), nn.ReLU(),
                                        nn.Linear(d_model // 2, d_model // 2))
        self.fusion = nn.Linear(d_model + d_model // 2, d_model)
        self.metric_proj = nn.Linear(d_model, d_model)
        self.register_buffer("m_keys", None, persistent=False)
        self.register_buffer("m_gains", None, persistent=False)
        self._m_dates = None

    def encode_key(self, z, state, dd, du):
        zp = z.mean(dim=1)
        sp = state.mean(dim=1)
        zp = torch.cat([zp, sp], dim=-1)
        cf = torch.cat([dd.mean(1, keepdim=True), dd.std(1, keepdim=True),
                        du.mean(1, keepdim=True), du.std(1, keepdim=True)], dim=1)
        ce = self.cand_proj(cf)
        k = self.fusion(torch.cat([self.key_net(zp), ce.detach()], -1))
        return F.normalize(self.metric_proj(k), dim=-1)

    def encode_raw(self, z, state, dd, du):
        """Pre-projection key for unified metric space (§E2)."""
        zp = z.mean(dim=1)
        sp = state.mean(dim=1)
        zp = torch.cat([zp, sp], dim=-1)
        cf = torch.cat([dd.mean(1, keepdim=True), dd.std(1, keepdim=True),
                        du.mean(1, keepdim=True), du.std(1, keepdim=True)], dim=1)
        ce = self.cand_proj(cf)
        k = self.fusion(torch.cat([self.key_net(zp), ce.detach()], -1))
        return k  # unnormalized, pre-projection

    def project_metric(self, raw_keys):
        """Apply metric projection exactly once (§E2)."""
        return F.normalize(self.metric_proj(raw_keys), dim=-1)

    def build(self, keys, gains, dates):
        self.m_keys = F.normalize(keys, dim=-1)
        self.m_gains = gains.clone()
        self._m_dates = list(dates) if not isinstance(dates, list) else dates

    def retrieve(self, qk, exclude_idx=None):
        if self.m_keys is None or len(self.m_keys) == 0:
            return None, None, None
        q = F.normalize(qk, dim=-1)
        sim = torch.mm(q, self.m_keys.T)
        if exclude_idx is not None:
            sim.scatter_(1, exclude_idx, -1e9)  # §F2: exclude self
        k = min(self.memory_k, sim.shape[1])
        tv, ti = sim.topk(k, dim=-1)
        w = F.softmax(tv / self.temperature, dim=-1)
        g = self.m_gains[ti]
        return w, g, ti

    def state_dict(self, *args, **kwargs):
        d = super().state_dict(*args, **kwargs)
        if self.m_keys is not None:
            d["_m_keys"] = self.m_keys
            d["_m_gains"] = self.m_gains
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        prefix = kwargs.pop("_prefix", "")
        m_keys = state_dict.pop("_m_keys", None)
        m_gains = state_dict.pop("_m_gains", None)
        super().load_state_dict(state_dict, *args, **kwargs)
        if m_keys is not None:
            self.m_keys = m_keys
        if m_gains is not None:
            self.m_gains = m_gains


# ======================== §9 DVG with LODO calibration ======================
class DVG(nn.Module):
    def __init__(self, cara_eta=1.0, kl_tau=1.0, gate_mode="soft_hard"):
        super().__init__()
        self.cara_eta = cara_eta
        self.kl_tau = kl_tau
        self.gate_mode = gate_mode

    def set_mode(self, mode):
        self.gate_mode = mode

    def calibrate(self, eta, tau, mode):
        """§F2: set parameters calibrated on S3."""
        self.cara_eta = eta
        self.kl_tau = tau
        self.gate_mode = mode

    def forward(self, weights, gains):
        if weights is None or gains is None:
            return None
        B, K = weights.shape
        if gains.dim() == 4:
            g = gains
        elif gains.dim() == 3:
            g = gains.unsqueeze(0)
            weights = weights[:1, :K]
        else:
            return None
        w = weights.reshape(-1, K, 1, 1)
        num = (w * torch.exp(-self.cara_eta * g)).sum(dim=1)
        den = w.sum(dim=1).clamp(min=EPS)
        ce = -torch.log(num / den + EPS) / self.cara_eta
        probs = F.softmax(ce / self.kl_tau, dim=-1)
        if self.gate_mode == "soft_hard":
            ch = torch.argmax(ce, dim=-1)
            am = F.one_hot(ch, 3).float()
            return {"action_prob": probs, "chosen_action": ch,
                    "action_value": ce, "action_mask": am}
        return {"action_prob": probs, "chosen_action": None,
                "action_value": ce, "action_mask": probs}


# ======================== state loss (§C2) ==================================
def state_loss_fn(s_pred, s_target):
    """MSE on continuous rank/scale targets."""
    return F.mse_loss(s_pred, s_target)


def compute_state_targets(y, s1_cdf, s1_median, s1_mad):
    """§C1: compute rank and scale targets from S1 stats.

    s_rank = 2*CDF_S1(y) - 1  ∈ [-1, 1]
    s_scale = log(1 + |y - median| / MAD)
    """
    import numpy as np
    yn = y.detach().cpu().numpy().ravel()
    rank = np.array([s1_cdf(v) for v in yn])
    rank = 2.0 * rank - 1.0
    scale = np.log1p(np.abs(yn - s1_median) / (s1_mad + 1e-8))
    return torch.tensor(np.stack([rank, scale], -1).reshape(y.shape[0], y.shape[1], 2),
                        dtype=torch.float32)


# ======================== candidate loss (§7) ===============================
def candidate_loss_fn(cand, target, host_pred, cfg):
    resid = target - host_pred
    r_d = F.relu(-resid).squeeze(-1)
    r_u = F.relu(resid).squeeze(-1)
    o_d = (resid < 0).float().squeeze(-1)
    o_u = (resid > 0).float().squeeze(-1)

    loss_occ = F.binary_cross_entropy(cand["p_down"].squeeze(-1), o_d) \
               + F.binary_cross_entropy(cand["p_up"].squeeze(-1), o_u)

    loss_mag = torch.tensor(0.0)
    n = 0
    for om, mp, mt in [(o_d > 0.5, cand["m_down"].squeeze(-1), r_d),
                       (o_u > 0.5, cand["m_up"].squeeze(-1), r_u)]:
        if om.sum() > 0:
            loss_mag = loss_mag + F.smooth_l1_loss(mp * om.float(), mt * om.float(),
                                                    reduction="sum") / om.sum().clamp(min=1)
            n += 1
    if n:
        loss_mag = loss_mag / n

    def _w1(p, q):
        return (torch.cumsum(p, -1) - torch.cumsum(q, -1)).abs().sum(-1)

    loss_loc = torch.tensor(0.0)
    for r_raw, d_raw in [(r_d, -cand["delta_down"].squeeze(-1)),
                         (r_u, cand["delta_up"].squeeze(-1))]:
        rs = r_raw.sum(-1).clamp(min=1e-8)
        mask = (rs > 1e-8).float()
        if mask.sum() == 0:
            continue
        ds = d_raw.abs().sum(-1).clamp(min=1e-8)
        loss_loc = loss_loc + (_w1(r_raw / rs.unsqueeze(-1),
                                    d_raw.abs() / ds.unsqueeze(-1))
                               * mask).sum() / mask.sum().clamp(min=1)

    return (cfg.occurrence_loss_weight * loss_occ
            + cfg.magnitude_loss_weight * loss_mag
            + cfg.location_loss_weight * loss_loc), {
        "occ": loss_occ.item(), "mag": loss_mag.item(), "loc": loss_loc.item(),
    }


# ======================== §G freeze bundle ==================================
class HCHV2Bundle:
    """Persistable bundle for freeze/reload (§G1-G3)."""

    def __init__(self):
        self.config = None
        self.model_state = None
        self.memory_keys = None
        self.memory_gains = None
        self.memory_dates = None
        self.s1_stats = None
        self.commit = None
        self.extra = {}

    def save(self, path):
        torch.save({
            "config": self.config,
            "model_state": self.model_state,
            "memory_keys": self.memory_keys,
            "memory_gains": self.memory_gains,
            "memory_dates": self.memory_dates,
            "s1_stats": self.s1_stats,
            "commit": self.commit,
            "extra": self.extra,
        }, path)

    @staticmethod
    def load(path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        b = HCHV2Bundle()
        b.config = data["config"]
        b.model_state = data["model_state"]
        b.memory_keys = data["memory_keys"]
        b.memory_gains = data["memory_gains"]
        b.memory_dates = data["memory_dates"]
        b.s1_stats = data["s1_stats"]
        b.commit = data["commit"]
        b.extra = data.get("extra", {})
        return b

    def hash(self):
        h = hashlib.sha256()
        for k in sorted(self.model_state.keys()):
            h.update(self.model_state[k].cpu().numpy().tobytes())
        return h.hexdigest()[:16]


# ======================== full HCHV2 model (§G2 freeze API) =================
class HCHV2(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or HCHV2Config()
        cfg = self.config
        sd = 2  # state_dim
        self.tok_enc = HourTokenEncoder(cfg.d_model, d_feat=cfg.n_exog_features,
                                         n_heads=cfg.n_heads, dropout=cfg.dropout)
        self.day_enc = DayEncoder(cfg.d_model, cfg.n_layers, cfg.dropout)
        self.state_head = ContinuousStateHead(cfg.d_model, cfg.dropout)
        self.biomc = BiOMC(cfg.d_model, state_dim=sd, dropout=cfg.dropout)
        self.memory = CAGMMemory(cfg.d_model, state_dim=sd,
                                  memory_k=cfg.memory_k,
                                  temperature=cfg.memory_temperature)
        self.dvg = DVG(cfg.cara_eta, cfg.kl_tau, cfg.gate_mode)
        self.built = False

    def encode(self, batch):
        h = self.tok_enc(batch.host_pred, batch.time_feat, batch.exog, batch.exog_mask)
        z = self.day_enc(h)
        state = self.state_head(z)
        return z, state

    def forward(self, batch):
        yhat = batch.host_pred.squeeze(-1)  # [B, H]
        z, state = self.encode(batch)
        cand = self.biomc(z, state)

        if not self.built:
            return {"y_base": yhat,
                    "y_down": yhat + cand["delta_down"].squeeze(-1),
                    "y_up": yhat + cand["delta_up"].squeeze(-1),
                    "delta_down": cand["delta_down"].squeeze(-1),
                    "delta_up": cand["delta_up"].squeeze(-1),
                    "state": state, "y_final": yhat}

        dk = self.memory.encode_key(z, state,
                                     cand["delta_down"].squeeze(-1),
                                     cand["delta_up"].squeeze(-1))
        w, g, ti = self.memory.retrieve(dk)
        gate = self.dvg(w, g)
        dd = cand["delta_down"].squeeze(-1)
        du = cand["delta_up"].squeeze(-1)
        if gate is not None:
            yf = yhat + gate["action_mask"][:, :, 1] * dd \
                      + gate["action_mask"][:, :, 2] * du
        else:
            yf = yhat
        out = {"y_base": yhat, "y_down": yhat + dd, "y_up": yhat + du,
               "delta_down": dd, "delta_up": du, "state": state, "y_final": yf}
        if gate is not None:
            out.update({"action_value": gate["action_value"],
                        "action_prob": gate["action_prob"],
                        "chosen_action": gate["chosen_action"],
                        "retrieval_idx": ti})
        return out

    # ======================== freeze API (§G2) ==============================
    def freeze(self, s1_stats=None, commit=None) -> HCHV2Bundle:
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        b = HCHV2Bundle()
        b.config = self.config
        sd = self.state_dict()
        for k in ["_m_keys", "_m_gains"]:
            sd.pop(k, None)
        b.model_state = {k: v.clone() for k, v in sd.items()}
        if self.memory.m_keys is not None:
            b.memory_keys = self.memory.m_keys.clone()
            b.memory_gains = self.memory.m_gains.clone()
            b.memory_dates = list(self.memory._m_dates) if self.memory._m_dates else []
        b.s1_stats = s1_stats
        b.commit = commit
        return b

    def calibrate_s3(self, s3_loader, s1_stats):
        """§F2: S3 leave-one-day-out grid search over k/eta/tau."""
        self.eval()
        K_grid = [5, 10, 15, 20]
        ETA_grid = [0.1, 0.5, 1.0, 2.0]
        TAU_grid = [0.2, 0.5, 1.0]

        all_keys, all_gains, all_dates = [], [], []
        with torch.no_grad():
            for batch in s3_loader:
                batch = _to_device(batch)
                z, state = self.encode(batch)
                cand = self.biomc(z, state)
                dd = cand["delta_down"].squeeze(-1)
                du = cand["delta_up"].squeeze(-1)
                yh = batch.host_pred.squeeze(-1)
                yt = batch.target.squeeze(-1)
                gain = compute_action_gain(yt, yh, yh + dd, yh + du)
                k_raw = self.memory.encode_raw(z, state, dd, du)
                all_keys.append(k_raw.cpu())
                all_gains.append(gain.cpu())
                if isinstance(batch.date_ids, list):
                    all_dates.extend(batch.date_ids)

        keys_pool = torch.cat(all_keys)
        gains_pool = torch.cat(all_gains)
        n_days = len(keys_pool)
        if n_days < 5:
            return

        best_score, best_cfg = -float("inf"), None
        for K in K_grid:
            self.memory.memory_k = K
            for eta in ETA_grid:
                self.dvg.cara_eta = eta
                for tau in TAU_grid:
                    self.dvg.kl_tau = tau
                    pos_gains = []
                    for i in range(n_days):
                        excl = torch.tensor([[i]])
                        q = self.memory.project_metric(keys_pool[i:i + 1])
                        mem_bak = self.memory.m_keys, self.memory.m_gains
                        self.memory.m_keys = F.normalize(self.memory.project_metric(keys_pool), dim=-1)
                        self.memory.m_gains = gains_pool
                        w, g, _ = self.memory.retrieve(q, exclude_idx=excl)
                        self.memory.m_keys, self.memory.m_gains = mem_bak
                        if w is None:
                            continue
                        gate = self.dvg(w, g)
                        if gate is None:
                            continue
                        ce = gate["action_value"]
                        best_a = ce.argmax(dim=-1)
                        gain_i = g[0, :, :, :].mean(dim=0)
                        chosen_gain = gain_i[range(24), best_a[0]].mean().item()
                        pos_gains.append(chosen_gain)
                    if pos_gains:
                        score = np.mean([g for g in pos_gains if g > 0]) if any(g > 0 for g in pos_gains) else 0.0
                        if score > best_score:
                            best_score = score
                            best_cfg = (K, eta, tau, self.dvg.gate_mode)

        self.config.memory_k = best_cfg[0]
        self.dvg.cara_eta = best_cfg[1]
        self.dvg.kl_tau = best_cfg[2]
        self.dvg.gate_mode = best_cfg[3]
        self.memory.memory_k = best_cfg[0]
        return best_cfg

    @classmethod
    def from_bundle(cls, bundle: HCHV2Bundle) -> "HCHV2":
        model = cls(bundle.config)
        model.load_state_dict(bundle.model_state)
        if bundle.memory_keys is not None:
            model.memory.m_keys = bundle.memory_keys.clone()
            model.memory.m_gains = bundle.memory_gains.clone()
            model.memory._m_dates = list(bundle.memory_dates)
        model.built = True
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model


def _to_device(batch):
    from hch_v2_data import DailyEpisodeBatch
    return DailyEpisodeBatch(
        host_pred=batch.host_pred, target=batch.target,
        exog=batch.exog, exog_mask=batch.exog_mask,
        time_feat=batch.time_feat, date_ids=batch.date_ids,
    )
