"""HCH v2 core model — Bi-OMC + CAGM + DVG."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

EPS = 1e-8


@dataclass
class HCHV2Config:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_exog: int = 16
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


class HourTokenEncoder(nn.Module):
    def __init__(self, d_model=64, d_time=7, d_exog=16, n_heads=4, dropout=0.1):
        super().__init__()
        self.y_proj = nn.Linear(1, d_model)
        self.t_proj = nn.Linear(d_time, d_model)
        self.exog_proj = nn.Linear(3, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, host_pred, time_feat, exog, exog_mask):
        B, H = host_pred.shape[:2]
        h = self.y_proj(host_pred) + self.t_proj(time_feat)
        N = exog.shape[2]
        if N > 0 and exog_mask.sum() > 0:
            e_flat = self.exog_proj(exog.reshape(B * H, N, 3))
            q_flat = h.reshape(B * H, 1, h.shape[-1])
            a_out, _ = self.cross_attn(q_flat, e_flat, e_flat)
            h = h + a_out.reshape(B, H, -1)
        return self.norm(self.dropout(h))


class DayEncoder(nn.Module):
    def __init__(self, d_model=64, n_layers=2, dropout=0.1):
        super().__init__()
        el = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model*2,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, n_layers)

    def forward(self, h):
        return self.enc(h)


class ContinuousStateHead(nn.Module):
    def __init__(self, d_model=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2), nn.ReLU())
        self.rank_head = nn.Sequential(nn.Linear(d_model//2, 1), nn.Tanh())
        self.zero_head = nn.Linear(d_model//2, 1)

    def forward(self, z):
        f = self.net(z)
        return self.rank_head(f), self.zero_head(f)


class BiOMC(nn.Module):
    def __init__(self, d_model=64, dropout=0.1):
        super().__init__()
        self.down_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.up_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.body = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2), nn.ReLU())
        self.occ_d = nn.Sequential(nn.Linear(d_model//2, 1), nn.Sigmoid())
        self.mag_d = nn.Sequential(nn.Linear(d_model//2, 1), nn.Softplus())
        self.occ_u = nn.Sequential(nn.Linear(d_model//2, 1), nn.Sigmoid())
        self.mag_u = nn.Sequential(nn.Linear(d_model//2, 1), nn.Softplus())

    def forward(self, z):
        B, H, D = z.shape
        zd = torch.cat([z, self.down_tok.expand(B, H, -1)], -1)
        zu = torch.cat([z, self.up_tok.expand(B, H, -1)], -1)
        fd, fu = self.body(zd), self.body(zu)
        pd, md = self.occ_d(fd), self.mag_d(fd)
        pu, mu = self.occ_u(fu), self.mag_u(fu)
        dd = -pd * md
        du = pu * mu
        return {"p_down": pd, "m_down": md, "p_up": pu, "m_up": mu,
                "delta_down": dd, "delta_up": du,
                "y_down": z[:,:,:1] + dd, "y_up": z[:,:,:1] + du}


class CAGMMemory(nn.Module):
    def __init__(self, d_model=64, memory_k=10, temperature=0.1):
        super().__init__()
        self.memory_k = memory_k
        self.temperature = temperature
        self.key_net = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                      nn.Linear(d_model, d_model))
        self.cand_proj = nn.Sequential(nn.Linear(4, d_model//2), nn.ReLU(),
                                        nn.Linear(d_model//2, d_model//2))
        self.fusion = nn.Linear(d_model + d_model//2, d_model)
        self.metric_proj = nn.Linear(d_model, d_model)  # §8.3 gain-aware projection
        self.register_buffer("m_keys", None, persistent=False)
        self.register_buffer("m_gains", None, persistent=False)
        self._m_dates = None

    def encode_key(self, z, dd, du):
        zp = z.mean(dim=1)
        cf = torch.cat([dd.mean(1, keepdim=True), dd.std(1, keepdim=True),
                        du.mean(1, keepdim=True), du.std(1, keepdim=True)], dim=1)
        ce = self.cand_proj(cf)
        k = self.fusion(torch.cat([self.key_net(zp), ce.detach()], -1))
        return self.metric_proj(F.normalize(k, dim=-1))

    def build(self, keys, gains, dates):
        self.m_keys = F.normalize(keys, dim=-1)
        self.m_gains = gains.clone()
        self._m_dates = list(dates) if not isinstance(dates, list) else dates

    def retrieve(self, qk):
        if self.m_keys is None or len(self.m_keys) == 0:
            return None, None, None
        sim = torch.mm(F.normalize(qk, dim=-1), self.m_keys.T)
        k = min(self.memory_k, len(self.m_keys))
        tv, ti = sim.topk(k, dim=-1)
        w = F.softmax(tv / self.temperature, dim=-1)
        g = self.m_gains[ti]
        return w, g, ti


class DVG(nn.Module):
    def __init__(self, cara_eta=1.0, kl_tau=1.0, gate_mode="soft_hard"):
        super().__init__()
        self.cara_eta = cara_eta
        self.kl_tau = kl_tau
        self.gate_mode = gate_mode

    def set_mode(self, mode):
        self.gate_mode = mode

    def forward(self, weights, gains):
        if weights is None or gains is None:
            return None
        B, K = weights.shape
        if gains.dim() == 4:
            g = gains
        elif gains.dim() == 3:
            g = gains.unsqueeze(0)
            weights = weights[:1]
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


class HCHV2(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or HCHV2Config()
        cfg = self.config
        self.tok_enc = HourTokenEncoder(cfg.d_model, d_exog=cfg.d_exog,
                                         n_heads=cfg.n_heads, dropout=cfg.dropout)
        self.day_enc = DayEncoder(cfg.d_model, cfg.n_layers, cfg.dropout)
        self.state = ContinuousStateHead(cfg.d_model, cfg.dropout)
        self.biomc = BiOMC(cfg.d_model, cfg.dropout)
        self.memory = CAGMMemory(cfg.d_model, cfg.memory_k, cfg.memory_temperature)
        self.dvg = DVG(cfg.cara_eta, cfg.kl_tau, cfg.gate_mode)
        self.built = False

    def encode(self, batch):
        h = self.tok_enc(batch.host_pred, batch.time_feat, batch.exog, batch.exog_mask)
        z = self.day_enc(h)
        sr, sz = self.state(z)
        return z, sr, sz

    def forward(self, batch):
        z, sr, sz = self.encode(batch)
        cand = self.biomc(z)
        yhat = batch.host_pred.squeeze(-1)
        if not self.built:
            return {"y_base": yhat, "y_down": cand["y_down"].squeeze(-1),
                    "y_up": cand["y_up"].squeeze(-1),
                    "delta_down": cand["delta_down"].squeeze(-1),
                    "delta_up": cand["delta_up"].squeeze(-1),
                    "state_low": sr.squeeze(-1), "y_final": yhat}
        dk = self.memory.encode_key(z, cand["delta_down"].squeeze(-1),
                                     cand["delta_up"].squeeze(-1))
        w, g, _ = self.memory.retrieve(dk)
        gate = self.dvg(w, g)
        if gate is not None:
            yf = yhat + gate["action_mask"][:,:,1] * cand["delta_down"].squeeze(-1) \
                      + gate["action_mask"][:,:,2] * cand["delta_up"].squeeze(-1)
        else:
            yf = yhat
        out = {"y_base": yhat, "y_down": cand["y_down"].squeeze(-1),
               "y_up": cand["y_up"].squeeze(-1),
               "delta_down": cand["delta_down"].squeeze(-1),
               "delta_up": cand["delta_up"].squeeze(-1),
               "state_low": sr.squeeze(-1), "y_final": yf}
        if gate is not None:
            out.update({"action_value": gate.get("action_value"),
                        "action_prob": gate.get("action_prob"),
                        "chosen_action": gate.get("chosen_action")})
        return out

    # ============================================ §5.1 cross-fitting =====
    def cross_fit_s2(self, block_loaders, cfg):
        """Blocked forward cross-fitting on S2.

        For each block i: train temp Bi-OMC on blocks 0..i-1,
        generate OOF candidates/keys/gains on block i.
        Returns (oof_keys, oof_gains, oof_dates) for gain-aware metric training.
        """
        oof_keys, oof_gains, oof_dates = [], [], []
        cum_blocks = []

        for i, loader in enumerate(block_loaders):
            if i == 0:
                # first block: no previous data, skip OOF
                cum_blocks.append(loader)
                continue

            # train temp model on all previous blocks
            self.train()
            opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr, weight_decay=1e-4)
            best, best_state, patience = float("inf"), None, 0

            for ep in range(cfg.epochs // 2):  # fewer epochs per block
                total = 0.0
                for prev_loader in cum_blocks:
                    for batch in prev_loader:
                        batch = _to_device(batch)
                        z, sr, sz = self.encode(batch)
                        cand = self.biomc(z)
                        loss = _candidate_loss(cand, batch.target, batch.host_pred, cfg)
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                        opt.step()
                        total += loss.item()
                avg = total / max(sum(len(l) for l in cum_blocks), 1)
                if avg < best - 1e-5:
                    best, patience = avg, 0
                    best_state = {k: v.clone() for k, v in self.state_dict().items()}
                else:
                    patience += 1
                    if patience >= cfg.patience // 2:
                        break
            if best_state:
                self.load_state_dict(best_state)

            self.eval()
            with torch.no_grad():
                for batch in loader:
                    batch = _to_device(batch)
                    z, sr, sz = self.encode(batch)
                    cand = self.biomc(z)
                    dk = self.memory.encode_key(
                        z, cand["delta_down"].squeeze(-1), cand["delta_up"].squeeze(-1),
                    )
                    base_ae = (batch.target.squeeze(-1) - batch.host_pred.squeeze(-1)).abs()
                    down_ae = (batch.target.squeeze(-1) - cand["y_down"].squeeze(-1)).abs()
                    up_ae = (batch.target.squeeze(-1) - cand["y_up"].squeeze(-1)).abs()
                    g = torch.stack([torch.zeros_like(base_ae), base_ae - down_ae,
                                     base_ae - up_ae], dim=-1)
                    oof_keys.append(dk.cpu())
                    oof_gains.append(g.cpu())
                    if hasattr(batch, "date_ids") and isinstance(batch.date_ids, list):
                        oof_dates.extend(batch.date_ids)

            cum_blocks.append(loader)

        if oof_keys:
            return torch.cat(oof_keys), torch.cat(oof_gains), oof_dates
        return None, None, None

    # ============================================ §8.3 gain-aware metric ==
    def train_gain_metric(self, oof_keys, oof_gains, cfg, epochs=10):
        """§8.3 Train metric_proj so similar gain profiles yield similar keys.

        Target:  softmax(-|G_i - G_j| / tau) from OOF gains
        Pred:    softmax(K_i @ K_j^T / tau)  from projected keys
        Loss:    L2 between target and pred similarity matrices.
        """
        if oof_keys is None or len(oof_keys) < 2:
            return

        n = min(len(oof_keys), 400)
        idx = torch.randperm(len(oof_keys))[:n]
        K0 = oof_keys[idx]
        G = oof_gains[idx]
        G_flat = G.reshape(n, -1)
        tau = 0.1

        D_g = torch.cdist(G_flat, G_flat)
        target = F.softmax(-D_g / tau, dim=-1).detach()

        opt = torch.optim.Adam(self.memory.metric_proj.parameters(), lr=1e-4)
        for ep in range(epochs):
            K = self.memory.metric_proj(K0)
            K = F.normalize(K, dim=-1)
            sim = torch.mm(K, K.T) / tau
            pred = F.softmax(sim, dim=-1)
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ============================================ fit final Bi-OMC =======
    def fit_final_biomc(self, s2_loader, cfg):
        """Train Bi-OMC on all S2 data (final model)."""
        self.train()
        opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr, weight_decay=1e-4)
        best, best_state, patience = float("inf"), None, 0

        for ep in range(cfg.epochs):
            total = 0.0
            for batch in s2_loader:
                batch = _to_device(batch)
                z, sr, sz = self.encode(batch)
                cand = self.biomc(z)
                loss = _candidate_loss(cand, batch.target, batch.host_pred, cfg)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
                total += loss.item()
            avg = total / max(len(s2_loader), 1)
            if avg < best - 1e-5:
                best, patience = avg, 0
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg.patience:
                    break
        if best_state:
            self.load_state_dict(best_state)
        return best


def _to_device(batch):
    from hch_v2_data import DailyEpisodeBatch
    DEV = torch.device("cpu")
    return DailyEpisodeBatch(
        host_pred=batch.host_pred.to(DEV),
        target=batch.target.to(DEV),
        exog=batch.exog.to(DEV),
        exog_mask=batch.exog_mask.to(DEV),
        time_feat=batch.time_feat.to(DEV),
        date_ids=batch.date_ids,
    )


def _candidate_loss(cand, target, host_pred, cfg):
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
    eps = 1e-8
    for r_raw, d_raw in [(r_d, -cand["delta_down"].squeeze(-1)),
                         (r_u, cand["delta_up"].squeeze(-1))]:
        rs = r_raw.sum(-1).clamp(min=eps)
        mask = (rs > eps).float()
        if mask.sum() == 0:
            continue
        ds = d_raw.abs().sum(-1).clamp(min=eps)
        loss_loc = loss_loc + (_w1(r_raw / rs.unsqueeze(-1),
                                    d_raw.abs() / ds.unsqueeze(-1))
                               * mask).sum() / mask.sum().clamp(min=1)

    return (cfg.occurrence_loss_weight * loss_occ
            + cfg.magnitude_loss_weight * loss_mag
            + cfg.location_loss_weight * loss_loc)
