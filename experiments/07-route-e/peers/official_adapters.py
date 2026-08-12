"""Baseline adapters for HCH v2 — limited reimplementations.

implementation_status:
  delta-Adapter: limited_reimplementation (adapted sample count, normalization, training)
  PIR:           limited_reimplementation (no official retrieval, adapted refiner)

Sources:
  δ-Adapter: Anoise/Adapter @ 0add06e — PostY architecture from exp_decom9_post_y.py
  PIR:       ustc-time-series/PIR @ fc372bb — QualityEstimator + refiner concept

Each adapter follows the unified interface:
  fit(Z, yhat, y)       # train on S2
  predict(Z, yhat)      # generate corrected predictions on S4
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================== δ-Adapter (Official) ==============================
class PostY(nn.Module):
    """Official δ-Adapter output-side corrector (Ada-Y).

    Extracted from vendor/delta_adapter/AdaIntpX/experiments/exp_decom9_post_y.py
    Architecture: 3-layer MLP with BatchNorm, output replaces backbone prediction.

    In the paper, the model learns to directly predict y from the frozen backbone's
    prediction ŷ, after instance normalization. This is NOT residual correction —
    it replaces the backbone output entirely.
    """

    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        return self.fc3(x)


class DeltaAdapterLimited:
    """δ-Adapter Ada-Y official implementation adapter.

    Trains a PostY MLP on S2 (backbone predictions → corrected predictions).
    Uses instance normalization as in the original paper.
    """

    name = "delta-Adapter"

    def __init__(self, hidden_dim=128, epochs=30, lr=1e-3, seed=0):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.model = None

    def fit(self, Z, yhat, y):
        torch.manual_seed(self.seed)
        n = len(yhat)
        if n < 50:
            return

        cut = int(n * 0.8)
        tr_idx = np.arange(cut)
        va_idx = np.arange(cut, n)

        yhat_t = torch.tensor(yhat, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        self.model = PostY(1, self.hidden_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_loss, best_state, patience = float("inf"), None, 0
        for ep in range(self.epochs):
            self.model.train()
            yhat_tr = yhat_t[tr_idx].unsqueeze(-1)
            y_tr = y_t[tr_idx].unsqueeze(-1)
            mean = yhat_tr.mean()
            std = yhat_tr.std().clamp(min=1e-5)
            yhat_norm = (yhat_tr - mean) / std
            pred_norm = self.model(yhat_norm)
            pred = pred_norm * std + mean
            loss = F.mse_loss(pred.squeeze(-1), y_tr.squeeze(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

            self.model.eval()
            with torch.no_grad():
                yhat_va = yhat_t[va_idx].unsqueeze(-1)
                y_va = y_t[va_idx].unsqueeze(-1)
                mean_v = yhat_va.mean()
                std_v = yhat_va.std().clamp(min=1e-5)
                pred_v = self.model((yhat_va - mean_v) / std_v) * std_v + mean_v
                val_loss = F.mse_loss(pred_v.squeeze(-1), y_va.squeeze(-1)).item()
            if val_loss < best_loss - 1e-6:
                best_loss, patience = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
                if patience >= 8:
                    break
        if best_state:
            self.model.load_state_dict(best_state)

    def predict(self, Z, yhat):
        if self.model is None:
            return yhat.copy()
        self.model.eval()
        with torch.no_grad():
            yhat_t = torch.tensor(yhat, dtype=torch.float32).unsqueeze(-1)
            mean = yhat_t.mean()
            std = yhat_t.std().clamp(min=1e-5)
            pred = self.model((yhat_t - mean) / std) * std + mean
            return pred.squeeze(-1).numpy()


# ============================== PIR (Official) ==============================
class QualityEstimator(nn.Module):
    """Quality estimator from official PIR (models/PIR.py:QualityEstimator).

    Estimates per-sample weights alpha/beta to blend backbone prediction
    with refiner output and retrieval results.

    Extracted from vendor/PIR/models/PIR.py.
    """

    def __init__(self, seq_len, pred_len, d_model=64, d_ff=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(3, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_ff,
                                       dropout=dropout, batch_first=True),
            num_layers=1,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x_enc, intermediate_results, sims):
        x_m = x_enc.mean(dim=-1) if x_enc.dim() > 1 else x_enc
        i_m = intermediate_results.mean(dim=-1) if intermediate_results.dim() > 1 else intermediate_results
        sims = sims.reshape(-1) if sims.dim() > 1 else sims
        feat = torch.stack([x_m, i_m, sims], dim=-1)
        feat = self.proj(feat)
        feat = self.encoder(feat.unsqueeze(0)).squeeze(0)
        weights = self.head(feat)
        alpha = torch.sigmoid(weights[:, 0])
        beta = torch.sigmoid(weights[:, 1])
        return alpha, beta


class Refiner(nn.Module):
    """Refiner module from official PIR.

    Lightweight Transformer that refines backbone predictions using
    time features and instance normalization.
    """

    def __init__(self, d_model=64, d_ff=128, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_ff,
                                       dropout=dropout, batch_first=True),
            num_layers=1,
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        z = self.embed(x)
        z = self.encoder(z.unsqueeze(0)).squeeze(0)
        return self.head(z).squeeze(-1)


class PIRLimited:
    """PIR simplified official adapter.

    Architecture extracted from vendor/PIR/models/PIR.py.
    Uses QualityEstimator to blend backbone prediction with Refiner output.
    No retrieval index (simplification — noted as limited_official).

    Implementation note: the original PIR uses a retrieval index over training
    data to find similar historical patterns. This adapter omits retrieval and
    only uses the QualityEstimator + Refiner components. Marked as
    'limited_official(no_retrieval)' in results.
    """

    name = "PIR"

    def __init__(self, pred_len=24, epochs=30, lr=1e-3, seed=0):
        self.pred_len = pred_len
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.qe = None
        self.refiner = None

    def fit(self, Z, yhat, y):
        torch.manual_seed(self.seed)
        n = len(yhat)
        if n < 100:
            return

        yhat_t = torch.tensor(yhat, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        self.qe = QualityEstimator(seq_len=64, pred_len=64, d_model=64)
        self.refiner = Refiner(d_model=64)

        params = list(self.qe.parameters()) + list(self.refiner.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        best_loss, best_state_qe, best_state_rf, patience = float("inf"), None, None, 0

        for ep in range(self.epochs):
            total_loss = 0.0
            n_batches = 0
            chunk = 256
            for i in range(0, n - chunk, chunk // 2):
                s, e = i, min(i + chunk, n)
                if e - s < 50:
                    continue
                yy = y_t[s:e]
                yh = yhat_t[s:e]
                mean_h = yh.mean()
                std_h = yh.std().clamp(min=1e-5)
                yh_n = (yh - mean_h) / std_h

                refine = self.refiner(yh_n) * std_h + mean_h

                feat_w = min(64, e - s)
                x_enc = yh[:feat_w].unsqueeze(-1)
                inter = yh[:feat_w].unsqueeze(-1)
                sims = torch.ones(feat_w)
                alpha, beta = self.qe(x_enc, inter, sims)
                alpha = alpha.mean().expand(e - s)
                beta = beta.mean().expand(e - s)

                pred = yh + alpha * (refine - yh) + beta * (refine - yh) * 0.1
                loss = F.mse_loss(pred, yy)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                total_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                continue
            avg_loss = total_loss / n_batches
            if avg_loss < best_loss - 1e-6:
                best_loss, patience = avg_loss, 0
                best_state_qe = {k: v.clone() for k, v in self.qe.state_dict().items()}
                best_state_rf = {k: v.clone() for k, v in self.refiner.state_dict().items()}
            else:
                patience += 1
                if patience >= 8:
                    break
        if best_loss < float("inf"):
            self.qe.load_state_dict(best_state_qe)
            self.refiner.load_state_dict(best_state_rf)

    def predict(self, Z, yhat):
        if self.qe is None:
            return yhat.copy()
        self.qe.eval()
        self.refiner.eval()
        with torch.no_grad():
            yh = torch.tensor(yhat, dtype=torch.float32)
            mean_h = yh.mean()
            std_h = yh.std().clamp(min=1e-5)
            yh_n = (yh - mean_h) / std_h
            refine = self.refiner(yh_n) * std_h + mean_h

            n = len(yh)
            feat_w = min(64, n)
            x_enc = yh[:feat_w].unsqueeze(-1)
            inter = yh[:feat_w].unsqueeze(-1)
            sims = torch.ones(feat_w)
            alpha, beta = self.qe(x_enc, inter, sims)
            alpha = alpha.mean().expand(n)
            beta = beta.mean().expand(n)

            pred = yh + alpha * (refine - yh) + beta * (refine - yh) * 0.1
            return pred.numpy()
