"""Project backbones: 5 heterogeneous frozen forecasters.

- LightGBM (GBDT): CPU only — GPU path deadlocks on this machine.
- PyTorch (LSTM, Transformer): GPU when CUDA available; CPU fallback.

Uniform contract
----------------
    m = make_backbone(name)
    m.fit(Xtr, ytr, seq_tr)
    yhat = m.predict(Xte, seq_te)

`X`   : (N, F) cutoff-safe tabular features
`seq` : (N, L) price window ending at t-24, used by LSTM / Transformer
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(int(os.environ.get("BECH_TORCH_THREADS", "4")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKBONES = ("Linear", "MLP", "LSTM", "Transformer", "GBDT")


# ------------------------------------------------------------- scikit-learn --
class _Sk:
    def __init__(self, kind: str, seed: int = 0):
        self.kind, self.seed = kind, seed
        self.sc = StandardScaler()
        self.scy = StandardScaler()

    def fit(self, X, y, seq=None):
        Xs = self.sc.fit_transform(X)
        ys = self.scy.fit_transform(y.reshape(-1, 1)).ravel()
        if self.kind == "Linear":
            self.m = Ridge(alpha=1.0).fit(Xs, ys)
        else:
            self.m = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300,
                                  early_stopping=True, n_iter_no_change=15,
                                  random_state=self.seed).fit(Xs, ys)
        return self

    def predict(self, X, seq=None):
        p = self.m.predict(self.sc.transform(X)).reshape(-1, 1)
        return self.scy.inverse_transform(p).ravel()


class _GBDT:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y, seq=None):
        import lightgbm as lgb
        self.m = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            min_child_samples=30, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, random_state=self.seed,
            n_jobs=8, verbose=-1)          # CPU only
        self.m.fit(X, y)
        return self

    def predict(self, X, seq=None):
        return self.m.predict(X)


# -------------------------------------------------------------------- torch --
class _SeqLSTM(nn.Module):
    def __init__(self, n_static: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def forward(self, seq, stat):
        h, _ = self.lstm(seq.unsqueeze(-1))
        return self.head(torch.cat([h[:, -1, :], stat], dim=1)).squeeze(-1)


class _SeqTransformer(nn.Module):
    """Patch-style encoder (PatchTST-lite): patchify -> linear embed -> encoder."""

    def __init__(self, n_static: int, seq_len: int, d_model: int = 64,
                 patch: int = 24, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.patch = patch
        self.n_patch = seq_len // patch
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                         dropout=0.1, batch_first=True,
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d_model + n_static, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def forward(self, seq, stat):
        b = seq.shape[0]
        p = seq[:, -self.n_patch * self.patch:].reshape(b, self.n_patch, self.patch)
        z = self.enc(self.embed(p) + self.pos).mean(dim=1)
        return self.head(torch.cat([z, stat], dim=1)).squeeze(-1)


class _Torch:
    def __init__(self, kind: str, seed: int = 0, epochs: int = 40, bs: int = 256):
        self.kind, self.seed, self.epochs, self.bs = kind, seed, epochs, bs
        self.sc_s = StandardScaler()
        self.sc_q = StandardScaler()
        self.sc_y = StandardScaler()

    def _prep(self, X, seq, fit=False):
        if fit:
            stat = self.sc_s.fit_transform(X)
            q = self.sc_q.fit_transform(seq.reshape(-1, 1)).reshape(seq.shape)
        else:
            stat = self.sc_s.transform(X)
            q = self.sc_q.transform(seq.reshape(-1, 1)).reshape(seq.shape)
        return (torch.tensor(q, dtype=torch.float32),
                torch.tensor(stat, dtype=torch.float32))

    def fit(self, X, y, seq):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        q, stat = self._prep(X, seq, fit=True)
        q, stat = q.to(DEVICE), stat.to(DEVICE)
        ys = torch.tensor(self.sc_y.fit_transform(y.reshape(-1, 1)).ravel(),
                          dtype=torch.float32, device=DEVICE)
        n = len(ys); cut = int(n * 0.9)
        idx = np.arange(n)
        tr, va = idx[:cut], idx[cut:]
        if self.kind == "LSTM":
            self.m = _SeqLSTM(X.shape[1]).to(DEVICE)
        else:
            self.m = _SeqTransformer(X.shape[1], seq.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(self.m.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.HuberLoss(delta=1.0)
        best, best_state, patience = np.inf, None, 0
        for ep in range(self.epochs):
            self.m.train()
            perm = np.random.permutation(tr)
            for i in range(0, len(perm), self.bs):
                b = perm[i:i + self.bs]
                opt.zero_grad()
                out = self.m(q[b], stat[b])
                loss = lossf(out, ys[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)
                opt.step()
            self.m.eval()
            with torch.no_grad():
                vl = float(lossf(self.m(q[va], stat[va]), ys[va]))
            if vl < best - 1e-5:
                best, patience = vl, 0
                best_state = {k: v.clone() for k, v in self.m.state_dict().items()}
            else:
                patience += 1
                if patience >= 8:
                    break
        if best_state is not None:
            self.m.load_state_dict(best_state)
        return self

    def predict(self, X, seq):
        q, stat = self._prep(X, seq, fit=False)
        q, stat = q.to(DEVICE), stat.to(DEVICE)
        self.m.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(q), 1024):
                outs.append(self.m(q[i:i + 1024], stat[i:i + 1024]).cpu().numpy())
        p = np.concatenate(outs).reshape(-1, 1)
        return self.sc_y.inverse_transform(p).ravel()


def make_backbone(name: str, seed: int = 0):
    if name in ("Linear", "MLP"):
        return _Sk(name, seed)
    if name == "GBDT":
        return _GBDT(seed)
    if name in ("LSTM", "Transformer"):
        return _Torch(name, seed)
    raise KeyError(name)


def needs_seq(name: str) -> bool:
    return name in ("LSTM", "Transformer")
