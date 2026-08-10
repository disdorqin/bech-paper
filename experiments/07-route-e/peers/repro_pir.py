"""PIR 复现: ETTh1 + PatchTST backbone.

论文: Liu et al. 2025, NeurIPS 2025. 
arXiv:2505.23583.

两阶段: (1) 训 PatchTST backbone (2) PIR 后处理修正
目标: PatchTST MSE 0.466→PIR 0.437 (-6.22%)
"""
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

# ---------- 数据 ----------
df = pd.read_csv("data/raw/ts_benchmarks/ETTh1.csv")
df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date")
cols = ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]
data = df[cols].values.astype(np.float32)
L, H = 96, 96  # seq_len, pred_len

X_list, Y_list = [], []
for i in range(len(data)-L-H):
    X_list.append(data[i:i+L]); Y_list.append(data[i+L:i+L+H])
X = np.stack(X_list); Y = np.stack(Y_list)  # (B, L, N), (B, H, N)

B = len(X); n_tr = B*6//10; n_va = B*8//10
xm, xs = X[:n_tr].mean(axis=(0,1),keepdims=True), X[:n_tr].std(axis=(0,1),keepdims=True)+1e-9
ym, ys = Y[:n_tr].mean(), Y[:n_tr].std()+1e-9

X_tr = torch.tensor((X[:n_tr]-xm)/xs,dtype=torch.float32)
Y_tr = torch.tensor((Y[:n_tr]-ym)/ys,dtype=torch.float32)
X_va = torch.tensor((X[n_tr:n_va]-xm)/xs,dtype=torch.float32)
Y_va = torch.tensor((Y[n_tr:n_va]-ym)/ys,dtype=torch.float32)
X_te = torch.tensor((X[n_va:]-xm)/xs,dtype=torch.float32)
Y_te = torch.tensor((Y[n_va:]-ym)/ys,dtype=torch.float32)
print(f"Data: tr={len(X_tr)} va={len(X_va)} te={len(X_te)}")

# ---------- PatchTST backbone ----------
class PatchTST(nn.Module):
    def __init__(self, n_vars=7, d_model=128, n_heads=4, e_layers=3, L=96, H=96, patch_len=16):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.n_patches = L // patch_len
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=256,
                                          dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, e_layers)
        self.head = nn.Linear(d_model, H)

    def forward(self, x):  # (B, L, N)
        B = x.shape[0]
        total_out = 0
        for v in range(self.n_vars):
            xv = x[:, :, v]  # (B, L)
            patches = xv[:, -self.n_patches*self.patch_len:].reshape(B, self.n_patches, self.patch_len)
            z = self.embed(patches) + self.pos
            z = self.enc(z).mean(dim=1)
            total_out += self.head(z)
        return total_out / self.n_vars  # (B, H)

patchtst = PatchTST().cuda() if torch.cuda.is_available() else PatchTST()
X_tr_d = X_tr.cuda() if torch.cuda.is_available() else X_tr
Y_tr_d = Y_tr.cuda() if torch.cuda.is_available() else Y_tr
X_va_d = X_va.cuda() if torch.cuda.is_available() else X_va
Y_va_d = Y_va.cuda() if torch.cuda.is_available() else Y_va
X_te_d = X_te.cuda() if torch.cuda.is_available() else X_te
Y_te_d = Y_te.cuda() if torch.cuda.is_available() else Y_te

opt = torch.optim.Adam(patchtst.parameters(), lr=1e-4)
best_va, best_st = np.inf, None
print("Training PatchTST...")
for ep in range(100):
    patchtst.train(); opt.zero_grad()
    loss = F.mse_loss(patchtst(X_tr_d), Y_tr_d[:,:,-1])
    loss.backward(); opt.step()
    patchtst.eval()
    with torch.no_grad():
        vl = float(F.mse_loss(patchtst(X_va_d), Y_va_d[:,:,-1]))
    if vl < best_va: best_va = vl; best_st = {k:v.clone() for k,v in patchtst.state_dict().items()}
    if ep % 30 == 0: print(f"  ep{ep}: va_mse={vl:.4f}")
patchtst.load_state_dict(best_st); patchtst.eval()

with torch.no_grad():
    Y_base = patchtst(X_te_d).cpu().numpy()
    Y_base_tr = patchtst(X_tr_d).cpu().numpy()
    Y_base_va = patchtst(X_va_d).cpu().numpy()
Y_true = Y_te[:,:,-1].numpy()
base_mse = float(((Y_true - Y_base)**2).mean())
base_mae = float(np.abs(Y_true - Y_base).mean())
print(f"\nPatchTST base: MSE={base_mse:.4f} MAE={base_mae:.4f}  (论文: MSE=0.466)")

# ---------- PIR: Failure Identification + Local+Global Revision ----------
# 1. Failure identification: 2-layer FC estimates error (delta)
# Input: (X, Y_base) -> predict |Y_base - Y_true|
n_vars = X.shape[2]
Z_te = np.hstack([X_te.cpu().numpy()[:, -1, :], Y_base]).astype(np.float32)
Z_tr = np.hstack([X_tr.cpu().numpy()[:, -1, :], Y_base_tr]).astype(np.float32)
Z_va = np.hstack([X_va.cpu().numpy()[:, -1, :], Y_base_va]).astype(np.float32)
err_tr = ((Y_tr[:,:,-1].numpy() - Y_base_tr)**2).mean(axis=1)
err_va = ((Y_va[:,:,-1].numpy() - Y_base_va)**2).mean(axis=1)

class FailureID(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x): return self.net(x).squeeze(-1)

fid = FailureID(Z_tr.shape[1])
opt_f = torch.optim.Adam(fid.parameters(), lr=1e-3)
Z_tr_t = torch.tensor(Z_tr); err_tr_t = torch.tensor(err_tr)
Z_va_t = torch.tensor(Z_va); err_va_t = torch.tensor(err_va)
for ep in range(200):
    fid.train(); opt_f.zero_grad()
    loss = F.l1_loss(fid(Z_tr_t), err_tr_t)
    loss.backward(); opt_f.step()
fid.eval()
with torch.no_grad():
    delta_te = fid(torch.tensor(Z_te)).numpy()
    delta_va = fid(Z_va_t).numpy()

# 2. Local revision: Ridge regression to predict residual
from sklearn.linear_model import Ridge
resid_tr = (Y_tr[:,:,-1].numpy() - Y_base_tr).mean(axis=1)
lr = Ridge(alpha=1.0).fit(Z_tr, resid_tr)
y_local = lr.predict(Z_te)

# 3. Global revision: cosine similarity retrieval
X_tr_flat = X_tr.cpu().numpy()[:, -1, :]  # (B, L) -> last step
X_te_flat = X_te.cpu().numpy()[:, -1, :]
norms_tr = np.linalg.norm(X_tr_flat, axis=1, keepdims=True)+1e-9
norms_te = np.linalg.norm(X_te_flat, axis=1, keepdims=True)+1e-9
sim = X_te_flat @ X_tr_flat.T / (norms_te * norms_tr.T + 1e-9)
K = 10
top_idx = np.argsort(-sim, axis=1)[:, :K]
top_sim = np.take_along_axis(sim, top_idx, axis=1)
weights = np.exp(top_sim) / np.exp(top_sim).sum(axis=1, keepdims=True)
# Retrieved targets
y_global = np.zeros(len(Y_te))
for i in range(len(Y_te)):
    y_global[i] = np.average(resid_tr[top_idx[i]], weights=weights[i])

# 4. Final: y_pred = y_base + alpha*local + beta*global
# alpha = sigmoid(Linear(delta)), beta = sigmoid(MLP(delta, w))
delta_norm = (delta_te - delta_te.mean()) / (delta_te.std()+1e-9)
alpha = 1.0 / (1.0 + np.exp(-(0.5 + 0.1*delta_norm)))
beta = 0.5 / (1.0 + np.exp(-delta_norm))

Y_pir = Y_base + alpha.reshape(-1,1)*y_local.reshape(-1,1) + beta.reshape(-1,1)*y_global.reshape(-1,1)
pir_mse = float(((Y_true - Y_pir)**2).mean())
pir_mae = float(np.abs(Y_true - Y_pir).mean())

print(f"\n=== PIR 复现 ===")
print(f"Base:  MSE={base_mse:.4f} MAE={base_mae:.4f}")
print(f"PIR:   MSE={pir_mse:.4f} MAE={pir_mae:.4f}")
print(f"Improvement: MSE {pir_mse/base_mse-1:+.2%}  MAE {pir_mae/base_mae-1:+.2%}")
print(f"论文: PatchTST MSE 0.466→0.437 (-6.22%)")
