"""WP-3 δ-Adapter fidelity — production DeltaAdapterLimited (PostY/Ada-Y) check.

The Ridge-based repro showed correct direction (-1.7%) vs paper (-9.6%) but a
small magnitude and a non-production adapter. This tests the ACTUAL production
implementation used for B3 in the matrix (official_adapters.DeltaAdapterLimited,
official PostY architecture) on the same Weather reference setting.

Interface under test: fit(Z, yhat, y) on val / predict(Z, yhat) on test,
flattened across horizons (the matrix applies the adapter per-day-vector).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from official_adapters import DeltaAdapterLimited  # noqa: E402

df = pd.read_csv("data/raw/ts_benchmarks/weather.csv", encoding="latin-1")
df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date")
price = df["OT"].values
L, H = 96, 96

s = pd.Series(price)
cols = [s.shift(lag).values for lag in [1, 2, 3, 7, 14, 28, 56, 168, 336, 720]]
X_all = np.column_stack(cols).astype(np.float64)
warm = 720
ok = np.isfinite(X_all[warm:len(price) - H]).all(axis=1)
valid = np.arange(warm, len(price) - H)[ok]
X, Y = X_all[valid], np.array([price[i:i + H] for i in valid])

N = len(X); n_tr = int(N * 0.7); n_va = int(N * 0.85)
my, sy = Y[:n_tr].mean(), Y[:n_tr].std() + 1e-9
Ytr_n = (Y[:n_tr] - my) / sy; Yva_n = (Y[n_tr:n_va] - my) / sy
Yte_n = (Y[n_va:] - my) / sy

y_tr = np.zeros((n_tr, H)); y_va = np.zeros((n_va - n_tr, H)); y_te = np.zeros((N - n_va, H))
for h in range(H):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                          random_state=42, n_jobs=8, verbose=-1)
    m.fit(X[:n_tr], Ytr_n[:, h])
    y_tr[:, h] = m.predict(X[:n_tr])
    y_va[:, h] = m.predict(X[n_tr:n_va])
    y_te[:, h] = m.predict(X[n_va:])

base_te = float(((Yte_n - y_te) ** 2).mean())
print(f"LGB base MSE (test): {base_te:.4f}")

# production PostY adapter, fitted on flattened val, applied to flattened test
ad = DeltaAdapterLimited(hidden_dim=128, epochs=30, lr=1e-3, seed=0)
ad.fit(np.zeros_like(y_va.ravel()), y_va.ravel(), Yva_n.ravel())
y_te_flat = y_te.ravel()
y_ada = ad.predict(None, y_te_flat)
ada_mse = float(((Yte_n.ravel() - y_ada) ** 2).mean())
print(f"PostY adapted MSE (test): {ada_mse:.4f}  ({ada_mse / base_te - 1:+.2%})")
print(f"Paper reference: Weather PatchTST+Ada-X+Y 0.178->0.161 (-9.6%) [pretrained base]")
