"""WP-3 δ-Adapter fidelity — CLEAN Weather rerun (v3).

v1 (ridge repro) got -1.7%, v2 (production PostY) got -98%. Both VOID:
the raw weather.csv train split contains a -9999 sentinel that inflates
train std to ~386 while val/test (late-2020 tail) have real std ~16, so
the test segment looks degenerate in train-normalized space and any
predict-the-mean adapter trivially "wins" (test corr(yhat,y)=0.08).

This rerun: drop -9999 sentinels, re-standardize on clean train, and
re-test BOTH the production PostY and a predict-the-mean reference so we
can separate adapter mechanism from test-set artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from official_adapters import DeltaAdapterLimited  # noqa: E402

df = pd.read_csv(ROOT / "data/raw/ts_benchmarks/weather.csv", encoding="latin-1")
df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date")
price = df["OT"].values.astype(float)
n_sentinel = int((price <= -9990).sum())
print(f"sentinel count (<=-9990): {n_sentinel}  -> dropping rows")
price[price <= -9990] = np.nan
df["OT_clean"] = pd.Series(price).interpolate(limit_direction="both").values

p = df["OT_clean"].values
H = 96
s = pd.Series(p)
cols = [s.shift(lag).values for lag in [1, 2, 3, 7, 14, 28, 56, 168, 336, 720]]
X_all = np.column_stack(cols).astype(np.float64)
warm = 720
ok = np.isfinite(X_all[warm:len(p) - H]).all(axis=1)
valid = np.arange(warm, len(p) - H)[ok]
X, Y = X_all[valid], np.array([p[i:i + H] for i in valid])

N = len(X); n_tr = int(N * 0.7); n_va = int(N * 0.85)
my, sy = Y[:n_tr].mean(), Y[:n_tr].std() + 1e-9
Ytr_n = (Y[:n_tr] - my) / sy; Yva_n = (Y[n_tr:n_va] - my) / sy
Yte_n = (Y[n_va:] - my) / sy
print(f"clean train std={sy:.2f}  val std={Yva_n.std():.3f}  test std={Yte_n.std():.3f}")

y_va = np.zeros((n_va - n_tr, H)); y_te = np.zeros((N - n_va, H))
for h in range(H):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                          random_state=42, n_jobs=8, verbose=-1)
    m.fit(X[:n_tr], Ytr_n[:, h])
    y_va[:, h] = m.predict(X[n_tr:n_va])
    y_te[:, h] = m.predict(X[n_va:])

base_te = float(((Yte_n - y_te) ** 2).mean())
print(f"LGB base MSE (test): {base_te:.4f}")

# predict-the-mean reference on the SAME test segment
mean_te = float(((Yte_n - y_te.mean()) ** 2).mean())
print(f"predict-mean MSE (test): {mean_te:.4f}")

# production PostY, flattened, fitted on val
ad = DeltaAdapterLimited(hidden_dim=128, epochs=30, lr=1e-3, seed=0)
ad.fit(np.zeros_like(y_va.ravel()), y_va.ravel(), Yva_n.ravel())
y_ada = ad.predict(None, y_te.ravel())
ada_mse = float(((Yte_n.ravel() - y_ada) ** 2).mean())
print(f"PostY adapted MSE (test): {ada_mse:.4f}  ({ada_mse / base_te - 1:+.1%})")

print(f"Paper reference: Weather PatchTST+Ada-X+Y 0.178->0.161 (-9.6%) [pretrained base]")
