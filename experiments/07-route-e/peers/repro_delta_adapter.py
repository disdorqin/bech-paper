"""delta-Adapter 复现: Weather, LightGBM backbone."""
import numpy as np, pandas as pd

df = pd.read_csv("data/raw/ts_benchmarks/weather.csv", encoding="latin-1")
df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date")
price = df["OT"].values; n = len(price)

L, H = 96, 96
# Build feature sequences: lags for lookback
cols = []; s = pd.Series(price)
for lag in [1,2,3,7,14,28,56,168,336,720]:
    cols.append(s.shift(lag).values)
X_all = np.column_stack(cols).astype(np.float64)
warm = 720; ok = np.isfinite(X_all[warm:n-H]).all(axis=1)
valid = np.arange(warm, n-H)[ok]
X, Y = X_all[valid], np.array([price[i:i+H] for i in valid])

N = len(X); n_tr = int(N*0.7); n_va = int(N*0.85)
my, sy = Y[:n_tr].mean(), Y[:n_tr].std()+1e-9
Ytr_n = (Y[:n_tr]-my)/sy; Yva_n = (Y[n_tr:n_va]-my)/sy; Yte_n = (Y[n_va:]-my)/sy

# LightGBM backbone: predict one value at each horizon step
import lightgbm as lgb
y_preds = np.zeros((len(Yte_n), H))
for h in range(H):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                          random_state=42, n_jobs=8, verbose=-1)
    m.fit(X[:n_tr], Ytr_n[:, h])
    y_preds[:, h] = m.predict(X[n_va:])

base_mse = float(((Yte_n - y_preds)**2).mean())
print(f"Base LGB MSE: {base_mse:.4f}")

# delta-Adapter: learn correction to each horizon step
from sklearn.linear_model import Ridge
# Train adapter on val: predict correction from base predictions + features
# Use per-horizon correction (simplified: 1 adapter for all horizons)
r_tr = np.zeros((n_va-n_tr, H))
y_tr_preds = np.zeros((n_va-n_tr, H))
for h in range(H):
    m_small = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                random_state=42, n_jobs=8, verbose=-1)
    m_small.fit(X[:n_tr], Ytr_n[:, h])
    y_tr_preds[:, h] = m_small.predict(X[n_tr:n_va])
r_tr = Yva_n - y_tr_preds

# Adapter features: base prediction + input summary
Z_tr = np.hstack([y_tr_preds.mean(axis=1, keepdims=True), X[n_tr:n_va].mean(axis=1, keepdims=True)])
Z_te = np.hstack([y_preds.mean(axis=1, keepdims=True), X[n_va:].mean(axis=1, keepdims=True)])

delta_val = 0.1; adapter = Ridge(alpha=1.0)
adapter.fit(Z_tr, r_tr.mean(axis=1))
r_pred = adapter.predict(Z_te)
bound = delta_val * np.std(r_tr)
r_pred = np.clip(r_pred, -bound, bound)

# Apply correction uniformly across all horizons
y_ada = y_preds + r_pred.reshape(-1, 1)
ada_mse = float(((Yte_n - y_ada)**2).mean())

print(f"Adapter MSE: {ada_mse:.4f} ({ada_mse/base_mse-1:+.1%})")
print(f"\nPaper: Weather PatchTST+Ada-X+Y: 0.178->0.161 (-9.6%)")
print(f"Paper: Weather Sundial-S+Ada-X:  0.427->0.025 (-95.6%) [pretrained]")
print(f"注: 我们使用 LGB 自训练基座，非预训练模型。")
