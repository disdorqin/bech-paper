"""Quantile Correction 复现: ETTh1 经典统计后处理.

方法: 分位回归直接预测目标，中位数作为点预测，
区间覆盖验证。不经过任何深度学习基座。
"""
import numpy as np, pandas as pd

# ---------- 加载 ETTh1 ----------
df = pd.read_csv("data/raw/ts_benchmarks/ETTh1.csv")
df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date")
y = df["OT"].values; n = len(y)

# ---------- 构建特征 ----------
cols, names = [], []; s = pd.Series(y)
for L in [1,2,3,7,14,28,56,168,336,720]:
    cols.append(s.shift(L).values); names.append(f"lag{L}")
hour = df["date"].dt.hour.values
cols += [np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)]
X_all = np.column_stack(cols).astype(np.float64)
warm = 720; ok = np.isfinite(X_all[warm:]).all(axis=1)
valid = np.arange(warm, n)[ok]
X, Y = X_all[valid], y[valid]

cut = int(len(X) * 0.8)
X_tr, Y_tr = X[:cut], Y[:cut]
X_te, Y_te = X[cut:], Y[cut:]

# ---------- Baseline: 直接回归 ----------
import lightgbm as lgb
base = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=63,
                          random_state=42, n_jobs=8, verbose=-1)
base.fit(X_tr, Y_tr)
y_base = base.predict(X_te)
base_mse = float(((Y_te-y_base)**2).mean()); base_mae = float(np.abs(Y_te-y_base).mean())

# ---------- Quantile 多分位回归 ----------
import math
results = []
for nq in [3, 5, 9, 19]:
    qs = np.linspace(0.05, 0.95, nq)
    q_models = {}
    for q in qs:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=300,
                              learning_rate=0.05, num_leaves=31,
                              random_state=42, n_jobs=8, verbose=-1)
        m.fit(X_tr, Y_tr); q_models[q] = m

    # 中位数预测
    mid_q = qs[len(qs)//2]
    y_mid = q_models[mid_q].predict(X_te)
    # 区间
    y_low = q_models[qs[0]].predict(X_te)
    y_high = q_models[qs[-1]].predict(X_te)
    coverage = float(((Y_te >= y_low) & (Y_te <= y_high)).mean())
    
    mse_q = float(((Y_te - y_mid)**2).mean())
    mae_q = float(np.abs(Y_te - y_mid).mean())
    results.append((nq, mse_q, mae_q, coverage))

print(f"Base (LGB): MSE={base_mse:.4f} MAE={base_mae:.4f}")
for r in results:
    print(f"  Quantile(nq={r[0]:2d}): MSE={r[1]:.4f} ({r[1]/base_mse-1:+.1%}) MAE={r[2]:.4f} ({r[2]/base_mae-1:+.1%}) cov={r[3]:.1%}")

best = min(results, key=lambda x: x[1])
print(f"\n验收: Quantile nq={best[0]} MSE={best[1]:.4f} vs Base MSE={base_mse:.4f}")
print(f"区间覆盖={best[3]:.1%} (预期≈90% for q0.05-q0.95)")
