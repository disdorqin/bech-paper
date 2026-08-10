"""Spike Regularization 复现: NEM SA1 替代新加坡 NEMS.

论文: Ponyuenyong et al. 2026, AAAI 2026 WS.
核心: spike penalty = 对 |residual| > threshold 的样本加权训练。
论文 MAPE -37.4% (新加坡 NEMS + TTM 预训练).
我们: NEM SA1 + LightGBM (数据不可获取 → 替代).
"""
import numpy as np, pandas as pd

# 加载 NEM SA1
df = pd.read_csv("data/raw/nem_aemo/clean/SA1_2024_hourly.csv", index_col=0, parse_dates=True)
price = df["price"].values; demand = df["TOTALDEMAND"].values; n = len(price)

# 构建特征
cols, names = [], []; s = pd.Series(price); d = pd.Series(demand)
for L in [1,2,3,24,48,72,168]:
    cols.append(s.shift(L).values); names.append(f"price_lag{L}")
for L in [1,24,168]:
    cols.append(d.shift(L).values); names.append(f"demand_lag{L}")
roll24 = s.shift(1).rolling(24, min_periods=12)
cols += [roll24.mean().values, roll24.std().values]
names += ["price_roll24_mean", "price_roll24_std"]
hour = df.index.hour.values
cols += [np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24), (df.index.dayofweek>=5).astype(float)]
names += ["hour_sin", "hour_cos", "is_weekend"]

X_all = np.column_stack(cols).astype(np.float64)
warm = 168; ok = np.isfinite(X_all[warm:]).all(axis=1)
X, Y = X_all[warm:][ok], price[warm:][ok]

cut = int(len(X)*0.8)
X_tr, Y_tr = X[:cut], Y[:cut]
X_te, Y_te = X[cut:], Y[cut:]

import lightgbm as lgb

# Baseline: standard L1 regression
base = lgb.LGBMRegressor(objective="regression_l1", n_estimators=500, learning_rate=0.03,
                          num_leaves=63, random_state=42, n_jobs=8, verbose=-1)
base.fit(X_tr, Y_tr)
y_base = base.predict(X_te)
base_mae = float(np.abs(Y_te - y_base).mean())
base_mape = float((np.abs((Y_te - y_base) / (np.abs(Y_te)+1e-6))).mean()) * 100

# Spike regularization: weighted L1, |residual| > p95 → extra weight
# First pass: get residual distribution
base2 = lgb.LGBMRegressor(objective="regression_l1", n_estimators=500, learning_rate=0.03,
                           num_leaves=63, random_state=42, n_jobs=8, verbose=-1)
base2.fit(X_tr, Y_tr)
r_tr = np.abs(Y_tr - base2.predict(X_tr))
spike_thr = float(np.quantile(r_tr, 0.95))
is_spike = np.abs(Y_tr - base2.predict(X_tr)) > spike_thr
sample_w = np.where(is_spike, 5.0, 1.0)  # spike 样本 5x 权重

spike_model = lgb.LGBMRegressor(objective="regression_l1", n_estimators=500, learning_rate=0.03,
                                 num_leaves=63, random_state=42, n_jobs=8, verbose=-1)
spike_model.fit(X_tr, Y_tr, sample_weight=sample_w)
y_spike = spike_model.predict(X_te)

spike_mae = float(np.abs(Y_te - y_spike).mean())
spike_mape = float((np.abs((Y_te - y_spike) / (np.abs(Y_te)+1e-6))).mean()) * 100

# 负价时段改善
neg = Y_te < 0
print(f"Test neg%: {neg.mean():.1%}")
print(f"Base MAE={base_mae:.1f} MAPE={base_mape:.1f}%")
print(f"SpikeReg MAE={spike_mae:.1f} MAPE={spike_mape:.1f}%")
print(f"  neg-hour MAE: Base={float(np.abs(Y_te[neg]-y_base[neg]).mean()):.1f} → Spike={float(np.abs(Y_te[neg]-y_spike[neg]).mean()):.1f}")
print(f"\n论文: Singapore NEMS + TTM, MAPE -37.4%")
print(f"我们: NEM SA1 + LGB, MAE {base_mae/spike_mae-1:+.1%}, MAPE {base_mape-spike_mape:+.1f}pp")
print(f"注: 新加坡 NEMS 不可获取 → 用 NEM SA1 替代. 与论文非直接对照.")

