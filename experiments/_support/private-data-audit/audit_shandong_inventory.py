# -*- coding: utf-8 -*-
"""山东数据盘点:数据概况 + 双尾可预测性 + 特征清单 + 实验规模评估"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

F = r"D:\作业\science\solar_leak_price_model\data\raw\provinces\shandong_pmos_hourly.csv"
df = pd.read_csv(F, encoding="gbk")
df["时刻"] = pd.to_datetime(df["时刻"])
df = df.sort_values("时刻").reset_index(drop=True)

price_cols = {"日前": "日前电价", "实时": "实时电价"}
for name, col in price_cols.items():
    s = df[col]
    n = len(s)
    neg = int((s < 0).sum())
    print(f"[{name}电价] n={n} 负价点数={neg} 负价占比={neg/n*100:.2f}% min={s.min():.2f} max={s.max():.2f} p99={s.quantile(0.99):.2f} p99.9={s.quantile(0.999):.2f}")

df["year"] = df["时刻"].dt.year
for name, col in price_cols.items():
    by_year = df.groupby("year")[col].apply(lambda x: (x < 0).mean() * 100).round(2)
    print(f"[{name} 逐年负价占比%] {by_year.to_dict()}")

print("\n=== 时间范围 ===")
print(df["时刻"].min(), "->", df["时刻"].max(), " 总小时数:", len(df))

# 四段切分(按 50/20/10/20)
n = len(df)
i1, i2, i3 = int(n * 0.5), int(n * 0.7), int(n * 0.8)
seg = np.zeros(n, dtype=int)
seg[i1:] = 1; seg[i2:] = 2; seg[i3:] = 3
print(f"\n=== 四段切分 === S1:[0,{i1}) S2:[{i1},{i2}) S3:[{i2},{i3}) S4:[{i3},{n})")
for k in range(4):
    m = seg == k
    print(f"  S{k+1}: {m.sum()} 小时, {df.loc[m, '时刻'].min()} ~ {df.loc[m, '时刻'].max()}")

# 特征工程(仅用滞后特征,防泄漏):残差历史、日前电价、日内形状、日历
X = pd.DataFrame(index=df.index)
X["da_lag1"] = df["日前电价"].shift(1)
X["da_lag24"] = df["日前电价"].shift(24)
X["da_lag168"] = df["日前电价"].shift(168)
X["rt_lag1"] = df["实时电价"].shift(1)
X["resid_lag1"] = df["日前电价"].shift(1) - df["实时电价"].shift(1)
X["wind_fc"] = df["风电总加预测值"]
X["solar_fc"] = df["光伏总加预测值"]
X["load_fc"] = df["直调负荷预测值"]
X["hour"] = df["时刻"].dt.hour
X["dow"] = df["时刻"].dt.dayofweek
X["month"] = df["时刻"].dt.month
X["is_holiday"] = df["时刻"].dt.dayofweek >= 5

print("\n=== 双尾可预测性(在 S1 段训练, S4 段测) ===")
s1 = seg == 0
s4 = seg == 3
for name, col in price_cols.items():
    y = df[col]
    neg_y = (y < 0).astype(int)
    p99 = y.quantile(0.99)
    spike_y = (y > p99).astype(int)
    Xtr = X[s1]; Xte = X[s4]
    for tname, tgt in [("负尾P(neg)", neg_y), (f"尖峰P(>{p99:.0f})", spike_y)]:
        ytr = tgt[s1].dropna(); yte = tgt[s4].dropna()
        Xtr2 = Xtr.loc[ytr.index]; Xte2 = Xte.loc[yte.index]
        Xtr2 = Xtr2.fillna(Xtr2.median()); Xte2 = Xte2.fillna(Xtr2.median())
        if ytr.sum() < 10 or (1 - ytr).sum() < 10:
            print(f"  [{name}][{tname}] 样本不足(S1正例{ytr.sum()})")
            continue
        clf = LogisticRegression(max_iter=2000).fit(Xtr2, ytr)
        p = clf.predict_proba(Xte2)[:, 1]
        try:
            auc = roc_auc_score(yte, p)
        except Exception:
            auc = float("nan")
        ap = average_precision_score(yte, p)
        print(f"  [{name}][{tname}] S1正例={ytr.sum()} S4正例={yte.sum()} AUC={auc:.3f} AP={ap:.3f}")

print("\n=== 特征清单与相关性 ===")
feat = ["日前电价", "实时电价", "风电总加预测值", "光伏总加预测值", "直调负荷预测值",
        "风电总加实际值", "光伏总加实际值", "直调负荷实际值", "联络线受电负荷预测值"]
corr = df[feat + ["日前电价"]].corr()
print(corr.round(2).to_string())
print("\n列数:", len(df.columns), " 特征列数(除时刻+双电价):", len(df.columns) - 3)
