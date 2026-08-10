# -*- coding: utf-8 -*-
"""
把 AEMO 逐月原始 CSV 合并成每个区域一条连续序列, 并同时产出小时聚合版。

原始字段: REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE
  SETTLEMENTDATE 为"区间结束时刻"(interval-ending), AEMO 惯例。
  2021-07 之前为 30min 结算, 之后为 5min 结算 -> 采样频率会在中途变化,
  因此额外产出统一的小时均值版本 (同行 NEM 论文的通行做法, 见
  KAN+XGBoost NEM 2026: "原始数据为五分钟分辨率, 聚合为小时值")。

输出:
  clean/{REGION}_price.csv        原生分辨率 (timestamp, price, demand)
  clean/{REGION}_hourly.csv       小时均值 (timestamp, price, demand)
  clean/{REGION}_hourly_max.csv   小时最大值 (保留尖峰, 用于极端事件研究)
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
CLEAN = os.path.join(HERE, "clean")
os.makedirs(CLEAN, exist_ok=True)


def build(region):
    files = sorted(f for f in os.listdir(RAW)
                   if f.startswith(region + "_") and f.endswith(".csv"))
    if not files:
        return None
    parts = []
    for f in files:
        d = pd.read_csv(os.path.join(RAW, f))
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["SETTLEMENTDATE"],
                                     format="%Y/%m/%d %H:%M:%S")
    df = (df.rename(columns={"RRP": "price", "TOTALDEMAND": "demand"})
            [["timestamp", "price", "demand"]]
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True))
    df.to_csv(os.path.join(CLEAN, f"{region}_price.csv"), index=False)

    h = df.set_index("timestamp")
    hm = h.resample("h").mean().dropna().reset_index()
    hx = h.resample("h").max().dropna().reset_index()
    hm.to_csv(os.path.join(CLEAN, f"{region}_hourly.csv"), index=False)
    hx.to_csv(os.path.join(CLEAN, f"{region}_hourly_max.csv"), index=False)

    n = len(df)
    neg = (df.price < 0).sum()
    return {
        "region": region, "n_native": n, "n_hourly": len(hm),
        "start": df.timestamp.iloc[0], "end": df.timestamp.iloc[-1],
        "min": df.price.min(), "max": df.price.max(),
        "median": df.price.median(),
        "neg_pct": neg / n * 100,
        "gt300_pct": (df.price > 300).sum() / n * 100,
    }


if __name__ == "__main__":
    regions = sorted({f.split("_")[0] for f in os.listdir(RAW)
                      if f.endswith(".csv")})
    rows = [r for r in (build(x) for x in regions) if r]
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
