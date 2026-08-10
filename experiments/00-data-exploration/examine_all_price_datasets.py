# -*- coding: utf-8 -*-
"""
统一口径检验全部公开电价数据集的"极端性"特征。

与 lago_benchmark/examine_datasets.py 保持完全相同的度量定义, 便于横向对比:
  - neg_pct        : 负电价占比 (%)
  - neg_min        : 最深负电价
  - spike_pct_p99  : 超过 p99 的占比 (定义上恒 ≈1%, 用于确认)
  - p99 / median   : 尖峰相对倍率 (量纲无关, 可跨市场比较)
  - max / p99      : 极端尾部厚度 (>2 说明 p99 之外还有长尾)
  - skew / kurt    : 偏度 / 峰度 (重尾证据)
  - spike_pct_abs  : 超过"绝对阈值"的占比。文献常用阈值:
        欧洲/美国 EUR|USD/MWh -> 100 (Lago/GEFCom 系列常用)
        澳洲 AUD/MWh          -> 300 (NEM 近年价格中枢抬升后的常用阈值;
                                      早期文献用 100, 见 Energy Economics 2023)

输出: data/ALL_price_characteristics.csv + ALL_price_characteristics.md
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def stats(name, s, unit, abs_thr, note=""):
    s = pd.Series(s).dropna().astype(float)
    n = len(s)
    p99 = s.quantile(0.99)
    med = s.median()
    neg = (s < 0).sum()
    return {
        "dataset": name,
        "unit": unit,
        "n": n,
        "min": round(s.min(), 2),
        "median": round(med, 2),
        "mean": round(s.mean(), 2),
        "p99": round(p99, 2),
        "max": round(s.max(), 2),
        "neg_n": int(neg),
        "neg_pct": round(neg / n * 100, 4),
        "neg_min": round(s.min(), 2) if neg else None,
        "p99_over_median": round(p99 / med, 2) if med > 0 else None,
        "max_over_p99": round(s.max() / p99, 2) if p99 > 0 else None,
        "skew": round(s.skew(), 2),
        "kurtosis": round(s.kurtosis(), 1),
        f"gt_{abs_thr}_pct": round((s > abs_thr).sum() / n * 100, 4),
        "abs_thr": abs_thr,
        "note": note,
    }


def load_lago():
    d = os.path.join(HERE, "lago_benchmark")
    out = []
    for mk in ["DE", "PJM", "FR", "BE", "NP"]:
        f = os.path.join(d, f"{mk}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        pc = [c for c in df.columns
              if "price" in c.lower() or c.lower() == "prices"]
        col = pc[0]
        unit = "USD/MWh" if mk == "PJM" else "EUR/MWh"
        out.append(stats(f"Lago-{mk}", df[col], unit, 100,
                         "Lago2021 AppliedEnergy 开放基准"))
    return out


def load_gefcom():
    f = os.path.join(HERE, "gefcom2014", "GEFCom2014P_hourly.csv")
    if not os.path.exists(f):
        return []
    df = pd.read_csv(f)
    return [stats("GEFCom2014-P", df["price"], "USD/MWh", 100,
                  "IJF2016 官方竞赛数据 2011-01~2013-12")]


def load_nem():
    d = os.path.join(HERE, "nem_aemo", "clean")
    out = []
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not f.endswith("_price.csv"):
            continue
        mk = f.replace("_price.csv", "")
        df = pd.read_csv(os.path.join(d, f))
        out.append(stats(f"NEM-{mk}", df["price"], "AUD/MWh", 300,
                         "AEMO 官方公开 5min 现货价 2021-2025"))
    return out


def main():
    rows = load_lago() + load_gefcom() + load_nem()
    if not rows:
        print("no dataset found")
        return
    df = pd.DataFrame(rows)
    csv = os.path.join(HERE, "ALL_price_characteristics.csv")
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    lines = ["# 公开电价数据集极端性特征（统一口径）", ""]
    lines.append("度量定义见 `examine_all_price_datasets.py` 顶部注释。")
    lines.append("")
    lines.append("| 数据集 | 单位 | 样本数 | 中位数 | p99 | 最大值 | 最小值 | "
                 "负价占比% | 最深负价 | p99/中位 | max/p99 | 偏度 | 峰度 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['unit']} | {r['n']} | {r['median']} | "
            f"{r['p99']} | {r['max']} | {r['min']} | {r['neg_pct']} | "
            f"{r['neg_min'] if pd.notna(r['neg_min']) else '—'} | "
            f"{r['p99_over_median']} | {r['max_over_p99']} | "
            f"{r['skew']} | {r['kurtosis']} |"
        )
    md = os.path.join(HERE, "ALL_price_characteristics.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(df.to_string(index=False))
    print(f"\nsaved -> {csv}\nsaved -> {md}")


if __name__ == "__main__":
    main()
