# -*- coding: utf-8 -*-
"""
量化"极端电价制度漂移"(regime shift): 经典 EPF 基准所刻画的市场,
与当下真实市场之间的分布距离有多大。

这是本文 problem statement 的核心实证依据:
  经典基准 (GEFCom2014-P 2011-2013, Lago 2011-2018) 几乎不含负电价,
  而同一批市场在 2019 年之后负电价占比抬升了一个数量级以上;
  澳洲 NEM 更是达到两位数百分比。
  => 只在经典基准上验证过的方法, 对"当下最主要的困难"结构性未经检验。

输出: data/regime_shift_evidence.md + regime_shift_yearly.csv
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def yearly(name, ts, price, unit):
    df = pd.DataFrame({"ts": pd.to_datetime(ts, utc=True, errors="coerce"),
                       "p": pd.to_numeric(price, errors="coerce")}).dropna()
    df["year"] = df.ts.dt.year
    g = df.groupby("year")["p"]
    out = pd.DataFrame({
        "dataset": name, "unit": unit,
        "n": g.size(),
        "neg_pct": (g.apply(lambda s: (s < 0).mean() * 100)).round(3),
        "min": g.min().round(2),
        "p99": g.quantile(0.99).round(2),
        "max": g.max().round(2),
        "median": g.median().round(2),
    }).reset_index()
    return out


def main():
    frames = []

    # --- Lago-DE (经典基准, 2012-2018) ---
    f = os.path.join(HERE, "lago_benchmark", "DE.csv")
    d = pd.read_csv(f)
    tcol = d.columns[0]
    pcol = next(c for c in d.columns if "price" in c.lower())
    frames.append(yearly("Lago-DE (经典基准)", d[tcol], d[pcol], "EUR/MWh"))

    # --- UniElecPrice-Germany (2015-2024, 同一市场向后延伸) ---
    f = os.path.join(HERE, "unielecprice", "by_country", "Germany.csv")
    if os.path.exists(f):
        d = pd.read_csv(f)
        frames.append(yearly("UniElec-Germany", d.timestamp, d.price,
                             "EUR/MWh"))

    # --- GEFCom2014-P (最经典竞赛集, 2011-2013) ---
    f = os.path.join(HERE, "gefcom2014", "GEFCom2014P_hourly.csv")
    if os.path.exists(f):
        d = pd.read_csv(f)
        frames.append(yearly("GEFCom2014-P (经典竞赛)", d.timestamp, d.price,
                             "USD/MWh"))

    # --- NEM SA1 (当代极端市场, 2021-2025) ---
    f = os.path.join(HERE, "nem_aemo", "clean", "SA1_price.csv")
    if os.path.exists(f):
        d = pd.read_csv(f)
        frames.append(yearly("NEM-SA1 (当代极端)", d.timestamp, d.price,
                             "AUD/MWh"))

    allx = pd.concat(frames, ignore_index=True)
    allx.to_csv(os.path.join(HERE, "regime_shift_yearly.csv"),
                index=False, encoding="utf-8-sig")

    lines = ["# 极端电价制度漂移：经典基准 vs 当代市场（逐年实证）", ""]
    lines.append("> 全部数字由本地公开数据直接统计，脚本 "
                 "`data/regime_shift_evidence.py`，可复现。")
    lines.append("")
    for name, g in allx.groupby("dataset", sort=False):
        u = g.unit.iloc[0]
        lines.append(f"## {name} （{u}）")
        lines.append("")
        lines.append("| 年份 | 样本 | 负电价占比% | 最深负价 | 中位数 | p99 | 最大值 |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            lines.append(f"| {int(r.year)} | {int(r.n)} | {r.neg_pct} | "
                         f"{r['min']} | {r['median']} | {r.p99} | {r['max']} |")
        lines.append("")

    # 汇总对照
    lines.append("## 一句话对照")
    lines.append("")
    lines.append("| 时代 | 代表数据 | 覆盖年份 | 负电价占比 |")
    lines.append("|---|---|---|---|")
    for nm in allx.dataset.unique():
        g = allx[allx.dataset == nm]
        tot_n = g.n.sum()
        tot_neg = (g.neg_pct / 100 * g.n).sum()
        lines.append(f"| — | {nm} | {int(g.year.min())}–{int(g.year.max())} | "
                     f"{tot_neg/tot_n*100:.3f}% |")

    md = os.path.join(HERE, "regime_shift_evidence.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print(allx.to_string(index=False))
    print(f"\nsaved -> {md}")


if __name__ == "__main__":
    main()
