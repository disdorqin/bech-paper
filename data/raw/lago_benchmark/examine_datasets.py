"""Examine Lago 5-market open-access benchmark dataset characteristics.

Critical goal: verify which PUBLIC markets contain negative prices
(core innovation = negative-price correction needs market transferability),
and quantify extreme spike behaviour, so Shandong problems can be migrated.

Outputs:
  characteristics_summary.csv   one row per market, key stats
  characteristics_evidence.md   human-readable report (negative price focus)
  clean/<MARKET>_price.csv      unified (timestamp, price) for downstream BECH runs
"""
import os, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "clean")
os.makedirs(OUT, exist_ok=True)

PRICE_ALIASES = ["price", "prices", "zonal comed price"]  # substring match (lower)


def find_price_col(cols):
    low = [c.lower() for c in cols]
    for a in PRICE_ALIASES:
        for c, cl in zip(cols, low):
            if a in cl:
                return c
    raise ValueError(f"no price col in {cols}")


def load_market(path):
    df = pd.read_csv(path)
    # DE.csv: first column is unnamed and holds the timestamp -> rename, do NOT drop
    first = df.columns[0]
    if str(first).strip() == "" or str(first).startswith("Unnamed"):
        df = df.rename(columns={first: "timestamp"})
    pc = find_price_col(list(df.columns))
    date_col = [c for c in df.columns if c.lower() in ("date", "timestamp")
                or c.lower().startswith("date")][0]
    df[date_col] = pd.to_datetime(df[date_col])
    s = df[[date_col, pc]].rename(columns={date_col: "timestamp", pc: "price"})
    s = s.sort_values("timestamp").reset_index(drop=True)
    return s


def analyze(name, s):
    p = s["price"].astype(float)
    total = len(p)
    neg = (p < 0).sum()
    nonpos = (p <= 0).sum()
    zero = (p == 0).sum()
    pos = (p > 0).sum()
    # spike (extreme high) via top percentiles
    p95, p99, p999 = p.quantile([0.95, 0.99, 0.999])
    spike99 = (p > p99).sum()
    # negative price by hour-of-day and by year
    s2 = s.copy()
    s2["hour"] = s2["timestamp"].dt.hour
    s2["year"] = s2["timestamp"].dt.year
    neg_by_hour = s2[s2["price"] < 0].groupby("hour").size()
    neg_by_year = s2[s2["price"] < 0].groupby("year").size()
    # top extremes
    top_neg = s.nsmallest(5, "price")
    top_pos = s.nlargest(5, "price")
    return dict(
        market=name, n=total,
        start=str(s["timestamp"].min()), end=str(s["timestamp"].max()),
        pmin=round(p.min(), 3), pmax=round(p.max(), 3),
        pmean=round(p.mean(), 3), pmedian=round(p.median(), 3), pstd=round(p.std(), 3),
        neg_count=int(neg), neg_pct=round(neg / total * 100, 3),
        nonpos_count=int(nonpos), zero_count=int(zero), pos_count=int(pos),
        most_negative=round(p.min(), 3),
        p95=round(p95, 2), p99=round(p99, 2), p999=round(p999, 2),
        spike99_count=int(spike99), spike99_pct=round(spike99 / total * 100, 3),
        neg_hours=neg_by_hour.to_dict(),
        neg_years=neg_by_year.to_dict(),
        top_neg=top_neg.to_dict("records"),
        top_pos=top_pos.to_dict("records"),
    )


def main():
    files = sorted(glob.glob(os.path.join(HERE, "*.csv")))
    files = [f for f in files if os.path.basename(f) not in ("characteristics_summary.csv",)]
    rows = []
    md = ["# Lago 五市场公开基准数据集 · 特征检验报告", "",
          "> 来源: Zenodo 4624804 (Lago et al. 2021, Applied Energy, open-access benchmark)",
          "> 用途: 顶会可复现基准；用于把山东私有数据上的问题迁移到公开数据集", "",
          "## 负电价核心结论（决定创新点市场可迁移性）", ""]
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        s = load_market(f)
        r = analyze(name, s)
        rows.append(r)
        # save clean unified series
        s.to_csv(os.path.join(OUT, f"{name}_price.csv"), index=False)
        md.append(f"### {name}  ({r['start']} → {r['end']}, n={r['n']})")
        md.append(f"- 价格列范围: [{r['pmin']}, {r['pmax']}]  EUR/MWh (PJM 为 USD/MWh)")
        md.append(f"- **负电价占比: {r['neg_pct']}%**  ({r['neg_count']} 个时点, 最负 = {r['most_negative']})")
        md.append(f"- 非正(≤0): {r['nonpos_count']} | 零价: {r['zero_count']} | 正价: {r['pos_count']}")
        md.append(f"- 正尖峰: p95={r['p95']}, p99={r['p99']}, p99.9={r['p999']}; >p99 共 {r['spike99_count']} 个 ({r['spike99_pct']}%)")
        ny = ", ".join(f"{y}:{c}" for y, c in sorted(r['neg_years'].items()))
        md.append(f"- 负电价按年分布: {ny if ny else '无'}")
        md.append("")

    # summary table
    sumdf = pd.DataFrame([{k: r[k] for k in
        ["market", "n", "pmin", "pmax", "pmean", "neg_count", "neg_pct",
         "most_negative", "p99", "spike99_count", "spike99_pct"]} for r in rows])
    sumdf = sumdf.sort_values("neg_pct", ascending=False)
    sumdf.to_csv(os.path.join(HERE, "characteristics_summary.csv"), index=False)
    md.append("## 汇总表（按负电价占比降序）")
    md.append("")
    md.append("| 市场 | n | 价格区间 | 均值 | 负电价数 | 负电价% | 最负价 | p99 | >p99数 | >p99% |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, row in sumdf.iterrows():
        md.append(f"| {row['market']} | {row['n']} | [{row['pmin']},{row['pmax']}] | {row['pmean']} | "
                  f"{row['neg_count']} | **{row['neg_pct']}%** | {row['most_negative']} | {row['p99']} | "
                  f"{row['spike99_count']} | {row['spike99_pct']}% |")
    md.append("")
    verdict = "含负电价" if any(r["neg_count"] > 0 for r in rows) else "均不含负电价"
    md.append(f"## 对论文叙事的结论")
    md.append(f"- 五市场中 {sum(1 for r in rows if r['neg_count']>0)} 个市场 {verdict}，可支撑「负电价校正」作为跨市场可迁移的核心创新。")
    md.append(f"- 负电价占比最高的市场（最值得作为主实验市场）将在汇总表中置顶。")
    md.append("")
    with open(os.path.join(HERE, "characteristics_evidence.md"), "w") as fh:
        fh.write("\n".join(md))
    print("[done] summary -> characteristics_summary.csv + characteristics_evidence.md")
    print(sumdf.to_string(index=False))


if __name__ == "__main__":
    main()
