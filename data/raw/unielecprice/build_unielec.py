# -*- coding: utf-8 -*-
"""
解包 UniElecPrice (Global Day-Ahead Electricity Price Dataset) 并生成
按国家的连续序列 + 负电价/尖峰概览。

来源(免认证公开镜像): Zenodo record 16284828
  DOI 10.17632/s54n4tyyz4.1 (Mendeley Data, 2025-07-21)
  论文: M. H. Ullah et al., "Descriptor: Unified Cross-Regional Time-Series
        Day-Ahead Electricity Price Dataset (UniElecPrice)",
        IEEE Data Descriptions, vol.2, pp.329-339, 2025.
        doi: 10.1109/IEEEDATA.2025.3609683
  注: IEEE DataPort 版需订阅, Mendeley 需登录, Zenodo 版可直连下载。

结构: <Country>/<Country>_<SourceOperator>_<Year>.csv
      + Price_Unit_by_Country.csv (各国计价单位)
"""
import os
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "UniElecPrice.zip")
EX = os.path.join(HERE, "extract")
OUT = os.path.join(HERE, "by_country")


def unpack():
    if not os.path.isdir(EX):
        zipfile.ZipFile(ZIP).extractall(EX)
    root = EX
    # 若解压后多套一层目录, 自动下钻
    entries = [e for e in os.listdir(root) if not e.startswith("__")]
    if len(entries) == 1 and os.path.isdir(os.path.join(root, entries[0])):
        root = os.path.join(root, entries[0])
    return root


def main():
    root = unpack()
    os.makedirs(OUT, exist_ok=True)

    unit_f = os.path.join(root, "Price_Unit_by_Country.csv")
    units = {}
    if os.path.exists(unit_f):
        u = pd.read_csv(unit_f)
        cols = list(u.columns)
        units = dict(zip(u[cols[0]].astype(str).str.strip(),
                         u[cols[1]].astype(str).str.strip()))
        print(f"units file cols={cols}, n={len(units)}")

    countries = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)))
    print(f"countries: {len(countries)}")

    def _collect_csvs(cdir):
        """递归收集国家目录下所有 csv(支持 <Country>/<Operator>/<Year>.csv 两层结构)。"""
        out = []
        for dirpath, _dirs, filenames in os.walk(cdir):
            for fn in sorted(filenames):
                if fn.endswith(".csv"):
                    out.append(os.path.join(dirpath, fn))
        return sorted(out)

    def _extract_series(f):
        """读单个 csv, 返回 (timestamp_series, price_series)。多节点列取均值。"""
        d = pd.read_csv(f)
        cols = list(d.columns)
        tcol = next((x for x in cols
                     if any(k in x.lower()
                            for k in ["date", "time", "utc", "period"])), cols[0])
        t = pd.to_datetime(d[tcol], errors="coerce", utc=True)
        # 价格列: 显式 price 列优先; 否则多节点取行均值, 单节点取该列
        pcands = [x for x in cols if "price" in x.lower()]
        if pcands:
            p = pd.to_numeric(d[pcands[0]], errors="coerce")
        else:
            numcols = [x for x in cols
                       if x != tcol
                       and pd.to_numeric(d[x], errors="coerce").notna().mean() > 0.5]
            if not numcols:
                return None, None
            p = (d[numcols].apply(pd.to_numeric, errors="coerce")
                           .mean(axis=1))
        return t, p

    rows = []
    for c in countries:
        cdir = os.path.join(root, c)
        files = _collect_csvs(cdir)
        if not files:
            continue
        parts_t, parts_p = [], []
        for f in files:
            try:
                t, p = _extract_series(f)
            except Exception:
                continue
            if t is None or len(t) == 0:
                continue
            parts_t.append(t)
            parts_p.append(p)
        if not parts_t:
            continue
        df = (pd.concat(parts_p, ignore_index=True)
                .to_frame("price"))
        df["timestamp"] = pd.concat(parts_t, ignore_index=True).values
        df = (df[["timestamp", "price"]].dropna()
                .drop_duplicates(subset="timestamp")
                .sort_values("timestamp").reset_index(drop=True))
        if len(df) < 100:
            continue
        df.to_csv(os.path.join(OUT, f"{c}.csv"), index=False)

        s = df.price
        n = len(s)
        neg = int((s < 0).sum())
        p99 = s.quantile(0.99)
        med = s.median()
        src = files[0].split("_")[1] if "_" in files[0] else "?"
        rows.append({
            "country": c, "source": src, "unit": units.get(c, "?"),
            "n": n,
            "years": f"{df.timestamp.dt.year.min()}-"
                     f"{df.timestamp.dt.year.max()}",
            "median": round(med, 2), "p99": round(p99, 2),
            "min": round(s.min(), 2), "max": round(s.max(), 2),
            "neg_n": neg, "neg_pct": round(neg / n * 100, 4),
            "max_over_p99": round(s.max() / p99, 2) if p99 > 0 else None,
        })

    sm = pd.DataFrame(rows).sort_values("neg_pct", ascending=False)
    sm.to_csv(os.path.join(HERE, "unielec_summary.csv"),
              index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 60)
    print(sm.to_string(index=False))
    print(f"\nsaved -> {os.path.join(HERE, 'unielec_summary.csv')}")
    print(f"per-country series -> {OUT}")


if __name__ == "__main__":
    main()
