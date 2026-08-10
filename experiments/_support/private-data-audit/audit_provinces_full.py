# -*- coding: utf-8 -*-
import os, pandas as pd

DATA = r"D:\作业\science\solar_leak_price_model\data\raw\provinces"

FILES = [
    ("宁夏", os.path.join(DATA, "宁夏24h电价数据集.xlsx")),
    ("甘肃", os.path.join(DATA, "甘肃24h电价数据集.xlsx")),
    ("陕西", os.path.join(DATA, "陕西24h电价数据集(1).xlsx")),
    ("青海", os.path.join(DATA, "青海24h电价数据集.xlsx")),
    ("山东", os.path.join(DATA, "shandong_pmos_hourly.csv")),
]

def detect(path):
    with open(path, "rb") as f:
        head = f.read(4)
    if head[:2] == b"PK":
        return "xlsx"
    return "text"

def load(path):
    t = detect(path)
    if t == "xlsx":
        xls = pd.ExcelFile(path)
        sh = xls.sheet_names[0]
        return xls.parse(sh), sh
    enc = None
    for e in ("utf-8", "gbk", "gb18030"):
        try:
            df = pd.read_csv(path, encoding=e)
            enc = e
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if enc is None:
        df = pd.read_csv(path, encoding="latin1")
        enc = "latin1"
    return df, enc

def analyze(prov, path):
    print("=" * 72)
    df, src = load(path)
    print(f"[{prov}] 实际格式: {'xlsx-sheet:'+src if src!='gbk' and src!='utf-8' and src!='latin1' and src!='gb18030' else 'csv-encoding:'+src}")
    print("  shape:", df.shape)
    print("  cols:", list(df.columns))
    tcol = df.columns[0]
    print("  首列 dtype:", df[tcol].dtype, "| 前2行:", list(df[tcol].head(2)))
    print("  末2行:", list(df[tcol].tail(2)))

    # 时间列解析
    t = pd.to_datetime(df[tcol], errors="coerce")
    tmin, tmax = t.min(), t.max()
    print(f"  时间范围: {tmin} ~ {tmax}  | 有效时间数: {t.notna().sum()} / {len(t)}")
    dt = pd.Series(t).diff().dropna()
    vals = sorted(dt.dt.total_seconds().value_counts().items())[:6]
    print("  相邻时间差 top:", vals)

    price_cols = [c for c in df.columns if "电价" in str(c)]
    for pc in price_cols:
        s = pd.to_numeric(df[pc], errors="coerce")
        neg = (s < 0).sum()
        print(f"  [{pc}] 非空 {s.notna().sum()}/{len(s)} (缺失率 {100*(1-s.notna().mean()):.2f}%) "
              f"| min {s.min()} max {s.max()} | 均值 {s.mean():.2f} 中位 {s.median():.2f} "
              f"| 负价 {neg} 点 ({100*neg/len(s):.3f}%) | 0值 {int((s==0).sum())}")

for prov, path in FILES:
    try:
        analyze(prov, path)
    except Exception as e:
        print("=" * 72)
        print(f"[{prov}] ERROR: {type(e).__name__}: {e}")
