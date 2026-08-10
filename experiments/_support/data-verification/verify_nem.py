# -*- coding: utf-8 -*-
import pandas as pd, os

base = r"D:\作业\science\solar_leak_price_model\data\raw\nem_aemo\clean"
print("=== NEM 5区 hourly 价格核验 ===")
for r in ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]:
    p = os.path.join(base, r + "_hourly.csv")
    df = pd.read_csv(p)
    print(f"\n{r}_hourly: shape={df.shape} cols={list(df.columns)}")
    tcol = df.columns[0]
    t = pd.to_datetime(df[tcol], errors="coerce")
    pc = df.columns[1] if len(df.columns) > 1 else None
    s = pd.to_numeric(df[pc], errors="coerce") if pc else None
    print(f"  时间: {t.min()} ~ {t.max()} 有效{t.notna().sum()}")
    if s is not None:
        neg = (s < 0).sum()
        print(f"  {pc}: min={s.min()} max={s.max()} 负价={neg} ({100*neg/len(s):.2f}%)")
