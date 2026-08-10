# -*- coding: utf-8 -*-
import pandas as pd, os, json

print("=== GEFCom2014-P 核验 ===")
p = r"D:\作业\science\solar_leak_price_model\data\raw\gefcom2014\GEFCom2014P_hourly.csv"
df = pd.read_csv(p)
print(f"shape={df.shape} cols={list(df.columns)}")
print(f"head:\n{df.head(3)}")
# 找价格列
pc = [c for c in df.columns if "price" in c.lower() or "P" == c]
for c in pc:
    s = pd.to_numeric(df[c], errors="coerce")
    print(f"  {c}: min={s.min()} max={s.max()} 负价={(s<0).sum()} ({100*(s<0).mean():.3f}%) 非空={s.notna().sum()}/{len(s)}")

print()
print("=== UniElecPrice 核验 ===")
base = r"D:\作业\science\solar_leak_price_model\data\raw\unielecprice"
meta = json.load(open(os.path.join(base, "meta.json"), encoding="utf-8"))
countries = [c for c in os.listdir(os.path.join(base, "by_country")) if c.endswith(".csv")]
print(f"by_country CSV 数: {len(countries)}")
print(f"meta.json 国家数: {len(meta) if isinstance(meta, list) else list(meta.keys())[:5]}")
if isinstance(meta, list):
    print("meta 是国家清单:")
    for m in meta[:5]:
        print("  ", m)
