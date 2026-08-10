# -*- coding: utf-8 -*-
import pandas as pd, os, json, zipfile

base = r"D:\作业\science\solar_leak_price_model\data\raw\unielecprice"
print("=== UniElecPrice meta.json 结构 ===")
meta = json.load(open(os.path.join(base, "meta.json"), encoding="utf-8"))
print("top keys:", list(meta.keys()))
print("title:", meta.get("title", {}))
print("doi:", meta.get("doi"))
print("publication:", meta.get("metadata", {}).get("publication_date", "?"))
# 文件清单
files = meta.get("files", [])
print(f"official files: {len(files)}")
for f in files[:20]:
    print(f"  {f['key']} ({f.get('size')}B) md5={f.get('checksum')}")
if len(files) > 20:
    print(f"  ... 共 {len(files)} 个")

print()
print("=== zip 内容核验 ===")
zp = os.path.join(base, "UniElecPrice.zip")
z = zipfile.ZipFile(zp)
names = z.namelist()
print(f"zip 条目数: {len(names)}")
for n in names[:15]:
    print(f"  {n}")
dirs = set()
for n in names:
    parts = n.split("/")
    dirs.add(parts[0] if parts[0] else "?")
print("zip 顶层目录:", dirs)

print()
print("=== by_country 核验 ===")
bc = os.path.join(base, "by_country")
csvs = sorted([f for f in os.listdir(bc) if f.endswith(".csv")])
print(f"by_country CSV: {len(csvs)}")
for c in csvs:
    p = os.path.join(bc, c)
    df = pd.read_csv(p)
    print(f"  {c}: {df.shape} cols={list(df.columns)[:6]}")
