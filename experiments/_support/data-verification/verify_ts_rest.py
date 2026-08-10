# -*- coding: utf-8 -*-
import os, zipfile
import pandas as pd

print("=== weather/exchange/illness 用 latin1/gbk 读 ===")
tb = r"D:\作业\science\solar_leak_price_model\data\raw\ts_benchmarks"
for f in ["weather.csv", "exchange_rate.csv", "illness.csv"]:
    p = os.path.join(tb, f)
    try:
        df = pd.read_csv(p, encoding="gbk", errors="replace")
        print(f"{f}: {df.shape} 首列={df.columns[0]} (gbk)")
    except Exception as e:
        df = pd.read_csv(p, encoding="latin1")
        print(f"{f}: {df.shape} 首列={df.columns[0]} (latin1)")

print()
print("=== UniElecPrice by_country 缺 Canada/USA 原因 ===")
base = r"D:\作业\science\solar_leak_price_model\data\raw\unielecprice"
z = zipfile.ZipFile(os.path.join(base, "UniElecPrice.zip"))
for c in ["Canada", "USA"]:
    cnt = sum(1 for n in z.namelist() if n.startswith(c + "/"))
    size = sum(z.getinfo(n).file_size for n in z.namelist() if n.startswith(c + "/"))
    print(f"{c}: zip 内 {cnt} 个文件, 总 {size/1e6:.1f}MB")
    print("  前3文件:", [n for n in z.namelist() if n.startswith(c+"/")][:3])

print()
print("=== by_country 是否有这些国家的替代(如 Canada_USA 合并) ===")
bc = [f[:-4] for f in os.listdir(os.path.join(base, "by_country")) if f.endswith(".csv")]
print("包含 Canada/USA/合并:", [x for x in bc if "canada" in x.lower() or "usa" in x.lower() or "americ" in x.lower() or "north" in x.lower()])
