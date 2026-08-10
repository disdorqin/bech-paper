# -*- coding: utf-8 -*-
import hashlib, os, json, zipfile
import pandas as pd

print("=== UniElecPrice zip MD5 核验 ===")
base = r"D:\作业\science\solar_leak_price_model\data\raw\unielecprice"
meta = json.load(open(os.path.join(base, "meta.json"), encoding="utf-8"))
off = meta["files"][0]
zp = os.path.join(base, "UniElecPrice.zip")
sz = os.path.getsize(zp)
h = hashlib.md5(open(zp, "rb").read()).hexdigest()
print(f"本地 {os.path.basename(zp)}: {sz}B / 官方 {off['size']}B, md5 {'MATCH' if h==off['checksum'].split(':')[1] else 'MISMATCH'}")
print(f"  本地={h}")
print(f"  官方={off['checksum']}")

print()
print("=== zip 国家目录 vs by_country 覆盖对账 ===")
z = zipfile.ZipFile(zp)
zip_dirs = {}
for n in z.namelist():
    parts = n.split("/")
    if len(parts) >= 1 and parts[0]:
        zip_dirs[parts[0]] = zip_dirs.get(parts[0], 0) + 1
bc = sorted(f[:-4] for f in os.listdir(os.path.join(base, "by_country")) if f.endswith(".csv"))
zip_countries = sorted(k for k in zip_dirs if k not in ("Price_Unit_by_Country.csv",))
print(f"zip 国家目录: {len(zip_countries)} 个, by_country CSV: {len(bc)} 个")
only_zip = sorted(set(zip_countries) - set(bc))
only_bc = sorted(set(bc) - set(zip_countries))
print(f"仅 zip 有: {only_zip}")
print(f"仅 by_country 有: {only_bc}")
print(f"两者一致: {not only_zip and not only_bc}")

print()
print("=== Price_Unit_by_Country.csv (zip内) ===")
try:
    pu = z.read("Price_Unit_by_Country.csv")
    print(pu.decode("utf-8", errors="replace")[:800])
except KeyError:
    for n in z.namelist():
        if "Price_Unit" in n or "Unit" in n:
            print("找到:", n)

print()
print("=== TS 基准核验 ===")
tb = r"D:\作业\science\solar_leak_price_model\data\raw\ts_benchmarks"
for f in ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv", "electricity.csv", "solar_AL.csv", "traffic.csv", "weather.csv", "exchange_rate.csv", "illness.csv"]:
    p = os.path.join(tb, f)
    if not os.path.exists(p):
        print(f"{f}: 不存在"); continue
    df = pd.read_csv(p)
    print(f"{f}: {df.shape} 首列={df.columns[0]}")
