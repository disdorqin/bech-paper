# -*- coding: utf-8 -*-
import hashlib, json, os, sys
import pandas as pd

base = r"D:\作业\science\solar_leak_price_model\data\raw\lago_benchmark"
meta = json.load(open(os.path.join(base, "zenodo_meta.json"), encoding="utf-8"))
print("=== Lago MD5 核验 (vs zenodo_meta.json 官方) ===")
for f in meta["files"]:
    p = os.path.join(base, f["key"])
    sz = os.path.getsize(p)
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    ok = (h == f["checksum"].split(":")[1]) and (sz == f["size"])
    print(f"{f['key']}: 本地 {sz}B / 官方 {f['size']}B | md5 {'MATCH' if ok else 'MISMATCH'} 本地={h}")
print()
print("=== Lago 价格特征核验 ===")
for name in ["DE", "BE", "FR", "NP", "PJM"]:
    p = os.path.join(base, name + ".csv")
    df = pd.read_csv(p)
    print(f"{name}: shape={df.shape} cols={list(df.columns)}")
    pc = [c for c in df.columns if "price" in c.lower()]
    if pc:
        s = pd.to_numeric(df[pc[0]], errors="coerce")
        neg = (s < 0).sum()
        print(f"  {pc[0]}: min={s.min()} max={s.max()} 负价={neg} ({100*neg/len(s):.3f}%)")
