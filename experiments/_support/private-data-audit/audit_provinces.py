# -*- coding: utf-8 -*-
import sys, json, os
import pandas as pd

DATA = r"D:\作业\science\solar_leak_price_model\data\raw\provinces"
OUT  = r"D:\作业\science\solar_leak_price_model\docs\paper_prep\07_省级数据集可用性审计.md"

FILES = [
    ("宁夏", os.path.join(DATA, "宁夏24h电价数据集.xlsx")),
    ("甘肃", os.path.join(DATA, "甘肃24h电价数据集.xlsx")),
    ("陕西", os.path.join(DATA, "陕西24h电价数据集(1).xlsx")),
    ("青海", os.path.join(DATA, "青海24h电价数据集.xlsx")),
    ("山东", os.path.join(DATA, "shandong_pmos_hourly.csv")),
]

def audit_file(prov, path):
    info = {"prov": prov, "path": os.path.basename(path)}
    if path.endswith(".xlsx"):
        xls = pd.ExcelFile(path)
        sheets = {}
        for sh in xls.sheet_names:
            df = xls.parse(sh, nrows=5)
            sheets[sh] = {
                "shape_preview": list(df.shape),
                "cols": [str(c) for c in df.columns],
            }
        info["type"] = "xlsx"
        info["sheets"] = sheets
    else:
        df = pd.read_csv(path, nrows=5)
        info["type"] = "csv"
        info["cols"] = [str(c) for c in df.columns]
    return info

for prov, path in FILES:
    try:
        info = audit_file(prov, path)
        print("=" * 70)
        print(info["prov"], info["type"])
        if "sheets" in info:
            for sh, meta in info["sheets"].items():
                print("  sheet:", repr(sh), "ncols(prev):", meta["shape_preview"][1], "cols:", meta["cols"])
        else:
            print("  cols:", info["cols"])
    except Exception as e:
        print("=" * 70)
        print(info["prov"], "ERROR:", repr(e))
