"""Phase-1: peek the 4-province 24h electricity price xlsx files.
Auto-detects the price column and prints schema + basic extreme stats.
Read-only; no model training.
"""
import os, glob, sys
import pandas as pd

DATA_DIR = r"D:/作业/science/solar_leak_price_model/data"
FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.xlsx")))

def find_price_col(cols):
    # prefer column whose name hints at price / 价 / price
    for c in cols:
        s = str(c)
        if any(k in s.lower() for k in ["价", "price", "price_", "elec"]):
            return c
    # else first numeric-ish column
    return cols[0]

for f in FILES:
    name = os.path.basename(f)
    try:
        xl = pd.ExcelFile(f)
        print(f"\n===== {name} =====")
        print("sheets:", xl.sheet_names)
        df = pd.read_excel(f, sheet_name=xl.sheet_names[0])
        print("shape:", df.shape)
        print("columns:", list(df.columns))
        pc = find_price_col(df.columns)
        print("detected price col:", pc)
        print(df.head(3).to_string())
        # basic numeric coercion attempt
        ser = pd.to_numeric(df[pc], errors="coerce")
        print("non-null price rows:", int(ser.notna().sum()),
              "min:", ser.min(), "max:", ser.max(), "mean:", round(ser.mean(), 3))
    except Exception as e:
        print(f"\n===== {name} ===== ERROR: {e}")
