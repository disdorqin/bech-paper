# -*- coding: utf-8 -*-
"""
从 AEMO 官方公开端点拉取 NEM 现货电价与需求数据。

来源(公开、无需认证):
  https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/
      data-nem/aggregated-data
  文件模板: https://aemo.com.au/aemo/data/nem/priceanddemand/
            PRICE_AND_DEMAND_{YYYYMM}_{REGION}.csv

字段: REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE
  RRP = Regional Reference Price (AUD/MWh)
  市场价格区间(AEMC 规定): 下限 -1000 AUD/MWh, 上限 ~16600 AUD/MWh(逐年上调)
  => 天然同时具备"深度负电价"与"极端尖峰", 是极端电价论文的黄金公开集

注: 2021-07 起 NEM 结算由 30min 转 5min, 早期文件为 30min 粒度。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://aemo.com.au/aemo/data/nem/priceanddemand"

REGIONS = ["NSW1", "VIC1", "SA1", "QLD1", "TAS1"]
YEARS = [2021, 2022, 2023, 2024, 2025]
MONTHS = list(range(1, 13))

RAW = os.path.join(HERE, "raw")
os.makedirs(RAW, exist_ok=True)


def fetch_one(region, ym):
    url = f"{BASE}/PRICE_AND_DEMAND_{ym}_{region}.csv"
    out = os.path.join(RAW, f"{region}_{ym}.csv")
    if os.path.exists(out) and os.path.getsize(out) > 5000:
        return True, os.path.getsize(out), "cached"
    r = subprocess.run(
        ["curl", "-sS", "-L", "--max-time", "90", "-o", out,
         "-w", "%{http_code}", url],
        capture_output=True, text=True,
    )
    code = (r.stdout or "").strip()
    size = os.path.getsize(out) if os.path.exists(out) else 0
    ok = code == "200" and size > 5000
    if not ok and os.path.exists(out):
        os.remove(out)
    return ok, size, code


def main():
    total_ok, total_fail = 0, 0
    fails = []
    for region in REGIONS:
        got = 0
        for y in YEARS:
            for m in MONTHS:
                ym = f"{y}{m:02d}"
                ok, size, code = fetch_one(region, ym)
                if ok:
                    got += 1
                    total_ok += 1
                else:
                    total_fail += 1
                    fails.append(f"{region}_{ym}(code={code})")
        print(f"[{region}] fetched {got} monthly files", flush=True)
    print(f"\nOK={total_ok}  FAIL={total_fail}")
    if fails:
        print("failed (通常是尚未发生的未来月份, 属正常):")
        print("  " + ", ".join(fails[:40]))
        if len(fails) > 40:
            print(f"  ... and {len(fails)-40} more")


if __name__ == "__main__":
    main()
