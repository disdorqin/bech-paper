# -*- coding: utf-8 -*-
"""
把 LSTNet 原始 .txt.gz 转成 Autoformer/PatchTST/iTransformer 通用的
`date + variables + OT` CSV 格式, 与已下载的 ETT/weather/traffic 对齐。

来源: github.com/laiguokun/multivariate-time-series-data (LSTNet, SIGIR'18)
  - electricity/electricity.txt.gz : UCI ElectricityLoadDiagrams20112014,
        321 客户, 小时粒度, 26304 行
  - solar-energy/solar_AL.txt.gz   : 阿拉巴马州 137 个光伏电站, 10min 粒度,
        52560 行 (2006 全年)

Autoformer 官方 electricity.csv 的时间基准: 2016-07-01 02:00:00 起, 小时步长。
Solar 通用做法: 2006-01-01 00:00:00 起, 10 分钟步长。
"""
import gzip
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

SPECS = [
    {
        "src": "electricity.txt.gz",
        "dst": "electricity.csv",
        "start": "2016-07-01 02:00:00",
        "freq": "h",
        "expect_rows": 26304,
        "expect_cols": 321,
    },
    {
        "src": "solar_AL.txt.gz",
        "dst": "solar_AL.csv",
        "start": "2006-01-01 00:00:00",
        "freq": "10min",
        "expect_rows": 52560,
        "expect_cols": 137,
    },
]


def convert(spec):
    src = os.path.join(HERE, spec["src"])
    with gzip.open(src, "rt") as f:
        arr = np.loadtxt(f, delimiter=",")
    n, m = arr.shape
    idx = pd.date_range(spec["start"], periods=n, freq=spec["freq"])
    cols = [str(i) for i in range(m - 1)] + ["OT"]
    df = pd.DataFrame(arr, columns=cols)
    df.insert(0, "date", idx)
    dst = os.path.join(HERE, spec["dst"])
    df.to_csv(dst, index=False)

    ok_rows = n == spec["expect_rows"]
    ok_cols = m == spec["expect_cols"]
    print(
        f"{spec['dst']}: rows={n} (expect {spec['expect_rows']}, "
        f"{'OK' if ok_rows else 'MISMATCH'}), "
        f"cols={m} (expect {spec['expect_cols']}, "
        f"{'OK' if ok_cols else 'MISMATCH'}), "
        f"range={idx[0]} -> {idx[-1]}, size={os.path.getsize(dst)/1e6:.2f}MB"
    )


if __name__ == "__main__":
    for s in SPECS:
        convert(s)
