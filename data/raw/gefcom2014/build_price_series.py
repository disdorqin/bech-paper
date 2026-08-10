# -*- coding: utf-8 -*-
"""
把 GEFCom2014-P (电价赛道) 的滚动任务文件合并成一条连续小时序列。

数据结构(已核实):
  Price/Task N/TaskN_P.csv  列 = ZONEID, timestamp, Forecasted Total Load,
                                 Forecasted Zonal Load, Zonal Price
  timestamp 格式 = MMDDYYYY H:00
  每个 Task N 文件包含"截至该任务前的全部历史" + 待预测 24h(Zonal Price 为空)
  Task 15 历史最长; 再拼 `Solution to Task15_P.csv` 的最后 24h 即得完整序列。

目标变量: Zonal Price = 小时级 locational marginal price (USD/MWh)
外生变量: Forecasted Total Load(系统负荷预测), Forecasted Zonal Load(分区负荷预测)
          —— 二者都是"预测值"而非实际值, 天然 cutoff-safe, 无未来信息泄露。

引用: Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli,
      Rob J. Hyndman. "Probabilistic energy forecasting: Global Energy
      Forecasting Competition 2014 and beyond." IJF 32(3):896-913, 2016.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PRICE = os.path.join(HERE, "price_track", "Price")


def parse_ts(s):
    return pd.to_datetime(s, format="%m%d%Y %H:%M")


def main():
    hist = pd.read_csv(os.path.join(PRICE, "Task 15", "Task15_P.csv"))
    sol = pd.read_csv(
        os.path.join(PRICE, "Solution to Task15", "Solution to Task15_P.csv")
    )

    hist["timestamp"] = parse_ts(hist["timestamp"])
    sol["timestamp"] = parse_ts(sol["timestamp"])

    # 用 solution 回填 Task15 末尾 24h 的空价格
    fill = sol.set_index("timestamp")["Zonal Price"]
    mask = hist["Zonal Price"].isna()
    hist.loc[mask, "Zonal Price"] = hist.loc[mask, "timestamp"].map(fill)

    df = hist.rename(
        columns={
            "Zonal Price": "price",
            "Forecasted Total Load": "load_system_fc",
            "Forecasted Zonal Load": "load_zonal_fc",
        }
    )[["timestamp", "price", "load_system_fc", "load_zonal_fc"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    out = os.path.join(HERE, "GEFCom2014P_hourly.csv")
    df.to_csv(out, index=False)

    n = len(df)
    span = pd.date_range(df.timestamp.iloc[0], df.timestamp.iloc[-1], freq="h")
    print(f"rows={n}, expected_by_span={len(span)}, "
          f"gaps={len(span)-n}, nan_price={df.price.isna().sum()}")
    print(f"range: {df.timestamp.iloc[0]} -> {df.timestamp.iloc[-1]}")
    print(f"price: min={df.price.min():.2f} max={df.price.max():.2f} "
          f"mean={df.price.mean():.2f} median={df.price.median():.2f}")
    neg = (df.price < 0).sum()
    print(f"negative: {neg} ({neg/n*100:.3f}%)")
    thr = df.price.quantile(0.99)
    print(f"p99={thr:.2f}, max/p99 ratio={df.price.max()/thr:.1f}x")
    print(f"saved -> {out} ({os.path.getsize(out)/1e6:.2f}MB)")


if __name__ == "__main__":
    main()
