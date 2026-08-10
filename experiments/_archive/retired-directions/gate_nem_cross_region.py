# -*- coding: utf-8 -*-
"""NEM 5区跨区图结构门禁:相关矩阵 + 格兰杰因果检验"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(r"D:\作业\science\solar_leak_price_model\data\raw\nem_aemo\clean")
REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]


def load_region(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}_hourly.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def main():
    frames = {r: load_region(r) for r in REGIONS}
    for r, df in frames.items():
        print(f"[{r}] price range {df['price'].min():.2f}..{df['price'].max():.2f}, "
              f"neg_pct={(df['price'] < 0).mean()*100:.2f}%, n={len(df)}, "
              f"dup_idx={df.index.duplicated().sum()}")

    price = pd.concat({r: frames[r]["price"] for r in REGIONS}, axis=1)
    price = price.dropna()
    print(f"\n对齐后样本量: {len(price)}, 时间范围 {price.index.min()} .. {price.index.max()}")

    print("\n=== Pearson 相关矩阵 ===")
    print(price.corr(method="pearson").round(3).to_string())

    print("\n=== Spearman 相关矩阵 ===")
    print(price.corr(method="spearman").round(3).to_string())

    pairs = [(a, b) for i, a in enumerate(REGIONS) for b in REGIONS[i + 1:]]
    cors = {p: price[p[0]].corr(price[p[1]], method="pearson") for p in pairs}
    most = max(cors, key=cors.get)
    least = min(cors, key=cors.get)
    print(f"\n最相关区对: {most} r={cors[most]:.3f}")
    print(f"最不相关区对: {least} r={cors[least]:.3f}")

    print("\n=== 格兰杰因果检验(手写嵌套F检验) ===")
    from scipy import stats as sstats
    import numpy.linalg as la

    def granger_test(x: np.ndarray, y: np.ndarray, k: int):
        """H0: x 的滞后对预测 y 无增量价值。返回 (F, p)。"""
        n = len(y)
        # 受限模型: y_t = a0 + sum a_i y_{t-i}
        Y = y[k:]
        Xr = np.column_stack([np.ones(n - k)] + [y[k - i: n - i] for i in range(1, k + 1)])
        Xf = np.column_stack([Xr] + [x[k - i: n - i] for i in range(1, k + 1)])
        betar, *_ = la.lstsq(Xr, Y, rcond=None)
        betaf, *_ = la.lstsq(Xf, Y, rcond=None)
        rss_r = float(np.sum((Y - Xr @ betar) ** 2))
        rss_f = float(np.sum((Y - Xf @ betaf) ** 2))
        df1, df2 = k, n - Xf.shape[1]
        f = ((rss_r - rss_f) / df1) / (rss_f / df2)
        p = 1 - sstats.f.cdf(f, df1, df2)
        return f, p

    lags = [1, 2, 3, 24]
    for a, b in [(most[0], most[1]), (least[0], least[1]), (least[1], least[0])]:
        df = pd.DataFrame({a: price[a], b: price[b]}).dropna()
        x = df[b].values.astype(float)
        y = df[a].values.astype(float)
        print(f"\n--- {b} -> {a} (n={len(df)}) ---")
        for lag in lags:
            f, p = granger_test(x, y, lag)
            print(f"  lag={lag:2d}: F={f:.2f}, p={p:.4g}")

    print("\n=== demand 与价格同期相关(每区) ===")
    for r in REGIONS:
        df = frames[r].dropna(subset=["price", "demand"])
        print(f"  {r}: Pearson={df['price'].corr(df['demand']):.3f}, "
              f"Spearman={df['price'].corr(df['demand'], method='spearman'):.3f}")

    print("\n=== 跨区 demand 与价格相关性(共享外生检验) ===")
    price_df = pd.concat({r: frames[r]["price"] for r in REGIONS}, axis=1)
    demand_df = pd.concat({r: frames[r]["demand"] for r in REGIONS}, axis=1)
    print("  " + "        " + "  ".join(f"{s:>7}" for s in REGIONS))
    for r in REGIONS:
        row = []
        for s in REGIONS:
            pr = price_df[r].dropna()
            dm = demand_df[s].reindex(pr.index).dropna()
            row.append(f"{pr.reindex(dm.index).corr(dm):7.3f}")
        print(f"  price[{r}]  " + "  ".join(row))


if __name__ == "__main__":
    main()
