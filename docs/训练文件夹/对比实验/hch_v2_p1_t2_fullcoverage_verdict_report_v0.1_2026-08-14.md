# P1 Round-2 · T2 方向判定报告 — full-coverage 均衡采样

- 日期: 2026-08-14
- git_sha: 0478cf7
- 方向: T2「市场/宿主均衡采样 = full-coverage」(修 day exposure 不均)
- 判定: **REJECT**（3/3 seed S2V 一致退化;S4 下游 96 对宏平均三指标全差,不反转）
- 完整证据: `experiments/08-hch-v2/results/P1_T2_0478cf7/VERDICT.md`(gitignored,本地留存)

## 假设与实现

T0 = P0-C 等域采样(K=median(N_g)=22 update/域/epoch)。短域 NEM_SA1(S2T 仅 58 天)被 5.5× 重复,长域 DE/PJM(S2T 348 天)全覆盖。T2 假设"NEM 过采样 = 暴露不均 = 需要修复",改为 **full-coverage**:每域全部 S2T batch 每 epoch 恰好一次(无截断无重复),总 update/epoch 264→192(−27%),NEM 梯度份额 33%→8.3%。

## 主指标(S2V 宏平均 CRPS,3 seed)

| seed | T2 | T0 | Δ |
|---|---|---|---|
| 0 | 0.2317 | 0.2275 | +0.0042 |
| 1 | 0.2346 | 0.2288 | +0.0058 |
| 2 | 0.2303 | 0.2238 | +0.0066 |

宏平均 Δ=+0.0055(+2.4%),3/3 方向一致 → 非噪声。

## 逐域拆解(S2V):故事反转

- DE ×4、PJM ×4 全部小幅改善(去 NEM 过采样不伤长域)
- **NEM 3/4 域退化**(Linear/MLP/PatchTST),退化量系统性大于 DE/PJM 改善 → 主导宏平均
- 结论:**NEM 的 5.5× 重复不是过拟合噪声,而是低数据量域的真实梯度需求**。移除即退化。

## S4 下游(96 对 = 32 cells × 3 seeds,weighted_mean)

| 指标 | Δ(T2−T0) | T2 胜率 |
|---|---|---|
| MAE | +0.233 | 45/96 |
| RMSE | +0.227 | 36/96 |
| SMAPE | +0.527 | 40/96 |

逐域新信号(与 S2V 不同):
- LAGO_DE 全改善(11/12 胜),与 S2V 一致
- **转移市场 NORD_DK1 重创**(mean +2.08,MLP 37.5→44.0 = +17%)—— 降低 NEM 梯度份额 + 总预算 −27% 伤通用头迁移能力,与 R1B"正迁移"结论形成张力
- NEM 由 LSTM 单独挽尊(−3.4~−6.4),其余全退化

诚实披露: S4 逐 seed 非 3/3 一致(seed1 反而更好),S4 证据更吵但方向不反转 REJECT。

## 对下一方向(T4)的教训

1. **"域失衡"轴可能选错了。** T0 的等域采样在短域上的重复是必要正则,不是要修的失衡。
2. T4(连续模式覆盖采样 / 软采样权重)若要推进,**必须先写清目标函数**:若目标是"让 NEM 这类低数据量域获得更多梯度",方向与 T2 的"均衡"相反,需按 §5.2 用 host-only 描述子(预测时可见统计)设计权重,并跑 target-free 对照。
3. **预算副作用不可分离**:full-coverage 把总 update 264→192,T2 检验的是"完整覆盖采样"整体设计,无法单独分离"域均衡"与"总预算"。T4 若用软权重,应保持总预算=12×K 不变,只改分配。

## 记录状态

- 代码: `p1_t2.py` / `p1_t2_s4.py` / `universal_trainer.py(sampling="full_coverage")` / `r1b_generalization_screen.py(epochs 参数)`
- 结果: `results/P1_T2_0478cf7/`(gitignored)
- 决策: **T2 REJECT,T0 等域采样继续作为训练范式基线**;T3 不触发(T0 CONVERGED_STABLE)
