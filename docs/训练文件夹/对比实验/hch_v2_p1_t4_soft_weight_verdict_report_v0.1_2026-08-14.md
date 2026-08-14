# P1 Round-3 — T4 方向判定(难度软权重采样 · 重定标)

convergence_status: CONVERGED_STABLE(6 个训练 = 2 arm × 3 seed 均 0 NaN / 0 scale-invalid,train loss 单调收敛,无健康异常)
T4_verdict: **REJECT**(两 arm 宏平均 3/3 seed 一致更差;等权分配是梯度分配轴上的局部最优)
arm_verdicts: host_s1r_mae = REJECT / inv_nbatch = REJECT
best_epoch: seed0=10~11 / seed1=10~11 / seed2=11(各 arm 内由 macro S2V 选)
candidate_crps: host_s1r_mae 0.2371 / 0.2349 / 0.2302;inv_nbatch 0.2383 / 0.2355 / 0.2316(vs T0 0.2275 / 0.2288 / 0.2238)
point_readout_selected: 沿用第一轮 —— weighted_mean(未变)
transfer_status: NOT_TESTED(第三轮)
negative_transfer_status: NOT_TESTED
SOTA_status: POINT_READOUT_ADVANTAGE(T4 被拒后 T0 训练范式不变,第一轮点读出优势结论不受影响)
next_recommendation: 梯度分配轴已双向证伪 —— T2(下移 NEM)与 T4(上移 NEM)都让宏平均变差,T0 等权是局部最优。下一方向 T5(国内混合比例,增量数据轴)按 T4 REJECT 衔接:国内域用等权,不改梯度分配。

---

## 结论(人可读)

T4 检验了"低数据量/高难度域(NEM)在等权采样下梯度不足,按宿主侧难度统计上调软权重应改善宏平均"的假设,实现为 **Case C 温度采样**(`rng.choice(p=w/Σw)`,总预算 n_domains×K 不变,只改分配)。配对结果否决了"重分配梯度能打败等权"。

### 1. 主指标:S2V 宏平均 CRPS,两 arm 3/3 seed 一致更差

| seed | host_s1r_mae | T0 equal | Δ | inv_nbatch | Δ |
|---|---|---|---|---|---|
| 0 | 0.23707 | 0.22750 | **+0.0096** | 0.23827 | **+0.0108** |
| 1 | 0.23493 | 0.22880 | **+0.0061** | 0.23554 | **+0.0067** |
| 2 | 0.23023 | 0.22376 | **+0.0065** | 0.23165 | **+0.0079** |

宏平均 Δ:host_s1r_mae **+0.0074**,inv_nbatch **+0.0085**。6/6 (arm,seed) 组合无一个持平或反超 → 不是噪声。

### 2. 逐域拆解:T4 的"故事反转"其实是 T2 的镜像,机制完全闭环

以 T0 同 seed 逐域 @best 为参照,两 arm 的逐域 Δ(负 = 该 arm 更好):

| arm | LAGO_DE ×4(sum Δ) | LAGO_PJM ×4 | NEM_SA1 ×4(改善数) |
|---|---|---|---|
| host_s1r_mae seed0 | **+0.0623**(0/4) | **+0.0466**(0/4) | +0.0059(**3/4** 改善) |
| inv_nbatch seed0 | +0.0591(0/4) | +0.0399(0/4) | +0.0303(2/4) |

- **host_s1r_mae 的权重"如预期"起作用了**:NEM 被上调到 1.7~2.9× 后,Linear/PatchTST/MLP 3/4 域改善 —— **NEM 确实响应更多梯度,这与 T2 的教训(去掉 NEM 重复→NEM 退化)互相印证**。
- **但代价是 DE/PJM 8/8 全部退化**:它们被下调到 ~0.4×,全表净差 +0.06~+0.11,损失系统性大于 NEM 的改善 → 主导宏平均。
- **inv_nbatch(数据量反比)更钝**:NEM 只在 2/4,0/4,2/4 seed 改善(不稳定),DE/PJM 同样 8/8 退化。
- **T0 等权(NEM 33%)是梯度分配轴的局部最优**:T2 把 NEM 从 33% 压到 8.3%(full_coverage)→ 宏平均 +0.0055 更差;T4 把 NEM 提到 ~50%+ → 宏平均 +0.007~0.011 更差。**两个方向都变差,等权就是最优点**。这不是"没找到正确的权重",而是"分配轴本身不该动"。

### 3. 收敛性:干净,不是训练没收敛

6 个训练 0 NaN / 0 scale-invalid,shift alive 正常,mass_entropy 无坍缩,train loss 单调收敛,best_epoch 与 T2/T0 同档(10~11)。标签 CONVERGED_STABLE —— **是"重分配梯度这个目标在 macro-S2V 下更差",不是训练过程异常**。

### 4. 诚实披露

1. **判据操作化**:文档 §5 的 REJECT 标准"任一 arm 与 T0 无差异或更差"。两 arm 6/6 (arm,seed) 全部更差 → REJECT。NEM 的改善(host_s1r_mae 3/4)是机制证据(T2 教训的正向确认),不构成"只有某 arm 成立"的 INCONCLUSIVE —— INCONCLUSIVE 的文档条件是"ΔCRPS < 0 但 seed 不一致",这里没有任何 arm 出现过 Δ<0。**(首次脚本判定给 INCONCLUSIVE 是操作化偏差,已修正为与文档一致,verdicts.json 已更新。)**
2. **T4 与 T2 同属"改梯度分配"轴**:T2 是数据量比例, T4 是难度/数据量权重。二者都被数据否决 → 该轴结论可合并表述为"等权采样是该轴的鲁棒最优"。
3. **host_s1r_mae 读 S1R 目标价格(训练-only,§5.2)**:它被 REJECT,不影响训练-only 标记的合规性 —— 但它的失败恰好说明"难度统计"不是有用的重分配信号,这本身削弱了"训练-only 难度信息"的价值主张。
4. S4 下游未重跑:两 arm 训练范式在 S2V 上已 3/3 更差,S4 冻结验证只会追加证据(与 T2 同规:若 S2V 方向明确,可先不跑 S4;此处 S2V 已一致否决)。

## 失败项 / 未解决

1. **梯度分配轴双向证伪**:上移(T4)和下移(T2)NEM 梯度份额都让宏平均变差 —— 无剩余正交方向可试(volatility 预检已显示噪声大,不再跑)。
2. **T4-C(volatility)未跑**:预检已显示它跨 host 方向不一致(NEM PatchTST 仅 0.58×),且主 arm 已 REJECT,补跑只有重复意义。

## 下一步建议

1. **T4 记为 REJECT**,T0 等权采样继续作为训练范式基线(不变)。
2. **T5 衔接(按 T5 假设文档 §5)**:T4 REJECT → T5 国内域加入训练池时**用等权**(与 T0 相同范式),不叠加 T4 的难度权重;只检验"增量数据(国内域)是资产还是污染"这一独立轴。
3. T5 前置已满足(D1 准入 10/10 PASS),可启动。

git_sha: be986f2
结果目录: experiments/08-hch-v2/results/P1_T4_be986f2/(verdicts.json / s2v_paired_comparison.csv / per_domain_comparison.csv / 每 arm 训练报告 + 曲线)
