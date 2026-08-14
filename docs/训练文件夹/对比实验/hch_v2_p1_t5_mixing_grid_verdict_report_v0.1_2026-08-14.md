# P1 第三阶段 — T5 方向判定(国内数据混合比例网格 · 增量数据)

convergence_status: CONVERGED_STABLE(6 个训练 = 2 r × 3 seed 均 0 NaN / 0 scale-invalid, 264 updates/epoch 预算与 T0 完全一致, 无健康异常)
T5_verdict: **INCONCLUSIVE**(r 之间不一致 —— r=0.15 满足 KEEP 的 doc §3 条件, r=0.30 触发 REJECT 条件; 主配置 T0 不动)
r_grid: {0(T0 equal 基线), 0.15, 0.30} × 3 seeds
training_pool: 12 国外源域(不变)+ 12 国内训练域(shandong_DA / gansu_DA / shaanxi_DA × 4 host)
domestic_gradient_share: r=0.15 → w_dom=0.1765/域, r=0.30 → w_dom=0.4286/域(国外=1.0/域)
updates_per_epoch: 24×K=24×11=264 = T0(12×22)预算精确保留
point_readout_selected: 沿用第一轮 —— weighted_mean(未变)
transfer_status: FOREIGN = 低 r 中性 / 高 r 负迁移(LAGO_NP 温和市场主导)
negative_transfer_status: r=0.30 明确负迁移(国外 2/3 seeds ΔMAE > +0.5%)
SOTA_status: POINT_READOUT_ADVANTAGE(主配置 T0 训练范式不变, 第一轮点读出优势结论不受影响)
next_recommendation: 国内数据入训需按"低比例 + 逐域温和市场风险评估"引入; 论文段落写为"条件性增量数据", 不写无条件泛化

---

## 结论(人可读)

T5 检验了"把已准入的国内域以受控梯度份额 r 加入通用修正头训练池, 应当不伤国外(transfer matrix)且改善国内(holdout)"的假设。配对结果给出**混合证据**:

- **r=0.15**: 国外 32-cell 宏平均 ΔMAE **+0.13%**(噪声带内, 无负迁移); 国内 40-cell **2/3 seeds 改善 −2.4~−3.0%**。单独看满足 doc §3 的 KEEP 条件。
- **r=0.30**: 国外宏平均 ΔMAE **+3.0%**, 且 **2/3 seeds 一致超 +0.5% 带**(seed0 +7.9%, seed2 +2.2%)—— 明确负迁移; 国内改善不随 r 增加(仍 2/3 seeds ≈ −2~−2.8%)。
- 既有 keep 达标的 r=0.15, 又有触发 REJECT 的 r=0.30 → doc §3 的"r 之间不一致" → **INCONCLUSIVE**。

**机制画像**: 国内数据是"有条件资产"。低比例(≤0.15)混入不伤国外、改善国内(且含未见模式, 非纯记忆); 比例升高后, 国内高波动分布污染国外**温和市场**(LAGO_NP 挪威, MAE~3 的最低波动市场, r=0.3 时 seed0/2 相对 +18~25%), 主导负迁移。这与 H5"为什么可能失败"分支(参考池污染)一致。

### 1. 国外 32-cell transfer matrix(国内数据入训是帮还是伤国外)

主指标 = 32 国外 headline cell S4 MAE 配对 Δ(r vs T0 同 seed), 宏平均(跨 seed):

| r | seed0 ΔMAE | seed1 ΔMAE | seed2 ΔMAE | 宏平均(跨 seed) |
|---|---|---|---|---|
| 0.15 | **+1.17%** | −0.84% | +0.07% | **+0.13%**(≤+0.5% 带) |
| 0.30 | **+7.85%** | −1.05% | **+2.24%** | **+3.01%**(>+0.5% 带) |

- 国外伤害集中在 **LAGO_NP**(挪威, 32 cell 中最低价/最低波动, MAE~3): r=0.30 时 seed0 Linear +25.4%/LSTM +18.6%, seed2 Linear +24.6%/LSTM +24.5%/PatchTST +21.8%。国内域(MAE 50~180, 高波动)的 scale-free 分布与 NP 差异最大, 入训比例升高后头被国内分布牵引 → NP 相对修正过度。
- 同度量 S2V(国外 12 源域 @best, 与 T0 同 metric)方向一致: r=0.15 ≈ T0(seed 平均差 +0.0004), r=0.30 略差(+0.0011)。**S2V 与 S4 不矛盾**。

### 2. 国内 holdout 40-cell(入训后国内是否更好)

次指标 = 40 国内 cell(10 模式 × 4 host)配对 ΔMAE:

| r | seed0 | seed1 | seed2 | ≥2/3 seeds ≤ −2%? |
|---|---|---|---|---|
| 0.15 | −0.23% | **−2.98%** | **−2.39%** | ✅(seed1/2) |
| 0.30 | +0.45% | **−2.04%** | **−2.83%** | ✅(seed1/2) |

- **国内改善含未见模式**: 入训域(shandong/gansu/shaanxi_DA ×4, 12 cell)与**未见模式**(shandong_RT/ningxia/gansu_RT/shaanxi_RT/qinghai ×4, 28 cell)在 r=0.15 时改善幅度相当(trained −0.3/−3.3/−3.0%, unseen −0.2/−2.9/−2.1%)→ 改善是**泛化而非记忆**。
- **seed0 从不改善**(国内 seed0 持平/微正, 国外 seed0 也最伤)—— seed0 对国内数据最敏感, 是 per-seed 不一致的来源。

### 3. 收敛性

6 个训练 0 NaN / 0 scale-invalid, 264 updates/epoch 全程恒定(预算 = T0 精确保留), train loss 单调收敛, best_epoch 正常(seed2 r=0.15 提前停在 ep8)。标签 CONVERGED_STABLE —— 是"混合比例 r 本身的效果差异", 不是训练异常。

### 4. 诚实披露

1. **判据操作化修正**(与 T4 同类问题): 初版脚本把 KEEP 的国外条件操作化为"每个 r 的每个 seed 都 ≤ +0.5%"(per-seed 全过), 得到 REJECT。复查 doc §3 字面, KEEP 的国外条件是"**32-cell 宏平均** ≤ +0.5%"、REJECT 的国外条件是"**2/3 seed 一致** > +0.5%"。按 doc 字面重算: r=0.15 宏平均 +0.13% 达标(keep_r), r=0.30 触发 2/3 seeds 超带(reject_r), 并存 → **INCONCLUSIVE**。脚本已修正, summary.json 已重算。**主结论从 REJECT 改为 INCONCLUSIVE, 数字不变。**
2. **per-seed 诚实呈现**: 即使 r=0.15 的宏平均无负迁移, seed0 国外 +1.17% 仍超带; 报告不把"宏平均中性"写成"逐域全无害"。
3. **训练 S2V 不同度量**: T5 训练宏平均在 24 域(含国内)上算, 与 T0(仅 12 国外)不可直接比; 报告用**国外-only 同度量**对比(S2V: r=0.15≈T0, r=0.30 略差)代替。
4. **负价指标**: 仅山东含负价(8 cell), 其余省份 0% 负价不报 negative-price; 报告所有 mae 均为全体小时口径。
5. **交叉验证**: T5 r=0 seed0 的 40 国内 cell 与 D2 基准**逐 cell 完全一致(40/40)** → eval 管道正确, 无实现漂移。
6. **"国内数据是资产不是污染"的 H5 表述被部分否定**: 低 r 支持"资产"侧(国内改善 + 国外中性), 高 r 支持"污染"侧(国外负迁移)。论文段落必须写为**条件性**, 不能写无条件正。

## 失败项 / 未解决

1. **r 不一致无法收敛到单一最优**: r=0.15 近 KEEP 但 seed0 国外超带; r=0.30 明确负。无单一 r 满足 KEEP 的逐 seed 稳健性。
2. **温和市场(型 LAGO_NP)对国内入训的脆弱性未建模**: 参考池污染的机制清楚, 但没有预防手段(不改 CAGM/DVG 核心, 不引入市场类别门)。
3. **seed0 的国内敏感性未解释**: 同一训练协议下 seed0 国外最伤且国内不改善, 其余 seed 相反 —— 可能是初始化路径对混合分布的敏感性, 未深挖。

## 下一步建议

1. **主配置 T0 不动**(12 国外源域等权)。国内数据不进入主配置训练池。
2. **若要在论文写"国内业务验证"段落**: 用 D2(冻结头转移, DOMESTIC_TRANSFER_POSITIVE, 37/40 帮助)+ T5 低 r 证据 —— "低比例国内数据入训对国外中性、对国内(含未见模式)改善; 比例升高后污染国外温和市场"。
3. **可选后续(不承诺)**: 国内专用修正头(不混入通用头)单独训练/评估; 或按市场类别(温和/剧烈)分层评估 transfer 风险。
4. **T6 维持暂缓**(host 消费外生, HCH 核心不复活)。

git_sha: 1dd1aa0
结果目录: experiments/08-hch-v2/results/P1_T5_1dd1aa0/(manifest.json / r{0.15,0.3}/ 每 arm 训练报告 + heads / eval_rows.json 648 行 / transfer_matrix.csv / summary.json)
