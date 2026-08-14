# P1 第三阶段 · T5 假设文档 — 国内数据混合比例网格(先写假设再跑)

- 日期: 2026-08-14
- 方向: T5「第三轮增量数据 —— 已准入国内域加入通用头训练,混合比例 r 网格」
- 前置: D1 准入 10/10 PASS(无未来电价泄漏、无回填实际值);D2 国内基准已建成(37/40 cell 修正帮助)
- 状态: **假设待验(未跑)**
- 触发: P1 doc §7.3 —— 新增数据源需走第三轮 transfer 验证;§5.4 —— r 选择用 S2V,S4 只做冻结确认。

## 1. 问题声明

> **H5**: 把已准入的国内域(山东/甘肃/陕西的 host 训练数据)以受控梯度份额 r 加入通用修正头的训练池,应当:
> (a) **不伤害** 国外 32-cell headline S4(transfer matrix: 国外域无负迁移或净中性),
> (b) **改善** 国内 holdout S4(入训后国内域预测更好),
> 从而在 r ∈ {0.15, 0.30} 上呈现"国内数据是资产不是污染"。r=0 即 source-only 基线(T0)。

**为什么可能成立**: 通用修正头是 schema-agnostic(D_VALUE=0),只依赖 host 输出的 scale-free 特征。国内域提供了**不同的电价分布形态**(0% 负价省的对称尖峰 + 山东含负价),这可能增强头对"宿主分布极端/长尾"的鲁棒性 —— 与 R1B"通用头对未见市场正迁移"一致。
**为什么可能失败**: 国内 host 特征方言不同(列集合不同),修正头虽不读特征,但 host rank/scale-free 统计的分布差异可能使 CAGM/DVG 参考池污染;且国内域 S2T 规模远小于长国外域,可能稀释梯度。

## 2. 实验设计

- **训练池**: 12 国外源域(固定,主配置不动)+ 国内训练域。
  - **国内训练域候选**: shandong_DA、gansu_DA、shaanxi_DA,各 × 4 host = 12 个 (模式×host) 域。
  - **排除**: ningxia/qinghai(S2T 仅 27~29d,且 qinghai 缺 17.4%;参考太薄,进训练会引入噪声)。
  - **shandong_RT 待定**: 它含 13.4% 负价 —— 若想给通用头注入负价体验这是唯一国内来源;但同省双市场可能重叠。默认**只进 DA**,RT 是否进由第 5 节判定在 sanity 后决定(不提前承诺)。
- **混合机制**(基于已核实的 `UniversalCoreTrainer` weights 语义): 每 epoch 总 update = n_domains×K 不变,K = median(N_g) 在含国内域的训练池上重算;rng.choice(p=w/Σw),国外域权重 1,每个国内域权重 w_d = r/(N_dom·(1−r)/N_for)… 具体数值由 p1_t5.py 实现时按实际 N_for/N_dom 计算,约束 = **国内域总梯度份额 = r**。
- **网格**: r ∈ {0(基线,即 T0), 0.15, 0.30},每 r 3 seeds(同 seed[0,1,2],同 trainer 协议)。
- **评估**:
  - **国外 transfer matrix**: 32 国外 headline cell S4(head 冻结,weighted_mean),r>0 vs r=0 配对 —— 国内数据入训是帮还是伤国外?
  - **国内 holdout S4**: 10 国内模式 × 4 host = 40 cell,入训后国内是否更好(参照 D2 的 r=0 即冻结头结果)。
- **主指标**: 国外 32-cell S4 宏平均 ΔMAE(配对,r>0 − r=0);次指标国内 40-cell。

## 3. 判定阈值

- **KEEP**: 国外 32-cell 宏平均无负迁移(ΔMAE ≤ 0 或 ≤ +0.5% 噪声带),且国内 holdout 至少 1 个 r 值显著改善(≥ +2% 且 ≥2/3 seed 一致)。
- **INCONCLUSIVE**: 国外微负但国内明显改善,或反之;r 之间不一致。
- **REJECT**: 国外 32-cell 明确负迁移(ΔMAE > +0.5% 且 2/3 seed 一致),或国内无改善。
- 若 mixed,如实保留,不冒充泛化(§7.3)。

## 4. 红线(全程)

- **r 选择只用 S2V**;S4 仅冻结确认(禁 S4 调参/筛数据)。
- 不改 IAH-CRPS 数学核、不改 query-dose replay、不改 double-event、不改 full-day action-value、不改 LCB 门、不改 6 固定比较器;不新建头/损失(不复活 T6)。
- 国外主配置(12 源域、HOSTS4、weighted_mean)不动;国内域只作为**新增训练数据**,不进主配置的 cells 列表。
- 国内特征仍只经 host 进入修正头;0% 负价省不报 negative-price 指标。
- 诚实: 若国内数据入训污染国外(transfer matrix 变差),如实记 REJECT。

## 5. 待 T4 结果的衔接

- T4(软权重)若 KEEP,则 r>0 的训练可叠加 T4 的难度权重(国内域按 host_s1r_mae 或等权);T4 若 REJECT/INCONCLUSIVE,则 T5 国内域用等权(与 T0 相同范式)。**此衔接在 T4 VERDICT 后更新本节的最终设置,不提前锁死。**
- 若 T5 KEEP,国内 holdout 改善 → 论文"第三轮增量数据"段落;若 REJECT,如实写"国内数据入训未通过 transfer 门"。

## 6. 产物

- `experiments/08-hch-v2/p1_t5.py`(镜像 p1_t4.py:sanity → r 网格 3 seeds)
- `results/domestic/p1_t5_<git>/`: 每 r 训练报告 + 国外 transfer_matrix.csv(32×r 配对)+ 国内 holdout.csv(40×r)+ VERDICT.md
- `docs/训练文件夹/对比实验/hch_v2_p1_t5_mixing_grid_verdict_report_v0.1_2026-08-14.md`
