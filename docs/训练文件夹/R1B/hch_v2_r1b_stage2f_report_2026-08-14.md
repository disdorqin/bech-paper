# R1B Stage-2F — Local-Core comparison 报告

- **日期**: 2026-08-14(服务器 2026-08-13 17:56→18:04:48)
- **产物**: `experiments/08-hch-v2/results/R1B_STAGE2F_20260813_175608/`
- **协议**: hch_v2_r1b_stage2_broad_frozen_generalization_and_action_chain_v0.1 §23
- **declared_source_commit**: `3cb6ce986216fdaf3cbfb496df4d64d8e35f8795`
- **parent**: R1B_STAGE2E_20260813_173357(§19 扩展 + 用户确认 CONTINUE)
- **runner**: `r1b_stage2f_localcore.py`(sha256_runner `0b65a864…`)

---

## 1. 方法(§23)

- **Universal 重训**:LearnedSig_main,seed=0,12 source 域,deterministic。**重训复现校验:best_macro_s2v = 0.2251746,与 Stage-2A 冻结版 0.22517 逐位一致** → 冻结候选可复现,2F 与 2A→2E 的候选是同一个。
- **LocalCore**:12 个单域候选(每域只用自己的 source 域数据训练,seed=0,12 epochs,patience=4)。= **candidate-level upper bound**,量化 universal sharing 的 source-fit cost。
- **DK1 full-shot**:DK1×4 host 单候选(4 域全量训练),只能标 `TARGET_TRAINED_FULLSHOT_UPPER_BOUND`(§23),不能当 transfer baseline。
- **cost[d] = universal_crps − local_crps**(负 = universal 更好)。

## 2. Source-fit cost 矩阵(12 源域)

| 域 | local_crps | universal_crps | cost = univ−local | 判读 |
|---|---|---|---|---|
| LAGO_DE:Linear | 0.183083 | **0.161651** | **−0.021432** | universal 好 |
| LAGO_DE:MLP | **0.124773** | 0.142412 | +0.017639 | local 好(唯一>0.01) |
| LAGO_DE:LSTM | 0.148779 | **0.141175** | −0.007604 | universal 好 |
| LAGO_DE:PatchTST | 0.168573 | **0.152646** | −0.015927 | universal 好 |
| LAGO_PJM:Linear | **0.077010** | 0.078481 | +0.001471 | local 略好 |
| LAGO_PJM:MLP | 0.047159 | **0.046859** | −0.000300 | 打平 |
| LAGO_PJM:LSTM | **0.046204** | 0.049102 | +0.002898 | local 略好 |
| LAGO_PJM:PatchTST | **0.054904** | 0.056008 | +0.001104 | local 略好 |
| NEM_SA1:Linear | 0.379282 | **0.279326** | **−0.099956** | universal 大胜 |
| NEM_SA1:MLP | 0.572166 | **0.455329** | **−0.116837** | universal 大胜 |
| NEM_SA1:LSTM | 0.692942 | **0.679366** | −0.013576 | universal 好 |
| NEM_SA1:PatchTST | 0.512660 | **0.459742** | −0.052918 | universal 好 |
| **macro** | | | **−0.025453** | **SOURCE_FIT_HELPS** |

**模式**:8/12 域 universal 更好,4/12 域 local 更好(LAGO_PJM×3 + LAGO_DE:MLP)。但:
- local 占优的全部是**微幅**(最大 +0.0176 的 LAGO_DE:MLP 仍在 ±0.02 容差内),合计仅 +0.0052;
- universal 占优的集中在**难市场 NEM_SA1**(合计 −0.283),单域训练在难市场明显欠拟合(如 NEM_SA1:MLP local 0.572 vs universal 0.455);
- **macro source-fit cost = −0.02545 → 共享在 source 上不花代价,反而 HELPS**。共享的收益最大处正是最难的 domain——这是"universal correction core"的正面证据,不是 cost。

## 3. DK1 full-shot(§23 upper bound 对标)

| 域 | universal | fullshot | gap (univ−full) | 判读 |
|---|---|---|---|---|
| NORD_DK1:Linear | **0.178824** | 0.185144 | −0.006320 | universal 好 |
| NORD_DK1:MLP | **0.218517** | 0.243491 | **−0.024974** | universal 大胜 |
| NORD_DK1:LSTM | **0.143082** | 0.150697 | −0.007615 | universal 好 |
| NORD_DK1:PatchTST | **0.137864** | 0.147692 | −0.009828 | universal 好 |
| **macro** | **0.169572** | 0.181756 | **−0.012184** | **TARGET_TRAINED_FULLSHOT_SURPASSED** |

**关键事实**:universal 候选**从没见过 DK1**(只训 3 个 source 市场),却在 4/4 host 上 CRPS 全部低于「用 DK1 自己的数据训练的 full-shot」。即:放弃 transfer、直接在 target 上训练得到的**本地最优**,被从未见 target 的 universal 候选反超 0.012 macro。

## 4. 判定

```text
macro source-fit cost = -0.025453  ->  SOURCE_FIT_HELPS
DK1 univ - fullshot gap  = -0.012184  ->  TARGET_TRAINED_FULLSHOT_SURPASSED
```

**§23 双绿。** 没有发现 universal sharing 的 source-fit cost;反而共享帮助了最难域,且 transfer 超越 target-trained upper bound。

## 5. 逐域观察与诚实 caveat

1. **LAGO_DE:MLP(+0.0176)是唯一有意义的 local 占优单元**,且在 ±0.02 容差内。DE:MLP 是 host 已强、域间相似度最高的单元;单域模型在这里微赢。属于"噪声级 source-fit cost",不触发任何 STOP。
2. **LAGO_PJM×3 的 local 微赢(+0.001~0.003)可忽略**,与 2D 里 LAGO_PJM 获益最小(host 已强)的观察一致。
3. **NEM_SA1 单域 LocalCore 显著弱**(MLP 0.572 vs 0.455):单域训练数据不足以拟合难市场,共享的泛化正则在此收益最大。这也解释了为什么 universal 在 source 上整体 HELPS。
4. **DK1 full-shot 本身已比 host 好**(fullshot_delta 全负,−0.033~−0.084),不是"full-shot 很弱";是 universal 更强。比较对 universal 有利面:**full-shot 是 4 域全量数据**,是慷慨的 upper bound;universal 在此仍反超 → 结论更稳。

## 6. §24 GENERALIZATION_LEDGER v2 条目

```text
experiment_id:            R1B_STAGE2F_20260813_175608
candidate_variant:        LearnedSig_main (frozen, seed=0) — 重训复现 0.2251746 ≡ 2A 0.22517
training_market_set:      LAGO_DE, LAGO_PJM, NEM_SA1 (12 domains)
evaluation_dataset:       source 12 domains (LocalCore) + NORD_DK1×4 (full-shot)
transfer_category:        LOCAL_CORE_UPPER_BOUND (source) / TARGET_TRAINED_FULLSHOT_UPPER_BOUND (DK1)
source_macro_delta:       universal − local = −0.025453 (SOURCE_FIT_HELPS)
holdout_delta:            DK1 universal − fullshot = −0.012184 (TARGET_TRAINED_FULLSHOT_SURPASSED)
worst_domain_delta:       LAGO_DE:MLP +0.017639 (≤ 0.02 容差)
seed_consistency:         3/3 (Stage-2C); universal 重训逐位复现
final_mae_effect:         0.0 (全阶段 mae_rel 恒等)
safety_effect:            SAFETY_FAILURE=False (2D/2E); 无 MAE 退化
complexity_added:         无
accepted_for_universal_core: YES (source 无 cost + DK1 反超 upper bound)
notes:                    universal 共享收益最大处 = 最难域 (NEM_SA1); 唯一 local 占优单元 LAGO_DE:MLP 在容差内
```

## 7. 下一步

Stage-2F 完成 → 全部 7 阶段证据链齐 → **§30 R1B FINAL VERDICT**(7 标签选 1)。
