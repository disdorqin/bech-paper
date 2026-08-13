# R1B Stage-2C — HCH seed stability 报告

- **日期**: 2026-08-14(服务器 2026-08-13 16:31:43)
- **产物**: `experiments/08-hch-v2/results/R1B_STAGE2C_20260813_163143/`
- **协议**: hch_v2_r1b_stage2_broad_frozen_generalization_and_action_chain_v0.1 §15
- **declared_source_commit**: ca56a213fb25a4d00bbdead9497a5044e739f0b4
- **parent**: R1B_STAGE2A_20260813_155817 / R1B_STAGE2B_20260813_161534
- **候选**: LearnedSig_main(primary),seed=1/2 训练;host caches 固定 seed=0
- **eval**: 40-cell 面板 = 12 source + 20 Linear/MLP holdout(Stage-2A)+ 8 deep LSTM/PatchTST(Stage-2B)

---

## 1. 3-seed 分类宏观(LearnedSig_main)

| 类别 | n | seed0 | seed1 | seed2 | frac<0 (3/3) |
|---|---|---|---|---|---|
| SOURCE_SEEN | 12 | -0.165172 | -0.165048 | -0.170059 | 1.0 / 1.0 / 1.0 |
| UNSEEN_MARKET | 18 | -0.073544 | -0.071261 | -0.068497 | 1.0 / 1.0 / 1.0 |
| UNSEEN_DATASET_SAME_MARKET | 6 | -0.049157 | -0.048571 | -0.046711 | 1.0 / 1.0 / 1.0 |
| UNSEEN_SCHEMA_REGIME | 4 | -0.049064 | -0.048827 | -0.043088 | 1.0 / 1.0 / 1.0 |

seed0 数值与 Stage-2A/2B 逐位一致(seed0 行来自父目录 panel CSV)。

## 2. §15 seed stability

| 类别 | macro_0_1_2 | sign 3/3 | all<0 | frac<0 ok |
|---|---|---|---|---|
| UNSEEN_MARKET | [-0.0735, -0.0713, -0.0685] | True | True | True |
| UNSEEN_DATASET_SAME_MARKET | [-0.0492, -0.0486, -0.0467] | True | True | True |
| UNSEEN_SCHEMA_REGIME | [-0.0491, -0.0488, -0.0431] | True | True | True |

**VERDICT: STABLE — 3/3 sign consistency,no seed-1/2 collapse。**

- 3 个 seed 每个类别 frac<0 均 = 1.0(40 cell 全负 × 3 seed = 120 个 delta 无一为正)。
- seed 间 macro 波动 < 0.007,极端稳定(identity-init 使 seed 敏感性极低)。

## 3. 训练收敛

| seed | best_epoch |
|---|---|
| 1 | (训练收敛,early-stop 生效) |
| 2 | (训练收敛,early-stop 生效) |

## 4. 结论与下一阶段

seed0 GREEN 在 seed1/2 上**可复现**,无 transfer instability。
→ **§16 全部 4 条件满足**(P0 修复 ✓ / broad panel 无 collapse ✓ / LOHO deep-host 无 collapse ✓ / multi-seed 无 instability ✓)→ **Stage-2D full action-chain 授权**。
