# R1B Stage-2A — Broad Linear/MLP frozen holdout panel 报告

- **日期**: 2026-08-14(服务器 2026-08-13 16:12:54)
- **产物**: `experiments/08-hch-v2/results/R1B_STAGE2A_20260813_155817/`
- **协议**: hch_v2_r1b_stage2_broad_frozen_generalization_and_action_chain_v0.1 §11-§12
- **declared_source_commit**: b95b11e33ebfa60a2fc4d5916fdf3f124b0d857d
- **sha256(runner)**: 0b65a8643b59…（config.json 内 5 项 sha256 齐全，含 data_signature）
- **候选**: LearnedSig_main / PlainCore_main(12 source,4 host,seed=0,冻结)
- **holdout**: H1(EPEX_FR/BE/NL, NORD_FI/NO/SE3/DK1) + H2(DE_EPEX, PJM_2020) + H3(GEFCOM14P)，全部 zero-gradient

---

## 1. 训练收敛

| 候选 | best_epoch | 说明 |
|---|---|---|
| LearnedSig_main | 9 / 12 | early-stop 生效 |
| PlainCore_main | 4 / 12 | early-stop 生效 |

## 2. 分类面板(LearnedSig_main)

| 类别 | n | macro_delta | frac<0 | p10 / p50 / p90 | worst |
|---|---|---|---|---|---|
| SOURCE_SEEN | 12 | **-0.165172** | 1.0 | -0.387744 / -0.096525 / -0.016024 | -0.011908 |
| UNSEEN_MARKET (H1) | 14 | **-0.079826** | 1.0 | -0.150437 / -0.060859 / -0.031020 | -0.027884 |
| UNSEEN_DATASET_SAME_MARKET (H2) | 4 | **-0.058396** | 1.0 | -0.077839 / -0.052020 / -0.044052 | -0.043070 |
| UNSEEN_SCHEMA_REGIME (H3) | 2 | **-0.047813** | 1.0 | -0.056257 / -0.047813 / -0.039369 | -0.037258 |

**所有 32 cell 全部 delta<0（frac<0=1.0），零个正 cell。** 无任何 cell 出现 candidate 劣于 host 的 CRPS 退化。

## 3. 泛化缺口(LearnedSig_main)

```
holdout macro(20 cells) = -0.072338
source macro(12 cells)  = -0.165172
G_vs12   = +0.092834    (holdout 较 source 衰减)
G_vsLL   = +0.119837
R_retain = 0.438        (source 收益的 ~44% 保留到 holdout)
```

Zero-gradient 转写下保留 ~44% 的 source 收益,holdout 全负,属健康转写。

## 4. DK1 可复现性交叉核对(与 Stage-1 逐位一致)

| 候选 × host | Stage-1 基线 | Stage-2A | 一致? |
|---|---|---|---|
| LearnedSig × Linear | -0.039749 | -0.039749 | ✅ 逐位一致 |
| LearnedSig × MLP | -0.109123 | -0.109123 | ✅ 逐位一致 |
| PlainCore × Linear | -0.037266 | -0.037266 | ✅ 逐位一致 |
| PlainCore × MLP | -0.112476 | -0.112476 | ✅ 逐位一致 |

面板 eval 管线与先前 screening 完全可复现,无 pipeline 漂移。

## 5. LearnedSig vs PlainCore

| 量 | LearnedSig_main | PlainCore_main |
|---|---|---|
| source macro | **-0.165172** | -0.153119 |
| holdout macro | -0.072338 | -0.072430 |
| source 正 cell | **0 / 12** | 1 / 12 (LAGO_PJM:MLP +0.0014) |
| holdout 正 cell | 0 / 20 | 0 / 20 |
| R_retain | 0.438 | 0.473 |

- LearnedSig 在 source 上优于 PlainCore(+0.012);holdout 上统计平局(±0.0001)。
- 无 SIGNATURE_NEGATIVE_TRANSFER(签名带来 source 收益且 holdout 不劣化)。

## 6. MAE 安全

所有 32 cell:`cand_mae == host_raw_mae`(逐位),`mae_rel_deg = 0.0`,`safety = ok`。
HCH-v2 设计使点预测(location)冻结为 host,签名只调制混合分布 spread → CRPS 改进不牺牲点预测 MAE。

## 7. §12 STOP rules

| 规则 | 判定 |
|---|---|
| MARKET_PANEL_COLLAPSE | False |
| DATASET_SHIFT_COLLAPSE | False |
| HISTORICAL_SCHEMA_COLLAPSE | False |
| SIGNATURE_NEGATIVE_TRANSFER | False |
| **VERDICT** | **CONTINUE — 无 §12 collapse** |

## 8. Provenance(P0-3)

- git_sha: unknown(服务器无 .git,预期 fallback)
- declared_source_commit: b95b11e33ebfa60a2fc4d5916fdf3f124b0d857d
- sha256_runner / iah_candidate / universal_trainer / iah_crps_loss / data_signature 全部在 config.json 中
- 修复项:Stage-2A 首次运行因 `torch.where` 广播怪癖崩溃(cond 需 `[B,1,1]` 而非 `[B,1]`),修复后重跑,本报告为修复后完整结果。

## 9. 结论与下一阶段

Stage-2A **healthy**:全部 holdout 类别 frac<0=1.0、DK1 逐位可复现、LearnedSig≥PlainCore 且无负转写、MAE 全安全。
→ 按 §13 授权推进 **Stage-2B(deep-host LSTM/PatchTST stress)**。
