# R1B Final Verdict 报告

- **日期**: 2026-08-14
- **协议**: hch_v2_r1b_stage2_broad_frozen_generalization_and_action_chain_v0.1 §30 / §31
- **候选**: LearnedSig_main(frozen, seed=0, 12 source 域;重训逐位复现 0.2251746 ≡ 0.22517)
- **依赖**: R1B_SCREEN + Stage-2A/2B/2C/2D/2E/2F(报告与产物全部已提交)

---

## 0. 判定

```text
NATIVE_GENERALIZATION_SUPPORTED
```

**一句话**:多个未见市场 / 未见数据 regime / 未见 host 上,candidate 方向一致(ΔCRPS<0 无一例外);无法获益域零代价安全退化;source 提升不仅没有以 unseen 退化为代价,反而 **source 上 universal 反超单域 LocalCore(SOURCE_FIT_HELPS)、DK1 上反超 target-trained full-shot(TARGET_TRAINED_FULLSHOT_SURPASSED)**。HCH-v2 universal correction core 雏形成立。

## 1. 证据链汇总(7 阶段,共 100+ 独立评估)

| # | 阶段 | 面板 | ΔCRPS<0 | macro 证据 | 判定 |
|---|---|---|---|---|---|
| 0 | R1B_SCREEN | 16 cells(seed0,四格) | 全负 | 四 cell mean<0;DK1+LOHO 无坍缩;LearnedSig>PlainCore | healthy |
| A | Stage-2A | Linear/MLP broad 32 cells | 32/32 (frac<0=1.0) | SOURCE −0.165 / UNSEEN_MARKET n=14 −0.080 / SAME_MARKET −0.058 / SCHEMA −0.048 | CONTINUE |
| B | Stage-2B | LSTM/PatchTST 深 host 15 new | 15/15 | UNSEEN_HOST n=3 **−0.176**;UNSEEN_MARKET_HOST n=4 −0.053 | CONTINUE |
| C | Stage-2C | seeds 1/2 × 40 cells | 120/120 | 3/3 种子宏值同号,无种子坍缩 | STABLE |
| D | Stage-2D | action-chain canonical 16 | 16/16 | source A1 +0.0257→A2 +0.0276;DK1 +0.0554→+0.0539;SAFETY False | CONTINUE |
| E | Stage-2E | 扩展 6(FR/PJM2020/GEFCOM) | 6/6 (−0.037~−0.063) | ext A1 +0.0335→A2 +0.0155;GATING_HURTS×1(EPEX_FR:PatchTST) | CONTINUE |
| F | Stage-2F | Local-Core 12 + DK1 full-shot 4 | — | source cost **−0.02545**;DK1 gap **−0.01218** | 双绿 |

**候选转移维度:没有任何一个 cell 的 ΔCRPS>0(除 PlainCore 源域 1/12 的 +0.0014 近持平)。**

## 2. §31 成功标准逐条对照

### ① 多个未见市场 / 未见数据 regime / 未见 host,candidate 方向一致 ✅

| 未见维度 | 证据 | 宏 ΔCRPS |
|---|---|---|
| **未见市场**(4 个) | NORD_DK1(2D,16 cells)、EPEX_FR/PJM_2020/GEFCOM14P(2E,6 cells) | −0.080(2A UNSEEN_MARKET n=14)→ −0.012~−0.026(2E 逐域) |
| **未见 host**(LSTM/PatchTST) | LOHO(2B,7 cells) | **−0.176**(UNSEEN_HOST n=3) |
| **未见数据 regime** | GEFCOM 公开基准、EPEX_FR 高负电价率(0.0686)、NEM_SA1 高波动 | −0.048~−0.058 |
| **未见市场×host 组合** | 2B n=4 | −0.053 |
| **方向一致性** | 全部面板 frac<0=1.0;2C 3/3 种子同号 | 无一例外 |

### ② 无法获益的 domain 可以安全退化 ✅

- **LAGO_PJM 四域**(host 已强,CRPS 0.046~0.078):候选 ΔCRPS 仅 −0.012~−0.037(获益最小),但 **MAE 恒等(mae_rel=0.0)**,action-chain A2 net 全正,DVG 无危害性 release 扩散。
- **GEFCOM14P:Linear**:A1 net +0.000066(≈零获益),不退化、不亏。
- **§21 全阶段检查**:16+6 域 `mae_rel=0.0`,`SAFETY_FAILURE=False`。

### ③ source 提升没有以 unseen 退化为代价 ✅(本轮最强)

- **2F**:universal 在 source 上 **反超单域 LocalCore**(macro cost −0.02545,SOURCE_FIT_HELPS)——共享不仅没花 source 的代价,反而帮助了最难域(NEM_SA1,单域 LocalCore 明显欠拟合)。
- **2F**:universal **从没见过 DK1**,却反超「用 DK1 自己数据训练的 full-shot」4/4 host(gap −0.01218,TARGET_TRAINED_FULLSHOT_SURPASSED)。
- 共享收益最大处 = 最难的 domain(source 收益与 unseen 泛化是**同一个机制**,不是零和)。

## 3. Caveats(诚实记录,均不推翻判定)

| caveat | 阶段 | 归类 | 影响 |
|---|---|---|---|
| EPEX_FR:PatchTST C3 短证据误授权 → GATING_HURTS | 2E | §22 LOCAL_ADAPTATION_LIMITED | 动作链局部适配失败(丢 0.108),**非候选失败**;候选 ΔCRPS −0.043 完好 |
| LAGO_DE:MLP 单域 local 占优 +0.0176 | 2F | 容差内(±0.02) | 唯一有意义的 local 微赢单元,噪声级 |
| S4 gate 罕见+危害性 release,NEM LCB coverage <0.90 | R1A.9 | 既有 caveat | 动作链保守化建议已记录 |
| C3 短证据域误授权风险(≈50 S3M 日) | 2D/2E 复现 | 设计级 | 建议:短证据域提高 OOS 门槛或部署前校验 full-fit vs rolling map |

## 4. §24 GENERALIZATION_LEDGER v2 最终条目

```text
experiment_id:            R1B_FINAL (SCREEN+2A+2B+2C+2D+2E+2F, 2026-08-13/14)
candidate_variant:        LearnedSig_main (frozen, seed=0, 12 source) — 确定性复现逐位一致
training_market_set:      LAGO_DE, LAGO_PJM, NEM_SA1
training_host_set:        Linear, MLP (LSTM/PatchTST 走 LOHO,未参与训练)
evaluation_dataset:       DK1×4, EPEX_FR/PJM_2020/GEFCOM14P × Linear/PatchTST, + 源 12 + LocalCore/全 shot 对照
transfer_category:        UNSEEN_MARKET (4) / UNSEEN_HOST (LOHO) / UNSEEN_SCHEMA_REGIME / LOCAL_CORE_UPPER_BOUND
source_macro_delta:       -0.165172 (SOURCE_SEEN n=12)
holdout_delta:            -0.079826 (UNSEEN_MARKET n=14) / -0.175594 (UNSEEN_HOST n=3)
worst_domain_delta:       -0.011908 (LAGO_PJM 近持平,安全) / EPEX_FR:PatchTST A2 门控丢 0.108 (动作链)
seed_consistency:         3/3 (2C) + universal 重训逐位复现 (2F)
final_mae_effect:         0.0 (全阶段 mae_rel=0.0)
safety_effect:            SAFETY_FAILURE=False; GATING_HURTS=1 (EPEX_FR:PatchTST, LOCAL_ADAPTATION_LIMITED)
complexity_added:         无 (frozen universal; action-chain 已有)
accepted_for_universal_core: YES — NATIVE_GENERALIZATION_SUPPORTED
notes:                    source 上 universal 反超 LocalCore; DK1 上 universal 反超 full-shot; 共享收益集中在最难域
```

## 5. 结论与后续建议

**HCH-v2 universal correction core 雏形成立**(§31 原文达标)。接下来：

1. **R1B 至此结束**,无更多必需阶段(§29:2F 后即 FINAL VERDICT)。
2. **动作链改进方向**(若有后续):短证据域(≈50 S3M 日)C3 授权需更保守 —— 提高 OOS 门槛、或部署前校验 full-fit map 与 rolling map 的 `D_map`(2E 事故链)。
3. **运行环境**:本阶段起回本地桌面(用户决定)。流水线纯 CPU 可跑(8.7K 参数模型),服务器 4090 基本闲置;本地环境 paramiko 4.0.0 / torch 2.13.0+cpu / numpy 2.4.6 已验证齐备。
4. **产物归档**:全部 7 阶段产物 + 报告已提交。
