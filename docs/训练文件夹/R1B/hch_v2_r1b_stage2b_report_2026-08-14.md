# R1B Stage-2B — Deep-host LSTM/PatchTST stress + LOHO unseen-host 报告

- **日期**: 2026-08-14(服务器 2026-08-13 16:26:26)
- **产物**: `experiments/08-hch-v2/results/R1B_STAGE2B_20260813_161534/`
- **协议**: hch_v2_r1b_stage2_broad_frozen_generalization_and_action_chain_v0.1 §13-§14
- **declared_source_commit**: 10bffe02bc68c3e62037d6dd2028f60c7487450a
- **parent_stage2a_dir**: R1B_STAGE2A_20260813_155817
- **候选**: 4 个 frozen — LearnedSig_main / PlainCore_main(12 source,4 host)+ LearnedSig_LOHO / PlainCore_LOHO(9 source: Linear/MLP/LSTM)
- **host 缓存**: 全部 seed=0;deep host caches(EPEX_FR/PJM_2020/GEFCOM14P × LSTM/PatchTST)在 Stage-2A 期间 CPU 并行预生成

---

## 1. Main deep-host panel(§13,zero-gradient,8 cells)

| cell | transfer 类别 | LearnedSig delta | PlainCore delta |
|---|---|---|---|
| EPEX_FR:LSTM | UNSEEN_MARKET | **-0.038093** | -0.037242 |
| EPEX_FR:PatchTST | UNSEEN_MARKET | **-0.042626** | -0.045171 |
| PJM_2020:LSTM | UNSEEN_DATASET_SAME_MARKET | **-0.024489** | -0.018851 |
| PJM_2020:PatchTST | UNSEEN_DATASET_SAME_MARKET | **-0.036869** | -0.030584 |
| GEFCOM14P:LSTM | UNSEEN_SCHEMA_REGIME | **-0.037673** | -0.029772 |
| GEFCOM14P:PatchTST | UNSEEN_SCHEMA_REGIME | **-0.062956** | -0.054724 |
| NORD_DK1:LSTM | UNSEEN_MARKET | **-0.052055** | -0.051522 |
| NORD_DK1:PatchTST | UNSEEN_MARKET | **-0.073456** | -0.073714 |

**8/8 全部 delta<0**(LearnedSig 与 PlainCore 均零正 cell)。Deep-host(LSTM/PatchTST)在 4 个代表市场全部转写成功。

## 2. 分类宏观(LearnedSig_main)

| 类别 | n | macro | frac<0 | worst |
|---|---|---|---|---|
| SOURCE_SEEN | 12 | -0.165172 | 1.0 | -0.011908 |
| UNSEEN_MARKET(deep) | 4 | -0.051557 | 1.0 | -0.038093 |
| UNSEEN_DATASET_SAME_MARKET(deep) | 2 | -0.030679 | 1.0 | -0.024489 |
| UNSEEN_SCHEMA_REGIME(deep) | 2 | -0.050314 | 1.0 | -0.037673 |

**GAP deep**: src(6 deep-source cells)=-0.138169,holdout=-0.046027,G_deep=+0.092142,R_retain=0.333。

## 3. 可复现性(与 Stage-2A 交叉核对)

- Stage-2B main 的 12 source cells 与 Stage-2A **完全一致**(SOURCE_SEEN macro 均为 -0.165172,每个 cell 逐位相同)。
- src-deep macro -0.138169 = Stage-2A 中 6 个 source LSTM/PatchTST cells 的均值(-0.051149-0.115351-0.014863-0.026477-0.231349-0.389824)/6 = -0.138169。**逐位一致**。

## 4. LOHO unseen-host 矩阵(§14,PatchTST 从未进入训练)

| cell | transfer 类别 | LearnedSig_LOHO | PlainCore_LOHO |
|---|---|---|---|
| LAGO_DE:PatchTST | UNSEEN_HOST | **-0.116972** | -0.111823 |
| LAGO_PJM:PatchTST | UNSEEN_HOST | **-0.024534** | -0.022391 |
| NEM_SA1:PatchTST | UNSEEN_HOST | **-0.385277** | -0.347154 |
| NORD_DK1:PatchTST | UNSEEN_MARKET_AND_HOST | **-0.074152** | -0.073321 |
| EPEX_FR:PatchTST | UNSEEN_MARKET_AND_HOST | **-0.042906** | -0.046340 |
| PJM_2020:PatchTST | UNSEEN_MARKET_AND_HOST | **-0.035782** | -0.033277 |
| GEFCOM14P:PatchTST | UNSEEN_MARKET_AND_HOST | **-0.057564** | -0.055521 |

- UNSEEN_HOST(source 市场、未见 host):macro **-0.175594**,frac<0=1.0。
- UNSEEN_MARKET_AND_HOST(市场+host 双未见):macro **-0.052601**,frac<0=1.0。
- **7/7 全部 delta<0** — true unseen-host × multiple market/dataset shift 矩阵零退化。

## 5. §13/§14 STOP rules

| 规则 | 判定 |
|---|---|
| DEEP_HOST_COLLAPSE(>1/3 of 8) | False(0/8 正) |
| LOHO_HOST_COLLAPSE(>1/3 of 4 holdout PatchTST) | False(0/4 正) |
| SIGNATURE_DEEP_NEGATIVE | False(LearnedSig source-deep -0.138 < PlainCore -0.124,holdout 也不劣化) |
| **VERDICT** | **CONTINUE — 无 §13/§14 deep-host collapse** |

## 6. MAE 安全

全部 19 cell:mae_rel_deg=0.0,safety=ok。签名仅调制 spread,点预测不变。

## 7. Provenance

- config.json 完整:declared 10bffe0 + 5 项 sha256(含 data_signature ff92d6c8…)。
- runner sha256(0b65a8643b59)与本地逐字节一致(服务器/本地 hash 比对 eb56d980cbb8)。
- 已知 cosmetic bug:summary.txt 的 `parent_stage2a` 字段读取了错误键名,config.json 的 `parent_stage2a_dir` 数据完好;已在本地 runner 修复。

## 8. 结论与下一阶段

Stage-2B **healthy**:8 deep-host + 7 LOHO 全负、DK1 LSTM/PatchTST 显著负(-0.052/-0.073)、可复现性逐位一致、无任何 collapse。
→ 按 §15 授权推进 **Stage-2C(HCH seeds 1/2)**。
