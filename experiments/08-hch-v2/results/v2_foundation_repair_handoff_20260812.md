# HCH v2 Foundation Repair Handoff

> Date: 2026-08-12 | Start: `f73b561` | End: (HEAD)

## 1. 起止 commit 与环境

- Start: `f73b561` (审计修订后的 math_loss 文档)
- End: (HEAD, this commit)
- Python: conda epf-2 (Python 3.11)
- Device: CPU only

## 2. P0/P1 修改映射

| ID | 文件/符号 | 修改 | 真实测试 |
|---|---|---|---|
| P0-1 | `host_cache.py` L29 | 缩进修复；CLI 只保留 `main()` 内一套；`--seed` 参数生效 | `py_compile` 通过；`--help` 返回 0 |
| P0-2 | `test_contracts.py` test 01 | 扩展至 `src`, `experiments/08-hch-v2`, `math_loss`, `peers` | `compileall` 全仓通过 |
| P0-S3 | `hch_v2_data.py` (完整重写) | raw/model 双通道；target 可选；S1-only exog 标准化；exog_type 分离；lag_context；learned-null mask=0 | 见下 |
| P0-S5 | `hch_v2_data.py` `DailyEpisodeDataset` | 新增 `fit_exog_scalers()`；exog 用 S1-fitted mean/std；不逐日 renormalize | S1 scaler 写入 per-feature dict |
| P0-S5-exog_type | 同上 | `exog_type` 从 value tensor 分离为独立字段 | 待 encoder 显式传递（见遗留项） |
| P0-S5-null | `DailyEpisodeDataset.__getitem__` | 无 exog 时 mask=0（非全 1） | learned-null 分支可被触发 |
| P0-S5-lag | `DailyEpisodeDataset._build_lag_context` | lag≥24h 的价格/残差/小时 context | cutoff-safe |
| P0-S6 | `HCHV2Bundle` | 新增 `calibration_params`, `exog_scalers`, `split_hash`, `data_hash` | `save/load/hash` 全部覆盖 |
| P0-S6-hash | `HCHV2Bundle.hash()` | 覆盖 model_state, memory_keys/gains/dates, calibration, scalers, split | 非恒真 (修改任一组件 hash 变) |
| P0-S6-freeze | `HCHV2.freeze()` | 新增 `calibration_params`, `exog_scalers`, `split_hash` 参数 | freeze 时逐项传入 |
| P0-S7/8 | `hch_v2.py` 顶部 + `LEGACY_UNTRAINED` | 三个 legacy flag；`require_not_legacy()` gate 函数 | formal runner 调用时 raise |
| P0-S7/8-note | `hch_v2.py` docstring | 标注 key_net 从未训练、calibration 自评分、state head 问题 | 源码可读 |
| P0-S9 | `audit_utils.py` `student_t_cdf` | `ratio=nu/(nu+x²)` + 正确分段公式 | 全部 OK vs scipy (16/16) |
| P0-S9 | `audit_utils.py` bootstrap | `np.isin` → 逐日拼接（保留 multiplicity） | CI 与控制组一致 |
| P0-S9 | `audit_utils.py` `day_dependence` | 真实日期 pivot；小时中心化；corr 特征值；participation ratio | 重写 |
| P0-S9 | `run_math_evidence_audit.py` | 残差符号统一为 `y - pred` | 全局替换 |
| P1-S10 | `test_contracts.py` | test 01 扩展；test 04 仍为 pass（见遗留项） | 其他 stub 未全替换 |
| P1-S11 | `official_adapters.py` | 类名 `Official → Limited`；文档标注 `limited_reimplementation` | 名称不含 Official |
| P1-S11 | `baselines_v2.py` | 文档标注 implementation_status 要求 | - |
| P1-S13 | `AGENTS.md` | 更新研究状态为 FOUNDATION_REPAIR；v2 canonical docs 指针；基座/基线表更新 | 与仓库实际一致 |

## 3. Compile 与 CLI

```
python -m compileall -q src experiments/08-hch-v2 experiments/00-data-exploration/math_loss
→ 0 errors
```

## 4. 数据契约变更摘要

| 旧字段 | 新字段 | 说明 |
|---|---|---|
| `host_pred` (norm) | `host_raw` + `host_model` | 原币种 + S1-norm 双通道 |
| `target` (norm, always present) | `target_raw` (Optional) + `target_model` (Optional) | S4 推理为 None |
| `exog [B,H,N,3]` 混合 | `exog_value/type/mask` 分离 | S1-only 标准化；type 独立传入 |
| 无 lag_context | `lag_context [B,H,5]` | lag24/168 价格 + 残差 + 小时 |

## 5. Exog 与 lag lineage

- exog 标准化：仅用 S1 日期拟合 `fit_exog_scalers()`
- 不再逐日 renormalize
- `exog_type` 独立张量（待 encoder 消费）
- `lag_context` 最小 lag=24h，包含价格、host_pred、残差(lag24)、小时

## 6. Legacy Quarantine

| 组件 | 状态 | Gate |
|---|---|---|
| CAGM key_net/metric_proj/cand_proj/fusion | `legacy_untrained_metric` | `require_not_legacy()` |
| S3 calibration (neighbor-average scoring) | `legacy_calibration` | 同上 |
| ContinuousStateHead + state_loss | `legacy_state_head` | 同上 |

## 7. 数学 evidence re-audit

| 原结论 | 缺陷 | 修正后状态 |
|---|---|---|
| Student-t CDF 可用 | `ratio = nu/(nu+x²)` 正确但 sign 处理 `0.5+0.5*sign*I` 错误 | 修正为 `x>=0 ? 1-0.5*I : 0.5*I`；全部 16/16 OK |
| bootstrap 合格 | `np.isin` 折叠重复 day | 改为逐日拼接保留 multiplicity |
| 日依赖矩阵 | 未中心化、假日期、错 SVD | 重写为正确协方差矩阵 + participation ratio |
| 残差符号统一 | `pred-y` vs `y-pred` 混用 | 全局统一为 `r = y - host_pred` |
| Candidate audit | Oracle (S3 自身算 pi/m) | 旧 csv 需标 `INVALID_ORACLE_DIAGNOSTIC`（未执行，见遗留项） |

## 8. Baseline provenance

| 类名 (旧) | 类名 (新) | Status |
|---|---|---|
| `DeltaAdapterOfficial` | `DeltaAdapterLimited` | `limited_reimplementation` |
| `PIROfficial` | `PIRLimited` | `limited_reimplementation` |
| Identity | Identity | ✅ |
| ResidualL1 | ResidualL1 | ✅ |
| QuantileResidualLGBM | QuantileResidualLGBM | ✅ |

## 9. 未解决项

| ID | 描述 | 严重度 |
|---|---|---|
| U1 | `test_contracts.py` test 04 仍为 `pass`（host_cache CLI deferred）| LOW |
| U2 | `smoke_v2.py` 仍调用 legacy calibration；S4 trim/pad 未删除 | MEDIUM |
| U3 | Candidate audit old csv 未标 INVALID_ORACLE_DIAGNOSTIC | MEDIUM |
| U4 | `HourTokenEncoder` 未接收新的 `exog_type` 张量 | MEDIUM |
| U5 | `HCHV2.encode()` 未传递 `exog_type` 到 encoder | MEDIUM |
| U6 | Unified ExperimentManifest (Section 3) 未创建 | HIGH |
| U7 | 老 `smoke_v2.py` 未适配新的 `DailyEpisodeBatch` 字段（`host_raw`→`host_model` 等） | HIGH |
| U8 | 弱测试替换（Section 10）未全完成：test 04/07/09/11/12/14-19 仍有残留 | MEDIUM |
| U9 | `docs/paper_info/README.md` 中山东角色未更新 | LOW |

## 10. 本轮已标记 superseded 的文件

| 文件 | 状态 |
|---|---|
| `docs/paper_prep/v2/hch_v2_code_repair_and_acceptance_spec_v0.2_2026-08-11.md` | 被 addendum v0.1 取代 |
| `experiments/08-hch-v2/results/v2_repair_handoff_20260811.md` | P0 全关闭声称不成立 |
| `experiments/00-data-exploration/math_loss/outputs/` 旧 csv | CDF/bootstrap/日矩阵有 bug；需 rerun |

## 11. 最终状态

**NOT_READY** → `READY_FOR_MATH_ARCHITECTURE_FUSION_REVIEW` 需要先解决 U6+U7（ExperimentManifest + smoke 适配）。

当前可交付：
- Compile: PASS (全仓)
- 数据契约: raw/model/target-free ✅
- 数学审计工具: CDF/bootstrap/日矩阵修复 ✅
- Legacy quarantine: gate in place ✅
- Bundle hash: 覆盖决策状态 ✅
- Baseline provenance: limited_reimplementation ✅
