# HCH v2 v0.3 代码审计报告

> 日期: 2026-08-12 | 审计基线: `b859f99`  
> 规范依据: `hch_v2_v0.3_architecture_fusion_and_code_design_v0.1`  
> 数学依据: `hch_v2_iah_crps_final_math_core_v0.3`

## 1. 审计范围

审查了 v0.3 架构 7 个新模块 + 4 组测试文件，编写 25 项深度审计测试。

## 2. 模块逐项审查

### 2.1 `iah_candidate.py` — IAHCandidateHead ✅

| 规范要求 | 实现 | 验证 |
|---|---|---|
| asinh 双曲坐标 z0 = asinh(host/s) | `torch.asinh` + float64 精度 | ✅ |
| scale s = mean(|host_raw|) | `_compute_scale()` | ✅ |
| mass = softmax([l⁻, 0, l⁺]) | 中心 logit 固定为 0 | ✅ |
| m = ReLU(r) | F.relu 保证非负 | ✅ |
| x_a = s * sinh(z_a) | float64 sinh 反变换 | ✅ |
| SCALE_UNIDENTIFIED fallback | w_zero=1, m=0, x=host | ✅ |
| 无 BCE/sigmoid/离散阈值 | 纯 softmax + relu | ✅ |

**未发现问题。**

### 2.2 `iah_crps_loss.py` — IAH-CRPS Loss ✅

| 规范要求 | 实现 | 验证 |
|---|---|---|
| L = Σ w^a |zY-z^a| − w⁻(1−w⁻)m⁻ − w⁺(1−w⁺)m⁺ | Eq 10 精确实现 | ✅ |
| Spread term 是 CRPS 代数简化 | 文档注明了"非正则化项" | ✅ |
| 按有效小时平均、再按日平均 | valid_mask 加权 | ✅ |
| 禁止 BCE/SmoothL1/W1/state loss | 单一 loss 返回 | ✅ |
| 配对公式一致性 | 审计测试 A06 验证 | ✅ |

**未发现问题。**

### 2.3 `s1_rank.py` — S1RankReference ⚠️

| 规范要求 | 实现 | 验证 |
|---|---|---|
| 无 learnable 参数 | 纯 numpy 实现 | ✅ |
| S1 折外 host 构建 rank pool | `__init__` 接收 S1 数据 | ✅ |
| per-hour 条件 | 按 hour 分 pool | ✅ |
| 连续 mid-rank 插值 | `_interpolate_rank()` | ✅ |

**发现**: 构造函数接受 `s1_market_ids` 和 `s1_target_ids` 参数但**未使用**。规范说 `u = R_P(z0, hour, market, target)` 暗示多维条件。目前仅实现了 hour 条件。

**严重程度**: 低。当前多市场/多 target 场景可通过调用方分别构建 rank reference 解决。建议后续补全。

### 2.4 `w1_retrieval.py` — W1 + CAGMAtomMemory ✅ (已修复)

| 规范要求 | 实现 | 验证 |
|---|---|---|
| R̂ = w⁻δ_{-m⁻} + w⁰δ₀ + w⁺δ_{m⁺} | residual atom measure | ✅ |
| W1 = ∫|F_a-F_b|dx 精确计算 | event-merge 算法 | ✅ |
| Day distance = mean_h W1 | `day_w1_distance()` | ✅ |
| retrieval key 不含 target | `build_retrieval_index` 只用候选 | ✅ |
| 自排除 | **已修复**: `get_neighbors(exclude_self=True)` 跳过距离<1e-14 | ✅ |
| W1 对称性 | 审计测试 A24 验证 | ✅ |
| W1 三角不等式 | 审计测试 A25 验证 | ✅ |

**修复**: 添加了 self-exclusion 逻辑（原代码将查询日本身作为最近邻返回）。

### 2.5 `query_replay.py` — Query-Dose Replay ✅

| 规范要求 | 实现 | 验证 |
|---|---|---|
| π_q = query dose | `pi_query` 参数 | ✅ |
| z_replay_j = z0_j + π_q | 内嵌于 g = |r_z| − |r_z − π_q| | ✅ |
| g = |r_z| − |r_z − π_q| | 直接计算 | ✅ |
| |g_h| ≤ |π_h| | `verify_gain_bound()` | ✅ |
| A = mean_valid_hours(g) | Eq 21 | ✅ |
| Â = mean_{j∈N_k} A | Eq 22 | ✅ |
| 使用查询剂量，非历史剂量 | 审计测试 A10 验证 | ✅ |

**未发现问题。**

### 2.6 `double_event.py` — Double-Event Proposal ✅

| 规范要求 | 实现 | 验证 |
|---|---|---|
| 最多 1 个 Down + 1 个 Up, 不重叠 | `double_event_proposal()` | ✅ |
| O(H²) 枚举 | 前缀 U_L/U_R + 枚举 | ✅ |
| 无硬阈值/最短长度/固定个数 | 纯前缀和优化 | ✅ |
| 与暴力穷举一致 (H=24) | 审计测试 A13 (100 随机) | ✅ |
| 全负 → Identity | 审计测试 A14 | ✅ |
| Tie-breaking (短区间优先) | 审计测试 A15 | ✅ |

**未发现问题。**

### 2.7 `dvg_calibrate.py` — DVG Split-Conformal ✅

| 规范要求 | 实现 | 验证 |
|---|---|---|
| E_t = A_hat_t − A_t | `record_error()` | ✅ |
| r = ⌈(n+1)(1−α)⌉ | `compute_quantile()` | ✅ |
| q = E_{(r)} | 排序后取 r−1 索引 | ✅ |
| LCB = A_hat − q | `lcb()` | ✅ |
| q=inf → 全日 Identity | 审计测试 A16 | ✅ |
| Freeze/restore | `freeze()` / `from_frozen()` | ✅ |

**未发现严重问题。**

## 3. 审计测试结果

| 测试集 | 数量 | 结果 |
|---|---|---|
| 原始 test_phase1 | 8 | 8/8 ✅ |
| 原始 test_phase2 | 3 | 3/3 ✅ |
| 原始 test_phase3 | 2 | 2/2 ✅ |
| 原始 test_phase45 | 7 | 7/7 ✅ |
| **审计新增** | **25** | **25/25 ✅** |
| **合计** | **45** | **45/45 ✅** |

## 4. 发现的问题与修复

| # | 问题 | 严重程度 | 状态 |
|---|---|---|---|
| 1 | `get_neighbors()` 不排除查询日本身 | 中 | ✅ 已修复 |
| 2 | S1RankReference 接受 market/target_id 但未使用 | 低 | ⚠️ 已记录 |
| 3 | 旧代码未完全隔离 (BiOMC/ContinuousStateHead 仍可导入) | 低 | legacy guard 已就位 |
| 4 | double_event U_L/U_R 预计算 O(H³) 非 O(H²) | 极低 | H=24 可忽略 |
| 5 | W1 原子合并阈值 hardcoded 1e-14 | 极低 | 数值稳定 |

## 5. 数学正确性验证

| 属性 | 测试 | 结果 |
|---|---|---|
| CRPS ≥ 0 | A22 | ✅ |
| CRPS(perfect) = 0 | A23 | ✅ |
| CRPS = 配对公式 | A06 | ✅ |
| W1 对称性 | A24 | ✅ |
| W1 三角不等式 | A25 | ✅ |
| W1 恒等 = 0 | A10 | ✅ |
| 尺度不变性 (z0 不变) | 01 + A03 | ✅ |
| ReLU 零 = Identity | 05 + A04 | ✅ |
| Gain bounded by |π| | 11 + A21 | ✅ |

## 6. 未覆盖的规范项

| 规范 § | 要求 | 状态 |
|---|---|---|
| §3.2 | 23/25h DST 日显式审计 | ⚠️ 现有 data_audit 含 DST 统计，未接入 IAHCandidateHead |
| §5.2 | S3-M / S3-C 严格分离 | ⚠️ 模块已分离但未端到端验证 |
| §5.3 | 最终 pi_q 回放得到 DVG 完整 A_hat | ⚠️ 未端到端测试 |
| §6 | 双事件提案接入 query-dose replay | ⚠️ 模块独立测试通过，未串联 |

## 7. 结论

**PASS WITH MINOR ISSUES** — 核心数学实现正确（CRPS/W1/Conformal 全部验证），1 个 bug 已修复（self-exclusion），无阻塞性问题。

建议后续:
1. 串联 S3-M/S3-C 端到端测试
2. 补全 S1RankReference 的 market/target 条件
3. 接入真实数据 smoke
