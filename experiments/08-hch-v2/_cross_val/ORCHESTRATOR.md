# HCH v0.3 Cross-Validation Orchestrator

> 每个 Phase 实现后，依次执行：编译 → 发派三 Agent → 汇总 → 放行/打回

## 工作流

```
实现代码 → compileall → 通过?
  ├─ NO  → 打回修编译
  └─ YES → 同时发派 Agent A/B/C
                │
         ┌──────┼──────┐
         A      B      C
         │      │      │
         └──────┼──────┘
                ↓
         汇总评分矩阵
                ↓
        全 ≥2 且均分 ≥2.5?
         /        \
       YES        NO
        ↓          ↓
    下一 Phase   打回+原因
```

## Phase 1 提示词模板 (示例)

发派时使用以下完整提示词，将 `{...}` 替换为实际路径：

---

### Agent A (Math Validator)

```
你是 HCH v0.3 数学验证 Agent。严格对照数学核心文档验证代码。

━━━━━━━━━━ 强制约束 ━━━━━━━━━━
1. 只依据 docs/paper_prep/v2_math/hch_v2_iah_crps_final_math_core_v0.3_2026-08-12.md (566行) 判定
2. 逐公式逐字符对照。变量名/符号/下标/括号必须完全匹配
3. 不许猜测、不许降低标准、给出具体行号和公式号
4. 区分严重度：符号错误=0，数值细微偏差=1-2

━━━━━━━━━━ 数学核心公式（速查） ━━━━━━━━━━
IAHCandidateHead:
  s = mean(|host_raw|) (式2)
  z0 = asinh(host_raw/s) (式3)
  w = softmax([l_minus, 0, l_plus]) — 中心 logit 固定为 0 (式6)
  m_minus = ReLU(r_minus), m_plus = ReLU(r_plus) (式7)
  z_minus = z0 - m_minus, z_plus = z0 + m_plus (式8)
  x_a = s * sinh(z_a) (式8下)

CRPS Loss:
  L = sum(w_a * |zY - z_a|) - w⁻(1-w⁻)m⁻ - w⁺(1-w⁺)m⁺ (式10)
  第二行是三原子下 -0.5*sum(w_a*w_b*|z_a-z_b|) 的化简，不是额外正则
  S2 唯一目标 (式11)。禁止 BCE/SmoothL1/MAE/state/tail loss

S1 Rank Reference:
  u = R(z0, hour, market, target) — 无学习参数，只用 S1 折外数据 (式5)

━━━━━━━━━━ Phase 1 验证（测试 1-8）━━━━━━━━━━━
请检查以下文件：src/iah_candidate.py src/iah_loss.py src/s1_rank.py test/test_phase1.py

对每个测试 (1-8)，独立判定。输出格式见 PROTOCOL.md。
不可跳过任何文件或测试。
```

---

### Agent B (Code Auditor)

```
你是 HCH v0.3 代码审计 Agent。检查工程正确性和接口契约。

━━━━━━━━━━ 强制约束 ━━━━━━━━━━
1. 编译优先。先跑 compileall，失败=全部 0 分
2. DailyEpisodeBatch 必须 13 字段
3. Legacy guard 阻断旧 loss/state/CARA/KL
4. NaN/inf/零值/空日/非24h日/全mask 必须处理
5. S4 不接受 target_raw/target_model
6. asinh/sinh 用 float64
7. SCALE_UNIDENTIFIED → Identity

━━━━━━━━━━ 检查清单 ━━━━━━━━━━
☐ compileall 0 errors
☐ IAHCandidateHead 输出: z0/z⁻/z⁺, w, m, scale, state_u, valid_mask
☐ CRPS 唯一 loss（无 BCE/SmoothL1/MAE/state/tail）
☐ S1 rank reference 无学习参数
☐ ContinuousStateHead 已删除或 legacy
☐ W1 不含 target/Y
☐ Query-dose: z_replay = z0_j + pi_q
☐ 不重叠: S1∩S2=∅

━━━━━━━━━━ Phase 1 审计 ━━━━━━━━━━
文件：src/iah_candidate.py src/iah_loss.py src/s1_rank.py test/test_phase1.py
测试：1-8

输出格式见 PROTOCOL.md。
```

---

### Agent C (Integration Tester)

```
你是 HCH v0.3 集成测试 Agent。运行代码并逐项核对输出。

━━━━━━━━━━ 强制约束 ━━━━━━━━━━
1. 实际运行代码，报告真实输出
2. 合成数据 + 固定 seed
3. 至少 1 个测试手工计算中间量并比对
4. 报告原始 stdout/assert/数值
5. 跳过 = 0 分

━━━━━━━━━━ Phase 1 测试（1-8）━━━━━━━━━━━
运行: python test/test_phase1.py

期望行为:
1. host×c → z0/w/m 不变, raw action ×c
2. host=0 → SCALE_UNIDENTIFIED + Identity
3. 改 model scaler 不改 CRPS
4. softmax(l⁻,0,l⁺) → sum=1
5. r⁻=r⁺=0 → Identity
6. x⁻ ≤ x⁰ ≤ x⁺
7. CRPS 与手工公式一致
8. 不等质量 W1 与手工 CDF 断点一致

输出格式见 PROTOCOL.md。必须包含 ≥1 个手工核对样例。
```

---

## 汇总评分矩阵 (Phase 1 示例)

| Test | Agent A | Agent B | Agent C | Min | Mean |
|---|---|---|---|---|---|
| 1 (等变) | 3 | 3 | 3 | 3 | 3.0 |
| 2 (零host) | 2 | 3 | 3 | 2 | 2.7 |
| 3 (scaler) | 3 | 2 | 3 | 2 | 2.7 |
| 4 (mass) | 3 | 3 | 3 | 3 | 3.0 |
| 5 (ReLU零) | 3 | 3 | 3 | 3 | 3.0 |
| 6 (单调) | 3 | 3 | 3 | 3 | 3.0 |
| 7 (CRPS) | 3 | 3 | 2 | 2 | 2.7 |
| 8 (W1) | 2 | 3 | 3 | 2 | 2.7 |
| **Agents** | **2.75** | **2.88** | **2.88** | | **2.83** |

**判定**: PASS → 进入 Phase 2

## 打回处理

若 PASS 失败：
1. 提取所有 Agent 的 <2 分项
2. 合并修复建议
3. 修复代码
4. 重新发派（只发相关 Phase 的 agent）
