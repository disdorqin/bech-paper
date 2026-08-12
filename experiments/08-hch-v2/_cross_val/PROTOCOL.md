# HCH v0.3 Cross-Validation Protocol

> 三 Agent 独立验证。每个 Phase 实现后，三个 Agent 并行审查。
> 全部通过 (均 ≥2, 均分 ≥2.5) 才进入下一 Phase。

## Agent 角色

| # | Agent | 参照文档 | 核心检查 |
|---|---|---|---|
| A | **Math Validator** | `math_core_v0.3.md` 全文 | 公式逐字符对照、符号、properness、等变性 |
| B | **Code Auditor** | `fusion_design_v0.1.md` §3-9 | compile、接口、edge case、legacy guard |
| C | **Integration Tester** | `fusion_design_v0.1.md` §10 | 合成数据运行、手工核对、逆变换检验 |

## 评分规则

对 Phase 的每个测试项，每个 Agent 给出 0-3 分：

| 分数 | 含义 |
|---|---|
| 3 | 通过，有清晰可复现证据 |
| 2 | 通过，有小问题但非阻塞（必须列出具体问题） |
| 1 | 失败，可修复（必须给出修复建议） |
| 0 | 失败，根因问题（必须给出根因分析） |

## 放行条件

```
所有 Agent 对每个测试项 ≥2
且所有 Agent 均分 ≥2.5
→ Phase 通过
```

任一 Agent 对任一项 <2 → 打回重做，附具体原因。

## Phase 与测试映射

| Phase | 实现内容 | 测试号 | 测试内容 |
|---|---|---|---|
| 1 | IAH 候选 + CRPS | 1-8 | 等变/零宿主/CRPS公式/mass/ReLU/不等式 |
| 2 | W1 + replay | 9-11 | key不含outcome/剂量回放/收益界限 |
| 3 | 双事件提案 | 12-13 | 穷举一致/不重叠/空提案 |
| 4 | S3-M/S3-C + LCB | 14-15 | 冻结不更新/conformal边界 |
| 5 | bundle + S4 + guard | 16-20 | target-free/round-trip/legacy/timestamp |

## 输出规范

每个 Agent 返回：

```
Phase: X
Agent: [A|B|C]
Test 1: [0-3] 理由/证据
Test 2: [0-3] 理由/证据
...
均分: X.X
判定: [PASS|FAIL]
问题清单: (如有)
修复建议: (如有)
```
