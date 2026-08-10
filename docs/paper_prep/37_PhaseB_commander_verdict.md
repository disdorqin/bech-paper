# 37 · Phase B 主窗口总裁决

> 日期：2026-08-09  
> 角色：主窗口（科学总指挥）  
> 输入：`35a` / `35b` / `35c`（Round-1 仅文档）  
> 状态：**Phase B Gate 1–2 关闭；真实 pilot 不授权。**

---

## 0. 终局裁决（唯一）

| 项 | 裁决 |
|---|---|
| **总裁决** | **`NO-GO`（A 会方法/理论路线）** |
| 问题锚点 | **保留**（P0：负价 episode + 完整漏报） |
| 现有 BECH 01–04 | **保留为 Route-E 应用证据** |
| Phase C / `experiments/07-*` | **禁止创建与运行** |
| 可选后续 | **Route-E 论文整理**；或用户明确指定后 **PIVOT 换对象**（须新开 Phase B，不得修补本候选） |

**不采用** `GO-to-next-stage`。  
**不采用**“三 Agent 投票成立创新”。  
A 还原闭合 **或** B 无合格 T1 任一成立即停实验；本次 **两者同时成立**。

---

## 1. 主窗口亲自复算（Gate 0）

### 1.1 生死公式

固定 \(x=(b,Z)\)、动作 \(a\)、编辑器 \(T\)、有界 \(L_{ep}\)：

\[
H(a;x,Y)=L_{ep}(T(b,a),Y)-L_{ep}(b,Y).
\]

第二项 \(L_{ep}(b,Y)\) **不依赖** \(a\)，故

\[
\arg\min_{a\in\mathcal A(x)}
\mathbb E[H(a;x,Y)\mid x]
=
\arg\min_{a\in\mathcal A(x)}
\mathbb E[L_{ep}(T(b,a),Y)\mid x].
\]

**主窗口确认：恒等式成立（identity）。**  
“直接学习相对 harm”在同一动作类、同一信息集下 **不产生新的 Bayes 动作**。

### 1.2 Risk-table 同序

有限菜单 \(\{a_i\}\cup\{\mathrm{id}\}\) 上，若

\[
r_a(x)=\mathbb E[L_{ep}(T(b,a),Y)\mid x],
\]

则

\[
r_a(x)-r_{\mathrm{id}}(x)
=
\mathbb E[H(a;x,Y)\mid x].
\]

故 \(\arg\min_a(r_a-r_{\mathrm{id}})=\arg\min_a r_a\)（identity 为公共偏移）。  
**主窗口确认：与 35a R2 / risk-table、35b (ID-BAYES)/(ID-RISK) 一致。**

### 1.3 三份报告是否打到协议要求

| 检查项 | 结果 |
|---|---|
| A 是否攻击 ABS-SET+VAL+RC（非逐点稻草人） | **是** |
| A 是否含 risk-table reduction | **是** |
| A 终局标签 | `REDUCTION PASS/KILL` |
| B T0 是否 KNOWN-FRAMEWORK | **是**，且 PROVABLE |
| B 是否仅一个 T1 | **是**（精确锚点有限样本优势） |
| B T1 是否非定义性新 lemma | **否** → `NOT JUSTIFIED` |
| C 是否禁止 A/B 未过前 pilot | **是** |
| C 默认预算 | **K 个 episode**（主报 K=1） |
| C 最强证伪 | DGP-4 + B3；预期等价 → NO-GO |
| 三者冻结对象是否一致 | **是**（24h / 冻结 b / \(L_{ep}\) / 相对风险） |
| 是否偷跑代码或 07 目录 | **否** |

---

## 2. 分 Agent 验收摘要

### 2.1 Agent A — `35a_AgentA_PhaseB_reduction_audit.md`

| 层 | 主窗口采纳 |
|---|---|
| Object | **KILL** — Bayes 层与表示层（固定 \(b\) 下 edit ↔ 绝对轨迹/区间）可还原 |
| Algorithm（公平有限菜单） | **KILL** — 同菜单 + 同 \(D_B\) + 同 S3 ⇒ 与 ABS-SET+VAL+RC / risk-table 决策等价 |
| Theory novelty | **KILL** — 成对 RC = 已知框架例化 |

**存活性质：** 主窗口同意 A —— **无绑定本候选的合格存活点**。  
“文献四元组组合差”“Bi-Hurdle 对齐”“无限连续动作开口”均 **不得** 记为 A 会对象存活。

### 2.2 Agent B — `35b_AgentB_PhaseB_theorem_sheet.md`

| 项 | 主窗口采纳 |
|---|---|
| T0 | **PROVABLE AS STATED** + **KNOWN-FRAMEWORK INSTANCE**（可作工程附录，非贡献） |
| T1 | **NOT JUSTIFIED** |
| 核心理由 | \(R_{rel}(\pi)=R_{abs}(\pi)-R_{abs}(\mathrm{BASE})\)；ABS\* 使用同一配对 \(\Delta\) 时半径 **identity**，无严格样本复杂度优势 |
| 弱化“配对优于未配对 dual-mean” | 经典配对常识，**非 new lemma** |
| Gate 2 | **KILL（A 会理论）** |

**主窗口复算 T1 塌缩：**  
在“最强合法 absolute 控制被允许使用 **同一** 有限菜单、**同一** 日级配对损失差、**同一** S3 选择规则”时，相对证书与“绝对风险减去 BASE 绝对风险”的证书 **重合**。  
故“精确基座锚点 ⇒ 更紧有限样本半径”在该公平比较下 **证不出严格优势**。同意 B。

### 2.3 Agent C — `35c_AgentC_PhaseBC_falsification_plan.md`

| 项 | 主窗口采纳 |
|---|---|
| 设计完整性 | **合格作为证伪档案** |
| 默认预算 K | **同意**（与 episode 问题锚点同构；利于 DGP-4 穷举） |
| DGP-4 + B3 | **正确的最强证伪点**；与 A/B 的 identity 一致，**预期等价** |
| Phase C 授权 | **本轮不授予**；C 文内闸门写法正确 |

C 未越权实现；其“获批后 07 范围”仅作档案，**不因本文存在而启动**。

---

## 3. 三闸结论

| Gate | 结果 | 说明 |
|---|---|---|
| **Gate 1 公式碰撞** | **KILL** | 同信息/动作/预算/损失/假设下，目标·推理·保证可被 ABS-SET+VAL+RC + risk-table 保持 |
| **Gate 2 理论诚实** | **KILL** | T0 正确但已知；T1 无非定义性 new lemma |
| **Gate 3 最小证伪** | **不进入** | A/B 未双过；禁止合成以外的真实 pilot；连 07 目录也不建 |

---

## 4. 明确保留 vs 明确删除

### 4.1 保留

1. **P0 问题锚点**：负价多小时 episode；冻结基座完整漏报（Linear/GBDT 审计量级）。  
2. **Route-E**：现有 BECH（BOM-SSC + SCARR）01–04 应用证据链；可写严格实证/领域方法论文。  
3. **工程证书语言**：T0 类有限策略成对 UCB/RC + identity fallback，可作 **附录实现**，标注已知框架。  
4. **Hurdle/Tweedie/ZILN**：可作为 **实现与 Related Work 对齐**，不写“电价首创双分量理论”。  
5. **数据策略**：主实验公开可复现；山东动机 + 内部补充。

### 4.2 删除 / 禁止写入贡献

1. base-relative episode edit 为 **A 会新学习对象**  
2. “学相对 harm”单独改变最优决策  
3. 精确锚点带来的 **严格** 有限样本优势（在公平 ABS\* 下）  
4. Hungarian / 连通分量 / exact fallback / 预算定义性满足 = 新定理  
5. Bi-Hurdle = 新定理（仅对齐）  
6. GNN/TAP/GAFE 贯穿、e-process、admin clamp、A4、双市场联合、旧 06 原型数值  
7. 经验测试集零退化 = 高概率总体非退化  
8. Phase A“八篇未见完整四元组”= 全球空白或创新证明  

---

## 5. 对用户目标的诚实映射

| 用户目标 | 本轮结论 |
|---|---|
| 对标 A 会方法/理论 | **本候选 NO-GO** |
| 即使 Route-E 也要创新点 | 应用论文可主张：**模型无关选择性极端校正 + 跨市场公开基准 + 事件级评估协议 + 安全回退**；**不是**新 Bayes 对象或新有限样本定理 |
| 冻结基座 | **保持**（Route-E 差异化与可部署性仍成立） |
| Hurdle 双损失 | **实现层保留**；理论层不升格 |
| GNN | **本轮不引入**；远期另案，不救援本候选 |
| 三子 Agent 启发式探索 | 已完成；结果为 **可证伪的否定**，优于包装性肯定 |

---

## 6. 允许的下一步（须用户点头）

### 选项 R — Route-E（推荐默认）

- 冻结 Phase B 候选为 **否决档案**（本文 + 35a/b/c）。  
- 整理应用论文：问题（负价 episode）+ 方法（BECH 后处理）+ 01–04 证据 + 事件级指标 + 局限。  
- 投稿定位：应用轨 / 能源预测期刊或 KDD 应用，**不以 A 会新定理卖点**。

### 选项 P — PIVOT（仅当用户要继续理论线）

必须 **换对象**，例如（示例，未论证）：

- 在 **不可实现** absolute oracle 或 **严格更小** 可实现策略类下的计算/样本分离（须新冻结规格）；  
- 预注册生成模型下 episode 依赖的 **正 excess-risk 下界**（针对明确策略类）；  
- 与预测正交的 **制度/规则可证安全层**（非 admin clamp 包装）。

**禁止：** 在同一对象上加大网络、加 GNN、加数据集、或只打败逐点弱基线来翻盘。

### 选项 S — 停止扩展

- 仅归档；不写论文。

---

## 7. 执行禁令（即时生效）

```text
❌ 创建 experiments/07-episode-relative-pilot/
❌ 运行真实 pilot / 全矩阵“验证创新”
❌ 修改 01–05 冻结证据以服务本候选叙事
❌ 修补 _archive/06-event-edit-prototype
❌ 将 35a/b/c 或本文写成“创新已成立”
✅ 可引用本文作为 Phase B NO-GO 的权威裁决
✅ 可开始 Route-E 文稿/图表整理（另开任务，非本裁决自动授权大改代码）
```

---

## 8. 验收清单（主窗口自检）

- [x] 亲自复算 argmin 等价与 risk-table 同序  
- [x] A 攻击最强组合，非稻草人  
- [x] B T1 非定义性失败，T0 诚实降级  
- [x] C 未偷跑 pilot；证伪点与 A/B 一致  
- [x] 唯一输出：`NO-GO`（A 会方法/理论）  
- [x] 问题锚点与 Route-E 证据边界清晰  

---

## 9. 一页给用户

**Phase B 单候选生死审计结束。**

- 候选「非可加 episode 损失下的 base-relative 负价事件校正」在 **对象 / 公平算法 / 理论新颖性** 三层被还原或证伪。  
- **A：`REDUCTION PASS/KILL`。B：T1 `NOT JUSTIFIED`。C：不授权 Phase C。**  
- **主窗口：`NO-GO` → 停真实实验；转 Route-E 或你指定的换对象 PIVOT。**  
- 负价 episode 问题仍真实；现有 BECH 仍可作应用贡献，但 **不能** 再把该候选写成 A 会方法/理论创新。

---

*End of commander verdict. Round-1 closed.*
