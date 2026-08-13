# HCH-v2 R1A.8 Decision Calibration & Safe Abstention Audit v0.1

**日期：2026-08-13**  
**仓库基线：** `main@7f8ede8666b6f73369e980959eb336b13bf679f6`  
**前置结论：** `UNIVERSAL_CANDIDATE_SUPPORTED`；`R1A.7 = FUSION_UNRESOLVED, primary CASE_A`。

本阶段继续暂停 R1B，不加新市场、不加新 Host、不启动 U0、不重新训练 IAH。

## 0. 为什么下一步不是立刻加 action head

R1A.7 的 PJM:MLP 表现为：host residual 很低、IAH distributional CRPS 仍改善、outer atoms 较小、pure IAH fire rate 约 8.7%，少数 fire days 方向错误，且 `AUC(w-, B-)≈0.523`。

对于三原子 residual distribution

\[
R_h\sim w^-_h\delta_{-m^-_h}+w^0_h\delta_0+w^+_h\delta_{m^+_h},
\]

full Down action \(a_h=-m^-_h\) 的 expected absolute-error improvement 为

\[
\boxed{g_{h,\downarrow}^{IAH}=m^-_h(2w^-_h-1)}.
\]

因此

\[
g_{h,\downarrow}^{IAH}>0\iff w^-_h>0.5,
\]

Up 同理。低 fire rate 本身可能正是 absolute-loss Bayes decision 的结果，而不是 bug。

所以本阶段核心问题改为：

> **PJM:MLP 是否存在 result-pre 可预测的 action signal？如果不存在，HCH 是否应该安全地 Identity？**

---

## 1. 重新解释 hindsight oracle

R1A.5 的 \(A^{oracle}>0\) 几乎每天成立，只证明 outcome 已知后，Down/Identity/Up 中常有更好的动作；它不证明该动作能由 \(\mathcal F^-\) 提前预测。

以后区分：

\[
\boxed{Oracle\ Actionability}\neq\boxed{Predictable\ Actionability}.
\]

若 residual 方向在合法 pre-outcome 信息下近似不可预测，则 oracle gain 很高和 ex-ante Identity 最优可以同时成立。

---

## 2. R1A.8 只回答四个问题

1. 当前 IAH full-atom action 是否就是 absolute-loss Bayes decision？
2. PJM:MLP 的 Down/Up benefit 能否由冻结的合法 pre-outcome features 预测？
3. 三原子分布是否包含连续方向信息，只是 full-atom decision 太离散？
4. 对没有可预测 action signal 的 domain，IAH-native value + DVG 是否可以安全 abstain？

---

## 3. A0 — Bayes-action equivalence audit

absolute-loss Bayes act 是 predictive median：

\[
a_h^*=\begin{cases}
-m^-_h,& w^-_h>0.5,\\
0,& w^-_h\le0.5,\ w^+_h\le0.5,\\
+m^+_h,& w^+_h>0.5.
\end{cases}
\]

逐 domain 输出：

- Bayes Identity / Down / Up fraction；
- current directional fire fraction；
- exact mismatch count；
- tie count 与 deterministic tie rule。

除预声明 tie 外，预期 mismatch = 0。

---

## 4. A1 — Materiality audit

不能只看 gain 正负。记录：

- host hyperbolic MAE；
- absolute true action gain；
- oracle gain；
- gain / host-error；
- p10/p50/p90/p95；
- fire-day realized gain；
- non-fire-day oracle gain。

重点对照：`PJM:Linear`, `PJM:MLP`, `NEM:Linear`, `NEM:MLP`。

要回答：PJM:MLP 是“强 host + 低可行动余量”，还是“仍有大量可行动余量但完全不可预测”。

---

## 5. A2 — Frozen Actionability Probe

这是**诊断模型**，不是 production component。

### Targets

\[
B^-_h=\mathbf 1[g^{true}_{h,\downarrow}>0],\qquad
B^+_h=\mathbf 1[g^{true}_{h,\uparrow}>0].
\]

可同时保留连续 gain \(G^-_h,G^+_h\)。

### Features

只用冻结的 result-pre 信息：

- \(w^-,w^0,w^+\)；
- \(m^-,m^+\)；
- \(g^{IAH}_{down},g^{IAH}_{up}\)；
- \(z^0\)；
- local rank \(u\)；
- legal scale-free lag context；
- cyclic hour features；
- frozen learned daily signature。

禁止 current/future target、realized residual、market ID shortcut。

### Probe 1

固定 L2 Logistic Regression。

### Probe 2

只有 Probe 1 不足时，允许一个极小 2-layer MLP diagnostic（如 width=32）。仍然只作为 predictability upper-bound probe。

### Chronology

- S3M prefix：probe train；
- S3M suffix / S3C：validation；
- S4：development diagnostic only；
- 禁止 S4 tuning。

### Metrics

Down / Up 分别报告：ROC-AUC、PR-AUC、Brier、reliability bins、top-decile benefit enrichment、per-domain 与 equal-domain macro。

### Interpretation

**Unpredictable regime：** 若 PJM:MLP logistic 与 shallow nonlinear probe 都约 `AUC <= 0.55`，则不设计 action calibrator，PJM:MLP 视为 safe-abstain domain。

**Mapping failure：** 若简单 probe 稳定达到约 `AUC >= 0.60`，说明冻结 representation 中有 actionability information，但当前 `(w,m)->action` mapping 没读出来，才授权 tiny action-calibration layer。

这些阈值是开发 gate，不是理论定理。

---

## 6. A3 — Continuous distribution-signal diagnostic

检查 predictive residual mean：

\[
\boxed{\mu_h^R=w^+_hm^+_h-w^-_hm^-_h}.
\]

它是 squared-error Bayes functional，不直接替代 MAE median；这里只诊断三原子是否在 \(w^\pm<0.5\) 时仍含连续方向信号。

报告：

- Spearman(\(\mu^R,r^{true}\))；
- sign accuracy；
- \(\mu^R\) decile vs realized residual；
- PJM:MLP 单独结果。

若 \(\mu^R\) 有信号而 full-atom action 无信号，说明 action set / decision functional 太离散，值得下一轮研究；若 \(\mu^R\) 也无信号，则不要从该分布强造 action。

---

## 7. 一个重要数学结论：简单缩小 dose 不能修复 ranking

Down fractional dose：

\[
a_h=-\gamma m^-_h,\quad 0\le\gamma\le1.
\]

当前三原子 distribution 下：

\[
\boxed{g_{h,\downarrow}^{IAH}(\gamma)=\gamma m^-_h(2w^-_h-1)}.
\]

因此 \(w^-<0.5\) 时，任何 \(0<\gamma\le1\) 仍然 expected gain < 0。

所以“更保守 dose”只能缩小误动作损失幅度，不能创造新的 action-ranking information。本轮禁止盲目 sweep \(\gamma\)。

---

## 8. A4 — 用 IAH-native value 重新校准 DVG

回到最简单 value estimator：

\[
\widehat A=\widehat A^{IAH}.
\]

必须重新用该 estimator 的 S3C error：

\[
E_t^{IAH}=\widehat A_t^{IAH}-A_t^{true}
\]

得到新的

\[
q^{IAH}=q_{1-\alpha}(E_{S3C}^{IAH}).
\]

执行：

\[
LCB^{IAH}=\widehat A^{IAH}-q^{IAH}>0.
\]

禁止复用旧 W1-DVG q。

### Safe-abstention metrics

每 domain：release rate、Identity rate、harmful release rate、realized gain | release、final MAE/rMAE、degradation vs host、coverage、q、Ahat distribution。

若 PJM:MLP probe 显示不可预测，而 DVG `release≈0` 且最终误差约等于 host，这应判为正确 safe degradation，而不是 failure。

---

## 9. 修订 success criteria

不再要求所有 domain 都有正 Spearman。

### Actionable domain

存在稳定 result-pre action signal。要求：positive enrichment、release 后 realized gain > 0、最终 point metric 改善。

### Non-actionable domain

现有合法信息无稳定 action signal。要求：low release、low harm、host 性能基本保持。

这更符合 selective safe corrector 的定位。

---

## 10. 若授权 Tiny Action Calibration Layer

只有 A2 判为 Mapping failure 才进入。

第一版结构：

```text
Frozen IAH outputs / frozen representation
        ↓
tiny shared action calibrator
        ↓
p_down / p_identity / p_up
        ↓
restricted action selection
        ↓
DVG
```

要求：universal/shared、参数极小、无 market ID、IAH core frozen、candidate distribution 语义不变。优先 logistic / isotonic-like decision calibration，不直接上 deep value net。

---

## 11. R1B status

继续暂停。只有以下之一成立才启动：

- Route A：safe-abstention end-to-end 健康；
- Route B：probe 证明 mapping failure，tiny calibration layer 经开发集验证有效。

然后才扩 LSTM、PatchTST、DK1 和 candidate ablations。

---

## 12. Server decision

从当前截图选择：

\[
\boxed{RTX\ 4090 / 24GB（海南）}
\]

截图配置：15 vCPU、100GB RAM、30GB 系统盘、50GB 数据盘、按量约 ¥1.28/h。

理由：比 3090 方案仅贵一小截，但后续 host training / embedding extraction 的 wall-clock 更划算；100GB RAM 也比福建 4090 的 58GB 更适合数据预处理。A100 40GB/80GB 对当前任务不值得。

**R1A.8 本地跑即可；R1B 获批后再租。**

磁盘是唯一警告：50GB 对 R1B 勉强，对 U0/HF feature bank 太小。若平台可扩盘，R1B 至少 200GB，U0 建议 500GB 起；若不能扩盘，U0 时换实例或挂外部存储。

---

## 13. Required output

建议新增：

`experiments/08-hch-v2/r1a8_decision_audit.py`

产物：

```text
R1A8_DECISION_<timestamp>/
├── bayes_equivalence.csv
├── materiality_by_domain.csv
├── actionability_probe.csv
├── probe_reliability.csv
├── continuous_signal.csv
├── iah_dvg_metrics.csv
├── safe_abstention_summary.csv
├── figures/
└── DECISION_VERDICT.md
```

最终 verdict 只能为：

```text
SAFE_ABSTENTION_SUPPORTED
ACTION_MAPPING_FAILURE
ACTION_SIGNAL_UNRESOLVED
```

---

## 14. 决策树

```text
PJM:MLP low fire
    ↓
Is it exact Bayes consequence?
    ├── no  -> implementation/math bug
    └── yes
         ↓
Can frozen legal features predict benefit?
         ├── no
         │    ↓
         │  SAFE ABSTENTION
         │    ↓
         │  test IAH-native DVG
         │
         └── yes
              ↓
          ACTION MAPPING FAILURE
              ↓
          authorize tiny decision-calibration layer
```

现在需要先判断“该不该行动”，再讨论“怎样行动”。
