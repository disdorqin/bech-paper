# HCH-v2 R1A.6 Retrieval / Value Estimation Recovery Plan v0.1

**日期：2026-08-13**  
**仓库基线：** `main@2f21a08224b80fb8a3c55bfe81eed168d97d7768`  
**前置结论：**

\[
\boxed{\texttt{UNIVERSAL\_CANDIDATE\_SUPPORTED}}
\]

\[
\boxed{\texttt{R1A.5\_BOTTLENECK = RETRIEVAL\_VALUE\_ESTIMATION}}
\]

**本阶段目标：** 在不重新训练 IAH candidate、不修改 CRPS、不修改 DVG 的前提下，恢复一个真正具有动作排序能力的 \(\widehat A\)。

---

## 0. Why R1A.6 instead of R1B

R1A.5 已经给出了很强的因果链：

1. D1：candidate dose 的 hindsight oracle 几乎每天都有正价值；
2. D2：当前 \(\widehat A\) 的排序能力近似于零；
3. D4：W1 retrieval 与 random retrieval 接近；
4. D5：部分市场存在 calibration drift，但它位于无效 value estimator 的下游。

因此目前最大的科学风险不是“host 不够多”，而是：

\[
\boxed{\text{HCH 有一个好的 correction distribution，但不知道如何从它和历史证据中判断今天应不应该执行}}
\]

所以：**R1B 暂停。先让 value estimation 在现有 6 个 domain 上恢复基本排序能力，再扩 Host / Market。**

---

# 1. R1A.5 code audit result

提交 `2f21a08` 的实现与 R1A.5 设计基本一致：

- candidate weights 冻结；
- D1–D5 retrospective only；
- D0 全链 before-freeze / after-reload audit 已加强；
- 变体已经修正为 `LearnedSig / Learned+DetSig`；
- 未运行 D6；
- 未启动 R1B；
- 未修改 IAH / CRPS / CAGM 数学核心。

R1A.5 的 `RETRIEVAL_VALUE_ESTIMATION` 结论可以作为下一阶段依据。

---

# 2. Two diagnostic caveats from R1A.5

## 2.1 D2 应按时间块单独报告

R1A.5 的总体 D2 会把：

```text
S3M-val
S3C
S4Q1
S4Q2
S4Q3
S4Q4
```

合并后计算 overall correlation。

R1A.6 必须额外计算：

\[
\rho_b=\operatorname{Spearman}(\widehat A,A\mid block=b).
\]

这样才能区分：

- S3M-val / S3C 已经 \(\rho\approx0\)：retrieval geometry 从一开始就缺 value information；
- S3M-val 有正相关、S4 逐步掉到 0/负值：static memory staleness / concept drift 可能才是主因。

## 2.2 R1A.5 `dist_weighted` 没有完整测试 weighted \(\widehat A\)

D4 中 distance weighting 作用于 hourly directional gain / proposal，但最终：

\[
\widehat A
\]

仍走当前 `estimate_action_value()` 的普通邻居均值。

因此 R1A.5 足以说明“weighted proposal 没救活结果”，但不能推出“weighted value estimator 无效”。

R1A.6 要显式测试：

\[
\widehat A_q^{weighted}
=
\sum_j \omega_j A_{q\to j}.
\]

---

# 3. New mathematical baseline: IAH-native expected action value

当前每小时 residual measure：

\[
\widehat R_h
=
w^-_h\delta_{-m^-_h}
+
w^0_h\delta_0
+
w^+_h\delta_{m^+_h}.
\]

对任意动作 dose \(\pi_h\)，预测分布自身定义：

\[
\widehat g^{IAH}_h(\pi_h)
=
\mathbb E_{R_h\sim\widehat R_h}
[
|R_h|-|R_h-\pi_h|
].
\]

这不是新 loss，而是对已经训练好的 IAH predictive distribution 直接计算 expected utility。

## 3.1 Down closed form

取：

\[
\pi_h=-m^-_h,
\]

则：

\[
\boxed{
\widehat g^{IAH}_{h,\downarrow}
=
m^-_h(2w^-_h-1)
}
\]

## 3.2 Up closed form

取：

\[
\pi_h=+m^+_h,
\]

则：

\[
\boxed{
\widehat g^{IAH}_{h,\uparrow}
=
m^+_h(2w^+_h-1)
}
\]

## 3.3 Whole-day value

将两组 expected directional gains 输入**现有 double-event optimizer**，得到：

\[
\pi_q^{IAH}.
\]

whole-day expected value：

\[
\boxed{
\widehat A_q^{IAH}
=
\frac1{|H_q|}
\sum_{h\in H_q}
\widehat g^{IAH}_h(\pi^{IAH}_{q,h})
}
\]

整个过程：

- 无历史 target；
- 无 CAGM；
- 无新增参数；
- 不改 IAH；
- 不改 CRPS；
- target-free。

这是 R1A.6 必须加入的零参数 baseline。

---

# 4. Why this baseline matters

它直接回答：

\[
\boxed{\text{candidate distribution 本身是否已经拥有 action-value information}}
\]

可能出现：

### A. IAH-native strong
CAGM 不适合作为 primary value estimator；未来可考虑 `IAH-native prior value + local evidence/calibration correction`。

### B. IAH-native weak, learned retrieval strong
dose 有用，但 action value 仍需 local context similarity；保留 CAGM 思想、替换 W1 key。

### C. Both weak
说明 distributional quality 与 ex-ante action ranking 仍有 gap；再考虑 distribution-aware dose 或独立轻量 value model。

---

# 5. Learned retrieval key v1

若 IAH-native 不足，第二个实验优先测试：

\[
\boxed{\text{learned temporal representation retrieval}}
\]

而不是继续微调 W1。

第一版直接复用 frozen HCH candidate 的 learned daily embedding：

\[
\boxed{
k_d
=
e^{learned}_d
=
\operatorname{ReLU}
(
W_{\rm sig}\operatorname{MeanPool}_h(h^{core}_{d,h})
)
}
\]

历史 day \(j\) 保存 result-pre key \(k_j\)，query 为 \(k_q\)。

第一版只用 cosine：

\[
D_{\rm emb}(q,j)
=
1-\cos(k_q,k_j).
\]

不要同时 sweep L1/L2/Pearson/DTW/Mahalanobis。

---

# 6. Memory freshness hypothesis

当前 local memory 在 S3-M 后冻结，覆盖较长 S4。

这对原 split-conformal 审计很干净，但工业上意味着新结果永远不进入 local evidence memory。

R1A.6 必须区分：

\[
\text{retrieval key problem}
\]

与：

\[
\text{memory staleness problem}.
\]

## Prequential memory diagnostic

candidate weights始终冻结。

对 S4 day \(t\)：

### before outcome
用：

\[
\mathcal M_{t^-}
\]

检索并产生 \(\widehat A_t\)。

### after outcome
将 day \(t\) 的：

- pre-outcome key；
- candidate atoms；
- target zY；
- valid mask

加入：

\[
\mathcal M_t.
\]

下一天才能使用。

这满足：

\[
Y_t\notin\mathcal F^-_t,\qquad
Y_t\in\mathcal F^-_{t+1}.
\]

本阶段只评价 value ranking，不重新声称 DVG safety。

---

# 7. True weighted value estimator

对于 neighbors \(N_k(q)\)，计算每个历史日的 query-dose realized value：

\[
A_{q\to j}.
\]

用自适应局部带宽：

\[
\tau_q
=
\operatorname{median}_{j\in N_k(q)}D(q,j)
\]

\[
\omega_j
=
\frac{
\exp[-D(q,j)/(\tau_q+\epsilon)]
}{
\sum_{\ell\in N_k(q)}
\exp[-D(q,\ell)/(\tau_q+\epsilon)]
}
\]

并定义：

\[
\boxed{
\widehat A_q^{weighted}
=
\sum_{j\in N_k(q)}
\omega_j A_{q\to j}.
}
\]

directional gain 也可以使用同一组 \(\omega_j\) 聚合。

---

# 8. R1A.6 experiment order

不要跑完整组合网格，按顺序证伪。

## V0 — Current baseline

\[
W1 + static\ memory + uniform\ mean.
\]

## V1 — IAH-native value

\[
IAH\ Expected\ Utility.
\]

无 retrieval、无 memory。

## V2 — Learned-key retrieval

保持 static memory + uniform mean，只把：

\[
W1 \to D_{\rm emb}.
\]

## V3 — Memory freshness

在 V1/V2 中历史检索型更好的分支上比较：

\[
static\ memory
\]

vs

\[
prequential\ expanding\ memory.
\]

## V4 — True weighted \(\widehat A\)

只在最佳 retrieval setup 上比较：

\[
uniform\ mean
\]

vs

\[
distance\ weighted\ mean.
\]

---

# 9. Do NOT test yet

R1A.6 禁止：

- retrain IAH；
- change CRPS；
- add BCE / tail loss；
- train value neural net；
- train metric-learning retriever；
- add MOMENT；
- add LSTM / PatchTST；
- add DK1；
- change DVG q；
- ACI / PID；
- tune action dose；
- add market ID；
- use Shandong。

---

# 10. Development-data status

R1A source S4 已经被 R1A.5 用于架构诊断，因此从现在开始：

\[
\boxed{R1A\ source\ S4 = DEVELOPMENT\ DATA}
\]

可以用它做 R1A.6 method discovery。

但选定 estimator 后，R1A source S4 不得再作为该 estimator 的 final confirmatory evidence。

真正确认留给：

- R1B unseen hosts；
- NORD_DK1；
- 后续其他市场；
- Shandong business holdout。

---

# 11. Required metrics

对每个 domain × estimator：

## Primary ranking

\[
\boxed{
Spearman(\widehat A,A^{true})
}
\]

并按 block 单独报告。

## Top-action enrichment

- \(P(A>0|\widehat A>0)\)
- \(P(A>0|\widehat A\text{ top10\%})\)
- mean/median \(A^{true}\) in top10%

## Secondary

- MAE(A_hat,A_true)
- bias

## Proposal

- realized A
- oracle efficiency
- missed-positive rate

## Stability

- per-domain
- macro
- worst-domain
- S4-quarter drift

---

# 12. Temporal uncertainty

daily observations 有序列相关，关键比较用 moving-block bootstrap：

- block length = 7 days；
- 500–1000 bootstrap samples；
- 95% CI for:
  - Spearman delta；
  - top10% realized-gain delta。

这是统计诊断，不是 conformal guarantee。

---

# 13. R1A.6 gates

这些是 engineering research gates，不是定理阈值。

## GREEN

优选 estimator 大致满足：

1. macro Spearman \(\ge 0.20\)；
2. 至少 5/6 domains Spearman > 0；
3. 无 domain \(\rho<-0.10\)；
4. top10% predicted-value days 的 realized A 明显高于全体；
5. 改善不是由单一 extreme market 拉动。

若多个方法通过，选择最简单者。

## YELLOW

\[
0.10\le\rho_{macro}<0.20
\]

或 domain 间差异明显。

## RED

若所有简单 estimator：

\[
\rho_{macro}<0.10,
\]

停止继续给 retrieval 加 trick，下一研究问题改为：

\[
\boxed{
candidate\ distribution
\rightarrow
action-dose\ design
}
\]

---

# 14. Architecture consequences

## Result A — IAH-native wins

可能简化为：

```text
Frozen Host
   ↓
Universal IAH
   ↓
analytic expected action value
   ↓
double-event
   ↓
local DVG
```

CAGM 降为 optional evidence / correction。

## Result B — learned retrieval wins

可能变为：

```text
Frozen Host
   ↓
Universal IAH
   ↓
learned pre-outcome daily representation
   ↓
local representation retrieval
   ↓
query-dose replay
   ↓
double-event
   ↓
DVG
```

## Result C — prequential memory decisive

正式区分：

\[
\theta_{\rm candidate}\quad\text{frozen}
\]

与：

\[
\mathcal M_t\quad\text{causally adaptive}.
\]

这对于工业应用非常自然：无需重训 universal weights，但 local evidence 随结果持续更新。

---

# 15. Literature anchors

R1A.6 只借鉴以下原则，不直接复制结构：

### Retrieval Augmented Time Series Forecasting / RAF
- historical motifs 可以提供有价值的外部上下文；
- retrieval 可以在 learned encoder representation space 中完成；
- similarity metric 与 time-causal retrieval database construction 重要。

### Nearest Neighbor Multivariate Time Series Forecasting
- 可直接使用 forecasting model representation 作为 cached-series nearest-neighbor key，不必训练独立 retriever。

### TS-RAG
- pretrained time-series encoder representation 可用于语义相关时间序列片段检索。

### FSNet
- 非平稳时序需要兼顾 recent adaptation 与 recurring-pattern memory；
- associative memory 可以因果地持续 read/write 更新。

这些工作共同支持优先测试：

\[
\boxed{
learned\ representation + causal\ updating\ memory
}
\]

而不是继续对静态 W1 做大量手工修补。

---

# 16. Required implementation

新增诊断文件：

```text
experiments/08-hch-v2/r1a6_value_recovery.py
```

不要先修改正式 HCH production modules。

建议 pluggable estimators：

```text
ValueEstimator
├── CurrentW1Static
├── IAHNativeValue
├── LearnedKeyStatic
├── LearnedKeyPrequential
└── LearnedKeyPrequentialWeighted
```

全部使用同一 frozen R1A candidate bundle。

---

# 17. Required artifacts

```text
R1A_VALUE_<timestamp>/
├── code_commit.txt
├── source_r1a_artifact.txt
├── estimator_config.json
├── value_by_day.csv
├── value_metrics_by_domain.csv
├── value_metrics_by_block.csv
├── top_decile_enrichment.csv
├── memory_growth.csv
├── bootstrap_intervals.csv
├── figures/
│   ├── ahat_atrue_*.png
│   ├── decile_gain_*.png
│   ├── spearman_by_block_*.png
│   └── memory_growth_*.png
└── VALUE_VERDICT.md
```

---

# 18. VALUE_VERDICT

最终只能选：

```text
IAH_NATIVE_VALUE
LEARNED_RETRIEVAL
PREQUENTIAL_MEMORY
WEIGHTED_LOCAL_VALUE
VALUE_ESTIMATION_UNRESOLVED
```

如果 winner 是组合，按“使结果第一次跨过 GREEN 的最小因果变化”命名。

---

# 19. After R1A.6

只有 R1A.6 GREEN 后：

1. freeze selected value-estimation rule；
2. 更新 architecture/training master doc；
3. 用修复后的 A_hat 再诊断 DVG；
4. 再进入 R1B：
   - LSTM；
   - PatchTST；
   - PlainCore/LearnedSig/DetSig；
   - Local-Core；
   - NORD_DK1。

若 R1A.6 RED：

> 不扩训练规模，回到 candidate-to-action 数学接口。

---

# 20. Current causal state

\[
\boxed{
\underbrace{\text{Universal Candidate}}_{\text{supported}}
\rightarrow
\underbrace{\text{Action Value Estimation}}_{\text{current bottleneck}}
\rightarrow
\underbrace{\text{Proposal}}_{\text{downstream}}
\rightarrow
\underbrace{\text{Safety Calibration}}_{\text{downstream}}
}
\]

现在正确的动作不是“加更多数据、更多模型”，而是：

\[
\boxed{\text{repair the first failed causal link, then re-test}}
\]
