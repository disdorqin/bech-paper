# HCH-v2 R1A 科学审查、R1A.5 诊断与 R1B 修订计划 v0.1

**日期：2026-08-13**  
**仓库审查基线：** `main@9e4a347b5e4d7d46bd4502d9743bfbedd4b58cb9`  
**当前裁决：**

\[
\boxed{
\text{Candidate Layer = GREEN}
\qquad
\text{Action / DVG Layer = NOT YET VALIDATED}
}
\]

R1A 已经足以证明继续研究，但还不足以证明完整 HCH 动作链已经成功。

---

## 0. 总体决策

R1A 最重要的结果是：

\[
\boxed{
\text{同一套 IAH candidate 权重在 6 个 market-host domains 上全部优于 frozen-host CRPS baseline}
}
\]

这首次实证支持了 HCH 的核心假设之一：

\[
\text{transferable host-relative correction prior}.
\]

但当前实验也同时显示：

\[
\boxed{
\text{candidate distribution 很好}
\not\Rightarrow
\text{当前 CAGM + proposal + DVG 动作链就一定有效}
}
\]

因此建议顺序调整为：

```text
R1A completed
    ↓
R1A.5 causal diagnostics
    ↓
定位 candidate dose / retrieval / proposal / DVG calibration 的主瓶颈
    ↓
仅做必要的最小修正
    ↓
R1B-native
    + LSTM
    + PatchTST
    + Local-Core
    + true signature ablation
    + NORD_DK1 frozen holdout
    ↓
native transfer verdict
    ↓
server-scale U0 foundation representation distillation
```

---

# 1. R1A 代码构建审查

当前 `r1a_run.py` 基本符合冻结协议：

- 3 markets × 2 hosts；
- `d_model=64`, `d_sig=32`；
- optional branch disabled；
- AdamW `3e-4 / 1e-4 / clip 1.0`；
- host/HCH seed = 0；
- equal-domain UniversalCoreTrainer；
- macro S2V IAH-CRPS checkpoint selection；
- S1R-only rank/signature；
- S3-M memory/k；
- S3-C DVG；
- target-free S4；
- 未重新引入 legacy multi-loss。

Candidate 主训练结果因此可以保留。

---

# 2. R1B 前必须修正的两个审计问题

## 2.1 当前 `Universal-NoSig` 不是“真正无 Signature”

当前 Data Signature 为：

\[
e=[e^{det}_{S1R}, e^{learned}_{day}]
\]

再经 FiLM：

\[
h'=(1+\Delta\gamma(e))h+\beta(e).
\]

R1A 当前两个变体实际是：

### 当前 NoSig

\[
e^{det}=0,\qquad e^{learned}_{day}\neq0.
\]

### 当前 Sig

\[
e^{det}=e^{det}_{S1R},\qquad e^{learned}_{day}\neq0.
\]

因此 R1A 的正确结论只能是：

\[
\boxed{
8\text{维冻结 S1R descriptor 在 learned daily signature 之上没有额外增益}
}
\]

不能写成：

\[
\text{Data Signature 无增益}.
\]

### R1B 修正

将当前变体重命名：

- `LearnedSig` = 旧 `Universal-NoSig`
- `Learned+DetSig` = 旧 `Universal-Sig`

并新增真正的：

### `PlainCore`

直接绕过 DataSignature modulation：

\[
h'=h.
\]

R1B 正式比较：

\[
\boxed{
PlainCore
\;vs\;
LearnedSig
\;vs\;
Learned+DetSig
}
\]

当前证据下，`LearnedSig` 可作为暂定主候选；`Learned+DetSig` 降级为消融。

---

## 2.2 当前 round-trip 验收过弱

当前 runner 先用 reload 后的 `pipe2` 产生 S4，然后 `_roundtrip_check()` 再调用同一个 `pipe2`，仅比较少量 `x_identity`。

这不能验证之前规定的完整 bundle contract：

\[
\text{candidate}
\rightarrow
\text{neighbors}
\rightarrow
\pi
\rightarrow
\widehat A
\rightarrow
q
\rightarrow
LCB
\rightarrow
\text{final action}.
\]

### R1B 前修正

对同一个 query 分别运行：

```text
pipe_before_freeze
pipe_after_reload
```

比较：

- scale；
- rank；
- atom masses；
- shifts；
- W1 distances；
- neighbor IDs；
- Down/Up intervals；
- final pi；
- A_hat；
- q；
- LCB；
- execute/Identity；
- final raw prediction。

至少 3 个固定 query 全链一致。

这属于审计修复，不改变数学。

---

# 3. 其他小修正

## 3.1 Bundle provenance

local bundle 必须分别记录：

```text
dataset_id
host_id
```

不要用 `dataset_id` 同时充当 host。

Universal checkpoint 记录全部 source datasets 与 source hosts。

## 3.2 sMAPE

最终评价按：

\[
sMAPE=
\frac1N\sum
\frac{2|y-\hat y|}
{|y|+|\hat y|}
\]

其中：

```text
(y, yhat)=(0,0) -> 0
```

不使用 price floor；不要让计算 epsilon 改变数学定义。

---

# 4. R1A 科学结果解释

## 4.1 Universal candidate 已经通过第一层 falsification

报告：

\[
L_{\rm IAH}^{macro}\approx0.1973
\]

对比：

\[
L_{\rm host}^{macro}\approx0.3862.
\]

而且 6/6 domains：

\[
\Delta_g=L_g^{IAH}-L_g^{host}<0.
\]

这不是某一个市场拉动平均值，而是跨 market × host 的共同改善。

## 4.2 Candidate 没有坍缩

报告：

\[
w^0\approx0.375,
\]

mass entropy 约 1.09，两侧 shift alive rate 约 0.88。

说明共享模型没有简单退化为：

\[
F=\delta_{z^0}.
\]

## 4.3 冻结 descriptor 很可能冗余

`0.19728 vs 0.19729` 基本相同。

当前最简解释：

> 已有 asinh 几何、local rank、scale-free history 和 learned per-day signature 已经提供了足够的 domain information；8维全局 S1R 统计量没有明显增加信息。

这更像“可以简化”的证据，而不是失败。

---

# 5. 为什么 candidate 成功而动作仍可能失败

IAH 优化的是概率分布：

\[
F=w^-\delta_{z^0-m^-}+w^0\delta_{z^0}+w^+\delta_{z^0+m^+}.
\]

CRPS 改善意味着 distribution 好。

但动作层把：

\[
-m^-,
\qquad
+m^+
\]

当成完整可执行 correction dose。

“outer atom 对 CRPS 有帮助”与“完整移动到 outer atom 能改善点预测”不是同一个命题。

所以当前失败可能来自四层：

1. candidate atoms 是好分布，但 full dose 不适合作为点动作；
2. W1 retrieval 找不到 action-value 相似的历史日；
3. replay gain 有信息，但 double-event proposal 丢失信息；
4. A_hat 有排序能力，但固定 split-conformal q 在时间漂移下失效。

必须按因果顺序诊断。

---

# 6. R1A.5：先诊断，不重新训练 candidate

R1A.5 使用已经完成的 R1A artifacts。

它是 retrospective development diagnostic。

一旦利用 R1A S4 来决定后续架构，R1A S4 就不能再被当成最终确认集。

---

# 7. D1 — Candidate Actionability Oracle

目标：

> 当前 m-/m+ 本身是否包含可执行价值？

对每个 S4 day，结果出现后离线计算真实逐时方向收益：

\[
g^{true}_{h,\downarrow}
=
|r_h|-|r_h+m^-_h|,
\]

\[
g^{true}_{h,\uparrow}
=
|r_h|-|r_h-m^+_h|,
\]

其中：

\[
r_h=z^Y_h-z^0_h.
\]

把真实 gain 输入同一个 double-event optimizer，得到：

\[
\pi^{oracle},
\qquad
A^{oracle}.
\]

报告：

- oracle positive-day rate；
- mean/median/p10/p90 `A_oracle`；
- Identity oracle-optimal 比例；
- oracle raw-MAE improvement；
- Down/Up event frequency。

### 判定

若多数日：

\[
A^{oracle}\le0,
\]

问题在 candidate→action interface，先不要改 DVG。

若 oracle 明显正，但当前动作亏，说明瓶颈在后面。

---

# 8. D2 — Retrieval / A_hat 质量

每个 S3M-val、S3C、S4 日记录：

\[
(\widehat A,A^{true},E=\widehat A-A^{true}).
\]

报告：

- Pearson corr；
- Spearman corr；
- MAE(A_hat, A_true)；
- bias；
- \(P(A>0\mid \widehat A>0)\)；
- \(P(A>0\mid \widehat A \text{ top 10\%})\)；
- A_true by A_hat decile。

重点不是 A_hat 平均误差，而是它是否有**动作排序能力**。

如果：

\[
\operatorname{Spearman}(\widehat A,A)\approx0,
\]

则 adaptive conformal 只能变得更保守，不能真正修复动作选择。

---

# 9. D3 — Proposal Efficiency

利用 D1 的 oracle proposal，对比当前 proposal。

定义：

\[
\eta_{\rm proposal}
=
\frac{\max(A^{proposal},0)}
{\max(A^{oracle},\epsilon)}.
\]

同时报告：

- Down IoU；
- Up IoU；
- direction mismatch；
- missed-positive-event rate。

若 hourly evidence 好而 proposal efficiency 差，瓶颈就是 event proposal。

---

# 10. D4 — Neighbor Evidence Diagnostic

先不替换 W1，只做冻结候选上的诊断。

在 S3M forward-validation 比较：

### Neighbor selection

1. W1 nearest-k；
2. recent-k；
3. random-k。

### Aggregation

1. mean；
2. median；
3. trimmed mean；
4. distance-weighted mean。

一个参数较少的 diagnostic weighting：

\[
\omega_j
\propto
\exp\left(
-\frac{D_j}{\operatorname{median}(D_{N_k})+\epsilon}
\right).
\]

报告：

- MAE(A_hat,A)；
- Spearman；
- positive-action precision；
- proposal efficiency。

判定：

```text
W1 ≈ random   -> retrieval geometry 不具 action information
recent > W1  -> temporal drift 比 geometry 更重要
robust/weighted > mean -> heavy-tail aggregation 是问题
```

NEM 尤其重点看。

---

# 11. D5 — DVG Calibration Drift

对：

\[
E_t=\widehat A_t-A_t
\]

按时间比较：

```text
S3M-validation
S3C
S4 quarter-1
S4 quarter-2
S4 quarter-3
S4 quarter-4
```

报告：

- median E；
- q90 / q95；
- max；
- S3C vs each S4 block 的 Wasserstein / KS diagnostic；
- 每个 block 的固定-q coverage。

绘制：

\[
E_t
\]

和固定 q 随时间的关系。

如果：

\[
q_{0.9}^{S4}
>
q_{0.9}^{S3C}
\]

且 A_hat 排序仍有信息，则主问题是 calibration drift。

---

# 12. D6 — Prequential calibration baselines

仅作为诊断，不立即替代 DVG。

每天结果出现后，只更新下一天的 calibration state。

比较：

### D6-A
当前 frozen split q。

### D6-B
Rolling empirical q：

\[
W\in\{30,60,90\}.
\]

### D6-C
Adaptive Conformal Inference (ACI)。

### D6-D
Conformal PID。

比较：

- coverage；
- release rate；
- harmful release rate；
- mean realized gain | execute。

不要在 R1A S4 选择最佳方法后，再把同一 S4 当 final evidence。

---

# 13. 文献依据

建议重点参考：

1. **Gibbs & Candès, NeurIPS 2021 — Adaptive Conformal Inference Under Distribution Shift**  
   面向未知 distribution shift 的在线 conformal adaptation。

2. **Angelopoulos, Candès & Tibshirani, NeurIPS 2023 — Conformal PID Control for Time Series Prediction**  
   针对 seasonality / trend / systematic shift 的时间序列校准。

3. **Zaffran et al. — Adaptive Conformal Predictions for Time Series**  
   研究 ACI/AgACI，并包含 day-ahead electricity price forecasting。

4. **Tibshirani et al., NeurIPS 2019 — Conformal Prediction Under Covariate Shift**  
   calibration/target covariate distribution 不同时的 weighted conformal 路线。

5. **Jin et al., 2026 preprint — Retrieval-Corrected Conformal Prediction for Time Series**  
   很新的预印本，只作为启发：local retrieved residual evidence 与 scalar conformal correction 可以结合，和 HCH 当前的“local A_hat + global q”问题高度相关。

这些文献说明 adaptive/local calibration 值得测试，但不能替代 R1A.5 的因果诊断。

---

# 14. Data Signature 下一步

当前证据更支持：

\[
\boxed{
\text{stable invariant geometry}
+
\text{learned daily adaptation}
}
\]

暂定保留：

- asinh geometry；
- local continuous rank；
- scale-free history；
- learned daily signature；
- identity-init FiLM。

暂时降级为 ablation / audit metadata：

- 8-dim frozen S1R descriptor。

不要立即删代码，等 PlainCore / LearnedSig / Learned+DetSig 完整消融后再裁决。

---

# 15. 修订后的 R1B

## Hosts

加入：

- LSTM；
- PatchTST。

继续：

- Linear；
- MLP。

TCN 继续不进第一版。

## Sources

继续：

- LAGO_DE；
- LAGO_PJM；
- NEM_SA1。

## Main frozen holdout

- NORD_DK1。

DK1 candidate 不接受任何 universal gradient。

## Candidate variants

### C0 PlainCore
无 DataSignature FiLM。

### C1 LearnedSig
learned daily signature + FiLM，det=0。

当前 R1A NoSig 就属于此类，暂定主模型。

### C2 Learned+DetSig
完整 signature；作为消融。

### C3 Local-Core
每域独立训练，用于回答 universal sharing 是否牺牲 local fit。

## Seeds

正式比较阶段：

\[
\{0,1,2\}.
\]

Host cache 第一版固定 host seed，隔离 HCH 随机性。

---

# 16. R1B 与 DVG 的关系

R1B candidate training 可以在 D1-D5 完成后启动。

它不必等待“新 DVG”完全定稿，因为：

\[
\text{candidate transfer}
\]

和：

\[
\text{safe action routing}
\]

是两个需要分开证伪的问题。

但完整 HCH S4 动作表在 DVG 问题解决前不能视为成熟主结果。

---

# 17. 服务器 U0 什么时候开始

暂时不要因为 R1A source candidate GREEN 就直接大规模蒸馏。

保持科学顺序：

\[
\boxed{
Native source success
\rightarrow
Native unseen-market/host transfer
\rightarrow
Foundation representation enhancement
}
\]

先让 R1B 的 DK1 + unseen-host 告诉我们 native prior 是否已经能迁移。

### 如果 native transfer 强

U0 是 enhancement：

> foundation prior 能否进一步提高已经有效的 universal corrector？

### 如果 native transfer 弱

U0 是 targeted repair hypothesis：

> distilled temporal representation 能否修复跨域动态信息不足？

这样 MOMENT 蒸馏的论文解释才干净。

---

# 18. R1A.5 决策树

```text
D1 oracle weak
    ↓
CANDIDATE_ACTIONABILITY
    ↓
检查 full atom dose 与 distribution-aware dose
    ↓
不要先改 conformal

D1 oracle strong
    ↓
D2 A_hat ranking weak
    ↓
RETRIEVAL_VALUE_ESTIMATION
    ↓
检查 W1 / recency / weighting / robust aggregation

D2 strong, D3 weak
    ↓
EVENT_PROPOSAL

D2/D3 strong, D5 coverage drift
    ↓
DVG_CALIBRATION_DRIFT
    ↓
rolling / ACI / PID / localized calibration
```

---

# 19. R1A.5 必需产物

```text
R1A_DIAG_<timestamp>/
├── code_commit.txt
├── source_r1a_run.txt
├── oracle_actionability_by_domain.csv
├── ahat_vs_atrue.csv
├── ahat_deciles.csv
├── proposal_efficiency.csv
├── neighbor_ablation.csv
├── calibration_error_timeseries.csv
├── calibration_drift_by_block.csv
├── prequential_calibration_baselines.csv
├── figures/
│   ├── ahat_vs_atrue_*.png
│   ├── error_vs_time_*.png
│   ├── calibration_q_drift_*.png
│   └── oracle_vs_actual_gain_*.png
└── DIAG_VERDICT.md
```

`DIAG_VERDICT.md` 最终只能选择：

```text
CANDIDATE_ACTIONABILITY
RETRIEVAL_VALUE_ESTIMATION
EVENT_PROPOSAL
DVG_CALIBRATION_DRIFT
MIXED
```

并用量化证据解释。

---

# 20. 当前项目状态

现在可以正式写：

\[
\boxed{
\texttt{UNIVERSAL\_CANDIDATE\_SUPPORTED}
}
\]

暂时不能写：

```text
SAFE_ACTION_GATE_CONFIRMED
FULL_HCH_EFFECTIVENESS_CONFIRMED
CROSS_MARKET_FROZEN_TRANSFER_CONFIRMED
FOUNDATION_DISTILLATION_NEEDED
```

下一步不是继续凭直觉加模块，而是：

\[
\boxed{
\text{先把 action chain 的瓶颈定位清楚，再进入 native transfer}
}
\]
