# HCH-v2 R1A.9 Action-Threshold / Utility Calibration Audit v0.1

**日期：2026-08-13**  
**仓库基线：** `main@32cca01ff8f57e79c03b30db9d0679f97b0d3931`  
**前置状态：** `UNIVERSAL_CANDIDATE_SUPPORTED`；`R1A.8 = ACTION_SIGNAL_UNRESOLVED`。

本阶段继续暂停 R1B，不增加市场、不增加 Host、不启动 U0、不重新训练 IAH candidate、不改变 IAH-CRPS。

本阶段只研究一个问题：

> 一个 CRPS 良好的三原子预测分布，能否通过极小、单调、action-relevant 的 post-hoc calibration，转化成可靠的动作效用估计？

---

## 0. 为什么下一步不是 action head

R1A.8 已经确认：PJM:MLP 的低 fire 是当前三原子分布在 absolute loss 下的 Bayes-median 决策，不是实现 bug；简单缩小 dose 不能改变 predicted expected-gain 的正负；冻结 representation 对 action benefit 只有灰区可预测性；IAH-native DVG 又不能自动把 harmful rare-fire 关掉。

因此目前最小假设不是“representation 不够强”，而是：

\[
\boxed{\text{distributional calibration}\neq\text{action-relevant calibration}}
\]

先验证 calibration mismatch；只有低容量单调 recalibration 也失败，才授权 richer action mapping。

---

## 1. 动作真正对应的 threshold event

记真实 transformed residual：

\[
r_h=z^Y_h-z^0_h.
\]

### Down

full-atom Down：

\[
a_h^-=-m^-_h.
\]

真实收益：

\[
g^-_h=|r_h|-|r_h+m^-_h|.
\]

当 \(m^-_h>0\) 时：

\[
\boxed{g^-_h>0\iff r_h<-m^-_h/2}
\]

定义：

\[
B^-_h=\mathbf1[r_h<-m^-_h/2].
\]

### Up

\[
a_h^+=+m^+_h,
\]

\[
\boxed{g^+_h>0\iff r_h>m^+_h/2}
\]

\[
B^+_h=\mathbf1[r_h>m^+_h/2].
\]

---

## 2. 为什么 w-/w+ 就是 IAH 隐含的动作阈值概率

IAH residual distribution：

\[
R_h\sim w^-_h\delta_{-m^-_h}+w^0_h\delta_0+w^+_h\delta_{m^+_h}.
\]

在该三点 support 下，Down-benefit threshold \(-m^-/2\) 左侧只有 atom \(-m^-\)，因此：

\[
\boxed{P_{IAH}(B^-_h=1)=w^-_h}
\]

同理：

\[
\boxed{P_{IAH}(B^+_h=1)=w^+_h}
\]

所以 R1A.9 直接审计：

\[
P(B^\pm=1\mid w^\pm=p)\stackrel{?}{\approx}p
\]

尤其关注真正触发动作的 \(p\approx0.5\) 区域。

---

## 3. B0 — Raw threshold-calibration audit

不拟合任何新组件。

对每个 domain、Down/Up 分别报告：

- reliability bins；
- Brier score；
- calibration intercept/slope；
- raw AUC/PR-AUC 仅作 discrimination diagnostic；
- \(w^\pm\in[0.40,0.60]\) 的细粒度 observed benefit frequency；
- \(w^\pm>0.5\) fire cases 的 empirical benefit rate；
- PJM:MLP、PJM:Linear、NEM:MLP、LAGO_DE:MLP 单独画图。

如果 PJM:MLP 在 raw \(w>0.5\) 时真实 benefit frequency 仍明显低于 0.5，则 rare harmful fire 可以直接归因于 action-threshold miscalibration。

---

## 4. Continuous utility calibration

仅校准 benefit probability 仍没有用到 gain magnitude，因此定义 normalized realized utility。

### Down

\[
Y^-_h=\frac{g^-_h}{m^-_h}\in[-1,1],\quad m^-_h>0.
\]

IAH 模型下：

\[
\boxed{E_{IAH}[Y^-_h]=2w^-_h-1}
\]

定义 raw utility score：

\[
s^-_h=2w^-_h-1.
\]

### Up

\[
Y^+_h=\frac{g^+_h}{m^+_h},\qquad s^+_h=2w^+_h-1.
\]

直接检查：

\[
\boxed{E[Y^\pm\mid s^\pm=s]\stackrel{?}{\approx}s}
\]

这比单独 binary AUC 更直接对应真实 action value。

---

## 5. Chronology

R1A source S4 已经是 development data。

本阶段严格使用：

```text
S3M-prefix  -> calibrator fit
S3M-suffix  -> calibrator selection / validation
S3C         -> calibrator frozen 后，单独拟合 DVG q
S4          -> development confirmation only
```

禁止 S4 fit、S4 hyperparameter selection，也禁止用 S3C 反向选择 calibrator。

---

## 6. Calibrator family：故意保持极小

### C0 — Raw IAH

\[
\tilde s^\pm=s^\pm.
\]

### C1 — Shared monotone affine utility calibration

Down/Up 各自拟合：

\[
\boxed{\tilde s^\pm=\operatorname{clip}(a_\pm s^\pm+b_\pm,-1,1)}
\]

约束：

\[
a_\pm\ge0.
\]

全模型只有四个 shared scalars：\(a_-,b_-,a_+,b_+\)。

source domains 必须 equal-weighted。

若 C1 成功，说明 action miscalibration 主要是 universal affine distortion，这是最理想的结果。

### C2 — Local monotone affine calibration

每个 `(market,host)` 自己拟合方向别 affine map。

只作为 diagnostic upper bound。

如果 C2 明显优于 C1，则 action calibration 是 domain-local 的，后续应归入 Local Evidence Layer，而不是伪装成 universal component。

### C3 — Local isotonic utility calibration

对 \(s^-\to Y^-\)、\(s^+\to Y^+\) 分别做 monotone isotonic regression。

仍然是低容量 upper bound。若 C3 也失败，则不再把问题称为“简单 calibration error”。

---

## 7. 从 calibrated utility 到动作

定义：

\[
\boxed{\tilde g^-_h=m^-_h\tilde s^-_h}
\]

\[
\boxed{\tilde g^+_h=m^+_h\tilde s^+_h}
\]

将 \(\tilde g^-\)、\(\tilde g^+\) 送入现有 double-event optimizer。

保持：

- at most one Down + one Up；
- contiguous event；
- 原 tie-break；
- final pi 语义不变。

whole-day value：

\[
\boxed{\widehat A_q^{cal}=\frac1{|H_q|}\sum_h\tilde g_h(\pi_{q,h}^{cal})}
\]

本阶段不重新引入 retrieval estimator。

---

## 8. 每个 calibrator 都必须重新拟合自己的 DVG

对于 C0/C1/C2/C3，分别计算 S3C error：

\[
E_t^{cal}=\widehat A_t^{cal}-A_t^{true}
\]

并得到自己的：

\[
q_{1-\alpha}^{cal}.
\]

执行：

\[
LCB_q^{cal}=\widehat A_q^{cal}-q^{cal}>0.
\]

禁止复用旧 W1 q 或其他 calibrator 的 q。

---

## 9. 评价指标：不再把 Spearman 当唯一主指标

strong-host domain 本来可能大量 Identity，因此本轮 primary 改成 selective decision value。

### 9.1 Net daily action value

abstain day 记 0：

\[
\boxed{V_g=\frac1{|S4_g|}\sum_t A_t^{true}\mathbf1[execute_t]}
\]

### 9.2 Safety

- release rate；
- harmful release rate；
- mean gain | release；
- worst-domain net value；
- DVG coverage。

### 9.3 Final forecast

- MAE；
- rMAE；
- no-floor sMAPE；
- degradation vs frozen host。

Spearman/AUC 继续保留为诊断，不再要求每个 low-actionability domain 都必须高。

---

## 10. PJM:MLP 的两种可接受成功方式

### Route A — Recalibrated action

若 calibration 后仍有非零 release，并且：

\[
E[A^{true}\mid release]>0
\]

且 net daily value > 0，则 action calibration 成功。

### Route B — Calibrated abstention

若 calibrator 将原 harmful fire 压回 \(\tilde g\le0\)，最终：

- release 很低；
- harmful actions 大幅减少；
- host MAE/rMAE 基本保持；

则同样判成功：

```text
CALIBRATED_SAFE_ABSTENTION
```

---

## 11. 不能为了修 PJM 杀死 NEM / DE

重点记录：

```text
NEM_SA1:MLP
NEM_SA1:Linear
LAGO_DE:MLP
```

相对 C0 的：

- release retention；
- net action value retention；
- final point metric retention。

如果 shared C1 把这些原本 actionable domains 全压成 Identity，说明 shared calibrator 不合适。

---

## 12. Decision tree

```text
raw threshold probability near 0.5 miscalibrated
        ↓
C1 shared affine works
        ↓
UNIVERSAL_ACTION_CALIBRATION_SUPPORTED

C1 fails
        ↓
C2/C3 local works
        ↓
LOCAL_ACTION_CALIBRATION_SUPPORTED

C1/C2/C3 all fail
        ↓
MONOTONE_CALIBRATION_INSUFFICIENT
        ↓
才授权 richer action mapping / richer information
```

---

## 13. Verdict labels

最终只能取：

```text
UNIVERSAL_ACTION_CALIBRATION_SUPPORTED
LOCAL_ACTION_CALIBRATION_SUPPORTED
ACTION_CALIBRATION_PARTIAL
MONOTONE_CALIBRATION_INSUFFICIENT
```

R1B 只有 GREEN-A 或 GREEN-B 且 protocol 冻结后才启动。

---

## 14. Literature anchors

本阶段主要受 decision-oriented calibration 启发，而不是再改 forecasting loss。

- Sahoo et al., NeurIPS 2021, **Reliable Decisions with Threshold Calibration**：普通平均 calibration 不保证 regression threshold decisions 的 loss 被正确预测；threshold-relevant calibration 更贴近 downstream decision。
- Zhao et al., NeurIPS 2021, **Calibrating Predictions to Decisions**：有限动作集可以直接研究 decision calibration，无需先达到完整 distribution calibration。
- Perez-Lebel et al., AISTATS 2025, **Decision from Suboptimal Classifiers**：recalibration 只能消除 calibration-induced regret；若主要问题是 grouping/discrimination，单纯 recalibration 不够。
- Rossellini et al., COLT 2025, **Can a calibration metric be both testable and actionable?**：cutoff-style calibration 与决策可操作性直接相关，并讨论 isotonic / Platt 等 post-hoc calibration。

这些文献只支撑“先试极小 post-hoc decision calibration”这一研究顺序，不证明 HCH 一定能被校准成功。

---

## 15. Required implementation

新增诊断脚本：

```text
experiments/08-hch-v2/r1a9_action_calibration.py
```

暂不修改 production HCH modules。

建议接口：

```text
ActionUtilityCalibrator
├── RawIAH
├── SharedAffine
├── LocalAffine
└── LocalIsotonic
```

全部使用 frozen R1A candidate。

---

## 16. Required artifacts

```text
R1A9_CAL_<timestamp>/
├── code_commit.txt
├── raw_threshold_calibration.csv
├── boundary_calibration.csv
├── normalized_utility_calibration.csv
├── calibrator_params.csv
├── calibration_validation.csv
├── s4_action_metrics.csv
├── dvg_metrics.csv
├── final_point_metrics.csv
├── figures/
│   ├── reliability_*.png
│   ├── utility_calibration_*.png
│   ├── boundary_zoom_*.png
│   └── net_value_*.png
└── CALIBRATION_VERDICT.md
```

---

## 17. Server / storage plan

服务器仍选择：

```text
RTX 4090 24GB（海南）
15 vCPU
100GB RAM
约 ¥1.28/h
```

R1A.9 继续本地即可。

### R1B 开始时

默认数据盘 50GB，建议把**总数据盘先扩到 200GB**。

也就是扩容 150GB，按 0.01 元/GB/日约：

\[
\boxed{1.5元/日}
\]

### U0 开始时

再扩到：

\[
\boxed{500GB\ total}
\]

相对默认 50GB，计费扩容量为 450GB，约：

\[
\boxed{4.5元/日}
\]

如果实际 HF cache / multi-teacher feature bank 超预期，再按真实磁盘占用扩到 750GB 或 1TB。由于平台不能缩容，不提前一次性上 1TB。

### 路径规划

系统盘 `/`：只放环境、repo、轻量日志。  
数据盘 `/root/rivermind-data/`：datasets、host cache、HF cache、teacher embeddings、active outputs。  
文件存储 `/root/rivermind-fs/`：长期备份 best checkpoints、manifests、重要结果。

建立实例后立即将：

```bash
HF_HOME=/root/rivermind-data/hf_cache
TRANSFORMERS_CACHE=/root/rivermind-data/hf_cache
TORCH_HOME=/root/rivermind-data/torch_cache
```

指向数据盘，避免 30GB 系统盘被 cache 填满。

---

## 18. Current research state

现在 HCH 已经不是在问“candidate 会不会预测”。

当前真正研究的是：

\[
\boxed{
\text{怎样让一个 proper-score 良好的 distribution
在有限动作集上给出可靠的 action utility}
}
\]

R1A.9 是当前最小、最直接、最可证伪的一刀。
