# HCH-v2 R1A.10 Evidence-Gated Local Action Calibration v0.1

**日期：2026-08-13**  
**仓库基线：** `main@adb8ec6c44867114cb25c7d0a162f7408bc16513`

前置状态：

\[
oxed{	exttt{UNIVERSAL\_CANDIDATE\_SUPPORTED}}
\]

\[
oxed{	exttt{R1A.9 = LOCAL\_ACTION\_CALIBRATION\_SUPPORTED}}
\]

本阶段继续暂停 R1B，不增加市场、不增加 Host、不启动 U0、不重新训练 IAH candidate。

本阶段唯一目标：

> 把 R1A.9 中“local isotonic 可以修某些 domain”这个 upper-bound 结果，变成一个在 S4 之前就能决定是否启用 local calibration 的可部署协议。

---

## 0. 为什么 R1A.9 还不能直接进入 R1B

R1A.9 已证明：

1. shared affine calibration 基本等于 identity，不存在统一 universal action-calibration distortion；
2. local isotonic C3 可以把 PJM:MLP harmful rare-fire 压成安全 Identity；
3. 但同一个 C3 若无条件套到 NEM，会严重损失原本真实存在的 action value。

因此正确结构不是：

```text
所有 domain 都套 C3
```

而更像：

```text
默认 Raw IAH (C0)
        ↓
只有 local evidence 足够且明确证明 C0 action-threshold 失准
        ↓
才启用 Local Isotonic (C3)
```

目前“PJM 用 C3、NEM 用 C0”仍是看过 development S4 后得到的科学发现，不能直接硬编码进方法，否则会变成 hindsight routing。

所以 R1A.10 必须建立：

\[
oxed{	ext{pre-S4 calibrator eligibility gate}}
\]

---

## 1. 架构原则

Universal candidate 继续冻结：

\[
	heta_{IAH}=	ext{frozen}.
\]

Local layer 允许：

\[
\mathcal C_g\in\{	ext{Identity},	ext{Local Isotonic}\}.
\]

但：

\[
oxed{	ext{C0 / Identity 是默认值}}
\]

只有当本地历史 evidence 足以证明 raw action translation 有系统性风险，并且 C3 在 forward validation 中改善该风险，才允许启用 C3。

因此 local calibration 是：

> evidence-authorized safety correction

而不是默认必装模块。

---

## 2. R1A.10 只保留 C0 与 C3

R1A.9 discovery 已完成：

- C1 shared affine：证伪，≈ identity；
- C2 local affine：有解释性，但不是当前最强 local upper bound；
- C3 local isotonic：当前最强 local calibration upper bound。

所以本轮固定：

### C0
Raw IAH action utility。

### C3
Local monotone isotonic utility calibration。

不增加第三种 calibrator，不做 AutoML selector。

---

## 3. Chronology：fit / select / calibrate / test 四段分离

```text
S3M-prefix
    ↓
fit C3 only

S3M-suffix
    ↓
select C0 vs C3

S3C
    ↓
selected calibrator frozen
fit its own DVG q

S4
    ↓
development confirmation only
```

禁止：

- 用 S3C 选择 C0/C3；
- 用 S4 选择 C0/C3；
- 看 S4 后改 eligibility rule；
- market name / host name 作为 selector feature。

---

## 4. Selection 必须是 decision-level，而不是只看 ECE/Brier

S3M-suffix 上直接比较真实 day-level action utility。

对 calibrator \(c\in\{C0,C3\}\)，定义无 DVG provisional policy：

\[
execute_t^c=\mathbf1[\widehat A_t^c>0].
\]

日价值：

\[
V_t^c=A_t^{true,c} execute_t^c.
\]

abstain day 记：

\[
V_t^c=0.
\]

paired difference：

\[
oxed{\Delta_t=V_t^{C3}-V_t^{C0}}.
\]

同一天比较，因此是 paired design。

---

## 5. Evidence Gate v1

本轮不学习 router，使用透明 deterministic gate。

### Gate A — Support sufficiency

第一版 engineering floor：

```text
n_fit_days >= 30
n_val_days >= 10
```

不足则：

```text
CALIBRATION_EVIDENCE_INSUFFICIENT
-> choose C0
```

这不是理论常数，后续在 R1B 继续验证。

它的主要作用是阻止类似 NEM 只有约 14 个 fit days 时直接部署高自由度 isotonic map。

### Gate B — Raw C0 必须先显示真实问题

S3M-suffix 上 C0 至少满足一个：

```text
val_net_C0 <= 0
OR
val_harmful_C0 >= 0.50
```

如果 raw C0 已经是正价值且 harm 可接受：

\[
oxed{	ext{do not calibrate a working policy}}
\]

直接保留 C0。

### Gate C — C3 必须改善真实 decision value

要求：

\[
val\_net_{C3}>val\_net_{C0}
\]

并且：

\[
val\_harmful_{C3}\le val\_harmful_{C0}.
\]

如果 C3 只是减少 release，但并没有改善 day-level net value，不授权。

### Gate D — 改善不能来自一两天偶然

对 \(\Delta_t\) 做 moving-block bootstrap：

- 7-day block；
- 1000 resamples；
- 若 n_val_days < 14，不做强证据声明，保持 C0；
- 报告 mean delta 和 one-sided 90% lower CI。

授权 C3：

\[
oxed{LCB_{0.90}(E[\Delta])>0}.
\]

如果 CI 跨 0：

```text
EVIDENCE_AMBIGUOUS
-> C0
```

这是保守 router：宁可暂时不校准，也不因少量 local evidence 破坏一个本来有效的 domain。

---

## 6. 为什么 default-to-C0 很重要

R1A.9 已显示：

- NEM:MLP 的 C0 action value 很强；
- C3 因 local evidence 少而把收益大幅压掉；
- PJM:MLP 的 C0 harmful，C3 abstention 有价值。

所以 local calibration 的风险不是单向的。

结构应该是：

\[
oxed{	ext{Raw universal decision prior}+	ext{optional local safety override}}
\]

而不是：

\[
	ext{every market must locally recalibrate}.
\]

---

## 7. Calibration support audit

每个 domain 记录：

- n_fit_days；
- n_val_days；
- n_fit_hours；
- raw C0 fire days；
- raw C0 fire hours；
- Down/Up 各自 \(w>0.5\) crossing day count；
- observed benefit positive/negative support；
- isotonic plateau 数；
- mapping range；
- maximum jump；
- S3M-prefix → suffix action-benefit rate drift。

关键：

> 小时观测数不能被当成独立 calibration sample 数；部署证据以天级历史和 action-support days 为主。

---

## 8. Optional support-size stress test

仅 diagnostic，不用于当前 selector tuning。

对历史较长的 LAGO domains 截取：

\[
W\in\{7,14,30,60\}	ext{ days}
\]

拟合 C3，保持之后 validation window 固定，记录：

- mapping stability；
- plateau 数；
- validation net value；
- harmful rate。

目标：

> 判断 local isotonic 大约从多少“天级 evidence”开始不再容易坍缩成极端 map。

不要用 source S4 挑最优 W。

---

## 9. 三个实验版本

### E0 — C0 everywhere
Raw IAH baseline。

### E1 — C3 everywhere
R1A.9 local-isotonic upper bound / negative control。

### E2 — Evidence-Gated C0/C3
使用 §5 pre-S4 rule 自动选择。

E2 是本轮 proposed deployment protocol。

---

## 10. S3C 与 DVG

E2 每个 domain 在 S3M-suffix 结束后固定：

```text
selected_calibrator ∈ {C0, C3}
```

之后不再改变。

S3C 只负责 selected estimator 对应：

\[
E_t=\widehat A_t-A_t^{true}
\]

及：

\[
q_{1-lpha}.
\]

S3C 不参与 calibrator selection。

---

## 11. Primary metrics

### Router correctness

每 domain 输出：

```text
selected = C0 / C3
reason_code
support gate
raw-harm gate
value-improvement gate
bootstrap gate
```

reason codes 至少：

```text
RAW_HEALTHY_KEEP_C0
INSUFFICIENT_LOCAL_EVIDENCE_KEEP_C0
C3_VALUE_NOT_BETTER_KEEP_C0
C3_IMPROVEMENT_UNCERTAIN_KEEP_C0
LOCAL_MISCALIBRATION_C3_AUTHORIZED
```

### Decision value

- S4 net daily action value；
- gain | release；
- harmful release rate；
- release rate；
- worst-domain net value。

### Forecast

- MAE；
- rMAE；
- no-floor sMAPE；
- degradation vs host。

### Retention

对原 C0 actionable domain：

\[
Retention_g=rac{V_g^{E2}}{V_g^{C0}+\epsilon}.
\]

特别关注 NEM。

---

## 12. Success criteria

### GREEN

Evidence-gated E2 在完全不看 S4 的 selector 下：

1. 不给 evidence-insufficient NEM 错启 C3；
2. 至少对一个已知 harmful/miscalibrated domain 启用 C3；
3. macro net action value 不低于 C0；
4. worst-domain net value 不比 C0 明显恶化；
5. NEM 的 C0 action value大部分保留；
6. PJM:MLP harmful rare-fire 被明显削弱或安全 abstain；
7. selection reason 完全来自 pre-S4 evidence。

Verdict：

```text
EVIDENCE_GATED_LOCAL_CALIBRATION_SUPPORTED
```

### YELLOW

```text
CALIBRATOR_ROUTING_UNRESOLVED
```

说明 router 大方向合理，但 support / CI 仍不稳定。下一步才研究 regularized / shrunk isotonic。

### RED

```text
LOCAL_CALIBRATION_NOT_DEPLOYABLE
```

pre-S4 evidence 无法区分“该校准”和“不该校准”的 domain。此时 R1A.9 只能保留为 upper bound。

---

## 13. 本轮禁止事项

禁止：

- market/host hard-coded list；
- S4-based selection；
- learned router；
- meta selector；
- new neural action head；
- change IAH/CRPS；
- retrieval；
- adaptive DVG；
- new datasets；
- server rental。

---

## 14. Literature interpretation

R1A.9 的 NEM 现象提醒我们：isotonic calibration 在校准数据不足时可能出现高方差或过拟合式映射，因此 calibration 文献中也存在对 isotonic 做额外结构约束或 regularization 以控制过拟合的工作。

R1A.10 暂时不引入新 regularized isotonic。

先回答更简单的问题：

> local calibration 能否作为一个 default-off、evidence-authorized component？

如果能，方法更简单；如果不能，再引入 shrinkage/regularization 才有实验依据。

---

## 15. 若 GREEN，对架构的影响

后半链暂定：

```text
Universal IAH Candidate
        ↓
analytic raw action utility (C0)
        ↓
Local Calibration Eligibility Gate
        ├── insufficient / raw healthy → C0
        └── supported local miscalibration → Local Isotonic C3
        ↓
Double Event
        ↓
S3C DVG
```

Gate 不预测 action，只决定是否授权 local recalibration。

这与：

\[
oxed{	ext{Universal Correction Core}+	ext{Local Evidence Layer}}
\]

保持一致。

---

## 16. R1B authorization

只有 R1A.10 GREEN 后进入 R1B。

届时冻结：

- IAH candidate；
- C0/C3 family；
- evidence-gate rule；
- chronology；
- DVG procedure。

R1B 才验证：

- LSTM；
- PatchTST；
- NORD_DK1；
- unseen host；
- unseen market；
- selector 是否能在新 domain 上正确判断 calibration eligibility。

---

## 17. Server plan

R1A.10 本地运行。

若 GREEN：

```text
租：海南 RTX 4090 24GB / 100GB RAM
数据盘 total = 200GB
```

进入 R1B。

U0 时再扩到：

```text
500GB total
```

不提前扩更大。

---

## 18. Required implementation

新增：

```text
experiments/08-hch-v2/r1a10_calibrator_router.py
```

不要先修改 production HCH。

建议结构：

```text
LocalCalibrationSelector
├── collect_support_stats()
├── evaluate_raw_health()
├── paired_value_bootstrap()
├── select()
└── reason_code
```

---

## 19. Required artifacts

```text
R1A10_ROUTER_<timestamp>/
├── selector_config.json
├── support_by_domain.csv
├── val_policy_comparison.csv
├── paired_bootstrap.csv
├── selected_calibrator_by_domain.csv
├── s3c_dvg.csv
├── s4_policy_metrics.csv
├── s4_point_metrics.csv
├── support_size_stress.csv
├── figures/
└── ROUTER_VERDICT.md
```

---

## 20. 当前研究状态

R1A.9 已把问题从：

> distribution 到 action 是否需要重做？

缩到了：

> local action calibration 是有用的，但如何只在证据可靠时启用？

下一步不再优化 calibrator 本身，而是验证：

\[
oxed{	ext{HCH 能否知道什么时候应该相信 local calibration。}}
\]
