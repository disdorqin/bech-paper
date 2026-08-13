# HCH-v2 R1A.11 Prequential Cross-Fitted Calibration Authorization v0.1

**日期：2026-08-13**  
**仓库基线：** `main@9f31ed241358ce07fa18523edd77533334c7b388`

前置结论：

\[
\boxed{\texttt{UNIVERSAL\_CANDIDATE\_SUPPORTED}}
\]

\[
\boxed{\texttt{R1A.9 = LOCAL\_ACTION\_CALIBRATION\_SUPPORTED}}
\]

\[
\boxed{\texttt{R1A.10 = CALIBRATOR\_ROUTING\_UNRESOLVED}}
\]

本阶段继续暂停 R1B，不增加 Host/Market、不启动 U0、不重新训练 IAH、不修改 C3 isotonic 本身。

本阶段唯一改变：

> **把 R1A.10 的单次 `S3M-prefix fit -> S3M-suffix validate`，改成严格 rolling-origin / prequential cross-fitting，从同一 S3M 中取得更多真正 out-of-sample 的 calibrator-selection evidence。**

---

## 0. 为什么现在不应该直接上 regularized / shrunk isotonic

R1A.10 的 PJM:MLP 结果：

- C0 suffix net < 0；
- C0 两次 fire 全 harmful；
- C3 suffix 为纯 abstention，net=0；
- paired mean \(\Delta>0\)；
- 81.5% bootstrap resamples 为正；
- 但 one-sided 90% LCB 恰好为 0，因此 Gate D 保守回退 C0。

同时 support-size stress 已显示：

\[
W\in\{7,14,30,60\}
\]

时 PJM:MLP 的 C3 都稳定地给出 abstention。

所以当前直接证据更像：

\[
\boxed{\text{calibrator map 已相对稳定，但固定 suffix 中 rare action events 太少}}
\]

而不是：

\[
\text{C3 因高方差导致 selector CI 失败}.
\]

regularizing C3 并不能创造新的 independent OOS action events，也不保证能把大量为 0 的 paired \(\Delta_t\) 的 LCB 从 0 推成正数。

因此先提高**时间序列 OOS evidence efficiency**，再决定是否需要 regularization。

---

# 1. Code audit of R1A.10

R1A.10 实现与计划主体一致：

- selector 只读取 S3M-suffix；
- S3M-prefix 只 fit C3；
- S3C 只拟合 selected estimator 的 DVG q；
- S4 不参与选择；
- 无 market/host hard-code；
- 无 learned router；
- Gate A/B/C/D 按冻结规则执行；
- NEM 因 evidence 不足被正确保留 C0。

因此 R1A.10 的 YELLOW 是有效结果，不需要重跑。

## 1.1 Minor reproducibility issue to fix

当前 moving-block bootstrap 使用 module-level global RNG，然后各 domain 顺序消费同一 RNG state。固定 domain order 下可复现，但 formal audit 更希望：

\[
\boxed{\text{domain-specific deterministic bootstrap seed}}
\]

例如：

```text
seed_g = hash(global_seed, domain, stage)
```

同一个 domain 无论执行顺序如何，都应得到相同 bootstrap 结果。R1A.11 必须修正。

---

# 2. Why rolling-origin evidence is the correct next move

R1A.10 目前：

```text
S3M
├── prefix: fit C3
└── suffix: evaluate once
```

对 rare-fire domain，suffix 可能只有 1–2 个 action events。

R1A.11 改成：

```text
S3M ordered days

first B days   -> fit C3_0
next H days    -> OOS evaluate C0 vs C3_0
expand history -> fit C3_1
next H days    -> OOS evaluate
...
```

每个被评价日都满足：

\[
Y_t\notin\text{calibrator fit set used for day }t.
\]

但相比一次 suffix split，可以使用 S3M 中更多日期形成 genuine forward OOS decisions。

这是 rolling-origin / prequential evaluation，不是数据泄漏。

---

# 3. R1A.11 fixed protocol

## 3.1 Initial burn-in

固定：

\[
\boxed{B=30\text{ days}}
\]

如果：

\[
|S3M|<B+14,
\]

则：

```text
INSUFFICIENT_PREQUENTIAL_EVIDENCE
-> C0
```

因此当前短 NEM 仍然不会获得 C3 权限。

## 3.2 Evaluation block

固定：

\[
\boxed{H=7\text{ days}}
\]

rolling procedure：

1. 用所有 block 开始前的 S3M days 拟合 C3；
2. 冻结 C3；
3. 在下一 7 天分别计算 C0/C3 provisional actions；
4. 记录 paired day value；
5. block outcome 全部可用后，历史扩展；
6. 进入下一 block。

最后不足 7 天的 remainder 可以作为最后一个短 block，但必须显式记录长度。

不在 block 内 day-by-day 更新 C3，避免把不同更新频率混入本轮实验。

---

# 4. Prequential OOS evidence table

对每个 OOS day \(t\)：

\[
V_t^{C0}=A_t^{true,C0}\mathbf1[\widehat A_t^{C0}>0]
\]

\[
V_t^{C3}=A_t^{true,C3}\mathbf1[\widehat A_t^{C3}>0]
\]

\[
\boxed{\Delta_t=V_t^{C3}-V_t^{C0}}
\]

保存：

- fit-end date；
- evaluation block id；
- fit-day count；
- C0/C3 A_hat；
- C0/C3 execute；
- C0/C3 A_true；
- paired delta；
- raw fire/harm status；
- C3 map diagnostics。

任何 selector 统计都只能来自这些 prequential OOS rows。

---

# 5. Selector v2

仍保持 default-to-C0，不训练 router。

## Gate A — Prequential support

要求：

```text
initial_fit_days >= 30
n_oos_eval_days >= 14
```

否则 C0。

## Gate B — Raw policy problem

在全部 prequential OOS rows 上：

```text
net_C0 <= 0
OR
harmful_C0 >= 0.50
```

否则：

```text
RAW_HEALTHY_KEEP_C0
```

## Gate C — C3 actual decision improvement

要求：

\[
net_{C3}>net_{C0}
\]

并：

\[
harm_{C3}\le harm_{C0}.
\]

## Gate D — Paired uncertainty

对完整 prequential OOS \(\Delta_t\) 做 moving-block bootstrap。

固定：

- block length = 7 days；
- 1000 resamples；
- one-sided 90% lower bound；
- domain-specific deterministic RNG seed。

仍使用同一预注册门槛：

\[
\boxed{LCB_{0.90}(E[\Delta])>0}.
\]

**不要因为 R1A.10 刚好 LCB=0 而修改 `>0` 为 `>=0`。**

---

# 6. Final calibrator fit after selection

若 selector 授权 C3：

1. 用**完整 S3M** 重新拟合 final local isotonic；
2. 冻结；
3. S3C 用 final C3 产生自己的 \(A_{hat}\) 与 calibration errors；
4. 拟合自己的 DVG q；
5. S4 development confirmation。

若 selector 选择 C0：S3C 只校准 C0 DVG。

---

# 7. Compare three routing protocols

### R0 — C0 everywhere
baseline。

### R10 — R1A.10 fixed-prefix selector
已完成 selector v1。

### R11 — Prequential cross-fitted selector
本轮 proposed selector v2。

重点不是让 R11 用更多自由度“赢”，而是检验：

> rolling-origin OOS evidence 是否足以把 sparse-evidence ambiguity 消掉？

---

# 8. 本轮禁止事项

禁止：

- regularized isotonic；
- shrunk isotonic；
- veto-only new calibrator；
- C3 hyperparameter tuning；
- change B=30 after result；
- change H=7 after result；
- change bootstrap alpha；
- learned router；
- use S3C/S4 in selection；
- add market/host；
- modify IAH / CRPS / DVG math。

本轮只改变：

\[
\boxed{\text{evidence extraction protocol}}
\]

---

# 9. Key diagnostics

每 domain 报告：

- total S3M days；
- initial burn-in days；
- number of OOS blocks；
- OOS evaluated days；
- raw C0 fire days；
- C3 fire days；
- C0 harm；
- C3 harm；
- C0 net；
- C3 net；
- mean paired delta；
- LCB90；
- fraction bootstrap > 0；
- selected calibrator；
- reason code。

同时画：

\[
\text{cumulative }\sum_{\tau\le t}\Delta_\tau
\]

随 OOS time 的曲线。

---

# 10. Map-stability diagnostic

rolling fit 过程中保存每个 C3 map：

- plateau count；
- map min/max；
- max jump；
- prediction on fixed grid \(s\in[-1,1]\)。

计算相邻 rolling fits：

\[
D_{map}(k,k+1)=\frac1M\sum_m|f_k(s_m)-f_{k+1}(s_m)|.
\]

解释：

- map 稳定但 CI 仍过不了 → rare-event evidence 不足，regularization 不是解；
- map 本身剧烈漂移 → 才支持下一步 regularized/shrunk isotonic。

---

# 11. Success criteria

## GREEN

R11 在完全 pre-S4 条件下：

1. NEM 继续因 insufficient evidence 保留 C0；
2. healthy domains 不被误校准；
3. PJM:MLP 获得 C3 授权，且由 rolling OOS evidence 支撑；
4. LCB90 > 0，而不是放宽门槛；
5. S4 development 上 PJM harmful rare-fire 明显减少；
6. macro / worst-domain net value 不低于 R0；
7. NEM C0 value retention 接近 1；
8. map stability 不显示严重滚动漂移。

Verdict：

```text
PREQUENTIAL_CALIBRATION_ROUTING_SUPPORTED
```

此时 R1A 系列可以结束，授权 R1B。

## YELLOW-A — Map stable, evidence still sparse

```text
RARE_EVENT_EVIDENCE_LIMITED
```

此时不要上 regularized isotonic。下一步讨论：

- longer local adaptation horizon；
- few-shot 30/60/90-day protocol；
- sequential evidence accumulation after deployment。

## YELLOW-B — Map unstable

```text
LOCAL_CALIBRATOR_VARIANCE_LIMITED
```

这时才授权：

- identity-anchored shrinkage；
- regularized isotonic；
- constrained monotone calibration。

## RED

```text
LOCAL_ISOTONIC_UPPER_BOUND_NOT_ROBUST
```

若更多 OOS evidence 表明 C3 对 PJM 没有稳定优势，则撤回 R1A.9 的 deployment interpretation。

---

# 12. Literature rationale

Rolling-origin evaluation 是时间序列 forecast evaluation 的经典方法。Tashman (2000) 系统讨论 fixed vs rolling origin、model recalibration 与 multiple test periods，并指出 rolling-origin 可以提高单序列 out-of-sample evaluation 的效率与可靠性。

这正对应 R1A.10 的问题：我们不是想增加训练数据，而是在严格 chronology 下获得更多 OOS action decisions。

另一方面，Berta, Bach & Jordan (AISTATS 2024) 的 ROC-regularized isotonic calibration 说明：当 calibration map 本身有过拟合/高方差证据时，约束 isotonic 是合理方向。但当前 PJM support-size stress 先显示 map 相对稳定，所以 regularization 应被保留到“rolling map instability 被实验证实”之后。

---

# 13. R1B authorization

R1B 继续暂停。

只有：

```text
PREQUENTIAL_CALIBRATION_ROUTING_SUPPORTED
```

才正式租服务器并进入：

- Linear / MLP / LSTM / PatchTST；
- NORD_DK1；
- unseen host / market；
- candidate signature ablations；
- local calibration eligibility generalization。

---

# 14. Server plan

R1A.11 本地完成。

若 GREEN：

```text
智川云：海南 RTX 4090 24GB / 100GB RAM
数据盘 total = 200GB
```

再启动 R1B。U0 时按需扩至 500GB。

---

# 15. Required implementation

新增：

```text
experiments/08-hch-v2/r1a11_prequential_calibration_router.py
```

复用 R1A.9 C0/C3，不改 production modules。

建议：

```text
PrequentialCalibrationEvaluator
├── build_rolling_folds()
├── fit_c3(history)
├── evaluate_next_block()
├── collect_paired_oos_rows()
├── map_stability()
└── select()
```

bootstrap 必须使用 per-domain deterministic seed。

---

# 16. Required artifacts

```text
R1A11_PREQ_<timestamp>/
├── prequential_config.json
├── rolling_folds.csv
├── paired_oos_value.csv
├── oos_summary_by_domain.csv
├── bootstrap_by_domain.csv
├── map_stability.csv
├── selected_calibrator_by_domain.csv
├── s3c_dvg.csv
├── s4_policy_metrics.csv
├── s4_point_metrics.csv
├── figures/
│   ├── cumulative_delta_*.png
│   └── rolling_map_*.png
└── PREQUENTIAL_VERDICT.md
```

---

# 17. Current project state

R1A.10 并没有证明 router 思路错了。

它证明的是：

\[
\boxed{\text{一次 fixed suffix 对 rare action event 的证据效率可能不够}}
\]

下一步先换一个更适合时间序列的方法选择协议，而不是因为一个 bootstrap LCB 恰好为 0 就修改 calibrator。

核心原则：

\[
\boxed{\text{先增加真正 OOS 的证据利用率，不增加模型复杂度。}}
\]
