# HCH-v2 R1B Universal Generalization Battery & R1A Closeout v0.1

**日期：2026-08-13**  
**R1A 结束基线：** `main@89b5e27569555ed294ebb181fb95a191003eb5f1`  
**阶段状态：**

\[
\boxed{\texttt{R1A\_CANDIDATE\_AND\_LOCAL\_ACTION\_MECHANISM\_SUPPORTED}}
\]

R1B 的目的不是继续针对 LAGO_DE / LAGO_PJM / NEM_SA1 优化，而是第一次系统检验：

\[
\boxed{
\text{HCH 是否真的是跨市场、跨 Host、可扩展到不同数据形态/预测类别的 universal correction module}
}
\]

---

## 0. R1A closeout — what is actually supported

R1A 系列支持以下结论：

1. **Universal IAH candidate 可学习。**  
   同一候选参数在多个 market × host domain 上改善 candidate CRPS，且没有退化成 Identity distribution。

2. **历史 W1 retrieval 不是有效主 value estimator。**  
   它已被实验削弱/降级，不应再作为 HCH 主叙事核心。

3. **IAH distribution 本身含 action utility 信息。**

4. **distribution → action translation 存在 domain-local threshold miscalibration。**

5. **Local isotonic calibration 有时有价值，但必须 default-off。**

6. **prequential evidence 可以在不使用 S4 的条件下授权 local calibration。**

当前后半链为：

```text
Universal IAH Candidate
        ↓
analytic action utility
        ↓
Prequential Local Calibration Eligibility
        ├── evidence insufficient / C0 healthy → C0
        └── local miscalibration sufficiently supported → C3 isotonic
        ↓
Double Event
        ↓
S3C DVG
```

---

# 1. R1A.11 governance note

R1A.11 的核心 routing result 可以接受为 GREEN。

但必须保留一个审计脚注：

原计划成功条件写：

> `map stability 不显示严重滚动漂移`

并未预先显式限定为“仅最终被部署 C3 的 domain”。

结果出来后发现全局最大 map drift 来自最终被 selector 留在 C0 的 `LAGO_PJM:Linear`；团队随后采用“deployed C3 domains 的 map stability 才影响 C3 deployment”的解释。

这一解释在部署逻辑上合理：

\[
C3_g\text{ never deployed}\Rightarrow
\text{its instability cannot corrupt final policy}.
\]

但它属于**post-result diagnostic-scope clarification**。

因此内部状态写成：

```text
R1A.11 GREEN
with documented post-hoc map-stability scope clarification
```

而不要写成：

```text
all criteria were perfectly pre-registered without interpretation changes
```

R1B 开始后，所有 success/failure gate 必须重新冻结，避免再次发生语义漂移。

---

# 2. R1B philosophy — move away from local optimization

R1B 不再问：

> “还能不能把当前三个市场多提升一点？”

而问四个更高层问题。

## G1 — Cross-host

同一个 universal corrector 是否适配：

- Linear
- MLP
- LSTM
- PatchTST

尤其：

\[
\boxed{\text{未参与 candidate training 的 host family}}
\]

是否仍能被 correction。

## G2 — Cross-market

Candidate 在完全不接收 NORD_DK1 gradient 时是否仍有：

- candidate quality；
- sensible action utility；
- safe local adaptation。

## G3 — Domain adaptation

Learned Data Signature 是否真的比 PlainCore 更能支持 unseen domain，而不是只拟合 source markets。

## G4 — Local mechanism generalization

R1A.11 的 prequential C0/C3 eligibility gate 能否在**新 host / 新 market**上自己判断：

- 该保持 C0；
- 还是有资格 local calibrate；

而不是依赖手工 market list。

---

# 3. What R1B does NOT yet prove

即使 R1B GREEN，也不能立即声称：

- rich-feature universality；
- DA ↔ RT universality；
- arbitrary frequency universality；
- generic non-electricity universality。

这些属于后续不同泛化轴。

所以整个路线明确拆成：

```text
R1B = host + unseen-market generalization
R1C = feature-schema + target-category (DA/RT) generalization
U0  = large-scale temporal representation prior
U2  = optional rich-feature interface
Final = multi-market / public DA-RT / Shandong business validation
```

避免把所有问题塞进一次实验。

---

# 4. R1B source and holdout design

## Source markets

仍固定：

```text
LAGO_DE
LAGO_PJM
NEM_SA1
```

第一版不要再加 source market。

原因：R1B 首先检验“已有 universal prior 能否迁移”，而不是通过增加数据掩盖迁移不足。

## Primary unseen market

```text
NORD_DK1
```

Universal candidate 在 DK1：

\[
\boxed{\text{zero gradient update}}
\]

允许 target-local：

- H0 host training；
- S1R rank/profile；
- local prequential calibration evidence；
- S3C DVG；

但 candidate \(\theta\) 不更新。

正确名称：

> **frozen-corrector / zero-gradient market transfer**

不是整个 forecasting system 的 zero-shot。

---

# 5. Host matrix

R1B 使用：

```text
Linear
MLP
LSTM
PatchTST
```

TCN 继续不加入。

原因不是永久排除，而是当前四类已经覆盖：

- linear/tabular
- nonlinear/tabular
- recurrent temporal
- Transformer/patch temporal

对第一版 model-agnostic stress 足够。

---

# 6. Two-level execution to control compute

不要直接做所有组合 × 3 seeds。

## R1B-A — one-seed screening

Host seed = 0  
HCH seed = 0

运行：

```text
3 source markets × 4 hosts
+
NORD_DK1 × 4 hosts
```

首先发现结构性 failure。

只有通过 screening 的 candidate variants 进入 R1B-B。

## R1B-B — confirmatory seeds

对 survivor：

\[
HCH\ seed\in\{0,1,2\}.
\]

Host cache 先固定 seed=0，隔离 correction randomness。

后续正式 paper benchmark 再增加 host-seed uncertainty。

---

# 7. Candidate variants

R1A 已经证明：

> 旧 `NoSig` 实际是 LearnedSig。

所以 R1B 必须完成真正的 signature ablation。

## C0 — PlainCore

完全绕过 Data Signature FiLM：

\[
h'=h.
\]

## C1 — LearnedSig

保留：

- learned daily signature；
- identity-init FiLM；

deterministic S1R descriptor = 0。

这是当前 provisional main candidate。

## C2 — Learned + Deterministic Signature

完整当前 signature。

R1A 已显示 8-d deterministic descriptor 在 source domains 没额外收益，因此 C2 只作为消融，不作为默认主模型。

## C3 — Local-Core

每个 source domain 独立训练同结构 candidate。

作用：

> 判断 universal sharing 是否通过共享知识获益，还是仅仅牺牲 local fit。

在 DK1：

Local-Core 属于 target-trained/full-shot comparison，不属于 frozen-transfer method。

---

# 8. Cross-host experiment must include a genuinely unseen host

仅训练时看过四种 host，然后分别在四种 host 上测试，不能充分支持 model-agnostic。

所以至少做一条：

## LOHO-PatchTST

Universal candidate training domains：

```text
Linear + MLP + LSTM
```

完全删除：

```text
PatchTST
```

然后：

\[
\boxed{\text{evaluate frozen candidate on PatchTST}}
\]

包括 source markets 和 DK1。

如果这条失败：

> 不要用“训练时包含 PatchTST 的 full model”掩盖 model-agnostic claim。

如果资源允许，正式 paper 阶段再扩展 leave-one-host-family-out 全部四折。

---

# 9. Market generalization experiment

## M0 — source-joint

训练：

```text
DE + PJM + NEM
```

测试 source + DK1。

## M1 — leave-one-source-market-family-out diagnostic

为了避免只检验一个 DK1，建议 R1B-B 再加一个轻量 LO-market：

例如：

```text
train: PJM + NEM
test: LAGO_DE
```

或循环 source family。

它不需要所有 fold 都成为主表，但至少能验证：

> DK1 结果不是单一市场巧合。

---

# 10. Training objective remains unchanged

Universal candidate：

\[
\boxed{
L=L_{\rm IAH-CRPS}
}
\]

不要因为现在开始做 generalization 就加入：

- domain classification；
- adversarial domain loss；
- GroupDRO loss；
- tail oversampling；
- market-weighted auxiliary objective；
- transfer-specific regularizer。

当前 equal-domain sampler 已经负责：

\[
\frac1{|G|}
\sum_gE[L_g].
\]

R1B 先观察泛化是否自然成立。

只有明确出现 domain domination / negative transfer，才研究新的 optimization strategy。

---

# 11. R1A action layer is frozen as protocol, not as universal truth

R1B 使用：

```text
C0 raw analytic utility
+
prequential eligibility
+
optional local C3 isotonic
+
S3C DVG
```

但它在 R1B 的身份是：

\[
\boxed{\text{mechanism under generalization test}}
\]

而不是“已经证明到所有 domain”。

对每个新 domain 都记录：

- S3M length；
- prequential OOS length；
- eligibility decision；
- C0/C3 choice；
- reason；
- C3 map stability；
- DVG q；
- release/harm/net value。

最关键的新结果之一就是：

> selector 在 unseen host / market 上会不会做出合理选择。

---

# 12. R1B evaluation axes

不要再只生成一张 overall MAE 表。

## Axis A — Candidate quality

- IAH-CRPS；
- deterministic host transformed loss；
- \(\Delta CRPS\)；
- per market；
- per host；
- seen/unseen host；
- source/unseen market。

## Axis B — Final correction quality

- MAE；
- rMAE；
- RMSE；
- no-floor sMAPE。

## Axis C — Safety

- release rate；
- harmful release；
- net daily action value；
- mean gain | release；
- normal-period degradation；
- Identity rate。

## Axis D — Transfer

单独报告：

```text
seen market + seen host
seen market + unseen host
unseen market + seen host
unseen market + unseen host
```

这张 2×2 transfer matrix 是 R1B 最重要的表之一。

---

# 13. Generalization dashboard

每个 candidate variant 计算：

## Macro

\[
M_{\rm macro}
=
\frac1{|G|}\sum_gM_g.
\]

## Worst-domain

\[
M_{\rm worst}.
\]

## Leave-group

- unseen-market performance；
- unseen-host performance。

## Failure count

明确统计：

```text
# domains candidate worse than host
# domains final MAE worse than host
# harmful local calibration authorizations
# catastrophic (> declared budget) degradations
```

不要只报平均改善。

---

# 14. R1B decision logic

R1B 的 GREEN 不要求所有 domain 必须提升。

更合理的是同时满足：

### Universal candidate

- macro candidate quality 明显优于 host baseline；
- unseen DK1 不发生 candidate collapse；
- unseen PatchTST 不发生系统性 collapse；
- improvement 不由 NEM 单独拉动。

### Safety

对于无法改善的新 domain：

\[
\boxed{\text{应优先退化到 Identity，而不是产生显著 harm}}
\]

### Local calibration

eligibility gate：

- 不应无证据地到处授权 C3；
- 新 domain 上的授权必须有 prequential evidence；
- insufficient-history target 应 default C0。

### Universal vs local

Universal candidate 至少应在 transfer/few-data 条件下体现相对 Local-Core 的优势，否则“universal”价值不足。

---

# 15. Anti-local-optimum guard

从 R1B 开始增加一个固定表：

```text
GENERALIZATION_LEDGER.csv
```

每一项架构改动必须填写：

```text
change
source-domain effect
unseen-market effect
unseen-host effect
worst-domain effect
complexity added
accepted/rejected
```

规则：

> **一个改动如果只改善 source domains，但损害 unseen market/host，不进入 universal core。**

它最多可以成为：

- local adapter；
- optional branch；
- dataset-specific baseline。

这样防止我们被 DE/PJM/NEM 开发集反复调优到局部最优。

---

# 16. Relation to recent universal time-series work

当前训练路线与同行有几个重要呼应，但我们不直接照搬。

## Moirai

Moirai 的 unified training 需要处理：

- cross-frequency；
- arbitrary variate counts；
- heterogeneous distributions；

说明真正 universal time-series model 的难点本来就不只是“多喂几个数据集”。

对 HCH 的启发：

> 后续 feature/frequency heterogeneity 必须被明确当成单独泛化轴，而不能把 R1B market transfer 当成 universality 的全部。

## UniTime

UniTime 明确指出 cross-domain 训练面临：

- variable-count differences；
- domain distinguishability；
- different convergence speeds。

对 HCH 的启发：

- role-based interface 有必要；
- balanced domain sampling 有必要；
- learned data signature 有合理动机；
- 但不要立即引入 language/domain ID。

## Moirai-MoE

该工作指出人为按 frequency/domain 做硬分组可能不是最佳方式，数据内部甚至短窗口级分布也可能差异明显。

对 HCH 的启发：

> 我们最初从 `market_id/target_id` predictive embedding 退回 learned data signature 是合理方向。

但 R1B 不引入 MoE；先验证轻量 FiLM adaptation 是否足够。

## EPF transfer learning

已有电价研究表明，多市场 source pretraining + target fine-tuning 可以改善 DA electricity-price forecasting。

我们的差异是：

> HCH 不希望对 target candidate 做 full fine-tune；我们重点测试 frozen correction prior + tiny/local evidence adaptation。

这使 R1B 的 frozen DK1 transfer 很重要。

---

# 17. R1C — prepare now, do not execute yet

用户的真正目标还包括：

- 特征不同；
- 市场不同；
- RT / DA 不同。

R1B 只解决前两者中的一部分。

所以现在并行准备 R1C 数据，但不把它混进 R1B。

## R1C-A — public DA/RT target pair

优先寻找：

### NYISO

官方公开页面同时提供：

- Day-Ahead Market zonal LBMP；
- Real-Time Market zonal LBMP；
- historical archived CSV。

这是非常适合做：

\[
\boxed{\text{同一市场，DA vs RT target-category transfer}}
\]

的候选。

### ERCOT

官方也同时公开：

- DAM Settlement Point Prices；
- RTM Settlement Point Prices；
- historical hub/load-zone prices。

可作为第二候选。

## PJM caution

PJM Data Miner 可访问丰富公共数据，但官方当前条款对数据 redistribution 有明确限制。

因此如果论文需要公开可复现数据 artifact：

> 不要默认把 PJM Data Miner 原始数据重新发布到仓库。

优先 NYISO / ERCOT 作为论文可复现 DA/RT 对。

---

# 18. Parallel agent task — DA/RT data audit

R1B 服务器训练期间，可以让另一个 Agent 做只读数据准备：

输出：

```text
docs/paper_prep/v2_final_prep/public_da_rt_dataset_audit_v0.1.md
```

对：

```text
NYISO
ERCOT
PJM (reference only)
```

记录：

- official source；
- DA target；
- RT target；
- frequency；
- historical coverage；
- zones/hubs；
- publication timing；
- forecast cutoff implications；
- legal pre-outcome covariates；
- license / redistribution；
- download/API method；
- estimated storage；
- exact hourly aggregation needed；
- DST handling；
- negative-price prevalence；
- recommendation。

不接入训练代码，等人工审核。

---

# 19. U0 / distillation remains in roadmap

R1B 不启动 foundation distillation。

因为现在最重要的是知道：

\[
\boxed{\text{native HCH prior 自己到底能 transfer 多远}}
\]

若 R1B native transfer 强：

> U0 是 enhancement。

若 native transfer 在 unseen host/market 明显弱：

> U0 是针对表示能力不足的 repair hypothesis。

之后 U0 仍按：

```text
MOMENT representation teacher
        ↓
HistorySignatureEncoder
        ↓
U1 IAH-CRPS specialization
```

而不是把 TSFM forecast 直接蒸馏成 HCH forecast。

---

# 20. Server plan — now authorized

R1A.11 GREEN 后，可以正式租：

```text
智川云
RTX 4090 24GB
海南
15 vCPU
100GB RAM
¥1.28/h
```

数据盘：

\[
\boxed{200GB\ total}
\]

开始。

不要一开始 500GB。

目录：

```text
/root/rivermind-data/
├── datasets/
├── host_cache/
├── hf_cache/
├── torch_cache/
├── experiments/
└── checkpoints/
```

设置：

```bash
HF_HOME=/root/rivermind-data/hf_cache
TORCH_HOME=/root/rivermind-data/torch_cache
```

R1B 完成后若进入 U0，再在线扩到：

\[
500GB.
\]

---

# 21. R1B execution order

不要一次启动所有任务。

## Step 0 — server reproducibility

- clone exact SHA；
- environment lock；
- run unit/P0 tests；
- run one existing R1A bundle replay；
- compare hash / metrics。

通过后服务器才算可信。

## Step 1 — new host caches

只生成：

```text
LSTM
PatchTST
```

source markets + DK1。

Linear/MLP 若现有 cache provenance 完整可复制；否则统一重建。

## Step 2 — candidate ablation one seed

```text
PlainCore
LearnedSig
Learned+DetSig
```

source joint training。

先看 candidate CRPS，不急跑全部 action chain。

## Step 3 — choose candidate architecture

依据：

- macro；
- worst；
- DK1 frozen；
- PatchTST unseen-host；

而不是 source mean。

## Step 4 — Local-Core comparison

对 source domains 做 local upper bound。

## Step 5 — full action chain

只对 survivor universal architecture + necessary baselines：

- prequential C0/C3 selector；
- DVG；
- final point metrics。

## Step 6 — confirmatory seeds

survivor：

\[
seed=0,1,2.
\]

---

# 22. When to stop R1B

若发现：

### candidate source strong, DK1 weak

不要立刻增加更多 market 到 source。

先判断：

- learned signature failure；
- host regime failure；
- temporal representation insufficiency。

这可能是 U0 应该提前介入的证据。

### unseen PatchTST weak, seen hosts strong

说明 model-agnostic claim 尚不成立。

优先研究 host-error representation，而不是 market data volume。

### DK1 candidate okay, action layer bad

说明 universal candidate 与 local action adaptation 要分开处理。

不要重训 candidate 去救 local calibration。

---

# 23. Exit criteria to R1C / U0

R1B 完成后必须选择一个状态：

```text
NATIVE_GENERALIZATION_SUPPORTED
MARKET_TRANSFER_LIMITED
HOST_TRANSFER_LIMITED
LOCAL_ADAPTATION_LIMITED
MIXED_GENERALIZATION_LIMIT
```

只有 `NATIVE_GENERALIZATION_SUPPORTED`：

- 才继续 R1C feature/DA-RT；
- 同时 U0 作为 enhancement。

如果是某个 LIMITED：

- U0 / interface redesign 必须针对具体失败轴设计。

---

# 24. Core research principle after R1A

从现在开始任何优化都要满足：

\[
\boxed{
\text{source performance}
+
\text{held-out transfer}
+
\text{worst-domain safety}
}
\]

而不是只追：

\[
\text{source average}.
\]

R1A 的价值是把方法内部链条磨清楚。

R1B 的价值应该是主动尝试**把它弄坏**：

> 如果在新的 host、新的 market 上仍然站住，HCH 才真正开始有“universal”资格。
