# HCH-v2 R1A.7 Prior–Local Value Fusion + PJM:MLP Failure Audit + Compute Plan v0.1

**日期：2026-08-13**  
**仓库基线：** `main@5b6eda689e22b92462020170d126c31518c78e95`  
**前置状态：** `UNIVERSAL_CANDIDATE_SUPPORTED`；`R1A.6 = VALUE_ESTIMATION_UNRESOLVED (YELLOW)`。

本阶段不扩 Host、不扩 Market、不启动 U0，不重新训练 IAH candidate。

目标：

\[
\boxed{\text{利用“全局解析 value + 新鲜局部 evidence”的互补性，恢复跨域稳定的 action-value ranking}}
\]

---

## 1. R1A.6 的核心解释

### Fact A — IAH 本身有 ex-ante action-value signal

V1 `IAHNativeValue`：macro Spearman = 0.176，且 top10 hit-lift = 0.292，为五个 estimator 中最高。

这说明 IAH 三原子分布不仅改善 CRPS，也携带可解析的 action-value information。

### Fact B — static retrieval 无效，memory freshness 有效

- V2 learned-key + static memory：macro rho = -0.086；
- V3 learned-key + prequential memory：macro rho = +0.146。

所以问题不只是 key 几何，**新鲜 local evidence 是主要有效信号来源之一**。

### Fact C — local weighting 提升 broad ranking，但稳定性仍不足

V4：macro rho = 0.205，5/6 domains 为正；但 `LAGO_PJM:MLP = -0.244`，且 excl-best = 0.113。

因此 V4 不能直接升级为正式 universal value estimator。

---

## 2. 下一步为何是 shrinkage fusion，而不是继续换 retriever

V1 与 V4 的优势互补：

- V1：stable global analytic prior；无 memory、无 retrieval、zero-parameter；top-value enrichment 最强；
- V4：prequential weighted local evidence；整体排序更强，但某些 domain 会被局部 evidence 带偏。

因此当前问题更像：

\[
\boxed{\text{stable global prior} + \text{high-variance local evidence}}
\]

下一步让 local evidence **修正** IAH prior，而不是完全替代它。

---

## 3. Proposed micro-adjustment: Prior–Local Shrinkage

定义：

- IAH-native analytic directional gain：\(g_h^I\)
- V4 prequential weighted local directional gain：\(g_h^L\)

构造：

\[
\boxed{g_h^{(\lambda)}=(1-\lambda)g_h^I+\lambda g_h^L},\qquad 0\le\lambda\le1.
\]

边界：

\[
\lambda=0\Rightarrow V1,\qquad \lambda=1\Rightarrow V4.
\]

### Proposal 与 value 必须使用同一套 fusion

先融合两个方向的逐时 gain：

\[
g_{\downarrow}^{(\lambda)}=(1-\lambda)g_{\downarrow}^{I}+\lambda g_{\downarrow}^{L}
\]

\[
g_{\uparrow}^{(\lambda)}=(1-\lambda)g_{\uparrow}^{I}+\lambda g_{\uparrow}^{L}
\]

输入现有 double-event optimizer 得到：

\[
\pi_q^{(\lambda)}.
\]

随后 IAH prior 和 local replay 都对**同一个 final action**评估：

\[
\widehat A_q^I(\pi_q^{(\lambda)}),\qquad \widehat A_q^L(\pi_q^{(\lambda)}).
\]

最终：

\[
\boxed{\widehat A_q^{(\lambda)}=(1-\lambda)\widehat A_q^I+\lambda\widehat A_q^L}
\]

这保证 proposal/value semantics 一致。

---

## 4. Lambda protocol

只测试：

\[
\lambda\in\{0,0.25,0.50,0.75,1.0\}.
\]

禁止：

- per-market lambda；
- per-host lambda；
- per-day learned lambda；
- Bayesian optimizer；
- S4 自动调参。

若需要 provisional selection，只允许用 `S3M-validation`：

1. 先比较 worst-domain Spearman；
2. 再比较 macro Spearman；
3. 再比较 top10 realized-gain enrichment。

不要为了 NEM 的大收益牺牲 PJM。

---

## 5. PJM:MLP Failure Audit

`LAGO_PJM:MLP` 在 V0–V4 全部为负，因此不能继续简单归因于 retrieval。

### 5.1 Same-market host contrast

严格比较：

`LAGO_PJM:Linear` vs `LAGO_PJM:MLP`

报告：

- host MAE / rMAE；
- hyperbolic host error；
- S2V IAH-CRPS delta；
- residual mean/std/IQR；
- residual sign balance；
- lag-1 / daily residual autocorrelation；
- large positive/negative residual rate；
- S1R → S2V → S3 → S4 residual drift。

### 5.2 IAH directional calibration

定义真实 benefit：

\[
B^-_h=\mathbf 1[g^{true}_{h,\downarrow}>0],\qquad
B^+_h=\mathbf 1[g^{true}_{h,\uparrow}>0].
\]

检查 \(w^-\) 对 \(B^-\)、\(w^+\) 对 \(B^+\) 的：

- rank/AUC；
- 10-bin reliability；
- \(g^{IAH}_{down}\) vs \(g^{true}_{down}\) Spearman；
- Up 同理。

如果 PJM:MLP 在 action-direction 上失准，而 CRPS 仍可接受，说明 distributional calibration 与 action calibration 在该 host regime 上分离。

### 5.3 Learned-key degeneracy

检查：

- key norm distribution；
- pairwise cosine distance；
- nearest-neighbor distance；
- distance concentration；
- distinct-neighbor count；
- 与 PJM:Linear / DE:MLP / NEM:MLP 对照。

### 5.4 Local weight reliability

对 V4 定义：

\[
ESS_q=\frac1{\sum_j\omega_j^2}.
\]

报告：

- ESS median/p10/p90；
- top neighbor weight；
- median distance；
- local estimation error vs ESS；
- Spearman by ESS tercile。

本阶段只诊断，不改 kernel。

---

## 6. R1A.7 experiment matrix

- F0: \(\lambda=0\)（V1）
- F1: \(\lambda=0.25\)
- F2: \(\lambda=0.50\)
- F3: \(\lambda=0.75\)
- F4: \(\lambda=1\)（V4）

全部保持：

- frozen candidate；
- same prequential memory；
- same weighted local estimator；
- same k；
- no DVG；
- no new trainable network。

---

## 7. Metrics and gate

Primary：

\[
\operatorname{Spearman}(\widehat A,A^{true}).
\]

必须报告：

- per-domain；
- macro；
- worst-domain；
- excl-best-domain；
- by temporal block；
- top10 hit-lift；
- mean/median A_true in top10；
- oracle efficiency；
- missed-positive rate；
- direction mismatch。

不确定性：7-day moving-block bootstrap，500+ resamples。

### GREEN

希望至少满足：

1. macro Spearman >= 0.20；
2. 6/6 domain rho > 0，或 worst-domain > -0.05；
3. excl-best >= 0.15；
4. top10 hit-lift >= 0.20；
5. PJM:MLP 不再是明显负相关 outlier。

若达不到，不因 macro > 0.20 就宣称 solved。

---

## 8. 若 fusion 仍 YELLOW

### Case A — atom/action calibration failure

若 PJM:MLP 的 \(w^\pm\) 对 directional benefit 无排序能力，下一问题改为：

\[
\boxed{\text{IAH probability mass} \rightarrow \text{action-value calibration}}
\]

此时才讨论更保守 dose 或极小 action calibration layer。

### Case B — representation/key degeneracy

才考虑 richer history representation / foundation representation / MOMENT U0。

### Case C — local evidence high variance

才考虑 reliability-adaptive shrinkage、ESS-based lambda、robust local estimator。

---

## 9. R1B status

R1B 继续暂停，直到 R1A.7 GREEN，或已经量化定位并裁决唯一失败机制。

原 R1B 方向保留：

- + LSTM；
- + PatchTST；
- PlainCore / LearnedSig / Learned+DetSig；
- Local-Core；
- NORD_DK1 frozen holdout。

---

## 10. 新数据现在是否需要

**不需要。**

R1A.7 使用现有 frozen R1A artifacts 即可。此时加数据会混淆 failure attribution。

### 可并行的只读准备任务

本地 Agent 可制作：

`docs/paper_prep/v2_final_prep/u0_external_corpus_inventory_v0.1.md`

只 inventory，不大规模下载：

- HF dataset/config；
- source organization；
- license；
- frequency；
- series count；
- variables；
- missingness；
- approximate size；
- energy/electricity relevance；
- potential overlap with MOMENT/Chronos pretraining；
- intended role：U0 representation / host diversity / final benchmark / exclude；
- exact HF revision if available。

优先盘点 Chronos/LOTSA 的 electricity/load/solar/wind、Monash Electricity、Australian Electricity Demand。

---

## 11. GPU server decision

### R1A.7

不租也可以。本阶段以 frozen inference、NumPy/SciPy retrieval 与 diagnostics 为主，本地 4060/CPU 足够。

### R1B

建议开始租服务器。R1B 要训练/缓存 Linear、MLP、LSTM、PatchTST，并做多 seed 和 holdout，服务器主要节省 wall-clock。

### U0

强烈建议服务器。需要批量 512-step windows、MOMENT embedding extraction、teacher feature bank 与 student HistorySignatureEncoder。

---

## 12. 推荐服务器规格

### Tier A — 当前性价比最合适

```text
GPU: 1 x RTX 4090 24GB
     或 RTX 5090 32GB（若租金接近）
CPU: 16-24 vCPU
RAM: 64GB minimum, 128GB preferred
Disk: 1TB NVMe minimum
OS: Ubuntu LTS
SSH: required
```

适合：R1B、LSTM/PatchTST host、MOMENT-small embedding、single-teacher U0、HCH student。

当前阶段不需要多 GPU。

### Tier B — 希望一台机器一路用到后面

```text
GPU: A6000 / A40 / L40S 48GB
CPU: 24-32 vCPU
RAM: 128GB
Disk: 1-2TB NVMe
```

如果 48GB 卡租金只比 4090 高不多，优先 Tier B，避免后面多 teacher / 大 batch 时重新迁移。

### Tier C — 目前不建议

```text
A100 80GB
H100/H200
multi-GPU
```

只有未来出现 multi-teacher joint extraction、very large encoder、极大窗口库或真正 foundation-scale pretraining 时再评估。

---

## 13. Server workflow

SSH Agent 工作流固定：

```text
1. clone/pull exact git commit
2. create isolated environment
3. verify CUDA/PyTorch
4. mount persistent data volume
5. raw/HF cache outside git
6. run tests
7. generate host/teacher cache + manifests
8. run experiment
9. commit only code + small audit artifacts
10. large outputs remain gitignored on persistent storage
```

每个正式 run 记录：

- git SHA；
- environment lock；
- GPU；
- CUDA/PyTorch；
- CPU/RAM；
- seeds；
- command；
- wall-clock；
- peak VRAM；
- peak host RAM。

---

## 14. 当前执行顺序

```text
R1A
candidate supported
  ↓
R1A.5
value-estimation bottleneck located
  ↓
R1A.6
IAH prior + fresh local evidence both contain signal, but unstable
  ↓
R1A.7
prior-local shrinkage + PJM:MLP audit
  ↓
if GREEN:
  rent server → R1B native transfer
  ↓
  U0 representation distillation
else:
  modify only the diagnosed failed component
```

原则：

\[
\boxed{\text{不为了推进而推进；每层先证明真实有效，再扩大实验规模}}
\]
