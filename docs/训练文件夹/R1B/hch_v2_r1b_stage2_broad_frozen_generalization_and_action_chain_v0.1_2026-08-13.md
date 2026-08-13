# HCH-v2 R1B Stage-2 — Broad Frozen Generalization Stress + Action-Chain Validation v0.1

**日期：2026-08-13**  
**当前实验分支：** `exp/r1b-screening-20260813 @ 0b49f7c33ebfa60a2fc4d5916fdf3f124b0d857d`  
**父协议：** `hch_v2_r1b_universal_generalization_battery_v0.1_2026-08-13.md`  
**上一阶段：** `R1B_SCREEN_HEALTHY`，单种子 candidate-level screening。

---

## 0. Stage-1 审查结论

R1B screening 的核心科学结论可以接受：

- DK1 未进入 universal candidate S2T 梯度；
- LOHO candidate 中 PatchTST 未进入 S2T 梯度或 S2V checkpoint selection；
- LearnedSig 和真正的 PlainCore 已经分离；
- DK1 与 LOHO-PatchTST 上 candidate delta CRPS 均未出现系统性 collapse。

当前可以写：

\[
oxed{	ext{native universal candidate passed the first unseen-market / unseen-host screen}}
\]

但还不能写成：

\[
	ext{cross-market / model-agnostic generalization confirmed}.
\]

进入下一阶段前先修三个实验治理问题。

---

## 1. P0-R1B-S1 — screen-specific `host_seen` 语义修复

当前 runner 的 `host_seen` 是按 host 名字静态判断：

```python
return self.host != "PatchTST"
```

但 seen/unseen host 应当相对于 **candidate training set** 定义。

### Main candidate

训练包含：

```text
Linear
MLP
LSTM
PatchTST
```

所以对 main candidate：

```text
4/4 hosts 全部 seen
```

不能把 PatchTST 标成 unseen host。

### LOHO candidate

训练仅：

```text
Linear
MLP
LSTM
```

所以只有 LOHO 中：

```text
PatchTST = unseen host
```

### Required fix

`market_seen / host_seen` 必须由 candidate 的训练 membership 动态计算，例如：

```python
eval_domain(..., trained_markets, trained_hosts)
```

### Recompute

不要求重训 Stage-1 candidate，仅重新生成正确聚合：

```text
four_cell_aggregate_corrected.csv
```

正确语义：

#### Main

```text
Seen market + Seen host      = 12
Unseen market + Seen host    = 4
```

没有 unseen-host cell。

#### LOHO

```text
Seen market + Seen host      = 9
Seen market + Unseen host    = 3
Unseen market + Seen host    = 3
Unseen market + Unseen host  = 1
```

原文件保留并写 correction note，不覆盖历史痕迹。

---

## 2. P0-R1B-S2 — selected checkpoint reporting

`UniversalCoreTrainer` 按 macro S2V 保存 best state，并在结束后 reload best state，这是正确的。

但当前 R1B `training_reports.json` 读取：

```python
rep["history"][-1]["per_domain"]
rep["history"][-1]["health"]
```

它可能对应最后 epoch，不是 best checkpoint。

虽然 `candidate_transfer_matrix.csv` 是 best state reload 后重新 eval，因此主 transfer 数字仍可信，但 report provenance 不够严谨。

### Required fix

增加：

```text
best_epoch
per_domain_at_best
host_baseline_at_best
delta_at_best
health_at_best
```

或者 best state reload 后显式：

```text
evaluate_selected_checkpoint()
```

并保存：

```text
selected_checkpoint_metrics.json
```

以后不得把 last-epoch diagnostics 当 best-checkpoint headline。

---

## 3. P0-R1B-S3 — server provenance

Stage-1 报告显示服务器目录无 `.git`，导致：

```text
config.json git_sha = unknown
```

下一正式 run 不再允许。

### Preferred

服务器直接：

```bash
git clone / git fetch
git checkout <exact commit>
```

保留 `.git`。

### 如果代码必须外部同步

至少保存：

```text
declared_source_commit
sha256(runner)
sha256(src/iah_candidate.py)
sha256(src/universal_trainer.py)
sha256(src/iah_crps_loss.py)
sha256(src/data_signature.py)
```

---

## 4. 为什么不直接进入 action-chain

Stage-1 真正独立的 unseen market 只有：

```text
NORD_DK1
```

报告中的“DK1 16 rows”来自：

```text
4 host domains × 4 candidate-screen configurations
```

不是 16 个独立市场证据。

因此下一步先做一个完全 zero-gradient 的 broad holdout panel，主动寻找 negative transfer。

---

## 5. Broad frozen holdout panel

Universal candidate source training set保持不变：

```text
LAGO_DE
LAGO_PJM
NEM_SA1
×
Linear / MLP / LSTM / PatchTST
```

所有新增 holdout：

```text
ZERO candidate gradient
ZERO candidate S2V selection signal
```

---

## 6. Holdout groups

### H1 — unseen market holdouts

```text
EPEX_FR
EPEX_BE
EPEX_NL
NORD_FI
NORD_NO
NORD_SE3
NORD_DK1
```

这些作为 repository holdout markets。

### H2 — same-market family / new dataset-regime

```text
DE_EPEX
PJM_2020
```

标签：

```text
SEEN_MARKET_FAMILY / UNSEEN_DATASET_REGIME
```

目的：区分“换市场”和“同市场换数据集/时间段”。

### H3 — historical / different-schema stress

```text
GEFCOM14P
```

标签：

```text
UNSEEN_DATASET_SCHEMA / HISTORICAL_REGIME
```

目的：测试 host-relative correction 是否只适合现代高波动/负价市场。

---

## 7. 不作为 primary 独立 holdout 的数据

暂不把：

```text
LAGO_BE
LAGO_FR
LAGO_NP
```

作为 primary independent unseen-market evidence，因为与 source LAGO_DE/PJM 同 benchmark family / processing convention。

可做 secondary diagnostic，但不能支撑主要 external-generalization claim。

---

## 8. Stage-2A — Fast panel

先只生成：

```text
Linear
MLP
```

host caches。

范围：

```text
H1 + H2 + H3
```

Host seed 固定：

```text
0
```

Candidate：

```text
LearnedSig_main seed=0
PlainCore_main seed=0
```

holdout 永远不参与 candidate optimization。

---

## 9. Transfer taxonomy 升级

每行必须显式记录：

```text
market_family_seen
dataset_seen
host_seen
schema_class
candidate_variant
```

建议类别：

```text
SOURCE_SEEN
UNSEEN_MARKET
UNSEEN_DATASET_SAME_MARKET
UNSEEN_SCHEMA_REGIME
UNSEEN_HOST
UNSEEN_MARKET_AND_HOST
```

未来 R1C 可继续加：

```text
UNSEEN_TARGET_CATEGORY_DA_RT
UNSEEN_OPTIONAL_FEATURE_SCHEMA
```

---

## 10. Candidate-level metrics

每 domain：

- host transformed baseline；
- IAH CRPS；
- delta CRPS；
- mass entropy；
- w0；
- m-/m+ alive；
- shift p50/p95；
- NaN / invalid scale；
- host transformed error；
- host raw MAE。

聚合：

- macro by transfer category；
- worst domain；
- fraction `delta_crps < 0`；
- p10/p50/p90 delta；
- source vs holdout gap。

---

## 11. Generalization gap

定义描述量：

\[
G_{transfer}
=
\overline{\Delta CRPS}_{holdout}
-
\overline{\Delta CRPS}_{source}.
\]

更负更好，所以：

```text
G > 0
```

表示 transfer gain 衰减。

同时报告：

\[
R_{retain}
=
rac{|\overline{\Delta}_{holdout}|}
{|\overline{\Delta}_{source}|+\epsilon}.
\]

只作解释，不作为优化目标。

---

## 12. Stage-2A STOP rules

### MARKET_PANEL_COLLAPSE

超过 1/3 primary unseen-market domains出现：

\[
\Delta CRPS>0.
\]

### DATASET_SHIFT_COLLAPSE

DE_EPEX / PJM_2020 两类同市场新数据 regime 大面积由改善转恶化。

### HISTORICAL_SCHEMA_COLLAPSE

GEFCOM14P 在 Linear/MLP 上均明显恶化。

### SIGNATURE_NEGATIVE_TRANSFER

LearnedSig：

- source 优于 PlainCore；
- holdout panel 却大面积弱于 PlainCore。

任一发生：

```text
STOP before full action-chain
```

不允许把这些 holdout market 加回 source training 来“修”。

---

## 13. Stage-2B — Predeclared deep-host stress

只有 Stage-2A healthy 才增加：

```text
LSTM
PatchTST-style
```

预先冻结 representative，不能看结果后挑容易市场：

```text
EPEX_FR
PJM_2020
GEFCOM14P
NORD_DK1
```

理由：

- EPEX_FR = new European market；
- PJM_2020 = same-market/new-dataset；
- GEFCOM14P = historical/schema shift；
- DK1 = existing Nordic unseen anchor。

---

## 14. LOHO semantics 单独报告

Main candidate 的 PatchTST 不是 unseen host。

真正 unseen-host 只使用：

```text
LearnedSig_LOHO_PatchTST
PlainCore_LOHO_PatchTST
```

训练：

```text
Linear + MLP + LSTM
```

评估 PatchTST-style 于：

```text
source markets
DK1
EPEX_FR
PJM_2020
GEFCOM14P
```

这样得到：

\[
oxed{	ext{unseen host × multiple market/dataset shifts}}
\]

---

## 15. Stage-2C — HCH seed stability

只有 Stage-2A/B 无结构性 collapse，才运行：

```text
HCH seed = 1, 2
```

Host cache固定：

```text
host_seed = 0
```

主确认对象：

```text
LearnedSig_main
```

至少覆盖：

- source 12 domains；
- DK1 4 domains；
- broad fast panel Linear/MLP。

报告：

- mean ± std；
- 3/3 sign consistency；
- worst seed；
- worst domain；
- transfer-category macro per seed。

若：

```text
seed0 good
seed1/2 bad
```

则 seed-0 GREEN 不算稳健 generalization evidence。

---

## 16. Action-chain authorization

仅在：

1. P0 semantics/provenance fixes完成；
2. broad holdout panel无 collapse；
3. LOHO deep-host stress无系统 collapse；
4. LearnedSig 多 seed 无明显 transfer instability；

以后才进入完整 action-chain。

---

## 17. Stage-2D — Full action-chain

Primary universal candidate：

```text
LearnedSig_main
```

比较：

### A0 — Host Identity

```text
pi = 0
```

### A1 — Raw IAH action

```text
analytic C0 utility
→ Double Event
→ own S3C DVG
```

### A2 — Evidence-gated local calibration

```text
analytic C0
→ prequential C0/C3 eligibility
→ selected utility map
→ Double Event
→ selected estimator's own S3C DVG
```

---

## 18. Canonical action-chain domains

第一批：

```text
3 source markets × 4 seen hosts
+
NORD_DK1 × 4 hosts
```

共 16 domain。

注意：

- main candidate 的 PatchTST 是 seen host；
- LOHO candidate → PatchTST 单独作为 unseen-host action-chain stress；
- 不把两者混成一个 macro claim。

---

## 19. Action-chain extension

Canonical 16 healthy 后，再扩：

```text
EPEX_FR:Linear
EPEX_FR:PatchTST-style

PJM_2020:Linear
PJM_2020:PatchTST-style

GEFCOM14P:Linear
GEFCOM14P:PatchTST-style
```

如果 local evidence 太短：

```text
selector must default C0
```

不降低 R1A.11 gate 来强迫 C3。

---

## 20. Action-chain metrics

每 domain × policy：

### Candidate
- CRPS；
- host baseline；
- delta。

### Selector
- S3M days；
- rolling OOS days；
- selected C0/C3；
- reason；
- map stability；
- LCB；
- evidence support。

### DVG
- q；
- coverage；
- release；
- Identity；
- harmful release；
- mean gain | release；
- net daily action value。

### Forecast
- MAE；
- rMAE；
- RMSE；
- no-floor sMAPE；
- negative-price metrics；
- high-tail metrics。

---

## 21. Safety red line

沿用项目讨论中的 normal-period degradation budget：

\[
oxed{15\%}
\]

任何 transfer domain：

\[
rac{MAE_{HCH}-MAE_{host}}
{MAE_{host}}
>15\%
\]

记：

```text
SAFETY_FAILURE
```

同时必须报告 continuous degradation，不只报是否跨阈值。

---

## 22. 结果解释

### Candidate good + action bad

```text
LOCAL_ADAPTATION_LIMITED
```

不要回头改 universal candidate。

### Candidate bad + Identity protects

```text
CANDIDATE_TRANSFER_LIMITED
```

安全不代表 candidate 泛化成功。

### Candidate good + selector因证据不足 default C0

```text
SAFE_FEW_EVIDENCE_FALLBACK
```

这是正常行为。

---

## 23. Local-Core comparison

只有 broad transfer 站住后才补 source-domain：

```text
Local-Core
```

candidate-level upper bound。

目标：

\[
oxed{	ext{量化 universal sharing 的 source-fit cost / benefit}}
\]

DK1 如果训练 Local-Core，只能标：

```text
TARGET_TRAINED_FULLSHOT_UPPER_BOUND
```

不能当 transfer baseline。

---

## 24. GENERALIZATION_LEDGER v2

至少字段：

```text
experiment_id
candidate_variant
training_market_set
training_host_set
evaluation_dataset
transfer_category
source_macro_delta
holdout_delta
worst_domain_delta
seed_consistency
final_mae_effect
safety_effect
complexity_added
accepted_for_universal_core
notes
```

以后任何 trick 都必须进 ledger。

---

## 25. Literature-inspired diagnostics — only diagnose

### D1 — Domain convergence imbalance

记录：

```text
per-domain S2V curve
best epoch by domain
global best macro epoch
```

判断 universal checkpoint 是否系统性牺牲慢收敛 domain。

无证据时不加 GroupDRO/动态权重。

### D2 — Signature shortcut probe

用 frozen learned signature 做：

```text
market-family classifier
host-family classifier
```

仅诊断。

若 source identity 几乎完美可分且 holdout transfer变差，才讨论：

```text
signature dropout / invariant representation
```

### D3 — Transfer vs host difficulty

分析：

\[
\Delta CRPS
\]

和：

- host transformed error；
- residual volatility；
- negative-price rate；

关系。

目的：

> 检查 HCH 是否主要只帮助弱 host / 极端市场。

---

## 26. 与 U0 的关系

不要把 broad holdout 自动加入 source training。

### Broad native transfer healthy

U0 = enhancement。

### unseen-market 弱、unseen-host 正常

U0 可以成为：

```text
temporal/domain representation repair
```

### unseen-host 弱

优先怀疑：

```text
host-error representation
```

而不是默认 U0。

### LearnedSig hurts holdout

先修 signature generalization，再谈 U0。

---

## 27. 与 R1C DA/RT 的关系

现有 DA/RT audit保留：

```text
NYISO primary
ERCOT secondary
```

本阶段不接入。

只有 R1B market/host/dataset 泛化大体成立后，才进入：

\[
oxed{
R1C =
same-market DA/RT target-category transfer
+
feature-schema stress
}
\]

---

## 28. Branch discipline

继续：

```text
exp/r1b-screening-20260813
```

建议：

```text
R1B Stage-2 candidate stress
→ action-chain
→ scientific review
→ squash/merge accepted infra + docs
```

避免 main 被 screening 中间状态污染。

---

## 29. Exact execution order

```text
P0-1 screen-relative seen/unseen semantics
P0-2 selected-checkpoint report fix
P0-3 server git/hash provenance
    ↓
Stage-2A
broad Linear/MLP zero-gradient holdout panel
    ↓
if healthy
Stage-2B
predeclared LSTM/PatchTST deep holdouts
    ↓
if healthy
Stage-2C
HCH seeds 1/2 for LearnedSig
    ↓
if stable
Stage-2D
full action-chain on canonical 16
    ↓
if healthy
action-chain extension to FR / PJM2020 / GEFCOM
    ↓
Local-Core upper bound
    ↓
R1B FINAL VERDICT
```

---

## 30. Final R1B verdict labels

只能选择：

```text
NATIVE_GENERALIZATION_SUPPORTED
MARKET_TRANSFER_LIMITED
DATASET_SHIFT_LIMITED
HOST_TRANSFER_LIMITED
LOCAL_ADAPTATION_LIMITED
SIGNATURE_GENERALIZATION_LIMITED
MIXED_GENERALIZATION_LIMIT
```

---

## 31. 什么才算真正有意义的 R1B 成功

不是：

> “DK1 上也提升了。”

而是：

\[
oxed{
	ext{多个未见市场、未见数据 regime、未见 host 上，
candidate 方向一致；
无法获益的 domain 可以安全退化；
source 提升没有以 unseen 退化为代价。}
}
\]

到这个程度，HCH 才真正有“universal correction core”雏形。
