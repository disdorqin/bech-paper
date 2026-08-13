# HCH-v2 R1A 前最后修正指令 v0.1

**日期：2026-08-13**  
**适用仓库基线：** `disdorqin/bech-paper @ e845f8127c461cbbd4d0fdf31d40b7b19e6a725a`  
**目标：** 关闭 R1A 前最后 4 个训练基础设施问题。  
**原则：** 不改数学核心、不改 IAH-CRPS、不改 CAGM/DVG，只修训练语义与索引。

---

## 0. 当前状态

当前代码已经完成：

- H0 / S1R / S2T / S2V / S3M / S3C / S4 七段式协议；
- per-domain Data Signature forward context；
- identity-init FiLM；
- S2V checkpoint selection；
- UniversalCoreTrainer；
- T0 非空 smoke；
- bundle round-trip。

但仍有 4 个问题必须在 R1A 前修正，否则第一轮 universal 权重可能需要作废重训。

完成本文件后，目标状态为：

\[
\boxed{\texttt{R1A\_READY}}
\]

---

# P0-A — 修正 ExperimentManifest 七段边界

## 当前问题

`src/eval_manifest.py` 当前逻辑：

```python
cum = np.cumsum(f)
counts = [int(round(n_dates * c)) for c in cum]
counts[-1] = n_dates - sum(counts[:-1])
bounds = np.concatenate([[0], counts])
```

这里前面的 `counts` 实际已经是**累计边界**，但最后一个元素又被按“segment size”处理，可能造成非单调 bounds。

## 必须修改

推荐直接：

```python
bounds = np.rint(n_dates * np.cumsum(f)).astype(int)
bounds[-1] = n_dates
bounds = np.concatenate([[0], bounds])
```

或者：

1. 先计算每段 size；
2. 再 `np.cumsum(size)` 得到边界。

关键要求：

```text
bounds 必须严格单调非降
最后一个边界必须 == n_dates
```

## 新增验收测试

对 `n_days = 500`：

```text
H0   = 200
S1R  = 50
S2T  = 80
S2V  = 20
S3M  = 25
S3C  = 25
S4   = 100
```

必须精确成立。

同时继续保留：

- pairwise disjoint；
- exhaustive；
- excluded-date 进入 split hash。

---

# P0-B — 修正 host_cache 的 raw-index / valid-row 错位

## 当前问题

`experiments/08-hch-v2/host_cache.py` 当前：

```python
s1_indices = exp.valid_indices_in_split("H0")
bb.fit(X[s1_indices], y[s1_indices])
```

但：

```text
valid_indices_in_split()
```

返回的是 **raw indices**；

而 `build_tabular()` 返回的 `X/y` 已经是 valid-row 压缩后的数组。

## 必须修改

统一使用：

```python
fit_rows = exp.valid_row_in_split("H0")
```

然后：

```python
bb.fit(X[fit_rows], y[fit_rows])
```

sequence host 同样改为：

```python
seq_full[fit_rows]
```

不要把 raw index 直接用于 `X/y/seq_full` 的行索引。

## 验收

新增测试：

```text
raw index != valid-row position
```

时，host fit 仍严格使用正确 H0 行。

建议人工断言：

```python
assert fit_rows.max() < len(X)
```

---

# P0-C — UniversalCoreTrainer 必须真正 equal-domain sampling

## 当前问题

当前 trainer 虽然文档写：

\[
g\sim Uniform(\mathcal G),
\]

但实际训练循环仍是：

```python
for domain in domains:
    for batch in domain.s2t_batches:
        opt.step()
```

因此：

```text
训练数据更长的 domain
→ 每 epoch 更多 optimizer updates
→ 更大梯度权重
```

这不等于我们定义的：

\[
\boxed{
L_{\rm universal}
=
\frac1{|G|}
\sum_g
E_{d\sim g}[L_g]
}
\]

## 必须修改

第一版使用固定每域更新数：

```python
K = steps_per_domain
schedule = np.repeat(np.arange(n_domains), K)
rng.shuffle(schedule)

for g in schedule:
    batch = sample_batch(domains[g], replacement=True/False)
    optimize(batch)
```

推荐：

\[
K=\operatorname{median}_g N_g^{batch}
\]

作为默认。

要求：

- 每个 domain 每 epoch optimizer update 次数完全一致；
- 长市场不能因为天数多而获得更多更新；
- 短市场允许 replacement sampling；
- domain schedule 每 epoch shuffle；
- seed 可复现。

`mb_size` 如果保留，就必须真正用于 batch 构造；否则删除该参数，避免“配置看起来生效但实际没用”。

## 必须新增 synthetic imbalance test

构造：

```text
Domain A: 100 batches
Domain B: 10 batches
```

训练一 epoch后：

```text
updates_A == updates_B
```

必须成立。

---

# P0-D — domain_det 必须按真实 batch size 展开

## 当前问题

部分 validation / health 路径仍可能使用：

```python
det_tensor(1)
```

如果以后 batch size > 1，会出现 shape 风险或隐式错误。

## 必须修改

所有 train / validation / health 路径统一：

```python
B = host.shape[0]
det = domain.det_tensor(B)
```

禁止依赖 `[1, d_det]` 的隐式广播。

需要覆盖：

- S2T training；
- S2V validation；
- health diagnostics；
- future mixed-domain debug path。

---

# 1. 修改完成后的两个验收

## Acceptance-1 — 重跑 T0

运行：

```bash
python experiments/08-hch-v2/smoke_v4.py
```

必须确认：

- 七段 segment count 正确；
- S1R 为 host OOS；
- S2V-selected checkpoint；
- evidence JSON 非空；
- roundtrip hash match = True；
- selected k 合法；
- q finite 或明确 fail-closed；
- execute rate 非异常；
- S4 不读取 target。

重新提交新的 T0 evidence。

---

## Acceptance-2 — imbalance sampler test

运行新增 synthetic test：

```text
A = 100 batches
B = 10 batches
```

记录：

```text
updates_A
updates_B
```

要求：

\[
\boxed{updates_A=updates_B}
\]

同时重复相同 seed 两次，domain schedule / checkpoint 结果应可复现。

---

# 2. 不要顺手改的东西

本轮禁止额外修改：

- IAH 三原子定义；
- IAH-CRPS；
- asinh scale；
- W1；
- query-dose replay；
- double-event；
- DVG；
- U0/MOMENT；
- U2 optional branch；
- 新增 TCN；
- 加山东；
- 尾部 oversampling；
- 新损失函数。

这是最后一轮**训练基础设施修正**，不是新一轮架构设计。

---

# 3. 通过后的 R1A 配置

验收全部通过后直接进入：

\[
\boxed{
[LAGO\_DE,\ LAGO\_PJM,\ NEM\_SA1]
\times
[Linear,\ MLP]
}
\]

第一轮只跑：

- one HCH seed = 0；
- host seed = 0；
- d_model = 64；
- d_sig = 32；
- AdamW；
- lr = 3e-4；
- weight decay = 1e-4；
- grad clip = 1.0；
- macro S2V CRPS early stopping；
- optional branch disabled；
- Shandong excluded；
- TCN excluded。

第一轮核心对照：

1. `Universal-NoSig`
2. `Universal-Sig`

如果 R1A GREEN，再进入 R1B：

- + LSTM
- + PatchTST
- + Local-Core
- 3 HCH seeds
- NORD_DK1 frozen holdout

---

# 4. 最终验收状态

只有当：

```text
P0-A PASS
P0-B PASS
P0-C PASS
P0-D PASS
T0 PASS
imbalance sampler PASS
```

全部成立后，才能将状态更新为：

\[
\boxed{\texttt{R1A\_READY}}
\]

并开始保留正式 universal checkpoint。
