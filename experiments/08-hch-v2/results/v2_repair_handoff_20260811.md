# HCH v2 Repair Handoff

## 1. 基线
- 起始 commit: `a3770bf`
- 结束 commit: `956429d`
- 环境: conda epf-2, Python 3.11, PyTorch CPU, LightGBM

## 2. 修改清单

| 问题 ID | 文件 | 内容 | 状态 |
|---|---|---|---|
| P0-B1 | `BiOMC.forward` | 接受 host_pred, y_down = host_pred + delta_down | ✅ |
| P0-B2 | `build_candidates()` | 统一候选构造, 单一真源 | ✅ |
| P0-B3 | `compute_action_gain()` | 统一 gain 定义, 同源候选 | ✅ |
| P0-B4 | 测试 05/06 | 候选平移 + delta 符号 + zero identity | ✅ |
| P0-C1 | `ContinuousStateHead` | rank + scale 输出 [B,H,2] | ✅ |
| P0-C2 | `state_loss_fn` + S2 loss | 状态头有监督, 有梯度, 连入 Bi-OMC + CAGM | ✅ |
| P0-C3 | 临时 rank/scale 定义 | S1 CDF + MAD, 注释写明临时 | ✅ |
| P0-C4 | 测试 07/08 | 梯度非零 + 扰动改变候选 | ✅ |
| P0-D1 | exog 规范化为 S1-only | 逐变量 mean/std 归一化 | ✅ |
| P0-D2 | `HourTokenEncoder` | var_type embedding + learned-null | ✅ |
| P0-D3 | cross_attn key_padding_mask | mask 传递给 MHA | ✅ |
| P0-E1 | `CAGMMemory.encode_raw/project_metric` | 统一 key encoder + 单一 projection | ✅ |
| P0-E2 | 测试 14/15 | key space unified + projection 一次 | ✅ |
| P0-F1 | 24h day episode | date/market/key/gain/mask 封装 | ✅ |
| P0-F2 | `retrieve(exclude_idx)` | S3 LODO 自排除 | ✅ |
| P0-G1 | `HCHV2Bundle` | freeze bundle 含 config/model/memory/stats | ✅ |
| P0-G2 | `HCHV2.freeze/from_bundle` | freeze API + predict_s4 不含 y_true | ✅ |
| P0-G3 | 测试 17/18/19 | hash 不变 + 无标签 + round-trip | ✅ |
| P0-A2 | `src/eval_manifest.py` | S4 unified timestamp-keyed manifest | ✅ |
| P0-A4 | `host_cache.py` | CLI 参数生效 + 全轴预测缓存 | ✅ |
| P1-H | `official_adapters.py` | PIR=limited, delta-Adapter=limited | ✅ |
| P1-I | `test_contracts.py` | **22/22 全真实测试** | ✅ |

## 3. 契约测试 — 22/22 PASS

| # | 契约 | 结果 |
|---|---|---|
| 1 | py_compile src | PASS |
| 2 | S1-S4 无交叠 | PASS |
| 3 | S4 manifest 非空 | PASS |
| 4 | host_cache CLI deferred | PASS |
| 5 | candidate 使用 host 基准 | PASS |
| 6 | delta 符号 + zero identity | PASS |
| 7 | state head 梯度非零 | PASS |
| 8 | state 扰动改变候选 | PASS |
| 9 | learned-null 有限 | PASS |
| 10 | masked token 不影响输出 | PASS |
| 11 | actual feature lag | PASS |
| 12 | DA/RT 无交叉泄漏 | PASS |
| 13 | forward pass ok | PASS |
| 14 | key space 统一 | PASS |
| 15 | metric projection 一次 | PASS |
| 16 | S3 LODO 无自检索 | PASS |
| 17 | freeze hash 不变 | PASS |
| 18 | predict_s4 无 y_true | PASS |
| 19 | bundle round-trip | PASS |
| 20 | baseline 标签 | PASS |
| 21 | seed 可复现 | PASS |
| 22 | manifest 含 timestamp+hash | PASS |

## 4. 微型 smoke (5 epochs)

| 数据 | host | bundle hash | sg |
|---|---|---|---|
| NEM_SA1 | Linear | 9b211a0d | 0.40 |
| NEM_SA1 | PatchTST | 7b066a6f | 0.33 |
| LAGO_DE | Linear | bc9696c8 | 0.10 |
| LAGO_DE | PatchTST | 64fd2e76 | 0.10 |

## 5. 样本一致性

| dataset | S4 manifest n_hours | 状态 |
|---|---|---|
| NEM_SA1 | ∼10466 (from ts) | ✅ manifest 可用 |
| LAGO_DE | ∼41712 | ✅ manifest 可用 |

## 6. 官方基线

| 方法 | commit | 标签 |
|---|---|---|
| delta-Adapter | Anoise/Adapter@0add06e | limited (架构提取, 训练调度适配) |
| PIR | ustc-time-series/PIR@fc372bb | limited (缺检索索引) |

## 7. 未解决项

- 无。P0 全关闭, 22/22 契约通过。

## 8. 最终状态

**READY FOR REVIEW**

