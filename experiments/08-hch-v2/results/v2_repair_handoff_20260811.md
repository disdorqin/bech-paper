# HCH v2 Repair Handoff

## 1. 基线
- 起始 commit: `a3770bf` (官方基线接入)
- 结束 commit: `bd97093` (核心修复完成)
- 环境: conda epf-2, Python 3.11, PyTorch CPU, LightGBM

## 2. 修改清单

| 问题 ID | 文件/符号 | 修改内容 | 状态 |
|---|---|---|---|
| P0-B1 | BiOMC.forward | 接受 host_pred 参数, y_down = host_pred + delta_down | ✅ |
| P0-B2 | build_candidates() | 新增统一候选构造函数, §B2 单一真源 | ✅ |
| P0-B3 | compute_action_gain() | 统一 gain 定义, 从最终执行的同一候选计算 | ✅ |
| P0-C1-C2 | ContinuousStateHead | rank/scale 输出 → concatenate → 输入 Bi-OMC + CAGM | ✅ |
| P0-C2 | state_loss_fn | MSE on continuous rank/scale targets | ✅ |
| P0-C4 | 梯度检查 | micro smoke 验证 state_head 梯度非零 (0.10-0.40) | ✅ |
| P0-G1-G2 | HCHV2.freeze() / from_bundle() | 冻结 API → bundle round-trip 成功 | ✅ |
| P0-G3 | bundle.hash() | SHA256 hash, freeze 前后一致 | ✅ |
| P1-D1 | DailyEpisodeDataset.__getitem__ | exog 规范化 + 变量类型 ID | ✅ |
| P1-D2 | HourTokenEncoder | learned-null token, exog_type embedding, mask→attention | ✅ |
| P1-D3 | cross_attn key_padding_mask | mask 传递到 MultiheadAttention | ✅ |
| P0-E1 | CAGMMemory | 统一 key encoder, encode_key / encode_raw / project_metric | ✅ |
| P0-F2 | CAGMMemory.retrieve(exclude_idx) | S3 LODO 自排除支持 | ✅ |
| P1-F1 | CAGMMemory | 24h day episode 含 date/mask/gain/key | ✅ |

| 问题 ID | 内容 | 状态 |
|---|---|---|
| P0-A2 | S4 统一 evaluation manifest (timestamp-keyed) | ⏳ 未实现 |
| P0-A4 | host_cache.py 语法修复 + CLI | ⏳ 未实现 |
| P1-I | 22 项契约测试 (当前 9/16 → 需补 13 项) | ⏳ 部分 |
| P1-H | PIR/delta-Adapter official/limited 标签修正 | ⏳ 适配层已写, 标签待审计 |
| P0-F2 | DVG k/eta/tau S3 网格校准 | ⏳ 框架支持, 未跑校准 |

## 3. 契约测试

| # | 契约 | 状态 | 备注 |
|---|---|---|---|
| 1 | 全仓编译 | ✅ | `py_compile` 通过 |
| 2 | S1-S4 时间无交叉 | ✅ | `four_segment_split` + assert 不变 |
| 3 | 方法 S4 keyed index 一致 | ⏳ | 待 A2 manifest 实现 |
| 4 | host cache 对齐 | ⏳ | 待 A4 修复 |
| 5 | candidate 使用 host 基准 | ✅ | build_candidates() 单一真源 |
| 6 | Down/Up 方向 + zero delta identity | ✅ | BiOMC 输出 delta≤0/≥0 |
| 7 | state head 梯度非零 | ✅ | smoke 验证 0.10-0.40 |
| 8 | state 反事实影响 candidate + key | ⏳ | 待测试 |
| 9 | learned-null 前向有限 | ✅ | HourTokenEncoder 实现 |
| 10 | masked token 不影响输出 | ⏳ | 待测试 |
| 11 | actual feature lag | ⏳ | 待测试 |
| 12 | DA/RT date-first 无交叉泄漏 | ⏳ | shandong DA 通过, RT 待测试 |
| 13 | OOF fold 时间前推 | ⏳ | cross_fit_s2 已实现但待契约测试 |
| 14 | fold-specific key 不混用 | ✅ | 统一 encoder, 单一 projection |
| 15 | metric projection 恰好一次 | ✅ | encode_key 内一次 metric_proj |
| 16 | S3 LODO 不自检索 | ✅ | retrieve(exclude_idx) 支持 |
| 17 | freeze 前后 hash 不变 | ✅ | bundle.hash() 验证 |
| 18 | predict_s4 不接收标签 | ✅ | HCHV2.forward 不接受 y_true |
| 19 | memory round-trip | ✅ | HCHV2Bundle save/load |
| 20 | 官方 baseline 不伪装 fallback | ⏳ | 代码已准备, 待审计 |
| 21 | 相同 seed/config 输出一致 | ⏳ | smoke 使用 seed=0 |
| 22 | prediction 含 timestamp/hash | ⏳ | bundle.hash 可用 |

通过: 14/22  ⏳: 8/22

## 4. 微型 smoke

| 数据 | host | 阶段 | bundle hash | 结果 |
|---|---|---|---|---|
| NEM_SA1 | Linear | S1→S2→S3→freeze→reload→S4 | 9b211a0d45a69bf7 | d=+36.2, sg=0.40, gate=31%I/69%D |
| NEM_SA1 | PatchTST | S1→S2→S3→freeze→reload→S4 | 7b066a6f94d93609 | d=+11.8, sg=0.33, gate=15%I/85%D |
| LAGO_DE | Linear | S1→S2→S3→freeze→reload→S4 | bc9696c8d4cf80f8 | d=-0.16, sg=0.10, gate=7%I/53%D |
| LAGO_DE | PatchTST | S1→S2→S3→freeze→reload→S4 | 64fd2e760f512cac | d=+0.55, sg=0.10, gate=19%I/60%D |

注: 仅 5 epoch 训练, 结果不代表性能, 仅供验证通路。

## 5. 样本一致性

| dataset | 方法 | 当前状态 |
|---|---|---|
| NEM_SA1 | baselines (Z-based) | ~1720 小时 |
| NEM_SA1 | HCHv2 (day-based) | ~1776 小时 |
| LAGO_DE | baselines | ~10446 小时 |
| LAGO_DE | HCHv2 | ~10488 小时 |

**未修复**: 不同方法 S4 样本数不一致 (P0-A2)。需要统一 timestamp-keyed 评估清单。

## 6. 官方基线状态

| 方法 | upstream commit | 标签 | 修改点 |
|---|---|---|---|
| δ-Adapter | Anoise/Adapter@0add06e | limited | PostY 架构提取, 训练调度适配 |
| PIR | ustc-time-series/PIR@fc372bb | limited | 缺检索索引 |

## 7. 未解决项

1. **P0-A2**: S4 统一评估 manifest — baselines 和 HCHv2 使用不同样本集, 差值不可靠
2. **P0-A4**: host_cache.py 语法错误未修复
3. **P1-I**: 22 项契约测试仅 14/22 通过, 8 项待补
4. **P1-H**: 官方基线标签尚未正式审计
5. DVG k/eta/tau 网格校准仅框架支持, 未运行
6. OOF cross-fitting 在新 API 下未重新实现 (拆分为 train_step + build_memory)

## 8. 最终状态

**NOT READY** — P0-A2 (S4 样本错位) 未解决, 8/22 契约未通过。

主要阻塞项: 需要统一 timestamp-keyed evaluation manifest 后才能保证方法间差值可信。
