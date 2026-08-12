# HCH v2 最新仓库复审与基础修复增量规范

> 版本：v0.1  
> 日期：2026-08-12  
> 仓库：https://github.com/disdorqin/bech-paper  
> 复审 commit：`f73b56149baa55e6c675fef072b39f1a1475f0b3`  
> 对照旧审计基线：`a3770bf813d56b2c597cd7917ee57fd46a8f654f`  
> 当前裁决：**FOUNDATION_REPAIR_REQUIRED**  
> 本轮目标：修复与新数学无关的基础契约，并纠正无效证据；**禁止实现 IAH-CRPS，禁止 half-exp，禁止正式消融**

---

## 0. 给代码 AI 的直接指令

请以最新 `main@f73b561` 为起点，仅执行本文列出的基础修复。不要依据仓库中现有的 Student-t/W2 文档继续设计方法，也不要依据用户新提供的 IAH-CRPS 文档改写 HCH 主模块；数学方案仍在交叉裁决。

本轮完成后，代码应达到：

1. 数据划分、原始价格、标准化输入、外生变量、时间戳与冻结 bundle 的契约真实可用；
2. 测试能证明契约，而不是只证明函数能运行；
3. 旧 HCH 路径被明确标成 `legacy_unvalidated`，不得产生论文结果；
4. 数学证据脚本中的符号、oracle、bootstrap、日矩阵和 CDF 错误被修正或隔离；
5. 输出一份新的修复 handoff，等待主架构窗口复审。

不要删除旧研究文件；新报告必须使用新文件名。

---

## 1. 复审结论

### 1.1 总结

仓库从旧基线增加了 7 个提交，确实完成了一些有效修改：

- Down/Up 最终候选已相对冻结宿主构造；
- 连续状态输出已接入旧 Bi-OMC；
- attention 收到了一个 mask 参数；
- PIR/δ-Adapter 的文档开始承认 limited implementation；
- 增加了 bundle、S3 calibration 和 evaluation manifest 的雏形。

但是，`v2_repair_handoff_20260811.md` 的“P0 全关闭、22/22 全真实测试、未解决项为空”不成立。当前 HEAD 至少存在以下直接反证：

- `experiments/08-hch-v2/host_cache.py` 第 29 行仍为顶层意外缩进，仓库级 compile 失败；
- 契约 04 的实现仍然只是 `pass`；
- freeze hash 测试只是比较 `b.hash() == b.hash()`；
- “predict_s4 无标签”测试仍把含 `target` 的 batch 传给通用 `forward`；
- S4 结果仍靠数组尾部截断或前部补 NaN 对齐；
- CAGM 的 key 网络没有进入任何 loss，保持随机初始化；
- S3 calibration 用邻居平均 gain 给自身打分，并丢弃负 gain；
- 校准后的 `eta/tau` 没有可靠封入 bundle，reload 后可能恢复默认值。

所以当前代码可作为原型素材，但不能进入数学融合、冻结实验或 half-exp。

### 1.2 旧修复项逐条复审

| 原问题 | 当前状态 | 源码证据 | 裁决 |
|---|---|---|---|
| 候选必须为 `host + delta` | 基本修正 | `build_candidates` 及 `HCHV2.forward` | PASS，但训练/记忆仍需统一调用 |
| Down≤Identity≤Up | 基本修正 | `delta_down=-p·m`、`delta_up=p·m` | PASS，仅限旧模型 |
| 连续状态不能是死模块 | 只修一半 | 状态有梯度，但 normalized target 被拿去查 raw S1 CDF | FAIL |
| exog mask / null / type | 只修一部分 | mask 传入 MHA，但 null 分支和 type embedding 实际未正确使用 | FAIL |
| S1-only 外生规范化 | 未修 | 每个目标日用当日 24 点自身均值/方差标准化 | FAIL |
| actual exog lag≥24h | HCH 数据层未实现 | `DailyEpisodeDataset` 只读取 `exog_fc` | FAIL |
| OOF 统一 key 空间 | 未落地 | blocked loader 无调用；key 网络无训练 loss | FAIL |
| projection 只执行一次 | API 表面修正 | 单次 API 存在，但未解决随机 metric | PARTIAL |
| S3 leave-one-day-out | 只有检索排己 | 校准评分不是 held-out day realized gain | FAIL |
| S4 统一时间戳 | 只有 manifest 雏形 | runner 仍 trim/pad；manifest 不连接各方法 keyed rows | FAIL |
| freeze/reload | 只有模型权重 round-trip | hash 不含 memory/校准/manifest；无 target-free predict API | FAIL |
| 官方基线真实性 | 文档部分修正 | 类名与结果名仍写 Official/PIR，缺强制 provenance 状态 | PARTIAL |
| 22 个真实契约测试 | 不成立 | 多项为 stub、弱断言或错误对象 | FAIL |
| machine-readable evidence | 未完成 | smoke 只保存汇总 JSON，无逐时间戳 action/retrieval/config | FAIL |

---

## 2. P0：先修仓库不能运行的问题

### P0-1. `host_cache.py` 必须先通过编译

当前文件在 `SEED = 0` 后存在意外缩进，导致：

```text
IndentationError: unexpected indent (host_cache.py, line 29)
```

修复要求：

- 模块级常量 `V2_BACKBONES/HERE/CACHE_ROOT` 回到正确缩进；
- 删除模块导入时执行的第一套重复 argparse；
- 只保留 `main()` 内一套 CLI；
- `--dataset`、`--backbone`、`--resume`、`--seed` 必须真实生效；
- import 模块时不得创建目录、解析参数或启动训练；
- 输出以 timestamp keyed rows 或 full-axis prediction + valid map 为准。

验收：

```powershell
python -m compileall src experiments/08-hch-v2 experiments/00-data-exploration/math_loss
python experiments/08-hch-v2/host_cache.py --help
```

两条均应返回 0。不得继续用“CLI deferred”算 PASS。

### P0-2. 把仓库级 compile 纳入测试

当前测试 01 只遍历 `src`，因此没有发现实验入口语法错误。改成至少覆盖：

- `src/**/*.py`；
- `experiments/08-hch-v2/**/*.py`；
- `experiments/00-data-exploration/math_loss/**/*.py`；
- 本轮实际调用的同行 baseline adapter。

允许归档目录不进入测试，但活跃目录不得跳过。

---

## 3. P0：建立唯一的日期划分与评估清单

### 3.1 当前存在两套不一致划分

宿主使用：

```python
valid = build_tabular(...)[3]
seg = four_segment_split(len(valid))
```

HCH daily loader 使用：

```python
splits = date_based_split(ds)
```

前者在 192 小时 warm-up 后按有效小时比例切，后者在完整原始日期上切。两者边界不一致，可能导致：

- HCH 的前几天 S2 落进宿主 S1，宿主预测并非样本外；
- S3/S4 边界错开；
- 同一日期被宿主和校正器赋予不同 split；
- 当前 manifest 虽名为统一，实际仍与宿主 `seg` 不同。

### 3.2 唯一合法流程

在 `src/eval_manifest.py` 或一个同等轻量文件中建立 `ExperimentManifest`，不要再新增复杂目录：

1. 从原始 timestamp 得到完整日历；
2. 显式处理 23/25 小时和缺失日；
3. 先按完整日期划 S1/S2/S3/S4；
4. 再把 `valid_idx`、宿主序列 warm-up 和各方法可用性映射进同一清单；
5. 宿主只用 manifest 中的 S1 keyed rows 训练；
6. HCH 只用同一 manifest 的 S2/S3；
7. 所有方法只在共同 S4 `(dataset, market, target, timestamp, horizon)` 上评分。

manifest 至少包含：

- `dataset_id`、`market_id`、`target_id`；
- timezone、DST policy；
- timestamp、date_id、horizon；
- raw index、valid index；
- split；
- host/method availability；
- data hash、split hash。

### 3.3 删除 trim/pad 对齐

必须删除 `smoke_v2.py` 中：

```python
hch_pred = hch_pred[-n_manifest:]
np.pad(...)
```

每个输出行必须携带键，最终用严格 inner join：

```text
(dataset_id, market_id, target_id, timestamp, horizon)
```

若缺行：

- 报告每方法缺失键；
- 正式评分前 fail closed；
- 不允许尾部截取、长度相等即视为对齐或静默 NaN。

### 3.4 必测

- 宿主 S1 与 HCH S2 日期无交叠；
- S1/S2/S3/S4 边界由同一 split hash 指认；
- 五种同行方法与 HCH 的 S4 key set bit-exact 相同；
- 打乱某方法数组顺序但保留 timestamp 后，join 结果不变；
- 删除一个 timestamp 后，正式评估必须失败；
- 23/25 小时日对所有方法采取同一策略。

---

## 4. P0：重写 DailyEpisodeBatch 的原始/模型双通道

### 4.1 当前问题

`DailyEpisodeBatch` 只有：

- normalized `host_pred`；
- normalized `target`；
- exog；
- time features；
- date ids。

这会导致：

- 无法在不反推 scaler 的情况下使用经济零点；
- 后续 IAH 数学可能误在均值中心化坐标中判断正负价格；
- `predict_s4` 被迫携带 target 字段；
- 输出无法证明 raw candidate 与模型坐标 candidate 一致。

### 4.2 目标 schema

本轮只建立数据契约，不实现 IAH：

```python
@dataclass
class DailyEpisodeBatch:
    host_raw: Tensor              # [B,H,1]，原币种
    host_model: Tensor            # [B,H,1]，仅 encoder 使用
    target_raw: Tensor | None     # [B,H,1]，S2/S3 train/eval；S4 predict 为 None
    target_model: Tensor | None   # 仅旧训练兼容，未来可删除
    exog_value: Tensor            # [B,H,N,1]，S1 规范化后
    exog_type: Tensor             # [B,H,N]，稳定类别 ID
    exog_mask: Tensor             # [B,H,N]
    lag_context: Tensor           # past price/residual/actual，严格 cutoff-safe
    time_feat: Tensor
    market_id: Tensor
    target_id: Tensor
    timestamps: ...
    date_ids: ...
```

允许为简单实现调整字段名，但 raw/model、target optional、feature type 和 keyed metadata 四个原则不能丢。

### 4.3 经济零点

- `host_raw=0` 与 `target_raw=0` 必须保持为零；
- 所有正负价判断只在 raw 坐标；
- encoder 的标准化可以中心化；
- 数学 loss、inverse transform、action gain 由未来数学融合文档决定；
- 本轮测试 raw side channel 的零点与单位，不引入 epsilon/floor/clip。

### 4.4 真正的 target-free 推理

新增独立的 `InferenceEpisodeBatch`，或允许 `target_raw=None`：

```python
predict_s4(batch_without_target, frozen_bundle)
```

函数签名、对象字段和执行图中均不能出现 S4 label。不能再以“输出字典没有 y_true”代替无标签证明。

---

## 5. P0：外生变量与历史上下文

### 5.1 S1-only 规范化

当前 `DailyEpisodeDataset.__getitem__` 对每个目标日的 24 点外生序列重新计算均值/方差。这不是文档声称的 S1-only normalization，并且会消除需求/风光水平信息。

修复：

- 每市场、每 `feature_id` 只用 S1 拟合 center/scale；
- scaler 封入 bundle；
- S2/S3/S4 只 transform；
- NaN 处理策略和 availability 单独记录；
- 不允许用测试日自身统计量拟合 scaler。

若研究上确实要 day-wise instance normalization，必须作为显式可见特征的变换版本，并保留 absolute-level token；不能暗中替换 S1 scaler。

### 5.2 feature type 目前未真正进入 embedding

当前数据把 `j+1` 写入 exog 第二维，但 `HCHV2.encode` 没把 `exog_type` 传给 `HourTokenEncoder`，因此 `var_type embedding` 实际未用。

修复：

- type id 从 value tensor 分离；
- `HCHV2.encode` 显式传递；
- 不把类别 ID 当连续数值输入 Linear；
- feature vocab/semantic group 写入 manifest；
- arbitrary number of exog 用 mask/padding 支持。

### 5.3 learned-null 目前未被测试

无外生数据时，dataset 把一个全零 token 标为 valid，导致 learned-null 分支不执行。测试 09 也传入全 1 mask。

修复：

- 无外生：`N=0`，或 dummy token 的 mask=0，由 encoder 插入 learned-null；
- 任意 batch row 全 mask 时，也必须逐行插入/替换 null，不能只看整个 batch 的 `mask.sum()`；
- 防止 MHA all-masked NaN。

测试必须覆盖：

- 整个 batch 无 exog；
- 同一 batch 中一行有 exog、另一行无 exog；
- masked value 改成 ±1e9，输出不变；
- null token 收到合法梯度。

### 5.4 实际量与残差历史

当前 HCH data layer 完全忽略 `exog_act`。加入：

- actual exog 只用 lag≥24h；
- host residual history 只用已经发生的 `y_{t-L}-hat y_{t-L}`，默认 L≥24h；
- 昨日、前周同小时等周期信息通过统一 lag context 输入，不手工使用当前标签；
- availability manifest 逐字段验证 cutoff。

NEM 的 demand 属于 actual；山东“实际值”同样必须 lag。山东当天可提前获得的“预测值”可作为 `exog_fc`。

---

## 6. P0：冻结 bundle 必须覆盖全部决策状态

### 6.1 当前 hash 不完整

`HCHV2Bundle.hash()` 目前只遍历 `model_state`。它没有覆盖：

- memory keys/gains/dates；
- `k/eta/tau` 与 gate mode；
- S1 scaler/state stats；
- feature availability；
- data/split/eval index；
- candidate definition；
- upstream baseline provenance。

因此 memory 或路由参数变化后 hash 仍可能不变。

### 6.2 校准参数 reload 丢失

`calibrate_s3` 修改 `self.dvg.cara_eta`、`self.dvg.kl_tau`，但它们不是 Parameter/Buffer；`from_bundle` 又从 config 构造 DVG。当前 config 并未可靠同步全部校准值。因此 smoke 报出的 best calibration 不等于 reload 后实际使用的 calibration。

修复：

- 把所有决策配置写入显式 serializable calibration object；
- freeze 时复制而非引用 mutable config；
- bundle deep hash 覆盖上述所有项目；
- reload 时逐字段验证；
- 缺字段/version/hash mismatch 时 fail closed。

### 6.3 正确 round-trip 测试

测试流程：

1. 建一个至少含 3 个 memory day 的模型；
2. 设非默认 calibration；
3. target-free batch 做一次预测，保存 candidate、neighbors、action、final；
4. freeze/save；
5. 新进程 load；
6. 同一 batch 重算；
7. 所有输出一致；
8. 修改一个 memory gain、date、eta 或 scaler，bundle hash 必须变化；
9. S4 推理前后 hash 完全一致。

删除 `b.hash()==b.hash()` 这种恒真测试。

---

## 7. P0：旧 S3 calibration 的逻辑错误

这一节用于修复旧原型的事实错误；未来 IAH 可能删除整个旧 DVG。

### 7.1 当前错误

当前 LODO 循环：

1. 排除 day i；
2. 用邻居 gain 估计动作；
3. 又用邻居平均 gain 给该动作打分；
4. 只平均其中正 gain，负 gain 被丢弃。

这不是 held-out calibration。正确评分必须看被排除 day i 上该选中动作的 realized gain。

此外：

- soft-hard 下 `tau` 只改变 softmax 概率，不改变 argmax，网格搜索 `tau` 没有决策作用；
- calibration 后 bundle 可能恢复默认 eta/tau；
- memory key 网络仍是随机的，见下一节。

### 7.2 临时修复或隔离

执行 AI 二选一，优先隔离：

**方案 A（推荐）**：把旧 `CAGMMemory+DVG` 标记为 `legacy_unvalidated`，正式 runner 遇到它直接拒绝；只保留代码供对照，等待 IAH 路由替换。

**方案 B（若必须维持旧 smoke）**：

- 选择动作只用 day i 之外的邻居；
- 评分使用 `gains_pool[i,h,chosen_action]`；
- 所有正负 realized gain 都进入平均；
- 只搜索能改变实际决策的参数；
- 每个 fold 输出 selected action、held-out gain、neighbor ids；
- calibration 配置进入完整 bundle。

即使选择 B，旧随机 metric 未解决前也只能标工程 smoke，不能作实验。

---

## 8. P0：CAGM key/OOF 声明不实

### 8.1 调用图

当前 S2 `train_step` 只调用：

```text
encode → state_head → biomc → candidate_loss + state_loss
```

没有调用：

```text
memory.key_net / cand_proj / fusion / metric_proj
```

因此这些层没有来自 loss 的梯度。`encode_key` 在 S3 才被调用，参数仍是随机初始化。

`build_blocked_s2_loaders` 虽存在，但没有被 active smoke/training 使用。测试 14/15 只证明两个 API 在同一初始化下数值一致，不能证明 key 被 OOF gain 训练。

### 8.2 本轮处理

由于新数学方案可能用确定性 atom distance 删除旧 metric，本轮不要重新投入实现旧 gain metric。请：

- 在旧类和 runner 上加 `legacy_untrained_metric` 状态；
- 正式/half-exp runner 禁止选择；
- 删除“OOF gain-aware metric 已完成”的当前结论；
- 保留源码等待融合后统一删除或迁移。

如果执行 AI 擅自训练旧 metric，本轮视为越界。

---

## 9. P0：纠正数学证据审计

仓库 `experiments/00-data-exploration/math_loss` 的结果只能作为待复核探索，不能继续标“audit-backed revision”。

### 9.1 残差符号统一

项目唯一 residual 定义：

$$
r=y-\hat y^{host}.
$$

当前脚本多处使用 `pred-y`，而 evaluator 又按 `y-host` 的公式解释。统一变量名、docstring、candidate sign 和测试。

### 9.2 Candidate audit 是 oracle，必须作废

当前 Phase C：

- 在 S3 自身估计 `pi/m`；
- 用 S3 的真实 residual sign 决定何时应用 Down/Up；
- 再在同一 S3 计算 gain。

这不是可部署 candidate，也不能证明 partial moment harm rate 低。必须：

- 将现有 `06_CANDIDATE_ACTIONS.csv` 标为 `INVALID_ORACLE_DIAGNOSTIC`；
- 从 executive verdict 删除“partial moment viable”的证据性结论；
- 若重做：只在 S2 拟合规则，S3 候选只能由 pre-outcome feature 产生，禁止用 S3 residual sign 路由；
- oracle 结果可以单列上界，但名称必须含 `oracle`。

### 9.3 Student-t CDF 实现错误

当前 `student_t_cdf` 在正尾趋近 0.5，而正确 CDF 应趋近 1；负尾同理。例如 ν=5、x=5：

- 当前实现约 0.502；
- 正确值约 0.998。

修复后与 `scipy.stats.t.cdf` 在：

- x∈{-20,-5,-1,0,1,5,20}；
- ν∈{2.1,3,5,10,100}

逐项比较。

### 9.4 Bootstrap 重复样本被折叠

当前 block bootstrap 使用：

```python
sampled = rng.choice(days, replace=True)
mask = np.isin(day_ids, sampled)
```

`np.isin` 会丢掉重复抽中的 day multiplicity，不是 bootstrap。必须按 sampled day 逐块拼接或按抽样计数加权。

同时修复：

- paired 方法差必须在共同日期上配对重抽；
- 不同方法不能独立抽两套日期后相减；
- 输出 seed、block unit、有效天数。

### 9.5 日依赖矩阵错误

当前实现：

- 把未减小时均值的 `E[r_h r_k]` 当 covariance；
- 再由该矩阵构造“correlation”；
- SVD 后把 singular values 再平方才算解释率；
- 人工 `arange(n)//24` 代替真实日期；
- “effective rank”实际只是特征值比例>0.01的计数。

修复：

1. 用真实 timestamp pivot 为 `day×hour`；
2. 只使用完整日；
3. 每小时跨日中心化；
4. 直接对 covariance/correlation 的 eigenvalues 计算解释率；
5. 报 participation ratio 或 entropy effective rank，并写清定义；
6. signed residual、absolute residual、squared residual 分别报告。

注意：正确重算后日依赖可能仍然很强，但这只能证明依赖存在；不能自动证明“共享 day scale latent”就是正确机制。共享尺度主要制造幅值依赖，不足以解释共同正负偏移。

### 9.6 分布结论降级

当前 Student-t 比较：

- 只覆盖部分 LAGO 组合；
- 是 unconditional residual fit；
- ν 为离散网格、μ 固定 median，不是完整 MLE；
- 未验证 state-conditional 模型；
- 未验证 W2 likelihood。

因此重命名为 `coarse_unconditional_fit`，不能写“验证了 M0+W2”。新 IAH 数学也不依赖这一结论。

### 9.7 新证据报告

生成全新文件：

`experiments/00-data-exploration/math_loss/outputs/00_EXECUTIVE_EVIDENCE_REAUDIT_20260812.md`

旧文件保留并在开头加 superseded/invalid 标记，不覆盖。

---

## 10. P1：测试套件必须证明真实契约

替换以下弱测试：

| 当前测试 | 为什么无效 | 新测试 |
|---|---|---|
| 04 `pass` | 完全未测试 CLI | subprocess `--help` + tiny cache + keyed rows |
| 07 用零 state target | 只证随机梯度存在 | 暂时标 legacy；不再声称目标正确 |
| 09 mask 全 1 | 没走 null | 全 mask 与混合 row null |
| 11 用 LAGO_DE | 没有 actual exog | NEM/Shandong lag manifest |
| 12 只比 DA/RT timestamp | 不证明无交叉真值 | feature lineage + forbidden source audit |
| 14/15 | 不证明 metric 受训练 | 标 legacy_untrained；禁止 formal |
| 16 随机 query 排 index0 | 只证 scatter | query=memory[0] 且结果绝不含自身 |
| 17 hash 与自身相等 | 恒真 | 修改 bundle 任一组件 hash 改变 |
| 18 batch 仍含 target | 不证明 target-free | `target=None` 的独立 predict API |
| 19 只比模型 hash | 不比决策输出 | 新进程 prediction/neighbors/action round-trip |
| 20 只比名称 | 不查 provenance | official/limited/unsupported fail-closed |
| 21 只复现一个 encoder call | 不证明 run | tiny end-to-end manifest/bundle 两次一致 |
| 22 只看 timestamp/hash 非空 | 不查各方法集合 | 多方法 keyed set equality |

新的测试报告不得写“22/22”，除非每个名称、断言和原始日志一一对应。跳过、deferred、环境缺依赖均不计 PASS。

---

## 11. P1：同行基线标签与固定方法集合

项目方法集合仍固定：

1. Identity；
2. Residual-L1；
3. QuantileResidual-LGBM；
4. PIR；
5. δ-Adapter Ada-Y；
6. HCH v2。

但当前 PIR 明确没有官方 retrieval，δ-Adapter 改了样本、归一化和训练流程，因此当前实现只能是 limited reimplementation。

代码要求：

```python
implementation_status in {
    "official",
    "limited_reimplementation",
    "unsupported_official"
}
```

- class 名称也不得含误导性 `Official`；
- report label 自动由 status 生成；
- official 运行失败时不得回退到 limited 并保留 official 名称；
- half-exp 前必须完成官方入口或明确 unsupported；
- 当前 contract smoke 可以不跑六方法，但报告必须写“只测基础通路”，不能暗示同行比较完成。

---

## 12. 暂缓修改的模块

数学窗口正在裁决 IAH-CRPS。以下旧模块很可能被重写/删除，本轮不要重复投资：

- `ContinuousStateHead/state_loss`；
- BCE occurrence + Smooth-L1 magnitude + location W1；
- `key_net/cand_proj/fusion/metric_proj`；
- CARA `eta`、KL `tau`、softmax temperature；
- 固定 k 的旧 DVG；
- 旧 memory 只存 raw gain 的 schema。

处理方式：

- 保留源码；
- 标 `legacy_unvalidated`；
- 不允许 formal runner 选用；
- 等数学窗口 v0.2 返回后由主架构窗口给出唯一迁移设计。

但它们依赖的数据、split、raw-price、freeze 和 evidence 基础必须在本轮修好。

---

## 13. 文档治理的最小修正

仓库当前入口文档仍有相互冲突：

- AGENTS.md 的宿主/基线与 v2 已锁定集合不同；
- `docs/paper_info/README.md` 写山东“主实验”，而当前决策是公开数据承担可复现主证据、山东作私有真实场景外部验证；
- 旧 report 写“OOF gain metric 已完成”，源码并未实现。

不要重构文档树，只做：

1. 在 AGENTS.md 顶部增加“v2 当前 canonical docs”指针；
2. 更新固定宿主、固定同行方法和山东角色；
3. 给被推翻报告加 superseded banner；
4. 不删除历史记录。

---

## 14. 本轮允许的验证

仅允许：

- 全仓 compile；
- unit/contract tests；
- tiny synthetic data；
- 每种数据契约各抽 2–5 天；
- Linear/PatchTST 各一个 target-free freeze/reload smoke；
- S2/S3-only 数学 evidence re-audit。

禁止：

- S4 指标调参；
- half-exp；
- 全方法/全市场性能比较；
- IAH-CRPS 实现；
- Student-t/W2 实现；
- 更新论文 SOTA 主张。

---

## 15. 验收 Gate

### Gate F0：可运行

- [ ] active Python 全部 compile；
- [ ] host_cache import 无副作用；
- [ ] CLI tiny run 可复现。

### Gate F1：数据

- [ ] 唯一 date-first split manifest；
- [ ] host S1 与 corrector S2 无交叠；
- [ ] raw/model 双通道；
- [ ] target-free S4 batch；
- [ ] exog S1 scaler、type、mask、null、actual lag；
- [ ] past residual context cutoff-safe。

### Gate F2：评估

- [ ] 禁止 trim/pad；
- [ ] keyed S4 set equality；
- [ ] DST 同策略；
- [ ] 缺键 fail closed。

### Gate F3：冻结

- [ ] full bundle hash；
- [ ] calibration/memory/scaler/split 均持久化；
- [ ] 新进程决策 round-trip；
- [ ] S4 前后 hash 不变。

### Gate F4：证据

- [ ] residual 符号统一；
- [ ] oracle candidate 明确作废；
- [ ] Student-t CDF 修正；
- [ ] block bootstrap 保留 multiplicity；
- [ ] 日依赖用真实日期和中心化 covariance；
- [ ] M0/W2 结论降级；
- [ ] 新 re-audit 文件完成。

### Gate F5：诚实状态

- [ ] weak/stub tests 已替换；
- [ ] legacy HCH 禁止 formal；
- [ ] baseline provenance fail-closed；
- [ ] 未执行 IAH 或 half-exp。

最终状态只能是：

- `READY_FOR_MATH_ARCHITECTURE_FUSION_REVIEW`；
- `NOT_READY`。

不得写 `READY_FOR_HALF_EXP`。

---

## 16. 执行 AI 回传格式

新建：

`experiments/08-hch-v2/results/v2_foundation_repair_handoff_20260812.md`

严格包含：

```markdown
# HCH v2 Foundation Repair Handoff
## 1. 起止 commit 与环境
## 2. P0/P1 修改映射
| ID | 文件/符号 | 修改 | 真实测试 |
## 3. Compile 与 CLI
## 4. 唯一 split/evaluation manifest
| dataset/target | S1 | S2 | S3 | S4 | split hash |
## 5. Raw/model/target-free 数据契约
## 6. Exog 与 lag lineage
## 7. Bundle round-trip 与 mutation hash
## 8. 数学 evidence re-audit
| 原结论 | 缺陷 | 修正后结论 |
## 9. Baseline provenance
## 10. Legacy quarantine
## 11. 所有测试及原始日志
## 12. 未解决项
## 13. 最终状态
READY_FOR_MATH_ARCHITECTURE_FUSION_REVIEW / NOT_READY
```

同时回传：

- `git diff --stat`；
- 修改文件列表；
- 全部测试原始日志；
- 一个 target-free batch schema 示例；
- 一个完整 bundle manifest；
- 一个 keyed prediction 示例；
- 修正前后 evidence 对照；
- 不得把“未运行”写成 PASS。

完成后停止，等待主架构窗口复审。
