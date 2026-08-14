# HCH-v2 Phase 4：CAVM 代码建设文档 v0.1

**用途：** 交给代码 Agent 执行 Phase4 的最小实现和诊断。

**重要：** 本文档只允许在现有 v0.4 代码通过基线复现后，增加可选 CAVM 分支。不得直接改写默认 `v0.4-core`。

**代码基线：**

- `main@5182ef4`：当前 v0.4 核心代码；
- `exp/r1b-screening-20260813@63b158a`：Phase3 实验和总报告；
- 数学唯一依据：`hch_v2_iah_crps_final_math_core_v0.3_2026-08-12.md`。
- Phase3 权威记录：`docs/训练文件夹/对比实验/hch_v2_phase3_results_master_report_v0.1_2026-08-14.md`。

---

## 1. 先读的现有文件

执行前必须读取并理解以下文件，不得凭文件名猜测接口：

| 文件 | 当前职责 | CAVM 处理原则 |
|---|---|---|
| `src/iah_candidate.py` | v0.4 Core + Data Signature + IAH 三原子 | 不改数学输出 |
| `src/hch_v2_context.py` | CoreContextEncoder、FiLM、optional branch | 只复用已有输出，不新增事件头 |
| `src/iah_crps_loss.py` | 单一 IAH-CRPS | 不改 |
| `src/universal_trainer.py` | 等域采样、宏平均 S2V 选点 | 不改训练目标和采样轴 |
| `src/w1_retrieval.py` | residual atom measure、W1、CAGM memory | 保留为 W1-only 控制组 |
| `src/query_replay.py` | query-dose replay、directional gain、final pi | CAVM 必须复用 |
| `src/double_event.py` | 双事件提案 | 不改 |
| `src/dvg_calibrate.py` | 整日 DVG 和 LCB | 第一版不改 q 的数学 |
| `src/hch_v2_pipeline.py` | 正式 S1→S2→S3-M→S3-C→S4 编排 | 以开关形式接入 CAVM |
| `src/hch_v2_bundle.py` | universal/local bundle 与 hash | 增加可选 CAVM state |
| `experiments/08-hch-v2/` | smoke、审计、Phase3 运行脚本 | 新增独立 Phase4 脚本，不覆盖旧结果 |

必须确认：正式 pipeline 不导入 `src/_legacy/` 路径。

---

## 2. 实现边界

### 2.1 不允许修改的内容

- `iah_crps_loss.py` 的公式和调用方式；
- `IAHCandidateHead` 的三原子参数化；
- `double_event_proposal()` 的数学逻辑；
- `DGVSplitConformal` 的现有 q/LCB 公式；
- `UniversalCoreTrainer` 的等域采样；
- 默认 `memory_mode="w1"` 的结果；
- S4 的 target-free 接口。

### 2.2 允许新增的内容

- 一个新的 `src/context_action_memory.py`；
- pipeline 的可选 `memory_mode="cavm"`；
- CAVM 专用实验脚本和测试；
- bundle 中可选的 CAVM local/global state；
- evidence JSON 中的 context distance、memory scope、effective count 等审计字段。

不需要新建复杂 package、插件系统、配置框架或数据库。

---

## 3. 新文件：`src/context_action_memory.py`

建议只增加一个文件，先不要把功能拆成多个目录。

### 3.1 数据结构

```python
@dataclass
class CAVMExperience:
    date: str
    context_key: np.ndarray
    z0: np.ndarray
    w_minus: np.ndarray
    w_zero: np.ndarray
    w_plus: np.ndarray
    m_minus: np.ndarray
    m_plus: np.ndarray
    target_zY: np.ndarray
    valid_mask: np.ndarray
    A_hat: float | None = None
    A_true: float | None = None
    action_error: float | None = None
    timestamp: str = ""
    audit_domain: str = ""
```

说明：

- `target_zY` 只存在于已经揭示标签的经验中，用于历史 replay；
- `context_key` 必须在标签揭示前构造完成；
- `audit_domain` 只用于报告和切分，不参与距离；
- 不保存当前日未来标签到 query key。

### 3.2 ContextKeyBuilder

```python
class ContextKeyBuilder:
    version = "cavm-key-v1"

    def build(
        self,
        candidate: dict,
        core_context: np.ndarray | torch.Tensor,
        valid_mask: np.ndarray | torch.Tensor,
        domain_det: np.ndarray | torch.Tensor | None = None,
        optional_values=None,
        optional_roles=None,
        optional_masks=None,
    ) -> np.ndarray:
        ...
```

实现要求：

1. 只能使用预测前信息；
2. 输出固定维度；
3. 所有统计只在 valid hours 上计算；
4. NaN、Inf、空 optional 必须有显式 fallback；
5. key 维度和版本写入 bundle；
6. 不读取 `target_raw`、`target_zY`、residual 或 action gain。

建议固定组成：

- `z0`：q10/q50/q90/IQR/mean_abs/std/sign_mass；
- `diff(z0)`：mean_abs/std/max_abs/positive_mass/negative_mass；
- `core_context`：按通道 mean/std，并限制为当前 schema 的确定顺序；
- `domain_det`：8 个已有 v0.4 descriptor；
- atom summary：`mean/max(w_minus,w_plus,m_minus,m_plus)`；
- optional：按 role 的 mean/std/missing ratio，缺失时全零并设置 mask。

第一版不使用可训练 Transformer key encoder。若固定 key 有效，后续才可另开实验替换为 learned key。

### 3.3 距离

实现两个独立距离，便于消融：

```python
context_distance(query_key, memory_keys) -> [M]
atom_w1_distance(query_candidate, memory_candidate) -> [M]
```

组合距离：

```python
d = lambda_atom * normalized_w1 + lambda_context * normalized_context
```

要求：

- `lambda_atom=1, lambda_context=0` 必须严格复现现有 W1-only；
- 组合权重只在 S3-M/S2V 选择并冻结；
- 不能看 S4 标签选权重；
- 不使用 market ID；
- 不把标签残差加入 query key。

第一版仍采用现有 top-k 邻居和均匀邻居 replay；不要同时引入 arbitrary weighted replay。

### 3.4 ContextActionMemory

```python
class ContextActionMemory:
    def __init__(self, scope: str, key_builder: ContextKeyBuilder):
        # scope: "global" or "local"

    def add_revealed_day(self, experience: CAVMExperience):
        ...

    def query(self, query_key, query_candidate, k, exclude_date=None):
        # return indices, distances, component distances

    def freeze(self) -> dict:
        ...

    @staticmethod
    def from_frozen(state: dict):
        ...
```

`query()` 的返回值必须包含：

- neighbor IDs；
- composite distance；
- W1 distance；
- context distance；
- source scope；
- effective neighbor count；
- self-exclusion 结果。

---

## 4. 与现有 replay 的衔接

CAVM 不能另写一套 action-value 公式。候选查询后仍然必须按以下顺序：

```text
context/atom retrieval
    → query-dose replay
    → directional gains
    → double_event_proposal
    → final pi replay
    → A_hat
    → existing DVG/LCB
```

直接复用：

- `full_replay_chain()`；
- `estimate_realized_A()`；
- `double_event_proposal()`；
- `DGVSplitConformal.lcb()`。

第一版 CAVM 只改变历史经验的检索候选集合和经验账本，不改 `full_replay_chain` 的内部公式。

`A_true` 只有在当天标签揭示后才能计算。

---

## 5. Pipeline 接入方式

在 `HCHV2UniversalPipeline.__init__()` 中增加可选参数：

```python
memory_mode: str = "w1"  # "w1" | "cavm"
```

默认必须是 `w1`，保证旧实验不变。

建议新增方法：

```python
def fit_cavm_memory(
    self,
    global_days: list[dict],
    local_days: list[dict] | None = None,
    key_builder=None,
) -> dict:
    """Build revealed-only CAVM state; no S4 target is accepted here."""

def observe_outcome(
    self,
    query_id: str,
    target_zY: np.ndarray,
    evidence: dict,
) -> dict:
    """Append one revealed day to local memory after prediction."""
```

### 5.1 `predict_s4()` 约束

`predict_s4()` 必须继续接受：

```python
predict_s4(host_raw, core_context, valid_mask=None,
           domain_det=None, optional_values=None, ...)
```

不得接受 `target_raw`、`target_zY`、`action_gain` 等参数。

### 5.1.1 Universal training 的 Data Signature 注意事项

当前 `HCHV2UniversalPipeline.fit_s1_signature()` 仍保留单域便利路径，会写入
`DataSignature.domain_det` buffer。它只能用于单域 local profile；UniversalCoreTrainer
的多域训练必须继续通过每个 batch 的 `domain_det=[B,d_det]` 显式传递，不能在混合域训练循环中反复调用
`fit_s1_signature()` 覆盖同一个 buffer。

代码 Agent 必须新增一个测试：交错训练域 A/B 后，A/B 的输出只由各自显式 descriptor 决定，最后一次写入的域不能污染另一域。

返回 evidence 至少增加：

```json
{
  "memory_mode": "cavm",
  "context_key_version": "cavm-key-v1",
  "neighbor_scopes": ["global", "local"],
  "neighbor_ids": [],
  "distance_total": [],
  "distance_context": [],
  "distance_w1": [],
  "effective_neighbor_count": 0,
  "A_hat": 0.0,
  "q": 0.0,
  "lcb": 0.0,
  "final_action": "identity",
  "fallback": null
}
```

### 5.2 `observe_outcome()` 约束

只有以下事件发生后才能调用：

1. query 已经完成预测；
2. 预测时间点已经过去；
3. 真实目标完整揭示；
4. 已经计算 `A_true`。

该方法：

- 只能更新 local memory；
- 默认不能更新 universal model parameters；
- 默认不能重新选择 (k)、(lambda) 或 q；
- 必须记录 timestamp 和 update reason；
- 必须支持关闭，从而得到严格 frozen S4。

---

## 6. Bundle 扩展

在 `HCHV2Bundle` 中增加可选字段：

```python
memory_mode: str = "w1"
cavm_key_version: str = ""
cavm_global_state: dict | None = None
cavm_local_state: dict | None = None
cavm_update_policy: dict = field(default_factory=dict)
cavm_global_hash: str = ""
cavm_local_hash: str = ""
```

规则：

- `extract_universal()` 不得包含 target-specific local memory；
- global memory 如果进入 universal bundle，必须明确来源域、时间范围和 leave-one-market-out 审核结果；
- local memory 属于 local package；
- bundle hash 必须覆盖 key version、memory mode、global/local state；
- 旧 v0.4 bundle 没有 CAVM 字段时必须仍可加载，并自动回退 `memory_mode="w1"`；
- round-trip 必须复现最终 action，而不是只复现模型参数。

---

## 7. 代码建设顺序

### P0：先证明旧路径没有被破坏

1. 在 Phase3 分支或最新同步分支上运行现有 v0.4 smoke；
2. 复现 `v0.4-core + W1 + static DVG`；
3. 保存旧结果 hash；
4. 运行旧测试；
5. 只有 P0 通过才能增加新文件。

### P1：实现只读 CAVM

1. 创建 `context_action_memory.py`；
2. 实现 key builder；
3. 实现 global/local memory 序列化；
4. 仅在离线 S3-M/S3-C 构建经验；
5. 不更新 S4；
6. 生成 CAVM evidence JSON。

### P2：接入 composite retrieval

1. 增加 `memory_mode="cavm"`；
2. 运行 W1-only、context-only、context+atom 三组；
3. 旧 W1 结果必须可以逐日复现；
4. 不更改 candidate、replay、proposal、DVG 公式。

### P3：实现 local observe，但默认关闭

1. 添加 `observe_outcome()`；
2. 只追加 local memory；
3. 运行严格 frozen 与 streaming 两种模式；
4. 输出冷启动/稳态曲线；
5. 不在线反向传播。

### P4：仅在实验通过后考虑 action-value state update

如果 P2/P3 已经证明有效，再另开分支研究 context-conditioned DVG q 或经验收缩。未经独立实验，不得把它写入默认 CAVM。

---

## 8. 必须新增的测试

### 8.1 信息隔离

- 修改 query target，query key 必须完全不变；
- 将 target/residual/action gain 传入 key builder 必须报错；
- `predict_s4()` 不能接收 target 参数；
- `observe_outcome()` 只能在 reveal 后调用。

### 8.2 市场 ID 隔离

- 只改变 `market_id` 审计字段，候选、key、邻居和最终输出不变；
- 不同市场使用相同上下文时，不能因 ID 直接改变动作。

### 8.3 global/local 隔离

- local update 不能改变 universal model state；
- 关闭 local memory 后结果等于 global-only；
- 清空 memory 后必须 fallback Identity。

### 8.4 时间顺序

- 当日预测前不能看到当日经验；
- 当日标签揭示后追加经验；
- 删除最后一天标签后，前一天预测结果不应改变。

### 8.5 兼容性

- 旧 v0.4 bundle 可以加载；
- CAVM bundle 保存/加载后 key、邻居、A_hat、q、LCB、final output 一致；
- optional covariates 为空时严格退化到 core-only。

### 8.6 数值健康

- key 无 NaN/Inf；
- 全部无效小时时安全 fallback；
- 邻居不足时明确记录 fallback；
- 不出现 silent truncation；
- 3 个 seed 均无 NaN/scale-invalid 异常增加。

---

## 9. 禁止代码 Agent 自行扩展

本轮禁止自行加入：

- 新 loss；
- BCE/event head；
- learned market embedding；
- hard price threshold；
- new expert/MoE；
- policy gradient/RL；
- S4 在线调参；
- S4 标签回写训练 universal core；
- 自动把新市场并入 global training；
- 删除或覆盖现有 Phase3 结果。

如果实现过程中发现 CAVM 需要上述内容才能工作，应停止并生成问题报告，而不是自行扩大范围。

---

## 10. AI 执行后必须返回

1. 修改文件清单；
2. 未修改文件清单；
3. 默认 W1 路径前后逐日 diff；
4. CAVM key schema 和维度；
5. global/local memory 的时间范围；
6. 所有测试命令和结果；
7. 是否读取 target 的审计结果；
8. smoke 的 loss/CRPS/MAE、candidate alive rate、action rate；
9. round-trip hash；
10. 明确说明：没有改动 IAH-CRPS、double-event、DVG 数学公式。
