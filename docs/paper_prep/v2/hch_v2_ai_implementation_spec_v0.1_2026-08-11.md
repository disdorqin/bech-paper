# HCH v2 AI 代码构建规范 v0.1

> 日期：2026-08-11  
> 目标仓库：`disdorqin/bech-paper`  
> 配套设计：`hch_v2_final_architecture_design_v0.5_2026-08-11.md`  
> 用途：把本文直接交给代码 Agent。它应能在不重构仓库、不覆盖 v1 的前提下，实现 HCH v2、跑完全部数据与宿主，并产出可审计报告。

## 0. 给执行 Agent 的任务

在现有仓库上新增 HCH v2 实验，不修改 v1 的定义与历史结果。先实现可验证的最小完整闭环，再运行全量矩阵。

必须交付：

1. HCH v2 的 Bi-OMC 候选生成器、CAGM 记忆和 DVG 路由；
2. 5 个固定冻结宿主；
3. 5 个固定同行/对照方法；
4. 所有可用公开数据集与山东私有外部验证；
5. 泄漏检查、单元测试、冒烟实验、全量实验、稳定性实验和消融；
6. 带时间戳的配置、逐行结果、聚合表和 Markdown 报告；
7. 代码修改清单、失败组合清单、复现实验命令。

优先级固定为：

> 数据与评估正确性 > 官方基线真实性 > 全矩阵覆盖 > 指标表现 > 运行速度。

禁止为了得到漂亮结果而改变切分、删除失败组合、使用测试标签调参或把近似复现写成“官方”。

## 1. 开工前必须阅读

按顺序读取：

1. 根目录 `AGENTS.md`；
2. `src/common.py`：数据集、时间切分、现有指标；
3. `src/backbones.py`：现有冻结宿主；
4. `src/selective_hurdle.py`：v1，仅用于理解，禁止覆盖；
5. `experiments/02-hch-v1/run_v1.py`；
6. `experiments/02-hch-v1/results/v1_full_report_20260810_163830.md`；
7. `experiments/07-route-e/peers/quantile.py`；
8. `experiments/07-route-e/peers/repro_pir.py`；
9. `experiments/07-route-e/peers/delta_adapter.py`；
10. 配套的 HCH v2 架构文档 v0.5。

先输出一次“仓库事实核对”，至少确认：

- 数据集实际数量和字段；
- 各市场时间分辨率、时区与缺失情况；
- 当前宿主是否真包含 TCN 与 PatchTST；
- PIR、δ-Adapter 当前文件是官方接入还是本地近似；
- 山东预测特征与实际特征的可用时点；
- 所有历史 v1 文件保持不变。

若仓库事实与本文冲突，停止该项实现并在报告中列出冲突，不可静默猜测。

## 2. 最小文件方案

保持当前仓库结构，不新建复杂框架。建议只新增或最小修改：

| 路径 | 动作 | 责任 |
|---|---|---|
| `src/hch_v2.py` | 新增 | Bi-OMC、CAGM、DVG 和总模型 |
| `src/hch_v2_data.py` | 新增 | 24 小时情节、外生 token、DST 与可用性处理 |
| `src/backbones.py` | 最小扩展 | 补真实 TCN、PatchTST，保留所有 v1 类 |
| `experiments/08-hch-v2/config_v2.json` | 新增 | 固定数据、宿主、方法、随机种子与网格 |
| `experiments/08-hch-v2/baselines_v2.py` | 新增 | Identity、Residual-L1、QuantileResidual-LGBM |
| `experiments/08-hch-v2/official_adapters.py` | 新增 | PIR、δ-Adapter 官方仓适配层 |
| `experiments/08-hch-v2/prepare_official.py` | 新增 | 拉取/校验固定 commit，不自动替换 |
| `experiments/08-hch-v2/run_v2.py` | 新增 | smoke/full/stability/ablation 统一入口 |
| `experiments/08-hch-v2/test_contracts.py` | 新增 | 形状、因果性、切分、冻结、指标测试 |
| `experiments/08-hch-v2/results/` | 新增 | 只写带时间戳的新产物 |

不得：

- 改写 `experiments/02-hch-v1` 的脚本或结果；
- 把 v2 逻辑塞回 `selective_hurdle.py`；
- 新建第二套仓库；
- 为形式完整而拆分大量空文件；
- 覆盖无时间戳的历史结果。

## 3. 冻结的实验对象

### 3.1 五个宿主

宿主固定：

1. Linear；
2. MLP；
3. LSTM；
4. TCN；
5. PatchTST。

要求：

- 宿主只在 S1 训练；
- S1 结束后参数永久冻结；
- 后处理方法获得完全相同的宿主预测缓存；
- TCN 必须是带 dilation 的因果卷积网络；
- PatchTST 必须实现 patching、channel-independent encoder 与预测头，不能用当前普通 Transformer 改名；
- 每个宿主记录参数量、训练时长、推理时长和随机种子；
- 宿主不是同行后处理基线，但必须覆盖传统到前沿的不同误差形态。

### 3.2 五个固定对照方法与 ours

主表固定为：

1. Base / Identity；
2. Residual-L1；
3. QuantileResidual-LGBM；
4. PIR（官方）；
5. δ-Adapter Ada-Y / output-side（官方）；
6. HCH v2（ours）。

除非用户另行明确变更，不得替换、删除或追加主表方法。额外方法只能进附录。

### 3.3 官方基线固定来源

PIR：

- 官方仓：[ustc-time-series/PIR](https://github.com/ustc-time-series/PIR)；
- 固定 commit：`fc372bb02090da887d4a20b614a6cfecbfd813d0`。

δ-Adapter：

- 官方仓：[Anoise/Adapter](https://github.com/Anoise/Adapter)；
- 固定 commit：`0add06ea7b4d2e0a84c364a8be72eef2676a92f2`；
- 只接 output-side / Ada-Y 对应实现。

官方接入规则：

1. 把第三方代码放到 `experiments/08-hch-v2/vendor/` 或通过外部路径加载；
2. 记录仓库 URL、commit、依赖版本、入口类/脚本和必要补丁；
3. 先复现官方最小样例，再接统一缓存接口；
4. 只允许输入/输出、数据形状和训练调度适配，不改其核心损失；
5. 某个数据×宿主组合不受官方实现支持时，写 `unsupported_official`；
6. 现有 `repro_pir.py` 和 `delta_adapter.py` 是本地近似，不得在主表标为官方，也不得静默回退。

## 4. 数据契约

### 4.1 先按日期切分，再构造 24 小时情节

每个市场按时间排序后，以日期为原子单位切分：

- S1：宿主训练；
- S2：HCH 候选生成器训练；
- S3：记忆构建、门控校准和所有超参数选择；
- S4：一次性测试。

比例可以随数据集配置，但必须满足：

- 四段严格连续且无交叠；
- 同一天 24 个小时不能跨段；
- 标准化器、分位数、经验 CDF、缺失填充值只在相应训练段拟合；
- 测试长度按数据集合理比例决定，而非强制统一天数；
- 配置和最终报告必须写明每段日期与样本数。

第一版对 DST 的处理：

- 23/25 小时日不直接 reshape；
- 将这些日从 day-episode 训练与评估中统一排除；
- 报告每个市场排除的日期与比例；
- 原始小时级缓存保留，便于下一版改为显式 mask；
- 若排除比例异常，直接标红，不得静默继续。

### 4.2 样本结构

建议数据类：

~~~python
@dataclass
class DailyEpisodeBatch:
    host_pred: Tensor       # [B, 24, 1]
    target: Tensor          # [B, 24, 1]，训练/评估时才可读
    known_exog: Tensor      # [B, 24, N, D]
    exog_mask: Tensor       # [B, 24, N]
    time_feat: Tensor       # [B, 24, Dt]
    market_id: Tensor       # [B]
    target_id: Tensor       # [B]
    date_id: list[str]
~~~

无外生变量时：

- `known_exog` 输入一个 learned-null token；
- mask 仍保留；
- 禁止通过补零让模型混淆“真实值为 0”和“变量不存在”。

外生变量数量 N 必须可变，核心时刻 token 对任意数量外生 token 做 cross-attention。

### 4.3 山东字段可用性

日前/实时电价均作为目标通道样本。可在预测日前已知的预测列可进入外生 token，例如：

- 地方电厂总加预测值；
- 联络线受电负荷预测值；
- 风电、光伏、核电预测值；
- 自备机组、试验机组预测值；
- 直调负荷、竞价空间、新能源预测值。

带“实际值”的列只能在其真实发布后作为滞后信息。第一版统一要求至少 lag 24h，并按业务时戳复核。

严禁：

- 同一自然日的实时目标真值帮助日前样本；
- 日前目标真值帮助同日实时样本；
- 用当天实际负荷/新能源结果作为 D-1 预测输入；
- 先池化日前与实时，再随机拆分。

正确流程：

1. 先按日期切 S1–S4；
2. 再把日前、实时构造成两个 `target_id` 通道；
3. 两通道可共同训练 global module；
4. 模型必须接收 target token；
5. 结果必须分别报告 DA、RT，同时可给 pooled 统计。

山东不可公开，因此只能作为私有真实场景外部验证。可公开市场承担主结论与可复现性。

## 5. HCH v2 接口契约

建议配置：

~~~python
@dataclass
class HCHV2Config:
    d_model: int
    n_heads: int
    n_layers: int
    memory_k: int
    memory_temperature: float
    action_temperature: float
    cara_eta: float
    kl_tau: float
    state_loss_weight: float
    occurrence_loss_weight: float
    magnitude_loss_weight: float
    location_loss_weight: float
    candidate_loss_weight: float
    rarity_weight_cap: float
    gate_mode: str              # soft_soft | soft_hard
    calibration_mode: str       # global_frozen | market_calibrated
    seed: int
~~~

总模型：

~~~python
class HCHV2(nn.Module):
    def fit_candidate(self, s2_loader): ...
    def fit_memory(self, s3_loader, host_cache): ...
    def calibrate_gate(self, s3_loader): ...
    def predict(self, batch, gate_mode=None): ...
    def export_state(self): ...
~~~

`predict` 至少返回：

~~~python
{
    "y_base": ...,              # [B, 24]
    "y_down": ...,              # [B, 24]
    "y_up": ...,                # [B, 24]
    "delta_down": ...,          # <= 0
    "delta_up": ...,            # >= 0
    "state_low": ...,           # [0, 1]
    "state_high": ...,          # [0, 1]
    "action_value": ...,        # [B, 24, 3]
    "action_prob": ...,         # [B, 24, 3]
    "chosen_action": ...,       # 0 / down / up
    "y_final": ...,
    "retrieved_date_ids": ...,
    "retrieval_weights": ...
}
~~~

运行时强制断言：

- `delta_down.max() <= eps`；
- `delta_up.min() >= -eps`；
- `action_prob.sum(-1)≈1`；
- Identity 的原始样本动作收益恒为 0；
- soft-hard 下，若两个修正动作价值均不大于 0，则必须 Identity；
- S4 的日期不能出现在 memory；
- 宿主参数在后处理训练前后哈希一致。

## 6. 模块实现

### 6.1 共享上下文编码

每小时建立核心 token：

\[
q_{d,h}=E_y(\hat y_{d,h})+E_t(t_{d,h})+E_m(m)+E_c(c).
\]

其中 m 是 market token，c 是 target/channel token。

外生变量 token：

\[
e_{d,h,j}=E_x(x_{d,h,j})+E_{\text{type}}(j).
\]

核心 token 对当小时任意数量外生 token 做 cross-attention：

\[
z_{d,h}=\operatorname{CrossAttn}(q_{d,h},\{e_{d,h,j}\}_{j=1}^{N}).
\]

再用一个轻量日内 encoder 在 24 小时维度传播上下文。无外生变量时使用 learned-null token。

第一版不要加入复杂跨日 Transformer；跨日知识由 CAGM 负责。

### 6.2 连续双尾状态

状态是上下文，不是动作。

宿主预测的因果 rank 与 zero anchor 只是可见输入特征：

\[
u^{host}_{d,h}=\widehat F_{d^-}(\hat y_{d,h}\mid m,h),
\qquad z^{host}_{d,h}=\frac{\hat y_{d,h}}
{\operatorname{MAD}(\mathcal H_{d^-})+\epsilon}.
\]

状态头从当前可见上下文预测真实价格的连续坐标：

\[
\hat s^{rank},\hat s^{zero}=f_{state}(z,u^{host},z^{host}),
\]

训练目标只在 S2 构造，且每一天只能使用此前成熟的统计量：

\[
t^{rank}=2\widehat F_{d^-}(y)-1,
\qquad
t^{zero}=\frac{y}{\operatorname{MAD}(\mathcal H_{d^-})+\epsilon}.
\]

由 \(\hat s^{rank}\) 连续表达相对低价—正常—高价，由 \(\hat s^{zero}\) 保留物理负价坐标。两者连同原始 context 共同服务 Down/Up，不离散切分。推理时状态头绝不能读取 y；经验 CDF、MAD 与温度也不能由 S4 得到。

禁止：

- 把状态硬切成低/正常/高三个标签；
- 把低状态直接映射成 Down；
- 把高状态直接映射成 Up；
- 把 y<0 二分类器当整个负价方案。

### 6.3 Bi-OMC 双向候选生成

共享主体产生每个方向的 occurrence 和 magnitude：

\[
p_a=\sigma(o_a(z)),\qquad
m_a=\operatorname{softplus}(r_a(z)), \quad a\in\{-,+\}.
\]

由全期望形式得到：

\[
\Delta^-=-p_-m_-,\qquad
\Delta^+=p_+m_+.
\]

这保证两个方向都存在，同时不依赖硬事件阈值。

监督残差：

\[
r=y-\hat y,\quad
o^-=\mathbb 1[r<0],\quad o^+=\mathbb 1[r>0],
\]

\[
m^-=\max(-r,0),\qquad m^+=\max(r,0).
\]

候选损失至少包含：

\[
\mathcal L_{\text{cand}}
=\lambda_o\sum_a \operatorname{BCE}(p_a,o^a)
+\lambda_m\sum_a o^a\,\operatorname{Huber}(m_a,m^a)
+\lambda_y\sum_a w_a\,\ell(\tilde y^a,y).
\]

为了直接约束一天内的尖峰/低谷位置，把每个方向的真实残差质量与候选质量归一化：

\[
p_h^a=\frac{r_h^a}{\sum_jr_j^a+\epsilon},\qquad
\hat p_h^a=\frac{|\Delta_h^a|}{\sum_j|\Delta_j^a|+\epsilon},
\]

并加入一维 Wasserstein 位置损失：

\[
\mathcal L_{loc}^a
=\sum_h\left|\operatorname{CDF}_{p^a}(h)
-\operatorname{CDF}_{\hat p^a}(h)\right|.
\]

当某方向当天真实质量为 0 时，该方向的 location/magnitude 项 mask 掉，但 occurrence 项仍训练。幅值项优先在 \(\log(1+x)\) 空间使用 Huber，降低个别异常峰的支配。

`w_a` 只能由 S2 可见统计产生，例如连续 rarity/state 权重，并设置 cap，避免少量极端样本支配训练。

必须记录：

- 两方向 occurrence AUPRC；
- 两方向 magnitude MAE；
- 候选 oracle upper bound；
- 候选方向命中率；
- 候选相对 Identity 的增益分布。

若候选 oracle 都无法改善，先修 Bi-OMC，不得把失败归因于 gate。

### 6.4 一日一情节的 CAGM

每一天只产生一个 day-level Key，但保留 24×3 Value：

\[
K_d=f_K(z_{d,1:24},\hat y_{d,1:24},
\operatorname{sg}(\Delta^-_{d,1:24}),
\operatorname{sg}(\Delta^+_{d,1:24})).
\]

`sg` 是 stop-gradient，防止检索目标通过候选签名反向操纵候选生成器。

动作收益按小时计算：

\[
G^a_{d,h}
=\ell(y_{d,h},\hat y_{d,h})
-\ell(y_{d,h},\tilde y^a_{d,h}),
\]

\[
V_d[h,:]=[0,G^-_{d,h},G^+_{d,h}].
\]

第一版只做同小时对齐的检索聚合：

\[
\bar G^a_{d,h}
=\sum_{i\in\mathcal N_k(d)}\omega_{di}G^a_{i,h}.
\]

不要在第一版引入跨小时动态对齐；若尖峰时序偏差仍明显，再做 v2.1。

检索距离必须是 gain-aware，而非仅用价格余弦相似度。推荐：

\[
d_\phi(d,i)=\|\phi(K_d)-\phi(K_i)\|_2^2,
\]

用 S2 的 OOF 动作收益监督 metric，使相似 Key 对应相似的 24×3 gain profile。可用对比损失或 gain-profile 距离回归，但必须做以下对照：

- learned gain-aware metric；
- cosine；
- Euclidean；
- random memory；
- no memory。

为每次预测导出 top-k 日期、距离和权重，以便审计。

### 6.5 DVG 风险价值门

对检索到的动作收益样本，以 CARA 效用计算确定性等价：

\[
\operatorname{CE}^a
=-\frac1\eta
\log\left(
\sum_i\omega_i\exp(-\eta G_i^a)
\right).
\]

Identity 的 CE 固定为 0。可加入有限样本不确定性罚项，但其系数只能在 S3 校准。

KL 正则化策略：

\[
\pi^*(a\mid z)
=\operatorname{softmax}
\left((\operatorname{CE}^a+b_a(z))/\tau\right).
\]

`b_a` 只能使用当前可见上下文，不能读 y。

必须验证两种部署：

1. `soft_soft`：训练 softmax，推理加权三候选；
2. `soft_hard`：训练 softmax，推理取 `argmax(0, CE_down, CE_up)`。

理论上 `soft_hard` 更严格实现“价值大于 Identity 才放行”，但不预选实验赢家。

第一版不实现 entmax。

## 7. 无泄漏训练协议

### Stage 0：宿主缓存

1. 每个数据×宿主只在 S1 训练；
2. 缓存 S1–S4 的预测；
3. 后续所有方法共用缓存；
4. 保存宿主 checkpoint hash 与预测文件 hash。

### Stage 1：S2 候选与 OOF 表征

为避免 CAGM metric 学到候选生成器的训练内收益：

1. 对 S2 做按日期的 blocked forward cross-fitting；
2. 每折只用过去块训练临时候选器；
3. 在下一块生成 OOF 候选、Key 和真实动作收益；
4. 汇总 OOF 样本，训练 gain-aware Key metric；
5. 最后用全部 S2 训练正式 Bi-OMC。

如果 S2 太短无法至少做 3 个有效块：

- 降级为 expanding 2-fold；
- 在报告中标记 `limited_oof`；
- 不允许随机 K-fold。

### Stage 2：S3 记忆与门控

1. 用正式且冻结的 Bi-OMC 在 S3 生成样本外候选；
2. 计算 S3 的动作收益；
3. 构建最终 memory bank；
4. 只用 S3 选择 k、温度、CARA η、KL τ 和 gate mode；
5. 同时得到 Global-Frozen 与 Market-Calibrated 两套 η/τ。

Global-Frozen：

- 在源市场 S3 联合校准；
- 留出目标市场时不接触其标签；
- 到目标市场直接冻结。

Market-Calibrated：

- 每个目标市场只用自身 S3 校准 η/τ；
- 其余网络权重保持冻结。

两者都报告，不预设赢家。

### Stage 3：S4 一次评估

- S4 标签只用于最终计分；
- 不进入 memory；
- 不改 threshold、k、η、τ 或早停轮次；
- 若 S4 结果不理想，允许设计下一版本，但不得回调本版本参数后仍称一次测试。

## 8. 多市场协议

必须至少运行两条轨道。

### 8.1 Within-market

每个市场各自按 S1–S4 切分并训练/校准，用来判断模块在常规设置中的有效性。

### 8.2 Leave-one-market-out

每次留一个公开市场作为目标：

- 其余公开市场为源市场；
- global candidate/key/gate 在源市场联合训练；
- 对目标市场比较 Global-Frozen 与 Market-Calibrated；
- 山东只作为额外目标，不参与公开主表汇总；
- 输出 macro-market 与 micro-sample 两种汇总。

市场特征数量不同依靠 variable-token + mask 兼容，不能以“共同特征交集为空”为由删除外生机制。

第一版允许 DA/RT 池化为两个通道样本，但所有结果必须拆开报告。

## 9. 对照方法定义

### Identity

\[
\tilde y=\hat y.
\]

### Residual-L1

- 输入与 HCH 当前时点可见信息一致；
- 一个不分事件的 LightGBM L1 / quantile-0.5 残差模型；
- 在 S2 训练；
- 直接输出 `hat_y + median_residual`；
- 不加选择性门；
- S3 只用于合法超参数选择。

### QuantileResidual-LGBM

- 在 S2 训练 q=0.1/0.5/0.9 三个残差模型；
- 主预测使用 q=0.5 残差修正；
- 若保留基于 interval width 的选择性规则，width cutoff 必须在 S3 冻结；
- 现有代码若从 S4 prediction 自身统计 cutoff，必须修复为非传导式；
- 名称统一为 QuantileResidual-LGBM。

### PIR 与 δ-Adapter

- 使用固定官方 commit；
- 严格共享 S1 宿主缓存、S2 训练、S3 选择、S4 测试；
- 若官方方法需要额外训练段，从 S2 内部按时间拆，不得读取 S4；
- 所有对官方代码的修改形成 patch 文件；
- 主表脚注说明 unsupported 组合，不能用本地 proxy 补格。

## 10. 第一版精简指标

主文先聚焦少而关键的指标，避免三个大表堆指标。

### 10.1 整体预测

- MAE；
- RMSE；
- rMAE：以同一 S4 上的 168 小时 seasonal-naive MAE 为分母。

sMAPE/MAPE 对负价与近零值不稳定，不进入第一版主表；如同行论文需要，只放附录并明确局限。

### 10.2 高价/尖峰

高尾集合使用 S1 冻结的市场内 p99，仅用于评估，不进入路由。

- High-tail MAE；
- Spike Recall；
- Peak Magnitude MAE；
- Peak Timing Error，单位小时。

### 10.3 低价/负价

- Adaptive Low-tail MAE：S1 冻结的市场内 q10；
- 若市场存在足量 y<0：Negative-price MAE；
- Negative-price Recall；
- Sign Error Rate。

无物理负价或样本不足时，不填 0，标 N/A；低尾指标仍必须报告。

### 10.4 安全与动作

- Normal-regime ΔMAE；
- Harm Rate：执行校正后误差变大的比例；
- Action Rate；
- Down/Identity/Up 比例；
- Oracle Gap：实际 HCH 相对三个候选 oracle 的差距。

### 10.5 汇总规则

- 每个数据×宿主先独立计分；
- 同时报 mean、median、win/tie/loss；
- 市场 macro average 为主，避免大数据集支配；
- 至少 5 个固定 seed；
- 给 paired bootstrap 95% CI；
- 对 HCH vs 每个同行方法做配对显著性检验，并进行多重比较校正；
- 不用单一总体平均掩盖高尾或低尾退化。

## 11. 必跑实验

### 11.1 Smoke

选至少三种互补场景：

1. 有明显负价的公开市场；
2. 尖峰显著但负价稀少的公开市场；
3. 山东一个 DA 与一个 RT 通道。

宿主至少 Linear 和 PatchTST。Smoke 必须通过全部 contracts 才能 full run。

### 11.2 Full matrix

所有可用数据 × 5 宿主 × 6 方法 × 5 seeds。

若官方方法不支持某组合，保留该行并给 status，不得删除分母。

### 11.3 Stability

- 5 seeds；
- 不同 S2/S3 合理比例；
- 不同 memory k；
- Global-Frozen vs Market-Calibrated；
- soft-soft vs soft-hard。

### 11.4 最小消融

只保留能回答贡献问题的消融：

1. Full HCH v2；
2. w/o CAGM；
3. random memory；
4. cosine retrieval；
5. w/o risk / η→0；
6. w/o Identity；
7. one-sided Up only；
8. one-sided Down only；
9. hard discrete state；
10. no exogenous token；
11. no candidate signature in Key；
12. separate Up/Down encoders vs shared encoder。

每个消融至少报告 overall、高尾、低尾、normal ΔMAE、harm rate、oracle gap。

## 12. 自动测试

`test_contracts.py` 至少覆盖：

1. 时间切分无日期交叠；
2. fit 类调用永远收不到 S4；
3. scaler/CDF/quantile 只在合法段 fit；
4. 23/25 小时日被显式审计；
5. 无外生变量能使用 learned-null token；
6. 不同 N 的外生 token 可同批 mask；
7. Down/Up 符号约束；
8. Identity gain 恒为 0；
9. S4 日期不在 memory；
10. OOF 折只用历史块；
11. 宿主训练后冻结且 hash 不变；
12. DA/RT 同日真值不互相泄漏；
13. QuantileResidual cutoff 不读取 S4；
14. official adapter 不允许 proxy fallback；
15. 结果文件名带时间戳且不覆盖；
16. 相同 seed 的 smoke 输出可重复。

任何一项失败都禁止启动 full run。

## 13. 统一命令

PowerShell 示例，保持单行，避免跨平台转义问题：

~~~powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY experiments/08-hch-v2/test_contracts.py
& $PY experiments/08-hch-v2/run_v2.py --mode smoke --config experiments/08-hch-v2/config_v2.json
& $PY experiments/08-hch-v2/run_v2.py --mode full --config experiments/08-hch-v2/config_v2.json
& $PY experiments/08-hch-v2/run_v2.py --mode stability --config experiments/08-hch-v2/config_v2.json
& $PY experiments/08-hch-v2/run_v2.py --mode ablation --config experiments/08-hch-v2/config_v2.json
~~~

Linux/macOS 只需把 `$PY` 换成相应 Python 路径，不改变脚本参数。

`run_v2.py` 需要支持：

- `--datasets`；
- `--backbones`；
- `--methods`；
- `--seeds`；
- `--resume`；
- `--fail-fast`；
- `--device`；
- `--num-workers`。

## 14. 结果产物

每次运行创建新时间戳，例如：

~~~text
experiments/08-hch-v2/results/20260811_153000/
  config_resolved.json
  environment.json
  data_audit.csv
  host_cache_manifest.csv
  official_baseline_manifest.json
  runs_long.csv
  aggregate_overall.csv
  aggregate_tail.csv
  aggregate_safety.csv
  retrieval_audit.jsonl
  failures.csv
  hch_v2_report_20260811_153000.md
~~~

`runs_long.csv` 每行至少包含：

- dataset、market、target_id、split dates；
- backbone、method、seed；
- official status 与 commit；
- overall/high/low/negative/normal 指标；
- action rate、harm rate、oracle gap；
- training/inference time；
- run status、error message、git commit。

Markdown 报告按以下顺序：

1. 执行摘要；
2. 数据与泄漏审计；
3. 官方基线接入审计；
4. 主表；
5. 高尾、低尾和安全表；
6. 候选 oracle 与 gate 分解；
7. Global-Frozen vs Market-Calibrated；
8. soft-soft vs soft-hard；
9. 消融；
10. 失败组合与限制；
11. 相对 v1 的确定改进与仍未解决问题；
12. 下一轮只列 3 个最有证据的改进方向。

## 15. 分阶段验收门

### Gate A：数据与宿主

通过标准：

- 数据审计完整；
- TCN/PatchTST 不是改名实现；
- 5 宿主均生成 S1–S4 冻结缓存；
- 无可用性泄漏。

### Gate B：候选生成器

通过标准：

- 两方向符号正确；
- occurrence、magnitude 和 oracle 指标齐全；
- 至少一个高尾市场和一个低尾市场的 candidate oracle 明显优于 Identity；
- 若不通过，停止调 gate，先修候选器。

### Gate C：记忆与门控

通过标准：

- memory 全来自 S3 样本外候选；
- top-k 可解释且 random/no-memory 对照齐全；
- harm rate、normal ΔMAE、oracle gap 同时报；
- Global-Frozen 与 Market-Calibrated 均完成；
- soft-soft 与 soft-hard 均完成。

### Gate D：全矩阵

通过标准：

- 全数据、5 宿主、6 方法、5 seeds 均有结果或显式失败状态；
- 官方基线没有 proxy 冒充；
- 所有聚合可由 `runs_long.csv` 重算；
- 报告不隐藏退化数据集。

## 16. 禁止捷径

- 禁止用 S4 选择任何参数；
- 禁止先看全量结果再改变极端阈值；
- 禁止用全数据拟合标准化或经验 CDF；
- 禁止随机拆小时；
- 禁止训练内 gain 写入最终 memory；
- 禁止把状态直接当动作；
- 禁止只训练 y<0 二分类器后宣称解决负价；
- 禁止只报告平均 MAE；
- 禁止删除 HCH 输掉的市场或宿主；
- 禁止把 private 山东结果当公开可复现主证据；
- 禁止把官方方法的不支持组合换成本地近似；
- 禁止因为计算成本删掉重要候选；先跑 smoke，再做有记录的资源裁剪。

## 17. 执行完成后的回复模板

代码 Agent 最终必须回复：

~~~markdown
## 完成情况
- Git commit：
- 新增/修改文件：
- 未修改的 v1 文件验证：

## 测试
- contracts：
- smoke：
- full：
- stability：
- ablation：

## 官方基线
- PIR commit / status：
- δ-Adapter commit / status：
- unsupported 组合：

## 关键结果
- Overall：
- High tail：
- Low/negative tail：
- Normal safety：
- Oracle gap：
- Global-Frozen vs Market-Calibrated：
- soft-soft vs soft-hard：

## 最大问题
1.
2.
3.

## 产物
- 结果目录：
- 主报告：
- 失败日志：
~~~

不要只回复“已完成”。所有结论必须能定位到结果文件与配置。

## 18. 本版实现边界

本版明确不做：

- 跨小时动态时间规整记忆；
- 在线写回 memory；
- 物理因果图；
- 共形动作风险保证；
- 端到端修改冻结宿主；
- entmax 路由；
- 为某个市场人工设专属硬阈值。

这些可以在 v2 结果明确暴露瓶颈后再决定，避免首版变成拼盘。
