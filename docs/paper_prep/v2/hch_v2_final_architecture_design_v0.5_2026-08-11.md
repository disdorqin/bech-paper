# HCH v2 最终候选架构设计 v0.5

> 日期：2026-08-11  
> 仓库：disdorqin/bech-paper  
> 目标会议语境：KDD 优先，WWW 备选  
> 状态：进入代码实现前的冻结候选；实验可以推翻具体实现，但不得在未记录原因时改变问题定义。  
> 版本规则：本文件不覆盖 v0.4；v0.4 保留为推导过程记录。

## 0. 一页结论

HCH v2 解决的不是“再训练一个更强预测器”，而是：

> 给定任意冻结电价预测器及其 24 小时预测轨迹，生成向下、保持、向上三类候选动作；从历史相似情节中估计每个候选动作的风险调整收益；只有修正动作的估计价值优于 Identity 时才执行。

论文只保留两个技术贡献：

1. **A：CAGM-DVG 主创新**  
   Counterfactual Action-Gain Memory + Dynamical Value Gate。以完整 24 小时为记忆情节，Value 存储 Identity/Down/Up 的逐时样本外动作收益，再由风险效用和 KL 正则推导动作路由。
2. **B：Bi-OMC 结构增强**  
   Bidirectional Occurrence–Magnitude Corrector。通过全期望公式分别生成 Down/Up 候选，双向共享主体、连续双尾状态与外生上下文。

以下内容是共享底座，不单列贡献：

- 连续低尾—正常—高尾状态；
- 任意数量外生变量 cross-attention；
- 24 小时 Day Token；
- 多市场联合训练；
- 山东日前/实时双目标池化。

## 1. 从 v1 实验出发，而不是从模块名称出发

仓库 v1 全矩阵实际结果为 17 个数据集 × 5 个宿主 = 85 组：

- HCH 正尖峰分支在 85/85 组中均为 λ_pos=0；
- 57/85 组完全 abstain；
- HCH 的平均 ΔMAE 为 -0.7123，但改善主要由山东与少数负价市场贡献；
- NEM_SA1 等组合虽改善负价，却明显损伤尖峰或正常期；
- 现有 v1 依赖 p(event)>0.5、S1 p99 和离散 λ 网格。

因此 v2 不是给 v1 增加更多阈值，而是修复三项结构性问题：

| v1 问题 | v2 裁决 |
|---|---|
| 负价头等同 Down、尖峰头等同 Up | 价格状态与校正动作解耦 |
| p99/0.5/λ 网格造成稀疏触发和边界跳变 | 连续状态、连续候选、价值路由 |
| S3 只判断一个分支整体是否安全 | 逐日逐时估计 Identity/Down/Up 条件收益 |

## 2. 问题定义

对市场 m、日期 d、小时 h，冻结宿主给出：

\[
\hat y_{m,d,h}=F_m(\mathcal I_{m,d}),
\]

其中 \(\mathcal I_{m,d}\) 只能包含发出预测时可见的信息。HCH v2 不修改 \(F_m\) 的参数。

动作集合固定为：

\[
\mathcal A=\{0,-,+\}
=\{\text{Identity},\text{Down},\text{Up}\}.
\]

三个候选预测为：

\[
\tilde y^0=\hat y,\qquad
\tilde y^-=\hat y+\Delta^-,\qquad
\tilde y^+=\hat y+\Delta^+,
\]

其中 \(\Delta^-\le 0\)，\(\Delta^+\ge 0\)。

最终目标不是无条件最小化某个整体残差，而是学习：

\[
a^*(Z)=\arg\max_{a\in\mathcal A}
\operatorname{CE}\!\left(G^a\mid Z\right),
\]

其中 \(G^a\) 是动作相对 Identity 的误差改善，CE 是风险调整确定性等价值。

## 3. 为什么整体残差头天然压平双尾

设条件残差是正常、上尾、下尾的混合：

\[
p(r\mid Z)
=(1-\rho_+-\rho_-)p_N
+\rho_+p_+
+\rho_-p_-,
\]

其中 \(r=y-\hat y\)、\(\rho_+,\rho_-\ll1\)，正常均值约为 0，上尾均值为 \(\mu_+>0\)，下尾均值为 \(-\mu_-<0\)。

整体 MSE 的 Bayes 最优校正为：

\[
c^*_{L2}(Z)=\mathbb E[r\mid Z]
\approx \rho_+\mu_+-\rho_-\mu_-.
\]

因此：

1. 稀有尾部修正被发生概率缩小；
2. 上下两尾可能互相抵消；
3. 即使两边误差都很大，整体校正仍可能接近 0。

整体 MAE 的最优解为：

\[
c^*_{L1}(Z)=\operatorname{Median}(r\mid Z).
\]

只要尾部质量不足以推动 0.5 分位离开正常主体，尾部幅值再大也可能完全不改变中位数。这解释了 Residual-L1 为什么应保留为重要基线，也解释了为什么 B 必须对 \(r^+\) 与 \(r^-\) 分解，而不是只换一个更复杂的通用残差网络。

## 4. 状态与动作必须解耦

残差方向定义：

\[
r^+=\max(r,0),\qquad
r^-=\max(-r,0),\qquad
r=r^+-r^-.
\]

- \(r>0\)：宿主预测偏低，需要 Up；
- \(r<0\)：宿主预测偏高，需要 Down。

但真实价格状态不决定动作方向：

| 真实价格状态 | 宿主失效 | 正确动作 |
|---|---|---|
| 低价/负价 | 宿主预测过高 | Down |
| 低价/负价 | 宿主预测更低 | Up |
| 高价尖峰 | 宿主压平尖峰 | Up |
| 高价尖峰 | 宿主制造伪峰 | Down |

因此不能实现“低价专家=Down、尖峰专家=Up”。连续双尾状态只作为两个动作候选的共同上下文。

## 5. Fig. 1：冻结候选架构

~~~mermaid
flowchart TD
    H["冻结宿主 24h 预测"] --> T["24 个 Hour Token + 1 个 Day Token"]
    X["D-1 可得外生 Token；空集用 Null Token"] --> T
    P["历史、日历、市场/宿主画像、目标通道"] --> T
    T --> S["连续双尾情景编码"]
    S --> B["Bi-OMC：Down/Up 候选"]
    S --> K["CAGM 情景表示"]
    B --> U["候选签名"]
    K --> Q["Day Key"]
    U --> Q
    M["成熟的 OOF 24×3 动作收益情节"] --> R["Top-k 情节检索"]
    Q --> R
    R --> C["CARA 风险调整动作价值"]
    B --> G["DVG 路由"]
    C --> G
    G --> O["Host + 执行动作"]
~~~

主线只有“候选—证据—决策”：

- B 提供可执行的 Down/Up；
- CAGM 提供相似历史中各动作是否有效的证据；
- DVG 决定保持还是修正。

## 6. 共享 24 小时上下文

### 6.1 Hour Token 与 Day Token

一天不是压成一个丢失峰位的标量。每个 episode 同时保留：

\[
E_d=(K_d,Z_{d,1:24},G_{d,1:24}^{0,-,+}).
\]

- Day Token \(K_d\)：用于检索整日情节；
- Hour Token \(Z_{d,1:24}\)：保留峰位、低价持续、峰前峰后形状；
- Gain Field \(G_{d,1:24}^{0,-,+}\)：三个动作逐时收益。

Pyraformer 式 \(24\rightarrow8\rightarrow4\rightarrow1\) 金字塔只作为 Day Key 聚合器消融，不作为记忆创新。主实现先使用 Day Token attention pooling。

### 6.2 任意数量外生变量

核心小时 token 作为 Query，对外生 token 集做 cross-attention：

\[
Z=Q+\operatorname{CrossAttn}(Q,X^{exo},X^{exo}).
\]

每个外生 token 编码：

- 变量类型；
- 目标小时；
- 数值；
- forecast/actual 属性；
- availability mask。

没有外生变量时使用 learned-null token，不改变网络拓扑。当前时刻只能输入目标日已知预测量；实际量必须至少滞后 24 小时。

### 6.3 连续双尾状态

状态头预测：

\[
s^{rank}_{d,h}\in[-1,1],\qquad
s^{zero}_{d,h}\in\mathbb R.
\]

训练监督仅由训练期因果统计构造：

\[
t^{rank}=2\widehat F_{d^-}(y)-1,\qquad
t^{zero}=\frac{y}{\operatorname{MAD}(\mathcal H_{d^-})+\epsilon}.
\]

- rank 坐标让没有负价的数据集也能学习“相对低价/高价”；
- zero 坐标保留物理负价意义；
- 两者联合使稠密低价知识向稀少 \(y<0\) 情节迁移。

推理时状态头只读取可见上下文，不读取当前真实价格。

## 7. B：Bi-OMC 双向发生—位置—幅值候选

### 7.1 全期望推导

对 \(a\in\{+,-\}\)：

\[
\mathbb E[r^a\mid Z]
=\Pr(r^a>0\mid Z)
\mathbb E[r^a\mid r^a>0,Z].
\]

模型输出：

\[
o^a=\sigma(f_{occ}(Z,e_a)),\qquad
m^a=\operatorname{softplus}(f_{mag}(Z,e_a)),
\]

\[
\Delta^+=o^+m^+,\qquad
\Delta^-=-o^-m^-.
\]

这里 occurrence 表示“冻结宿主存在该方向残差”的概率，不表示价格是否越过某个人工极端阈值。推理阶段不使用 \(o>0.5\)。

### 7.2 参数共享

Down/Up 使用：

- 完全共享的 24 小时主体；
- 一个方向 token；
- 可选的低秩方向适配器。

必须比较：

1. 两方向完全分离；
2. 完全共享；
3. 共享主体 + 方向适配器。

### 7.3 时序位置与幅值

将一天的真实方向残差与候选归一为质量分布：

\[
p_h^a=\frac{r_h^a}{\sum_jr_j^a+\epsilon},\qquad
\hat p_h^a=\frac{|\Delta_h^a|}{\sum_j|\Delta_j^a|+\epsilon}.
\]

位置损失：

\[
\mathcal L_{loc}^a
=W_1(p^a,\hat p^a)
=\sum_h|\operatorname{CDF}_{p^a}(h)
-\operatorname{CDF}_{\hat p^a}(h)|.
\]

幅值损失：

\[
\mathcal L_{mag}^a
=\sum_h\operatorname{Huber}
\left(\log(1+r_h^a)-\log(1+|\Delta_h^a|)\right).
\]

这同时约束：

- 尖峰发生时刻与幅值；
- 低价/负价 episode 的开始、持续与深度。

### 7.4 连续稀有度加权

训练不按 p1/p99 切样本，使用因果密度得到连续权重：

\[
w=\frac{(\widehat p(t^{rank})+\epsilon)^{-\beta}}
{\mathbb E[(\widehat p(t^{rank})+\epsilon)^{-\beta}]}.
\]

\(\beta\) 只能由开发段选择并冻结，不能解释成理论常数。

## 8. A：CAGM 动作增益记忆

### 8.1 记忆情节

\[
\mathcal M_i=
(K_i,Z_{i,1:24},G_{i,1:24}^{0,-,+},m_i^{meta}).
\]

Key 只能含预测时可见信息。Value 不存“未来价格”，而存同一真实结果下三个备选预测的样本外收益：

\[
G^a_{i,h}
=\ell(y_{i,h},\hat y_{i,h})
-\ell(y_{i,h},\tilde y^a_{i,h}),
\qquad G^0=0.
\]

第一版 \(\ell\) 使用市场尺度归一的 Smooth-L1。没有真实交易回测前，不把经济收益混入训练标签。

“counterfactual”只指对同一结果比较备选预测动作，不主张因果处理效应。

### 8.2 Key 包含候选签名

仅看价格形状可能检索到候选尺度完全不同的历史日。定义：

\[
u_d^a=\operatorname{Pool}(o^a,m^a,\Delta^a),
\]

\[
K_d=E_{day}
\left(Z_{d,1:24},
\operatorname{sg}(u_d^-),
\operatorname{sg}(u_d^+)\right).
\]

候选签名不需要真实标签，因而推理可用。第一版对它 stop-gradient，防止候选头为了便于检索而扭曲候选。

### 8.3 相似性学习

情景相似不等于宿主错误相似。训练期用 OOF gain field 定义目标邻域：

\[
\bar\alpha_{ij}
=\operatorname{softmax}_j
\left(-D_G(G_i,G_j)/\tau_g\right),
\]

\[
\mathcal L_{metric}
=D_{KL}(\bar\alpha_i\Vert\alpha_i).
\]

推理只使用 Key；gain 只在历史真值成熟后写入记忆。

### 8.4 第一版检索

第一版主模型使用：

\[
\alpha_i(q)=\operatorname{softmax}_{i\in TopK}
\left(
\frac{q^\top K_i+b_\omega(\Delta d_i)
+b_{profile}(q,i)}{\tau_m}
\right).
\]

周期偏置使用可学习 Fourier 特征，不手写“昨天/上周/去年”规则。

第一版按相同 delivery hour 读取检索日的 \(G_{i,h}^a\)。跨小时软对齐保留为后续消融，不作为首轮代码阻塞项。

## 9. DVG：由效用与动力学推出价值门控

### 9.1 CARA 确定性等价值

对动作 a：

\[
C^a_{d,h}
=-\frac1\eta
\log\sum_i\alpha_i(q)
\exp(-\eta G^a_{i,h}).
\]

性质：

\[
\eta\to0:\ C^a\to\mathbb E[G^a],
\]

\[
C^a\approx\mu_a-\frac{\eta}{2}\sigma_a^2.
\]

Identity 恒有 \(C^0=0\)。收益均值低或不稳定的动作会自然降权。

### 9.2 KL 正则闭式解

\[
\pi^*=\arg\max_{\pi\in\Delta^3}
\left[
\sum_a\pi_aC^a
-\tau D_{KL}(\pi\Vert\pi_0)
\right],
\]

\[
\pi_a^*
=\frac{\pi_{0,a}\exp(C^a/\tau)}
{\sum_b\pi_{0,b}\exp(C^b/\tau)}.
\]

这不是经验拼出的 sigmoid。若概率服从复制子动力学：

\[
\dot\pi_a
=\frac1\tau\pi_a
\left(C^a-\sum_b\pi_bC^b\right),
\]

其解析解正是上述 softmax；二动作时退化为 logistic 微分方程及 sigmoid 解。

### 9.3 训练与部署的两个候选

需要完整比较：

1. **Soft/Soft**：softmax 训练，softmax 加权推理；
2. **Soft/Hard**：softmax 训练，部署取

\[
a^*=
\begin{cases}
\arg\max_{a\in\{-,+\}}C^a,&\max(C^-,C^+)>0,\\
0,&\text{otherwise}.
\end{cases}
\]

Soft/Hard 更符合“正价值才修正”，也避免 Down/Up 同时加权抵消；Soft/Soft 更平滑。论文主方案由实验决定。第一版不加入 entmax。

### 9.4 \(\eta,\tau\) 的裁决实验

比较：

- **Global-Frozen**：所有源市场联合学习一套 \(\eta,\tau\)，目标市场直接冻结；
- **Market-Calibrated**：每个市场在 S3 单独校准并冻结。

不提前指定赢家。裁决依据：

- 跨市场总体与双尾收益；
- 正常期退化；
- positive-value precision；
- leave-one-market-out 泛化。

## 10. 训练与防泄漏

保持仓库四段时间协议：

| 段 | 用途 |
|---|---|
| S1 50% | 宿主训练并冻结；因果尺度/分布统计 |
| S2 20% | Bi-OMC 训练；blocked forward cross-fit 产生 metric 监督 |
| S3 10% | 冻结候选产生最终版本 OOS gain；构建 CAGM；校准 \(\eta,\tau\) |
| S4 20% | 一次性最终评估 |

详细流程：

1. 宿主只在 S1 训练，S2–S4 预测均为样本外；
2. S2 按日期做 forward blocks，前块训练、后块预测，生成 OOF 候选收益训练 Key metric；
3. Bi-OMC 最终模型在全部 S2 训练后冻结；
4. 最终 Bi-OMC 在 S3 生成候选，S3 真值成熟后建立与最终候选版本一致的 memory bank；
5. S3 内以 leave-one-day-out 查询校准 \(\eta,\tau\) 和推理模式；
6. S4 不进入候选训练、Key 训练、memory、超参数或模式选择。

记忆条目必须记录：

- dataset/market；
- target channel；
- backbone；
- candidate checkpoint hash；
- split/fold；
- truth maturity time；
- Key 与 gain 的生成版本。

## 11. 多市场、DA/RT 与外生特征

### 11.1 多市场联合训练

- 用市场训练段的稳健尺度归一；
- batch 对 market × host 近似均衡；
- 不依赖训练市场离散 ID；使用波动、偏度、峰度、负价率、宿主残差等连续画像；
- 共享 Bi-OMC、Key 编码器和 DVG；
- leave-one-market-out 时，目标市场标签不得进入 corrector、memory 或调参。

若宿主在目标市场 S1 训练、corrector 未见目标市场标签，主张只能是：

> zero-shot corrector transfer

不能称整个预测系统 zero-shot。

### 11.2 山东日前/实时

本版把日前和实时当作同一 global corrector 的两类样本：

- 必须先按日期划分 S1–S4，再展开为 DA/RT 两条 episode；
- 使用 target-channel token；
- 两通道共享所有模型参数与损失；
- 禁止同日 DA/RT 真实值互相作为输入；
- 预测列在 D-1 可得时可输入；
- 实际列只允许滞后至少 24 小时。

### 11.3 私有数据承诺

- 公开市场是主表、主权重和可复现结论；
- 山东只作真实场景外部验证；
- 公开 schema、availability manifest、划分与汇总统计；
- 主结论不得依赖混入山东私有数据的不可公开权重。

## 12. 固定比较对象

### 12.1 冻结宿主

v2 固定：

1. Linear；
2. MLP；
3. LSTM；
4. TCN；
5. PatchTST。

当前仓库的 Transformer-lite 与 GBDT 仅保留给 v1，不替代 TCN/PatchTST。

### 12.2 后处理方法

固定：

1. Identity / Base；
2. Residual-L1；
3. QuantileResidual-LGBM；
4. PIR 官方实现；
5. \(\delta\)-Adapter 官方 output-residual 版本；
6. HCH v2。

当前仓库的 VahediStyle、SpikeRegularization、CRC 和代理 DeltaAdapter 不进入 v2 最终主表，但旧结果保留。

官方来源必须固定版本：

- PIR：ustc-time-series/PIR，commit fc372bb02090da887d4a20b614a6cfecbfd813d0；
- \(\delta\)-Adapter：Anoise/Adapter，commit 0add06ea7b4d2e0a84c364a8be72eef2676a92f2。

项目内 repro_pir.py 使用手写 alpha/beta，不得标记成官方 PIR。

## 13. 指标与裁决

### 13.1 主预测指标

- MAE；
- RMSE；
- rMAE，以同一 S4 的 t-168h seasonal naive 为分母。

### 13.2 双尾与时序

- physical negative：neg_n、MAE_neg、negative miss rate；
- adaptive low tail：S1 q10 冻结后的 MAE_low；
- high spike：S1 p99 冻结后的 MAE_spike、spike miss rate；
- normal：MAE_normal；
- episode recall、complete miss、boundary MAE；
- 日内峰值 timing error 与 magnitude error。

阈值只用于评估，不进入 v2 路由。

### 13.3 门控诊断

- touch rate；
- Down/Up/Identity 占比；
- hindsight positive-value precision；
- harmful-touch rate；
- mean realized gain；
- risk–coverage / coverage–gain 曲线；
- CARA value calibration。

## 14. 必须消融

| 消融 | 回答问题 |
|---|---|
| v1 vs v2 | 新结构是否解决 λ_pos=0 与大量 abstain |
| 单通用残差 vs 双向全期望分解 | B 是否超出参数量收益 |
| 状态与动作硬绑定 vs 解耦 | 方向独立建模是否必要 |
| 无位置损失 vs W1 | 尖峰时序和低价持续是否改善 |
| 无 memory vs future/residual/gain memory | 动作收益 Value 是否必要 |
| 价格相似 Key vs gain-aware Key | 相似性监督是否有效 |
| 无候选签名 vs 有候选签名 | A/B 是否真正闭合 |
| 小时记忆 vs Day Key + 24×3 Value | 日情节是否必要 |
| 期望 gain vs CARA gain | 风险项是否必要 |
| Soft/Soft vs Soft/Hard | 严格放行是否更稳 |
| Global-Frozen vs Market-Calibrated | 参数如何跨市场冻结 |
| 完全分离/完全共享/方向适配 | 上下尾知识是否正迁移 |

## 15. 创新边界

最严格差异化声明：

> HCH v2 does not retrieve historical futures, directly regress one generic residual, or route between normal and extreme forecasters. It constructs symmetric signed correction candidates for a frozen host, retrieves cross-fitted 24-hour action-gain episodes under action-effect similarity, and chooses Identity/Down/Up through a risk-sensitive value rule derived from KL-regularized utility.

暂不允许：

- 首次负电价预测；
- 首次研究高低双尾；
- 首次 occurrence–magnitude；
- 首次无硬切分极端专家；
- 首次 24 小时轨迹或时序检索；
- 首次冻结宿主后处理；
- 首次“有收益才修正”；
- 理论保证绝不退化；
- 整个系统 zero-shot；
- 负价精度必然等于交易收益。

## 16. 第一轮实现边界

必须实现：

- 连续 rank/zero 状态；
- 双向 occurrence–magnitude 候选；
- W1 位置与幅值损失；
- 任意外生 token + null token；
- Day Key + 候选签名；
- OOF 24×3 action-gain memory；
- gain-aware metric；
- CARA value；
- KL-softmax 训练；
- Soft/Soft 与 Soft/Hard；
- Global-Frozen 与 Market-Calibrated；
- DA/RT date-first pooling；
- 完整防泄漏与诊断指标。

暂缓：

- 跨小时软对齐；
- 在线本地记忆写回；
- EMA 联合微调；
- entmax；
- 交易收益训练；
- 动态图/因果图。

进入论文主张前，必须先由全矩阵与消融证明：

1. 尖峰分支不再系统性失活；
2. 负价/低尾改善不依赖山东；
3. 正常期伤害受控；
4. gain memory 优于普通残差与未来值记忆；
5. 价值估计与真实动作收益具有可测的校准关系。

## 17. 主要文献边界

- [PIR, NeurIPS 2025](https://github.com/ustc-time-series/PIR)：实例识别 + 局部/全局 revision；CAGM 不检索历史 future，而检索三动作 OOF gain。
- [δ-Adapter, ICLR 2026](https://github.com/Anoise/Adapter)：冻结宿主的 output residual correction；HCH v2 的差异是双向候选、gain episode 与 value routing。
- [RAFT, ICML 2025](https://arxiv.org/abs/2505.04163)：检索相似历史后续值；不能声称首次时序检索。
- [TS-Memory, KDD 2026](https://arxiv.org/abs/2602.11550)：泄漏安全检索教师和 improvement gate；不能声称首次 memory + gain gate。
- [Pyraformer, ICLR 2022](https://openreview.net/forum?id=0EXmFzUn5I)：只启发多尺度 Day Key 编码。
- [Loizidis et al., Applied Energy 2025](https://doi.org/10.1016/j.apenergy.2025.126013)：已有负价两阶段预测，不能声称首次负价。

---

本文件定义“要实现什么以及为什么”。逐文件、逐接口、逐测试的执行要求见配套的 HCH v2 AI implementation spec v0.1。
