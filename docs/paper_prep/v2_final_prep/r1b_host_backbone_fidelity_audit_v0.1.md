# R1B Host Backbone Fidelity Audit

- **版本:** v0.1
- **日期:** 2026-08-13
- **仓库基线:** `ccd715d` (chore: add requirements.txt)
- **审计对象:** `src/backbones.py` 全量 host 骨架
- **审计类型:** 只读审计,不改代码
- **R1B host 集合:** `V2_BACKBONES = ("Linear", "MLP", "LSTM", "TCN", "PatchTST")`(`experiments/08-hch-v2/host_cache.py:32`);另有 `Transformer`(`_SeqTransformer`)注册于 `BACKBONES`,当前未进入 V2 host 集,但存在命名问题,一并审计。
- **序列窗口:** `SEQ_LEN = 168`(`src/common.py:117`,7 天小时级价格窗,截至 t-24)。

---

## 1. 审计结论摘要

| Host 名 | 代码类 | 组件要点 | 分类 |
|---|---|---|---|
| `Linear` | `_Sk("Linear")` | Ridge(alpha=1.0) on standardized X | **canonical/basic implementation** |
| `MLP` | `_Sk("MLP")` | sklearn MLPRegressor(128,64), early stopping, seed | **canonical/basic implementation** |
| `GBDT` | `_GBDT` | LightGBM 600 树 / lr0.05 / 63 leaves,CPU only | **canonical/basic implementation** |
| `LSTM` | `_SeqLSTM` | 2 层 LSTM(h=64)+ 静态特征 MLP head | **canonical/basic implementation** |
| `TCN` | `_TCN` | 因果空洞卷积(1,2,4,8)+ 全局池化 MLP head | **architecture-inspired local implementation** |
| `PatchTST` | `_PatchTST` | patch(16/8)+ 线性嵌入 + pre-LN Transformer + mean-pool MLP head,单通道,无 RevIN | **architecture-inspired local implementation(PatchTST-style)** |
| `Transformer` | `_SeqTransformer` | **非重叠 patch** 化(24h)+ TransformerEncoder + mean-pool MLP head | **architecture-inspired local implementation(命名不诚实,见 §6)** |

---

## 2. 逐 host 组件表与分类

### 2.1 `_Sk`(Linear / MLP)

**构造参数:** `kind`(Linear|MLP)、`seed=0`。
**预处理:** `StandardScaler` 对 X 与 y 分别标准化(预测后反标准化)。
**Linear:** `Ridge(alpha=1.0)`。岭回归,标准线性基线,命名与实现一致。
**MLP:** `MLPRegressor(hidden_layer_sizes=(128,64), max_iter=300, early_stopping=True, n_iter_no_change=15, random_state=seed)`。scikit-learn 标准 MLP,Adam,两层 128→64。

**分类:canonical/basic implementation。**
备注:Linear 实为 L2 正则化的岭回归;作为 "Linear" 基线在领域惯例中可接受(文献常以 Ridge 代指线性),若需更严格可注明 ridge-regularized。

### 2.2 `_GBDT`

**构造参数:** `seed=0`。
**模型:** `LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=30, subsample=0.9, subsample_freq=1, colsample_bytree=0.9, n_jobs=8, verbose=-1)`。
**备注:** CPU only(注释说明本机 GPU 路径死锁)。标准 LightGBM 配置,属于 GBDT 族的常规基线。

**分类:canonical/basic implementation。**

### 2.3 `_SeqLSTM`

**构造参数:** `n_static`、`hidden=64`。
**网络结构:**
- `nn.LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.1)` —— 输入 1 通道(单一价格序列),2 层,隐层 64,层间 dropout 0.1。
- head:`Linear(64 + n_static → 64) → ReLU → Linear(64 → 1)`。
- forward:取最后一个时间步 `h[:,-1,:]`,拼接静态特征后过 MLP,输出单步标量。

**与标准 LSTM 基线约定对比(文献惯例):**
- 层数:1–2 层为主流;本地取 2 层,在惯例内。
- hidden size:32–128 常见;本地 64,在惯例内。
- 输入窗:本地为 `SEQ_LEN=168`(7 天小时价),在短期负荷/电价预测常用窗(24–336h)内。
- 输出:单步(24h 前)标量,匹配本项目预测目标。
- 与若干公开实现相比,本地 LSTM 属"朴素单序列 LSTM + 静态特征 MLP head",无 embedding、无双向、无 attention 增强——均不是缺陷,而是标准基础形态。

**分类:canonical/basic implementation。**

### 2.4 `_TCN`

**构造参数:** `n_static`、`seq_len`、`hidden=64`、`kernel_size=3`、`dropout=0.1`。
**网络结构:**
- 4 层因果 Conv1d,膨胀率 `[1,2,4,8]`,核 3,`padding=(k-1)*d`(左侧补零,保因果)。
- 每层后 `Dropout → ReLU`;通道从 1 → 64 → 64 → 64 → 64。
- 全局均值池化(时间维)→ `Linear(64 + n_static → 64) → ReLU → Linear(64 → 1)`。

**与经典 TCN(Bai et al., 2018)对比:**
| 组件 | 经典 TCN | 本地 `_TCN` |
|---|---|---|
| 因果卷积 | ✓ | ✓ |
| 空洞膨胀 | 指数 2^i | [1,2,4,8](前 4 层指数) |
| 残差连接 | 每个 block 内残差 | **无** |
| 权值归一化 weight-norm | ✓ | **无** |
| 感受野设计 | 按任务显式覆盖输入长 | 未显式保证覆盖 168 |
| head | 逐时间步输出 | 全局池化 + 静态 MLP(单步) |

**分类:architecture-inspired local implementation。**
结论:是一族"因果空洞卷积"的简化本地实现,不具备经典 TCN 的残差 + weight-norm 结构,不应在论文中表述为"复现了 TCN (Bai et al. 2018)"。

### 2.5 `_PatchTST`

见 §3 详细逐项对比。结论先行:

**分类:architecture-inspired local implementation(PatchTST-style)。**

### 2.6 `_SeqTransformer`("Transformer")

**构造参数:** `n_static`、`seq_len`、`d_model=64`、`patch=24`、`nhead=4`、`layers=2`。
**网络结构:**
- **非重叠 patch 化**:`n_patch = 168 // 24 = 7`,把输入切成 7 个 24 小时 patch(丢弃尾部不足 24 的部分,`seq[:, -7*24:]`)。
- `Linear(24 → 64)` 嵌入 + learnable 位置编码(zeros 初始化)。
- 2 层 PyTorch `TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, norm_first=True)`。
- mean-pool over patches → `Linear(64 + n_static → 64) → ReLU → Linear(64 → 1)`。

**关键发现:该模型是 patch-based transformer,不是 vanilla Transformer。**
Vanilla Transformer 做时间序列的通常做法是把**每个时间步当作一个 token**(L=168 个 token,线性嵌入到 d_model);而本实现把输入先切成 24 小时块再送入编码器——这正是 PatchTST 的核心设计元素(patching)。因此命名为 `Transformer` 会误导读者以为它是逐点 token 的经典 Transformer,而它实际是 `PatchTST-style`(非重叠变体)。

**分类:architecture-inspired local implementation(命名不诚实)。**

---

## 3. PatchTST 保真度逐项对比

**对照基准:** Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023;官方仓库 `yuqinie98/PatchTST`(PatchTST_supervised)。

| # | 组件 | 论文 / 官方实现 | 本仓库 `_PatchTST` | 判定 |
|---|---|---|---|---|
| 1 | Patching | P=16, S=8;尾部用 S 个重复值 padding,使 N≈L/S(论文 N=42/64 对应 L=336/512) | P=16, S=8;`n_patches=(168-16)//8+1=20`,**无尾部 padding**(官方会得 21) | 部分一致(丢尾 patch) |
| 2 | Patch embedding | 线性投影 W_p ∈ R^(D×P) | `Linear(16, 128)` | **一致** |
| 3 | 位置编码 | learnable 加法,`nn.init.normal_` 初始化 | learnable `Parameter(zeros)` | 一致(初始化次要差异) |
| 4 | Encoder 层 | vanilla Transformer,e_layers=3, n_heads=4;官方自定义 `TSTEncoderLayer`(**post-LN**:attention → dropout → residual → LayerNorm,FFN 同理) | 3 层 PyTorch 默认 `TransformerEncoderLayer(d_model=128, nhead=4, d_ff=256, **norm_first=True**, batch_first)`(**pre-LN**) | **不一致**(norm 位置、d_ff) |
| 5 | 通道独立 channel-independence | 多变量输入拆成 M 条单变量序列,共享同一套 embedding + Transformer 权重;官方代码 [B,L,D]→[B,D,L]→patch→[B*D,N,D]→encoder→[B,D,N,D] | 输入为**单一价格序列**(单通道),天然只处理 1 条通道;**未实现多通道共享权重的通道独立机制** | 对本项目单变量 host 无影响;机制本身缺失 |
| 6 | Instance normalization(RevIN) | 每个通道进入 patch 前按该序列 mean/std 标准化,输出后反标准化 | **无** | **缺失** |
| 7 | 输出 head | 每通道独立输出后拼接;官方 `FlattenHead`:`flatten(d_model×N) → Linear(pred_len)` | **mean-pool over 20 patches → MLP(128+n_static → 128 → 1)** | **不一致**(池化 + 静态特征 MLP,无 FlattenHead) |
| 8 | 预测步数 | 多步 `pred_len`(96/192/336/720) | 单步标量(24h 前) | 不一致(适配本项目) |
| 9 | 训练损失 | MSE | `HuberLoss(delta=1.0)` | 不一致 |
| 10 | 优化器/早停 | Adam, lr≈1e-4, epochs≈100, patience 3, bs 128 | AdamW lr=1e-3 wd=1e-4, 40 ep, patience 8, bs 256, 90/10 时序切分, grad-clip 1.0 | 不一致(训练协议差异) |
| 11 | 静态特征注入 | 论文无(纯序列模型) | head 注入 `n_static` 外生特征 | **本地扩展** |
| 12 | 自监督 masked pretraining | 论文另一半(mask ratio 0.4 预训练) | 不适用(监督 host) | N/A |

**逐项结论:**
- 具备 PatchTST 的两个"签名"之一:patching(P=16/S=8)+ 线性嵌入 + learnable 位置编码 + 共享 vanilla Transformer encoder → 家族上成立。
- 但缺失/改动:**RevIN 实例归一化(缺失)**、**FlattenHead 多步 head(改为 mean-pool + 静态 MLP)**、**encoder 归一化位置(pre-LN vs 官方 post-LN)**、**尾部 patch padding(丢弃)**、**多通道通道独立机制(单变量 N/A)**;另有本地扩展(静态特征注入)。

---

## 4. 五个问题的回答

**(a) 本仓库 PatchTST 是否够格叫 PatchTST?**
家族上够格,严格复现上不够格。它拥有核心的 "patching + 共享 Transformer encoder" 结构,说它是 PatchTST 一族的模型没问题;但它不是论文模型的忠实复现(缺 RevIN、缺 FlattenHead/多步预测、encoder 用 pre-LN 而非论文 post-LN、无尾部 padding、单通道)。**结论:可称 "PatchTST-style" / "PatchTST 风格",不宜在论文中表述为"复现了 PatchTST"。**

**(b) 缺了/改了哪些主要组件?**
1. **Instance normalization(RevIN 式)—— 缺失**(这是 PatchTST 缓解分布漂移的关键组件之一);
2. **head —— 改了**(mean-pool + 静态特征 MLP、单步输出,而非官方 FlattenHead 多步);
3. **encoder 归一化位置 —— 改了**(PyTorch pre-LN vs 官方 post-LN TSTEncoderLayer);
4. **尾部 patch padding —— 缺失**(`n_patches=(L-P)//S+1` 截断,官方为 +2);
5. **通道独立机制 —— 单变量 N/A,多变量共享权重机制未实现**;
6. **位置编码初始化 —— 次要差异**(zeros vs normal init);
7. **自监督 masked pretraining —— 非监督 host 范围,N/A**;
8. **静态特征注入 head —— 本地扩展**(论文无此设计)。

**(c) 论文里是否应改叫 PatchTST-style?**
**是。** 理由:(i) 科学诚实——当前实现不是论文/官方复现,直呼 "PatchTST" 会让审稿人默认是忠实复现;(ii) 论文不能把该 host 当作"外部 SOTA 基线 PatchTST"来对比;(iii) 本项目叙事是"corrector 跨 host 泛化",host 的**精确家族命名**比"是否叫 PatchTST"更重要,用 "PatchTST-style" 即可准确传达"patch + Transformer"族而不过度承诺。

**(d) 做一个官方/公平 host 复现大概多少工作量?**
**中等偏小,约 1–2 人日。** 拆解:
- 新增模型代码 ~150–250 行(instance norm 前/后处理、FlattenHead、post-LN encoder 层或明确标注 pre-LN 偏差、n_patches padding 修正);
- 集成进 `backbones.py` + `host_cache.py` 并重建 host cache;
- 重跑 host-cache + corrector 全管线并重验 R1A/R1B 结论(~0.5–1 人日,模型本身很小,d_model=128、3 层、20 patches,单卡数分钟级/数据集)。
总成本主要是"重跑 + 重验"而非写代码。

**(e) 影响"corrector 与 host 无关"声明,还是只影响基线命名?**
**主要影响基线命名;独立声明本身不失效,但"host 族多样性"论证的强度受影响。** 依据:
- 正确性层面:声明依赖的是"corrector 能跨**不同误差族**泛化",而误差族来自 host 在数据上的拟合/误差分布,与 host 是否忠实复现某篇论文**无逻辑关系**。host 改名不推翻该声明。
- 命名层面:论文必须把 host 如实标注为 "PatchTST-style"/"Ridge"/"LSTM" 等,不能把 host 呈现为"复现了 PatchTST 官方基线";否则审稿人会发现对比对象名不副实。**此点必须修。**
- 多样性层面:若把 `Transformer`(patch-based)与 `PatchTST`(patch-based)同时计入 host 集并当作两个独立家族,则二者其实是**同一族**(patch-token Transformer,仅超参不同),会削弱 "distinct host families" 与 leave-one-host-family-out(§7.2)的论证强度。**当前 V2 host 集(Linear/MLP/LSTM/TCN/PatchTST)不含 `Transformer`,不存在该问题**;但一旦 R1B 后续加入 `Transformer`,必须先处理其命名/结构(见 §6)。

---

## 5. `_SeqTransformer` 命名建议

**问题:** "Transformer" 名不副实——它先做非重叠 24h patch 化再进编码器,是 patch-based transformer,不是逐点 token 的 vanilla Transformer。

**方案 A(最小改名,仅命名诚实):**
- 改为 `PatchTransformer` 或 `PatchTST-style (non-overlapping)`。
- 优点:零代码改动、命名诚实。缺点:与 `_PatchTST` 仍是同族,两个 host 不算两个独立家族。

**方案 B(改名 + 真正多样化,推荐):**
- 把 `_SeqTransformer` 改为**逐点 token 的 vanilla Transformer**:`seq (N,168)` 线性嵌入到 d_model(即 168 个 token),加位置编码,过 `TransformerEncoder`,mean-pool 后接静态 MLP head(与现有 head 一致)。
- 优点:(i) "Transformer" 名实相符;(ii) 与 `_PatchTST`(patch-token)构成**真正不同的 host 族**(point-token vs patch-token),增强 model-independence / leave-one-host-family-out 论证;(iii) 改动量小(替换 patch 化逻辑为线性逐点嵌入)。
- 代价:需重跑该 host 的 cache 与相关结论。

**建议:采用方案 B。** 若工期不允许,至少采用方案 A 的命名。

---

## 6. 结论

1. R1B 五个 V2 host 中,`Linear/MLP/GBDT/LSTM` 为 canonical/basic,可放心直接使用与命名。
2. `TCN` 为架构启发的本地简化实现(无残差/无 weight-norm),不应表述为复现经典 TCN。
3. `_PatchTST` 具备 PatchTST 家族核心(patching + 共享 Transformer),但缺 RevIN、缺 FlattenHead/多步、encoder 为 pre-LN、无尾部 padding;**应命名为 "PatchTST-style"**。
4. `_SeqTransformer`("Transformer")实为 patch-based transformer,命名不诚实;**建议改为逐点 token 的 vanilla Transformer(方案 B)或至少改名(方案 A)**。
5. 上述全部为**基线命名与分类问题**,不推翻 "corrector 与 host 无关" 的声明;唯一需要警惕的是不要把 patch-token 的 `Transformer` 与 `PatchTST` 当作两个独立 host 族计入多样性论证。

---

## 附:审计信息源

- 本地代码:`src/backbones.py`;`src/common.py`(SEQ_LEN=168);`experiments/08-hch-v2/host_cache.py`(V2_BACKBONES)。
- PatchTST 论文:Nie, Nguyen, Sinthong, Kalagnanam, *A Time Series is Worth 64 Words*, ICLR 2023(arXiv:2211.14730)。
- 官方仓库结构与实现要点(经 web 检索核对,非逐行比对源码):
  - patching P=16/S=8,尾部 pad stride 个重复值;
  - 线性 patch embedding + learnable 位置编码;
  - 共享 Transformer encoder(e_layers=3, n_heads=4, 自定义 post-LN TSTEncoderLayer);
  - 通道独立:[B,L,D]→[B,D,L]→[B*D,N,D],每通道独立过同一编码器;
  - RevIN 式 instance normalization(前标准化 / 后反标准化);
  - FlattenHead:flatten(d_model×N)→Linear(pred_len),多步输出;
  - 监督训练 MSE,Adam,约 100 epochs,patience 3。
- 已知局限:本机网络受限,未能直接抓取官方 `PatchTST.py` 源码逐行比对;上述官方实现要点来自论文正文与多个二次实现(TSLib、HF blog、CSDN 解析、tsai)交叉确认,置信度高。若需 100% 逐行级,建议在有外网的环境 clone `yuqinie98/PatchTST` 复核 encoder 层与 FlattenHead 细节。
