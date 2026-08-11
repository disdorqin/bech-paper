# 官方基线接入文档

> 日期: 2026-08-11  
> 对应规范: `hch_v2_ai_implementation_spec_v0.1` §3.3

## 1. 来源

| 基线 | 官方仓库 | 固定 commit | 适配文件 |
|---|---|---|---|
| δ-Adapter | [Anoise/Adapter](https://github.com/Anoise/Adapter) | `0add06ea7b4d2e0a84c364a8be72eef2676a92f2` | `official_adapters.py::DeltaAdapterOfficial` |
| PIR | [ustc-time-series/PIR](https://github.com/ustc-time-series/PIR) | `fc372bb02090da887d4a20b614a6cfecbfd813d0` | `official_adapters.py::PIROfficial` |

仓库位置: `experiments/07-route-e/peers/vendor/`

## 2. δ-Adapter 适配

**提取的官方模块**: `PostY` 类 (来自 `vendor/delta_adapter/AdaIntpX/experiments/exp_decom9_post_y.py`)

**架构**: 3层 MLP (BatchNorm + ReLU), 输入冻结宿主预测 → 输出直接替代预测

**论文中的行为**: PostY 接收 instance-normalized 的宿主预测, 直接回归目标价格 (不做残差, 直接替换)

**适配修改**:
| 项目 | 官方实现 | 我们的适配 |
|---|---|---|
| 数据格式 | TSLib DataLoader (batch_x/batch_y/batch_mark) | fit(yhat, y) 直接传入 numpy |
| 训练 | 120 epoch + lr schedule + early stop | 30 epoch + patience=8 |
| 推理 | vali_post() 内嵌 test loop | predict() 返回 numpy |
| 归一化 | 逐样本 instance norm | 全段 mean/std |

**未改变**: PostY 网络结构 (Linear→BN→ReLU→Linear→BN→ReLU→Linear)

## 3. PIR 适配

**提取的官方模块**: `QualityEstimator` + `Refiner` (来自 `vendor/PIR/models/PIR.py`)

**架构**: 
- QualityEstimator: TransformerEncoder → 双头输出 α/β 权重
- Refiner: TransformerEncoder → 修正预测

**论文中的行为**: PIR = 主干预测 + α × refine_out + β × retrieval_out

**适配修改**:
| 项目 | 官方实现 | 我们的适配 |
|---|---|---|
| 检索索引 | construct_index + prepare_retrieval_index | **省略** (标记为 limited_official) |
| 主干 | 预训练 PatchTST/iTransformer/TimeMixer | 冻结的 Linear/MLP/LSTM 宿主 |
| 数据格式 | TSLib DataLoader | fit(yhat, y) numpy |
| 归一化 | 逐样本 instance norm | 全段 mean/std |

**已知差异**: 官方 PIR 需要训练集检索索引, 我们在适配中省略了检索组件, 仅保留 QualityEstimator + Refiner。结果中标记为 `limited_official(no_retrieval)`。

## 4. 接口统一

所有官方基线遵循与 v2 基线相同的接口:
```python
method.fit(Z, yhat, y)      # S2 训练 (Z 可为 None)
method.predict(Z, yhat)     # S4 预测
```

## 5. 无法支持的组合

PIR 和 δ-Adapter 的官方实现原本绑定 TSLib 数据管道 (仅支持 ETTh/ETTm/Weather/ECL/Solar 等数据集)。我们对所有自有电价数据集进行了适配, 理论上全覆盖。

若特定 数据×宿主 组合因样本量不足 (<100) 导致训练失败, 标记为 `unsupported_official(sample_too_small)`。

## 6. 文件结构

```
experiments/07-route-e/peers/
├── vendor/                     # 官方仓库 (git clone, 不可修改)
│   ├── PIR/                    # fc372bb
│   └── delta_adapter/          # 0add06e
├── official_adapters.py        # 适配代码 (本文件配套)
└── official_baselines.md       # 本文档
```
