# 调研报告：有限特征扩展方法 (Feature Expansion for Sparse Data)

> 日期：2026-08-09 | 目的：解决公开数据集仅有 price+demand 时 HCH 特征不足的问题

---

## 核心发现

当前领域有 5 条处理路线：

| 路线 | 核心思想 | 代表工作 |
|---|---|---|
| **领域知识编码** | 将电价形成的物理/市场结构先验编码为特征 | Lago 2023 (arXiv:2306.14186) |
| **时间特征穷举** | 日历/周期信息穷尽编码 | arXiv:2412.01557 |
| **时序分解** | EMD/VMD/STL/Wavelet 将序列分解为多模态子序列 | arXiv:2408.16122, VMDNet |
| **自监督预训练** | 对比学习/掩码自编码学习表征 | CoST (ICLR 2023), Ti-MAE |
| **数据增强** | 生成合成样本扩展训练数据 | FrAug, STAug |

**Gap 确认**：没有论文专门研究"只有 price+demand 时如何自动扩展特征提升极端电价预测"——这是我们的空白。

---

## Top 10 推荐文献

| # | 论文 | 会议/期刊 | 核心方法 | 为什么重要 |
|---|---|---|---|---|
| 1 | **CoST** | ICLR 2023 | 对比学习解耦季节-趋势表征 | SSL 做特征扩展的奠基性方法论 |
| 2 | **Ti-MAE** | arXiv:2301.08871 | 掩码自编码器时序表征提取 | 操作简单效果好 |
| 3 | **Structural approach** | arXiv:2306.14186 | 领域知识编码替代大数据 | 直接论证"有限数据+领域知识"有效 |
| 4 | **BasisFormer** | NeurIPS 2023 | 可学习基函数自动发现特征 | "自动构建特征"的典范 |
| 5 | **VMDNet** | arXiv:2509.15394 | 无泄漏 VMD 做特征分解 | 指出 decomposition 泄漏问题 |
| 6 | **Time-related Features** | arXiv:2412.01557 | 量化"仅时间特征"的上限 | 提供特征增强的必要性基准 |
| 7 | **Pathformer** | ICLR 2024 | 自适应多尺度路径 Transformer | 多粒度特征建模 |
| 8 | **Simple Model** | arXiv:2405.14893 | 仅供需关系预测电价 | price+demand 基线的文献支撑 |
| 9 | **VMD + Linear** | arXiv:2408.16122 | VMD+线性模型, 13数据集验证 | 分解即特征工程的实证 |
| 10 | **Wavelet + Transformer** | arXiv:2403.08630 | 小波特征提取系统性对比 | 小波方向指导 |

## 对 HCH 的建议

最可行的两条路：
1. **Decomposition-based**: VMD/STL 分解 price 序列 → 5-10 模态作为额外特征 → 修改 `build_corrector_features` 加入
2. **Calendar + rolling statistics**: 穷举日历编码 + 波动率/动量/分位统计 → 已在 NEM 诊断中确认 is_weekend/dow 是最强特征
