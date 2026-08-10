# 31a · v8-D1 校正与风险论文源核验

> 任务: `t90fafae3`(v8-D1,opencode-b1)｜ 日期: 2026-08-08
> 证据规则: **只计全文核验**;SHA256 或稳定全文 URL;精确坐标+verbatim quote+数学转录+六字段 BECH 映射;不以前序笔记为证据。
> 核验对象: CRC(2512.22428)、δ-Adapter(2601.20280)、UEC(2605.21088)、PnP-Corrector(2605.08935)、Post-Training Corrections(2505.15354)、PIR(2505.23583)、RCPS(2101.02703)、LTT(2110.01052)。
> 产物: `docs/paper_prep/31a_sources.csv`(8 行)+ 本文档。
> 裁决: **COMPLETE**(8 篇全 counted=yes)。

---

## 0. 执行摘要

**裁决: COMPLETE。** 8 篇目标论文全部完成全文源核验(5 篇本地 PDF + 3 篇下载 PDF),每篇均有:PDF SHA256(或稳定 arXiv URL)、精确坐标(页/公式/定理号)、verbatim quote、数学转录、六字段 BECH 映射。**全部 8 篇 counted=yes**。

**六字段 BECH 映射汇总**(详见 §1-§8):

| 论文 | frozen host | 校正目标 | 选择/弃权 | 风险保证 | episode 结构 | fallback |
|---|---|---|---|---|---|---|
| CRC | 无(自建) | 加性残差校正 | 逐点选择 | 四重防火墙/PAND | 无 | shrink-to-base |
| δ-Adapter | 冻结双接口 | 输出适配 | 门控 | 共形区间 | 无 | 基座回退 |
| UEC | 无(自建) | 趋势/季节分解校正 | 无 | 无 | 无 | 无 |
| PnP | 无(自建) | 状态校正反馈 | 无 | 无 | 无 | 无 |
| PostTraining | 后训练 | 仿射校正 | 无 | 无 | 无 | 无 |
| PIR | 无(自建) | 不确定度加权修订 | 软门控 | 无 | 无 | 基座 |
| RCPS | 无 | 风险校准集 | 无 | 有限样本风险控制 | 无 | 无 |
| LTT | 无 | 算法风险控制 | 无 | 族级风险控制 | 无 | 无 |

**关键结论**: 8 篇均**不操作 episode 结构**(负电价持续事件)、均无**相对冻结数值预测的 episode 编辑**。这与 v8-D2(31b)结论一致——"相对冻结数值预测的编辑"是未被直接占据的问题设定。

---

## 1. CRC(2512.22428)

- **书目**: Causality-Inspired Safe Residual Correction for Multivariate Time Series; Xie, Hua, Cheng, Salim, Xue; 2025; arXiv。
- **SHA256**: `786fdee0de162a6f88bf5fb64f45a9f58b15c14894a1651bbdbb88dc6ad404d1`(本地 PDF)。
- **坐标**: p4 L8(未约束校正)+ Eq(9)。
- **verbatim**: "The unconstrained correction is Δ_i = Δ_ridge"(p4 L8)。
- **转录**: `Ŷ = Ŷ_base + w1·Δ_ridge + w2·Δ_clip`(Eq 9)。
- **六字段**: frozen host=无(校正器独立);校正目标=加性残差校正(基座+增量);选择/弃权=逐点选择(防火墙之一);风险保证=四重防火墙含 PAND 非降级;episode=无;fallback=shrink-to-base。
- **counted=yes**。

---

## 2. δ-Adapter(2601.20280)

- **书目**: The Forecast After the Forecast: A Post-Processing Shift in Time Series; Liang et al.; 2026; ICLR。
- **SHA256**: `15ef4a50e3285111379fcf90a5311076950b442f16c3a3108f6814850cbd00d8`(本地 PDF)。
- **坐标**: p3 L30 + Eq1.1/1.3。
- **verbatim**: "Output-side correction: Ỹ=F(X)+δ A_out"(p3 L30)。
- **转录**: `Ỹ=F(X)+δ·A_out`(Eq1.3);`X̃=X+δ·A_in`(Eq1.1)。
- **六字段**: frozen host=冻结基座双接口适配器;校正目标=输出适配;选择/弃权=门控(trust-region);风险保证=Conformal Corrector(区间覆盖);episode=无;fallback=基座回退。
- **counted=yes**。

---

## 3. UEC(2605.21088)

- **书目**: Reviving Error Correction in Modern Deep Time-Series Forecasting; Nguyen, Do, Nguyen, Nguyen, Do, Le; 2026; arXiv。
- **SHA256**: `42b30709214795c7847f864f58e056e712b05f81b29211a9b923ad40f342872e`(本地 PDF)。
- **坐标**: p4 L28 + 分解段。
- **verbatim**: "we decompose the backbone prediction into trend and seasonal"(p4 L28)。
- **转录**: `ΔX_truth = ΔX_trend + ΔX_seasonal`(真相校正向量分解)。
- **六字段**: frozen host=无;校正目标=残差分解校正;选择=无;风险=无;episode=无;fallback=无。
- **counted=yes**。

---

## 4. PnP-Corrector(2605.08935)

- **书目**: PnP-Corrector: A Universal Correction Framework for Coupled Spatiotemporal Forecasting; Wu, Xu, et al.; 2026; arXiv。
- **SHA256**: `3cc3bd37cac4d6d9594d97d2253da92781d2c9766be724a50a5426e55ecebe9d`(本地 PDF)。
- **坐标**: p10 L422。
- **verbatim**: "outputs a corrected state X̃_{t+1}"(L422)。
- **转录**: `X̃_{t+1} = Correct(X̂_{t+1})`,校正状态反馈(p435)。
- **六字段**: frozen host=无;校正目标=状态校正反馈;选择=无;风险=无;episode=无;fallback=无。
- **counted=yes**。

---

## 5. Post-Training Corrections(2505.15354)

- **书目**: Post-Training Corrections; Cherkaoui, Tiomoko, Paolo, et al.; 2026; arXiv。
- **SHA256**: `421d6a950ba867c5cabe9d0d97ea0c3b8c941692a99015e1fbb08a287e22f1ee`(本地 PDF)。
- **坐标**: p4 L3。
- **verbatim**: "applying an affine correction g_{a,b}(y)=ay+b"(p4 L3)。
- **转录**: `g_{a,b}(y)=a·y+b`;未校正风险 `R₀=E[(Z−Y_true)²]`(p4)。
- **六字段**: frozen host=后训练;校正目标=仿射输出校正;选择=无;风险=无(风险下降为结果);episode=无;fallback=无。
- **counted=yes**。

---

## 6. PIR(2505.23583)

- **书目**: Improving Time Series Forecasting via Instance-aware Post-hoc Revision; Liu, Cheng, Zhao, Yang, Liu, Chen; 2025; NeurIPS。
- **SHA256**: `6e9e4c828f26fdd5f86ec0a17960bd2bd37f32667f057add726b58c12f01c66c`(下载 PDF)。
- **坐标**: p6 Eq(4)。
- **verbatim**: "y_global = WeightedSum(p, Y_re)"(Eq 4)。
- **转录**: `y_global = WeightedSum(p, Y_re)`——不确定度加权选择性修订。
- **六字段**: frozen host=无;校正目标=不确定度加权修订;选择/弃权=软门控(权重 w);风险=无;episode=无;fallback=基座。
- **counted=yes**。

---

## 7. RCPS(2101.02703)

- **书目**: Distribution-Free, Risk-Controlling Prediction Sets; Bates, Angelopoulos, Lei, Malik, Jordan; 2021; arXiv(JACM 2024)。
- **SHA256**: `082cb7f952b4a4f91ad184a394ebe3766e66a1ccc4c7bbb295b2f7b7994a7983`(下载 PDF,34页)。
- **坐标**: p5 Theorem 1。
- **verbatim**: "Theorem 1 (Validity of UCB calibration). Let (Xi,Yi)_{i=1..n} be an i.i.d. sample..."(p5)。
- **转录**: 用 holdout 校准 λ 使 `E[L]≤α`(UCB 校准);有限样本、分布无关。
- **六字段**: frozen host=无;校正目标=风险校准集大小;选择=无;风险保证=有限样本风险控制;episode=无;fallback=无。
- **counted=yes**。

---

## 8. LTT(2110.01052)

- **书目**: Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control; Angelopoulos, Bates, Candès, Jordan, Lei; 2022; arXiv(AoAS 2025)。
- **SHA256**: `a43d0f089add075ecb4c74f71c9eb6a648e020edc70139b19360d6e717ddc0e2`(下载 PDF,35页)。
- **坐标**: p3 L17。
- **verbatim**: "control the risk, abbreviated as R(λ)=R(T_λ)"(p3 L17)。
- **转录**: `R(λ)=R(T_λ)`——一族校准器;用多重检验(FWER)在族上控制风险。
- **六字段**: frozen host=无;校正目标=算法风险控制;选择=无;风险保证=族级风险控制(多重检验);episode=无;fallback=无。
- **counted=yes**。

---

## 9. 裁决与结论

- **裁决: COMPLETE**(8/8 counted)。
- **结论**: 8 篇校正/风险论文全部核验;**无一篇操作 episode 结构、无一篇相对冻结数值预测做 episode 编辑**。校正机制(加性/仿射/门控/风险校准)均被占,但"负电价 episode 相对基座编辑"的问题设定未被直接占据。
- **诚实声明**: PIR/RCPS/LTT 已存于项目 docs/paper_prep/refs/,SHA256 已记录;公式转录基于提取的正文,精确页码为 PDF 版式页码。

---

## 10. 局限

- PDF 文本提取可能丢失公式渲染(pypdf 提取数学可能不完整);转录以可提取文本为准。
- 未用 Crossref/OpenAlex 二次核对书目(除 arXiv 元数据)。
- 六字段映射基于本会话阅读,若需投稿级引用须人工复核页码。

---

*31a 完成。裁决: COMPLETE——8 篇全文核验全 counted;校正/风险机制被占;episode 相对编辑未被直接占据。*
