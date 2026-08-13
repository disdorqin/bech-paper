# BECH / HCH-v2 研究推进札记 — 2026-08-13 晚：第一次看到“后半段”真正成形

R1A.9 的结果是到目前为止最让我放心的一次“小胜利”。

不是因为数字有多漂亮。恰恰相反，它又给了我们一个很不舒服的副作用：PJM:MLP 被 C3 修好了，NEM 却差点被 C3 杀死。

但这次和前几轮不同。我不再觉得“问题又换了一个地方”。我开始觉得 HCH 的后半段结构真的在自己长出来。

最初我们设计 CAGM，是希望历史 evidence 能告诉今天的 correction 是否值得执行。后来实验告诉我们 W1 retrieval 根本不会排序 action value。继续往下拆，发现 IAH distribution 自己有 value signal；local memory 有时有帮助、有时有害；融合也没有自动变好。那些当时看起来像挫折的结果，现在反而把结构越削越清楚。

R1A.9 最有价值的一点，是把“CRPS 好但动作坏”变成了一个非常具体的统计现象：PJM:MLP 的 Down mass 跨过 0.5 时，真实 benefit rate 只有 0.168。这个事实比任何抽象的“action calibration 不够”都更有说服力。

更让我满意的是 shared affine 没有被硬救活。四个 shared scalar 自己回到了 identity。它像是在提醒我们：不要为了维持“所有东西都 universal”这个漂亮故事，而强行让一个全局参数解释方向完全不同的 local distortion。

这和我们的三层框架反而越来越吻合。

Universal core 学的是 correction distribution。

Local layer 学的是：在这个市场、这个 host、这段历史里，我到底能不能相信这些 action masses。

但 NEM 又提醒了我们另一件同样重要的事：Local 不是天然更可靠。

它只有十几天 fit evidence，C3 一拟合，就把一个本来 action value 很强的 domain 几乎全部压成 Identity。这个错误如果出现在真实部署里，不是“少赚一点”，而是安全模块把真正的机会关掉。

所以我现在最关心的问题已经不是“怎么把 isotonic 调得更好”，而是一个权限问题：

**凭什么允许某个 domain 启用 local calibrator？**

默认应该相信 universal IAH 的 raw action utility。只有本地历史给出足够证据，证明 raw translation 的确在伤人，而且 local calibration 在前向验证里确实改善，才允许它覆盖 raw policy。

我很喜欢这个 default-off 的结构。

C0 是默认。
C3 是 override。

这样 HCH 不会慢慢退化成“每个市场重新训练一套本地后处理”。通用部分仍然是主体，本地部分只在有证据时介入。

当然，现在还不能手工写“PJM 用 C3、NEM 用 C0”。那是看完 S4 后我们知道的答案。如果把它写成方法，那就是 hindsight routing，再漂亮也不可信。

所以 R1A.10 的任务很朴素：在 S4 之前，让系统自己决定 C0 还是 C3。

如果它做得到，我觉得这是 HCH 后半段第一次有资格从“实验组件”升级成“部署协议”。

如果做不到，也没关系。那说明 local isotonic 只是一个 upper bound，我们就继续研究 regularized calibration，而不是把最好的事后组合包装成方法。

这段研究到现在，我越来越觉得好实验不是给方案加分，而是不断削掉我们没有资格声称的东西。W1 被削掉了，shared action calibration 被削掉了，强制所有 domain 都修正的想法也被削掉了。留下来的东西反而越来越清楚。

今晚我会这样描述项目：

> 我们已经知道 local calibration 有时能救命，现在要证明系统能在不知道未来的情况下知道什么时候该用它。

如果这一步过了，R1B 加 LSTM、PatchTST 和 DK1 才真正有意义。那时我们扩大测试的，不再是一个还在成形的想法，而是一条经过多轮失败和修正之后已经基本闭合的机制。
