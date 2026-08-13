# BECH / HCH-v2 研究推进札记 — 2026-08-13 傍晚

今天这轮让我觉得，项目真正开始从“设计一个漂亮结构”进入“被实验逼着长成它应该有的样子”了。

最开始我们很自然地相信一个逻辑：IAH 给出候选，CAGM 去历史里找证据，DVG 再安全放行。纸面上这个故事非常顺。R1A 真跑出来后，第一层确实是对的——共享 candidate 在六个 domain 上都改善了 CRPS，而且没有坍缩。这是一个很重要的好消息，因为它说明最核心的“跨市场 correction prior”不是幻想。

随后事情变得更有意思。R1A.5 发现历史检索几乎不会给动作排序；R1A.6 又发现 candidate 自己其实知道一点 action value，prequential memory 也知道一点。于是我们很自然地猜，二者融合也许会变好。R1A.7 直接否掉了这个想法：lambda 越往 local evidence 走，整体越差。这个结果其实很漂亮，因为它不给我们“再调一点 lambda 也许就能救回来”的借口。

PJM:MLP 是今天最有价值的反例。它的 host 已经很强，residual 很低。IAH 的 CRPS 还能改善，但 outer atoms 小，fire rate 只有 8.7%，而真正触发时方向还很差。一开始我也把它理解成“atom/action calibration 断裂”。继续往下想，我意识到这里还有一层更基础的问题：我们是不是把 hindsight oracle 可以修，误当成了模型应该能提前知道怎么修？

这两件事不是一回事。

结果出现以后，当然几乎每天都能从 Down/Identity/Up 里挑出更好的动作；但预测时只能看到 \(\mathcal F^-\)。如果给定这些信息后 residual 的方向本来就接近不可预测，那么正确决策完全可能是 Identity。

更关键的是，当前

\[
g_{down}=m^-(2w^- -1)
\]

不是随手设计出来的阈值。它就是三原子分布下 full-atom Down 对 absolute error 的 expected gain。也就是说 tail mass 不超过 0.5 时，不触发其实正是这个 predictive distribution 给出的 Bayes 决策。于是 PJM:MLP 的“91% 不动”本身不能再被我当成错误。真正异常的是：剩下那少数它认为该动的日子，为什么方向还错？

这把问题从“如何让它更多地行动”改成了“这里究竟有没有可预测的行动信息”。我觉得这个改变很重要。

所以下一轮我不想马上造一个 action head。我更想做一个朴素的 predictability probe：把冻结的 IAH 输出、rank、lag context、learned signature 都给一个很简单的 logistic regression，看它能不能预测 Down/Up 到底有没有真实收益。如果一个直接拿真实监督做开发诊断的简单模型都只有 0.52、0.54，那就别骗自己说“再设计个巧妙模块一定能学出来”。这种 domain 就该 abstain。反过来，如果 logistic 都能做到 0.65，而当前 \(w,m\) 做不到，那才真的证明 action mapping 没把已有 representation 里的信息读出来。

这也让我重新看待“安全退化”。以前安全退化比较像保险丝：不确定就别改。现在我觉得它可能是 HCH 的一个核心统计角色——不同 market-host regime 的可行动性本来就不同。模型无关模块如果能识别“这个 host 已经很好，我没有足够证据继续碰它”，这可能比“所有模型都强行提升”更可信。

换句话说，未来完整 HCH 的成功标准可能不应该是“六个 domain 全都正 Spearman”，而应该是：有可预测收益的地方做对动作，没有可预测收益的地方几乎不动，而且不伤 host。这更像我们最初想要的 selective safe corrector。

服务器也顺便定下来了。用户给的列表里我会选海南的 4090 24GB / 100GB RAM，1.28 元每小时。不是现在马上租——R1A.8 本地就够——而是等真正进入 R1B，LSTM、PatchTST、多 seed、DK1 开始以后再租。A100 当前没必要。唯一不满意的是 50GB 数据盘，后面 U0 做 HF / MOMENT feature bank 肯定得扩。

今天我并不觉得 R1A.7 的 YELLOW 是拖慢进度。相反，它让我们少走了一条“继续融合、继续调 retrieval、继续加参数”的路。项目现在越来越像真的研究：每一次失败都把可解释空间切掉一块，最后留下来的结构才有资格写进论文。

今天最想记住的一句话是：

> 不要因为 oracle 事后知道答案，就要求模型事前也必须知道答案。

下一轮真正要回答的是：PJM:MLP 到底是 action mapping 失败，还是它本来就是一个应该安全 abstain 的强 host regime？这个答案会决定 HCH 后半段到底需不需要增加一个新的部件。
