# BECH / HCH-v2 研究推进札记 — 2026-08-13 夜：终于该从显微镜前抬头了

R1A.11 绿了以后，我第一反应不是“终于可以扩模型了”，反而是重新想了一遍：我们是不是在一个很小的世界里待得太久了。

从 R1A 到 R1A.11，我们一直围绕 DE、PJM、NEM，Linear、MLP，把 Candidate、W1、value、calibration、router 一层层拆。这个过程很值，因为如果这些内部机制都没弄明白，直接堆更多市场和 Host，只会得到一大张没人知道为什么好或坏的表。

但用户这次提醒得很对。HCH 从一开始就不是为了把 PJM:MLP 修漂亮。真正的目标一直是一个可迁移模块：换市场、换 host、换特征条件，甚至以后 DA 换 RT，它仍然知道自己应该怎么工作。

所以现在到了该从显微镜前抬头的时候。

R1A.11 本身还有一个我想诚实留下的细节。最早的成功条件只写了 map stability 不能严重漂移，没有明确说只看最终被部署 C3 的 domain。第一次 verdict 因为 PJM:Linear 一个很大的 map jump 判成了 YELLOW，后来把它重新解释成“这个 domain 根本不会部署 C3，所以它的 C3 map 稳不稳定不应该阻断 deployment”。我觉得这个解释在工程和统计职责上是对的，但它毕竟是看到结果之后才把语义说清楚。

这不至于推翻 R1A.11。selector 选择本身完全是 pre-S4 的，PJM:MLP 的严格 LCB 也真的跨过了 0。但是它是一个很好的实验治理提醒：到了 R1B，我们不能再让 success criteria 留这种解释空间。以后 seen/unseen、source/holdout、什么叫 collapse、什么叫 safe degradation，都要在跑之前写清楚。

更重要的是，我不想把 R1B 做成“再加 LSTM 和 PatchTST，然后平均分再高一点”。

那样很容易局部最优。

真正值得看的，是四个格子：

seen market + seen host；
seen market + unseen host；
unseen market + seen host；
unseen market + unseen host。

如果最后一个格子还能站住，我才会开始相信 model-agnostic 和 cross-market 这两个词。

这也让我重新看了最近 universal time-series model 的工作。Moirai 真正面对的是 27B observations、九个 domain、不同 frequency、不同 variate count 和不同 distribution；UniTime 甚至把 variable count、domain distinguishability、不同 domain 收敛速度专门列成 cross-domain training 的三个问题。Moirai-MoE 后来又指出人工按 frequency 去划分专家未必靠谱，短窗口里都可能有完全不同的 distribution。

这些研究给我的感觉不是“我们也该上 MoE”，恰恰相反。

它们是在提醒我们：universal 本身是一组问题，不是一个开关。

所以我现在更坚定把后面的实验拆轴：

R1B 先测 market 和 host；
R1C 再测 feature schema 和 DA/RT；
U0 再问大规模 representation prior 能不能进一步帮忙。

如果所有东西一次混进去，最后即使变好了，我们也不知道 universal 到底来自哪里。

电价领域本身也有类似提醒。已有多市场 transfer learning 工作表明跨市场 pretrain/fine-tune 是有价值的；但最近跨境电价工作也明确发现“更多市场不总是更好”，并且 frequent recalibration 很重要。这和我们这一路实验其实很像：local evidence 有价值，但不能无条件相信；更多数据也不能无条件相信。

所以 R1B 我准备给自己定一个规则：

任何改动如果只让 source domains 更漂亮、却让 DK1 或 unseen PatchTST 更差，就不准进入 universal core。

最多把它放到 local adapter 或 optional branch。

这条规则可能比某个具体网络结构更重要。

服务器现在也终于值得租了。不是因为 4090 会让某个小实验显得更高级，而是因为从这一阶段开始，算力真的开始服务“扩大证据面”：LSTM、PatchTST、DK1、3 seeds。200GB 先够用。等真正到 MOMENT feature bank，再扩 500GB。

还有一个我想现在就并行准备的东西：公开 DA/RT 数据。用户最早就说过，不希望最后只证明“几个 DA 市场可以”。NYISO 官方同时有 DA 和 RT zonal LBMP，ERCOT 也同时有 DAM 和 RTM settlement prices。这两类数据很适合未来 R1C，因为我们可以在同一 market 下只改变 target category，少一个 market confound。

但现在不把它们接进 R1B。

我越来越喜欢这种节奏：一边跑当前最小实验，一边准备下一层证据，但不把未来的问题提前塞进当前方法。

今晚项目的状态应该是：

> R1A 解决了“模块内部到底怎么工作”；R1B 要开始回答“离开我们熟悉的三个市场，它还算不算同一个模块”。

如果答案是否定的，也没关系。那时 MOMENT、更多市场、Data Signature、optional interface 才会有明确的用武之地。

如果答案是肯定的，那我们才真正拥有了一个值得继续做大的 universal correction core。
