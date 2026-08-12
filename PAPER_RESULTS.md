# DimABSA 论文数据与本项目对比

## 数据来源

数据来自任务总览论文 *SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DimABSA)*（arXiv:2604.07066v1）：

- Table 1：数据集规模，论文第 4 页；
- Table 2：Track A 三个子任务的前两名和官方基线，论文第 7 页；
- Table 9–11：Track A 完整排行榜，论文第 20–21 页。

仓库中的 `results/paper_track_a_table2.csv` 完整录入了 Table 2 的 Track A 数据；`results/paper_eng_rest_comparison.csv` 单独整理了与本项目一致的英文 Restaurant 数据，便于直接比较。

## 数据集简称是什么意思

数据集名字由“语言-领域”构成。例如 `eng-rest` 是英文餐厅评论，`zho-rest` 是中文餐厅评论。

| 缩写 | 含义 |
|---|---|
| eng / jpn / rus / tat / ukr / zho | 英文 / 日文 / 俄文 / 鞑靼文 / 乌克兰文 / 中文 |
| rest / lap / hot / fin | 餐厅 / 笔记本电脑 / 酒店 / 金融 |

本项目本轮 LoRA 实验使用的是 `eng-rest`，所以应与论文的 `eng-rest` 列比较，不能拿中文或日文数字直接比较。

## 三个任务和指标

| 任务 | 模型需要输出什么 | 论文指标 | 如何理解 |
|---|---|---|---|
| Task 1 / DimASR | 已给定 Aspect，预测 Valence 和 Arousal 两个 1–9 分数 | RMSE | 预测分数与标准答案的距离，越低越好 |
| Task 2 / DimASTE | 抽取 `(Aspect, Opinion, VA)` | cF1 | 同时考虑三元组结构是否正确以及 VA 距离，越高越好 |
| Task 3 / DimASQP | 抽取 `(Aspect, Category, Opinion, VA)` | cF1 | 比 Task 2 多预测 Category，越高越好，通常更难 |

### RMSE 表示什么

论文将全部样本的 V、A 平方误差汇总：

`RMSE_VA = sqrt((1/N) × Σ[(V_pred - V_gold)^2 + (A_pred - A_gold)^2])`

其中 `N` 是实例数。数值越接近 0，表示 V/A 程度预测越准确。它不是“准确率”，因此 `1.10` 不是 110%，而是 V/A 二维误差的均方根尺度约为 1.10。

### cF1 表示什么

普通 F1 只判断抽取结构能否精确匹配。论文的 continuous F1 进一步用 V/A 距离给已匹配结构一个 0–1 的连续权重：VA 越接近金标准，这个匹配贡献越接近 1；VA 差得越远，贡献越小。因此 cF1 同时考查“有没有抽对”和“程度分数准不准”。

## 英文 Restaurant：论文与本项目

| Task | 论文第一名 | 论文第二名 | Kimi-K2 基线 | Qwen3-14B 基线 | 本项目 |
|---|---:|---:|---:|---:|---:|
| Task 1 RMSE ↓ | LogSigma 1.1035 | BertKittens 1.1812 | 2.1461 | 2.6427 | **1.2421** |
| Task 2 cF1 ↑ | Takoyaki 0.7021 | nchellwig 0.6985 | 0.4920 | 0.4483 | **0.5397** |
| Task 3 cF1 ↑ | Takoyaki 0.6514 | nchellwig 0.6403 | 0.3746 | 0.2673 | **0.4962** |

这些数据说明：

- Task 1：本项目比 LogSigma 高 `0.1386` RMSE，所以没有超过第一名；也比第二名高 `0.0609`。但明显优于论文里的两个官方大模型基线。
- Task 2：本项目比第一名低 `0.1624` cF1，但比 Kimi-K2 基线高 `0.0477`，比 Qwen3-14B 基线高 `0.0914`。
- Task 3：本项目比第一名低 `0.1552` cF1，但比 Kimi-K2 基线高 `0.1216`，比 Qwen3-14B 基线高 `0.2289`。
- Task 3 通常比 Task 2 难，因为除了 Aspect、Opinion 和 VA，还必须把 Category 完全预测正确。本项目也呈现这一规律：Task 3 比 Task 2 低约 `0.0434` cF1。

这些是比赛结束后的本地复现实验，不是提交到当年 Codabench 的官方参赛成绩，所以不能据此声称官方名次。

## 论文中几个重要团队的方法

### LogSigma：Task 1 英文第一名

论文描述其核心是把 V 和 A 当成两个回归任务：共享一个语言 Transformer 编码器，但分别使用两个回归头。训练时不固定两种损失的权重，而是学习任务各自的 log-variance（对数方差）参数，让噪声更大的目标自动获得更低权重；最后再用多个随机种子训练的模型做集成。这也是 `LogSigma` 名字的来源。

它和本项目的相似之处是都使用 V/A 双输出；差别是我们的双头回归 LoRA 使用普通联合误差，没有加入可学习的 log-variance 损失权重，也只正式训练了一个随机种子。

### Takoyaki：Task 2/3 英文第一名

论文描述其方法不是普通 LoRA：先用多种 BM25 方法从训练集检索相似示例，交给 Gemini 3.0 Pro 生成四元组；再对多个变体做一致性集成，保留高一致性的结果；最后用 LLM 挖掘的纠正规则修复抽取和 Category 错误，并平均重复四元组的 VA 分数。

这说明 Task 2/3 的难点不仅是提示词，还包括检索质量、多个预测器的互补性、结构纠错和长尾 Category。

### 官方基线

论文的 Kimi-K2 Thinking 和 Qwen3-14B 是官方组织者提供的参考系统，不代表这两个基础模型在所有提示词或微调方案下的能力上限。我们的 4B LoRA 超过这些基线，说明监督微调对当前数据集有效；但它不能证明 4B 模型普遍强于 14B 模型。

## 论文数据集规模：英文 Restaurant

Table 1 使用 `文本数 / 实例数`：一条文本可能出现多个 Aspect 或多个三元组/四元组，所以实例数通常大于文本数。

| 切分 | Task 1 | Task 2/3 |
|---|---:|---:|
| Train | 2,284 / 3,659 | 2,284 / 3,659 |
| Dev | 200 / 340 | 200 / 408 |
| Test | 1,000 / 1,504 | 1,000 / 2,129 |

Task 1 的“实例”是给定 Aspect；Task 2/3 的“实例”是完整关系。因此同一批文本在不同任务中的实例数会不同。
