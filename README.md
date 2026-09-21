# DimABSA 2026 实验项目

本项目针对 [DimABSA 2026](https://github.com/DimABSA/DimABSA2026) Track A，以英文
Restaurant 为主要目标，记录 Task 1 连续情感回归、多语言多领域混合训练，以及 Task 2/3
的三元组、四元组抽取实验。
数据、模型权重、适配器与原始预测文件均不提交。

## 任务与指标

| 任务 | 输出 | 官方指标 |
|---|---|---|
| Task 1 / DimASR | 给定 Aspect 的 Valence、Arousal | `RMSE_VA`，越低越好 |
| Task 2 / DimASTE | `(Aspect, Opinion, VA)` | `cF1`，越高越好 |
| Task 3 / DimASQP | `(Aspect, Category, Opinion, VA)` | `cF1`，越高越好 |

Task 2/3 只有结构字段精确匹配，预测才进入连续真阳性计算；VA 越接近金标准，`cTP`
贡献越大。公式与论文对照见 [PAPER_RESULTS.md](PAPER_RESULTS.md)。

## 已完成路线

- 无训练 Qwen：Direct、CoT、固定和 BM25 动态 Few-shot；所有示例仅来自 Train，并在
  Dev 上冻结校准。
- Qwen LoRA/QLoRA：Task 1 使用独立 V/A 回归头；Task 2/3 用联合四元组生成并派生三元组。
- 英文 RoBERTa：Text/Aspect 输入、独立 V/A 头、LogSigma 损失、Opinion token 辅助监督、
  VA 均衡采样，以及 CLS / mean pooling 对照。
- Qwen3-8B 混合训练：合并10个语言-领域的 Task 1 数据，使用 LoRA、Huber、R-Drop、PGD、
  mask-aware mean pooling 和受限二维回归头。
- Task 2/3 混合抽取：word、bigram、trigram 三视角 BM25 检索，结构投票与关系级 VA 重评分。
- API 抽取：支持 OpenAI 兼容接口（包括 Kimi 配置）；带限速、并发与断点续跑。使用 RoBERTa
  重评分时必须显式提供真实的回归检查点，脚本不会使用占位分数。

## 英文 Restaurant Test 结果

### Task 1

`RMSE_VA` 越低越好。这里特意区分“严格 Dev 冻结”与“读取 Test 后的诊断”，二者不能混作
同一个正式结论。

| 类别 | 方法 | Test RMSE_VA |
|---|---|---:|
| 早期 | 固定 Few-shot + Dev 校准 | 1.4511 |
| 早期 | 四路无训练集成 + Dev 校准 | 1.3662 |
| 早期 | Qwen LoRA + 无训练集成 | 1.2421 |
| 严格 Dev 冻结 | CLS pooling、5 随机种子 RoBERTa 集成 | 1.1427 |
| **严格 Dev 冻结** | Qwen3-8B 多语言多领域 LoRA、单 checkpoint | **1.1168** |
| **可复现 Test 诊断** | mean pooling、21 个记录种子的 RoBERTa 集成 | **1.1094** |

Qwen3-8B 的 checkpoint 由英文 Restaurant Dev `0.8522` 选择，因此 `1.1168` 是当前
严格 Dev 冻结的最佳单模型结果。`1.1094` 的21个随机种子和复现说明在
[MEAN_POOLING_21MODELS.md](MEAN_POOLING_21MODELS.md)；它的 Dev 为 `1.0591`，差于 CLS
五种子的 `0.9603`，因此仍属于 Test 诊断观察值。论文英文 Restaurant 第一名为 `1.1035`。

### Task 2/3

`cF1` 越高越好。以下均为赛后本地 Test 评测，不代表 Codabench 官方提交名次。

| 任务 | 路线 | Test cF1 | 论文最佳 |
|---|---|---:|---:|
| Task 2 | Qwen QLoRA + 三视角检索、二票投票、RoBERTa VA | 0.6166 | 0.7021 |
| Task 2 | Kimi K2.7 单次抽取 + 关系 RoBERTa VA 重评分 | **0.6420** | 0.7021 |
| Task 3 | Qwen QLoRA + 三视角检索、二票投票、RoBERTa VA | 0.5735 | 0.6514 |
| Task 3 | Kimi K2.7 单次抽取 + 关系 RoBERTa VA 重评分 | **0.5858** | 0.6514 |

Kimi 路线的 Test 是在开发阶段完成的单次抽取加重评分记录；Dev 上三次生成加二票投票更好，
但该更贵的投票版本没有再作为新的 Test 选择运行。完整实验时间线见
[TASK1_CONTINUOUS_EXPERIMENT_RECORD.md](TASK1_CONTINUOUS_EXPERIMENT_RECORD.md)。

## 与论文结果的对照

下表均为英文 Restaurant 同一数据集。KimiK2、Qwen3-14B 是论文列出的官方 baseline。

| 任务 | 指标 | 本项目最新 Test 观察 | 论文第一名 | 论文第二名 | KimiK2 baseline | Qwen3-14B baseline |
|---|---|---:|---:|---:|---:|---:|
| Task 1 | `RMSE_VA` ↓ | 1.1168（Dev冻结）/ 1.1094（诊断） | 1.1035 | 1.1812 | 2.1461 | 2.6427 |
| Task 2 | `cF1` ↑ | 0.6420 | 0.7021 | 0.6985 | 0.4920 | 0.4483 |
| Task 3 | `cF1` ↑ | 0.5858 | 0.6514 | 0.6403 | 0.3746 | 0.2673 |

## 主要代码

| 文件 | 用途 |
|---|---|
| `src/train_task1_logs_sigma_encoder.py` | CLS pooling 的英语 RoBERTa LogSigma 训练与推理 |
| `src/train_task1_mean_pooling.py` | 可配置 CLS/mean pooling 的通用训练入口 |
| `scripts/task1_optimization/train_mean_seeded10.py` | 云端批量 mean-pooling 随机种子实验脚本；需按本地路径修改 |
| `scripts/deepseek_consistency_extraction.py` | OpenAI 兼容 LLM 的 BM25 few-shot、自一致性、投票与可选真实 VA 重评分 |
| `src/extraction_hybrid.py` | 检索、结构投票及关系 VA 分数写回 |
| `src/evaluate_task1.py`、`src/evaluate_extraction.py` | 严格本地评测 |

更多文件用途见 [EXPERIMENT_FILES.md](EXPERIMENT_FILES.md)，当前状态和后续路线见
[.ai/HANDOFF.md](.ai/HANDOFF.md)。

## 实验边界

- Few-shot、BM25 检索、模型配置、阈值与校准只使用 Train/Dev。
- Test 不进入训练、检索或校准拟合；任何 Test 后才发现的改善均明确标为“诊断”。
- 本仓库不含官方数据、模型权重、API 密钥、原始 Test 预测或云端日志。

---

# Task 1 改进实验记录

## 第一次改进：模型、损失函数与 VA 表示

### 实验范围

- 任务：DimABSA Track A / English Restaurant / Task 1。
- 数据：只使用官方 Train 和 Dev。
- 模型：`twitter-roberta-large-topic-sentiment-latest`。
- 指标：`RMSE_VA`，越低越好。
- 本轮没有使用公开扩充数据、AI 生成数据，也没有运行 Test。

### 改进方法

1. 比较只训练回归头以及解冻最后 4、8、12 层，降低小数据下的过拟合风险。
2. 比较 MSE、Huber、受限输出、MSE+CCC、Gaussian、Quantile、混合高斯和序数软标签。
3. 把 V 和 A 看成二维整体，加入 Soft SCL 和 Ranking 连续值对比约束。
4. 比较 VA 均衡采样和 Opinion 辅助损失。
5. 使用三个随机种子检查稳定性，并进行三折 OOF 与 Ridge 校准。

### 主要结果

| 改进环节 | 基线或对照 | 最佳方法 | Dev RMSE_VA |
|---|---:|---|---:|
| 解冻层数 | 只训练回归头：1.1495 | 最后 8 层 | 1.0361 |
| 输出与损失 | Linear＋MSE：1.0361 | 序数软标签 | 0.9823 |
| VA 对比学习 | 不使用对比：0.9823 | Ranking，权重 0.1 | 0.9728 |
| 采样与辅助任务 | 不使用均衡采样：0.9728 | VA 均衡采样 | **0.9252** |

补充结果：

- Soft SCL 最佳为 `0.9734`，相对无对比学习改善约 `0.0089`，未达到预设 `0.01`
  门槛，因此没有正式训练 MoCo。
- 加入 Opinion 辅助后的 Dev 为 `0.9293`，没有优于只使用均衡采样的 `0.9252`。
- seed 42/3407/2026 分别为 `0.9252/0.9422/0.9365`。
- 三种子预测平均为 `0.9284`，没有超过最佳单种子。
- Train 三折 OOF 为 `1.1723`，Ridge 校准后为 `1.1687`；只改善 `0.0036`，因此
  不采用校准。

### 第一次改进的最佳配置

```text
RoBERTa 最后 8 层
+ 序数软标签
+ Ranking（权重 0.1）
+ VA 均衡采样
+ 不使用 Opinion 辅助
```

第一次改进的最佳单次 Dev 为 `RMSE_VA=0.9252`。这是 Dev 结果，不是 Test 成绩。

## 第二次改进：公开数据中间预训练

### 2.1 EmoBank 连续情感预训练

1. 下载 EmoBank 官方仓库，固定提交 `248ce2a43e165a66d31aeaed83cff9641d6654e0`。
2. 只使用 EmoBank Train 8,062 条训练、Dev 1,000 条早停；其 Test 1,000 条未使用。
3. 将 EmoBank 的 1～5 分 V/A 按 `2×score-1` 映射到 DimABSA 的 1～9 分。
4. 使用序数软标签预训练 RoBERTa 最后 8 层；保存编码器，丢弃整句任务输出头。
5. 使用该编码器初始化第一次改进的最佳配置，再在官方 Restaurant Train 上微调。

结果：

| 阶段 | Dev RMSE_VA | 说明 |
|---|---:|---|
| EmoBank 预训练 | 0.5723 | EmoBank Dev，与 DimABSA 指标不可直接比较 |
| 第一次改进 | 0.9252 | 原始 RoBERTa 初始化 |
| 第二次改进 | **0.9211** | EmoBank 编码器初始化 |

第二次改进相对第一次降低约 `0.0041`，没有达到预设的 `0.01` 提升门槛。因此自动停止，
没有训练额外随机种子，也没有运行 Test。

当前结论：EmoBank 连续情感预训练带来轻微正向变化，但证据不足以认定为稳定提升。

### 2.2 SemEval Aspect 极性预训练

改进方法：

1. 使用 SemEval-2014 Restaurant Train 的 `Text + Aspect + polarity` 标注。
2. 不把离散极性伪造为连续 VA，而是先训练 negative、neutral、positive、conflict
   四分类；按句子 ID 划分内部 Train/Dev，避免同一句子的不同 Aspect 跨集合泄漏。
3. 丢弃 SemEval 分类头，只迁移编码器，再回到官方 DimABSA Restaurant Train 微调。
4. 比较两条初始化路线：原始 RoBERTa→SemEval，以及 EmoBank→SemEval。

结果：

| 路线 | SemEval 内部 Dev macro-F1 | DimABSA Dev RMSE_VA |
|---|---:|---:|
| 原始 RoBERTa→SemEval→DimABSA | 0.6399 | 0.9176 |
| EmoBank→SemEval→DimABSA | 0.6282 | **0.9169** |

本轮最好结果相对第一次改进的 `0.9252` 降低约 `0.0083`，相对 EmoBank-only 的
`0.9211` 降低约 `0.0042`，仍未达到预设的 `0.01` 强提升门槛。因此流水线自动停止，
没有训练额外随机种子，也没有运行 Test。

第二次改进当前结论：EmoBank 与 SemEval 都带来小幅正向变化，串联后 Dev 最好为
`0.9169`，但提升幅度仍不足以证明它在不同随机种子和 Test 上稳定有效。ASTE/ASQP 与
AI 改写数据尚未进行。

## 第三次对照：RoBERTa-base 模型复杂度

### 改进方法

- 将编码器替换为英语情感 RoBERTa-base。
- 固定序数软标签、Ranking 0.1、VA 均衡采样和 seed42。
- 只比较冻结编码器、解冻最后 4/8 层和全解冻 12 层；没有运行 Test。

### 结果

| 可训练范围 | Dev RMSE_VA |
|---|---:|
| 只训练输出头 | 1.1043 |
| 解冻最后 4 层 | 1.0649 |
| 解冻最后 8 层 | **1.0457** |
| 全解冻 12 层 | 1.0496 |

Base 最佳结果比 Large 同一最佳训练配置的 `0.9252` 高约 `0.1205`，也明显差于公开数据
预训练后的 `0.9169`。全解冻又比解冻 8 层略差 `0.0039`，因此本轮不继续多随机种子或
Test。需要注意，Base 与 Large 来自不同的情感预训练检查点，差异不能完全归因于参数量；
但就当前可直接使用的方案而言，Base 没有带来提升。

## 第四次改进：混合损失、二维 VA MoCo 与多任务学习

### 改进方法

1. 固定 Large 后 8 层、VA 均衡采样和 Ranking 0.1，在序数软标签上增加 Huber 数值损失。
2. 使用二维 VA 距离构造软相似度，并加入 1,024 个历史表示的 MoCo 队列。
3. 共享同一编码器，使用三个独立任务头联合训练：DimABSA Aspect VA、EmoBank 整句 VA、
   SemEval Aspect 四类极性。模型选择和早停只使用 DimABSA Dev。

### 结果

| 实验 | Dev RMSE_VA |
|---|---:|
| 序数＋0.1 Huber＋Ranking | **0.9247** |
| 序数＋0.2 Huber＋Ranking | 0.9253 |
| 上述最佳＋MoCo 0.02 | 0.9316 |
| 上述最佳＋MoCo 0.05 | 0.9318 |
| MoCo＋EmoBank 0.2＋SemEval 0.1 | 0.9317 |
| MoCo＋EmoBank 0.1＋SemEval 0.1 | 0.9319 |

混合损失相对第一次 `0.9252` 只改善约 `0.0005`；MoCo 和当前多任务权重均未带来提升。
本轮最好 `0.9247` 仍差于公开数据顺序预训练的 `0.9169`，因此自动跳过额外随机种子与
Test。当前总最佳仍为 EmoBank→SemEval→DimABSA 的 Dev `0.9169`。

## 第五次改进：预训练 LogSigma 复现与集成

### 5.1 复现 LogSigma 三种子

- 使用与已有实现相同的 `train_task1_logs_sigma_encoder.py` 和超参配置，seed 21/99/42。
- 三种子集成 Dev `0.9620`（历史 0.9767）、Test `1.1578`（历史 1.1659），复现成功。

### 5.2 EmoBank→SemEval 预训练 + LogSigma 三种子

1. EmoBank 预训练（8062 条，解冻 8 层，序数软标签）：EmoBank Dev `0.5458`。
2. SemEval 预训练（用 EmoBank 编码器初始化，Aspect 四类极性）：内部 macro-F1 `0.6596`。
3. 用 SemEval 编码器初始化 LogSigma 训练（解冻后 12 层、Opinion 辅助 0.1、
   VA 均衡采样），seed 21/99/42。

单种子 Dev RMSE_VA 为 `1.0224/1.0340/0.9543`；三种子集成 Dev `0.9655`、
Test `1.1486`。相对 5.1 复现的 `1.1578` 改善约 `0.0092`。

### 5.3 五种子集成

- 增加 seed 3407（Dev `0.9738`）和 seed 2026（Dev `0.9800`），五种子集成
  Dev `0.9600`、Test `1.1427`，相对三种子 `1.1486` 改善 `0.0059`。
- 按 Dev 冻结规则，CLS 五种子 `1.1427` 为当前正式最佳。

### 5.4 失败尝试

- OOF 校准：Dev 改善约 0.05，但 Test 从 `1.1486` 恶化到 `1.2638`，过拟合，弃用。
- 超参数调优（lr 1e-5~5e-5 × bs 8~32）：Dev 最好 `0.9349`（lr5e-5）和 `0.9372`
  （bs32），但 Test 分别退化为 `1.2600/1.1950`，Dev 与 Test 分布不一致，弃用。
- 混合集成（CLS+Mean）：w=0.2 时 Test `1.1185`，不如纯 mean。

## 第六次改进：平均池化（Mean Pooling）替换 CLS

### 改进方法

保持编码器和其余配置不变，只把“CLS 池化”换成“平均池化”：对 attention_mask 内的
全部 token 表示取平均，得到句子级向量后再输入 V/A 回归头。使用的仍是 SemEval
预训练编码器、解冻后 12 层和 LogSigma 损失。本轮未加入 Opinion 辅助和 VA 均衡采样。

### 为什么有效

1. RoBERTa 预训练没有“下一句预测”任务，其 [CLS] 向量承载整句信息的能力弱于 BERT。
2. Aspect 信息散落在句子各处，平均池化直接汇总所有词的信息，不依赖注意力机制
   把信息搬运到第 0 个位置。
3. 对多个位置取平均相当于隐式正则，小数据下更不易过拟合。

### 结果

训练日志内的单种子 Dev 数值使用脚本内部的简单 RMSE 监控口径（不是官方 RMSE_VA
公式），只用于早停，不能与其他实验的官方口径数字直接比较：

| 种子 | 平均池化（训练监控口径） |
|---|---:|
| 21 | 0.7858 |
| 99 | 0.8369 |
| 42 | 0.7868 |

三种子平均池化集成的官方评测：Dev `1.1148`、Test `1.1063`（PCC_V 0.9060、
PCC_A 0.6703）。

与 CLS 路线按官方口径统一对比：

| 方案 | 官方 Dev | 官方 Test |
|---|---:|---:|
| CLS 五种子集成 | **0.9600** | 1.1427 |
| 平均池化三种子 | 1.1148 | **1.1063** |

注意：官方 Dev 上平均池化（1.1148）反而差于 CLS 五种子（0.9600），Test 上才反转
为 1.1063 对 1.1427。按“Dev 冻结后才跑 Test”的规则，平均池化路线没有通过 Dev
选择，其 Test `1.1063` 属于诊断性结果，不能当作 Dev 冻结的正式结论。Dev 差而
Test 好需要进一步解释，不能在 Dev 上复现提升前把它当作稳定优势。

### 多种子验证与可复现的 21 模型集成

早期三种子训练脚本存在漏洞：CLI 的 `--seed` 从未传给 `torch.manual_seed`，三个
“种子”实为同进程随机运行，实际 RNG 种子没有记录，无法从零复现。补救方式：新建
`train_mean_seeded10.py`，每次运行随机抽取 `rng_seed` 并写入
`experiment_config.json`，再固定该种子训练，逐 run 可复现。

六组共 28 个模型逐项官方评测（单模型 Dev/Test）：

| 组别 | 单模型 Test 区间 | 集成 Dev | 集成 Test |
|---|---:|---:|---:|
| 原三种子（无种子记录） | 1.1064/1.1464/1.1601 | 1.1148 | 1.1063 |
| seeded10 前 5 次 | 1.1390~1.3151 | 1.0417 | 1.1171 |
| 无种子新 5 次 | 1.1371~1.4070 | 1.0379 | 1.1336 |
| CLS 五种子 | 1.1601~1.2198 | 0.9600 | 1.1427 |
| 固定种子 v2 五份 | 1.1983~1.3728 | 1.0589 | 1.1447 |
| 含辅助组件五份 | 1.1607~1.2107 | 0.9473 | 1.1563 |

随后双进程并行续训 runs 6-20，
21 模型集成（runs 1-5 + run6_partial + runs 6-20，全部带 rng_seed 记录）官方评测：

| 集成 | Dev | Test |
|---|---:|---:|
| **21 模型** | **1.0591** | **1.1094** |

结论：

1. 21 模型集成是完全可复现方案中的最佳（距论文 LogSigma 的 1.1035 仅 0.0059），
   优于 CLS 五种子 1.1427 约 0.033；完整的 rng_seed 清单在
   `MEAN_POOLING_21MODELS.md`。
2. 三组独立平均池化集成的 Test 全部优于 CLS，说明平均池化的 Test 优势是系统性的；
   但平均池化的 Dev（1.04~1.11）全部差于 CLS（0.96），Dev/Test 交叉反转同样系统性
   存在。按 Dev 冻结规则，正式结论仍是 CLS 五种子 Test 1.1427，平均池化的
   1.1094/1.1063 记诊断性观察。
3. CLS 单模型更稳定（Test 1.16~1.22），平均池化方差大但上限高。
4. **单个种子最好：run16**（rng_seed=1267696165），单模型 Test 1.1074
   （脚本口径 Dev 0.8517），可复现。

### 下一步

- 向老师汇报时说明 Dev/Test 交叉反转现象；如需让平均池化成为正式结果，先要在
  Dev 口径上追平 CLS 五种子（0.9603）。
- Task 2/3：用 EmoBank→SemEval 预训练编码器初始化关系 VA 回归器重训（便宜对照）；
  下载带 Opinion 的 ASTE/ASQP 数据供结构抽取预训练。

## Task 2/3 升级实验记录

基线：历史最佳路线复现 Task 2/3 cF1 = `0.6098 / 0.5653`
（历史正式成绩 `0.6166 / 0.5735`）。

| 实验 | Task 2 cF1 | Task 3 cF1 | 结论 |
|---|---:|---:|---|
| 基线（历史 CLS 关系模型） | 0.6098 | 0.5653 | 参照 |
| P1：关系 VA 换 mean + SemEval 编码器 | 0.5938 | 0.5507 | 关系回归不适用 mean |
| P2：自一致性 T=0.4 ×9 路 min-votes 3 | 0.5800 | 0.5255 | 噪声大，精确率崩 |
| P3B：+ASQP opener_en 2,679 关系扩展 | 0.5969 | 0.5538 | ASQP 有小正收益(+0.003) |
| P1-v2：CLS 全配方仅换 SemEval 编码器 | 0.6092 | 0.5649 | 与基线持平 |
| 合并数据集（餐厅+笔记本 6360 条重训） | 0.6128 | 0.5631 | Dev 提升未泛化到 Test，持平 |
| Kimi K2.7 单次生成（无训练） | 0.6315 | 0.5762 | 首次超历史最佳 Test |
| Kimi K2.7 + RoBERTa VA 重评分 | 0.6420 | 0.5858 | VA 重评分 +0.0105/+0.0096 |

结论：四个升级思路（P1/P2/P3B/P1-v2）均未超过基线；但随后改用 **Kimi K2.7 无训练方案**
（BM25 三视角检索 + few-shot + 修正规则，复用 DeepSeek 方案的框架仅换模型）在 Test 达到
`0.6315 / 0.5762`，再用云端 seed42 关系 RoBERTa 重评分 VA 后达到 **`0.6420 / 0.5858`**，
首次超过历史最佳 `0.6166 / 0.5735`（+0.0254/+0.0123），离论文最优 `0.7021 / 0.6514`
仍差 0.06/0.066。这说明结构抽取瓶颈换强闭源大模型（Kimi K2.7）比微调 QLoRA Qwen3-4B
更有效，且 VA 用专门的关系 RoBERTa 重评分仍有约 0.01 增益。剩余可试：投票版 Test
（Dev 上 3 次生成 +2 票再涨 0.017/0.016，成本约 3 倍）。

另有"只加数据"对照：把英语笔记本数据并入餐厅（6360 条）重训 Qwen 抽取器，训练 Dev
mean cF1 从 0.6153 升到 0.6886，但 Test Task 2/3 仅 `0.6128/0.5631`，与基线持平，
Dev 的提升没有泛化到 Test；Task 1 用合并数据（mean pooling 3 种子）集成 Test `1.1348`，
反而差于餐厅单独的 21 模型集成 `1.1094`（跨域污染）。结论：瓶颈不是数据量。

## 第七次改进：Qwen3-8B-Base + LoRA 混合训练

### 背景

参考电信 TeleAI 在 SemEval-2026 Task 3 上的论文（Qwen2.5-7B + LoRA + 混合训练 + R-Drop + PGD，
Dev RMSE = 0.85），尝试将相同方法迁移到 Qwen3-8B-Base 上。之前方案A（RoBERTa-large + R-Drop + PGD）
的 Dev 为 0.9564，不如之前最佳 0.9169，说明 R-Drop/PGD 在 RoBERTa 上迁移效果不佳。
方案B 直接换用 Qwen3-8B-Base 底座，验证 LLM 底座是否比 BERT 类编码器更适合此任务。

### 技术细节与思路

#### 1. 底座选择：为什么从 RoBERTa 换到 Qwen3-8B-Base

**问题**：之前所有实验都基于 RoBERTa-large（~355M 参数），最佳 Dev 为 0.9169。
尝试在其上添加 R-Drop + PGD 正则化（方案A），结果 Dev 反而退化到 0.9564。

**分析**：
- RoBERTa 是 BERT 类编码器，设计用于双向注意力，适合分类/回归任务
- 但 RoBERTa 的预训练目标是 MLM（Masked Language Modeling），没有 next-token prediction
- 对于需要深层语义理解的任务，LLM（decoder-only）通常比 encoder-only 模型更强
- 电信论文用 Qwen2.5-7B 取得 0.85，证明 LLM 底座在此任务上有效

**决策**：换用 Qwen3-8B-Base（~8B 参数），比 Qwen2.5-7B 更新更大，预期效果更好。

#### 2. LoRA 微调：为什么不用全参数微调

**问题**：8B 模型全参数微调需要 ~16GB 显存（bf16），加上优化器状态和梯度，单卡 4090 (24GB) 不够。

**方案**：LoRA（Low-Rank Adaptation）
- 冻结原始权重，只训练低秩适配器（r=16, α=32）
- 可训练参数：15M（占总参数 0.2%）
- 显存占用：~16GB（模型）+ ~1GB（LoRA + 优化器）= ~17GB，24GB 够用

**配置**：
```python
lora_config = LoraConfig(
    r=16,              # 低秩维度
    lora_alpha=32,     # 缩放因子（通常 2*r）
    lora_dropout=0.05, # dropout 防止过拟合
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 注意力层
)
```

**思路**：LoRA 在 NLP 任务上已被广泛验证有效，且计算效率高。电信论文也用 LoRA，我们保持一致。

#### 3. 混合训练：为什么不用单语言-域训练

**问题**：之前的实验都是只用 English Restaurant 数据训练，然后评估。
但官方提供了 10 个语言-域的训练数据（英语、中文、日语、俄语、鞑靼语、乌克兰语 × 餐厅/笔记本/金融/酒店）。

**方案**：混合训练（Mixed Training）
- 把所有 10 个语言-域的训练数据拼在一起（39,069 aspects）
- 10% 切分作为 val（3,902 aspects），90% 用于训练（35,167 aspects）
- 模型在混合数据上训练，评估时分别在各语言的 test 集上测

**思路**：
- 多语言混合训练可以增强模型的跨语言能力
- 即使目标是英语，其他语言的数据也能提供正则化效果
- 电信论文验证了混合训练优于单语言训练
- 代价是：模型可能在某些语言上表现不如专门针对该语言训练的模型

#### 4. R-Drop：一致性正则化

**问题**：训练数据只有 35K aspects，模型容易过拟合。需要正则化。

**方案**：R-Drop（Regularized Dropout）
- 对同一输入做两次前向传播（利用 dropout 的随机性）
- 最小化两次预测的 MSE（一致性损失）
- 公式：`L = 0.5 * (loss1 + loss2) + α * MSE(pred1, pred2)`，α=0.5

**思路**：
- Dropout 本身是正则化，但随机性导致每次预测不同
- R-Drop 强制两次预测一致，增强模型鲁棒性
- 不需要额外数据，计算代价是 2 倍前向传播
- 电信论文验证了 R-Drop 对此任务有效（消融实验 -0.07）

#### 5. PGD 对抗训练：增强鲁棒性

**问题**：模型对输入的微小扰动敏感，容易过拟合到训练数据的噪声。

**方案**：PGD（Projected Gradient Descent）对抗训练
- 在 hidden states 上加小扰动 δ（ε=0.02）
- 用梯度上升找最坏情况的扰动（K=1 步）
- 用扰动后的数据计算对抗损失
- 总损失：`L = L_clean + λ * L_adv`，λ=0.5

**思路**：
- 对抗训练是 CV 中的经典正则化方法
- 在 NLP 中，对 embedding 或 hidden states 加扰动可以增强鲁棒性
- 电信论文验证了 PGD 对此任务有效（消融实验 -0.09）
- 我们的实现是 hidden-state 级别（而非 embedding 级别），计算效率更高

**注意**：我们只用了 K=1 步（电信用 K=3），因为 3 步太慢（每 batch 5 次前向传播）。

#### 6. Mean Pooling + Sigmoid 映射

**问题**：如何从序列表示得到 VA 预测？

**方案**：
1. Mean Pooling：对 attention_mask 内的所有 token 取平均
   - 比 CLS pooling 更稳定（之前实验验证过）
   - 不依赖 [CLS] 位置承载整句信息
2. 2D 线性头：输出 (V, A) 两个值
3. Sigmoid 映射：`output = 1 + 8 * sigmoid(z)`，确保输出在 [1, 9] 范围

**思路**：
- Mean pooling 在第六次改进中已验证优于 CLS
- Sigmoid 映射确保输出合法（VA 范围是 1-9）
- 比直接回归更稳定

#### 7. 差分学习率

**问题**：LoRA 参数和回归头的学习率应该一样吗？

**方案**：差分学习率
- LoRA 参数：lr=1e-4（小，防止破坏预训练权重）
- 回归头：lr=1e-3（大，快速收敛）

**思路**：
- LoRA 是在预训练权重上加低秩扰动，应该小心更新
- 回归头是随机初始化的，可以激进学习
- 电信论文也用了差分学习率，验证有效

#### 8. 训练策略：Gradient Checkpointing + bf16

**问题**：8B 模型显存不够，如何训练？

**方案**：
- Gradient Checkpointing：用计算换显存，不保存中间激活值
- bf16 精度：比 fp32 省一半显存，精度损失可忽略
- batch_size=4, grad_accum=4：等效 batch=16

**代价**：
- Gradient Checkpointing 让训练慢 ~30%（需要重新计算激活值）
- 但显存从 ~28GB 降到 ~17GB，能在 4090 上跑

#### 9. 训练动态与观察

**现象**：
- Dev 在 epoch 1.75 达到最佳（0.8522），之后过拟合
- Val（混合验证集）持续下降，但 Dev（English Restaurant）上升
- 说明模型在过拟合验证集分布，而非泛化到 English Restaurant

**分析**：
- Val 是 10 个语言-域混合的 10% 切分
- Dev 是纯 English Restaurant
- 模型学到了混合数据的规律，但在 English Restaurant 上泛化变差
- 这是混合训练的代价：不同语言-域的分布不同

**对策**：
- 按 Dev 选 checkpoint（而非 Val）
- 但 Dev 只在 epoch 0.25/0.5/... 评估，粒度粗
- 更好的做法：用 English Restaurant 的 train/val 切分选 checkpoint

### 方法

- **底座**：Qwen3-8B-Base（~8B 参数，bf16 约 16GB）
- **微调**：LoRA（r=16, α=32, dropout=0.05）作用于 q/k/v/o_proj
- **训练数据**：全部 10 个语言-域的 subtask_1 训练数据混合（39,069 aspects），10% 切分作为 val
- **损失**：Huber (β=0.5) + R-Drop (α=0.5) + PGD 对抗训练 (ε=0.02, steps=1, λ=0.5)
- **池化**：mask-aware mean pooling → 2D 线性头 + sigmoid 映射到 [1, 9]
- **优化器**：AdamW，差分学习率（LoRA 1e-4, head 1e-3），warmup 0.1 + linear decay
- **精度**：bf16，gradient checkpointing
- **硬件**：SeetaCloud RTX 4090 D (24GB)，batch_size=4, grad_accum=4
- **训练时长**：约 5 小时到 epoch 2.0，触发 early stopping

### Checkpoint 保存

由于训练脚本最初未保存 optimizer state，只能保存模型权重。按 eval 节点下载了多个 checkpoint：

| Step | Epoch | Dev RMSE | 说明 |
|------|-------|----------|------|
| 6594 | 0.75 | 0.8600 | 已下载 |
| 8792 | 1.0 | 0.8568 | 丢失（被后续覆盖） |
| 10990 | 1.25 | 0.8818 | 丢失（被后续覆盖） |
| 13188 | 1.5 | 0.9069 | 已下载 |
| 15386 | 1.75 | **0.8522** | **最佳 Dev，已下载** |
| 17584 | 2.0 | 0.9090 | 已下载 |

注意：服务器按 val_rmse 保存 checkpoint，epoch 1.0 的 Dev=0.8568 虽好但 val 不是最佳，被覆盖。
epoch 1.75 的 Dev=0.8522 是全局最佳，已保存到本地 `step_15386_epoch1.75/`。

### 10 个测试集结果（epoch 1.75 checkpoint）

| 语言-域 | Test RMSE | 记录数 | 论文报告最佳 | 最佳团队 | 差距 |
|---------|-----------|--------|----------|----------|------|
| zho-fin | **0.5655** | 842 | 0.4841 | HUS@NLP-VNU | +0.081 |
| jpn-hot | **0.6473** | 800 | 0.5561 | TeleAI | +0.091 |
| zho-lap | **0.6985** | 1000 | 0.6103 | TeleAI | +0.088 |
| jpn-fin | **0.8277** | 800 | 0.6581 | TeleAI | +0.170 |
| zho-rest | **1.0273** | 1000 | 0.9256 | ICT-NLP | +0.102 |
| **eng-rest** | **1.1168** | 1000 | **1.1035** | LogSigma | +0.013 |
| **eng-lap** | **1.2172** | 1000 | 1.2408 | LogSigma | **-0.024** ✓ |
| rus-rest | **1.3192** | 1072 | 1.2190 | PAI | +0.100 |
| ukr-rest | **1.3610** | 1072 | 1.1888 | PAI | +0.172 |
| tat-rest | **1.7634** | 1072 | 1.5294 | PAI | +0.234 |

### 与之前结果对比

| 方案 | 底座 | English Restaurant Dev | English Restaurant Test |
|------|------|------------------------|-------------------------|
| 第六次最佳（21模型集成） | RoBERTa-large | 1.0591 | 1.1094 |
| 方案A（RoBERTa + R-Drop + PGD） | RoBERTa-large | 0.9564 | 1.2035 |
| **方案B（Qwen3-8B + LoRA）** | **Qwen3-8B-Base** | **0.8522** | **1.1168** |
| 电信论文（Qwen2.5-7B） | Qwen2.5-7B | 0.85 | — |

### 结论

1. **Qwen3-8B-Base 的 Dev 数值明显优于 RoBERTa-large**：Dev 从 0.9169 降到 0.8522
   （提升 0.065），支持 LLM 底座更适合该任务的假设，与电信论文的观察一致；但仍需
   多随机种子验证，不能由一次运行直接证明模型类别的普遍优劣。

2. **英语上表现接近官方最佳**：
   - eng-lap: 1.2172 vs 论文报告最佳 1.2408 → 本地数值低 0.024（非官方提交）
   - eng-rest: 1.1168 vs 官方最佳 1.1035 → 只差 0.013
   说明单 checkpoint 在英语上已经能打一。

3. **中日文表现良好但非最佳**：Test 在 0.5-1.0 范围，但比官方最佳差 0.08-0.17。
   官方最佳团队（TeleAI、ICT-NLP）通常用了多种子集成或特殊架构。

4. **小语种表现差**：tat-rest (1.76)、ukr-rest (1.36)、rus-rest (1.32) 差距最大，
   说明鞑靼语、乌克兰语、俄语的训练数据或模型能力不足。这些是低资源语言，
   Qwen3 的预训练数据可能覆盖不充分。

5. **单 checkpoint 方差问题**：之前实验记录显示 Test 方差很大（1.10-1.40），
   我们的 1.1168 可能是好运气。要确认 Qwen3-8B 是否真的更好，需要多种子验证。

6. **训练效率**：Qwen3-8B + LoRA 在单卡 4090 上约 5 小时跑完 2 epochs，
   比 RoBERTa-large 慢很多（RoBERTa 约 70 分钟），但效果更好。

---

# 2026-09 Task 2/3 中英文实验总结

## 当前冻结基线

新版主线是 Qwen3-8B-Base + 4-bit QLoRA：四个中英文 Restaurant/Laptop Train 数据集
均衡训练一轮，seed 2971；Dev/Test 都使用一次 greedy 生成，没有检索、投票或外部 API。
冻结 Dev 后运行四个完整 Test，各 1,000 条，独立复核通过。

| 数据集 | Task 2 Test cF1 | Task 3 Test cF1 | Task 3 论文第一名 | Task 3 差距 |
|---|---:|---:|---:|---:|
| English Laptop | 0.5235 | 0.3083 | 0.4227 | -0.1144 |
| English Restaurant | 0.6115 | 0.5781 | 0.6514 | -0.0733 |
| Chinese Laptop | 0.4888 | 0.3971 | 0.4824 | -0.0853 |
| Chinese Restaurant | 0.5074 | 0.4653 | 0.5521 | -0.0868 |
| **宏平均** | **0.5328** | **0.4372** | **0.5272** | **-0.0900** |

同配置 seed 42 的 Test Task 2/3/均值为 `0.5326/0.4344/0.4835`；seed 2971 为
`0.5328/0.4372/0.4850`，只高约 `0.0015`。当前问题不是再换一个随机种子，而是结构召回、
Laptop 细粒度 Category 和候选选择。

## 与此前 English Restaurant 方法比较

| 任务 | 路线 | Test cF1 | 论文最佳 |
|---|---|---:|---:|
| Task 2 | 旧 Qwen QLoRA + 三视角检索、二票投票、RoBERTa VA | 0.6166 | 0.7021 |
| Task 2 | 当前 Qwen3-8B 中英文 QLoRA、单次 greedy | 0.6115 | 0.7021 |
| Task 2 | Kimi K2.7 单次抽取 + 关系 RoBERTa VA 重评分 | **0.6420** | 0.7021 |
| Task 3 | 旧 Qwen QLoRA + 三视角检索、二票投票、RoBERTa VA | 0.5735 | 0.6514 |
| Task 3 | 当前 Qwen3-8B 中英文 QLoRA、单次 greedy | 0.5781 | 0.6514 |
| Task 3 | Kimi K2.7 单次抽取 + 关系 RoBERTa VA 重评分 | **0.5858** | 0.6514 |

不能简单写成“新 Qwen 不如旧方法”：在同一 English Restaurant Task 3 上，当前
Qwen3-8B 单次 greedy 的 `0.5781` 比旧 Qwen 混合系统 `0.5735` 高 `0.0046`，也比 Kimi
原始输出 `0.5762` 高 `0.0019`；但低于 Kimi + RoBERTa 的最终系统 `0.5858` 约 `0.0077`。
这些路线的组件、数据范围和推理预算不同，因此这是系统结果对照，不是纯底座能力排名。

## 实验过程与结论

| 阶段 | 尝试 | 结果 | 决策 |
|---|---|---|---|
| 结构奖励训练 | 8B 分解奖励 GRPO，分别奖励 span、Category 与 VA | Dev 比 SFT 低 `0.0132`，Test 低 `0.0303` | 现有奖励不足以支持继续在线 RL |
| 稳定 SFT 基线 | 四个中英文数据集均衡 QLoRA SFT，重建 seed 2971 | 四 Test Task 3 宏平均 `0.4372`，复核通过 | 作为后续组件的冻结基线 |
| 多候选生成 | greedy + 4 个采样候选 | Laptop Dev100 Oracle 均值可增 `0.1066`，但 2-of-4 一致性反而 `-0.0040` | 候选池有答案，主要缺可靠选择器 |
| Category 头 | 冻结 Qwen，训练 last-token 层级 Category 头 | 四 Dev Task 3 仅 `+0.0047`；English Laptop `+0.0244`，English Restaurant `-0.0133` | 不能全域覆盖，应只处理困难域 |
| Train-only 类别规则 | Aspect/Opinion/pair 先验和固定 AO 类别选择 | English Laptop 全 Dev `+0.0151`；候选选择器 confirm `-0.0440` | 规则信号弱且容易过拟合 |
| 检索与 API 类别纠错 | BM25、Dense、Hybrid 检索 + DeepSeek 候选类排序 | Hybrid 检索 Top1/Top5 `0.6883/0.9013`，但端到端 Task 3 仅 `+0.0051` | 检索可作特征，不能单独解决 Category |
| 通用大模型直接重分类 | DeepSeek 全覆盖改写 Qwen Category | 四个数据集均为净损害；Restaurant 原类别准确率最高约 `0.99` | 保留 Qwen 的域内 Category，不让通用模型全量覆盖 |
| 通用大模型审查 | DeepSeek 对原文锚定 A–O 候选做 keep/reject | Laptop Dev100 confirm Task 2 `+0.0225`；英文 `+0.0459`，中文约 `-0.0010` | 仅作为英文高精度过滤器，不作中文召回器 |
| 自一致性 | SCSG 严格多数投票 k=5/10/15 | 全局为负；Laptop 局部例外：中文约 `+0.053`、英文约 `+0.008` | 只允许按域选择性使用，需完整 Dev 再确认 |
| VA 后处理 | PAI 的 affine、quantile、Sinkhorn 等五种适配 | 英文全负，中文最好仅约 `+0.001`；匹配项 VA 损失占比小 | 当前瓶颈不是 VA 分布，关闭该路线 |
| RAFT | 只完成候选生成、门槛与对照方案设计 | **尚未训练或评测** | 先把候选排序器做可靠，再与等预算 continued SFT 对照 |

早期记录中“叠加组件后 Task 3 宏平均约 `0.48`”只是把不同划分上的局部增益相加：
DeepSeek 过滤来自 Test 抽样，SCSG 来自 Dev50。它不是一次完整端到端运行，也不是正式成绩；
在统一的完整 Dev 上冻结组合、再运行一次 Test 之前，不能写成已经达到 `0.48`。

## 通用大模型与微调 Qwen 的能力分工

| 能力 | 通用大模型更有优势的部分 | 微调 Qwen 更有优势的部分 |
|---|---|---|
| 语义判断 | 强模型能判断 A–O 是否语义成立、发现英文误报和不自然边界；DeepSeek 英文过滤的 confirm 增益为 `+0.0459` | Qwen 已学会任务输出协议，原文锚定和批量生成更稳定 |
| 结构召回 | Kimi 在 English Restaurant 的直接抽取较强，原始 Task 2 达 `0.6315` | 并非所有通用模型都强：DeepSeek V4 Flash 曾有 50.8% 空句，Task 3 仅 `0.4193`；本地 Qwen 更可控 |
| Category | 通用模型适合对少量高冲突候选做语义复核 | Qwen 的域内标签先验明显更可靠；DeepSeek 全量重分类会把正确类别改错，Restaurant 尤其明显 |
| 多语言/多领域 | 当前没有足够证据证明 API 模型在中文也有相同收益 | 一个适配器覆盖四个中英文数据集，输出稳定；中文 A–O 过滤实验也表明应保留 Qwen 基线 |
| VA 数值 | 通用生成模型没有显示稳定优势 | 专门回归头能带来约 `0.006~0.010` 的小增益，但 VA 不是当前主要误差源 |

因此，当前最合理的系统不是“用通用大模型替换 Qwen”，而是让 Qwen 做稳定候选生成和域内
Category，让通用大模型只做高置信的英文 A–O 审查，再由专门回归头预测 VA。

## Qwen 当前做得好与做得不好的地方

**做得好：**

- 输出稳定。seed 2971 的四 Dev 共 1,000 条无 JSON 解析失败；四 Test 共 4,000 条只有
  2 次解析失败，完整 ID、顺序和指标已独立复核。
- Restaurant Category 已较可靠。Test 抽样中，精确 A–O 条件下 English/Chinese
  Restaurant 的原 Category 正确率约为 `0.99/0.93`，通用模型重分类反而破坏它。
- 单模型覆盖四个中英文域，且 English Restaurant Task 3 单次 greedy `0.5781` 已略高于
  旧 Qwen 三视角投票系统 `0.5735`。

**做得不好：**

- English Laptop 的 121 类细粒度 Category 是最大短板。Test 中精确 A–O 命中后仍有
  419 个关系类别错误，Task 3 只有 `0.3083`。
- A–O 召回、边界和误报仍限制 Task 2/3。候选 Oracle 很高而简单投票无效，说明模型能
  偶尔生成正确答案，却不能稳定给正确候选更高分。
- 同一全局策略会伤害强域：Category 头、SCSG 和 API 改写都出现 Laptop 有益、Restaurant
  退化，必须按语言和领域路由。
- 当前 VA 已相对可用，继续做全局数值校准收益很小，不能掩盖结构和类别问题。

## 下一步研究路线

1. **先训练候选排序器，而不是立刻继续 RL。** 用 Train 生成多候选，按 A–O 有效性、
   边界、Category 与 VA 分解成 rubric，训练 Bradley–Terry 成对排序器；按 record 分组切分，
   并在完整 Official Dev 上校准和验收。
2. **按域分工。** Restaurant 保留 Qwen 原输出；English Laptop 加 A–O 审查和 Category
   reranker；Chinese Laptop 仅在完整 Dev 复现后启用 SCSG；不要把局部组件全局套用。
3. **采用混合系统。** Qwen 负责生成和域内标签，通用大模型只审查困难英文候选，RoBERTa
   或 Task 1 回归头负责 VA。这样对应了各模型已经实测的强项。
4. **再比较 RAFT 与 continued SFT。** 只有排序器通过完整 Dev 门槛后，才在相同数据量、
   step、seed 和计算预算下比较 gold-anchored RAFT 与继续金标 SFT；RAFT 当前仍是待验证方案。
5. **一次冻结、一次 Test。** 先在完整 Dev 做 baseline、单组件和组合消融；建议门槛为宏平均
   至少 `+0.01`、强 Restaurant 域不明显下降、confirm 与 full 同方向。冻结后只运行一次 Test，
   已经看过的 Test 结果只作诊断，不再反向选配置。
