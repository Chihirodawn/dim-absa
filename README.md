# DimABSA 2026 实验项目

本项目使用 `Qwen3-4B-Instruct-2507` 和英语情感 RoBERTa 完成
[DimABSA 2026](https://github.com/DimABSA/DimABSA2026) Track A 的三个子任务。实验从
不训练的 Instruct 基线开始，随后加入动态检索、LoRA/QLoRA、模型集成和关系级重评分。
本文只展示已经运行完整 Test 且相对前一阶段取得提升的实验。

## 任务与指标

| 任务 | 输出 | 官方指标 |
|---|---|---|
| Task 1 / DimASR | 给定 Aspect 的 Valence、Arousal | `RMSE_VA`，越低越好 |
| Task 2 / DimASTE | `(Aspect, Opinion, VA)` | `cF1`，越高越好 |
| Task 3 / DimASQP | `(Aspect, Category, Opinion, VA)` | `cF1`，越高越好 |

Task 2/3 只有结构字段完全匹配时才计算连续真阳性；VA 距离越小，对 `cTP` 的贡献越高。

## 方法概览

### 1. 无训练基线

- 使用 Qwen3-4B 比较 Direct、CoT、固定 Few-shot 和动态 Few-shot。
- 动态 Few-shot 只从 Train 检索相似示例；Dev 用于选择 Prompt、集成权重及 V/A
  仿射或 Ridge 校准，Test 不参与调参。
- 英文 Task 1 从固定 Few-shot Test `RMSE_VA=1.4511` 提升到四路无训练集成
  `1.3662`，说明检索和校准有效，但仍不能充分学习英文 VA 标尺。

### 2. Qwen LoRA/QLoRA

- Task 1 在 Qwen 隐藏表示后增加独立 V/A 回归头，使用 Dev 早停；LoRA 与无训练结果
  按 Dev 冻结的 `90%/10%` 集成，Test `RMSE_VA=1.2421`。
- Task 2/3 使用 4-bit QLoRA 联合生成四元组，并由程序派生三元组。初始 Test cF1
  分别为 `0.5397/0.4962`。

### 3. Task 1 英语编码器增强

- 使用 `twitter-roberta-large-topic-sentiment-latest` 的 Text/Aspect 文本对和英语分词器。
- 只解冻后 12/24 层，使用独立 V/A 头、LogSigma 动态损失、Opinion token 辅助监督和
  VA 均衡采样。
- 训练 seed `21/99/42` 三个模型并等权平均；Train 内使用按记录分组的三折 OOF Ridge
  校准，随后在 Dev 冻结 RoBERTa/Qwen 集成权重。

严格按照 Dev 预选得到的 `30% RoBERTa + 70% Qwen` 在 Test 上达到
`RMSE_VA=1.1883`。纯 RoBERTa 三种子平均的 Test 为 `1.1659`，但这是评测 Test 后才发现
更好，因此只作为诊断结果，不反向替换预先冻结的正式方案。

### 4. Task 2/3 三视角抽取与关系重评分

- 复用 Qwen QLoRA 抽取器，分别使用 word、bigram、trigram BM25 从英文 Train 动态
  检索 3 条示例。
- 对三路 Aspect/Opinion/Category 结构进行至少二票的精确多数投票，减少单路生成噪声。
- 投票后的关系由英语 RoBERTa 单独预测 V/A，不再使用 Qwen 自由生成的分数。

该方案在 Dev 上达到 `0.7412/0.7175`，冻结后 Test cF1 达到
`0.6166/0.5735`；三路共 3,000 次 Test 生成均为 `parse_failures=0`。

## 英文 Restaurant Test 逐步提升结果

Task 1 的 `RMSE_VA` 越低越好：

| 阶段 | 方法 | Test RMSE_VA |
|---|---|---:|
| 1 | 固定 Few-shot + Dev 校准 | 1.4511 |
| 2 | Direct/CoT/固定与动态 Few-shot 四路集成 | 1.3662 |
| 3 | Qwen LoRA 双头回归 + 无训练集成 | 1.2421 |
| 4 | Qwen3 动态 5-shot + LogSigma | 1.2358 |
| 5 | **Dev 冻结 30% RoBERTa + 70% Qwen** | **1.1883** |
| 诊断 | 纯 RoBERTa 三种子平均 | 1.1659 |

`1.1659` 是已经运行 Test 后发现的最佳观察值，因此保留为诊断结果；没有利用该结果
反向调整 Dev 选择。论文英文 Restaurant Task 1 最佳结果为 `1.1035`。

Task 2/3 的 `cF1` 越高越好：

| 任务 | 阶段 | 方法 | Test cF1 | 论文最佳 |
|---|---|---|---:|---:|
| Task 2 | 1 | Qwen联合QLoRA | 0.5397 | 0.7021 |
| Task 2 | 2 | **三路检索 + 二票投票 + RoBERTa VA** | **0.6166** | 0.7021 |
| Task 3 | 1 | Qwen联合QLoRA | 0.4962 | 0.6514 |
| Task 3 | 2 | **三路检索 + 二票投票 + RoBERTa VA** | **0.5735** | 0.6514 |

项目严格评测器与官方 `metrics_subtask_1_2_3.py` 的最终结果一致。本项目为比赛结束后的
本地实验，不代表 Codabench 官方提交名次。

## 主要代码

| 文件 | 作用 |
|---|---|
| `src/train_task1_gemma_regression.py` | Qwen Task 1 LoRA回归、独立V/A头与动态Few-shot |
| `src/train_task1_logs_sigma_encoder.py` | 英语RoBERTa、独立V/A头与LogSigma训练/推理 |
| `src/crossfit_calibrate_task1.py` | 按记录分组的OOF Ridge校准 |
| `src/combine_task1_predictions.py` | 多种子预测平均与组合 |
| `src/weighted_ensemble_task1.py` | Dev选择RoBERTa/Qwen集成权重 |
| `src/train_extraction_lora.py` | Task 2/3 QLoRA训练、动态检索与生成 |
| `src/extraction_hybrid.py` | 三路投票、关系数据转换和VA重评分 |
| `src/evaluate_task1.py`、`src/evaluate_extraction.py` | 严格本地评测 |

论文使用的评测指标与公式见 [PAPER_RESULTS.md](PAPER_RESULTS.md)。

## 实验边界

- Few-shot 和 BM25 示例只来自 Train。
- 模型、阈值、集成和校准只根据 Train/Dev 选择。
- Test 不进入训练、Prompt 检索或校准拟合。
- 读取 Test 后得到的诊断结果不会反向用于替换预先冻结方案。
- 数据集、模型权重、适配器和原始预测不提交到本仓库。
