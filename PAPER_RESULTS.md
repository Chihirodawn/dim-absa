# 论文评测指标与本项目结果

来源：*SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DimABSA)*，第 5.1 节与附录 C。

## 1. 论文使用的评测指标

| 任务 | 最终指标 | 方向 |
|---|---|---|
| Task 1 / DimASR | `RMSE_VA` | 越低越好 |
| Task 2 / DimASTE | `cF1` | 越高越好 |
| Task 3 / DimASQP | `cF1` | 越高越好 |

### Task 1：RMSE_VA

```text
RMSE_VA = sqrt((1/N) × Σ[(V_pred - V_gold)² + (A_pred - A_gold)²])
```

- `N`：方面实例总数。
- `V_pred`、`A_pred`：预测的 Valence、Arousal。
- `V_gold`、`A_gold`：金标准 Valence、Arousal。

### Task 2/3：cF1

只有结构字段完全匹配时，预测才进入连续真阳性计算：

- Task 2：`Aspect`、`Opinion` 完全匹配；
- Task 3：`Aspect`、`Category`、`Opinion` 完全匹配。

```text
D_max = sqrt(8² + 8²) = sqrt(128)

distance = sqrt((V_pred - V_gold)² + (A_pred - A_gold)²) / sqrt(128)

cTP = 1 - distance                结构完全匹配
cTP = 0                           结构不匹配

cPrecision = ΣcTP / 预测关系数
cRecall    = ΣcTP / 金标准关系数
cF1        = 2 × cPrecision × cRecall / (cPrecision + cRecall)
```

## 2. 论文附录 C 的计算示例

| 数据 | 论文示例值 |
|---|---:|
| 预测关系数 | 4 |
| 金标准关系数 | 3 |
| `ΣcTP` | 1.375 |
| `cRecall` | 0.458 |
| `cPrecision` | 0.344 |
| `cF1` | 0.393 |

## 3. 本项目英文 Restaurant Test 结果

| 任务 | 论文最终指标 | 严格 Dev 冻结结果 | 最新 Test 观察值 | 论文同数据集最佳结果 |
|---|---|---:|---:|---:|
| Task 1 | `RMSE_VA` ↓ | **1.1427** | 1.1094（mean 21 模型，诊断） | 1.1035 |
| Task 2 | `cF1` ↑ | **0.616578** | 0.6420（Kimi + 关系 RoBERTa） | 0.7021 |
| Task 3 | `cF1` ↑ | **0.573456** | 0.5858（Kimi + 关系 RoBERTa） | 0.6514 |

Task 1 的 1.1094 是记录了 21 个随机种子后可复现的 Test 诊断结果，但它并非由 Dev
预先选出的方案：其 Dev 为 1.0591，而 CLS 五种子方案的 Dev 为 0.9603。因此严格
Dev 冻结结论仍是 1.1427。Task 2/3 的 0.6420/0.5858 是赛后本地 Test 运行结果，
并非 Codabench 官方提交名次。

## 4. 论文英文 Restaurant 前两名与官方 Baseline

| 任务 | 指标 | 论文第一名 | 论文第二名 | 官方 Baseline KimiK2 | 官方 Baseline Qwen3-14B |
|---|---|---:|---:|---:|---:|
| Task 1 | `RMSE_VA` ↓ | 1.1035 | 1.1812 | 2.1461 | 2.6427 |
| Task 2 | `cF1` ↑ | 0.7021 | 0.6985 | 0.4920 | 0.4483 |
| Task 3 | `cF1` ↑ | 0.6514 | 0.6403 | 0.3746 | 0.2673 |

Task 2 的评测明细：

| 指标 | 数值 |
|---|---:|
| 金标准关系数 | 2,129 |
| 预测关系数 | 1,962 |
| 结构 TP | 1,351 |
| FP | 611 |
| FN | 778 |
| `ΣcTP` | 1,261.209488 |
| `cPrecision` | 0.642818 |
| `cRecall` | 0.592395 |
| `cF1` | **0.616578** |

Task 3 的评测明细：

| 指标 | 数值 |
|---|---:|
| 金标准关系数 | 2,129 |
| 预测关系数 | 1,977 |
| 结构 TP | 1,261 |
| FP | 716 |
| FN | 868 |
| `ΣcTP` | 1,177.305764 |
| `cPrecision` | 0.595501 |
| `cRecall` | 0.552985 |
| `cF1` | **0.573456** |

机器可读数据见 [`results/evaluation_metrics.csv`](results/evaluation_metrics.csv)。本项目结果是赛后本地实验，不是 Codabench 官方提交成绩。
