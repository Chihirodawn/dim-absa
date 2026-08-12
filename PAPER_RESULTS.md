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

| 任务 | 论文最终指标 | 本项目结果 | 论文同数据集最佳结果 |
|---|---|---:|---:|
| Task 1 | `RMSE_VA` ↓ | **1.242073** | 1.1035 |
| Task 2 | `cF1` ↑ | **0.539676** | 0.7021 |
| Task 3 | `cF1` ↑ | **0.496232** | 0.6514 |

Task 2 的评测明细：

| 指标 | 数值 |
|---|---:|
| 金标准关系数 | 2,129 |
| 预测关系数 | 1,812 |
| 结构 TP | 1,138 |
| FP | 674 |
| FN | 991 |
| `ΣcTP` | 1,063.431364 |
| `cPrecision` | 0.586883 |
| `cRecall` | 0.499498 |
| `cF1` | **0.539676** |

Task 3 的评测明细：

| 指标 | 数值 |
|---|---:|
| 金标准关系数 | 2,129 |
| 预测关系数 | 1,857 |
| 结构 TP | 1,058 |
| FP | 799 |
| FN | 1,071 |
| `ΣcTP` | 988.989592 |
| `cPrecision` | 0.532574 |
| `cRecall` | 0.464532 |
| `cF1` | **0.496232** |

机器可读数据见 [`results/evaluation_metrics.csv`](results/evaluation_metrics.csv)。本项目结果是赛后本地实验，不是 Codabench 官方提交成绩。
