# DimABSA Task 1 测试结果

本项目使用 `Qwen3-4B-Instruct-2507` 完成 [DimABSA2026](https://github.com/DimABSA/DimABSA2026) Track A / Task 1 的中文餐厅领域测试。

任务要求根据文本和指定方面，分别预测：

- Valence：正负程度，范围为 1.00～9.00；
- Arousal：情绪激烈程度，范围为 1.00～9.00。

官方评判指标是 `RMSE_VA`，表示预测分数与标准答案之间的总体误差，数值越低越好。

## 最终方法

- 使用同一个 Qwen3-4B-Instruct 模型，不进行 LoRA 微调或参数训练；
- Few-shot Prompt 中加入 5 条 Train 训练集示例；
- 在 Dev 验证集上拟合 Valence 和 Arousal 的线性尺度校准；
- 在确定方法和校准参数后，只运行一次 Test 模型推理；
- Test 原始预测生成后，应用提前冻结的 Dev 校准参数。

## Test 结果

Test 数据包含 1,000 条文本和 1,929 个待预测方面。

| 方法 | Test RMSE |
|---|---:|
| Train 均值基线 | 1.4761 |
| Few-shot 原始预测 | 2.0312 |
| **Few-shot + Dev 线性校准** | **1.1149** |

最终结果由项目评测器和官方评测脚本共同复核：

```text
RMSE_VA = 1.1149206501
```

最终方法相较 Train 均值基线降低约 24.5% RMSE，相较未校准的 Few-shot 原始预测降低约 45.1% RMSE。完整 Test 推理没有出现格式错误、解析失败或均值回退。

## 数据使用说明

- Train：选择 5 条 Few-shot 示例，并计算均值基线；
- Dev：比较方法并拟合线性校准参数；
- Test：在方法与参数冻结后进行最终预测和评测。

Test 标准答案没有进入模型 Prompt，也没有用于修改 Prompt 或重新拟合校准参数。
