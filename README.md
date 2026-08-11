# DimABSA 中文餐厅测试结果

本项目使用 `Qwen3-4B-Instruct-2507` 完成 [DimABSA2026](https://github.com/DimABSA/DimABSA2026) Track A 中文餐厅领域的 Task 1、Task 2 和 Task 3 测试。模型不进行 LoRA 微调或参数训练。

## 实验方法

- 使用官方 `Qwen3-4B-Instruct-2507` BF16，通过 Transformers 直接推理，不训练或微调模型。
- Task 1 在 Dev 上比较 Direct、CoT 和 Few-shot CoT；Few-shot 将 5 条 Train 标注样本放入 Prompt，最终选择 Few-shot，并在 Dev 上分别拟合 V/A 线性校准参数。
- Task 2/3 使用 Few-shot CoT，在 Prompt 中加入 8 条 Train 标注样本。模型一次生成 `(Aspect, Category, Opinion, VA)` 四元组，再去掉 Category 得到 Task 2 三元组。
- 程序检查 Aspect 和 Opinion 是否为原文片段、Category 是否合法，并记录格式或解析错误。Task 2/3 还根据 Dev 过滤低置信度抽取，并拟合 V/A 线性校准。
- 所有 Prompt、过滤规则和校准参数均在 Train/Dev 上确定并冻结，Test 只进行一次最终推理和评测。

## Task 1：程度预测

Task 1 根据文本和给定方面预测 Valence（正负程度）和 Arousal（激烈程度）。官方指标是 `RMSE_VA`，越低越好。

Test 包含 1,000 条文本和 1,929 个方面。

| 方法 | RMSE_VA |
|---|---:|
| Train 均值基线 | 1.4761 |
| Few-shot 原始预测 | 2.0312 |
| **Few-shot + Dev 线性校准** | **1.1149** |

## Task 2/3：三元组与四元组抽取

Task 2 从文本中抽取 `(Aspect, Opinion, VA)`；Task 3 进一步抽取 `(Aspect, Category, Opinion, VA)`。官方指标是连续 F1（`continuous F1`），越高越好。

同一次 Task 3 推理自动去掉 Category 生成 Task 2，避免重复运行模型。Test 包含 1,000 条文本和 2,861 条金标准关系。

| 任务 | 方法 | 精确结构 F1 | 官方连续 F1 |
|---|---|---:|---:|
| Task 2 | Few-shot 原始预测 | 0.3055 | 0.2810 |
| Task 2 | **Dev 过滤 + VA 校准** | **0.3075** | **0.2869** |
| Task 3 | Few-shot 原始预测 | 0.2671 | 0.2455 |
| Task 3 | **Dev 过滤 + VA 校准** | **0.2719** | **0.2535** |

Task 2/3 完整 Test 推理 `parse_failures=0`，耗时约 15 分 18 秒，峰值显存约 10.44 GiB。项目评测器与官方评测脚本结果一致。

## 结果说明

Few-shot 示例来自 Train。Prompt、过滤规则和 VA 校准参数在 Dev 上确定后冻结，Test 标准答案没有进入模型 Prompt，也没有用于重新拟合参数。
