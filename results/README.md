# 实验结果目录

模型原始输出、诊断 JSONL 和元数据 JSON 默认保存在这里，但不会提交到个人 GitHub 仓库。

只有经过完整性检查、确认不是 smoke 且明确标注解析失败数的汇总指标，才填写到 `metrics.csv`。

2026-08-11 正式结果：dev 选择的最终方法为 `fewshot_calibrated`；test 1,000 条、1,929 个方面，`parse_failures=0`，官方与严格评测器均得到 `RMSE_VA=1.1149206501`。最终预测是 `zho_restaurant_test_task1_fewshot_calibrated.jsonl`，参数来自 `zho_restaurant_dev_task1_fewshot_calibration.json`。

同日完成中文餐厅 Task 2/3：使用一次 Task 3 few-shot 推理同时生成四元组和派生三元组，dev 冻结不确定分过滤及 V/A 仿射校准后，test Task 2 `continuous_F1=0.2869350972`，Task 3 `continuous_F1=0.2535017417`。1,000 条 test `parse_failures=0`，官方脚本与严格评测器一致；详细汇总见 `extraction_metrics.csv`。
