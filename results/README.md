# 实验结果目录

模型原始输出、诊断 JSONL 和元数据 JSON 默认保存在这里，但不会提交到个人 GitHub 仓库。

只有经过完整性检查、确认不是 smoke 且明确标注解析失败数的汇总指标，才填写到 `metrics.csv`。

2026-08-11 正式结果：dev 选择的最终方法为 `fewshot_calibrated`；test 1,000 条、1,929 个方面，`parse_failures=0`，官方与严格评测器均得到 `RMSE_VA=1.1149206501`。最终预测是 `zho_restaurant_test_task1_fewshot_calibrated.jsonl`，参数来自 `zho_restaurant_dev_task1_fewshot_calibration.json`。
