# 最近一次 AI 交接

## 1. 基本信息

- 更新时间：2026-08-11
- 上一个 Agent：Codex
- Git：已初始化 `main`，远程公开仓库为 `https://github.com/Chihirodawn/dim-absa`。
- 状态：Task 1、Task 2、Task 3 云端实验、结果回传、文档更新与 GitHub 发布均已完成。

## 2. 本次完成内容

- SSH 检查 SeetaCloud：RTX 5090 32 GB 空闲；Python 3.12.3、Torch 2.11.0+cu130、Transformers 5.5.0、SciPy 可用。
- 现有 Unsloth 2026.8.5 导入触发 Triton `duplicate template name`；未修改基础环境，给推理脚本增加原生 Transformers 后端。
- 通过 `https://hf-mirror.com` 下载官方 `Qwen/Qwen3-4B-Instruct-2507` BF16 到 `/root/autodl-tmp/models/Qwen3-4B-Instruct-2507`。
- direct、CoT、few-shot 的 smoke 和 300 条完整 dev 均完成；通过动态方面编号、精确输出数量和二维 JSON 协议实现正式运行 0 解析失败。
- 新增最多两次格式纠正重试并完整记录；正式 dev/test 均未触发重试。
- 新增 `src/calibrate_task1.py`，在 dev 上分别拟合 V/A 的斜率和截距，应用时裁剪到 `[1,9]`。
- dev 选择 `few-shot + calibration`，之后只运行一次 1,000 条 test；官方和严格评测器结果一致。
- 云端 `results/` 的 44 个文件已同步回本地，共约 1.3 MB。

## 3. 正式指标

### Dev（300 条、685 个方面）

| 方法 | 原始 RMSE | 校准 RMSE |
|---|---:|---:|
| direct | 1.9864848828 | 0.9546792177 |
| CoT | 1.9353279961 | 0.9611840326 |
| few-shot | 1.9835888363 | 0.8940462438 |
| train mean | 1.2879411511 | — |

### Test（1,000 条、1,929 个方面）

- train mean：`RMSE_VA=1.4761066966`。
- few-shot raw：`RMSE_VA=2.0311753244`。
- few-shot calibrated：`RMSE_VA=1.1149206501`，`PCC_V=0.7745251911`，`PCC_A=0.4314290922`。
- 最终推理：`parse_failures=0`、`format_retry_recoveries=0`、峰值 CUDA allocated 10.61 GiB、耗时 164.75 秒。

## 4. 关键文件

| 文件 | 作用 |
|---|---|
| `src/run_instruct.py` | Transformers/Unsloth 可选后端、批量推理、费用门禁、格式重试 |
| `src/dimabsa_prompts.py` | 编号方面、direct/CoT/few-shot 与严格输出协议 |
| `src/calibrate_task1.py` | dev 拟合和 test 应用仿射校准 |
| `results/metrics.csv` | 可提交 GitHub 的完整汇总 |
| `results/zho_restaurant_test_task1_fewshot_calibrated.jsonl` | 最终官方格式预测，本地保留但被 Git 忽略 |
| `results/zho_restaurant_dev_task1_fewshot_calibration.json` | 冻结校准参数，本地保留但被 Git 忽略 |

## 5. 验证

- 本地：`py_compile` 通过，unittest 7/7，Ruff 通过。
- 云端：模型实际加载、三种 full dev、唯一 few-shot test 均成功，无 OOM/traceback。
- 最终 test 严格评测器：`RMSE_VA=1.1149206500857176`。
- 最终 test 官方 SciPy 脚本：`RMSE_VA=1.1149206500857172`。

## 6. 后续注意事项

- 不要再用 test 调 Prompt、few-shot 示例或校准参数。
- 本项目没有执行 Qwen 微调；“训练”仅指 dev 上 4 个标量的线性校准拟合。
- 若发布 GitHub，遵守 README 的发布边界，只提交代码、文档和 `results/metrics.csv`，不提交官方数据、模型权重、原始预测/诊断。
- 最终检查无 `python`/`python3`/`hf` 进程，GPU 2 MiB、利用率 0%、温度 30°C；仍需在 SeetaCloud 控制台停止或释放实例以停止计费。
- GitHub 已仅发布代码、测试、文档和 `results/metrics.csv`；官方数据、模型权重、原始预测及诊断仍在 `.gitignore` 中。

## 7. 后续更新（2026-08-11）：Task 2/3 联合抽取

- 新增 `dimabsa_extraction.py`、`dimabsa_extraction_prompts.py`、`run_extraction.py`、`evaluate_extraction.py`、`calibrate_extraction.py`、`repair_extraction.py`。
- Qwen 一次生成 Task 3 `(Aspect, Category, Opinion, VA)`，程序自动去 Category 派生 Task 2，未重复运行 test。
- dev 使用官方完整餐厅类别协议；对截断但含完整 item 的输出可安全恢复。最终 dev Task 2/3 continuous F1 为 `0.3719793303` / `0.3120061572`。
- 冻结 dev 过滤和 V/A 参数后只运行一次 1,000 条 test：Task 2 `continuous_F1=0.2869350972`，Task 3 `continuous_F1=0.2535017417`；官方脚本一致。
- test 生成 3,549 个原始四元组，过滤校准后 2,883 个；金标准 2,861 个。`parse_failures=0`、`discarded_invalid_items=53`、耗时 918.38 秒、峰值显存 10.44 GiB。
- 云端最终 GPU 2 MiB、利用率 0%；结果已同步到 `results/extraction_schema_v2/` 与 `results/extraction_final/`，原始 JSON/JSONL 被 Git 忽略。
- 本地验证：9 项 unittest 通过，`py_compile` 通过，Ruff 通过。Task 2/3 代码、README、测试与汇总指标已按安全范围发布；原始预测和官方数据未上传。
- README 已补充简短实验方法：说明 Task 1 的 Direct/CoT/Few-shot 对照、Train 示例、Dev 校准，以及 Task 2/3 的联合生成、过滤与校准流程；未加入冗长复现命令。

## 8. 后续更新（2026-08-12）：英文 Restaurant Task 1

- 新主机：SeetaCloud RTX 5090 D 32 GB；复用官方 Qwen3-4B-Instruct-2507 BF16 与原有 Transformers 推理代码，没有训练或微调模型。
- 云端 `/etc/network_turbo` 使权重断点下载由亚 MB/s 提升至约 51–65 MiB/s；三块权重 SHA-256 与旧主机完整副本完全一致。
- 英文 dev（200 条、340 个方面）校准 RMSE：Direct `1.1717095354`、CoT `1.1722364247`、Few-shot `1.0998732547`；据此冻结 Few-shot 及其 dev 校准参数。
- 英文 test（1,000 条、1,504 个方面）只运行一次：raw `2.3629920685`，calibrated `1.4511021582`，`PCC_V=0.8070043842`，`PCC_A=0.3900397818`。
- 官方 SciPy 脚本与项目严格评测器一致；相对 LogSigma 公布的 `1.1035` 高 `0.3476` RMSE，因此未赢。
- 推理耗时 119.64 秒，峰值 CUDA allocated 10.35 GiB，`parse_failures=0`、`format_retry_recoveries=0`；结果保存在 `results/english/` 并被 Git 忽略，汇总已写入 `results/metrics.csv`。
- 本次英文结果和交接尚未提交或推送 GitHub；发布前仍需遵守只上传代码、文档和汇总指标的边界。

## 9. 后续更新（2026-08-12）：同模型无训练动态检索与集成

- 新增 `select_similar_examples`：用文本和方面词构造无答案 BM25 风格词法特征，为每条英文输入从 Train 检索 5 条短示例；新增 `dynamic_fewshot` 英文专用提示。
- 动态单路校准 Dev `RMSE_VA=1.1079851878`，没有直接替代固定 Few-shot；其差异用于四路集成。
- 冻结方法前按记录 ID 做 5 折分组交叉验证：固定 Few-shot `1.1133`，四路等权 `1.0415`；复杂自由权重未采用。
- 新增 `src/ensemble_task1.py`，对 Direct/CoT/固定 Few-shot/动态 Few-shot 原始输出各取 25%，再应用 Dev 冻结的 V/A 仿射校准。
- 完整 Dev `RMSE_VA=1.0262814085`；Test `RMSE_VA=1.3661908335`，`PCC_V=0.8448344992`、`PCC_A=0.5057372064`；V/A RMSE `1.1296`/`0.7685`。
- 官方与严格脚本一致；补跑 Test 的 Direct/CoT/动态 Few-shot 均 `parse_failures=0`、`format_retry_recoveries=0`，GPU 最终 2 MiB、0%。
- 本地通过 11 项 unittest、py_compile、Ruff；原始结果位于 `results/english_dynamic/` 和 `results/english_ensemble_test/` 且被 Git 忽略，汇总写入 `results/metrics.csv`。
- 本次代码、交接和汇总尚未提交或推送 GitHub。

## 10. 后续更新（2026-08-12）：英文 Restaurant 三任务 LoRA

- 新增 `train_task1_lora_regression.py`：Qwen LoRA 特征主干 + V/A 双输出回归头；正式训练在 epoch 1 达到最佳 Dev，随后早停，完整训练 335.14 秒、峰值 CUDA allocated 8.20 GiB。
- Task 1 最佳检查点 Test raw `1.2859560`，Dev 仿射后 `1.2578152`；按 Dev 冻结的 90% LoRA + 10% 无训练集成最终 `RMSE_VA=1.2420729499`，官方脚本一致。
- 新增 `train_extraction_lora.py`：Task 2/3 共用 4-bit QLoRA。batch 20、梯度累积 1，epoch 2 最优；完整训练 1630.98 秒、平均 GPU 利用率 93.03%、峰值 CUDA allocated 25.10 GiB。
- 新增 `calibrate_extraction_affine.py`：英文抽取只做 VA 仿射校准，保留全部预测项。最终 Test Task 2/3 `continuous_F1=0.5396759014` / `0.4962316066`，官方脚本一致，1000 条生成 `parse_failures=0`。
- 云端最终无训练进程，GPU 0%、2 MiB。两个约 46 MB 的 LoRA 适配器、预测、指标、Dev epoch 输出及 GPU 日志已同步回本地 `outputs/`、`results/`、`logs/`；这些大部分按 `.gitignore` 不发布。
- 本地验证：13 项 unittest、py_compile 与 Ruff 已通过；正式 GPU smoke、训练、检查点重载和 Test 推理全部通过。当前修改尚未 commit 或 push。

## 11. 后续更新（2026-08-12）：论文数据入库

- 使用用户提供的 `2604.07066v1` 任务总览论文，依据第 4 页 Table 1、第 7 页 Table 2 和第 20–21 页完整排行榜核对数据。
- 新增 `PAPER_RESULTS.md`：解释数据集缩写、三个任务、RMSE/cF1、英文 Restaurant 对比、LogSigma/Takoyaki 方法及官方基线含义。
- 新增 `results/paper_track_a_table2.csv`（104 条论文成绩）、`paper_eng_rest_comparison.csv`（15 条论文/本项目对比）和 `paper_results.xlsx`（说明、原始表、对比表）。
- XLSX 三个工作表均完成渲染检查；CSV 已检查行数、任务分布、唯一性和关键数字。当前修改尚未 commit 或 push。

## 12. 后续更新（2026-08-12）：收缩为论文评测指标

- 用户澄清老师需要的是论文评测相关数据，不是完整排行榜。
- `PAPER_RESULTS.md` 现只含 RMSE、VA 距离、cTP、cPrecision、cRecall、cF1、论文附录 C 示例和本项目对应结果。
- 新增 `results/evaluation_metrics.csv`；删除此前的完整 Track A 表、团队方法、数据集介绍和 XLSX 排行榜。
