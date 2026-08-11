# 最近一次 AI 交接

## 1. 基本信息

- 更新时间：2026-08-11
- 上一个 Agent：Codex
- Git：已初始化 `main`，远程公开仓库为 `https://github.com/Chihirodawn/dim-absa`。
- 状态：Task 1 云端实验、最终 test、结果回传、文档更新与 GitHub 首次发布均已完成。

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
