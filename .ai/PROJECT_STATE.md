# 项目状态

## 1. 项目简介

- 项目名称：dim absa
- 项目目标：使用同一个 Qwen3-4B-Instruct-2507，在不训练模型的条件下完成中文餐厅 DimASR、DimASTE 与 DimASQP。
- 当前阶段：Task 1、Task 2、Task 3 云端实验、结果回传与 GitHub 发布均已完成。
- 主要使用者：项目所有者与协作 AI。

## 2. 技术栈

- Python 3.12、PyTorch、Transformers、Accelerate、SciPy。
- 云端：SeetaCloud 单卡 RTX 5090 32 GB，官方 Qwen3-4B-Instruct-2507 BF16。
- 原生 Transformers 后端已验证；云端 Unsloth 因 Torch 2.11/Triton 模板冲突未使用。
- 官方评测脚本与项目严格评测器的最终 test RMSE 一致。

## 3. 目录与模块

| 路径 | 作用 | 备注 |
|---|---|---|
| `src/dimabsa_data.py` | 官方 schema 归一化、均值、输出 | 保留方面原文、顺序和重复项 |
| `src/dimabsa_prompts.py` | direct/CoT/few-shot Prompt | 金标准不进入待预测 Prompt |
| `src/run_instruct.py` | Qwen 无训练推理 | 默认 smoke，完整推理有费用门禁 |
| `src/calibrate_task1.py` | VA 尺度校准 | dev 拟合、test 只应用冻结参数 |
| `src/evaluate_task1.py` | 严格 Task 1 评测 | 支持 smoke 子集 |
| `src/run_extraction.py` | Task 2/3 联合 Instruct 推理 | 一次生成 Task 3，同时派生 Task 2 |
| `src/dimabsa_extraction.py` | 抽取数据、解析与官方输出 | 精确原文片段与合法类别检查 |
| `src/calibrate_extraction.py` | Dev 不确定分过滤与 VA 校准 | 参数在 test 前冻结 |
| `src/evaluate_extraction.py` | Task 2/3 严格连续 F1 | 已与官方脚本一致 |
| `resources/DimABSA2026/` | 完整官方仓库快照 | 上游提交 `bdc93be...`，不提交个人仓库 |
| `results/` | 预测、诊断、元数据与汇总 | 原始 JSON/JSONL 被忽略 |

## 4. 启动与验证方式

详见根目录 `README.md`。正式运行使用：

```bash
RUN_MODE=full CONFIRM_FULL_RUN=YES /root/miniconda3/bin/python \
  src/run_instruct.py --prompt-mode fewshot --backend transformers \
  --model-name /root/autodl-tmp/models/Qwen3-4B-Instruct-2507
```

## 5. 当前已实现功能

- [x] 完整官方仓库本地下载与来源记录
- [x] 中文餐厅 Task 1 train/dev/test 解析
- [x] direct、CoT、few-shot CoT Prompt
- [x] Qwen 批量生成、严格 JSON 解析、失败诊断和均值回退
- [x] 官方 JSONL 重建与严格本地评测
- [x] smoke/full 费用保护
- [x] 离线测试与 dev 均值基线
- [x] 云端三种 smoke 与正式 dev，最终全部 0 解析失败
- [x] dev 仿射尺度校准与唯一 few-shot test
- [x] 官方脚本复核及 44 个云端结果文件回传本地
- [x] Task 2/3 四元组联合生成、三元组派生、截断结果恢复
- [x] Task 2/3 dev 过滤与 VA 校准、唯一 test、官方连续 F1 复核

## 6. 正式结果

- dev 最优：few-shot + dev 校准，`RMSE_VA=0.8940462438`。
- test 最终：1,000 条、1,929 个方面，`RMSE_VA=1.1149206501`，`PCC_V=0.7745251911`，`PCC_A=0.4314290922`。
- test 训练均值基线：`RMSE_VA=1.4761066966`。
- 最终 test 推理 `parse_failures=0`、`format_retry_recoveries=0`，峰值 CUDA allocated 10.61 GiB。
- Task 2 test：`exact_F1=0.3074512535`，`continuous_F1=0.2869350972`。
- Task 3 test：`exact_F1=0.2719359331`，`continuous_F1=0.2535017417`。
- Task 2/3 共用一次 1,000 条 test 推理：`parse_failures=0`，耗时 918.38 秒，峰值 CUDA allocated 10.44 GiB。

## 7. 重要约束

- test 金标准不用于选择 Prompt。
- 有解析回退的结果不作为正式模型结果。
- 不再根据公开 test 修改 Prompt 或校准参数。

## 8. 最近更新

- 2026-08-10：Codex 初始化项目、完整资源、推理/评测代码、离线测试、README 与均值基线。
- 2026-08-11：完成云端正式 Task 1；新增 Transformers 后端、稳定输出协议、格式重试和线性校准，最终 test RMSE 1.1149206501。
- 2026-08-11：发布公开仓库 `https://github.com/Chihirodawn/dim-absa`；只包含安全发布范围。
- 2026-08-11：完成中文餐厅 Task 2/3 Instruct 基线、dev 冻结后唯一 test、官方连续 F1 复核及本地结果回传，并以安全范围发布代码、文档与汇总指标到 GitHub。
