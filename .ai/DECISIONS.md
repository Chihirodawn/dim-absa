# 技术决策记录

## DECISION-001：第一阶段只使用同一个 Instruct 模型

- 日期：2026-08-10
- 状态：已采用
- 背景：用户希望先验证无需训练的大模型判断能力。
- 最终选择：固定 Qwen3-4B-Instruct-2507，对比 direct、CoT 和 few-shot CoT，不创建 LoRA。
- 选择原因：控制模型变量，先验证 Prompt 与上下文示例的实际作用，避免过早引入训练成本。
- 影响范围：当前 Track A / Task 1 / 中文餐厅实验。

## DECISION-002：模型只生成数值，程序重建官方结构

- 日期：2026-08-10
- 状态：已采用
- 背景：官方要求 ID、方面原文、顺序及重复项准确，生成完整 JSON 容易改写文本。
- 最终选择：模型输出 `{"scores":[["V","A"],...]}`；代码使用原始 ID/方面词生成 `Aspect_VA`。
- 选择原因：将模型任务集中到 VA 判断，并降低格式错误。
- 后续注意事项：方面词显式编号，Prompt 写出精确数量；任何解析失败先做最多两次纯格式纠正并写入 diagnostics，仍失败才用训练均值补齐；有回退的结果不得报告。

## DECISION-003：公开 test 只做最终一次评测

- 日期：2026-08-10
- 状态：已采用
- 背景：比赛结束后 test 金标准已公开，容易发生无意的数据泄漏。
- 最终选择：所有 Prompt、few-shot 数量和生成参数只在 train/dev 确定，test 不用于调参。
- 选择原因：保留离线实验的可信度。

## DECISION-004：完整 GPU 推理继续使用费用门禁

- 日期：2026-08-10
- 状态：已采用
- 最终选择：默认 smoke；Qwen full 必须同时设置 `RUN_MODE=full CONFIRM_FULL_RUN=YES`。
- 选择原因：避免上传云端后误启动完整付费推理。

## DECISION-005：使用官方 BF16 与原生 Transformers 后端

- 日期：2026-08-11
- 状态：已采用
- 背景：新云端没有模型；已有 Unsloth 2026.8.5 与 Torch 2.11/Triton 组合导入失败。
- 最终选择：不修改共享基础环境，下载官方 `Qwen/Qwen3-4B-Instruct-2507` BF16，并给脚本增加 `--backend transformers`。
- 选择原因：RTX 5090 32 GB 足以容纳 4B BF16；官方明确支持 Transformers，实测峰值显存仅 10.63 GiB。

## DECISION-006：用 dev 仿射校准解决 VA 刻度偏差

- 日期：2026-08-11
- 状态：已采用
- 背景：raw LLM 对变化方向有相关性，但过度使用 3/5/7/9，RMSE 甚至弱于训练均值基线，尤其高估 Arousal。
- 最终选择：在 dev 上为 V/A 分别拟合 `gold = slope * pred + intercept`，裁剪到 `[1,9]`；比较三种校准 dev RMSE 后锁定 few-shot，再仅应用冻结参数到 test。
- 选择原因：官方 Task 1 以 RMSE 为主指标；线性校准只修正模型尺度，不使用 test 标签，不训练 Qwen。
- 验证依据：few-shot dev RMSE 从 1.9836 降至 0.8940；test 从 2.0312 降至 1.1149，并优于 test 均值基线 1.4761。
