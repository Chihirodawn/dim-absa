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

## DECISION-007：Task 3 一次生成并派生 Task 2

- 日期：2026-08-11
- 状态：已采用
- 背景：中文餐厅 Task 2 与 Task 3 的输入 ID、文本和关系数量一致，Task 3 只比 Task 2 多 Category。
- 最终选择：模型只运行一次 Task 3；程序去掉 Category 并按 Aspect/Opinion 去重后生成 Task 2。
- 原因：避免对相同 1,000 条 test 重复付费推理，并确保两个任务使用相同的方面与评价边界。

## DECISION-008：抽取任务采用官方完整类别协议与严格原文片段

- 日期：2026-08-11
- 状态：已采用
- 最终选择：Category 按官方 Restaurant entity/attribute 组合校验；Aspect 与 Opinion 必须是原文连续片段；非法单项丢弃并记录诊断。
- 原因：Task 2/3 的结构必须精确匹配；训练集中未出现的组合仍可能是官方合法类别。

## DECISION-009：Dev 冻结抽取过滤与 VA 校准

- 日期：2026-08-11
- 状态：已采用
- 最终选择：在 dev 丢弃 Qwen 常用于低置信事实片段的 4 个中心网格分，并对 Task 2 精确匹配关系拟合 V/A 仿射参数；test 只应用冻结规则。
- 验证依据：最终 dev Task 2/3 continuous F1 分别为 0.3720/0.3120；唯一 test 分别为 0.2869/0.2535。
