# 任务清单

## 进行中

- 暂无。

## 待处理

- 如需向老师提交，再根据 README 与 `results/metrics.csv` 整理方法报告。

## 已完成

### TASK-001：DimABSA Instruct 无训练项目初始化

- 状态：已完成
- 完成时间：2026-08-10
- 完成内容：完整官方资源、数据/Prompt/推理/评测代码、5 项离线测试、云端说明与均值基线。
- 验证结果：`py_compile`、`unittest`、Ruff 全部通过；300 条 dev CPU 均值基线 `RMSE_VA=1.2879411511`。

### TASK-002：云端 Instruct Prompt 对照与 Task 1 最终结果

- 状态：已完成
- 完成时间：2026-08-11
- 完成内容：
  - [x] 检查 RTX 5090、Python、CUDA、依赖、磁盘与模型状态
  - [x] 使用镜像下载官方 Qwen3-4B-Instruct-2507 BF16，原生 Transformers 加载成功
  - [x] direct、CoT、few-shot smoke 与完整 dev 全部完成
  - [x] 修正动态方面数量协议，正式运行 `parse_failures=0`
  - [x] dev 拟合 V/A 仿射尺度校准，few-shot calibrated `RMSE_VA=0.8940462438`
  - [x] dev 选定后只运行一次 few-shot test，最终校准 `RMSE_VA=1.1149206501`
  - [x] 官方评测脚本与严格评测器一致；44 个结果文件已回传本地
- 资源结果：test 1,000 条、1,929 个方面，峰值 CUDA allocated 10.61 GiB，无 OOM、无回退、无格式重试。

### TASK-003：GitHub 安全发布

- 状态：已完成
- 完成时间：2026-08-11
- 仓库：`https://github.com/Chihirodawn/dim-absa`
- 发布范围：代码、测试、README、依赖、复现脚本、AI 交接与 `results/metrics.csv`。
- 排除范围：完整官方仓库/数据、模型权重、原始预测、诊断和元数据 JSON/JSONL。

### TASK-004：中文餐厅 Task 2/3 Instruct 联合抽取

- 状态：已完成
- 完成时间：2026-08-11
- 完成内容：
  - [x] 一次 Task 3 四元组生成自动派生 Task 2 三元组，避免重复 GPU 推理
  - [x] 精确原文片段、官方类别、V/A 范围、去重与截断输出恢复
  - [x] 8 条 Train few-shot CoT，300 条完整 dev 与 1,000 条唯一 test
  - [x] Dev 冻结 4 个不确定中心分过滤及 V/A 仿射校准
  - [x] 严格评测器与官方连续 F1 脚本一致
  - [x] 9 项 unittest、py_compile、Ruff 全部通过
  - [x] 代码、README、测试与汇总指标按安全范围发布 GitHub
  - [x] README 补充 Direct、CoT、Few-shot、Dev 校准及 Task 2/3 联合推理方法
- 最终结果：Task 2 test `continuous_F1=0.2869350972`；Task 3 test `continuous_F1=0.2535017417`；`parse_failures=0`。

### TASK-005：英文餐厅 Task 1 原方案复现与 LogSigma 对比

- 状态：已完成
- 完成时间：2026-08-12
- 完成内容：
  - [x] 在新 RTX 5090 D 主机复用同一 Qwen3-4B-Instruct-2507 与既有代码
  - [x] 英文 dev 对比 Direct、CoT、Few-shot，并分别仅用 dev 拟合尺度校准
  - [x] dev 选择 Few-shot + calibration 后只运行一次 1,000 条 test
  - [x] 项目严格评测器与官方 SciPy 脚本结果一致
  - [x] 模型三分片 SHA-256 与旧主机完整副本一致；原始结果回传本地
- 最终结果：英文 Restaurant test `RMSE_VA=1.4511021582`；未超过 LogSigma 公布的 `1.1035`。

### TASK-006：英文 Task 1 无训练同模型动态检索与集成

- 状态：已完成
- 完成时间：2026-08-12
- 完成内容：
  - [x] 增加 `dynamic_fewshot`：每条输入仅按文本/方面词从 Train 检索 5 条相似标注示例
  - [x] 增加英文专用 V/A 评分提示，不读取 Dev/Test 标签进行检索
  - [x] 按整条记录分组做 5 折交叉验证，冻结四路等权平均与 V/A 仿射校准
  - [x] 新增独立集成脚本并通过 11 项 unittest、py_compile 与 Ruff
  - [x] 唯一新方案 Test 由 Direct、CoT、固定 Few-shot、动态 Few-shot 各 25% 组成
  - [x] 严格评测器与官方 SciPy 脚本一致；所有模型推理均 0 解析失败
- 结果：Dev 交叉验证从固定 Few-shot `1.1133` 降至集成 `1.0415`；完整 Dev `1.0263`；Test 从 `1.4511` 降至 `1.3662`，但仍未超过 LogSigma `1.1035`。

### TASK-007：英文 Restaurant 三任务 LoRA 微调

- 状态：已完成
- 完成时间：2026-08-12
- 完成内容：
  - [x] Task 1 Qwen LoRA + 双输出回归头，Dev 早停并冻结最优检查点
  - [x] Task 1 将校准 LoRA 与无训练集成按 Dev 选择的 90% / 10% 合并
  - [x] Task 2/3 使用 4-bit 联合 LoRA，生成四元组并派生三元组
  - [x] Task 2/3 batch 20、梯度累积 1，训练平均 GPU 利用率 93.03%，无 OOM
  - [x] Dev 仅拟合 VA 仿射参数，不沿用中文预测项过滤规则
  - [x] 三个 Test 均完成，Task 2/3 1000 条生成 `parse_failures=0`
  - [x] 项目严格评测器与官方脚本复核一致
  - [x] 适配器、原始预测、指标与 GPU 日志回传本地
- 结果：Task 1 Test `RMSE_VA=1.2420729499`；Task 2/3 Test `continuous_F1=0.5396759014` / `0.4962316066`。

### TASK-008：论文官方数据入库与中文解释

- 状态：已完成
- 完成时间：2026-08-12
- 完成内容：
  - [x] 从论文第 7 页 Table 2 录入 Track A 三任务前两名与两个官方基线，共 104 条成绩
  - [x] 单独整理英文 Restaurant 论文成绩与本项目三任务结果
  - [x] 说明数据集缩写、三个任务、RMSE、cF1、V/A 和主要团队方法
  - [x] 明确本项目为比赛结束后的本地实验，不虚构官方名次
  - [x] 提供 CSV 与格式化 XLSX，并检查数值类型、行数、重复项和公式错误
- 文件：`PAPER_RESULTS.md`、`results/paper_track_a_table2.csv`、`results/paper_eng_rest_comparison.csv`、`results/paper_results.xlsx`。
