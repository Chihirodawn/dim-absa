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
- 最终结果：Task 2 test `continuous_F1=0.2869350972`；Task 3 test `continuous_F1=0.2535017417`；`parse_failures=0`。
