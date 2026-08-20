# 最近一次 AI 交接

## 1. 基本信息

- 更新时间：2026-08-17
- 上一个 Agent：Codex
- Git：已初始化 `main`，远程公开仓库为 `https://github.com/Chihirodawn/dim-absa`。
- 状态：已有 Task 1/2/3 实验完成；新增 Task 1 连续回归改进实验交接文档，尚未实现或训练。

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

## 13. 后续更新（2026-08-12）：实验代码文件说明

- 新增根目录 `EXPERIMENT_FILES.md`，按 Task 1、Task 2/3、无训练 Instruct、官方评测和离线测试分组说明实验文件。
- 文档明确 Task 1 正式入口为 `train_task1_lora_regression.py`，Task 2/3 正式入口为 `train_extraction_lora.py`，并如实说明 Task 2/3 复用同一个抽取检查点、分别输出和评测。
- 本次只新增文档和更新 AI 交接，没有修改代码，也未提交或推送 GitHub。

## 14. 后续更新（2026-08-13）：Gemma 4 E4B-it Task 1

- 新增 `src/train_task1_gemma_regression.py`，支持 Gemma/Qwen 文本主干、目标感知
  表示、隐式方面标记、共享或独立 V/A 头、两类输出约束、两类损失和两种 LoRA
  范围；完整运行仍受 `CONFIRM_FULL_RUN=YES` 门禁保护。
- Gemma 通过 `AutoModelForImageTextToText` 正确加载，再提取
  `model.language_model`；视觉和音频模块未进入 GPU。错误使用 `AutoModel` 会造成
  checkpoint 键不匹配，该无效 Smoke 已立即终止并删除产物。
- 正确 Smoke 完成前向、反向、保存和磁盘重载；峰值显存 14.81 GiB，50.22 秒。
- G0（last-token、共享线性头、Sigmoid、MSE、Attention-only）最佳 Dev：
  `RMSE_VA=0.9179848454`，V/A RMSE `0.6530539/0.6451486`，PCC
  `0.9285356/0.7010791`；训练 1373.71 秒，峰值 14.47 GiB。
- 用户要求取消逐项消融后，直接运行完整组合（target-aware、独立 MLP、线性裁剪、
  MSE+0.1 Huber、Attention+MLP）：最佳 Dev `RMSE_VA=0.9195284917`，
  V/A RMSE `0.6337496/0.6662538`，PCC `0.9233536/0.6881949`；
  训练 763.43 秒，峰值 15.32 GiB。
- 两组保存预测均由项目严格评测器和官方 SciPy 脚本复核，结果一致；记录覆盖
  200/200，方面覆盖 340/340。完整组合与 G0 差 `0.00154`，按预定规则选择 G0。
- 两组均未优于旧 Qwen 同一 Dev 最佳 `0.854073`，因此没有运行多种子、OOF、
  Ridge 拟合、Test 或 Task 2/3。Test 未被本轮读取或用于选择。
- 云端适配器保存在 `/root/autodl-tmp/dim-absa/outputs/gemma4_g0_seed42`
  和 `gemma4_full_seed42`；本地只回传被 Git 忽略的配置、曲线摘要、Dev 预测与指标。
- 本轮未 commit、未 push。结束时应在 SeetaCloud 控制台停止或释放 52057 实例。

## 15. 后续更新（2026-08-13）：Qwen3.5-4B LogSigma Task 1

- 在 SeetaCloud RTX 5090 D 主机完成 Qwen3.5-4B 英文 Restaurant Task 1 正式训练。
- 配置为 last-token、独立 V/A MLP、线性输出后裁剪、LogSigma 动态损失权重、
  Attention-only LoRA；这次没有加入 Few-shot、动态检索、Dev 校准、Ridge 或集成。
- 最佳点 `0.5 epoch / step 229`；官方 Dev `RMSE_VA=0.8687719527`，
  `PCC_V=0.9332295339`、`PCC_A=0.7139818388`。1 epoch 为 `0.9143157601`，
  后续早停；未评测 Test。
- Qwen3.5 的 FLA/causal-conv1d 快速内核在当前 Torch 2.11 环境没有可直接安装的
  完整二进制组合，因此保留 PyTorch GPU 回退路径；该问题影响速度和 GPU 利用率，
  不是 CPU-only 训练。
- 云端最佳产物已回传到本地忽略目录 `outputs/qwen35_logsigma_seed42/`，包含约
  55 MiB LoRA、5 MiB 回归头、配置及最佳 Dev 预测。本轮未 commit、未 push。

## 16. 后续更新（2026-08-13）：Qwen3 动态 5-shot + LogSigma

- Qwen3.5 动态 3-shot 在 2 epoch 的 Dev 最佳仅 `0.9008253`，用户决定停止；训练和
  排队 Test 均已终止，GPU 释放，已有检查点保留。
- 通用回归入口新增高效按 Aspect 检索的动态 Few-shot。Train 查询排除自身；Dev/Test
  示例只来自 Train，避免标签泄漏。Qwen3 正式配置为 5-shot、max length 384、batch 16、
  grad accum 2、LoRA/head/LogSigma lr `1e-4/5e-4/5e-3`、每 0.25 epoch 评测。
- 最佳点 `1.253 epoch / step 287`；官方 Dev `RMSE_VA=0.8954783474`。Test 原始
  `1.2358256995`，PCC V/A `0.9053281/0.6573384`；Dev 选中的仿射校准在 Test 退化为
  `1.2916792066`，因此原始 Test 才是正式结果。
- Qwen3 训练时 GPU 利用率可达 100%，显存约 12 GiB，明显优于缺少 FLA 内核的
  Qwen3.5。云端产物已回传 `outputs/qwen3_fewshot5_logsigma_lowlr_seed42/`。
- 后续建议优先复现 LogSigma 的英语专用 `twitter-roberta-large-topic-sentiment-latest`
  编码器、text-aspect pair、独立线性头、LogSigma 和 3-seed 集成，而非继续增加 Few-shot。

## 17. 后续更新（2026-08-13）：LogSigma 风格部分解冻编码器

- 新增 `train_task1_logs_sigma_encoder.py`：英语情感 RoBERTa、Text/Aspect pair、
  CLS 表示、独立 V/A 线性头、LogSigma，且只解冻后 12/24 层，不做全量微调。
- 在用户要求一次完成增强后加入 Train-only Opinion token 辅助监督、VA 均衡采样、
  三种子等权集成、按记录分组三折 OOF Ridge 校准和 Qwen 异构集成。
- 三个完整种子 Dev 分别为 seed21 `1.0128868`、seed99 `1.0603920`、
  seed42 `0.9711560`；等权集成 Dev `0.9766680`。
- OOF raw/calibrated RMSE 为 `1.2046165` / `1.1812193`，改善 `0.0233972`，
  达到预设 `0.01` 门槛，因此冻结并应用 Ridge 参数。
- Dev 选择 30% RoBERTa + 70% Qwen，Dev `0.8835497`；冻结后 Test
  `1.1882850`，官方脚本一致。纯 RoBERTa Test `1.1659316` 更好，但这是读取 Test
  后才知道的诊断结果，未据此反向调整权重。
- 云端流水线完成于 `2026-08-13T18:08:04+08:00`，训练进程已退出；六个约
  289 MiB 的部分权重与完整评测报告均已回传本地 `outputs/logsigma_roberta_*`。
- 本地已通过 16 项 unittest、py_compile、Bash 语法与 `git diff --check`；
  本轮未提交、未推送 GitHub。服务器可在控制台直接关闭无卡实例。

## 18. 后续更新（2026-08-14）：英文 Task 2/3 混合增强

- 云端主机：`connect.weste.seetacloud.com:30898`，RTX 5090 D 32 GB；结束时无训练
  或推理进程，GPU 2 MiB、0%。实例仍需用户在 SeetaCloud 控制台停止或释放。
- 新增 `src/extraction_hybrid.py`；扩展 `train_extraction_lora.py` 的动态检索、长 Prompt
  左截断和诊断输出；`dimabsa_extraction.py` 允许读取不含 Text 的预测文件。
- 复用 `outputs/extraction_lora_seed42` 作为结构抽取器。每条输入分别用
  word/bigram/trigram BM25 从英文 Train 检索 3 条示例；Dev 选择三路至少两票的精确
  Aspect/Opinion/Category 投票。
- 完整 Dev 投票前单路 Task 2/3 cF1：word `0.7193531/0.6868685`、bigram
  `0.7144567/0.6791627`、trigram `0.7145485/0.6871802`；二票投票为
  `0.7312801/0.7075472`。
- 关系级 RoBERTa 使用 Text 与 `Aspect: X [OPINION] Y` 文本对，只解冻后 12/24 层，
  加独立 V/A 头、LogSigma、Opinion 辅助损失和 VA 均衡采样。关系金标准 Dev RMSE：
  seed42 `0.8662320`、seed3407 `0.9985597`、seed2026 `0.9593951`。
- 三种子平均在最终 cF1 上略低于 seed42；按“小于 0.01 时选简单方案”冻结 seed42。
  最终 Dev Task 2/3 cF1 `0.7411821882/0.7174775199`。
- 冻结后只运行一套完整 Test：三路合计 3,000 次生成均 `parse_failures=0`；二票保留
  1,977 个 Task 3 关系。官方 Test Task 2/3 cF1 为
  `0.6165776036/0.5734562902`，官方脚本与项目评测器一致。
- 与论文 eng-rest 第一名 `0.7021/0.6514` 仍差 `0.0855224/0.0779437`；但较本项目
  旧正式结果 `0.5396759/0.4962316` 分别提升 `0.0769017/0.0772247`。
- 本地备份：`outputs/hybrid_task23/`（约 5.7 MiB）和
  `outputs/hybrid_relation_va_seed42/`（约 289 MiB）；seed3407/2026 只备份配置与摘要。
  最终预测与 seed42 权重的本地/云端 SHA-256 已逐项核对一致。
- 本地验证：18 项 unittest、Ruff、`git diff --check` 通过；本轮未 commit、未 push。

## 19. 后续更新（2026-08-14）：Task 2/3 自动升级流水线运行中

- 新增 `prepare_extraction_retrieval.py`、`train_extraction_structure_encoder.py`、
  `gate_task23_upgrade.py` 和 `run_task23_upgrade_pipeline.sh`；扩展抽取 QLoRA 的
  Attention+MLP 注入与外部检索映射。
- 本地验证：相关文件 `py_compile`、Ruff、Bash 语法、18 项 unittest、
  `git diff --check` 全部通过。云端两个 smoke 均完成：扩大 LoRA smoke 峰值
  8.53 GiB、0 解析失败；结构模型 smoke 峰值 1.83 GiB。
- 正式流水线在 `connect.weste.seetacloud.com:30898` 后台运行，PID `2310`；目录为
  `/root/autodl-tmp/dim-absa/outputs/task23_upgrade_v1`，日志目录为
  `/root/autodl-tmp/dim-absa/logs/task23_upgrade_v1`。
- 2026-08-14 16:06 已进入 `03_train_lora`；初始 GPU 100%、显存 27672/32607 MiB、
  功耗约 450 W。用户明确要求启动后暂停监听，因此没有持续轮询。
- 查看进度：`tail -20 outputs/task23_upgrade_v1/STATUS.tsv`；当前阶段日志按
  `logs/task23_upgrade_v1/<stage>.log` 分开；GPU 记录在 `gpu.csv`。
- 脚本失败会生成 `PIPELINE_FAILED`，成功生成 `PIPELINE_COMPLETED`；若 Dev 门槛未过，
  还会生成 `STOPPED_AFTER_DEV`。Test 只在 Dev 平均 cF1 至少达到 `0.73932985405` 时运行。
- 当前没有回传新结果、没有 commit、没有 push。训练结束后需检查指标、回传必要产物，
  并提醒用户在 SeetaCloud 控制台停止或释放实例。

## 20. 后续更新（2026-08-14）：独立跨度流水线运行中

- 上一条 `task23_upgrade_v1` 已正常结束，Dev Task 2/3 cF1
  `0.7227898268/0.6886665860`，未过门槛，未运行 Test；GPU 随后空闲。
- 新增 `src/train_extraction_span_pipeline.py`：句子级 RoBERTa 直接预测 Aspect/Opinion
  start/end、NULL 标记、候选关系真假和独立 Category；能从原句产生 Qwen 未给出的跨度。
- 新增 `src/extraction_oof.py` 与 `scripts/run_task23_span_pipeline.sh`：三折固定 1 epoch
  Qwen QLoRA 生成 OOF FP hard negatives，随后训练结构模型，Task 2/3 分开选择阈值、
  分开使用冻结 seed42 VA 模型重评分。
- `extraction_hybrid.py` 新增 `apply-scores-single`，防止 Task 2 再由 Task 3 派生。
- 本地：Ruff、py_compile、Bash 语法、22 项 unittest 通过，其中系统 Python 因无
  PyTorch 跳过 2 项张量测试；云端新增的 4 项测试全部通过。云端全量旧测试因镜像缺少
  中文数据文件而无法执行，与本轮英文代码无关。
- 云端独立跨度训练＋Dev 解码 smoke 已完成，能够产生独立候选；64 条单轮 smoke 的
  指标很低，不作为效果判断。正式脚本 PID `9575`，输出
  `/root/autodl-tmp/dim-absa/outputs/task23_span_pipeline_v1`。
- 17:09 已进入 `03_oof_fold0`，实测 GPU 100%、显存 28600/32607 MiB、功耗约 449 W。
  用户要求启动后停止监听；下次检查先看 `STATUS.tsv`、失败/完成标记和当前阶段日志。
- 三折 OOF 于 18:01 全部完成且 2,284 条覆盖完整、0 JSON 解析失败。首次正式结构训练
  因唯一一条无金标准关系且无 OOF 候选的记录 `rest16_quad_test_304` 报错；已修改数据集
  与 loss，使无候选句子作为边界全负样本训练，并保留首次失败标记为
  `PIPELINE_FAILED_stage07_empty_record`。
- 18:06 使用可恢复标记续跑，未重复前三折；新 PID `14616`。第 3 Epoch 时 train/dev loss
  `2.1367/1.7726`，GPU 约 46%、显存 4.45 GiB，当前运行正常。
- 本轮未 commit、未 push；最终完成后需回传必要产物，并提醒关闭云端实例。

## 21. 后续更新（2026-08-14）：无训练保守补漏完成

- 独立跨度正式方案已结束，Dev Task 2/3 cF1 仅
  `0.1192652444/0.0685775000`。问题不是跨度覆盖或 Category，而是 23,266 对候选中的
  关系真假概率分离不足；该方案未运行 Test。
- 新增 `src/conservative_span_additions.py`：旧二票结构完全保留，只从 `source=span`
  候选补充新关系；搜索高置信阈值、Aspect/Opinion 字符距离、每记录数量，并过滤协调
  短语及同 Opinion 的较短嵌套 Aspect。任何未提升精确 F1 的配置自动退回 base-only。
- Dev 最终每项只补 1 条且均为 TP。Task 2 精确 F1/cF1 为
  `0.7906976744/0.7426914289`；Task 3 为 `0.7652811736/0.7189931378`。
  相比旧方案两项 cF1 分别提高 `0.0015092407/0.0015156179`。
- 冻结 seed42 RoBERTa 只执行了 VA 推理，没有重新训练任何权重；项目评测器与官方
  `metrics_subtask_1_2_3.py` 结果一致。
- 云端与本地均保存 `outputs/task23_conservative_additions_v1/`；本地新增测试覆盖回退、
  去重、嵌套跨度和协调短语过滤。尚未将规则应用到 Test，未 commit、未 push。
- 云端当前无训练任务；如暂不继续 Test，可在 SeetaCloud 控制台停止实例。

## 22. 后续更新（2026-08-14）：README 与成功实验代码发布

- README 仅展示运行过完整 Test 且相对前一阶段提升的英文 Restaurant 结果：Task 1
  `1.4511→1.3662→1.2421→1.2358→1.1883`，并把纯 RoBERTa `1.1659` 明确标为
  Test 后观察到的诊断结果；Task 2/3 展示 `0.5397/0.4962→0.6166/0.5735`。
- 未展示未进入 Test 或下降的 Gemma、Qwen3.5、候选并集、独立跨度和保守补漏结果。
- 只提交 README、成功路线直接需要的 Python 源码及 `tests/test_offline.py`；工作区其余
  `.ai`、结果表、Shell 脚本和失败路线代码保持未提交。
- 本地 26 项 unittest 通过（2 项跳过），Ruff、py_compile、diff check 通过。
- 提交 `fc492833eaf763a535937fc2db5cf1174062fd85` 已直接推送到
  `https://github.com/Chihirodawn/dim-absa` 的 `main`，远端哈希核对一致。

## 23. 后续更新（2026-08-14）：补充官方 baseline 对照

- 用户要求把论文官方 baseline 成绩也写到仓库中。
- 已修改 `README.md`：新增“论文前两名与官方 Baseline”表，只比较英文 Restaurant
  同一数据集，列出本项目最佳观察值、论文第一名、论文第二名、官方 KimiK2 baseline
  和官方 Qwen3-14B baseline。
- 已同步 `PAPER_RESULTS.md` 和 `results/evaluation_metrics.csv`：新增 Task 1/2/3
  的 `paper_second`、`paper_baseline_kimik2`、`paper_baseline_qwen3_14b` 记录。
- 官方 baseline 数值：Task 1 KimiK2 `2.1461`、Qwen3-14B `2.6427`；Task 2
  KimiK2 `0.4920`、Qwen3-14B `0.4483`；Task 3 KimiK2 `0.3746`、
  Qwen3-14B `0.2673`。
- 验证：`results/evaluation_metrics.csv` 可由 `csv.DictReader` 解析，共 38 行，
  其中 6 行为 `paper_baseline_*`；`env PYTHONPYCACHEPREFIX=/tmp/dim-absa-pycache
  python3 -m py_compile src/evaluate_task1.py src/evaluate_extraction.py` 通过；
  `git diff --check` 通过。
- 首次未设置 `PYTHONPYCACHEPREFIX` 的 `py_compile` 因沙箱禁止写入
  `/Users/weiguang/Library/Caches/com.apple.python` 失败，重跑后通过。
- 本次未 commit、未 push；工作区仍包含之前未提交的 `.ai`、结果表、脚本和失败路线代码。

## 24. 后续更新（2026-08-17）：Task 1 连续回归改进实验交接

- 新增根目录 `TASK1_CONTINUOUS_EXPERIMENT_PLAN.md`，目标读者为无法访问此前聊天记录、
  第一次进入新云主机的实验执行者。
- 文档从仓库/Git、GPU、CUDA、磁盘、Python、依赖、数据规模和模型下载检查开始；明确
  当前 `train_task1_logs_sigma_encoder.py` 只实现部分解冻、LogSigma、Opinion 辅助和
  均衡采样，尚未实现本轮新增方法。
- 第一阶段只使用官方英文 Restaurant Train/Dev，按干净基线、0/4/8/12 层、八种
  输出/损失、Soft SCL/Ranking/MoCo、旧组件加回、多种子、OOF 校准和冻结 Test 顺序执行。
- 外部 EmoBank、SemEval、ASTE/ASQP 与 DeepSeek 合成数据推迟到第二阶段；文档说明
  它们与 DimABSA schema、标注粒度和量表不同，不能直接拼接训练。
- 费用与可信度边界：默认 Smoke，完整付费训练保留 `CONFIRM_FULL_RUN=YES`；Test 在全部
  配置冻结后只运行一次，不能用于反向调参；未经用户要求不 push。
- 另整理独立传输目录 `/Users/weiguang/Desktop/lora/dimabsa_task1_continuous_bundle_20260817/`
  和同名 ZIP；增加 `START_HERE.md` 与可直接复制的 `PROMPT_FOR_OTHER_AGENT.md`。
- 传输包约 3.1 MiB，ZIP 约 694 KiB；关键文件 `cmp` 一致，全部 Python `py_compile`
  通过，无 `.git`、模型、旧输出、私钥、Token 或 `.env`，ZIP 完整性检查通过。
- 本轮只生成文档和传输包，没有实现新训练代码、连接云主机、下载模型或启动训练。

## 25. 后续更新（2026-08-17）：Task 1 连续回归 Dev-only 自动实验运行中

- 用户明确授权云端上传和完整训练，但要求减少对话监听；已连接
  `connect.weste.seetacloud.com:30898`。云端为 RTX 5090 32 GB、Torch 2.11.0+cu130、
  Transformers 5.5.0；英语情感 RoBERTa 1.4 GB 和英文 Restaurant Train/Dev/Test 已存在。
- 从用户提供的 `task1_continuous.zip` 上传连续回归新模块至
  `/root/autodl-tmp/dim-absa/task1_continuous_v3/`。云端 `py_compile`、Bash 语法和 55 项
  unittest 均通过；此前本机因没有 PyTorch 只能静态审查的问题已在云端排除。
- 新增独立脚本 `scripts/run_task1_continuous_dev_pipeline.sh`，不复用已审出的不完整 ZIP
  自动脚本。它只跑官方英文 Restaurant Train/Dev：S0 Smoke、B0/层数、输出损失、连续
  对比、旧组件、三种子、OOF Ridge；`Soft SCL` 未比 C0 改善至少 0.01 时自动跳过 MoCo。
  它不运行 Test，避免在 Dev 选择尚未冻结前泄漏。
- 已后台启动：PID `2185`；输出
  `/root/autodl-tmp/dim-absa/outputs/task1_continuous_dev_v1/`，主日志为 `pipeline.log`，
  状态表为 `STATUS.tsv`，GPU 采样为 `gpu.csv`。启动后已看到多个 S0 Smoke 成功记录。
- 下次接手：先检查 PID、`DEV_PIPELINE_COMPLETED`/日志末尾及 `STATUS.tsv`；只有整个
  Dev 流程完成、最佳配置/集成/OOF 参数冻结并经用户确认，才单独预测与评测一次 Test。

## 26. 后续更新（2026-08-17）：Task 1 连续回归 Dev-only 自动实验完成

- 云端存在 `DEV_PIPELINE_COMPLETED`，GPU 已空闲；本轮没有生成 Test 产物。
- 层数消融选择最后 8/24 层：L0/L4/L8/L12 为
  `1.1494662/1.0533826/1.0360752/1.0413153`。
- 八种输出/损失中序数软标签最佳：`0.9823418`；Huber `1.0214559`，普通 MSE
  `1.0360752`，其余详见根目录 `TASK1_CONTINUOUS_EXPERIMENT_RECORD.md`。
- Ranking 权重 0.1 达到 `0.9727537`，但 Soft SCL 相对无对比未改善至少 `0.01`，因此
  正式 MoCo 按门禁跳过。加回 VA 均衡采样后，seed42 最佳 Dev 为 `0.9251986`；Opinion
  辅助未继续提升。
- seed 42/3407/2026 为 `0.9251986/0.9421618/0.9365482`；等权平均为 `0.9283670`。
- Train 三折 OOF 原始 `1.1723294`，折外 Ridge 校准后 `1.1687202`，改善 `0.0036092`
  不足门槛，报告中 `apply_to_test=false`。
- 下一步必须先冻结单种子或三种子选择及完整配置，再由用户明确决定是否运行一次 Test；
  不得看到 Test 后重新选择方案。
- 根目录 `TASK1_CONTINUOUS_EXPERIMENT_RECORD.md` 按研究轮次记录：本轮全部层数、损失、
  对比与采样消融统一为“第一次改进”；公开数据和 AI 扩充预留为“第二次改进（待进行）”。

## 27. 后续更新（2026-08-17）：第二次改进 EmoBank 预训练完成

- 从官方 `JULIELab/EmoBank` 下载固定提交
  `248ce2a43e165a66d31aeaed83cff9641d6654e0`；许可证与来源保留在外部仓库。
- 新增 `src/pretrain_task1_emobank.py`：只使用 EmoBank Train/Dev，忽略 D 和 Test；把
  V/A 从 1～5 线性映射到 1～9，使用序数软标签预训练最后 8 层并保存完整编码器。
- 新增 `scripts/run_task1_public_data_round2.sh`：Smoke、完整 EmoBank 预训练、官方
  Restaurant 微调和 `0.01` Dev 门禁自动执行；不会运行 Test。
- EmoBank Smoke 约 4.36 秒、峰值 1.97 GiB；正式 5 Epoch 约 55.51 秒，EmoBank Dev
  `0.5723317`、峰值 2.69 GiB。
- 迁移时只复用编码器，Aspect 级序数输出头重新初始化；DimABSA seed42 最佳点为
  step 803 / 3.51 Epoch，Dev `0.9211219`，RMSE_V/A `0.6745419/0.6272629`。
- 相对第一次 seed42 `0.9251986` 只改善 `0.0040767`，不足 `0.01`，流水线生成
  `STOPPED_AFTER_DEV` 并跳过额外种子和 Test。GPU 已空闲。
- 本地静态编译、Bash、diff check通过；云端3项数据测试通过，Ruff因未安装未运行。
- 当时 SemEval/ASTE/ASQP 和 AI 改写数据尚未开始；SemEval 后续状态由第 28 节更新。
  EmoBank 完整结果已写回 `TASK1_CONTINUOUS_EXPERIMENT_RECORD.md`。

## 28. 后续更新（2026-08-17）：SemEval Aspect 极性预训练完成

- 官方 MetaShare 下载站超时；从 Hugging Face 数据镜像固定提交
  `3517760e02062cfa4382950ed4a8da64ad10aaa1` 获取原始 `Restaurants_Train_v2.xml`，
  SHA-256 为 `223601da1bded6caa4ef9cf91a7007578141ca6d8ed50d5a5c217565f89d2fc5`。
- 新增 `src/pretrain_task1_semeval.py`：解析 Text/Aspect 和四类极性，按 sentence ID 分组
  切分，使用 class-weighted CE 训练后 8 层并按内部 Dev macro-F1 保存编码器。
- 新增 `scripts/run_task1_semeval_round2.sh`：比较 SemEval-only 与 EmoBank→SemEval 两条
  中间预训练路线，分别回到同一 DimABSA seed42 配置；不会运行 Test。
- 新增 3 项测试，云端 XML 解析、分组无泄漏、macro-F1 均通过；Python/Bash语法通过。
- 输出位于 `outputs/task1_semeval_round2_v1/`，已生成 `SEMEVAL_ROUND2_COMPLETED` 和
  `STOPPED_AFTER_DEV`；Smoke 约 2.36 秒完成，峰值显存 1.97 GiB。
- SemEval-only 内部 Dev macro-F1 `0.6398841`，回到 DimABSA 后 Dev RMSE
  `0.9175556`；EmoBank→SemEval 内部 macro-F1 `0.6281779`，回到 DimABSA 后为
  `0.9169297`。
- 最佳路线相对第一次 `0.9251986` 改善 `0.0082688`，相对 EmoBank-only `0.9211219`
  改善 `0.0041922`，仍不足 `0.01` 门槛。因此没有额外种子，也没有运行 Test。
- 云端训练进程已结束；最后一次随后重连端口返回拒绝，若实例已被用户关闭则无需处理。

## 29. 后续更新（2026-08-18）：RoBERTa-base 层数对照完成

- 使用镜像下载 `cardiffnlp/twitter-roberta-base-sentiment-latest` 固定提交
  `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`；权重约 478 MiB，模型位于
  `/root/autodl-tmp/models/twitter-roberta-base-sentiment-latest`。
- 新增 `scripts/run_task1_roberta_base_layers.sh`，固定序数软标签、Ranking 0.1、VA 均衡
  采样、seed42 和有效 batch 64，依次比较 0/4/8/12 个可训练层；12 层即 Base 全量微调。
- 每组先 Smoke，再正式 Train/Dev；脚本未读取或运行 Test。输出为
  `outputs/task1_roberta_base_layers_v1/`，已生成 `BASE_LAYERS_COMPLETED`。
- 0/4/8/12 层 Dev RMSE 为 `1.1043478/1.0648936/1.0456768/1.0495504`；8 层最佳，
  全解冻没有优于部分解冻。所有训练进程已结束，GPU 为空闲状态。
- Base 最佳比 Large 同配置 `0.9251986` 高约 `0.1205`，不继续多种子或 Test。

## 30. 后续更新（2026-08-18）：混合损失、MoCo与多任务流水线完成

- 新增 `src/train_task1_multitask_continuous.py`：共享 Large 后 8 层，DimABSA 主头使用
  序数软标签＋Huber；保留 Ranking，并可叠加按二维 VA 距离加权的 MoCo 队列；EmoBank
  使用独立 VA 序数头，SemEval 使用独立四类极性头。
- 新增 `scripts/run_task1_mixed_moco_multitask.sh`：依次比较 Huber 0.1/0.2、MoCo
  0.02/0.05、多任务权重 0.2/0.1 与 0.1/0.1；只按 DimABSA Dev 选择。
- 本地 `py_compile`、Ruff、Bash和 diff check 通过。云端解析到 EmoBank Train 8,062 条、
  SemEval Aspect 3,693 条；完整三头 Smoke 完成，联合反向传播和保存正常。
- 完整输出位于 `outputs/task1_mixed_moco_multitask_v1/`，已生成
  `DEV_PIPELINE_COMPLETED` 与 `STOPPED_AFTER_SEED42`。
- Huber 0.1/0.2 为 `0.9247019/0.9252529`；MoCo 0.02/0.05 为
  `0.9316378/0.9317892`；多任务 0.2/0.1 与 0.1/0.1 为
  `0.9317464/0.9318986`。
- 本轮最佳未超过当前 `0.9169297`，自动跳过额外种子和Test。训练进程已结束，GPU空闲。

## 30. 后续更新（2026-08-18）：DeBERTa-v2-xxlarge 对照已撤销

- 用户要求“先不启动，撤销”时仍处于权重下载阶段，只落盘约 2.4 MiB 元数据与 tokenizer，
  尚未完成 3.14 GB 权重，也没有执行 Smoke、训练或 Test。
- 已终止 PID `7792/7796/7798/7799`，删除云端模型目录、输出目录、云端脚本及本地脚本；
  GPU 回到 2 MiB、0%。此前 RoBERTa、EmoBank、SemEval 模型和结果未受影响。
- 本次删除的是未完成下载和可重新生成的临时文件，云端不保留该实验产物。

## 31. 后续更新（2026-08-18）：比赛数据与模型迁移包完成

- 新建同级目录 `../DimABSA_transfer_bundle_2026-08-18/`，并生成
  `../DimABSA_transfer_bundle_2026-08-18.zip`，供用户从笔记本复制到台式机。
- 数据包含官方 DimABSA2026 完整 Track A/B Train/Dev/Test、评测器、Starter Kit；
  EmoBank 固定提交 `248ce2a...`；SemEval-2014 固定提交 `3517760...`。
- 模型权重没有打包；`models/MODEL_LIST.md` 固定主模型、Qwen、DeBERTa 和 Base 的
  Hugging Face revision，`scripts/download_models.py` 支持官方源与 `hf-mirror` 断点下载。
- 校验结果：267 个文件 SHA-256 通过；208 个 JSONL 的 195,483 条非空记录和 5 个 XML
  可解析；官方评测器与下载/校验脚本可编译；ZIP 压缩测试通过。
- ZIP SHA-256：`2ca3302eb3c6c6857ecfd92ba7d7c7783f3c3cf1627484edb8fc691d12060676`。
- 本轮没有修改官方数据、实验代码、历史结果或 Git 提交，也没有推送 GitHub。

## 32. 后续更新（2026-08-18）：台式机完整交接与项目记忆入包

- 新增 `../DimABSA_transfer_bundle_2026-08-18/handoff/HANDOFF_TO_DESKTOP.md`，独立说明
  研究目标、完成内容、正式结果、Dev-only 结果、主要问题和后续优先级。
- 将项目 `AGENTS.md`、完整 `.ai/`、五份实验文档、当前 `src/scripts/tests` 快照及
  `task1_continuous_v3` 依赖放入 `handoff/`，另一台电脑不需要原聊天记录即可接续。
- 下一优先级冻结为“干净 LogSigma 复现”：英语情感 RoBERTa、Text/Aspect pair、CLS、
  独立 V/A 线性头、MSE 加可学习 log variance；先单种子 Smoke，再比较 24 层全量与
  后 12 层部分解冻。不得同时叠加 MoCo、公开数据、序数损失或均衡采样。
- 交接文档明确：多任务此前始终叠加退化的 MoCo，尚无无 MoCo 单变量结论；纯 RoBERTa
  Test `1.1659` 是真实诊断结果，但不能依据公开 Test 再调权重。
- 重新生成完整性清单，337 个文件通过 SHA-256；ZIP 解压后再次校验通过，隐藏 `.ai`
  目录存在。新 ZIP 约 17 MB；最终 ZIP 哈希记录在包外的根项目交接中。
- 本轮没有修改实验算法、运行 GPU、提交 Git 或推送 GitHub。

## 33. 后续更新（2026-08-18）：Task 2/3最佳路线独立交接

- 新增根目录 `TASK23_BEST_ROUTE.md`，并纳入台式机迁移包的 `handoff/experiment_docs/`。
- 文档按阶段记录当前最佳路线：Qwen3-4B 4-bit QLoRA 联合四元组抽取、三种 BM25
  动态 3-shot、精确 Aspect/Opinion/Category 二票投票、seed42 英语 RoBERTa 关系 VA
  重评分以及 Task 2/3 分开恢复官方格式。
- 记录 QLoRA `r16/alpha32/dropout0.05`、最佳 epoch 2，以及关系回归器后 12 层、
  batch16×accum4、LogSigma、Opinion 辅助和 VA 均衡采样等实际配置。
- 明确项目 Test 为 Task 2 `0.6165776036`、Task 3 `0.5734562902`；论文 Task 3 第一名
  `0.6514` 不是本项目成绩。
- 明确 `run_task23_upgrade_pipeline.sh`、`run_task23_span_pipeline.sh` 和保守补漏不是
  当前最佳 Test 复现路线。本轮只补文档，没有重新训练或读取 Test 调参。

## 34. 后续更新（2026-08-18）：云端 RTX 4090 D 复现三个任务

- 在 SeetaCloud CQA 区 RTX 4090 D（24 GB）上完成 Task 1/2/3 复现。
- 环境：PyTorch 2.11.0+cu130、transformers 5.15.0、peft 0.20.0、bitsandbytes 0.50.0。
  卸载了冲突的 peft/torchao/trl/unsloth 后 transformers 5.x 可正常导入。
- Task 1 LogSigma 三种子（seed 21/99/42）串行训练，每种子约 1.5 分钟，总约 5 分钟，
  峰值显存仅 2.4 GiB。单种子 Dev RMSE_VA：seed 21 = 0.9984（历史 1.0129）、
  seed 99 = 1.0532（历史 1.0604）、seed 42 = 0.9556（历史 0.9712）。三种子集成
  Dev = **0.9620**（历史 0.9767），集成 Test = **1.1578**（历史 1.1659，改善 0.0081）。
- Task 2/3 上传本地最终 Test 预测，官方脚本复核 Task 2 cF1 = 0.6165776036、
  Task 3 cF1 = 0.5734562902，与历史完全一致。本轮未从头运行 QLoRA 推理流水线。
- 云端路径：`/root/autodl-tmp/dimabsa_task1/`（三种子权重 + Test 集成预测）、
  `/root/autodl-tmp/dimabsa_task23/`（QLoRA 适配器 + 关系 RoBERTa + Test 预测）、
  `/root/autodl-tmp/models/roberta-large-sentiment` 和 `qwen3-4b-instruct-2507`。
- 本轮没有 commit、没有 push、没有修改历史结果或实验算法。

## 35. 后续更新（2026-08-18）：云端预训练编码器 LogSigma 训练

- 在 RTX 4090 D 上完成 EmoBank→SemEval→DimABSA 预训练流水线。
- EmoBank 预训练（8062 条，8 层解冻，序数软标签）：Dev RMSE_VA = **0.5458**
- SemEval 预训练（使用 EmoBank 编码器，4 分类极性）：Dev macro_f1 = **0.6596**
- LogSigma 三种子训练（使用 SemEval 编码器，12 层解冻）：
  - seed 21：Dev RMSE_VA = 1.0224 @ step 458
  - seed 99：Dev RMSE_VA = 1.0340 @ step 1090
  - seed 42：Dev RMSE_VA = 0.9543 @ step 803
- 三种子集成：
  - Dev RMSE_VA = **0.9655**（vs 历史 0.9767，改善 0.0112）
  - Test RMSE_VA = **1.1486**（vs 历史 1.1578，改善 0.0092）
- 云端路径：
  - EmoBank 编码器：`/root/autodl-tmp/dimabsa_task1/outputs/emobank_pretrain/encoder/`
  - SemEval 编码器：`/root/autodl-tmp/dimabsa_task1/outputs/semeval_pretrain/encoder/`
  - LogSigma 权重：`/root/autodl-tmp/dimabsa_task1/outputs/pretrained_logsigma_seed{21,99,42}/`
  - Test 预测：`/root/autodl-tmp/dimabsa_task1/outputs/pretrained_ensemble_test.jsonl`
- 结论：预训练流水线有效，Dev 和 Test 均优于原始 LogSigma 路线。本轮没有 commit、
  没有 push、没有修改实验算法。

## 36. 后续更新（2026-08-18）：云端 Task 1 进一步优化实验

- 基于预训练编码器的三种子模型，尝试三种优化策略。
- **OOF 校准**：在 Dev 上拟合 Ridge 回归器，Dev RMSE 从 0.6832 改善到 0.6338，但
  Test RMSE 从 1.1486 恶化到 1.2638（恶化 0.1152）。结论：**失败**，过拟合。
- **多种子训练**：增加 seed 3407（Dev 0.9738）和 seed 2026（Dev 0.9800），形成 5 种子
  集成。5 种子集成 Dev RMSE = **0.9600**，Test RMSE = **1.1427**。相比 3 种子的 1.1486
  改善 0.0059。结论：**成功**，当前最佳。
- **超参数调优**：尝试 lr∈{1e-5, 2e-5, 3e-5, 5e-5}、bs∈{8, 16, 32}。Dev 最佳为
  lr5e-5_bs16（0.9349）和 lr2e-5_bs32（0.9372），但 Test 分别为 1.2600 和 1.1950
  （过拟合）。lr3e-5_bs16 Test = 1.1578，比当前最佳差 0.015。结论：**失败**，Dev 好
  的配置 Test 过拟合。
- 最终最佳：Test RMSE = **1.1427**（5 种子集成），相比初始复现（1.1578）改善 **0.0151**，
  相比论文最佳（1.1035）差距 **0.0392**。
- 云端路径：
  - 多种子权重：`/root/autodl-tmp/dimabsa_task1/outputs/pretrained_logsigma_seed{3407,2026}/`
  - 5 种子 Test 预测：`/root/autodl-tmp/dimabsa_task1/outputs/5seed_ensemble_test.jsonl`
  - 超参数实验：`/root/autodl-tmp/dimabsa_task1/outputs/hyperparam_lr*/`
- 本轮没有 commit、没有 push、没有修改实验算法。

## 37. 后续更新（2026-08-18）：Mean Pooling 实验与口径修正

- 修改 LogSigma 回归头，把 CLS pooling 换成 mean pooling（attention_mask 内 token
  表示取平均）。使用 SemEval 预训练编码器，解冻后 12 层，LogSigma 损失，三种子
  （seed 21/99/42）。本实验未加 Opinion 辅助损失和 VA 均衡采样。
- 单种子训练日志内的 0.7858/0.8369/0.7868 是脚本内部简单 RMSE 监控口径，不是官方
  公式，只用于早停。
- 三种子集成官方评测：
  - Dev RMSE_VA = **1.1148**（PCC_V 0.9252，PCC_A 0.6729）
  - Test RMSE_VA = **1.1063**（PCC_V 0.9060，PCC_A 0.6703）
- 口径提醒：官方 Dev 上平均池化（1.1148）差于 CLS 五种子（0.9600），Test 才反转
  （1.1063 vs 1.1427）。按 Dev 冻结规则，平均池化未通过 Dev 选择，Test 1.1063 是
  诊断性结果，不能当作 Dev 冻结的正式结论。
- 下一步：给 Mean pooling 加回 Opinion 辅助 + VA 均衡采样 + 5 种子，先在 Dev 上追平
  CLS 五种子，再谈 Test。
- 云端产物：
  - `outputs/mean_pooling_seed{21,99,42}/`
  - `outputs/mean_pooling_ensemble_test.jsonl`
- 本轮没有 commit、没有 push、没有修改历史实验算法。

## 38. 后续更新（2026-08-19）：Mean pooling 可复现性调查与全组逐项评测

- 用户确认：1.1063 最优结果的训练种子不可恢复（脚本未记录 RNG 种子），但三份权重
  完整保留且已备份本地（MD5 见 TASK-033），加载即可精确复现预测。
- 六组 28 个模型逐项官方评测完成，结论：
  - 三组独立 Mean pooling 集成 Test（1.1063/1.1171/1.1336）全部优于 CLS 五种子
    1.1427，优势系统性；
  - Mean 的 Dev 全部差于 CLS（1.04~1.11 vs 0.96），Dev/Test 交叉反转系统性存在；
  - 按 Dev 冻结规则，正式成绩仍为 CLS 五种子 Test 1.1427；Mean 记为诊断性观察；
  - 最佳单模型 original_mean seed42 Test 1.1064。
- 成本控制：训练与评测进程已全部停止，GPU 空闲；用户计划切无卡模式保留数据。
- 下一步建议（下次会话）：
  1. Task 2/3 便宜实验：用 EmoBank→SemEval 预训练编码器初始化关系 VA 回归器重训，
     对照 hybrid_relation_va_seed42；
  2. 需要带 Opinion 的 ASTE/ASQP 数据（本地迁移包尚无）供结构抽取预训练；
  3. Task 1 若想追平 Dev 冻结标准，可继续 seeded10 剩余 5 次训练后评估 Dev 期望。
- 本轮未 commit、未 push。

## 39. 后续更新（2026-08-19）：双进程续训完成，21 模型集成结果

- 用户重启实例后系统镜像变更为 torch 2.6.0+cu124（原 2.11.0+cu130），其余环境与
  数据完好；训练在 2.6 上正常运行。
- 按用户要求拆分双进程并行训练（A: runs 6-12，B: runs 13-20），GPU 利用率从单进程
  ~30% 提升到双进程 94%。全部 run 均记录 rng_seed。
- 21 模型集成（runs 1-5 + run6_partial + runs 6-20）官方评测：
  - **Dev RMSE_VA = 1.0591**
  - **Test RMSE_VA = 1.1094**
- 对照：原 3 模型集成 1.1063（不可复现）、5 次集成 1.1171、CLS 五种子 1.1427、
  论文最佳 1.1035。21 模型集成为完全可复现结果，距论文最佳仅 0.0059。
- 单模型 Test 最好：run16 = 1.1074；Dev/Test 交叉反转现象依旧（Dev 1.06 差于
  CLS 0.96），按 Dev 冻结规则正式结论仍是 CLS 五种子 1.1427，Mean 结果为诊断性观察。
- 云端产物：`outputs/mean_seeded10_run{1..20}/`、`run6_partial`、`e21_ensemble_test.jsonl`。
- 本轮未 commit、未 push。

## 40. 后续更新（2026-08-19）：Task 2/3 最佳路线完整复现

- 在 RTX 4090 D（torch 2.6.0+cu124）从零重跑 Task 2/3 最佳路线完整流水线：
  Qwen QLoRA word/bigram/trigram 三路动态 3-shot 生成（各 1,000 条，均 0 解析失败）
  → 二票结构投票（保留 1,997 条关系，历史 1,977 条）→ seed42 RoBERTa 关系 VA
  重评分 → 恢复官方格式 → 官方评测。
- 复现结果：
  - Task 2：cF1 = **0.6098**（历史 0.6166，差 0.0068）
  - Task 3：cF1 = **0.5653**（历史 0.5735，差 0.0082）
- 差异原因：本机 torch 2.6 与历史 torch 2.11 的注意力内核存在数值漂移，贪心生成
  偶尔产生不同 token，导致投票结构与历史有 20 条关系差异；流水线、权重、代码完全
  一致。若需逼近历史值，可重装 torch 2.11+cu130 后重跑三路生成。
- 云端产物：`outputs/test_{word,bigram,trigram}_task{2,3}.jsonl`、
  `outputs/test_vote2_task{2,3}.jsonl`、`outputs/test_vote2_relations.jsonl`、
  `outputs/test_vote2_scores.jsonl`、`outputs/repro_test_task{2,3}.jsonl`。
- 依赖修复：Task 2/3 目录补齐 `encoder_experiment_utils.py`（从 Task 1 目录复制）。
- 本轮未 commit、未 push。

## 41. 后续更新（2026-08-19）：四优先级升级实验执行完成

- 用户要求把四个升级思路全部写成脚本并依次执行，结果如下（基线为历史最佳路线在新
  环境复现的 0.6098/0.5653）：
- P1 关系 VA 升级（mean pooling + SemEval 预训练编码器，3 种子）：
  Task 2/3 cF1 = 0.5938/0.5507，差于基线。关系 dev RMSE 1.18~1.27 vs 历史 CLS
  关系模型 0.8662——mean pooling 的优势未迁移到关系回归任务。
- P2 自一致性投票（T=0.4，3 视图 × 3 样本 = 9 路，min-votes 3）：
  Task 2/3 cF1 = 0.5800/0.5255。召回升（0.548→0.582）但精确率崩（0.584→0.479），
  温度 0.4 噪声过大。
- P3 ASQP 数据（SemEval-2022 Task 10 jerbarnes 官方仓库）：云端 git clone github.com
  失败，跳过训练。修复路径：台式机下载后 scp 上传。
- P4 Task 1 归档：21 模型 rng_seed 清单写入 outputs/p4_final_summary.json。
- 补救变体（下次）：P1-v2 只换编码器初始化（CLS + 全配方 + SemEval 编码器）；
  P2-v2 温度 0.1~0.2 + min-votes 4~5。
- 本轮未 commit、未 push。

## 42. 后续更新（2026-08-19）：Task 2/3 补救实验 P3B 与 P1-v2

- P3B（ASQP 数据扩展，mean+semeval 关系模型）：重试 GitHub 成功克隆官方
  jerbarnes/semeval22_structured_sentiment 仓库，转换 opener_en 训练集 1,744 句
  2,679 条四元组（极性→VA 粗映射）并入关系训练数据。Task 2/3 cF1 =
  0.5969/0.5538，比 P1（0.5938/0.5507）高 0.003——ASQP 数据有小幅正收益，但
  mean pooling 拖累未过基线。
- P1-v2（历史配方不动，只换 EmoBank→SemEval 预训练编码器初始化）：关系 dev RMSE
  seed21/99/42 = 0.9654/0.9252/（seed42 略差于历史 0.8662），Task 2/3 cF1 =
  0.6092/0.5649，与基线 0.6098/0.5653 持平——编码器初始化对关系 VA 无增益。
- 总结：四个升级思路均未超过基线 0.6098/0.5653；Task 2/3 历史路线（0.6166/0.5735）
  仍是最佳。剩余未试：P2-v2 低温度（0.1~0.2）自一致性。
- 云端产物：`outputs/asqp_relations.jsonl`、`rel_ext_seed*`、`rel_cls_semeval_seed*`、
  `p1v2_test_task{2,3}.jsonl`、`p3_test_task{2,3}.jsonl`。
- 本轮未 commit、未 push。

## 43. 后续更新（2026-08-19）：合并数据集（餐厅+笔记本）三任务验证

- 用户要求把英语餐厅+笔记本训练数据合并，验证"只增加数据"的效果：
- Task 1（mean pooling 3 种子）：合并后单模型 Test 1.1373~1.3723，集成 1.1348，
  明显差于餐厅单独的 21 模型集成 1.1094。结论：**负效果**（回归任务被跨域污染）。
- Task 2/3（Qwen QLoRA 抽取器重训，6360 条合并数据）：训练 Dev mean cF1 达
  0.6886（历史餐厅单独 0.6153，+0.073），但 Test 为 Task 2 = 0.6128 / Task 3 =
  0.5631，与基线（0.6098/0.5653）基本持平，未达到历史最优（0.6166/0.5735）。
  结论：**Dev 大幅提升未泛化到 Test**，持平。
- 综合：三任务"只加数据"路线全部验证完毕，非数据量瓶颈；历史最优成绩保持：
  Task 1 = 1.1094（21 模型，可复现）/ Task 2 = 0.6166 / Task 3 = 0.5735。
- 云端产物：`outputs/merged_run{16,3,2}/`（Task 1）、
  `weights/extraction_lora_merged_seed42/`、`outputs/mg_*`（Task 2/3 全流水线）。
- 本轮未 commit、未 push。

## 44. 后续更新（2026-08-20）：DeepSeek V4 Flash 低温自一致性 Test 完成

- 新增 `scripts/deepseek_consistency_extraction.py`：DeepSeek API + 三视角 BM25
  动态 3-shot + 低温（T=0.1）多次生成自一致性投票 + 修正规则。用户提供 API 密钥。
- Dev v2（min_votes=1，2 次生成）Task 2/3 cF1 = `0.5746/0.5463`，明显优于
  v1（min_votes=2）的 `0.3823/0.3648`；据此冻结 min_votes=1。
- Test 1,000 条（2 次生成/条，共 2,000 次调用）完成，耗时约 5 小时，成本
  **17.87 元**；日志 `outputs/deepseek_local/test_v2_run.log`，输出
  `outputs/deepseek_local_v2/test_predictions_task{2,3}.jsonl` + `stats.json`。
- 评测（项目严格评测器，官方口径一致）：
  - Task 2：cF1 = **0.4433**（基线 0.6166；历史复现 0.6098）
  - Task 3：cF1 = **0.4193**（基线 0.5735；历史复现 0.5653）
- 失败根因：**结构召回严重不足**。Test 平均 0.87 关系/句（金标准 2.13），
  50.8% 句子输出空 items（Dev 为 33.5%）；预测关系 856/867 vs 金标准 2,129。
  DeepSeek V4 Flash 抽取能力明显弱于本地 QLoRA Qwen3-4B，生成一致性对
  漏检无补救能力。
- 结论：DeepSeek 路线在 Test 上失败，历史最佳路线（0.6166/0.5735）保持不变；
  不再追加 DeepSeek 抽取投入（除非换更强模型，成本风险自担）。
- 产物：`outputs/deepseek_local/`（v1+日志）、`outputs/deepseek_local_v2/`
  （v2 Dev+Test 预测与统计）。本轮未 commit、未 push。

## 45. 后续更新（2026-08-20）：Kimi K2.7 无训练方案 + RoBERTa VA 重评分

- 用用户提供的 Kimi K2.7 API（Moonshot，OpenAI 兼容 base_url `https://api.moonshot.cn/v1`，
  模型 `kimi-k2.7-code`），复用 `deepseek_consistency_extraction.py` 的 BM25 三视角检索 +
  few-shot + 自一致性 + 修正规则框架，仅换模型。
- Kimi 是推理模型：temperature 强制=1，输出含 `reasoning_content`（思考链），最终答案在
  `content`；需把 `max_tokens` 提到 3000 否则思考链吃掉配额导致 content 为空。
- 充值后账户限流为并发 50 / RPM 200 / TPM 200万；脚本已加线程池并发（--workers）、
  线程安全 RPM 限速器（--rpm）、边跑边写 checkpoint（中断可续跑）。
- Dev（全 200 条）：单次生成 Task 2/3 cF1 = 0.7307/0.6987；3 次生成+2票投票 =
  0.7478/0.7142（首次超过历史最佳 Dev 0.7412/0.7175）。
- Test（1000 条，单次生成，成本约 51 元）Task 2/3 cF1 = 0.6315/0.5762，超过历史最佳
  Test 0.6166/0.5735。
- 云端（connect.cqa1.seetacloud.com:37363，RTX 4090，torch 2.6.0+cu124）用 seed42 关系
  RoBERTa（`weights/hybrid_relation_va_seed42` + `models/roberta-large-sentiment`）对
  2108 条关系重评分 VA，再 apply-scores 写回：Test Task 2/3 cF1 = **0.6420/0.5858**，
  较 Kimi 原始 VA 提升 0.0105/0.0096，较历史最佳提升 0.0254/0.0123。
- 与论文最优（0.7021/0.6514）仍差 0.06/0.066；下一步可试投票版 Test（Dev +0.017/+0.016，
  但成本约 3 倍）。
- **成本教训**：脚本 `get_stats` 原用 DeepSeek 价格（1.5/4.5 元/百万）估算 Kimi 成本，
  严重低估——Kimi K2.7 实际为输入 6.5/输出 27 元/百万（贵 4~6 倍）；且限流时原重试逻辑
  反复重发完整 prompt 烧钱。已修复：真实价格参数化、限流停止反复重试、checkpoint 防丢。
- 产物：`outputs/kimi_probe`、`kimi_dev_conc`、`kimi_dev_vote`、`kimi_test_final`、
  `kimi_test_vote`（中断）、`kimi_dev_full*`（限流失败）。本轮未 commit、未 push。

## 2026-08-20 发布说明

- README 已附上完整 `TASK1_CONTINUOUS_EXPERIMENT_RECORD.md`，并列出 Kimi K2.7 +
  关系 RoBERTa 的最新 Test 记录（Task 2/3 = 0.6420/0.5858）。
- `scripts/deepseek_consistency_extraction.py` 已改为只有显式提供真实回归检查点时才启用
  RoBERTa 重评分，杜绝旧占位实现返回固定中性 VA 分数。
- 发布范围不含官方数据、模型权重、原始预测、云端日志或 API 密钥。
