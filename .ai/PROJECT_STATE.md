# 项目状态

## 1. 项目简介

- 项目名称：dim absa
- 项目目标：使用同一个 Qwen3-4B-Instruct-2507 完成中文与英文 Restaurant 的 DimASR、DimASTE 与 DimASQP，并比较 Instruct 与 LoRA 路线。
- 当前阶段：中文 Instruct、英文 Qwen LoRA 三任务、Gemma 4 E4B-it Task 1 对照，以及
  英文 Task 2/3 三视角动态检索、投票与关系 RoBERTa 重评分均已完成；英文 Task 1 的
  连续回归第一阶段 Dev-only 消融也已完成；云端 RTX 4090 D 复现已完成（Task 1 Test
  RMSE 1.1578，Task 2/3 cF1 0.6166/0.5735），结果与本地历史一致或更优。
  本次修改尚未发布 GitHub。
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
| `src/train_task1_gemma_regression.py` | Gemma/Qwen Task 1 通用 LoRA 回归 | 目标感知表示、独立双头、文本主干 |
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
- 英文 Restaurant Task 1：dev 选择 few-shot + calibration（`RMSE_VA=1.0998732547`）；唯一 test `RMSE_VA=1.4511021582`，未超过 LogSigma 的 `1.1035`。
- 英文 test 1,000 条、1,504 个方面，`parse_failures=0`、`format_retry_recoveries=0`，推理 119.64 秒，峰值 CUDA allocated 10.35 GiB；官方脚本复核一致。
- 英文无训练增强版：动态检索 Few-shot 加入 Direct/CoT/固定 Few-shot，四路原始预测等权平均后应用冻结 Dev 校准；5 折分组 CV `1.0415`，完整 Dev `1.0263`，Test `1.3661908335`。
- 新 Test 的 V/A RMSE 分别为 `1.1296`/`0.7685`；相对原 Few-shot + calibration 降低 `0.0849`（约 5.85%），但仍高于 LogSigma `1.1035`。
- 英文 LoRA Task 1：双头回归 + Dev 校准 Test `RMSE_VA=1.2578151703`；Dev 冻结的 90% LoRA + 10% 无训练集成最终为 `1.2420729499`。
- 英文联合 QLoRA Task 2/3：Test `continuous_F1=0.5396759014` / `0.4962316066`，两项平均 `0.5179537540`，官方脚本一致，生成 `parse_failures=0`。
- Gemma 4 E4B-it 英文 Task 1：简单 G0 Dev `RMSE_VA=0.9179848454`；一次性完整
  改进组合 Dev `0.9195284917`。均未优于旧 Qwen Dev `0.854073`，故未评测 Test。
- Qwen3.5-4B 英文 Task 1：独立 V/A MLP + 线性输出 + LogSigma + Attention-only
  LoRA，最佳 Dev `RMSE_VA=0.8687719527`，未超过旧 Qwen3 `0.854073`；当前环境
  缺少线性注意力快速内核，GPU 利用率偏低，未评测 Test。
- Qwen3 动态 5-shot + LogSigma 低学习率：最佳 Dev `0.8954783474`；Test 原始
  `1.2358256995`，Dev 仿射后 Test 退化为 `1.2916792066`。相对旧 Qwen3 最终
  `1.2420729499` 仅小幅改善，后续不继续增加 Few-shot。
- LogSigma 风格英语 RoBERTa（后 12 层部分解冻、Opinion 辅助、VA 均衡、三种子）：
  纯编码器集成 Dev/Test 为 `0.9766679549` / `1.1659316485`；按 OOF 校准并在
  Dev 冻结 30% RoBERTa + 70% Qwen 后，Dev/Test 为 `0.8835496590` /
  `1.1882849869`。冻结方案相对 Qwen3 动态 5-shot Test 改善约 3.85%，但仍未达到
  LogSigma 官方 `1.1035`。
- 英文 Task 2/3 混合增强：复用 Qwen 抽取 LoRA，用 word/bigram/trigram 动态 3-shot、
  二票精确结构投票和 seed42 关系 RoBERTa 重评分；Dev cF1
  `0.7411821882/0.7174775199`，冻结后的 Test cF1
  `0.6165776036/0.5734562902`，三路生成共 3,000 条且 0 解析失败。
- Task 2/3 正式训练 batch 20、梯度累积 1，最佳 epoch 2；训练约 27.18 分钟，平均 GPU 利用率 93.03%，峰值 CUDA allocated 25.10 GiB。
- 已按论文第 5.1 节与附录 C 整理论文正式评测指标、公式、示例值和本项目对应结果；完整多语言排行榜已从仓库移除。

## 7. 重要约束

- test 金标准不用于选择 Prompt。
- 有解析回退的结果不作为正式模型结果。
- 不再根据公开 test 修改 Prompt 或校准参数。

## 8. 最近更新

- 2026-08-10：Codex 初始化项目、完整资源、推理/评测代码、离线测试、README 与均值基线。
- 2026-08-11：完成云端正式 Task 1；新增 Transformers 后端、稳定输出协议、格式重试和线性校准，最终 test RMSE 1.1149206501。
- 2026-08-11：发布公开仓库 `https://github.com/Chihirodawn/dim-absa`；只包含安全发布范围。
- 2026-08-11：完成中文餐厅 Task 2/3 Instruct 基线、dev 冻结后唯一 test、官方连续 F1 复核及本地结果回传，并以安全范围发布代码、文档与汇总指标到 GitHub。
- 2026-08-12：只复用原 Qwen Instruct 方法完成英文餐厅 Task 1；dev 选 Few-shot + calibration，test RMSE 1.4511021582；结果已回传但尚未发布 GitHub。
- 2026-08-12：在不训练、不换模型下加入动态 Train 示例检索和四路等权集成；冻结前 5 折 CV 验证提升，最终 test RMSE 1.3661908335；结果已回传但尚未发布 GitHub。
- 2026-08-12：完成英文 Restaurant 三任务 LoRA；Task 1 RMSE 1.2421，Task 2/3 连续 F1 0.5397/0.4962；适配器、结果和 GPU 日志已回传，尚未发布 GitHub。
- 2026-08-12：将论文内容收缩为正式评测指标与结果：RMSE、VA 距离、cTP、cPrecision、cRecall、cF1；移除无关的完整多语言排行榜。
- 2026-08-13：完成 Gemma 4 E4B-it Task 1 通用回归代码、Smoke、G0 与完整组合；
  官方 Dev 评测一致，因未超过 Qwen 而停止后续昂贵实验。
- 2026-08-13：完成 Qwen3.5-4B LogSigma Task 1 对照，官方 Dev `0.868772`；
  最佳权重已回传，因未超过旧 Qwen3 `0.854073` 而不运行 Test。
- 2026-08-13：停止低效的 Qwen3.5 动态 Few-shot，改用 Qwen3 完成动态 5-shot、
  LogSigma、Dev 校准和 Test；正式原始 Test `1.235826`，校准迁移失败。
- 2026-08-13：完成 LogSigma 风格英语专用编码器增强流水线；三种子、三折 OOF、
  Ridge 和 Qwen 异构集成全部完成，冻结方案 Test `1.188285`，结果与权重已回传。
- 2026-08-14：完成英文 Task 2/3 三视角动态检索、二票投票和关系 RoBERTa 重评分；
  Test 提升至 `0.616578/0.573456`，最终预测和 seed42 权重已回传，尚未发布。
- 2026-08-14：启动 Task 2/3 自动升级实验：Attention+MLP QLoRA、dense/hybrid-MMR
  差异化检索、三路候选并集、RoBERTa 边界与关系过滤以及显式/NULL 分支。两项 smoke
  已通过，正式 QLoRA 初始实测 GPU 100%、显存 27.7/32.6 GiB；结果尚未产生。
- 2026-08-14：上述候选过滤方案 Dev 平均 cF1 `0.7057282064`，未过门槛并跳过 Test。
  随后完成真正独立跨度系统；候选池召回较高，但关系判别失败，Task 2/3 Dev cF1
  仅 `0.119265/0.068578`，因此未运行 Test。
- 2026-08-14：在不训练的前提下改为保守补漏：冻结旧二票结果，只加入独立跨度池中
  高置信、短距离且去除嵌套噪声的新关系。Dev 每项仅新增 1 个 TP，Task 2/3 cF1
  小幅升至 `0.742691/0.718993`；官方脚本一致，尚未应用到 Test。
- 2026-08-14：README 改为只展示完整运行 Test 且逐步提升的英文实验；成功路线所需
  Python 文件随提交 `fc49283` 直接发布到 GitHub `main`，失败路线和本地产物未发布。
- 2026-08-14：README、`PAPER_RESULTS.md` 和 `results/evaluation_metrics.csv` 补充
  论文英文 Restaurant 前两名与官方 KimiK2、Qwen3-14B baseline 对照；本次尚未提交或发布。
- 2026-08-17：云端 30898 已验证 RTX 5090、英语 RoBERTa 权重、官方英文 Train/Dev
  数据和 55 项连续回归单元测试；启动独立 Dev-only 自动消融。当前只使用官方数据，流程
  在 Test 前结束，输出位于 `outputs/task1_continuous_dev_v1/`。
- 2026-08-17：连续回归 Dev-only 自动消融完成。最佳单次 Dev 为 L8、序数软标签、
  Ranking 0.1 与均衡采样的 `0.9251986`；三种子平均 `0.9283670`。Train 三折 OOF
  原始/校准为 `1.1723294/1.1687202`，校准改善不足 `0.01`，未采用；MoCo 按门槛跳过，
  未运行 Test。完整记录见 `TASK1_CONTINUOUS_EXPERIMENT_RECORD.md`。
- 2026-08-17：第二次改进完成 EmoBank 公开数据中间预训练。8,062 条 Train、1,000 条
  Dev，1～5 分映射到 1～9，EmoBank Dev `0.5723317`；编码器回到官方 Restaurant 微调后
  Dev `0.9211219`，相对第一次 `0.9251986` 只改善 `0.0040767`，未过 `0.01` 门槛。
  自动跳过额外种子和 Test；SemEval 与 AI 生成数据尚未进行。
- 2026-08-17：完成 SemEval-2014 Restaurant Aspect 极性中间预训练。SemEval-only 与
  EmoBank→SemEval 的内部 macro-F1 分别为 `0.6398841/0.6281779`；回到 DimABSA 后 Dev
  RMSE 分别为 `0.9175556/0.9169297`。最佳路线相对第一次 `0.9251986` 改善 `0.0082688`，
  未达到 `0.01` 门槛，自动跳过额外种子和 Test。
- 2026-08-18：完成英语情感 RoBERTa-base 0/4/8/12 层对照，Dev RMSE 分别为
  `1.1043478/1.0648936/1.0456768/1.0495504`；8 层最佳，全解冻没有提升，整体明显差于
  Large 与公开数据预训练路线，因此未运行额外种子或 Test。
- 2026-08-18：完成 Large 后 8 层的序数＋Huber、二维 VA MoCo及EmoBank/SemEval三头
  多任务实验。混合损失最佳 `0.9247019`，MoCo最佳 `0.9316378`，多任务最佳
  `0.9317464`，均未超过顺序预训练 `0.9169297`；未运行额外种子或Test。
- 2026-08-18：用户在 DeBERTa-v2-xxlarge 下载阶段撤销实验；下载与后台进程已停止，
  云端部分模型、输出和脚本及本地脚本均已删除，没有执行 Smoke、训练或 Test。
- 2026-08-18：在项目同级建立 `DimABSA_transfer_bundle_2026-08-18`，保存完整官方数据、
  EmoBank、SemEval-2014、模型固定版本清单和跨平台下载/校验脚本；另生成 16 MB ZIP。
- 2026-08-18：迁移包补入台式机交接文档、项目 `AGENTS.md`、完整 `.ai` 共享记忆、
  实验记录和当前代码快照；337 个文件与解压后的 ZIP 均通过校验。新版 ZIP 约 17 MB，
  最终 ZIP 哈希记录在包外的根项目交接中。
- 2026-08-18：新增 `TASK23_BEST_ROUTE.md`，把英文 Task 2/3 的 Qwen QLoRA、三视角
  BM25、二票结构投票和 seed42 RoBERTa VA 重评分完整路线纳入台式机交接。
- 2026-08-18：台式机首次资产核验完成。确认 Task 1 最佳路线（LogSigma RoBERTa-large
  三种子集成，Test `RMSE_VA=1.1659316485`）权重已保存于
  `outputs/logsigma_roberta_seed21/42/99`；Task 2/3 最佳权重
  `outputs/extraction_lora_seed42/` 与 `outputs/hybrid_relation_va_seed42/` 完好。
  三份迁移 ZIP 完整性通过；本机无 PyTorch，权重未做完整加载复测。
- 2026-08-18：云端 RTX 4090 D 完成 EmoBank→SemEval→DimABSA 预训练流水线。EmoBank
  预训练 Dev RMSE_VA = 0.5458；SemEval 预训练 Dev macro_f1 = 0.6596；LogSigma 三种子
  训练（使用 SemEval 编码器初始化）Dev RMSE_VA：seed 21 = 1.0224，seed 99 = 1.0340，
  seed 42 = 0.9543。三种子集成 Dev = **0.9655**（vs 历史 0.9767，改善 0.0112），
  Test = **1.1486**（vs 历史 1.1578，改善 0.0092）。预训练流水线有效，Dev 和 Test
  均优于原始 LogSigma 路线。
- 2026-08-18：云端 Task 1 进一步优化实验完成。OOF 校准失败（Test 恶化到 1.2638）；
  多种子训练成功，5 种子集成（seed 21/99/42/3407/2026）Dev = **0.9600**，
  Test = **1.1427**（vs 3 种子 1.1486，改善 0.0059）；超参数调优失败（Dev 好的配置
  Test 过拟合）。当前最佳 Test RMSE = **1.1427**，相比初始复现（1.1578）改善 **0.0151**，
  相比论文最佳（1.1035）差距 **0.0392**。
- 2026-08-18：Mean Pooling 实验完成。平均池化三种子官方 Dev = 1.1148、Test = 1.1063。
  口径提醒：官方 Dev 上平均池化差于 CLS 五种子（0.9600），Test 才反转，按 Dev 冻结
  规则属于诊断性结果；当前 Dev 冻结的正式最佳仍为 CLS 五种子（Dev 0.9600 /
  Test 1.1427）。下一步先在 Dev 上验证"平均池化 + Opinion 辅助 + VA 均衡采样 +
  5 种子"能否追平 CLS。
- 2026-08-19：Task 1 Mean pooling 可复现性调查完成。早期三种子脚本未记录 RNG 种子，
  1.1063 训练过程不可复现但权重已备份本地（MD5 校验通过）。新建带种子记录的
  `train_mean_seeded10.py`，六组 28 模型逐项评测确认 Mean 的 Test 优势系统性但
  Dev 全部差于 CLS（交叉反转）。双进程续训 runs 6-20 后，21 模型集成官方
  Dev = **1.0591** / Test = **1.1094**（可复现最佳，距论文 1.1035 仅 0.0059）。
  按 Dev 冻结规则正式结论仍为 CLS 五种子 Test 1.1427；种子清单见
  `MEAN_POOLING_21MODELS.md`。
- 2026-08-20：已整理另一台机器的交接包。README 直接附上完整连续实验记录；Task 1
  正式 Dev 冻结结果为 1.1427，21 模型 mean 的 1.1094 仅作可复现 Test 诊断；Task 2/3
  最新本地 Test 记录为 Kimi K2.7 + 关系 RoBERTa 的 0.6420/0.5858。
