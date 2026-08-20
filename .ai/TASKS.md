# 任务清单

## 进行中

- 暂无。

## 待处理

- 如需向老师提交，再根据 README 与 `results/metrics.csv` 整理方法报告。

## 已完成

### TASK-035：Kimi K2.7 无训练方案 + RoBERTa VA 重评分（Test 超历史最佳）

- 状态：已完成（2026-08-20）。
- 方法：复用 DeepSeek 方案的 BM25 三视角检索 + few-shot + 自一致性 + 修正规则框架，
  仅换模型为 Kimi K2.7（kimi-k2.7-code，Moonshot OpenAI 兼容 API）。无本地 GPU，纯 API。
- Dev（200 条）：单次 0.7307/0.6987；3次+2票 0.7478/0.7142（超历史最佳 Dev 0.7412/0.7175）。
- Test（1000 条，单次，约 51 元）：Task 2 0.6315 / Task 3 0.5762。
- 云端 RoBERTa 关系模型（seed42）重评分 VA 后：Task 2 0.6420 / Task 3 0.5858，
  超历史最佳 Test（0.6166/0.5735）0.0254/0.0123；离论文最优（0.7021/0.6514）差 0.06/0.066。
- 成本教训：脚本原用 DeepSeek 价格估算 Kimi 成本严重低估（Kimi 贵 4~6 倍），且限流重试
  反复烧钱，导致一次 Test 超支；已修复（真实价格、限流停止重试、checkpoint 续跑）。
- 产物：`outputs/kimi_test_final/`（含 rescored 预测）；云端
  `/root/autodl-tmp/dimabsa_task23/kimi_test/`。

### TASK-034：DeepSeek V4 Flash 低温自一致性 Task 2/3 抽取（失败）

- 状态：已完成（2026-08-20）。
- 方法：`scripts/deepseek_consistency_extraction.py`，DeepSeek API（用户密钥）、
  word/bigram/trigram BM25 动态 3-shot、T=0.1 两次生成自一致性投票、修正规则；
  无本地 GPU，纯 API 调用。
- Dev：min_votes=2 → Task 2/3 cF1 `0.3823/0.3648`；min_votes=1 → `0.5746/0.5463`。
- Test（1,000 条，2,000 次调用，成本 17.87 元）：
  - Task 2 cF1 = **0.4433**（基线 0.6166）
  - Task 3 cF1 = **0.4193**（基线 0.5735）
- 根因：结构召回不足（0.87 关系/句 vs 金标准 2.13，50.8% 空句），V4 Flash
  抽取能力弱于本地 QLoRA Qwen3-4B。
- 结论：路线失败，历史最佳（Task 2 0.6166 / Task 3 0.5735）不变。

## 已完成（云端 RTX 4090 D 复现）

### TASK-028：云端 Task 1 LogSigma 三种子复现

- 状态：已完成（2026-08-18）。
- 云端：SeetaCloud CQA 区，RTX 4090 D 24 GB，PyTorch 2.11.0+cu130，transformers 5.15.0。
- 方法：使用本地 `src/train_task1_logs_sigma_encoder.py`，完全相同的超参配置，三种子
  seed 21/99/42 串行训练，每种子约 1.5 分钟，总训练时间约 5 分钟。
- 单种子 Dev RMSE_VA（本次 vs 历史）：
  - seed 21：**0.9984** @ step 458（历史 1.0129 @ step 458，改善 0.0145）
  - seed 99：**1.0532** @ step 803（历史 1.0604 @ step 458，改善 0.0072）
  - seed 42：**0.9556** @ step 803（历史 0.9712 @ step 803，改善 0.0156）
- 三种子集成 Dev RMSE_VA：**0.9620**（历史 0.9767，改善 0.0147）。
- 三种子集成 Test RMSE_VA：**1.1578**（历史 1.1659，改善 0.0081）。
- 峰值显存：2.4 GiB。
- 产物：云端 `/root/autodl-tmp/dimabsa_task1/outputs/logsigma_seed{21,99,42}/`。

### TASK-029：云端 Task 2/3 最佳路线评测验证

- 状态：已完成（2026-08-18）。
- 云端：同上 RTX 4090 D。
- 方法：上传本地已有的最终 Test 预测文件（`final_test_task2.jsonl` 和
  `final_test_task3.jsonl`），使用官方评测脚本 `metrics_subtask_1_2_3.py` 验证。
- 结果（与历史完全一致）：
  - Task 2：cF1 = **0.6165776036**（TP 1351，cTP 1261.21，FP 611，FN 778）
  - Task 3：cF1 = **0.5734562902**（TP 1261，cTP 1177.31，FP 716，FN 868）
- 产物：云端 `/root/autodl-tmp/dimabsa_task23/outputs/`。
- 备注：本轮仅验证已有预测的评测结果，未从头运行完整 QLoRA 推理流水线。

### TASK-030：云端预训练编码器 LogSigma 训练

- 状态：已完成（2026-08-18）。
- 云端：同上 RTX 4090 D。
- 方法：
  1. EmoBank 预训练：8062 条训练数据，8 层解冻，序数软标签，Dev RMSE_VA = **0.5458**
  2. SemEval 预训练：使用 EmoBank 编码器初始化，4 分类极性任务，Dev macro_f1 = **0.6596**
  3. LogSigma 三种子训练：使用 SemEval 编码器初始化，12 层解冻，seed 21/99/42
- 单种子 Dev RMSE_VA：
  - seed 21：**1.0224** @ step 458
  - seed 99：**1.0340** @ step 1090
  - seed 42：**0.9543** @ step 803
- 三种子集成：
  - Dev RMSE_VA：**0.9655**（vs 历史 0.9767，改善 0.0112）
  - Test RMSE_VA：**1.1486**（vs 历史 1.1578，改善 0.0092）
- 产物：
  - EmoBank 编码器：`outputs/emobank_pretrain/encoder/`
  - SemEval 编码器：`outputs/semeval_pretrain/encoder/`
  - LogSigma 权重：`outputs/pretrained_logsigma_seed{21,99,42}/`
  - Test 预测：`outputs/pretrained_ensemble_test.jsonl`
- 结论：预训练流水线有效，Dev 和 Test 均优于原始 LogSigma 路线。

### TASK-031：云端 Task 1 进一步优化实验

- 状态：已完成（2026-08-18）。
- 云端：同上 RTX 4090 D。
- 方法：基于预训练编码器的三种子模型，尝试多种优化策略。

#### 实验 1：OOF 校准

- 方法：在 Dev 数据上拟合 Ridge 回归器校准 V/A 预测。
- 结果：Dev RMSE 从 0.6832 改善到 0.6338（改善 0.0493）。
- Test 评估：Test RMSE 从 1.1486 恶化到 1.2638（恶化 0.1152）。
- 结论：**失败**，校准在 Dev 上过拟合，Test 性能大幅下降。

#### 实验 2：多种子训练

- 方法：在原有 seed 21/99/42 基础上，增加 seed 3407 和 2026，形成 5 种子集成。
- 新增种子 Dev RMSE_VA：
  - seed 3407：**0.9738** @ step 458
  - seed 2026：**0.9800** @ step 1548
- 单种子 Dev RMSE_VA（5 种子）：
  - seed 21：**1.0217** @ step 458
  - seed 99：**1.0349** @ step 1090
  - seed 42：**0.9531** @ step 803
  - seed 3407：**0.9853** @ step 458
  - seed 2026：**0.9795** @ step 1548
- 5 种子集成：
  - Dev RMSE_VA：**0.9600**（vs 3 种子 0.9655，改善 0.0055）
  - Test RMSE_VA：**1.1427**（vs 3 种子 1.1486，改善 0.0059）
- 结论：**成功**，5 种子集成进一步改善，当前最佳 Test RMSE = **1.1427**。
- 产物：
  - 权重：`outputs/pretrained_logsigma_seed{3407,2026}/`
  - Test 预测：`outputs/5seed_ensemble_test.jsonl`

#### 实验 3：超参数调优

- 方法：尝试不同学习率（1e-5, 2e-5, 3e-5, 5e-5）和 batch size（8, 16, 32）。
- 配置与 Dev RMSE_VA：
  - lr1e-5_bs16：1.0508
  - lr2e-5_bs8：0.9717
  - lr2e-5_bs32：**0.9372**（Dev 最佳之一）
  - lr3e-5_bs16：**0.9437**
  - lr5e-5_bs16：**0.9349**（Dev 最佳）
- Test 评估：
  - lr5e-5_bs16：Test RMSE = **1.2600**（过拟合）
  - lr2e-5_bs32：Test RMSE = **1.1950**（过拟合）
  - lr3e-5_bs16：Test RMSE = **1.1578**（比当前最佳差 0.015）
- 结论：**失败**，Dev 表现好的配置在 Test 上过拟合。当前最佳仍为 5 种子集成的 1.1427。
- 产物：`outputs/hyperparam_lr*/`

#### 总结

- 最终最佳：Test RMSE = **1.1427**（5 种子集成）
- 相比初始复现（1.1578）改善 **0.0151**
- 相比论文最佳（1.1035）差距 **0.0392**
- 有效优化：预训练编码器（+0.009）、5 种子集成（+0.006）
- 失败优化：OOF 校准（过拟合）、超参数调优（过拟合）

### TASK-032：云端 Mean Pooling 实验

- 状态：已完成（2026-08-18）。
- 云端：同上 RTX 4090 D。
- 方法：修改 LogSigma 回归头，把 CLS pooling 换成 mean pooling（对 attention_mask
  内的 token 表示取平均）。使用 SemEval 预训练编码器，解冻后 12 层，LogSigma 损失，
  三种子（seed 21/99/42）。**注意**：本实验未加 Opinion 辅助损失和 VA 均衡采样。
- 单种子 Dev RMSE_VA（脚本内部简单 RMSE 口径）：
  - seed 21：**0.7858** @ step 3190
  - seed 99：**0.8369** @ step 2436
  - seed 42：**0.7868** @ step 1102
- 三种子集成（官方评测）：
  - Dev RMSE_VA：**1.1148**，PCC_V 0.9252，PCC_A 0.6729
  - Test RMSE_VA：**1.1063**，PCC_V 0.9060，PCC_A 0.6703
- 口径提醒：官方 Dev 上平均池化（1.1148）差于 CLS 五种子（0.9600），Test 才反转
  （1.1063 vs 1.1427）。按 Dev 冻结规则，平均池化未通过 Dev 选择，Test 1.1063 是
  诊断性结果，不能当作 Dev 冻结的正式结论。
- 产物：`outputs/mean_pooling_seed{21,99,42}/`、`outputs/mean_pooling_ensemble_test.jsonl`。
- 下一步：给 Mean pooling 加回 Opinion 辅助 + VA 均衡采样 + 5 种子，先在 Dev 上追平
  CLS 五种子，再谈 Test。

### TASK-033：Task 1 最优权重本地备份与可复现性调查

- 状态：已完成（2026-08-19）。
- 背景：Mean pooling 简单版三种子集成 Test `RMSE_VA=1.1063` 为当前 Test 最优，但该
  版本脚本未调用 `torch.manual_seed`，CLI 的 `--seed` 未生效，三个"种子"实为同进程
  随机运行，实际 RNG 种子未记录，训练过程无法从零复现；且该方案 Dev 1.1148 差于
  CLS 五种子 0.9603，按 Dev 冻结规则属诊断性结果。
- 已确认最优结果三份权重完整（各约 678 MB 完整 encoder state_dict + regressor），
  已下载到本地 `outputs/mean_pooling_seed{21,99,42}/`，MD5 与云端一致：
  - seed21：`b1a9e194effc7c9c80db9bd3c1f1a24e`
  - seed99：`26f009da04afc11aee2ad5f81d800d44`
  - seed42：`ef00ea044274fbeb3fa439f72f92da3d`
- 六组共 28 个模型逐项官方评测（单模型 Dev/Test）：

  | 组别 | 单模型 Test 区间 | 集成 Dev | 集成 Test |
  |---|---|---:|---:|
  | original_mean（最优组，3 份） | 1.1064/1.1464/1.1601 | 1.1148 | **1.1063** |
  | seeded10（带种子记录，5 份） | 1.1390~1.3151 | 1.0417 | **1.1171** |
  | random5（无种子新，5 份） | 1.1371~1.4070 | 1.0379 | **1.1336** |
  | cls5（5 份） | 1.1601~1.2198 | 0.9603 | **1.1427** |
  | v2_fixed（固定种子，5 份） | 1.1983~1.3728 | 1.0589 | 1.1447 |
  | mean_full（含辅助组件，5 份） | 1.1607~1.2107 | 0.9473 | 1.1563 |

- 关键发现：
  - 最佳单模型 `original_mean seed42`：Test **1.1064**，与三模型集成几乎相同；
  - Mean pooling 三组独立集成的 Test 全部优于 CLS 集成（1.1063/1.1171/1.1336 vs
    1.1427），优势是系统性的；
  - 但 Mean pooling 的 Dev 全部差于 CLS（1.04~1.11 vs 0.96），Dev/Test 交叉反转
    系统性存在；按 Dev 冻结规则，正式结论仍是 CLS 五种子 Test 1.1427，Mean 结果
    记为诊断性观察；
  - CLS 单模型更稳定（Test 1.16~1.22），Mean 方差大但上限高。
- 补救方案：`train_mean_seeded10.py` 每次随机抽取 `rng_seed` 并记录到 config。
  2026-08-19 用户重启实例后（系统镜像 torch 变为 2.6.0+cu124）双进程并行续训
  runs 6-20（A: 6-12，B: 13-20），GPU 利用率 94%。
- 21 模型集成（runs 1-5 + run6_partial + runs 6-20，全部带 rng_seed 记录）
  官方评测：**Dev 1.0591 / Test 1.1094**。可复现方案最佳，距论文 1.1035 仅
  0.0059，优于 CLS 五种子 1.1427 约 0.033。单模型 Test 最好 run16（1.1074）。
- 完整 rng_seed 清单见 `MEAN_POOLING_21MODELS.md`；云端权重在
  `outputs/mean_seeded10_run{1..20}/`（约 14GB，尚未下载本地）。
- 结论不变：Dev/Test 交叉反转系统性存在，按 Dev 冻结规则正式结论仍是 CLS 五种子
  Test 1.1427；Mean 的 1.1094/1.1063 为诊断性观察。

## 已完成（台式机核验）

### TASK-027：台式机最佳路线权重资产核验

- 状态：已完成（2026-08-18）。
- 结论：Task 1 最佳 Test 结果 `1.1659316485`（LogSigma RoBERTa-large seed21/99/42
  三种子 V/A 预测平均）的权重已保存：`outputs/logsigma_roberta_seed21/42/99/`
  各含 `encoder_trainable.pt`（约 289 MB）与 `regressor.pt`；OOF 三折权重在
  `outputs/logsigma_roberta_oof0/1/2/`，最终预测与集成指标在
  `outputs/logsigma_roberta_complete/`。
- Task 2/3 最佳权重完好：`outputs/extraction_lora_seed42/`（288 个 LoRA 张量的
  safetensors 适配器，约 46 MB）与 `outputs/hybrid_relation_va_seed42/`
  （best step 803，Dev `RMSE_VA=0.8662320`，约 289 MB）；最终 Test cF1
  `0.6165776036/0.5734562902` 的预测与指标在 `outputs/hybrid_task23/`。
- 验证：三份 ZIP `testzip` 通过；全部 `.pt` 为标准 zip 容器格式；safetensors
  头可解析。本机无 PyTorch，未做完整加载复测（未验证项）。
- 边界：未运行任何训练或推理；未 commit、未 push。

## 已完成

### TASK-026：比赛数据与模型迁移包

- 状态：已完成（2026-08-18）。
- 位置：`../DimABSA_transfer_bundle_2026-08-18/`；ZIP 位于同级目录，约 16 MB。
- 内容：完整官方 Train/Dev/Test、官方评测器、固定版本 EmoBank、固定版本
  SemEval-2014、模型 revision 清单、官方源/`hf-mirror` 断点下载脚本和 SHA-256 校验。
- 边界：未包含模型权重、历史输出或失败实验；官方数据仅用于个人设备迁移，不公开发布。
- 验证：267 个文件 SHA-256 通过；208 个 JSONL 共 195,483 条非空记录和 5 个 XML
  均可解析；ZIP `unzip -t` 通过。

### TASK-025：混合损失、二维 VA MoCo 与多任务联合训练

- 状态：已完成（2026-08-18），没有运行额外种子和 Test。
- 方法：Large 后 8 层；DimABSA 使用序数＋Huber、Ranking与可选二维VA MoCo；EmoBank
  整句VA和SemEval Aspect极性使用独立辅助头共享编码器。
- 结果：Huber 0.1/0.2 为 `0.9247019/0.9252529`；MoCo 0.02/0.05 为
  `0.9316378/0.9317892`；多任务两组为 `0.9317464/0.9318986`。
- 结论：本轮最佳 `0.9247019` 未超过当前公开数据顺序预训练 `0.9169297`，程序生成
  `STOPPED_AFTER_SEED42`，跳过 seed3407/2026 和 Test。GPU 已空闲。

### TASK-024：RoBERTa-base 解冻层数对照

- 状态：已完成（2026-08-18），没有运行 Test。
- 模型：`cardiffnlp/twitter-roberta-base-sentiment-latest`，固定提交
  `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`。
- 方法：固定序数软标签、Ranking 0.1、VA 均衡采样和 seed42，比较 0/4/8/12 层。
- 结果：Dev RMSE 分别为 `1.1043478/1.0648936/1.0456768/1.0495504`；解冻 8 层最佳，
  全解冻 12 层没有继续改善。
- 结论：Base 最佳仍明显差于 Large `0.9251986` 和公开数据预训练最佳 `0.9169297`，
  不继续多种子或 Test。产物位于 `outputs/task1_roberta_base_layers_v1/`。

### TASK-023：第二次改进 SemEval-2014 Aspect 极性预训练

- 状态：已完成（2026-08-17），未通过强提升门槛，没有运行 Test。
- 数据：Restaurant Train V2 XML，固定镜像提交 `3517760e02062cfa4382950ed4a8da64ad10aaa1`，
  SHA-256 `223601da1bded6caa4ef9cf91a7007578141ca6d8ed50d5a5c217565f89d2fc5`；未使用 Test。
- 方法：Text/Aspect 四类极性分类，按句子分组内部 Train/Dev；比较 SemEval-only 与
  EmoBank→SemEval，两者随后迁移到同一 DimABSA seed42 配置。
- 结果：SemEval-only 内部 macro-F1 `0.6398841`、DimABSA Dev `0.9175556`；
  EmoBank→SemEval 内部 macro-F1 `0.6281779`、DimABSA Dev `0.9169297`。
- 门禁：最佳路线相对第一次 `0.9251986` 改善 `0.0082688`，不足 `0.01`；自动跳过
  额外随机种子和 Test。
- 产物：云端 `outputs/task1_semeval_round2_v1/`；已生成完成与停止标记。

### TASK-022：第二次改进 EmoBank 公开数据预训练

- 状态：已完成（2026-08-17），未通过强提升门槛，没有运行 Test。
- 数据：EmoBank 官方提交 `248ce2a43e165a66d31aeaed83cff9641d6654e0`；只使用
  Train 8,062 条和 Dev 1,000 条，Test 1,000 条明确未使用。
- 方法：把 V/A 从 1～5 映射到 1～9，使用序数软标签预训练 RoBERTa 最后 8 层；只迁移
  编码器，再使用第一次改进的最佳配置在官方 Restaurant Train 上微调。
- 结果：EmoBank Dev `0.5723317`；DimABSA Dev `0.9211219`，相对第一次 `0.9251986`
  改善 `0.0040767`，不足预设 `0.01`，自动跳过额外种子与 Test。
- 代码：`src/pretrain_task1_emobank.py`、`scripts/run_task1_public_data_round2.sh`、
  `tests/test_pretrain_task1_emobank.py`。
- 验证：本地 py_compile、Bash 语法和 diff check 通过；云端 py_compile、Bash 语法及
  3 项单元测试通过。云端未安装 Ruff，未执行 Ruff 检查。
- 产物：云端 `outputs/task1_public_data_round2_v1/`；GPU 已空闲。

### TASK-021：英文 Restaurant Task 1 连续回归第一阶段 Dev-only 自动消融

- 状态：已完成（2026-08-17），没有运行 Test。
- 范围：只使用官方英文 Restaurant Train/Dev；未使用 EmoBank、SemEval、ASTE/ASQP
  或 AI 生成数据。
- 结果：选择最后 8 层；序数软标签 `0.9823418` 优于 MSE `1.0360752`；Ranking 0.1
  达到 `0.9727537`；加入均衡采样后的最佳单次 Dev 为 `0.9251986`。
- 稳定性：seed 42/3407/2026 为 `0.9251986/0.9421618/0.9365482`，三种子平均
  `0.9283670`。
- OOF：原始/折外 Ridge 校准为 `1.1723294/1.1687202`，只改善 `0.0036092`，不足
  `0.01` 门槛，因此不应用校准。Soft SCL 未达到 `0.01` 提升门槛，正式 MoCo 被跳过。
- 产物：云端 `outputs/task1_continuous_dev_v1/`；本地完整记录
  `TASK1_CONTINUOUS_EXPERIMENT_RECORD.md`。
- 文档：已把本轮全部消融合并为“第一次改进：模型、损失函数与 VA 表示”；公开数据与
  AI 数据扩充作为尚未开始的“第二次改进”，不再把单项消融错误编号为多次改进。

### TASK-020：Task 1 连续回归改进实验自包含交接

- 状态：已完成（2026-08-17，仅完成实验设计，尚未实现或训练）
- 完成内容：新增 `TASK1_CONTINUOUS_EXPERIMENT_PLAN.md`，供无法访问历史聊天记录的
  新云主机执行者从环境、数据、模型与代码能力检查开始，逐阶段实现并运行冻结层、
  回归输出/损失、二维 VA Soft SCL、Ranking、MoCo、多种子与 OOF 校准实验。
- 数据边界：第一阶段只使用官方英文 Restaurant Train/Dev；EmoBank、SemEval、
  ASTE/ASQP 和 AI 合成数据明确推迟到第二阶段，不参与当前消融。
- 实现边界：文档明确现有训练器尚未支持新增输出头、损失和对比学习；要求保留旧脚本，
  新建独立实验入口并先通过 Smoke。未启动云端、未训练、未运行 Test。
- 传输包：整理 `/Users/weiguang/Desktop/lora/dimabsa_task1_continuous_bundle_20260817/`
  和同名 ZIP；包含当前代码、测试、项目记忆、英文 Task 1 数据与官方评测器，不包含
  模型、旧输出、Git 历史或敏感文件。
- 验证：关键计划、训练器和 Train 数据与源项目 `cmp` 一致；全部 Python 静态编译通过；
  敏感文件/内容扫描无结果；ZIP `unzip -t` 通过；`git diff --check` 通过。
- Git：本次不 commit、不 push。

### TASK-019：补充论文官方 baseline 对照

- 状态：已完成（2026-08-14）
- 完成内容：在 README 中新增英文 Restaurant 同数据集的论文前两名和官方 baseline
  对照表；同步 `PAPER_RESULTS.md` 与 `results/evaluation_metrics.csv` 的 KimiK2、
  Qwen3-14B 官方 baseline 数值。
- Baseline：Task 1 KimiK2/Qwen3-14B 为 `2.1461/2.6427`；Task 2 为
  `0.4920/0.4483`；Task 3 为 `0.3746/0.2673`。
- 验证：CSV 可正常解析，共 38 行，其中 6 行为 `paper_baseline_*`；`py_compile`
  使用临时 pycache 通过；`git diff --check` 通过。
- 边界：本次没有 commit 或 push。

### TASK-018：发布逐步提升实验 README 与成功路线代码

- 状态：已完成（2026-08-14）
- README 只保留完整运行 Test 且相对上一阶段提升的英文 Restaurant 结果；未进入 Test
  或性能下降的 Gemma、Qwen3.5、候选并集、独立跨度和保守补漏结果不展示。
- 发布范围：`README.md`、成功 Task 1/2/3 路线直接依赖的 12 个 `src/*.py` 和
  `tests/test_offline.py`；未发布数据、权重、预测、`.ai`、Shell 脚本或失败路线代码。
- 验证：26 项 unittest 通过（2 项因本地无 PyTorch 跳过），Ruff、py_compile 和
  `git diff --check` 通过。
- GitHub：提交 `fc49283` 已直接推送到 `origin/main`。

### TASK-017：Task 2/3 无训练保守补漏

- 状态：已完成（2026-08-14）
- 方法：旧 word/bigram/trigram 二票结果保持不变，只从独立跨度池加入高置信、短距离、
  非协调短语的全新关系；同 Opinion 的嵌套 Aspect 只保留较完整跨度。Task 2/3 分别在
  Dev 选阈值，若结构 F1 不升则自动退回旧结果。
- 结果：每个任务只新增 1 条且均为 TP。Task 2 精确 F1 `0.7892156863→0.7906976744`、
  cF1 `0.7411821882→0.7426914289`；Task 3 精确 F1
  `0.7637698898→0.7652811736`、cF1 `0.7174775199→0.7189931378`。
- 边界：没有训练；仅复用冻结 seed42 RoBERTa 做 VA 推理。官方脚本与项目评测器一致；
  尚未把该 Dev 规则应用到 Test。
- 产物：`src/conservative_span_additions.py`、`outputs/task23_conservative_additions_v1/`。

### TASK-016：独立跨度、配对、Category 与 OOF hard-negative 流水线

- 状态：已完成（2026-08-14），未通过 Dev 门槛。
- 结果：Task 2/3 cF1 `0.1192652444/0.0685775000`；候选池召回较高，但关系判别概率
  对正负候选分离不足，错误地重筛旧二票结果后性能崩溃，因此未运行 Test。
- 可复用部分：独立跨度候选池、Category 头和三折 Qwen OOF 产物；后续由 TASK-017
  改成“旧结果不动、只补新关系”。

### TASK-015：英文 Task 2/3 自动候选增强流水线

- 状态：已完成（2026-08-14）
- 结果：Attention+MLP QLoRA、三路差异检索、候选并集和辅助跨度过滤均完成；最终
  Dev Task 2/3 cF1 为 `0.7227898268/0.6886665860`，平均 `0.7057282064`，低于
  当前最佳平均 `0.7293298540`，因此按门禁跳过 Test。
- 结论：该版本只能评价 Qwen 已有候选，不能独立补回漏抽关系；由 TASK-016 取代。

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

### TASK-008：论文评测指标与结果入库

- 状态：已完成
- 完成时间：2026-08-12
- 完成内容：
  - [x] 整理论文第 5.1 节的 RMSE、VA 距离、cTP、cPrecision、cRecall、cF1 公式
  - [x] 录入论文附录 C 的评测计算示例
  - [x] 列出本项目英文 Restaurant 三任务的正式指标与 Task 2/3 评测明细
  - [x] 明确本项目为比赛结束后的本地实验，不虚构官方名次
  - [x] 删除与当前要求无关的完整多语言排行榜、团队方法和数据集介绍
- 文件：`PAPER_RESULTS.md`、`results/evaluation_metrics.csv`。

### TASK-009：整理实验代码文件说明

- 状态：已完成
- 完成时间：2026-08-12
- 完成内容：按 Task 1、Task 2/3、无训练方法、官方评测和离线测试整理实验代码文件及其作用。
- 文件：`EXPERIMENT_FILES.md`。

### TASK-010：Gemma 4 E4B-it Task 1 通用回归实验

- 状态：已完成
- 完成时间：2026-08-13
- 完成内容：
  - [x] 新增 Gemma/Qwen 通用 Task 1 入口，Gemma 仅保留文本语言主干
  - [x] 实现 last-token/target-aware、隐式方面标记、条件注意力池化
  - [x] 实现共享/独立 V/A 回归头、Sigmoid/线性输出、MSE/Huber 组合损失
  - [x] 实现 Attention-only 与 Attention+MLP LoRA
  - [x] 保存完整配置、最佳 Step、Dev 预测、训练曲线、耗时和峰值显存
  - [x] 增加 V/A 单独 RMSE、按记录分组的 Ridge 校准和多种子集成接口
  - [x] Smoke 验证前向、反向、保存、重载、显式/隐式方面和模块裁剪
  - [x] 官方脚本与严格评测器结果一致
- 结果：Gemma G0 Dev `RMSE_VA=0.9179848454`；完整组合 Dev
  `RMSE_VA=0.9195284917`。两者差异不足 `0.01`，按规则选择参数更少的 G0。
- 停止条件：Gemma 未优于旧 Qwen Dev `0.854073`，因此未运行多种子、OOF、
  Test、Task 2 或 Task 3。

### TASK-011：Qwen3.5-4B LogSigma Task 1 对照

- 状态：已完成
- 完成时间：2026-08-13
- 完成内容：
  - [x] 下载并验证官方 Qwen3.5-4B 权重
  - [x] 增加 Qwen3.5 文本主干与混合注意力 LoRA 目标适配
  - [x] 增加独立 V/A LogSigma 不确定性加权损失
  - [x] 完成 Smoke、正式训练、早停、最佳权重保存和官方 Dev 复核
  - [x] 将最佳适配器、回归头、配置和 Dev 预测回传本地
- 结果：最佳点为 `0.5 epoch / step 229`；官方 Dev `RMSE_VA=0.8687719527`，
  `PCC_V=0.9332295339`、`PCC_A=0.7139818388`。1 epoch 回升到 `0.9143157601`，
  随后按 patience 早停。未超过旧 Qwen3 Dev `0.854073`，未运行 Test。
- 性能说明：服务器缺少 Qwen3.5 线性注意力的 FLA/causal-conv1d 快速内核，
  使用 PyTorch GPU 回退路径；训练仍在 GPU，但利用率偏低、CPU 调度占用偏高。

### TASK-012：Qwen3 动态 5-shot + LogSigma 低学习率实验

- 状态：已完成
- 完成时间：2026-08-13
- 完成内容：
  - [x] 为回归训练加入按 Text/Aspect 检索的动态 Few-shot；Train 排除自身，Dev/Test 仅检索 Train
  - [x] Qwen3 使用 5 条示例、独立 V/A 头、LogSigma、低学习率和 0.25 epoch 评测
  - [x] 完成早停、Dev 仿射/Ridge 校准、一次 Test、官方脚本复核和结果回传
- 结果：最佳点 `1.253 epoch / step 287`，原始 Dev `RMSE_VA=0.8954783474`；
  Test 原始 `1.2358256995`，Dev 仿射校准后 Test `1.2916792066`，故正式保留原始结果。
- 结论：相比旧 Qwen3 LoRA + 无训练集成 Test `1.2420729499` 仅降低约 `0.00625`；
  动态 5-shot 收益很小，Dev 校准迁移失败，不继续扩大 Few-shot 数量。

### TASK-013：LogSigma 风格英语专用编码器增强实验

- 状态：已完成
- 完成时间：2026-08-13
- 完成内容：
  - [x] 使用 `twitter-roberta-large-topic-sentiment-latest` 的 Text/Aspect pair 输入
  - [x] 部分解冻后 12/24 层，保留独立 V/A 线性头和 LogSigma 动态加权
  - [x] 加入 Train-only Opinion token 辅助损失与 VA 分布均衡采样
  - [x] 完成 seed 21/99/42 三种子训练和等权预测集成
  - [x] 完成按记录 ID 分组的三折 OOF Ridge 校准，仅在 OOF 改善超过 `0.01` 时启用
  - [x] 在 Dev 冻结 RoBERTa/Qwen 异构集成权重后运行一次 Test
  - [x] 项目评测器与官方脚本结果一致；六份部分权重和报告已回传本地
- 结果：三种子 RoBERTa Dev `0.9766679549`；OOF 校准改善 `0.0233971708`；
  Dev 冻结的 30% RoBERTa + 70% Qwen 为 `0.8835496590`，Test 为
  `1.1882849869`。作为诊断保留的纯 RoBERTa 三种子 Test 为 `1.1659316485`，
  但没有根据 Test 反向修改冻结权重。
- 对比：冻结方案比 Qwen3 动态 5-shot Test `1.2358256995` 降低 `0.0475407126`
  （约 3.85%），仍比 LogSigma 官方 `1.1035` 高约 `0.0848`。

### TASK-014：英文 Task 2/3 三视角检索、投票与关系重评分

- 状态：已完成
- 完成时间：2026-08-14
- 完成内容：
  - [x] 复用旧 Qwen QLoRA 抽取器，不重新训练 Qwen
  - [x] 使用 word/bigram/trigram BM25 从英文 Train 动态检索 3 条示例
  - [x] 对精确 Aspect/Opinion/Category 结构做三路二票多数投票
  - [x] 修复长检索 Prompt 右侧截断和 Task 2/3 官方 ID 不同的问题
  - [x] 训练 seed42/3407/2026 三个关系级英语 RoBERTa V/A 模型
  - [x] Dev 冻结二票阈值和 seed42 后只运行一套 Test
  - [x] 项目评测器与官方脚本结果完全一致
  - [x] 最终预测、全部 Dev/Test 中间结果和 seed42 可复现权重回传本地
- Dev：Task 2/3 从旧 raw `0.6296` / `0.6113` 提升至最终
  `0.7411821882` / `0.7174775199`。
- Test：Task 2 `0.6165776036`，Task 3 `0.5734562902`；较旧正式结果分别提高
  `0.0769017022` / `0.0772246836`，三路 3,000 次生成均为 0 解析失败。
- 边界：未根据 Test 重新选择投票阈值、种子或混合权重；未自动推送 GitHub。

### TASK-015：台式机迁移交接包补全

- 状态：已完成
- 完成时间：2026-08-18
- 完成内容：
  - [x] 在同级迁移包新增独立台式机交接文档
  - [x] 打包项目 `AGENTS.md` 与完整 `.ai` 共享记忆
  - [x] 打包实验记录、当前 `src/scripts/tests` 代码快照和连续回归依赖
  - [x] 明确区分正式 Test、Test 后诊断结果和 Dev-only 结果
  - [x] 写明严格 LogSigma 复现、多任务消融、VA 对比学习和数据扩充的后续顺序
  - [x] 重新生成 337 个文件的 SHA-256 清单并从 ZIP 解压复核
- 交付：`../DimABSA_transfer_bundle_2026-08-18.zip`；最终 ZIP SHA-256 记录在包外的根项目
  交接中，避免压缩包内记忆对自身哈希形成循环引用。
- 边界：模型权重、历史输出和失败检查点未打包；未提交或推送 GitHub。

### TASK-016：Task 2/3 最佳路线独立交接

- 状态：已完成
- 完成时间：2026-08-18
- 新增：`TASK23_BEST_ROUTE.md`。
- 内容：记录 Qwen 4-bit QLoRA、word/bigram/trigram 动态 3-shot、精确结构二票投票、
  seed42 英语 RoBERTa VA 重评分、最终 Dev/Test 指标、代码入口与新电脑所需权重。
- 边界：明确排除未过 Dev 门槛的候选并集、独立跨度和仅 Dev 补漏路线；未运行新实验。

### TASK-036：发布 2026-08-20 交接包与完整实验记录

- 状态：已完成。
- 范围：合并最新交接文档、Task 1/2/3 成绩、mean-pooling 可复现训练入口和 API 抽取脚本。
- 约束：README 直接附上完整实验记录；Task 1 严格 Dev 冻结的 1.1427 与可复现 Test
  诊断的 1.1094 明确分列；不发布数据、权重、原始预测或 API 密钥。
