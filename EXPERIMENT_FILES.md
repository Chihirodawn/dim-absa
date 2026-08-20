# 实验相关代码文件说明

本文档整理本项目实验使用的主要代码文件。代码位于项目根目录的 `src/` 文件夹中。

## 1. Task 1：VA 程度预测

### 正式 LoRA 训练入口

#### `src/train_task1_lora_regression.py`

Task 1 最主要的训练文件，负责：

- 加载 `Qwen3-4B-Instruct-2507`；
- 给 Qwen 添加 LoRA；
- 添加同时输出 Valence 和 Arousal 的双输出回归头；
- 使用英文 Restaurant Train 进行训练；
- 在 Dev 上评测、早停并保存最佳检查点；
- 加载最佳检查点生成 Task 1 预测结果。

#### `src/train_task1_gemma_regression.py`

Gemma 4 E4B-it 改进实验入口，同时兼容 Qwen，负责：

- 只提取 Gemma 的文本语言主干，不把视觉和音频模块放入 GPU；
- 支持最后 Token 和目标方面感知表示；
- 将 `NULL` 方面转换为专门的 `[IMPLICIT_ASPECT]` 标记；
- 支持 Aspect 条件注意力池化、共享头和独立 V/A MLP 头；
- 支持 Sigmoid 输出、线性输出后裁剪、MSE 和 Huber 组合损失；
- 支持 Attention-only 和 Attention+MLP LoRA；
- 每半个 Epoch 评测，保存最佳 Step、配置、曲线、Dev 预测、显存和耗时；
- Smoke 模式实际验证保存后重新加载。

本轮 Dev 结果：简单 G0 为 `0.9179848454`，全部改进组合为
`0.9195284917`。两者差异不足 `0.01`，选择参数更少的 G0；由于没有超过
旧 Qwen Dev `0.854073`，没有继续评测 Test。

### Task 1 校准、集成与评测

#### `src/calibrate_task1.py`

使用 Dev 分别拟合 V、A 的校准参数，再把冻结参数应用到 Test。既支持普通仿射，
也支持按记录 ID 分组交叉验证选择强度的 Ridge：

```text
V_new = slope_V × V_raw + intercept_V
A_new = slope_A × A_raw + intercept_A
```

#### `src/ensemble_task1.py`

把以下四种无训练预测等权平均，再进行 Dev 校准：

- Direct；
- CoT；
- 固定 Few-shot；
- Dynamic Few-shot。

每一路占四路集成结果的 `25%`。

#### `src/blend_task1.py`

把校准后的 LoRA 结果与无训练四路集成结果再次加权：

```text
最终预测 = 90% × LoRA + 10% × 无训练四路集成
```

权重作用于每个 Aspect 的 V、A 预测值，不是对最终 RMSE 做加权。

#### `src/evaluate_task1.py`

Task 1 本地严格评测文件，计算：

- `RMSE_VA`；
- `RMSE_V`、`RMSE_A`；
- `RMSE_VA_NORMALIZED`；
- `PCC_V`；
- `PCC_A`；
- 记录数、Aspect 数和预测覆盖率。

Task 1 的最终官方排名指标是 `RMSE_VA`，越低越好。

## 2. Task 2/3：三元组与四元组抽取

### 正式 LoRA 训练入口

#### `src/train_extraction_lora.py`

Task 2/3 最主要的训练文件，负责：

- 以 4-bit 方式加载 Qwen；
- 进行生成式 QLoRA 微调；
- 根据文本生成 Task 3 四元组：

```text
(Aspect, Category, Opinion, VA)
```

- 去除 `Category` 后生成 Task 2 三元组：

```text
(Aspect, Opinion, VA)
```

- 在 Dev 上计算 Task 2/3 的连续 F1；
- 早停并保存最佳 LoRA 检查点；
- 加载检查点生成 Test 预测。

实际实验中，Task 2 和 Task 3 复用同一个抽取 LoRA 检查点；二者分别生成官方格式的结果文件并分别评测。

当前脚本还支持预测阶段的 BM25 动态示例检索：用 `word`、`bigram` 或
`trigram` 视角从英文 Train 选择 3 条标注示例，并记录逐条生成诊断。训练逻辑和
旧适配器保持不变。

#### `src/extraction_hybrid.py`

英文 Task 2/3 当前最佳后处理入口，负责：

- 建立 word/bigram/trigram 三种 BM25 检索视角；
- 恢复模型生成片段在原文中的真实大小写，并保持精确跨度；
- 对三路完全一致的 Aspect/Opinion/Category 做二票多数投票；
- 将投票后的关系转换成关系级 V/A 回归数据；
- 把 RoBERTa 的 V/A 分数写回 Task 2 和 Task 3；
- 分别使用官方 Task 2/3 模板恢复不同的记录 ID。

#### `src/train_task1_logs_sigma_encoder.py`

在本实验中复用为“关系 V/A 重评分器”。输入是原文与
`Aspect: ... [OPINION] ...` 组成的文本对，只解冻英语情感 RoBERTa 的后 12/24
层，使用独立 V/A 回归头、LogSigma 动态权重、Opinion 辅助损失和 VA 均衡采样。
Dev 比较 seed42、3407、2026 后，按预设简化规则冻结 seed42 用于 Test。

当前冻结方案的英文 Restaurant Test cF1 为 Task 2 `0.6165776036`、Task 3
`0.5734562902`；两项均由官方脚本复核。

### Task 2/3 数据、校准与评测

#### `src/dimabsa_extraction.py`

抽取任务的数据和输出处理文件，负责：

- 读取 Task 2/3 数据；
- 统一三元组、四元组结构；
- 解析模型生成的 JSON；
- 检查 Aspect 和 Opinion 是否为合法原文片段；
- 检查 Category 是否合法；
- 检查 V/A 是否位于 `[1, 9]`；
- 去重并写出官方格式预测文件。

#### `src/calibrate_extraction_affine.py`

使用 Dev 中结构匹配的关系拟合 V/A 线性校准参数，然后分别应用到 Task 2 和 Task 3。

英文 LoRA 实验只校准 VA，不删除任何预测关系。

#### `src/evaluate_extraction.py`

Task 2/3 本地严格评测文件，计算：

- 结构 `TP`、`FP`、`FN`；
- Exact Precision、Recall、F1；
- `cTP`；
- `cPrecision`；
- `cRecall`；
- `cF1`。

Task 2 和 Task 3 的最终官方排名指标是 `cF1`，越高越好。

## 3. 无训练 Instruct 实验

### `src/run_instruct.py`

Task 1 无训练推理入口，负责运行：

- Direct；
- CoT；
- 固定 Few-shot；
- Dynamic Few-shot；
- Qwen 批量推理；
- 输出解析、格式重试和诊断记录。

### `src/dimabsa_prompts.py`

负责构造 Task 1 的 Direct、CoT、固定 Few-shot 和 Dynamic Few-shot 提示词。

### `src/dimabsa_data.py`

负责：

- 读取和标准化 Task 1 数据；
- 读取 Aspect 和 V/A；
- 选择固定 Few-shot 示例；
- 使用文本和 Aspect 词法相似度检索 Dynamic Few-shot 示例；
- 写出官方格式预测。

### `src/run_extraction.py`

早期中文 Task 2/3 的无训练 Instruct 推理入口，不是本次英文 LoRA 的训练文件。

### `src/dimabsa_extraction_prompts.py`

为早期中文 Task 2/3 无训练实验构造抽取提示词。

## 4. 官方评测程序

官方仓库提供的评测文件位于：

```text
resources/DimABSA2026/evaluation_script/metrics_subtask_1_2_3.py
```

本项目最终结果同时经过本地严格评测器和官方评测脚本核对：

- Task 1 使用官方 `RMSE_VA`；
- Task 2/3 使用官方 `cF1`。

## 5. 离线测试

### `tests/test_offline.py`

不加载 Qwen 的离线测试文件，检查：

- 数据解析；
- Prompt 结构；
- 模型输出解析；
- Dynamic Few-shot 检索；
- Task 1 集成和校准；
- 按记录分组的 Ridge 校准；
- Task 2/3 抽取解析；
- 英文 `NULL` Aspect/Opinion；
- 抽取 VA 校准不删除预测关系。
- 三视角 BM25 检索、原文大小写恢复和结构投票；
- Task 2/3 官方模板 ID 不同情况下的对齐输出。

## 6. 最重要的实验文件

如果只需要向老师介绍主要代码，可以重点说明：

| 实验部分 | 主要文件 |
|---|---|
| Task 1 LoRA 训练 | `src/train_task1_lora_regression.py` |
| Gemma Task 1 改进实验 | `src/train_task1_gemma_regression.py` |
| Task 1 校准与最终混合 | `src/calibrate_task1.py`、`src/blend_task1.py` |
| Task 1 无训练四路集成 | `src/run_instruct.py`、`src/ensemble_task1.py` |
| Task 1 评测 | `src/evaluate_task1.py` |
| Task 2/3 LoRA 训练 | `src/train_extraction_lora.py` |
| Task 2/3 动态检索、投票与重评分 | `src/extraction_hybrid.py`、`src/train_task1_logs_sigma_encoder.py` |
| Task 2/3 数据与输出处理 | `src/dimabsa_extraction.py` |
| Task 2/3 VA 校准 | `src/calibrate_extraction_affine.py` |
| Task 2/3 评测 | `src/evaluate_extraction.py` |
| 官方评测 | `resources/DimABSA2026/evaluation_script/metrics_subtask_1_2_3.py` |
