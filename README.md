# dim absa

使用同一个 `Qwen3-4B-Instruct-2507` 完成 DimABSA2026 Track A / Task 1 / 中文餐厅领域实验，对比普通 Prompt、CoT Prompt 和 Few-shot CoT，并用 dev 金标准拟合 VA 线性尺度校准。当前版本**不训练 Qwen、不创建 LoRA、不反向传播**；校准阶段只拟合 4 个标量参数。

官方仓库：<https://github.com/DimABSA/DimABSA2026>

## 1. 当前实验范围

- Track：A（DimABSA）
- Subtask：1（DimASR）
- 语言：中文 `zho`
- 领域：餐厅 `restaurant`
- 输入：`Text` 与有序方面词列表
- 输出：每个方面的 `Valence#Arousal`，两项都在 `[1.00, 9.00]`
- 指标：`RMSE_VA`，越低越好
- 模型：官方 `Qwen/Qwen3-4B-Instruct-2507` BF16
- 后端：原生 Transformers；云端现有 Unsloth 与 Torch 2.11 不兼容，未修改基础环境

## 2. 项目结构

```text
dim absa/
├── AGENTS.md
├── .ai/                         # AI 项目状态与交接
├── src/
│   ├── dimabsa_data.py          # 两种官方 schema 归一化、均值、JSONL 输出
│   ├── dimabsa_prompts.py       # direct / CoT / few-shot Prompt
│   ├── run_instruct.py          # 无训练 Qwen 推理，默认 smoke
│   ├── calibrate_task1.py        # dev 拟合/应用 V、A 仿射尺度校准
│   └── evaluate_task1.py        # 严格检查 ID/方面顺序/重复方面并计算指标
├── tests/test_offline.py        # 不加载模型的离线测试
├── resources/DimABSA2026/       # 完整官方仓库本地快照，不提交个人仓库
├── results/                     # 预测、诊断、元数据和汇总表
├── scripts/download_official_resources.sh
└── requirements.txt
```

本地官方资源对应提交：`bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`。

中文餐厅 Task 1 当前规模：

| split | 文本数 |
|---|---:|
| train | 6,050 |
| dev | 300 |
| test | 1,000 |

训练文件使用 `Quadruplet`，dev/test 金标准使用 `Aspect_VA`；`dimabsa_data.py` 会统一转换为 Task 1 的有序方面与 VA。模型 Prompt 永远只接收文本和方面词，不接收金标准 VA。

## 3. 四种运行模式

| `--prompt-mode` | 是否加载模型 | 说明 |
|---|---:|---|
| `mean` | 否 | 所有方面预测训练集 VA 均值，用于最低基线和链路检查 |
| `direct` | 是 | 评分标尺 + 严格 JSON 输出，不含 CoT 句子 |
| `cot` | 是 | 在 direct 的基础上加入 `Let's think step by step` |
| `fewshot` | 是 | CoT 再加 5 个训练集同领域锚点示例 |

为了公平比较，`direct` 与 `cot` 使用相同模型、数据、评分标尺和生成参数；CoT 版本只增加逐步思考提示。模型只生成数值数组，程序使用原始 ID 与方面词重建官方 JSONL，避免模型改写方面词。

## 4. 验证

```bash
cd '/Users/weiguang/Desktop/lora/dim absa'

PYTHONPYCACHEPREFIX=/tmp/dimabsa-pycache \
  python3 -m py_compile \
  src/dimabsa_data.py src/dimabsa_prompts.py \
  src/run_instruct.py src/calibrate_task1.py \
  src/evaluate_task1.py tests/test_offline.py

python3 -m unittest discover -s tests -v
ruff check src tests
```

离线测试不下载模型，当前 7/7 通过，已覆盖：

- 6,050 条 train 与 300 条 dev 的两种官方 schema
- 训练集均值和 few-shot 示例选择
- direct/CoT Prompt 区别及金标准不进入 Prompt
- 模型 JSON 解析、数值范围和数量检查
- 动态方面编号、输出数量协议与线性校准/裁剪
- 官方预测文件不含 `Text` 的正常读取

当前训练集均值为 `5.8137#5.7572`。完整 dev 均值基线实测：

```text
records=300
aspects=685
RMSE_VA=1.2879411511
RMSE_VA_NORMALIZED=0.1138389902
```

均值预测是常数，因此 `PCC_V/PCC_A` 无定义；这不是模型结果。

## 5. 云端环境与复现命令

实测环境：SeetaCloud RTX 5090 32 GB、Python 3.12.3、Torch 2.11.0+cu130、Transformers 5.5.0。官方 BF16 模型位于：

```text
/root/autodl-tmp/models/Qwen3-4B-Instruct-2507
```

完整 dev 推理命令（付费 GPU，保留门禁）：

```bash
cd '/root/autodl-tmp/lora/dim absa'

for mode in direct cot fewshot; do
  RUN_MODE=full CONFIRM_FULL_RUN=YES /root/miniconda3/bin/python \
    src/run_instruct.py \
    --prompt-mode "$mode" \
    --backend transformers \
    --model-name /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
    --batch-size 16
done
```

拟合 dev 校准并应用到预测：

```bash
/root/miniconda3/bin/python src/calibrate_task1.py fit \
  --gold resources/DimABSA2026/task-dataset/track_a/subtask_1/zho/zho_restaurant_dev_task1.jsonl \
  --pred results/zho_restaurant_dev_task1_fewshot_full.jsonl \
  --output-params results/zho_restaurant_dev_task1_fewshot_calibration.json \
  --output-pred results/zho_restaurant_dev_task1_fewshot_calibrated.jsonl
```

程序为每个方面显式编号，并要求输出数量严格等于方面数；解析失败时最多进行两次只纠正格式的重试。最终正式 dev/test 都是 `parse_failures=0`、`format_retry_recoveries=0`，没有使用均值回退。

## 6. 正式结果

官方 Task 1 主指标是 `RMSE_VA`，越低越好。dev 共 300 条、685 个方面：

| 方法 | PCC V | PCC A | 原始 RMSE | 校准后 RMSE |
|---|---:|---:|---:|---:|
| direct | 0.7123 | 0.4803 | 1.9865 | 0.9547 |
| CoT | 0.7078 | 0.4684 | **1.9353** | 0.9612 |
| few-shot CoT | **0.7717** | **0.5036** | 1.9836 | **0.8940** |
| train mean | 无定义 | 无定义 | 1.2879 | — |

原始预测中 CoT 的 RMSE 最低；校准后 few-shot 最优，因此在查看 test 结果前锁定“few-shot + dev 仿射校准”。只对这一种模型方案运行一次 test。test 共 1,000 条、1,929 个方面：

| test 方法 | PCC V | PCC A | RMSE |
|---|---:|---:|---:|
| train mean | 无定义 | 无定义 | 1.4761 |
| few-shot 原始 | 0.7745 | 0.4316 | 2.0312 |
| **few-shot + dev 校准** | **0.7745** | **0.4314** | **1.1149** |

校准方案比 test 训练均值基线降低约 24.5% RMSE，比未校准 few-shot 降低约 45.1%。项目严格评测器与官方 SciPy 脚本均得到 `RMSE_VA=1.1149206501`。

最终预测文件：`results/zho_restaurant_test_task1_fewshot_calibrated.jsonl`；校准参数：`results/zho_restaurant_dev_task1_fewshot_calibration.json`。test 标签只用于最终一次报告，没有用于选择 Prompt 或修改校准参数。

## 7. 资源重新下载与代理

当前完整官方仓库已经下载，不需要重复执行。若以后在新的本机目录重建：

```bash
PROXY_URL=http://127.0.0.1:7897 bash scripts/download_official_resources.sh
```

脚本只给本次 `git clone` 设置代理，不修改全局 Git 配置。云端不需要本机代理时省略 `PROXY_URL`。

## 8. GitHub 发布边界

个人仓库建议提交：

- `src/`、`tests/`、`scripts/`
- README、依赖、AI 项目记忆
- `results/metrics.csv` 和经过整理的方法说明

不要提交：

- 官方原始数据和完整上游仓库
- 模型权重、checkpoint、`.env`
- 含完整文本的原始诊断与预测文件
- smoke 结果冒充正式结果
