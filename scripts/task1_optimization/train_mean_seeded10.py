"""
Mean pooling 实验：修改 LogSigma 训练脚本使用 mean pooling 而不是 CLS。
在云端运行：python scripts/task1_optimization/train_mean_pooling.py
"""
import json
import torch
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer
import os
import sys

# 添加 src 目录到路径
BASE_DIR = '/root/autodl-tmp/dimabsa_task1'
sys.path.insert(0, f'{BASE_DIR}/src')

from train_task1_logs_sigma_encoder import (
    EncoderDataset, EncoderCollator, LogSigmaRegressor,
    load_task1_records, examples_from_records
)

DEVICE = torch.device('cuda')
DTYPE = torch.bfloat16

class MeanPoolingRegressor(nn.Module):
    """使用 mean pooling 而不是 CLS"""
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.v_head = nn.Linear(hidden_size, 1)
        self.a_head = nn.Linear(hidden_size, 1)
        self.opinion_head = nn.Linear(hidden_size, 1)
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, hidden, attention_mask):
        # Mean pooling: 对所有 token 表示取平均（忽略 padding）
        # hidden: (batch, seq_len, hidden_size)
        # attention_mask: (batch, seq_len)

        # 扩展 attention_mask 以匹配 hidden 维度（保持 dtype）
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (batch, seq_len, 1)

        # 计算有效 token 数量（保持 dtype）
        token_counts = mask.sum(dim=1)  # (batch, 1)
        token_counts = token_counts.clamp(min=torch.tensor(1.0, dtype=hidden.dtype, device=hidden.device))

        # 对 hidden 进行 mask 并求和
        masked_hidden = hidden * mask
        summed = masked_hidden.sum(dim=1)  # (batch, hidden_size)

        # 除以有效 token 数量得到平均值（保持 dtype）
        mean_pooled = (summed / token_counts).to(hidden.dtype)  # (batch, hidden_size)

        cls = self.dropout(mean_pooled)
        scores = torch.cat([self.v_head(cls), self.a_head(cls)], dim=-1)
        return scores

def train_mean_pooling(seed, output_dir):
    """训练 mean pooling 模型（随机抽种子、记录、固定，可复现）"""
    import random as py_random
    rng_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    py_random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)
    print(f"Run {seed}: rng_seed={rng_seed}")
    print(f"\n{'='*60}")
    print(f"Training mean pooling model, seed={seed}")
    print('='*60)

    # 加载数据
    train_records = load_task1_records(f'{BASE_DIR}/data/DimABSA2026/task-dataset/track_a/subtask_1/eng/eng_restaurant_train_alltasks.jsonl')
    dev_records = load_task1_records(f'{BASE_DIR}/data/DimABSA2026/task-dataset/track_a/subtask_1/eng/eng_restaurant_dev_task1.jsonl')

    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/roberta-large-sentiment', use_fast=True)

    train_dataset = EncoderDataset(examples_from_records(train_records), tokenizer, max_length=128)
    dev_dataset = EncoderDataset(examples_from_records(dev_records), tokenizer, max_length=128)

    generator = torch.Generator().manual_seed(rng_seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        collate_fn=EncoderCollator(tokenizer), generator=generator
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=16, shuffle=False, collate_fn=EncoderCollator(tokenizer)
    )

    # 使用预训练编码器初始化
    model = AutoModel.from_pretrained(
        f'{BASE_DIR}/outputs/semeval_pretrain/encoder',
        dtype=DTYPE
    ).to(DEVICE)

    # 解冻最后 12 层
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.encoder.layer[-12:]:
        for param in layer.parameters():
            param.requires_grad = True

    # 使用 MeanPoolingRegressor
    regressor = MeanPoolingRegressor(model.config.hidden_size).to(DEVICE).to(DTYPE)

    # 训练参数（排除 log_vars 避免重复）
    regressor_params = [p for name, p in regressor.named_parameters()
                       if p.requires_grad and name != 'log_vars']
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if p.requires_grad], 'lr': 2e-5},
        {'params': regressor_params, 'lr': 2e-5},
        {'params': [regressor.log_vars], 'lr': 0.05, 'weight_decay': 0.0}
    ])

    # 训练循环（简化版）
    best_dev_rmse = float('inf')
    best_step = 0
    global_step = 0

    for epoch in range(25):
        model.train()
        regressor.train()

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels', 'opinion_labels', 'opinion_supervised']}
            # 转换 labels 到 bf16 与模型一致
            batch['labels'] = batch['labels'].to(DTYPE)
            batch['opinion_labels'] = batch['opinion_labels'].to(DTYPE)

            hidden = model(batch['input_ids'], batch['attention_mask'], return_dict=True).last_hidden_state
            scores = regressor(hidden, batch['attention_mask'])

            # LogSigma loss
            mse_v = torch.nn.functional.mse_loss(scores[:, 0], batch['labels'][:, 0])
            mse_a = torch.nn.functional.mse_loss(scores[:, 1], batch['labels'][:, 1])

            with torch.no_grad():
                log_vars = regressor.log_vars.clamp(
                    torch.tensor(-5.0, dtype=DTYPE, device=DEVICE),
                    torch.tensor(5.0, dtype=DTYPE, device=DEVICE)
                )
                prec_v = torch.exp(-log_vars[0])
                prec_a = torch.exp(-log_vars[1])

            va_loss = torch.tensor(0.5, dtype=DTYPE, device=DEVICE) * (prec_v * mse_v + log_vars[0] + prec_a * mse_a + log_vars[1])
            loss = va_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # 每 58 步评估一次（约 0.25 epoch）
            if global_step % 58 == 0:
                model.eval()
                regressor.eval()

                dev_preds = []
                dev_labels = []

                with torch.no_grad():
                    for batch in dev_loader:
                        batch = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels', 'opinion_labels', 'opinion_supervised']}
                        hidden = model(batch['input_ids'], batch['attention_mask'], return_dict=True).last_hidden_state
                        scores = regressor(hidden, batch['attention_mask']).clamp(1.0, 9.0)
                        dev_preds.extend(scores.float().cpu().numpy())
                        dev_labels.extend(batch['labels'].float().cpu().numpy())

                dev_preds = np.array(dev_preds)
                dev_labels = np.array(dev_labels)

                rmse_va = np.sqrt(np.mean((dev_preds - dev_labels)**2))
                rmse_v = np.sqrt(np.mean((dev_preds[:, 0] - dev_labels[:, 0])**2))
                rmse_a = np.sqrt(np.mean((dev_preds[:, 1] - dev_labels[:, 1])**2))

                print(f'Epoch {epoch+1}, Step {global_step}: dev RMSE_VA={rmse_va:.6f}, RMSE_V={rmse_v:.6f}, RMSE_A={rmse_a:.6f}')

                if rmse_va < best_dev_rmse:
                    best_dev_rmse = rmse_va
                    best_step = global_step

                    # 保存模型
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(model.state_dict(), f'{output_dir}/encoder_trainable.pt')
                    torch.save(regressor.state_dict(), f'{output_dir}/regressor.pt')

                    config = {
                        'model_name': 'semeval_pretrain/encoder',
                        'pooling': 'mean',
                        'run_label': seed,
                        'rng_seed': rng_seed,
                        'unfreeze_last_layers': 12,
                        'best_step': int(best_step),
                        'best_dev_rmse_va': float(best_dev_rmse)
                    }
                    with open(f'{output_dir}/experiment_config.json', 'w') as f:
                        json.dump(config, f, indent=2)

                model.train()
                regressor.train()

    print(f"\nBest Dev RMSE_VA: {best_dev_rmse:.6f} at step {best_step}")
    return best_dev_rmse

if __name__ == '__main__':
    # 10 次随机初始化，每次记录 rng_seed，可复现
    for run in range(1, 11):
        output_dir = f'{BASE_DIR}/outputs/mean_seeded10_run{run}'
        train_mean_pooling(run, output_dir)

    print("\n" + "="*60)
    print("Mean pooling training completed for all seeds")
    print("="*60)
