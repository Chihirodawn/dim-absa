"""LoRA dual-regression training for DimABSA Task 1."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from dimabsa_data import Task1Record, load_task1_records, write_task1_predictions


@dataclass(frozen=True)
class AspectKey:
    record_id: str
    aspect_index: int


class AspectRegressionDataset(Dataset):
    def __init__(
        self,
        records: list[Task1Record],
        tokenizer,
        max_length: int,
        limit: int | None = None,
    ) -> None:
        self.items = []
        for record in records:
            if record.gold_scores is None:
                raise ValueError(f"ID {record.record_id!r}: VA scores are required")
            for aspect_index, (aspect, score) in enumerate(
                zip(record.aspects, record.gold_scores)
            ):
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Represent the sentiment toward the specified aspect. "
                            "Valence and Arousal will be predicted by regression heads."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Text: {record.text}\nAspect: {aspect}",
                    },
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                encoded = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False,
                )
                self.items.append(
                    {
                        **encoded,
                        "labels": score,
                        "key": AspectKey(record.record_id, aspect_index),
                    }
                )
                if limit is not None and len(self.items) >= limit:
                    return

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


class RegressionCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        keys = [feature["key"] for feature in features]
        labels = torch.tensor(
            [feature["labels"] for feature in features], dtype=torch.float32
        )
        model_features = [
            {key: value for key, value in feature.items() if key not in {"key", "labels"}}
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["labels"] = labels
        batch["keys"] = keys
        return batch


class DualRegressionHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # The bounded output matches the official [1, 9] VA scale.
        return 1.0 + 8.0 * torch.sigmoid(self.projection(self.dropout(hidden)))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_base(model_name: str, *, train: bool):
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = not train
    if train:
        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, config)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def _last_hidden(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    last_positions = batch["attention_mask"].sum(dim=1) - 1
    return hidden[
        torch.arange(hidden.shape[0], device=hidden.device), last_positions
    ]


@torch.inference_mode()
def predict(model, head, loader, device) -> tuple[list[tuple[float, float]], list[AspectKey], float]:
    model.eval()
    head.eval()
    values: list[tuple[float, float]] = []
    keys: list[AspectKey] = []
    squared_error = 0.0
    aspect_count = 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch_keys = batch.pop("keys")
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = head(_last_hidden(model, inputs)).float()
        squared_error += torch.square(scores - labels).sum().item()
        aspect_count += labels.shape[0]
        values.extend(tuple(map(float, row)) for row in scores.cpu().tolist())
        keys.extend(batch_keys)
    return values, keys, math.sqrt(squared_error / aspect_count)


def prediction_map(
    records: list[Task1Record],
    values: list[tuple[float, float]],
    keys: list[AspectKey],
) -> dict[str, tuple[tuple[float, float], ...]]:
    collected: dict[str, list[tuple[float, float] | None]] = {
        record.record_id: [None] * len(record.aspects) for record in records
    }
    for key, value in zip(keys, values):
        collected[key.record_id][key.aspect_index] = value
    output = {}
    for record_id, scores in collected.items():
        if any(score is None for score in scores):
            raise ValueError(f"ID {record_id!r}: incomplete regression predictions")
        output[record_id] = tuple(score for score in scores if score is not None)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen LoRA dual regression for Task 1")
    parser.add_argument("--mode", choices=["smoke", "train", "predict"], default="smoke")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-file")
    parser.add_argument("--dev-file")
    parser.add_argument("--input-file")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-pred")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _check_args(args: argparse.Namespace) -> None:
    if args.mode == "train" and os.environ.get("CONFIRM_FULL_RUN") != "YES":
        raise RuntimeError("Full paid-GPU training requires CONFIRM_FULL_RUN=YES")
    if args.mode in {"smoke", "train"} and (not args.train_file or not args.dev_file):
        raise ValueError("Training requires --train-file and --dev-file")
    if args.mode == "predict" and (not args.input_file or not args.output_pred):
        raise ValueError("Prediction requires --input-file and --output-pred")


def main() -> None:
    args = parse_args()
    _check_args(args)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.time()
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "predict":
        records = load_task1_records(args.input_file, require_gold=True)
        dataset = AspectRegressionDataset(records, tokenizer, args.max_length)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            collate_fn=RegressionCollator(tokenizer),
        )
        base = _load_base(args.model_name, train=False)
        model = PeftModel.from_pretrained(base, args.adapter_dir).to(device)
        head = DualRegressionHead(model.config.hidden_size).to(device)
        head.load_state_dict(
            torch.load(Path(args.adapter_dir) / "regression_head.pt", map_location=device)
        )
        values, keys, rmse = predict(model, head, loader, device)
        write_task1_predictions(records, prediction_map(records, values, keys), args.output_pred)
        print(json.dumps({"records": len(records), "aspects": len(dataset), "rmse": rmse}))
        return

    train_records = load_task1_records(args.train_file, require_gold=True)
    dev_records = load_task1_records(args.dev_file, require_gold=True)
    limit = 64 if args.mode == "smoke" else None
    epochs = 1 if args.mode == "smoke" else args.epochs
    train_dataset = AspectRegressionDataset(
        train_records, tokenizer, args.max_length, limit=limit
    )
    dev_dataset = AspectRegressionDataset(
        dev_records, tokenizer, args.max_length, limit=32 if args.mode == "smoke" else None
    )
    collator = RegressionCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    model = _load_base(args.model_name, train=True).to(device)
    head = DualRegressionHead(model.config.hidden_size).to(device)
    lora_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.learning_rate},
            {"params": head.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=0.01,
    )
    update_steps = math.ceil(len(train_loader) / args.grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(update_steps * 0.05)),
        num_training_steps=update_steps,
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "train_aspects": len(train_dataset),
                "dev_aspects": len(dev_dataset),
                "trainable_parameters": sum(p.numel() for p in lora_parameters)
                + sum(p.numel() for p in head.parameters()),
                "updates": update_steps,
            }
        ),
        flush=True,
    )
    best_rmse = float("inf")
    bad_epochs = 0
    history = []
    adapter_dir = Path(args.adapter_dir)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch.pop("keys")
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = head(_last_hidden(model, inputs)).float()
                loss = torch.square(scores - labels).mean()
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            running_loss += loss.item()
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [*lora_parameters, *head.parameters()], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        values, keys, dev_rmse = predict(model, head, dev_loader, device)
        row = {
            "epoch": epoch,
            "train_mse": running_loss / len(train_loader),
            "dev_rmse": dev_rmse,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if dev_rmse < best_rmse:
            best_rmse = dev_rmse
            bad_epochs = 0
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            adapter_dir.mkdir(parents=True)
            model.save_pretrained(adapter_dir)
            torch.save(head.state_dict(), adapter_dir / "regression_head.pt")
            if len(dev_dataset) == sum(len(record.aspects) for record in dev_records):
                write_task1_predictions(
                    dev_records,
                    prediction_map(dev_records, values, keys),
                    adapter_dir / "best_dev_predictions.jsonl",
                )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "best_dev_rmse": best_rmse,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    (adapter_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
