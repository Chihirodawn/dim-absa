"""English Task 1 encoder regression with LogSigma and opinion supervision."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from dimabsa_data import load_task1_records, write_task1_predictions
from encoder_experiment_utils import (
    AspectExample,
    balanced_weight_values,
    examples_from_records,
    load_train_examples,
    opinion_spans,
    split_oof,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EncoderDataset(Dataset):
    def __init__(self, examples: list[AspectExample], tokenizer, max_length: int) -> None:
        self.items: list[dict] = []
        for example in examples:
            aspect = "[IMPLICIT_ASPECT]" if example.aspect == "NULL" else example.aspect
            encoded = tokenizer(
                example.text,
                aspect,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")
            sequence_ids = encoded.sequence_ids()
            spans = opinion_spans(example.text, example.opinion)
            opinion_mask, text_mask = [], []
            for sequence_id, (left, right) in zip(sequence_ids, offsets):
                is_text = sequence_id == 0 and right > left
                text_mask.append(int(is_text))
                opinion_mask.append(
                    int(
                        is_text
                        and any(right > start and left < end for start, end in spans)
                    )
                )
            self.items.append(
                {
                    **encoded,
                    "labels": example.score,
                    "opinion_labels": opinion_mask,
                    "text_mask": text_mask,
                    "has_opinion_supervision": example.opinion is not None,
                    "record_id": example.record_id,
                    "aspect_index": example.aspect_index,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


class EncoderCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        labels = torch.tensor([item["labels"] for item in features], dtype=torch.float32)
        record_ids = [item["record_id"] for item in features]
        aspect_indices = [item["aspect_index"] for item in features]
        supervised = torch.tensor(
            [item["has_opinion_supervision"] for item in features], dtype=torch.bool
        )
        model_features = [
            {k: v for k, v in item.items() if k in {"input_ids", "attention_mask"}}
            for item in features
        ]
        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        opinion_labels = torch.zeros_like(batch["attention_mask"], dtype=torch.float32)
        text_mask = torch.zeros_like(batch["attention_mask"], dtype=torch.bool)
        for row, item in enumerate(features):
            length = len(item["opinion_labels"])
            opinion_labels[row, :length] = torch.tensor(item["opinion_labels"])
            text_mask[row, :length] = torch.tensor(item["text_mask"], dtype=torch.bool)
        return {
            **batch,
            "labels": labels,
            "opinion_labels": opinion_labels,
            "text_mask": text_mask,
            "opinion_supervised": supervised,
            "record_ids": record_ids,
            "aspect_indices": aspect_indices,
        }


class LogSigmaRegressor(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.v_head = nn.Linear(hidden_size, 1)
        self.a_head = nn.Linear(hidden_size, 1)
        self.opinion_head = nn.Linear(hidden_size, 1)
        initial_variance = torch.empty(2).uniform_(0.2, 1.0)
        self.log_vars = nn.Parameter(torch.log(initial_variance))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.dropout(hidden[:, 0])
        scores = torch.cat([self.v_head(cls), self.a_head(cls)], dim=-1)
        opinion_logits = self.opinion_head(self.dropout(hidden)).squeeze(-1)
        return scores, opinion_logits

    def initialize_bias(self, means: tuple[float, float]) -> None:
        with torch.no_grad():
            self.v_head.bias.fill_(means[0])
            self.a_head.bias.fill_(means[1])


def freeze_encoder_layers(model, unfreeze_last_layers: int) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = model.encoder.layer
    if not 1 <= unfreeze_last_layers <= len(layers):
        raise ValueError(f"unfreeze_last_layers must be within 1..{len(layers)}")
    for layer in layers[-unfreeze_last_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    return {
        "encoder_layers": len(layers),
        "unfrozen_layers": unfreeze_last_layers,
        "trainable_encoder_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }


def metric_dict(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    error = scores - labels
    result = {
        "rmse_va": float(torch.sqrt(torch.square(error).sum() / len(labels))),
        "rmse_v": float(torch.sqrt(torch.square(error[:, 0]).mean())),
        "rmse_a": float(torch.sqrt(torch.square(error[:, 1]).mean())),
    }
    for index, name in enumerate(("pcc_v", "pcc_a")):
        left, right = scores[:, index].double(), labels[:, index].double()
        left, right = left - left.mean(), right - right.mean()
        denominator = torch.sqrt(torch.square(left).sum() * torch.square(right).sum())
        result[name] = float((left * right).sum() / denominator) if denominator else 0.0
    return result


@torch.inference_mode()
def predict(model, regressor, loader, device):
    model.eval()
    regressor.eval()
    all_scores, all_labels, keys = [], [], []
    for batch in loader:
        labels = batch.pop("labels")
        record_ids = batch.pop("record_ids")
        aspect_indices = batch.pop("aspect_indices")
        batch.pop("opinion_labels")
        batch.pop("text_mask")
        batch.pop("opinion_supervised")
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(**inputs, return_dict=True).last_hidden_state
            scores, _ = regressor(hidden)
        all_scores.append(scores.clamp(1.0, 9.0).float().cpu())
        all_labels.append(labels)
        keys.extend(zip(record_ids, aspect_indices))
    scores, labels = torch.cat(all_scores), torch.cat(all_labels)
    return scores, labels, keys, metric_dict(scores, labels)


def prediction_map(records, scores, keys):
    collected = {record.record_id: [None] * len(record.aspects) for record in records}
    for key, score in zip(keys, scores.tolist()):
        collected[key[0]][key[1]] = tuple(map(float, score))
    if any(value is None for row in collected.values() for value in row):
        raise ValueError("Prediction coverage is incomplete")
    return {key: tuple(values) for key, values in collected.items()}


def save_checkpoint(path: Path, model, regressor, config: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in trainable},
        path / "encoder_trainable.pt",
    )
    torch.save(regressor.state_dict(), path / "regressor.pt")
    (path / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_checkpoint(path: Path, model, regressor) -> None:
    state = torch.load(path / "encoder_trainable.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected encoder checkpoint keys: {unexpected[:5]}")
    if not state or not missing:
        raise ValueError("Partial encoder checkpoint validation failed")
    regressor.load_state_dict(
        torch.load(path / "regressor.pt", map_location="cpu", weights_only=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "train", "predict"], default="smoke")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-file")
    parser.add_argument("--dev-file")
    parser.add_argument("--input-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-pred")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--patience",
        type=int,
        default=12,
        help="Evaluation events without improvement; 12 equals 3 epochs at 4 evals/epoch",
    )
    parser.add_argument("--evals-per-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=2e-5)
    parser.add_argument("--logsigma-learning-rate", type=float, default=5e-2)
    parser.add_argument("--opinion-loss-weight", type=float, default=0.1)
    parser.add_argument("--opinion-positive-weight", type=float, default=10.0)
    parser.add_argument("--unfreeze-last-layers", type=int, default=12)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oof-fold", type=int)
    parser.add_argument("--oof-folds", type=int, default=3)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "train" and os.environ.get("CONFIRM_FULL_RUN") != "YES":
        raise RuntimeError("Full paid-GPU training requires CONFIRM_FULL_RUN=YES")
    if args.mode in {"smoke", "train"} and not args.train_file:
        raise ValueError("Training requires --train-file")
    if args.mode in {"smoke", "train"} and args.oof_fold is None and not args.dev_file:
        raise ValueError("Non-OOF training requires --dev-file")
    if args.mode == "predict" and (not args.input_file or not args.output_pred):
        raise ValueError("Prediction requires --input-file and --output-pred")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name, dtype=torch.bfloat16)
    layer_config = freeze_encoder_layers(model, args.unfreeze_last_layers)
    regressor = LogSigmaRegressor(model.config.hidden_size)

    if args.mode == "predict":
        records = load_task1_records(args.input_file, require_gold=True)
        dataset = EncoderDataset(examples_from_records(records), tokenizer, args.max_length)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=EncoderCollator(tokenizer),
            pin_memory=True,
            num_workers=2,
        )
        load_checkpoint(output_dir, model, regressor)
        model, regressor = model.to(device), regressor.to(device)
        scores, _, keys, metrics = predict(model, regressor, loader, device)
        write_task1_predictions(records, prediction_map(records, scores, keys), args.output_pred)
        print(json.dumps({**metrics, "records": len(records), "aspects": len(dataset)}))
        return

    all_records, all_examples = load_train_examples(args.train_file)
    if args.oof_fold is not None:
        train_records, train_examples, dev_records, dev_examples = split_oof(
            all_records, all_examples, args.oof_folds, args.oof_fold
        )
    else:
        train_records, train_examples = all_records, all_examples
        dev_records = load_task1_records(args.dev_file, require_gold=True)
        dev_examples = examples_from_records(dev_records)
    if args.mode == "smoke":
        train_examples, dev_examples = train_examples[:64], dev_examples[:64]
        train_ids, dev_ids = {e.record_id for e in train_examples}, {e.record_id for e in dev_examples}
        train_records = [r for r in train_records if r.record_id in train_ids]
        dev_records = [r for r in dev_records if r.record_id in dev_ids]
    train_dataset = EncoderDataset(train_examples, tokenizer, args.max_length)
    dev_dataset = EncoderDataset(dev_examples, tokenizer, args.max_length)
    collator = EncoderCollator(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    shuffle = True
    if args.balanced_sampling:
        sampler = WeightedRandomSampler(
            torch.tensor(balanced_weight_values(train_examples), dtype=torch.double),
            len(train_examples),
            generator=generator,
        )
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collator,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size * 2,
        collate_fn=collator,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )
    labels = torch.tensor([example.score for example in train_examples])
    means = tuple(map(float, labels.mean(dim=0)))
    regressor.initialize_bias(means)
    model, regressor = model.to(device), regressor.to(device)
    encoder_parameters = [p for p in model.parameters() if p.requires_grad]
    head_parameters = [p for name, p in regressor.named_parameters() if name != "log_vars"]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
            {"params": [regressor.log_vars], "lr": args.logsigma_learning_rate, "weight_decay": 0.0},
        ],
        weight_decay=0.01,
    )
    epochs = 1 if args.mode == "smoke" else args.epochs
    update_steps = math.ceil(len(train_loader) / args.grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, int(update_steps * 0.1)), update_steps
    )
    eval_interval = max(1, math.ceil(len(train_loader) / args.evals_per_epoch))
    config = {
        **vars(args),
        **layer_config,
        "hidden_size": model.config.hidden_size,
        "train_label_means": means,
        "train_records": len(train_records),
        "train_aspects": len(train_dataset),
        "dev_records": len(dev_records),
        "dev_aspects": len(dev_dataset),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"mode": args.mode, **layer_config, "train_aspects": len(train_dataset), "dev_aspects": len(dev_dataset)}), flush=True)
    best, bad_evals, global_step, optimizer_step = float("inf"), 0, 0, 0
    history: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    stop = False
    for epoch in range(epochs):
        model.train()
        regressor.train()
        loss_sum = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            opinion_labels = batch.pop("opinion_labels").to(device)
            text_mask = batch.pop("text_mask").to(device)
            supervised = batch.pop("opinion_supervised").to(device)
            batch.pop("record_ids")
            batch.pop("aspect_indices")
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model(**inputs, return_dict=True).last_hidden_state
                scores, opinion_logits = regressor(hidden)
                mse_by_dimension = torch.square(scores.float() - labels).mean(dim=0)
                va_loss = 0.5 * torch.sum(
                    torch.exp(-regressor.log_vars) * mse_by_dimension + regressor.log_vars
                )
                active = text_mask & supervised.unsqueeze(1)
                opinion_loss = F.binary_cross_entropy_with_logits(
                    opinion_logits.float()[active],
                    opinion_labels[active],
                    pos_weight=torch.tensor(args.opinion_positive_weight, device=device),
                )
                loss = va_loss + args.opinion_loss_weight * opinion_loss
            (loss / args.grad_accum).backward()
            loss_sum += float(loss.detach())
            global_step += 1
            if batch_index % args.grad_accum == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_([*encoder_parameters, *regressor.parameters()], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            if batch_index % eval_interval and batch_index != len(train_loader):
                continue
            scores_dev, _, keys, metrics = predict(model, regressor, dev_loader, device)
            log_vars = regressor.log_vars.detach().float().cpu()
            row = {
                "epoch": epoch + batch_index / len(train_loader),
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                "train_loss": loss_sum / batch_index,
                **metrics,
                "sigma2_v": float(torch.exp(log_vars[0])),
                "sigma2_a": float(torch.exp(log_vars[1])),
                "precision_v": float(torch.exp(-log_vars[0])),
                "precision_a": float(torch.exp(-log_vars[1])),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if metrics["rmse_va"] < best:
                best, bad_evals = metrics["rmse_va"], 0
                config["best_step"] = global_step
                save_checkpoint(output_dir, model, regressor, config)
                if len(dev_dataset) == sum(len(record.aspects) for record in dev_records):
                    write_task1_predictions(
                        dev_records,
                        prediction_map(dev_records, scores_dev, keys),
                        output_dir / "best_dev_predictions.jsonl",
                    )
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    stop = True
            model.train()
            regressor.train()
            if stop:
                break
        if stop:
            break
    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "best_dev_rmse_va": best,
        "best_step": config.get("best_step"),
        "history": history,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
