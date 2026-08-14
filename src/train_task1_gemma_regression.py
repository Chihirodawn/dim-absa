"""Configurable Qwen/Gemma LoRA regression experiments for DimABSA Task 1."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from dimabsa_data import Task1Record, load_task1_records, write_task1_predictions


@dataclass(frozen=True)
class AspectKey:
    record_id: str
    aspect_index: int


@dataclass(frozen=True)
class FewShotAspect:
    record: Task1Record
    aspect_index: int


_RETRIEVAL_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


class AspectFewShotRetriever:
    """Fast answer-free BM25-style retrieval over Train aspect examples."""

    def __init__(self, records: list[Task1Record]) -> None:
        self.examples: list[FewShotAspect] = []
        tokenized: list[list[str]] = []
        for record in records:
            if record.gold_scores is None or len(record.text) > 220:
                continue
            text_tokens = _RETRIEVAL_TOKEN.findall(record.text.lower())
            for aspect_index, aspect in enumerate(record.aspects):
                aspect_tokens = _RETRIEVAL_TOKEN.findall(aspect.lower())
                self.examples.append(FewShotAspect(record, aspect_index))
                tokenized.append(text_tokens + aspect_tokens * 3)
        if not self.examples:
            raise ValueError("Few-shot retrieval pool is empty")
        self.lengths = [len(tokens) for tokens in tokenized]
        self.average_length = sum(self.lengths) / len(self.lengths)
        document_frequency = Counter(term for tokens in tokenized for term in set(tokens))
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, tokens in enumerate(tokenized):
            for term, frequency in Counter(tokens).items():
                self.postings[term].append((index, frequency))
        count = len(self.examples)
        self.inverse_frequency = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def retrieve(
        self, text: str, aspect: str, count: int, exclude_record_id: str
    ) -> list[FewShotAspect]:
        query_terms = set(
            _RETRIEVAL_TOKEN.findall(text.lower())
            + _RETRIEVAL_TOKEN.findall(aspect.lower()) * 3
        )
        scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            inverse_frequency = self.inverse_frequency.get(term)
            if inverse_frequency is None:
                continue
            for index, frequency in self.postings[term]:
                example = self.examples[index]
                if example.record.record_id == exclude_record_id:
                    continue
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * self.lengths[index] / self.average_length
                )
                scores[index] += inverse_frequency * frequency * 2.5 / denominator
        ranked = sorted(
            scores,
            key=lambda index: (-scores[index], self.examples[index].record.record_id),
        )
        selected: list[FewShotAspect] = []
        used_records: set[str] = set()
        for index in ranked:
            example = self.examples[index]
            if example.record.record_id in used_records:
                continue
            selected.append(example)
            used_records.add(example.record.record_id)
            if len(selected) == count:
                return selected
        for example in self.examples:
            if (
                example.record.record_id != exclude_record_id
                and example.record.record_id not in used_records
            ):
                selected.append(example)
                used_records.add(example.record.record_id)
                if len(selected) == count:
                    return selected
        raise ValueError("Not enough distinct Train records for Few-shot retrieval")


def _format_few_shot(examples: list[FewShotAspect]) -> str:
    blocks = ["Training examples:"]
    for number, example in enumerate(examples, start=1):
        record, index = example.record, example.aspect_index
        if record.gold_scores is None:
            raise ValueError("Few-shot example has no gold VA")
        v, a = record.gold_scores[index]
        aspect = "[IMPLICIT_ASPECT]" if record.aspects[index] == "NULL" else record.aspects[index]
        blocks.append(
            f"Example {number}\nText: {record.text}\nAspect: {aspect}\nVA: {v:.2f}#{a:.2f}"
        )
    return "\n\n".join(blocks)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _last_overlap_mask(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    mask = [int(right > start and left < end) for left, right in offsets]
    if not any(mask):
        raise ValueError("Target aspect was truncated or could not be tokenized")
    return mask


class AspectRegressionDataset(Dataset):
    def __init__(
        self,
        records: list[Task1Record],
        tokenizer,
        max_length: int,
        representation: str,
        limit: int | None = None,
        few_shot_retriever: AspectFewShotRetriever | None = None,
        few_shot_examples: int = 0,
    ) -> None:
        self.items: list[dict] = []
        for record in records:
            if record.gold_scores is None:
                raise ValueError(f"ID {record.record_id!r}: VA scores are required")
            for aspect_index, (aspect, score) in enumerate(
                zip(record.aspects, record.gold_scores)
            ):
                implicit = aspect == "NULL"
                target = "[IMPLICIT_ASPECT]" if implicit else aspect
                query = f"Now predict this target:\nText: {record.text}\nAspect: {target}"
                if few_shot_examples:
                    if few_shot_retriever is None:
                        raise ValueError("Few-shot examples require a Train retriever")
                    demonstrations = few_shot_retriever.retrieve(
                        record.text, target, few_shot_examples, record.record_id
                    )
                    content = f"{_format_few_shot(demonstrations)}\n\n{query}"
                else:
                    content = query
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Represent the sentiment toward the specified aspect. "
                            "Valence and Arousal are predicted by regression heads."
                        ),
                    },
                    {"role": "user", "content": content},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                encoded = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offsets = encoded.pop("offset_mapping")
                if representation == "target-aware":
                    target_start = prompt.rfind(target)
                    if target_start < 0:
                        raise ValueError(f"Target {target!r} is missing from prompt")
                    target_mask = _last_overlap_mask(
                        offsets, target_start, target_start + len(target)
                    )
                else:
                    target_mask = [0] * len(encoded["input_ids"])
                self.items.append(
                    {
                        **encoded,
                        "target_mask": target_mask,
                        "implicit": implicit,
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
        keys = [item["key"] for item in features]
        labels = torch.tensor([item["labels"] for item in features], dtype=torch.float32)
        implicit = torch.tensor([item["implicit"] for item in features], dtype=torch.bool)
        target_masks = [item["target_mask"] for item in features]
        model_features = [
            {k: v for k, v in item.items() if k not in {"key", "labels", "implicit", "target_mask"}}
            for item in features
        ]
        batch = self.tokenizer.pad(
            model_features, padding=True, pad_to_multiple_of=8, return_tensors="pt"
        )
        padded = torch.zeros_like(batch["attention_mask"])
        for row, mask in enumerate(target_masks):
            padded[row, : len(mask)] = torch.tensor(mask)
        batch.update(labels=labels, keys=keys, target_mask=padded, implicit=implicit)
        return batch


class Task1Regressor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        representation: str,
        head_type: str,
        output_mode: str,
        loss_type: str = "mse",
        dropout: float = 0.1,
        attention_dim: int = 256,
        fusion_dim: int = 512,
        head_hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.representation = representation
        self.output_mode = output_mode
        if loss_type == "logsigma":
            initial_variance = torch.empty(2).uniform_(0.2, 1.0)
            self.log_vars = nn.Parameter(torch.log(initial_variance))
        else:
            self.register_parameter("log_vars", None)
        rep_size = hidden_size
        if representation == "target-aware":
            self.query = nn.Linear(hidden_size, attention_dim, bias=False)
            self.key = nn.Linear(hidden_size, attention_dim, bias=False)
            self.fusion = nn.Sequential(
                nn.Linear(hidden_size * 4, fusion_dim),
                nn.GELU(),
                nn.LayerNorm(fusion_dim),
                nn.Dropout(dropout),
            )
            rep_size = fusion_dim
        if head_type == "shared":
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(rep_size, 2))
            self.v_head = self.a_head = None
        else:
            def make_head() -> nn.Sequential:
                return nn.Sequential(
                    nn.LayerNorm(rep_size),
                    nn.Linear(rep_size, head_hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_size, 1),
                )

            self.head = None
            self.v_head, self.a_head = make_head(), make_head()

    def initialize_linear_bias(self, means: tuple[float, float]) -> None:
        if self.output_mode != "linear":
            return
        with torch.no_grad():
            if self.head is not None:
                self.head[-1].bias.copy_(torch.tensor(means))
            else:
                self.v_head[-1].bias.fill_(means[0])
                self.a_head[-1].bias.fill_(means[1])

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = attention_mask.sum(dim=1) - 1
        last = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        if self.representation == "target-aware":
            target_weights = target_mask.unsqueeze(-1).to(hidden.dtype)
            if torch.any(target_weights.sum(dim=1) == 0):
                raise ValueError("target-aware representation received an empty target mask")
            target = (hidden * target_weights).sum(dim=1) / target_weights.sum(dim=1)
            sequence_weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
            global_mean = (hidden * sequence_weights).sum(dim=1) / sequence_weights.sum(dim=1)
            scores = torch.einsum("bld,bd->bl", self.key(hidden), self.query(target))
            scores = scores / math.sqrt(self.query.out_features)
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores.float(), dim=1).to(hidden.dtype)
            context = torch.einsum("bl,blh->bh", attention, hidden)
            representation = self.fusion(torch.cat([last, target, global_mean, context], dim=-1))
        else:
            representation = last
        if self.head is not None:
            raw = self.head(representation)
        else:
            raw = torch.cat([self.v_head(representation), self.a_head(representation)], dim=-1)
        return 1.0 + 8.0 * torch.sigmoid(raw) if self.output_mode == "sigmoid" else raw


def _extract_text_backbone(model_name: str, train: bool):
    config = AutoConfig.from_pretrained(model_name)
    removed_modalities: list[str] = []
    if config.model_type == "gemma4":
        full = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model = full.model.language_model
        full.model.language_model = None
        removed_modalities = ["vision_tower", "audio_tower", "multi_modal_projector"]
        del full
        gc.collect()
    elif config.model_type == "qwen3_5":
        full = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model = full.model.language_model
        full.model.language_model = None
        removed_modalities = ["visual"]
        del full
        gc.collect()
    else:
        model = AutoModel.from_pretrained(
            model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
    model.config.use_cache = not train
    return model, removed_modalities, config.model_type


def load_train_model(model_name: str, lora_scope: str):
    model, removed, model_type = _extract_text_backbone(model_name, train=True)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if model_type == "qwen3_5":
        targets.extend(
            ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
        )
    if lora_scope == "attention-mlp":
        targets.extend(["gate_proj", "up_proj", "down_proj"])
    lora = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model, removed, model_type


def _forward_hidden(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
    ).last_hidden_state


def metric_dict(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    error = scores - labels
    result = {
        "rmse_va": float(torch.sqrt(torch.square(error).sum() / len(labels))),
        "rmse_v": float(torch.sqrt(torch.square(error[:, 0]).mean())),
        "rmse_a": float(torch.sqrt(torch.square(error[:, 1]).mean())),
    }
    for index, name in enumerate(("pcc_v", "pcc_a")):
        x, y = scores[:, index].double(), labels[:, index].double()
        x, y = x - x.mean(), y - y.mean()
        denominator = torch.sqrt(torch.square(x).sum() * torch.square(y).sum())
        result[name] = float((x * y).sum() / denominator) if denominator > 0 else 0.0
    return result


@torch.inference_mode()
def predict(model, regressor, loader, device):
    model.eval()
    regressor.eval()
    all_scores, all_labels, keys = [], [], []
    for batch in loader:
        labels = batch.pop("labels")
        keys.extend(batch.pop("keys"))
        batch.pop("implicit")
        target_mask = batch.pop("target_mask").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = regressor(
                _forward_hidden(model, inputs), inputs["attention_mask"], target_mask
            ).float()
        all_scores.append(scores.clamp(1.0, 9.0).cpu())
        all_labels.append(labels)
    scores, labels = torch.cat(all_scores), torch.cat(all_labels)
    return scores.tolist(), keys, metric_dict(scores, labels)


def prediction_map(records, values, keys):
    collected = {r.record_id: [None] * len(r.aspects) for r in records}
    for key, value in zip(keys, values):
        collected[key.record_id][key.aspect_index] = tuple(map(float, value))
    if any(value is None for scores in collected.values() for value in scores):
        raise ValueError("Incomplete predictions")
    return {key: tuple(values) for key, values in collected.items()}


def _save_config(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "train", "predict"], default="smoke")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-file")
    parser.add_argument("--dev-file")
    parser.add_argument("--input-file")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-pred")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2, help="Evaluation events without improvement")
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--few-shot-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representation", choices=["last-token", "target-aware"], default="last-token")
    parser.add_argument("--head-type", choices=["shared", "independent"], default="shared")
    parser.add_argument("--output-mode", choices=["sigmoid", "linear"], default="sigmoid")
    parser.add_argument(
        "--loss-type", choices=["mse", "mse-huber", "logsigma"], default="mse"
    )
    parser.add_argument("--huber-weight", type=float, default=0.1)
    parser.add_argument("--logsigma-learning-rate", type=float, default=5e-2)
    parser.add_argument("--lora-scope", choices=["attention", "attention-mlp"], default="attention")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode == "train" and os.environ.get("CONFIRM_FULL_RUN") != "YES":
        raise RuntimeError("Full paid-GPU training requires CONFIRM_FULL_RUN=YES")
    if args.mode in {"smoke", "train"} and (not args.train_file or not args.dev_file):
        raise ValueError("Training requires --train-file and --dev-file")
    if args.mode == "predict" and (not args.input_file or not args.output_pred):
        raise ValueError("Prediction requires --input-file and --output-pred")
    if args.few_shot_examples < 0:
        raise ValueError("--few-shot-examples cannot be negative")


def _build_regressor(hidden_size: int, cfg: dict) -> Task1Regressor:
    return Task1Regressor(
        hidden_size,
        cfg["representation"],
        cfg["head_type"],
        cfg["output_mode"],
        cfg.get("loss_type", "mse"),
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    started = time.time()
    adapter_dir = Path(args.adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = (
        "left" if args.representation == "target-aware" or args.few_shot_examples else "right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "predict":
        cfg = json.loads((adapter_dir / "experiment_config.json").read_text())
        few_shot_count = int(cfg.get("few_shot_examples", 0))
        tokenizer.truncation_side = (
            "left" if cfg["representation"] == "target-aware" or few_shot_count else "right"
        )
        records = load_task1_records(args.input_file, require_gold=True)
        retriever = None
        if few_shot_count:
            source = cfg.get("few_shot_source") or cfg.get("train_file")
            if not source:
                raise ValueError("Saved Few-shot model has no Train retrieval source")
            retriever = AspectFewShotRetriever(
                load_task1_records(source, require_gold=True)
            )
        dataset = AspectRegressionDataset(
            records,
            tokenizer,
            cfg["max_length"],
            cfg["representation"],
            few_shot_retriever=retriever,
            few_shot_examples=few_shot_count,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size * 2, collate_fn=RegressionCollator(tokenizer))
        base, _, _ = _extract_text_backbone(args.model_name, train=False)
        model = PeftModel.from_pretrained(base, adapter_dir).to(device)
        regressor = _build_regressor(model.config.hidden_size, cfg).to(device)
        regressor.load_state_dict(torch.load(adapter_dir / "regression_model.pt", map_location=device))
        values, keys, metrics = predict(model, regressor, loader, device)
        write_task1_predictions(records, prediction_map(records, values, keys), args.output_pred)
        print(json.dumps({**metrics, "records": len(records), "aspects": len(dataset)}))
        return

    train_records = load_task1_records(args.train_file, require_gold=True)
    dev_records = load_task1_records(args.dev_file, require_gold=True)
    few_shot_retriever = (
        AspectFewShotRetriever(train_records) if args.few_shot_examples else None
    )
    limit = 64 if args.mode == "smoke" else None
    train_dataset = AspectRegressionDataset(
        train_records,
        tokenizer,
        args.max_length,
        args.representation,
        limit,
        few_shot_retriever,
        args.few_shot_examples,
    )
    dev_dataset = AspectRegressionDataset(
        dev_records,
        tokenizer,
        args.max_length,
        args.representation,
        32 if args.mode == "smoke" else None,
        few_shot_retriever,
        args.few_shot_examples,
    )
    collator = RegressionCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size * 2, collate_fn=collator)
    model, removed_modalities, model_type = load_train_model(args.model_name, args.lora_scope)
    model = model.to(device)
    config = vars(args).copy()
    config.update(model_type=model_type, removed_modalities=removed_modalities, hidden_size=model.config.hidden_size)
    config["few_shot_source"] = args.train_file if args.few_shot_examples else None
    regressor = _build_regressor(model.config.hidden_size, config).to(device)
    labels = torch.tensor([item["labels"] for item in train_dataset.items])
    means = tuple(map(float, labels.mean(dim=0)))
    regressor.initialize_linear_bias(means)
    config["train_label_means"] = means
    if args.representation == "target-aware":
        if any(not any(item["target_mask"]) for item in train_dataset.items):
            raise AssertionError("Explicit target mask validation failed")
        if not any(item["implicit"] for item in train_dataset.items):
            raise AssertionError("Smoke/train sample did not exercise implicit aspect branch")
    lora_params = [p for p in model.parameters() if p.requires_grad]
    head_params = [
        parameter
        for name, parameter in regressor.named_parameters()
        if name != "log_vars"
    ]
    parameter_groups = [
        {"params": lora_params, "lr": args.learning_rate},
        {"params": head_params, "lr": args.head_learning_rate},
    ]
    if regressor.log_vars is not None:
        parameter_groups.append(
            {
                "params": [regressor.log_vars],
                "lr": args.logsigma_learning_rate,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)
    epochs = 1 if args.mode == "smoke" else args.epochs
    updates = math.ceil(len(train_loader) / args.grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(updates * 0.05)), updates)
    eval_interval = max(1, math.ceil(len(train_loader) / args.evals_per_epoch))
    adapter_dir.mkdir(parents=True, exist_ok=True)
    _save_config(adapter_dir / "experiment_config.json", config)
    print(json.dumps({"mode": args.mode, "train_aspects": len(train_dataset), "dev_aspects": len(dev_dataset), "removed_modalities": removed_modalities, "trainable_parameters": sum(p.numel() for p in lora_params) + sum(p.numel() for p in regressor.parameters())}), flush=True)
    best, bad_evals, global_step, optimizer_step = float("inf"), 0, 0, 0
    history: list[dict] = []
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(epochs):
        model.train()
        regressor.train()
        loss_sum = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch.pop("keys")
            batch.pop("implicit")
            target_mask = batch.pop("target_mask").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = regressor(_forward_hidden(model, inputs), inputs["attention_mask"], target_mask).float()
                mse = F.mse_loss(scores, labels)
                if args.loss_type == "mse-huber":
                    loss = mse + args.huber_weight * F.huber_loss(scores, labels)
                elif args.loss_type == "logsigma":
                    mse_by_dimension = torch.square(scores - labels).mean(dim=0)
                    loss = 0.5 * torch.sum(
                        torch.exp(-regressor.log_vars) * mse_by_dimension
                        + regressor.log_vars
                    )
                else:
                    loss = mse
            (loss / args.grad_accum).backward()
            loss_sum += float(loss.detach())
            global_step += 1
            if batch_index % args.grad_accum == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_([*lora_params, *regressor.parameters()], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            should_eval = batch_index % eval_interval == 0 or batch_index == len(train_loader)
            if not should_eval:
                continue
            values, keys, metrics = predict(model, regressor, dev_loader, device)
            row = {"epoch": epoch + batch_index / len(train_loader), "global_step": global_step, "optimizer_step": optimizer_step, "train_loss": loss_sum / batch_index, **metrics}
            if regressor.log_vars is not None:
                log_vars = regressor.log_vars.detach().float().cpu()
                row["log_variance_v"] = float(log_vars[0])
                row["log_variance_a"] = float(log_vars[1])
                row["precision_weight_v"] = float(torch.exp(-log_vars[0]))
                row["precision_weight_a"] = float(torch.exp(-log_vars[1]))
            history.append(row)
            print(json.dumps(row), flush=True)
            if metrics["rmse_va"] < best:
                best, bad_evals = metrics["rmse_va"], 0
                model.save_pretrained(adapter_dir)
                torch.save(regressor.state_dict(), adapter_dir / "regression_model.pt")
                if len(dev_dataset) == sum(len(r.aspects) for r in dev_records):
                    write_task1_predictions(dev_records, prediction_map(dev_records, values, keys), adapter_dir / "best_dev_predictions.jsonl")
                config["best_step"] = global_step
                _save_config(adapter_dir / "experiment_config.json", config)
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
    reload_metrics = None
    if args.mode == "smoke":
        del optimizer, scheduler, model, regressor
        gc.collect()
        torch.cuda.empty_cache()
        base, _, _ = _extract_text_backbone(args.model_name, train=False)
        reloaded_model = PeftModel.from_pretrained(base, adapter_dir).to(device)
        reloaded_regressor = _build_regressor(
            reloaded_model.config.hidden_size, config
        ).to(device)
        reloaded_regressor.load_state_dict(
            torch.load(adapter_dir / "regression_model.pt", map_location=device)
        )
        reloaded_values, _, reload_metrics = predict(
            reloaded_model, reloaded_regressor, dev_loader, device
        )
        if len(reloaded_values) != len(dev_dataset):
            raise AssertionError("Reloaded smoke model produced a wrong prediction count")
        if any(value < 1.0 or value > 9.0 for row in reloaded_values for value in row):
            raise AssertionError("Reloaded smoke predictions are outside [1, 9]")
    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "best_dev_rmse_va": best,
        "best_step": config.get("best_step"),
        "history": history,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "removed_modalities": removed_modalities,
        "reload_metrics": reload_metrics,
    }
    _save_config(adapter_dir / "training_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
