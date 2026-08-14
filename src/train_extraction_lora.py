"""Joint QLoRA generation training for English DimABSA Tasks 2 and 3."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from json import JSONDecoder
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from dimabsa_extraction import (
    ExtractionItem,
    ExtractionRecord,
    RESTAURANT_CATEGORIES,
    load_extraction_records,
    parse_extraction_payload,
    write_extraction_predictions,
)
from evaluate_extraction import evaluate
from extraction_hybrid import BM25Retriever, recover_payload_spans


SYSTEM = """You extract dimensional aspect sentiment quadruplets from restaurant reviews.
Return exact aspect and opinion spans, a valid restaurant category, and continuous Valence/Arousal scores."""

CATEGORIES = ", ".join(RESTAURANT_CATEGORIES)


def user_prompt(text: str) -> str:
    return f"""Extract every (aspect, category, opinion, V, A) relation from the text.
- Copy explicit aspect and opinion spans exactly from the text.
- Use the literal string NULL only when an aspect or opinion is implicit.
- V and A are continuous values in [1,9].
- Valid categories: {CATEGORIES}
- Output only JSON: {{"items":[{{"aspect":"...","opinion":"...","category":"FOOD#QUALITY","V":"7.25","A":"6.50"}}]}}
- If there is no relation, output {{"items":[]}}.
Text: {text}"""


def gold_answer(record: ExtractionRecord) -> str:
    if record.gold_items is None:
        raise ValueError("Training record has no gold items")
    items = [
        {
            "aspect": item.aspect,
            "opinion": item.opinion,
            "category": item.category,
            "V": f"{item.score[0]:.2f}",
            "A": f"{item.score[1]:.2f}",
        }
        for item in record.gold_items
    ]
    return json.dumps({"items": items}, ensure_ascii=False, separators=(",", ":"))


def prompt_messages(
    record: ExtractionRecord,
    examples: list[ExtractionRecord] | None = None,
) -> list[dict[str, str]]:
    """Build optional retrieval demonstrations without exposing query labels."""

    messages = [{"role": "system", "content": SYSTEM}]
    for example in examples or []:
        messages.extend(
            [
                {"role": "user", "content": user_prompt(example.text)},
                {"role": "assistant", "content": gold_answer(example)},
            ]
        )
    messages.append({"role": "user", "content": user_prompt(record.text)})
    return messages


class ExtractionSFTDataset(Dataset):
    def __init__(
        self,
        records: list[ExtractionRecord],
        tokenizer,
        max_length: int,
        limit: int | None = None,
    ) -> None:
        self.items = []
        for record in records:
            messages = prompt_messages(record)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, truncation=False
            ).input_ids
            answer_ids = tokenizer(
                gold_answer(record), add_special_tokens=False, truncation=False
            ).input_ids + [tokenizer.eos_token_id]
            input_ids = (prompt_ids + answer_ids)[:max_length]
            answer_start = min(len(prompt_ids), max_length)
            labels = [-100] * answer_start + input_ids[answer_start:]
            if not any(label != -100 for label in labels):
                raise ValueError(f"ID {record.record_id!r}: prompt consumes max_length")
            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )
            if limit is not None and len(self.items) >= limit:
                break

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


class SFTCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        maximum = math.ceil(maximum / 8) * 8
        input_ids = []
        attention_masks = []
        labels = []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_quantized_base(
    model_name: str, *, train: bool, lora_scope: str = "attention"
):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = not train
    if train:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if lora_scope == "attention_mlp":
            target_modules.extend(["gate_proj", "up_proj", "down_proj"])
        elif lora_scope != "attention":
            raise ValueError("lora_scope must be attention or attention_mlp")
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=target_modules,
                bias="none",
            ),
        )
    return model


def _json_candidates(text: str) -> list[object]:
    decoder = JSONDecoder()
    output = []
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            output.append(value)
        except json.JSONDecodeError:
            pass
    return output


def parse_generation(text: str, source_text: str) -> tuple[ExtractionItem, ...]:
    for candidate in reversed(_json_candidates(text)):
        try:
            items, _ = parse_extraction_payload(
                recover_payload_spans(candidate, source_text),
                source_text,
                allow_null=True,
            )
            return items
        except ValueError:
            continue
    return ()


@torch.inference_mode()
def generate_predictions(
    model,
    tokenizer,
    records: list[ExtractionRecord],
    *,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    examples_by_id: dict[str, list[ExtractionRecord]] | None = None,
    temperature: float = 0.0,
    diagnostics_path: str | Path | None = None,
) -> tuple[dict[str, tuple[ExtractionItem, ...]], int]:
    model.eval()
    previous_padding_side = tokenizer.padding_side
    previous_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    predictions = {}
    failures = 0
    diagnostics = []
    order = sorted(range(len(records)), key=lambda index: len(records[index].text), reverse=True)
    for start in range(0, len(order), batch_size):
        batch_records = [records[index] for index in order[start : start + batch_size]]
        prompts = [
            tokenizer.apply_chat_template(
                prompt_messages(
                    record,
                    None if examples_by_id is None else examples_by_id[record.record_id],
                ),
                tokenize=False,
                add_generation_prompt=True,
            )
            for record in batch_records
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)
        width = encoded["input_ids"].shape[1]
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generation_args["temperature"] = temperature
        generated = model.generate(**encoded, **generation_args)
        texts = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
        for record, text in zip(batch_records, texts):
            items = parse_generation(text, record.text)
            if not _json_candidates(text):
                failures += 1
            predictions[record.record_id] = items
            diagnostics.append(
                {
                    "ID": record.record_id,
                    "raw_output": text,
                    "parsed_items": len(items),
                    "has_json_candidate": bool(_json_candidates(text)),
                }
            )
    tokenizer.padding_side = previous_padding_side
    tokenizer.truncation_side = previous_truncation_side
    if diagnostics_path is not None:
        destination = Path(diagnostics_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for row in diagnostics:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return predictions, failures


def evaluate_dev(
    model,
    tokenizer,
    records: list[ExtractionRecord],
    task2_gold: str,
    task3_gold: str,
    output_dir: Path,
    epoch: int,
    batch_size: int,
    max_new_tokens: int,
    allow_subset: bool = False,
) -> dict:
    predictions, failures = generate_predictions(
        model,
        tokenizer,
        records,
        batch_size=batch_size,
        max_input_length=512,
        max_new_tokens=max_new_tokens,
    )
    task3_path = output_dir / f"dev_epoch{epoch}_task3.jsonl"
    task2_path = output_dir / f"dev_epoch{epoch}_task2.jsonl"
    write_extraction_predictions(records, predictions, task3_path)
    task2_records = load_extraction_records(task2_gold, require_gold=True)
    if allow_subset:
        task2_records = task2_records[: len(records)]
    if len(task2_records) != len(records) or any(
        left.text != right.text for left, right in zip(task2_records, records)
    ):
        raise ValueError("Task 2 and Task 3 templates are not text-aligned")
    task2_predictions = {
        task2_record.record_id: predictions[task3_record.record_id]
        for task2_record, task3_record in zip(task2_records, records)
    }
    task2_dummy = output_dir / f"dev_epoch{epoch}_task2_quadruplet_view.jsonl"
    write_extraction_predictions(
        task2_records, task2_predictions, task2_dummy, task2_path
    )
    task2_dummy.unlink()
    metrics2 = evaluate(
        task2_gold, task2_path, task=2, allow_subset=allow_subset
    )
    metrics3 = evaluate(
        task3_gold, task3_path, task=3, allow_subset=allow_subset
    )
    return {
        "parse_failures": failures,
        "task2_continuous_f1": metrics2["continuous_F1"],
        "task3_continuous_f1": metrics3["continuous_F1"],
        "mean_continuous_f1": (
            metrics2["continuous_F1"] + metrics3["continuous_F1"]
        )
        / 2.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint Task 2/3 English QLoRA")
    parser.add_argument("--mode", choices=["smoke", "train", "predict"], default="smoke")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-file")
    parser.add_argument("--dev-task2-file")
    parser.add_argument("--dev-task3-file")
    parser.add_argument("--input-file")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-task2")
    parser.add_argument("--output-task3")
    parser.add_argument("--task2-template-file")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--retrieval-train-file")
    parser.add_argument("--retrieval-map-file")
    parser.add_argument("--retrieval-k", type=int, default=0)
    parser.add_argument(
        "--retrieval-variant",
        choices=("word", "bigram", "trigram"),
        default="word",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--diagnostics")
    parser.add_argument(
        "--lora-scope",
        choices=("attention", "attention_mlp"),
        default="attention",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if args.mode == "train" and os.environ.get("CONFIRM_FULL_RUN") != "YES":
        raise RuntimeError("Full paid-GPU training requires CONFIRM_FULL_RUN=YES")
    if args.mode in {"smoke", "train"} and not all(
        (args.train_file, args.dev_task2_file, args.dev_task3_file)
    ):
        raise ValueError("Training requires Train and both Dev files")
    if args.mode == "predict" and not all(
        (
            args.input_file,
            args.task2_template_file,
            args.output_task2,
            args.output_task3,
        )
    ):
        raise ValueError("Prediction requires input and Task 2/3 outputs")
    if args.retrieval_map_file and not args.retrieval_train_file:
        raise ValueError("A retrieval map still requires --retrieval-train-file")
    if args.retrieval_map_file and args.retrieval_k:
        raise ValueError("Use either --retrieval-map-file or --retrieval-k, not both")
    if not args.retrieval_map_file and (
        (args.retrieval_k > 0) != bool(args.retrieval_train_file)
    ):
        raise ValueError("BM25 retrieval requires both --retrieval-k and --retrieval-train-file")
    if args.temperature < 0:
        raise ValueError("temperature cannot be negative")


def main() -> None:
    args = parse_args()
    check_args(args)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left" if args.mode == "predict" else "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "predict":
        records = load_extraction_records(args.input_file, require_gold=True)
        examples_by_id = None
        if args.retrieval_k or args.retrieval_map_file:
            retrieval_records = load_extraction_records(
                args.retrieval_train_file, require_gold=True
            )
            if args.retrieval_map_file:
                payload = json.loads(Path(args.retrieval_map_file).read_text())
                selections = payload.get("selections", payload)
                train_by_id = {record.record_id: record for record in retrieval_records}
                missing = [record.record_id for record in records if record.record_id not in selections]
                if missing:
                    raise ValueError(f"Retrieval map misses query IDs: {missing[:5]}")
                examples_by_id = {}
                for record in records:
                    selected_ids = selections[record.record_id]
                    if not selected_ids or any(key not in train_by_id for key in selected_ids):
                        raise ValueError(f"Invalid retrieval selection for {record.record_id!r}")
                    examples_by_id[record.record_id] = [train_by_id[key] for key in selected_ids]
            else:
                retriever = BM25Retriever(retrieval_records, args.retrieval_variant)
                examples_by_id = {
                    record.record_id: retriever.select(
                        record.text, args.retrieval_k, exclude_id=record.record_id
                    )
                    for record in records
                }
        base = load_quantized_base(args.model_name, train=False)
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        predictions, failures = generate_predictions(
            model,
            tokenizer,
            records,
            batch_size=args.eval_batch_size,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            examples_by_id=examples_by_id,
            temperature=args.temperature,
            diagnostics_path=args.diagnostics,
        )
        write_extraction_predictions(records, predictions, args.output_task3)
        task2_records = load_extraction_records(
            args.task2_template_file, require_gold=True
        )
        if len(task2_records) != len(records) or any(
            left.text != right.text for left, right in zip(task2_records, records)
        ):
            raise ValueError("Task 2 and Task 3 templates are not text-aligned")
        task2_predictions = {
            task2_record.record_id: predictions[task3_record.record_id]
            for task2_record, task3_record in zip(task2_records, records)
        }
        dummy = Path(args.output_task2).with_suffix(".quadruplet_view.tmp")
        write_extraction_predictions(
            task2_records, task2_predictions, dummy, args.output_task2
        )
        dummy.unlink()
        print(
            json.dumps(
                {
                    "records": len(records),
                    "parse_failures": failures,
                    "retrieval_k": (
                        len(next(iter(examples_by_id.values()))) if examples_by_id else 0
                    ),
                    "retrieval_variant": (
                        "map" if args.retrieval_map_file else args.retrieval_variant
                    ),
                    "temperature": args.temperature,
                }
            )
        )
        return

    train_records = load_extraction_records(args.train_file, require_gold=True)
    dev_records = load_extraction_records(args.dev_task3_file, require_gold=True)
    limit = 64 if args.mode == "smoke" else None
    epochs = 1 if args.mode == "smoke" else args.epochs
    dataset = ExtractionSFTDataset(
        train_records, tokenizer, args.max_length, limit=limit
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=SFTCollator(tokenizer),
        num_workers=0,
    )
    model = load_quantized_base(
        args.model_name, train=True, lora_scope=args.lora_scope
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    total_updates = math.ceil(len(loader) / args.grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * 0.05)),
        num_training_steps=total_updates,
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "train_records": len(dataset),
                "micro_batch": args.batch_size,
                "effective_batch": args.batch_size * args.grad_accum,
                "updates": total_updates,
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            }
        ),
        flush=True,
    )
    output_dir = Path(args.adapter_dir).parent / (Path(args.adapter_dir).name + "_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    bad_epochs = 0
    history = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            running += loss.item()
            if step % args.grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        dev_subset = dev_records[:16] if args.mode == "smoke" else dev_records
        result = evaluate_dev(
            model,
            tokenizer,
            dev_subset,
            args.dev_task2_file,
            args.dev_task3_file,
            output_dir,
            epoch,
            args.eval_batch_size,
            args.max_new_tokens,
            allow_subset=args.mode == "smoke",
        )
        row = {"epoch": epoch, "train_loss": running / len(loader), **result}
        history.append(row)
        print(json.dumps(row), flush=True)
        if result["mean_continuous_f1"] > best_score:
            best_score = result["mean_continuous_f1"]
            bad_epochs = 0
            adapter_dir = Path(args.adapter_dir)
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            model.save_pretrained(adapter_dir)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "lora_scope": args.lora_scope,
        "best_mean_continuous_f1": best_score,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
