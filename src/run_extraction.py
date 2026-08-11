"""Cost-guarded Qwen inference for joint DimABSA Tasks 2 and 3."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from json import JSONDecoder
from pathlib import Path
from typing import Any, Sequence

from dimabsa_data import select_smoke_records
from dimabsa_extraction import (
    ExtractionItem,
    ExtractionRecord,
    load_extraction_records,
    parse_extraction_payload,
    select_extraction_examples,
    write_extraction_predictions,
)
from dimabsa_extraction_prompts import build_extraction_messages
from run_instruct import _generate, _load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "resources"
    / "DimABSA2026"
    / "task-dataset"
    / "track_a"
    / "subtask_3"
    / "zho"
)
LOGGER = logging.getLogger(Path(__file__).name)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen joint Task 2/3 extraction; Task 2 is derived from Task 3"
    )
    parser.add_argument(
        "--prompt-mode", choices=["direct", "cot", "fewshot"], default="fewshot"
    )
    parser.add_argument(
        "--run-mode",
        choices=["smoke", "full"],
        default=os.environ.get("RUN_MODE", "smoke").lower(),
    )
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_DATA_ROOT / "zho_restaurant_dev_task3.jsonl"),
    )
    parser.add_argument(
        "--train-file",
        default=str(DEFAULT_DATA_ROOT / "zho_restaurant_train_alltasks.jsonl"),
    )
    parser.add_argument("--result-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507"),
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "unsloth"],
        default=os.environ.get("MODEL_BACKEND", "transformers"),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--smoke-samples", type=int, default=8)
    parser.add_argument("--few-shot-examples", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-format-retries", type=int, default=2)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.run_mode == "full" and os.environ.get("CONFIRM_FULL_RUN") != "YES":
        raise RuntimeError(
            "Full paid-GPU inference requires RUN_MODE=full and CONFIRM_FULL_RUN=YES"
        )
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.smoke_samples < 1 or args.few_shot_examples < 1:
        raise ValueError("sample counts must be positive")
    if args.max_seq_length < 512 or args.max_new_tokens < 64:
        raise ValueError("token limits are unexpectedly small")


def _json_candidates(text: str) -> list[Any]:
    decoder = JSONDecoder()
    values = []
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values


def parse_model_output(
    text: str, source_text: str
) -> tuple[tuple[ExtractionItem, ...], list[str]]:
    errors = []
    candidates = _json_candidates(text)
    for candidate in reversed(candidates):
        try:
            return parse_extraction_payload(candidate, source_text)
        except ValueError as exc:
            errors.append(str(exc))
    # A very long answer can hit max_new_tokens after several complete item
    # objects but before the outer list closes. Salvage only those complete
    # objects; each item still passes exact-span/category/VA validation.
    complete_items = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and any(key in candidate for key in ("aspect", "Aspect"))
        and any(key in candidate for key in ("opinion", "Opinion"))
    ]
    if complete_items:
        return parse_extraction_payload(complete_items, source_text)
    detail = errors[-1] if errors else "no JSON object or array was found"
    raise ValueError(f"Could not parse model output: {detail}")


def _format_batch(
    tokenizer,
    records: Sequence[ExtractionRecord],
    *,
    prompt_mode: str,
    examples: Sequence[ExtractionRecord],
) -> list[str]:
    conversations = [
        build_extraction_messages(
            record, prompt_mode=prompt_mode, examples=examples
        )
        for record in records
    ]
    return tokenizer.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=True
    )


def _retry_prompt(
    tokenizer,
    record: ExtractionRecord,
    *,
    prompt_mode: str,
    examples: Sequence[ExtractionRecord],
    previous_output: str,
    parse_error: str,
) -> str:
    messages = build_extraction_messages(
        record, prompt_mode=prompt_mode, examples=examples
    )
    messages.extend(
        [
            {"role": "assistant", "content": previous_output},
            {
                "role": "user",
                "content": (
                    f"上面的 JSON 格式无法解析：{parse_error}。"
                    "请只重新输出合法 JSON：{\"items\":[...]}，不要解释。"
                ),
            },
        ]
    )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    started = time.perf_counter()
    input_records = load_extraction_records(args.input_file, require_gold=False)
    train_records = load_extraction_records(args.train_file, require_gold=True)
    records = (
        select_smoke_records(input_records, args.smoke_samples)
        if args.run_mode == "smoke"
        else input_records
    )
    examples = (
        select_extraction_examples(train_records, args.few_shot_examples)
        if args.prompt_mode == "fewshot"
        else []
    )
    batch_size = args.batch_size or (2 if args.run_mode == "smoke" else 8)
    LOGGER.info(
        "mode=%s prompt=%s rows=%s batch=%s examples=%s",
        args.run_mode,
        args.prompt_mode,
        len(records),
        batch_size,
        len(examples),
    )
    if args.run_mode == "smoke":
        LOGGER.warning("Smoke output is only a pipeline check, not a formal result")

    model, tokenizer, torch = _load_model(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    predictions: dict[str, tuple[ExtractionItem, ...]] = {}
    diagnostics = []
    order = sorted(
        range(len(records)), key=lambda index: len(records[index].text), reverse=True
    )
    for start in range(0, len(order), batch_size):
        batch_records = [records[index] for index in order[start : start + batch_size]]
        prompts = _format_batch(
            tokenizer,
            batch_records,
            prompt_mode=args.prompt_mode,
            examples=examples,
        )
        raw_outputs = _generate(
            model,
            tokenizer,
            torch,
            prompts,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
        )
        for record, initial_output in zip(batch_records, raw_outputs):
            current_output = initial_output
            attempt_outputs = [initial_output]
            parse_error = None
            invalid_items: list[str] = []
            for attempt in range(args.max_format_retries + 1):
                try:
                    items, invalid_items = parse_model_output(
                        current_output, record.text
                    )
                    status = "ok" if attempt == 0 else "ok_after_format_retry"
                    break
                except ValueError as exc:
                    parse_error = str(exc)
                    if attempt >= args.max_format_retries:
                        items = ()
                        status = "fallback_empty"
                        break
                    prompt = _retry_prompt(
                        tokenizer,
                        record,
                        prompt_mode=args.prompt_mode,
                        examples=examples,
                        previous_output=current_output,
                        parse_error=parse_error,
                    )
                    current_output = _generate(
                        model,
                        tokenizer,
                        torch,
                        [prompt],
                        max_seq_length=args.max_seq_length,
                        max_new_tokens=args.max_new_tokens,
                    )[0]
                    attempt_outputs.append(current_output)
            predictions[record.record_id] = items
            diagnostics.append(
                {
                    "ID": record.record_id,
                    "status": status,
                    "parse_error": parse_error,
                    "invalid_items": invalid_items,
                    "format_retry_count": len(attempt_outputs) - 1,
                    "attempt_outputs": attempt_outputs,
                    "valid_item_count": len(items),
                }
            )
        LOGGER.info("Inference progress: %s/%s", min(start + batch_size, len(order)), len(order))

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    input_stem = Path(args.input_file).stem
    base_stem = f"{input_stem}_{args.prompt_mode}_{args.run_mode}"
    task3_path = result_dir / f"{base_stem}.jsonl"
    task2_path = result_dir / f"{base_stem.replace('task3', 'task2')}_derived.jsonl"
    diagnostic_path = result_dir / f"{base_stem}_diagnostics.jsonl"
    metadata_path = result_dir / f"{base_stem}_metadata.json"
    write_extraction_predictions(records, predictions, task3_path, task2_path)
    with diagnostic_path.open("w", encoding="utf-8") as handle:
        for item in diagnostics:
            json.dump(item, handle, ensure_ascii=False)
            handle.write("\n")
    metadata = {
        "input_file": str(Path(args.input_file).resolve()),
        "train_file": str(Path(args.train_file).resolve()),
        "model_name": args.model_name,
        "backend": args.backend,
        "prompt_mode": args.prompt_mode,
        "run_mode": args.run_mode,
        "records": len(records),
        "predicted_task3_items": sum(len(items) for items in predictions.values()),
        "parse_failures": sum(item["status"] == "fallback_empty" for item in diagnostics),
        "format_retry_recoveries": sum(
            item["status"] == "ok_after_format_retry" for item in diagnostics
        ),
        "discarded_invalid_items": sum(
            len(item["invalid_items"]) for item in diagnostics
        ),
        "batch_size": batch_size,
        "few_shot_examples": len(examples),
        "few_shot_example_ids": [record.record_id for record in examples],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "task3_prediction_file": str(task3_path.resolve()),
        "task2_prediction_file": str(task2_path.resolve()),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Saved Task 3 %s and derived Task 2 %s; parse_failures=%s; discarded_items=%s",
        task3_path,
        task2_path,
        metadata["parse_failures"],
        metadata["discarded_invalid_items"],
    )


if __name__ == "__main__":
    main()
