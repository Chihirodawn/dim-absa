"""Cost-guarded Qwen3-4B-Instruct inference for DimABSA Track A / Task 1."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from json import JSONDecoder
from pathlib import Path
from typing import Any, Sequence

from dimabsa_data import (
    Score,
    Task1Record,
    load_task1_records,
    mean_gold_score,
    select_anchor_examples,
    select_similar_examples,
    select_smoke_records,
    write_task1_predictions,
)
from dimabsa_prompts import build_messages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "resources"
    / "DimABSA2026"
    / "task-dataset"
    / "track_a"
    / "subtask_1"
    / "zho"
)
DEFAULT_MODEL = os.environ.get(
    "MODEL_NAME",
    "Qwen/Qwen3-4B-Instruct-2507",
)
DEFAULT_BACKEND = os.environ.get("MODEL_BACKEND", "transformers")

LOGGER = logging.getLogger(Path(__file__).name)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-4B-Instruct direct/CoT/few-shot DimABSA inference"
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["mean", "direct", "cot", "fewshot", "dynamic_fewshot"],
        default="direct",
    )
    parser.add_argument(
        "--run-mode",
        choices=["smoke", "full"],
        default=os.environ.get("RUN_MODE", "smoke").lower(),
    )
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_DATA_ROOT / "zho_restaurant_dev_task1.jsonl"),
    )
    parser.add_argument(
        "--train-file",
        default=str(DEFAULT_DATA_ROOT / "zho_restaurant_train_alltasks.jsonl"),
    )
    parser.add_argument("--result-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--backend",
        choices=["transformers", "unsloth"],
        default=DEFAULT_BACKEND,
        help="Model loader. Transformers supports the official BF16 model directly.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--smoke-samples", type=int, default=8)
    parser.add_argument("--few-shot-examples", type=int, default=5)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-format-retries", type=int, default=2)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if (
        args.run_mode == "full"
        and args.prompt_mode != "mean"
        and os.environ.get("CONFIRM_FULL_RUN") != "YES"
    ):
        raise RuntimeError(
            "Full paid-GPU inference requires RUN_MODE=full and CONFIRM_FULL_RUN=YES"
        )
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.smoke_samples < 1:
        raise ValueError("smoke-samples must be positive")
    if args.max_seq_length < 128 or args.max_new_tokens < 8:
        raise ValueError("token limits are unexpectedly small")
    if args.max_format_retries < 0:
        raise ValueError("max-format-retries cannot be negative")


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


def _score_item(item: Any) -> Score:
    if isinstance(item, dict):
        if "VA" in item:
            item = item["VA"]
        else:
            v = item.get("V", item.get("Valence"))
            a = item.get("A", item.get("Arousal"))
            item = [v, a]
    if isinstance(item, str):
        if "#" in item:
            item = item.split("#")
        elif "," in item:
            item = item.split(",")
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        raise ValueError(f"score item must contain exactly V and A, got {item!r}")
    v, a = float(item[0]), float(item[1])
    if not math.isfinite(v) or not math.isfinite(a):
        raise ValueError("score contains a non-finite value")
    if not 1.0 <= v <= 9.0 or not 1.0 <= a <= 9.0:
        raise ValueError(f"score {(v, a)!r} is outside [1, 9]")
    return v, a


def _scores_from_payload(payload: Any, expected_count: int) -> tuple[Score, ...]:
    if isinstance(payload, dict):
        if "scores" in payload:
            payload = payload["scores"]
        elif "Aspect_VA" in payload:
            payload = payload["Aspect_VA"]
    if not isinstance(payload, list):
        raise ValueError("JSON payload does not contain a score list")
    scores = tuple(_score_item(item) for item in payload)
    if len(scores) != expected_count:
        raise ValueError(f"expected {expected_count} scores, got {len(scores)}")
    return scores


def parse_model_output(text: str, expected_count: int) -> tuple[Score, ...]:
    """Accept the requested JSON and a few harmless equivalent structures."""

    errors = []
    for candidate in reversed(_json_candidates(text)):
        try:
            return _scores_from_payload(candidate, expected_count)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    detail = errors[-1] if errors else "no JSON object or array was found"
    raise ValueError(f"Could not parse model output: {detail}; raw={text!r}")


def _format_prompt_batch(
    tokenizer,
    records: Sequence[Task1Record],
    *,
    prompt_mode: str,
    examples: Sequence[Task1Record],
    train_records: Sequence[Task1Record] = (),
    few_shot_examples: int = 5,
) -> list[str]:
    conversations = [
        build_messages(
            record,
            prompt_mode=prompt_mode,
            examples=(
                select_similar_examples(record, train_records, few_shot_examples)
                if prompt_mode == "dynamic_fewshot"
                else examples
            ),
        )
        for record in records
    ]
    return tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
    )


def _format_retry_prompt(
    tokenizer,
    record: Task1Record,
    *,
    prompt_mode: str,
    examples: Sequence[Task1Record],
    previous_output: str,
    parse_error: str,
    train_records: Sequence[Task1Record] = (),
    few_shot_examples: int = 5,
) -> str:
    if prompt_mode == "dynamic_fewshot":
        examples = select_similar_examples(record, train_records, few_shot_examples)
    messages = build_messages(record, prompt_mode=prompt_mode, examples=examples)
    messages.extend(
        [
            {"role": "assistant", "content": previous_output},
            {
                "role": "user",
                "content": (
                    "上面的回答格式不合格，请重新输出。"
                    f"解析问题：{parse_error}\n"
                    f"本样本有 {len(record.aspects)} 个带 index 的方面；"
                    "必须为每个 index 输出一组 [V,A]，即 scores 是二维数组，"
                    f"并且外层必须恰好有 {len(record.aspects)} 项。"
                    "不要合并或遗漏，不要输出字母占位符。"
                    "只输出纠正后的合法 JSON，不要解释。"
                ),
            },
        ]
    )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _load_model(args: argparse.Namespace):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Qwen inference must run on the GPU server")
    LOGGER.info("Loading model: %s (backend=%s)", args.model_name, args.backend)
    if args.backend == "unsloth":
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            load_in_4bit=True,
            max_seq_length=args.max_seq_length,
            dtype=None,
        )
        FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    model.eval()
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, torch


def _generate(
    model,
    tokenizer,
    torch,
    prompts: Sequence[str],
    *,
    max_seq_length: int,
    max_new_tokens: int,
) -> list[str]:
    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    ).to(model.device)
    input_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[:, input_width:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    started = time.perf_counter()

    input_records = load_task1_records(args.input_file, require_gold=False)
    train_records = load_task1_records(args.train_file, require_gold=True)
    fallback_score = mean_gold_score(train_records)
    records = (
        select_smoke_records(input_records, args.smoke_samples)
        if args.run_mode == "smoke"
        else input_records
    )
    batch_size = args.batch_size or (4 if args.run_mode == "smoke" else 16)
    LOGGER.info(
        "mode=%s prompt=%s rows=%s aspects=%s batch=%s fallback_mean=%.4f#%.4f",
        args.run_mode,
        args.prompt_mode,
        len(records),
        sum(len(record.aspects) for record in records),
        batch_size,
        fallback_score[0],
        fallback_score[1],
    )
    if args.run_mode == "smoke":
        LOGGER.warning("Smoke output is only a pipeline check, not a formal result")

    predictions: dict[str, tuple[Score, ...]] = {}
    diagnostics = []
    peak_allocated_gib = 0.0
    peak_reserved_gib = 0.0

    if args.prompt_mode == "mean":
        for record in records:
            predictions[record.record_id] = tuple(
                fallback_score for _ in record.aspects
            )
    else:
        examples = (
            select_anchor_examples(train_records, args.few_shot_examples)
            if args.prompt_mode == "fewshot"
            else []
        )
        model, tokenizer, torch = _load_model(args)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()

        order = sorted(
            range(len(records)),
            key=lambda index: len(records[index].text),
            reverse=True,
        )
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch_records = [records[index] for index in batch_indices]
            prompts = _format_prompt_batch(
                tokenizer,
                batch_records,
                prompt_mode=args.prompt_mode,
                examples=examples,
                train_records=train_records,
                few_shot_examples=args.few_shot_examples,
            )
            raw_outputs = _generate(
                model,
                tokenizer,
                torch,
                prompts,
                max_seq_length=args.max_seq_length,
                max_new_tokens=args.max_new_tokens,
            )
            if len(raw_outputs) != len(batch_records):
                raise RuntimeError("Generated output count does not match batch size")
            for record, raw_output in zip(batch_records, raw_outputs):
                attempt_outputs = [raw_output]
                attempt_errors = []
                current_output = raw_output
                for attempt_index in range(args.max_format_retries + 1):
                    try:
                        scores = parse_model_output(
                            current_output, len(record.aspects)
                        )
                        status = (
                            "ok"
                            if attempt_index == 0
                            else "ok_after_format_retry"
                        )
                        error = None
                        if attempt_index:
                            LOGGER.info(
                                "ID %s recovered after %s format retry(s)",
                                record.record_id,
                                attempt_index,
                            )
                        break
                    except ValueError as exc:
                        attempt_errors.append(str(exc))
                        if attempt_index >= args.max_format_retries:
                            scores = tuple(fallback_score for _ in record.aspects)
                            status = "fallback_train_mean"
                            error = str(exc)
                            LOGGER.warning(
                                "ID %s parse failure after %s retry(s): %s",
                                record.record_id,
                                args.max_format_retries,
                                exc,
                            )
                            break
                        retry_prompt = _format_retry_prompt(
                            tokenizer,
                            record,
                            prompt_mode=args.prompt_mode,
                            examples=examples,
                            previous_output=current_output,
                            parse_error=str(exc),
                            train_records=train_records,
                            few_shot_examples=args.few_shot_examples,
                        )
                        current_output = _generate(
                            model,
                            tokenizer,
                            torch,
                            [retry_prompt],
                            max_seq_length=args.max_seq_length,
                            max_new_tokens=args.max_new_tokens,
                        )[0]
                        attempt_outputs.append(current_output)
                predictions[record.record_id] = scores
                diagnostics.append(
                    {
                        "ID": record.record_id,
                        "status": status,
                        "raw_output": current_output,
                        "error": error,
                        "initial_error": attempt_errors[0] if attempt_errors else None,
                        "format_retry_count": len(attempt_outputs) - 1,
                        "attempt_outputs": attempt_outputs,
                        "scores": [[v, a] for v, a in scores],
                    }
                )
            LOGGER.info(
                "Inference progress: %s/%s",
                min(start + batch_size, len(order)),
                len(order),
            )

        peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    input_stem = Path(args.input_file).stem
    result_stem = f"{input_stem}_{args.prompt_mode}_{args.run_mode}"
    prediction_path = result_dir / f"{result_stem}.jsonl"
    diagnostic_path = result_dir / f"{result_stem}_diagnostics.jsonl"
    metadata_path = result_dir / f"{result_stem}_metadata.json"
    write_task1_predictions(records, predictions, prediction_path)

    if diagnostics:
        with diagnostic_path.open("w", encoding="utf-8") as handle:
            for item in diagnostics:
                json.dump(item, handle, ensure_ascii=False)
                handle.write("\n")

    failure_count = sum(
        item["status"] == "fallback_train_mean" for item in diagnostics
    )
    retry_recovery_count = sum(
        item["status"] == "ok_after_format_retry" for item in diagnostics
    )
    metadata = {
        "input_file": str(Path(args.input_file).resolve()),
        "train_file": str(Path(args.train_file).resolve()),
        "model_name": None if args.prompt_mode == "mean" else args.model_name,
        "backend": None if args.prompt_mode == "mean" else args.backend,
        "prompt_mode": args.prompt_mode,
        "run_mode": args.run_mode,
        "records": len(records),
        "aspects": sum(len(record.aspects) for record in records),
        "parse_failures": failure_count,
        "format_retry_recoveries": retry_recovery_count,
        "max_format_retries": args.max_format_retries,
        "fallback_train_mean": [fallback_score[0], fallback_score[1]],
        "batch_size": batch_size,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "few_shot_examples": args.few_shot_examples if args.prompt_mode in {"fewshot", "dynamic_fewshot"} else 0,
        "few_shot_retrieval": "per_record_bm25_lexical" if args.prompt_mode == "dynamic_fewshot" else None,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_gib": peak_allocated_gib,
        "peak_cuda_reserved_gib": peak_reserved_gib,
        "prediction_file": str(prediction_path.resolve()),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    LOGGER.info(
        "Saved %s; parse_failures=%s; format_retry_recoveries=%s; "
        "peak_cuda_allocated=%.2f GiB",
        prediction_path,
        failure_count,
        retry_recovery_count,
        peak_allocated_gib,
    )
    if failure_count:
        LOGGER.warning(
            "This run used the train mean for %s malformed outputs; inspect %s before reporting",
            failure_count,
            diagnostic_path,
        )


if __name__ == "__main__":
    main()
