#!/usr/bin/env python
"""
OpenAI-compatible LLM + self-consistency extraction for DimABSA Task 2/3.

Usage:
  # Dev evaluation
  python scripts/deepseek_consistency_extraction.py \
    --mode dev \
    --api-key YOUR_API_KEY \
    --temperature 0.1 \
    --num-generations 3 \
    --min-votes 3

  # Test (full run)
  python scripts/deepseek_consistency_extraction.py \
    --mode test \
    --api-key YOUR_API_KEY

Outputs:
  outputs/deepseek_results/dev_predictions_task3.jsonl
  outputs/deepseek_results/dev_predictions_task2.jsonl
  outputs/deepseek_results/test_predictions_task3.jsonl
  outputs/deepseek_results/test_predictions_task2.jsonl
  outputs/deepseek_results/stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add project src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from openai import OpenAI

from dimabsa_extraction import (
    ExtractionItem,
    ExtractionRecord,
    RESTAURANT_CATEGORIES,
    load_extraction_records,
    parse_extraction_payload,
    write_extraction_predictions,
)
from extraction_hybrid import BM25Retriever, recover_exact_span


# ==================== Configuration ====================

@dataclass
class Config:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    temperature: float = 0.1
    max_tokens: int = 500
    num_generations: int = 3
    min_votes: int = 3
    retrieval_k: int = 3  # per variant
    limit: Optional[int] = None  # only process first N records (probe runs)
    workers: int = 1  # concurrent API workers (1 = serial)
    rpm: int = 200  # account requests-per-minute limit
    input_price: float = 6.5  # yuan per million input tokens
    output_price: float = 27.0  # yuan per million output tokens
    output_dir: str = "outputs/deepseek_results"
    use_roberta_rescore: bool = False
    roberta_model_name: Optional[str] = None
    roberta_checkpoint: Optional[str] = None
    correction_rules: bool = True


# ==================== Prompt Template ====================

SYSTEM_PROMPT = """You extract dimensional aspect sentiment quadruplets from restaurant reviews.
Return exact aspect and opinion spans, a valid restaurant category, and continuous Valence/Arousal scores."""

CATEGORIES = ", ".join(RESTAURANT_CATEGORIES)


def build_user_prompt(text: str) -> str:
    return f"""Extract every (aspect, category, opinion, V, A) relation from the text.
- Copy explicit aspect and opinion spans exactly from the text.
- Use the literal string NULL only when an aspect or opinion is implicit.
- V and A are continuous values in [1,9].
- Valid categories: {CATEGORIES}
- Output only JSON: {{"items":[{{"aspect":"...","opinion":"...","category":"FOOD#QUALITY","V":"7.25","A":"6.50"}}]}}
- If there is no relation, output {{"items":[]}}.
Text: {text}"""


def build_fewshot_messages(
    record: ExtractionRecord,
    examples: List[ExtractionRecord],
) -> List[Dict[str, str]]:
    """Build multi-turn messages with few-shot examples."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add examples as user/assistant pairs
    for example in examples:
        messages.append({"role": "user", "content": build_user_prompt(example.text)})
        gold_items = [
            {
                "aspect": item.aspect,
                "opinion": item.opinion,
                "category": item.category,
                "V": f"{item.score[0]:.2f}",
                "A": f"{item.score[1]:.2f}",
            }
            for item in example.gold_items
        ]
        messages.append({
            "role": "assistant",
            "content": json.dumps({"items": gold_items}, ensure_ascii=False)
        })

    # Add current query
    messages.append({"role": "user", "content": build_user_prompt(record.text)})
    return messages


# ==================== DeepSeek API ====================

class RateLimiter:
    """Thread-safe sliding-window RPM limiter."""

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self.lock = threading.Lock()
        self.window: List[float] = []

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            self.window = [t for t in self.window if now - t < 60.0]
            if len(self.window) >= self.rpm:
                sleep_for = 60.0 - (now - self.window[0]) + 0.05
                time.sleep(sleep_for)
                now = time.time()
                self.window = [t for t in self.window if now - t < 60.0]
            self.window.append(now)


class DeepSeekAPI:
    def __init__(self, config: Config):
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.config = config
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.stats_lock = threading.Lock()
        self.rate_limiter = RateLimiter(config.rpm)

    def call(self, messages: List[Dict[str, str]], temperature: float = None) -> Tuple[str, Dict]:
        """Call API once, honoring the RPM limit and retrying transient errors.

        Every attempt (including retries) passes through the rate limiter so a
        burst of failures cannot turn into a thundering-herd of re-requests.
        """
        if temperature is None:
            temperature = self.config.temperature

        max_retries = 4
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            self.rate_limiter.wait()
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.config.max_tokens
                )

                usage = response.usage
                with self.stats_lock:
                    self.total_input_tokens += usage.prompt_tokens
                    self.total_output_tokens += usage.completion_tokens
                    self.total_calls += 1

                content = response.choices[0].message.content
                stats = {
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                }
                return content, stats
            except Exception as exc:
                last_error = exc
                wait = 20.0 * (2 ** attempt)
                print(
                    f"    API error (attempt {attempt + 1}/{max_retries}, "
                    f"wait {wait:.0f}s): {type(exc).__name__}"
                )
                time.sleep(wait)
        raise last_error

    def generate_with_consistency(
        self, messages: List[Dict[str, str]]
    ) -> List[Optional[str]]:
        """Generate N times with low temperature for self-consistency."""
        results = []
        for i in range(self.config.num_generations):
            try:
                content, _ = self.call(messages, temperature=self.config.temperature)
                results.append(content)
                print(f"    Generation {i+1}/{self.config.num_generations} OK")
            except Exception as e:
                print(f"    Generation {i+1}/{self.config.num_generations} FAILED: {e}")
                results.append(None)
        return results

    def get_stats(self) -> Dict:
        """Estimate cost using the configured per-model token prices."""
        input_cost = (self.total_input_tokens / 1e6) * self.config.input_price
        output_cost = (self.total_output_tokens / 1e6) * self.config.output_price
        return {
            "total_calls": self.total_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_yuan": round(input_cost + output_cost, 4)
        }


# ==================== JSON Parsing ====================

def parse_json_output(text: str) -> Optional[List[Dict]]:
    """Parse model JSON output, handling various formats."""
    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        if isinstance(data, list):
            return data
    except:
        pass

    # Try extracting JSON from text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except:
            pass

    # Try extracting from markdown code block
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "items" in data:
                return data["items"]
            if isinstance(data, list):
                return data
        except:
            pass

    return None


def normalize_item(item: Dict, text: str) -> Optional[ExtractionItem]:
    """Normalize a parsed item into ExtractionItem, recovering spans."""
    try:
        aspect = item.get("aspect", item.get("Aspect", "")).strip()
        opinion = item.get("opinion", item.get("Opinion", "")).strip()
        category = item.get("category", item.get("Category", "")).strip().upper()

        if not aspect or not opinion:
            return None

        # Recover exact spans from text
        aspect_recovered = recover_exact_span(text, aspect)
        opinion_recovered = recover_exact_span(text, opinion)
        if aspect_recovered is None or opinion_recovered is None:
            return None

        # Parse V/A
        v_raw = item.get("V", item.get("v", item.get("Valence")))
        a_raw = item.get("A", item.get("a", item.get("Arousal")))
        va_str = item.get("VA", item.get("va"))

        if va_str and isinstance(va_str, str) and "#" in va_str:
            v, a = map(float, va_str.split("#"))
        elif v_raw is not None and a_raw is not None:
            v, a = float(v_raw), float(a_raw)
        else:
            return None

        # Validate range
        if not (1.0 <= v <= 9.0 and 1.0 <= a <= 9.0):
            return None

        return ExtractionItem(
            aspect=aspect_recovered,
            opinion=opinion_recovered,
            category=category,
            score=(v, a)
        )
    except:
        return None


# ==================== Voting ====================

def vote_items(
    generations: List[Optional[List[Dict]]],
    text: str,
    min_votes: int,
) -> Tuple[ExtractionItem, ...]:
    """Vote on items across multiple generations.

    Key: (aspect.lower(), opinion.lower(), category.upper())
    Only keep items appearing >= min_votes times.
    VA scores are averaged across matching items.
    """
    item_counter = Counter()
    item_data = {}
    item_scores = {}

    for gen_idx, gen in enumerate(generations):
        if gen is None:
            continue

        # Deduplicate within this generation
        seen_in_gen = set()
        for raw_item in gen:
            item = normalize_item(raw_item, text)
            if item is None:
                continue

            key = (item.aspect.lower(), item.opinion.lower(), item.category)
            if key in seen_in_gen:
                continue
            seen_in_gen.add(key)

            item_counter[key] += 1
            if key not in item_data:
                item_data[key] = item
                item_scores[key] = []
            item_scores[key].append(item.score)

    # Filter by vote threshold and average VA
    result = []
    for key, count in item_counter.items():
        if count >= min_votes:
            item = item_data[key]
            avg_v = sum(s[0] for s in item_scores[key]) / len(item_scores[key])
            avg_a = sum(s[1] for s in item_scores[key]) / len(item_scores[key])
            result.append(ExtractionItem(
                aspect=item.aspect,
                opinion=item.opinion,
                category=item.category,
                score=(avg_v, avg_a)
            ))

    return tuple(result)


# ==================== Correction Rules ====================

class CorrectionRules:
    def __init__(self):
        self.rules = []

    def add_rule(self, condition, action):
        self.rules.append((condition, action))

    def apply(self, items: Tuple[ExtractionItem, ...]) -> Tuple[ExtractionItem, ...]:
        result = []
        for item in items:
            item = self._apply_rules(item)
            if item is not None:
                result.append(item)
        return tuple(result)

    def _apply_rules(self, item: ExtractionItem) -> Optional[ExtractionItem]:
        for condition, action in self.rules:
            if condition(item):
                item = action(item)
                if item is None:
                    return None
        return item


def create_default_rules() -> CorrectionRules:
    """Create default correction rules based on common error patterns."""
    rules = CorrectionRules()

    # Rule 1: Aspect="price" but Category not PRICE#* → force PRICE#GENERAL
    def cond1(item):
        return item.aspect.lower() == "price" and "PRICE#" not in item.category

    def action1(item):
        return ExtractionItem(
            aspect=item.aspect,
            opinion=item.opinion,
            category="PRICE#GENERAL",
            score=item.score
        )

    rules.add_rule(cond1, action1)

    # Rule 2: Opinion="expensive" but V>7 → correct V to 3
    def cond2(item):
        return item.opinion.lower() == "expensive" and item.score[0] > 7

    def action2(item):
        return ExtractionItem(
            aspect=item.aspect,
            opinion=item.opinion,
            category=item.category,
            score=(3.0, item.score[1])
        )

    rules.add_rule(cond2, action2)

    # Rule 3: Opinion="cheap" but V<3 → correct V to 7
    def cond3(item):
        return item.opinion.lower() == "cheap" and item.score[0] < 3

    def action3(item):
        return ExtractionItem(
            aspect=item.aspect,
            opinion=item.opinion,
            category=item.category,
            score=(7.0, item.score[1])
        )

    rules.add_rule(cond3, action3)

    # Rule 4: Category missing "#" → try to fix or delete
    def cond4(item):
        return "#" not in item.category

    def action4(item):
        cat_upper = item.category.upper()
        if "FOOD" in cat_upper:
            return ExtractionItem(item.aspect, item.opinion, "FOOD#QUALITY", item.score)
        elif "SERVICE" in cat_upper:
            return ExtractionItem(item.aspect, item.opinion, "SERVICE#GENERAL", item.score)
        elif "AMBIEN" in cat_upper:
            return ExtractionItem(item.aspect, item.opinion, "AMBIENCE#GENERAL", item.score)
        elif "LOCATION" in cat_upper or "LOCAT" in cat_upper:
            return ExtractionItem(item.aspect, item.opinion, "LOCATION#GENERAL", item.score)
        elif "DRINK" in cat_upper:
            return ExtractionItem(item.aspect, item.opinion, "DRINKS#QUALITY", item.score)
        elif "RESTAURANT" in cat_upper or "RESTAUR" in cat_upper:
            return ExtractionItem(item.aspect, item.opinion, "RESTAURANT#GENERAL", item.score)
        return None  # Can't fix, delete

    rules.add_rule(cond4, action4)

    return rules


# ==================== RoBERTa VA Rescore ====================

class RoBERTaRescorer:
    """Load a real Task-1 encoder checkpoint and predict relation-level VA."""

    def __init__(self, model_name: str, checkpoint_path: str, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from train_task1_mean_pooling import LogSigmaRegressor, load_checkpoint

        checkpoint = Path(checkpoint_path)
        config_path = checkpoint / "experiment_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing experiment config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        pooling = config.get("pooling", "cls")
        self.max_length = int(config.get("max_length", 128))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" else None
        model_kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        self.encoder = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.regressor = LogSigmaRegressor(self.encoder.config.hidden_size, pooling=pooling)
        load_checkpoint(checkpoint, self.encoder, self.regressor)
        self.encoder.to(self.device).eval()
        self.regressor.to(self.device, dtype=next(self.encoder.parameters()).dtype).eval()
        print(f"RoBERTa VA rescorer loaded from {checkpoint}")

    def rescore(self, text: str, aspect: str, opinion: str) -> Tuple[float, float]:
        import torch

        del opinion  # The published checkpoint is trained on a Text/Aspect pair.
        aspect = "[IMPLICIT_ASPECT]" if aspect == "NULL" else aspect
        inputs = self.tokenizer(
            text,
            aspect,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.encoder(**inputs, return_dict=True).last_hidden_state
            scores, _ = self.regressor(hidden, inputs["attention_mask"])
        value = scores[0].float().clamp(1.0, 9.0).cpu().tolist()
        return float(value[0]), float(value[1])


# ==================== Main Pipeline ====================

class ExtractionPipeline:
    def __init__(self, config: Config, train_file: str):
        self.config = config
        self.api = DeepSeekAPI(config)
        self.correction_rules = (
            create_default_rules() if config.correction_rules else None
        )

        # Load training data for BM25 retrieval
        print(f"Loading training data from {train_file}...")
        self.train_records = load_extraction_records(train_file, require_gold=True)
        print(f"Loaded {len(self.train_records)} training records")

        # Build BM25 retrievers for 3 variants
        print("Building BM25 retrievers (word/bigram/trigram)...")
        self.retrievers = {
            "word": BM25Retriever(self.train_records, variant="word"),
            "bigram": BM25Retriever(self.train_records, variant="bigram"),
            "trigram": BM25Retriever(self.train_records, variant="trigram"),
        }
        print("BM25 retrievers ready")

        # Optional: Load RoBERTa rescorer
        self.roberta_rescorer = None
        if config.use_roberta_rescore:
            if not config.roberta_model_name or not config.roberta_checkpoint:
                raise ValueError(
                    "--use-roberta-rescore requires --roberta-model-name and --roberta-checkpoint"
                )
            self.roberta_rescorer = RoBERTaRescorer(
                config.roberta_model_name, config.roberta_checkpoint
            )

    def retrieve_examples(self, record: ExtractionRecord, k: int = 3) -> List[ExtractionRecord]:
        """Retrieve k examples from each BM25 variant (total 3k examples)."""
        examples = []
        for variant, retriever in self.retrievers.items():
            try:
                selected = retriever.select(record.text, count=k, exclude_id=record.record_id)
                examples.extend(selected)
            except Exception as e:
                print(f"    Warning: BM25 {variant} retrieval failed: {e}")
        return examples

    def extract_single(self, record: ExtractionRecord) -> Tuple[ExtractionItem, ...]:
        """Extract quadruplets for a single record with self-consistency."""
        # Retrieve examples
        examples = self.retrieve_examples(record, k=self.config.retrieval_k)
        print(f"  Retrieved {len(examples)} examples")

        # Build few-shot prompt
        messages = build_fewshot_messages(record, examples[:9])  # Limit to 9 examples

        # Generate with self-consistency
        print(f"  Generating {self.config.num_generations} times (T={self.config.temperature})...")
        generations_raw = self.api.generate_with_consistency(messages)

        # Parse JSON outputs
        generations_parsed = []
        for raw in generations_raw:
            if raw is None:
                generations_parsed.append(None)
            else:
                parsed = parse_json_output(raw)
                generations_parsed.append(parsed)

        # Vote
        voted = vote_items(
            generations_parsed,
            record.text,
            min_votes=self.config.min_votes
        )
        print(f"  Voted: {len(voted)} items retained")

        # Apply correction rules
        if self.correction_rules:
            voted = self.correction_rules.apply(voted)
            print(f"  After correction: {len(voted)} items")

        return voted

    def extract_batch(
        self, records: List[ExtractionRecord]
    ) -> Dict[str, Tuple[ExtractionItem, ...]]:
        """Extract for a batch of records, serially or with concurrent workers.

        Results are appended to a checkpoint file as they finish so a crash or
        kill does not lose completed work; re-running resumes from the file.
        """
        results = {}
        total = len(records)
        checkpoint_path = Path(self.config.output_dir) / "checkpoint.jsonl"
        ckpt_lock = threading.Lock()

        done_ids: set = set()
        if checkpoint_path.exists():
            for line in checkpoint_path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    items = tuple(
                        ExtractionItem(
                            it["aspect"], it["opinion"], it["category"],
                            (float(it["V"]), float(it["A"])),
                        )
                        for it in row["items"]
                    )
                    results[row["ID"]] = items
                    done_ids.add(row["ID"])
                except Exception:
                    continue
            print(f"Resumed from checkpoint: {len(done_ids)} records already done")

        pending = [record for record in records if record.record_id not in done_ids]

        def save_one(record_id: str, items: Tuple[ExtractionItem, ...]) -> None:
            row = {
                "ID": record_id,
                "items": [
                    {
                        "aspect": it.aspect,
                        "opinion": it.opinion,
                        "category": it.category,
                        "V": it.score[0],
                        "A": it.score[1],
                    }
                    for it in items
                ],
            }
            with ckpt_lock:
                with checkpoint_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        def report(done: int) -> None:
            stats = self.api.get_stats()
            print(f"\n  --- Progress: {done}/{total} records ---")
            print(f"  API calls: {stats['total_calls']}")
            print(f"  Est. cost: {stats['estimated_cost_yuan']:.2f} yuan")

        if self.config.workers <= 1:
            for i, record in enumerate(pending):
                print(f"\n[{i + 1 + len(done_ids)}/{total}] ID={record.record_id}")
                items = self.extract_single(record)
                results[record.record_id] = items
                save_one(record.record_id, items)
                if (i + 1) % 10 == 0:
                    report(i + 1 + len(done_ids))
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
                future_to_record = {
                    pool.submit(self.extract_single, record): record
                    for record in pending
                }
                done = len(done_ids)
                for future in as_completed(future_to_record):
                    record = future_to_record[future]
                    try:
                        items = future.result()
                    except Exception as exc:
                        print(f"  ERROR on {record.record_id}: {exc}")
                        items = ()
                    results[record.record_id] = items
                    save_one(record.record_id, items)
                    done += 1
                    if done % 10 == 0:
                        report(done)

        return results

    def rescore_with_roberta(
        self,
        results: Dict[str, Tuple[ExtractionItem, ...]],
        records: List[ExtractionRecord],
    ) -> Dict[str, Tuple[ExtractionItem, ...]]:
        """Optional: Use RoBERTa to rescore VA predictions."""
        if self.roberta_rescorer is None:
            print("\nRoBERTa rescorer not available, skipping VA rescore")
            return results

        print("\nRescoring VA with RoBERTa...")
        rescored = {}
        for record in records:
            items = results.get(record.record_id, ())
            new_items = []
            for item in items:
                new_v, new_a = self.roberta_rescorer.rescore(
                    record.text, item.aspect, item.opinion
                )
                new_items.append(ExtractionItem(
                    aspect=item.aspect,
                    opinion=item.opinion,
                    category=item.category,
                    score=(new_v, new_a)
                ))
            rescored[record.record_id] = tuple(new_items)

        print(f"Rescored {len(rescored)} records")
        return rescored

    def run(
        self,
        test_file: str,
        output_task3: str,
        output_task2: str,
        test_records: Optional[List[ExtractionRecord]] = None,
        task2_template_file: Optional[str] = None,
    ) -> Dict:
        """Run full pipeline on test/dev set."""
        # Load data
        if test_records is None:
            print(f"\nLoading test data from {test_file}...")
            test_records = load_extraction_records(test_file, require_gold=True)
        if self.config.limit:
            test_records = test_records[: self.config.limit]
        print(f"Loaded {len(test_records)} test records")

        # Extract
        print("\nStarting extraction pipeline...")
        results = self.extract_batch(test_records)

        # Optional: RoBERTa rescore
        results = self.rescore_with_roberta(results, test_records)

        # Write outputs
        print(f"\nWriting outputs to {output_task3} and {output_task2}...")
        if task2_template_file:
            # Task 2 uses different official IDs; align by text order
            task2_records = load_extraction_records(task2_template_file, require_gold=True)
            if self.config.limit:
                task2_records = task2_records[: self.config.limit]
            if len(task2_records) != len(test_records):
                raise ValueError("Task 2 template and Task 3 records differ in length")
            task2_predictions = {}
            for task2_record, task3_record in zip(task2_records, test_records):
                if task2_record.text != task3_record.text:
                    raise ValueError(
                        f"Task 2/3 text mismatch: {task2_record.record_id} vs {task3_record.record_id}"
                    )
                task2_predictions[task2_record.record_id] = results[task3_record.record_id]
            write_extraction_predictions(
                task2_records, task2_predictions, output_task2, None
            )
            write_extraction_predictions(test_records, results, output_task3, None)
        else:
            write_extraction_predictions(test_records, results, output_task3, output_task2)

        # Final stats
        stats = self.api.get_stats()
        print(f"\n{'='*60}")
        print(f"Pipeline complete!")
        print(f"Total records: {len(test_records)}")
        print(f"Total API calls: {stats['total_calls']}")
        print(f"Input tokens: {stats['input_tokens']}")
        print(f"Output tokens: {stats['output_tokens']}")
        print(f"Estimated cost: {stats['estimated_cost_yuan']:.2f} yuan")
        print(f"{'='*60}")

        # Save stats
        stats_path = Path(output_task3).parent / "stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        return stats


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible LLM + self-consistency extraction for DimABSA Task 2/3"
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "test"],
        required=True,
        help="Run on dev (300 records) or test (1000 records)"
    )
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="API base URL")
    parser.add_argument("--model", default="deepseek-v4-flash", help="Model name")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--num-generations", type=int, default=3, help="Self-consistency generations")
    parser.add_argument("--min-votes", type=int, default=3, help="Minimum votes to retain")
    parser.add_argument("--retrieval-k", type=int, default=3, help="BM25 examples per variant")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N records")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent API workers")
    parser.add_argument("--rpm", type=int, default=200, help="Account requests-per-minute limit")
    parser.add_argument("--use-roberta-rescore", action="store_true", help="Apply a trained RoBERTa VA checkpoint")
    parser.add_argument("--roberta-model-name", help="Base model used by the VA checkpoint")
    parser.add_argument("--roberta-checkpoint", help="Directory containing encoder_trainable.pt and regressor.pt")
    parser.add_argument("--no-correction-rules", action="store_true", help="Disable correction rules")
    parser.add_argument("--output-dir", default="outputs/deepseek_local", help="Output directory")

    args = parser.parse_args()

    # Data paths (local absolute paths)
    DATA_ROOT = Path(__file__).parent.parent / "resources/DimABSA2026/task-dataset/track_a/subtask_3/eng"
    train_file = str(DATA_ROOT / "eng_restaurant_train_alltasks.jsonl")
    if args.mode == "dev":
        test_file = str(DATA_ROOT / "eng_restaurant_dev_task3.jsonl")
    else:
        test_file = str(DATA_ROOT / "eng_restaurant_test_task3.jsonl")

    # Create config
    config = Config(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_generations=args.num_generations,
        min_votes=args.min_votes,
        retrieval_k=args.retrieval_k,
        limit=args.limit,
        workers=args.workers,
        rpm=args.rpm,
        output_dir=args.output_dir,
        use_roberta_rescore=args.use_roberta_rescore,
        roberta_model_name=args.roberta_model_name,
        roberta_checkpoint=args.roberta_checkpoint,
        correction_rules=not args.no_correction_rules,
    )

    # Create output paths
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_task3 = output_dir / f"{args.mode}_predictions_task3.jsonl"
    output_task2 = output_dir / f"{args.mode}_predictions_task2.jsonl"

    # Create pipeline
    pipeline = ExtractionPipeline(config, train_file)

    # Task 2 template (uses different official IDs)
    task2_template_file = None
    if args.mode == "dev":
        task2_template_file = str(
            Path(__file__).parent.parent
            / "resources/DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl"
        )
    else:
        task2_template_file = str(
            Path(__file__).parent.parent
            / "resources/DimABSA2026/task-dataset/track_a/subtask_2/eng/eng_restaurant_test_task2.jsonl"
        )

    # Run
    stats = pipeline.run(
        test_file,
        str(output_task3),
        str(output_task2),
        task2_template_file=task2_template_file,
    )

    print("\nDone! Outputs:")
    print(f"  Task 3: {output_task3}")
    print(f"  Task 2: {output_task2}")
    print(f"  Stats:  {output_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
