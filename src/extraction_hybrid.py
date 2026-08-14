"""Hybrid utilities for retrieval, structure voting, and relation VA regression."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dimabsa_data import parse_va
from dimabsa_extraction import (
    ExtractionItem,
    ExtractionRecord,
    load_extraction_records,
    write_extraction_predictions,
)


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def retrieval_tokens(text: str, variant: str) -> list[str]:
    """Tokenize answer-free text for one of the three BM25 retrieval views."""

    lowered = text.lower()
    if variant == "word":
        return _WORD.findall(lowered)
    compact = " ".join(_WORD.findall(lowered))
    width = {"bigram": 2, "trigram": 3}.get(variant)
    if width is None:
        raise ValueError("variant must be word, bigram, or trigram")
    return [compact[index : index + width] for index in range(len(compact) - width + 1)]


class BM25Retriever:
    """Small deterministic BM25 index over labeled Train sentences."""

    def __init__(self, records: Sequence[ExtractionRecord], variant: str) -> None:
        self.records = [record for record in records if record.gold_items]
        if not self.records:
            raise ValueError("BM25 retrieval requires labeled records")
        self.variant = variant
        self.documents = [retrieval_tokens(record.text, variant) for record in self.records]
        self.frequencies = [Counter(tokens) for tokens in self.documents]
        self.document_frequency = Counter(
            token for tokens in self.documents for token in set(tokens)
        )
        self.average_length = sum(map(len, self.documents)) / len(self.documents)

    def score_all(self, text: str) -> list[float]:
        """Return BM25 scores aligned with ``self.records``."""

        query_terms = set(retrieval_tokens(text, self.variant))
        population = len(self.records)

        def score(index: int) -> float:
            tokens = self.documents[index]
            frequencies = self.frequencies[index]
            value = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                frequency_docs = self.document_frequency[term]
                inverse_frequency = math.log(
                    1.0 + (population - frequency_docs + 0.5) / (frequency_docs + 0.5)
                )
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * len(tokens) / self.average_length
                )
                value += inverse_frequency * frequency * 2.5 / denominator
            return value

        return [score(index) for index in range(len(self.records))]

    def select(self, text: str, count: int, *, exclude_id: str | None = None):
        if count < 1:
            raise ValueError("BM25 example count must be positive")
        scores = self.score_all(text)

        candidates = [
            index
            for index, record in enumerate(self.records)
            if record.record_id != exclude_id
        ]
        ranked = sorted(
            candidates,
            key=lambda index: (-scores[index], self.records[index].record_id),
        )
        if len(ranked) < count:
            raise ValueError("Not enough retrieval candidates")
        return [self.records[index] for index in ranked[:count]]


def recover_exact_span(text: str, candidate: str) -> str | None:
    """Restore original casing when a generated span matches case-insensitively."""

    candidate = candidate.strip()
    if candidate == "NULL":
        return candidate
    if candidate in text:
        return candidate
    lowered_text, lowered_candidate = text.lower(), candidate.lower()
    start = lowered_text.find(lowered_candidate)
    if start < 0:
        return None
    return text[start : start + len(candidate)]


def recover_payload_spans(payload: object, text: str) -> object:
    """Return a JSON-compatible payload with source casing restored where possible."""

    cloned = json.loads(json.dumps(payload, ensure_ascii=False))
    if isinstance(cloned, dict):
        raw_items = cloned.get("items", cloned.get("Quadruplet", cloned.get("quadruplets")))
    else:
        raw_items = cloned
    if not isinstance(raw_items, list):
        return cloned
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        for lower_key, upper_key in (("aspect", "Aspect"), ("opinion", "Opinion")):
            key = lower_key if lower_key in item else upper_key if upper_key in item else None
            if key is None or not isinstance(item[key], str):
                continue
            recovered = recover_exact_span(text, item[key])
            if recovered is not None:
                item[key] = recovered
    return cloned


def relation_label(aspect: str, opinion: str) -> str:
    return f"Aspect: {aspect} [OPINION] {opinion}"


def write_relation_dataset(
    input_path: str | Path,
    output_path: str | Path,
    schema: str,
    template_path: str | Path | None = None,
) -> int:
    """Convert gold/predicted Task 3 relations into the existing Task 1 regression schema."""

    records = load_extraction_records(
        input_path, require_gold=True, require_text=template_path is None
    )
    if template_path is not None:
        templates = load_extraction_records(template_path, require_gold=True)
        text_by_id = {record.record_id: record.text for record in templates}
        if set(text_by_id) != {record.record_id for record in records}:
            raise ValueError("Input and template ID sets differ")
        records = [
            ExtractionRecord(
                record.record_id, text_by_id[record.record_id], record.gold_items
            )
            for record in records
        ]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    relations = 0
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            items = record.gold_items or ()
            if not items:
                continue
            relations += len(items)
            if schema == "train":
                row = {
                    "ID": record.record_id,
                    "Text": record.text,
                    "Quadruplet": [
                        {
                            "Aspect": relation_label(item.aspect, item.opinion),
                            "Opinion": item.opinion,
                            "Category": item.category,
                            "VA": f"{item.score[0]:.2f}#{item.score[1]:.2f}",
                        }
                        for item in items
                    ],
                }
            elif schema == "task1":
                row = {
                    "ID": record.record_id,
                    "Text": record.text,
                    "Aspect_VA": [
                        {
                            "Aspect": relation_label(item.aspect, item.opinion),
                            "VA": f"{item.score[0]:.2f}#{item.score[1]:.2f}",
                        }
                        for item in items
                    ],
                }
            else:
                raise ValueError("schema must be train or task1")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return relations


def _load_task1_scores(path: str | Path) -> dict[str, list[tuple[str, tuple[float, float]]]]:
    rows: dict[str, list[tuple[str, tuple[float, float]]]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["ID"]] = [
                (item["Aspect"], parse_va(item["VA"])) for item in row["Aspect_VA"]
            ]
    return rows


def apply_relation_scores(
    extraction_path: str | Path,
    score_path: str | Path,
    task3_output: str | Path,
    task2_output: str | Path,
    task2_template_path: str | Path | None = None,
    task3_template_path: str | Path | None = None,
) -> int:
    records = load_extraction_records(
        extraction_path, require_gold=True, require_text=False
    )
    if task3_template_path is not None:
        templates = load_extraction_records(task3_template_path, require_gold=True)
        if [record.record_id for record in records] != [
            record.record_id for record in templates
        ]:
            raise ValueError("Extraction and Task 3 template IDs or order differ")
        records = [
            ExtractionRecord(record.record_id, template.text, record.gold_items)
            for record, template in zip(records, templates)
        ]
    predicted_scores = _load_task1_scores(score_path)
    output: dict[str, tuple[ExtractionItem, ...]] = {}
    relations = 0
    for record in records:
        original = record.gold_items or ()
        if not original:
            output[record.record_id] = ()
            continue
        scores = predicted_scores.get(record.record_id)
        if scores is None or len(scores) != len(original):
            raise ValueError(f"ID {record.record_id!r}: relation score coverage mismatch")
        rescored = []
        for item, (label, score) in zip(original, scores):
            expected = relation_label(item.aspect, item.opinion)
            if label != expected:
                raise ValueError(f"ID {record.record_id!r}: relation order mismatch")
            rescored.append(ExtractionItem(item.aspect, item.opinion, item.category, score))
        output[record.record_id] = tuple(rescored)
        relations += len(rescored)
    _write_aligned_predictions(
        records,
        output,
        task3_output,
        task2_output,
        task2_template_path,
    )
    return relations


def apply_single_task_scores(
    extraction_path: str | Path,
    score_path: str | Path,
    template_path: str | Path,
    output_path: str | Path,
    task: int,
) -> int:
    """Apply relation VA scores without forcing Task 2 and Task 3 to share structures."""

    if task not in {2, 3}:
        raise ValueError("task must be 2 or 3")
    records = load_extraction_records(
        extraction_path, require_gold=True, require_text=False
    )
    templates = load_extraction_records(template_path, require_gold=True)
    if [record.record_id for record in records] != [
        record.record_id for record in templates
    ]:
        raise ValueError("Extraction and template IDs or order differ")
    predicted_scores = _load_task1_scores(score_path)
    output: dict[str, tuple[ExtractionItem, ...]] = {}
    relations = 0
    for record, template in zip(records, templates):
        original = record.gold_items or ()
        if not original:
            output[template.record_id] = ()
            continue
        scores = predicted_scores.get(record.record_id)
        if scores is None or len(scores) != len(original):
            raise ValueError(f"ID {record.record_id!r}: relation score coverage mismatch")
        rescored = []
        for item, (label, score) in zip(original, scores):
            expected = relation_label(item.aspect, item.opinion)
            if label != expected:
                raise ValueError(f"ID {record.record_id!r}: relation order mismatch")
            rescored.append(
                ExtractionItem(item.aspect, item.opinion, item.category, score)
            )
        output[template.record_id] = tuple(rescored)
        relations += len(rescored)
    if task == 3:
        write_extraction_predictions(templates, output, output_path)
    else:
        dummy = Path(output_path).with_name(Path(output_path).name + ".task3.tmp")
        write_extraction_predictions(templates, output, dummy, output_path)
        dummy.unlink()
    return relations


def _write_aligned_predictions(
    task3_records: Sequence[ExtractionRecord],
    predictions: dict[str, Sequence[ExtractionItem]],
    task3_output: str | Path,
    task2_output: str | Path,
    task2_template_path: str | Path | None,
) -> None:
    """Write aligned Task 2/3 files even when their official IDs differ."""

    if task2_template_path is None:
        write_extraction_predictions(
            task3_records, predictions, task3_output, task2_output
        )
        return
    task2_records = load_extraction_records(task2_template_path, require_gold=True)
    if len(task2_records) != len(task3_records) or any(
        task2.text != task3.text
        for task2, task3 in zip(task2_records, task3_records)
    ):
        raise ValueError("Task 2 and Task 3 templates are not text-aligned")
    write_extraction_predictions(task3_records, predictions, task3_output)
    task2_predictions = {
        task2.record_id: predictions[task3.record_id]
        for task2, task3 in zip(task2_records, task3_records)
    }
    dummy = Path(task2_output).with_name(Path(task2_output).name + ".quadruplet.tmp")
    write_extraction_predictions(
        task2_records, task2_predictions, dummy, task2_output
    )
    dummy.unlink()


@dataclass
class VoteEntry:
    first_order: tuple[int, int]
    appearances: int
    item: ExtractionItem
    scores: list[tuple[float, float]]


def vote_prediction_files(
    template_path: str | Path,
    prediction_paths: Sequence[str | Path],
    minimum_votes: int,
    task3_output: str | Path,
    task2_output: str | Path,
    task2_template_path: str | Path | None = None,
) -> dict[str, int]:
    """Vote on exact categorical structure and average VA only among matching views."""

    if not 1 <= minimum_votes <= len(prediction_paths):
        raise ValueError("minimum_votes must be within the number of views")
    template = load_extraction_records(template_path, require_gold=True)
    views = [
        load_extraction_records(path, require_gold=True, require_text=False)
        for path in prediction_paths
    ]
    template_ids = [record.record_id for record in template]
    if any([record.record_id for record in view] != template_ids for view in views):
        raise ValueError("Prediction view IDs or order differ")
    voted: dict[str, tuple[ExtractionItem, ...]] = {}
    retained = 0
    for record_index, record in enumerate(template):
        grouped: dict[tuple[str, str, str], VoteEntry] = {}
        for view_index, view in enumerate(views):
            seen_in_view: set[tuple[str, str, str]] = set()
            for item_index, raw_item in enumerate(view[record_index].gold_items or ()):
                aspect = recover_exact_span(record.text, raw_item.aspect)
                opinion = recover_exact_span(record.text, raw_item.opinion)
                if aspect is None or opinion is None:
                    continue
                key = (aspect.lower(), opinion.lower(), raw_item.category.upper())
                if key in seen_in_view:
                    continue
                seen_in_view.add(key)
                if key not in grouped:
                    normalized = ExtractionItem(
                        aspect, opinion, raw_item.category.upper(), raw_item.score
                    )
                    grouped[key] = VoteEntry(
                        (view_index, item_index), 0, normalized, []
                    )
                grouped[key].appearances += 1
                grouped[key].scores.append(raw_item.score)
        selected = []
        for entry in sorted(grouped.values(), key=lambda value: value.first_order):
            if entry.appearances < minimum_votes:
                continue
            mean_score = (
                sum(score[0] for score in entry.scores) / len(entry.scores),
                sum(score[1] for score in entry.scores) / len(entry.scores),
            )
            selected.append(
                ExtractionItem(
                    entry.item.aspect,
                    entry.item.opinion,
                    entry.item.category,
                    mean_score,
                )
            )
        voted[record.record_id] = tuple(selected)
        retained += len(selected)
    _write_aligned_predictions(
        template,
        voted,
        task3_output,
        task2_output,
        task2_template_path,
    )
    return {"records": len(template), "views": len(views), "retained_relations": retained}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-relations")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--schema", choices=("train", "task1"), required=True)
    build.add_argument("--template")
    apply_parser = subparsers.add_parser("apply-scores")
    apply_parser.add_argument("--extraction", required=True)
    apply_parser.add_argument("--scores", required=True)
    apply_parser.add_argument("--task2-template")
    apply_parser.add_argument("--task3-template")
    apply_parser.add_argument("--output-task2", required=True)
    apply_parser.add_argument("--output-task3", required=True)
    single = subparsers.add_parser("apply-scores-single")
    single.add_argument("--extraction", required=True)
    single.add_argument("--scores", required=True)
    single.add_argument("--template", required=True)
    single.add_argument("--output", required=True)
    single.add_argument("--task", type=int, choices=(2, 3), required=True)
    vote = subparsers.add_parser("vote")
    vote.add_argument("--template", required=True)
    vote.add_argument("--task2-template")
    vote.add_argument("--pred", nargs="+", required=True)
    vote.add_argument("--minimum-votes", type=int, required=True)
    vote.add_argument("--output-task2", required=True)
    vote.add_argument("--output-task3", required=True)
    args = parser.parse_args()
    if args.command == "build-relations":
        result = {
            "relations": write_relation_dataset(
                args.input, args.output, args.schema, args.template
            )
        }
    elif args.command == "apply-scores":
        result = {
            "relations": apply_relation_scores(
                args.extraction,
                args.scores,
                args.output_task3,
                args.output_task2,
                args.task2_template,
                args.task3_template,
            )
        }
    elif args.command == "apply-scores-single":
        result = {
            "relations": apply_single_task_scores(
                args.extraction,
                args.scores,
                args.template,
                args.output,
                args.task,
            )
        }
    else:
        result = vote_prediction_files(
            args.template,
            args.pred,
            args.minimum_votes,
            args.output_task3,
            args.output_task2,
            args.task2_template,
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
