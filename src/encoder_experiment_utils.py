"""Framework-free data helpers for Task 1 encoder experiments."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dimabsa_data import Task1Record, parse_va


@dataclass(frozen=True)
class AspectExample:
    record_id: str
    text: str
    aspect: str
    score: tuple[float, float]
    aspect_index: int
    opinion: str | None = None


def stable_fold(record_id: str, folds: int) -> int:
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def load_train_examples(path: str | Path) -> tuple[list[Task1Record], list[AspectExample]]:
    records: list[Task1Record] = []
    examples: list[AspectExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            entries = row.get("Quadruplet")
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"{path}:{line_number}: Quadruplet is required")
            aspects, scores = [], []
            for index, entry in enumerate(entries):
                aspect = entry["Aspect"]
                score = parse_va(entry["VA"])
                opinion = entry.get("Opinion", "NULL")
                aspects.append(aspect)
                scores.append(score)
                examples.append(
                    AspectExample(row["ID"], row["Text"], aspect, score, index, opinion)
                )
            records.append(
                Task1Record(row["ID"], row["Text"], tuple(aspects), tuple(scores))
            )
    return records, examples


def examples_from_records(records: list[Task1Record]) -> list[AspectExample]:
    examples: list[AspectExample] = []
    for record in records:
        if record.gold_scores is None:
            raise ValueError(f"ID {record.record_id!r}: VA labels are required")
        for index, (aspect, score) in enumerate(zip(record.aspects, record.gold_scores)):
            examples.append(
                AspectExample(record.record_id, record.text, aspect, score, index)
            )
    return examples


def split_oof(
    records: list[Task1Record], examples: list[AspectExample], folds: int, fold: int
) -> tuple[list[Task1Record], list[AspectExample], list[Task1Record], list[AspectExample]]:
    if not 0 <= fold < folds:
        raise ValueError("OOF fold must satisfy 0 <= fold < folds")
    train_records = [record for record in records if stable_fold(record.record_id, folds) != fold]
    valid_records = [record for record in records if stable_fold(record.record_id, folds) == fold]
    train_ids = {record.record_id for record in train_records}
    valid_ids = {record.record_id for record in valid_records}
    return (
        train_records,
        [example for example in examples if example.record_id in train_ids],
        valid_records,
        [example for example in examples if example.record_id in valid_ids],
    )


def opinion_spans(text: str, opinion: str | None) -> list[tuple[int, int]]:
    if not opinion or opinion == "NULL":
        return []
    spans, lowered, needle, start = [], text.lower(), opinion.lower(), 0
    while True:
        index = lowered.find(needle, start)
        if index < 0:
            break
        spans.append((index, index + len(needle)))
        start = index + len(needle)
    if not spans:
        raise ValueError(f"Opinion {opinion!r} is not present in text {text!r}")
    return spans


def balanced_weight_values(examples: list[AspectExample]) -> list[float]:
    def bucket(score: tuple[float, float]) -> tuple[int, int]:
        def one(value: float) -> int:
            return 0 if value < 4.0 else 1 if value < 6.5 else 2

        return one(score[0]), one(score[1])

    counts = Counter(bucket(example.score) for example in examples)
    weights = [
        math.sqrt(len(examples) / counts[bucket(example.score)]) for example in examples
    ]
    mean = sum(weights) / len(weights)
    return [value / mean for value in weights]
