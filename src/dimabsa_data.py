"""DimABSA Track A / Task 1 JSONL loading and output helpers."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


Score = tuple[float, float]


@dataclass(frozen=True)
class Task1Record:
    """One Task 1 text with ordered aspects and optional gold VA scores."""

    record_id: str
    text: str
    aspects: tuple[str, ...]
    gold_scores: tuple[Score, ...] | None


def parse_va(value: str) -> Score:
    """Parse the official ``V#A`` representation and validate its range."""

    if not isinstance(value, str) or value.count("#") != 1:
        raise ValueError(f"VA must be a 'V#A' string, got {value!r}")
    v_text, a_text = value.split("#")
    v, a = float(v_text), float(a_text)
    if not math.isfinite(v) or not math.isfinite(a):
        raise ValueError(f"VA contains a non-finite value: {value!r}")
    if not 1.0 <= v <= 9.0 or not 1.0 <= a <= 9.0:
        raise ValueError(f"VA must stay within [1, 9], got {value!r}")
    return v, a


def _record_from_object(
    row: dict,
    *,
    source: Path,
    line_number: int,
    require_gold: bool,
    require_text: bool,
) -> Task1Record:
    record_id = row.get("ID")
    text = row.get("Text", "")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"{source}:{line_number}: missing non-empty ID")
    if not isinstance(text, str) or (require_text and not text):
        raise ValueError(f"{source}:{line_number}: missing non-empty Text")

    entries = None
    if "Aspect_VA" in row:
        entries = row["Aspect_VA"]
    elif "Quadruplet" in row:
        # Chinese restaurant/laptop training files are shared by all three tasks.
        # Task 1 only needs Aspect and VA from each quadruplet.
        entries = row["Quadruplet"]

    if entries is not None:
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{source}:{line_number}: aspect entries must be a non-empty list")
        aspects: list[str] = []
        scores: list[Score] = []
        missing_score = False
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{source}:{line_number}: aspect entry {index} is not an object"
                )
            aspect = entry.get("Aspect")
            if not isinstance(aspect, str) or not aspect:
                raise ValueError(
                    f"{source}:{line_number}: aspect entry {index} has no valid Aspect"
                )
            aspects.append(aspect)
            if "VA" in entry:
                scores.append(parse_va(entry["VA"]))
            else:
                missing_score = True
        if missing_score and scores:
            raise ValueError(f"{source}:{line_number}: only some aspect entries contain VA")
        if require_gold and missing_score:
            raise ValueError(f"{source}:{line_number}: gold VA is required but missing")
        gold_scores = None if missing_score else tuple(scores)
    else:
        raw_aspects = row.get("Aspect")
        if not isinstance(raw_aspects, list) or not raw_aspects:
            raise ValueError(
                f"{source}:{line_number}: expected Aspect, Aspect_VA, or Quadruplet"
            )
        if not all(isinstance(aspect, str) and aspect for aspect in raw_aspects):
            raise ValueError(f"{source}:{line_number}: Aspect contains an invalid value")
        if require_gold:
            raise ValueError(f"{source}:{line_number}: gold VA is required but missing")
        aspects = list(raw_aspects)
        gold_scores = None

    if gold_scores is not None and len(aspects) != len(gold_scores):
        raise ValueError(f"{source}:{line_number}: aspect/score count mismatch")
    return Task1Record(record_id, text, tuple(aspects), gold_scores)


def load_task1_records(
    path: str | Path,
    *,
    require_gold: bool = False,
    require_text: bool = True,
) -> list[Task1Record]:
    """Load JSONL while preserving ID, aspect order, and duplicate aspects."""

    source = Path(path)
    records: list[Task1Record] = []
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: each line must be a JSON object")
            record = _record_from_object(
                row,
                source=source,
                line_number=line_number,
                require_gold=require_gold,
                require_text=require_text,
            )
            if record.record_id in seen_ids:
                raise ValueError(f"{source}:{line_number}: duplicate ID {record.record_id!r}")
            seen_ids.add(record.record_id)
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {source}")
    return records


def mean_gold_score(records: Iterable[Task1Record]) -> Score:
    """Compute a deterministic train-set mean used as baseline and parse fallback."""

    scores = [
        score
        for record in records
        for score in (record.gold_scores or ())
    ]
    if not scores:
        raise ValueError("Cannot compute a mean without gold scores")
    return (
        sum(score[0] for score in scores) / len(scores),
        sum(score[1] for score in scores) / len(scores),
    )


def select_smoke_records(
    records: Sequence[Task1Record], sample_size: int
) -> list[Task1Record]:
    """Select longest texts for a conservative GPU-memory smoke test."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if sample_size >= len(records):
        return list(records)
    selected_indices = sorted(
        sorted(range(len(records)), key=lambda index: len(records[index].text), reverse=True)[
            :sample_size
        ]
    )
    return [records[index] for index in selected_indices]


def select_anchor_examples(
    records: Sequence[Task1Record], count: int
) -> list[Task1Record]:
    """Choose deterministic, short few-shot examples across the VA space."""

    if count < 0:
        raise ValueError("few-shot example count cannot be negative")
    if count == 0:
        return []
    anchors = [(2.0, 8.0), (4.0, 3.0), (5.0, 5.0), (7.0, 4.0), (8.0, 8.0)]
    candidates = [
        record
        for record in records
        if record.gold_scores
        and len(record.aspects) <= 3
        and len(record.text) <= 180
    ]
    if not candidates:
        raise ValueError("No suitable records are available for few-shot examples")

    selected: list[Task1Record] = []
    used_ids: set[str] = set()
    while len(selected) < count:
        anchor = anchors[len(selected) % len(anchors)]

        def distance(record: Task1Record) -> tuple[float, str]:
            assert record.gold_scores is not None
            mean_v = sum(score[0] for score in record.gold_scores) / len(record.gold_scores)
            mean_a = sum(score[1] for score in record.gold_scores) / len(record.gold_scores)
            return (mean_v - anchor[0]) ** 2 + (mean_a - anchor[1]) ** 2, record.record_id

        available = [record for record in candidates if record.record_id not in used_ids]
        if not available:
            break
        chosen = min(available, key=distance)
        selected.append(chosen)
        used_ids.add(chosen.record_id)
    return selected


_ENGLISH_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def _retrieval_tokens(record: Task1Record) -> list[str]:
    """Return answer-free lexical features for English example retrieval."""

    text_tokens = _ENGLISH_TOKEN.findall(record.text.lower())
    aspect_tokens = [
        token
        for aspect in record.aspects
        for token in _ENGLISH_TOKEN.findall(aspect.lower())
    ]
    # Repeating aspect tokens gives the requested target more influence than
    # generic restaurant words without using any VA label.
    return text_tokens + aspect_tokens * 3


def select_similar_examples(
    query: Task1Record,
    records: Sequence[Task1Record],
    count: int,
) -> list[Task1Record]:
    """Select deterministic per-query Train examples using BM25-style overlap."""

    if count < 1:
        raise ValueError("dynamic few-shot example count must be positive")
    candidates = [
        record
        for record in records
        if record.gold_scores
        and record.record_id != query.record_id
        and len(record.aspects) <= 3
        and len(record.text) <= 220
    ]
    if len(candidates) < count:
        raise ValueError("Not enough suitable records for dynamic few-shot retrieval")

    query_terms = set(_retrieval_tokens(query))
    tokenized = [_retrieval_tokens(record) for record in candidates]
    document_frequency = Counter(
        term for tokens in tokenized for term in set(tokens) if term in query_terms
    )
    average_length = sum(map(len, tokenized)) / len(tokenized)

    def score(item: tuple[Task1Record, list[str]]) -> tuple[float, str]:
        record, tokens = item
        frequencies = Counter(tokens)
        value = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            frequency_docs = document_frequency[term]
            inverse_frequency = math.log(
                1.0 + (len(candidates) - frequency_docs + 0.5) / (frequency_docs + 0.5)
            )
            denominator = frequency + 1.5 * (
                0.25 + 0.75 * len(tokens) / average_length
            )
            value += inverse_frequency * frequency * 2.5 / denominator
        return value, record.record_id

    ranked = sorted(
        zip(candidates, tokenized),
        key=lambda item: (-score(item)[0], score(item)[1]),
    )
    return [record for record, _ in ranked[:count]]


def write_task1_predictions(
    records: Sequence[Task1Record],
    predictions: dict[str, Sequence[Score]],
    path: str | Path,
) -> None:
    """Write the exact official Task 1 JSONL structure."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            if record.record_id not in predictions:
                raise ValueError(f"Missing prediction for ID {record.record_id!r}")
            scores = list(predictions[record.record_id])
            if len(scores) != len(record.aspects):
                raise ValueError(
                    f"ID {record.record_id!r}: expected {len(record.aspects)} scores, "
                    f"got {len(scores)}"
                )
            aspect_va = []
            for aspect, (v, a) in zip(record.aspects, scores):
                if not math.isfinite(v) or not math.isfinite(a):
                    raise ValueError(f"ID {record.record_id!r}: non-finite prediction")
                if not 1.0 <= v <= 9.0 or not 1.0 <= a <= 9.0:
                    raise ValueError(f"ID {record.record_id!r}: prediction outside [1, 9]")
                aspect_va.append({"Aspect": aspect, "VA": f"{v:.2f}#{a:.2f}"})
            json.dump(
                {"ID": record.record_id, "Aspect_VA": aspect_va},
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")
