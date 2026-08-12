"""Data and output helpers for DimABSA Track A Tasks 2 and 3."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from dimabsa_data import Score, parse_va


RESTAURANT_ENTITIES = (
    "RESTAURANT",
    "FOOD",
    "DRINKS",
    "AMBIENCE",
    "SERVICE",
    "LOCATION",
)
RESTAURANT_ATTRIBUTES = (
    "GENERAL",
    "PRICES",
    "QUALITY",
    "STYLE_OPTIONS",
    "MISCELLANEOUS",
)
RESTAURANT_CATEGORIES = tuple(
    f"{entity}#{attribute}"
    for entity in RESTAURANT_ENTITIES
    for attribute in RESTAURANT_ATTRIBUTES
)


@dataclass(frozen=True)
class ExtractionItem:
    aspect: str
    opinion: str
    category: str
    score: Score


@dataclass(frozen=True)
class ExtractionRecord:
    record_id: str
    text: str
    gold_items: tuple[ExtractionItem, ...] | None


def load_extraction_records(
    path: str | Path, *, require_gold: bool = False
) -> list[ExtractionRecord]:
    """Load Task 2/3 JSONL; shared train files use the Task 3 schema."""

    source = Path(path)
    records: list[ExtractionRecord] = []
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
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            record_id = row.get("ID")
            text = row.get("Text")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{source}:{line_number}: missing non-empty ID")
            if record_id in seen_ids:
                raise ValueError(f"{source}:{line_number}: duplicate ID {record_id!r}")
            if not isinstance(text, str) or not text:
                raise ValueError(f"{source}:{line_number}: missing non-empty Text")

            raw_items = row.get("Quadruplet", row.get("Triplet"))
            if raw_items is None:
                if require_gold:
                    raise ValueError(f"{source}:{line_number}: gold extraction is required")
                gold_items = None
            else:
                if not isinstance(raw_items, list):
                    raise ValueError(f"{source}:{line_number}: extraction field is not a list")
                parsed_items = []
                for index, item in enumerate(raw_items):
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"{source}:{line_number}: item {index} is not an object"
                        )
                    aspect = item.get("Aspect")
                    opinion = item.get("Opinion")
                    category = item.get("Category", "")
                    if not isinstance(aspect, str) or not aspect:
                        raise ValueError(f"{source}:{line_number}: invalid Aspect")
                    if not isinstance(opinion, str) or not opinion:
                        raise ValueError(f"{source}:{line_number}: invalid Opinion")
                    if not isinstance(category, str):
                        raise ValueError(f"{source}:{line_number}: invalid Category")
                    parsed_items.append(
                        ExtractionItem(aspect, opinion, category, parse_va(item["VA"]))
                    )
                gold_items = tuple(parsed_items)

            seen_ids.add(record_id)
            records.append(ExtractionRecord(record_id, text, gold_items))
    if not records:
        raise ValueError(f"No records found in {source}")
    return records


def select_extraction_examples(
    records: Sequence[ExtractionRecord], count: int
) -> list[ExtractionRecord]:
    """Choose short deterministic examples while covering category diversity."""

    if count < 0:
        raise ValueError("few-shot example count cannot be negative")
    if count == 0:
        return []
    candidates = [
        record
        for record in records
        if record.gold_items
        and len(record.gold_items) <= 4
        and len(record.text) <= 100
        and all(item.category for item in record.gold_items)
    ]
    selected: list[ExtractionRecord] = []
    covered: set[str] = set()
    while len(selected) < count:
        available = [record for record in candidates if record not in selected]
        if not available:
            break

        def rank(record: ExtractionRecord) -> tuple[int, int, int, str]:
            assert record.gold_items is not None
            categories = {item.category for item in record.gold_items}
            return (
                -len(categories - covered),
                -min(len(record.gold_items), 3),
                len(record.text),
                record.record_id,
            )

        chosen = min(available, key=rank)
        selected.append(chosen)
        assert chosen.gold_items is not None
        covered.update(item.category for item in chosen.gold_items)
    return selected


def _pick(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _parse_score(item: dict[str, Any]) -> Score:
    raw_va = _pick(item, "VA", "va")
    if raw_va is not None:
        if isinstance(raw_va, str):
            return parse_va(raw_va)
        if isinstance(raw_va, (list, tuple)) and len(raw_va) == 2:
            raw_v, raw_a = raw_va
        else:
            raise ValueError("VA must be a V#A string or a two-value list")
    else:
        raw_v = _pick(item, "V", "v", "Valence", "valence")
        raw_a = _pick(item, "A", "a", "Arousal", "arousal")
    try:
        v, a = float(raw_v), float(raw_a)
    except (TypeError, ValueError) as exc:
        raise ValueError("missing or invalid V/A") from exc
    if not math.isfinite(v) or not math.isfinite(a):
        raise ValueError("V/A must be finite")
    if not 1.0 <= v <= 9.0 or not 1.0 <= a <= 9.0:
        raise ValueError("V/A must stay within [1, 9]")
    return v, a


def parse_extraction_payload(
    payload: Any, text: str, *, allow_null: bool = False
) -> tuple[tuple[ExtractionItem, ...], list[str]]:
    """Parse and validate model output, discarding only invalid individual items."""

    if isinstance(payload, dict):
        payload = _pick(payload, "items", "Quadruplet", "quadruplets")
    if not isinstance(payload, list):
        raise ValueError("JSON payload does not contain an items list")

    valid: list[ExtractionItem] = []
    errors: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw_item in enumerate(payload):
        try:
            if not isinstance(raw_item, dict):
                raise ValueError("item is not an object")
            aspect = _pick(raw_item, "aspect", "Aspect")
            opinion = _pick(raw_item, "opinion", "Opinion")
            category = _pick(raw_item, "category", "Category")
            if not isinstance(aspect, str) or not aspect.strip():
                raise ValueError("missing Aspect")
            if not isinstance(opinion, str) or not opinion.strip():
                raise ValueError("missing Opinion")
            if not isinstance(category, str) or not category.strip():
                raise ValueError("missing Category")
            aspect = aspect.strip()
            opinion = opinion.strip()
            category = category.strip().upper()
            if aspect == "NULL" and not allow_null:
                raise ValueError("NULL Aspect is not allowed for this dataset")
            if opinion == "NULL" and not allow_null:
                raise ValueError("NULL Opinion is not allowed for this dataset")
            if aspect != "NULL" and aspect not in text:
                raise ValueError(f"Aspect {aspect!r} is not an exact text span")
            if opinion != "NULL" and opinion not in text:
                raise ValueError(f"Opinion {opinion!r} is not an exact text span")
            if category not in RESTAURANT_CATEGORIES:
                raise ValueError(f"illegal restaurant category {category!r}")
            score = _parse_score(raw_item)
            key = (aspect.lower(), opinion.lower(), category)
            if key in seen:
                raise ValueError("duplicate Aspect/Opinion/Category")
            seen.add(key)
            valid.append(ExtractionItem(aspect, opinion, category, score))
        except ValueError as exc:
            errors.append(f"item {index}: {exc}")
    return tuple(valid), errors


def write_extraction_predictions(
    records: Sequence[ExtractionRecord],
    predictions: dict[str, Sequence[ExtractionItem]],
    task3_path: str | Path,
    task2_path: str | Path | None = None,
) -> None:
    """Write official Task 3 output and an optional de-duplicated Task 2 view."""

    task3_destination = Path(task3_path)
    task3_destination.parent.mkdir(parents=True, exist_ok=True)
    task2_destination = Path(task2_path) if task2_path is not None else None
    task2_handle = (
        task2_destination.open("w", encoding="utf-8")
        if task2_destination is not None
        else None
    )
    try:
        with task3_destination.open("w", encoding="utf-8") as task3_handle:
            for record in records:
                items = predictions.get(record.record_id)
                if items is None:
                    raise ValueError(f"Missing prediction for ID {record.record_id!r}")
                quadruplets = []
                triplets = []
                seen_task2: set[tuple[str, str]] = set()
                for item in items:
                    va = f"{item.score[0]:.2f}#{item.score[1]:.2f}"
                    quadruplets.append(
                        {
                            "Aspect": item.aspect,
                            "Category": item.category,
                            "Opinion": item.opinion,
                            "VA": va,
                        }
                    )
                    task2_key = (item.aspect.lower(), item.opinion.lower())
                    if task2_key not in seen_task2:
                        seen_task2.add(task2_key)
                        triplets.append(
                            {"Aspect": item.aspect, "Opinion": item.opinion, "VA": va}
                        )
                json.dump(
                    {"ID": record.record_id, "Quadruplet": quadruplets},
                    task3_handle,
                    ensure_ascii=False,
                )
                task3_handle.write("\n")
                if task2_handle is not None:
                    json.dump(
                        {"ID": record.record_id, "Triplet": triplets},
                        task2_handle,
                        ensure_ascii=False,
                    )
                    task2_handle.write("\n")
    finally:
        if task2_handle is not None:
            task2_handle.close()
