"""Strict continuous-F1 evaluation for DimABSA Tasks 2 and 3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: str | Path, task: int) -> dict[str, list[dict[str, Any]]]:
    key = "Triplet" if task == 2 else "Quadruplet"
    fallback_key = "Quadruplet" if task == 2 else key
    rows: dict[str, list[dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row.get("ID")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{line_number}: invalid ID")
            if record_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate ID")
            items = row.get(key, row.get(fallback_key, []))
            if not isinstance(items, list):
                raise ValueError(f"{path}:{line_number}: {key} is not a list")
            rows[record_id] = items
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _match_key(item: dict[str, Any], task: int) -> tuple[str, ...]:
    fields = ("Aspect", "Opinion") if task == 2 else ("Aspect", "Opinion", "Category")
    values = []
    for field in fields:
        value = item.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"missing {field}")
        values.append(value.lower())
    return tuple(values)


def _score(item: dict[str, Any]) -> tuple[float, float]:
    value = item.get("VA")
    if not isinstance(value, str) or value.count("#") != 1:
        raise ValueError("invalid VA")
    v, a = (float(part) for part in value.split("#"))
    if not 1.0 <= v <= 9.0 or not 1.0 <= a <= 9.0:
        raise ValueError("VA outside [1, 9]")
    return v, a


def evaluate(
    gold_path: str | Path,
    prediction_path: str | Path,
    *,
    task: int,
    allow_subset: bool = False,
) -> dict[str, float | int]:
    """Mirror the official metric while also reporting exact structural F1."""

    if task not in {2, 3}:
        raise ValueError("task must be 2 or 3")
    gold = _load(gold_path, task)
    predictions = _load(prediction_path, task)
    if allow_subset:
        unknown = set(predictions) - set(gold)
        if unknown:
            raise ValueError(f"prediction has unknown IDs: {sorted(unknown)[:3]}")
        gold = {record_id: gold[record_id] for record_id in predictions}
    elif set(gold) != set(predictions):
        raise ValueError(
            f"ID sets differ: missing={len(set(gold) - set(predictions))}, "
            f"extra={len(set(predictions) - set(gold))}"
        )

    exact_tp = 0
    continuous_tp = 0.0
    false_positive = 0
    false_negative = 0
    maximum_distance = math.sqrt(128.0)
    for record_id, gold_items in gold.items():
        predicted_items = predictions.get(record_id, [])
        predicted_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
        for item in predicted_items:
            key = _match_key(item, task)
            if key in predicted_by_key:
                raise ValueError(f"ID {record_id}: duplicate prediction key {key}")
            predicted_by_key[key] = item
        matched = 0
        for gold_item in gold_items:
            key = _match_key(gold_item, task)
            predicted_item = predicted_by_key.get(key)
            if predicted_item is None:
                false_negative += 1
                continue
            gold_v, gold_a = _score(gold_item)
            pred_v, pred_a = _score(predicted_item)
            distance = math.hypot(pred_v - gold_v, pred_a - gold_a)
            continuous_tp += max(0.0, 1.0 - distance / maximum_distance)
            exact_tp += 1
            matched += 1
        false_positive += len(predicted_items) - matched

    exact_precision = exact_tp / (exact_tp + false_positive) if exact_tp + false_positive else 0.0
    exact_recall = exact_tp / (exact_tp + false_negative) if exact_tp + false_negative else 0.0
    exact_f1 = (
        2 * exact_precision * exact_recall / (exact_precision + exact_recall)
        if exact_precision + exact_recall
        else 0.0
    )
    continuous_precision = (
        continuous_tp / (exact_tp + false_positive)
        if exact_tp + false_positive
        else 0.0
    )
    continuous_recall = (
        continuous_tp / (exact_tp + false_negative)
        if exact_tp + false_negative
        else 0.0
    )
    continuous_f1 = (
        2
        * continuous_precision
        * continuous_recall
        / (continuous_precision + continuous_recall)
        if continuous_precision + continuous_recall
        else 0.0
    )
    return {
        "records": len(gold),
        "gold_items": exact_tp + false_negative,
        "predicted_items": exact_tp + false_positive,
        "exact_TP": exact_tp,
        "FP": false_positive,
        "FN": false_negative,
        "exact_precision": exact_precision,
        "exact_recall": exact_recall,
        "exact_F1": exact_f1,
        "continuous_TP": continuous_tp,
        "continuous_precision": continuous_precision,
        "continuous_recall": continuous_recall,
        "continuous_F1": continuous_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file", required=True)
    parser.add_argument("--prediction-file", required=True)
    parser.add_argument("--task", type=int, choices=[2, 3], required=True)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--output-file")
    args = parser.parse_args()
    metrics = evaluate(
        args.gold_file,
        args.prediction_file,
        task=args.task,
        allow_subset=args.allow_subset,
    )
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_file:
        destination = Path(args.output_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
