"""Strict local Task 1 evaluator with duplicate-aspect and order validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean

from dimabsa_data import Task1Record, load_task1_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DimABSA Task 1 JSONL")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Evaluate only predicted IDs; intended for smoke runs",
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def evaluate(
    gold_records: list[Task1Record],
    pred_records: list[Task1Record],
    *,
    allow_subset: bool,
) -> dict[str, float | int | None]:
    gold_by_id = {record.record_id: record for record in gold_records}
    pred_ids = [record.record_id for record in pred_records]
    gold_ids = [record.record_id for record in gold_records]
    unknown = [record_id for record_id in pred_ids if record_id not in gold_by_id]
    if unknown:
        raise ValueError(f"Predictions contain unknown IDs: {unknown[:10]}")
    if not allow_subset and pred_ids != gold_ids:
        raise ValueError("Full evaluation requires the same IDs in the same order as gold")

    gold_v: list[float] = []
    gold_a: list[float] = []
    pred_v: list[float] = []
    pred_a: list[float] = []
    squared_error = 0.0
    squared_error_v = 0.0
    squared_error_a = 0.0
    aspect_count = 0
    for prediction in pred_records:
        gold = gold_by_id[prediction.record_id]
        if prediction.aspects != gold.aspects:
            raise ValueError(
                f"ID {prediction.record_id!r}: aspect text/order differs from gold"
            )
        if gold.gold_scores is None or prediction.gold_scores is None:
            raise ValueError(f"ID {prediction.record_id!r}: missing VA values")
        for (gv, ga), (pv, pa) in zip(gold.gold_scores, prediction.gold_scores):
            gold_v.append(gv)
            gold_a.append(ga)
            pred_v.append(pv)
            pred_a.append(pa)
            squared_error += (pv - gv) ** 2 + (pa - ga) ** 2
            squared_error_v += (pv - gv) ** 2
            squared_error_a += (pa - ga) ** 2
            aspect_count += 1
    if aspect_count == 0:
        raise ValueError("No aspects were evaluated")
    rmse = math.sqrt(squared_error / aspect_count)
    return {
        "records": len(pred_records),
        "aspects": aspect_count,
        "coverage": len(pred_records) / len(gold_records),
        "PCC_V": _pearson(pred_v, gold_v),
        "PCC_A": _pearson(pred_a, gold_a),
        "RMSE_V": math.sqrt(squared_error_v / aspect_count),
        "RMSE_A": math.sqrt(squared_error_a / aspect_count),
        "RMSE_VA": rmse,
        "RMSE_VA_NORMALIZED": rmse / math.sqrt(128),
    }


def main() -> None:
    args = parse_args()
    gold = load_task1_records(args.gold, require_gold=True)
    predictions = load_task1_records(
        args.pred, require_gold=True, require_text=False
    )
    metrics = evaluate(gold, predictions, allow_subset=args.allow_subset)
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        destination = Path(args.output_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
