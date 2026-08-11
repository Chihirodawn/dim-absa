"""Dev-only confidence filtering and VA calibration for Tasks 2 and 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from calibrate_task1 import calibrate_score, fit_affine
from dimabsa_data import parse_va
from dimabsa_extraction import (
    ExtractionItem,
    ExtractionRecord,
    load_extraction_records,
    write_extraction_predictions,
)


# Qwen repeatedly uses these four central grid points for uncertain factual spans.
# The set is selected on dev and then frozen before test application.
DEFAULT_UNCERTAIN_VA = (
    "5.00#5.00",
    "5.00#5.50",
    "5.50#5.00",
    "5.50#5.50",
)


def load_task3_predictions(
    path: str | Path,
) -> tuple[list[ExtractionRecord], dict[str, tuple[ExtractionItem, ...]]]:
    source = Path(path)
    records = []
    predictions = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row.get("ID")
            raw_items = row.get("Quadruplet")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{source}:{line_number}: invalid ID")
            if record_id in predictions:
                raise ValueError(f"{source}:{line_number}: duplicate ID")
            if not isinstance(raw_items, list):
                raise ValueError(f"{source}:{line_number}: Quadruplet is not a list")
            items = []
            for item in raw_items:
                items.append(
                    ExtractionItem(
                        item["Aspect"],
                        item["Opinion"],
                        item["Category"],
                        parse_va(item["VA"]),
                    )
                )
            records.append(ExtractionRecord(record_id, "", None))
            predictions[record_id] = tuple(items)
    if not records:
        raise ValueError(f"No predictions found in {source}")
    return records, predictions


def _fit_parameters(
    gold_records: Sequence[ExtractionRecord],
    predictions: dict[str, tuple[ExtractionItem, ...]],
) -> dict[str, Any]:
    if {record.record_id for record in gold_records} != set(predictions):
        raise ValueError("Gold and prediction ID sets differ")
    pred_v: list[float] = []
    pred_a: list[float] = []
    gold_v: list[float] = []
    gold_a: list[float] = []
    uncertain = set(DEFAULT_UNCERTAIN_VA)
    for record in gold_records:
        if record.gold_items is None:
            raise ValueError(f"ID {record.record_id}: gold items are missing")
        gold_by_task2_key = {
            (item.aspect.lower(), item.opinion.lower()): item
            for item in record.gold_items
        }
        for prediction in predictions[record.record_id]:
            raw_va = f"{prediction.score[0]:.2f}#{prediction.score[1]:.2f}"
            if raw_va in uncertain:
                continue
            gold = gold_by_task2_key.get(
                (prediction.aspect.lower(), prediction.opinion.lower())
            )
            if gold is None:
                continue
            pred_v.append(prediction.score[0])
            pred_a.append(prediction.score[1])
            gold_v.append(gold.score[0])
            gold_a.append(gold.score[1])
    slope_v, intercept_v = fit_affine(pred_v, gold_v)
    slope_a, intercept_a = fit_affine(pred_a, gold_a)
    return {
        "method": "drop_frozen_uncertain_grid_then_task2_matched_dev_affine",
        "fit_matches": len(pred_v),
        "drop_exact_va": list(DEFAULT_UNCERTAIN_VA),
        "V": {"slope": slope_v, "intercept": intercept_v},
        "A": {"slope": slope_a, "intercept": intercept_a},
    }


def _apply(
    predictions: dict[str, tuple[ExtractionItem, ...]], parameters: dict[str, Any]
) -> dict[str, tuple[ExtractionItem, ...]]:
    uncertain = set(parameters["drop_exact_va"])
    output = {}
    for record_id, items in predictions.items():
        calibrated = []
        for item in items:
            raw_va = f"{item.score[0]:.2f}#{item.score[1]:.2f}"
            if raw_va in uncertain:
                continue
            calibrated.append(
                ExtractionItem(
                    item.aspect,
                    item.opinion,
                    item.category,
                    calibrate_score(item.score, parameters),
                )
            )
        output[record_id] = tuple(calibrated)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate DimABSA Task 2/3 output")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--gold", required=True)
    fit_parser.add_argument("--pred", required=True)
    fit_parser.add_argument("--output-params", required=True)
    fit_parser.add_argument("--output-task3", required=True)
    fit_parser.add_argument("--output-task2", required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--pred", required=True)
    apply_parser.add_argument("--params", required=True)
    apply_parser.add_argument("--output-task3", required=True)
    apply_parser.add_argument("--output-task2", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, predictions = load_task3_predictions(args.pred)
    if args.command == "fit":
        gold_records = load_extraction_records(args.gold, require_gold=True)
        parameters = _fit_parameters(gold_records, predictions)
        parameter_path = Path(args.output_params)
        parameter_path.parent.mkdir(parents=True, exist_ok=True)
        parameter_path.write_text(
            json.dumps(parameters, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        parameters = json.loads(Path(args.params).read_text(encoding="utf-8"))
    write_extraction_predictions(
        records,
        _apply(predictions, parameters),
        args.output_task3,
        args.output_task2,
    )
    print(json.dumps(parameters, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
