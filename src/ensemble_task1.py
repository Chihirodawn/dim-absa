"""Equal-weight Task 1 ensemble with dev-only affine calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from calibrate_task1 import calibrate_score, fit_affine
from dimabsa_data import Score, Task1Record, load_task1_records, write_task1_predictions


def _validate_alignment(reference: Sequence[Task1Record], other: Sequence[Task1Record]) -> None:
    if len(reference) != len(other):
        raise ValueError("Ensemble prediction files have different record counts")
    for expected, actual in zip(reference, other):
        if expected.record_id != actual.record_id or expected.aspects != actual.aspects:
            raise ValueError("Ensemble prediction IDs, order, or aspects differ")
        if actual.gold_scores is None:
            raise ValueError(f"ID {actual.record_id!r}: prediction has no VA scores")


def average_predictions(
    prediction_sets: Sequence[Sequence[Task1Record]],
) -> tuple[list[Task1Record], dict[str, tuple[Score, ...]]]:
    if len(prediction_sets) < 2:
        raise ValueError("Ensemble requires at least two prediction files")
    reference = list(prediction_sets[0])
    for records in prediction_sets:
        _validate_alignment(reference, records)
    averaged: dict[str, tuple[Score, ...]] = {}
    for record_index, record in enumerate(reference):
        assert record.gold_scores is not None
        scores = []
        for aspect_index in range(len(record.aspects)):
            values = [
                records[record_index].gold_scores[aspect_index]
                for records in prediction_sets
            ]
            scores.append(
                (
                    sum(score[0] for score in values) / len(values),
                    sum(score[1] for score in values) / len(values),
                )
            )
        averaged[record.record_id] = tuple(scores)
    return reference, averaged


def fit_parameters(
    gold: Sequence[Task1Record],
    averaged_records: Sequence[Task1Record],
) -> dict:
    _validate_alignment(gold, averaged_records)
    gold_v: list[float] = []
    gold_a: list[float] = []
    pred_v: list[float] = []
    pred_a: list[float] = []
    for target, prediction in zip(gold, averaged_records):
        if target.gold_scores is None or prediction.gold_scores is None:
            raise ValueError("Gold and predictions must contain VA scores")
        for (gv, ga), (pv, pa) in zip(target.gold_scores, prediction.gold_scores):
            gold_v.append(gv)
            gold_a.append(ga)
            pred_v.append(pv)
            pred_a.append(pa)
    slope_v, intercept_v = fit_affine(pred_v, gold_v)
    slope_a, intercept_a = fit_affine(pred_a, gold_a)
    return {
        "method": "equal_weight_raw_ensemble_then_dev_affine_clip_1_9",
        "ensemble_size": 0,
        "aspects": len(gold_v),
        "V": {"slope": slope_v, "intercept": intercept_v},
        "A": {"slope": slope_a, "intercept": intercept_a},
    }


def _load_ensemble(paths: Sequence[str]) -> tuple[list[Task1Record], dict[str, tuple[Score, ...]]]:
    prediction_sets = [
        load_task1_records(path, require_gold=True, require_text=False) for path in paths
    ]
    return average_predictions(prediction_sets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Equal-weight DimABSA Task 1 ensemble")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--gold", required=True)
    fit_parser.add_argument("--pred", nargs="+", required=True)
    fit_parser.add_argument("--output-params", required=True)
    fit_parser.add_argument("--output-pred", required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--pred", nargs="+", required=True)
    apply_parser.add_argument("--params", required=True)
    apply_parser.add_argument("--output-pred", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, averaged = _load_ensemble(args.pred)
    write_task1_predictions(records, averaged, args.output_pred + ".raw")
    averaged_records = load_task1_records(
        args.output_pred + ".raw", require_gold=True, require_text=False
    )
    if args.command == "fit":
        gold = load_task1_records(args.gold, require_gold=True)
        parameters = fit_parameters(gold, averaged_records)
        parameters["ensemble_size"] = len(args.pred)
        parameters["prediction_files"] = [Path(path).name for path in args.pred]
        destination = Path(args.output_params)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(parameters, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        parameters = json.loads(Path(args.params).read_text(encoding="utf-8"))
        if parameters["ensemble_size"] != len(args.pred):
            raise ValueError("Prediction count does not match frozen ensemble size")
    calibrated = {
        record.record_id: tuple(
            calibrate_score(score, parameters) for score in averaged[record.record_id]
        )
        for record in records
    }
    write_task1_predictions(records, calibrated, args.output_pred)
    Path(args.output_pred + ".raw").unlink()
    print(json.dumps(parameters, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
