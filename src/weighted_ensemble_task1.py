"""Select and apply a two-model Task 1 ensemble using Dev only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dimabsa_data import load_task1_records, write_task1_predictions
from evaluate_task1 import evaluate


def _validate(left, right) -> None:
    if len(left) != len(right):
        raise ValueError("Prediction record counts differ")
    for first, second in zip(left, right):
        if first.record_id != second.record_id or first.aspects != second.aspects:
            raise ValueError("Prediction IDs, aspect text, or order differ")


def blend(left, right, left_weight: float):
    if not 0.0 <= left_weight <= 1.0:
        raise ValueError("Ensemble weight must be within [0, 1]")
    _validate(left, right)
    output = {}
    for first, second in zip(left, right):
        if first.gold_scores is None or second.gold_scores is None:
            raise ValueError("Ensemble inputs must contain predictions")
        output[first.record_id] = tuple(
            (
                left_weight * left_score[0] + (1.0 - left_weight) * right_score[0],
                left_weight * left_score[1] + (1.0 - left_weight) * right_score[1],
            )
            for left_score, right_score in zip(first.gold_scores, second.gold_scores)
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--gold", required=True)
    fit.add_argument("--left", required=True)
    fit.add_argument("--right", required=True)
    fit.add_argument("--params", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument(
        "--weights", type=float, nargs="+", default=[i / 10 for i in range(11)]
    )
    apply = subparsers.add_parser("apply")
    apply.add_argument("--left", required=True)
    apply.add_argument("--right", required=True)
    apply.add_argument("--params", required=True)
    apply.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left = load_task1_records(args.left, require_gold=True, require_text=False)
    right = load_task1_records(args.right, require_gold=True, require_text=False)
    if args.command == "fit":
        gold = load_task1_records(args.gold, require_gold=True)
        scores = {}
        for weight in args.weights:
            predictions = blend(left, right, weight)
            temporary = [
                record.__class__(record.record_id, "", record.aspects, predictions[record.record_id])
                for record in gold
            ]
            scores[str(weight)] = evaluate(gold, temporary, allow_subset=False)["RMSE_VA"]
        best = min(args.weights, key=lambda weight: (scores[str(weight)], -weight))
        parameters = {
            "left_weight": best,
            "right_weight": 1.0 - best,
            "dev_rmse_by_left_weight": scores,
        }
        Path(args.params).write_text(
            json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        parameters = json.loads(Path(args.params).read_text(encoding="utf-8"))
    write_task1_predictions(left, blend(left, right, parameters["left_weight"]), args.output)
    print(json.dumps(parameters, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
