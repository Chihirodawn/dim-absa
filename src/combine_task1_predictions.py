"""Average aligned predictions or merge disjoint Task 1 prediction shards."""

from __future__ import annotations

import argparse

from dimabsa_data import load_task1_records, write_task1_predictions
from ensemble_task1 import average_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["average", "merge"], required=True)
    parser.add_argument("--pred", nargs="+", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_task1_records(args.gold, require_gold=True)
    prediction_sets = [
        load_task1_records(path, require_gold=True, require_text=False)
        for path in args.pred
    ]
    if args.mode == "average":
        _, predictions = average_predictions(prediction_sets)
    else:
        by_id = {}
        for records in prediction_sets:
            for record in records:
                if record.record_id in by_id:
                    raise ValueError(f"Duplicate prediction ID {record.record_id!r}")
                by_id[record.record_id] = record
        if set(by_id) != {record.record_id for record in gold}:
            raise ValueError("Merged prediction shards do not exactly cover gold IDs")
        predictions = {
            record.record_id: by_id[record.record_id].gold_scores for record in gold
        }
    write_task1_predictions(gold, predictions, args.output)


if __name__ == "__main__":
    main()
