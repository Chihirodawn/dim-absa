"""Blend two aligned Task 1 prediction files with a frozen weight."""

from __future__ import annotations

import argparse

from dimabsa_data import load_task1_records, write_task1_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--secondary", required=True)
    parser.add_argument("--primary-weight", type=float, required=True)
    parser.add_argument("--output-pred", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.primary_weight <= 1.0:
        raise ValueError("primary weight must be in [0, 1]")
    primary = load_task1_records(args.primary, require_gold=True, require_text=False)
    secondary = load_task1_records(args.secondary, require_gold=True, require_text=False)
    if len(primary) != len(secondary):
        raise ValueError("prediction files have different lengths")
    output = {}
    for left, right in zip(primary, secondary):
        if left.record_id != right.record_id or left.aspects != right.aspects:
            raise ValueError("prediction files are not aligned")
        assert left.gold_scores is not None and right.gold_scores is not None
        output[left.record_id] = tuple(
            (
                args.primary_weight * lv + (1.0 - args.primary_weight) * rv,
                args.primary_weight * la + (1.0 - args.primary_weight) * ra,
            )
            for (lv, la), (rv, ra) in zip(left.gold_scores, right.gold_scores)
        )
    write_task1_predictions(primary, output, args.output_pred)


if __name__ == "__main__":
    main()
