"""Fit and apply VA-only affine calibration for Task 2/3 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calibrate_task1 import calibrate_score, fit_affine
from dimabsa_data import parse_va


def load_rows(path: str | Path, task: int) -> list[dict[str, Any]]:
    key = "Triplet" if task == 2 else "Quadruplet"
    rows = []
    seen_ids = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row.get("ID")
            items = row.get(key)
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{line_number}: invalid ID")
            if record_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate ID")
            if not isinstance(items, list):
                raise ValueError(f"{path}:{line_number}: {key} is not a list")
            seen_ids.add(record_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def fit_parameters(
    gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], task: int
) -> dict[str, Any]:
    key = "Triplet" if task == 2 else "Quadruplet"
    fields = ("Aspect", "Opinion") if task == 2 else (
        "Aspect",
        "Opinion",
        "Category",
    )
    gold_by_id = {row["ID"]: row for row in gold_rows}
    pred_by_id = {row["ID"]: row for row in pred_rows}
    if set(gold_by_id) != set(pred_by_id):
        raise ValueError("Gold and prediction ID sets differ")

    predicted_v: list[float] = []
    predicted_a: list[float] = []
    gold_v: list[float] = []
    gold_a: list[float] = []
    for record_id, gold_row in gold_by_id.items():
        gold_items = {
            tuple(str(item[field]).lower() for field in fields): item
            for item in gold_row[key]
        }
        for prediction in pred_by_id[record_id][key]:
            match = gold_items.get(
                tuple(str(prediction[field]).lower() for field in fields)
            )
            if match is None:
                continue
            pv, pa = parse_va(prediction["VA"])
            gv, ga = parse_va(match["VA"])
            predicted_v.append(pv)
            predicted_a.append(pa)
            gold_v.append(gv)
            gold_a.append(ga)
    slope_v, intercept_v = fit_affine(predicted_v, gold_v)
    slope_a, intercept_a = fit_affine(predicted_a, gold_a)
    return {
        "method": "matched_dev_affine_keep_all_predictions",
        "fit_task": task,
        "fit_matches": len(predicted_v),
        "V": {"slope": slope_v, "intercept": intercept_v},
        "A": {"slope": slope_a, "intercept": intercept_a},
    }


def apply_parameters(
    rows: list[dict[str, Any]], parameters: dict[str, Any], task: int
) -> list[dict[str, Any]]:
    key = "Triplet" if task == 2 else "Quadruplet"
    output = json.loads(json.dumps(rows, ensure_ascii=False))
    for row in output:
        for item in row[key]:
            v, a = calibrate_score(parse_va(item["VA"]), parameters)
            item["VA"] = f"{v:.2f}#{a:.2f}"
    return output


def write_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate extraction VA values")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--gold", required=True)
    fit_parser.add_argument("--pred", required=True)
    fit_parser.add_argument("--task", type=int, choices=(2, 3), required=True)
    fit_parser.add_argument("--output-params", required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--pred", required=True)
    apply_parser.add_argument("--params", required=True)
    apply_parser.add_argument("--task", type=int, choices=(2, 3), required=True)
    apply_parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_rows = load_rows(args.pred, args.task)
    if args.command == "fit":
        parameters = fit_parameters(load_rows(args.gold, args.task), prediction_rows, args.task)
        destination = Path(args.output_params)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(parameters, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        parameters = json.loads(Path(args.params).read_text(encoding="utf-8"))
        write_rows(apply_parameters(prediction_rows, parameters, args.task), args.output)
    print(json.dumps(parameters, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
