"""Fit and apply dev-only affine calibration for DimABSA Task 1 scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Sequence

from dimabsa_data import Score, Task1Record, load_task1_records, write_task1_predictions


def fit_affine(predicted: Sequence[float], gold: Sequence[float]) -> tuple[float, float]:
    """Return least-squares slope and intercept for gold ~= slope * pred + intercept."""

    if len(predicted) != len(gold) or not predicted:
        raise ValueError("predicted and gold must be non-empty sequences of equal length")
    pred_mean = fmean(predicted)
    gold_mean = fmean(gold)
    variance = sum((value - pred_mean) ** 2 for value in predicted)
    if variance == 0.0:
        return 0.0, gold_mean
    covariance = sum(
        (pred - pred_mean) * (target - gold_mean)
        for pred, target in zip(predicted, gold)
    )
    slope = covariance / variance
    return slope, gold_mean - slope * pred_mean


def fit_ridge(
    predicted: Sequence[float], gold: Sequence[float], alpha: float
) -> tuple[float, float]:
    """Fit one-dimensional Ridge with an unregularized intercept."""

    if len(predicted) != len(gold) or not predicted:
        raise ValueError("predicted and gold must be non-empty sequences of equal length")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    pred_mean, gold_mean = fmean(predicted), fmean(gold)
    centered = [value - pred_mean for value in predicted]
    scale = (sum(value * value for value in centered) / len(centered)) ** 0.5
    if scale == 0.0:
        return 0.0, gold_mean
    standardized = [value / scale for value in centered]
    numerator = sum(
        value * (target - gold_mean) for value, target in zip(standardized, gold)
    )
    denominator = sum(value * value for value in standardized) + alpha * len(gold)
    slope = numerator / denominator / scale
    return slope, gold_mean - slope * pred_mean


def select_ridge(
    predicted: Sequence[float],
    gold: Sequence[float],
    groups: Sequence[str],
    alphas: Sequence[float],
    folds: int = 3,
) -> tuple[float, float, float, dict[str, float]]:
    """Select alpha with grouped cross-validation, then refit on all examples."""

    if len(groups) != len(gold):
        raise ValueError("groups must align with predicted and gold")
    unique_groups = list(dict.fromkeys(groups))
    if folds < 2 or len(unique_groups) < folds:
        raise ValueError("Not enough record groups for grouped Ridge selection")
    fold_by_group = {group: index % folds for index, group in enumerate(unique_groups)}
    cv_rmse: dict[str, float] = {}
    for alpha in alphas:
        errors: list[float] = []
        for fold in range(folds):
            train_indices = [
                i for i, group in enumerate(groups) if fold_by_group[group] != fold
            ]
            valid_indices = [
                i for i, group in enumerate(groups) if fold_by_group[group] == fold
            ]
            slope, intercept = fit_ridge(
                [predicted[i] for i in train_indices],
                [gold[i] for i in train_indices],
                alpha,
            )
            errors.extend(
                (slope * predicted[i] + intercept - gold[i]) ** 2
                for i in valid_indices
            )
        cv_rmse[str(alpha)] = (sum(errors) / len(errors)) ** 0.5
    best_alpha = min(alphas, key=lambda value: (cv_rmse[str(value)], value))
    slope, intercept = fit_ridge(predicted, gold, best_alpha)
    return slope, intercept, best_alpha, cv_rmse


def calibrate_score(score: Score, parameters: dict) -> Score:
    values = []
    for value, dimension in zip(score, ("V", "A")):
        config = parameters[dimension]
        calibrated = config["slope"] * value + config["intercept"]
        values.append(min(9.0, max(1.0, calibrated)))
    return values[0], values[1]


def _validate_alignment(
    gold_records: Sequence[Task1Record], pred_records: Sequence[Task1Record]
) -> None:
    if [record.record_id for record in pred_records] != [
        record.record_id for record in gold_records
    ]:
        raise ValueError("Prediction IDs/order do not exactly match calibration gold")
    for gold, prediction in zip(gold_records, pred_records):
        if gold.aspects != prediction.aspects:
            raise ValueError(f"ID {gold.record_id!r}: aspect text/order differs from gold")


def _fit_parameters(
    gold_records: Sequence[Task1Record],
    pred_records: Sequence[Task1Record],
    *,
    method: str = "affine",
    ridge_alphas: Sequence[float] = (0.0, 0.01, 0.1, 1.0, 10.0),
    folds: int = 3,
) -> dict:
    _validate_alignment(gold_records, pred_records)
    gold_v: list[float] = []
    gold_a: list[float] = []
    pred_v: list[float] = []
    pred_a: list[float] = []
    groups: list[str] = []
    for gold, prediction in zip(gold_records, pred_records):
        if gold.gold_scores is None or prediction.gold_scores is None:
            raise ValueError(f"ID {gold.record_id!r}: VA scores are missing")
        for (gv, ga), (pv, pa) in zip(gold.gold_scores, prediction.gold_scores):
            gold_v.append(gv)
            gold_a.append(ga)
            pred_v.append(pv)
            pred_a.append(pa)
            groups.append(gold.record_id)
    if method == "ridge":
        slope_v, intercept_v, alpha_v, cv_v = select_ridge(
            pred_v, gold_v, groups, ridge_alphas, folds
        )
        slope_a, intercept_a, alpha_a, cv_a = select_ridge(
            pred_a, gold_a, groups, ridge_alphas, folds
        )
        dimensions = {
            "V": {
                "slope": slope_v,
                "intercept": intercept_v,
                "alpha": alpha_v,
                "cv_rmse": cv_v,
            },
            "A": {
                "slope": slope_a,
                "intercept": intercept_a,
                "alpha": alpha_a,
                "cv_rmse": cv_a,
            },
        }
    else:
        slope_v, intercept_v = fit_affine(pred_v, gold_v)
        slope_a, intercept_a = fit_affine(pred_a, gold_a)
        dimensions = {
            "V": {"slope": slope_v, "intercept": intercept_v},
            "A": {"slope": slope_a, "intercept": intercept_a},
        }
    return {
        "method": f"per_dimension_{method}_then_clip_1_9",
        "aspects": len(gold_v),
        **dimensions,
    }


def _apply(records: Sequence[Task1Record], parameters: dict) -> dict[str, tuple[Score, ...]]:
    predictions: dict[str, tuple[Score, ...]] = {}
    for record in records:
        if record.gold_scores is None:
            raise ValueError(f"ID {record.record_id!r}: predictions contain no VA scores")
        predictions[record.record_id] = tuple(
            calibrate_score(score, parameters) for score in record.gold_scores
        )
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate DimABSA Task 1 scores")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--gold", required=True)
    fit_parser.add_argument("--pred", required=True)
    fit_parser.add_argument("--output-params", required=True)
    fit_parser.add_argument("--output-pred", required=True)
    fit_parser.add_argument("--method", choices=["affine", "ridge"], default="affine")
    fit_parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.1, 1.0, 10.0],
    )
    fit_parser.add_argument("--folds", type=int, default=3)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--pred", required=True)
    apply_parser.add_argument("--params", required=True)
    apply_parser.add_argument("--output-pred", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_records = load_task1_records(
        args.pred, require_gold=True, require_text=False
    )
    if args.command == "fit":
        gold_records = load_task1_records(args.gold, require_gold=True)
        parameters = _fit_parameters(
            gold_records,
            pred_records,
            method=args.method,
            ridge_alphas=args.ridge_alphas,
            folds=args.folds,
        )
        parameter_path = Path(args.output_params)
        parameter_path.parent.mkdir(parents=True, exist_ok=True)
        parameter_path.write_text(
            json.dumps(parameters, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        parameters = json.loads(Path(args.params).read_text(encoding="utf-8"))
    write_task1_predictions(
        pred_records,
        _apply(pred_records, parameters),
        args.output_pred,
    )
    print(json.dumps(parameters, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
