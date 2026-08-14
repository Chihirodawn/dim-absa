"""Cross-fit Task 1 Ridge calibration on Train OOF predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from calibrate_task1 import _apply, _fit_parameters
from dimabsa_data import load_task1_records, write_task1_predictions
from evaluate_task1 import evaluate


def _fold(record_id: str, folds: int) -> int:
    digest = hashlib.sha256(("calibration:" + record_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output-params", required=True)
    parser.add_argument("--output-crossfit-pred", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--ridge-alphas", type=float, nargs="+", default=[0.0, 0.01, 0.1, 1.0, 10.0]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_task1_records(args.gold, require_gold=True)
    pred = load_task1_records(args.pred, require_gold=True, require_text=False)
    pred_by_id = {record.record_id: record for record in pred}
    crossfit = {}
    for fold in range(args.folds):
        fit_gold = [record for record in gold if _fold(record.record_id, args.folds) != fold]
        fit_pred = [pred_by_id[record.record_id] for record in fit_gold]
        valid_gold = [record for record in gold if _fold(record.record_id, args.folds) == fold]
        valid_pred = [pred_by_id[record.record_id] for record in valid_gold]
        params = _fit_parameters(
            fit_gold,
            fit_pred,
            method="ridge",
            ridge_alphas=args.ridge_alphas,
            folds=args.folds,
        )
        crossfit.update(_apply(valid_pred, params))
    write_task1_predictions(gold, crossfit, args.output_crossfit_pred)
    crossfit_records = load_task1_records(
        args.output_crossfit_pred, require_gold=True, require_text=False
    )
    raw_metrics = evaluate(gold, pred, allow_subset=False)
    crossfit_metrics = evaluate(gold, crossfit_records, allow_subset=False)
    final_params = _fit_parameters(
        gold,
        pred,
        method="ridge",
        ridge_alphas=args.ridge_alphas,
        folds=args.folds,
    )
    improvement = raw_metrics["RMSE_VA"] - crossfit_metrics["RMSE_VA"]
    report = {
        "raw_oof": raw_metrics,
        "crossfit_calibrated_oof": crossfit_metrics,
        "rmse_improvement": improvement,
        "apply_to_test": improvement >= 0.01,
        "minimum_required_improvement": 0.01,
    }
    Path(args.output_params).write_text(
        json.dumps(final_params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path(args.output_report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
