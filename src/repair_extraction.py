"""Repair fallback rows from already saved complete item objects in diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrate_extraction import load_task3_predictions
from dimabsa_extraction import load_extraction_records, write_extraction_predictions
from run_extraction import parse_model_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", required=True)
    parser.add_argument("--base-task3", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output-task3", required=True)
    parser.add_argument("--output-task2", required=True)
    args = parser.parse_args()

    source_records = load_extraction_records(args.source_data, require_gold=False)
    source_by_id = {record.record_id: record for record in source_records}
    output_records, predictions = load_task3_predictions(args.base_task3)
    repaired = 0
    with Path(args.diagnostics).open("r", encoding="utf-8") as handle:
        for line in handle:
            diagnostic = json.loads(line)
            if diagnostic.get("status") != "fallback_empty":
                continue
            record_id = diagnostic["ID"]
            record = source_by_id[record_id]
            for raw_output in diagnostic.get("attempt_outputs", []):
                items, _ = parse_model_output(raw_output, record.text)
                if items:
                    predictions[record_id] = items
                    repaired += 1
                    break
    write_extraction_predictions(
        output_records,
        predictions,
        args.output_task3,
        args.output_task2,
    )
    print(json.dumps({"repaired_fallback_rows": repaired}, indent=2))


if __name__ == "__main__":
    main()
