"""Offline tests that do not import or download the Qwen model."""

from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dimabsa_data import (  # noqa: E402
    load_task1_records,
    mean_gold_score,
    select_anchor_examples,
)
from calibrate_task1 import calibrate_score, fit_affine  # noqa: E402
from dimabsa_prompts import build_user_prompt  # noqa: E402
from run_instruct import parse_model_output  # noqa: E402


DATA_ROOT = (
    PROJECT_ROOT
    / "resources"
    / "DimABSA2026"
    / "task-dataset"
    / "track_a"
    / "subtask_1"
    / "zho"
)


class OfflinePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = load_task1_records(
            DATA_ROOT / "zho_restaurant_train_alltasks.jsonl", require_gold=True
        )
        cls.dev = load_task1_records(
            DATA_ROOT / "zho_restaurant_dev_task1.jsonl", require_gold=True
        )

    def test_official_schemas_are_normalized(self) -> None:
        self.assertEqual(len(self.train), 6050)
        self.assertEqual(len(self.dev), 300)
        self.assertTrue(self.train[0].aspects)
        self.assertEqual(len(self.train[0].aspects), len(self.train[0].gold_scores or ()))

    def test_mean_and_anchor_examples(self) -> None:
        mean_v, mean_a = mean_gold_score(self.train)
        self.assertTrue(1.0 <= mean_v <= 9.0)
        self.assertTrue(1.0 <= mean_a <= 9.0)
        examples = select_anchor_examples(self.train, 5)
        self.assertEqual(len(examples), 5)
        self.assertEqual(len({example.record_id for example in examples}), 5)

    def test_cot_prompt_is_explicit(self) -> None:
        direct = build_user_prompt(self.dev[0], prompt_mode="direct")
        cot = build_user_prompt(self.dev[0], prompt_mode="cot")
        self.assertNotIn("Let's think step by step", direct)
        self.assertIn("Let's think step by step", cot)
        gold_v, gold_a = self.dev[0].gold_scores[0]
        self.assertNotIn(f"{gold_v:.2f}#{gold_a:.2f}", direct)

    def test_output_schema_matches_aspect_count(self) -> None:
        record = next(item for item in self.dev if len(item.aspects) == 3)
        prompt = build_user_prompt(record, prompt_mode="direct")
        self.assertIn('"index":3', prompt)
        self.assertNotIn("V3", prompt)
        self.assertIn("V1、A1 等是格式占位符", prompt)
        self.assertIn("scores 必须恰好包含 3 组实际数值", prompt)

    def test_model_output_parser(self) -> None:
        parsed = parse_model_output(
            '分析完成。\n{"scores":[["7.25","6.50"],["3.00","7.75"]]}',
            2,
        )
        self.assertEqual(parsed, ((7.25, 6.5), (3.0, 7.75)))
        with self.assertRaises(ValueError):
            parse_model_output('{"scores":[["10.0","5.0"]]}', 1)

    def test_affine_calibration_and_clipping(self) -> None:
        slope, intercept = fit_affine([1.0, 2.0, 3.0], [3.0, 5.0, 7.0])
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 1.0)
        parameters = {
            "V": {"slope": 2.0, "intercept": 1.0},
            "A": {"slope": 0.5, "intercept": 1.0},
        }
        self.assertEqual(calibrate_score((5.0, 2.0), parameters), (9.0, 2.0))

    def test_prediction_file_does_not_require_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "ID": "example-1",
                        "Aspect_VA": [{"Aspect": "服务", "VA": "7.00#6.00"}],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_task1_records(
                path, require_gold=True, require_text=False
            )
            self.assertEqual(records[0].text, "")


if __name__ == "__main__":
    unittest.main()
