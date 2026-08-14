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
    select_similar_examples,
)
from calibrate_task1 import (  # noqa: E402
    calibrate_score,
    fit_affine,
    fit_ridge,
    select_ridge,
)
from ensemble_task1 import average_predictions  # noqa: E402
from dimabsa_prompts import build_user_prompt  # noqa: E402
from run_instruct import parse_model_output  # noqa: E402
from run_extraction import parse_model_output as parse_extraction_output  # noqa: E402
from dimabsa_extraction import (  # noqa: E402
    ExtractionItem,
    ExtractionRecord,
    load_extraction_records,
    parse_extraction_payload,
    select_extraction_examples,
)
from extraction_hybrid import (  # noqa: E402
    BM25Retriever,
    recover_exact_span,
    relation_label,
    vote_prediction_files,
    write_relation_dataset,
)
from dimabsa_extraction_prompts import build_extraction_user_prompt  # noqa: E402
from calibrate_extraction import DEFAULT_UNCERTAIN_VA  # noqa: E402
from calibrate_extraction_affine import (  # noqa: E402
    apply_parameters as apply_extraction_affine,
    fit_parameters as fit_extraction_affine,
)
from encoder_experiment_utils import (  # noqa: E402
    balanced_weight_values,
    load_train_examples,
    opinion_spans,
    split_oof,
    stable_fold,
)


DATA_ROOT = (
    PROJECT_ROOT
    / "resources"
    / "DimABSA2026"
    / "task-dataset"
    / "track_a"
    / "subtask_1"
    / "zho"
)
EXTRACTION_ROOT = DATA_ROOT.parents[1] / "subtask_3" / "zho"


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

    def test_dynamic_examples_are_query_specific_and_answer_free(self) -> None:
        query = self.dev[0]
        examples = select_similar_examples(query, self.train, 5)
        self.assertEqual(len(examples), 5)
        self.assertEqual(len({example.record_id for example in examples}), 5)
        prompt = build_user_prompt(
            query, prompt_mode="dynamic_fewshot", examples=examples
        )
        self.assertIn("retrieved for their similarity", prompt)
        self.assertIn("Think step by step internally", prompt)
        gold_v, gold_a = query.gold_scores[0]
        self.assertNotIn(f"{gold_v:.2f}#{gold_a:.2f}", prompt.split("Now", 1)[-1])

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

    def test_grouped_ridge_calibration(self) -> None:
        predicted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        gold = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0]
        groups = ["a", "a", "b", "b", "c", "c"]
        slope, intercept = fit_ridge(predicted, gold, alpha=0.0)
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 1.0)
        slope, intercept, alpha, scores = select_ridge(
            predicted, gold, groups, [0.0, 0.1, 1.0], folds=3
        )
        self.assertEqual(alpha, 0.0)
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 1.0)
        self.assertEqual(set(scores), {"0.0", "0.1", "1.0"})

    def test_equal_weight_ensemble(self) -> None:
        left = [
            self.dev[0].__class__(
                self.dev[0].record_id,
                "",
                self.dev[0].aspects,
                tuple((3.0, 5.0) for _ in self.dev[0].aspects),
            )
        ]
        right = [
            self.dev[0].__class__(
                self.dev[0].record_id,
                "",
                self.dev[0].aspects,
                tuple((7.0, 9.0) for _ in self.dev[0].aspects),
            )
        ]
        _, averaged = average_predictions([left, right])
        self.assertEqual(
            averaged[self.dev[0].record_id],
            tuple((5.0, 7.0) for _ in self.dev[0].aspects),
        )

    def test_encoder_training_examples_and_opinion_spans(self) -> None:
        records, examples = load_train_examples(
            DATA_ROOT / "zho_restaurant_train_alltasks.jsonl"
        )
        self.assertEqual(len(records), len(self.train))
        self.assertEqual(len(examples), sum(len(record.aspects) for record in records))
        explicit = next(example for example in examples if example.opinion != "NULL")
        spans = opinion_spans(explicit.text, explicit.opinion)
        self.assertTrue(spans)
        self.assertEqual(
            explicit.text[spans[0][0] : spans[0][1]].lower(), explicit.opinion.lower()
        )
        self.assertEqual(opinion_spans(explicit.text, "NULL"), [])

    def test_encoder_oof_is_grouped_and_sampling_is_balanced(self) -> None:
        records, examples = load_train_examples(
            DATA_ROOT / "zho_restaurant_train_alltasks.jsonl"
        )
        train_records, train_examples, valid_records, valid_examples = split_oof(
            records, examples, folds=3, fold=0
        )
        train_ids = {record.record_id for record in train_records}
        valid_ids = {record.record_id for record in valid_records}
        self.assertFalse(train_ids & valid_ids)
        self.assertEqual(train_ids, {example.record_id for example in train_examples})
        self.assertEqual(valid_ids, {example.record_id for example in valid_examples})
        self.assertTrue(all(stable_fold(record_id, 3) == 0 for record_id in valid_ids))
        weights = balanced_weight_values(examples)
        self.assertEqual(len(weights), len(examples))
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)
        self.assertGreater(max(weights), min(weights))

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

    def test_extraction_data_prompt_and_parser(self) -> None:
        train = load_extraction_records(
            EXTRACTION_ROOT / "zho_restaurant_train_alltasks.jsonl",
            require_gold=True,
        )
        dev = load_extraction_records(
            EXTRACTION_ROOT / "zho_restaurant_dev_task3.jsonl",
            require_gold=True,
        )
        self.assertEqual(len(train), 6050)
        self.assertEqual(len(dev), 300)
        examples = select_extraction_examples(train, 8)
        self.assertEqual(len(examples), 8)
        prompt = build_extraction_user_prompt(
            dev[0], prompt_mode="fewshot", examples=examples
        )
        self.assertIn("Let's think step by step", prompt)
        self.assertIn("FOOD#QUALITY", prompt)
        self.assertNotIn('"6.25#5.50"', prompt.split("现在处理新文本：", 1)[1])
        items, errors = parse_extraction_payload(
            {
                "items": [
                    {
                        "aspect": "配料",
                        "opinion": "覺得特別",
                        "category": "food#quality",
                        "V": "6.25",
                        "A": "5.50",
                    },
                    {
                        "aspect": "不存在",
                        "opinion": "覺得特別",
                        "category": "FOOD#QUALITY",
                        "V": 6,
                        "A": 5,
                    },
                ]
            },
            dev[0].text,
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].category, "FOOD#QUALITY")
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(DEFAULT_UNCERTAIN_VA), 4)

    def test_truncated_extraction_output_salvages_complete_items(self) -> None:
        text = "食物很好吃，服務也親切。"
        truncated = (
            '{"items":['
            '{"aspect":"食物","opinion":"很好吃","category":"FOOD#QUALITY",'
            '"V":"6.50","A":"6.00"},'
            '{"aspect":"服務","opinion":"親切","category":"SERVICE#GENERAL",'
            '"V":"6.00"'
        )
        items, errors = parse_extraction_output(truncated, text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].aspect, "食物")
        self.assertEqual(errors, [])

    def test_english_null_extraction_is_opt_in(self) -> None:
        payload = {
            "items": [
                {
                    "aspect": "NULL",
                    "opinion": "NULL",
                    "category": "RESTAURANT#GENERAL",
                    "V": "6.75",
                    "A": "6.38",
                }
            ]
        }
        rejected, errors = parse_extraction_payload(payload, "Can't wait to return")
        self.assertEqual(rejected, ())
        self.assertEqual(len(errors), 1)
        accepted, errors = parse_extraction_payload(
            payload, "Can't wait to return", allow_null=True
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(errors, [])

    def test_extraction_affine_keeps_every_prediction(self) -> None:
        gold = [
            {
                "ID": "one",
                "Triplet": [
                    {"Aspect": "food", "Opinion": "great", "VA": "7.00#5.00"}
                ],
            }
        ]
        prediction = [
            {
                "ID": "one",
                "Triplet": [
                    {"Aspect": "food", "Opinion": "great", "VA": "5.00#5.00"},
                    {"Aspect": "staff", "Opinion": "slow", "VA": "3.00#6.00"},
                ],
            }
        ]
        parameters = fit_extraction_affine(gold, prediction, 2)
        calibrated = apply_extraction_affine(prediction, parameters, 2)
        self.assertEqual(parameters["fit_matches"], 1)
        self.assertEqual(len(calibrated[0]["Triplet"]), 2)
        self.assertEqual(calibrated[0]["Triplet"][0]["VA"], "7.00#5.00")

    def test_bm25_retrieval_and_span_recovery(self) -> None:
        records = [
            ExtractionRecord(
                "food",
                "The pizza was delicious.",
                (ExtractionItem("pizza", "delicious", "FOOD#QUALITY", (8.0, 6.0)),),
            ),
            ExtractionRecord(
                "service",
                "The waiter was painfully slow.",
                (ExtractionItem("waiter", "slow", "SERVICE#GENERAL", (2.0, 7.0)),),
            ),
        ]
        for variant in ("word", "bigram", "trigram"):
            selected = BM25Retriever(records, variant).select(
                "Service was slow.", 1
            )
            self.assertEqual(selected[0].record_id, "service")
        self.assertEqual(recover_exact_span("The Service was slow.", "service"), "Service")
        self.assertIsNone(recover_exact_span("The service was slow.", "waiter"))

    def test_relation_conversion_and_structure_vote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.jsonl"
            template.write_text(
                json.dumps(
                    {
                        "ID": "one",
                        "Text": "The Service was painfully slow.",
                        "Quadruplet": [
                            {
                                "Aspect": "Service",
                                "Category": "SERVICE#GENERAL",
                                "Opinion": "slow",
                                "VA": "2.00#7.00",
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            views = []
            for index, opinion in enumerate(("slow", "slow", "painfully slow")):
                path = root / f"view{index}.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "ID": "one",
                            "Quadruplet": [
                                {
                                    "Aspect": "service",
                                    "Category": "SERVICE#GENERAL",
                                    "Opinion": opinion,
                                    "VA": f"{2 + index:.2f}#7.00",
                                }
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                views.append(path)
            task2_template = root / "task2_template.jsonl"
            task2_template.write_text(
                json.dumps(
                    {
                        "ID": "task2-one",
                        "Text": "The Service was painfully slow.",
                        "Triplet": [
                            {
                                "Aspect": "Service",
                                "Opinion": "slow",
                                "VA": "2.00#7.00",
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            task2, task3 = root / "task2.jsonl", root / "task3.jsonl"
            result = vote_prediction_files(
                template, views, 2, task3, task2, task2_template
            )
            self.assertEqual(result["retained_relations"], 1)
            output = json.loads(task3.read_text(encoding="utf-8"))
            self.assertEqual(output["Quadruplet"][0]["Aspect"], "Service")
            self.assertEqual(output["Quadruplet"][0]["VA"], "2.50#7.00")
            task2_output = json.loads(task2.read_text(encoding="utf-8"))
            self.assertEqual(task2_output["ID"], "task2-one")
            relation_path = root / "relations.jsonl"
            self.assertEqual(
                write_relation_dataset(template, relation_path, "task1"), 1
            )
            relation = json.loads(relation_path.read_text(encoding="utf-8"))
            self.assertEqual(
                relation["Aspect_VA"][0]["Aspect"],
                relation_label("Service", "slow"),
            )


if __name__ == "__main__":
    unittest.main()
