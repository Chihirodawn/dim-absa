"""Prompt variants for zero-shot and few-shot DimABSA inference."""

from __future__ import annotations

import json
from collections.abc import Sequence

from dimabsa_data import Task1Record


SYSTEM_PROMPT = """你是维度方面级情感分析（DimABSA）专家。
你的任务是针对给定的每个方面词分别预测 Valence 和 Arousal，而不是判断整句话的总体情感。
必须严格遵守用户指定的数值范围、方面顺序和输出格式。"""

RUBRIC = """评分标准：
- Valence（V，正负程度）：1=极度负面，3=明显负面，5=中性，7=明显正面，9=极度正面。
- Arousal（A，激烈程度）：1=非常平静，3=较弱，5=中等，7=强烈，9=极度强烈。
- Arousal 不是置信度；平静的正面评价可以是高 V、低 A，激烈的负面评价可以是低 V、高 A。
- 注意否定词、程度副词、转折、感叹号，以及评价词究竟对应哪个方面。
- V 和 A 都必须在 1.00 到 9.00 之间，并保留两位小数。"""

COT_LINE = (
    "Let's think step by step. 请在内部逐个分析方面对应的评价、正负方向和激烈程度，"
    "但不要输出分析过程。"
)


def _format_aspects(record: Task1Record) -> str:
    indexed = [
        {"index": index, "aspect": aspect}
        for index, aspect in enumerate(record.aspects, start=1)
    ]
    return json.dumps(indexed, ensure_ascii=False, separators=(",", ":"))


def _output_rule(record: Task1Record) -> str:
    count = len(record.aspects)
    return f"""最终只能输出一个合法 JSON 对象，不要输出 Markdown、解释或额外文字：
{{"scores":[["V1","A1"],["V2","A2"]]}}
上面只演示二维 JSON 结构，不代表固定有两项。V1、A1 等是格式占位符，实际输出必须替换成 1.00 到 9.00 之间的小数。
本样本共有 {count} 项，scores 必须恰好包含 {count} 组实际数值并严格对应 index 1 到 {count}。
即使方面词相同或相似，也必须按 index 分别输出，不能合并、去重或遗漏。"""


def _format_gold(record: Task1Record) -> str:
    if record.gold_scores is None:
        raise ValueError("Few-shot examples require gold scores")
    scores = [[f"{v:.2f}", f"{a:.2f}"] for v, a in record.gold_scores]
    return json.dumps({"scores": scores}, ensure_ascii=False, separators=(",", ":"))


def build_user_prompt(
    record: Task1Record,
    *,
    prompt_mode: str,
    examples: Sequence[Task1Record] = (),
) -> str:
    """Build direct, CoT, or few-shot-CoT prompts with a shared rubric."""

    if prompt_mode not in {"direct", "cot", "fewshot"}:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
    parts = [RUBRIC]
    if prompt_mode in {"cot", "fewshot"}:
        parts.append(COT_LINE)
    if prompt_mode == "fewshot":
        if not examples:
            raise ValueError("fewshot mode requires at least one example")
        example_blocks = []
        for index, example in enumerate(examples, start=1):
            example_blocks.append(
                "\n".join(
                    [
                        f"示例 {index}：",
                        f"文本：{example.text}",
                        f"方面词（共 {len(example.aspects)} 项，顺序固定）："
                        + _format_aspects(example),
                        "正确输出：" + _format_gold(example),
                    ]
                )
            )
        parts.append("下面是同领域标注示例：\n\n" + "\n\n".join(example_blocks))
    parts.extend(
        [
            "现在预测新样本：",
            f"文本：{record.text}",
            f"方面词（共 {len(record.aspects)} 项，顺序固定）："
            + _format_aspects(record),
            _output_rule(record),
        ]
    )
    return "\n\n".join(parts)


def build_messages(
    record: Task1Record,
    *,
    prompt_mode: str,
    examples: Sequence[Task1Record] = (),
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                record,
                prompt_mode=prompt_mode,
                examples=examples,
            ),
        },
    ]
