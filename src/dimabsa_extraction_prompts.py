"""Prompt construction for joint Task 2/3 extraction with Qwen Instruct."""

from __future__ import annotations

import json
from collections.abc import Sequence

from dimabsa_extraction import ExtractionRecord


SYSTEM_PROMPT = """你是中文餐厅评论的维度方面级情感分析专家。
从文本中抽取显式的方面词、对应评价词、餐厅类别和 Valence-Arousal 分数。
方面词和评价词必须原样复制文本中的连续片段；最终只输出指定 JSON。"""

RUBRIC = """抽取规则：
- 每个 item 表示一条 (Aspect, Category, Opinion, V, A) 关系。
- aspect 是被评价对象，opinion 是直接描述它的评价片段；二者必须逐字出现在原文中，不得改写或补全。
- 同一 aspect 有多个 opinion 时分别输出；同一 opinion 同时评价多个 aspect 时也分别输出。
- 不要抽取没有明确评价关系的普通事实，不要输出重复 item。
- V：1 极负面，5 中性，9 极正面。A：1 平静，5 中等，9 极强烈；A 不是置信度。
- V/A 均在 1.00 到 9.00 之间并保留两位小数。

类别必须使用 `ENTITY#ATTRIBUTE`。ENTITY 只能是 RESTAURANT、FOOD、DRINKS、AMBIENCE、SERVICE、LOCATION；
ATTRIBUTE 只能是 GENERAL、PRICES、QUALITY、STYLE_OPTIONS、MISCELLANEOUS。常见含义如下：
FOOD#QUALITY=食物味道、口感、品质；FOOD#STYLE_OPTIONS=菜式、份量、外观或选择；FOOD#PRICES=具体食物价格；
DRINKS#QUALITY=饮料品质；DRINKS#STYLE_OPTIONS=饮料种类或份量；DRINKS#PRICES=饮料价格；
SERVICE#GENERAL=服务、店员、上菜或外送；AMBIENCE#GENERAL=环境、气氛、卫生；LOCATION#GENERAL=位置或交通；
RESTAURANT#GENERAL=餐厅总体；RESTAURANT#PRICES=整体消费或价格；RESTAURANT#MISCELLANEOUS=其他餐厅相关评价。"""

COT_LINE = (
    "Let's think step by step. 请在内部先定位方面和评价片段，再检查关系、类别和 VA，"
    "但不要输出思考过程。"
)


def _gold_json(record: ExtractionRecord) -> str:
    if record.gold_items is None:
        raise ValueError("few-shot examples require gold items")
    items = [
        {
            "aspect": item.aspect,
            "opinion": item.opinion,
            "category": item.category,
            "V": f"{item.score[0]:.2f}",
            "A": f"{item.score[1]:.2f}",
        }
        for item in record.gold_items
    ]
    return json.dumps({"items": items}, ensure_ascii=False, separators=(",", ":"))


def build_extraction_user_prompt(
    record: ExtractionRecord,
    *,
    prompt_mode: str,
    examples: Sequence[ExtractionRecord] = (),
) -> str:
    if prompt_mode not in {"direct", "cot", "fewshot"}:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
    parts = [RUBRIC]
    if prompt_mode in {"cot", "fewshot"}:
        parts.append(COT_LINE)
    if prompt_mode == "fewshot":
        if not examples:
            raise ValueError("fewshot mode requires examples")
        blocks = []
        for index, example in enumerate(examples, start=1):
            blocks.append(
                f"示例 {index}\n文本：{example.text}\n正确输出：{_gold_json(example)}"
            )
        parts.append("同领域标注示例：\n\n" + "\n\n".join(blocks))
    parts.extend(
        [
            f"现在处理新文本：\n{record.text}",
            "最终只能输出一个合法 JSON 对象，格式为："
            '{"items":[{"aspect":"原文片段","opinion":"原文片段",'
            '"category":"FOOD#QUALITY","V":"6.25","A":"5.50"}]}。'
            "没有可抽取关系时输出 {\"items\":[]}。不要输出 Markdown 或解释。",
        ]
    )
    return "\n\n".join(parts)


def build_extraction_messages(
    record: ExtractionRecord,
    *,
    prompt_mode: str,
    examples: Sequence[ExtractionRecord] = (),
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_extraction_user_prompt(
                record, prompt_mode=prompt_mode, examples=examples
            ),
        },
    ]
