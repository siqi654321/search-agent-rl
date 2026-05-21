#!/usr/bin/env python3
"""Convert verl QA data to search+visit tool prompts.

The script preserves all existing columns and rewrites the prompt column to:
  [
    {"role": "system", "content": search+visit system prompt + date},
    {"role": "user", "content": question}
  ]

Supported input/output: parquet, json, jsonl.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
from typing import Any

import pandas as pd


SYSTEM_PROMPT = """* You are Deep AI Research Assistant

The question I give you is a complex question that requires a *deep research* to answer.

I will provide you with two tools to help you answer the question:
* A web search tool to help you perform google search. 
* A webpage browsing tool to help you get new page content.

You don't have to answer the question now, but you should first think about the research plan or what to search next.

Your output format should be one of the following two formats:

<think>
YOUR THINKING PROCESS
</think>
<answer>
YOUR ANSWER AFTER GETTING ENOUGH INFORMATION
</answer>

or

<think>
YOUR THINKING PROCESS
</think>
<tool_call>
YOUR TOOL CALL WITH CORRECT FORMAT
</tool_call>

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform local corpus searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit local webpage(s)/document(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s), doc:// document id(s), or raw document id(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

You should always follow the above two formats strictly.

Final answer rule for evaluation:
* The text inside <answer></answer> must be ONLY the final short answer: an entity, name, date, number, yes/no, or short phrase.
* Do NOT include explanations, evidence, reasoning, citations, markdown, bullet points, or full sentences inside <answer></answer>.
* Do NOT use markdown such as **bold** inside <answer></answer>.
* If this is a yes-or-no question, answer only yes or no.
* Good: <answer>Rona Weerasuriya</answer>
* Bad: <answer>The full name of the other co-author is **Rona Weerasuriya** because ...</answer>

Current date: """


def _read(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".json"):
        return pd.read_json(path)
    raise ValueError(f"Unsupported input extension: {path}")


def _write(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".jsonl"):
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    elif path.endswith(".json"):
        df.to_json(path, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported output extension: {path}")


def _maybe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") or s.startswith("{"):
            try:
                return json.loads(s)
            except Exception:
                return value
    return value


def _is_present(value: Any) -> bool:
    """pd.notna is ambiguous for list/dict/ndarray, so keep this small helper."""
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return True


def _content_to_text(content: Any) -> str:
    content = _maybe_json_loads(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(content or "").strip()


def _strip_prompt_scaffold(text: str) -> str:
    """Best-effort extraction from existing Search-R1-style prompt text."""
    text = str(text or "").strip()
    if not text:
        return ""

    # Common templates contain an explicit User:/Question: marker. Use the last
    # marker because system prompts may mention these words in instructions.
    patterns = [
        r"(?:^|\n)User\s*:\s*(.+)\s*$",
        r"(?:^|\n)Question\s*:\s*(.+)\s*$",
        r"(?:^|\n)Query\s*:\s*(.+)\s*$",
        r"(?:^|\n)问题\s*[:：]\s*(.+)\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            candidate = matches[-1].strip()
            if candidate:
                return candidate

    # If it looks like a single standalone user message, keep it. Avoid using a
    # long system/tool instruction as the question.
    lower = text.lower()
    instruction_markers = ("you are a helpful ai assistant", "<tools>", "tool_call", "your output format")
    if not any(marker in lower for marker in instruction_markers):
        return text
    return ""


def _extract_question_from_extra_info(extra_info: Any) -> str:
    extra_info = _maybe_json_loads(extra_info)
    if not isinstance(extra_info, dict):
        return ""
    for key in (
        "question",
        "query",
        "user_query",
        "problem",
        "input",
        "prompt",
        "original_question",
        "raw_question",
    ):
        if key in extra_info and _is_present(extra_info[key]):
            q = _strip_prompt_scaffold(_content_to_text(extra_info[key]))
            if q:
                return q
    return ""


def _extract_question_from_messages(messages: Any) -> str:
    messages = _maybe_json_loads(messages)
    if not isinstance(messages, list):
        # parquet may store python objects as numpy arrays; make list if possible
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        elif isinstance(messages, tuple):
            messages = list(messages)

    if isinstance(messages, list):
        # Prefer an explicit user message.
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            q = _strip_prompt_scaffold(_content_to_text(msg.get("content", "")))
            if q:
                return q

        # Some existing data has a single prompt message without role=user, with
        # the actual query embedded after User:/Question:. Try all messages.
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            q = _strip_prompt_scaffold(_content_to_text(msg.get("content", "")))
            if q:
                return q
    return ""


def _extract_question(row: pd.Series, question_key: str, prompt_key: str) -> str:
    for key in (question_key, "question", "query", "user_query"):
        if key in row and _is_present(row[key]):
            q = _strip_prompt_scaffold(_content_to_text(row[key]))
            if q:
                return q

    if "extra_info" in row:
        q = _extract_question_from_extra_info(row["extra_info"])
        if q:
            return q

    if prompt_key in row:
        q = _extract_question_from_messages(row[prompt_key])
        if q:
            return q
    if "messages" in row:
        q = _extract_question_from_messages(row["messages"])
        if q:
            return q

    available = ", ".join(map(str, row.index.tolist()))
    raise ValueError(
        "Could not extract question from row; tried question/query/user_query, "
        "extra_info.question/query, prompt, and messages. Available columns: " + available
    )


def _extract_answer(row: pd.Series, answer_key: str) -> Any:
    if answer_key in row and pd.notna(row[answer_key]):
        return row[answer_key]
    if "answer" in row and pd.notna(row["answer"]):
        return row["answer"]
    if "reward_model" in row and isinstance(row["reward_model"], dict):
        return row["reward_model"].get("ground_truth", "")
    return ""


def _ensure_reward_model(row: pd.Series, answer: Any) -> Any:
    reward_model = row.get("reward_model", None)
    if isinstance(reward_model, str):
        reward_model = _maybe_json_loads(reward_model)
    if not isinstance(reward_model, dict):
        reward_model = {}
    reward_model.setdefault("style", "rule")
    reward_model["ground_truth"] = answer
    return reward_model


def _safe_query_id(row: pd.Series, index: int) -> str:
    for key in ("query_id", "question_id", "id"):
        if key in row and pd.notna(row[key]):
            value = str(row[key]).strip()
            if value:
                return value
    extra_info = row.get("extra_info", None)
    if isinstance(extra_info, str):
        extra_info = _maybe_json_loads(extra_info)
    if isinstance(extra_info, dict):
        for key in ("query_id", "question_id", "id", "index"):
            if key in extra_info:
                value = str(extra_info[key]).strip()
                if value:
                    return value
    return str(index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--data-source", default="searchR1_openresearcher")
    parser.add_argument("--date", default=_dt.date.today().isoformat())
    args = parser.parse_args()

    df = _read(args.input)
    rows = []
    for idx, row in df.iterrows():
        row = row.copy()
        question = _extract_question(row, args.question_key, args.prompt_key)
        answer = _extract_answer(row, args.answer_key)
        row[args.prompt_key] = [
            {"role": "system", "content": SYSTEM_PROMPT + args.date},
            {"role": "user", "content": question},
        ]
        if "data_source" not in row or pd.isna(row.get("data_source")):
            row["data_source"] = args.data_source
        row["reward_model"] = _ensure_reward_model(row, answer)
        extra_info = row.get("extra_info", None)
        if isinstance(extra_info, str):
            extra_info = _maybe_json_loads(extra_info)
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info.setdefault("index", idx)
        extra_info.setdefault("query_id", _safe_query_id(row, idx))
        extra_info.setdefault("question", question)
        row["extra_info"] = extra_info
        rows.append(row)

    out_df = pd.DataFrame(rows)
    _write(out_df, args.output)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
