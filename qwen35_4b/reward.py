# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py


from __future__ import annotations

import os
import re
import string

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _safe_str(x) -> str:
    if x is None:
        return ""
    return str(x)

def _normalized_token_f1(pred: str, ref: str) -> float:
    """Token F1 after lowercasing, punctuation and article normalization."""
    normalized_pred = normalize_answer(_safe_str(pred))
    normalized_ref = normalize_answer(_safe_str(ref))
    if not normalized_pred or not normalized_ref:
        return 0.0

    character_mode = " " not in normalized_pred and " " not in normalized_ref
    if character_mode:
        pred_tokens = list(normalized_pred)
        ref_tokens = list(normalized_ref)
    else:
        pred_tokens = normalized_pred.split()
        ref_tokens = normalized_ref.split()

    from collections import Counter

    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return float(2 * precision * recall / (precision + recall))

_ROLE_MARKER_RE = re.compile(r"(?:^|(?<=\n)|(?<=>))(assistant|user|tool|system)\n")

_RUNTIME_USER_PROMPT_RE = re.compile(
    r"user\n(?=(?:The search phase is closed by the runtime\.|"
    r"Your previous finalization response was invalid|"
    r"Your previous response was rejected|"
    r"This is the final answer-rescue attempt\.))"
)

def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

def _as_refs(targets) -> list[str]:
    if isinstance(targets, str):
        return [targets]
    if isinstance(targets, list):
        return [str(value) for value in targets]
    return [str(targets)]

def _assistant_projection(solution_str: str) -> str:
    """Return assistant-role text from decoded multi-turn rollout text.

    ``batch_decode(..., skip_special_tokens=True)`` removes ChatML boundary
    tokens but leaves role names such as ``user\n`` and ``assistant\n``.  The
    first response segment is an assistant turn and later tool responses or
    runtime-finalization prompts are user turns.  Counting answer tags on the
    full flattened string therefore misclassifies a valid answer whenever a
    runtime user prompt contains the literal example ``<answer></answer>``.

    Plain single-turn strings have no role markers and remain unchanged, so
    the helper is backward compatible with reward smoke tests and callers
    outside the agent loop.
    """
    text = _safe_str(solution_str)
    marker_matches = list(_ROLE_MARKER_RE.finditer(text))
    runtime_user_matches = list(_RUNTIME_USER_PROMPT_RE.finditer(text))
    markers: list[tuple[int, int, str]] = [
        (match.start(1), match.end(), match.group(1).lower())
        for match in marker_matches
    ]
    markers.extend(
        (match.start(), match.end(), "user") for match in runtime_user_matches
    )
    markers.sort(key=lambda item: (item[0], item[1]))

    # Remove duplicate/overlapping detections of the same flattened marker.
    deduped: list[tuple[int, int, str]] = []
    for marker in markers:
        if deduped and marker[0] < deduped[-1][1]:
            continue
        deduped.append(marker)
    if not deduped:
        return text

    role = "assistant"
    cursor = 0
    assistant_parts: list[str] = []
    for start, end, next_role in deduped:
        if role == "assistant" and start > cursor:
            assistant_parts.append(text[cursor:start])
        role = next_role
        cursor = end
    if role == "assistant" and cursor < len(text):
        assistant_parts.append(text[cursor:])
    return "\n".join(assistant_parts)

def _tool_name(call_text: str) -> str:
    native = re.search(r"<function\s*=\s*([A-Za-z0-9_.-]+)\s*>", call_text)
    if native:
        return native.group(1).strip().lower()
    hermes = re.search(
        r'["\']name["\']\s*:\s*["\']([^"\']+)["\']',
        call_text,
    )
    return hermes.group(1).strip().lower() if hermes else ""

def _normalized_ref_in_text(text: str, refs: list[str]) -> bool:
    haystack = normalize_answer(_safe_str(text))
    if not haystack:
        return False
    padded_haystack = f" {haystack} "
    for ref in refs:
        needle = normalize_answer(ref)
        if needle and f" {needle} " in padded_haystack:
            return True
    return False

def _target_in_open_or_find(solution_str: str, refs: list[str]) -> bool:
    pair_pattern = re.compile(
        r"<tool_call>(.*?)</tool_call>.*?<tool_response>(.*?)</tool_response>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    for call_text, response_text in pair_pattern.findall(_safe_str(solution_str)):
        if _tool_name(call_text) in {"open", "find", "visit", "browse"} and _normalized_ref_in_text(
            response_text, refs
        ):
            return True
    return False

def strict_terminal_reward_details(solution_str: str, targets) -> dict[str, object]:
    """Return auditable components of the strict terminal reward."""
    text = _safe_str(solution_str)
    assistant_text = _assistant_projection(text)
    answers = re.findall(
        r"<answer>(.*?)</answer>",
        assistant_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    open_count = len(re.findall(r"<answer>", assistant_text, flags=re.IGNORECASE))
    close_count = len(re.findall(r"</answer>", assistant_text, flags=re.IGNORECASE))
    tool_call_bodies = re.findall(
        r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL | re.IGNORECASE
    )
    tool_call_open_count = len(re.findall(r"<tool_call>", text, flags=re.IGNORECASE))
    tool_call_close_count = len(re.findall(r"</tool_call>", text, flags=re.IGNORECASE))
    tool_response_open_count = len(re.findall(r"<tool_response>", text, flags=re.IGNORECASE))
    tool_response_close_count = len(re.findall(r"</tool_response>", text, flags=re.IGNORECASE))
    pseudo_answer_tool = any(_tool_name(body) == "answer" for body in tool_call_bodies)
    details: dict[str, object] = {
        "assistant_answer_open_count": open_count,
        "assistant_answer_close_count": close_count,
        "assistant_answer_pair_count": len(answers),
        "tool_call_count": tool_call_open_count,
        "tool_call_close_count": tool_call_close_count,
        "tool_response_count": tool_response_open_count,
        "tool_response_close_count": tool_response_close_count,
        "pseudo_answer_tool": pseudo_answer_tool,
        "trailing_assistant_text": False,
        "protocol_valid": False,
        "answer": "",
        "base_f1": 0.0,
        "target_in_open_find": False,
        "evidence_coefficient": 1.0,
        "reward": 0.0,
    }
    if open_count != 1 or close_count != 1 or len(answers) != 1:
        return details

    answer = answers[0].strip()
    details["answer"] = answer
    if not normalize_answer(answer):
        return details

    answer_match = re.search(
        r"<answer>.*?</answer>", assistant_text, flags=re.DOTALL | re.IGNORECASE
    )
    trailing_assistant_text = bool(
        answer_match and assistant_text[answer_match.end() :].strip()
    )
    details["trailing_assistant_text"] = trailing_assistant_text
    if trailing_assistant_text or pseudo_answer_tool:
        return details

    if len(
        {
            tool_call_open_count,
            tool_call_close_count,
            tool_response_open_count,
            tool_response_close_count,
        }
    ) != 1:
        return details

    refs = _as_refs(targets)
    base_f1 = max((_normalized_token_f1(answer, ref) for ref in refs), default=0.0)
    target_in_open_find = _target_in_open_or_find(text, refs) if base_f1 > 0 else False
    coefficient = 1.0
    if base_f1 > 0 and _truthy_env("SEARCH_STRICT_EVIDENCE_COEF_ENABLED"):
        if not target_in_open_find:
            coefficient = float(os.getenv("SEARCH_STRICT_SEARCH_ONLY_COEF", "0.9"))
            coefficient = max(0.0, min(1.0, coefficient))
    reward = float(base_f1 * coefficient)
    details.update(
        {
            "protocol_valid": True,
            "base_f1": float(base_f1),
            "target_in_open_find": target_in_open_find,
            "evidence_coefficient": coefficient,
            "reward": reward,
        }
    )
    return details


def compute_score(data_source=None, solution_str="", ground_truth=None, extra_info=None, **kwargs):
    """VERL custom reward: fixed fail-closed terminal F1, no shaping/dispatch."""
    targets = ground_truth.get("target") if isinstance(ground_truth, dict) else ground_truth
    return float(strict_terminal_reward_details(solution_str, targets)["base_f1"])
