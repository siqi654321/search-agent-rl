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

import random
import re
import string
from typing import Iterable


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


def _normalize_for_long_answer(text: str) -> str:
    """Normalization for long answers (less aggressive than EM normalization)."""
    text = _safe_str(text)
    text = text.strip()
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def _simple_tokenize(text: str, max_tokens: int = 2048) -> list[str]:
    """A lightweight tokenizer that works for both EN and CJK.

    - If there are spaces: split on spaces after basic cleanup.
    - Otherwise: fall back to per-character tokens (helps CJK).
    """
    text = _normalize_for_long_answer(text)
    if not text:
        return []

    if " " in text:
        toks = [t for t in text.split(" ") if t]
    else:
        toks = [c for c in text if not c.isspace()]
    return toks[:max_tokens]


def _lcs_len(a: list[str], b: list[str]) -> int:
    """Compute LCS length with O(min(n,m)) memory."""
    if not a or not b:
        return 0
    # Ensure b is the shorter one for memory
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(cur[-1], prev[j]))
        prev = cur
    return prev[-1]


def _rouge_l_f1(pred: str, ref: str, max_tokens: int = 2048) -> float:
    """ROUGE-L F1 on token sequences."""
    p = _simple_tokenize(pred, max_tokens=max_tokens)
    r = _simple_tokenize(ref, max_tokens=max_tokens)
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    prec = lcs / max(len(p), 1)
    rec = lcs / max(len(r), 1)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def _token_f1(pred: str, ref: str, max_tokens: int = 4096) -> float:
    """Token-level F1 on multisets (bag-of-tokens)."""
    p = _simple_tokenize(pred, max_tokens=max_tokens)
    r = _simple_tokenize(ref, max_tokens=max_tokens)
    if not p or not r:
        return 0.0
    from collections import Counter

    pc = Counter(p)
    rc = Counter(r)
    common = pc & rc
    tp = sum(common.values())
    if tp == 0:
        return 0.0
    prec = tp / max(len(p), 1)
    rec = tp / max(len(r), 1)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def _ngram_repetition_ratio(tokens: list[str], n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    uniq = len(set(ngrams))
    return 1.0 - (uniq / len(ngrams))


def _apply_anti_gibberish_penalty(score: float, pred: str) -> float:
    """Penalize obvious degeneration for long answers (repetition / extremely long outputs)."""
    pred = _normalize_for_long_answer(pred)
    toks = _simple_tokenize(pred, max_tokens=8192)
    if not toks:
        return 0.0

    rep4 = _ngram_repetition_ratio(toks, n=4)
    # Allow some repetition; penalize when it becomes severe.
    rep_penalty = max(0.0, rep4 - 0.2)  # starts after 20%
    score = score * (1.0 - min(0.7, rep_penalty))

    # Overlong penalty (soft)
    if len(toks) > 3000:
        score = score * 0.85
    if len(toks) > 6000:
        score = score * 0.7

    return float(max(0.0, min(1.0, score)))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """Search-R1-like scoring function.

    By default it uses EM on <answer>...</answer>.
    If ground_truth is a dict with `metric` specified, it supports long-answer reward:
    - metric=rouge_l: ROUGE-L F1 between extracted answer and reference.
    - metric=token_f1: bag-of-tokens F1.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # Normalize ground_truth
    if isinstance(ground_truth, dict):
        targets = ground_truth.get("target")
        metric = ground_truth.get("metric")
    else:
        targets = ground_truth
        metric = None

    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Metric: {metric}")
        print(f"Golden answers: {targets}")
        print(f"Solution str: {repr(solution_str)}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")

    if answer is None:
        return 0.0

    # prevent output a lot of </answer>
    if open_count > 10 or close_count > 10:
        score = score / 4

    # Long-answer metrics
    if metric in {"rouge_l", "rouge-l", "rouge_l_f1", "rouge"}:
        # support multi-reference: take max
        refs: Iterable[str]
        if isinstance(targets, str):
            refs = [targets]
        elif isinstance(targets, list):
            refs = [str(x) for x in targets]
        else:
            refs = [str(targets)]

        best = 0.0
        for ref in refs:
            best = max(best, _rouge_l_f1(answer, ref))
        best = _apply_anti_gibberish_penalty(best, answer)
        return float(best * score)

    if metric in {"token_f1", "f1", "bag_f1"}:
        if isinstance(targets, str):
            refs = [targets]
        elif isinstance(targets, list):
            refs = [str(x) for x in targets]
        else:
            refs = [str(targets)]
        best = 0.0
        for ref in refs:
            best = max(best, _token_f1(answer, ref))
        best = _apply_anti_gibberish_penalty(best, answer)
        return float(best * score)

    # Default EM metric
    if isinstance(targets, dict) and "target" in targets:
        targets = targets["target"]

    if em_check(answer, targets):
        return float(score)
    return float(format_score)


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score

