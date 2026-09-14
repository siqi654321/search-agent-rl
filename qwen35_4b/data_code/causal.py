#!/usr/bin/env python3
"""Build fail-closed causal protocol SFT data from framework traces.

The previous search-preserving dataset balanced *rows* and trained every token
in the selected assistant completion.  That is not enough to internalize a
runtime intervention: a long piece of ordinary reasoning can dominate a short
protocol correction, and a duplicate-loop trace may condition the correction
on the duplicate action that should have been avoided.

This builder keeps the successful search policy while changing only the
supervision contract:

* ordinary successful tool turns train only the exact ``<tool_call>`` action
  and the assistant turn-end token;
* natural final answers and runtime corrections train only the exact
  ``<answer>`` action and the assistant turn-end token;
* duplicate-loop corrections are re-parented to the context immediately
  before the rejected duplicate action, using an exact substring rewind;
* all other pre-generation corrections must not contain their rejected output
  in the supplied context;
* balancing and capacity gates use *actual tokenizer loss tokens*, not rows.

The output is consumed by ``PromptCompletionSFTDataset`` with
``loss_spans_key=loss_spans_json``.  No BrowseComp labels are used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


TOOL_REASON = "successful_tool_call"
ANSWER_REASON = "successful_final_answer"
TURN_END = "<|im_end|>"

CORRECTION_REASONS = {
    "answer_pseudo_tool",
    "missing_answer",
    "generic_nonanswer",
    "duplicate_loop",
    "turn_limit",
    "response_reserve",
}
PRE_ERROR_CONTEXT_REASONS = {
    "answer_pseudo_tool",
    "missing_answer",
    "generic_nonanswer",
}
ANSWER_RESCUE_REASONS = {"missing_answer", "generic_nonanswer"}

TOOL_ACTION_RE = re.compile(r"<tool_call\b[^>]*>.*?</tool_call>", re.IGNORECASE | re.DOTALL)
ANSWER_ACTION_RE = re.compile(r"<answer\b[^>]*>.*?</answer>", re.IGNORECASE | re.DOTALL)


class CandidateRejected(ValueError):
    """A candidate is unsafe or structurally unsuitable for causal SFT."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="v2/v3 candidate parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer path")
    parser.add_argument("--max-length", type=int, default=127000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--tool-loss-ratio", type=float, default=0.575)
    parser.add_argument("--answer-loss-ratio", type=float, default=0.225)
    parser.add_argument("--correction-loss-ratio", type=float, default=0.20)
    parser.add_argument("--loss-share-tolerance", type=float, default=0.03)
    parser.add_argument("--max-tool-actions-per-question", type=int, default=4)
    parser.add_argument("--min-deep-tool-fraction", type=float, default=0.70)

    # These defaults are deliberately lower than the v3 raw row gates because
    # v4 removes correlated repeats and balances by loss-token capacity.
    parser.add_argument("--min-tool-questions", type=int, default=500)
    parser.add_argument("--min-answer-questions", type=int, default=800)
    parser.add_argument("--min-correction-questions", type=int, default=250)
    parser.add_argument("--min-pseudo-answer-corrections", type=int, default=75)
    parser.add_argument("--min-answer-rescue-corrections", type=int, default=75)
    parser.add_argument("--min-duplicate-loop-corrections", type=int, default=30)
    parser.add_argument("--min-duplicate-rewind-rate", type=float, default=0.70)
    return parser.parse_args()


def _string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _string(value).strip().lower() in {"1", "true", "yes", "on"}


def _stable_hash(*parts: Any) -> str:
    material = "\0".join(_string(part) for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _question_key(row: pd.Series | dict[str, Any]) -> str:
    query_id = _string(row.get("query_id")).strip()
    if query_id:
        return f"query:{query_id}"
    question = _string(row.get("question")).strip()
    if question:
        return "question:" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:24]
    return "source:" + _stable_hash(
        row.get("source_path"), row.get("source_line"), row.get("sample_id")
    )[:24]


def _exact_action_spans(completion: str, *, action_kind: str) -> list[tuple[int, int]]:
    """Return exact action and turn-end spans, rejecting ambiguous completions."""

    pattern = TOOL_ACTION_RE if action_kind == "tool" else ANSWER_ACTION_RE
    matches = list(pattern.finditer(completion))
    if len(matches) != 1:
        raise CandidateRejected(f"{action_kind}_action_count_{len(matches)}")
    action = matches[0]
    turn_end_start = completion.find(TURN_END, action.end())
    if turn_end_start < 0:
        raise CandidateRejected("missing_assistant_turn_end")
    if completion[action.end() : turn_end_start].strip():
        raise CandidateRejected("trailing_assistant_text_before_turn_end")
    turn_end_end = turn_end_start + len(TURN_END)
    if completion[turn_end_end:].strip():
        raise CandidateRejected("trailing_text_after_turn_end")
    return [(action.start(), action.end()), (turn_end_start, turn_end_end)]


def _rewind_duplicate_context(context: str, rejected: str) -> tuple[str, str]:
    """Cut a duplicate-loop event back to the exact rejected assistant action.

    The runtime records the duplicate model action in ``failed_assistant_output``
    and normally records the correction context after the blocked tool response.
    Rewinding by exact text is safe; guessing a turn boundary is not.
    """

    candidates: list[str] = []
    if rejected:
        candidates.append(rejected)
        stripped = rejected.strip()
        if stripped and stripped != rejected:
            candidates.append(stripped)
    for candidate in candidates:
        index = context.rfind(candidate)
        if index < 0:
            continue
        suffix = context[index + len(candidate) :]
        # This guard prevents an accidental match against an older, legitimate
        # occurrence of the same text.
        lowered = suffix.lower()
        if not any(
            marker in lowered
            for marker in (
                "duplicate_call_blocked",
                "duplicate call",
                "<tool_response",
                "<|im_start|>tool",
            )
        ):
            continue
        rewound = context[:index]
        if not rewound.strip():
            raise CandidateRejected("duplicate_rewind_empty_context")
        return rewound, "exact_failed_output"
    raise CandidateRejected("duplicate_rewind_exact_match_missing")


def _correction_rule(reason: str) -> str:
    if reason == "answer_pseudo_tool":
        return "answer_pseudo_tool_redirect"
    if reason in ANSWER_RESCUE_REASONS:
        return "generic_answer_rescue"
    if reason == "duplicate_loop":
        return "duplicate_loop_break"
    if reason in {"turn_limit", "response_reserve"}:
        return "late_final_answer"
    return reason


def _materialize_candidate(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    reason = _string(row.get("intervention_reason")).strip().lower()
    context = _string(row.get("context_text"))
    chosen = _string(row.get("chosen_text"))
    rejected = _string(row.get("rejected_text"))
    if not context.strip() or not chosen.strip():
        raise CandidateRejected("empty_context_or_completion")

    duplicate_rewound = False
    rewind_method = "not_applicable"
    if reason == TOOL_REASON:
        category = "tool"
        causal_rule = "first_tool_call_boundary"
        spans = _exact_action_spans(chosen, action_kind="tool")
    elif reason == ANSWER_REASON:
        category = "answer"
        causal_rule = "final_answer_boundary"
        spans = _exact_action_spans(chosen, action_kind="answer")
    elif reason in CORRECTION_REASONS:
        category = "correction"
        causal_rule = _correction_rule(reason)
        if reason == "duplicate_loop":
            context, rewind_method = _rewind_duplicate_context(context, rejected)
            duplicate_rewound = True
        elif reason in PRE_ERROR_CONTEXT_REASONS and rejected.strip():
            if context.rfind(rejected) >= 0 or context.rfind(rejected.strip()) >= 0:
                raise CandidateRejected("pre_error_context_contains_rejected_output")
        spans = _exact_action_spans(chosen, action_kind="answer")
    else:
        raise CandidateRejected("unsupported_intervention_reason")

    materialized = dict(row)
    materialized.update(
        {
            "context_text": context,
            "chosen_text": chosen,
            "rejected_text": rejected,
            "intervention_reason": reason,
            "causal_category": category,
            "causal_rule": causal_rule,
            "question_key": _question_key(row),
            "loss_spans_json": json.dumps(spans, separators=(",", ":")),
            "duplicate_context_rewound": duplicate_rewound,
            "context_rewind_method": rewind_method,
            "approx_total_chars": len(context) + len(chosen),
        }
    )
    materialized["causal_sample_id"] = _stable_hash(
        materialized.get("sample_id"), category, causal_rule, context, chosen
    )[:24]
    return materialized


def _parse_spans(raw: Any) -> list[tuple[int, int]]:
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, (list, tuple)):
        raise CandidateRejected("invalid_loss_span_json")
    spans: list[tuple[int, int]] = []
    previous_end = 0
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise CandidateRejected("invalid_loss_span")
        start, end = int(item[0]), int(item[1])
        if start < previous_end or end <= start:
            raise CandidateRejected("invalid_loss_span_order")
        spans.append((start, end))
        previous_end = end
    if not spans:
        raise CandidateRejected("empty_loss_spans")
    return spans


def _tokenize_candidate(
    sample: dict[str, Any], tokenizer: Any, *, max_length: int
) -> dict[str, Any]:
    context = _string(sample["context_text"])
    completion = _string(sample["chosen_text"])
    context_ids = list(tokenizer.encode(context, add_special_tokens=False))
    try:
        encoded = tokenizer(
            completion,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError) as exc:
        raise ValueError(
            "causal protocol SFT requires a fast tokenizer with offset mappings"
        ) from exc
    completion_ids = list(encoded["input_ids"])
    offsets = list(encoded.get("offset_mapping") or [])
    if len(completion_ids) != len(offsets):
        raise CandidateRejected("token_offset_count_mismatch")
    spans = _parse_spans(sample["loss_spans_json"])
    loss_mask: list[int] = []
    for raw_offset in offsets:
        if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 2:
            raise CandidateRejected("invalid_token_offset")
        token_start, token_end = int(raw_offset[0]), int(raw_offset[1])
        loss_mask.append(
            int(
                token_end > token_start
                and any(
                    token_start < span_end and token_end > span_start
                    for span_start, span_end in spans
                )
            )
        )
    loss_tokens = int(sum(loss_mask))
    if not completion_ids or loss_tokens <= 0:
        raise CandidateRejected("empty_completion_or_loss")
    total_tokens = len(context_ids) + len(completion_ids)
    if total_tokens > max_length:
        raise CandidateRejected("overlength")
    sample.update(
        {
            "context_tokens": len(context_ids),
            "completion_tokens": len(completion_ids),
            "total_tokens": total_tokens,
            "loss_tokens": loss_tokens,
            "masked_completion_tokens": len(completion_ids) - loss_tokens,
        }
    )
    return sample


def _evenly_spaced(rows: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(rows) <= maximum:
        return rows
    if maximum <= 0:
        return rows.iloc[0:0]
    positions = np.linspace(0, len(rows) - 1, num=maximum)
    indices: list[int] = []
    for position in positions:
        index = int(round(float(position)))
        if index not in indices:
            indices.append(index)
    if len(indices) < maximum:
        indices.extend(index for index in range(len(rows)) if index not in indices)
    return rows.iloc[indices[:maximum]]


def _cap_independent_states(
    frame: pd.DataFrame, *, max_tool_actions_per_question: int
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    tools = frame.loc[frame["causal_category"] == "tool"]
    for _question, group in tools.groupby("question_key", sort=True):
        deep = group.loc[group["trajectory_is_deep_search"].map(_bool)]
        shallow = group.loc[~group.index.isin(deep.index)]
        deep_limit = int(math.ceil(max_tool_actions_per_question * 0.75))
        shallow_limit = max_tool_actions_per_question - deep_limit
        selected_deep = _evenly_spaced(
            deep.sort_values(
                ["assistant_turn", "approx_total_chars", "causal_sample_id"],
                kind="stable",
            ),
            deep_limit,
        )
        selected_shallow = _evenly_spaced(
            shallow.sort_values(
                ["assistant_turn", "approx_total_chars", "causal_sample_id"],
                kind="stable",
            ),
            shallow_limit,
        )
        chosen = pd.concat([selected_deep, selected_shallow])
        remaining = max_tool_actions_per_question - len(chosen)
        if remaining > 0:
            unused = group.loc[~group.index.isin(chosen.index)].sort_values(
                ["assistant_turn", "approx_total_chars", "causal_sample_id"],
                kind="stable",
            )
            chosen = pd.concat([chosen, _evenly_spaced(unused, remaining)])
        selected.append(chosen)

    answers = frame.loc[frame["causal_category"] == "answer"]
    for _question, group in answers.groupby("question_key", sort=True):
        ordered = group.assign(
            _deep_priority=~group["trajectory_is_deep_search"].map(_bool)
        ).sort_values(
            ["_deep_priority", "approx_total_chars", "causal_sample_id"],
            kind="stable",
        )
        selected.append(
            ordered.drop(columns=["_deep_priority"]).head(1)
        )

    corrections = frame.loc[frame["causal_category"] == "correction"]
    for (_question, _rule), group in corrections.groupby(
        ["question_key", "causal_rule"], sort=True
    ):
        selected.append(
            group.sort_values(
                ["approx_total_chars", "causal_sample_id"], kind="stable"
            ).head(1)
        )
    if not selected:
        return frame.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True)


def _shuffled_indices(frame: pd.DataFrame, *, seed: int) -> list[int]:
    indices = frame.index.to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return [int(index) for index in indices]


def _tool_indices(frame: pd.DataFrame, *, seed: int) -> list[int]:
    """Return a deterministic ~3:1 deep/shallow interleaving."""

    deep = frame.loc[frame["trajectory_is_deep_search"].map(_bool)]
    shallow = frame.loc[~frame.index.isin(deep.index)]
    deep_indices = _shuffled_indices(deep, seed=seed + 11)
    shallow_indices = _shuffled_indices(shallow, seed=seed + 29)
    output: list[int] = []
    deep_cursor = shallow_cursor = 0
    while deep_cursor < len(deep_indices) or shallow_cursor < len(shallow_indices):
        for _ in range(3):
            if deep_cursor < len(deep_indices):
                output.append(deep_indices[deep_cursor])
                deep_cursor += 1
        if shallow_cursor < len(shallow_indices):
            output.append(shallow_indices[shallow_cursor])
            shallow_cursor += 1
        elif deep_cursor >= len(deep_indices):
            break
    return output


def _correction_indices(frame: pd.DataFrame, *, seed: int) -> list[int]:
    """Round-robin independent correction rules before exhausting large pools."""

    reason_order = [
        "answer_pseudo_tool",
        "missing_answer",
        "generic_nonanswer",
        "duplicate_loop",
        "turn_limit",
        "response_reserve",
    ]
    groups: dict[str, list[int]] = {}
    for offset, reason in enumerate(reason_order):
        group = frame.loc[frame["intervention_reason"] == reason]
        if len(group):
            groups[reason] = _shuffled_indices(group, seed=seed + 37 * (offset + 1))
    output: list[int] = []
    cursors = {reason: 0 for reason in groups}
    while any(cursors[reason] < len(groups[reason]) for reason in groups):
        for reason in reason_order:
            if reason not in groups:
                continue
            cursor = cursors[reason]
            if cursor < len(groups[reason]):
                output.append(groups[reason][cursor])
                cursors[reason] += 1
    return output


def _select_to_loss_budget(
    frame: pd.DataFrame,
    *,
    target_loss_tokens: int,
    ordered_indices: Sequence[int],
) -> pd.DataFrame:
    selected: list[int] = []
    accumulated = 0
    for index in ordered_indices:
        if accumulated >= target_loss_tokens:
            break
        selected.append(int(index))
        accumulated += int(frame.at[index, "loss_tokens"])
    return frame.loc[selected].copy()


def _distribution(values: Iterable[int]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.int64)
    if not len(array):
        return {"min": 0, "p50": 0.0, "p95": 0.0, "max": 0}
    return {
        "min": int(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": int(array.max()),
    }


def _load_tokenizer(path: str, *, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("--tokenizer must resolve to a fast tokenizer")
    return tokenizer


def _validate_args(args: argparse.Namespace) -> dict[str, float]:
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.max_tool_actions_per_question <= 0:
        raise ValueError("--max-tool-actions-per-question must be positive")
    ratios = {
        "tool": float(args.tool_loss_ratio),
        "answer": float(args.answer_loss_ratio),
        "correction": float(args.correction_loss_ratio),
    }
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("all loss-token ratios must be positive")
    if abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError(f"loss-token ratios must sum to 1, got {sum(ratios.values())}")
    if not 0 <= args.loss_share_tolerance < 1:
        raise ValueError("--loss-share-tolerance must be in [0, 1)")
    if not 0 <= args.min_deep_tool_fraction <= 1:
        raise ValueError("--min-deep-tool-fraction must be in [0, 1]")
    if not 0 <= args.min_duplicate_rewind_rate <= 1:
        raise ValueError("--min-duplicate-rewind-rate must be in [0, 1]")
    for name in (
        "min_tool_questions",
        "min_answer_questions",
        "min_correction_questions",
        "min_pseudo_answer_corrections",
        "min_answer_rescue_corrections",
        "min_duplicate_loop_corrections",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    return ratios


def build(
    args: argparse.Namespace,
    *,
    tokenizer: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ratios = _validate_args(args)
    tokenizer = tokenizer or _load_tokenizer(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    source = pd.read_parquet(args.input)
    print(
        f"[causal-v4] loaded candidate_rows={len(source)} from {args.input}",
        file=sys.stderr,
        flush=True,
    )
    required_columns = {
        "assistant_turn",
        "context_text",
        "chosen_text",
        "intervention_reason",
        "trajectory_is_deep_search",
    }
    missing_columns = sorted(required_columns - set(source.columns))
    if missing_columns:
        raise ValueError(f"candidate parquet lacks columns: {missing_columns}")

    counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    materialized: list[dict[str, Any]] = []
    for source_index, (_, row) in enumerate(source.iterrows(), start=1):
        counts["candidate_rows"] += 1
        input_reason = _string(row.get("intervention_reason")).strip().lower()
        counts[f"input_reason__{input_reason or 'missing'}"] += 1
        try:
            sample = _materialize_candidate(row)
        except CandidateRejected as exc:
            rejection_counts[str(exc)] += 1
        else:
            materialized.append(sample)
            counts[f"materialized_{sample['causal_category']}"] += 1
            if sample["duplicate_context_rewound"]:
                counts["duplicate_exact_rewinds"] += 1
        if source_index % 5000 == 0 or source_index == len(source):
            print(
                f"[causal-v4] materialized={len(materialized)}/{source_index}",
                file=sys.stderr,
                flush=True,
            )

    if not materialized:
        raise ValueError("no causal candidates survived materialization")
    materialized_frame = pd.DataFrame(materialized)
    capped_pre_token = _cap_independent_states(
        materialized_frame,
        max_tool_actions_per_question=args.max_tool_actions_per_question,
    )
    if capped_pre_token.empty:
        raise ValueError("no causal candidates survived independent-state caps")
    print(
        f"[causal-v4] independent_state_cap={len(capped_pre_token)}/{len(materialized_frame)}",
        file=sys.stderr,
        flush=True,
    )

    tokenized: list[dict[str, Any]] = []
    for token_index, (_, row) in enumerate(capped_pre_token.iterrows(), start=1):
        try:
            sample = _tokenize_candidate(
                dict(row), tokenizer, max_length=args.max_length
            )
        except CandidateRejected as exc:
            rejection_counts[str(exc)] += 1
        else:
            tokenized.append(sample)
            counts[f"tokenized_{sample['causal_category']}"] += 1
        if token_index % 500 == 0 or token_index == len(capped_pre_token):
            print(
                f"[causal-v4] exact_tokenized={len(tokenized)}/{token_index}",
                file=sys.stderr,
                flush=True,
            )
    if not tokenized:
        raise ValueError("no causal candidates survived exact tokenizer checks")
    capped = pd.DataFrame(tokenized).reset_index(drop=True)

    capacity_loss_tokens = {
        category: int(
            capped.loc[capped["causal_category"] == category, "loss_tokens"].sum()
        )
        for category in ratios
    }
    if any(value <= 0 for value in capacity_loss_tokens.values()):
        raise ValueError(f"empty category loss-token capacity: {capacity_loss_tokens}")
    total_loss_budget = int(
        math.floor(
            min(
                capacity_loss_tokens[category] / ratios[category]
                for category in ratios
            )
        )
    )
    target_loss_tokens = {
        category: max(1, int(math.floor(total_loss_budget * ratio)))
        for category, ratio in ratios.items()
    }

    selected_parts: list[pd.DataFrame] = []
    for offset, category in enumerate(("tool", "answer", "correction")):
        category_frame = capped.loc[capped["causal_category"] == category]
        if category == "tool":
            ordering = _tool_indices(category_frame, seed=args.seed)
        elif category == "correction":
            ordering = _correction_indices(category_frame, seed=args.seed)
        else:
            ordering = _shuffled_indices(category_frame, seed=args.seed + 101 * (offset + 1))
        selected_parts.append(
            _select_to_loss_budget(
                capped,
                target_loss_tokens=target_loss_tokens[category],
                ordered_indices=ordering,
            )
        )
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    output_loss_tokens = {
        category: int(
            selected.loc[
                selected["causal_category"] == category, "loss_tokens"
            ].sum()
        )
        for category in ratios
    }
    output_loss_total = sum(output_loss_tokens.values())
    realized_loss_shares = {
        category: output_loss_tokens[category] / max(output_loss_total, 1)
        for category in ratios
    }
    output_row_counts = Counter(selected["causal_category"].astype(str))
    output_reason_counts = Counter(selected["intervention_reason"].astype(str))
    unique_questions = {
        category: int(
            selected.loc[
                selected["causal_category"] == category, "question_key"
            ].nunique()
        )
        for category in ratios
    }
    tool_rows = selected.loc[selected["causal_category"] == "tool"]
    deep_tool_fraction = (
        float(tool_rows["trajectory_is_deep_search"].map(_bool).mean())
        if len(tool_rows)
        else 0.0
    )
    pseudo_count = int(output_reason_counts.get("answer_pseudo_tool", 0))
    rescue_count = sum(int(output_reason_counts.get(reason, 0)) for reason in ANSWER_RESCUE_REASONS)
    duplicate_count = int(output_reason_counts.get("duplicate_loop", 0))

    failures: list[str] = []
    for category, requested in ratios.items():
        actual = realized_loss_shares[category]
        if abs(actual - requested) > args.loss_share_tolerance:
            failures.append(
                f"{category}_loss_share={actual:.6f} differs from requested "
                f"{requested:.6f} by more than {args.loss_share_tolerance:.6f}"
            )
    for category, minimum in (
        ("tool", args.min_tool_questions),
        ("answer", args.min_answer_questions),
        ("correction", args.min_correction_questions),
    ):
        if unique_questions[category] < minimum:
            failures.append(
                f"{category}_questions={unique_questions[category]} < required {minimum}"
            )
    for name, actual, minimum in (
        ("pseudo_answer", pseudo_count, args.min_pseudo_answer_corrections),
        ("answer_rescue", rescue_count, args.min_answer_rescue_corrections),
        ("duplicate_loop", duplicate_count, args.min_duplicate_loop_corrections),
    ):
        if actual < minimum:
            failures.append(f"{name}_corrections={actual} < required {minimum}")
    if deep_tool_fraction + 1e-12 < args.min_deep_tool_fraction:
        failures.append(
            f"deep_tool_fraction={deep_tool_fraction:.6f} < required "
            f"{args.min_deep_tool_fraction:.6f}"
        )

    correction_pool = capped.loc[capped["causal_category"] == "correction"]
    duplicate_input_count = int(counts.get("input_reason__duplicate_loop", 0))
    duplicate_exact_rewinds = int(counts.get("duplicate_exact_rewinds", 0))
    duplicate_rewind_rate = (
        duplicate_exact_rewinds / duplicate_input_count
        if duplicate_input_count
        else 0.0
    )
    if (
        duplicate_input_count
        and duplicate_rewind_rate + 1e-12 < args.min_duplicate_rewind_rate
    ):
        failures.append(
            f"duplicate_rewind_rate={duplicate_rewind_rate:.6f} < required "
            f"{args.min_duplicate_rewind_rate:.6f}"
        )
    report: dict[str, Any] = {
        "schema": "causal_protocol_sft_v1",
        "input": str(args.input),
        "output": str(args.output),
        "tokenizer": str(args.tokenizer),
        "seed": args.seed,
        "max_length": args.max_length,
        "loss_policy": {
            "context": 0,
            "assistant_reasoning": 0,
            "selected_tool_call": 1,
            "selected_final_answer": 1,
            "assistant_turn_end": 1,
            "tool_response": 0,
            "loss_spans_key": "loss_spans_json",
        },
        "causal_policy": {
            "duplicate_loop": "exact rewind before failed_assistant_output",
            "pre_error_corrections": sorted(PRE_ERROR_CONTEXT_REASONS),
            "pre_error_context_must_exclude_rejected_output": True,
            "unsupported_reasons_fail_closed": True,
        },
        "counts": dict(sorted(counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rows": {
            "input": int(len(source)),
            "materialized": int(len(materialized_frame)),
            "after_independent_state_caps": int(len(capped_pre_token)),
            "after_exact_token_filter": int(len(capped)),
            "output": int(len(selected)),
        },
        "independent_state_caps": {
            "max_tool_actions_per_question": args.max_tool_actions_per_question,
            "answer_actions_per_question": 1,
            "corrections_per_question_per_rule": 1,
            "correction_selection": "round_robin_by_intervention_reason",
        },
        "requested_loss_shares": ratios,
        "loss_token_capacity_after_caps": capacity_loss_tokens,
        "target_loss_tokens": target_loss_tokens,
        "output_loss_tokens": output_loss_tokens,
        "realized_loss_shares": realized_loss_shares,
        "loss_share_tolerance": args.loss_share_tolerance,
        "output_row_counts": dict(sorted(output_row_counts.items())),
        "output_reason_counts": dict(sorted(output_reason_counts.items())),
        "output_unique_question_counts": unique_questions,
        "deep_tool_fraction": deep_tool_fraction,
        "duplicate_rewind": {
            "input_candidates": duplicate_input_count,
            "materialized_exact_rewinds": duplicate_exact_rewinds,
            "materialized_exact_rewind_rate": duplicate_rewind_rate,
            "minimum_exact_rewind_rate": args.min_duplicate_rewind_rate,
            "materialized_failures": rejection_counts.get(
                "duplicate_rewind_exact_match_missing", 0
            ),
            "pool_after_caps": int(
                (correction_pool["intervention_reason"] == "duplicate_loop").sum()
            ),
            "output": duplicate_count,
        },
        "token_distributions": {
            "total_tokens": _distribution(selected["total_tokens"]),
            "completion_tokens": _distribution(selected["completion_tokens"]),
            "loss_tokens": _distribution(selected["loss_tokens"]),
            "masked_completion_tokens": _distribution(selected["masked_completion_tokens"]),
        },
        "protection_rule_training_coverage": {
            "first_tool_call_boundary": int(output_reason_counts.get(TOOL_REASON, 0)),
            "final_answer_boundary": int(output_reason_counts.get(ANSWER_REASON, 0)),
            "answer_pseudo_tool_redirect": pseudo_count,
            "generic_answer_rescue": rescue_count,
            "duplicate_loop_break": duplicate_count,
            "late_final_answer": int(output_reason_counts.get("turn_limit", 0))
            + int(output_reason_counts.get("response_reserve", 0)),
        },
        "final_gate": {"passed": not failures, "failures": failures},
    }
    return selected, report


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output, report = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    report_path = args.report or args.output.with_suffix(".report.json")
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["final_gate"]["passed"]:
        raise SystemExit(
            "causal protocol SFT gate failed: "
            + "; ".join(report["final_gate"]["failures"])
        )


if __name__ == "__main__":
    main()
