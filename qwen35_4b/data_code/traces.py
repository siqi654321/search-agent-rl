#!/usr/bin/env python3
"""Build prompt/completion SFT candidates from opt-in agent-loop corrections.

Rollouts must be produced with ``VERL_TOOL_AGENT_DISTILL_TRACE=1``.  The agent
loop records the exact rendered context immediately before a runtime
intervention, the rejected model output, and the corrected output that
eventually terminated successfully.  This script keeps only successful scored
rows and writes a compact parquet consumable by PromptCompletionSFTDataset.

Naturally successful rollouts may contribute their verified tool and final-answer
turns.  A rollout that only succeeded after runtime finalization/rescue may
contribute the successful correction pair, but not the preceding ordinary tool
turns: those turns often contain the exact search loop that the student should
learn to leave.

Ground-truth answers are used only through the rollout's existing score; they
are never inserted into a prompt or completion by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _iter_jsonl(paths: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if isinstance(row, dict):
                    yield path, line_number, row


def _score(row: dict[str, Any]) -> float:
    for key in ("score", "reward", "acc"):
        try:
            return float(row.get(key))
        except (TypeError, ValueError):
            continue
    return 0.0


def _event_id(row: dict[str, Any], event: dict[str, Any]) -> str:
    material = "\0".join(
        [
            str(row.get("query_id") or row.get("dataset_index") or ""),
            str(event.get("reason") or ""),
            str(event.get("context_text") or ""),
            str(event.get("corrected_assistant_output") or ""),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _runtime_assisted_success(row: dict[str, Any]) -> bool:
    """Return whether framework intervention was needed to finish this row.

    ``finalization_reason`` covers turn/response limits, duplicate-loop
    finalization, pseudo-answer redirection and other calls to
    ``_request_final_answer``.  The counters are retained as fail-closed
    fallbacks for dumps produced by intermediate loop versions.
    """

    if str(row.get("finalization_reason") or "").strip():
        return True
    return any(
        _positive_number(row.get(key))
        for key in (
            "final_answer_attempts",
            "generic_answer_rescue_triggers",
            "answer_pseudo_tool_redirects",
        )
    )


_FUNCTION_TOOL_RE = re.compile(r"<function\s*=\s*([A-Za-z0-9_.:-]+)", re.IGNORECASE)
_JSON_TOOL_RE = re.compile(
    r"[\"']name[\"']\s*:\s*[\"']([A-Za-z0-9_.:-]+)[\"']",
    re.IGNORECASE,
)


def _tool_name(text: Any) -> str:
    """Best-effort tool name extraction for telemetry and stratified reports."""

    value = str(text or "")
    match = _FUNCTION_TOOL_RE.search(value) or _JSON_TOOL_RE.search(value)
    return match.group(1).lower() if match else ""


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _trajectory_metadata(
    row: dict[str, Any],
    turns: list[Any],
    events: list[Any],
    *,
    runtime_assisted: bool,
    deep_search_min_tool_turns: int,
) -> dict[str, Any]:
    valid_turns = [turn for turn in turns if isinstance(turn, dict)]
    tool_turns = [turn for turn in valid_turns if str(turn.get("kind") or "") == "tool_call"]
    answer_turns = [turn for turn in valid_turns if str(turn.get("kind") or "") == "final_answer"]
    assistant_turn_values = [
        _positive_int(item.get("assistant_turn"))
        for item in [*valid_turns, *[event for event in events if isinstance(event, dict)]]
    ]
    assistant_turn_values = [value for value in assistant_turn_values if value > 0]
    search_calls = _positive_int(row.get("search_calls"))
    open_calls = _positive_int(row.get("open_calls"))
    visit_calls = _positive_int(row.get("visit_calls"))
    find_calls = _positive_int(row.get("find_calls"))
    # Prefer executed-call telemetry.  Older rollout dumps may only carry the
    # distillation turn list, so retain a fail-safe count from verified turns.
    executed_tool_turns = search_calls + open_calls + visit_calls + find_calls
    trajectory_tool_turns = max(len(tool_turns), executed_tool_turns)
    trajectory_assistant_turns = max(
        assistant_turn_values,
        default=len(valid_turns),
    )
    return {
        "runtime_assisted": bool(runtime_assisted),
        "natural_success": not runtime_assisted,
        "trajectory_tool_turns": trajectory_tool_turns,
        "trajectory_assistant_turns": trajectory_assistant_turns,
        "trajectory_success_tool_turns": len(tool_turns),
        "trajectory_success_answer_turns": len(answer_turns),
        "trajectory_search_calls": search_calls,
        "trajectory_open_calls": open_calls,
        "trajectory_visit_calls": visit_calls,
        "trajectory_find_calls": find_calls,
        "trajectory_inspection_calls": open_calls + visit_calls + find_calls,
        "trajectory_finalization_reason": str(row.get("finalization_reason") or "natural"),
        "trajectory_is_deep_search": bool(
            not runtime_assisted and trajectory_tool_turns >= deep_search_min_tool_turns
        ),
        "deep_search_min_tool_turns": deep_search_min_tool_turns,
    }


def _turn_metadata(
    assistant_turn: int,
    *,
    trajectory_assistant_turns: int,
) -> dict[str, Any]:
    denominator = max(trajectory_assistant_turns, assistant_turn, 1)
    progress = min(1.0, max(0.0, assistant_turn / denominator))
    if progress <= 1 / 3:
        phase = "early"
    elif progress <= 2 / 3:
        phase = "middle"
    else:
        phase = "late"
    return {
        "turn_position": assistant_turn,
        "turn_progress": progress,
        "turn_phase": phase,
    }


def _append_turn_end(text: str, turn_end_text: str | None) -> str:
    if not turn_end_text or not text.strip() or text.rstrip().endswith(turn_end_text):
        return text
    return text + turn_end_text


_STRATEGY_CORRECTION_REASONS = {
    "duplicate_loop",
    "turn_limit",
    "response_reserve",
    "missing_answer",
    "generic_nonanswer",
}


def _correction_group(reason: Any) -> str:
    value = str(reason or "unknown").strip().lower()
    return "strategy" if value in _STRATEGY_CORRECTION_REASONS else "protocol"


def build_rows(
    paths: Iterable[Path],
    *,
    min_score: float,
    allow_truncated_context: bool,
    include_runtime_assisted_success_turns: bool = False,
    deep_search_min_tool_turns: int = 12,
    final_turn_end_text: str | None = None,
    source_label: str | None = None,
    schema_version: str = "v1",
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if schema_version not in {"v1", "v2"}:
        raise ValueError("schema_version must be 'v1' or 'v2'")
    if schema_version == "v1" and (final_turn_end_text is not None or source_label is not None):
        raise ValueError("final_turn_end_text/source_label require schema_version='v2'")
    emit_v2_metadata = schema_version == "v2"
    output: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for path, line_number, row in _iter_jsonl(paths):
        counts["rollout_rows"] += 1
        score = _score(row)
        if score < min_score:
            counts["low_score_rows"] += 1
            continue
        events = row.get("distill_trace_events")
        turns = row.get("distill_success_turns")
        if not isinstance(events, list):
            events = []
        if not isinstance(turns, list):
            turns = []
        if not events and not turns:
            counts["rows_without_distill_data"] += 1
            continue
        runtime_assisted = _runtime_assisted_success(row)
        trajectory = _trajectory_metadata(
            row,
            turns,
            events,
            runtime_assisted=runtime_assisted,
            deep_search_min_tool_turns=deep_search_min_tool_turns,
        )
        if runtime_assisted:
            counts["runtime_assisted_success_rows"] += 1
        else:
            counts["natural_success_rows"] += 1
            if emit_v2_metadata:
                if trajectory["trajectory_is_deep_search"]:
                    counts["natural_deep_search_success_rows"] += 1
                else:
                    counts["natural_shallow_success_rows"] += 1
        for event_index, event in enumerate(events):
            counts["trace_events"] += 1
            if not isinstance(event, dict) or not event.get("correction_succeeded"):
                counts["unsuccessful_corrections"] += 1
                continue
            if event.get("context_truncated") and not allow_truncated_context:
                counts["truncated_contexts"] += 1
                continue
            context = str(event.get("context_text") or "")
            chosen = str(event.get("corrected_assistant_output") or "")
            rejected = str(event.get("failed_assistant_output") or "")
            if not context.strip() or not chosen.strip():
                counts["empty_context_or_completion"] += 1
                continue
            if emit_v2_metadata:
                chosen = _append_turn_end(chosen, final_turn_end_text)
            event_id = _event_id(row, event)
            if event_id in seen:
                counts["duplicate_events"] += 1
                continue
            seen.add(event_id)
            distilled_row = {
                    "sample_id": event_id,
                    "query_id": str(row.get("query_id") or row.get("dataset_index") or ""),
                    "question": str(row.get("question") or ""),
                    "context_text": context,
                    "chosen_text": chosen,
                    "rejected_text": rejected,
                    "intervention_reason": str(event.get("reason") or "unknown"),
                    "score": score,
                    "assistant_turn": int(event.get("assistant_turn") or 0),
                    "source_path": str(path),
                    "source_line": line_number,
                    "source_event_index": event_index,
                }
            if emit_v2_metadata:
                distilled_row.update(
                    {
                        **_turn_metadata(
                            int(event.get("assistant_turn") or 0),
                            trajectory_assistant_turns=int(
                                trajectory["trajectory_assistant_turns"]
                            ),
                        ),
                        "assistant_output_kind": "correction",
                        "correction_group": _correction_group(event.get("reason")),
                        "tool_name": _tool_name(chosen),
                        "candidate_source": str(source_label or path.stem),
                        **trajectory,
                    }
                )
            output.append(distilled_row)
            counts["selected_events"] += 1
        if runtime_assisted and not include_runtime_assisted_success_turns:
            counts["suppressed_runtime_assisted_success_turns"] += len(turns)
            continue
        for turn_index, turn in enumerate(turns):
            counts["success_turns"] += 1
            if not isinstance(turn, dict):
                counts["invalid_success_turns"] += 1
                continue
            if turn.get("context_truncated") and not allow_truncated_context:
                counts["truncated_contexts"] += 1
                continue
            if turn.get("kind") == "tool_call" and turn.get("tool_execution_succeeded") is not True:
                counts["unverified_or_failed_tool_turns"] += 1
                continue
            context = str(turn.get("context_text") or "")
            chosen = str(turn.get("chosen_assistant_output") or "")
            if not context.strip() or not chosen.strip():
                counts["empty_context_or_completion"] += 1
                continue
            if emit_v2_metadata and turn.get("kind") == "final_answer":
                chosen = _append_turn_end(chosen, final_turn_end_text)
            distilled = {
                "reason": f"successful_{turn.get('kind') or 'assistant_turn'}",
                "context_text": context,
                "corrected_assistant_output": chosen,
            }
            event_id = _event_id(row, distilled)
            if event_id in seen:
                counts["duplicate_events"] += 1
                continue
            seen.add(event_id)
            distilled_row = {
                    "sample_id": event_id,
                    "query_id": str(row.get("query_id") or row.get("dataset_index") or ""),
                    "question": str(row.get("question") or ""),
                    "context_text": context,
                    "chosen_text": chosen,
                    "rejected_text": "",
                    "intervention_reason": str(distilled["reason"]),
                    "score": score,
                    "assistant_turn": int(turn.get("assistant_turn") or 0),
                    "source_path": str(path),
                    "source_line": line_number,
                    "source_event_index": turn_index,
                }
            if emit_v2_metadata:
                distilled_row.update(
                    {
                        **_turn_metadata(
                            int(turn.get("assistant_turn") or 0),
                            trajectory_assistant_turns=int(
                                trajectory["trajectory_assistant_turns"]
                            ),
                        ),
                        "assistant_output_kind": str(turn.get("kind") or "assistant_turn"),
                        "correction_group": "",
                        "tool_name": (
                            _tool_name(chosen) if turn.get("kind") == "tool_call" else ""
                        ),
                        "candidate_source": str(source_label or path.stem),
                        **trajectory,
                    }
                )
            output.append(distilled_row)
            counts["selected_success_turns"] += 1
    return output, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--schema-version",
        choices=("v1", "v2"),
        default="v1",
        help="v1 exactly preserves the previous candidate format; v2 enables round-two metadata.",
    )
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--allow-truncated-context", action="store_true")
    parser.add_argument(
        "--include-runtime-assisted-success-turns",
        action="store_true",
        help=(
            "Also keep ordinary tool/answer turns from rollouts that needed "
            "runtime finalization or answer rescue. Disabled by default because "
            "such turns often reinforce late search loops."
        ),
    )
    parser.add_argument(
        "--source-label",
        help="Stable source label such as base4b_anchor or sft368_new.",
    )
    parser.add_argument(
        "--final-turn-end-text",
        help=(
            "Append this exact tokenizer special token/text to final answers and corrections "
            "when absent. For Qwen3.5 second-round distillation use '<|im_end|>'."
        ),
    )
    parser.add_argument(
        "--deep-search-min-tool-turns",
        type=int,
        default=12,
        help="Mark natural successes with at least this many executed tool turns as deep search.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, counts = build_rows(
        args.input,
        min_score=args.min_score,
        allow_truncated_context=args.allow_truncated_context,
        include_runtime_assisted_success_turns=args.include_runtime_assisted_success_turns,
        deep_search_min_tool_turns=args.deep_search_min_tool_turns,
        final_turn_end_text=args.final_turn_end_text,
        source_label=args.source_label,
        schema_version=args.schema_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    reason_counts = Counter(row["intervention_reason"] for row in rows)
    report = {
        "schema": (
            "framework_distillation_candidates_v2"
            if args.schema_version == "v1"
            else "framework_distillation_candidates_v3"
        ),
        "inputs": [str(path) for path in args.input],
        "output": str(args.output),
        "min_score": args.min_score,
        "allow_truncated_context": args.allow_truncated_context,
        "include_runtime_assisted_success_turns": args.include_runtime_assisted_success_turns,
        "counts": dict(sorted(counts.items())),
        "selected_reason_counts": dict(sorted(reason_counts.items())),
    }
    if args.schema_version == "v2":
        report.update(
            {
                "schema_version": "v2",
                "deep_search_min_tool_turns": args.deep_search_min_tool_turns,
                "final_turn_end_text": args.final_turn_end_text,
                "source_label": args.source_label,
            }
        )
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
