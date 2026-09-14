#!/usr/bin/env python3
"""Diagnose whether strict long-chain rollouts match the BC+ 371 behavior.

The analyzer intentionally separates three questions that were previously
conflated:

* Is the n=8 group informative under fail-closed normalized terminal F1?
* Is its *absolute* turn/token length close to the hard BC+ subset?
* Does a higher-reward continuation finish materially earlier after inspecting
  designated evidence?

New rollouts should contain exact ``num_turns`` and ``response_token_count``
telemetry emitted by ``verl/trainer/main_agent_eval.py``.  Legacy dumps remain
readable for reporting, but cannot pass the exact-telemetry scale-up gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "browsecomp_safe_long_chain_rollout_diagnostic_v2"


def _as_python(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _mapping(value: Any) -> dict[str, Any]:
    value = _as_python(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _targets(ground_truth: Any) -> Any:
    mapping = _mapping(ground_truth)
    return mapping.get("target") if mapping else ground_truth


def _docids(value: Any) -> set[str]:
    value = _as_python(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = (value,)
    output = set()
    for item in values:
        docid = str(item or "").strip().casefold()
        if docid.startswith("doc://"):
            docid = docid[6:]
        if docid:
            output.add(docid)
    return output


def _question(row: dict[str, Any]) -> str:
    for key in ("question", "query", "user_query"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _group_id(row: dict[str, Any]) -> str:
    for key in ("query_id", "dataset_index"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{str(value).strip()}"
    question = _question(row)
    if not question:
        return ""
    return "question_sha256:" + hashlib.sha256(question.casefold().encode()).hexdigest()


def _load_reward(path: Path):
    module_name = "long_chain_strict_reward"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reward script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "strict_terminal_reward_details"):
        raise RuntimeError(f"reward lacks strict_terminal_reward_details(): {path}")
    return module


def _read_jsonl(paths: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                yield path, line_number, row


def _number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _tool_count(row: dict[str, Any], names: Sequence[str]) -> int:
    total = 0
    for name in names:
        try:
            total += int(row.get(name, 0) or 0)
        except (TypeError, ValueError):
            pass
    return total


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "mean": 0.0, "max": 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "min": min(ordered),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "mean": statistics.mean(ordered),
        "max": max(ordered),
    }


def _tier(items: list[dict[str, Any]], value: float) -> list[dict[str, Any]]:
    return [item for item in items if math.isclose(item["base_f1"], value, abs_tol=1e-12)]


def _mean(items: Sequence[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in items]
    return statistics.mean(values) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-jsonl", type=Path, required=True)
    parser.add_argument("--selected-jsonl", type=Path)
    parser.add_argument("--reward-script", type=Path, default=root / "reward.py")
    parser.add_argument("--expected-rollouts-per-question", type=int, default=8)
    parser.add_argument(
        "--selection-mode",
        choices=("success_profile",),
        default="success_profile",
        help="Diagnostic success profile; batches.py selects the final length-evidence core",
    )
    parser.add_argument(
        "--best-evidence-mode",
        choices=("answer_doc_and_target",),
        default="answer_doc_and_target",
        help="How the best valid reward tier must demonstrate evidence use",
    )
    parser.add_argument("--min-f1-range", type=float, default=1e-6)
    parser.add_argument("--min-group-mean-turns", type=float, default=65.0)
    parser.add_argument("--max-group-mean-turns", type=float, default=85.0)
    parser.add_argument("--min-group-mean-response-tokens", type=float, default=35000.0)
    parser.add_argument("--max-group-mean-response-tokens", type=float, default=65000.0)
    parser.add_argument("--min-failure-success-turn-gap", type=float, default=10.0)
    parser.add_argument("--min-failure-success-token-reduction", type=float, default=0.20)
    parser.add_argument("--min-group-mean-search-calls", type=float, default=0.0)
    parser.add_argument("--max-group-inspection-search-ratio", type=float, default=1e9)
    parser.add_argument("--min-best-tier-mean-turns", type=float, default=60.0)
    parser.add_argument("--max-best-tier-mean-turns", type=float, default=105.0)
    parser.add_argument("--min-best-tier-mean-response-tokens", type=float, default=40000.0)
    parser.add_argument("--max-best-tier-mean-response-tokens", type=float, default=70000.0)
    parser.add_argument("--min-best-tier-mean-search-calls", type=float, default=20.0)
    parser.add_argument("--max-best-tier-inspection-search-ratio", type=float, default=0.45)
    parser.add_argument("--max-absolute-failure-success-turn-gap", type=float, default=20.0)
    parser.add_argument("--max-absolute-failure-success-token-gap", type=float, default=15000.0)
    parser.add_argument("--long-assistant-threshold", type=int, default=8192)
    parser.add_argument("--min-protocol-valid-rate", type=float, default=0.75)
    parser.add_argument("--min-informative-rate", type=float, default=0.15)
    parser.add_argument("--min-behavior-match-rate", type=float, default=0.08)
    parser.add_argument("--target-selected-groups", type=int, default=1000)
    parser.add_argument("--require-exact-telemetry", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_rollouts_per_question <= 0:
        raise ValueError("--expected-rollouts-per-question must be positive")
    if args.min_best_tier_mean_turns > args.max_best_tier_mean_turns:
        raise ValueError("best-tier turn range is inverted")
    if args.min_best_tier_mean_response_tokens > args.max_best_tier_mean_response_tokens:
        raise ValueError("best-tier response-token range is inverted")
    reward = _load_reward(args.reward_script)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    questions: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for path, line_number, row in _read_jsonl(args.rollouts):
        group_id = _group_id(row)
        question = _question(row)
        if not group_id or not question:
            raise ValueError(f"{path}:{line_number}: missing stable group id/question")
        details = reward.strict_terminal_reward_details(
            str(row.get("output") or ""), _targets(row.get("gts"))
        )
        exact_turns = _number(row, "num_turns")
        if exact_turns is None:
            exact_turns = float(
                2
                * _tool_count(
                    row,
                    ("search_calls", "open_calls", "find_calls", "visit_calls", "browse_calls"),
                )
                + 1
            )
            turn_source = "tool_call_proxy"
            counts["rows_missing_exact_num_turns"] += 1
        else:
            turn_source = "exact"
        response_tokens = _number(row, "response_token_count")
        if response_tokens is None:
            # Retain useful legacy diagnostics without pretending that decoded
            # whitespace words are model tokens.  Such rows cannot pass the
            # exact telemetry gate.
            response_tokens = float(len(str(row.get("output") or "").split()))
            token_source = "serialized_word_proxy"
            counts["rows_missing_exact_response_tokens"] += 1
        else:
            token_source = "exact"
        ground_truth = _mapping(row.get("gts"))
        evidence_docids = _docids(ground_truth.get("evidence_docids"))
        answer_docids = _docids(ground_truth.get("answer_docid"))
        inspected_docids = set()
        for key in ("opened_docids", "find_docids", "visit_docids", "visited_docids", "browse_docids"):
            inspected_docids.update(_docids(row.get(key)))
        search_calls = _tool_count(row, ("search_calls",))
        inspection_calls = _tool_count(row, ("open_calls", "find_calls", "visit_calls", "browse_calls"))
        max_assistant = int(_number(row, "max_assistant_generation_tokens") or 0)
        item = {
            "base_f1": float(details["base_f1"]),
            "protocol_valid": bool(details["protocol_valid"]),
            "target_in_open_find": bool(details["target_in_open_find"]),
            "num_turns": exact_turns,
            "turn_source": turn_source,
            "response_tokens": response_tokens,
            "token_source": token_source,
            "assistant_generated_tokens": _number(row, "assistant_generated_token_count"),
            "max_assistant_generation_tokens": max_assistant,
            "search_calls": search_calls,
            "inspection_calls": inspection_calls,
            "opened_evidence_docids": len(evidence_docids & inspected_docids),
            "opened_answer_docids": len(answer_docids & inspected_docids),
            "has_designated_answer_docid": bool(answer_docids),
            "designated_evidence_docids": len(evidence_docids),
            "finalization_reason": str(row.get("finalization_reason") or "natural"),
            "source": str(path),
            "line": line_number,
        }
        grouped[group_id].append(item)
        questions.setdefault(group_id, question)
        counts["rows"] += 1
        counts["protocol_valid_rows"] += int(item["protocol_valid"])

    size_counts: Counter[int] = Counter()
    details_rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for group_id, items in grouped.items():
        size_counts[len(items)] += 1
        scores = [item["base_f1"] for item in items]
        min_score = min(scores)
        max_score = max(scores)
        valid_items = [item for item in items if item["protocol_valid"]]
        valid_scores = [item["base_f1"] for item in valid_items]
        valid_range = max(valid_scores) - min(valid_scores) if len(valid_scores) >= 2 else 0.0
        informative = valid_range >= args.min_f1_range
        best = _tier(valid_items, max(valid_scores)) if valid_items else []
        worst = _tier(valid_items, min(valid_scores)) if valid_items else []
        group_mean_turns = _mean(items, "num_turns")
        group_mean_tokens = _mean(items, "response_tokens")
        group_mean_search = _mean(items, "search_calls")
        group_mean_inspection = _mean(items, "inspection_calls")
        inspection_search_ratio = (
            group_mean_inspection / group_mean_search if group_mean_search else 1e9
        )
        best_turns = _mean(best, "num_turns")
        worst_turns = _mean(worst, "num_turns")
        best_tokens = _mean(best, "response_tokens")
        worst_tokens = _mean(worst, "response_tokens")
        best_search = _mean(best, "search_calls")
        best_inspection = _mean(best, "inspection_calls")
        best_inspection_search_ratio = (
            best_inspection / best_search if best_search else 1e9
        )
        turn_gap = worst_turns - best_turns
        token_gap = worst_tokens - best_tokens
        token_reduction = (worst_tokens - best_tokens) / worst_tokens if worst_tokens else 0.0
        exact_telemetry = all(
            item["turn_source"] == "exact" and item["token_source"] == "exact"
            for item in items
        )
        turn_matched = args.min_group_mean_turns <= group_mean_turns <= args.max_group_mean_turns
        token_matched = (
            args.min_group_mean_response_tokens
            <= group_mean_tokens
            <= args.max_group_mean_response_tokens
        )
        efficient_contrast = (
            turn_gap >= args.min_failure_success_turn_gap
            and token_reduction >= args.min_failure_success_token_reduction
        )
        search_behavior_matched = (
            group_mean_search >= args.min_group_mean_search_calls
            and inspection_search_ratio <= args.max_group_inspection_search_ratio
        )
        best_target_in_open_find = any(item["target_in_open_find"] for item in best)
        best_answer_doc_opened = any(item["opened_answer_docids"] > 0 for item in best)
        best_answer_doc_target_supported = any(
            item["target_in_open_find"]
            and (
                item["opened_answer_docids"] > 0
                if item["has_designated_answer_docid"]
                else True
            )
            for item in best
        )
        best_evidence_supported = best_answer_doc_target_supported
        best_length_matched = (
            args.min_best_tier_mean_turns
            <= best_turns
            <= args.max_best_tier_mean_turns
            and args.min_best_tier_mean_response_tokens
            <= best_tokens
            <= args.max_best_tier_mean_response_tokens
        )
        best_search_behavior_matched = (
            best_search >= args.min_best_tier_mean_search_calls
            and best_inspection_search_ratio
            <= args.max_best_tier_inspection_search_ratio
        )
        success_failure_gap_matched = (
            informative
            and abs(turn_gap) <= args.max_absolute_failure_success_turn_gap
            and abs(token_gap) <= args.max_absolute_failure_success_token_gap
        )
        worst_budget_failure = any(
            item["finalization_reason"] in {"turn_limit", "response_reserve"} for item in worst
        )
        worst_long_assistant = any(
            item["max_assistant_generation_tokens"] >= args.long_assistant_threshold
            for item in worst
        )
        worst_low_conversion = any(
            item["search_calls"] >= 5
            and item["inspection_calls"] / max(item["search_calls"], 1) <= 0.20
            for item in worst
        )
        failure_signature = worst_budget_failure or worst_long_assistant or worst_low_conversion
        core_match = (
            informative
            and best_evidence_supported
            and best_length_matched
            and best_search_behavior_matched
            and success_failure_gap_matched
            and (exact_telemetry or not args.require_exact_telemetry)
        )
        selected_match = core_match
        strong_match = core_match and failure_signature
        if max_score - min_score >= args.min_f1_range:
            category_counts["raw_f1_varying"] += 1
        if informative:
            category_counts["valid_f1_varying"] += 1
        if turn_matched:
            category_counts["turn_matched"] += 1
        if token_matched:
            category_counts["token_matched"] += 1
        if turn_matched and token_matched:
            category_counts["turn_and_token_matched"] += 1
        if efficient_contrast:
            category_counts["efficient_contrast"] += 1
        if search_behavior_matched:
            category_counts["search_behavior_matched"] += 1
        if best_target_in_open_find:
            category_counts["best_target_in_open_find"] += 1
        if best_answer_doc_opened:
            category_counts["best_answer_doc_opened"] += 1
        if best_answer_doc_target_supported:
            category_counts["best_answer_doc_target_supported"] += 1
        if best_length_matched:
            category_counts["best_length_matched"] += 1
        if best_search_behavior_matched:
            category_counts["best_search_behavior_matched"] += 1
        if success_failure_gap_matched:
            category_counts["success_failure_gap_matched"] += 1
        if core_match:
            category_counts["core_behavior_matched"] += 1
        if strong_match:
            category_counts["strong_behavior_matched"] += 1
        if selected_match:
            category_counts["selected_behavior_matched"] += 1
        details_rows.append(
            {
                "group_id": group_id,
                "question": questions[group_id],
                "rollouts": len(items),
                "min_base_f1": min_score,
                "max_base_f1": max_score,
                "valid_base_f1_range": valid_range,
                "protocol_valid_rollouts": len(valid_items),
                "group_mean_turns": group_mean_turns,
                "group_mean_response_tokens": group_mean_tokens,
                "group_mean_search_calls": group_mean_search,
                "group_mean_inspection_calls": group_mean_inspection,
                "group_inspection_search_ratio": inspection_search_ratio,
                "best_tier_mean_turns": best_turns,
                "worst_tier_mean_turns": worst_turns,
                "failure_minus_success_turns": turn_gap,
                "best_tier_mean_response_tokens": best_tokens,
                "worst_tier_mean_response_tokens": worst_tokens,
                "best_tier_mean_search_calls": best_search,
                "best_tier_mean_inspection_calls": best_inspection,
                "best_tier_inspection_search_ratio": best_inspection_search_ratio,
                "failure_to_success_token_reduction": token_reduction,
                "failure_minus_success_response_tokens": token_gap,
                "best_target_in_open_find": best_target_in_open_find,
                "best_answer_doc_opened": best_answer_doc_opened,
                "best_answer_doc_target_supported": best_answer_doc_target_supported,
                "best_evidence_supported": best_evidence_supported,
                "best_length_matched": best_length_matched,
                "best_search_behavior_matched": best_search_behavior_matched,
                "success_failure_gap_matched": success_failure_gap_matched,
                "worst_budget_failure": worst_budget_failure,
                "worst_long_assistant": worst_long_assistant,
                "worst_low_search_to_open_conversion": worst_low_conversion,
                "failure_signature": failure_signature,
                "exact_telemetry": exact_telemetry,
                "informative": informative,
                "turn_matched": turn_matched,
                "token_matched": token_matched,
                "efficient_contrast": efficient_contrast,
                "search_behavior_matched": search_behavior_matched,
                "core_behavior_matched": core_match,
                "strong_behavior_matched": strong_match,
                "selected_behavior_matched": selected_match,
            }
        )

    args.details_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.details_jsonl.open("w", encoding="utf-8") as handle:
        for row in details_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    selected = [row for row in details_rows if row["selected_behavior_matched"]]
    selected.sort(
        key=lambda row: (
            -row["group_mean_turns"],
            -row["group_mean_response_tokens"],
            row["group_id"],
        )
    )
    if args.selected_jsonl:
        args.selected_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.selected_jsonl.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    questions_count = len(grouped)
    protocol_valid_rate = counts["protocol_valid_rows"] / counts["rows"] if counts["rows"] else 0.0
    informative_rate = category_counts["valid_f1_varying"] / questions_count if questions_count else 0.0
    behavior_rate = category_counts["selected_behavior_matched"] / questions_count if questions_count else 0.0
    exact_telemetry_groups = sum(row["exact_telemetry"] for row in details_rows)
    failures: list[str] = []
    if any(size != args.expected_rollouts_per_question for size in size_counts):
        failures.append(
            f"unexpected_group_sizes={dict(sorted(size_counts.items()))}; "
            f"expected={args.expected_rollouts_per_question}"
        )
    if args.require_exact_telemetry and exact_telemetry_groups != questions_count:
        failures.append(
            f"exact_telemetry_groups={exact_telemetry_groups} != questions={questions_count}"
        )
    if protocol_valid_rate < args.min_protocol_valid_rate:
        failures.append(
            f"protocol_valid_rate={protocol_valid_rate:.4f} < {args.min_protocol_valid_rate:.4f}"
        )
    if informative_rate < args.min_informative_rate:
        failures.append(
            f"informative_rate={informative_rate:.4f} < {args.min_informative_rate:.4f}"
        )
    if behavior_rate < args.min_behavior_match_rate:
        failures.append(
            f"selected_behavior_match_rate={behavior_rate:.4f} < {args.min_behavior_match_rate:.4f}"
        )
    estimated_raw = (
        math.ceil(args.target_selected_groups / behavior_rate) if behavior_rate else None
    )
    report = {
        "schema": SCHEMA,
        "inputs": {
            "rollouts": [str(path) for path in args.rollouts],
            "reward_script": str(args.reward_script),
        },
        "rows": counts["rows"],
        "questions": questions_count,
        "group_size_counts": {str(k): v for k, v in sorted(size_counts.items())},
        "telemetry": {
            "exact_groups": exact_telemetry_groups,
            "rows_missing_exact_num_turns": counts["rows_missing_exact_num_turns"],
            "rows_missing_exact_response_tokens": counts[
                "rows_missing_exact_response_tokens"
            ],
            "fallback_turn_metric": "2 * tool_calls + 1",
            "fallback_length_metric": "serialized whitespace words",
        },
        "thresholds": {
            "selection_mode": args.selection_mode,
            "best_evidence_mode": args.best_evidence_mode,
            "min_f1_range": args.min_f1_range,
            "group_mean_turns": [args.min_group_mean_turns, args.max_group_mean_turns],
            "group_mean_response_tokens": [
                args.min_group_mean_response_tokens,
                args.max_group_mean_response_tokens,
            ],
            "min_failure_success_turn_gap": args.min_failure_success_turn_gap,
            "min_failure_success_token_reduction": args.min_failure_success_token_reduction,
            "min_group_mean_search_calls": args.min_group_mean_search_calls,
            "max_group_inspection_search_ratio": args.max_group_inspection_search_ratio,
            "best_tier_mean_turns": [
                args.min_best_tier_mean_turns,
                args.max_best_tier_mean_turns,
            ],
            "best_tier_mean_response_tokens": [
                args.min_best_tier_mean_response_tokens,
                args.max_best_tier_mean_response_tokens,
            ],
            "min_best_tier_mean_search_calls": args.min_best_tier_mean_search_calls,
            "max_best_tier_inspection_search_ratio": (
                args.max_best_tier_inspection_search_ratio
            ),
            "max_absolute_failure_success_turn_gap": (
                args.max_absolute_failure_success_turn_gap
            ),
            "max_absolute_failure_success_token_gap": (
                args.max_absolute_failure_success_token_gap
            ),
            "long_assistant_threshold": args.long_assistant_threshold,
        },
        "rates": {
            "protocol_valid": protocol_valid_rate,
            "valid_f1_informative": informative_rate,
            "best_target_in_open_find": (
                category_counts["best_target_in_open_find"] / questions_count
                if questions_count
                else 0.0
            ),
            "best_answer_doc_target_supported": (
                category_counts["best_answer_doc_target_supported"] / questions_count
                if questions_count
                else 0.0
            ),
            "best_length_matched": (
                category_counts["best_length_matched"] / questions_count
                if questions_count
                else 0.0
            ),
            "best_search_behavior_matched": (
                category_counts["best_search_behavior_matched"] / questions_count
                if questions_count
                else 0.0
            ),
            "success_failure_gap_matched": (
                category_counts["success_failure_gap_matched"] / questions_count
                if questions_count
                else 0.0
            ),
            "core_behavior_matched": (
                category_counts["core_behavior_matched"] / questions_count
                if questions_count
                else 0.0
            ),
            "strong_behavior_matched": (
                category_counts["strong_behavior_matched"] / questions_count
                if questions_count
                else 0.0
            ),
            "selected_behavior_matched": behavior_rate,
        },
        "counts": dict(sorted(category_counts.items())),
        "distributions": {
            "group_mean_turns": _quantiles(
                [row["group_mean_turns"] for row in details_rows]
            ),
            "group_mean_response_tokens": _quantiles(
                [row["group_mean_response_tokens"] for row in details_rows]
            ),
            "group_mean_search_calls": _quantiles(
                [row["group_mean_search_calls"] for row in details_rows]
            ),
            "group_inspection_search_ratio": _quantiles(
                [row["group_inspection_search_ratio"] for row in details_rows]
            ),
            "best_tier_mean_turns_informative": _quantiles(
                [
                    row["best_tier_mean_turns"]
                    for row in details_rows
                    if row["informative"]
                ]
            ),
            "best_tier_mean_response_tokens_informative": _quantiles(
                [
                    row["best_tier_mean_response_tokens"]
                    for row in details_rows
                    if row["informative"]
                ]
            ),
            "best_tier_mean_search_calls_informative": _quantiles(
                [
                    row["best_tier_mean_search_calls"]
                    for row in details_rows
                    if row["informative"]
                ]
            ),
            "best_tier_inspection_search_ratio_informative": _quantiles(
                [
                    row["best_tier_inspection_search_ratio"]
                    for row in details_rows
                    if row["informative"]
                    and row["best_tier_mean_search_calls"] > 0
                ]
            ),
            "failure_minus_success_turns_informative": _quantiles(
                [
                    row["failure_minus_success_turns"]
                    for row in details_rows
                    if row["informative"]
                ]
            ),
            "failure_to_success_token_reduction_informative": _quantiles(
                [
                    row["failure_to_success_token_reduction"]
                    for row in details_rows
                    if row["informative"]
                ]
            ),
            "failure_minus_success_response_tokens_informative": _quantiles(
                [
                    row["failure_minus_success_response_tokens"]
                    for row in details_rows
                    if row["informative"]
                ]
            ),
        },
        "scale_projection": {
            "target_behavior_matched_groups": args.target_selected_groups,
            "observed_selected_behavior_match_rate": behavior_rate,
            "estimated_raw_questions_required": estimated_raw,
        },
        "gate": {
            "passed": not failures,
            "failures": failures,
            "recommendation": (
                "run_next_pilot_stage" if not failures else "revise_synthesis_before_scaling"
            ),
        },
        "outputs": {
            "details_jsonl": str(args.details_jsonl),
            "selected_jsonl": str(args.selected_jsonl) if args.selected_jsonl else None,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.fail_on_gate and failures:
        raise SystemExit(2)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
