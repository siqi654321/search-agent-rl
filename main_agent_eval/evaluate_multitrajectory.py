#!/usr/bin/env python3
"""Target-blind aggregation for two to eight BrowseComp+ trajectories.

``prepare`` aligns completed validation dumps and groups trajectories by their
distinct normalized final answers.  The emitted model input contains only the
question, candidate answers, generic run status, and evidence visible in the
original trajectories.  It never loads qrels and never writes targets.

``run`` asks an OpenAI-compatible endpoint to choose one existing candidate.
The action space is a minimal JSON label (A-H); free-form answer synthesis and
pairwise elimination are deliberately excluded.

``evaluate`` is the only stage that loads targets from the original dumps.  It
reports every single-trajectory score, the N-trajectory oracle, and the real
target-blind aggregate score.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts import evaluate_browsecomp_answer_verifier as verifier
    from scripts import evaluate_browsecomp_candidate_selector as common
    from scripts import evaluate_browsecomp_multitrajectory as pair
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "BrowseComp aggregation helpers were not found. Run this script with the "
        "VERL checkout as the current directory and ensure its scripts/ directory "
        "contains evaluate_browsecomp_answer_verifier.py, "
        "evaluate_browsecomp_candidate_selector.py, and "
        "evaluate_browsecomp_multitrajectory.py."
    ) from exc


JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
MAX_TRAJECTORIES = 8


@dataclass(frozen=True)
class TrajectorySpec:
    label: str
    path: Path
    public_id: str


def parse_trajectory_specs(values: list[str]) -> list[TrajectorySpec]:
    if not 2 <= len(values) <= MAX_TRAJECTORIES:
        raise ValueError(f"Expected 2-{MAX_TRAJECTORIES} --trajectory values, got {len(values)}")
    specs: list[TrajectorySpec] = []
    labels: set[str] = set()
    paths: set[Path] = set()
    for index, value in enumerate(values, start=1):
        if "=" not in value:
            raise ValueError(f"Trajectory must be LABEL=PATH, got: {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
        if not label or not str(path):
            raise ValueError(f"Trajectory must be LABEL=PATH, got: {value!r}")
        if label in labels:
            raise ValueError(f"Duplicate trajectory label: {label}")
        resolved = path.resolve()
        if resolved in paths:
            raise ValueError(f"Duplicate trajectory path: {path}")
        labels.add(label)
        paths.add(resolved)
        specs.append(TrajectorySpec(label=label, path=path, public_id=f"T{index}"))
    return specs


def _candidate_label(index: int) -> str:
    if not 0 <= index < MAX_TRAJECTORIES:
        raise ValueError(f"Candidate index out of range: {index}")
    return chr(ord("A") + index)


def _candidate_by_label(record: dict[str, Any], label: str) -> dict[str, Any]:
    wanted = str(label or "").strip().upper()
    for candidate in record.get("candidates", []):
        if str(candidate.get("label", "")).upper() == wanted:
            return candidate
    raise KeyError(f"Record {record.get('task_id')} lacks candidate {wanted}")


def _build_candidates(
    sources: list[dict[str, Any]],
    *,
    anchor_public_id: str,
) -> tuple[list[dict[str, Any]], str, str]:
    nonempty_groups: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    for source in sources:
        normalized = common.normalize_answer(source.get("answer", ""))
        if not normalized:
            continue
        if normalized not in nonempty_groups:
            nonempty_groups[normalized] = []
            group_order.append(normalized)
        nonempty_groups[normalized].append(source)

    if not group_order:
        return (
            [
                {
                    "label": "A",
                    "answer": "",
                    "supporting_trajectories": [source["id"] for source in sources],
                    "sources": sources,
                }
            ],
            "A",
            "all_answers_empty",
        )

    anchor_group = ""
    for normalized, grouped_sources in nonempty_groups.items():
        if any(source["id"] == anchor_public_id for source in grouped_sources):
            anchor_group = normalized
            break
    ordered_groups = list(group_order)
    if anchor_group:
        ordered_groups.remove(anchor_group)
        ordered_groups.insert(0, anchor_group)

    candidates: list[dict[str, Any]] = []
    for index, normalized in enumerate(ordered_groups):
        grouped_sources = nonempty_groups[normalized]
        representative = next(
            (
                source
                for source in grouped_sources
                if source["id"] == anchor_public_id
            ),
            grouped_sources[0],
        )
        candidates.append(
            {
                "label": _candidate_label(index),
                "answer": representative["answer"],
                "supporting_trajectories": [source["id"] for source in grouped_sources],
                "sources": grouped_sources,
            }
        )
    auto_reason = "single_distinct_nonempty_answer" if len(candidates) == 1 else "model_required"
    return candidates, "A", auto_reason


def prepare(args: argparse.Namespace) -> None:
    specs = parse_trajectory_specs(args.trajectory)
    dumps = {spec.public_id: pair.load_dump(spec.path) for spec in specs}
    expected_ids = set(dumps[specs[0].public_id])
    for spec in specs[1:]:
        ids = set(dumps[spec.public_id])
        if ids != expected_ids:
            raise ValueError(
                f"Query-id mismatch for {spec.label}: "
                f"missing={len(expected_ids - ids)} extra={len(ids - expected_ids)}"
            )

    anchor_spec = next((spec for spec in specs if spec.label == args.anchor), None)
    if anchor_spec is None:
        available = ", ".join(spec.label for spec in specs)
        raise ValueError(f"Anchor {args.anchor!r} is not present; available: {available}")

    records: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    candidate_counts: Counter[int] = Counter()
    evidence_chars: list[int] = []
    sort_key = lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)
    for query_id in sorted(expected_ids, key=sort_key):
        first_row = dumps[specs[0].public_id][query_id]
        question = str(first_row.get("question", "")).strip()
        sources: list[dict[str, Any]] = []
        for spec in specs:
            row = dumps[spec.public_id][query_id]
            if str(row.get("question", "")).strip() != question:
                raise ValueError(f"Question mismatch for query id {query_id}: {spec.label}")
            answer = common.extract_answer(row.get("output"))
            evidence = verifier.extract_evidence_pack(
                row.get("output"),
                answer,
                max_chars=args.evidence_max_chars,
                open_block_max_chars=args.open_block_max_chars,
                search_block_max_chars=args.search_block_max_chars,
            )
            sources.append(
                {
                    "id": spec.public_id,
                    "answer": answer,
                    "telemetry": pair._telemetry(row),
                    "evidence": evidence,
                }
            )
            evidence_chars.append(len(evidence))

        candidates, fallback_candidate, auto_reason = _build_candidates(
            sources,
            anchor_public_id=anchor_spec.public_id,
        )
        reasons[auto_reason] += 1
        candidate_counts[len(candidates)] += 1
        record = {
            "schema": "browsecomp_multitrajectory_n_input_v1",
            "task_id": query_id,
            "query_id": query_id,
            "dataset_index": str(first_row.get("dataset_index", "")),
            "question": question,
            "trajectory_count": len(specs),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "fallback_candidate": fallback_candidate,
            "auto_reason": auto_reason,
        }
        leaked = pair._forbidden_keys(record)
        if leaked:
            raise AssertionError(f"Forbidden model-input keys for {query_id}: {sorted(leaked)}")
        records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "schema": "browsecomp_multitrajectory_n_prepare_report_v1",
                "trajectories": [
                    {"label": spec.label, "path": str(spec.path), "public_id": spec.public_id}
                    for spec in specs
                ],
                "anchor": args.anchor,
                "output": str(args.output),
                "rows": len(records),
                "model_required_rows": reasons["model_required"],
                "automatic_rows": len(records) - reasons["model_required"],
                "auto_reasons": dict(reasons),
                "candidate_count_rows": {
                    str(count): rows for count, rows in sorted(candidate_counts.items())
                },
                "evidence_chars_per_trajectory": {
                    "mean": sum(evidence_chars) / len(evidence_chars) if evidence_chars else 0,
                    "max": max(evidence_chars, default=0),
                },
                "targets_and_qrels_are_not_loaded_or_written": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _render_source(source: dict[str, Any], source_number: int) -> str:
    telemetry = source.get("telemetry") or {}
    return f"""Evidence source {source_number}:
Run status: finalization={telemetry.get('finalization_reason', 'unknown')}, searches={telemetry.get('search_calls', 0)}, opens={telemetry.get('open_calls', 0)}, finds={telemetry.get('find_calls', 0)}
Visible evidence:
{source.get('evidence', '') or '(no usable evidence extracted)'}"""


def render_prompt(record: dict[str, Any]) -> str:
    cards: list[str] = []
    for candidate in record.get("candidates", []):
        evidence = "\n\n".join(
            _render_source(source, index)
            for index, source in enumerate(candidate.get("sources", []), start=1)
        )
        cards.append(
            f"""Candidate {candidate.get('label')}:
Proposed answer: {candidate.get('answer', '')}
{evidence}"""
        )
    labels = [str(candidate.get("label", "")) for candidate in record.get("candidates", [])]
    examples = " or ".join(f'{{"decision":"{label}"}}' for label in labels)
    return f"""Question:
{record.get('question', '')}

Candidate answers and their model-visible evidence:

{chr(10).join(cards)}

Compare every candidate using only the evidence shown. Determine all constraints in the question and choose
the candidate supported by the most complete, direct, and internally consistent evidence chain. Do not prefer
an answer because more trajectories proposed it, because its evidence is longer, or because it appears first.
Opaque document identifiers have no semantic meaning. You may not invent, rewrite, or combine answers.

Do all comparison before responding. Return one minimal JSON object and no explanation:
{examples}
"""


def resolve_response(content: Any, record: dict[str, Any]) -> tuple[str, str, str]:
    text = str(content or "").strip()
    payload: Any = None
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    brace = JSON_OBJECT_RE.search(text)
    if brace:
        candidates.append(brace.group(0))
    for candidate_text in candidates:
        try:
            parsed = json.loads(candidate_text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    allowed = {
        str(candidate.get("label", "")).upper()
        for candidate in record.get("candidates", [])
    }
    decision = str(payload.get("decision", "")).strip().upper() if isinstance(payload, dict) else ""
    if not decision and re.fullmatch(r"[A-H]", text.upper()):
        decision = text.upper()
    if decision in allowed:
        answer = str(_candidate_by_label(record, decision).get("answer", ""))
        return answer, decision, "parsed"
    fallback = str(record.get("fallback_candidate", "A")).upper()
    answer = str(_candidate_by_label(record, fallback).get("answer", ""))
    return answer, fallback, "parse_fallback"


def _prediction(
    record: dict[str, Any],
    *,
    endpoint: str,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    started = time.time()
    if str(record.get("auto_reason")) != "model_required":
        decision = str(record.get("fallback_candidate", "A"))
        answer = str(_candidate_by_label(record, decision).get("answer", ""))
        return {
            "schema": "browsecomp_multitrajectory_n_prediction_v1",
            "task_id": record["task_id"],
            "query_id": record["query_id"],
            "decision": decision,
            "final_answer": answer,
            "resolution": str(record.get("auto_reason", "automatic")),
            "raw_response": "",
            "latency_seconds": 0.0,
            "error": None,
        }

    request_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the final evidence judge for several independent deep-research "
                    "trajectories. Use only supplied evidence and return the required minimal JSON."
                ),
            },
            {"role": "user", "content": render_prompt(record)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request_data = json.dumps(request_payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_error = ""
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                endpoint, data=request_data, headers=headers, method="POST"
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            message = result["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
            answer, decision, resolution = resolve_response(content, record)
            return {
                "schema": "browsecomp_multitrajectory_n_prediction_v1",
                "task_id": record["task_id"],
                "query_id": record["query_id"],
                "decision": decision,
                "final_answer": answer,
                "resolution": resolution,
                "raw_response": content,
                "latency_seconds": time.time() - started,
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            last_error = f"HTTPError: HTTP {exc.code}: {body[:2000]}"
        except (
            KeyError,
            IndexError,
            json.JSONDecodeError,
            urllib.error.URLError,
            TimeoutError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(8.0, 1.5 * (2**attempt)))

    fallback = str(record.get("fallback_candidate", "A"))
    answer = str(_candidate_by_label(record, fallback).get("answer", ""))
    return {
        "schema": "browsecomp_multitrajectory_n_prediction_v1",
        "task_id": record["task_id"],
        "query_id": record["query_id"],
        "decision": fallback,
        "final_answer": answer,
        "resolution": "request_error_fallback",
        "raw_response": "",
        "latency_seconds": time.time() - started,
        "error": last_error,
    }


def run(args: argparse.Namespace) -> None:
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        records = records[: args.limit]
    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[str(row["task_id"])] = row
    pending = [
        record
        for record in records
        if str(record["task_id"]) not in existing
        or (args.retry_errors and bool(existing[str(record["task_id"])].get("error")))
    ]
    endpoint = pair._chat_endpoint(args.endpoint)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def work(record: dict[str, Any]) -> dict[str, Any]:
        return _prediction(
            record,
            endpoint=endpoint,
            model=args.model,
            api_key=api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            retries=args.retries,
        )

    completed = errors = model_calls = automatic = 0
    with args.output.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(work, record) for record in pending]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                completed += 1
                errors += bool(result.get("error"))
                if float(result.get("latency_seconds") or 0) > 0:
                    model_calls += 1
                else:
                    automatic += 1
                if completed % 20 == 0 or completed == len(pending):
                    print(
                        f"completed={completed}/{len(pending)} model_calls={model_calls} "
                        f"automatic={automatic} errors={errors}",
                        flush=True,
                    )
    print(
        json.dumps(
            {
                "input_rows": len(records),
                "resumed_rows": len(records) - len(pending),
                "completed_rows": completed,
                "model_calls": model_calls,
                "automatic_rows": automatic,
                "errors": errors,
                "output": str(args.output),
                "endpoint": endpoint,
            },
            indent=2,
        )
    )


def evaluate(args: argparse.Namespace) -> None:
    specs = parse_trajectory_specs(args.trajectory)
    dumps = {spec.public_id: pair.load_dump(spec.path) for spec in specs}
    inputs = {
        str(row["task_id"]): row
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    predictions: dict[str, dict[str, Any]] = {}
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            predictions[str(row["task_id"])] = row

    trajectory_sums = Counter()
    trajectory_exacts = Counter()
    aggregate_sum = oracle_sum = 0.0
    aggregate_exact = oracle_exact = 0
    counts = Counter()
    decisions = Counter()
    resolutions = Counter()
    for task_id, record in inputs.items():
        query_id = str(record["query_id"])
        first_row = dumps[specs[0].public_id][query_id]
        target = common.get_target(first_row.get("gts"))
        scores: dict[str, float] = {}
        for spec in specs:
            row = dumps[spec.public_id][query_id]
            other_target = common.get_target(row.get("gts"))
            if common.normalize_answer(target) != common.normalize_answer(other_target):
                raise ValueError(f"Target mismatch for query id {query_id}: {spec.label}")
            answer = common.extract_answer(row.get("output"))
            score = common.normalized_token_f1(answer, target)
            scores[spec.label] = score
            trajectory_sums[spec.label] += score
            trajectory_exacts[spec.label] += score == 1.0

        oracle_score = max(scores.values())
        prediction = predictions.get(task_id)
        if prediction is None:
            counts["missing_predictions"] += 1
            fallback = str(record.get("fallback_candidate", "A"))
            final_answer = str(_candidate_by_label(record, fallback).get("answer", ""))
            decision = fallback
            resolution = "missing_prediction_fallback"
        else:
            final_answer = str(prediction.get("final_answer", ""))
            decision = str(prediction.get("decision", ""))
            resolution = str(prediction.get("resolution", "unknown"))
            counts["prediction_errors"] += bool(prediction.get("error"))
        aggregate_score = common.normalized_token_f1(final_answer, target)
        aggregate_sum += aggregate_score
        oracle_sum += oracle_score
        aggregate_exact += aggregate_score == 1.0
        oracle_exact += oracle_score == 1.0
        opportunities = max(scores.values()) > min(scores.values()) + 1e-12
        counts["oracle_choice_opportunities"] += opportunities
        counts["oracle_choice_wins"] += opportunities and abs(aggregate_score - oracle_score) <= 1e-12
        counts["model_required_rows"] += record.get("auto_reason") == "model_required"
        counts["automatic_rows"] += record.get("auto_reason") != "model_required"
        counts["rows"] += 1
        decisions[decision] += 1
        resolutions[resolution] += 1

    rows = int(counts["rows"])
    trajectory_means = {
        spec.label: trajectory_sums[spec.label] / rows for spec in specs
    }
    best_label = max(trajectory_means, key=trajectory_means.get)
    best_single = trajectory_means[best_label]
    aggregate_mean = aggregate_sum / rows
    oracle_mean = oracle_sum / rows
    gap = oracle_mean - best_single
    recovered = aggregate_mean - best_single
    report = {
        "schema": "browsecomp_multitrajectory_n_eval_v1",
        "input": str(args.input),
        "predictions": str(args.predictions),
        "targets_loaded_only_during_evaluation": True,
        "rows": rows,
        "trajectory_count": len(specs),
        "normalized_f1": {
            "per_trajectory": trajectory_means,
            "best_single_label": best_label,
            "best_single": best_single,
            "n_trajectory_oracle": oracle_mean,
            "aggregate": aggregate_mean,
            "aggregate_delta_vs_best": recovered,
            "available_oracle_gap": gap,
            "oracle_gap_recovery_rate": recovered / gap if gap > 0 else 0.0,
            "delta_830_points": recovered * 100,
        },
        "exact_rows": {
            "per_trajectory": {
                spec.label: int(trajectory_exacts[spec.label]) for spec in specs
            },
            "n_trajectory_union": oracle_exact,
            "aggregate": aggregate_exact,
        },
        "paired": {
            "oracle_choice_wins": int(counts["oracle_choice_wins"]),
            "oracle_choice_opportunities": int(counts["oracle_choice_opportunities"]),
        },
        "execution": {
            "model_required_rows": int(counts["model_required_rows"]),
            "automatic_rows": int(counts["automatic_rows"]),
            "missing_predictions": int(counts["missing_predictions"]),
            "prediction_errors": int(counts["prediction_errors"]),
            "decisions": dict(decisions),
            "resolutions": dict(resolutions),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--trajectory", action="append", required=True, help="Repeat LABEL=PATH two to eight times."
    )
    prepare_parser.add_argument("--anchor", required=True, help="Global fallback trajectory label.")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--evidence-max-chars", type=int, default=32_000)
    prepare_parser.add_argument("--open-block-max-chars", type=int, default=10_000)
    prepare_parser.add_argument("--search-block-max-chars", type=int, default=4_000)
    prepare_parser.set_defaults(func=prepare)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--endpoint", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--api-key", default="")
    run_parser.add_argument("--max-tokens", type=int, default=64)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--timeout", type=float, default=300.0)
    run_parser.add_argument("--retries", type=int, default=2)
    run_parser.add_argument("--concurrency", type=int, default=4)
    run_parser.add_argument("--retry-errors", action="store_true")
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.set_defaults(func=run)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--trajectory", action="append", required=True, help="Repeat the same LABEL=PATH values."
    )
    evaluate_parser.add_argument("--input", type=Path, required=True)
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
