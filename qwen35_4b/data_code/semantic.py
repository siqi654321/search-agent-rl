#!/usr/bin/env python3
"""Semantically audit and minimally repair V4 atomic fact-chain questions.

The V4 generator already verifies lexical grounding, answer leakage, and the
tool-facing parquet schema.  Its proper-name heuristic cannot, however, prove
that a question asking for (for example) an organization is paired with an
organization rather than a film title.  This post-generation pass checks that
semantic contract without changing the grounded chain construction.

Only the final interrogative request may be repaired.  Everything before the
last WH request is retained byte-for-byte.  A repaired row is accepted only
after a second semantic audit.  The source JSONL is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple, Sequence

try:
    from data_code import generate as generator
    from data_code import qa_common as base
except ImportError:  # direct execution via ``python3 data_code/...``
    import generate as generator  # type: ignore
    import qa_common as base  # type: ignore


SCHEMA = "browsecomp_safe_atomic_fact_chain_semantic_clean_v2"
INPUT_SCHEMA = "browsecomp_safe_atomic_fact_chain_qa_v4"
INPUT_SOURCE = "browsecomp_safe_atomic_fact_chain_synth_v4"
ANSWER_TYPES = {
    "person",
    "organization",
    "work",
    "event",
    "place",
    "name_fragment",
    "acronym",
    "short_phrase",
    "other",
}
WH_REQUEST_RE = re.compile(r"\b(?:who|what|which|where|when)\b", re.I)
GENERIC_ROLE_WORDS = {
    "a",
    "an",
    "the",
    "american",
    "british",
    "canadian",
    "hollywood",
    "international",
    "specific",
    "actor",
    "actress",
    "artist",
    "author",
    "company",
    "department",
    "event",
    "faculty",
    "film",
    "foundation",
    "group",
    "host",
    "individual",
    "institution",
    "journal",
    "location",
    "member",
    "movie",
    "organization",
    "person",
    "place",
    "player",
    "program",
    "scholarship",
    "site",
    "star",
    "university",
    "website",
    "work",
    "writer",
}
GENERIC_ROLE_HEADS = {
    "actor",
    "actress",
    "artist",
    "author",
    "company",
    "department",
    "event",
    "foundation",
    "group",
    "host",
    "individual",
    "institution",
    "journal",
    "location",
    "member",
    "movie",
    "organization",
    "person",
    "place",
    "player",
    "program",
    "scholarship",
    "site",
    "star",
    "university",
    "website",
    "work",
    "writer",
}


class RequestSpec(NamedTuple):
    candidate_id: str


def norm_space(value: Any) -> str:
    return base.norm_space(value)


def norm_key(value: Any) -> str:
    return base.norm_key(value)


def _normalize_type(value: Any) -> str:
    raw = norm_key(value)
    aliases = {
        "person": "person",
        "individual": "person",
        "human": "person",
        "name": "person",
        "organization": "organization",
        "organisation": "organization",
        "company": "organization",
        "institution": "organization",
        "foundation": "organization",
        "work": "work",
        "film": "work",
        "movie": "work",
        "book": "work",
        "novel": "work",
        "song": "work",
        "title": "work",
        "event": "event",
        "place": "place",
        "location": "place",
        "first name": "name_fragment",
        "surname": "name_fragment",
        "last name": "name_fragment",
        "name fragment": "name_fragment",
        "abbreviation": "acronym",
        "initialism": "acronym",
        "short phrase": "short_phrase",
    }
    return aliases.get(raw, raw)


def is_obvious_generic_role(answer: str) -> bool:
    """Catch labels such as ``Hollywood actor`` that capitalization admits."""

    words = norm_key(answer).split()
    return bool(
        words
        and words[-1] in GENERIC_ROLE_HEADS
        and all(word in GENERIC_ROLE_WORDS for word in words)
    )


def split_final_request(question: str) -> tuple[str, str] | None:
    """Split at the final WH word so an upstream grounded prefix is immutable."""

    text = norm_space(question)
    if not text.endswith("?"):
        return None
    matches = list(WH_REQUEST_RE.finditer(text))
    if not matches:
        return None
    # A final request should occur in the latter half of these long questions.
    # This avoids treating relative clauses such as "a player who ..." as the
    # replaceable request when a malformed question lacks a final WH clause.
    candidates = [match for match in matches if match.start() >= len(text) * 0.45]
    if not candidates:
        return None
    match = candidates[-1]
    return text[: match.start()], text[match.start() :]


def build_audit_prompt(row: dict[str, Any], *, repaired: bool = False) -> str:
    phase = "repaired candidate" if repaired else "generated candidate"
    request_split = split_final_request(norm_space(row.get("question")))
    final_request = request_split[1] if request_split else ""
    return f"""You are a conservative local-contract auditor for one deep-research QA {phase}.

Audit ONLY whether the supplied short answer can grammatically and semantically
fill the final request in the question.  Do not solve or fact-check the research
chain.  The upstream premises are supported by separate documents that are not
shown here.  The terminal quote is intentionally only a short local excerpt; it
is not supposed to prove every premise in the question.

Rules:
- Never reject because the terminal quote does not restate an upstream clue,
  relation, date, occupation, or the complete multi-hop chain.
- "Insufficient evidence", "unrelated premises", and factual disagreement are
  not valid rejection reasons in this local audit.  The quote may be used only
  to disambiguate whether the answer looks like a person, work, organization,
  event, place, fragment, acronym, or short phrase.
- Never use outside knowledge to dispute a premise, identity, publication year,
  or attribution.  If a mismatch is not directly observable from the supplied
  answer and final request, pass it for downstream strict rollout.
- Named titles, award/category names, program names, acronyms, first names, and
  surnames are allowed when the final request explicitly asks for that form.
  Do not require every answer to be a globally famous full entity name.
- Reject an unusable target such as a role/category (for example "Hollywood
  actor"), placeholder, UI fragment, or incomplete answer only when the final
  request explicitly requires a fuller form such as "full name".
- A film/work cannot fill a request for an organization, a person cannot fill
  a request for an event, and analogous directly visible type mismatches fail.
- Use verdict "repair" only when the answer itself is usable and the sole
  directly visible problem can be fixed by replacing the final WH request.
  Do not change upstream facts, the answer, or evidence.
- A repair request must not contain the supplied answer or add a new factual
  claim.  It should retain the descriptive constraints from the original
  request while asking for the correct type.
- For a repaired candidate, compare against original_question and reject it if
  the replacement request dropped or changed a descriptive constraint.

Candidate:
{json.dumps({
    "question": row.get("question"),
    "final_request": final_request,
    "original_question": row.get("semantic_original_question"),
    "answer": row.get("answer"),
    "terminal_local_type_context_do_not_fact_check": row.get("answer_evidence_quote"),
}, ensure_ascii=False, indent=2)}

Return exactly one JSON object:
{{
  "verdict": "pass|repair|reject",
  "requested_answer_type": "person|organization|work|event|place|name_fragment|acronym|short_phrase|other",
  "answer_semantic_type": "person|organization|work|event|place|name_fragment|acronym|short_phrase|other",
  "answer_is_usable_target": true,
  "answer_is_self_contained_for_requested_form": true,
  "answer_fits_final_request": true,
  "request_only_repairable": false,
  "repair_preserves_original_constraints": true,
  "replacement_request": "empty unless verdict is repair; one WH clause ending in ?",
  "reason": "short concrete explanation"
}}
Do not return Markdown or commentary."""


def parse_audit_response(response: str) -> tuple[dict[str, Any] | None, str]:
    parsed = base.parse_json_object(response)
    if not parsed:
        return None, "malformed_audit_json"
    verdict = norm_key(parsed.get("verdict"))
    if verdict not in {"pass", "repair", "reject"}:
        return None, "invalid_audit_verdict"
    normalized = dict(parsed)
    normalized["verdict"] = verdict
    normalized["requested_answer_type"] = _normalize_type(
        parsed.get("requested_answer_type")
    )
    normalized["answer_semantic_type"] = _normalize_type(
        parsed.get("answer_semantic_type")
    )
    for key in (
        "answer_is_usable_target",
        "answer_is_self_contained_for_requested_form",
        "answer_fits_final_request",
        "request_only_repairable",
        "repair_preserves_original_constraints",
    ):
        if not isinstance(parsed.get(key), bool):
            return None, f"invalid_audit_boolean:{key}"
    normalized["replacement_request"] = norm_space(parsed.get("replacement_request"))
    normalized["reason"] = norm_space(parsed.get("reason"))
    return normalized, ""


def audit_action(row: dict[str, Any], audit: dict[str, Any]) -> tuple[str, str]:
    if is_obvious_generic_role(norm_space(row.get("answer"))):
        return "reject", "deterministic_generic_role_answer"
    if not base.contains_answer(
        norm_space(row.get("answer_evidence_quote")), norm_space(row.get("answer"))
    ):
        return "reject", "deterministic_answer_missing_from_terminal_quote"
    usable = audit["answer_is_usable_target"] is True
    self_contained = audit["answer_is_self_contained_for_requested_form"] is True
    fits = audit["answer_fits_final_request"] is True
    answer_type = _normalize_type(audit.get("answer_semantic_type"))
    if (
        audit["verdict"] == "pass"
        and usable
        and self_contained
        and fits
        and answer_type in ANSWER_TYPES
        and (
            norm_space(row.get("semantic_clean_status")) != "repaired"
            or audit["repair_preserves_original_constraints"] is True
        )
    ):
        return "pass", "semantic_contract_passed"
    if (
        audit["verdict"] == "repair"
        and usable
        and self_contained
        and not fits
        and audit["request_only_repairable"] is True
        and answer_type in ANSWER_TYPES
        and audit.get("replacement_request")
    ):
        return "repair", "request_only_repair"
    return "reject", norm_space(audit.get("reason")) or "semantic_contract_failed"


def apply_request_repair(
    row: dict[str, Any], audit: dict[str, Any], *, min_words: int, max_words: int
) -> tuple[dict[str, Any] | None, str]:
    split = split_final_request(norm_space(row.get("question")))
    if split is None:
        return None, "final_wh_request_not_found"
    prefix, _old_request = split
    replacement = norm_space(audit.get("replacement_request"))
    if not replacement or replacement.count("?") != 1 or not replacement.endswith("?"):
        return None, "invalid_replacement_question_mark"
    if not WH_REQUEST_RE.match(replacement):
        return None, "replacement_not_wh_request"
    if len(generator.WORD_RE.findall(replacement)) > 55:
        return None, "replacement_request_too_long"
    question = norm_space(prefix + replacement)
    answer = norm_space(row.get("answer"))
    word_count = len(generator.WORD_RE.findall(question))
    if not min_words <= word_count <= max_words:
        return None, "repaired_question_length"
    if question.count("?") != 1 or generator.MULTI_DELIVERABLE_RE.search(question):
        return None, "repaired_multiple_deliverables"
    if base.contains_answer(question, answer):
        return None, "repaired_answer_leak"
    if generator.SYNTHETIC_META_RE.search(question):
        return None, "repaired_synthetic_meta_language"
    for bridge in row.get("bridge_entities") or []:
        if base.contains_answer(question, norm_space(bridge)):
            return None, "repaired_bridge_name_leak"

    repaired = dict(row)
    repaired["semantic_original_question"] = norm_space(row.get("question"))
    repaired["question"] = question
    repaired["question_words"] = word_count
    repaired["answer_type"] = _normalize_type(audit.get("answer_semantic_type"))
    repaired["semantic_clean_status"] = "repaired"
    return repaired, ""


def _cache_key(stage: str, query_id: str, prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{stage}:{query_id}:{digest}"


def _read_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    cache: dict[str, str] = {}
    for row in base.read_jsonl(path):
        request_id = norm_space(row.get("request_id"))
        if request_id:
            cache[request_id] = str(row.get("response") or "")
    return cache


def run_cached_stage(
    *,
    stage: str,
    rows: Sequence[dict[str, Any]],
    prompts: Sequence[str],
    cache_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    if len(rows) != len(prompts):
        raise ValueError("rows/prompts length mismatch")
    cache = _read_cache(cache_path)
    request_ids = [
        _cache_key(
            f"{stage}:{args.model}", norm_space(row.get("query_id")), prompt
        )
        for row, prompt in zip(rows, prompts, strict=True)
    ]
    outputs = [cache.get(request_id, "") for request_id in request_ids]
    missing_indices = [index for index, value in enumerate(outputs) if not value.strip()]
    if not missing_indices:
        print(f"[BCP-ATOMIC-SEMANTIC] stage={stage} cache_hit={len(rows)}", flush=True)
        return outputs

    client = base.ChatClient(
        args.base_url,
        args.model,
        args.concurrency,
        args.timeout,
        args.retries,
        args.temperature,
        False,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = 0
    with cache_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(missing_indices), args.batch_size):
            indices = missing_indices[start : start + args.batch_size]
            specs = [RequestSpec(request_ids[index]) for index in indices]
            batch_prompts = [prompts[index] for index in indices]
            responses = client.generate(specs, batch_prompts, args.max_tokens)
            for index, response in zip(indices, responses, strict=True):
                outputs[index] = response
                handle.write(
                    json.dumps(
                        {"request_id": request_ids[index], "response": response},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            completed += len(indices)
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"[BCP-ATOMIC-SEMANTIC] stage={stage} "
                f"completed={completed}/{len(missing_indices)} rate={completed / elapsed:.2f}/s",
                flush=True,
            )
    print(
        f"[BCP-ATOMIC-SEMANTIC] stage={stage} request_stats="
        + json.dumps(dict(sorted(client.stats.items())), ensure_ascii=False),
        flush=True,
    )
    return outputs


def _validate_input(rows: Sequence[dict[str, Any]], path: Path) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        query_id = norm_space(row.get("query_id"))
        if not query_id:
            raise ValueError(f"{path}:{index}: missing query_id")
        if query_id in seen:
            raise ValueError(f"{path}:{index}: duplicate query_id={query_id}")
        seen.add(query_id)
        if row.get("schema") != INPUT_SCHEMA:
            raise ValueError(f"{path}:{index}: unexpected schema={row.get('schema')!r}")
        if row.get("source") != INPUT_SOURCE or row.get("answer_mode") != "terminal_atomic_fact":
            raise ValueError(f"{path}:{index}: not a V4 terminal atomic-fact row")
        for key in ("question", "answer", "answer_type", "answer_evidence_quote"):
            if not norm_space(row.get(key)):
                raise ValueError(f"{path}:{index}: missing {key}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    destination = (args.output_dir / "questions.jsonl").resolve()
    if args.input.resolve() == destination:
        raise ValueError("--output-dir must not overwrite the source questions.jsonl")
    rows = base.read_jsonl(args.input)
    _validate_input(rows, args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "semantic.responses.cache.jsonl"

    primary_prompts = [build_audit_prompt(row) for row in rows]
    primary_responses = run_cached_stage(
        stage="primary",
        rows=rows,
        prompts=primary_prompts,
        cache_path=cache_path,
        args=args,
    )

    audits: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    passed: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    repair_audits: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row, response in zip(rows, primary_responses, strict=True):
        parsed, parse_reason = parse_audit_response(response)
        if parsed is None:
            action, reason = "reject", parse_reason
            parsed = {"verdict": "invalid", "reason": parse_reason}
        else:
            action, reason = audit_action(row, parsed)
        counts[f"primary.{action}"] += 1
        audit_row = {
            "query_id": row["query_id"],
            "question": row["question"],
            "answer": row["answer"],
            "stage": "primary",
            "action": action,
            "reason": reason,
            "audit": parsed,
        }
        audits.append(audit_row)
        if action == "pass":
            clean = dict(row)
            clean["semantic_clean_status"] = "passed"
            clean["semantic_audit"] = parsed
            passed.append(clean)
        elif action == "repair":
            repair_rows.append(row)
            repair_audits.append(parsed)
        else:
            rejected.append(audit_row)

    repaired_candidates: list[dict[str, Any]] = []
    for row, audit in zip(repair_rows, repair_audits, strict=True):
        repaired, reason = apply_request_repair(
            row,
            audit,
            min_words=args.min_question_words,
            max_words=args.max_question_words,
        )
        if repaired is None:
            counts[f"repair_rejected.{reason}"] += 1
            rejected.append(
                {
                    "query_id": row["query_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "action": "reject",
                    "reason": reason,
                    "audit": audit,
                }
            )
            continue
        repaired_candidates.append(repaired)

    if repaired_candidates:
        verify_prompts = [build_audit_prompt(row, repaired=True) for row in repaired_candidates]
        verify_responses = run_cached_stage(
            stage="verify_repair",
            rows=repaired_candidates,
            prompts=verify_prompts,
            cache_path=cache_path,
            args=args,
        )
        for row, response in zip(repaired_candidates, verify_responses, strict=True):
            parsed, parse_reason = parse_audit_response(response)
            if parsed is None:
                action, reason = "reject", parse_reason
            else:
                action, reason = audit_action(row, parsed)
            audits.append(
                {
                    "query_id": row["query_id"],
                    "question": row["question"],
                    "original_question": row.get("semantic_original_question"),
                    "answer": row["answer"],
                    "stage": "verify_repair",
                    "action": action,
                    "reason": reason,
                    "audit": parsed
                    or {"verdict": "invalid", "reason": parse_reason},
                }
            )
            if action != "pass":
                counts[f"repair_rejected.{reason}"] += 1
                rejected.append(
                    {
                        "query_id": row["query_id"],
                        "question": row["question"],
                        "answer": row["answer"],
                        "action": "reject",
                        "reason": reason,
                        "audit": parsed or {"verdict": "invalid", "reason": parse_reason},
                    }
                )
                continue
            row["semantic_audit"] = parsed
            passed.append(row)
            counts["repair_accepted"] += 1

    # A request-only rewrite can still collide with another question.  Keep
    # uniqueness fail-closed while preserving deterministic source order.
    by_query_id = {row["query_id"]: index for index, row in enumerate(rows)}
    passed.sort(key=lambda row: by_query_id[row["query_id"]])
    clean_rows: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for row in passed:
        key = norm_key(row["question"])
        if key in seen_questions:
            counts["final_rejected.duplicate_question"] += 1
            rejected.append(
                {
                    "query_id": row["query_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "action": "reject",
                    "reason": "duplicate_question_after_repair",
                }
            )
            continue
        seen_questions.add(key)
        clean_rows.append(row)

    questions_jsonl = args.output_dir / "questions.jsonl"
    rejected_jsonl = args.output_dir / "rejected.jsonl"
    audit_jsonl = args.output_dir / "semantic.audit.jsonl"
    base.write_jsonl(questions_jsonl, clean_rows)
    base.write_jsonl(rejected_jsonl, rejected)
    base.write_jsonl(audit_jsonl, audits)

    raw_parquet: Path | None = None
    prepared_parquet: Path | None = None
    if clean_rows:
        raw_parquet, prepared_parquet = generator._prepare_parquet(clean_rows, args)

    final_status_counts = Counter(
        norm_space(row.get("semantic_clean_status")) or "unknown" for row in clean_rows
    )
    rejection_counts = Counter(norm_space(row.get("reason")) for row in rejected)
    gate_failures: list[str] = []
    if len(clean_rows) < args.minimum_clean_questions:
        gate_failures.append(
            f"clean_questions={len(clean_rows)} < required={args.minimum_clean_questions}"
        )
    report = {
        "schema": SCHEMA,
        "inputs": {
            "questions": str(args.input),
            "rows": len(rows),
            "model": args.model,
            "base_urls": args.base_url,
        },
        "policy": {
            "audit_scope": "final_request_answer_contract_only",
            "upstream_chain_factuality_judged": False,
            "outside_knowledge_allowed": False,
            "terminal_quote_full_chain_support_required": False,
            "terminal_quote_answer_presence_checked_deterministically": True,
            "chain_construction_changed": False,
            "answer_changed": False,
            "evidence_changed": False,
            "repair_scope": "final_wh_request_only",
            "one_repair_maximum": True,
            "repaired_rows_reaudited": True,
            "source_overwritten": False,
        },
        "counts": {
            "input_questions": len(rows),
            "clean_questions": len(clean_rows),
            "rejected_questions": len(rejected),
            "primary_actions": {
                key.split(".", 1)[1]: value
                for key, value in sorted(counts.items())
                if key.startswith("primary.")
            },
            "final_status": dict(sorted(final_status_counts.items())),
            "repair_accepted": counts["repair_accepted"],
            "rejection_reasons": dict(sorted(rejection_counts.items())),
        },
        "gate": {
            "passed": not gate_failures,
            "minimum_clean_questions": args.minimum_clean_questions,
            "failures": gate_failures,
        },
        "outputs": {
            "questions_jsonl": str(questions_jsonl),
            "rejected_jsonl": str(rejected_jsonl),
            "audit_jsonl": str(audit_jsonl),
            "raw_parquet": str(raw_parquet) if raw_parquet else None,
            "prepared_parquet": str(prepared_parquet) if prepared_parquet else None,
            "response_cache": str(cache_path),
        },
    }
    report_path = args.output_dir / "semantic_clean.report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.fail_on_gate and gate_failures:
        raise SystemExit("semantic clean gate failed")
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--min-question-words", type=int, default=70)
    parser.add_argument("--max-question-words", type=int, default=140)
    parser.add_argument("--minimum-clean-questions", type=int, default=48)
    parser.add_argument("--data-source", default="searchR1_openresearcher")
    parser.add_argument("--date", default="2026-09-04")
    parser.add_argument(
        "--prepare-script", type=Path, default=root / "prepare.py"
    )
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.concurrency <= 0 or args.batch_size <= 0:
        raise SystemExit("--concurrency and --batch-size must be positive")
    if not 12 <= args.min_question_words <= args.max_question_words <= 180:
        raise SystemExit("question word limits must satisfy 12 <= min <= max <= 180")
    run(args)


if __name__ == "__main__":
    main()
