#!/usr/bin/env python3
"""Materialize a small, grounded BrowseComp-safe anchor bundle pilot.

The input is the allow-list and title-anchor table emitted by
``corpus.py``.  This script selects a
deterministic anchor sample, scans only the referenced safe documents, and
keeps anchors supported by non-duplicate answer-bearing evidence snippets from
multiple documents/domains.

It does not generate questions, call an LLM, build a retrieval index, or expose
benchmark questions/answers to a generator.  The output is an offline capacity
artifact for deciding whether question synthesis is worth implementing.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from data_code.corpus import (
        canonical_domain,
        corpus_payload,
        iter_corpus_rows,
        load_benchmark_firewall,
        load_qrels,
        normalize_docid,
        normalized_text,
        normalized_tokens,
        stable_hash,
        write_jsonl,
    )
except ImportError:  # direct execution via ``python3 data_code/...``
    from corpus import (  # type: ignore
        canonical_domain,
        corpus_payload,
        iter_corpus_rows,
        load_benchmark_firewall,
        load_qrels,
        normalize_docid,
        normalized_text,
        normalized_tokens,
        stable_hash,
        write_jsonl,
    )


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
CAPITALIZED_RE = re.compile(r"\b(?:[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+){0,4})\b")
RELATION_WORDS = (
    "born",
    "died",
    "founded",
    "established",
    "created",
    "launched",
    "located",
    "based",
    "headquartered",
    "directed",
    "written",
    "authored",
    "produced",
    "owned",
    "operated",
    "member",
    "served",
    "worked",
    "married",
    "won",
    "awarded",
    "released",
    "published",
    "renamed",
    "known",
)
RELATION_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, RELATION_WORDS)) + r")\b", re.I)
BOILERPLATE_RE = re.compile(
    r"\b(?:cookie policy|privacy policy|all rights reserved|javascript is disabled|"
    r"subscribe now|sign in|log in|terms of service)\b",
    re.I,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_docids(path: Path) -> set[str]:
    return {
        docid
        for line in path.read_text(encoding="utf-8").splitlines()
        if (docid := normalize_docid(line))
    }


def phrase_pattern(answer: str) -> re.Pattern[str]:
    tokens = normalized_tokens(answer)
    if not tokens:
        return re.compile(r"(?!x)x")
    return re.compile(
        r"(?<!\w)" + r"[\W_]+".join(re.escape(token) for token in tokens) + r"(?!\w)",
        re.I,
    )


def _window_around_match(text: str, match: re.Match[str], max_chars: int) -> str:
    if len(text) <= max_chars:
        return text.strip()
    radius = max(80, (max_chars - len(match.group(0))) // 2)
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return text[start:end].strip(" \t\r\n,;:-")


def evidence_signatures(quote: str, answer: str) -> dict[str, list[str]]:
    answer_key = normalized_text(answer)
    entities: list[str] = []
    seen: set[str] = set()
    for value in CAPITALIZED_RE.findall(quote):
        key = normalized_text(value)
        if not key or key == answer_key or key in seen:
            continue
        seen.add(key)
        entities.append(value)
        if len(entities) >= 8:
            break
    return {
        "years": sorted(set(YEAR_RE.findall(quote)))[:8],
        "numbers": sorted(set(NUMBER_RE.findall(quote)))[:8],
        "relations": sorted({match.group(0).casefold() for match in RELATION_RE.finditer(quote)})[:8],
        "other_entities": entities,
    }


def quote_score(quote: str, answer: str) -> float:
    tokens = normalized_tokens(quote)
    answer_tokens = set(normalized_tokens(answer))
    non_answer = [token for token in tokens if token not in answer_tokens]
    if len(tokens) < 8 or len(non_answer) < 6 or BOILERPLATE_RE.search(quote):
        return -1.0
    score = min(len(set(non_answer)), 40) * 0.25
    score += 4.0 if YEAR_RE.search(quote) else 0.0
    score += 2.0 if NUMBER_RE.search(quote) else 0.0
    score += 4.0 if RELATION_RE.search(quote) else 0.0
    score += min(len(CAPITALIZED_RE.findall(quote)), 4) * 0.75
    score -= max(0, len(tokens) - 100) * 0.1
    return score


def best_evidence_quote(text: str, answer: str, max_chars: int) -> tuple[str, float] | None:
    pattern = phrase_pattern(answer)
    candidates: list[tuple[float, str]] = []
    for segment in SENTENCE_SPLIT_RE.split(text):
        segment = re.sub(r"\s+", " ", segment).strip()
        if not segment:
            continue
        match = pattern.search(segment)
        if not match:
            continue
        quote = _window_around_match(segment, match, max_chars)
        score = quote_score(quote, answer)
        if score >= 0:
            candidates.append((score, quote))
    if not candidates:
        match = pattern.search(text)
        if match:
            quote = re.sub(r"\s+", " ", _window_around_match(text, match, max_chars)).strip()
            score = quote_score(quote, answer)
            if score >= 0:
                candidates.append((score, quote))
    if not candidates:
        return None
    score, quote = max(candidates, key=lambda item: (item[0], -len(item[1]), item[1]))
    return quote, score


def evidence_token_set(quote: str, answer: str) -> set[str]:
    answer_tokens = set(normalized_tokens(answer))
    return {token for token in normalized_tokens(quote) if token not in answer_tokens}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def select_independent_evidence(
    candidates: list[dict[str, Any]],
    *,
    answer: str,
    max_items: int,
    max_jaccard: float,
) -> tuple[list[dict[str, Any]], int]:
    ordered = sorted(
        candidates,
        key=lambda row: (-float(row["score"]), str(row["domain"]), str(row["docid"])),
    )
    selected: list[dict[str, Any]] = []
    selected_sets: list[set[str]] = []
    rejected_near_duplicates = 0
    while ordered and len(selected) < max_items:
        used_domains = {str(row["domain"]) for row in selected}
        best_index: int | None = None
        best_key: tuple[Any, ...] | None = None
        for index, row in enumerate(ordered):
            tokens = evidence_token_set(str(row["quote"]), answer)
            similarity = max((jaccard(tokens, prior) for prior in selected_sets), default=0.0)
            if similarity > max_jaccard:
                continue
            key = (
                str(row["domain"]) not in used_domains,
                float(row["score"]),
                -similarity,
                str(row["docid"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None:
            rejected_near_duplicates += len(ordered)
            break
        row = ordered.pop(best_index)
        selected.append(row)
        selected_sets.append(evidence_token_set(str(row["quote"]), answer))
    return selected, rejected_near_duplicates


def signature_key(signatures: dict[str, list[str]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            normalized_text(value)
            for key in ("years", "numbers", "relations", "other_entities")
            for value in signatures.get(key, [])
            if normalized_text(value)
        )
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_docids = load_docids(Path(args.safe_docids))
    anchors = load_jsonl(Path(args.anchors))
    benchmark = load_benchmark_firewall(Path(args.benchmark_jsonl))
    if len(benchmark.query_ids) < args.required_benchmark_queries:
        raise ValueError(
            f"benchmark source has only {len(benchmark.query_ids)} unique query_ids; "
            f"required {args.required_benchmark_queries}. Do not use a pilot64 dump."
        )
    benchmark_answers = {" ".join(pattern) for pattern in benchmark.answer_patterns}
    gold_qrels = load_qrels(Path(args.qrel_golds))
    evidence_qrels = load_qrels(Path(args.qrel_evidence))
    positive_docids = (
        (set().union(*gold_qrels.values()) if gold_qrels else set())
        | (set().union(*evidence_qrels.values()) if evidence_qrels else set())
    )
    positive_allowlist_overlap = safe_docids & positive_docids
    if positive_allowlist_overlap:
        raise ValueError(
            f"safe allow-list contains {len(positive_allowlist_overlap)} qrel-positive documents"
        )

    rejection_counts: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for row in anchors:
        answer = str(row.get("answer") or "").strip()
        answer_key = normalized_text(answer)
        doc_count = int(row.get("mention_doc_count") or 0)
        domain_count = int(row.get("mention_domain_count") or 0)
        if not answer_key:
            rejection_counts["empty_answer"] += 1
        elif answer_key in benchmark_answers:
            rejection_counts["benchmark_answer_overlap"] += 1
        elif not args.min_source_doc_count <= doc_count <= args.max_source_doc_count:
            rejection_counts["source_doc_count"] += 1
        elif domain_count < args.min_source_domain_count:
            rejection_counts["source_domain_count"] += 1
        else:
            eligible.append(row)
    eligible.sort(key=lambda row: stable_hash(str(row["normalized_answer"]), seed=args.seed))
    selected = eligible[: args.anchor_sample_size]
    if len(selected) < args.anchor_sample_size:
        rejection_counts["sample_capacity_shortfall"] += args.anchor_sample_size - len(selected)

    requested_docids: set[str] = set()
    docid_to_anchor_indices: dict[str, list[int]] = defaultdict(list)
    for anchor_index, row in enumerate(selected):
        values = list(row.get("example_mention_docids") or [])
        values.extend(row.get("title_source_docids") or [])
        seen: set[str] = set()
        for value in values:
            docid = normalize_docid(value)
            if not docid or docid in seen or docid not in safe_docids:
                continue
            seen.add(docid)
            requested_docids.add(docid)
            docid_to_anchor_indices[docid].append(anchor_index)

    candidates_by_anchor: list[list[dict[str, Any]]] = [[] for _ in selected]
    found_requested_docids: set[str] = set()
    corpus_rows = 0
    for row in iter_corpus_rows(
        args.corpus,
        split=args.corpus_split,
        parquet_batch_size=args.parquet_batch_size,
        max_docs=args.max_docs,
    ):
        corpus_rows += 1
        if args.progress_every > 0 and corpus_rows % args.progress_every == 0:
            print(
                f"[bundle-scan] documents={corpus_rows} "
                f"requested_found={len(found_requested_docids)}/{len(requested_docids)}",
                flush=True,
            )
        docid, title, text, url = corpus_payload(row)
        if docid not in requested_docids or docid in found_requested_docids:
            continue
        found_requested_docids.add(docid)
        if docid in positive_docids:
            raise ValueError(f"qrel-positive document reached materialization: {docid}")
        for anchor_index in docid_to_anchor_indices[docid]:
            answer = str(selected[anchor_index]["answer"])
            evidence = best_evidence_quote(text, answer, args.max_quote_chars)
            if evidence is None:
                continue
            quote, score = evidence
            candidates_by_anchor[anchor_index].append(
                {
                    "docid": docid,
                    "url": url,
                    "domain": canonical_domain(url),
                    "title": title,
                    "quote": quote,
                    "score": score,
                    "signatures": evidence_signatures(quote, answer),
                }
            )

    bundles: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    near_duplicate_rejections = 0
    evidence_doc_histogram: Counter[int] = Counter()
    evidence_domain_histogram: Counter[int] = Counter()
    for anchor_index, row in enumerate(selected):
        answer = str(row["answer"])
        candidates = candidates_by_anchor[anchor_index]
        if len(candidates) < args.min_evidence_docs:
            reason = "insufficient_answer_bearing_docs"
            rejection_counts[reason] += 1
            rejected_rows.append({"anchor_id": row.get("anchor_id"), "reason": reason})
            continue
        evidence, rejected_duplicates = select_independent_evidence(
            candidates,
            answer=answer,
            max_items=args.max_evidence_docs,
            max_jaccard=args.max_evidence_jaccard,
        )
        near_duplicate_rejections += rejected_duplicates
        domains = {str(item["domain"]) for item in evidence if item.get("domain")}
        if len(evidence) < args.min_evidence_docs:
            reason = "insufficient_nonduplicate_evidence"
        elif len(domains) < args.min_evidence_domains:
            reason = "insufficient_evidence_domains"
        else:
            signatures = {signature_key(item["signatures"]) for item in evidence}
            signatures.discard(())
            if len(signatures) < args.min_signature_diversity:
                reason = "insufficient_signature_diversity"
            else:
                reason = ""
        if reason:
            rejection_counts[reason] += 1
            rejected_rows.append({"anchor_id": row.get("anchor_id"), "reason": reason})
            continue
        evidence_doc_histogram[len(evidence)] += 1
        evidence_domain_histogram[len(domains)] += 1
        bundles.append(
            {
                "bundle_id": f"bcp-safe-bundle-{len(bundles):05d}",
                "anchor_id": row.get("anchor_id"),
                "answer": answer,
                "normalized_answer": normalized_text(answer),
                "source_capacity": {
                    "mention_doc_count": row.get("mention_doc_count"),
                    "mention_domain_count": row.get("mention_domain_count"),
                },
                "evidence": evidence,
                "evidence_doc_count": len(evidence),
                "evidence_domain_count": len(domains),
                "signature_diversity": len(
                    {signature_key(item["signatures"]) for item in evidence} - {()}
                ),
                "benchmark_answer_overlap": False,
                "qrel_positive_overlap": False,
            }
        )

    bundles_path = output_dir / "grounded_anchor_bundles.jsonl"
    rejected_path = output_dir / "rejected_anchors.jsonl"
    write_jsonl(bundles_path, bundles)
    write_jsonl(rejected_path, rejected_rows)
    failures: list[str] = []
    if len(selected) < args.anchor_sample_size:
        failures.append(f"selected_anchors={len(selected)} < requested {args.anchor_sample_size}")
    if len(bundles) < args.required_output_bundles:
        failures.append(
            f"grounded_bundles={len(bundles)} < required {args.required_output_bundles}"
        )
    if len(found_requested_docids) != len(requested_docids):
        failures.append(
            f"requested_docs_found={len(found_requested_docids)} != requested {len(requested_docids)}"
        )
    report = {
        "schema": "browsecomp_safe_anchor_bundles_v1",
        "inputs": {
            "corpus": args.corpus,
            "safe_docids": str(Path(args.safe_docids)),
            "anchors": str(Path(args.anchors)),
            "benchmark_jsonl": str(Path(args.benchmark_jsonl)),
            "qrel_golds": str(Path(args.qrel_golds)),
            "qrel_evidence": str(Path(args.qrel_evidence)),
        },
        "policy": {
            "llm_calls": 0,
            "questions_generated": 0,
            "benchmark_answers_allowed_as_anchors": False,
            "qrel_positive_docs_allowed": False,
            "evidence_near_duplicate_metric": "answer-excluded token Jaccard",
            "max_evidence_jaccard": args.max_evidence_jaccard,
        },
        "counts": {
            "benchmark_query_ids": len(benchmark.query_ids),
            "benchmark_answer_patterns": len(benchmark.answer_patterns),
            "qrel_positive_docids": len(positive_docids),
            "positive_allowlist_overlap": len(positive_allowlist_overlap),
            "safe_docids": len(safe_docids),
            "input_anchors": len(anchors),
            "eligible_anchors": len(eligible),
            "selected_anchors": len(selected),
            "requested_docids": len(requested_docids),
            "found_requested_docids": len(found_requested_docids),
            "corpus_rows_scanned": corpus_rows,
            "grounded_bundles": len(bundles),
            "rejected_selected_anchors": len(rejected_rows),
            "near_duplicate_evidence_rejections": near_duplicate_rejections,
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "evidence_doc_count_distribution": dict(sorted(evidence_doc_histogram.items())),
        "evidence_domain_count_distribution": dict(sorted(evidence_domain_histogram.items())),
        "gate": {
            "passed": not failures,
            "failures": failures,
            "required_output_bundles": args.required_output_bundles,
        },
        "outputs": {
            "bundles": str(bundles_path),
            "rejected": str(rejected_path),
        },
    }
    report_path = output_dir / "bundle.report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.fail_on_gate and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--safe-docids", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--benchmark-jsonl", required=True)
    parser.add_argument("--required-benchmark-queries", type=int, default=830)
    parser.add_argument("--qrel-golds", required=True)
    parser.add_argument("--qrel-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchor-sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-source-doc-count", type=int, default=3)
    parser.add_argument("--max-source-doc-count", type=int, default=50)
    parser.add_argument("--min-source-domain-count", type=int, default=2)
    parser.add_argument("--min-evidence-docs", type=int, default=3)
    parser.add_argument("--max-evidence-docs", type=int, default=5)
    parser.add_argument("--min-evidence-domains", type=int, default=2)
    parser.add_argument("--min-signature-diversity", type=int, default=2)
    parser.add_argument("--max-evidence-jaccard", type=float, default=0.75)
    parser.add_argument("--max-quote-chars", type=int, default=1200)
    parser.add_argument("--required-output-bundles", type=int, default=500)
    parser.add_argument("--parquet-batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-docs", type=int, default=-1)
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
