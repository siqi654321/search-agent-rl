#!/usr/bin/env python3
"""Audit whether the BrowseComp-Plus corpus can support safe synthetic QA.

This is deliberately a *capacity audit*, not a data generator.  It builds a
one-way decontamination firewall, emits a safe document allow-list, and counts
stable page-title anchors that occur in multiple independent web documents.

The benchmark questions, answers, and qrels are used only to reject documents.
They are never written to the anchor table and never select a positive example.

Supported corpus inputs:

* a JSON/JSONL file;
* one or more Parquet files (glob patterns are accepted);
* a directory containing JSONL or Parquet shards;
* a Hugging Face dataset repository/path accepted by ``datasets.load_dataset``.

The audit makes two streaming passes over the corpus.  The first pass builds
the firewall and title vocabulary.  The second pass measures exact, token-level
title mentions across safe documents.  No LLM or embedding model is required.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import heapq
import json
import re
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TextIO
from urllib.parse import urlparse


WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
SPACE_RE = re.compile(r"\s+")
METADATA_FIELD_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}:\s*\S")
METADATA_PREFIXES = (
    "url:",
    "date:",
    "author:",
    "authors:",
    "by:",
    "layout:",
    "tags:",
    "tag:",
    "category:",
    "categories:",
    "summary:",
    "description:",
    "published:",
    "modified:",
    "draft:",
)
GENERIC_TITLES = {
    "home",
    "homepage",
    "home page",
    "index",
    "untitled",
    "welcome",
    "contact",
    "contact us",
    "about",
    "about us",
    "privacy policy",
    "terms of service",
    "page not found",
    "404 not found",
    "access denied",
    "login",
    "sign in",
    "search",
}
DOCID_KEYS = ("docid", "doc_id", "id", "_id")
TEXT_KEYS = ("text", "contents", "body", "document", "content")
URL_KEYS = ("url", "source_url", "link")
TITLE_KEYS = ("title", "name")
QUERY_ID_KEYS = ("query_id", "qid", "id", "_id")
QUESTION_KEYS = ("query", "question", "problem", "prompt")
ANSWER_KEYS = ("answer", "target", "gold_answer", "gts", "answers")


def normalize_docid(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("doc://"):
        text = text[6:]
    return text.rstrip(".,;:)]}")


def normalized_tokens(value: Any) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return tuple(WORD_RE.findall(text))


def normalized_text(value: Any) -> str:
    return " ".join(normalized_tokens(value))


def stable_hash(value: str, *, seed: int = 0) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def content_hash(tokens: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _first_present(row: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _stringify_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_stringify_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        for key in TEXT_KEYS:
            if key in value:
                return _stringify_text(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "")


def _looks_like_metadata(line: str) -> bool:
    lowered = line.casefold()
    return (
        lowered.startswith(METADATA_PREFIXES)
        or line in {"---", "..."}
        or line.startswith("http://")
        or line.startswith("https://")
        or bool(METADATA_FIELD_RE.match(line))
    )


def extract_title(row: dict[str, Any], text: str, url: str) -> str:
    explicit = _first_present(row, TITLE_KEYS)
    if explicit:
        title = SPACE_RE.sub(" ", str(explicit)).strip()
        if title:
            return title[:240]
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    for line in lines:
        if line.casefold().startswith("title:"):
            title = line.split(":", 1)[1].strip()
            if title:
                return title[:240]
    for line in lines:
        if not _looks_like_metadata(line):
            return line[:240]
    return url[:240] if url else "Untitled"


def canonical_domain(url: str) -> str:
    try:
        host = (urlparse(str(url or "")).hostname or "").casefold().strip(".")
    except ValueError:
        host = ""
    return host[4:] if host.startswith("www.") else host


def corpus_payload(row: dict[str, Any]) -> tuple[str, str, str, str]:
    docid = normalize_docid(_first_present(row, DOCID_KEYS))
    text = _stringify_text(_first_present(row, TEXT_KEYS))
    url = str(_first_present(row, URL_KEYS) or "").strip()
    title = extract_title(row, text, url)
    return docid, title, text, url


def _json_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                yield value
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        rows = value.get("data") or value.get("rows")
        if rows is None:
            yield value
            return
        value = rows
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON object/list in {path}")
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"Expected object rows in {path}")
        yield row


def _parquet_rows(paths: Sequence[Path], batch_size: int) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as error:  # pragma: no cover - depends on remote environment
        raise ModuleNotFoundError(
            "Reading Parquet requires pyarrow. Install pyarrow or point --corpus at JSONL."
        ) from error
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def _local_corpus_files(source: str) -> list[Path]:
    expanded = [Path(item) for item in sorted(glob.glob(source))]
    if not expanded and Path(source).exists():
        expanded = [Path(source)]
    files: list[Path] = []
    for path in expanded:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.parquet")))
            files.extend(sorted(path.rglob("*.jsonl")))
            files.extend(sorted(path.rglob("*.json")))
    # Some Hugging Face cache directories contain metadata JSON in addition to
    # data shards.  Prefer actual data shards when present.
    data_files = [path for path in files if path.suffix.casefold() in {".parquet", ".jsonl"}]
    return data_files or files


def iter_corpus_rows(
    source: str,
    *,
    split: str,
    parquet_batch_size: int,
    max_docs: int,
) -> Iterator[dict[str, Any]]:
    files = _local_corpus_files(source)
    emitted = 0

    def bounded(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        nonlocal emitted
        for row in rows:
            if max_docs > 0 and emitted >= max_docs:
                return
            emitted += 1
            yield row

    if files:
        parquet = [path for path in files if path.suffix.casefold() == ".parquet"]
        json_files = [path for path in files if path.suffix.casefold() in {".json", ".jsonl"}]
        if parquet and json_files:
            raise ValueError(
                f"Corpus source {source!r} contains both Parquet and JSON data files; "
                "point --corpus at one representation to avoid duplicate rows"
            )
        rows: Iterable[dict[str, Any]]
        if parquet:
            rows = _parquet_rows(parquet, parquet_batch_size)
        elif json_files:
            rows = (row for path in json_files for row in _json_rows(path))
        else:
            raise ValueError(f"No supported corpus files found under {source!r}")
        yield from bounded(rows)
        return

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as error:  # pragma: no cover - depends on remote environment
        raise ModuleNotFoundError(
            f"Corpus {source!r} is not local; loading it requires the datasets package"
        ) from error
    dataset = load_dataset(source, split=split)
    yield from bounded(dataset)


def load_qrels(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 4:
                raise ValueError(f"Malformed qrel line {path}:{line_number}: {line!r}")
            qid, _, docid, relevance = fields[:4]
            if float(relevance) > 0 and (normalized := normalize_docid(docid)):
                result[str(qid)].add(normalized)
    return dict(result)


def _flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        return [item for child in value for item in _flatten_strings(child)]
    if isinstance(value, dict):
        for key in ("target", "answer", "text", "value"):
            if key in value:
                return _flatten_strings(value[key])
    return []


def _docids_from_records(value: Any) -> set[str]:
    if value is None:
        return set()
    rows = value if isinstance(value, list) else [value]
    output: set[str] = set()
    for row in rows:
        raw = _first_present(row, DOCID_KEYS) if isinstance(row, dict) else row
        if docid := normalize_docid(raw):
            output.add(docid)
    return output


@dataclass(frozen=True)
class BenchmarkFirewall:
    query_ids: frozenset[str]
    question_hashes: frozenset[str]
    answer_patterns: tuple[tuple[str, ...], ...]
    body_scan_patterns: tuple[tuple[str, ...], ...]
    skipped_short_body_patterns: int
    gold_docids_from_rows: frozenset[str]
    evidence_docids_from_rows: frozenset[str]


def load_benchmark_firewall(path: Path) -> BenchmarkFirewall:
    query_ids: set[str] = set()
    question_hashes: set[str] = set()
    answer_patterns: set[tuple[str, ...]] = set()
    body_scan_patterns: set[tuple[str, ...]] = set()
    skipped_short_body_patterns = 0
    gold_docids: set[str] = set()
    evidence_docids: set[str] = set()
    for row in _json_rows(path):
        qid = str(_first_present(row, QUERY_ID_KEYS) or "").strip()
        if qid:
            query_ids.add(qid)
        question = _first_present(row, QUESTION_KEYS)
        if question:
            question_hashes.add(stable_hash(normalized_text(question)))
        answers: list[str] = []
        for key in ANSWER_KEYS:
            if key in row:
                answers.extend(_flatten_strings(row[key]))
        for answer in answers:
            tokens = normalized_tokens(answer)
            if not tokens:
                continue
            answer_patterns.add(tokens)
            # A one-character answer is meaningful for exact title equality,
            # but useless as a document-body collision pattern.
            if len(tokens) == 1 and len(tokens[0]) < 2:
                skipped_short_body_patterns += 1
            else:
                body_scan_patterns.add(tokens)
        gold_docids.update(_docids_from_records(row.get("gold_docs")))
        evidence_docids.update(_docids_from_records(row.get("evidence_docs")))
    return BenchmarkFirewall(
        query_ids=frozenset(query_ids),
        question_hashes=frozenset(question_hashes),
        answer_patterns=tuple(sorted(answer_patterns)),
        body_scan_patterns=tuple(sorted(body_scan_patterns)),
        skipped_short_body_patterns=skipped_short_body_patterns,
        gold_docids_from_rows=frozenset(gold_docids),
        evidence_docids_from_rows=frozenset(evidence_docids),
    )


class TokenAhoMatcher:
    """Small dependency-free Aho-Corasick matcher over normalized tokens."""

    def __init__(self, patterns: Sequence[Sequence[str]]):
        self.children: list[dict[str, int]] = [{}]
        self.fail: list[int] = [0]
        self.outputs: list[list[int]] = [[]]
        for pattern_id, pattern in enumerate(patterns):
            node = 0
            for token in pattern:
                child = self.children[node].get(token)
                if child is None:
                    child = len(self.children)
                    self.children[node][token] = child
                    self.children.append({})
                    self.fail.append(0)
                    self.outputs.append([])
                node = child
            self.outputs[node].append(pattern_id)
        queue: deque[int] = deque()
        for child in self.children[0].values():
            queue.append(child)
        while queue:
            node = queue.popleft()
            for token, child in self.children[node].items():
                queue.append(child)
                fallback = self.fail[node]
                while fallback and token not in self.children[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.children[fallback].get(token, 0)
                self.outputs[child].extend(self.outputs[self.fail[child]])

    def find_unique(self, tokens: Iterable[str]) -> set[int]:
        node = 0
        found: set[int] = set()
        for token in tokens:
            while node and token not in self.children[node]:
                node = self.fail[node]
            node = self.children[node].get(token, 0)
            found.update(self.outputs[node])
        return found


@dataclass
class DocumentMeta:
    docid: str
    title: str
    normalized_title: str
    url: str
    domain: str
    content_hash: str
    is_gold: bool
    is_evidence: bool
    title_target_exact: bool
    target_match_count: int


def anchor_rejection_reason(title: str, normalized: str) -> str | None:
    tokens = normalized.split()
    if not normalized:
        return "empty"
    if normalized in GENERIC_TITLES:
        return "generic"
    if len(title) < 4 or len(title) > 100:
        return "length"
    if not 1 <= len(tokens) <= 9:
        return "token_count"
    if len(tokens) == 1 and len(tokens[0]) < 5:
        return "short_single_token"
    if not any(character.isalpha() for character in title):
        return "no_letters"
    if title.count("|") >= 3 or title.count(">") >= 3:
        return "boilerplate"
    return None


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def _progress(stage: str, count: int, every: int) -> None:
    if every > 0 and count % every == 0:
        print(f"[{stage}] documents={count}", flush=True)


def _select_anchor_titles(
    records: Sequence[DocumentMeta],
    safe_docids: set[str],
    *,
    max_patterns: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    by_title: dict[str, dict[str, Any]] = {}
    rejections: Counter[str] = Counter()
    for record in records:
        if record.docid not in safe_docids:
            continue
        reason = anchor_rejection_reason(record.title, record.normalized_title)
        if reason:
            rejections[reason] += 1
            continue
        entry = by_title.setdefault(
            record.normalized_title,
            {
                "answer": record.title,
                "normalized_answer": record.normalized_title,
                "title_source_docids": [],
                "title_source_domains": set(),
            },
        )
        if len(entry["title_source_docids"]) < 8:
            entry["title_source_docids"].append(record.docid)
        if record.domain:
            entry["title_source_domains"].add(record.domain)
    unique_before_cap = len(by_title)
    entries = list(by_title.values())
    if max_patterns > 0 and len(entries) > max_patterns:
        # Bottom-k by a stable hash gives a deterministic, corpus-order-neutral
        # sample while keeping memory bounded for the token automaton.
        entries = heapq.nsmallest(
            max_patterns,
            entries,
            key=lambda item: stable_hash(item["normalized_answer"], seed=seed),
        )
    entries.sort(key=lambda item: item["normalized_answer"])
    return entries, rejections, unique_before_cap


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark_firewall(Path(args.benchmark_jsonl))
    if len(benchmark.query_ids) < args.required_benchmark_queries:
        raise ValueError(
            f"benchmark source has only {len(benchmark.query_ids)} unique query_ids; "
            f"required {args.required_benchmark_queries}. Do not use a pilot64 dump."
        )
    gold_qrels = load_qrels(Path(args.qrel_golds))
    evidence_qrels = load_qrels(Path(args.qrel_evidence))
    gold_docids = set().union(*gold_qrels.values()) if gold_qrels else set()
    evidence_docids = set().union(*evidence_qrels.values()) if evidence_qrels else set()
    gold_docids.update(benchmark.gold_docids_from_rows)
    evidence_docids.update(benchmark.evidence_docids_from_rows)
    positive_docids = gold_docids | evidence_docids

    benchmark_answer_set = set(benchmark.answer_patterns)
    target_matcher = TokenAhoMatcher(benchmark.body_scan_patterns)
    records: list[DocumentMeta] = []
    seen_docids: set[str] = set()
    duplicate_docids = 0
    missing_docids = 0
    truncated_scan_docs = 0
    first_pass_rows = 0
    print(
        f"[firewall] answers={len(benchmark.answer_patterns)} "
        f"body_patterns={len(benchmark.body_scan_patterns)} "
        f"gold_docids={len(gold_docids)} evidence_docids={len(evidence_docids)}",
        flush=True,
    )
    for row in iter_corpus_rows(
        args.corpus,
        split=args.corpus_split,
        parquet_batch_size=args.parquet_batch_size,
        max_docs=args.max_docs,
    ):
        first_pass_rows += 1
        _progress("firewall", first_pass_rows, args.progress_every)
        docid, title, text, url = corpus_payload(row)
        if not docid:
            missing_docids += 1
            continue
        if docid in seen_docids:
            duplicate_docids += 1
            continue
        seen_docids.add(docid)
        scanned_text = text
        if args.scan_max_chars > 0 and len(scanned_text) > args.scan_max_chars:
            scanned_text = scanned_text[: args.scan_max_chars]
            truncated_scan_docs += 1
        tokens = normalized_tokens(scanned_text)
        matches = target_matcher.find_unique(tokens)
        records.append(
            DocumentMeta(
                docid=docid,
                title=title,
                normalized_title=normalized_text(title),
                url=url,
                domain=canonical_domain(url),
                content_hash=content_hash(tokens),
                is_gold=docid in gold_docids,
                is_evidence=docid in evidence_docids,
                title_target_exact=normalized_tokens(title) in benchmark_answer_set,
                target_match_count=len(matches),
            )
        )

    positive_hashes = {
        record.content_hash for record in records if record.docid in positive_docids
    }
    blocked_rows: list[dict[str, Any]] = []
    safe_docids: set[str] = set()
    blocked_reason_counts: Counter[str] = Counter()
    primary_blocked_counts: Counter[str] = Counter()
    found_gold_docids = {record.docid for record in records if record.is_gold}
    found_evidence_docids = {record.docid for record in records if record.is_evidence}
    reason_order = (
        "gold_qrel",
        "evidence_qrel",
        "positive_normalized_exact_duplicate",
        "benchmark_target_exact_title",
        "benchmark_target_exact",
    )
    for record in records:
        reasons: list[str] = []
        if record.is_gold:
            reasons.append("gold_qrel")
        if record.is_evidence:
            reasons.append("evidence_qrel")
        if record.title_target_exact:
            reasons.append("benchmark_target_exact_title")
        if args.body_target_policy == "block" and record.target_match_count:
            reasons.append("benchmark_target_exact")
        if record.content_hash in positive_hashes and not (record.is_gold or record.is_evidence):
            reasons.append("positive_normalized_exact_duplicate")
        if reasons:
            blocked_reason_counts.update(reasons)
            primary = next(reason for reason in reason_order if reason in reasons)
            primary_blocked_counts[primary] += 1
            blocked_rows.append({"docid": record.docid, "reasons": reasons})
        else:
            safe_docids.add(record.docid)

    safe_docids_path = output_dir / "safe_docids.txt"
    safe_docids_path.write_text(
        "".join(f"{docid}\n" for docid in sorted(safe_docids)), encoding="utf-8"
    )
    write_jsonl(output_dir / "blocked_docids.jsonl", blocked_rows)

    anchors, anchor_rejections, unique_anchors_before_cap = _select_anchor_titles(
        records,
        safe_docids,
        max_patterns=args.max_anchor_patterns,
        seed=args.seed,
    )
    anchor_patterns = [tuple(item["normalized_answer"].split()) for item in anchors]
    anchor_matcher = TokenAhoMatcher(anchor_patterns)
    doc_counts = [0] * len(anchors)
    domains: list[set[str]] = [set() for _ in anchors]
    examples: list[list[str]] = [[] for _ in anchors]
    second_pass_rows = 0
    second_pass_safe_rows = 0
    safe_corpus_handle: TextIO | None = None
    safe_corpus_path = Path(args.write_safe_corpus).resolve() if args.write_safe_corpus else None
    if safe_corpus_path:
        safe_corpus_path.parent.mkdir(parents=True, exist_ok=True)
        safe_corpus_handle = safe_corpus_path.open("w", encoding="utf-8")
    try:
        seen_second_pass: set[str] = set()
        for row in iter_corpus_rows(
            args.corpus,
            split=args.corpus_split,
            parquet_batch_size=args.parquet_batch_size,
            max_docs=args.max_docs,
        ):
            second_pass_rows += 1
            _progress("anchor-scan", second_pass_rows, args.progress_every)
            docid, title, text, url = corpus_payload(row)
            if not docid or docid in seen_second_pass:
                continue
            seen_second_pass.add(docid)
            if docid not in safe_docids:
                continue
            second_pass_safe_rows += 1
            if safe_corpus_handle is not None:
                json.dump(
                    {"docid": docid, "text": text, "url": url},
                    safe_corpus_handle,
                    ensure_ascii=False,
                )
                safe_corpus_handle.write("\n")
            scanned_text = text[: args.scan_max_chars] if args.scan_max_chars > 0 else text
            matched = anchor_matcher.find_unique(normalized_tokens(scanned_text))
            domain = canonical_domain(url)
            for anchor_id in matched:
                doc_counts[anchor_id] += 1
                if domain:
                    domains[anchor_id].add(domain)
                if len(examples[anchor_id]) < args.max_anchor_examples:
                    examples[anchor_id].append(docid)
    finally:
        if safe_corpus_handle is not None:
            safe_corpus_handle.close()

    viable_rows: list[dict[str, Any]] = []
    anchor_capacity_counts: Counter[str] = Counter()
    for anchor_id, item in enumerate(anchors):
        docs = doc_counts[anchor_id]
        domain_count = len(domains[anchor_id])
        if docs >= 2:
            anchor_capacity_counts["document_ge_2"] += 1
        if docs >= 3:
            anchor_capacity_counts["document_ge_3"] += 1
        if docs >= 2 and domain_count >= 2:
            anchor_capacity_counts["document_ge_2_domain_ge_2"] += 1
        if docs >= 3 and domain_count >= 2:
            anchor_capacity_counts["document_ge_3_domain_ge_2"] += 1
        if docs > args.max_anchor_docs:
            anchor_capacity_counts["too_common_above_max_docs"] += 1
        if 2 <= docs <= args.max_anchor_docs and domain_count >= 2:
            anchor_capacity_counts["usable_document_ge_2_domain_ge_2"] += 1
        if 3 <= docs <= args.max_anchor_docs and domain_count >= 2:
            anchor_capacity_counts["usable_document_ge_3_domain_ge_2"] += 1
        if (
            docs < args.min_anchor_docs
            or docs > args.max_anchor_docs
            or domain_count < args.min_anchor_domains
        ):
            continue
        viable_rows.append(
            {
                "anchor_id": f"bcp-safe-anchor-{anchor_id:06d}",
                "answer": item["answer"],
                "normalized_answer": item["normalized_answer"],
                "mention_doc_count": docs,
                "mention_domain_count": domain_count,
                "title_source_docids": item["title_source_docids"],
                "example_mention_docids": examples[anchor_id],
                "capacity_tier": "three_doc" if docs >= 3 else "two_doc",
            }
        )
    viable_rows.sort(
        key=lambda row: (
            -int(row["mention_domain_count"]),
            -int(row["mention_doc_count"]),
            str(row["normalized_answer"]),
        )
    )
    anchors_path = output_dir / "anchor_candidates.jsonl"
    write_jsonl(anchors_path, viable_rows)

    capacity_failures: list[str] = []
    if len(safe_docids) < args.required_safe_docs:
        capacity_failures.append(
            f"safe_docs={len(safe_docids)} < required {args.required_safe_docs}"
        )
    two_doc_domains = anchor_capacity_counts["usable_document_ge_2_domain_ge_2"]
    if two_doc_domains < args.required_two_doc_anchors:
        capacity_failures.append(
            f"two_doc_multi_domain_anchors={two_doc_domains} < required "
            f"{args.required_two_doc_anchors}"
        )
    three_doc_domains = anchor_capacity_counts["usable_document_ge_3_domain_ge_2"]
    if three_doc_domains < args.required_three_doc_anchors:
        capacity_failures.append(
            f"three_doc_multi_domain_anchors={three_doc_domains} < required "
            f"{args.required_three_doc_anchors}"
        )

    contamination_failures: list[str] = []
    missing_gold = sorted(gold_docids - found_gold_docids)
    missing_evidence = sorted(evidence_docids - found_evidence_docids)
    if missing_gold:
        contamination_failures.append(f"missing_gold_qrel_docids={len(missing_gold)}")
    if missing_evidence:
        contamination_failures.append(f"missing_evidence_qrel_docids={len(missing_evidence)}")
    if not benchmark.answer_patterns:
        contamination_failures.append("no_benchmark_answer_patterns_loaded")
    if args.body_target_policy == "block" and benchmark.skipped_short_body_patterns:
        contamination_failures.append(
            f"unfilterable_short_answers={benchmark.skipped_short_body_patterns}"
        )
    if truncated_scan_docs:
        contamination_failures.append(
            f"truncated_contamination_scans={truncated_scan_docs}; rerun with --scan-max-chars=0"
        )
    if second_pass_safe_rows != len(safe_docids):
        contamination_failures.append(
            f"safe_second_pass_rows={second_pass_safe_rows} != safe_docids={len(safe_docids)}"
        )

    report = {
        "schema": "browsecomp_safe_corpus_capacity_v2",
        "inputs": {
            "corpus": args.corpus,
            "corpus_split": args.corpus_split,
            "benchmark_jsonl": str(Path(args.benchmark_jsonl)),
            "qrel_golds": str(Path(args.qrel_golds)),
            "qrel_evidence": str(Path(args.qrel_evidence)),
        },
        "policy": {
            "benchmark_metadata_usage": "one_way_exclusion_only",
            "generator_receives_benchmark_text": False,
            "qrels_or_targets_select_positive_examples": False,
            "semantic_near_duplicate_filter_implemented": False,
            "normalized_exact_positive_duplicate_filter_implemented": True,
            "body_target_policy": args.body_target_policy,
            "body_target_matches_are_report_only": args.body_target_policy == "report",
            "exact_target_title_is_always_blocked": True,
            "partial_match_negative_capacity_measured": False,
            "capacity_interpretation": "title-anchor lower-bound, not final QA yield",
        },
        "benchmark": {
            "query_ids": len(benchmark.query_ids),
            "question_fingerprints": len(benchmark.question_hashes),
            "answer_patterns": len(benchmark.answer_patterns),
            "body_scan_patterns": len(benchmark.body_scan_patterns),
            "skipped_short_body_patterns": benchmark.skipped_short_body_patterns,
            "required_query_ids": args.required_benchmark_queries,
            "gold_docids": len(gold_docids),
            "evidence_docids": len(evidence_docids),
            "positive_docids_union": len(positive_docids),
        },
        "corpus": {
            "first_pass_rows": first_pass_rows,
            "unique_docids": len(records),
            "missing_docids": missing_docids,
            "duplicate_docids": duplicate_docids,
            "truncated_scan_docs": truncated_scan_docs,
            "found_gold_docids": len(found_gold_docids),
            "found_evidence_docids": len(found_evidence_docids),
            "missing_gold_docids": len(missing_gold),
            "missing_evidence_docids": len(missing_evidence),
            "safe_docs": len(safe_docids),
            "safe_fraction": len(safe_docids) / len(records) if records else 0.0,
            "strict_no_body_target_safe_docs": sum(
                record.docid in safe_docids and record.target_match_count == 0
                for record in records
            ),
            "body_target_match_docs": sum(record.target_match_count > 0 for record in records),
            "safe_docs_with_body_target_match": sum(
                record.docid in safe_docids and record.target_match_count > 0
                for record in records
            ),
            "exact_target_title_docs": sum(record.title_target_exact for record in records),
            "blocked_docs": len(blocked_rows),
            "blocked_reason_counts": dict(sorted(blocked_reason_counts.items())),
            "primary_blocked_reason_counts": dict(sorted(primary_blocked_counts.items())),
        },
        "anchors": {
            "unique_quality_titles_before_cap": unique_anchors_before_cap,
            "patterns_scanned": len(anchors),
            "selection_cap": args.max_anchor_patterns,
            "title_rejection_counts": dict(sorted(anchor_rejections.items())),
            "capacity_counts": dict(sorted(anchor_capacity_counts.items())),
            "viable_output_rows": len(viable_rows),
            "minimum_output_docs": args.min_anchor_docs,
            "maximum_output_docs": args.max_anchor_docs,
            "minimum_output_domains": args.min_anchor_domains,
        },
        "gates": {
            "capacity": {
                "passed": not capacity_failures,
                "failures": capacity_failures,
                "requirements": {
                    "safe_docs": args.required_safe_docs,
                    "two_doc_multi_domain_anchors": args.required_two_doc_anchors,
                    "three_doc_multi_domain_anchors": args.required_three_doc_anchors,
                },
            },
            "contamination_firewall": {
                "passed": not contamination_failures,
                "failures": contamination_failures,
                "level": (
                    "qrel+target-body+target-title+normalized-exact-duplicate_v2"
                    if args.body_target_policy == "block"
                    else "qrel+target-title+normalized-exact-duplicate_v2"
                ),
            },
            "overall_passed": not capacity_failures and not contamination_failures,
        },
        "outputs": {
            "safe_docids": str(safe_docids_path),
            "blocked_docids": str(output_dir / "blocked_docids.jsonl"),
            "anchor_candidates": str(anchors_path),
            "safe_corpus": str(safe_corpus_path) if safe_corpus_path else None,
        },
    }
    report_path = output_dir / "capacity.report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.fail_on_gate and not report["gates"]["overall_passed"]:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="Corpus file, directory, glob, or HF dataset ID")
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--benchmark-jsonl", required=True, help="Decrypted benchmark JSONL")
    parser.add_argument(
        "--required-benchmark-queries",
        type=int,
        default=830,
        help="Fail before corpus scanning if a partial evaluation dump is supplied",
    )
    parser.add_argument("--qrel-golds", required=True)
    parser.add_argument("--qrel-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-docs", type=int, default=-1)
    parser.add_argument(
        "--scan-max-chars",
        type=int,
        default=0,
        help="0 scans full documents; positive values make a non-clean diagnostic run",
    )
    parser.add_argument("--parquet-batch-size", type=int, default=256)
    parser.add_argument(
        "--body-target-policy",
        choices=("report", "block"),
        default="report",
        help=(
            "report keeps non-qrel pages that merely mention a benchmark answer; "
            "block reproduces the intentionally over-conservative v1 lower bound"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-anchor-patterns", type=int, default=50000)
    parser.add_argument("--max-anchor-examples", type=int, default=8)
    parser.add_argument("--min-anchor-docs", type=int, default=2)
    parser.add_argument(
        "--max-anchor-docs",
        type=int,
        default=100,
        help="Reject overly common title anchors above this document frequency",
    )
    parser.add_argument("--min-anchor-domains", type=int, default=2)
    parser.add_argument("--required-safe-docs", type=int, default=70000)
    parser.add_argument("--required-two-doc-anchors", type=int, default=5000)
    parser.add_argument("--required-three-doc-anchors", type=int, default=2000)
    parser.add_argument(
        "--write-safe-corpus",
        default="",
        help="Optional JSONL output. Omit it for the low-disk audit.",
    )
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_audit(args)


if __name__ == "__main__":
    main()
