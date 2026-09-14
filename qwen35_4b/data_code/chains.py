#!/usr/bin/env python3
"""Build grounded two-hop chain bundles from BrowseComp-safe anchor bundles.

The v1 safe-data pipeline groups several pages that all mention the final
answer.  That is useful for contamination-safe capacity, but it creates flat
intersection questions: every clue can retrieve the answer directly.  This
builder adds a real bridge layer without consulting BrowseComp questions,
answers, or qrels:

    bridge profile page -> hidden bridge entity -> answer-bearing edge page

For each named entity mentioned next to an anchor answer, the script looks for
a safe corpus page whose title identifies that entity.  A retained target has
at least two independent bridge/edge chains.  The downstream question writer
can therefore describe unnamed intermediate entities before asking for the
single target answer.

The script performs no LLM calls and never invents evidence.  It only consumes
the already-firewalled safe docid allow-list and grounded anchor bundles.
"""

from __future__ import annotations

import argparse
import hashlib
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
        normalize_docid,
        normalized_text,
        normalized_tokens,
    )
except ImportError:  # direct execution via ``python3 data_code/...``
    from corpus import (  # type: ignore
        canonical_domain,
        corpus_payload,
        iter_corpus_rows,
        normalize_docid,
        normalized_text,
        normalized_tokens,
    )


SCHEMA = "browsecomp_safe_chain_bundles_v2"
SPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
RELATION_RE = re.compile(
    r"\b(?:born|died|founded|established|created|launched|located|based|"
    r"headquartered|directed|written|authored|produced|owned|operated|"
    r"served|worked|married|won|awarded|released|published|renamed)\b",
    re.I,
)
BOILERPLATE_RE = re.compile(
    r"\b(?:cookie policy|privacy policy|all rights reserved|javascript is disabled|"
    r"subscribe now|sign in|log in|terms of service)\b",
    re.I,
)
GENERIC_ENTITY_KEYS = {
    "the",
    "this",
    "that",
    "for",
    "here",
    "home",
    "about",
    "contact",
    "read more",
    "learn more",
    "official website",
}


def norm_space(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "")).strip()


def stable_int(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_docids(path: Path) -> set[str]:
    return {
        docid
        for line in path.read_text(encoding="utf-8").splitlines()
        if (docid := normalize_docid(line))
    }


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _title_keys(title: str) -> set[str]:
    """Return conservative title aliases for exact bridge-page matching."""

    title = norm_space(title)
    variants = {title}
    for separator in (" - ", " | ", " — ", " – "):
        if separator in title:
            variants.add(title.split(separator, 1)[0])
    output = {normalized_text(value) for value in variants}
    return {value for value in output if value and value not in GENERIC_ENTITY_KEYS}


def _entity_aliases(entity: str) -> set[str]:
    """Create conservative aliases such as ``Jensen Huang`` from a role phrase."""

    key = normalized_text(entity)
    tokens = key.split()
    if not key or key in GENERIC_ENTITY_KEYS or len(key) < 4 or len(tokens) > 10:
        return set()
    aliases = {key}
    # Capitalized entity extraction sometimes captures a short role prefix,
    # e.g. "NVIDIA CEO Jensen Huang".  Two-to-five-token suffixes recover the
    # canonical page title while avoiding unsafe one-token surname matches.
    for width in range(2, min(5, len(tokens)) + 1):
        alias = " ".join(tokens[-width:])
        if len(alias) >= 6:
            aliases.add(alias)
    if tokens and tokens[0] in {"the", "a", "an"} and len(tokens) >= 2:
        aliases.add(" ".join(tokens[1:]))
    return {value for value in aliases if value not in GENERIC_ENTITY_KEYS}


def _entity_values(evidence: dict[str, Any]) -> list[str]:
    signatures = evidence.get("signatures")
    if not isinstance(signatures, dict):
        return []
    values = signatures.get("other_entities")
    if not isinstance(values, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        entity = norm_space(raw)
        key = normalized_text(entity)
        if not _entity_aliases(entity) or key in seen:
            continue
        seen.add(key)
        output.append(entity)
    return output


def _excerpt_score(sentence: str, title: str) -> float:
    tokens = normalized_tokens(sentence)
    if len(tokens) < 10 or BOILERPLATE_RE.search(sentence):
        return -1.0
    title_tokens = set(normalized_tokens(title))
    non_title = [token for token in tokens if token not in title_tokens]
    if len(non_title) < 7:
        return -1.0
    score = min(len(set(non_title)), 50) * 0.15
    score += 3.0 if YEAR_RE.search(sentence) else 0.0
    score += 1.0 if NUMBER_RE.search(sentence) else 0.0
    score += 3.0 if RELATION_RE.search(sentence) else 0.0
    score -= max(0, len(tokens) - 120) * 0.03
    return score


def bridge_excerpt(text: str, title: str, max_chars: int) -> str:
    """Select several informative sentences from a bridge profile page."""

    candidates: list[tuple[float, int, str]] = []
    for index, raw in enumerate(SENTENCE_SPLIT_RE.split(str(text or ""))):
        sentence = norm_space(raw)
        score = _excerpt_score(sentence, title)
        if score >= 0:
            candidates.append((score, index, sentence))
    if not candidates:
        return norm_space(text)[:max_chars]
    selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[:4]
    selected.sort(key=lambda item: item[1])
    excerpt = " ".join(item[2] for item in selected)
    return excerpt[:max_chars].rstrip()


def _edge_occurrences(
    bundles: Sequence[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]], Counter[str]]:
    by_alias: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entity_aliases: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for bundle_index, bundle in enumerate(bundles):
        answer = norm_space(bundle.get("answer"))
        answer_key = normalized_text(answer)
        bundle_id = norm_space(bundle.get("bundle_id") or bundle.get("anchor_id"))
        if not answer or not bundle_id:
            counts["missing_answer_or_bundle_id"] += 1
            continue
        seen_in_bundle: set[tuple[str, str]] = set()
        for evidence_index, raw in enumerate(_as_list(bundle.get("evidence"))):
            if not isinstance(raw, dict):
                continue
            edge_docid = normalize_docid(raw.get("docid"))
            quote = norm_space(raw.get("quote"))
            if not edge_docid or not quote:
                continue
            for entity in _entity_values(raw):
                entity_key = normalized_text(entity)
                if entity_key == answer_key:
                    continue
                identity = (edge_docid, entity_key)
                if identity in seen_in_bundle:
                    continue
                seen_in_bundle.add(identity)
                aliases = _entity_aliases(entity)
                entity_aliases[entity_key].update(aliases)
                occurrence = {
                    "bundle_index": bundle_index,
                    "bundle_id": bundle_id,
                    "answer": answer,
                    "answer_key": answer_key,
                    "entity": entity,
                    "entity_key": entity_key,
                    "edge_evidence_index": evidence_index,
                    "edge_docid": edge_docid,
                    "edge_url": norm_space(raw.get("url")),
                    "edge_domain": norm_space(raw.get("domain")),
                    "edge_title": norm_space(raw.get("title")),
                    "edge_quote": quote,
                    "edge_score": float(raw.get("score") or 0.0),
                    "edge_signatures": raw.get("signatures") or {},
                }
                for alias in aliases:
                    by_alias[alias].append(occurrence)
                counts["edge_entity_occurrences"] += 1
    counts["edge_entity_keys"] = len(entity_aliases)
    counts["edge_alias_keys"] = len(by_alias)
    return dict(by_alias), dict(entity_aliases), counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundles = read_jsonl(args.anchor_bundles)
    safe_docids = load_docids(args.safe_docids)
    by_alias, entity_aliases, counts = _edge_occurrences(bundles)

    bridge_docs_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_bridge_docids: set[str] = set()
    corpus_rows = 0
    title_alias_hits = 0
    for row in iter_corpus_rows(
        str(args.corpus),
        split=args.corpus_split,
        parquet_batch_size=args.parquet_batch_size,
        max_docs=args.max_docs,
    ):
        corpus_rows += 1
        if args.progress_every > 0 and corpus_rows % args.progress_every == 0:
            print(
                f"[chain-bundle-scan] documents={corpus_rows} "
                f"title_alias_hits={title_alias_hits}",
                flush=True,
            )
        docid, title, text, url = corpus_payload(row)
        if not docid or docid not in safe_docids or docid in seen_bridge_docids:
            continue
        matching_aliases = _title_keys(title).intersection(by_alias)
        if not matching_aliases:
            continue
        excerpt = bridge_excerpt(text, title, args.max_bridge_chars)
        if len(normalized_tokens(excerpt)) < args.min_bridge_words:
            continue
        seen_bridge_docids.add(docid)
        title_alias_hits += 1
        entity_keys: set[str] = set()
        for alias in matching_aliases:
            entity_keys.update(item["entity_key"] for item in by_alias[alias])
        for entity_key in entity_keys:
            bridge_docs_by_entity[entity_key].append(
                {
                    "docid": docid,
                    "title": title,
                    "url": url,
                    "domain": canonical_domain(url),
                    "excerpt": excerpt,
                }
            )

    occurrences_by_bundle: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for occurrences in by_alias.values():
        for occurrence in occurrences:
            if occurrence["edge_docid"] not in safe_docids:
                counts["unsafe_edge_occurrences_rejected"] += 1
                continue
            bridge_docs = bridge_docs_by_entity.get(occurrence["entity_key"], [])
            for bridge_doc in bridge_docs:
                if bridge_doc["docid"] == occurrence["edge_docid"]:
                    continue
                candidate = dict(occurrence)
                candidate["bridge_doc"] = bridge_doc
                occurrences_by_bundle[occurrence["bundle_index"]].append(candidate)

    # Other answers connected to the same bridge are genuine partial-match
    # distractors.  They are optional metadata, not a capacity gate.
    targets_by_bridge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in occurrences_by_bundle.values():
        for row in rows:
            targets_by_bridge[row["entity_key"]].append(row)

    output: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    chain_histogram: Counter[int] = Counter()
    distractor_histogram: Counter[int] = Counter()
    ordered_indices = sorted(
        range(len(bundles)),
        key=lambda index: stable_int(args.seed, bundles[index].get("bundle_id"), "chain"),
    )
    if args.max_target_bundles > 0:
        ordered_indices = ordered_indices[: args.max_target_bundles]
    for bundle_index in ordered_indices:
        bundle = bundles[bundle_index]
        answer_key = normalized_text(bundle.get("answer"))
        candidates = occurrences_by_bundle.get(bundle_index, [])
        deduplicated: dict[tuple[str, str, str], dict[str, Any]] = {}
        for candidate in candidates:
            key = (
                candidate["entity_key"],
                candidate["edge_docid"],
                candidate["bridge_doc"]["docid"],
            )
            deduplicated.setdefault(key, candidate)
        candidates = list(deduplicated.values())
        candidates.sort(
            key=lambda row: (
                row["edge_domain"] == row["bridge_doc"]["domain"],
                -row["edge_score"],
                stable_int(args.seed, row["bundle_id"], row["entity_key"]),
            )
        )
        selected: list[dict[str, Any]] = []
        used_entities: set[str] = set()
        used_edge_docids: set[str] = set()
        used_bridge_docids: set[str] = set()
        for candidate in candidates:
            if candidate["entity_key"] in used_entities:
                continue
            if candidate["edge_docid"] in used_edge_docids:
                continue
            if candidate["bridge_doc"]["docid"] in used_bridge_docids:
                continue
            used_entities.add(candidate["entity_key"])
            used_edge_docids.add(candidate["edge_docid"])
            used_bridge_docids.add(candidate["bridge_doc"]["docid"])
            selected.append(candidate)
            if len(selected) >= args.max_chains_per_bundle:
                break
        if len(selected) < args.min_chains_per_bundle:
            rejected.append(
                {
                    "bundle_id": bundle.get("bundle_id"),
                    "answer": bundle.get("answer"),
                    "reason": "insufficient_independent_bridge_chains",
                    "candidate_chains": len(selected),
                }
            )
            continue

        chains: list[dict[str, Any]] = []
        for index, candidate in enumerate(selected, start=1):
            distractors: list[dict[str, Any]] = []
            seen_answers: set[str] = {answer_key}
            for other in targets_by_bridge[candidate["entity_key"]]:
                other_answer_key = other["answer_key"]
                if other_answer_key in seen_answers:
                    continue
                seen_answers.add(other_answer_key)
                distractors.append(
                    {
                        "answer": other["answer"],
                        "edge_docid": other["edge_docid"],
                        "edge_title": other["edge_title"],
                        "edge_quote": other["edge_quote"],
                    }
                )
                if len(distractors) >= args.max_distractors_per_chain:
                    break
            chains.append(
                {
                    "chain_id": f"C{index}",
                    "bridge_entity": candidate["entity"],
                    "bridge_entity_key": candidate["entity_key"],
                    "bridge_doc": candidate["bridge_doc"],
                    "edge_doc": {
                        "docid": candidate["edge_docid"],
                        "title": candidate["edge_title"],
                        "url": candidate["edge_url"],
                        "domain": candidate["edge_domain"],
                        "quote": candidate["edge_quote"],
                        "score": candidate["edge_score"],
                        "signatures": candidate["edge_signatures"],
                    },
                    "partial_match_distractors": distractors,
                }
            )
        used_edge_docids = {chain["edge_doc"]["docid"] for chain in chains}
        verification_evidence: list[dict[str, Any]] = []
        for raw in _as_list(bundle.get("evidence")):
            if not isinstance(raw, dict):
                continue
            docid = normalize_docid(raw.get("docid"))
            if not docid or docid not in safe_docids or docid in used_edge_docids:
                continue
            quote = norm_space(raw.get("quote"))
            if not quote:
                continue
            verification_evidence.append(
                {
                    "docid": docid,
                    "title": norm_space(raw.get("title")),
                    "url": norm_space(raw.get("url")),
                    "domain": norm_space(raw.get("domain")),
                    "quote": quote,
                    "score": float(raw.get("score") or 0.0),
                    "signatures": raw.get("signatures") or {},
                }
            )
            if len(verification_evidence) >= 2:
                break
        chain_histogram[len(chains)] += 1
        distractor_count = sum(len(chain["partial_match_distractors"]) for chain in chains)
        distractor_histogram[distractor_count] += 1
        evidence_docids = sorted(
            {
                docid
                for chain in chains
                for docid in (chain["bridge_doc"]["docid"], chain["edge_doc"]["docid"])
            }
            | {row["docid"] for row in verification_evidence}
        )
        output.append(
            {
                "schema": SCHEMA,
                "chain_bundle_id": f"bcp-safe-chain-{len(output):05d}",
                "source_bundle_id": bundle.get("bundle_id"),
                "answer": bundle.get("answer"),
                "normalized_answer": answer_key,
                "chains": chains,
                "chain_count": len(chains),
                "verification_evidence": verification_evidence,
                "evidence_docids": evidence_docids,
                "benchmark_answer_overlap": bool(bundle.get("benchmark_answer_overlap", False)),
                "qrel_positive_overlap": bool(bundle.get("qrel_positive_overlap", False)),
            }
        )

    bundles_path = output_dir / "grounded_chain_bundles.jsonl"
    rejected_path = output_dir / "rejected_chain_bundles.jsonl"
    write_jsonl(bundles_path, output)
    write_jsonl(rejected_path, rejected)
    failures: list[str] = []
    if len(output) < args.required_output_bundles:
        failures.append(
            f"grounded_chain_bundles={len(output)} < required={args.required_output_bundles}"
        )
    unsafe_rows = sum(
        bool(row["benchmark_answer_overlap"] or row["qrel_positive_overlap"])
        for row in output
    )
    if unsafe_rows:
        failures.append(f"unsafe_source_bundles={unsafe_rows}")
    report = {
        "schema": SCHEMA,
        "inputs": {
            "corpus": str(args.corpus),
            "safe_docids": str(args.safe_docids),
            "anchor_bundles": str(args.anchor_bundles),
        },
        "policy": {
            "benchmark_metadata_used": False,
            "qrels_used": False,
            "bridge_match": "safe corpus title exact/conservative role-prefix suffix",
            "minimum_independent_chains": args.min_chains_per_bundle,
            "maximum_chains": args.max_chains_per_bundle,
            "partial_match_distractors_are_optional": True,
        },
        "counts": {
            "input_anchor_bundles": len(bundles),
            "safe_docids": len(safe_docids),
            "corpus_rows_scanned": corpus_rows,
            "title_alias_hits": title_alias_hits,
            "bridge_entity_keys_with_docs": len(bridge_docs_by_entity),
            "grounded_chain_bundles": len(output),
            "rejected_chain_bundles": len(rejected),
            **dict(sorted(counts.items())),
        },
        "chain_count_distribution": {str(k): v for k, v in sorted(chain_histogram.items())},
        "distractor_count_distribution": {
            str(k): v for k, v in sorted(distractor_histogram.items())
        },
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
    report_path = output_dir / "chain_bundle.report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.fail_on_gate and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--safe-docids", type=Path, required=True)
    parser.add_argument("--anchor-bundles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-chains-per-bundle", type=int, default=2)
    parser.add_argument("--max-chains-per-bundle", type=int, default=3)
    parser.add_argument("--max-distractors-per-chain", type=int, default=2)
    parser.add_argument("--max-bridge-chars", type=int, default=2400)
    parser.add_argument("--min-bridge-words", type=int, default=20)
    parser.add_argument("--max-target-bundles", type=int, default=-1)
    parser.add_argument("--required-output-bundles", type=int, default=1)
    parser.add_argument("--parquet-batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-docs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.min_chains_per_bundle < 2:
        raise SystemExit("--min-chains-per-bundle must be at least 2")
    if args.max_chains_per_bundle < args.min_chains_per_bundle:
        raise SystemExit("--max-chains-per-bundle must be >= --min-chains-per-bundle")
    if args.required_output_bundles <= 0:
        raise SystemExit("--required-output-bundles must be positive")
    run(args)


if __name__ == "__main__":
    main()
