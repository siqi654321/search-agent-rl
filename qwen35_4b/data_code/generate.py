#!/usr/bin/env python3
"""Generate long-chain QA from grounded BrowseComp-safe bridge bundles.

Unlike ``qa_common.py`` v1, this generator does not
turn several answer-bearing snippets into a flat clue intersection.  Every
question is backed by two unnamed bridge entities.  Each bridge has a profile
page and a separate answer-bearing edge page, so solving the question requires
at least four grounded evidence roles.  A fifth independent verification clue
is used by the 5-constraint variant when available.

The generator never receives BrowseComp benchmark questions, answers, qrels,
or positive documents.  Runtime behavior is still calibrated by a later n=8
strict rollout; generation-time quality labels are deliberately soft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence

import pandas as pd

try:
    from data_code import qa_common as base
except ImportError:  # direct execution via ``python3 data_code/...``
    import qa_common as base  # type: ignore


ATOMIC_FACT_SCHEMA = "browsecomp_safe_atomic_fact_chain_qa_v4"
WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
MULTI_DELIVERABLE_RE = re.compile(
    r"\b(?:what|which|who|when|where)\b[^?]{0,100}\band\s+"
    r"(?:what|which|who|when|where)\b|\b(?:provide|name|identify)\s+(?:both|two)\b",
    re.I,
)
UI_ANSWER_RE = re.compile(
    r"^(?:comment|comments|cancel|reply|submit|menu|home|about|contact|"
    r"read more|learn more|click here|sign in|log in|login|register|"
    r"privacy policy|cookie policy|terms(?: of (?:use|service))?|"
    r"next|previous|back|share|search|subscribe|download)$|"
    r"\b(?:leave a comment|comment cancel reply|all rights reserved|"
    r"javascript is disabled)\b",
    re.I,
)
PLACEHOLDER_RE = re.compile(r"\[(?:shared entity|intermediate|final target)[^\]]*\]", re.I)
SYNTHETIC_META_RE = re.compile(
    r"\b(?:"
    r"terminal\s+(?:passage|page|article|account|record|source)|"
    r"shared\s+(?:entity|target|subject)|"
    r"hidden\s+(?:answer|entity|subject|target)|"
    r"both\s+(?:chains|clues|branches)|"
    r"(?:first|second)\s+(?:chain|clue|unnamed\s+(?:entity|person|subject))|"
    r"independent\s+(?:terminal\s+)?(?:passage|page|account|record|source)|"
    r"the\s+two\s+(?:chains|clues|figures)\s+(?:point|lead|link|refer)|"
    r"(?:after|once)\s+(?:identifying|finding|locating|determining|"
    r"establishing|cross-checking)"
    r")\b",
    re.I,
)
ATOMIC_ANSWER_TYPES = {"person", "organization", "work", "event", "place"}
ATOMIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
DATE_LIKE_ANSWER_RE = re.compile(
    r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})$"
)


def norm_space(value: Any) -> str:
    return base.norm_space(value)


def norm_key(value: Any) -> str:
    return base.norm_key(value)


def stable_int(*parts: Any) -> int:
    return base.stable_int(*parts)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return []


def _word_tokens(value: Any) -> list[str]:
    return WORD_RE.findall(norm_space(value))


def _longest_verbatim_run(question: str, source_texts: Sequence[str]) -> int:
    """Return the longest normalized contiguous token run copied from evidence."""

    question_tokens = norm_key(question).split()
    if not question_tokens:
        return 0
    best = 0
    for source_text in source_texts:
        source_tokens = norm_key(source_text).split()
        if not source_tokens:
            continue
        previous = [0] * (len(source_tokens) + 1)
        for question_token in question_tokens:
            current = [0] * (len(source_tokens) + 1)
            for index, source_token in enumerate(source_tokens, start=1):
                if question_token == source_token:
                    current[index] = previous[index - 1] + 1
                    best = max(best, current[index])
            previous = current
    return best


def _atomic_answer_rejection(answer: str, answer_type: str) -> str:
    """Keep V4 rewards close to binary by accepting proper-name-like targets."""

    normalized_type = norm_space(answer_type).casefold()
    if normalized_type not in ATOMIC_ANSWER_TYPES:
        return "atomic_answer_type"
    tokens = _word_tokens(answer)
    if not 1 <= len(tokens) <= 8:
        return "atomic_answer_length"
    if DATE_LIKE_ANSWER_RE.fullmatch(answer.strip()) or answer.strip().isdigit():
        return "atomic_answer_date_or_number"
    if any(mark in answer for mark in (";", "\n", "\r", "•")):
        return "atomic_answer_list"
    if normalized_type != "place" and answer.count(",") >= 1:
        return "atomic_answer_list"
    if normalized_type == "person" and re.search(r"\b(?:and|or)\b", answer, re.I):
        return "atomic_answer_list"

    content_tokens = [token for token in tokens if token.casefold() not in ATOMIC_STOPWORDS]
    if not content_tokens:
        return "atomic_answer_not_named"
    named_tokens = sum(
        1
        for token in content_tokens
        if token[:1].isupper() or (len(token) >= 2 and token.isupper())
    )
    # This rejects sentence fragments such as "in both French and English" and
    # "a pot of boiling coffee to the face", while retaining ordinary titles
    # such as "Brothers in Arms".
    if answer[:1].islower() or named_tokens / len(content_tokens) < 0.5:
        return "atomic_answer_not_named"
    return ""


class LongChainSpec(NamedTuple):
    candidate_id: str
    chain_bundle_id: str
    source_bundle_id: str
    variant_index: int
    constraint_count: int
    answer: str
    chains: tuple[dict[str, Any], ...]
    verification: tuple[dict[str, Any], ...]
    answer_mode: str


def _valid_chain(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    chain_id = norm_space(raw.get("chain_id")).upper()
    bridge_entity = norm_space(raw.get("bridge_entity"))
    bridge_doc = raw.get("bridge_doc")
    edge_doc = raw.get("edge_doc")
    if not chain_id or not bridge_entity or not isinstance(bridge_doc, dict) or not isinstance(edge_doc, dict):
        return None
    if not norm_space(bridge_doc.get("docid")) or not norm_space(bridge_doc.get("excerpt")):
        return None
    if not norm_space(edge_doc.get("docid")) or not norm_space(edge_doc.get("quote")):
        return None
    row = dict(raw)
    row["chain_id"] = chain_id
    row["bridge_entity"] = bridge_entity
    row["bridge_doc"] = dict(bridge_doc)
    row["edge_doc"] = dict(edge_doc)
    return row


def _valid_verification(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    if not norm_space(raw.get("docid")) or not norm_space(raw.get("quote")):
        return None
    return dict(raw)


def parse_constraint_pattern(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item not in {4, 5} for item in values):
        raise argparse.ArgumentTypeError("constraint pattern must contain only 4 and 5")
    return values


def build_specs(
    bundles: Sequence[dict[str, Any]],
    *,
    variants_per_bundle: int,
    constraint_pattern: Sequence[int],
    seed: int,
    max_questions: int,
    answer_mode: str = "terminal_atomic_fact",
) -> tuple[list[LongChainSpec], Counter[str]]:
    if answer_mode != "terminal_atomic_fact":
        raise ValueError("This package only generates terminal_atomic_fact questions")
    counts: Counter[str] = Counter()
    specs: list[LongChainSpec] = []
    ordered = sorted(
        bundles,
        key=lambda row: stable_int(seed, row.get("chain_bundle_id"), row.get("answer")),
    )
    for bundle in ordered:
        answer = norm_space(bundle.get("answer"))
        chain_bundle_id = norm_space(bundle.get("chain_bundle_id"))
        source_bundle_id = norm_space(bundle.get("source_bundle_id"))
        chains = [row for raw in _as_list(bundle.get("chains")) if (row := _valid_chain(raw))]
        verification = [
            row
            for raw in _as_list(bundle.get("verification_evidence"))
            if (row := _valid_verification(raw))
        ]
        if not answer or not chain_bundle_id:
            counts["missing_answer_or_bundle_id"] += 1
            continue
        if len(chains) < 2:
            counts["fewer_than_two_chains"] += 1
            continue
        for variant_index in range(variants_per_bundle):
            constraint_count = int(constraint_pattern[variant_index % len(constraint_pattern)])
            if not verification:
                counts["missing_fifth_verification"] += 1
                continue
            chain_order = sorted(
                chains,
                key=lambda row: stable_int(
                    seed, chain_bundle_id, variant_index, row["chain_id"], "chain-order"
                ),
            )
            selected_chains = tuple(chain_order[:2])
            verification_order = sorted(
                verification,
                key=lambda row: stable_int(
                    seed, chain_bundle_id, variant_index, row.get("docid"), "verification"
                ),
            )
            selected_verification = (verification_order[0],)
            candidate_prefix = "bcp-safe-atomic-fact-chain-"
            identity = "terminal-atomic-fact-v4"
            candidate_payload = (
                f"{seed}\x1f{identity}\x1f{chain_bundle_id}\x1f"
                f"{variant_index}\x1f{constraint_count}"
            )
            candidate_id = candidate_prefix + hashlib.sha256(
                candidate_payload.encode("utf-8")
            ).hexdigest()[:16]
            specs.append(
                LongChainSpec(
                    candidate_id=candidate_id,
                    chain_bundle_id=chain_bundle_id,
                    source_bundle_id=source_bundle_id,
                    variant_index=variant_index,
                    constraint_count=constraint_count,
                    answer=answer,
                    chains=selected_chains,
                    verification=selected_verification,
                    answer_mode=answer_mode,
                )
            )
    if max_questions > 0:
        specs = specs[:max_questions]
    counts["candidate_specs"] = len(specs)
    return specs, counts


def _mask(text: str, phrase: str, replacement: str) -> str:
    original = norm_space(text)
    masked = base.mask_answer(original, phrase)
    return masked.replace("[HIDDEN ANSWER]", replacement)


def _masked_chain_payload(spec: LongChainSpec) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for position, chain in enumerate(spec.chains, start=1):
        chain_id = f"C{position}"
        bridge = chain["bridge_entity"]
        bridge_doc = chain["bridge_doc"]
        edge_doc = chain["edge_doc"]
        bridge_profile = _mask(bridge_doc.get("excerpt", ""), spec.answer, "[FINAL TARGET]")
        bridge_profile = _mask(bridge_profile, bridge, f"[INTERMEDIATE {chain_id}]")
        bridge_title = _mask(bridge_doc.get("title", ""), spec.answer, "[FINAL TARGET]")
        bridge_title = _mask(bridge_title, bridge, f"[INTERMEDIATE {chain_id}]")
        edge_quote = _mask(edge_doc.get("quote", ""), spec.answer, "[FINAL TARGET]")
        edge_quote = _mask(edge_quote, bridge, f"[INTERMEDIATE {chain_id}]")
        edge_title = _mask(edge_doc.get("title", ""), spec.answer, "[FINAL TARGET]")
        edge_title = _mask(edge_title, bridge, f"[INTERMEDIATE {chain_id}]")
        distractors: list[dict[str, str]] = []
        for raw in _as_list(chain.get("partial_match_distractors"))[:2]:
            if not isinstance(raw, dict):
                continue
            quote = _mask(raw.get("edge_quote", ""), spec.answer, "[FINAL TARGET]")
            quote = _mask(quote, bridge, f"[INTERMEDIATE {chain_id}]")
            quote = _mask(quote, norm_space(raw.get("answer")), "[PARTIAL MATCH]")
            distractors.append(
                {
                    "page_title": _mask(raw.get("edge_title", ""), raw.get("answer", ""), "[PARTIAL MATCH]"),
                    "passage": quote,
                }
            )
        payload.append(
            {
                "chain_id": chain_id,
                "bridge_profile": {
                    "page_title": bridge_title,
                    "passage": bridge_profile,
                },
                "bridge_to_target_edge": {
                    "page_title": edge_title,
                    "passage": edge_quote,
                },
                "optional_partial_match_edges": distractors,
            }
        )
    return payload


def _fact_terminal_payload(spec: LongChainSpec) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    bridge_names = [chain["bridge_entity"] for chain in spec.chains]
    for index, row in enumerate(spec.verification, start=1):
        title = _mask(row.get("title", ""), spec.answer, "[SHARED ENTITY]")
        passage = _mask(row.get("quote", ""), spec.answer, "[SHARED ENTITY]")
        for bridge_index, bridge in enumerate(bridge_names, start=1):
            title = _mask(title, bridge, f"[INTERMEDIATE C{bridge_index}]")
            passage = _mask(passage, bridge, f"[INTERMEDIATE C{bridge_index}]")
        payload.append(
            {
                "verification_id": f"V{index}",
                "page_title": title,
                "passage": passage,
            }
        )
    return payload


def _build_terminal_atomic_fact_prompt(spec: LongChainSpec) -> str:
    """V4 prompt: natural surface form plus a proper-name terminal target."""

    chain_payload = json.loads(
        json.dumps(_masked_chain_payload(spec), ensure_ascii=False).replace(
            "[FINAL TARGET]", "[SHARED ENTITY]"
        )
    )
    terminal_payload = [
        {
            "verification_id": row["verification_id"],
            # Deliberately omit the terminal page title.  V3 often copied or
            # paraphrased that title into the question, creating a direct
            # lookup shortcut that bypassed the two bridge chains.
            "body_passage": row["passage"],
        }
        for row in _fact_terminal_payload(spec)
    ]
    # Variant zero is intentionally byte-for-byte compatible with the V4
    # pilot cache.  Additional full-scale variants receive a soft target slot
    # so repeated calls over one evidence bundle do not all select the same
    # first salient name.  It remains a preference: grounding/type checks are
    # unchanged and the model may fall back when the passage lacks that type.
    diversity_instruction = ""
    if spec.variant_index > 0:
        type_preferences = ("person", "organization", "work", "event", "place")
        position_preferences = ("early", "middle", "late", "any")
        preferred_type = type_preferences[(spec.variant_index - 1) % len(type_preferences)]
        preferred_position = position_preferences[
            ((spec.variant_index - 1) // len(type_preferences))
            % len(position_preferences)
        ]
        diversity_instruction = f"""

Diversity slot {spec.variant_index}: when several valid atomic answers exist,
prefer a {preferred_type} name from the {preferred_position} portion of the
final body evidence.  If no such valid name exists, select another allowed
proper-name type.  This is an internal generation preference; never mention
the slot, preferred type, or passage position in the question."""
    return f"""You write natural deep-research questions grounded in a local web corpus.

Construct ONE English question of 70 to 140 words.  A solver must first infer
two different people, works, organizations, or places from the two profile
descriptions.  Separate relationship evidence connects each inferred item to
the same hidden subject.  Only after recovering that subject can the solver
look up one proper-name fact in the final body evidence.

Select the answer yourself from the final body evidence.  It must:
- be one atomic proper name of type person, organization, work, event, or place;
- contain 1 to 8 words and appear verbatim in the final body evidence;
- not be the hidden subject, an intermediate name, a page title, a date,
  a number, a quotation fragment, a list, or generic page/navigation text;
- not occur in any profile or relationship evidence.

Question-writing rules:
- Ask for exactly that one proper name.
- Do not reveal either inferred item or the hidden subject.
- Express the final relation naturally, without referring to a page, passage,
  article, account, record, source, shared entity, chain, clue, branch,
  document, placeholder, or research procedure.
- Do not say "unnamed", "both clues", "both chains", "shared entity",
  "terminal passage", "after identifying", or "once located".
- Do not copy any sequence of 8 or more words from the supplied evidence.
  Preserve factual meaning while varying syntax and wording.
- Keep both routes factually necessary.  Do not introduce a title, quotation,
  or distinctive phrase that lets the final body evidence be found directly
  from the full question.
- Use only supplied facts and write one fluent question, not numbered steps.

Profile and relationship evidence:
{json.dumps(chain_payload, ensure_ascii=False, indent=2)}

Final body evidence (its page title is intentionally withheld):
{json.dumps(terminal_payload, ensure_ascii=False, indent=2)}{diversity_instruction}

Return exactly one JSON object:
{{
  "question": "one natural question ending in exactly one question mark",
  "answer": "one exact proper name from V1",
  "answer_type": "person|organization|work|event|place",
  "answer_is_named_entity": true,
  "answer_evidence_quote": "a short exact span from V1 containing the answer",
  "single_answer": true,
  "used_chain_ids": ["C1", "C2"],
  "used_verification_ids": ["V1"],
  "chain_clues": [
    {{
      "chain_id": "C1",
      "bridge_description": "the natural description used in the question",
      "edge_relation": "how this item connects to the hidden subject",
      "grounded_terms": ["short supported terms that occur in the question"]
    }},
    {{
      "chain_id": "C2",
      "bridge_description": "the natural description used in the question",
      "edge_relation": "how this item connects to the hidden subject",
      "grounded_terms": ["short supported terms that occur in the question"]
    }}
  ],
  "retrieval_plan": [
    "resolve one profile",
    "follow its relationship",
    "resolve and cross-check the other profile",
    "retrieve the requested proper-name fact"
  ]
}}
Do not return commentary outside the JSON object."""


def build_prompt(spec: LongChainSpec) -> str:
    return _build_terminal_atomic_fact_prompt(spec)


def _normalized_terms(value: Any) -> list[str]:
    terms: list[str] = []
    for raw in _as_list(value):
        term = norm_space(raw)
        if len(norm_key(term)) >= 3:
            terms.append(term)
    return terms


def _terminal_fact(
    spec: LongChainSpec, parsed: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    """Validate a terminal answer as a non-title fact unique to terminal evidence."""

    if len(spec.verification) != 1:
        return None, "terminal_evidence_count"
    answer = norm_space(parsed.get("answer"))
    answer_words = WORD_RE.findall(answer)
    if not answer or not 1 <= len(answer_words) <= 12 or len(answer) > 160:
        return None, "terminal_answer_length"
    if PLACEHOLDER_RE.search(answer) or UI_ANSWER_RE.search(answer):
        return None, "terminal_answer_ui_or_placeholder"
    answer_type = norm_space(parsed.get("answer_type") or "other").casefold()
    atomic_reason = _atomic_answer_rejection(answer, answer_type)
    if atomic_reason:
        return None, atomic_reason
    if parsed.get("answer_is_named_entity") is not True:
        return None, "atomic_answer_not_confirmed"

    forbidden_entities = [spec.answer] + [chain["bridge_entity"] for chain in spec.chains]
    for entity in forbidden_entities:
        if base.contains_answer(answer, entity) or base.contains_answer(entity, answer):
            return None, "terminal_answer_is_hidden_entity"

    terminal = spec.verification[0]
    terminal_quote = norm_space(terminal.get("quote"))
    if not base.contains_answer(terminal_quote, answer):
        return None, "terminal_answer_not_in_body"

    all_titles = [norm_space(terminal.get("title"))]
    all_titles.extend(norm_space(chain["bridge_doc"].get("title")) for chain in spec.chains)
    all_titles.extend(norm_space(chain["edge_doc"].get("title")) for chain in spec.chains)
    if any(base.contains_answer(title, answer) for title in all_titles if title):
        return None, "terminal_answer_appears_in_title"

    nonterminal_text = " ".join(
        [norm_space(chain["bridge_doc"].get("excerpt")) for chain in spec.chains]
        + [norm_space(chain["edge_doc"].get("quote")) for chain in spec.chains]
    )
    if base.contains_answer(nonterminal_text, answer):
        return None, "terminal_answer_shortcut_in_chain"

    evidence_quote = norm_space(parsed.get("answer_evidence_quote"))
    if not evidence_quote or not base.contains_answer(evidence_quote, answer):
        return None, "terminal_answer_quote_missing"
    masked_terminal = _fact_terminal_payload(spec)[0]["passage"]
    quote_key = norm_key(evidence_quote)
    masked_key = norm_key(masked_terminal)
    quote_exact = bool(quote_key and f" {quote_key} " in f" {masked_key} ")
    answer_tokens = set(norm_key(answer).split())
    supporting_tokens = set(quote_key.split()) - answer_tokens - {
        "shared",
        "entity",
        "intermediate",
        "c1",
        "c2",
    }
    terminal_tokens = set(masked_key.split())
    supporting_overlap = len(supporting_tokens & terminal_tokens)
    if not quote_exact and supporting_overlap < 3:
        return None, "terminal_answer_quote_not_grounded"

    return (
        {
            "answer": answer,
            "answer_type": answer_type,
            "answer_docid": norm_space(terminal.get("docid")),
            "answer_evidence_quote": evidence_quote,
            "answer_quote_exact": quote_exact,
            "answer_quote_supporting_overlap": supporting_overlap,
        },
        "",
    )


def validate_response(
    spec: LongChainSpec,
    response: str,
    *,
    min_question_words: int = 70,
    max_question_words: int = 140,
    max_verbatim_source_run: int = 7,
) -> tuple[dict[str, Any] | None, str]:
    if not response.strip():
        return None, "empty_response"
    parsed = base.parse_json_object(response)
    if not parsed:
        return None, "malformed_json"
    question = norm_space(parsed.get("question"))
    word_count = len(WORD_RE.findall(question))
    if not question or not min_question_words <= word_count <= max_question_words:
        return None, "question_length"
    terminal_fact, reason = _terminal_fact(spec, parsed)
    if terminal_fact is None:
        return None, reason
    fact_metadata = terminal_fact
    final_answer = terminal_fact["answer"]
    if base.contains_answer(question, final_answer):
        return None, "terminal_answer_leak"
    if base.contains_answer(question, spec.answer) or PLACEHOLDER_RE.search(question):
        return None, "shared_entity_leak"
    if SYNTHETIC_META_RE.search(question):
        return None, "synthetic_meta_language"
    terminal_title = norm_space(spec.verification[0].get("title"))
    if len(_word_tokens(terminal_title)) >= 3 and base.contains_answer(
        question, terminal_title
    ):
        return None, "terminal_title_leak"
    source_texts = [
        norm_space(chain["bridge_doc"].get("excerpt"))
        for chain in spec.chains
    ] + [
        norm_space(chain["edge_doc"].get("quote"))
        for chain in spec.chains
    ] + [norm_space(spec.verification[0].get("quote"))]
    longest_run = _longest_verbatim_run(question, source_texts)
    if max_verbatim_source_run >= 0 and longest_run > max_verbatim_source_run:
        return None, "verbatim_evidence_span"
    for chain in spec.chains:
        if base.contains_answer(question, chain["bridge_entity"]):
            return None, "bridge_name_leak"
    if question.count("?") != 1 or MULTI_DELIVERABLE_RE.search(question):
        return None, "multiple_deliverables"
    if parsed.get("single_answer") is not True:
        return None, "single_answer_not_confirmed"

    valid_chain_ids = {"C1", "C2"}
    used_chain_ids = {norm_space(value).upper() for value in _as_list(parsed.get("used_chain_ids"))}
    clue_rows = [row for row in _as_list(parsed.get("chain_clues")) if isinstance(row, dict)]
    clue_ids = {norm_space(row.get("chain_id")).upper() for row in clue_rows}
    if used_chain_ids != valid_chain_ids or clue_ids != valid_chain_ids:
        return None, "invalid_chain_references"
    valid_verification_ids = {f"V{index}" for index in range(1, len(spec.verification) + 1)}
    used_verification_ids = {
        norm_space(value).upper() for value in _as_list(parsed.get("used_verification_ids"))
    }
    if used_verification_ids != valid_verification_ids:
        return None, "invalid_verification_references"

    question_key = norm_key(question)
    grounded_chains = 0
    normalized_clues: list[dict[str, Any]] = []
    chain_by_id = {f"C{index}": chain for index, chain in enumerate(spec.chains, start=1)}
    for clue in clue_rows:
        chain_id = norm_space(clue.get("chain_id")).upper()
        chain = chain_by_id[chain_id]
        terms = _normalized_terms(clue.get("grounded_terms"))
        source_text = " ".join(
            (
                norm_space(chain["bridge_doc"].get("excerpt")),
                norm_space(chain["edge_doc"].get("quote")),
            )
        )
        supported = [
            term
            for term in terms
            if norm_key(term) in norm_key(source_text)
            and norm_key(term) in question_key
            and not base.contains_answer(term, final_answer)
            and not base.contains_answer(term, spec.answer)
            and not base.contains_answer(term, chain["bridge_entity"])
        ]
        grounded = bool(supported)
        grounded_chains += int(grounded)
        normalized_clues.append(
            {
                "chain_id": chain_id,
                "bridge_description": norm_space(clue.get("bridge_description")),
                "edge_relation": norm_space(clue.get("edge_relation")),
                "grounded_terms": terms,
                "supported_grounded_terms": supported,
                "soft_grounded": grounded,
            }
        )

    retrieval_plan = [norm_space(value) for value in _as_list(parsed.get("retrieval_plan"))]
    retrieval_plan = [value for value in retrieval_plan if value]
    grounding_fraction = grounded_chains / len(spec.chains)
    generic_answer = bool(base.GENERIC_ANSWER_RE.search(final_answer))
    distractor_count = sum(
        len(_as_list(chain.get("partial_match_distractors"))) for chain in spec.chains
    )
    if grounding_fraction == 1.0 and len(retrieval_plan) >= 3 and not generic_answer:
        quality_tier = "A"
    elif grounding_fraction >= 0.5 and len(retrieval_plan) >= 2:
        quality_tier = "B"
    else:
        quality_tier = "C"
    bridge_docids = [norm_space(chain["bridge_doc"].get("docid")) for chain in spec.chains]
    edge_docids = [norm_space(chain["edge_doc"].get("docid")) for chain in spec.chains]
    verification_docids = [norm_space(row.get("docid")) for row in spec.verification]
    evidence_docids = list(dict.fromkeys(bridge_docids + edge_docids + verification_docids))
    evidence_domains = list(
        dict.fromkeys(
            [norm_space(chain["bridge_doc"].get("domain")) for chain in spec.chains]
            + [norm_space(chain["edge_doc"].get("domain")) for chain in spec.chains]
            + [norm_space(row.get("domain")) for row in spec.verification]
        )
    )
    return (
        {
            "schema": ATOMIC_FACT_SCHEMA,
            "query_id": spec.candidate_id,
            "question": question,
            "question_words": word_count,
            "answer": final_answer,
            "answer_type": norm_space(parsed.get("answer_type") or "other").casefold(),
            "source": "browsecomp_safe_atomic_fact_chain_synth_v4",
            "answer_mode": spec.answer_mode,
            "chain_bundle_id": spec.chain_bundle_id,
            "source_bundle_id": spec.source_bundle_id,
            "variant_index": spec.variant_index,
            "constraint_count": spec.constraint_count,
            "chain_count": len(spec.chains),
            "quality_tier": quality_tier,
            "soft_grounding_fraction": grounding_fraction,
            "retrieval_plan_steps": len(retrieval_plan),
            "generic_answer_proxy": generic_answer,
            "partial_match_distractor_count": distractor_count,
            "max_verbatim_source_run": longest_run,
            "bridge_entities": [chain["bridge_entity"] for chain in spec.chains],
            "bridge_docids": bridge_docids,
            "edge_docids": edge_docids,
            "verification_docids": verification_docids,
            "evidence_docids": evidence_docids,
            "evidence_domains": [value for value in evidence_domains if value],
            "chain_clues": normalized_clues,
            "retrieval_plan": retrieval_plan,
            **fact_metadata,
        },
        "",
    )


def _load_response_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    cache: dict[str, str] = {}
    for row in base.read_jsonl(path):
        candidate_id = norm_space(row.get("candidate_id"))
        if candidate_id:
            cache[candidate_id] = str(row.get("response") or "")
    return cache


def _generate_missing(
    specs: Sequence[LongChainSpec],
    args: argparse.Namespace,
    response_cache: Path,
    *,
    force_ids: set[str] | None = None,
    repair_reasons: dict[str, str] | None = None,
    generation_attempt: int = 0,
) -> dict[str, str]:
    cache = _load_response_cache(response_cache)
    force_ids = force_ids or set()
    repair_reasons = repair_reasons or {}
    missing = [
        spec
        for spec in specs
        if spec.candidate_id in force_ids or not cache.get(spec.candidate_id, "").strip()
    ]
    if not missing:
        print(f"[BCP-LONG-CHAIN-QA] response cache complete: {len(cache)}", flush=True)
        return cache
    client = base.ChatClient(
        args.base_url,
        args.model,
        args.concurrency,
        args.timeout,
        args.retries,
        args.temperature,
        args.enable_thinking,
    )
    response_cache.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = 0
    with response_cache.open("a", encoding="utf-8") as handle:
        for start in range(0, len(missing), args.batch_size):
            batch = missing[start : start + args.batch_size]
            prompts: list[str] = []
            for spec in batch:
                prompt = build_prompt(spec)
                reason = repair_reasons.get(spec.candidate_id)
                if reason:
                    prompt += (
                        "\n\nYour previous response was rejected for: "
                        f"{reason}. Return a corrected JSON object obeying every rule."
                    )
                prompts.append(prompt)
            outputs = client.generate(
                batch, prompts, args.max_tokens, generation_attempt=generation_attempt
            )
            for spec, response in zip(batch, outputs, strict=True):
                cache[spec.candidate_id] = response
                handle.write(
                    json.dumps(
                        {"candidate_id": spec.candidate_id, "response": response},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            completed += len(batch)
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"[BCP-LONG-CHAIN-QA] generated={completed}/{len(missing)} "
                f"rate={completed / elapsed:.2f}/s",
                flush=True,
            )
    print(
        "[BCP-LONG-CHAIN-QA] request_stats="
        + json.dumps(dict(sorted(client.stats.items())), ensure_ascii=False),
        flush=True,
    )
    return cache


def _prepare_parquet(
    accepted: Sequence[dict[str, Any]], args: argparse.Namespace
) -> tuple[Path, Path]:
    raw_path = args.output_dir / "questions.raw.parquet"
    output_path = args.output_dir / "questions.parquet"
    raw_rows = [
        {
            "query_id": row["query_id"],
            "question": row["question"],
            "answer": row["answer"],
            "source": row["source"],
            "ability": "deep_research",
            "metadata": {
                "dataset_name": row["source"],
                "chain_bundle_id": row["chain_bundle_id"],
                "constraint_count": row["constraint_count"],
                "quality_tier": row["quality_tier"],
            },
        }
        for row in accepted
    ]
    pd.DataFrame(raw_rows).to_parquet(raw_path, index=False)
    prepare_script = args.prepare_script.resolve()
    if not prepare_script.is_file():
        raise FileNotFoundError(f"prepare script is missing: {prepare_script}")
    source = prepare_script.read_text(encoding="utf-8", errors="ignore")
    if "native-tool-profile" not in source:
        raise RuntimeError(
            f"prepare script lacks --native-tool-profile: {prepare_script}; "
            "use prepare.py"
        )
    subprocess.run(
        [
            sys.executable,
            str(prepare_script),
            "--input",
            str(raw_path),
            "--output",
            str(output_path),
            "--data-source",
            args.data_source,
            "--tool-prompt-mode",
            "native",
            "--native-tool-profile",
            "search_open_find",
            "--date",
            args.date,
        ],
        check=True,
    )
    by_id = {row["query_id"]: row for row in accepted}
    prepared = pd.read_parquet(output_path)
    patched: list[dict[str, Any]] = []
    for raw in prepared.to_dict(orient="records"):
        query_id = norm_space(raw.get("query_id"))
        source_row = by_id[query_id]
        ground_truth = {
            "metric": "token_f1",
            "target": source_row["answer"],
            "answer_aliases": [source_row["answer"]],
            "synthetic_source": source_row["source"],
            "chain_bundle_id": source_row["chain_bundle_id"],
            "evidence_docids": source_row["evidence_docids"],
            "bridge_docids": source_row["bridge_docids"],
            "edge_docids": source_row["edge_docids"],
            "verification_docids": source_row["verification_docids"],
            "constraint_count": source_row["constraint_count"],
        }
        if source_row.get("answer_docid"):
            ground_truth["answer_docid"] = source_row["answer_docid"]
            ground_truth["answer_mode"] = source_row.get("answer_mode", "terminal_atomic_fact")
        raw["reward_model"] = {"style": "rule", "ground_truth": ground_truth}
        extra_info = raw.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info.update(
            {
                "query_id": query_id,
                "question": source_row["question"],
                "ground_truth": source_row["answer"],
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "search": {
                        "create_kwargs": {
                            "data_source": args.data_source,
                            "ground_truth": ground_truth,
                            "question": source_row["question"],
                        }
                    }
                },
                "bcp_safe_long_chain": {
                    "chain_bundle_id": source_row["chain_bundle_id"],
                    "constraint_count": source_row["constraint_count"],
                    "bridge_docids": source_row["bridge_docids"],
                    "edge_docids": source_row["edge_docids"],
                    "verification_docids": source_row["verification_docids"],
                    "quality_tier": source_row["quality_tier"],
                    "answer_mode": source_row.get("answer_mode", "terminal_atomic_fact"),
                    "answer_docid": source_row.get("answer_docid", ""),
                },
            }
        )
        raw["extra_info"] = extra_info
        patched.append(raw)
    pd.DataFrame(patched).to_parquet(output_path, index=False)
    return raw_path, output_path


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--answer-mode",
        choices=("terminal_atomic_fact",),
        default="terminal_atomic_fact",
        help="Only the final atomic proper-name terminal target is supported",
    )
    parser.add_argument(
        "--unique-answers",
        action="store_true",
        help="retain at most one accepted question per normalized answer",
    )
    parser.add_argument(
        "--unique-bundles",
        action="store_true",
        help="retain at most one accepted question per chain bundle",
    )
    parser.add_argument(
        "--max-accepted-per-bundle",
        type=int,
        default=0,
        help=(
            "retain at most this many accepted questions per chain bundle; "
            "zero disables the cap. --unique-bundles remains the backward-compatible "
            "alias for a cap of one"
        ),
    )
    parser.add_argument("--variants-per-bundle", type=int, default=2)
    parser.add_argument("--constraint-pattern", type=parse_constraint_pattern, default=[4, 5])
    parser.add_argument("--max-bundles", type=int, default=-1)
    parser.add_argument("--max-questions", type=int, default=-1)
    parser.add_argument("--output-question-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--semantic-retry-passes", type=int, default=2)
    parser.add_argument("--min-question-words", type=int, default=70)
    parser.add_argument("--max-question-words", type=int, default=140)
    parser.add_argument(
        "--max-verbatim-source-run",
        type=int,
        default=7,
        help=(
            "V4 maximum contiguous normalized evidence tokens allowed in the "
            "question; negative disables the check"
        ),
    )
    parser.add_argument("--min-generated-questions", type=int, default=64)
    parser.add_argument("--target-behavior-matched-groups", type=int, default=1000)
    parser.add_argument("--assumed-behavior-match-rate", type=float, default=0.15)
    parser.add_argument(
        "--assumed-clean-question-rate",
        type=float,
        default=1.0,
        help=(
            "estimated fraction of generated candidate specifications that survive "
            "generation validation and semantic cleaning; capacity planning only"
        ),
    )
    parser.add_argument(
        "--rollout-plan",
        choices=("pilot_staged", "full_n8"),
        default="pilot_staged",
        help="reporting-only label for the intended downstream rollout policy",
    )
    parser.add_argument("--data-source", default="searchR1_openresearcher")
    parser.add_argument("--date", default="2026-09-03")
    parser.add_argument(
        "--prepare-script", type=Path, default=root / "prepare.py"
    )
    return parser


def resolve_bundle_output_cap(*, unique_bundles: bool, requested_cap: int) -> int:
    """Resolve the legacy boolean and additive numeric bundle-cap controls."""
    if requested_cap < 0:
        raise ValueError("--max-accepted-per-bundle must be non-negative")
    if unique_bundles and requested_cap not in {0, 1}:
        raise ValueError(
            "--unique-bundles cannot be combined with "
            "--max-accepted-per-bundle greater than one"
        )
    return 1 if unique_bundles else requested_cap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.variants_per_bundle <= 0:
        raise SystemExit("--variants-per-bundle must be positive")
    try:
        effective_bundle_cap = resolve_bundle_output_cap(
            unique_bundles=args.unique_bundles,
            requested_cap=args.max_accepted_per_bundle,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.output_question_limit < 0:
        raise SystemExit("--output-question-limit must be non-negative")
    if not 0 < args.assumed_behavior_match_rate <= 1:
        raise SystemExit("--assumed-behavior-match-rate must be in (0, 1]")
    if not 0 < args.assumed_clean_question_rate <= 1:
        raise SystemExit("--assumed-clean-question-rate must be in (0, 1]")
    if not 12 <= args.min_question_words <= args.max_question_words <= 180:
        raise SystemExit("question word limits must satisfy 12 <= min <= max <= 180")
    bundles = base.read_jsonl(args.bundles)
    if args.max_bundles > 0:
        bundles = bundles[: args.max_bundles]
    specs, spec_counts = build_specs(
        bundles,
        variants_per_bundle=args.variants_per_bundle,
        constraint_pattern=args.constraint_pattern,
        seed=args.seed,
        max_questions=args.max_questions,
        answer_mode=args.answer_mode,
    )
    if not specs:
        raise SystemExit("no long-chain candidate specifications could be built")
    required_clean_questions = math.ceil(
        args.target_behavior_matched_groups / args.assumed_behavior_match_rate
    )
    required_candidate_specs = math.ceil(
        required_clean_questions / args.assumed_clean_question_rate
    )
    if len(specs) < required_candidate_specs:
        print(
            f"[BCP-LONG-CHAIN-QA] capacity warning: specs={len(specs)} "
            f"< estimated_specs_for_target={required_candidate_specs}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    response_cache = args.output_dir / "responses.cache.jsonl"
    cache = _generate_missing(specs, args, response_cache)
    retry_counts: list[int] = []
    for retry_index in range(1, args.semantic_retry_passes + 1):
        rejected_ids: dict[str, str] = {}
        for spec in specs:
            row, reason = validate_response(
                spec,
                cache.get(spec.candidate_id, ""),
                min_question_words=args.min_question_words,
                max_question_words=args.max_question_words,
                max_verbatim_source_run=args.max_verbatim_source_run,
            )
            if row is None:
                rejected_ids[spec.candidate_id] = reason
        retry_counts.append(len(rejected_ids))
        if not rejected_ids:
            break
        print(
            f"[BCP-LONG-CHAIN-QA] semantic_retry={retry_index}/"
            f"{args.semantic_retry_passes} candidates={len(rejected_ids)}",
            flush=True,
        )
        cache = _generate_missing(
            specs,
            args,
            response_cache,
            force_ids=set(rejected_ids),
            repair_reasons=rejected_ids,
            generation_attempt=retry_index,
        )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    seen_bundles: set[str] = set()
    accepted_per_bundle: Counter[str] = Counter()
    for spec in specs:
        row, reason = validate_response(
            spec,
            cache.get(spec.candidate_id, ""),
            min_question_words=args.min_question_words,
            max_question_words=args.max_question_words,
            max_verbatim_source_run=args.max_verbatim_source_run,
        )
        if row is None:
            rejection_counts[reason] += 1
            rejected.append(
                {
                    "candidate_id": spec.candidate_id,
                    "chain_bundle_id": spec.chain_bundle_id,
                    "reason": reason,
                }
            )
            continue
        question_key = norm_key(row["question"])
        if question_key in seen_questions:
            rejection_counts["duplicate_question"] += 1
            rejected.append(
                {
                    "candidate_id": spec.candidate_id,
                    "chain_bundle_id": spec.chain_bundle_id,
                    "reason": "duplicate_question",
                }
            )
            continue
        answer_key = norm_key(row["answer"])
        if args.unique_answers and answer_key in seen_answers:
            rejection_counts["duplicate_answer"] += 1
            rejected.append(
                {
                    "candidate_id": spec.candidate_id,
                    "chain_bundle_id": spec.chain_bundle_id,
                    "reason": "duplicate_answer",
                }
            )
            continue
        if (
            effective_bundle_cap > 0
            and accepted_per_bundle[spec.chain_bundle_id] >= effective_bundle_cap
        ):
            rejection_reason = (
                "duplicate_chain_bundle"
                if args.unique_bundles
                else "bundle_output_cap"
            )
            rejection_counts[rejection_reason] += 1
            rejected.append(
                {
                    "candidate_id": spec.candidate_id,
                    "chain_bundle_id": spec.chain_bundle_id,
                    "reason": rejection_reason,
                }
            )
            continue
        seen_questions.add(question_key)
        seen_answers.add(answer_key)
        seen_bundles.add(spec.chain_bundle_id)
        accepted_per_bundle[spec.chain_bundle_id] += 1
        accepted.append(row)
    accepted_before_limit = len(accepted)
    if args.output_question_limit and len(accepted) > args.output_question_limit:
        accepted.sort(key=lambda row: (stable_int(args.seed, "output", row["query_id"]), row["query_id"]))
        accepted = accepted[: args.output_question_limit]
    output_bundle_counts = Counter(row["chain_bundle_id"] for row in accepted)
    base.write_jsonl(args.output_dir / "questions.jsonl", accepted)
    base.write_jsonl(args.output_dir / "rejected.jsonl", rejected)
    raw_parquet: Path | None = None
    prepared_parquet: Path | None = None
    if accepted:
        raw_parquet, prepared_parquet = _prepare_parquet(accepted, args)

    report = {
        "schema": ATOMIC_FACT_SCHEMA,
        "inputs": {"bundles": str(args.bundles), "bundles_loaded": len(bundles)},
        "generation": {
            "model": args.model,
            "base_urls": args.base_url,
            "seed": args.seed,
            "answer_mode": args.answer_mode,
            "max_verbatim_source_run": args.max_verbatim_source_run,
            "candidate_specs": len(specs),
            "accepted_questions": len(accepted),
            "accepted_before_output_limit": accepted_before_limit,
            "output_question_limit": args.output_question_limit,
            "variants_per_bundle": args.variants_per_bundle,
            "max_accepted_per_bundle": effective_bundle_cap,
            "rejected_questions": len(rejected),
            "unique_answers": len({norm_key(row["answer"]) for row in accepted}),
            "unique_chain_bundles": len({row["chain_bundle_id"] for row in accepted}),
            "accepted_per_bundle_distribution": dict(
                sorted(
                    Counter(str(count) for count in output_bundle_counts.values()).items()
                )
            ),
            "constraint_counts": dict(
                sorted(Counter(str(row["constraint_count"]) for row in accepted).items())
            ),
            "quality_tier_counts": dict(
                sorted(Counter(row["quality_tier"] for row in accepted).items())
            ),
            "question_words": {
                "min": min((row["question_words"] for row in accepted), default=0),
                "mean": (
                    sum(row["question_words"] for row in accepted) / len(accepted)
                    if accepted
                    else 0.0
                ),
                "max": max((row["question_words"] for row in accepted), default=0),
            },
            "semantic_retry_candidate_counts": retry_counts,
            "spec_counts": dict(sorted(spec_counts.items())),
            "rejection_counts": dict(sorted(rejection_counts.items())),
        },
        "hard_gate_policy": [
            "parseable_json",
            f"question_length_{args.min_question_words}_to_{args.max_question_words}_words",
            "single_requested_answer",
            "no_final_answer_or_bridge_name_leak",
            "all_chain_and_verification_ids_referenced",
            "unique_question_text",
            "terminal_answer_exactly_in_body",
            "terminal_answer_absent_from_titles_and_prior_chain",
            "terminal_answer_not_ui_or_placeholder",
            "unique_answer_and_chain_bundle_when_requested",
            "atomic_proper_name_answer_type",
            "no_date_number_fragment_or_multi_entity_answer",
            "no_synthetic_process_language",
            f"question_verbatim_evidence_run_le_{args.max_verbatim_source_run}",
            "terminal_page_title_withheld_and_not_repeated",
        ],
        "soft_only_quality": [
            "grounded_terms",
            "retrieval_plan_steps",
            "generic_answer_proxy",
            "partial_match_distractor_capacity",
            "direct_retrieval_difficulty_requires_strict_rollout",
        ],
        "pilot_policy": {
            "rollout_plan": args.rollout_plan,
            "generate_first_then_rollout_in_64_and_192_row_stages": (
                args.rollout_plan == "pilot_staged"
            ),
            "target_behavior_matched_groups": args.target_behavior_matched_groups,
            "assumed_behavior_match_rate": args.assumed_behavior_match_rate,
            "assumed_clean_question_rate": args.assumed_clean_question_rate,
            # Backward-compatible name retained for report consumers.  It has
            # always represented post-generation questions, not LLM specs.
            "estimated_raw_questions_required": required_clean_questions,
            "estimated_clean_questions_required": required_clean_questions,
            "estimated_candidate_specs_required": required_candidate_specs,
            "note": "Only strict n=8 rollout establishes actual mixed and behavior-match rates.",
        },
        "gate": {
            "passed": len(accepted) >= args.min_generated_questions,
            "minimum_generated_questions": args.min_generated_questions,
            "failures": (
                []
                if len(accepted) >= args.min_generated_questions
                else [
                    f"accepted_questions={len(accepted)} < required={args.min_generated_questions}"
                ]
            ),
        },
        "outputs": {
            "questions_jsonl": str(args.output_dir / "questions.jsonl"),
            "rejected_jsonl": str(args.output_dir / "rejected.jsonl"),
            "response_cache": str(response_cache),
            "raw_parquet": str(raw_parquet) if raw_parquet else None,
            "prepared_parquet": str(prepared_parquet) if prepared_parquet else None,
        },
    }
    report_path = args.output_dir / "generation.report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
