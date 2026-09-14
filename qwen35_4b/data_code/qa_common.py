#!/usr/bin/env python3
"""Generate minimally-gated research QA from BrowseComp-safe evidence bundles.

The generator never consumes BrowseComp questions, answers, or qrels.  Its only
input is the safe bundle artifact produced by
``anchors.py``.  An OpenAI-compatible local model
turns two or three independent evidence snippets into a natural question whose
short answer is the bundle anchor.

Only failures that make a sample unusable are hard-rejected: malformed JSON,
an empty/implausible question, answer leakage, invalid evidence references, or
duplicate question text.  Grounding strength and answer specificity are
reported as soft quality tiers instead of being used as a global gate.  This is
intentional: the downstream n=8 rollout and continuous-F1 variance filter are
the actual trainability gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


SCHEMA = "browsecomp_safe_synthetic_qa_v1"
GENERIC_ANSWER_RE = re.compile(
    r"\b(?:directory|admissions?|awards?|tournaments?|website|platform|"
    r"program(?:me)?s?|services?|resources?|association|organization|"
    r"foundation|institute|university|college|school|journal|magazine|"
    r"festival|conference|competition|company|group|project|initiative)\b",
    re.IGNORECASE,
)
TITLE_ONLY_RE = re.compile(r"^\s*title\s*:\s*", re.IGNORECASE)
WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)


def norm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def norm_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", norm_space(value)).casefold()
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


def stable_int(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def parse_json_object(text: str) -> dict[str, Any]:
    value = JSON_FENCE_RE.sub("", str(text or "").strip())
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        objects: list[dict[str, Any]] = []
        for match in re.finditer(r"\{", value):
            try:
                parsed, _ = decoder.raw_decode(value[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                objects.append(parsed)
        return objects[-1] if objects else {}


def contains_answer(question: str, answer: str) -> bool:
    q_key = norm_key(question)
    a_key = norm_key(answer)
    return bool(a_key and f" {a_key} " in f" {q_key} ")


def mask_answer(text: str, answer: str) -> str:
    answer_tokens = [re.escape(token) for token in WORD_RE.findall(norm_space(answer))]
    if not answer_tokens:
        return norm_space(text)
    pattern = re.compile(
        r"(?<!\w)" + r"[\W_]+".join(answer_tokens) + r"(?!\w)",
        re.IGNORECASE,
    )
    return norm_space(pattern.sub("[HIDDEN ANSWER]", str(text or "")))




class CandidateSpec(NamedTuple):
    candidate_id: str
    bundle_id: str
    variant_index: int
    constraint_count: int
    answer: str
    evidence: tuple[dict[str, Any], ...]
    source_capacity: dict[str, Any]










class ChatClient:
    def __init__(
        self,
        base_urls: Sequence[str],
        model: str,
        concurrency: int,
        timeout: float,
        retries: int,
        temperature: float,
        enable_thinking: bool,
    ) -> None:
        urls: list[str] = []
        for raw in base_urls:
            base = raw.rstrip("/")
            if base.endswith("/chat/completions"):
                urls.append(base)
            elif base.endswith("/v1"):
                urls.append(base + "/chat/completions")
            else:
                urls.append(base + "/v1/chat/completions")
        if not urls:
            raise ValueError("at least one --base-url is required")
        self.urls = urls
        self.model = model
        self.concurrency = max(1, concurrency)
        self.timeout = timeout
        self.retries = max(0, retries)
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.stats: Counter[str] = Counter()
        self._lock = threading.Lock()

    def _stat(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def _one(
        self, spec: CandidateSpec, prompt: str, max_tokens: int, generation_attempt: int
    ) -> str:
        url = self.urls[stable_int(spec.candidate_id) % len(self.urls)]
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return only the requested JSON object. Do not reason aloud, "
                            "add Markdown fences, or add prose before or after the JSON. "
                            "/no_think"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                # Qwen3.5 thinks by default.  With the old 768-token budget it can
                # spend the whole completion inside <think> and never emit the JSON.
                # SGLang exposes this chat-template switch on its OpenAI endpoint.
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                "temperature": max(
                    0.05, self.temperature - 0.10 * generation_attempt
                ),
                "top_p": 0.9,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        last_error = ""
        for _ in range(self.retries + 1):
            request = Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                choice = (body.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = str(message.get("content") or "").strip()
                reasoning_content = str(message.get("reasoning_content") or "").strip()
                self._stat("requests")
                finish_reason = norm_space(choice.get("finish_reason")) or "missing"
                self._stat(f"finish_reason.{finish_reason}")
                if reasoning_content:
                    self._stat("responses_with_reasoning_content")
                if "<think>" in content.casefold():
                    self._stat("content_with_think_tag")
                if content:
                    return content
                # Keep this only as a diagnostic fallback.  A complete JSON object
                # can occasionally land in reasoning_content when a server was
                # launched with a reasoning parser; validation remains fail-closed.
                if reasoning_content:
                    self._stat("reasoning_content_fallback")
                    return reasoning_content
                last_error = f"empty_response; finish_reason={finish_reason}"
            except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                self._stat("request_errors")
        self._stat("failed")
        if self.stats["failed"] <= 5:
            print(f"[BCP-SAFE-QA] request failed for {spec.candidate_id}: {last_error}", flush=True)
        return ""

    def generate(
        self,
        specs: Sequence[CandidateSpec],
        prompts: Sequence[str],
        max_tokens: int,
        generation_attempt: int = 0,
    ) -> list[str]:
        outputs = [""] * len(specs)
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(specs))) as pool:
            futures = {
                pool.submit(
                    self._one, spec, prompt, max_tokens, generation_attempt
                ): index
                for index, (spec, prompt) in enumerate(zip(specs, prompts, strict=True))
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    outputs[index] = future.result()
                except Exception as exc:  # defensive isolation across thousands of calls
                    self._stat("future_errors")
                    outputs[index] = ""
                    if self.stats["future_errors"] <= 5:
                        print(f"[BCP-SAFE-QA] worker failed: {exc}", flush=True)
        return outputs
