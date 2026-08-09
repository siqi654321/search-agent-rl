# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone BrowseComp-Plus retrieval server with resilient summarization and safer short-title handling.

这个版本专门解决两个问题：
- summarize 偶发 400/502：失败时自动缩短 `max_doc_chars` 重试，最后降级成 extractive summary；
- 超长标题污染 prompt：标题提取更保守，且 summarize/fallback 中 title 会单独截断。
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import re
import threading
import time
import textwrap
from dataclasses import dataclass
from collections import Counter
from typing import Any, Optional

try:
    import datasets  # type: ignore
except Exception:  # pragma: no cover
    datasets = None

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

try:
    import uvicorn  # type: ignore
except Exception:  # pragma: no cover
    uvicorn = None

try:
    from fastapi import FastAPI, HTTPException  # type: ignore
    from fastapi.responses import PlainTextResponse  # type: ignore
except Exception:  # pragma: no cover
    FastAPI = None
    HTTPException = Exception
    PlainTextResponse = None

try:
    from huggingface_hub import snapshot_download  # type: ignore
except Exception:  # pragma: no cover
    snapshot_download = None

try:
    from pydantic import BaseModel  # type: ignore
except Exception:  # pragma: no cover
    BaseModel = object

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    import torch  # type: ignore
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    torch = None
    AutoModel = None
    AutoTokenizer = None


def _require(dep, name: str):
    if dep is None:
        raise ModuleNotFoundError(f"Missing dependency '{name}'. Please install it first.")
    return dep


short_title_max_chars = 120

_METADATA_PREFIXES = (
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
_METADATA_FIELD_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}:\s*\S")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those", "it", "its",
    "their", "his", "her", "he", "she", "they", "them", "who", "whom", "whose", "which", "what", "when",
    "where", "why", "how", "please", "tell", "me", "name", "following", "criteria", "criterion", "particular",
    "certain", "about", "into", "after", "before", "between", "during", "over", "under", "than", "then", "also",
    "same", "first", "second", "third", "identify", "find", "provide", "could", "would", "should", "user", "goal",
}


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _shorten_title(text: str, max_chars: int) -> str:
    text = _normalize_spaces(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _looks_like_metadata_line(line: str) -> bool:
    low = line.lower()
    if low.startswith(_METADATA_PREFIXES):
        return True
    if line in {"---", "..."}:
        return True
    if line.startswith("http://") or line.startswith("https://"):
        return True
    if _METADATA_FIELD_RE.match(line):
        return True
    return False


def _extract_title(text: str, url: str) -> str:
    text = str(text or "")
    url = str(url or "")
    lines = [_normalize_spaces(line) for line in text.splitlines()]
    lines = [line for line in lines if line]

    for line in lines:
        if line.lower().startswith("title:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate:
                return _shorten_title(candidate, short_title_max_chars)

    for line in lines:
        if _looks_like_metadata_line(line):
            continue
        return _shorten_title(line, short_title_max_chars)

    if url:
        return _shorten_title(url, short_title_max_chars)
    return "Untitled"


def _format_corpus_doc(doc: dict[str, Any]) -> dict[str, Any]:
    docid = str(doc.get("docid") or "")
    text = str(doc.get("text") or "")
    url = str(doc.get("url") or "")
    title = _extract_title(text=text, url=url)
    contents = f"{title}\n{text}".strip()
    return {
        "docid": docid,
        "id": docid,
        "title": title,
        "text": text,
        "url": url,
        "contents": contents,
    }


class BrowseCompCorpus:
    def __init__(
        self,
        *,
        corpus_repo_id: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        max_docs: int = -1,
    ):
        _require(datasets, "datasets")
        self.dataset = datasets.load_dataset(corpus_repo_id, split=split, cache_dir=cache_dir)
        total = len(self.dataset)
        limit = total if max_docs is None or max_docs <= 0 else min(total, int(max_docs))
        self.docid_to_idx: dict[str, int] = {}
        for idx in range(limit):
            row = self.dataset[idx]
            self.docid_to_idx[str(row.get("docid"))] = idx
        self.size = limit

    def get_doc(self, docid: str) -> Optional[dict[str, Any]]:
        idx = self.docid_to_idx.get(str(docid))
        if idx is None:
            return None
        return _format_corpus_doc(self.dataset[idx])

    def get_docs(self, docids: list[str]) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for docid in docids:
            doc = self.get_doc(str(docid))
            if doc is not None:
                docs.append(doc)
        return docs


class QueryEncoder:
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 32,
        query_instruction: Optional[str] = None,
        backend: str = "sentence_transformers",
        max_length: int = 8192,
    ):
        _require(np, "numpy")
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.backend = backend
        self.max_length = max(1, int(max_length))
        self.tokenizer = None

        if backend == "sentence_transformers":
            _require(SentenceTransformer, "sentence-transformers")
            self.model = SentenceTransformer(model_name_or_path)
        elif backend == "transformer_eos":
            _require(torch, "torch")
            _require(AutoModel, "transformers")
            _require(AutoTokenizer, "transformers")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                padding_side="left",
            )
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model.eval()
        else:
            raise ValueError(f"Unsupported query encoder backend: {backend}")

    def encode_queries(self, queries: list[str]) -> "np.ndarray":
        if self.backend == "transformer_eos":
            return self._encode_queries_transformer_eos(queries)

        kwargs = {
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.query_instruction:
            kwargs["prompt"] = self.query_instruction
        else:
            kwargs["prompt_name"] = "query"

        try:
            emb = self.model.encode(queries, **kwargs)
        except Exception:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("prompt_name", None)
            if "prompt" not in fallback_kwargs:
                fallback_kwargs["prompt"] = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
            emb = self.model.encode(queries, **fallback_kwargs)
        return np.asarray(emb, dtype=np.float32)

    def _encode_queries_transformer_eos(self, queries: list[str]) -> "np.ndarray":
        """AgentIR-compatible left-padded, last-token query encoding."""

        if not queries:
            return np.empty((0, 0), dtype=np.float32)
        prefix = self.query_instruction or ""
        batches: list["np.ndarray"] = []
        for start in range(0, len(queries), max(1, self.batch_size)):
            texts = [prefix + query for query in queries[start : start + self.batch_size]]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            device = self.model.device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                hidden = self.model(**encoded, return_dict=True).last_hidden_state
                embeddings = torch.nn.functional.normalize(hidden[:, -1], p=2, dim=-1)
            batches.append(embeddings.float().cpu().numpy())
        return np.asarray(np.concatenate(batches, axis=0), dtype=np.float32)


class BaseRetriever:
    def __init__(self, config: "Config"):
        self.config = config
        self.topk = int(config.retrieval_topk)

    def search(self, query: str, num: Optional[int] = None, return_score: bool = False):
        raise NotImplementedError

    def batch_search(self, query_list: list[str], num: Optional[int] = None, return_score: bool = False):
        raise NotImplementedError


class BM25Retriever(BaseRetriever):
    def __init__(self, config: "Config"):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.index_dir = resolve_index_dir(config=config)
        self.searcher = LuceneSearcher(self.index_dir)
        self.corpus = BrowseCompCorpus(
            corpus_repo_id=config.corpus_repo_id,
            split=config.corpus_split,
            cache_dir=config.cache_dir,
            max_docs=config.max_docs,
        )

    def search(self, query: str, num: Optional[int] = None, return_score: bool = False):
        topk = int(num or self.topk)
        hits = self.searcher.search(query, topk)
        docids = [str(hit.docid) for hit in hits]
        scores = [float(hit.score) for hit in hits]
        docs = self.corpus.get_docs(docids)
        if return_score:
            return docs, scores[: len(docs)]
        return docs

    def batch_search(self, query_list: list[str], num: Optional[int] = None, return_score: bool = False):
        all_docs = []
        all_scores = []
        for query in query_list:
            docs, scores = self.search(query=query, num=num, return_score=True)
            all_docs.append(docs)
            all_scores.append(scores)
        if return_score:
            return all_docs, all_scores
        return all_docs


class DenseRetriever(BaseRetriever):
    def __init__(self, config: "Config"):
        super().__init__(config)
        _require(faiss, "faiss")
        _require(np, "numpy")

        self.index_dir = resolve_index_dir(config=config)
        self.corpus = BrowseCompCorpus(
            corpus_repo_id=config.corpus_repo_id,
            split=config.corpus_split,
            cache_dir=config.cache_dir,
            max_docs=config.max_docs,
        )
        self.encoder = QueryEncoder(
            model_name_or_path=config.query_encoder_model,
            batch_size=config.query_encoder_batch_size,
            query_instruction=config.query_instruction,
            backend=config.query_encoder_backend,
            max_length=config.query_encoder_max_length,
        )
        self.index: "faiss.Index" = None  # type: ignore[assignment]
        self.lookup: list[str] = []
        self._build_faiss_index(faiss_gpu=config.faiss_gpu, max_docs=config.max_docs)

    def _iter_pkl_files(self) -> list[str]:
        paths = sorted(glob.glob(os.path.join(self.index_dir, "*.pkl")))
        if not paths:
            raise FileNotFoundError(f"No .pkl files found under index_dir={self.index_dir}")
        return paths

    def _build_faiss_index(self, *, faiss_gpu: bool, max_docs: int):
        arrays = []
        total = 0
        for path in self._iter_pkl_files():
            with open(path, "rb") as f:
                embeddings, lookup = pickle.load(f)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings)

            if max_docs and max_docs > 0:
                remaining = max_docs - total
                if remaining <= 0:
                    break
                if embeddings.shape[0] > remaining:
                    embeddings = embeddings[:remaining]
                    lookup = lookup[:remaining]

            arrays.append(embeddings)
            self.lookup.extend([str(x) for x in lookup])
            total += int(embeddings.shape[0])
            if max_docs and max_docs > 0 and total >= max_docs:
                break

        if not arrays:
            raise RuntimeError(f"No embedding shards loaded from {self.index_dir}")

        matrix = np.concatenate(arrays, axis=0)
        index = faiss.IndexFlatIP(int(matrix.shape[1]))
        index.add(matrix)

        if faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            index = faiss.index_cpu_to_all_gpus(index, co=co)

        self.index = index

    def _lookup_docs(self, idxs: list[int]) -> list[dict[str, Any]]:
        docids = []
        for idx in idxs:
            if idx < 0 or idx >= len(self.lookup):
                continue
            docids.append(self.lookup[idx])
        return self.corpus.get_docs(docids)

    def search(self, query: str, num: Optional[int] = None, return_score: bool = False):
        topk = int(num or self.topk)
        query_emb = self.encoder.encode_queries([query])
        scores, idxs = self.index.search(query_emb, topk)
        idx_list = [int(x) for x in idxs[0].tolist()]
        score_list = [float(x) for x in scores[0].tolist()]
        docs = self._lookup_docs(idx_list)
        if return_score:
            return docs, score_list[: len(docs)]
        return docs

    def batch_search(self, query_list: list[str], num: Optional[int] = None, return_score: bool = False):
        topk = int(num or self.topk)
        query_emb = self.encoder.encode_queries(list(query_list))
        scores, idxs = self.index.search(query_emb, topk)

        all_docs = []
        all_scores = []
        for row_idxs, row_scores in zip(idxs.tolist(), scores.tolist(), strict=True):
            docs = self._lookup_docs([int(x) for x in row_idxs])
            all_docs.append(docs)
            all_scores.append([float(x) for x in row_scores][: len(docs)])

        if return_score:
            return all_docs, all_scores
        return all_docs


def resolve_index_dir(config: "Config") -> str:
    if config.index_path:
        path = os.path.expanduser(config.index_path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"index_path 不存在或不是目录: {path}")
        return path

    _require(snapshot_download, "huggingface_hub")
    local_root = snapshot_download(
        repo_id=config.indexes_repo_id,
        repo_type="dataset",
        allow_patterns=[f"{config.index_name}/*"],
        cache_dir=config.cache_dir,
    )
    index_dir = os.path.join(local_root, config.index_name)
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"Downloaded index dir not found: {index_dir}")
    return index_dir


def get_retriever(config: "Config") -> BaseRetriever:
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    if config.retrieval_method == "dense":
        return DenseRetriever(config)
    raise ValueError(f"Unsupported retrieval_method: {config.retrieval_method}")


class Config:
    def __init__(
        self,
        *,
        retrieval_method: str,
        retrieval_topk: int,
        indexes_repo_id: str,
        index_name: str,
        index_path: Optional[str],
        corpus_repo_id: str,
        corpus_split: str,
        cache_dir: Optional[str],
        query_encoder_model: str,
        query_encoder_batch_size: int,
        query_instruction: Optional[str],
        query_encoder_backend: str,
        query_encoder_max_length: int,
        faiss_gpu: bool,
        max_docs: int,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.indexes_repo_id = indexes_repo_id
        self.index_name = index_name
        self.index_path = index_path
        self.corpus_repo_id = corpus_repo_id
        self.corpus_split = corpus_split
        self.cache_dir = cache_dir
        self.query_encoder_model = query_encoder_model
        self.query_encoder_batch_size = query_encoder_batch_size
        self.query_instruction = query_instruction
        self.query_encoder_backend = query_encoder_backend
        self.query_encoder_max_length = query_encoder_max_length
        self.faiss_gpu = faiss_gpu
        self.max_docs = max_docs


def _build_chat_completions_url(base_url: str) -> str:
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        raise ValueError("sglang_base_url 不能为空")
    if base_url.endswith("/v1/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _parse_sglang_base_urls(base_url: Optional[Any]) -> list[str]:
    if base_url is None:
        return []
    if isinstance(base_url, (list, tuple)):
        ret: list[str] = []
        for item in base_url:
            ret.extend(_parse_sglang_base_urls(item))
        return ret
    return [item.strip() for item in str(base_url).split(",") if item.strip()]


def _extract_chat_completion_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not choices:
        raise ValueError(f"SGLang 返回缺少 choices 字段: {payload}")
    choice0 = choices[0]
    if isinstance(choice0, dict):
        msg = choice0.get("message")
        if isinstance(msg, dict) and msg.get("content") is not None:
            return str(msg.get("content"))
        if choice0.get("text") is not None:
            return str(choice0.get("text"))
    raise ValueError(f"SGLang 返回不含可解析内容: {payload}")


def _truncate_text(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def _format_docs_for_llm(docs: list[dict[str, Any]], scores: Optional[list[float]], max_doc_chars: int) -> str:
    parts: list[str] = []
    for i, doc in enumerate(docs):
        contents = _truncate_text(str(doc.get("contents") or ""), max_chars=max_doc_chars)
        title = _shorten_title(str(doc.get("title") or ""), short_title_max_chars)
        body = contents.split("\n", 1)[1].strip() if "\n" in contents else contents
        score_part = ""
        if scores is not None and i < len(scores):
            try:
                score_part = f" (score={float(scores[i]):.4f})"
            except Exception:
                score_part = ""
        parts.append(f"Doc {i + 1}{score_part} (Title: {title})\n{body}".strip())
    return "\n\n".join([p for p in parts if p])


def _build_summary_user_prompt(query: str, docs_text: str) -> str:
    return (
        "You are given a user query and top retrieved documents. "
        "Write a concise, factual summary grounded ONLY in the retrieved documents.\n\n"
        f"User query: {query}\n\n"
        f"Retrieved documents:\n{docs_text}\n\n"
        "Requirements:\n"
        "- Cover the key facts (who/what/when/where) if present in the docs.\n"
        "- Do NOT invent facts not supported by the docs.\n"
        "- End with one sentence explaining why the results match the query."
    )


VISIT_EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
Return ONLY one valid JSON object. Do not wrap it in markdown. Escape internal double quotes in string values. Do not output unescaped newlines inside JSON strings. The schema must be exactly:
{{"rational": "...", "evidence": "...", "summary": "..."}}
"""


def _build_visit_extractor_prompt(webpage_content: str, goal: str) -> str:
    return VISIT_EXTRACTOR_PROMPT.format(webpage_content=webpage_content, goal=goal)


def _clean_llm_response(content: str) -> str:
    content = str(content or "").strip()
    if not content:
        return ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
    return content.replace("```json", "").replace("```", "").strip()


def _parse_visit_extractor_json(raw: str) -> dict[str, Any]:
    cleaned = _clean_llm_response(raw)
    left, right = cleaned.find("{"), cleaned.rfind("}")
    if left != -1 and right != -1 and left <= right:
        cleaned = cleaned[left : right + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError(f"Visit extractor returned non-object JSON: {type(parsed)}")
    return parsed


def _extract_error_detail(resp: Any) -> str:
    try:
        text = resp.text.strip()
    except Exception:
        text = ""
    return text[:2000] if text else ""


def _call_sglang_summarizer(
    *,
    base_url: str,
    model: str,
    api_key: Optional[str],
    system_prompt: Optional[str],
    user_prompt: str,
    timeout: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    _require(requests, "requests")
    url = _build_chat_completions_url(base_url)
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        detail = _extract_error_detail(resp)
        raise RuntimeError(f"HTTP {resp.status_code} from {url}; body={detail}")
    return _extract_chat_completion_content(resp.json()).strip()


_sglang_rr_lock = threading.Lock()
_sglang_rr_idx = 0


def _call_sglang_summarizer_multi(
    *,
    base_urls: list[str] | str,
    model: str,
    api_key: Optional[str],
    system_prompt: Optional[str],
    user_prompt: str,
    timeout: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    base_urls = _parse_sglang_base_urls(base_urls)
    if not base_urls:
        raise ValueError("sglang_base_url 不能为空")

    global _sglang_rr_idx
    with _sglang_rr_lock:
        start = _sglang_rr_idx % len(base_urls)
        _sglang_rr_idx += 1

    errors: list[str] = []
    for offset in range(len(base_urls)):
        base_url = base_urls[(start + offset) % len(base_urls)]
        try:
            return _call_sglang_summarizer(
                base_url=base_url,
                model=model,
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as e:
            errors.append(f"{base_url}: {e}")
    raise RuntimeError("all SGLang backends failed: " + " | ".join(errors))


def _parse_retry_doc_chars(raw: Optional[str], initial_value: int) -> list[int]:
    values: list[int] = []
    if initial_value > 0:
        values.append(int(initial_value))
    if raw:
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            value = int(part)
            if value > 0 and value not in values:
                values.append(value)
    for candidate in [2000, 1000, 500, 250]:
        if candidate > 0 and candidate not in values and candidate < max(values or [candidate]) + 1:
            values.append(candidate)
    return sorted(set(values), reverse=True)


def _build_extractive_fallback(query: str, docs: list[dict[str, Any]], scores: Optional[list[float]], max_doc_chars: int) -> str:
    snippets: list[str] = []
    for idx, doc in enumerate(docs[: min(3, len(docs))]):
        title = _shorten_title(str(doc.get("title") or "Untitled"), short_title_max_chars)
        url = str(doc.get("url") or "")
        contents = _truncate_text(str(doc.get("contents") or ""), max_chars=max_doc_chars)
        body = contents.split("\n", 1)[1].strip() if "\n" in contents else contents
        snippet = body.replace("\n", " ").strip()
        score_part = ""
        if scores is not None and idx < len(scores):
            score_part = f" score={scores[idx]:.4f}."
        snippets.append(f"Doc {idx + 1}: {title}.{score_part} {snippet[:500]} URL: {url}".strip())
    joined = " ".join(snippets).strip()
    if not joined:
        return f"No reliable summary was generated for query: {query}."
    return (
        f"Fallback summary for query '{query}': "
        f"The following high-ranked retrieved evidence may help answer the question. {joined}"
    ).strip()


def _normalize_lite_values(value: Any) -> list[str]:
    if value is None:
        values = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    return [str(v).strip() for v in values if str(v).strip()]


def _lite_doc_url(doc: dict[str, Any]) -> str:
    docid = str(doc.get("docid") or doc.get("id") or "").strip()
    url = str(doc.get("url") or "").strip()
    # In the local BrowseComp-Plus corpus, docid is the stable address.  Keep
    # Keep the field name as "url" to match the tool schema, but expose
    # doc:// links in search results so the model copies deterministic local
    # identifiers instead of long web URLs that later need reverse lookup.
    if docid:
        return f"doc://{docid}"
    if url:
        return url
    return "doc://unknown"


def _search_link_field(doc: dict[str, Any]) -> str:
    return f"{search_result_link_label}: {_lite_doc_url(doc)}"


def _search_open_instruction() -> str:
    return (
        f"Use {search_result_link_label} values with the "
        f"{search_result_open_tool_name} tool to inspect the source document."
    )


def _snippet_from_doc(doc: dict[str, Any], max_chars: Optional[int] = None) -> str:
    if max_chars is None:
        max_chars = lite_search_snippet_chars
    contents = str(doc.get("contents") or doc.get("text") or "")
    if "\n" in contents:
        contents = contents.split("\n", 1)[1]
    snippet = re.sub(r"\s+", " ", contents).strip()
    return _truncate_text(snippet, max_chars=max_chars)


def _content_terms(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", str(text or "").lower())
    return {t for t in toks if t not in _STOPWORDS and len(t) >= 2}


def _split_text_chunks(text: str, chunk_chars: int, overlap_chars: int) -> list[tuple[int, str]]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    chunk_chars = max(200, int(chunk_chars))
    overlap_chars = max(0, min(int(overlap_chars), chunk_chars // 2))
    step = max(1, chunk_chars - overlap_chars)
    chunks: list[tuple[int, str]] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_chars].strip()
        if chunk:
            chunks.append((start, chunk))
        if start + chunk_chars >= len(text):
            break
    return chunks


def _extract_answer_like_spans(text: str, max_items: int = 80) -> list[str]:
    """No-gold candidate entity/fact inventory for search/visit externalization."""
    text = re.sub(r"\s+", " ", str(text or " ")).strip()
    candidates: list[str] = []

    # Quoted work titles / exact phrases.
    candidates.extend(m.group(1).strip() for m in re.finditer(r"[\"“”']([^\"“”']{3,90})[\"“”']", text))

    # Title-case names/titles/organizations plus acronym-like spans.
    title_span_re = re.compile(r"\b(?:[A-Z][A-Za-z0-9&.\-]*(?:\s+|$)){1,8}")
    for m in title_span_re.finditer(text):
        span = re.sub(r"\s+", " ", m.group(0)).strip(" ,.;:()[]{}")
        if len(span) >= 3 and span.lower() not in _STOPWORDS:
            candidates.append(span)

    # Dates, years, percentages, measurements, compact numeric answers.
    candidates.extend(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", text))
    candidates.extend(re.findall(r"\b\d+(?:\.\d+)?\s?%\b", text))
    candidates.extend(
        re.findall(
            r"\b\d+(?:\.\d+)?\s?(?:kg|kilograms|km|m|cm|mm|ft|feet|in|inch|inches|miles|years|episodes|goals|points)\b",
            text,
            flags=re.I,
        )
    )
    candidates.extend(re.findall(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", text))

    bad = {
        "Retrieved", "Evidence", "Summary", "Contents", "Skip", "Privacy Policy",
        "Terms", "Home", "Menu", "Search", "Read More", "Click Here",
    }
    out: list[str] = []
    seen = set()
    for cand in candidates:
        cand = re.sub(r"\s+", " ", str(cand)).strip(" -–—,.;:()[]{}")
        if len(cand) < 2 or len(cand) > 120 or cand in bad:
            continue
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= max_items:
            break
    return out


def _externalize_chunk_score(chunk: str, *, terms: set[str], years: set[str]) -> float:
    low = chunk.lower()
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", low)
    counts = Counter(t for t in toks if t in terms)
    unique_hits = len(counts)
    overlap = sum(counts.values())
    year_hits = sum(1 for y in years if y in low)
    proper_spans = re.findall(r"\b[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,5}", chunk)
    quoted = re.findall(r"[\"“”'][^\"“”']{3,90}[\"“”']", chunk)
    numerics = re.findall(r"\b\d+(?:\.\d+)?\b", chunk)
    return (
        4.0 * unique_hits
        + 1.0 * overlap
        + 5.0 * year_hits
        + min(len(proper_spans), 20) * 0.35
        + min(len(quoted), 8) * 0.75
        + min(len(numerics), 20) * 0.15
    )


def _selected_externalized_chunks(
    *,
    text: str,
    context: str,
    chunk_chars: int,
    overlap_chars: int,
    include_head: bool,
    max_chunks: int,
) -> list[tuple[int, str]]:
    terms = _content_terms(context)
    years = set(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", context))
    chunks = _split_text_chunks(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    scored: list[tuple[float, int, str]] = []
    for idx, (start, chunk) in enumerate(chunks):
        score = _externalize_chunk_score(chunk, terms=terms, years=years)
        # Small position prior keeps intro/infobox-like sections visible without
        # swamping query/goal relevance.
        score += 1.0 / math.sqrt(idx + 1)
        scored.append((score, start, chunk))

    selected: list[tuple[int, str]] = []
    used_starts = set()
    if include_head and chunks:
        selected.append(chunks[0])
        used_starts.add(chunks[0][0])
    for _score, start, chunk in sorted(scored, key=lambda x: x[0], reverse=True):
        if start in used_starts:
            continue
        selected.append((start, chunk))
        used_starts.add(start)
        if len(selected) >= max_chunks:
            break
    return selected


def _build_source_card(
    *,
    query: str,
    doc: dict[str, Any],
    budget_chars: int,
    mode: str,
) -> str:
    """Build a compact high-recall /search result card without using gold."""
    docid = str(doc.get("docid") or doc.get("id") or "").strip()
    title = _shorten_title(str(doc.get("title") or docid or "Untitled"), short_title_max_chars)
    contents = str(doc.get("contents") or doc.get("text") or "")
    body = contents.split("\n", 1)[1].strip() if "\n" in contents else contents
    include_inventory = "inventory" in (mode or "")
    include_head = "head" in (mode or "")

    header = f"{title} | {_search_link_field(doc)} | DocID: {docid}".strip()
    sections: list[str] = []
    if include_inventory:
        inventory = _extract_answer_like_spans(
            body[: max(30000, int(budget_chars) * 6)],
            max_items=search_card_max_entity_items,
        )
        if inventory:
            sections.append("Candidate answer-like spans: " + "; ".join(inventory))

    selected = _selected_externalized_chunks(
        text=body,
        context=query,
        chunk_chars=search_card_chunk_chars,
        overlap_chars=search_card_overlap_chars,
        include_head=include_head,
        max_chunks=3,
    )
    if selected:
        snippet_text = "\n".join(f"[snippet @ char {start}] {chunk}" for start, chunk in selected)
        sections.append("Relevant snippets: " + snippet_text)
    else:
        sections.append("Snippet: " + _snippet_from_doc(doc, max_chars=lite_search_snippet_chars))

    return _truncate_text(header + "\n" + "\n".join(sections), max_chars=max(300, int(budget_chars)))


def _build_compact_search_line(
    *,
    query: str,
    doc: dict[str, Any],
    mode: str,
    snippet_chars: int,
) -> str:
    """Very compact search line used by global title/title-inventory output."""
    docid = str(doc.get("docid") or doc.get("id") or "").strip()
    title = _shorten_title(str(doc.get("title") or docid or "Untitled"), short_title_max_chars)
    docid_part = f"DocID: {docid} | " if docid else ""
    mode = mode or "snippet"
    if mode == "title":
        return f"{title} | {_search_link_field(doc)} | {docid_part}".strip()

    contents = str(doc.get("contents") or doc.get("text") or "")
    body = contents.split("\n", 1)[1].strip() if "\n" in contents else contents
    if mode == "title_inventory":
        inventory = _extract_answer_like_spans(
            body[: max(20000, int(snippet_chars) * 100)],
            max_items=max(5, min(search_card_max_entity_items, 20)),
        )
        if inventory and snippet_chars > 0:
            snippet = _truncate_text("Candidates: " + "; ".join(inventory), max_chars=snippet_chars)
            return f"{title} | {_search_link_field(doc)} | {docid_part}{snippet}".strip()
        return f"{title} | {_search_link_field(doc)} | {docid_part}".strip()

    if mode == "query":
        selected = _selected_externalized_chunks(
            text=body,
            context=query,
            chunk_chars=max(200, snippet_chars),
            overlap_chars=0,
            include_head=False,
            max_chunks=1,
        )
        snippet = selected[0][1] if selected else _snippet_from_doc(doc, max_chars=snippet_chars)
        snippet = _truncate_text(re.sub(r"\s+", " ", snippet).strip(), max_chars=snippet_chars)
        return f"{title} | {_search_link_field(doc)} | {docid_part}Snippet: {snippet}".strip()

    snippet = _snippet_from_doc(doc, max_chars=snippet_chars)
    return f"{title} | {_search_link_field(doc)} | {docid_part}Snippet: {snippet}".strip()


def _doc_key(doc: dict[str, Any]) -> str:
    return str(doc.get("docid") or doc.get("id") or doc.get("url") or hash(str(doc)[:1000]))


def _normalize_frontier_docid(value: Any) -> str:
    docid = str(value or "").strip()
    if docid.lower().startswith("doc://"):
        docid = docid[6:]
    return docid.rstrip(".,;:)]}").strip().strip("/")


def _global_doc_score(rec: dict[str, Any], mode: str) -> float:
    rank = max(1, int(rec.get("rank", 10**9)))
    query_count = max(1, int(rec.get("query_count", 1)))
    rrf = float(rec.get("rrf", 1.0 / (60.0 + rank)))
    dense_score = float(rec.get("score") or 0.0)
    if mode == "min_rank":
        return 1.0 / math.sqrt(rank) + 0.05 * query_count + 0.0001 * dense_score
    if mode == "rrf":
        return rrf + 0.01 * query_count + 0.0001 * dense_score
    if mode == "rrf_lexical":
        text = str(rec.get("doc", {}).get("contents") or rec.get("doc", {}).get("text") or "")[:8000]
        title = str(rec.get("doc", {}).get("title") or "")
        query = " ".join(rec.get("queries", []))
        terms = _content_terms(query)
        title_l = title.lower()
        text_l = text.lower()
        lex = sum(5.0 for t in terms if t in title_l) + sum(1.0 for t in terms if t in text_l)
        return 5.0 * rrf + 0.05 * query_count + 0.1 * lex + 0.0001 * dense_score
    return 1.0 / math.sqrt(rank)


def _collect_global_search_records(
    *,
    queries: list[str],
    results: list[list[dict[str, Any]]],
    scores: Optional[list[list[float]]] = None,
    rank_cutoff: int,
) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for qi, docs in enumerate(results):
        query = queries[qi] if qi < len(queries) else ""
        for rank, doc in enumerate(docs, start=1):
            if rank > max(1, int(rank_cutoff)):
                continue
            key = _doc_key(doc)
            score = None
            if scores is not None and qi < len(scores) and rank - 1 < len(scores[qi]):
                score = scores[qi][rank - 1]
            rec = by_key.get(key)
            if rec is None:
                rec = {
                    "doc": doc,
                    "rank": rank,
                    "query": query,
                    "queries": [],
                    "query_count": 0,
                    "rrf": 0.0,
                    "score": score,
                }
                by_key[key] = rec
            rec["queries"].append(query)
            rec["query_count"] = int(rec.get("query_count", 0)) + 1
            rec["rrf"] = float(rec.get("rrf", 0.0)) + 1.0 / (60.0 + rank)
            if rank < int(rec.get("rank", 10**9)):
                rec["doc"] = doc
                rec["rank"] = rank
                rec["query"] = query
                rec["score"] = score
    return by_key


def _rank_global_search_records(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records.values(),
        key=lambda rec: _global_doc_score(rec, search_global_rerank_mode),
        reverse=True,
    )


def _format_lite_search_global_result(
    *,
    queries: list[str],
    results: list[list[dict[str, Any]]],
    scores: Optional[list[list[float]]] = None,
    frontier_enable: bool = False,
    frontier_exclude_docids: Optional[list[str]] = None,
    frontier_base_rank_cutoff: int = 10,
    frontier_output_topk: Optional[int] = None,
    frontier_candidate_topk: int = 50,
    frontier_overlap_threshold: float = 0.90,
    frontier_compact_snippet_chars: Optional[int] = None,
) -> tuple[str, dict[str, Any]]:
    """Render a base top page or an unseen page from a larger retrieval frontier.

    The base page is ranked only from each query's original top-N pool.  This is
    intentional: adding rank 11-50 candidates must not perturb the established
    top-10 control.  The full pool participates only after the base page has
    already been exposed and substantially overlaps the trajectory history.
    """

    full_rank_cutoff = max(1, int(search_global_rank_cutoff))
    base_rank_cutoff = min(full_rank_cutoff, max(1, int(frontier_base_rank_cutoff)))
    output_topk = max(
        1,
        int(
            frontier_output_topk
            if frontier_output_topk is not None
            else search_global_output_topk
        ),
    )
    threshold = min(1.0, max(0.0, float(frontier_overlap_threshold)))

    base_records = _collect_global_search_records(
        queries=queries,
        results=results,
        scores=scores,
        rank_cutoff=base_rank_cutoff,
    )
    full_records = _collect_global_search_records(
        queries=queries,
        results=results,
        scores=scores,
        rank_cutoff=full_rank_cutoff,
    )
    base_ranked = _rank_global_search_records(base_records)
    full_ranked = _rank_global_search_records(full_records)[: max(1, int(frontier_candidate_topk))]
    base_page = base_ranked[:output_topk]

    excluded = {
        docid
        for value in (frontier_exclude_docids or [])
        if (docid := _normalize_frontier_docid(value))
    }
    base_page_docids = [_normalize_frontier_docid(_doc_key(rec["doc"])) for rec in base_page]
    base_overlap = (
        sum(docid in excluded for docid in base_page_docids) / len(base_page_docids)
        if base_page_docids
        else 0.0
    )
    frontier_applied = bool(frontier_enable and excluded and base_overlap >= threshold)
    unseen_ranked = [
        rec
        for rec in full_ranked
        if _normalize_frontier_docid(_doc_key(rec["doc"])) not in excluded
    ]
    selected = unseen_ranked[:output_topk] if frontier_applied else base_page
    frontier_exhausted = bool(frontier_applied and not selected)

    if frontier_applied:
        header = [
            f"A local corpus search for {len(queries)} query/query variants found "
            f"{len(full_records)} internal candidate documents.",
        ]
        if frontier_exhausted:
            header.extend(
                [
                    "SEARCH_FRONTIER_EXHAUSTED",
                    (
                        "No unseen candidates remain in this search frontier. "
                        "Open a previously returned cursor or add a genuinely new entity, relation, date, "
                        "title, or quoted phrase to the next search."
                    ),
                ]
            )
        else:
            header.extend(
                [
                    "SEARCH_FRONTIER_PAGE",
                    (
                        f"Returning up to {output_topk} candidates not previously shown in this trajectory. "
                        f"{_search_open_instruction()} Compact candidates may be equally relevant; "
                        "open a cursor to inspect full evidence."
                    ),
                    "Results:",
                ]
            )
    else:
        # Preserve the original base-page wording and candidate pool so the
        # first page remains directly comparable with the 51-point control.
        header = [
            f"A local corpus search for {len(queries)} query/query variants found "
            f"{len(base_records)} unique documents.",
            (
                f"Returning a budgeted global top list. {_search_open_instruction()} "
                f"Top {search_global_rich_topk} results include richer evidence; "
                "the rest are compact title/candidate lines."
            ),
            "Results:",
        ]

    lines: list[str] = []
    compact_snippet_chars = int(search_global_compact_snippet_chars)
    if frontier_applied and frontier_compact_snippet_chars is not None:
        compact_snippet_chars = max(0, int(frontier_compact_snippet_chars))
    for out_rank, rec in enumerate(selected, start=1):
        doc = rec["doc"]
        # Select passages using only the query variants that actually retrieved
        # this document; unrelated batch queries otherwise dilute lexical focus.
        query_context = " ".join(rec.get("queries") or queries)
        if out_rank <= int(search_global_rich_topk):
            text = _build_source_card(
                query=query_context,
                doc=doc,
                budget_chars=search_global_rich_budget,
                mode=search_global_rich_mode,
            )
        else:
            text = _build_compact_search_line(
                query=query_context,
                doc=doc,
                mode=search_global_compact_mode,
                snippet_chars=compact_snippet_chars,
            )
        line = f"{out_rank}. {text}".strip()
        current = "\n".join(header + lines)
        if search_global_max_output_chars > 0 and len(current) + len(line) + 2 > int(search_global_max_output_chars):
            break
        lines.append(line)
    selected_records = selected[: len(lines)]
    metadata = {
        "enabled": bool(frontier_enable),
        "applied": frontier_applied,
        "exhausted": frontier_exhausted,
        "base_overlap": base_overlap,
        "base_rank_cutoff": base_rank_cutoff,
        "full_rank_cutoff": full_rank_cutoff,
        "base_candidate_count": len(base_records),
        "candidate_count": len(full_ranked),
        "full_unique_candidate_count": len(full_records),
        "unseen_count": len(unseen_ranked),
        "excluded_count": len(excluded),
        "selected_count": len(selected_records),
        "selected_docids": [
            _normalize_frontier_docid(_doc_key(rec["doc"])) for rec in selected_records
        ],
        "selected_original_ranks": [int(rec.get("rank", 0)) for rec in selected_records],
    }
    return "\n".join(header + lines).strip(), metadata


def _rank_route_records(
    *,
    queries: list[str],
    results: list[list[dict[str, Any]]],
    scores: Optional[list[list[float]]],
    rank_cutoff: int,
    rerank_mode: str,
) -> list[dict[str, Any]]:
    records = _collect_global_search_records(
        queries=queries,
        results=results,
        scores=scores,
        rank_cutoff=rank_cutoff,
    )
    return sorted(
        records.values(),
        key=lambda rec: _global_doc_score(rec, rerank_mode),
        reverse=True,
    )


def _fuse_dual_ranked_records(
    *,
    primary_ranked: list[dict[str, Any]],
    secondary_docids: list[str],
    corpus: Any,
    output_topk: int,
    rrf_k: int,
    primary_weight: float,
    secondary_weight: float,
    queries: list[str],
) -> list[dict[str, Any]]:
    """Fuse route-level ranked lists using weighted reciprocal rank fusion."""

    records: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    next_seen = 0
    for rank, rec in enumerate(primary_ranked, start=1):
        doc = rec.get("doc") or {}
        docid = _normalize_frontier_docid(_doc_key(doc))
        if not docid:
            continue
        if docid not in first_seen:
            first_seen[docid] = next_seen
            next_seen += 1
        records[docid] = {
            "doc": doc,
            "queries": list(queries),
            "primary_rank": rank,
            "secondary_rank": None,
            "fusion_score": float(primary_weight) / (max(0, int(rrf_k)) + rank),
        }
    for rank, raw_docid in enumerate(secondary_docids, start=1):
        docid = _normalize_frontier_docid(raw_docid)
        if not docid:
            continue
        if docid not in first_seen:
            first_seen[docid] = next_seen
            next_seen += 1
        rec = records.get(docid)
        if rec is None:
            doc = corpus.get_doc(docid) if corpus is not None else None
            if doc is None:
                continue
            rec = {
                "doc": doc,
                "queries": list(queries),
                "primary_rank": None,
                "secondary_rank": rank,
                "fusion_score": 0.0,
            }
            records[docid] = rec
        rec["secondary_rank"] = rank
        rec["fusion_score"] = float(rec.get("fusion_score", 0.0)) + float(
            secondary_weight
        ) / (max(0, int(rrf_k)) + rank)

    ranked = sorted(
        records.items(),
        key=lambda item: (
            -float(item[1].get("fusion_score", 0.0)),
            min(
                int(item[1].get("primary_rank") or 10**9),
                int(item[1].get("secondary_rank") or 10**9),
            ),
            first_seen[item[0]],
            item[0],
        ),
    )[: max(1, int(output_topk))]
    fused: list[dict[str, Any]] = []
    for rank, (_, rec) in enumerate(ranked, start=1):
        copied = dict(rec)
        copied["rank"] = rank
        fused.append(copied)
    return fused


def _format_dual_ranked_result(
    *,
    queries: list[str],
    ranked: list[dict[str, Any]],
    frontier_enable: bool,
    frontier_exclude_docids: Optional[list[str]],
    frontier_output_topk: int,
    frontier_overlap_threshold: float,
    frontier_compact_snippet_chars: Optional[int],
) -> tuple[str, dict[str, Any]]:
    """Render a pre-fused route ranking without reordering it a second time."""

    output_topk = max(1, int(frontier_output_topk))
    base_page = ranked[:output_topk]
    excluded = {
        docid
        for value in (frontier_exclude_docids or [])
        if (docid := _normalize_frontier_docid(value))
    }
    base_docids = [
        _normalize_frontier_docid(_doc_key(rec.get("doc") or {})) for rec in base_page
    ]
    base_overlap = (
        sum(docid in excluded for docid in base_docids) / len(base_docids)
        if base_docids
        else 0.0
    )
    threshold = min(1.0, max(0.0, float(frontier_overlap_threshold)))
    frontier_applied = bool(frontier_enable and excluded and base_overlap >= threshold)
    unseen_ranked = [
        rec
        for rec in ranked
        if _normalize_frontier_docid(_doc_key(rec.get("doc") or {})) not in excluded
    ]
    selected = unseen_ranked[:output_topk] if frontier_applied else base_page
    frontier_exhausted = bool(frontier_applied and not selected)

    if frontier_applied:
        header = [
            f"A dual retrieval search for {len(queries)} query/query variants found "
            f"{len(ranked)} fused candidate documents."
        ]
        if frontier_exhausted:
            header.extend(
                [
                    "SEARCH_FRONTIER_EXHAUSTED",
                    (
                        "No unseen candidates remain in this fused search frontier. "
                        "Open a previously returned cursor or add a genuinely new entity, relation, date, "
                        "title, or quoted phrase to the next search."
                    ),
                ]
            )
        else:
            header.extend(
                [
                    "SEARCH_FRONTIER_PAGE",
                    (
                        f"Returning up to {output_topk} unseen fused candidates. "
                        f"{_search_open_instruction()} Compact candidates may be equally relevant; "
                        "open a cursor to inspect full evidence."
                    ),
                    "Results:",
                ]
            )
    else:
        header = [
            f"A dual retrieval search for {len(queries)} query/query variants found "
            f"{len(ranked)} fused candidate documents.",
            (
                "Results combine the original query route with a reasoning-aware route. "
                f"{_search_open_instruction()} Top {search_global_rich_topk} results include richer "
                "evidence; the rest are compact candidate lines."
            ),
            "Results:",
        ]

    lines: list[str] = []
    compact_chars = int(search_global_compact_snippet_chars)
    if frontier_applied and frontier_compact_snippet_chars is not None:
        compact_chars = max(0, int(frontier_compact_snippet_chars))
    query_context = " ".join(queries)
    for out_rank, rec in enumerate(selected, start=1):
        doc = rec.get("doc") or {}
        if out_rank <= int(search_global_rich_topk):
            text = _build_source_card(
                query=query_context,
                doc=doc,
                budget_chars=search_global_rich_budget,
                mode=search_global_rich_mode,
            )
        else:
            text = _build_compact_search_line(
                query=query_context,
                doc=doc,
                mode=search_global_compact_mode,
                snippet_chars=compact_chars,
            )
        line = f"{out_rank}. {text}".strip()
        current = "\n".join(header + lines)
        if (
            search_global_max_output_chars > 0
            and len(current) + len(line) + 2 > int(search_global_max_output_chars)
        ):
            break
        lines.append(line)

    selected_records = selected[: len(lines)]
    metadata = {
        "enabled": bool(frontier_enable),
        "applied": frontier_applied,
        "exhausted": frontier_exhausted,
        "base_overlap": base_overlap,
        "base_rank_cutoff": output_topk,
        "full_rank_cutoff": len(ranked),
        "base_candidate_count": min(output_topk, len(ranked)),
        "candidate_count": len(ranked),
        "full_unique_candidate_count": len(ranked),
        "unseen_count": len(unseen_ranked),
        "excluded_count": len(excluded),
        "selected_count": len(selected_records),
        "selected_docids": [
            _normalize_frontier_docid(_doc_key(rec.get("doc") or {}))
            for rec in selected_records
        ],
        "selected_original_ranks": [int(rec.get("rank", 0)) for rec in selected_records],
        "dual_retrieval": True,
    }
    return "\n".join(header + lines).strip(), metadata


def _secondary_ranked_docids(*, queries: list[str], topk: int) -> list[str]:
    if not dual_secondary_rank_url:
        raise ValueError("dual_secondary_rank_url is empty")
    response = requests.post(
        dual_secondary_rank_url,
        json={"queries": queries, "topk": topk, "return_scores": True},
        timeout=dual_secondary_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success", False):
        raise RuntimeError(str(payload.get("error") or "secondary rank endpoint failed"))
    return [
        docid
        for value in payload.get("ranked_docids", [])
        if (docid := _normalize_frontier_docid(value))
    ]


def _build_externalized_visit_evidence(
    *,
    doc: dict[str, Any],
    goal: str,
    budget_chars: int,
    mode: str,
) -> str:
    """Build high-recall /browse evidence without using gold."""
    docid = str(doc.get("docid") or doc.get("id") or "").strip()
    title = _shorten_title(str(doc.get("title") or docid or "Untitled"), short_title_max_chars)
    contents = str(doc.get("contents") or doc.get("text") or "")
    body = contents.split("\n", 1)[1].strip() if "\n" in contents else contents
    include_inventory = "inventory" in (mode or "")
    include_head = "head" in (mode or "")

    header = f"Page title: {title}\nDocID: {docid}\n"
    sections: list[str] = []
    if include_inventory:
        inventory = _extract_answer_like_spans(
            body[: max(50000, int(budget_chars) * 4)],
            max_items=visit_externalize_max_entity_items,
        )
        if inventory:
            sections.append("Candidate answer-like spans:\n- " + "\n- ".join(inventory))

    selected = _selected_externalized_chunks(
        text=body,
        context=goal,
        chunk_chars=visit_externalize_chunk_chars,
        overlap_chars=visit_externalize_overlap_chars,
        include_head=include_head,
        max_chunks=8,
    )
    if selected:
        chunk_lines = []
        for i, (start, chunk) in enumerate(selected, start=1):
            chunk_lines.append(f"[chunk {i} @ char {start}]\n{chunk}")
        sections.append("Selected raw evidence chunks:\n" + "\n\n".join(chunk_lines))
    else:
        sections.append(_truncate_text(body, max_chars=min(int(budget_chars), default_max_doc_chars)))

    evidence = header + "\n\n".join(sections)
    return _truncate_text(evidence, max_chars=max(1000, int(budget_chars)))


def _build_broad_visit_goal(goal: str) -> str:
    goal = re.sub(r"\s+", " ", str(goal or "")).strip()
    return (
        "Extract high-recall evidence from this page for the user's research goal. "
        "Preserve exact entity names, page titles, work titles, institution names, roles, dates, years, "
        "numbers, measurements, and short answer-like phrases. Include candidate facts even if they only "
        "partially match; do not filter out names/titles just because relevance is uncertain. "
        f"User visit goal/search context: {goal}"
    )


def _format_lite_search_result(query: str, docs: list[dict[str, Any]], scores: Optional[list[float]] = None) -> str:
    # Keep the search response compact so top-10 results fit under verl's
    # max_tool_response_length.  Expose only doc:// identifiers to discourage
    # the model from copying long real URLs; /browse can visit doc:// directly.
    lines = [
        f"A local corpus search for '{query}' found {len(docs)} results:",
        f"{_search_open_instruction()} Results:",
    ]
    for idx, doc in enumerate(docs, start=1):
        docid = str(doc.get("docid") or doc.get("id") or "").strip()
        title = _shorten_title(str(doc.get("title") or docid or "Untitled"), short_title_max_chars)
        docid_part = f"DocID: {docid} | " if docid else ""
        use_card = bool(search_card_enable) and (
            search_card_rich_topk <= 0 or idx <= int(search_card_rich_topk)
        )
        if use_card:
            card = _build_source_card(
                query=query,
                doc=doc,
                budget_chars=search_card_budget_per_doc,
                mode=search_card_mode,
            )
            lines.append(f"{idx}. {card}".strip())
        else:
            snippet = _snippet_from_doc(doc)
            lines.append(f"{idx}. {title} | {_search_link_field(doc)} | {docid_part}Snippet: {snippet}".strip())
    return "\n".join(lines).strip()


def _document_lines(doc: dict[str, Any], line_chars: Optional[int] = None) -> list[str]:
    """Convert a corpus document into deterministic, addressable display lines."""
    width = max(40, int(line_chars or browser_line_chars))
    text = str(doc.get("text") or doc.get("contents") or "")
    raw_lines = text.splitlines() or [text]
    lines: list[str] = []
    for raw in raw_lines:
        normalized = re.sub(r"\s+", " ", raw).strip()
        if not normalized:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        wrapped = textwrap.wrap(
            normalized,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        lines.extend(wrapped or [normalized])
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _line_relevance_score(line: str, *, terms: set[str], years: set[str]) -> float:
    low = line.lower()
    hits = [term for term in terms if term in low]
    year_hits = [year for year in years if year in low]
    return 3.0 * len(set(hits)) + 1.0 * len(hits) + 5.0 * len(set(year_hits))


def _best_open_loc(lines: list[str], goal: str, num_lines: int) -> int:
    if not lines or not str(goal or "").strip():
        return 0
    terms = _content_terms(goal)
    years = set(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", goal))
    if not terms and not years:
        return 0
    line_scores = [_line_relevance_score(line, terms=terms, years=years) for line in lines]
    window = max(1, min(int(num_lines), len(lines)))
    running = sum(line_scores[:window])
    best_score = running
    best_start = 0
    for start in range(1, max(1, len(lines) - window + 1)):
        running += line_scores[start + window - 1] - line_scores[start - 1]
        if running > best_score:
            best_score = running
            best_start = start
    return best_start if best_score > 0 else 0


def _format_open_document(
    *,
    doc: dict[str, Any],
    loc: Optional[int],
    num_lines: Optional[int],
    goal: str = "",
) -> str:
    docid = str(doc.get("docid") or doc.get("id") or "").strip()
    cursor = f"doc://{docid}" if docid else str(doc.get("url") or "")
    title = _shorten_title(str(doc.get("title") or docid or "Untitled"), short_title_max_chars)
    lines = _document_lines(doc)
    requested = browser_default_num_lines if num_lines is None or int(num_lines) <= 0 else int(num_lines)
    count = max(1, min(requested, browser_max_num_lines))
    if loc is None or int(loc) < 0:
        start = _best_open_loc(lines, goal=goal, num_lines=count)
        positioned = "query-relevant"
    else:
        start = min(max(0, int(loc)), max(0, len(lines) - 1))
        positioned = "explicit"
    end = min(len(lines), start + count)
    rendered = [f"L{i}: {lines[i]}" for i in range(start, end)]
    next_hint = f" Next: open cursor={cursor!r} loc={end}." if end < len(lines) else " End of document."
    header = (
        f"Cursor: {cursor}\nTitle: {title}\nURL: {str(doc.get('url') or '').strip()}\n"
        f"Showing lines {start}-{max(start, end - 1)} of {len(lines)} ({positioned} window).{next_hint}"
    )
    return header + "\n" + "\n".join(rendered)


def _format_find_results(
    *,
    doc: dict[str, Any],
    pattern: str,
    context_lines: Optional[int],
    max_hits: Optional[int],
) -> str:
    docid = str(doc.get("docid") or doc.get("id") or "").strip()
    cursor = f"doc://{docid}" if docid else str(doc.get("url") or "")
    title = _shorten_title(str(doc.get("title") or docid or "Untitled"), short_title_max_chars)
    lines = _document_lines(doc)
    needle = re.sub(r"\s+", " ", str(pattern or "")).strip().lower()
    if not needle:
        return f"Find error: pattern is empty. Cursor: {cursor}"
    indices = [idx for idx, line in enumerate(lines) if needle in line.lower()]
    limit = max(1, min(int(max_hits or browser_find_max_hits), browser_find_max_hits))
    radius = max(0, min(int(context_lines or browser_find_context_lines), 5))
    if not indices:
        return (
            f"Cursor: {cursor}\nTitle: {title}\nPattern {pattern!r} was not found. "
            "Try a shorter exact name, year, or phrase, or open another window."
        )
    blocks: list[str] = []
    for hit_no, idx in enumerate(indices[:limit], start=1):
        lo = max(0, idx - radius)
        hi = min(len(lines), idx + radius + 1)
        body = "\n".join(f"L{i}: {lines[i]}" for i in range(lo, hi))
        blocks.append(f"Hit {hit_no} at line {idx}:\n{body}")
    return (
        f"Cursor: {cursor}\nTitle: {title}\nFound {len(indices)} line(s) containing {pattern!r}; "
        f"showing {min(len(indices), limit)}. Use open with cursor={cursor!r} and loc=<line> for a larger window.\n"
        + "\n---\n".join(blocks)
    )


def _extract_docid_from_lite_url(raw_url: str) -> str:
    value = str(raw_url or "").strip()
    if value.startswith("doc://"):
        return value[len("doc://") :].strip().strip("/")
    if value.startswith("docid:"):
        return value.split(":", 1)[1].strip()
    # Search results expose either doc://<docid> or raw docids. If a real URL is
    # copied back by the model, try to find a matching corpus URL before failing.
    if retriever is not None and getattr(retriever, "corpus", None) is not None:
        corpus = retriever.corpus
        if hasattr(corpus, "docid_to_idx") and value in getattr(corpus, "docid_to_idx", {}):
            return value
        if value.startswith("http://") or value.startswith("https://"):
            for docid in list(getattr(corpus, "docid_to_idx", {}).keys()):
                doc = corpus.get_doc(docid)
                if doc and str(doc.get("url") or "").strip() == value:
                    return docid
    return value


def _summarize_visit_doc(goal: str, doc: dict[str, Any], max_doc_chars: int) -> SummaryAttemptResult:
    query = goal or "Extract information relevant to the user's research question from this page."
    return _summarize_with_fallback(query=query, docs=[doc], scores=None, request_max_doc_chars=max_doc_chars)


def _format_visit_result(raw_url: str, doc: dict[str, Any], goal: str, max_doc_chars: int, summarize: bool) -> str:
    title = _shorten_title(str(doc.get("title") or "Untitled"), short_title_max_chars)
    docid = str(doc.get("docid") or doc.get("id") or "")
    url = _lite_doc_url(doc)
    contents = str(doc.get("contents") or doc.get("text") or "")
    body = contents.split("\n", 1)[1].strip() if "\n" in contents else contents

    if visit_externalize_enable:
        # High-recall no-gold evidence externalization: surface candidate
        # answer-like spans and selected raw chunks directly instead of relying
        # on the LLM extractor summary to preserve short names/titles/numbers.
        broad_goal = _build_broad_visit_goal(goal)
        evidence = _build_externalized_visit_evidence(
            doc=doc,
            goal=broad_goal,
            budget_chars=visit_externalize_budget_chars,
            mode=visit_externalize_mode,
        )
        if summarize and visit_externalize_with_summary:
            extracted = _summarize_visit_doc_with_extractor(goal=broad_goal, doc=doc, max_doc_chars=max_doc_chars)
            summary = _truncate_text(str(extracted.summary), visit_summary_max_chars)
        elif summarize:
            summary = (
                "High-recall candidate spans and selected raw evidence chunks are provided above; "
                "LLM summary generation was disabled by server configuration."
            )
        else:
            summary = "Summarization is disabled. High-recall raw evidence chunks are provided above."
    elif summarize:
        extracted = _summarize_visit_doc_with_extractor(goal=goal, doc=doc, max_doc_chars=max_doc_chars)
        # For long multi-turn rollouts (50-100 tool interactions), bound each
        # visit response.  The extractor still sees max_doc_chars of page text,
        # but the model only receives compact evidence/summary.
        evidence = _truncate_text(str(extracted.evidence), visit_evidence_max_chars)
        summary = _truncate_text(str(extracted.summary), visit_summary_max_chars)
    else:
        evidence = _truncate_text(body, max_doc_chars)
        summary = "Summarization is disabled. The relevant page content is provided in Evidence in page above."

    # Keep the established browser output contract so the model sees the same
    # Evidence/Summary framing used during training and evaluation.
    return (
        f"The useful information in {raw_url} for user goal {goal} as follows: \n\n"
        f"Evidence in page: \n{evidence}\n\n"
        f"Summary: \n{summary}\n\n"
    ).strip()


@dataclass
class SummaryAttemptResult:
    summary: str
    used_backend: Optional[str]
    used_max_doc_chars: int
    mode: str
    attempts: list[dict[str, Any]]


@dataclass
class LiteExtractorAttemptResult:
    evidence: str
    summary: str
    rational: str
    used_backend: Optional[str]
    used_max_doc_chars: int
    mode: str
    attempts: list[dict[str, Any]]


def _build_visit_extractive_fallback(goal: str, doc: dict[str, Any], max_doc_chars: int) -> LiteExtractorAttemptResult:
    title = _shorten_title(str(doc.get("title") or "Untitled"), short_title_max_chars)
    contents = str(doc.get("contents") or doc.get("text") or "")
    body = contents.split("\n", 1)[1].strip() if "\n" in contents else contents
    evidence = _truncate_text(body, max_doc_chars).strip()
    summary = _build_extractive_fallback(query=goal, docs=[doc], scores=None, max_doc_chars=max_doc_chars)
    return LiteExtractorAttemptResult(
        evidence=evidence or f"No relevant evidence was extracted from {title}.",
        summary=summary,
        rational="Extractor LLM was unavailable or returned invalid JSON; fell back to truncated page content.",
        used_backend=None,
        used_max_doc_chars=max_doc_chars,
        mode="extractive_fallback",
        attempts=[{"ok": True, "mode": "extractive_fallback", "max_doc_chars": max_doc_chars}],
    )


def _summarize_visit_doc_with_extractor(goal: str, doc: dict[str, Any], max_doc_chars: int) -> LiteExtractorAttemptResult:
    query = goal or "Extract information relevant to the user's research question from this page."
    attempts: list[dict[str, Any]] = []
    retry_chars = _parse_retry_doc_chars(summarize_retry_max_doc_chars, max_doc_chars)

    if sglang_model and sglang_base_urls:
        for current_max_doc_chars in retry_chars:
            contents = _truncate_text(str(doc.get("contents") or doc.get("text") or ""), max_chars=current_max_doc_chars)
            user_prompt = _build_visit_extractor_prompt(webpage_content=contents, goal=query)
            try:
                raw = _call_sglang_summarizer_multi(
                    base_urls=sglang_base_urls,
                    model=sglang_model,
                    api_key=sglang_api_key,
                    system_prompt=None,
                    user_prompt=user_prompt,
                    timeout=sglang_timeout,
                    temperature=sglang_temperature,
                    top_p=sglang_top_p,
                    max_tokens=sglang_max_tokens,
                )
                parsed = _parse_visit_extractor_json(raw)
                evidence = str(parsed.get("evidence", "N/A"))
                summary = str(parsed.get("summary", "N/A"))
                rational = str(parsed.get("rational", ""))
                attempts.append(
                    {
                        "ok": True,
                        "mode": "visit_extractor",
                        "max_doc_chars": current_max_doc_chars,
                        "goal_chars": len(query),
                        "content_chars": len(contents),
                        "user_prompt_chars": len(user_prompt),
                    }
                )
                return LiteExtractorAttemptResult(
                    evidence=evidence,
                    summary=summary,
                    rational=rational,
                    used_backend="sglang",
                    used_max_doc_chars=current_max_doc_chars,
                    mode="visit_extractor",
                    attempts=attempts,
                )
            except Exception as e:
                attempts.append(
                    {
                        "ok": False,
                        "mode": "visit_extractor",
                        "max_doc_chars": current_max_doc_chars,
                        "goal_chars": len(query),
                        "user_prompt_chars": len(user_prompt),
                        "error": str(e),
                    }
                )
                print(
                    f"[visit-extractor-retry] goal={query[:80]!r} max_doc_chars={current_max_doc_chars} "
                    f"user_prompt_chars={len(user_prompt)} error={e}"
                )

    fallback_chars = min(retry_chars[-1] if retry_chars else max_doc_chars, extractive_fallback_doc_chars)
    fallback = _build_visit_extractive_fallback(goal=query, doc=doc, max_doc_chars=fallback_chars)
    fallback.attempts = attempts + fallback.attempts
    return fallback


def _summarize_with_fallback(
    *,
    query: str,
    docs: list[dict[str, Any]],
    scores: Optional[list[float]],
    request_max_doc_chars: int,
) -> SummaryAttemptResult:
    attempts: list[dict[str, Any]] = []
    retry_chars = _parse_retry_doc_chars(summarize_retry_max_doc_chars, request_max_doc_chars)

    if sglang_model and sglang_base_urls:
        for max_doc_chars in retry_chars:
            docs_text = _format_docs_for_llm(docs, scores, max_doc_chars)
            user_prompt = _build_summary_user_prompt(query, docs_text)
            try:
                summary = _call_sglang_summarizer_multi(
                    base_urls=sglang_base_urls,
                    model=sglang_model,
                    api_key=sglang_api_key,
                    system_prompt=sglang_system_prompt,
                    user_prompt=user_prompt,
                    timeout=sglang_timeout,
                    temperature=sglang_temperature,
                    top_p=sglang_top_p,
                    max_tokens=sglang_max_tokens,
                )
                attempts.append(
                    {
                        "ok": True,
                        "mode": "sglang",
                        "max_doc_chars": max_doc_chars,
                        "query_chars": len(query),
                        "docs_text_chars": len(docs_text),
                        "user_prompt_chars": len(user_prompt),
                    }
                )
                return SummaryAttemptResult(
                    summary=summary,
                    used_backend="sglang",
                    used_max_doc_chars=max_doc_chars,
                    mode="sglang",
                    attempts=attempts,
                )
            except Exception as e:
                attempts.append(
                    {
                        "ok": False,
                        "mode": "sglang",
                        "max_doc_chars": max_doc_chars,
                        "query_chars": len(query),
                        "docs_text_chars": len(docs_text),
                        "user_prompt_chars": len(user_prompt),
                        "error": str(e),
                    }
                )
                print(
                    f"[summarize-retry] query={query[:80]!r} max_doc_chars={max_doc_chars} "
                    f"docs_text_chars={len(docs_text)} user_prompt_chars={len(user_prompt)} error={e}"
                )

    fallback_chars = min(retry_chars[-1] if retry_chars else request_max_doc_chars, extractive_fallback_doc_chars)
    summary = _build_extractive_fallback(query=query, docs=docs, scores=scores, max_doc_chars=fallback_chars)
    attempts.append(
        {
            "ok": True,
            "mode": "extractive_fallback",
            "max_doc_chars": fallback_chars,
            "query_chars": len(query),
        }
    )
    return SummaryAttemptResult(
        summary=summary,
        used_backend=None,
        used_max_doc_chars=fallback_chars,
        mode="extractive_fallback",
        attempts=attempts,
    )


if FastAPI is not None:
    app = FastAPI()
else:  # pragma: no cover
    class _DummyApp:
        def post(self, *args, **kwargs):
            def deco(fn):
                return fn

            return deco

    app = _DummyApp()


class RetrieveRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


class RetrieveSummarizeCompatRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = True
    max_doc_chars: Optional[int] = None


class RetrieveSummarizeRequest(BaseModel):
    query: str
    topk: Optional[int] = None
    return_scores: bool = True
    summarize: bool = True
    max_doc_chars: Optional[int] = None


class LiteSearchRequest(BaseModel):
    # The search prompt emits {"query": [...]}, while verl's existing
    # SearchTool transport posts {"queries": [...]}. Accept both so this
    # endpoint can be used by the model prompt and by the verl tool wrapper.
    query: Optional[Any] = None
    queries: Optional[list[str]] = None
    topk: Optional[int] = None
    return_scores: bool = True
    frontier_enable: bool = False
    frontier_exclude_docids: Optional[list[str]] = None
    frontier_base_rank_cutoff: int = 10
    frontier_output_topk: int = 10
    frontier_candidate_topk: int = 50
    frontier_overlap_threshold: float = 0.90
    frontier_compact_snippet_chars: Optional[int] = None
    reasoning: str = ""


class RankSearchRequest(BaseModel):
    query: Optional[Any] = None
    queries: Optional[list[str]] = None
    topk: Optional[int] = None
    return_scores: bool = True


class LiteBrowseRequest(BaseModel):
    url: Any
    goal: str = ""
    max_doc_chars: Optional[int] = None
    summarize: bool = True


class OpenDocumentRequest(BaseModel):
    url: Optional[Any] = None
    docid: Optional[str] = None
    id: Optional[Any] = None
    cursor: Optional[str] = None
    loc: Optional[int] = None
    num_lines: Optional[int] = None
    goal: str = ""


class FindDocumentRequest(BaseModel):
    cursor: Optional[str] = None
    url: Optional[Any] = None
    docid: Optional[str] = None
    pattern: str
    context_lines: Optional[int] = None
    max_hits: Optional[int] = None


config: Optional[Config] = None
retriever: Any = None

sglang_base_urls: list[str] = []
sglang_model: Optional[str] = None
sglang_api_key: Optional[str] = None
sglang_system_prompt: Optional[str] = None
sglang_timeout: float = 60.0
sglang_temperature: float = 0.2
sglang_top_p: float = 0.9
sglang_max_tokens: int = 256
default_max_doc_chars: int = 2000
summarize_retry_max_doc_chars: Optional[str] = None
extractive_fallback_doc_chars: int = 500
lite_search_snippet_chars: int = 120
visit_evidence_max_chars: int = 1200
visit_summary_max_chars: int = 600
search_card_enable: bool = False
search_card_budget_per_doc: int = 700
search_card_mode: str = "inventory_chunks"
search_card_chunk_chars: int = 900
search_card_overlap_chars: int = 150
search_card_max_entity_items: int = 40
search_card_rich_topk: int = -1
search_global_enable: bool = False
search_global_output_topk: int = 200
search_global_rank_cutoff: int = 50
search_global_rerank_mode: str = "rrf"
search_global_rich_topk: int = 5
search_global_rich_budget: int = 500
search_global_rich_mode: str = "inventory_chunks"
search_global_compact_mode: str = "title_inventory"
search_global_compact_snippet_chars: int = 80
search_global_max_output_chars: int = 32000
dual_retrieval_enable: bool = False
dual_secondary_rank_url: str = ""
dual_candidate_topk: int = 50
dual_rrf_k: int = 60
dual_primary_weight: float = 1.0
dual_secondary_weight: float = 2.0
dual_primary_rerank_mode: str = "rrf"
dual_secondary_timeout: float = 120.0
visit_externalize_enable: bool = False
visit_externalize_budget_chars: int = 8000
visit_externalize_mode: str = "inventory_chunks"
visit_externalize_chunk_chars: int = 1200
visit_externalize_overlap_chars: int = 200
visit_externalize_max_entity_items: int = 80
visit_externalize_with_summary: bool = True
search_result_link_label: str = "VisitURL"
search_result_open_tool_name: str = "visit"
browser_line_chars: int = 160
browser_default_num_lines: int = 40
browser_max_num_lines: int = 80
browser_find_context_lines: int = 1
browser_find_max_hits: int = 10


@app.post("/retrieve")
def retrieve_endpoint(request: RetrieveRequest):
    if retriever is None or config is None:
        raise HTTPException(status_code=500, detail="server 未初始化（retriever/config 为空）")
    if not request.queries:
        raise HTTPException(status_code=400, detail="queries 不能为空")

    topk = request.topk or config.retrieval_topk
    results, scores = retriever.batch_search(query_list=request.queries, num=topk, return_score=request.return_scores)

    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            combined = []
            for doc, score in zip(single_result, scores[i], strict=True):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


@app.post("/rank")
def rank_search_endpoint(request: RankSearchRequest):
    """Return a compact route-level ranking for a dual-retrieval gateway."""

    start = time.time()
    if retriever is None or config is None:
        raise HTTPException(status_code=500, detail="server 未初始化（retriever/config 为空）")
    raw_queries = request.query if request.query is not None else request.queries
    queries = _normalize_lite_values(raw_queries or [])
    if not queries:
        return {
            "success": False,
            "ranked_docids": [],
            "elapsed_ms": 0.0,
            "error": "query/queries is empty",
        }
    try:
        topk = max(1, int(request.topk or config.retrieval_topk))
        results, scores = retriever.batch_search(
            query_list=queries,
            num=topk,
            return_score=True,
        )
        ranked = _rank_route_records(
            queries=queries,
            results=results,
            scores=scores,
            rank_cutoff=topk,
            rerank_mode=search_global_rerank_mode,
        )[:topk]
        return {
            "success": True,
            "ranked_docids": [
                _normalize_frontier_docid(_doc_key(rec.get("doc") or {})) for rec in ranked
            ],
            "candidate_count": len(ranked),
            "elapsed_ms": (time.time() - start) * 1000,
            "error": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "ranked_docids": [],
            "elapsed_ms": (time.time() - start) * 1000,
            "error": str(exc),
        }


@app.post("/search")
def search_endpoint(request: LiteSearchRequest):
    """Local search endpoint compatible with the configured search tool.

    Accepts {"query": str | list[str]} and returns a Serper-like text block
    containing titles, snippets, and doc:// identifiers that can be passed to
    the visit tool.
    """
    start = time.time()
    if retriever is None or config is None:
        raise HTTPException(status_code=500, detail="server 未初始化（retriever/config 为空）")
    raw_queries = request.query if request.query is not None else request.queries
    queries = _normalize_lite_values(raw_queries or [])
    if not queries:
        return {"success": False, "result": "", "elapsed_ms": 0.0, "error": "query/queries is empty"}

    try:
        dual_error = ""
        if dual_retrieval_enable and str(request.reasoning or "").strip():
            candidate_topk = max(
                int(dual_candidate_topk),
                int(request.frontier_candidate_topk),
                int(request.topk or config.retrieval_topk),
            )
            primary_results, primary_scores = retriever.batch_search(
                query_list=queries,
                num=candidate_topk,
                return_score=True,
            )
            primary_ranked = _rank_route_records(
                queries=queries,
                results=primary_results,
                scores=primary_scores,
                rank_cutoff=candidate_topk,
                rerank_mode=dual_primary_rerank_mode,
            )[:candidate_topk]
            reasoning = re.sub(r"\s+", " ", str(request.reasoning or "")).strip()
            secondary_queries = [
                f"Reasoning: {reasoning}\nQuery: {query}" for query in queries
            ]
            try:
                secondary_docids = _secondary_ranked_docids(
                    queries=secondary_queries,
                    topk=candidate_topk,
                )
                fused_ranked = _fuse_dual_ranked_records(
                    primary_ranked=primary_ranked,
                    secondary_docids=secondary_docids,
                    corpus=getattr(retriever, "corpus", None),
                    output_topk=candidate_topk,
                    rrf_k=dual_rrf_k,
                    primary_weight=dual_primary_weight,
                    secondary_weight=dual_secondary_weight,
                    queries=queries,
                )
                result_text, frontier_metadata = _format_dual_ranked_result(
                    queries=queries,
                    ranked=fused_ranked,
                    frontier_enable=bool(request.frontier_enable),
                    frontier_exclude_docids=request.frontier_exclude_docids,
                    frontier_output_topk=request.frontier_output_topk,
                    frontier_overlap_threshold=request.frontier_overlap_threshold,
                    frontier_compact_snippet_chars=request.frontier_compact_snippet_chars,
                )
                elapsed = (time.time() - start) * 1000
                return {
                    "success": True,
                    "result": result_text,
                    "elapsed_ms": elapsed,
                    "error": "",
                    "frontier": frontier_metadata,
                    "fusion": {
                        "enabled": True,
                        "applied": True,
                        "primary_candidates": len(primary_ranked),
                        "secondary_candidates": len(secondary_docids),
                        "fused_candidates": len(fused_ranked),
                        "rrf_k": dual_rrf_k,
                        "primary_weight": dual_primary_weight,
                        "secondary_weight": dual_secondary_weight,
                    },
                }
            except Exception as exc:
                # Preserve the established 59-point route if AgentIR is down.
                dual_error = f"{type(exc).__name__}: {exc}"
                print(f"[dual-retrieval-fallback] {dual_error}", flush=True)

        topk = request.topk or config.retrieval_topk
        results, scores = retriever.batch_search(query_list=queries, num=topk, return_score=request.return_scores)
        frontier_metadata: dict[str, Any] = {
            "enabled": False,
            "applied": False,
            "exhausted": False,
        }
        if search_global_enable:
            result_text, frontier_metadata = _format_lite_search_global_result(
                queries=queries,
                results=results,
                scores=scores if request.return_scores else None,
                frontier_enable=bool(request.frontier_enable),
                frontier_exclude_docids=request.frontier_exclude_docids,
                frontier_base_rank_cutoff=request.frontier_base_rank_cutoff,
                frontier_output_topk=request.frontier_output_topk,
                frontier_candidate_topk=request.frontier_candidate_topk,
                frontier_overlap_threshold=request.frontier_overlap_threshold,
                frontier_compact_snippet_chars=request.frontier_compact_snippet_chars,
            )
        else:
            blocks: list[str] = []
            for idx, query in enumerate(queries):
                docs = results[idx] if idx < len(results) else []
                score_list = scores[idx] if request.return_scores and scores is not None and idx < len(scores) else None
                blocks.append(_format_lite_search_result(query=query, docs=docs, scores=score_list))
            result_text = "\n=======\n".join(blocks)
        elapsed = (time.time() - start) * 1000
        return {
            "success": True,
            "result": result_text,
            "elapsed_ms": elapsed,
            "error": "",
            "frontier": frontier_metadata,
            "fusion": {
                "enabled": bool(dual_retrieval_enable),
                "applied": False,
                "fallback_error": dual_error,
            },
        }
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return {"success": False, "result": "", "elapsed_ms": elapsed, "error": str(e)}


@app.post("/browse")
def browse_endpoint(request: LiteBrowseRequest):
    """Local browse endpoint compatible with the configured open tool.

    Accepts {"url": str | list[str], "goal": str}. URLs may be doc://<docid>,
    raw docids, or corpus URLs. The response uses the established browser
    evidence format.
    """
    start = time.time()
    if retriever is None or config is None:
        raise HTTPException(status_code=500, detail="server 未初始化（retriever/config 为空）")
    if getattr(retriever, "corpus", None) is None:
        raise HTTPException(status_code=500, detail="retriever 缺少 corpus，无法 browse")

    urls = _normalize_lite_values(request.url)
    if not urls:
        return {"success": False, "result": "", "elapsed_ms": 0.0, "error": "url is empty"}

    max_doc_chars = default_max_doc_chars if request.max_doc_chars is None else int(request.max_doc_chars)
    goal = str(request.goal or "").strip()
    blocks: list[str] = []
    errors: list[str] = []
    for raw_url in urls:
        docid = _extract_docid_from_lite_url(raw_url)
        doc = retriever.corpus.get_doc(docid)
        if doc is None:
            msg = f"The provided webpage/document could not be accessed: {raw_url}"
            errors.append(msg)
            blocks.append(
                f"The useful information in {raw_url} for user goal {goal} as follows: \n\n"
                "Evidence in page: \nThe provided webpage content could not be accessed.\n\n"
                "Summary: \nThe webpage content could not be processed.\n"
            )
            continue
        blocks.append(
            _format_visit_result(
                raw_url=raw_url,
                doc=doc,
                goal=goal,
                max_doc_chars=max_doc_chars,
                summarize=bool(request.summarize),
            )
        )

    elapsed = (time.time() - start) * 1000
    return {
        "success": len(blocks) > 0,
        "result": "\n=======\n".join(blocks).strip(),
        "elapsed_ms": elapsed,
        "error": " | ".join(errors),
    }


def _request_docid(*values: Any) -> str:
    for value in values:
        normalized = _normalize_lite_values(value)
        if normalized:
            return _extract_docid_from_lite_url(normalized[0])
    return ""


@app.post("/open")
def open_document_endpoint(request: OpenDocumentRequest):
    """Open a deterministic line window from a corpus document."""
    start_time = time.time()
    if retriever is None or config is None or getattr(retriever, "corpus", None) is None:
        raise HTTPException(status_code=500, detail="server/retriever corpus 未初始化")
    docid = _request_docid(request.cursor, request.url, request.docid, request.id)
    if not docid:
        return {"success": False, "result": "", "elapsed_ms": 0.0, "error": "cursor/url/docid is empty"}
    doc = retriever.corpus.get_doc(docid)
    if doc is None:
        return {
            "success": False,
            "result": "",
            "elapsed_ms": (time.time() - start_time) * 1000,
            "error": f"document not found: {docid}",
        }
    result = _format_open_document(
        doc=doc,
        loc=request.loc,
        num_lines=request.num_lines,
        goal=str(request.goal or "").strip(),
    )
    return {
        "success": True,
        "result": result,
        "cursor": f"doc://{docid}",
        "elapsed_ms": (time.time() - start_time) * 1000,
        "error": "",
    }


@app.post("/find")
def find_document_endpoint(request: FindDocumentRequest):
    """Find exact text inside a corpus document and return line-addressed hits."""
    start_time = time.time()
    if retriever is None or config is None or getattr(retriever, "corpus", None) is None:
        raise HTTPException(status_code=500, detail="server/retriever corpus 未初始化")
    docid = _request_docid(request.cursor, request.url, request.docid)
    if not docid:
        return {"success": False, "result": "", "elapsed_ms": 0.0, "error": "cursor/url/docid is empty"}
    doc = retriever.corpus.get_doc(docid)
    if doc is None:
        return {
            "success": False,
            "result": "",
            "elapsed_ms": (time.time() - start_time) * 1000,
            "error": f"document not found: {docid}",
        }
    result = _format_find_results(
        doc=doc,
        pattern=request.pattern,
        context_lines=request.context_lines,
        max_hits=request.max_hits,
    )
    return {
        "success": True,
        "result": result,
        "cursor": f"doc://{docid}",
        "elapsed_ms": (time.time() - start_time) * 1000,
        "error": "",
    }


@app.post("/retrieve_summarize")
def retrieve_and_summarize_endpoint(request: RetrieveSummarizeRequest):
    if retriever is None or config is None:
        raise HTTPException(status_code=500, detail="server 未初始化（retriever/config 为空）")

    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    topk = request.topk or config.retrieval_topk
    docs, scores = retriever.search(query=query, num=topk, return_score=True)
    max_doc_chars = default_max_doc_chars if request.max_doc_chars is None else int(request.max_doc_chars)

    summary_result = SummaryAttemptResult(summary="", used_backend=None, used_max_doc_chars=max_doc_chars, mode="disabled", attempts=[])
    if request.summarize:
        summary_result = _summarize_with_fallback(
            query=query,
            docs=docs,
            scores=scores if request.return_scores else None,
            request_max_doc_chars=max_doc_chars,
        )

    result_text = f"Your query is: {query}. The search results are summarized as following: {summary_result.summary}".strip()
    return {
        "result": result_text,
        "summary": summary_result.summary,
        "summary_meta": {
            "mode": summary_result.mode,
            "used_max_doc_chars": summary_result.used_max_doc_chars,
            "attempts": summary_result.attempts,
        },
        "retrieval": {
            "query": query,
            "topk": topk,
            "docs": docs,
            "scores": scores if request.return_scores else None,
        },
    }


@app.post("/retrieve_summarize_compat")
def retrieve_summarize_compat_endpoint(request: RetrieveSummarizeCompatRequest):
    if retriever is None or config is None:
        raise HTTPException(status_code=500, detail="server 未初始化（retriever/config 为空）")
    if not request.queries:
        raise HTTPException(status_code=400, detail="queries 不能为空")

    topk = request.topk or config.retrieval_topk
    all_results: list[list[dict[str, Any]]] = []
    for q in request.queries:
        query = (q or "").strip()
        if not query:
            all_results.append([])
            continue

        docs, scores = retriever.search(query=query, num=topk, return_score=True)
        max_doc_chars = default_max_doc_chars if request.max_doc_chars is None else int(request.max_doc_chars)
        summary_result = _summarize_with_fallback(
            query=query,
            docs=docs,
            scores=scores if request.return_scores else None,
            request_max_doc_chars=max_doc_chars,
        )

        contents = f"Your query is: {query}. The search results are summarized as following: {summary_result.summary}".strip()
        all_results.append([
            {
                "document": {
                    "id": "summary",
                    "contents": contents,
                    "summary_mode": summary_result.mode,
                    "used_max_doc_chars": summary_result.used_max_doc_chars,
                },
                "score": 0.0,
            }
        ])
    return {"result": all_results}


@app.post("/retrieve_summarize_raw", response_class=PlainTextResponse)
def retrieve_and_summarize_raw_endpoint(request: RetrieveSummarizeRequest):
    payload = retrieve_and_summarize_endpoint(request)
    if isinstance(payload, dict) and isinstance(payload.get("result"), str):
        return payload["result"]
    return str(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone BrowseComp-Plus retriever with resilient summarization and short-title handling.")
    parser.add_argument("--retrieval_method", choices=["bm25", "dense"], default="dense")
    parser.add_argument("--indexes_repo_id", type=str, default="Tevatron/browsecomp-plus-indexes")
    parser.add_argument("--index_name", type=str, default="qwen3-embedding-0.6b")
    parser.add_argument("--index_path", type=str, default=None, help="Optional local index dir, overrides auto-download.")
    parser.add_argument("--corpus_repo_id", type=str, default="Tevatron/browsecomp-plus-corpus")
    parser.add_argument("--corpus_split", type=str, default="train")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional HF cache dir.")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_docs", type=int, default=-1, help="For smoke test / debug, only load first N docs.")

    parser.add_argument("--query_encoder_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--query_encoder_batch_size", type=int, default=32)
    parser.add_argument("--query_instruction", type=str, default=None)
    parser.add_argument(
        "--query_encoder_backend",
        choices=["sentence_transformers", "transformer_eos"],
        default="sentence_transformers",
        help=(
            "Query embedding backend. transformer_eos uses left padding and the final token, "
            "as required by AgentIR-4B."
        ),
    )
    parser.add_argument("--query_encoder_max_length", type=int, default=8192)
    parser.add_argument("--faiss_gpu", action="store_true")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_doc_chars", type=int, default=2000)
    parser.add_argument(
        "--max_title_chars",
        type=int,
        default=120,
        help="Summarize prompt 与 fallback 中每篇文档 title 的最大字符数。",
    )
    parser.add_argument(
        "--summarize_retry_max_doc_chars",
        type=str,
        default=None,
        help="逗号分隔的重试裁剪阶梯，例如 '4000,2500,1500,800'；默认会自动附加 2000/1000/500/250。",
    )
    parser.add_argument(
        "--extractive_fallback_doc_chars",
        type=int,
        default=500,
        help="所有 SGLang summarize 尝试失败后，extractive fallback 每篇文档保留的最大字符数。",
    )
    parser.add_argument(
        "--lite_search_snippet_chars",
        type=int,
        default=120,
        help="/search 每条结果正文 snippet 的最大字符数。",
    )
    parser.add_argument(
        "--visit_evidence_max_chars",
        type=int,
        default=1200,
        help="/browse 返回给模型的 Evidence 最大字符数。",
    )
    parser.add_argument(
        "--visit_summary_max_chars",
        type=int,
        default=600,
        help="/browse 返回给模型的 Summary 最大字符数。",
    )
    parser.add_argument(
        "--search_card_enable",
        action="store_true",
        help="开启 /search source-card 外显：candidate answer-like spans + query-aware snippets。",
    )
    parser.add_argument(
        "--search_card_budget_per_doc",
        type=int,
        default=700,
        help="/search source-card 每条结果最大字符数；建议 512/700/1024。",
    )
    parser.add_argument(
        "--search_card_mode",
        choices=["chunks", "inventory_chunks", "inventory_head_chunks"],
        default="inventory_chunks",
        help="/search source-card 模式。res5 中 chunks/inventory_chunks 优于 head。",
    )
    parser.add_argument("--search_card_chunk_chars", type=int, default=900)
    parser.add_argument("--search_card_overlap_chars", type=int, default=150)
    parser.add_argument("--search_card_max_entity_items", type=int, default=40)
    parser.add_argument(
        "--search_card_rich_topk",
        type=int,
        default=-1,
        help="只对前 N 条结果使用 rich source-card；<=0 表示全部 topk 都 rich。",
    )
    parser.add_argument(
        "--search_global_enable",
        action="store_true",
        help="开启 /search 多 query 全局去重/rerank/budgeted 输出，用于 global top200 title/title_inventory。",
    )
    parser.add_argument("--search_global_output_topk", type=int, default=200)
    parser.add_argument("--search_global_rank_cutoff", type=int, default=50)
    parser.add_argument(
        "--search_global_rerank_mode",
        choices=["rrf", "min_rank", "rrf_lexical"],
        default="rrf",
    )
    parser.add_argument("--search_global_rich_topk", type=int, default=5)
    parser.add_argument("--search_global_rich_budget", type=int, default=500)
    parser.add_argument(
        "--search_global_rich_mode",
        choices=["chunks", "inventory_chunks", "inventory_head_chunks"],
        default="inventory_chunks",
    )
    parser.add_argument(
        "--search_global_compact_mode",
        choices=["title", "title_inventory", "query", "snippet"],
        default="title_inventory",
    )
    parser.add_argument("--search_global_compact_snippet_chars", type=int, default=80)
    parser.add_argument("--search_global_max_output_chars", type=int, default=32000)
    parser.add_argument(
        "--dual_retrieval_enable",
        action="store_true",
        help="Fuse local raw-query retrieval with a remote reasoning-aware /rank endpoint.",
    )
    parser.add_argument("--dual_secondary_rank_url", type=str, default="")
    parser.add_argument("--dual_candidate_topk", type=int, default=50)
    parser.add_argument("--dual_rrf_k", type=int, default=60)
    parser.add_argument("--dual_primary_weight", type=float, default=1.0)
    parser.add_argument("--dual_secondary_weight", type=float, default=2.0)
    parser.add_argument(
        "--dual_primary_rerank_mode",
        choices=["rrf", "min_rank", "rrf_lexical"],
        default="rrf",
    )
    parser.add_argument("--dual_secondary_timeout", type=float, default=120.0)
    parser.add_argument(
        "--search_result_link_label",
        choices=["VisitURL", "OpenURL"],
        default="VisitURL",
        help="Label used for stable doc:// links in search results.",
    )
    parser.add_argument(
        "--search_result_open_tool_name",
        choices=["visit", "open"],
        default="visit",
        help="Tool name advertised by search results for source inspection.",
    )
    parser.add_argument("--browser_line_chars", type=int, default=160)
    parser.add_argument("--browser_default_num_lines", type=int, default=40)
    parser.add_argument("--browser_max_num_lines", type=int, default=80)
    parser.add_argument("--browser_find_context_lines", type=int, default=1)
    parser.add_argument("--browser_find_max_hits", type=int, default=10)
    parser.add_argument(
        "--visit_externalize_enable",
        action="store_true",
        help="开启 /browse high-recall 外显：candidate answer-like spans + selected raw chunks。",
    )
    parser.add_argument(
        "--visit_externalize_budget_chars",
        type=int,
        default=8000,
        help="/browse 外显 Evidence 最大字符数；res6 建议优先试 8000。",
    )
    parser.add_argument(
        "--visit_externalize_mode",
        choices=["chunks", "inventory_chunks", "inventory_head_chunks"],
        default="inventory_chunks",
        help="/browse 外显模式；res6 中 inventory_chunks/inventory_head_chunks 效果最好。",
    )
    parser.add_argument("--visit_externalize_chunk_chars", type=int, default=1200)
    parser.add_argument("--visit_externalize_overlap_chars", type=int, default=200)
    parser.add_argument("--visit_externalize_max_entity_items", type=int, default=80)
    parser.add_argument(
        "--visit_externalize_no_summary",
        action="store_true",
        help="开启 visit externalization 时不再调用 LLM summary，只返回外显 evidence。",
    )

    parser.add_argument(
        "--sglang_base_url",
        type=str,
        default=None,
        help="单个 SGLang base URL，或逗号分隔的多个 base URL（会轮询并在失败时自动切换）。",
    )
    parser.add_argument("--sglang_model", type=str, default=None)
    parser.add_argument("--sglang_api_key", type=str, default=None)
    parser.add_argument("--sglang_system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--sglang_timeout", type=float, default=60.0)
    parser.add_argument("--sglang_temperature", type=float, default=0.2)
    parser.add_argument("--sglang_top_p", type=float, default=0.9)
    parser.add_argument("--sglang_max_tokens", type=int, default=12000)

    args = parser.parse_args()

    short_title_max_chars = int(args.max_title_chars)

    _require(uvicorn, "uvicorn")
    _require(FastAPI, "fastapi")

    config = Config(
        retrieval_method=args.retrieval_method,
        retrieval_topk=args.topk,
        indexes_repo_id=args.indexes_repo_id,
        index_name=args.index_name,
        index_path=args.index_path,
        corpus_repo_id=args.corpus_repo_id,
        corpus_split=args.corpus_split,
        cache_dir=args.cache_dir,
        query_encoder_model=args.query_encoder_model,
        query_encoder_batch_size=args.query_encoder_batch_size,
        query_instruction=args.query_instruction,
        query_encoder_backend=args.query_encoder_backend,
        query_encoder_max_length=args.query_encoder_max_length,
        faiss_gpu=args.faiss_gpu,
        max_docs=args.max_docs,
    )
    retriever = get_retriever(config)

    sglang_base_urls = _parse_sglang_base_urls(args.sglang_base_url)
    sglang_model = args.sglang_model
    sglang_api_key = args.sglang_api_key
    sglang_system_prompt = args.sglang_system_prompt
    sglang_timeout = args.sglang_timeout
    sglang_temperature = args.sglang_temperature
    sglang_top_p = args.sglang_top_p
    sglang_max_tokens = args.sglang_max_tokens
    default_max_doc_chars = args.max_doc_chars
    summarize_retry_max_doc_chars = args.summarize_retry_max_doc_chars
    extractive_fallback_doc_chars = args.extractive_fallback_doc_chars
    lite_search_snippet_chars = args.lite_search_snippet_chars
    visit_evidence_max_chars = args.visit_evidence_max_chars
    visit_summary_max_chars = args.visit_summary_max_chars
    search_card_enable = bool(args.search_card_enable)
    search_card_budget_per_doc = int(args.search_card_budget_per_doc)
    search_card_mode = args.search_card_mode
    search_card_chunk_chars = int(args.search_card_chunk_chars)
    search_card_overlap_chars = int(args.search_card_overlap_chars)
    search_card_max_entity_items = int(args.search_card_max_entity_items)
    search_card_rich_topk = int(args.search_card_rich_topk)
    search_global_enable = bool(args.search_global_enable)
    search_global_output_topk = int(args.search_global_output_topk)
    search_global_rank_cutoff = int(args.search_global_rank_cutoff)
    search_global_rerank_mode = args.search_global_rerank_mode
    search_global_rich_topk = int(args.search_global_rich_topk)
    search_global_rich_budget = int(args.search_global_rich_budget)
    search_global_rich_mode = args.search_global_rich_mode
    search_global_compact_mode = args.search_global_compact_mode
    search_global_compact_snippet_chars = int(args.search_global_compact_snippet_chars)
    search_global_max_output_chars = int(args.search_global_max_output_chars)
    dual_retrieval_enable = bool(args.dual_retrieval_enable)
    dual_secondary_rank_url = str(args.dual_secondary_rank_url or "").strip()
    dual_candidate_topk = max(1, int(args.dual_candidate_topk))
    dual_rrf_k = max(0, int(args.dual_rrf_k))
    dual_primary_weight = max(0.0, float(args.dual_primary_weight))
    dual_secondary_weight = max(0.0, float(args.dual_secondary_weight))
    dual_primary_rerank_mode = args.dual_primary_rerank_mode
    dual_secondary_timeout = max(1.0, float(args.dual_secondary_timeout))
    if dual_retrieval_enable and not dual_secondary_rank_url:
        parser.error("--dual_retrieval_enable requires --dual_secondary_rank_url")
    search_result_link_label = args.search_result_link_label
    search_result_open_tool_name = args.search_result_open_tool_name
    browser_line_chars = max(40, int(args.browser_line_chars))
    browser_default_num_lines = max(1, int(args.browser_default_num_lines))
    browser_max_num_lines = max(browser_default_num_lines, int(args.browser_max_num_lines))
    browser_find_context_lines = max(0, int(args.browser_find_context_lines))
    browser_find_max_hits = max(1, int(args.browser_find_max_hits))
    visit_externalize_enable = bool(args.visit_externalize_enable)
    visit_externalize_budget_chars = int(args.visit_externalize_budget_chars)
    visit_externalize_mode = args.visit_externalize_mode
    visit_externalize_chunk_chars = int(args.visit_externalize_chunk_chars)
    visit_externalize_overlap_chars = int(args.visit_externalize_overlap_chars)
    visit_externalize_max_entity_items = int(args.visit_externalize_max_entity_items)
    visit_externalize_with_summary = not bool(args.visit_externalize_no_summary)

    uvicorn.run(app, host=args.host, port=args.port)
