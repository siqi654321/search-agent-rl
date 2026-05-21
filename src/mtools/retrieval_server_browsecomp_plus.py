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
import os
import pickle
import re
import threading
import time
from dataclasses import dataclass
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
    def __init__(self, model_name_or_path: str, batch_size: int = 32, query_instruction: Optional[str] = None):
        _require(SentenceTransformer, "sentence-transformers")
        _require(np, "numpy")
        self.model = SentenceTransformer(model_name_or_path)
        self.batch_size = batch_size
        self.query_instruction = query_instruction

    def encode_queries(self, queries: list[str]) -> "np.ndarray":
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
        raise ValueError(f"visit extractor returned non-object JSON: {type(parsed)}")
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
    values = [value] if isinstance(value, str) else list(value or [])
    return [str(v).strip() for v in values if str(v).strip()]


def _lite_doc_url(doc: dict[str, Any]) -> str:
    docid = str(doc.get("docid") or doc.get("id") or "").strip()
    url = str(doc.get("url") or "").strip()
    # In the local BrowseComp-Plus corpus, docid is the stable address.  Keep
    # the field name stays as "url" to match the tool schema, but expose
    # doc:// links in search results so the model copies deterministic local
    # identifiers instead of long web URLs that later need reverse lookup.
    if docid:
        return f"doc://{docid}"
    if url:
        return url
    return "doc://unknown"


def _snippet_from_doc(doc: dict[str, Any], max_chars: Optional[int] = None) -> str:
    if max_chars is None:
        max_chars = lite_search_snippet_chars
    contents = str(doc.get("contents") or doc.get("text") or "")
    if "\n" in contents:
        contents = contents.split("\n", 1)[1]
    snippet = re.sub(r"\s+", " ", contents).strip()
    return _truncate_text(snippet, max_chars=max_chars)


def _format_lite_search_result(query: str, docs: list[dict[str, Any]], scores: Optional[list[float]] = None) -> str:
    # Keep the search response compact so top-10 results fit under verl's
    # max_tool_response_length.  Expose only doc:// identifiers to discourage
    # the model from copying long real URLs; /browse can visit doc:// directly.
    lines = [
        f"A local corpus search for '{query}' found {len(docs)} results:",
        "Use VisitURL values with the visit tool. Results:",
    ]
    for idx, doc in enumerate(docs, start=1):
        docid = str(doc.get("docid") or doc.get("id") or "").strip()
        title = _shorten_title(str(doc.get("title") or docid or "Untitled"), short_title_max_chars)
        visit_url = _lite_doc_url(doc)
        snippet = _snippet_from_doc(doc)
        docid_part = f"DocID: {docid} | " if docid else ""
        lines.append(f"{idx}. {title} | VisitURL: {visit_url} | {docid_part}Snippet: {snippet}".strip())
    return "\n".join(lines).strip()


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

    if summarize:
        extracted = _summarize_visit_doc_with_extractor(goal=goal, doc=doc, max_doc_chars=max_doc_chars)
        # For long multi-turn rollouts (50-100 tool interactions), bound each
        # visit response.  The extractor still sees max_doc_chars of page text,
        # but the model only receives compact evidence/summary.
        evidence = _truncate_text(str(extracted.evidence), visit_evidence_max_chars)
        summary = _truncate_text(str(extracted.summary), visit_summary_max_chars)
    else:
        evidence = _truncate_text(body, max_doc_chars)
        summary = "Summarization is disabled. The relevant page content is provided in Evidence in page above."

    # Keep the existing browser output contract so the
    # model sees the same Evidence/Summary framing as it was trained/evaluated on.
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
    # The prompt may emit {"query": [...]}, while verl's existing
    # SearchTool transport posts {"queries": [...]}. Accept both so this
    # endpoint can be used by the model prompt and by the verl tool wrapper.
    query: Optional[Any] = None
    queries: Optional[list[str]] = None
    topk: Optional[int] = None
    return_scores: bool = True


class LiteBrowseRequest(BaseModel):
    url: Any
    goal: str = ""
    max_doc_chars: Optional[int] = None
    summarize: bool = True


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


@app.post("/search")
def mtool_search_endpoint(request: LiteSearchRequest):
    """Search+visit-compatible local search endpoint.

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
        topk = request.topk or config.retrieval_topk
        results, scores = retriever.batch_search(query_list=queries, num=topk, return_score=request.return_scores)
        blocks: list[str] = []
        for idx, query in enumerate(queries):
            docs = results[idx] if idx < len(results) else []
            score_list = scores[idx] if request.return_scores and scores is not None and idx < len(scores) else None
            blocks.append(_format_lite_search_result(query=query, docs=docs, scores=score_list))
        elapsed = (time.time() - start) * 1000
        return {"success": True, "result": "\n=======\n".join(blocks), "elapsed_ms": elapsed, "error": ""}
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return {"success": False, "result": "", "elapsed_ms": elapsed, "error": str(e)}


@app.post("/browse")
def mtool_browse_endpoint(request: LiteBrowseRequest):
    """Search+visit-compatible local visit endpoint.

    Accepts {"url": str | list[str], "goal": str}. URLs may be doc://<docid>,
    raw docids, or corpus URLs. The response shape mirrors the existing
    browser_server.py.
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

    uvicorn.run(app, host=args.host, port=args.port)
