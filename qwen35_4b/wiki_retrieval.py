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
"""Wiki E5 search/open/find/Frontier for protected SFT trace collection.
Retrieval and browser behavior are preserved; unused summarization routes removed.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
import time
from typing import Any, Optional

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def load_docs(corpus, doc_idxs):
    return [corpus[int(idx)] for idx in doc_idxs]


def load_model(model_path: str, use_fp16: bool = False):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pooling_method == "cls":
        return last_hidden_state[:, 0]
    if pooling_method == "pooler":
        return pooler_output
    raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: list[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower() and is_query:
            query_list = [
                f"Represent this sentence for searching relevant passages: {query}" for query in query_list
            ]

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy().astype(np.float32, order="C")

        del inputs, output
        torch.cuda.empty_cache()
        return query_emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: list[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        enriched_results = []
        for doc, idx in zip(results, idxs, strict=False):
            if isinstance(doc, dict):
                item = dict(doc)
            else:
                item = {"contents": str(doc)}
            item.setdefault("docid", str(int(idx)))
            item.setdefault("id", str(int(idx)))
            enriched_results.append(item)
        return (enriched_results, scores.tolist()) if return_score else enriched_results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Retrieval process: "):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            enriched_batch_results = []
            for doc, idx in zip(batch_results, flat_idxs, strict=False):
                if isinstance(doc, dict):
                    item = dict(doc)
                else:
                    item = {"contents": str(doc)}
                item.setdefault("docid", str(int(idx)))
                item.setdefault("id", str(int(idx)))
                enriched_batch_results.append(item)
            batch_results = [enriched_batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()

        return (results, scores) if return_score else results


class Config:
    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


# ------------------------
# SGLang summarize helpers
# ------------------------


def _truncate_text(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


# ------------------------
# FastAPI
# ------------------------


class LiteSearchRequest(BaseModel):
    query: Optional[Any] = None
    queries: Optional[list[str]] = None
    query_list: Optional[list[str]] = None
    topk: Optional[int] = None
    return_scores: bool = True
    frontier_enable: bool = False
    frontier_exclude_docids: Optional[list[str]] = None
    frontier_base_rank_cutoff: int = 10
    frontier_output_topk: int = 10
    frontier_candidate_topk: int = 50
    frontier_overlap_threshold: float = 0.90
    frontier_compact_snippet_chars: Optional[int] = None


class OpenDocumentRequest(BaseModel):
    cursor: Optional[Any] = None
    url: Optional[Any] = None
    docid: Optional[str] = None
    id: Optional[str] = None
    loc: Optional[int] = None
    num_lines: Optional[int] = None
    goal: str = ""


class FindDocumentRequest(BaseModel):
    cursor: Optional[Any] = None
    url: Optional[Any] = None
    docid: Optional[str] = None
    pattern: str
    context_lines: Optional[int] = None
    max_hits: Optional[int] = None


def _normalize_query_list(*values: Any) -> list[str]:
    queries: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, list):
            candidates = value
        else:
            candidates = [str(value)]
        for item in candidates:
            text = str(item or "").strip()
            if text:
                queries.append(text)
    return queries


def _doc_contents(doc: dict[str, Any]) -> str:
    contents = doc.get("contents")
    if contents is not None:
        return str(contents)
    title = str(doc.get("title") or "").strip()
    text = str(doc.get("text") or "").strip()
    if title or text:
        return (title + "\n" + text).strip()
    return json.dumps(doc, ensure_ascii=False)


def _doc_title(doc: dict[str, Any]) -> str:
    title = str(doc.get("title") or "").strip()
    if title:
        return _truncate_text(re.sub(r"\s+", " ", title), 120)
    contents = _doc_contents(doc)
    first = contents.split("\n", 1)[0].strip() if contents else "Untitled"
    return _truncate_text(re.sub(r"\s+", " ", first), 120)


def _docid(doc: dict[str, Any]) -> str:
    return str(doc.get("docid") or doc.get("id") or "").strip()


def _lite_doc_url(doc: dict[str, Any]) -> str:
    did = _docid(doc)
    if did:
        return f"doc://{did}"
    url = str(doc.get("url") or "").strip()
    return url or "doc://unknown"


def _snippet_from_doc(doc: dict[str, Any], max_chars: Optional[int] = None) -> str:
    if max_chars is None:
        max_chars = lite_search_snippet_chars
    contents = _doc_contents(doc)
    if "\n" in contents:
        contents = contents.split("\n", 1)[1]
    snippet = re.sub(r"\s+", " ", contents).strip()
    return _truncate_text(snippet, max_chars=max_chars)


def _format_lite_search_result(query: str, docs: list[dict[str, Any]], scores: Optional[list[float]] = None) -> str:
    lines = [
        f"A local corpus search for '{query}' found {len(docs)} results:",
        "Use doc:// cursors with open (or the legacy visit tool). Results:",
    ]
    for idx, doc in enumerate(docs, start=1):
        did = _docid(doc)
        docid_part = f"DocID: {did} | " if did else ""
        lines.append(
            f"{idx}. {_doc_title(doc)} | VisitURL: {_lite_doc_url(doc)} | {docid_part}Snippet: {_snippet_from_doc(doc)}"
        )
    return "\n".join(lines).strip()


def _normalize_frontier_docid(value: Any) -> str:
    text = str(value or "").strip().rstrip(".,;:)]}")
    if text.lower().startswith("doc://"):
        text = text[len("doc://") :]
    elif text.lower().startswith("docid:"):
        text = text.split(":", 1)[1]
    return text.strip().strip("/")


def _rank_frontier_records(
    queries: list[str],
    results: list[list[dict[str, Any]]],
    scores: list[list[float]],
    rank_cutoff: int,
) -> list[dict[str, Any]]:
    """Merge query variants without letting rank 11-50 perturb the base page."""

    by_docid: dict[str, dict[str, Any]] = {}
    for query_index, query in enumerate(queries):
        query_docs = results[query_index] if query_index < len(results) else []
        query_scores = scores[query_index] if query_index < len(scores) else []
        for rank, doc in enumerate(query_docs[: max(1, rank_cutoff)], start=1):
            docid = _normalize_frontier_docid(_docid(doc))
            if not docid:
                continue
            record = by_docid.get(docid)
            score = query_scores[rank - 1] if rank - 1 < len(query_scores) else None
            if record is None:
                record = {
                    "doc": doc,
                    "docid": docid,
                    "rank": rank,
                    "queries": [],
                    "rrf": 0.0,
                    "score": score,
                }
                by_docid[docid] = record
            record["queries"].append(query)
            record["rrf"] += 1.0 / (60.0 + rank)
            if rank < record["rank"]:
                record["doc"] = doc
                record["rank"] = rank
                record["score"] = score
    return sorted(
        by_docid.values(),
        key=lambda item: (len(item["queries"]), item["rrf"], -(item["rank"])),
        reverse=True,
    )


def _format_frontier_search_result(
    *,
    queries: list[str],
    results: list[list[dict[str, Any]]],
    scores: list[list[float]],
    request: LiteSearchRequest,
) -> tuple[str, dict[str, Any]]:
    base_cutoff = max(1, int(request.frontier_base_rank_cutoff))
    candidate_cutoff = max(base_cutoff, int(request.frontier_candidate_topk))
    output_topk = max(1, int(request.frontier_output_topk))
    threshold = min(1.0, max(0.0, float(request.frontier_overlap_threshold)))
    base_ranked = _rank_frontier_records(queries, results, scores, base_cutoff)
    full_ranked = _rank_frontier_records(queries, results, scores, candidate_cutoff)[
        :candidate_cutoff
    ]
    base_page = base_ranked[:output_topk]
    excluded = {
        docid
        for value in (request.frontier_exclude_docids or [])
        if (docid := _normalize_frontier_docid(value))
    }
    base_docids = [record["docid"] for record in base_page]
    base_overlap = (
        sum(docid in excluded for docid in base_docids) / len(base_docids)
        if base_docids
        else 0.0
    )
    applied = bool(excluded and base_overlap >= threshold)
    unseen = [record for record in full_ranked if record["docid"] not in excluded]
    selected = unseen[:output_topk] if applied else base_page
    exhausted = bool(applied and not selected)

    if exhausted:
        header = [
            "SEARCH_FRONTIER_EXHAUSTED",
            "No unseen candidates remain. Open an existing doc:// cursor or search a genuinely new clue.",
        ]
    elif applied:
        header = [
            "SEARCH_FRONTIER_PAGE",
            f"Returning up to {output_topk} previously unseen candidates. Open promising doc:// cursors before repeating this search.",
            "Results:",
        ]
    else:
        header = [
            f"A local corpus search for {len(queries)} query variant(s) found {len(base_ranked)} base candidates.",
            "Open promising doc:// cursors to inspect full evidence. Results:",
        ]
    snippet_chars = request.frontier_compact_snippet_chars
    if snippet_chars is None:
        snippet_chars = lite_search_snippet_chars
    lines: list[str] = []
    for rank, record in enumerate(selected, start=1):
        doc = record["doc"]
        lines.append(
            f"{rank}. {_doc_title(doc)} | Cursor: {_lite_doc_url(doc)} | "
            f"DocID: {record['docid']} | Snippet: {_snippet_from_doc(doc, max_chars=max(0, int(snippet_chars)))}"
        )
    selected = selected[: len(lines)]
    metadata = {
        "enabled": True,
        "applied": applied,
        "exhausted": exhausted,
        "base_overlap": base_overlap,
        "base_rank_cutoff": base_cutoff,
        "full_rank_cutoff": candidate_cutoff,
        "base_candidate_count": len(base_ranked),
        "candidate_count": len(full_ranked),
        "unseen_count": len(unseen),
        "excluded_count": len(excluded),
        "selected_count": len(selected),
        "selected_docids": [record["docid"] for record in selected],
        "selected_original_ranks": [int(record["rank"]) for record in selected],
    }
    return "\n".join(header + lines).strip(), metadata


def _extract_docid_from_lite_url(raw_url: str) -> str:
    value = str(raw_url or "").strip()
    if value.startswith("doc://"):
        return value[len("doc://") :].strip().strip("/")
    if value.startswith("docid:"):
        return value.split(":", 1)[1].strip()
    if value.isdigit():
        return value
    return ""


def _load_doc_by_id(docid: str) -> dict[str, Any]:
    if retriever is None or not hasattr(retriever, "corpus"):
        raise ValueError("retriever corpus is not available")
    doc = retriever.corpus[int(docid)]
    if isinstance(doc, dict):
        item = dict(doc)
    else:
        item = {"contents": str(doc)}
    item.setdefault("docid", str(docid))
    item.setdefault("id", str(docid))
    return item


def _document_lines(doc: dict[str, Any], width: int = 160) -> list[str]:
    lines: list[str] = []
    for raw_line in _doc_contents(doc).splitlines() or [""]:
        normalized = re.sub(r"\s+", " ", raw_line).strip()
        if not normalized:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.extend(
            textwrap.wrap(
                normalized,
                width=max(40, width),
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [normalized]
        )
    while lines and lines[-1] == "":
        lines.pop()
    return lines or [""]


def _best_open_loc(lines: list[str], goal: str, num_lines: int) -> int:
    terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", str(goal or ""))
        if token.lower() not in {"the", "and", "for", "from", "with", "that", "this", "find"}
    }
    if not terms:
        return 0
    window = max(1, min(num_lines, len(lines)))
    scores = [sum(term in line.lower() for term in terms) for line in lines]
    running = sum(scores[:window])
    best_score, best_start = running, 0
    for start in range(1, max(1, len(lines) - window + 1)):
        running += scores[start + window - 1] - scores[start - 1]
        if running > best_score:
            best_score, best_start = running, start
    return best_start if best_score else 0


def _format_open_document(
    doc: dict[str, Any],
    *,
    loc: Optional[int],
    num_lines: Optional[int],
    goal: str,
) -> str:
    cursor = _lite_doc_url(doc)
    lines = _document_lines(doc)
    count = max(1, min(int(num_lines or 20), 100))
    if loc is None or int(loc) < 0:
        start = _best_open_loc(lines, goal, count)
        positioned = "query-relevant"
    else:
        start = min(max(0, int(loc)), max(0, len(lines) - 1))
        positioned = "explicit"
    end = min(len(lines), start + count)
    body = "\n".join(f"L{index}: {lines[index]}" for index in range(start, end))
    next_hint = f" Next: open cursor={cursor!r} loc={end}." if end < len(lines) else " End of document."
    return (
        f"Cursor: {cursor}\nTitle: {_doc_title(doc)}\n"
        f"Showing lines {start}-{max(start, end - 1)} of {len(lines)} ({positioned} window).{next_hint}\n"
        f"{body}"
    )


def _format_find_results(
    doc: dict[str, Any],
    *,
    pattern: str,
    context_lines: Optional[int],
    max_hits: Optional[int],
) -> str:
    cursor = _lite_doc_url(doc)
    lines = _document_lines(doc)
    needle = re.sub(r"\s+", " ", str(pattern or "")).strip().lower()
    if not needle:
        return f"Find error: pattern is empty. Cursor: {cursor}"
    indices = [index for index, line in enumerate(lines) if needle in line.lower()]
    if not indices:
        return f"Cursor: {cursor}\nTitle: {_doc_title(doc)}\nPattern {pattern!r} was not found."
    radius = max(0, min(int(context_lines or 1), 5))
    limit = max(1, min(int(max_hits or 8), 20))
    blocks: list[str] = []
    for hit_no, index in enumerate(indices[:limit], start=1):
        lo, hi = max(0, index - radius), min(len(lines), index + radius + 1)
        body = "\n".join(f"L{line_no}: {lines[line_no]}" for line_no in range(lo, hi))
        blocks.append(f"Hit {hit_no} at line {index}:\n{body}")
    return (
        f"Cursor: {cursor}\nTitle: {_doc_title(doc)}\n"
        f"Found {len(indices)} line(s) containing {pattern!r}; showing {min(len(indices), limit)}.\n"
        + "\n---\n".join(blocks)
    )


app = FastAPI()


# Runtime globals initialized in __main__.
config: Optional[Config] = None
retriever: Any = None

lite_search_snippet_chars: int = 120


@app.get("/health")
def health():
    # Observability only: never changes search/ranking/open behavior.
    import hashlib
    from pathlib import Path
    source = Path(__file__).resolve()
    return {"success": retriever is not None, "runtime": {
        "corpus_source_path": str(Path(config.corpus_path).resolve()) if config else "",
        "index_path": str(Path(config.index_path).resolve()) if config else "",
        "query_encoder_model": str(Path(config.retrieval_model_path).resolve()) if config else "",
        "server_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "service_kind": "wiki_e5",
    }}


@app.post("/search")
def search_endpoint(request: LiteSearchRequest):
    if retriever is None or config is None:
        raise HTTPException(status_code=500, detail="server 未初始化（retriever/config 为空）")
    queries = _normalize_query_list(request.query, request.queries, request.query_list)
    if not queries:
        raise HTTPException(status_code=400, detail="query/queries 不能为空")
    requested_topk = request.topk or config.retrieval_topk
    retrieval_topk = max(
        requested_topk,
        int(request.frontier_candidate_topk) if request.frontier_enable else 0,
    )
    results, scores = retriever.batch_search(
        query_list=queries,
        num=retrieval_topk,
        return_score=True,
    )
    if request.frontier_enable:
        rendered, frontier = _format_frontier_search_result(
            queries=queries,
            results=results,
            scores=scores,
            request=request,
        )
        return {
            "success": True,
            "result": rendered,
            "frontier": frontier,
            "error": "",
        }
    blocks = [
        _format_lite_search_result(
            query,
            docs[:requested_topk],
            scores[index][:requested_topk] if request.return_scores else None,
        )
        for index, (query, docs) in enumerate(zip(queries, results, strict=True))
    ]
    return {"success": True, "result": "\n\n=======\n\n".join(blocks), "error": ""}


def _request_docid(*values: Any) -> str:
    for value in values:
        normalized = _normalize_query_list(value)
        if not normalized:
            continue
        docid = _extract_docid_from_lite_url(normalized[0])
        if docid:
            return docid
    return ""


@app.post("/open")
def open_document_endpoint(request: OpenDocumentRequest):
    """Open a deterministic, line-addressed Wiki-18 document window."""

    started = time.time()
    if retriever is None or config is None:
        raise HTTPException(status_code=500, detail="server 未初始化（retriever/config 为空）")
    docid = _request_docid(request.cursor, request.url, request.docid, request.id)
    if not docid:
        return {"success": False, "result": "", "error": "cursor/url/docid is empty"}
    try:
        doc = _load_doc_by_id(docid)
        result = _format_open_document(
            doc,
            loc=request.loc,
            num_lines=request.num_lines,
            goal=str(request.goal or "").strip(),
        )
        return {
            "success": True,
            "result": result,
            "cursor": f"doc://{docid}",
            "elapsed_ms": (time.time() - started) * 1000,
            "error": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "result": "",
            "elapsed_ms": (time.time() - started) * 1000,
            "error": f"document not found: {docid}: {exc}",
        }


@app.post("/find")
def find_document_endpoint(request: FindDocumentRequest):
    """Find a literal pattern in a Wiki-18 document without an LLM extractor."""

    started = time.time()
    if retriever is None or config is None:
        raise HTTPException(status_code=500, detail="server 未初始化（retriever/config 为空）")
    docid = _request_docid(request.cursor, request.url, request.docid)
    if not docid:
        return {"success": False, "result": "", "error": "cursor/url/docid is empty"}
    try:
        doc = _load_doc_by_id(docid)
        result = _format_find_results(
            doc,
            pattern=request.pattern,
            context_lines=request.context_lines,
            max_hits=request.max_hits,
        )
        return {
            "success": True,
            "result": result,
            "cursor": f"doc://{docid}",
            "elapsed_ms": (time.time() - started) * 1000,
            "error": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "result": "",
            "elapsed_ms": (time.time() - started) * 1000,
            "error": f"document not found: {docid}: {exc}",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index_path', required=True)
    parser.add_argument('--corpus_path', required=True)
    parser.add_argument('--retriever_name', choices=['e5'], default='e5')
    parser.add_argument('--retriever_model', required=True)
    parser.add_argument('--faiss_gpu', action='store_true')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=1249)
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--lite_search_snippet_chars', type=int, default=800)
    args = parser.parse_args()
    config = Config(retrieval_method=args.retriever_name, index_path=args.index_path,
                    corpus_path=args.corpus_path, retrieval_topk=args.topk, faiss_gpu=args.faiss_gpu,
                    retrieval_model_path=args.retriever_model, retrieval_pooling_method='mean',
                    retrieval_query_max_length=256, retrieval_use_fp16=True, retrieval_batch_size=512)
    retriever = DenseRetriever(config)
    lite_search_snippet_chars = args.lite_search_snippet_chars
    uvicorn.run(app, host=args.host, port=args.port)
