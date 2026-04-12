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
"""Standalone retrieval server with optional SGLang summarization.

目标：一个“单文件”服务，不依赖同目录的 `retrieval_server.py`。

能力：
- `/retrieve`：和 Search-R1-like 的 retrieval server 保持一致（批量 queries）。
- `/retrieve_summarize`：对单条 query 先检索 topk，再调用 SGLang（OpenAI-compatible）模型服务做 summarize。

启动示例：
  python3 retrieval_server_sglang_summarize.py \
    --index_path /path/to/index \
    --corpus_path /path/to/corpus.jsonl \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu \
    --sglang_base_url http://127.0.0.1:30000 \
    --sglang_model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import warnings
from typing import Any, Optional

import datasets
import faiss
import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
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


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            return ([], []) if return_score else []

        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!", stacklevel=2)
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results = [
                {
                    "title": content.split("\n")[0].strip('"'),
                    "text": "\n".join(content.split("\n")[1:]),
                    "contents": content,
                }
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        return (results, scores) if return_score else results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        return (results, scores) if return_score else results


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
        return (results, scores.tolist()) if return_score else results

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
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()

        return (results, scores) if return_score else results


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    return DenseRetriever(config)


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


def _build_chat_completions_url(base_url: str) -> str:
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        raise ValueError("sglang_base_url 不能为空")
    if base_url.endswith("/v1/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


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
        if not isinstance(doc, dict):
            doc = {"contents": str(doc)}

        contents = doc.get("contents")
        if contents is None:
            title = doc.get("title")
            text = doc.get("text")
            if title or text:
                contents = (str(title or "").strip() + "\n" + str(text or "").strip()).strip()
            else:
                contents = json.dumps(doc, ensure_ascii=False)

        contents = _truncate_text(str(contents), max_chars=max_doc_chars)
        title = contents.split("\n", 1)[0].strip() if contents else ""
        body = contents.split("\n", 1)[1].strip() if "\n" in contents else ""

        score_part = ""
        if scores is not None and i < len(scores):
            try:
                score_part = f" (score={float(scores[i]):.4f})"
            except Exception:
                score_part = ""

        parts.append(f"Doc {i + 1}{score_part} (Title: {title})\n{body}".strip())
    return "\n\n".join([p for p in parts if p])


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
        "chat_template_kwargs": {"enable_thinking": False}
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return _extract_chat_completion_content(resp.json()).strip()


# ------------------------
# FastAPI
# ------------------------


class RetrieveRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


class RetrieveSummarizeCompatRequest(BaseModel):
    """与 Search-R1 `SearchTool` 完全兼容的请求 schema。

    SearchTool 的请求 payload 是：
      {"queries": [...], "topk": <int>, "return_scores": true}
    """

    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = True
    max_doc_chars: int = 2000


class RetrieveSummarizeRequest(BaseModel):
    query: str
    topk: Optional[int] = None
    return_scores: bool = True

    summarize: bool = True
    max_doc_chars: int = 2000


app = FastAPI()


# Runtime globals initialized in __main__.
config: Optional[Config] = None
retriever: Any = None

sglang_base_url: Optional[str] = None
sglang_model: Optional[str] = None
sglang_api_key: Optional[str] = None
sglang_system_prompt: Optional[str] = None
sglang_timeout: float = 60.0
sglang_temperature: float = 0.2
sglang_top_p: float = 0.9
sglang_max_tokens: int = 256


@app.post("/retrieve")
def retrieve_endpoint(request: RetrieveRequest):
    if retriever is None or config is None:
        raise HTTPException(status_code=500, detail="server 未初始化（retriever/config 为空）")
    if not request.queries:
        raise HTTPException(status_code=400, detail="queries 不能为空")

    topk = request.topk or config.retrieval_topk
    results, scores = retriever.batch_search(
        query_list=request.queries, num=topk, return_score=request.return_scores
    )

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


@app.post("/retrieve_summarize")
def retrieve_and_summarize_endpoint(request: RetrieveSummarizeRequest):
    if retriever is None or config is None:
        raise HTTPException(status_code=500, detail="server 未初始化（retriever/config 为空）")

    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    topk = request.topk or config.retrieval_topk
    docs, scores = retriever.search(query=query, num=topk, return_score=True)

    docs_text = _format_docs_for_llm(docs, scores if request.return_scores else None, request.max_doc_chars)
    user_prompt = (
        "You are given a user query and top retrieved documents. "
        "Write a concise, factual summary grounded ONLY in the retrieved documents.\n\n"
        f"User query: {query}\n\n"
        f"Retrieved documents:\n{docs_text}\n\n"
        "Requirements:\n"
        "- Cover the key facts (who/what/when/where) if present in the docs.\n"
        "- Do NOT invent facts not supported by the docs.\n"
        "- End with one sentence explaining why the results match the query."
    )

    summary = ""
    if request.summarize:
        if not sglang_model or not sglang_base_url:
            raise HTTPException(status_code=500, detail="未配置 SGLang summarizer（--sglang_base_url / --sglang_model）")
        try:
            summary = _call_sglang_summarizer(
                base_url=sglang_base_url,
                model=sglang_model,
                api_key=sglang_api_key,
                system_prompt=sglang_system_prompt,
                user_prompt=user_prompt,
                timeout=sglang_timeout,
                temperature=sglang_temperature,
                top_p=sglang_top_p,
                max_tokens=sglang_max_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"调用 SGLang summarizer 失败: {e}")

    result_text = f"Your query is: {query}. The search results are summarized as following: {summary}".strip()

    return {
        "result": result_text,
        "summary": summary,
        "retrieval": {
            "query": query,
            "topk": topk,
            "docs": docs,
            "scores": scores if request.return_scores else None,
        },
    }


@app.post("/retrieve_summarize_compat")
def retrieve_summarize_compat_endpoint(request: RetrieveSummarizeCompatRequest):
    """Search-R1 训练可直接替换的 retrieval endpoint。

    - 入参：与 `/retrieve` 完全一致（queries/topk/return_scores）。
    - 出参：保持 `/retrieve` 的 JSON shape：{"result": [per_query_results...]}
      但每个 query 只返回 1 条“伪 doc”，其 `document.contents` 为 summary 字符串。

    这样 `verl.tools.search_tool.SearchTool` 不用改代码，只要把
    `retrieval_service_url` 指到本 endpoint 即可。
    """

    if retriever is None or config is None:
        raise HTTPException(status_code=500, detail="server 未初始化（retriever/config 为空）")
    if not request.queries:
        raise HTTPException(status_code=400, detail="queries 不能为空")
    if not sglang_model or not sglang_base_url:
        raise HTTPException(status_code=500, detail="未配置 SGLang summarizer（--sglang_base_url / --sglang_model）")

    topk = request.topk or config.retrieval_topk
    all_results: list[list[dict[str, Any]]] = []

    for q in request.queries:
        query = (q or "").strip()
        if not query:
            all_results.append([])
            continue

        docs, scores = retriever.search(query=query, num=topk, return_score=True)
        docs_text = _format_docs_for_llm(docs, scores, request.max_doc_chars)
        user_prompt = (
            "You are given a user query and top retrieved documents. "
            "Write a concise, factual summary grounded ONLY in the retrieved documents.\n\n"
            f"User query: {query}\n\n"
            f"Retrieved documents:\n{docs_text}\n\n"
            "Requirements:\n"
            "- Cover the key facts (who/what/when/where) if present in the docs.\n"
            "- Do NOT invent facts not supported by the docs.\n"
            "- End with one sentence explaining why the results match the query."
        )

        try:
            summary = _call_sglang_summarizer(
                base_url=sglang_base_url,
                model=sglang_model,
                api_key=sglang_api_key,
                system_prompt=sglang_system_prompt,
                user_prompt=user_prompt,
                timeout=sglang_timeout,
                temperature=sglang_temperature,
                top_p=sglang_top_p,
                max_tokens=sglang_max_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"调用 SGLang summarizer 失败: {e}")

        contents = f"Your query is: {query}. The search results are summarized as following: {summary}".strip()

        # 兼容 SearchTool：每个 query 返回 list[ {document, score}, ... ]
        # 这里返回 1 条即可。
        all_results.append(
            [
                {
                    "document": {"id": "summary", "contents": contents},
                    "score": 0.0,
                }
            ]
        )

    return {"result": all_results}


@app.post("/retrieve_summarize_raw", response_class=PlainTextResponse)
def retrieve_and_summarize_raw_endpoint(request: RetrieveSummarizeRequest):
    """Same as `/retrieve_summarize` but returns plain text (NOT JSON)."""

    # Reuse the JSON endpoint logic and only return the `result` string.
    payload = retrieve_and_summarize_endpoint(request)
    if isinstance(payload, dict) and isinstance(payload.get("result"), str):
        return payload["result"]
    # Fallback
    return str(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone local retriever with optional SGLang summarizer.")

    parser.add_argument(
        "--index_path",
        type=str,
        default="data/e5_Flat.index",
        help="Corpus indexing file.",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data/wiki-18.jsonl",
        help="Local corpus file.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="intfloat/e5-base-v2",
        help="Path of the retriever model.",
    )
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for computation")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")

    # SGLang summarizer args
    parser.add_argument(
        "--sglang_base_url",
        type=str,
        default=None,
        help="SGLang OpenAI-compatible base URL, e.g. http://127.0.0.1:30000",
    )
    parser.add_argument("--sglang_model", type=str, default=None, help="Model name served by SGLang")
    parser.add_argument("--sglang_api_key", type=str, default=None, help="Optional API key")
    parser.add_argument(
        "--sglang_system_prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt for summarizer",
    )
    parser.add_argument("--sglang_timeout", type=float, default=60.0)
    parser.add_argument("--sglang_temperature", type=float, default=0.2)
    parser.add_argument("--sglang_top_p", type=float, default=0.9)
    parser.add_argument("--sglang_max_tokens", type=int, default=256)

    args = parser.parse_args()

    # init globals
    config = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )
    retriever = get_retriever(config)

    sglang_base_url = args.sglang_base_url
    sglang_model = args.sglang_model
    sglang_api_key = args.sglang_api_key
    sglang_system_prompt = args.sglang_system_prompt
    sglang_timeout = args.sglang_timeout
    sglang_temperature = args.sglang_temperature
    sglang_top_p = args.sglang_top_p
    sglang_max_tokens = args.sglang_max_tokens

    uvicorn.run(app, host=args.host, port=args.port)
