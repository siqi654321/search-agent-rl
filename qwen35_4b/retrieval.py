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
"""Dense BrowseComp-Plus server for the fixed dual-retrieval recipe.

Only /health, /rank, /search, /open and /find are served. Safe allow-list,
Frontier, ranking, source-card formatting and browser windows are preserved.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import gc
import glob
import hashlib
import json
import math
import os
import pickle
import re
import threading
import time
import textwrap
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


SERVER_SOURCE_PATH = os.path.abspath(__file__)
SERVER_SOURCE_SHA256 = _sha256_file(SERVER_SOURCE_PATH)

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
    import pyarrow as pa  # type: ignore
    import pyarrow.dataset as pa_dataset  # type: ignore
    import pyarrow.parquet as pa_parquet  # type: ignore
except Exception:  # pragma: no cover
    pa = None
    pa_dataset = None
    pa_parquet = None

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
except Exception:  # pragma: no cover
    FastAPI = None
    HTTPException = Exception

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


def _natural_sort_key(value: str) -> list[tuple[int, Any]]:
    """Sort shard-2 before shard-10 while keeping mixed path pieces stable."""

    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", str(value))
    ]


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
        docid_is_row_index: bool = False,
        row_lookup_mode: str = "dataset_take",
        doc_cache_size: int = 0,
        row_group_read_workers: int = 1,
    ):
        self.docid_is_row_index = bool(docid_is_row_index)
        self.row_lookup_mode = str(row_lookup_mode)
        if self.row_lookup_mode not in {"dataset_take", "parquet_row_group"}:
            raise ValueError(
                "row_lookup_mode must be one of {'dataset_take', 'parquet_row_group'}, "
                f"got {self.row_lookup_mode!r}"
            )
        self.doc_cache_size = max(0, int(doc_cache_size))
        self._doc_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._doc_cache_lock = threading.Lock()
        self.row_group_read_workers = max(1, int(row_group_read_workers))
        self._row_group_executor: Optional[ThreadPoolExecutor] = None
        self.arrow_dataset = None
        self.dataset = None
        self._parquet_files: list[str] = []
        self._parquet_file_starts: list[int] = []
        self._parquet_file_row_counts: list[int] = []
        self._parquet_row_group_starts: list[list[int]] = []
        self.parquet_file_count = 0
        self.parquet_row_group_count = 0
        self.parquet_max_row_group_rows = 0
        self.parquet_mean_row_group_rows = 0.0
        local_parquet_files: list[str] = []
        expanded = os.path.expanduser(corpus_repo_id)
        if os.path.isfile(expanded) and expanded.casefold().endswith(".parquet"):
            local_parquet_files = [expanded]
        elif os.path.isdir(expanded):
            local_parquet_files = sorted(
                glob.glob(os.path.join(expanded, "**", "*.parquet"), recursive=True),
                key=_natural_sort_key,
            )

        if self.row_lookup_mode == "parquet_row_group" and not (
            self.docid_is_row_index and local_parquet_files
        ):
            raise ValueError(
                "row_lookup_mode='parquet_row_group' requires "
                "docid_is_row_index=True and a local Parquet file/directory"
            )

        if self.docid_is_row_index and local_parquet_files and self.row_lookup_mode == "parquet_row_group":
            total = self._initialize_parquet_row_lookup(local_parquet_files)
            if self.row_group_read_workers > 1:
                self._row_group_executor = ThreadPoolExecutor(
                    max_workers=self.row_group_read_workers,
                    thread_name_prefix="corpus-row-group",
                )
        elif self.docid_is_row_index and local_parquet_files:
            _require(pa, "pyarrow")
            _require(pa_dataset, "pyarrow.dataset")
            # Reading the downloaded Parquet shards directly avoids creating a
            # second ~30 GB Arrow cache for OpenResearcher-Corpus.
            self.arrow_dataset = pa_dataset.dataset(local_parquet_files, format="parquet")
            total = int(self.arrow_dataset.count_rows())
        else:
            _require(datasets, "datasets")
            if local_parquet_files:
                self.dataset = datasets.load_dataset(
                    "parquet",
                    data_files=local_parquet_files,
                    split=split,
                    cache_dir=cache_dir,
                )
            else:
                self.dataset = datasets.load_dataset(corpus_repo_id, split=split, cache_dir=cache_dir)
            total = len(self.dataset)
        limit = total if max_docs is None or max_docs <= 0 else min(total, int(max_docs))
        self.docid_to_idx: dict[str, int] = {}
        self.size = limit
        if self.docid_is_row_index:
            probes = sorted({0, 1, max(0, limit // 2), max(0, limit - 1)}) if limit else []
            rows = self._take_direct(probes)
            for idx, row in zip(probes, rows):
                if str(row.get("docid")) != str(idx):
                    raise RuntimeError(
                        "corpus_docid_is_row_index validation failed: "
                        f"row={idx} docid={row.get('docid')!r}"
                    )
            print(
                f"[CORPUS] direct row-index mode enabled: rows={self.size} "
                f"probes={len(probes)} lookup={self.row_lookup_mode} "
                f"doc_cache_size={self.doc_cache_size} "
                f"row_group_read_workers={self.row_group_read_workers}",
                flush=True,
            )
        else:
            for idx in range(limit):
                row = self.dataset[idx]
                self.docid_to_idx[str(row.get("docid"))] = idx

    def _initialize_parquet_row_lookup(self, parquet_files: list[str]) -> int:
        """Build an in-memory global-row to Parquet row-group offset map.

        ``pyarrow.dataset.Dataset.take`` locates global row indices by scanning
        through the dataset.  On the 14.9M-row OpenResearcher corpus that makes
        latency grow with the numeric docid.  Parquet footer metadata already
        contains every file and row-group length, so prefix offsets let us
        locate only the row groups that contain the requested documents.
        """

        _require(pa, "pyarrow")
        _require(pa_parquet, "pyarrow.parquet")
        total = 0
        total_row_groups = 0
        max_row_group_rows = 0
        for path in parquet_files:
            parquet_file = pa_parquet.ParquetFile(path)
            try:
                metadata = parquet_file.metadata
                file_rows = int(metadata.num_rows)
                row_group_starts: list[int] = []
                local_start = 0
                for row_group_index in range(int(metadata.num_row_groups)):
                    row_group_starts.append(local_start)
                    row_group_rows = int(metadata.row_group(row_group_index).num_rows)
                    local_start += row_group_rows
                    max_row_group_rows = max(max_row_group_rows, row_group_rows)
                if local_start != file_rows:
                    raise RuntimeError(
                        f"Parquet metadata row mismatch for {path}: "
                        f"row_groups={local_start} file_rows={file_rows}"
                    )
            finally:
                close = getattr(parquet_file, "close", None)
                if callable(close):
                    close()

            if file_rows <= 0:
                continue

            self._parquet_files.append(path)
            self._parquet_file_starts.append(total)
            self._parquet_file_row_counts.append(file_rows)
            self._parquet_row_group_starts.append(row_group_starts)
            total += file_rows
            total_row_groups += len(row_group_starts)

        if total <= 0:
            raise RuntimeError("No rows found in local Parquet corpus")
        self.parquet_file_count = len(self._parquet_files)
        self.parquet_row_group_count = total_row_groups
        self.parquet_max_row_group_rows = max_row_group_rows
        self.parquet_mean_row_group_rows = total / total_row_groups if total_row_groups else 0.0
        print(
            f"[CORPUS] parquet row-group map ready: files={len(self._parquet_files)} "
            f"row_groups={total_row_groups} rows={total}",
            flush=True,
        )
        return total

    def _locate_parquet_row(self, index: int) -> tuple[int, int, int]:
        file_index = bisect_right(self._parquet_file_starts, index) - 1
        if file_index < 0 or file_index >= len(self._parquet_files):
            raise IndexError(f"corpus row index out of range: {index}")
        file_start = self._parquet_file_starts[file_index]
        local_index = index - file_start
        if local_index < 0 or local_index >= self._parquet_file_row_counts[file_index]:
            raise IndexError(f"corpus row index out of range: {index}")
        row_group_starts = self._parquet_row_group_starts[file_index]
        row_group_index = bisect_right(row_group_starts, local_index) - 1
        if row_group_index < 0:
            raise RuntimeError(f"failed to locate row group for corpus row {index}")
        row_in_group = local_index - row_group_starts[row_group_index]
        return file_index, row_group_index, row_in_group

    def _take_parquet_row_groups(self, indices: list[int]) -> list[dict[str, Any]]:
        """Read requested global rows while touching only their row groups."""

        grouped: dict[int, dict[int, list[tuple[int, int]]]] = {}
        for output_position, index in enumerate(indices):
            file_index, row_group_index, row_in_group = self._locate_parquet_row(int(index))
            grouped.setdefault(file_index, {}).setdefault(row_group_index, []).append(
                (output_position, row_in_group)
            )

        work_items = list(grouped.items())
        if self._row_group_executor is not None:
            futures = [
                self._row_group_executor.submit(self._read_parquet_file_rows, file_index, row_groups)
                for file_index, row_groups in work_items
            ]
            materialized = [future.result() for future in futures]
        else:
            materialized = [
                self._read_parquet_file_rows(file_index, row_groups)
                for file_index, row_groups in work_items
            ]

        output: list[Optional[dict[str, Any]]] = [None] * len(indices)
        for rows_for_file in materialized:
            for output_position, row in rows_for_file:
                output[output_position] = row

        if any(row is None for row in output):
            raise RuntimeError("Parquet row-group lookup did not materialize every requested row")
        return [row for row in output if row is not None]

    def _read_parquet_file_rows(
        self,
        file_index: int,
        row_groups: dict[int, list[tuple[int, int]]],
    ) -> list[tuple[int, dict[str, Any]]]:
        """Read all requested row groups from one file with a single file open."""

        parquet_file = pa_parquet.ParquetFile(self._parquet_files[file_index])
        materialized: list[tuple[int, dict[str, Any]]] = []
        try:
            for row_group_index, requests_for_group in row_groups.items():
                table = parquet_file.read_row_group(
                    row_group_index,
                    columns=["docid", "text", "url"],
                    use_threads=True,
                )
                offsets = pa.array(
                    [row_in_group for _, row_in_group in requests_for_group],
                    type=pa.int64(),
                )
                selected_rows = table.take(offsets).to_pylist()
                materialized.extend(
                    (output_position, row)
                    for (output_position, _), row in zip(requests_for_group, selected_rows)
                )
        finally:
            close = getattr(parquet_file, "close", None)
            if callable(close):
                close()
        return materialized

    def _get_cached_doc(self, docid: str) -> Optional[dict[str, Any]]:
        if self.doc_cache_size <= 0:
            return None
        with self._doc_cache_lock:
            doc = self._doc_cache.get(docid)
            if doc is not None:
                self._doc_cache.move_to_end(docid)
            return doc

    def _put_cached_doc(self, docid: str, doc: dict[str, Any]) -> None:
        if self.doc_cache_size <= 0:
            return
        with self._doc_cache_lock:
            self._doc_cache[docid] = doc
            self._doc_cache.move_to_end(docid)
            while len(self._doc_cache) > self.doc_cache_size:
                self._doc_cache.popitem(last=False)

    def _take_direct(self, indices: list[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        if self._parquet_files:
            return self._take_parquet_row_groups(indices)
        if self.arrow_dataset is not None:
            table = self.arrow_dataset.take(
                pa.array(indices, type=pa.int64()),
                columns=["docid", "text", "url"],
            )
            return table.to_pylist()
        return [self.dataset[index] for index in indices]

    def get_doc(self, docid: str) -> Optional[dict[str, Any]]:
        docid = str(docid)
        cached = self._get_cached_doc(docid)
        if cached is not None:
            return cached
        if self.docid_is_row_index:
            try:
                idx = int(docid)
            except (TypeError, ValueError):
                return None
            if idx < 0 or idx >= self.size:
                return None
            row = self._take_direct([idx])[0]
            if str(row.get("docid")) != str(docid):
                raise RuntimeError(
                    f"corpus direct lookup mismatch: requested={docid!r} got={row.get('docid')!r}"
                )
            formatted = _format_corpus_doc(row)
            self._put_cached_doc(docid, formatted)
            return formatted
        idx = self.docid_to_idx.get(docid)
        if idx is None:
            return None
        formatted = _format_corpus_doc(self.dataset[idx])
        self._put_cached_doc(docid, formatted)
        return formatted

    def get_docs(self, docids: list[str]) -> list[dict[str, Any]]:
        if self.docid_is_row_index:
            requested: list[tuple[str, int]] = []
            for docid in docids:
                try:
                    idx = int(str(docid))
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < self.size:
                    requested.append((str(docid), idx))
            docs_by_docid: dict[str, dict[str, Any]] = {}
            missing: list[tuple[str, int]] = []
            missing_seen: set[str] = set()
            for docid, idx in requested:
                cached = self._get_cached_doc(docid)
                if cached is not None:
                    docs_by_docid[docid] = cached
                elif docid not in missing_seen:
                    missing.append((docid, idx))
                    missing_seen.add(docid)

            rows = self._take_direct([idx for _, idx in missing])
            for (docid, _), row in zip(missing, rows):
                if str(row.get("docid")) != docid:
                    raise RuntimeError(
                        f"corpus direct lookup mismatch: requested={docid!r} got={row.get('docid')!r}"
                    )
                formatted = _format_corpus_doc(row)
                docs_by_docid[docid] = formatted
                self._put_cached_doc(docid, formatted)
            return [docs_by_docid[docid] for docid, _ in requested]
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
            docid_is_row_index=config.corpus_docid_is_row_index,
            row_lookup_mode=config.corpus_row_lookup_mode,
            doc_cache_size=config.corpus_doc_cache_size,
            row_group_read_workers=config.corpus_row_group_read_workers,
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
        self.lookup_is_row_index = False
        self._build_faiss_index(
            faiss_gpu=config.faiss_gpu,
            max_docs=config.max_docs,
            stream_index_build=config.stream_index_build,
            index_lookup_mode=config.index_lookup_mode,
        )

    def _iter_pkl_files(self) -> list[str]:
        paths = sorted(glob.glob(os.path.join(self.index_dir, "*.pkl")), key=_natural_sort_key)
        if not paths:
            raise FileNotFoundError(f"No .pkl files found under index_dir={self.index_dir}")
        return paths

    def _build_faiss_index(
        self,
        *,
        faiss_gpu: bool,
        max_docs: int,
        stream_index_build: bool = False,
        index_lookup_mode: str = "stored",
    ):
        if index_lookup_mode not in {"stored", "row_index", "auto"}:
            raise ValueError(f"Unsupported index_lookup_mode={index_lookup_mode!r}")
        if stream_index_build:
            self._build_faiss_index_streaming(
                faiss_gpu=faiss_gpu,
                max_docs=max_docs,
                index_lookup_mode=index_lookup_mode,
            )
            return
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

    def _build_faiss_index_streaming(
        self,
        *,
        faiss_gpu: bool,
        max_docs: int,
        index_lookup_mode: str,
    ) -> None:
        """Add one embedding shard at a time, optionally straight to sharded GPU FAISS.

        The legacy path remains the default.  This mode avoids retaining both a
        122 GB float16 source and a ~244 GB concatenated float32 matrix while
        loading OpenResearcher-Indexes.
        """

        total = 0
        index = None
        row_index_candidate = index_lookup_mode in {"row_index", "auto"}
        self.lookup = []
        for shard_number, path in enumerate(self._iter_pkl_files(), start=1):
            with open(path, "rb") as handle:
                embeddings, lookup = pickle.load(handle)
            embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
            if embeddings.ndim != 2 or embeddings.shape[0] != len(lookup):
                raise RuntimeError(
                    f"Malformed embedding shard {path}: shape={embeddings.shape} lookup={len(lookup)}"
                )

            if max_docs and max_docs > 0:
                remaining = max_docs - total
                if remaining <= 0:
                    break
                if embeddings.shape[0] > remaining:
                    embeddings = embeddings[:remaining]
                    lookup = lookup[:remaining]

            faiss.normalize_L2(embeddings)
            if index is None:
                cpu_index = faiss.IndexFlatIP(int(embeddings.shape[1]))
                if faiss_gpu:
                    co = faiss.GpuMultipleClonerOptions()
                    co.useFloat16 = True
                    co.shard = True
                    index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
                else:
                    index = cpu_index

            lookup_values = [str(value) for value in lookup]
            if row_index_candidate:
                mismatch = next(
                    (
                        (offset, value)
                        for offset, value in enumerate(lookup_values)
                        if value != str(total + offset)
                    ),
                    None,
                )
                if mismatch is not None:
                    if index_lookup_mode == "row_index":
                        offset, value = mismatch
                        raise RuntimeError(
                            "index_lookup_mode=row_index validation failed: "
                            f"global_index={total + offset} lookup={value!r} shard={path}"
                        )
                    # Auto mode falls back without losing already validated
                    # identity shards.
                    self.lookup = [str(value) for value in range(total)]
                    row_index_candidate = False
            if not row_index_candidate:
                self.lookup.extend(lookup_values)

            index.add(embeddings)
            total += int(embeddings.shape[0])
            print(
                f"[INDEX] streamed_shard={shard_number} rows={embeddings.shape[0]} total={total}",
                flush=True,
            )
            del embeddings, lookup, lookup_values
            gc.collect()
            if max_docs and max_docs > 0 and total >= max_docs:
                break

        if index is None or total <= 0:
            raise RuntimeError(f"No embedding shards loaded from {self.index_dir}")
        if int(index.ntotal) != total:
            raise RuntimeError(f"FAISS ntotal={index.ntotal} does not match loaded rows={total}")
        self.lookup_is_row_index = row_index_candidate
        if not self.lookup_is_row_index and len(self.lookup) != total:
            raise RuntimeError(f"lookup_len={len(self.lookup)} does not match loaded rows={total}")
        print(
            f"[INDEX] ready ntotal={total} lookup_mode="
            f"{'row_index' if self.lookup_is_row_index else 'stored'}",
            flush=True,
        )
        self.index = index

    def _lookup_docids(self, idxs: list[int]) -> list[str]:
        docids: list[str] = []
        for idx in idxs:
            if self.lookup_is_row_index:
                if idx < 0 or idx >= int(self.index.ntotal):
                    continue
                docids.append(str(idx))
            else:
                if idx < 0 or idx >= len(self.lookup):
                    continue
                docids.append(self.lookup[idx])
        return docids

    def _lookup_docs(self, idxs: list[int]) -> list[dict[str, Any]]:
        return self.corpus.get_docs(self._lookup_docids(idxs))

    def materialize_docids(self, docid_rows: list[list[str]]) -> list[list[dict[str, Any]]]:
        """Resolve ranked IDs in one corpus batch, preserving per-query order.

        A tool call may carry up to three query variants.  Flattening their IDs
        lets the Parquet corpus coalesce shared files/row groups once instead of
        running three independent decompression waves.
        """

        normalized_rows = [[str(docid) for docid in row] for row in docid_rows]
        unique_docids: list[str] = []
        seen: set[str] = set()
        for row in normalized_rows:
            for docid in row:
                if docid not in seen:
                    seen.add(docid)
                    unique_docids.append(docid)
        materialized = self.corpus.get_docs(unique_docids)
        docs_by_id = {_doc_key(doc): doc for doc in materialized}
        return [
            [docs_by_id[docid] for docid in row if docid in docs_by_id]
            for row in normalized_rows
        ]

    def batch_search_ids(
        self,
        query_list: list[str],
        num: Optional[int] = None,
    ) -> tuple[list[list[str]], list[list[float]], dict[str, float]]:
        """Run query encoding and FAISS search without materializing corpus rows.

        OpenResearcher stores roughly 30k documents per Parquet row group.  The
        legacy ``batch_search`` eagerly decompressed every Top-K document even
        when the first Frontier page only needed ranks 1-10.  Keeping the ID
        search separate lets the HTTP layer defer ranks 11-50 until Frontier is
        actually selected, while the legacy API below remains unchanged.
        """

        topk = int(num or self.topk)
        encode_started = time.perf_counter()
        query_emb = self.encoder.encode_queries(list(query_list))
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        index_started = time.perf_counter()
        scores, idxs = self.index.search(query_emb, topk)
        index_ms = (time.perf_counter() - index_started) * 1000.0
        all_docids = [
            self._lookup_docids([int(value) for value in row_idxs])
            for row_idxs in idxs.tolist()
        ]
        all_scores = [
            [float(value) for value in row_scores][: len(docids)]
            for docids, row_scores in zip(all_docids, scores.tolist())
        ]
        return all_docids, all_scores, {
            "encode": encode_ms,
            "index": index_ms,
        }

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
        all_docids, all_scores, _ = self.batch_search_ids(query_list=query_list, num=num)
        all_docs = self.materialize_docids(all_docids)
        all_scores = [
            scores[: len(docs)]
            for docs, scores in zip(all_docs, all_scores)
        ]
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
        stream_index_build: bool = False,
        corpus_docid_is_row_index: bool = False,
        index_lookup_mode: str = "stored",
        corpus_row_lookup_mode: str = "dataset_take",
        corpus_doc_cache_size: int = 0,
        corpus_row_group_read_workers: int = 1,
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
        self.stream_index_build = bool(stream_index_build)
        self.corpus_docid_is_row_index = bool(corpus_docid_is_row_index)
        self.index_lookup_mode = str(index_lookup_mode)
        self.corpus_row_lookup_mode = str(corpus_row_lookup_mode)
        self.corpus_doc_cache_size = max(0, int(corpus_doc_cache_size))
        self.corpus_row_group_read_workers = max(1, int(corpus_row_group_read_workers))


def _truncate_text(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


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


def _docid_is_allowed(value: Any) -> bool:
    """Return whether an optional training-corpus allow-list admits a docid."""

    docid = _normalize_frontier_docid(value)
    return bool(docid) and (allowed_docids is None or docid in allowed_docids)


def _filter_ranked_results(
    results: list[list[dict[str, Any]]],
    scores: Optional[list[list[float]]],
) -> tuple[list[list[dict[str, Any]]], Optional[list[list[float]]]]:
    if allowed_docids is None:
        return results, scores
    filtered_results: list[list[dict[str, Any]]] = []
    filtered_scores: Optional[list[list[float]]] = [] if scores is not None else None
    for row_index, docs in enumerate(results):
        row_scores = scores[row_index] if scores is not None and row_index < len(scores) else []
        kept_docs: list[dict[str, Any]] = []
        kept_scores: list[float] = []
        for rank, doc in enumerate(docs):
            if not _docid_is_allowed(_doc_key(doc)):
                continue
            kept_docs.append(doc)
            if scores is not None and rank < len(row_scores):
                kept_scores.append(float(row_scores[rank]))
        filtered_results.append(kept_docs)
        if filtered_scores is not None:
            filtered_scores.append(kept_scores)
    return filtered_results, filtered_scores


def _filter_docid_score_rows(
    docid_rows: list[list[str]], score_rows: list[list[float]]
) -> tuple[list[list[str]], list[list[float]]]:
    if allowed_docids is None:
        return docid_rows, score_rows
    kept_docid_rows: list[list[str]] = []
    kept_score_rows: list[list[float]] = []
    for row_index, docids in enumerate(docid_rows):
        scores = score_rows[row_index] if row_index < len(score_rows) else []
        kept_docids: list[str] = []
        kept_scores: list[float] = []
        for rank, docid in enumerate(docids):
            if not _docid_is_allowed(docid):
                continue
            kept_docids.append(docid)
            if rank < len(scores):
                kept_scores.append(float(scores[rank]))
        kept_docid_rows.append(kept_docids)
        kept_score_rows.append(kept_scores)
    return kept_docid_rows, kept_score_rows


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
            if not _docid_is_allowed(_doc_key(doc)):
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


def _unique_ranked_docids(
    docid_rows: list[list[str]],
    *,
    rank_cutoff: int,
) -> list[str]:
    """Return first-seen IDs from the ranked per-query candidate lists."""

    seen: set[str] = set()
    ordered: list[str] = []
    cutoff = max(1, int(rank_cutoff))
    for row in docid_rows:
        for value in row[:cutoff]:
            docid = _normalize_frontier_docid(value)
            if _docid_is_allowed(docid) and docid not in seen:
                seen.add(docid)
                ordered.append(docid)
    return ordered


def _lazy_frontier_search_result(
    *,
    query_list: list[str],
    topk: int,
    return_scores: bool,
    frontier_enable: bool,
    frontier_exclude_docids: Optional[list[str]],
    frontier_base_rank_cutoff: int,
    frontier_output_topk: int,
    frontier_candidate_topk: int,
    frontier_overlap_threshold: float,
    frontier_compact_snippet_chars: Optional[int],
) -> tuple[str, dict[str, Any], dict[str, float]]:
    """Render the exact base page before deciding whether Top50 is needed.

    The base page only depends on ranks up to ``frontier_base_rank_cutoff``.
    We therefore materialize that prefix first.  If its overlap activates the
    Frontier, the full candidate set is materialized and passed through the
    unchanged formatter, preserving the established Frontier ordering.  If it
    does not activate, the model-visible text and selected document IDs are
    identical to eager hydration while ranks 11-50 stay as IDs.
    """

    id_search_started = time.perf_counter()
    docid_rows, all_scores, search_timing = retriever.batch_search_ids(
        query_list=query_list,
        num=topk,
    )
    docid_rows, all_scores = _filter_docid_score_rows(docid_rows, all_scores)
    id_search_ms = (time.perf_counter() - id_search_started) * 1000.0

    full_rank_cutoff = max(1, int(search_global_rank_cutoff))
    base_rank_cutoff = min(
        full_rank_cutoff,
        max(1, int(frontier_base_rank_cutoff)),
    )
    base_docid_rows = [row[:base_rank_cutoff] for row in docid_rows]
    base_scores = [row[: len(docids)] for row, docids in zip(all_scores, base_docid_rows)]

    base_hydrate_started = time.perf_counter()
    base_results = retriever.materialize_docids(base_docid_rows)
    base_hydrate_ms = (time.perf_counter() - base_hydrate_started) * 1000.0

    render_started = time.perf_counter()
    result_text, frontier_metadata = _format_lite_search_global_result(
        queries=query_list,
        results=base_results,
        scores=base_scores if return_scores else None,
        frontier_enable=frontier_enable,
        frontier_exclude_docids=frontier_exclude_docids,
        frontier_base_rank_cutoff=frontier_base_rank_cutoff,
        frontier_output_topk=frontier_output_topk,
        frontier_candidate_topk=frontier_candidate_topk,
        frontier_overlap_threshold=frontier_overlap_threshold,
        frontier_compact_snippet_chars=frontier_compact_snippet_chars,
    )
    render_ms = (time.perf_counter() - render_started) * 1000.0

    frontier_hydrate_ms = 0.0
    full_hydration = bool(frontier_metadata.get("applied", False))
    if full_hydration:
        frontier_hydrate_started = time.perf_counter()
        full_results = retriever.materialize_docids(docid_rows)
        frontier_hydrate_ms = (time.perf_counter() - frontier_hydrate_started) * 1000.0
        render_started = time.perf_counter()
        result_text, frontier_metadata = _format_lite_search_global_result(
            queries=query_list,
            results=full_results,
            scores=all_scores if return_scores else None,
            frontier_enable=frontier_enable,
            frontier_exclude_docids=frontier_exclude_docids,
            frontier_base_rank_cutoff=frontier_base_rank_cutoff,
            frontier_output_topk=frontier_output_topk,
            frontier_candidate_topk=frontier_candidate_topk,
            frontier_overlap_threshold=frontier_overlap_threshold,
            frontier_compact_snippet_chars=frontier_compact_snippet_chars,
        )
        render_ms += (time.perf_counter() - render_started) * 1000.0
    else:
        # The formatter only saw the hydrated prefix.  Preserve useful full-pool
        # counters without forcing body reads.  When exclusions are non-empty,
        # ``unseen_count`` is an ID-level count rather than a lexical-reranked
        # count; it is diagnostic-only and never controls Frontier behavior.
        full_docids = _unique_ranked_docids(docid_rows, rank_cutoff=full_rank_cutoff)
        excluded = {
            docid
            for value in (frontier_exclude_docids or [])
            if (docid := _normalize_frontier_docid(value))
        }
        candidate_count = min(len(full_docids), max(1, int(frontier_candidate_topk)))
        frontier_metadata["candidate_count"] = candidate_count
        frontier_metadata["full_unique_candidate_count"] = len(full_docids)
        frontier_metadata["unseen_count"] = min(
            sum(docid not in excluded for docid in full_docids),
            candidate_count,
        )
        frontier_metadata["full_metadata_exact"] = not excluded

    hydrated_docids = _unique_ranked_docids(
        docid_rows if full_hydration else base_docid_rows,
        rank_cutoff=full_rank_cutoff if full_hydration else base_rank_cutoff,
    )
    retrieved_docids = _unique_ranked_docids(docid_rows, rank_cutoff=full_rank_cutoff)
    frontier_metadata["hydration"] = {
        "mode": "lazy_frontier",
        "full_hydration": full_hydration,
        "retrieved_unique_docids": len(retrieved_docids),
        "hydrated_unique_docids": len(hydrated_docids),
    }
    timing = {
        "id_search": id_search_ms,
        "encode": float(search_timing.get("encode", 0.0)),
        "index": float(search_timing.get("index", 0.0)),
        "base_hydrate": base_hydrate_ms,
        "frontier_hydrate": frontier_hydrate_ms,
        "render": render_ms,
    }
    return result_text, frontier_metadata, timing


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
        if not _docid_is_allowed(docid):
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
        if not _docid_is_allowed(docid):
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

    ranked = [
        rec
        for rec in ranked
        if _docid_is_allowed(_doc_key(rec.get("doc") or {}))
    ]
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


def _format_lite_search_result(query: str, docs: list[dict[str, Any]], scores: Optional[list[float]] = None) -> str:
    # Keep the search response compact so top-10 results fit under verl's
    # max_tool_response_length.  Expose only doc:// identifiers to discourage
    # the model from copying long real URLs; /open resolves doc:// directly.
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


if FastAPI is not None:
    app = FastAPI()
else:  # pragma: no cover
    class _DummyApp:
        def post(self, *args, **kwargs):
            def deco(fn):
                return fn

            return deco

        get = post

    app = _DummyApp()


class LiteSearchRequest(BaseModel):
    # The search prompt emits {"query": [...]}, while VERL's existing
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

lite_search_snippet_chars: int = 120
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
search_lazy_frontier_hydration: bool = False
search_request_concurrency: int = 0
search_request_gate: Optional[threading.BoundedSemaphore] = None
dual_retrieval_enable: bool = False
dual_secondary_rank_url: str = ""
dual_candidate_topk: int = 50
dual_rrf_k: int = 60
dual_primary_weight: float = 1.0
dual_secondary_weight: float = 2.0
dual_primary_rerank_mode: str = "rrf"
dual_secondary_timeout: float = 120.0
search_result_link_label: str = "VisitURL"
search_result_open_tool_name: str = "visit"
browser_line_chars: int = 160
browser_default_num_lines: int = 40
browser_max_num_lines: int = 80
browser_find_context_lines: int = 1
browser_find_max_hits: int = 10
allowed_docids: Optional[set[str]] = None
allowed_docids_path: str = ""


@app.get("/health")
def health_endpoint():
    """Expose the effective retrieval configuration for launch preflight."""

    corpus = getattr(retriever, "corpus", None)
    return {
        "success": retriever is not None and config is not None,
        "runtime": {
            "search_lazy_frontier_hydration": bool(search_lazy_frontier_hydration),
            "search_request_concurrency": int(search_request_concurrency),
            "search_global_enable": bool(search_global_enable),
            "search_global_rank_cutoff": int(search_global_rank_cutoff),
            "search_global_output_topk": int(search_global_output_topk),
            "search_global_rerank_mode": str(search_global_rerank_mode),
            "query_encoder_model": os.path.realpath(str(getattr(config, "query_encoder_model", ""))),
            "index_path": os.path.realpath(str(getattr(config, "index_path", "") or "")),
            "dual_retrieval_enable": bool(dual_retrieval_enable),
            "dual_secondary_rank_url": str(dual_secondary_rank_url),
            "allowed_docids_path": allowed_docids_path or None,
            "allowed_docids_count": len(allowed_docids) if allowed_docids is not None else None,
            "corpus_row_lookup_mode": getattr(corpus, "row_lookup_mode", None),
            "corpus_source_path": os.path.realpath(
                os.path.expanduser(str(getattr(config, "corpus_repo_id", "")))
            ),
            "corpus_rows": int(getattr(corpus, "size", 0)),
            "corpus_parquet_files": int(getattr(corpus, "parquet_file_count", 0)),
            "corpus_parquet_row_groups": int(
                getattr(corpus, "parquet_row_group_count", 0)
            ),
            "corpus_max_row_group_rows": int(
                getattr(corpus, "parquet_max_row_group_rows", 0)
            ),
            "corpus_mean_row_group_rows": float(
                getattr(corpus, "parquet_mean_row_group_rows", 0.0)
            ),
            "corpus_doc_cache_size": int(getattr(corpus, "doc_cache_size", 0)),
            "corpus_row_group_read_workers": int(
                getattr(corpus, "row_group_read_workers", 0)
            ),
            "server_source_path": SERVER_SOURCE_PATH,
            "server_source_sha256": SERVER_SOURCE_SHA256,
        },
    }


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


def _search_impl(request: LiteSearchRequest):
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
            primary_results, primary_scores = _filter_ranked_results(
                primary_results, primary_scores
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

        topk = max(1, int(request.topk or config.retrieval_topk))
        # Frontier renders a small visible page from a deeper candidate pool.
        # Promote the internal retrieval depth even when a tool schema asks for
        # only the visible Top-K; otherwise Frontier is nominally enabled but
        # has no unseen candidates beyond the first page.
        if search_global_enable and request.frontier_enable:
            topk = max(topk, int(request.frontier_candidate_topk))
        frontier_metadata: dict[str, Any] = {
            "enabled": False,
            "applied": False,
            "exhausted": False,
        }
        timing_ms: dict[str, float] = {}
        can_lazy_hydrate = bool(
            search_lazy_frontier_hydration
            and search_global_enable
            and request.frontier_enable
            and hasattr(retriever, "batch_search_ids")
            and hasattr(retriever, "materialize_docids")
        )
        if can_lazy_hydrate:
            result_text, frontier_metadata, timing_ms = _lazy_frontier_search_result(
                query_list=queries,
                topk=topk,
                return_scores=bool(request.return_scores),
                frontier_enable=bool(request.frontier_enable),
                frontier_exclude_docids=request.frontier_exclude_docids,
                frontier_base_rank_cutoff=request.frontier_base_rank_cutoff,
                frontier_output_topk=request.frontier_output_topk,
                frontier_candidate_topk=request.frontier_candidate_topk,
                frontier_overlap_threshold=request.frontier_overlap_threshold,
                frontier_compact_snippet_chars=request.frontier_compact_snippet_chars,
            )
        else:
            eager_started = time.perf_counter()
            results, scores = retriever.batch_search(
                query_list=queries,
                num=topk,
                return_score=request.return_scores,
            )
            results, scores = _filter_ranked_results(results, scores)
            timing_ms["eager_search_and_hydrate"] = (
                time.perf_counter() - eager_started
            ) * 1000.0

        if search_global_enable and not can_lazy_hydrate:
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
        elif not search_global_enable:
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
            "timing_ms": timing_ms,
            "fusion": {
                "enabled": bool(dual_retrieval_enable),
                "applied": False,
                "fallback_error": dual_error,
            },
        }
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return {
            "success": False,
            "result": "",
            "elapsed_ms": elapsed,
            "error": str(e),
        }


@app.post("/search")
def search_endpoint(request: LiteSearchRequest):
    """Apply one authoritative server-side search concurrency gate."""

    request_started = time.perf_counter()
    acquired_at = request_started
    gate = search_request_gate
    if gate is not None:
        gate.acquire()
        acquired_at = time.perf_counter()
    try:
        response = _search_impl(request)
    finally:
        if gate is not None:
            gate.release()

    if isinstance(response, dict):
        completed_at = time.perf_counter()
        timing = dict(response.get("timing_ms") or {})
        timing["queue"] = (acquired_at - request_started) * 1000.0
        timing["total"] = (completed_at - request_started) * 1000.0
        response["timing_ms"] = timing
        # Keep the established field meaningful under backpressure: callers
        # should observe queueing as part of server latency.
        response["elapsed_ms"] = timing["total"]
    return response


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
    if not _docid_is_allowed(docid):
        return {
            "success": False,
            "result": "",
            "elapsed_ms": (time.time() - start_time) * 1000,
            "error": f"document outside configured training allow-list: {docid}",
        }
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
    if not _docid_is_allowed(docid):
        return {
            "success": False,
            "result": "",
            "elapsed_ms": (time.time() - start_time) * 1000,
            "error": f"document outside configured training allow-list: {docid}",
        }
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense BrowseComp-Plus retrieval: dual search, rank, open and find.")
    parser.add_argument("--retrieval_method", choices=["dense"], default="dense")
    parser.add_argument("--indexes_repo_id", type=str, default="Tevatron/browsecomp-plus-indexes")
    parser.add_argument("--index_name", type=str, default="qwen3-embedding-0.6b")
    parser.add_argument("--index_path", type=str, default=None, help="Optional local index dir, overrides auto-download.")
    parser.add_argument("--corpus_repo_id", type=str, default="Tevatron/browsecomp-plus-corpus")
    parser.add_argument("--corpus_split", type=str, default="train")
    parser.add_argument(
        "--allowed_docids_path",
        type=str,
        default=None,
        help=(
            "Optional newline-delimited docid allow-list. Search, rank, "
            "open, and find hide every document outside it. Omit to preserve "
            "the legacy full-corpus behavior."
        ),
    )
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
    parser.add_argument(
        "--stream_index_build",
        action="store_true",
        help="Add embedding shards incrementally; with --faiss_gpu, write directly to sharded GPU FAISS.",
    )
    parser.add_argument(
        "--corpus_docid_is_row_index",
        action="store_true",
        help="Use numeric docid as the Parquet row index and skip the full Python docid map.",
    )
    parser.add_argument(
        "--corpus_row_lookup_mode",
        choices=["dataset_take", "parquet_row_group"],
        default="dataset_take",
        help=(
            "dataset_take preserves the legacy global-row scan; parquet_row_group builds "
            "file/row-group offsets for direct random access."
        ),
    )
    parser.add_argument(
        "--corpus_doc_cache_size",
        type=int,
        default=0,
        help="Number of formatted corpus documents retained in an in-process LRU; <=0 disables it.",
    )
    parser.add_argument(
        "--corpus_row_group_read_workers",
        type=int,
        default=1,
        help=(
            "Shared worker limit for independent Parquet-file reads in parquet_row_group mode; "
            "1 preserves serial reads."
        ),
    )
    parser.add_argument(
        "--index_lookup_mode",
        choices=["stored", "row_index", "auto"],
        default="stored",
        help="Store lookup strings, require identity lookup, or auto-detect identity lookup.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max_title_chars",
        type=int,
        default=120,
        help="Maximum characters in displayed document titles.",
    )
    parser.add_argument(
        "--lite_search_snippet_chars",
        type=int,
        default=120,
        help="/search 每条结果正文 snippet 的最大字符数。",
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
        "--search_lazy_frontier_hydration",
        action="store_true",
        help=(
            "Materialize only the base rank prefix first; read the full candidate "
            "pool only when Frontier is actually applied. Disabled by default."
        ),
    )
    parser.add_argument(
        "--search_request_concurrency",
        type=int,
        default=0,
        help=(
            "Authoritative server-side limit for concurrent /search requests; "
            "<=0 preserves the unbounded legacy behavior."
        ),
    )
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


    args = parser.parse_args()

    short_title_max_chars = int(args.max_title_chars)

    _require(uvicorn, "uvicorn")
    _require(FastAPI, "fastapi")

    if args.allowed_docids_path:
        allow_path = os.path.realpath(os.path.expanduser(args.allowed_docids_path))
        if not os.path.isfile(allow_path):
            parser.error(f"--allowed_docids_path does not exist: {allow_path}")
        with open(allow_path, encoding="utf-8") as handle:
            loaded_allowed_docids = {
                docid
                for line in handle
                if (docid := _normalize_frontier_docid(line))
            }
        if not loaded_allowed_docids:
            parser.error(f"--allowed_docids_path is empty: {allow_path}")
        allowed_docids = loaded_allowed_docids
        allowed_docids_path = allow_path
        print(
            f"[CORPUS-ALLOWLIST] path={allowed_docids_path} docids={len(allowed_docids)}",
            flush=True,
        )

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
        stream_index_build=args.stream_index_build,
        corpus_docid_is_row_index=args.corpus_docid_is_row_index,
        index_lookup_mode=args.index_lookup_mode,
        corpus_row_lookup_mode=args.corpus_row_lookup_mode,
        corpus_doc_cache_size=args.corpus_doc_cache_size,
        corpus_row_group_read_workers=args.corpus_row_group_read_workers,
    )
    retriever = get_retriever(config)

    lite_search_snippet_chars = args.lite_search_snippet_chars
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
    search_lazy_frontier_hydration = bool(args.search_lazy_frontier_hydration)
    search_request_concurrency = max(0, int(args.search_request_concurrency))
    search_request_gate = (
        threading.BoundedSemaphore(search_request_concurrency)
        if search_request_concurrency > 0
        else None
    )
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

    print(
        "[SEARCH] "
        f"lazy_frontier_hydration={search_lazy_frontier_hydration} "
        f"request_concurrency={search_request_concurrency or 'unbounded'} "
        f"rank_cutoff={search_global_rank_cutoff} "
        f"output_topk={search_global_output_topk} "
        f"rerank={search_global_rerank_mode} "
        f"allowed_docids={len(allowed_docids) if allowed_docids is not None else 'all'}",
        flush=True,
    )

    uvicorn.run(app, host=args.host, port=args.port)
