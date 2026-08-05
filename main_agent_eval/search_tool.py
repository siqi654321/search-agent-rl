# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import json
import logging
import os
import re
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor

from verl.tools.utils.search_r1_like_utils import perform_single_search_batch
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_SEARCH_EXPANSION_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those", "it", "its",
    "their", "his", "her", "he", "she", "they", "them", "who", "whom", "whose", "which", "what", "when",
    "where", "why", "how", "please", "tell", "me", "name", "following", "criteria", "criterion",
    "about", "into", "after", "before", "between", "during", "over", "under", "than", "then", "also",
    "same", "first", "second", "third", "identify", "find", "provide", "could", "would", "should",
    "you", "your",
}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _dedupe_keep_order(items: list[str], max_items: int | None = None) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        item = re.sub(r"\s+", " ", str(item or "")).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if max_items is not None and len(out) >= max_items:
            break
    return out


def _extract_original_user_question(agent_data: Any) -> str:
    messages = getattr(agent_data, "messages", []) or []
    # Use the last real user message before tool_response messages begin.
    candidates = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        stripped = content.strip()
        if stripped.startswith("<tool_response>"):
            continue
        candidates.append(stripped)
    return candidates[-1] if candidates else ""


def _question_clauses(question: str, max_clauses: int) -> list[str]:
    q = re.sub(r"\s+", " ", str(question or "")).strip()
    if not q:
        return []
    chunks = re.split(r"\s+(?=[A-H]\.\s+)|\s+-\s+", q)
    clauses: list[str] = []
    for chunk in chunks:
        chunk = re.sub(r"^[A-H]\.\s+", "", chunk).strip(" .")
        for sent in re.split(r"(?<=[.!?])\s+", chunk):
            sent = sent.strip(" .")
            if len(sent) >= 20:
                clauses.append(sent)
    return _dedupe_keep_order(clauses, max_clauses)


def _keyword_query(text: str, max_terms: int = 18) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    years = re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", text)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text.lower())
    content = [
        w
        for w in words
        if w not in _SEARCH_EXPANSION_STOPWORDS
        and len(w) >= 3
        and not re.fullmatch(r"[a-h]", w)
    ]
    return " ".join(_dedupe_keep_order(years + content, max_terms))

# Keep a strong reference to the named search execution actor in the current
# Python process to avoid Ray GC killing it when tool instances are destroyed.
_SEARCH_POOL_HANDLE: Optional[ray.actor.ActorHandle] = None

T = TypeVar("T")


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


@ray.remote
class SearchExecutionWorker:
    """Worker for executing search operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing search: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_search_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize search execution pool."""
    global _SEARCH_POOL_HANDLE
    if mode == PoolMode.ThreadMode:
        if _SEARCH_POOL_HANDLE is None:
            _SEARCH_POOL_HANDLE = (
                SearchExecutionWorker.options(
                    max_concurrency=num_workers,
                    name="search-pool",
                    get_if_exists=True,
                ).remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
            )
        return _SEARCH_POOL_HANDLE
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class SearchTool(BaseTool):
    """Search tool for retrieving information using external retrieval services.

    This tool provides search functionality with rate limiting and concurrent execution
    support through Ray. It integrates with external retrieval services to perform
    semantic search operations.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Searches for relevant information based on queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries"
                            }
                        },
                        "required": ["query_list"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        # Retrieval service configuration
        self.retrieval_service_url = os.getenv(
            "BROWSECOMP_SEARCH_SERVICE_URL", config.get("retrieval_service_url")
        )
        assert self.retrieval_service_url, "Configuration must include 'retrieval_service_url'"
        self.topk = int(os.getenv("BROWSECOMP_SEARCH_TOPK", config.get("topk", 3)))
        if self.retrieval_service_url == "":
            raise ValueError("retrieval_service_url is not set")
        self.query_expansion_enable = _truthy(
            config.get("query_expansion_enable", os.getenv("SEARCH_QUERY_EXPANSION_ENABLE", "0"))
        )
        self.query_expansion_mode = str(
            config.get("query_expansion_mode", os.getenv("SEARCH_QUERY_EXPANSION_MODE", "trajectory_clauses"))
        )
        self.query_expansion_max_clauses = int(
            config.get("query_expansion_max_clauses", os.getenv("SEARCH_QUERY_EXPANSION_MAX_CLAUSES", "4"))
        )
        self.query_expansion_max_total = int(
            config.get("query_expansion_max_total", os.getenv("SEARCH_QUERY_EXPANSION_MAX_TOTAL", "16"))
        )
        self.search_frontier_enabled = _truthy(
            os.getenv(
                "VERL_TOOL_AGENT_SEARCH_FRONTIER",
                config.get("search_frontier_enable", "0"),
            )
        )
        self.search_frontier_base_rank_cutoff = max(
            1,
            int(
                os.getenv(
                    "VERL_TOOL_AGENT_SEARCH_FRONTIER_BASE_TOPK",
                    config.get("search_frontier_base_topk", 10),
                )
            ),
        )
        self.search_frontier_output_topk = max(
            1,
            int(
                os.getenv(
                    "VERL_TOOL_AGENT_SEARCH_FRONTIER_OUTPUT_TOPK",
                    config.get("search_frontier_output_topk", 10),
                )
            ),
        )
        self.search_frontier_candidate_topk = max(
            self.search_frontier_output_topk,
            int(
                os.getenv(
                    "VERL_TOOL_AGENT_SEARCH_FRONTIER_CANDIDATE_TOPK",
                    config.get("search_frontier_candidate_topk", 50),
                )
            ),
        )
        self.search_frontier_overlap_threshold = min(
            1.0,
            max(
                0.0,
                float(
                    os.getenv(
                        "VERL_TOOL_AGENT_SEARCH_FRONTIER_OVERLAP_THRESHOLD",
                        config.get("search_frontier_overlap_threshold", 0.90),
                    )
                ),
            ),
        )
        self.search_frontier_compact_snippet_chars = max(
            0,
            int(
                os.getenv(
                    "VERL_TOOL_AGENT_SEARCH_FRONTIER_COMPACT_SNIPPET_CHARS",
                    config.get("search_frontier_compact_snippet_chars", 800),
                )
            ),
        )
        self.search_frontier_exclude_limit = max(
            self.search_frontier_output_topk,
            int(
                os.getenv(
                    "VERL_TOOL_AGENT_SEARCH_FRONTIER_EXCLUDE_LIMIT",
                    config.get("search_frontier_exclude_limit", 1000),
                )
            ),
        )

        logger.info(f"Initialized SearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    def execute_search(self, instance_id: str, query_list: list, retrieval_service_url: str, topk: int, timeout: int):
        """Execute search operation using retrieval service.

        Args:
            instance_id: Tool instance ID
            query_list: List of search queries
            retrieval_service_url: URL of the retrieval service
            topk: Number of top results to return
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        logger.debug(f"Search result for instance {instance_id}: {result_text}")
        return result_text, metadata

    def _expand_queries(self, queries: list[str], agent_data: Any | None) -> list[str]:
        if not self.query_expansion_enable:
            return queries
        question = _extract_original_user_question(agent_data)
        clauses = _question_clauses(question, max_clauses=self.query_expansion_max_clauses)
        keyword_queries = _dedupe_keep_order(
            [_keyword_query(question)] + [_keyword_query(c) for c in clauses],
            max_items=self.query_expansion_max_clauses + 1,
        )
        mode = self.query_expansion_mode
        if mode == "trajectory":
            expanded = queries
        elif mode == "trajectory_clauses":
            expanded = queries + clauses
        elif mode == "trajectory_keywords":
            expanded = queries + keyword_queries
        elif mode == "all_no_gold":
            expanded = queries + ([question] if question else []) + clauses + keyword_queries
        else:
            logger.warning("Unknown query_expansion_mode=%s; using trajectory_clauses", mode)
            expanded = queries + clauses
        return _dedupe_keep_order(expanded, max_items=max(len(queries), self.query_expansion_max_total))

    def _frontier_request_options(self, agent_data: Any | None) -> dict[str, Any]:
        if not self.search_frontier_enabled:
            return {}
        exposed: list[str] = []
        for value in sorted(getattr(agent_data, "searched_docids", set()) or set()):
            docid = str(value or "").strip()
            if docid.lower().startswith("doc://"):
                docid = docid[6:]
            docid = docid.rstrip(".,;:)]}").strip()
            if docid:
                exposed.append(docid)
        if len(exposed) > self.search_frontier_exclude_limit:
            exposed = exposed[-self.search_frontier_exclude_limit :]
        return {
            "frontier_enable": True,
            "frontier_exclude_docids": exposed,
            "frontier_base_rank_cutoff": self.search_frontier_base_rank_cutoff,
            "frontier_output_topk": self.search_frontier_output_topk,
            "frontier_candidate_topk": self.search_frontier_candidate_topk,
            "frontier_overlap_threshold": self.search_frontier_overlap_threshold,
            "frontier_compact_snippet_chars": self.search_frontier_compact_snippet_chars,
        }

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        timeout = self.timeout

        # Backward/compat handling:
        # - Search-R1 tool schema historically used `query_list: list[str]`.
        # - Some schemas/tools (e.g., Hermes style) may emit `query: str`.
        # - Some callers may emit `queries: list[str]`.
        query_list_from_params = parameters.get("query_list")
        if query_list_from_params is None:
            if isinstance(parameters.get("query"), str):
                query_list_from_params = [parameters.get("query")]
            elif isinstance(parameters.get("query"), list):
                query_list_from_params = parameters.get("query")
            elif isinstance(parameters.get("queries"), list):
                query_list_from_params = parameters.get("queries")
            elif isinstance(parameters.get("queries"), str):
                query_list_from_params = [parameters.get("queries")]

        if isinstance(query_list_from_params, str):
            query_list_from_params = [query_list_from_params]

        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        # Normalize and drop empty queries
        normalized_queries: list[str] = []
        for q in query_list_from_params:
            if q is None:
                continue
            qs = str(q).strip()
            if qs:
                normalized_queries.append(qs)

        if not normalized_queries:
            error_msg = "Error: query_list is empty after normalization."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}
        normalized_queries = self._expand_queries(normalized_queries, kwargs.get("agent_data"))
        frontier_request_options = self._frontier_request_options(kwargs.get("agent_data"))

        # Execute search using Ray execution pool
        try:
            result_text, metadata = await self.execution_pool.execute.remote(
                perform_single_search_batch,
                self.retrieval_service_url,
                normalized_queries,
                self.topk,
                None,
                timeout,
                frontier_request_options,
            )

            # Compatibility: older helper returned a JSON string like {"result": "..."}.
            # Newer helper returns plain text. If we still receive JSON, unwrap it.
            try:
                if isinstance(result_text, str) and result_text.lstrip().startswith("{"):
                    parsed = json.loads(result_text)
                    if isinstance(parsed, dict) and isinstance(parsed.get("result"), str):
                        result_text = parsed["result"]
            except Exception:
                pass

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # Convert metadata to metrics
            api_response = metadata.get("api_response")
            frontier_metadata = (
                api_response.get("frontier", {})
                if isinstance(api_response, dict) and isinstance(api_response.get("frontier"), dict)
                else {}
            )
            metrics = {
                "query_count": metadata.get("query_count", 0),
                "status": metadata.get("status", "unknown"),
                "total_results": frontier_metadata.get(
                    "selected_count", metadata.get("total_results", 0)
                ),
                "api_request_error": metadata.get("api_request_error"),
                "search_frontier_enabled": bool(frontier_metadata.get("enabled", False)),
                "search_frontier_applied": bool(frontier_metadata.get("applied", False)),
                "search_frontier_exhausted": bool(frontier_metadata.get("exhausted", False)),
                "search_frontier_base_overlap": float(frontier_metadata.get("base_overlap", 0.0)),
                "search_frontier_candidate_count": int(frontier_metadata.get("candidate_count", 0)),
                "search_frontier_unseen_count": int(frontier_metadata.get("unseen_count", 0)),
                "search_frontier_selected_docids": list(
                    frontier_metadata.get("selected_docids", []) or []
                ),
                "search_frontier_selected_original_ranks": list(
                    frontier_metadata.get("selected_original_ranks", []) or []
                ),
            }

            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
