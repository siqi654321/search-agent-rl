# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Search+visit-compatible visit tool.

The model emits tool calls in the form:

<tool_call>
{"name": "visit", "arguments": {"url": ["..."], "goal": "..."}}
</tool_call>

This native verl tool forwards those calls to a local browser endpoint, such as
`src/mtools/retrieval_server_browsecomp_plus.py:/browse`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor
import requests

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_VISIT_POOL_HANDLE: Optional[ray.actor.ActorHandle] = None
T = TypeVar("T")


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


@ray.remote
class VisitExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        return TokenBucketWorker.options(name="visit-rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                return fn(*fn_args, **fn_kwargs)
        return fn(*fn_args, **fn_kwargs)


def init_visit_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    global _VISIT_POOL_HANDLE
    if mode != PoolMode.ThreadMode:
        raise NotImplementedError("Process mode is not implemented yet")
    if _VISIT_POOL_HANDLE is None:
        _VISIT_POOL_HANDLE = (
            VisitExecutionWorker.options(max_concurrency=num_workers, name="visit-pool", get_if_exists=True)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    return _VISIT_POOL_HANDLE


class VisitTool(BaseTool):
    """Visit local webpage/document(s) and return content relevant to a goal."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.browser_service_url = config.get("browser_service_url") or config.get("visit_service_url")
        assert self.browser_service_url, "Configuration must include 'browser_service_url'"
        self.timeout = config.get("timeout", 120)
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_visit_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"response": "", "reward": []}
        return instance_id, ToolResponse()

    @staticmethod
    def _normalize_urls(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = value
        else:
            values = [str(value)]
        return [str(v).strip() for v in values if str(v).strip()]

    @staticmethod
    def _call_browser_api(browser_service_url: str, urls: list[str], goal: str, timeout: int) -> tuple[str, dict]:
        payload = {"url": urls, "goal": goal}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        resp = requests.post(browser_service_url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        try:
            data = resp.json()
        except json.JSONDecodeError:
            return resp.text, {"status": "non_json_response"}

        if isinstance(data, dict):
            if data.get("success") is False:
                return f"[Visit] Error: {data.get('error', 'unknown error')}", {"status": "api_error", "api_response": data}
            if isinstance(data.get("result"), str):
                return data["result"], {"status": "success", "api_response": data}
        return str(data), {"status": "unknown_response", "api_response": data}

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        urls = self._normalize_urls(parameters.get("url") or parameters.get("urls"))
        goal = str(parameters.get("goal") or "").strip()
        if not urls:
            error_msg = "Error: 'url' is missing or empty in visit parameters."
            logger.error("[VisitTool] %s Received parameters: %s", error_msg, parameters)
            return ToolResponse(text=error_msg), 0.0, {"status": "bad_arguments"}

        try:
            result_text, metrics = await self.execution_pool.execute.remote(
                VisitTool._call_browser_api, self.browser_service_url, urls, goal, self.timeout
            )
            self._instance_dict[instance_id]["reward"].append(result_text.strip())
            metrics.update({"url_count": len(urls)})
            return ToolResponse(text=result_text), 0.0, metrics
        except Exception as e:
            error_msg = f"Visit execution failed: {e}"
            logger.error("[VisitTool] %s", error_msg)
            return ToolResponse(text=error_msg), 0.0, {"status": "exception", "error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs):
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

