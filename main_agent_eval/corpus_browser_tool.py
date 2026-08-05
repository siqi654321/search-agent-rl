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

"""Deterministic open/find tools for the local BrowseComp-Plus corpus.

The retrieval service owns document contents and stable ``doc://`` cursors.  This
wrapper intentionally performs no LLM extraction: ``open`` returns an addressable
line window and ``find`` returns exact in-document matches.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

import requests

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from .visit_tool import init_visit_execution_pool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _latest_cursor(agent_data: Any) -> str:
    """Recover the most recently opened cursor when the model omits it."""
    messages = getattr(agent_data, "messages", []) or []
    cursor_re = re.compile(r"(?im)^Cursor:\s*(doc://[^\s]+)")
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        matches = cursor_re.findall(content)
        if matches:
            return matches[-1].rstrip(".,;)")
    return ""


def _normalize_cursor(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value or "").strip()


class CorpusBrowserTool(BaseTool):
    """Native verl wrapper shared by the ``open`` and ``find`` schemas."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}
        configured_base = config.get("browser_service_url") or config.get("retrieval_service_url")
        self.browser_service_url = os.getenv("BROWSECOMP_BROWSER_SERVICE_URL", configured_base or "").rstrip("/")
        if not self.browser_service_url:
            raise ValueError("CorpusBrowserTool requires browser_service_url")
        if self.name not in {"open", "find"}:
            raise ValueError(f"CorpusBrowserTool only supports open/find, got {self.name!r}")
        self.timeout = int(config.get("timeout", 30))
        self.execution_pool = init_visit_execution_pool(
            num_workers=int(config.get("num_workers", 120)),
            enable_global_rate_limit=bool(config.get("enable_global_rate_limit", True)),
            rate_limit=int(config.get("rate_limit", 120)),
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        self._instance_dict[instance_id] = {"reward": []}
        return instance_id, ToolResponse()

    @staticmethod
    def _call_api(url: str, payload: dict[str, Any], timeout: int) -> tuple[str, dict[str, Any]]:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        try:
            data = response.json()
        except json.JSONDecodeError:
            return response.text, {"status": "non_json_response"}
        if not isinstance(data, dict):
            return str(data), {"status": "unknown_response"}
        if data.get("success") is False:
            error = str(data.get("error") or "unknown browser error")
            return f"[{url.rsplit('/', 1)[-1]}] Error: {error}", {"status": "api_error", "error": error}
        result = data.get("result")
        if isinstance(result, str):
            return result, {
                "status": "success",
                "cursor": data.get("cursor"),
                "elapsed_ms": data.get("elapsed_ms"),
            }
        return str(data), {"status": "unknown_response"}

    def _payload(self, parameters: dict[str, Any], agent_data: Any) -> tuple[dict[str, Any], Optional[str]]:
        cursor = _normalize_cursor(
            parameters.get("cursor") or parameters.get("url") or parameters.get("docid") or parameters.get("id")
        )
        if not cursor:
            cursor = _latest_cursor(agent_data)
        if not cursor:
            return {}, "A doc:// cursor is required. Open a search result first or pass cursor explicitly."

        if self.name == "open":
            payload: dict[str, Any] = {"cursor": cursor}
            for key in ("loc", "num_lines", "goal"):
                if parameters.get(key) is not None:
                    payload[key] = parameters[key]
            return payload, None

        pattern = str(parameters.get("pattern") or "").strip()
        if not pattern:
            return {}, "The find tool requires a non-empty exact pattern."
        payload = {"cursor": cursor, "pattern": pattern}
        for key in ("context_lines", "max_hits"):
            if parameters.get(key) is not None:
                payload[key] = parameters[key]
        return payload, None

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        payload, error = self._payload(parameters, kwargs.get("agent_data"))
        if error:
            return ToolResponse(text=f"[{self.name}] Error: {error}"), 0.0, {"status": "bad_arguments"}
        try:
            result, metrics = await self.execution_pool.execute.remote(
                CorpusBrowserTool._call_api,
                f"{self.browser_service_url}/{self.name}",
                payload,
                self.timeout,
            )
            self._instance_dict[instance_id]["reward"].append(result.strip())
            metrics["tool"] = self.name
            return ToolResponse(text=result), 0.0, metrics
        except Exception as exc:
            logger.error("[%s] execution failed: %s", self.name, exc)
            return (
                ToolResponse(text=f"{self.name} execution failed: {exc}"),
                0.0,
                {"status": "exception", "error": str(exc)},
            )

    async def calc_reward(self, instance_id: str, **kwargs):
        return self._instance_dict.get(instance_id, {}).get("reward", [])

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
