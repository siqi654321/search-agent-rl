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
import asyncio
import json
import logging
import os
import re
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_FINAL_ANSWER_INSTRUCTION = """The search phase is closed by the runtime. Tools are no longer available.
Do not output a tool call and do not continue researching. Use only the evidence already collected.
Return the single best-supported short answer inside <answer></answer>. Even if uncertain, provide the
best-supported candidate now. Do not restate the question, evidence, or explanation."""

_FINAL_ANSWER_RETRY_INSTRUCTION = """Your previous finalization response was invalid because it did not contain
a non-empty final answer. Tool calls are disabled and will not be executed. Do not continue researching or explain
your uncertainty. Return only the best-supported short answer inside <answer></answer> now."""

_FINAL_ANSWER_ONLY_INSTRUCTION = """The search phase is closed by the runtime. Tools are no longer available.
Return one concrete entity, title, name, date, number, or short phrase. "Unknown", "Unable to identify",
explanations, and tool calls are invalid. Even if uncertain, choose the best-supported concrete candidate from the
evidence already collected. The runtime has already emitted <answer>; output only the candidate and </answer>."""

_FINAL_ANSWER_ONLY_RETRY_INSTRUCTION = """Your previous finalization response was invalid. Tools are disabled
and will not be executed. Do not research, explain uncertainty, or answer with "Unknown" or "Unable to identify".
The runtime has already emitted <answer>; output only one concrete candidate from the collected evidence followed
by </answer>."""

_GENERIC_ANSWER_RESCUE_INSTRUCTION = """Your previous response was rejected because it did not provide a
concrete answer. "Unknown", "Not found", "Unable to determine", "insufficient information", and similar
non-answers are invalid. Tools are now disabled. Use the evidence already collected and choose the single
best-supported concrete entity, title, name, date, number, or short phrase, even if uncertain. Return only that
candidate inside <answer></answer>."""

_GENERIC_ANSWER_ONLY_RESCUE_INSTRUCTION = """Your previous response did not provide a concrete answer. Tools
are disabled. Generic non-answers such as "Unknown", "Not found", and "Unable to determine" are invalid. The
runtime has already emitted <answer>; output only the single best-supported concrete candidate from the collected
evidence followed by </answer>."""

_DOC_CURSOR_RE = re.compile(r"doc://([^\s|<>'\"]+)", flags=re.IGNORECASE)


def _has_nonempty_final_answer(content: str) -> bool:
    matches = re.findall(r"<answer>(.*?)</answer>", content, flags=re.DOTALL | re.IGNORECASE)
    return any(match.strip() for match in matches)


def _has_nonempty_prefilled_final_answer(content: str) -> bool:
    """Validate a completion generated after a runtime-provided ``<answer>`` prefix."""

    match = re.search(r"(.*?)</answer>", content, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return False
    candidate = re.sub(r"<[^>]+>", " ", match.group(1)).strip()
    return bool(candidate)


def _final_answer_text(content: str, *, prefilled: bool = False) -> str:
    """Extract the final candidate used by answer validation."""

    if prefilled:
        match = re.search(r"(.*?)</answer>", content, flags=re.DOTALL | re.IGNORECASE)
        candidate = match.group(1) if match else ""
    else:
        matches = re.findall(r"<answer>(.*?)</answer>", content, flags=re.DOTALL | re.IGNORECASE)
        candidate = matches[-1] if matches else ""
    return re.sub(r"<[^>]+>", " ", candidate).strip()


def _is_generic_non_answer(candidate: str) -> bool:
    """Conservatively reject answer strings that cannot receive BrowseComp credit."""

    text = " ".join(str(candidate or "").strip().lower().split())
    text = text.strip(" *`_\"'.,;:!?()[]{}")
    if not text:
        return False
    exact = {
        "unknown",
        "not found",
        "unable to determine",
        "unable to identify",
        "cannot determine",
        "cannot identify",
        "could not determine",
        "could not identify",
        "insufficient evidence",
        "insufficient information",
        "not enough evidence",
        "not enough information",
        "no answer",
        "i don't know",
        "i do not know",
    }
    if text in exact:
        return True
    patterns = (
        r"^unknown\s*(?:[-—:,(]|\b(?:due|because|from|based|insufficient|not|unable)\b)",
        r"^not found(?:\s+(?:in|from|within|among|using|based)\b)",
        r"^(?:unable|not able) to (?:determine|identify|find|verify|answer)\b",
        r"^(?:cannot|can't|could not) (?:determine|identify|find|verify|answer)\b",
        r"^(?:insufficient|inadequate|not enough) (?:evidence|information|context)\b",
        r"^no (?:conclusive |definitive |specific )?(?:answer|evidence|information)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _final_answer_status(content: str, *, prefilled: bool = False) -> str:
    candidate = _final_answer_text(content, prefilled=prefilled)
    if not candidate:
        return "missing"
    if _is_generic_non_answer(candidate):
        return "generic"
    return "concrete"


def _extract_doc_cursor_sequence(text: str) -> list[str]:
    """Extract doc cursors in display order while removing repeats."""

    cursors: list[str] = []
    seen: set[str] = set()
    for match in _DOC_CURSOR_RE.findall(text or ""):
        docid = match.rstrip(".,;:)]}")
        cursor = f"doc://{docid}" if docid else ""
        if cursor and cursor not in seen:
            seen.add(cursor)
            cursors.append(cursor)
    return cursors


def _extract_doc_cursors(text: str) -> set[str]:
    return set(_extract_doc_cursor_sequence(text))


def _extract_requested_doc_cursors(tool_call: FunctionCall) -> set[str]:
    """Best-effort cursor extraction from open/find/visit arguments."""

    try:
        arguments = json.loads(tool_call.arguments)
    except Exception:
        return set()
    if not isinstance(arguments, dict):
        return set()

    cursors: set[str] = set()
    for key in ("cursor", "url", "urls", "docid", "id"):
        values = arguments.get(key)
        if values is None:
            continue
        if not isinstance(values, list):
            values = [values]
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            extracted = _extract_doc_cursors(text)
            if extracted:
                cursors.update(extracted)
            elif key in {"docid", "id"}:
                cursors.add(f"doc://{text}")
    return cursors


def _canonical_tool_call_key(tool_call: FunctionCall) -> Optional[str]:
    """Return a conservative duplicate key for deterministic browser calls.

    Only exact normalized search/visit/open/find requests are deduplicated. A
    narrower search, another document window, or a different find pattern remains
    executable even when it is related to an earlier request.
    """

    name = str(tool_call.name or "").strip().lower()
    if name not in {"search", "visit", "open", "find"}:
        return None

    try:
        arguments = json.loads(tool_call.arguments)
    except Exception:
        return None
    if not isinstance(arguments, dict):
        return None

    if name == "search":
        values = arguments.get("query")
        if values is None:
            values = arguments.get("query_list")
        if values is None:
            values = arguments.get("queries")
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return None
        queries = sorted({" ".join(str(value).lower().split()) for value in values if str(value).strip()})
        if not queries:
            return None
        payload = {"name": name, "query": queries}
    elif name == "visit":
        values = arguments.get("url")
        if values is None:
            values = arguments.get("urls")
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return None
        urls = sorted({" ".join(str(value).split()) for value in values if str(value).strip()})
        if not urls:
            return None
        goal = " ".join(str(arguments.get("goal") or "").lower().split())
        payload = {"name": name, "url": urls, "goal": goal}
    elif name == "open":
        value = (
            arguments.get("cursor")
            or arguments.get("url")
            or arguments.get("docid")
            or arguments.get("id")
        )
        if isinstance(value, list):
            value = value[0] if value else ""
        cursor = " ".join(str(value or "").split())
        if not cursor:
            return None
        payload = {
            "name": name,
            "cursor": cursor,
            "loc": arguments.get("loc"),
            "num_lines": arguments.get("num_lines"),
            "goal": " ".join(str(arguments.get("goal") or "").lower().split()),
        }
    else:  # find
        # An omitted cursor is resolved dynamically from the latest open result.
        # Do not deduplicate that form because the same pattern on another current
        # document is a distinct request.
        cursor = " ".join(str(arguments.get("cursor") or "").split())
        pattern = " ".join(str(arguments.get("pattern") or "").lower().split())
        if not cursor or not pattern:
            return None
        payload = {
            "name": name,
            "cursor": cursor,
            "pattern": pattern,
            "context_lines": arguments.get("context_lines"),
            "max_hits": arguments.get("max_hits"),
        }

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        # Read-only handoff for reasoning-aware retrieval.  This does not alter
        # prompt construction or any controller/finalization state.
        self.current_assistant_content = ""
        self.executed_tool_call_keys: set[str] = set()
        self.duplicate_tool_calls = 0
        self.duplicate_only_streak = 0
        self.max_duplicate_only_streak = 0
        self.force_final_answer = False
        self.final_answer_attempts = 0
        self.finalization_reason: Optional[str] = None
        self.final_answer_prefilled = False
        self.finalization_exhausted = False
        self.finalization_rejected_tool_calls = 0
        self.finalization_balance_failures = 0
        self.finalization_tool_compactions = 0
        self.generic_answer_rescue_active = False
        self.generic_answer_rescue_triggers = 0
        self.generic_answer_rescue_rejections = 0
        self.generic_answer_rescue_missing_rejections = 0
        self.generic_answer_rescue_generic_rejections = 0
        self.generic_answer_rescue_successes = 0
        self.generic_answer_rescue_exhausted = False
        self.generic_answer_rescue_events: list[dict[str, Any]] = []

        # Read-only trajectory telemetry.  These fields do not affect search or
        # browser policy; they make later qrels/failure-stage analysis possible.
        self.search_calls = 0
        self.open_calls = 0
        self.visit_calls = 0
        self.find_calls = 0
        self.search_doc_refs = 0
        self.searched_docids: set[str] = set()
        self.opened_docids: set[str] = set()
        self.find_docids: set[str] = set()
        self.document_flow: list[dict[str, Any]] = []
        self.search_result_overlap_events = 0
        self.max_search_result_overlap = 0.0
        self.search_frontier_requests = 0
        self.search_frontier_base_pages = 0
        self.search_frontier_unseen_pages = 0
        self.search_frontier_exhausted_calls = 0
        self.search_frontier_selected_docids: set[str] = set()
        self.search_frontier_events: list[dict[str, Any]] = []
        # Internal duplicate-control state. Exact repeated searches are useful
        # while they advance an unseen frontier, but become ordinary duplicates
        # after that call key has exhausted its candidate pool.
        self.search_frontier_exhausted_call_keys: set[str] = set()

        # Opt-in frontier open-first state.  Unlike scaffold-v2, this controller
        # consumes the server's frontier metadata directly: an unseen page is
        # novel by construction, so its visible-doc overlap cannot distinguish
        # useful inspection from continued pagination.
        self.frontier_open_first_active = False
        self.frontier_open_first_pending_docids: list[str] = []
        self.frontier_open_first_pending_docid_set: set[str] = set()
        self.frontier_open_first_ignore_streak = 0
        self.frontier_open_first_block_next_search = False
        self.frontier_open_first_blocks_in_cycle = 0
        self.frontier_open_first_triggers = 0
        self.frontier_open_first_prompts = 0
        self.frontier_open_first_ignored_unseen_pages = 0
        self.frontier_open_first_blocked_calls = 0
        self.frontier_open_first_compliances = 0
        self.frontier_open_first_pending_compliances = 0
        self.frontier_open_first_other_inspection_releases = 0
        self.frontier_open_first_novel_search_releases = 0
        self.frontier_open_first_suggested_docids: set[str] = set()
        self.frontier_open_first_events: list[dict[str, Any]] = []

        # Opt-in scaffold-v2 controller state. The controller only becomes
        # active after consecutive search-only turns return mostly documents
        # already exposed earlier in the trajectory.
        self.search_candidate_docids: list[str] = []
        self.search_candidate_docid_set: set[str] = set()
        self.consecutive_search_only_turns = 0
        self.max_consecutive_search_only_turns = 0
        self.last_search_turn_overlap = 0.0
        self.search_to_open_required = False
        self.search_to_open_block_streak = 0
        self.search_overlap_notices = 0
        self.search_to_open_triggers = 0
        self.search_to_open_interventions = 0
        self.search_to_open_blocked_calls = 0
        self.search_to_open_compliances = 0
        self.search_to_open_soft_releases = 0
        self.search_to_open_suggested_docids: set[str] = set()
        self.search_to_open_last_suggestions: list[str] = []
        self.search_to_open_events: list[dict[str, Any]] = []

        # Opt-in one-shot local verification checkpoint.  This is deliberately
        # separate from finalization: tools remain available after the model
        # audits its current requirements, evidence, and most critical gap.
        self.verification_checkpoint_triggered = False
        self.verification_checkpoint_pending = False
        self.verification_checkpoint_reason: Optional[str] = None
        self.verification_checkpoint_triggers = 0
        self.verification_checkpoint_responses = 0
        self.verification_checkpoint_structured_responses = 0
        self.verification_checkpoint_soft_releases = 0
        self.verification_checkpoint_consecutive_exhausted_turns = 0
        self.verification_checkpoint_consecutive_search_only_turns = 0
        self.verification_checkpoint_last_search_overlap = 0.0
        self.verification_checkpoint_max_search_overlap = 0.0
        self.verification_checkpoint_suggested_docids: set[str] = set()
        self.verification_checkpoint_action_counts = {
            "open_existing": 0,
            "search_gap": 0,
            "answer": 0,
            "other": 0,
        }
        self.verification_checkpoint_events: list[dict[str, Any]] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.duplicate_only_streak_limit = max(
            1, int(os.getenv("VERL_TOOL_AGENT_DUPLICATE_STREAK_LIMIT", "2"))
        )
        self.final_answer_reserve_tokens = min(
            max(1, self.response_length - 1),
            max(1, int(os.getenv("VERL_TOOL_AGENT_FINAL_RESERVE_TOKENS", "8192"))),
        )
        self.final_answer_max_attempts = max(
            1, int(os.getenv("VERL_TOOL_AGENT_FINAL_MAX_ATTEMPTS", "2"))
        )
        self.final_answer_attempt_max_tokens = max(
            1, int(os.getenv("VERL_TOOL_AGENT_FINAL_ATTEMPT_TOKENS", "2048"))
        )
        self.final_answer_prefill = os.getenv("VERL_TOOL_AGENT_FINAL_ANSWER_PREFILL", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.generic_answer_rescue_enabled = os.getenv(
            "VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.generic_answer_rescue_prefill = os.getenv(
            "VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE_PREFILL", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.finalization_structure_closure = os.getenv(
            "VERL_TOOL_AGENT_FINALIZATION_STRUCTURE_CLOSURE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.scaffold_v2_enabled = os.getenv("VERL_TOOL_AGENT_SCAFFOLD_V2", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.search_frontier_enabled = os.getenv(
            "VERL_TOOL_AGENT_SEARCH_FRONTIER", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.frontier_open_first_enabled = bool(
            self.search_frontier_enabled
            and os.getenv(
                "VERL_TOOL_AGENT_FRONTIER_OPEN_FIRST", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.frontier_open_first_prompt_limit = max(
            1, int(os.getenv("VERL_TOOL_AGENT_FRONTIER_OPEN_FIRST_PROMPT_LIMIT", "3"))
        )
        self.frontier_open_first_escalate_after = max(
            1, int(os.getenv("VERL_TOOL_AGENT_FRONTIER_OPEN_FIRST_ESCALATE_AFTER", "2"))
        )
        self.frontier_open_first_block_limit = max(
            0, int(os.getenv("VERL_TOOL_AGENT_FRONTIER_OPEN_FIRST_BLOCK_LIMIT", "1"))
        )
        self.frontier_open_first_suggestion_limit = min(
            10,
            max(
                1,
                int(os.getenv("VERL_TOOL_AGENT_FRONTIER_OPEN_FIRST_SUGGESTIONS", "10")),
            ),
        )
        self.scaffold_v2_mode = os.getenv("VERL_TOOL_AGENT_SCAFFOLD_V2_MODE", "hard").strip().lower()
        if self.scaffold_v2_mode not in {"hard", "soft"}:
            raise ValueError(
                "VERL_TOOL_AGENT_SCAFFOLD_V2_MODE must be either 'hard' or 'soft', "
                f"got {self.scaffold_v2_mode!r}"
            )
        soft_v2 = self.scaffold_v2_mode == "soft"
        self.search_overlap_threshold = min(
            1.0,
            max(
                0.0,
                float(
                    os.getenv(
                        "VERL_TOOL_AGENT_SEARCH_OVERLAP_THRESHOLD",
                        "0.90" if soft_v2 else "0.75",
                    )
                ),
            ),
        )
        self.search_only_turn_limit = max(
            1,
            int(
                os.getenv(
                    "VERL_TOOL_AGENT_SEARCH_ONLY_TURN_LIMIT",
                    "3" if soft_v2 else "2",
                )
            ),
        )
        self.search_to_open_suggestion_limit = max(
            1, int(os.getenv("VERL_TOOL_AGENT_SEARCH_TO_OPEN_SUGGESTIONS", "5"))
        )
        self.search_to_open_block_limit = max(
            1, int(os.getenv("VERL_TOOL_AGENT_SEARCH_TO_OPEN_BLOCK_LIMIT", "2"))
        )
        self.search_to_open_prompt_limit = max(
            1, int(os.getenv("VERL_TOOL_AGENT_SEARCH_TO_OPEN_PROMPT_LIMIT", "2"))
        )
        self.verification_checkpoint_enabled = os.getenv(
            "VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.verification_checkpoint_exhausted_turns = max(
            1,
            int(
                os.getenv(
                    "VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_EXHAUSTED_TURNS", "2"
                )
            ),
        )
        self.verification_checkpoint_search_only_turns = max(
            1,
            int(
                os.getenv(
                    "VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_SEARCH_ONLY_TURNS", "3"
                )
            ),
        )
        self.verification_checkpoint_search_overlap_threshold = min(
            1.0,
            max(
                0.0,
                float(
                    os.getenv(
                        "VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_SEARCH_OVERLAP_THRESHOLD",
                        "0.90",
                    )
                ),
            ),
        )
        self.verification_checkpoint_duplicate_streak = max(
            1,
            int(
                os.getenv(
                    "VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_DUPLICATE_STREAK", "1"
                )
            ),
        )
        self.verification_checkpoint_suggestion_limit = min(
            12,
            max(
                1,
                int(
                    os.getenv(
                        "VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_SUGGESTIONS", "9"
                    )
                ),
            ),
        )

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields={},
        )
        output.extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "duplicate_tool_calls": agent_data.duplicate_tool_calls,
                "max_duplicate_only_streak": agent_data.max_duplicate_only_streak,
                "final_answer_attempts": agent_data.final_answer_attempts,
                "finalization_reason": agent_data.finalization_reason,
                "finalization_exhausted": agent_data.finalization_exhausted,
                "finalization_rejected_tool_calls": agent_data.finalization_rejected_tool_calls,
                "finalization_balance_failures": agent_data.finalization_balance_failures,
                "finalization_tool_compactions": agent_data.finalization_tool_compactions,
                "generic_answer_rescue_enabled": getattr(
                    self, "generic_answer_rescue_enabled", False
                ),
                "generic_answer_rescue_prefill": getattr(
                    self, "generic_answer_rescue_prefill", True
                ),
                "generic_answer_rescue_triggers": agent_data.generic_answer_rescue_triggers,
                "generic_answer_rescue_rejections": agent_data.generic_answer_rescue_rejections,
                "generic_answer_rescue_missing_rejections": (
                    agent_data.generic_answer_rescue_missing_rejections
                ),
                "generic_answer_rescue_generic_rejections": (
                    agent_data.generic_answer_rescue_generic_rejections
                ),
                "generic_answer_rescue_successes": agent_data.generic_answer_rescue_successes,
                "generic_answer_rescue_exhausted": agent_data.generic_answer_rescue_exhausted,
                "generic_answer_rescue_events": agent_data.generic_answer_rescue_events,
                "final_answer_prefill_enabled": getattr(self, "final_answer_prefill", False),
                "finalization_structure_closure_enabled": getattr(
                    self, "finalization_structure_closure", True
                ),
                "search_calls": agent_data.search_calls,
                "open_calls": agent_data.open_calls,
                "visit_calls": agent_data.visit_calls,
                "find_calls": agent_data.find_calls,
                "search_doc_refs": agent_data.search_doc_refs,
                "searched_docids": sorted(agent_data.searched_docids),
                "opened_docids": sorted(agent_data.opened_docids),
                "find_docids": sorted(agent_data.find_docids),
                "document_flow": agent_data.document_flow,
                "unique_search_doc_ratio": (
                    len(agent_data.searched_docids) / agent_data.search_doc_refs
                    if agent_data.search_doc_refs
                    else 0.0
                ),
                "search_result_overlap_events": agent_data.search_result_overlap_events,
                "max_search_result_overlap": agent_data.max_search_result_overlap,
                "search_frontier_enabled": getattr(self, "search_frontier_enabled", False),
                "search_frontier_requests": agent_data.search_frontier_requests,
                "search_frontier_base_pages": agent_data.search_frontier_base_pages,
                "search_frontier_unseen_pages": agent_data.search_frontier_unseen_pages,
                "search_frontier_exhausted_calls": agent_data.search_frontier_exhausted_calls,
                "search_frontier_selected_docids": sorted(agent_data.search_frontier_selected_docids),
                "search_frontier_events": agent_data.search_frontier_events,
                "frontier_open_first_enabled": getattr(
                    self, "frontier_open_first_enabled", False
                ),
                "frontier_open_first_prompt_limit": getattr(
                    self, "frontier_open_first_prompt_limit", 3
                ),
                "frontier_open_first_escalate_after": getattr(
                    self, "frontier_open_first_escalate_after", 2
                ),
                "frontier_open_first_block_limit": getattr(
                    self, "frontier_open_first_block_limit", 1
                ),
                "frontier_open_first_active": agent_data.frontier_open_first_active,
                "frontier_open_first_pending_docids": agent_data.frontier_open_first_pending_docids,
                "frontier_open_first_ignore_streak": agent_data.frontier_open_first_ignore_streak,
                "frontier_open_first_triggers": agent_data.frontier_open_first_triggers,
                "frontier_open_first_prompts": agent_data.frontier_open_first_prompts,
                "frontier_open_first_ignored_unseen_pages": (
                    agent_data.frontier_open_first_ignored_unseen_pages
                ),
                "frontier_open_first_blocked_calls": agent_data.frontier_open_first_blocked_calls,
                "frontier_open_first_compliances": agent_data.frontier_open_first_compliances,
                "frontier_open_first_pending_compliances": (
                    agent_data.frontier_open_first_pending_compliances
                ),
                "frontier_open_first_other_inspection_releases": (
                    agent_data.frontier_open_first_other_inspection_releases
                ),
                "frontier_open_first_novel_search_releases": (
                    agent_data.frontier_open_first_novel_search_releases
                ),
                "frontier_open_first_suggested_docids": sorted(
                    agent_data.frontier_open_first_suggested_docids
                ),
                "frontier_open_first_events": agent_data.frontier_open_first_events,
                "scaffold_v2_enabled": getattr(self, "scaffold_v2_enabled", False),
                "scaffold_v2_mode": getattr(self, "scaffold_v2_mode", "hard"),
                "search_overlap_threshold": getattr(self, "search_overlap_threshold", 0.75),
                "search_only_turn_limit": getattr(self, "search_only_turn_limit", 2),
                "search_to_open_prompt_limit": getattr(self, "search_to_open_prompt_limit", 2),
                "consecutive_search_only_turns": agent_data.consecutive_search_only_turns,
                "max_consecutive_search_only_turns": agent_data.max_consecutive_search_only_turns,
                "last_search_turn_overlap": agent_data.last_search_turn_overlap,
                "search_to_open_required": agent_data.search_to_open_required,
                "search_overlap_notices": agent_data.search_overlap_notices,
                "search_to_open_triggers": agent_data.search_to_open_triggers,
                "search_to_open_interventions": agent_data.search_to_open_interventions,
                "search_to_open_blocked_calls": agent_data.search_to_open_blocked_calls,
                "search_to_open_compliances": agent_data.search_to_open_compliances,
                "search_to_open_soft_releases": agent_data.search_to_open_soft_releases,
                "search_to_open_suggested_docids": sorted(agent_data.search_to_open_suggested_docids),
                "search_to_open_events": agent_data.search_to_open_events,
                "verification_checkpoint_enabled": getattr(
                    self, "verification_checkpoint_enabled", False
                ),
                "verification_checkpoint_exhausted_turns": getattr(
                    self, "verification_checkpoint_exhausted_turns", 2
                ),
                "verification_checkpoint_search_only_turns": getattr(
                    self, "verification_checkpoint_search_only_turns", 3
                ),
                "verification_checkpoint_search_overlap_threshold": getattr(
                    self, "verification_checkpoint_search_overlap_threshold", 0.90
                ),
                "verification_checkpoint_duplicate_streak": getattr(
                    self, "verification_checkpoint_duplicate_streak", 1
                ),
                "verification_checkpoint_triggered": agent_data.verification_checkpoint_triggered,
                "verification_checkpoint_pending": agent_data.verification_checkpoint_pending,
                "verification_checkpoint_reason": agent_data.verification_checkpoint_reason,
                "verification_checkpoint_triggers": agent_data.verification_checkpoint_triggers,
                "verification_checkpoint_responses": agent_data.verification_checkpoint_responses,
                "verification_checkpoint_structured_responses": (
                    agent_data.verification_checkpoint_structured_responses
                ),
                "verification_checkpoint_soft_releases": (
                    agent_data.verification_checkpoint_soft_releases
                ),
                "verification_checkpoint_consecutive_exhausted_turns": (
                    agent_data.verification_checkpoint_consecutive_exhausted_turns
                ),
                "verification_checkpoint_consecutive_search_only_turns": (
                    agent_data.verification_checkpoint_consecutive_search_only_turns
                ),
                "verification_checkpoint_last_search_overlap": (
                    agent_data.verification_checkpoint_last_search_overlap
                ),
                "verification_checkpoint_max_search_overlap": (
                    agent_data.verification_checkpoint_max_search_overlap
                ),
                "verification_checkpoint_suggested_docids": sorted(
                    agent_data.verification_checkpoint_suggested_docids
                ),
                "verification_checkpoint_action_counts": (
                    agent_data.verification_checkpoint_action_counts
                ),
                "verification_checkpoint_events": agent_data.verification_checkpoint_events,
            }
        )
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        # Keep a hard reserve for final synthesis.  Without this guard, a normal
        # thinking turn can consume the complete response budget and terminate
        # mid-<think> before the agent gets a chance to ask for an answer.
        reserve_tokens = getattr(self, "final_answer_reserve_tokens", 0)
        search_response_limit = max(0, self.response_length - reserve_tokens)
        if (
            not agent_data.force_final_answer
            and not ignore_termination
            and len(agent_data.response_mask) >= search_response_limit
        ):
            return await self._request_final_answer(agent_data, reason="response_reserve")

        remaining_response_tokens = self.response_length - len(agent_data.response_mask)
        if remaining_response_tokens <= 0:
            return AgentState.TERMINATED

        turn_sampling_params = dict(sampling_params)
        if agent_data.force_final_answer:
            max_new_tokens = min(
                remaining_response_tokens,
                getattr(self, "final_answer_attempt_max_tokens", remaining_response_tokens),
            )
        else:
            max_new_tokens = search_response_limit - len(agent_data.response_mask)
        if max_new_tokens <= 0:
            return await self._request_final_answer(agent_data, reason="response_reserve")
        configured_max_tokens = turn_sampling_params.get("max_new_tokens", turn_sampling_params.get("max_tokens"))
        if configured_max_tokens is not None:
            max_new_tokens = min(max_new_tokens, int(configured_max_tokens))
        turn_sampling_params.pop("max_tokens", None)
        turn_sampling_params["max_new_tokens"] = max_new_tokens

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=turn_sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Extract tool calls and retain the non-tool content so a completed final
        # answer is distinguishable from a max-turn response containing only
        # reasoning text.
        response_content, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        agent_data.current_assistant_content = str(response_content or "")

        if agent_data.verification_checkpoint_pending:
            self._record_verification_checkpoint_response(
                agent_data,
                response_content=response_content,
                tool_calls=agent_data.tool_calls,
            )

        if agent_data.force_final_answer:
            agent_data.final_answer_attempts += 1
            was_prefilled = agent_data.final_answer_prefilled
            answer_status = _final_answer_status(response_content, prefilled=was_prefilled)
            if getattr(self, "generic_answer_rescue_enabled", False):
                has_final_answer = answer_status == "concrete"
            else:
                has_final_answer = (
                    _has_nonempty_prefilled_final_answer(response_content)
                    if was_prefilled
                    else _has_nonempty_final_answer(response_content)
                )
            agent_data.final_answer_prefilled = False
            if has_final_answer:
                if getattr(self, "generic_answer_rescue_enabled", False):
                    self._record_generic_answer_success(agent_data)
                return AgentState.TERMINATED
            if getattr(self, "generic_answer_rescue_enabled", False):
                self._record_generic_answer_rejection(
                    agent_data,
                    status=answer_status,
                    phase="finalization",
                )
            if agent_data.final_answer_attempts >= getattr(self, "final_answer_max_attempts", 1):
                agent_data.finalization_exhausted = True
                if agent_data.generic_answer_rescue_active:
                    agent_data.generic_answer_rescue_exhausted = True
                    agent_data.generic_answer_rescue_events.append(
                        {
                            "kind": "rescue_exhausted",
                            "assistant_turn": agent_data.assistant_turns,
                        }
                    )
                if agent_data.tool_calls and getattr(self, "finalization_structure_closure", True):
                    appended = await self._append_finalization_messages(
                        agent_data,
                        rejected_tool_calls=agent_data.tool_calls,
                        instruction=None,
                        prefill_answer=False,
                    )
                    if not appended:
                        agent_data.finalization_balance_failures += len(agent_data.tool_calls)
                return AgentState.TERMINATED
            # A tool call emitted during finalization is rejected rather than
            # executed.  Add balanced synthetic tool responses followed by a
            # higher-priority user correction, then allow one bounded retry.
            return await self._request_final_answer(
                agent_data,
                reason=agent_data.finalization_reason or "final_retry",
                rejected_tool_calls=agent_data.tool_calls,
                retry=True,
                generic_rescue=agent_data.generic_answer_rescue_active,
            )

        if (
            getattr(self, "generic_answer_rescue_enabled", False)
            and not agent_data.tool_calls
            and not self.interaction_config_file
        ):
            answer_status = _final_answer_status(response_content)
            if answer_status != "concrete":
                self._record_generic_answer_rejection(
                    agent_data,
                    status=answer_status,
                    phase="natural_termination",
                )
                return await self._request_final_answer(
                    agent_data,
                    reason=(
                        "generic_nonanswer" if answer_status == "generic" else "missing_answer"
                    ),
                    retry=True,
                    generic_rescue=True,
                )

        if not ignore_termination and len(agent_data.response_mask) >= search_response_limit:
            if _has_nonempty_final_answer(response_content):
                return AgentState.TERMINATED
            if agent_data.tool_calls:
                self._begin_finalization(agent_data, "response_reserve")
                return AgentState.PROCESSING_TOOLS
            return await self._request_final_answer(agent_data, reason="response_reserve")

        assistant_limit_reached = bool(
            self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns
        )
        user_limit_reached = bool(self.max_user_turns and agent_data.user_turns >= self.max_user_turns)
        if assistant_limit_reached or user_limit_reached:
            if _has_nonempty_final_answer(response_content):
                return AgentState.TERMINATED
            if agent_data.tool_calls:
                # Execute the last allowed tool turn, then append a synthesis-only
                # instruction and generate once more.  Previously the state machine
                # terminated here before even parsing this final tool call.
                self._begin_finalization(agent_data, "turn_limit")
                return AgentState.PROCESSING_TOOLS
            return await self._request_final_answer(agent_data, reason="turn_limit")

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        duplicate_flags = []
        selected_tool_calls = agent_data.tool_calls[: self.max_parallel_calls]
        executed_before_turn = set(agent_data.executed_tool_call_keys)
        frontier_repeat_keys_this_turn: set[str] = set()
        searched_docids_before_turn = set(agent_data.searched_docids)
        fresh_inspection_requested = any(
            str(tool_call.name or "").strip().lower() in {"open", "visit"}
            and (
                (call_key := _canonical_tool_call_key(tool_call)) is None
                or call_key not in agent_data.executed_tool_call_keys
            )
            for tool_call in selected_tool_calls
        )
        fresh_frontier_inspection_requested = any(
            str(tool_call.name or "").strip().lower() in {"open", "visit", "find"}
            and (
                (call_key := _canonical_tool_call_key(tool_call)) is None
                or call_key not in agent_data.executed_tool_call_keys
            )
            for tool_call in selected_tool_calls
        )
        block_searches_for_open = bool(
            getattr(self, "scaffold_v2_enabled", False)
            and getattr(self, "scaffold_v2_mode", "hard") == "hard"
            and agent_data.search_to_open_required
            and not fresh_inspection_requested
        )
        block_searches_for_frontier_open = bool(
            getattr(self, "frontier_open_first_enabled", False)
            and agent_data.frontier_open_first_active
            and agent_data.frontier_open_first_block_next_search
            and not fresh_frontier_inspection_requested
        )
        for tool_call in selected_tool_calls:
            tool_name = str(tool_call.name or "").strip().lower()
            call_key = _canonical_tool_call_key(tool_call)
            allow_frontier_repeat = bool(
                getattr(self, "search_frontier_enabled", False)
                and tool_name == "search"
                and call_key is not None
                and call_key in executed_before_turn
                and call_key not in agent_data.search_frontier_exhausted_call_keys
                and call_key not in frontier_repeat_keys_this_turn
            )
            if block_searches_for_frontier_open and tool_name == "search":
                duplicate_flags.append(False)
                tasks.append(
                    self._frontier_open_first_blocked_response(
                        agent_data.frontier_open_first_pending_docids
                    )
                )
            elif block_searches_for_open and tool_name == "search":
                duplicate_flags.append(False)
                tasks.append(
                    self._search_to_open_blocked_response(
                        agent_data.search_to_open_last_suggestions
                        or self._suggest_unopened_docids(agent_data)
                    )
                )
            elif (
                call_key is not None
                and call_key in agent_data.executed_tool_call_keys
                and not allow_frontier_repeat
            ):
                agent_data.duplicate_tool_calls += 1
                duplicate_flags.append(True)
                tasks.append(
                    self._duplicate_tool_call_response(
                        tool_call,
                        repeat_count=agent_data.duplicate_tool_calls,
                    )
                )
            else:
                duplicate_flags.append(False)
                if allow_frontier_repeat and call_key is not None:
                    frontier_repeat_keys_this_turn.add(call_key)
                if call_key is not None:
                    agent_data.executed_tool_call_keys.add(call_key)
                tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        if duplicate_flags and all(duplicate_flags):
            agent_data.duplicate_only_streak += 1
            agent_data.max_duplicate_only_streak = max(
                agent_data.max_duplicate_only_streak,
                agent_data.duplicate_only_streak,
            )
        else:
            agent_data.duplicate_only_streak = 0

        if agent_data.duplicate_only_streak >= getattr(self, "duplicate_only_streak_limit", 2):
            self._begin_finalization(agent_data, "duplicate_loop")

        with simple_timer("tool_calls", agent_data.metrics):
            responses = list(await asyncio.gather(*tasks))

        for tool_call, (_tool_response, _tool_reward, tool_metadata) in zip(
            selected_tool_calls, responses, strict=True
        ):
            self._record_search_frontier_telemetry(agent_data, tool_call, tool_metadata)

        controller_instructions: list[str] = []
        if getattr(self, "scaffold_v2_enabled", False):
            scaffold_instruction = self._update_search_to_open_policy(
                agent_data,
                selected_tool_calls,
                responses,
                searched_docids_before_turn=searched_docids_before_turn,
            )
            if scaffold_instruction:
                controller_instructions.append(scaffold_instruction)
        if getattr(self, "frontier_open_first_enabled", False):
            frontier_instruction = self._update_frontier_open_first_policy(
                agent_data,
                selected_tool_calls,
                responses,
            )
            if frontier_instruction:
                controller_instructions.insert(0, frontier_instruction)
        verification_instruction = None
        if (
            getattr(self, "verification_checkpoint_enabled", False)
            and not agent_data.force_final_answer
        ):
            verification_instruction = self._update_verification_checkpoint_policy(
                agent_data,
                selected_tool_calls,
                responses,
            )
        # The checkpoint is a single, higher-level audit.  Do not dilute it with
        # simultaneous controller prompts; those policies continue normally on
        # later turns because the checkpoint does not mutate their state.
        controller_instruction = (
            verification_instruction
            or ("\n\n".join(controller_instructions) or None)
        )

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_call, (tool_response, tool_reward, tool_metadata) in zip(
            selected_tool_calls, responses, strict=True
        ):
            controller_blocked = isinstance(tool_metadata, dict) and tool_metadata.get(
                "search_to_open_blocked"
            )
            if isinstance(tool_metadata, dict) and tool_metadata.get(
                "frontier_open_first_blocked"
            ):
                controller_blocked = True
            if (
                not isinstance(tool_metadata, dict)
                or not tool_metadata.get("duplicate_tool_call")
            ) and not controller_blocked:
                self._record_tool_telemetry(agent_data, tool_call, tool_response)

            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        if controller_instruction and not agent_data.force_final_answer:
            add_messages.append({"role": "user", "content": controller_instruction})

        # Finalization is a separate user turn.  Putting this instruction inside
        # a tool response proved too weak for Qwen3.5: it treated the text as an
        # observation and emitted another tool call in most trajectories.
        if agent_data.force_final_answer and self.tool_parser_name != "gpt-oss":
            add_messages.append({"role": "user", "content": self._final_answer_instruction(retry=False)})

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

            # A tool observation can move the sequence into the reserved final
            # budget.  Re-render the suffix with a user-level final instruction
            # now, before an empty assistant generation prefix is appended.
            search_response_limit = self.response_length - getattr(self, "final_answer_reserve_tokens", 0)
            if (
                not agent_data.force_final_answer
                and len(agent_data.response_mask) + len(response_ids) >= search_response_limit
            ):
                self._begin_finalization(agent_data, "response_reserve")
                add_messages.append({"role": "user", "content": self._final_answer_instruction(retry=False)})
                response_ids = await self.apply_chat_template(
                    add_messages,
                    images=images,
                    videos=videos,
                    remove_system_prompt=True,
                )

            if agent_data.force_final_answer and getattr(self, "finalization_structure_closure", True):
                remaining_tokens = self.response_length - len(agent_data.response_mask)
                remaining_attempts = max(
                    1,
                    getattr(self, "final_answer_max_attempts", 1) - agent_data.final_answer_attempts,
                )
                # Hold the complete retry budget plus a small structural margin:
                # a failed final attempt may itself emit tool calls, which still
                # need synthetic tool responses and a retry instruction.
                final_generation_tokens = min(
                    getattr(self, "final_answer_attempt_max_tokens", 1) * remaining_attempts + 1024,
                    max(1, remaining_tokens - 1),
                )
                suffix_token_limit = remaining_tokens - final_generation_tokens
                fitted = await self._fit_finalization_suffix(
                    add_messages,
                    token_limit=suffix_token_limit,
                    images=images,
                    videos=videos,
                )
                if fitted is None:
                    agent_data.finalization_balance_failures += len(selected_tool_calls)
                    return AgentState.TERMINATED
                response_ids, compactions = fitted
                agent_data.finalization_tool_compactions += compactions

            if agent_data.force_final_answer and getattr(self, "final_answer_prefill", False):
                response_ids = await self._append_answer_prefill(response_ids, agent_data)

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        agent_data.messages.extend(add_messages)

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _fit_finalization_suffix(
        self,
        messages: list[dict[str, Any]],
        *,
        token_limit: int,
        images: Optional[list[Any]],
        videos: Optional[list[Any]],
    ) -> Optional[tuple[list[int], int]]:
        """Fit balanced tool observations while preserving room for final synthesis."""

        if token_limit <= 0:
            return None

        async def render(candidate_messages: list[dict[str, Any]]) -> list[int]:
            return await self.apply_chat_template(
                candidate_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        rendered = await render(messages)
        if len(rendered) < token_limit:
            return rendered, 0

        text_tool_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "tool" and isinstance(message.get("content"), str)
        ]
        if not text_tool_indices:
            return None

        original_contents = {index: str(messages[index].get("content") or "") for index in text_tool_indices}
        max_chars = max((len(content) for content in original_contents.values()), default=0)
        best: Optional[tuple[list[int], list[dict[str, Any]]]] = None
        low, high = 0, max_chars
        while low <= high:
            char_limit = (low + high) // 2
            candidate_messages = [dict(message) for message in messages]
            for index, content in original_contents.items():
                if len(content) <= char_limit:
                    continue
                marker = "\n...[finalization context compacted]...\n"
                if char_limit <= len(marker):
                    compacted = marker[:char_limit]
                else:
                    side = (char_limit - len(marker)) // 2
                    compacted = content[:side] + marker + content[-(char_limit - len(marker) - side) :]
                candidate_messages[index]["content"] = compacted
            candidate_ids = await render(candidate_messages)
            if len(candidate_ids) < token_limit:
                best = candidate_ids, candidate_messages
                low = char_limit + 1
            else:
                high = char_limit - 1

        if best is None:
            return None
        response_ids, fitted_messages = best
        messages[:] = fitted_messages
        compactions = sum(
            fitted_messages[index].get("content") != original_contents[index] for index in text_tool_indices
        )
        return response_ids, compactions

    def _suggest_unopened_docids(
        self,
        agent_data: AgentData,
        preferred: Optional[list[str]] = None,
    ) -> list[str]:
        """Return a small, stable list of visible search cursors not yet opened."""

        suggestions: list[str] = []
        seen: set[str] = set()
        candidates = [*(preferred or []), *agent_data.search_candidate_docids]
        for cursor in candidates:
            if cursor in seen or cursor in agent_data.opened_docids:
                continue
            seen.add(cursor)
            suggestions.append(cursor)
            if len(suggestions) >= getattr(self, "search_to_open_suggestion_limit", 5):
                break
        return suggestions

    def _verification_checkpoint_suggestions(
        self,
        agent_data: AgentData,
        preferred: Optional[list[str]] = None,
    ) -> list[str]:
        """Return visible, uninspected cursors for the one-shot audit prompt."""

        suggestions: list[str] = []
        seen: set[str] = set()
        inspected = agent_data.opened_docids | agent_data.find_docids
        candidates = [
            *(preferred or []),
            *reversed(agent_data.search_candidate_docids),
            *sorted(agent_data.searched_docids),
        ]
        for cursor in candidates:
            cursor = str(cursor or "").strip().rstrip(".,;:)]}")
            if not cursor:
                continue
            if not cursor.lower().startswith("doc://"):
                cursor = f"doc://{cursor}"
            if cursor in seen or cursor in inspected:
                continue
            seen.add(cursor)
            suggestions.append(cursor)
            if len(suggestions) >= getattr(
                self, "verification_checkpoint_suggestion_limit", 9
            ):
                break
        return suggestions

    @staticmethod
    def _verification_checkpoint_instruction(
        *,
        reason: str,
        suggestions: list[str],
    ) -> str:
        cursor_text = ", ".join(suggestions) if suggestions else "none currently visible"
        return (
            "LOCAL_VERIFICATION_CHECKPOINT\n"
            f"trigger: {reason}\n"
            "The current research path is no longer making reliably grounded progress. Tools remain "
            "available, but before the next action perform one concise local audit using only evidence "
            "already present in this conversation. Do not invent a quotation or claim.\n"
            "Write an ordinary plain-text state block, not XML and not a tool call. Never call a tool or "
            "function named research_checkpoint. Use exactly this envelope and these fields:\n"
            "RESEARCH_CHECKPOINT_STATE_BEGIN\n"
            "requirements: split the original question into atomic constraints, each marked supported, "
            "contradicted, or missing\n"
            "candidates: each live bridge/answer candidate with supporting doc:// cursor and exact short "
            "phrase; mark unsupported candidates explicitly\n"
            "discarded_hypotheses: failed directions and the evidence/reason that rejects them\n"
            "critical_gap: the single most important unresolved constraint\n"
            "next_action: OPEN_EXISTING, SEARCH_GAP, or ANSWER\n"
            "next_queries: only for SEARCH_GAP, at most three complementary queries that test different "
            "constraints rather than paraphrasing an earlier query\n"
            "RESEARCH_CHECKPOINT_STATE_END\n"
            f"Visible uninspected cursors (optional, not mandatory): {cursor_text}.\n"
            "Immediately carry out that next_action in the same response: call open/find/visit, issue the "
            "new search queries, or return one concrete short answer inside <answer></answer>. Do not stop "
            "after writing the audit."
        )

    def _update_verification_checkpoint_policy(
        self,
        agent_data: AgentData,
        tool_calls: list[FunctionCall],
        responses: list[tuple[ToolResponse, float, dict]],
    ) -> Optional[str]:
        """Update stalled-progress counters and optionally emit one audit prompt.

        The current tool turn has already executed.  The checkpoint therefore
        preserves its evidence and only steers the next model action.  It never
        blocks a call and can trigger at most once per trajectory.
        """

        actual_search_calls = 0
        exhausted_search_calls = 0
        successful_inspection = False
        preferred_docids: list[str] = []
        current_search_docids: set[str] = set()

        for tool_call, (tool_response, _tool_reward, tool_metadata) in zip(
            tool_calls, responses, strict=True
        ):
            metadata = tool_metadata if isinstance(tool_metadata, dict) else {}
            if any(
                metadata.get(key)
                for key in (
                    "duplicate_tool_call",
                    "search_to_open_blocked",
                    "frontier_open_first_blocked",
                )
            ):
                continue

            tool_name = str(tool_call.name or "").strip().lower()
            response_text = str(tool_response.text or "")
            response_failed = response_text.startswith("Error when executing tool:")
            if tool_name == "search" and not response_failed:
                actual_search_calls += 1
                if metadata.get("search_frontier_exhausted"):
                    exhausted_search_calls += 1
                current_cursors = _extract_doc_cursor_sequence(response_text)
                preferred_docids.extend(current_cursors)
                current_search_docids.update(current_cursors)
            elif tool_name in {"open", "visit", "find"} and not response_failed:
                successful_inspection = True

        if actual_search_calls and exhausted_search_calls == actual_search_calls:
            agent_data.verification_checkpoint_consecutive_exhausted_turns += 1
        elif actual_search_calls or successful_inspection:
            agent_data.verification_checkpoint_consecutive_exhausted_turns = 0

        if actual_search_calls and not successful_inspection:
            agent_data.verification_checkpoint_consecutive_search_only_turns += 1
        elif successful_inspection:
            agent_data.verification_checkpoint_consecutive_search_only_turns = 0

        if current_search_docids:
            overlap = len(current_search_docids & agent_data.searched_docids) / len(
                current_search_docids
            )
            agent_data.verification_checkpoint_last_search_overlap = overlap
            agent_data.verification_checkpoint_max_search_overlap = max(
                agent_data.verification_checkpoint_max_search_overlap,
                overlap,
            )
        elif actual_search_calls:
            agent_data.verification_checkpoint_last_search_overlap = 0.0

        if agent_data.verification_checkpoint_triggered:
            return None

        suggestions = self._verification_checkpoint_suggestions(
            agent_data,
            preferred=preferred_docids,
        )
        reason: Optional[str] = None
        if agent_data.duplicate_only_streak >= getattr(
            self, "verification_checkpoint_duplicate_streak", 1
        ):
            reason = "duplicate_only_streak"
        elif agent_data.verification_checkpoint_consecutive_exhausted_turns >= getattr(
            self, "verification_checkpoint_exhausted_turns", 2
        ):
            reason = "frontier_exhausted"
        elif (
            suggestions
            and agent_data.verification_checkpoint_consecutive_search_only_turns
            >= getattr(self, "verification_checkpoint_search_only_turns", 3)
            and agent_data.verification_checkpoint_last_search_overlap
            >= getattr(
                self,
                "verification_checkpoint_search_overlap_threshold",
                0.90,
            )
        ):
            reason = "search_stall_with_uninspected_candidates"

        if reason is None:
            return None

        agent_data.verification_checkpoint_triggered = True
        agent_data.verification_checkpoint_pending = True
        agent_data.verification_checkpoint_reason = reason
        agent_data.verification_checkpoint_triggers += 1
        agent_data.verification_checkpoint_suggested_docids.update(suggestions)
        agent_data.verification_checkpoint_events.append(
            {
                "kind": "triggered",
                "reason": reason,
                "assistant_turn": agent_data.assistant_turns,
                "exhausted_turns": (
                    agent_data.verification_checkpoint_consecutive_exhausted_turns
                ),
                "search_only_turns": (
                    agent_data.verification_checkpoint_consecutive_search_only_turns
                ),
                "search_overlap": agent_data.verification_checkpoint_last_search_overlap,
                "duplicate_only_streak": agent_data.duplicate_only_streak,
                "suggested_docids": list(suggestions),
            }
        )
        return self._verification_checkpoint_instruction(
            reason=reason,
            suggestions=suggestions,
        )

    @staticmethod
    def _record_verification_checkpoint_response(
        agent_data: AgentData,
        *,
        response_content: str,
        tool_calls: list[FunctionCall],
    ) -> None:
        """Record checkpoint compliance once, then unconditionally soft-release."""

        content = str(response_content or "")
        normalized = content.lower()
        structured = bool(
            "research_checkpoint_state_begin" in normalized
            and "research_checkpoint_state_end" in normalized
            and all(
                field in normalized
                for field in (
                    "requirements:",
                    "candidates:",
                    "discarded_hypotheses:",
                    "critical_gap:",
                    "next_action:",
                    "next_queries:",
                )
            )
        )
        tool_names = {
            str(tool_call.name or "").strip().lower() for tool_call in tool_calls
        }
        if tool_names & {"open", "visit", "find"}:
            action = "open_existing"
        elif "search" in tool_names:
            action = "search_gap"
        elif _has_nonempty_final_answer(content):
            action = "answer"
        else:
            action = "other"

        agent_data.verification_checkpoint_pending = False
        agent_data.verification_checkpoint_responses += 1
        agent_data.verification_checkpoint_action_counts[action] += 1
        if structured:
            agent_data.verification_checkpoint_structured_responses += 1
        else:
            agent_data.verification_checkpoint_soft_releases += 1
        agent_data.verification_checkpoint_events.append(
            {
                "kind": "response",
                "reason": agent_data.verification_checkpoint_reason,
                "assistant_turn": agent_data.assistant_turns,
                "structured": structured,
                "action": action,
            }
        )

    def _search_to_open_instruction(
        self,
        agent_data: AgentData,
        *,
        overlap: Optional[float] = None,
    ) -> str:
        suggestions = agent_data.search_to_open_last_suggestions or self._suggest_unopened_docids(agent_data)
        cursor_text = ", ".join(suggestions) if suggestions else "a previously returned OpenURL cursor"
        overlap_text = (
            f" ({overlap:.0%} of this turn's returned documents were already seen)"
            if overlap is not None
            else ""
        )
        if getattr(self, "scaffold_v2_mode", "hard") == "soft":
            return (
                "SEARCH_TO_OPEN_RECOMMENDED\n"
                f"Recent search-only turns are adding few new documents{overlap_text}. "
                "Prefer inspecting at least one promising unvisited result with open before searching again. "
                f"Suggested exact cursors: {cursor_text}. "
                "If none addresses the missing constraint, you may instead issue one genuinely new, "
                "discriminative search; do not merely paraphrase an earlier query."
            )
        return (
            "SEARCH_TO_OPEN_REQUIRED\n"
            f"Recent search-only turns have stopped adding enough new evidence{overlap_text}. "
            "Do not issue another search-only turn. Inspect at least one unvisited result with open first, "
            "or return the final answer if the existing evidence is sufficient. "
            f"Suggested exact cursors: {cursor_text}."
        )

    @staticmethod
    async def _search_to_open_blocked_response(
        suggestions: list[str],
    ) -> tuple[ToolResponse, float, dict]:
        cursor_text = ", ".join(suggestions) if suggestions else "an unvisited OpenURL from an earlier result"
        return (
            ToolResponse(
                text=(
                    "SEARCH_TO_OPEN_BLOCKED\n"
                    "This search was not executed because recent search results substantially overlap earlier "
                    "results and no new document has been opened since the runtime requested inspection. "
                    f"Open one of these candidates first: {cursor_text}. "
                    "Search becomes available again after a fresh open call."
                )
            ),
            0.0,
            {"search_to_open_blocked": True},
        )

    def _frontier_open_first_suggestions(
        self,
        agent_data: AgentData,
        values: list[Any],
    ) -> list[str]:
        """Normalize an unseen page into ordered, still-uninspected cursors."""

        inspected = agent_data.opened_docids | agent_data.find_docids
        suggestions: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip().rstrip(".,;:)]}")
            if not text:
                continue
            cursor = text if text.lower().startswith("doc://") else f"doc://{text}"
            if cursor in seen or cursor in inspected:
                continue
            seen.add(cursor)
            suggestions.append(cursor)
            if len(suggestions) >= getattr(
                self, "frontier_open_first_suggestion_limit", 10
            ):
                break
        return suggestions

    @staticmethod
    def _clear_frontier_open_first_cycle(agent_data: AgentData) -> None:
        agent_data.frontier_open_first_active = False
        agent_data.frontier_open_first_pending_docids = []
        agent_data.frontier_open_first_pending_docid_set.clear()
        agent_data.frontier_open_first_ignore_streak = 0
        agent_data.frontier_open_first_block_next_search = False
        agent_data.frontier_open_first_blocks_in_cycle = 0

    def _frontier_open_first_instruction(
        self,
        agent_data: AgentData,
        *,
        kind: str,
    ) -> str:
        suggestions = agent_data.frontier_open_first_pending_docids
        cursor_text = ", ".join(suggestions) if suggestions else "an OpenURL from the latest unseen page"
        if kind == "blocked":
            return (
                "FRONTIER_OPEN_FIRST_REQUIRED\n"
                "A further search-only call was blocked once because multiple unseen frontier pages were "
                "requested without inspecting any candidate. Inspect one promising result with open or find "
                f"before continuing this line of research. Exact cursors: {cursor_text}. "
                "After an inspection, search remains available; if this direction is irrelevant, pivot using "
                "a genuinely new entity, relation, date, title, or quoted phrase."
            )
        if kind == "ignored":
            return (
                "FRONTIER_OPEN_FIRST_REPEATED\n"
                "Another unseen frontier page was requested without inspecting a candidate from the prior page. "
                "Before continuing the same search direction, inspect at least one promising result with open "
                f"or find. Review all ten positions; entries 6-10 are compact, not lower-confidence. "
                f"Exact cursors: {cursor_text}. A genuinely new discriminative query is still allowed."
            )
        return (
            "FRONTIER_OPEN_FIRST_RECOMMENDED\n"
            "The latest search returned previously unseen candidates from the current frontier. Before "
            "continuing the same search direction, inspect at least one promising result with open or find. "
            "Review all ten positions; entries 6-10 are compact, not lower-confidence. "
            f"Exact cursors: {cursor_text}. If none addresses the missing constraint, you may instead pivot "
            "with a genuinely new, discriminative query."
        )

    @staticmethod
    async def _frontier_open_first_blocked_response(
        suggestions: list[str],
    ) -> tuple[ToolResponse, float, dict]:
        cursor_text = ", ".join(suggestions) if suggestions else "an OpenURL from the latest unseen page"
        return (
            ToolResponse(
                text=(
                    "FRONTIER_OPEN_FIRST_BLOCKED\n"
                    "This search was not executed because multiple unseen frontier pages were requested without "
                    "inspecting a returned candidate. This is a one-shot evidence-use intervention, not a "
                    f"permanent search ban. Inspect one promising cursor first: {cursor_text}."
                )
            ),
            0.0,
            {"frontier_open_first_blocked": True},
        )

    def _update_frontier_open_first_policy(
        self,
        agent_data: AgentData,
        tool_calls: list[FunctionCall],
        responses: list[tuple[ToolResponse, float, dict]],
    ) -> Optional[str]:
        """Prompt for evidence use after unseen pages without blocking novel pivots.

        The first follow-up search is always executed.  Its frontier metadata is
        the online discriminator: another unseen page means continued pagination,
        while a base page means the query found a genuinely novel candidate set.
        Only repeated unseen pagination arms a one-shot search block.
        """

        active_before_turn = agent_data.frontier_open_first_active
        pending_before_turn = set(agent_data.frontier_open_first_pending_docid_set)
        successful_inspection = False
        inspected_pending = False
        unseen_pages = 0
        unseen_docids: list[str] = []
        saw_base_page = False
        saw_exhausted = False
        blocked_search_calls = 0

        for tool_call, (tool_response, _tool_reward, tool_metadata) in zip(
            tool_calls, responses, strict=True
        ):
            metadata = tool_metadata if isinstance(tool_metadata, dict) else {}
            if metadata.get("duplicate_tool_call") or metadata.get("search_to_open_blocked"):
                continue
            if metadata.get("frontier_open_first_blocked"):
                blocked_search_calls += 1
                continue

            tool_name = str(tool_call.name or "").strip().lower()
            response_text = str(tool_response.text or "")
            response_failed = response_text.startswith("Error when executing tool:")
            if tool_name in {"open", "visit", "find"} and not response_failed:
                successful_inspection = True
                inspected_docids = _extract_requested_doc_cursors(tool_call) | _extract_doc_cursors(
                    response_text
                )
                if inspected_docids & pending_before_turn:
                    inspected_pending = True
                continue

            if tool_name != "search" or not metadata.get("search_frontier_enabled"):
                continue
            if metadata.get("search_frontier_exhausted"):
                saw_exhausted = True
            elif metadata.get("search_frontier_applied"):
                unseen_pages += 1
                unseen_docids.extend(metadata.get("search_frontier_selected_docids") or [])
            else:
                saw_base_page = True

        if blocked_search_calls:
            agent_data.frontier_open_first_blocked_calls += blocked_search_calls
            agent_data.frontier_open_first_block_next_search = False
            agent_data.frontier_open_first_blocks_in_cycle += 1
            agent_data.frontier_open_first_ignore_streak = 0
            agent_data.frontier_open_first_events.append(
                {
                    "kind": "blocked_once",
                    "assistant_turn": agent_data.assistant_turns,
                    "blocked_calls": blocked_search_calls,
                    "suggested_docids": list(agent_data.frontier_open_first_pending_docids),
                }
            )
            return self._frontier_open_first_instruction(agent_data, kind="blocked")

        if successful_inspection and active_before_turn:
            agent_data.frontier_open_first_compliances += 1
            event_kind = "opened_pending" if inspected_pending else "other_inspection"
            if inspected_pending:
                agent_data.frontier_open_first_pending_compliances += 1
            else:
                agent_data.frontier_open_first_other_inspection_releases += 1
            agent_data.frontier_open_first_events.append(
                {
                    "kind": event_kind,
                    "assistant_turn": agent_data.assistant_turns,
                }
            )
            self._clear_frontier_open_first_cycle(agent_data)
            return None

        if active_before_turn:
            if unseen_pages:
                suggestions = self._frontier_open_first_suggestions(
                    agent_data, unseen_docids
                )
                if suggestions:
                    agent_data.frontier_open_first_pending_docids = suggestions
                    agent_data.frontier_open_first_pending_docid_set = set(suggestions)
                    agent_data.frontier_open_first_suggested_docids.update(suggestions)
                agent_data.frontier_open_first_ignore_streak += 1
                agent_data.frontier_open_first_ignored_unseen_pages += unseen_pages
                block_limit = getattr(self, "frontier_open_first_block_limit", 1)
                if (
                    agent_data.frontier_open_first_ignore_streak
                    >= getattr(self, "frontier_open_first_escalate_after", 2)
                    and agent_data.frontier_open_first_blocks_in_cycle < block_limit
                ):
                    agent_data.frontier_open_first_block_next_search = True
                agent_data.frontier_open_first_events.append(
                    {
                        "kind": "ignored_with_unseen_page",
                        "assistant_turn": agent_data.assistant_turns,
                        "unseen_pages": unseen_pages,
                        "ignore_streak": agent_data.frontier_open_first_ignore_streak,
                        "block_armed": agent_data.frontier_open_first_block_next_search,
                        "suggested_docids": list(agent_data.frontier_open_first_pending_docids),
                    }
                )
                if agent_data.frontier_open_first_prompts < getattr(
                    self, "frontier_open_first_prompt_limit", 3
                ):
                    agent_data.frontier_open_first_prompts += 1
                    return self._frontier_open_first_instruction(agent_data, kind="ignored")
                return None

            if saw_base_page:
                agent_data.frontier_open_first_novel_search_releases += 1
                agent_data.frontier_open_first_events.append(
                    {
                        "kind": "released_by_base_page",
                        "assistant_turn": agent_data.assistant_turns,
                    }
                )
                self._clear_frontier_open_first_cycle(agent_data)
                return None

            if saw_exhausted:
                agent_data.frontier_open_first_ignore_streak += 1
                block_limit = getattr(self, "frontier_open_first_block_limit", 1)
                if (
                    agent_data.frontier_open_first_ignore_streak
                    >= getattr(self, "frontier_open_first_escalate_after", 2)
                    and agent_data.frontier_open_first_blocks_in_cycle < block_limit
                ):
                    agent_data.frontier_open_first_block_next_search = True
                agent_data.frontier_open_first_events.append(
                    {
                        "kind": "exhausted_while_pending",
                        "assistant_turn": agent_data.assistant_turns,
                        "ignore_streak": agent_data.frontier_open_first_ignore_streak,
                        "block_armed": agent_data.frontier_open_first_block_next_search,
                    }
                )
                if agent_data.frontier_open_first_prompts < getattr(
                    self, "frontier_open_first_prompt_limit", 3
                ):
                    agent_data.frontier_open_first_prompts += 1
                    return self._frontier_open_first_instruction(agent_data, kind="ignored")
                return None

        if unseen_pages:
            suggestions = self._frontier_open_first_suggestions(agent_data, unseen_docids)
            if not suggestions:
                return None
            if agent_data.frontier_open_first_prompts >= getattr(
                self, "frontier_open_first_prompt_limit", 3
            ):
                agent_data.frontier_open_first_events.append(
                    {
                        "kind": "prompt_limit_suppressed",
                        "assistant_turn": agent_data.assistant_turns,
                        "unseen_pages": unseen_pages,
                    }
                )
                return None
            agent_data.frontier_open_first_active = True
            agent_data.frontier_open_first_pending_docids = suggestions
            agent_data.frontier_open_first_pending_docid_set = set(suggestions)
            agent_data.frontier_open_first_ignore_streak = 0
            agent_data.frontier_open_first_block_next_search = False
            agent_data.frontier_open_first_blocks_in_cycle = 0
            agent_data.frontier_open_first_triggers += 1
            agent_data.frontier_open_first_prompts += 1
            agent_data.frontier_open_first_suggested_docids.update(suggestions)
            agent_data.frontier_open_first_events.append(
                {
                    "kind": "recommended",
                    "assistant_turn": agent_data.assistant_turns,
                    "unseen_pages": unseen_pages,
                    "suggested_docids": list(suggestions),
                }
            )
            return self._frontier_open_first_instruction(agent_data, kind="recommended")

        return None

    def _update_search_to_open_policy(
        self,
        agent_data: AgentData,
        tool_calls: list[FunctionCall],
        responses: list[tuple[ToolResponse, float, dict]],
        *,
        searched_docids_before_turn: set[str],
    ) -> Optional[str]:
        """Update scaffold-v2 state and return an optional user-level instruction.

        Search calls in the current turn have already completed when this runs.
        A requirement raised here therefore affects only the next turn. This keeps
        new queries executable until their actual document overlap is observable.
        """

        actual_search_indices: list[int] = []
        turn_cursor_sequence: list[str] = []
        turn_cursor_set: set[str] = set()
        successful_document_inspection = False
        successful_open_or_visit = False
        blocked_search_calls = 0
        executed_nonblocked_calls = 0

        for index, (tool_call, response) in enumerate(zip(tool_calls, responses, strict=True)):
            tool_response, _tool_reward, tool_metadata = response
            metadata = tool_metadata if isinstance(tool_metadata, dict) else {}
            if metadata.get("frontier_open_first_blocked"):
                continue
            if metadata.get("search_to_open_blocked"):
                blocked_search_calls += 1
                continue
            if metadata.get("duplicate_tool_call"):
                continue

            executed_nonblocked_calls += 1
            tool_name = str(tool_call.name or "").strip().lower()
            response_text = str(tool_response.text or "")
            response_failed = response_text.startswith("Error when executing tool:")
            if tool_name == "search":
                cursors = _extract_doc_cursor_sequence(response_text)
                if cursors:
                    actual_search_indices.append(index)
                    for cursor in cursors:
                        if cursor not in turn_cursor_set:
                            turn_cursor_set.add(cursor)
                            turn_cursor_sequence.append(cursor)
                        if cursor not in agent_data.search_candidate_docid_set:
                            agent_data.search_candidate_docid_set.add(cursor)
                            agent_data.search_candidate_docids.append(cursor)
            elif tool_name in {"open", "visit", "find"} and not response_failed:
                successful_document_inspection = True
                if tool_name in {"open", "visit"}:
                    successful_open_or_visit = True

        if successful_open_or_visit and agent_data.search_to_open_required:
            agent_data.search_to_open_required = False
            agent_data.search_to_open_block_streak = 0
            agent_data.search_to_open_compliances += 1
            agent_data.search_to_open_events.append(
                {
                    "kind": "complied",
                    "assistant_turn": agent_data.assistant_turns,
                }
            )

        if actual_search_indices and not successful_document_inspection:
            agent_data.consecutive_search_only_turns += 1
            agent_data.max_consecutive_search_only_turns = max(
                agent_data.max_consecutive_search_only_turns,
                agent_data.consecutive_search_only_turns,
            )
        elif executed_nonblocked_calls:
            agent_data.consecutive_search_only_turns = 0

        overlap = 0.0
        if turn_cursor_set:
            overlap = len(turn_cursor_set & searched_docids_before_turn) / len(turn_cursor_set)
            agent_data.last_search_turn_overlap = overlap

        controller_mode = getattr(self, "scaffold_v2_mode", "hard")
        if (
            controller_mode == "soft"
            and agent_data.search_to_open_required
            and turn_cursor_set
            and overlap < getattr(self, "search_overlap_threshold", 0.90)
        ):
            # A genuinely novel search is evidence that the model found a useful
            # new constraint. Release the soft recommendation without requiring
            # an open; future saturation can start another cycle.
            agent_data.search_to_open_required = False
            agent_data.consecutive_search_only_turns = 1
            agent_data.search_to_open_soft_releases += 1
            agent_data.search_to_open_events.append(
                {
                    "kind": "released_by_novel_search",
                    "assistant_turn": agent_data.assistant_turns,
                    "overlap": overlap,
                }
            )

        suggestions = self._suggest_unopened_docids(agent_data, preferred=turn_cursor_sequence)
        high_overlap = bool(
            turn_cursor_set
            and overlap >= getattr(self, "search_overlap_threshold", 0.75)
            and suggestions
            and not successful_document_inspection
        )

        prompt_limit_reached = bool(
            controller_mode == "soft"
            and agent_data.search_to_open_triggers
            >= getattr(self, "search_to_open_prompt_limit", 2)
        )
        newly_required = bool(
            high_overlap
            and not successful_document_inspection
            and agent_data.consecutive_search_only_turns
            >= getattr(self, "search_only_turn_limit", 2)
            and not agent_data.search_to_open_required
            and not prompt_limit_reached
        )

        # Hard-v2 preserves its original per-overlap notices for exact rollback.
        # Soft-v2 emits only the notice attached to a new recommendation cycle.
        emit_overlap_notice = bool(
            high_overlap
            and actual_search_indices
            and (controller_mode == "hard" or newly_required)
        )
        if emit_overlap_notice:
            agent_data.search_overlap_notices += 1
            notice = (
                "\n\nSEARCH_RESULT_OVERLAP_NOTICE\n"
                f"historical_overlap: {overlap:.0%} "
                f"({len(turn_cursor_set & searched_docids_before_turn)}/{len(turn_cursor_set)} documents)\n"
                f"unvisited_candidates: {', '.join(suggestions)}\n"
                "Prefer opening one of these exact cursors before issuing another search."
            )
            notice_index = actual_search_indices[-1]
            tool_response, tool_reward, tool_metadata = responses[notice_index]
            tool_response.text = f"{tool_response.text or ''}{notice}"
            responses[notice_index] = (tool_response, tool_reward, tool_metadata)

        if newly_required:
            agent_data.search_to_open_required = True
            agent_data.search_to_open_block_streak = 0
            agent_data.search_to_open_last_suggestions = suggestions
            agent_data.search_to_open_suggested_docids.update(suggestions)
            agent_data.search_to_open_triggers += 1
            agent_data.search_to_open_interventions += 1
            agent_data.search_to_open_events.append(
                {
                    "kind": "required",
                    "assistant_turn": agent_data.assistant_turns,
                    "overlap": overlap,
                    "suggested_docids": suggestions,
                }
            )
            return self._search_to_open_instruction(agent_data, overlap=overlap)

        if blocked_search_calls:
            agent_data.search_to_open_block_streak += 1
            agent_data.search_to_open_blocked_calls += blocked_search_calls
            agent_data.search_to_open_interventions += 1
            suggestions = agent_data.search_to_open_last_suggestions or self._suggest_unopened_docids(agent_data)
            agent_data.search_to_open_suggested_docids.update(suggestions)
            agent_data.search_to_open_events.append(
                {
                    "kind": "blocked",
                    "assistant_turn": agent_data.assistant_turns,
                    "blocked_calls": blocked_search_calls,
                    "suggested_docids": suggestions,
                }
            )
            if agent_data.search_to_open_block_streak >= getattr(
                self, "search_to_open_block_limit", 2
            ):
                self._begin_finalization(agent_data, "search_to_open_stall")
                return None
            return self._search_to_open_instruction(agent_data)

        return None

    @staticmethod
    def _record_search_frontier_telemetry(
        agent_data: AgentData,
        tool_call: FunctionCall,
        tool_metadata: Any,
    ) -> None:
        if str(tool_call.name or "").strip().lower() != "search" or not isinstance(
            tool_metadata, dict
        ):
            return
        if not tool_metadata.get("search_frontier_enabled"):
            return

        applied = bool(tool_metadata.get("search_frontier_applied"))
        exhausted = bool(tool_metadata.get("search_frontier_exhausted"))
        selected_docids = [
            f"doc://{str(value).removeprefix('doc://')}"
            for value in (tool_metadata.get("search_frontier_selected_docids") or [])
            if str(value or "").strip()
        ]
        agent_data.search_frontier_requests += 1
        if exhausted:
            agent_data.search_frontier_exhausted_calls += 1
        elif applied:
            agent_data.search_frontier_unseen_pages += 1
        else:
            agent_data.search_frontier_base_pages += 1
        if exhausted:
            call_key = _canonical_tool_call_key(tool_call)
            if call_key is not None:
                agent_data.search_frontier_exhausted_call_keys.add(call_key)
        agent_data.search_frontier_selected_docids.update(selected_docids)
        agent_data.search_frontier_events.append(
            {
                "kind": "exhausted" if exhausted else ("unseen_page" if applied else "base_page"),
                "assistant_turn": agent_data.assistant_turns,
                "base_overlap": float(tool_metadata.get("search_frontier_base_overlap", 0.0)),
                "candidate_count": int(tool_metadata.get("search_frontier_candidate_count", 0)),
                "unseen_count": int(tool_metadata.get("search_frontier_unseen_count", 0)),
                "selected_docids": selected_docids,
                "selected_original_ranks": list(
                    tool_metadata.get("search_frontier_selected_original_ranks") or []
                ),
                "dual_retrieval_enabled": bool(
                    tool_metadata.get("dual_retrieval_enabled", False)
                ),
                "dual_retrieval_applied": bool(
                    tool_metadata.get("dual_retrieval_applied", False)
                ),
                "dual_retrieval_fallback_error": tool_metadata.get(
                    "dual_retrieval_fallback_error"
                ),
                "dual_primary_candidates": int(
                    tool_metadata.get("dual_retrieval_primary_candidates", 0)
                ),
                "dual_secondary_candidates": int(
                    tool_metadata.get("dual_retrieval_secondary_candidates", 0)
                ),
                "dual_fused_candidates": int(
                    tool_metadata.get("dual_retrieval_fused_candidates", 0)
                ),
            }
        )

    @staticmethod
    def _record_tool_telemetry(
        agent_data: AgentData,
        tool_call: FunctionCall,
        tool_response: ToolResponse,
    ) -> None:
        """Record document flow without changing tool execution or agent policy."""

        tool_name = str(tool_call.name or "").strip().lower()
        response_text = str(tool_response.text or "")
        # Scaffold notices can repeat earlier candidate cursors for the model.
        # Keep retrieval telemetry tied to the actual server payload only.
        retrieval_response_text = response_text.split("\n\nSEARCH_RESULT_OVERLAP_NOTICE\n", 1)[0]
        response_docids = _extract_doc_cursors(retrieval_response_text)
        requested_docids = _extract_requested_doc_cursors(tool_call)

        if tool_name in {"search", "visit", "open", "find"}:
            agent_data.document_flow.append(
                {
                    "tool": tool_name,
                    "docids": sorted(response_docids | requested_docids),
                }
            )

        if tool_name == "search":
            agent_data.search_calls += 1
            if response_docids:
                overlap = len(response_docids & agent_data.searched_docids) / len(response_docids)
                agent_data.max_search_result_overlap = max(agent_data.max_search_result_overlap, overlap)
                if overlap >= 0.75:
                    agent_data.search_result_overlap_events += 1
                agent_data.search_doc_refs += len(response_docids)
                agent_data.searched_docids.update(response_docids)
            return

        if tool_name == "open":
            agent_data.open_calls += 1
            agent_data.opened_docids.update(response_docids | requested_docids)
            return

        if tool_name == "visit":
            agent_data.visit_calls += 1
            agent_data.opened_docids.update(response_docids | requested_docids)
            return

        if tool_name == "find":
            agent_data.find_calls += 1
            agent_data.find_docids.update(response_docids | requested_docids)

    async def _append_answer_prefill(self, response_ids: list[int], agent_data: AgentData) -> list[int]:
        """Append a loss-free answer prefix for an optional answer-only finalization ablation."""

        prefix_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode("<answer>", add_special_tokens=False)
        )
        if not prefix_ids:
            return response_ids
        agent_data.final_answer_prefilled = True
        return [*response_ids, *prefix_ids]

    async def _append_finalization_messages(
        self,
        agent_data: AgentData,
        *,
        rejected_tool_calls: Optional[list[FunctionCall]],
        instruction: Optional[str],
        prefill_answer: bool,
        force_prefill: bool = False,
    ) -> bool:
        """Append balanced rejected calls and an optional final-answer instruction."""

        add_messages: list[dict[str, Any]] = []
        rejected_tool_calls = rejected_tool_calls or []
        for tool_call in rejected_tool_calls:
            add_messages.append(
                {
                    "role": "tool",
                    "content": (
                        f"FINALIZATION_TOOL_DISABLED: the requested {tool_call.name} call was not executed. "
                        "The search phase is closed."
                    ),
                }
            )
        if instruction is not None:
            add_messages.append({"role": "user", "content": instruction})
        if not add_messages:
            return True

        response_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
        if prefill_answer and (
            getattr(self, "final_answer_prefill", False) or force_prefill
        ):
            response_ids = await self._append_answer_prefill(response_ids, agent_data)
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            agent_data.final_answer_prefilled = False
            return False

        agent_data.messages.extend(add_messages)
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.finalization_rejected_tool_calls += len(rejected_tool_calls)
        if instruction is not None:
            agent_data.user_turns += 1
        return True

    @staticmethod
    async def _duplicate_tool_call_response(
        tool_call: FunctionCall,
        repeat_count: int = 1,
    ) -> tuple[ToolResponse, float, dict]:
        return (
            ToolResponse(
                text=(
                    "DUPLICATE_CALL_BLOCKED\n"
                    f"tool: {tool_call.name}\n"
                    f"repeat_count: {repeat_count}\n"
                    "This exact request already completed and is permanently blocked for this trajectory. "
                    "Do not retry it or cycle back to another previously issued request. Use the result already "
                    "present in the conversation, issue a request with genuinely new constraints, or return the "
                    "required final-answer format."
                )
            ),
            0.0,
            {"duplicate_tool_call": True},
        )

    def _begin_finalization(self, agent_data: AgentData, reason: str) -> None:
        agent_data.force_final_answer = True
        if agent_data.finalization_reason is None:
            agent_data.finalization_reason = reason

    @staticmethod
    def _record_generic_answer_rejection(
        agent_data: AgentData,
        *,
        status: str,
        phase: str,
    ) -> None:
        if not agent_data.generic_answer_rescue_active:
            agent_data.generic_answer_rescue_active = True
            agent_data.generic_answer_rescue_triggers += 1
        agent_data.generic_answer_rescue_rejections += 1
        if status == "generic":
            agent_data.generic_answer_rescue_generic_rejections += 1
        else:
            agent_data.generic_answer_rescue_missing_rejections += 1
        agent_data.generic_answer_rescue_events.append(
            {
                "kind": f"rejected_{status}",
                "phase": phase,
                "assistant_turn": agent_data.assistant_turns,
            }
        )

    @staticmethod
    def _record_generic_answer_success(agent_data: AgentData) -> None:
        if not agent_data.generic_answer_rescue_active:
            return
        agent_data.generic_answer_rescue_successes += 1
        agent_data.generic_answer_rescue_active = False
        agent_data.generic_answer_rescue_events.append(
            {
                "kind": "concrete_retry",
                "assistant_turn": agent_data.assistant_turns,
            }
        )

    def _final_answer_instruction(self, *, retry: bool, generic_rescue: bool = False) -> str:
        if generic_rescue:
            return (
                _GENERIC_ANSWER_ONLY_RESCUE_INSTRUCTION
                if (
                    getattr(self, "final_answer_prefill", False)
                    or getattr(self, "generic_answer_rescue_prefill", True)
                )
                else _GENERIC_ANSWER_RESCUE_INSTRUCTION
            )
        if getattr(self, "final_answer_prefill", False):
            return _FINAL_ANSWER_ONLY_RETRY_INSTRUCTION if retry else _FINAL_ANSWER_ONLY_INSTRUCTION
        return _FINAL_ANSWER_RETRY_INSTRUCTION if retry else _FINAL_ANSWER_INSTRUCTION

    async def _request_final_answer(
        self,
        agent_data: AgentData,
        reason: str,
        rejected_tool_calls: Optional[list[FunctionCall]] = None,
        retry: bool = False,
        generic_rescue: bool = False,
    ) -> AgentState:
        """Append a user-level synthesis request, balancing rejected final tool calls."""

        self._begin_finalization(agent_data, reason)
        instruction = self._final_answer_instruction(
            retry=retry,
            generic_rescue=generic_rescue,
        )
        appended = await self._append_finalization_messages(
            agent_data,
            rejected_tool_calls=rejected_tool_calls,
            instruction=instruction,
            prefill_answer=True,
            force_prefill=(
                generic_rescue
                and getattr(self, "generic_answer_rescue_prefill", True)
            ),
        )
        if not appended:
            agent_data.finalization_balance_failures += len(rejected_tool_calls or [])
            return AgentState.TERMINATED
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map
