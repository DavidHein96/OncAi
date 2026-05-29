"""
Function-calling client for multi-turn extraction.

Handles the conversation loop with tool calls and error recovery.
Supports both Azure OpenAI Responses API and vLLM.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from .models import FinishExtraction, NoteInfo, StopWorkflow
from .tools import ToolRegistry


class Timeline:
    """No-op timeline used by single-note extraction.

    The original multi-note implementation tracked per-patient ledger state
    here; single-note runs never use the ledger, but ``extract_note`` still
    accepts a ``timeline`` parameter so the call signatures stay uniform.
    """

    def __init__(self, mrn: str = "") -> None:
        self.mrn = mrn

    def format_for_prompt(self, registry: ToolRegistry | None = None) -> str:  # noqa: ARG002
        return ""


logger = logging.getLogger(__name__)

# Jitter for rate limiting
BASE_WAIT = 0.5
JITTER = 0.1


class FCBackend(StrEnum):
    """Supported backends for function-calling extraction."""

    AZURE_RESPONSES = "azure-responses"
    VLLM = "vllm"
    VLLM_CHAT = "vllm-chat"


def _jitter_wait(base: float = BASE_WAIT):
    """Wait with jitter to avoid rate limits."""
    time.sleep(max(0.0, base + random.uniform(-JITTER, JITTER)))


def _get_reasoning_tokens(usage: Any) -> int:
    """Extract reasoning tokens from a usage object, returning 0 if unavailable."""
    if usage is None:
        return 0
    # Responses API: usage.output_tokens_details.reasoning_tokens
    details = getattr(usage, "output_tokens_details", None)
    if details is not None:
        return getattr(details, "reasoning_tokens", 0) or 0
    # Chat Completions API: usage.completion_tokens_details.reasoning_tokens
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        return getattr(details, "reasoning_tokens", 0) or 0
    return 0


@dataclass
class FCClientConfig:
    """Configuration for function-calling client."""

    max_rounds_per_note: int = 12
    reasoning_effort: str = "medium"  # low, medium, high, or "none" to disable
    text_verbosity: str = "low"  # low, medium, high (controls text output verbosity)
    parallel_tool_calls: bool = True
    retry_on_validation_error: bool = True
    max_validation_retries: int = 3
    temperature: float = 0.0  # For vLLM
    top_p: float | None = None  # vLLM only (nucleus sampling)
    top_k: int | None = None  # vLLM only (sent via extra_body)


@dataclass
class NoteExtractionResult:
    """Result of extracting from a single note."""

    note_id: str
    success: bool
    finish: FinishExtraction | None = None
    stop_workflow: StopWorkflow | None = None
    events: list[tuple[str, BaseModel]] = field(default_factory=list)
    rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    error: str | None = None


class FunctionCallingClient:
    """
    Client for function-calling based extraction.

    Uses Azure OpenAI Responses API with tools for multi-turn extraction.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        api_version: str = "2025-03-01-preview",
        config: FCClientConfig | None = None,
        debug_dir: str | Path | None = None,
    ):
        """
        Initialize the function-calling client.

        Args:
            endpoint: Azure OpenAI endpoint (or AZURE_OPENAI_ENDPOINT env var)
            api_key: API key (or AZURE_OPENAI_API_KEY env var)
            deployment: Model deployment name (or AZURE_OPENAI_DEPLOYMENT env var)
            api_version: API version
            config: Client configuration
            debug_dir: If set, save full request/response JSON for each API call
        """
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e

        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.config = config or FCClientConfig()
        self._debug_dir: Path | None = Path(debug_dir) if debug_dir else None
        self._call_count = 0

        if self._debug_dir:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint required")
        if not self.api_key:
            raise ValueError("Azure OpenAI API key required")
        if not self.deployment:
            raise ValueError("Azure OpenAI deployment name required")

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=api_version,
            timeout=120,
            max_retries=3,
        )

    def _reasoning_kwargs(self) -> dict[str, Any]:
        """Return the reasoning kwarg dict, or empty if reasoning is disabled."""
        if (
            self.config.reasoning_effort
            and self.config.reasoning_effort.lower() != "none"
        ):
            return {"reasoning": {"effort": self.config.reasoning_effort}}
        return {}

    def _text_kwargs(self) -> dict[str, Any]:
        """Return the text config kwarg dict, or empty if not set."""
        if self.config.text_verbosity and self.config.text_verbosity.lower() != "none":
            return {
                "text": {
                    "format": {"type": "text"},
                    "verbosity": self.config.text_verbosity,
                }
            }
        return {}

    def _save_debug(
        self,
        note_id: str,
        round_idx: int,
        request_data: dict[str, Any],
        response_obj: Any,
        fn_calls: list[dict] | None = None,
        tool_results: list[Any] | None = None,
    ) -> None:
        """
        Save full request/response JSON for a single API call.

        Args:
            note_id: Note being processed
            round_idx: Conversation round (0 = initial call)
            request_data: Dict of request parameters (input, tools, etc.)
            response_obj: Raw response object from the API
            fn_calls: Parsed function calls from the response
            tool_results: Tool execution results for each call
        """
        if not self._debug_dir:
            return

        self._call_count += 1
        debug_file = (
            self._debug_dir / f"call_{self._call_count:04d}_{note_id}_r{round_idx}.json"
        )

        # Serialize response
        response_data: dict[str, Any] = {}
        try:
            response_data["model_dump"] = response_obj.model_dump()
        except Exception:
            response_data["repr"] = repr(response_obj)

        # Extract output items — best-effort; missing/odd shapes are normal in
        # debug captures and shouldn't break the write.
        with suppress(Exception):
            output_items = []
            for item in getattr(response_obj, "output", []) or []:
                if hasattr(item, "model_dump"):
                    output_items.append(item.model_dump())
                else:
                    output_items.append(str(item))
            response_data["output_items"] = output_items

        # Usage — best-effort, same rationale.
        with suppress(Exception):
            response_data["usage"] = {
                "input_tokens": getattr(response_obj.usage, "input_tokens", None),
                "output_tokens": getattr(response_obj.usage, "output_tokens", None),
            }

        debug_record = {
            "call_number": self._call_count,
            "note_id": note_id,
            "round": round_idx,
            "backend": getattr(self, "backend", "azure-responses"),
            "model": getattr(self, "deployment", getattr(self, "model", "unknown")),
            "request": request_data,
            "response": response_data,
            "parsed_function_calls": fn_calls,
            "tool_results": tool_results,
        }

        try:
            with debug_file.open("w") as f:
                json.dump(debug_record, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save debug file {debug_file}: {e}")

    def _make_api_call(
        self,
        instructions: str,
        input_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        prev_response_id: str | None = None,
    ) -> Any:
        """Make a single API call to the Azure Responses API."""
        kwargs: dict[str, Any] = dict(
            model=self.deployment,
            instructions=instructions,
            input=input_messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=self.config.parallel_tool_calls,
            **self._reasoning_kwargs(),
            **self._text_kwargs(),
        )
        if prev_response_id:
            kwargs["previous_response_id"] = prev_response_id
        return self.client.responses.create(**kwargs)

    def extract_note(
        self,
        note_text: str,
        note_info: NoteInfo,
        timeline: Timeline,
        registry: ToolRegistry,
        system_prompt: str,
        _user_prompt: str | None = None,
        include_timeline: bool = True,
    ) -> NoteExtractionResult:
        """
        Extract events from a single note using multi-turn function calling.

        Args:
            note_text: The clinical note text
            note_info: Metadata about the note
            timeline: Current timeline state (for context)
            registry: Tool registry with available tools
            system_prompt: System instructions for extraction
            _user_prompt: Pre-built user prompt to use instead of running
                ``_build_note_prompt``. Used by ``extract_single_note`` to
                substitute a single-note prompt that omits multi-note framing.
            include_timeline: When False, omit the ledger context from the
                prompt (stateless multi-note mode).

        Returns:
            NoteExtractionResult with extracted events
        """
        result = NoteExtractionResult(note_id=note_info.note_id, success=False)

        # Build initial prompt (or use override from extract_single_note)
        user_prompt = _user_prompt or self._build_note_prompt(
            note_text, note_info, timeline, registry, include_timeline=include_timeline
        )
        input_messages = [{"role": "user", "content": user_prompt}]
        tools = registry.to_openai_tools()

        _jitter_wait()
        try:
            response = self._make_api_call(system_prompt, input_messages, tools)
        except Exception as e:
            logger.exception("Initial API call failed for note %s", note_info.note_id)
            result.error = str(e)
            return result

        result.input_tokens += (
            getattr(response.usage, "input_tokens", 0) if response.usage else 0
        )
        result.output_tokens += (
            getattr(response.usage, "output_tokens", 0) if response.usage else 0
        )
        result.reasoning_tokens += _get_reasoning_tokens(response.usage)

        self._save_debug(
            note_id=note_info.note_id,
            round_idx=0,
            request_data={
                "input": input_messages,
                "tools": [t["name"] for t in tools],
                "reasoning_effort": self.config.reasoning_effort,
                "temperature": self.config.temperature,
            },
            response_obj=response,
        )

        # Multi-turn loop
        prev_response = response
        validation_retries = 0

        for round_idx in range(1, self.config.max_rounds_per_note + 1):
            result.rounds = round_idx

            # Refresh tools each round (enables gated registries to swap tools)
            tools = registry.to_openai_tools()

            fn_calls, _ = self._extract_calls_and_text(prev_response)

            if not fn_calls:
                logger.warning(
                    f"No tool calls in round {round_idx} for note {note_info.note_id}"
                )
                tool_outputs: list[dict[str, Any]] = []
                extra_messages: list[dict[str, Any]] = [
                    {
                        "role": "user",
                        "content": (
                            "You did not call any tools. Please call a tool to record findings, "
                            "or call finish_note_extraction if there is nothing relevant in this note."
                        ),
                    }
                ]
            else:
                tool_outputs = []
                had_errors = False
                recorded_tools = []

                for call in fn_calls:
                    tool_name = call["name"]
                    call_id = call.get("call_id")
                    args = call.get("arguments", {})

                    if "note_id" not in args:
                        args["note_id"] = note_info.note_id

                    tool_result = registry.execute(tool_name, args)

                    if tool_result.success and tool_result.obj is not None:
                        result.events.append((tool_name, tool_result.obj))
                        recorded_tools.append(tool_name)

                        if tool_name == "finish_note_extraction":
                            result.finish = cast(FinishExtraction, tool_result.obj)
                            result.success = True
                            return result
                        elif tool_name == "stop_workflow":
                            result.stop_workflow = cast(StopWorkflow, tool_result.obj)
                            result.success = True
                            return result
                    else:
                        had_errors = True
                        logger.warning(f"Tool {tool_name} failed: {tool_result.error}")

                    if call_id:
                        tool_outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": str(call_id),
                                "output": json.dumps(tool_result.status),
                            }
                        )
                    else:
                        logger.warning(
                            "Missing call_id for %s in note %s round %d — "
                            "tool output will not be sent back to model",
                            tool_name,
                            note_info.note_id,
                            round_idx,
                        )

                if had_errors and self.config.retry_on_validation_error:
                    validation_retries += 1
                    if validation_retries > self.config.max_validation_retries:
                        logger.error(
                            f"Max validation retries exceeded for note {note_info.note_id}"
                        )
                        result.error = "Max validation retries exceeded"
                        return result

                    error_summary = self._summarize_errors(fn_calls, registry)
                    logger.info(
                        "validation retry %d/%d for note %s:\n%s",
                        validation_retries,
                        self.config.max_validation_retries,
                        note_info.note_id,
                        error_summary,
                    )
                    extra_messages = [
                        {
                            "role": "user",
                            "content": (
                                f"Some tool calls had validation errors:\n{error_summary}\n"
                                "Please fix and retry, or call finish_note_extraction."
                            ),
                        }
                    ]
                else:
                    recorded_summary = (
                        ", ".join(recorded_tools) if recorded_tools else "none"
                    )
                    extra_messages = [
                        {
                            "role": "user",
                            "content": (
                                f"Tools recorded this round: {recorded_summary}. "
                                "Do NOT call these again for the same findings. "
                                "Extract any remaining new findings, or call finish_note_extraction if done with this note."
                            ),
                        }
                    ]

                logger.debug(
                    "Round %d for note %s: %d tool_outputs, %d fn_calls processed",
                    round_idx,
                    note_info.note_id,
                    len(tool_outputs),
                    len(fn_calls),
                )

            # Use previous_response_id for stateful chaining if available,
            # otherwise fall back to including the full response history in input.
            _jitter_wait()
            try:
                prev_response_id = getattr(prev_response, "id", None) or None
                if prev_response_id:
                    next_input = tool_outputs + extra_messages
                else:
                    carry = self._to_input_from_prev_output(prev_response)
                    next_input = carry + tool_outputs + extra_messages

                prev_response = self._make_api_call(
                    system_prompt, next_input, tools, prev_response_id
                )

                result.input_tokens += (
                    getattr(prev_response.usage, "input_tokens", 0)
                    if prev_response.usage
                    else 0
                )
                result.output_tokens += (
                    getattr(prev_response.usage, "output_tokens", 0)
                    if prev_response.usage
                    else 0
                )
                result.reasoning_tokens += _get_reasoning_tokens(prev_response.usage)

                new_fn_calls, _ = self._extract_calls_and_text(prev_response)
                self._save_debug(
                    note_id=note_info.note_id,
                    round_idx=round_idx,
                    request_data={
                        "tool_outputs": tool_outputs,
                        "extra_messages": extra_messages,
                        "processed_calls_from_prev_round": [
                            c["name"] for c in fn_calls
                        ],
                    },
                    response_obj=prev_response,
                    fn_calls=new_fn_calls,
                    tool_results=[t.get("output") for t in tool_outputs]
                    if tool_outputs
                    else None,
                )

            except Exception as e:
                logger.exception(
                    "API call failed in round %s for note %s",
                    round_idx,
                    note_info.note_id,
                )
                result.error = str(e)
                return result

        logger.warning(f"Max rounds reached for note {note_info.note_id}")
        result.error = "Max rounds reached without finish_note_extraction"
        return result

    def _build_single_note_prompt(
        self,
        note_text: str,
        note_id: str,
        registry: ToolRegistry | None = None,  # noqa: ARG002
    ) -> str:
        """Build the prompt for a one-shot single-note extraction.

        Differs from ``_build_note_prompt`` (multi-note) in that there is no
        timeline context, no progress framing, no "final note" marker, and no
        mention of ``stop_workflow`` — none of those concepts apply when each
        note is processed independently.
        """
        parts = [
            "## Current Note to Review",
            f"**Note ID:** {note_id}",
            "",
            "### Note Text",
            "```",
            note_text,
            "```",
            "",
            "## Instructions",
            "1. Review the note and extract relevant events using the available tools",
            "2. Call tools for each distinct event/finding (can call multiple)",
            "3. When done with this note, call `finish_note_extraction`",
        ]
        return "\n".join(parts)

    def extract_single_note(
        self,
        note_text: str,
        system_prompt: str,
        registry: ToolRegistry,
        note_id: str = "",
    ) -> NoteExtractionResult:
        """One-shot single-note extraction (no patient timeline context).

        Uses ``_build_single_note_prompt`` for the user prompt and threads it
        through the standard multi-turn ``extract_note`` loop. Constructs a
        synthetic ``NoteInfo``/``Timeline`` only because the loop's debug and
        logging paths need a ``note_id`` and a no-op timeline object.
        """
        user_prompt = self._build_single_note_prompt(note_text, note_id, registry)
        note_info = NoteInfo(
            note_id=note_id,
            note_date="unknown",
            note_index=0,
            total_notes=1,
            is_final_note=False,
        )
        timeline = Timeline(mrn="")
        return self.extract_note(
            note_text=note_text,
            note_info=note_info,
            timeline=timeline,
            registry=registry,
            system_prompt=system_prompt,
            _user_prompt=user_prompt,
        )

    def _build_note_prompt(
        self,
        note_text: str,
        note_info: NoteInfo,
        timeline: Timeline,
        registry: ToolRegistry | None = None,
        include_timeline: bool = True,
    ) -> str:
        """Build the prompt for a single note.

        When ``include_timeline`` is False, the ledger section is omitted so the
        model sees each note independently (stateless multi-note mode).
        """
        parts = []

        if include_timeline:
            # Timeline context - pass registry for field filtering
            parts.append(timeline.format_for_prompt(registry=registry))
            parts.append("")

        # Note metadata
        parts.append("## Current Note to Review")
        parts.append(f"**Note ID:** {note_info.note_id}")
        parts.append(f"**Note Date:** {note_info.note_date}")
        if note_info.note_type:
            parts.append(f"**Note Type:** {note_info.note_type}")
        if note_info.department:
            parts.append(f"**Department:** {note_info.department}")
        parts.append(
            f"**Progress:** Note {note_info.note_index + 1} of {note_info.total_notes}"
        )

        if note_info.is_final_note:
            parts.append("")
            parts.append("**THIS IS THE FINAL NOTE IN THE PATIENT'S RECORD**")

        parts.append("")
        parts.append("### Note Text")
        parts.append("```")
        parts.append(note_text)
        parts.append("```")
        parts.append("")

        # Instructions
        parts.append("## Instructions")
        parts.append(
            "1. Review the note and extract relevant events using the available tools"
        )
        parts.append(
            "2. Call tools for each distinct event/finding (can call multiple)"
        )
        parts.append("3. When done with this note, call `finish_note_extraction`")
        parts.append(
            "4. If you have found all needed information, call `stop_workflow`"
        )

        return "\n".join(parts)

    def _extract_calls_and_text(self, response) -> tuple[list[dict], list[str]]:
        """Extract function calls and text from a response."""
        fn_calls, texts = [], []

        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)

            if item_type == "text" and getattr(item, "text", None):
                texts.append(item.text)

            elif item_type == "function_call":
                call_id = (
                    getattr(item, "call_id", None)
                    or getattr(item, "id", None)
                    or getattr(item, "tool_call_id", None)
                )
                try:
                    args = json.loads(getattr(item, "arguments", "{}") or "{}")
                except json.JSONDecodeError:
                    args = {}

                fn_calls.append(
                    {
                        "name": item.name,
                        "call_id": call_id,
                        "arguments": args,
                    }
                )

            elif item_type == "message" and getattr(item, "content", None):
                for part in item.content:
                    if getattr(part, "type", None) == "text" and getattr(
                        part, "text", None
                    ):
                        texts.append(part.text)
                    elif getattr(part, "type", None) == "tool_call":
                        call_id = (
                            getattr(part, "call_id", None)
                            or getattr(part, "id", None)
                            or getattr(part, "tool_call_id", None)
                        )
                        try:
                            args = json.loads(getattr(part, "arguments", "{}") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        fn_calls.append(
                            {
                                "name": part.name,
                                "call_id": call_id,
                                "arguments": args,
                            }
                        )

        return fn_calls, texts

    def _to_input_from_prev_output(self, prev_response) -> list[dict]:
        """Convert previous response output to input format for manual chaining."""
        items = []
        id_keys = {"id", "message_id", "item_id", "output_id", "refusal_id"}

        def strip_ids(x):
            if isinstance(x, dict):
                for k in list(x.keys()):
                    if k in id_keys:
                        x.pop(k, None)
                for v in x.values():
                    strip_ids(v)
            elif isinstance(x, list):
                for v in x:
                    strip_ids(v)

        for part in getattr(prev_response, "output", []) or []:
            if hasattr(part, "model_dump"):
                d = part.model_dump()
            else:
                d = json.loads(
                    json.dumps(part, default=lambda o: getattr(o, "__dict__", str(o)))
                )
            strip_ids(d)
            items.append(d)

        return items

    def _summarize_errors(self, fn_calls: list[dict], registry: ToolRegistry) -> str:
        """Summarize validation errors for the model."""
        lines = []
        for call in fn_calls:
            result = registry.execute(call["name"], call.get("arguments", {}))
            if not result.success and result.error:
                fields = result.error.get("fields", [])
                for e in fields[:5]:
                    path = ".".join(str(p) for p in e.get("loc", []))
                    msg = e.get("msg", e.get("type", "validation_error"))
                    lines.append(f"- {call['name']}.{path}: {msg}")
        return "\n".join(lines) if lines else "Unknown validation errors"


class VLLMFunctionCallingClient(FunctionCallingClient):
    """
    vLLM function-calling client using the Responses API.

    vLLM supports the OpenAI Responses API format for tool calling.
    Requires vLLM server started with --enable-auto-tool-choice --tool-call-parser <parser>
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str = "not-needed",
        config: FCClientConfig | None = None,
        debug_dir: str | Path | None = None,
    ):
        """
        Initialize vLLM function-calling client.

        Args:
            base_url: vLLM server URL (e.g., http://localhost:8000/v1)
            model: Model name
            api_key: API key (vLLM usually doesn't require one)
            config: Client configuration
            debug_dir: If set, save full request/response JSON for each API call
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e

        self.base_url = base_url or os.getenv(
            "VLLM_BASE_URL", "http://localhost:8000/v1"
        )
        self.model = model or os.getenv("VLLM_MODEL")
        self.config = config or FCClientConfig()
        self._debug_dir: Path | None = Path(debug_dir) if debug_dir else None
        self._call_count = 0

        if self._debug_dir:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

        if not self.model:
            raise ValueError(
                "vLLM model name required (set VLLM_MODEL env var or pass model=)"
            )

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=120,
            max_retries=3,
        )
        self.deployment = self.model  # For compatibility with base class

    def _apply_sampling(self, kw: dict[str, Any]) -> None:
        """Inject top_p / top_k into a vLLM request kwargs dict if configured.

        top_p is a standard OpenAI param. top_k is vLLM-specific and must be
        sent via extra_body so the OpenAI SDK passes it through unchanged.
        """
        if self.config.top_p is not None:
            kw["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            extra_body = kw.setdefault("extra_body", {})
            extra_body["top_k"] = self.config.top_k

    def extract_note(
        self,
        note_text: str,
        note_info: NoteInfo,
        timeline: Timeline,
        registry: ToolRegistry,
        system_prompt: str,
        _user_prompt: str | None = None,
        include_timeline: bool = True,
    ) -> NoteExtractionResult:
        """
        Extract events from a single note using function calling.

        Same as Azure client but without reasoning parameter (vLLM may not support it).
        """
        result = NoteExtractionResult(note_id=note_info.note_id, success=False)

        # Build initial prompt (or use override from extract_single_note)
        user_prompt = _user_prompt or self._build_note_prompt(
            note_text, note_info, timeline, registry, include_timeline=include_timeline
        )

        # Initial request
        input_messages = [{"role": "user", "content": user_prompt}]
        tools = registry.to_openai_tools(for_responses_api=True)

        _jitter_wait()
        try:
            # vLLM Responses API - no reasoning parameter
            _kw: dict[str, Any] = dict(
                model=self.model,
                instructions=system_prompt,
                input=input_messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=self.config.parallel_tool_calls,
                temperature=self.config.temperature,
            )
            self._apply_sampling(_kw)
            response = self.client.responses.create(**_kw)
        except Exception as e:
            logger.exception("Initial API call failed for note %s", note_info.note_id)
            result.error = str(e)
            return result

        result.input_tokens += (
            getattr(response.usage, "input_tokens", 0) if response.usage else 0
        )
        result.output_tokens += (
            getattr(response.usage, "output_tokens", 0) if response.usage else 0
        )
        result.reasoning_tokens += _get_reasoning_tokens(response.usage)

        # Debug: save initial API call
        self._save_debug(
            note_id=note_info.note_id,
            round_idx=0,
            request_data={
                "input": input_messages,
                "tools": [t["name"] for t in tools],
                "temperature": self.config.temperature,
            },
            response_obj=response,
        )

        # Multi-turn loop
        prev_response = response
        validation_retries = 0

        for round_idx in range(1, self.config.max_rounds_per_note + 1):
            result.rounds = round_idx

            # Refresh tools each round (enables gated registries to swap tools)
            tools = registry.to_openai_tools(for_responses_api=True)

            # Extract function calls from response
            fn_calls, _texts = self._extract_calls_and_text(prev_response)

            if not fn_calls:
                # No tool calls - nudge the model
                logger.warning(
                    f"No tool calls in round {round_idx} for note {note_info.note_id}"
                )
                tool_outputs = []
                extra_messages = [
                    {
                        "role": "user",
                        "content": (
                            "You did not call any tools. Please call a tool to record findings, "
                            "or call finish_note_extraction if there is nothing relevant in this note."
                        ),
                    }
                ]
            else:
                # Process each tool call
                tool_outputs = []
                had_errors = False
                recorded_tools = []

                for call in fn_calls:
                    tool_name = call["name"]
                    call_id = call.get("call_id")
                    args = call.get("arguments", {})

                    # Add note_id if not present
                    if "note_id" not in args:
                        args["note_id"] = note_info.note_id

                    # Execute tool
                    tool_result = registry.execute(tool_name, args)

                    if tool_result.success and tool_result.obj is not None:
                        # Record event
                        result.events.append((tool_name, tool_result.obj))
                        recorded_tools.append(tool_name)

                        # Check for terminal tools
                        if tool_name == "finish_note_extraction":
                            result.finish = cast(FinishExtraction, tool_result.obj)
                            result.success = True
                            return result
                        elif tool_name == "stop_workflow":
                            result.stop_workflow = cast(StopWorkflow, tool_result.obj)
                            result.success = True
                            return result
                    else:
                        had_errors = True
                        logger.warning(f"Tool {tool_name} failed: {tool_result.error}")

                    # Build tool output for next round
                    if call_id:
                        tool_outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": str(call_id),
                                "output": json.dumps(tool_result.status),
                            }
                        )

                # Build extra messages based on errors
                if had_errors and self.config.retry_on_validation_error:
                    validation_retries += 1
                    if validation_retries > self.config.max_validation_retries:
                        logger.error(
                            f"Max validation retries exceeded for note {note_info.note_id}"
                        )
                        result.error = "Max validation retries exceeded"
                        return result

                    error_summary = self._summarize_errors(fn_calls, registry)
                    logger.info(
                        "validation retry %d/%d for note %s:\n%s",
                        validation_retries,
                        self.config.max_validation_retries,
                        note_info.note_id,
                        error_summary,
                    )
                    extra_messages = [
                        {
                            "role": "user",
                            "content": (
                                f"Some tool calls had validation errors:\n{error_summary}\n"
                                "Please fix and retry, or call finish_note_extraction."
                            ),
                        }
                    ]
                else:
                    recorded_summary = (
                        ", ".join(recorded_tools) if recorded_tools else "none"
                    )
                    extra_messages = [
                        {
                            "role": "user",
                            "content": (
                                f"Tools recorded this round: {recorded_summary}. "
                                "Do NOT call these again for the same findings. "
                                "Extract any remaining new findings, or call finish_note_extraction if done with this note."
                            ),
                        }
                    ]

            # Make next API call
            _jitter_wait()
            try:
                # Try to use previous_response_id for efficiency
                if hasattr(prev_response, "id") and prev_response.id:
                    _kw = dict(
                        model=self.model,
                        instructions=system_prompt,
                        previous_response_id=prev_response.id,
                        input=tool_outputs + extra_messages,
                        tools=tools,
                        tool_choice="auto",
                        parallel_tool_calls=self.config.parallel_tool_calls,
                        temperature=self.config.temperature,
                    )
                    self._apply_sampling(_kw)
                    prev_response = self.client.responses.create(**_kw)
                else:
                    # Fall back to manual chaining
                    carry = self._to_input_from_prev_output(prev_response)
                    _kw = dict(
                        model=self.model,
                        instructions=system_prompt,
                        input=carry + tool_outputs + extra_messages,
                        tools=tools,
                        tool_choice="auto",
                        parallel_tool_calls=self.config.parallel_tool_calls,
                        temperature=self.config.temperature,
                    )
                    self._apply_sampling(_kw)
                    prev_response = self.client.responses.create(**_kw)

                result.input_tokens += (
                    getattr(prev_response.usage, "input_tokens", 0)
                    if prev_response.usage
                    else 0
                )
                result.output_tokens += (
                    getattr(prev_response.usage, "output_tokens", 0)
                    if prev_response.usage
                    else 0
                )
                result.reasoning_tokens += _get_reasoning_tokens(prev_response.usage)

                # Debug: save round API call with tool results
                self._save_debug(
                    note_id=note_info.note_id,
                    round_idx=round_idx,
                    request_data={
                        "tool_outputs": tool_outputs,
                        "extra_messages": extra_messages,
                    },
                    response_obj=prev_response,
                    fn_calls=fn_calls,
                    tool_results=[t.get("output") for t in tool_outputs]
                    if tool_outputs
                    else None,
                )

            except Exception as e:
                logger.exception(
                    "API call failed in round %s for note %s",
                    round_idx,
                    note_info.note_id,
                )
                result.error = str(e)
                return result

        # Max rounds reached without finish
        logger.warning(f"Max rounds reached for note {note_info.note_id}")
        result.error = "Max rounds reached without finish_note_extraction"
        return result


class VLLMChatFunctionCallingClient(VLLMFunctionCallingClient):
    """
    vLLM function-calling client using the Chat Completions API.

    Use this when /v1/responses on the vLLM server is unreliable.
    Requires vLLM started with --enable-auto-tool-choice --tool-call-parser <parser>.
    """

    def extract_note(
        self,
        note_text: str,
        note_info: NoteInfo,
        timeline: Timeline,
        registry: ToolRegistry,
        system_prompt: str,
        _user_prompt: str | None = None,
        include_timeline: bool = True,
    ) -> NoteExtractionResult:
        result = NoteExtractionResult(note_id=note_info.note_id, success=False)

        user_prompt = _user_prompt or self._build_note_prompt(
            note_text, note_info, timeline, registry, include_timeline=include_timeline
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        validation_retries = 0

        for round_idx in range(self.config.max_rounds_per_note + 1):
            result.rounds = round_idx + 1

            # Refresh tools each round (enables gated registries to swap tools)
            tools = registry.to_openai_tools(for_responses_api=False)

            _jitter_wait()
            try:
                _kw: dict[str, Any] = dict(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=self.config.parallel_tool_calls,
                    temperature=self.config.temperature,
                )
                self._apply_sampling(_kw)
                response = self.client.chat.completions.create(**_kw)
            except Exception as e:
                logger.exception(
                    "API call failed in round %s for note %s",
                    round_idx,
                    note_info.note_id,
                )
                result.error = str(e)
                return result

            # Chat Completions usage naming differs from Responses API
            if response.usage:
                result.input_tokens += getattr(response.usage, "prompt_tokens", 0) or 0
                result.output_tokens += (
                    getattr(response.usage, "completion_tokens", 0) or 0
                )
                result.reasoning_tokens += _get_reasoning_tokens(response.usage)

            choice = response.choices[0]
            msg = choice.message
            tool_calls = getattr(msg, "tool_calls", None) or []

            # Parse tool calls into the same internal shape used elsewhere
            fn_calls: list[dict[str, Any]] = []
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                fn_calls.append(
                    {
                        "name": tc.function.name,
                        "call_id": tc.id,
                        "arguments": args,
                    }
                )

            self._save_debug(
                note_id=note_info.note_id,
                round_idx=round_idx,
                request_data={
                    "messages_len": len(messages),
                    "tools": [t["function"]["name"] for t in tools],
                    "temperature": self.config.temperature,
                },
                response_obj=response,
                fn_calls=fn_calls,
            )

            # Append the assistant message to history (must precede tool results)
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            if not fn_calls:
                logger.warning(
                    f"No tool calls in round {round_idx} for note {note_info.note_id}"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You did not call any tools. Please call a tool to record findings, "
                            "or call finish_note_extraction if there is nothing relevant in this note."
                        ),
                    }
                )
                continue

            had_errors = False
            recorded_tools: list[str] = []

            for call in fn_calls:
                tool_name = call["name"]
                call_id = call["call_id"]
                args = call["arguments"]

                if "note_id" not in args:
                    args["note_id"] = note_info.note_id

                tool_result = registry.execute(tool_name, args)

                if tool_result.success and tool_result.obj is not None:
                    result.events.append((tool_name, tool_result.obj))
                    recorded_tools.append(tool_name)

                    if tool_name == "finish_note_extraction":
                        result.finish = cast(FinishExtraction, tool_result.obj)
                        result.success = True
                        return result
                    elif tool_name == "stop_workflow":
                        result.stop_workflow = cast(StopWorkflow, tool_result.obj)
                        result.success = True
                        return result
                else:
                    had_errors = True
                    logger.warning(f"Tool {tool_name} failed: {tool_result.error}")

                # Tool result message must reference the call_id
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call_id),
                        "content": json.dumps(tool_result.status),
                    }
                )

            if had_errors and self.config.retry_on_validation_error:
                validation_retries += 1
                if validation_retries > self.config.max_validation_retries:
                    logger.error(
                        f"Max validation retries exceeded for note {note_info.note_id}"
                    )
                    result.error = "Max validation retries exceeded"
                    return result

                error_summary = self._summarize_errors(fn_calls, registry)
                logger.info(
                    "validation retry %d/%d for note %s:\n%s",
                    validation_retries,
                    self.config.max_validation_retries,
                    note_info.note_id,
                    error_summary,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Some tool calls had validation errors:\n{error_summary}\n"
                            "Please fix and retry, or call finish_note_extraction."
                        ),
                    }
                )
            else:
                recorded_summary = (
                    ", ".join(recorded_tools) if recorded_tools else "none"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tools recorded this round: {recorded_summary}. "
                            "Do NOT call these again for the same findings. "
                            "Extract any remaining new findings, or call finish_note_extraction if done with this note."
                        ),
                    }
                )

        logger.warning(f"Max rounds reached for note {note_info.note_id}")
        result.error = "Max rounds reached without finish_note_extraction"
        return result


def get_fc_client(
    backend: str | FCBackend = FCBackend.AZURE_RESPONSES,
    config: FCClientConfig | None = None,
    **kwargs,
) -> FunctionCallingClient:
    """
    Factory function to get a function-calling client.

    Args:
        backend: Backend type: "azure-responses", "vllm", or "vllm-chat"
        config: Client configuration
        **kwargs: Backend-specific arguments

    Returns:
        Configured FunctionCallingClient
    """
    if isinstance(backend, str):
        backend = FCBackend(backend)

    if backend == FCBackend.AZURE_RESPONSES:
        return FunctionCallingClient(config=config, **kwargs)
    elif backend == FCBackend.VLLM:
        return VLLMFunctionCallingClient(config=config, **kwargs)
    elif backend == FCBackend.VLLM_CHAT:
        return VLLMChatFunctionCallingClient(config=config, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
