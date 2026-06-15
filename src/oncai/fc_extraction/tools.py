"""
Tool registry and definition helpers for function-calling extraction.

Tools are defined as Pydantic models and registered with handlers.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from .models import (
    ExtractionPlan,
    FinishExtraction,
    FinishSingleExtraction,
    StopWorkflow,
)

logger = logging.getLogger(__name__)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline all $defs/$ref references in a JSON schema.

    Pydantic emits enum types as $ref pointers into a $defs block. Many local
    LLMs (and most vLLM tool-call parsers) don't dereference $ref, so the
    enum values become invisible to the model. Inlining yields a self-contained
    schema where enum values appear directly on the field.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {}) or {}

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                ref = node["$ref"]
                if ref.startswith("#/$defs/"):
                    name = ref.rsplit("/", 1)[-1]
                    target = defs.get(name)
                    if target is not None:
                        return resolve(copy.deepcopy(target))
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


@dataclass
class ToolDefinition:
    """A tool definition with its Pydantic model and handler."""

    name: str
    description: str
    model: type[BaseModel]
    handler: Callable[[dict], tuple[dict, BaseModel | None]] | None = None
    is_terminal: bool = False  # If True, this tool ends the note extraction
    event_identity_fields: tuple[str, ...] = field(default_factory=tuple)
    # Fields that identify the same real-world event across separate runs.
    # Used by review/disagreement tooling to align repeated calls of the same
    # function before comparing their non-identity fields.
    comparison_fields: tuple[str, ...] = field(default_factory=tuple)
    # Fields that should be compared after events are aligned. Text fields are
    # intentionally not special-cased; compare only fields explicitly listed
    # here plus event_identity_fields.


@dataclass
class ToolResult:
    """Result of executing a tool."""

    success: bool
    status: dict[str, Any]
    obj: BaseModel | None = None
    error: dict[str, Any] | None = None


class ToolRegistry:
    """
    Registry of tools for a specific extraction task.

    Tools are Pydantic models that define the schema for function calling.
    Handlers process the extracted data and return status.
    """

    def __init__(self, single_note: bool = False):
        self._tools: dict[str, ToolDefinition] = {}
        self._single_note = single_note
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register built-in tools that are always available."""
        if self._single_note:
            self.register(
                name="finish_note_extraction",
                description=(
                    "Call this as the FINAL step when all relevant information has been "
                    "extracted from this note. "
                    "This MUST be called for every note."
                ),
                model=FinishSingleExtraction,
                is_terminal=True,
            )
        else:
            self.register(
                name="finish_note_extraction",
                description=(
                    "Call this as the FINAL step when all relevant information has been "
                    "extracted from the current note. This signals you are ready for the next note. "
                    "This MUST be called for every note."
                ),
                model=FinishExtraction,
                is_terminal=True,
            )
            self.register(
                name="stop_workflow",
                description=(
                    "Call this to stop processing additional notes. Use when you are confident "
                    "that all required information has been found and no more notes are needed. "
                    "This will end the extraction for this patient."
                ),
                model=StopWorkflow,
                is_terminal=True,
            )

    def register(
        self,
        name: str,
        description: str,
        model: type[BaseModel],
        handler: Callable[[dict], tuple[dict, BaseModel | None]] | None = None,
        is_terminal: bool = False,
        event_identity_fields: Iterable[str] = (),
        comparison_fields: Iterable[str] = (),
    ) -> None:
        """
        Register a tool.

        Args:
            name: Tool name (used in function calling)
            description: Description shown to the LLM
            model: Pydantic model defining the tool's parameters
            handler: Optional handler function; if None, default validation handler is used
            is_terminal: If True, calling this tool ends extraction for current note
            event_identity_fields: Fields that identify the same real-world
                event across separate runs when this tool can be called more
                than once for a note. For example, IHC results are identified by
                specimen_id + standardized_test_name.
            comparison_fields: Fields to compare after events are aligned.
                Omit noisy/context fields like comment or free-text rationale.
        """
        identity_fields = tuple(str(field_name) for field_name in event_identity_fields)
        fields_to_compare = tuple(str(field_name) for field_name in comparison_fields)
        model_fields = set(model.model_fields)
        unknown_fields = [
            field_name
            for field_name in (*identity_fields, *fields_to_compare)
            if field_name not in model_fields
        ]
        if unknown_fields:
            missing = ", ".join(unknown_fields)
            raise ValueError(
                f"Tool {name!r} comparison metadata fields not found on "
                f"{model.__name__}: {missing}"
            )

        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            model=model,
            handler=handler,
            is_terminal=is_terminal,
            event_identity_fields=identity_fields,
            comparison_fields=fields_to_compare,
        )

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def to_openai_tools(self, for_responses_api: bool = True) -> list[dict[str, Any]]:
        """
        Convert all tools to OpenAI function calling format.

        Args:
            for_responses_api: If True, use Responses API format (flat structure).
                              If False, use Chat Completions format (nested function).

        Returns:
            List of tool definitions in OpenAI format
        """
        tools = []
        for tool_def in self._tools.values():
            params = _inline_refs(tool_def.model.model_json_schema())
            if for_responses_api:
                # Responses API format: flat structure with name at top level
                tools.append(
                    {
                        "type": "function",
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameters": params,
                    }
                )
            else:
                # Chat Completions API format: nested under "function"
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_def.name,
                            "description": tool_def.description,
                            "parameters": params,
                        },
                    }
                )
        return tools

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """
        Execute a tool with the given arguments.

        Args:
            name: Tool name
            arguments: Arguments from the LLM

        Returns:
            ToolResult with success status and parsed object
        """
        tool_def = self._tools.get(name)
        if tool_def is None:
            return ToolResult(
                success=False,
                status={"ok": False},
                error={"type": "UnknownTool", "message": f"Unknown tool: {name}"},
            )

        # Validate with Pydantic
        try:
            obj = tool_def.model.model_validate(arguments)
        except ValidationError as ve:
            return ToolResult(
                success=False,
                status={"ok": False},
                error={
                    "type": "ValidationError",
                    "message": "Pydantic validation failed",
                    "fields": ve.errors(),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                status={"ok": False},
                error={"type": "ExecutionError", "message": str(e)},
            )

        # Run custom handler if provided
        if tool_def.handler is not None:
            try:
                status, obj = tool_def.handler(arguments)
                return ToolResult(
                    success=status.get("ok", False),
                    status=status,
                    obj=obj,
                    error=status.get("error") if not status.get("ok") else None,
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    status={"ok": False},
                    error={"type": "HandlerError", "message": str(e)},
                )

        # Default: return the validated object with a confirmation summary
        # so the model knows what was recorded and doesn't repeat the call.
        event_dict = obj.model_dump(mode="json")
        summary_exclude = {"comment", "evidence", "note_id", "review_anchor"}
        summary_fields = [
            field_name
            for field_name in event_dict
            if field_name not in summary_exclude
        ]
        summary = {
            field_name: event_dict[field_name]
            for field_name in summary_fields
            if field_name in event_dict
        }

        is_plan = isinstance(obj, ExtractionPlan)
        kind = "Plan" if is_plan else "Event"
        return ToolResult(
            success=True,
            status={
                "ok": True,
                "recorded": name,
                "summary": summary,
                "message": f"{kind} recorded. Do not call {name} again for this same finding.",
            },
            obj=obj,
        )

    def is_terminal(self, name: str) -> bool:
        """Check if a tool is terminal (ends note extraction)."""
        tool_def = self._tools.get(name)
        return tool_def.is_terminal if tool_def else False


def create_registry_from_models(
    models: list[tuple[str, str, type[BaseModel]]],
) -> ToolRegistry:
    """
    Create a tool registry from a list of (name, description, model) tuples.

    Args:
        models: List of (name, description, pydantic_model) tuples

    Returns:
        Configured ToolRegistry
    """
    registry = ToolRegistry()
    for name, description, model in models:
        registry.register(name=name, description=description, model=model)
    return registry


class GatedToolRegistry(ToolRegistry):
    """
    Two-phase tool registry with a classification gate.

    Phase "gate": Only gate tools + finish_note_extraction are available.
    Phase "extract": Matched phase-2 tools + shared tools + finish_note_extraction.

    The gate tool must produce a model with a field (default: ``report_type``)
    whose value is looked up in ``phase_map`` to select the extract-phase tools.
    """

    def __init__(
        self,
        gate_tools: dict[str, ToolDefinition],
        phase_map: dict[str, dict[str, ToolDefinition]],
        shared_tools: dict[str, ToolDefinition] | None = None,
        gate_field: str = "report_type",
        single_note: bool = True,
    ):
        # Skip ToolRegistry.__init__ (we build _tools ourselves)
        self._single_note = single_note
        self._tools: dict[str, ToolDefinition] = {}

        # Store originals for fresh()
        self._gate_tools = gate_tools
        self._phase_map = phase_map
        self._shared_tools = shared_tools or {}
        self._gate_field = gate_field

        # State
        self._phase: str = "gate"
        self._gate_result: str | None = None
        self._gate_event: BaseModel | None = None

        # Register builtins then set up gate phase
        self._register_builtin_tools()
        self._rebuild_tools()

    def _rebuild_tools(self) -> None:
        """Rebuild ``_tools`` based on current phase."""
        # Preserve builtin terminal tools (finish/stop)
        builtins = {k: v for k, v in self._tools.items() if v.is_terminal}

        if self._phase == "gate":
            self._tools = {**self._gate_tools, **builtins}
        else:
            # extract phase: matched tools + shared + builtins
            matched = self._phase_map.get(self._gate_result or "", {})
            self._tools = {**matched, **self._shared_tools, **builtins}

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def gate_result(self) -> str | None:
        return self._gate_result

    @property
    def gate_event(self) -> BaseModel | None:
        return self._gate_event

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool. If it's a gate tool, transition to extract phase."""
        result = super().execute(name, arguments)

        # Check for gate transition
        if (
            result.success
            and result.obj is not None
            and self._phase == "gate"
            and name in self._gate_tools
        ):
            # Read the classification value
            classification = getattr(result.obj, self._gate_field, None)
            if classification is not None:
                classification_str = (
                    classification.value
                    if hasattr(classification, "value")
                    else str(classification)
                )

                if classification_str in self._phase_map:
                    self._gate_result = classification_str
                    self._gate_event = result.obj
                    self._phase = "extract"
                    self._rebuild_tools()
                    logger.info(
                        "Gate transition: %s -> extract (classified as %s)",
                        name,
                        classification_str,
                    )
                    # Override status to inform LLM
                    result.status = {
                        "ok": True,
                        "classified_as": classification_str,
                        "message": "Now extract findings using the tools now available.",
                    }
                else:
                    logger.warning(
                        "Gate tool %s returned unknown classification %r; "
                        "valid keys: %s",
                        name,
                        classification_str,
                        list(self._phase_map.keys()),
                    )
                    result = ToolResult(
                        success=False,
                        status={"ok": False},
                        error={
                            "type": "GateError",
                            "message": (
                                f"Unknown classification: {classification_str}. "
                                f"Valid: {list(self._phase_map.keys())}"
                            ),
                        },
                    )

        return result

    def fresh(self) -> GatedToolRegistry:
        """Return a new instance reset to the gate phase (for thread safety)."""
        return GatedToolRegistry(
            gate_tools=self._gate_tools,
            phase_map=self._phase_map,
            shared_tools=self._shared_tools,
            gate_field=self._gate_field,
            single_note=self._single_note,
        )
