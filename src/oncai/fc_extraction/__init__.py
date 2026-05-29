"""
Function-calling based extraction for single notes.

This module provides a single-note extraction approach using LLM function
calling / tool use. Each note (e.g. a pathology report) is processed
independently — the model calls tools to record findings, then signals
completion. There is no cross-note state; designed for reports where one
input = one set of extractions.

Usage:
    from pydantic import Field
    from oncai.fc_extraction import (
        ExtractionEvent,
        SingleNoteConfig,
        ToolRegistry,
        run_fc_single_batch,
    )

    # Tool payloads inherit from ExtractionEvent so ``note_id`` / ``comment``
    # are auto-populated; you only declare the task-specific fields.
    class MyEvent(ExtractionEvent):
        event_date: str = Field(..., description="When the event occurred")
        details: str = Field(..., description="What happened")

    # Register tools — single_note=True wires the right finish_extraction tool
    # and drops the multinote-only stop_workflow.
    registry = ToolRegistry(single_note=True)
    registry.register(
        name="record_my_event",
        description="Record an event when...",
        model=MyEvent,
    )

    # Configure single-note extraction
    config = SingleNoteConfig(
        name="MyDefinition",
        system_prompt="You are extracting...",
    )

    # Run batch
    result = run_fc_single_batch(
        registry=registry,
        config=config,
        client=fc_client,
        db_path="oncai.duckdb",
        source_table="raw.pathology",
        output_dir="outputs",
        batch_name="v1",
    )
"""

from .batch_single import (
    SingleNoteBatchResult,
    SingleNoteConfig,
    run_fc_single_batch,
)
from .client import (
    FCBackend,
    FCClientConfig,
    FunctionCallingClient,
    NoteExtractionResult,
    VLLMFunctionCallingClient,
    get_fc_client,
)
from .load import (
    WideLoadResult,
    jsonl_to_wide_parquet,
)
from .manifest import (
    get_code_version,
    get_git_info,
    hash_string,
)
from .models import (
    ApproxDate,
    ApproxDateStr,
    ExtractionEvent,
    ExtractionPlan,
    FinishExtraction,
    NoteInfo,
)
from .tools import (
    ToolDefinition,
    ToolRegistry,
    ToolResult,
    create_registry_from_models,
)

__all__ = [
    # Models
    "ApproxDate",
    "ApproxDateStr",
    "ExtractionEvent",
    "ExtractionPlan",
    "FinishExtraction",
    "NoteInfo",
    # Tools
    "ToolDefinition",
    "ToolRegistry",
    "ToolResult",
    "create_registry_from_models",
    # Client
    "FCBackend",
    "FCClientConfig",
    "FunctionCallingClient",
    "NoteExtractionResult",
    "VLLMFunctionCallingClient",
    "get_fc_client",
    # Batch single
    "SingleNoteBatchResult",
    "SingleNoteConfig",
    "run_fc_single_batch",
    # Manifest helpers
    "get_code_version",
    "get_git_info",
    "hash_string",
    # Load
    "WideLoadResult",
    "jsonl_to_wide_parquet",
]
