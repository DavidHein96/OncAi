"""Tests for function-calling tool validation and schema generation."""

from __future__ import annotations

import pytest

from oncai.fc_extraction.models import ApproxDate, ExtractionEvent
from oncai.fc_extraction.tools import ToolRegistry


class ExampleEvent(ExtractionEvent):
    finding: str
    finding_date: ApproxDate


def _registry() -> ToolRegistry:
    registry = ToolRegistry(single_note=True)
    registry.register(
        name="record_example",
        description="Record an example finding.",
        model=ExampleEvent,
        event_identity_fields=("finding",),
        comparison_fields=("finding_date",),
    )
    return registry


def test_tool_registry_stores_comparison_metadata() -> None:
    tool = _registry().get("record_example")

    assert tool is not None
    assert tool.event_identity_fields == ("finding",)
    assert tool.comparison_fields == ("finding_date",)


def test_tool_registry_rejects_unknown_comparison_metadata_fields() -> None:
    registry = ToolRegistry(single_note=True)

    with pytest.raises(ValueError, match="missing_field"):
        registry.register(
            name="record_example",
            description="Record an example finding.",
            model=ExampleEvent,
            event_identity_fields=("missing_field",),
        )

    with pytest.raises(ValueError, match="missing_compare_field"):
        registry.register(
            name="record_example",
            description="Record an example finding.",
            model=ExampleEvent,
            comparison_fields=("missing_compare_field",),
        )


def test_tool_execution_rejects_extra_fields() -> None:
    result = _registry().execute(
        "record_example",
        {
            "note_id": "N1",
            "finding": "RCC",
            "finding_date": {"date": "2024-01-01", "precision": 3},
            "unexpected": "do not persist this",
        },
    )

    assert result.success is False
    assert result.error is not None
    assert result.error["type"] == "ValidationError"
    assert result.error["fields"][0]["type"] == "extra_forbidden"
    assert result.error["fields"][0]["loc"] == ("unexpected",)


def test_tool_execution_rejects_nested_extra_fields() -> None:
    result = _registry().execute(
        "record_example",
        {
            "note_id": "N1",
            "finding": "RCC",
            "finding_date": {
                "date": "2024-01-01",
                "precision": 3,
                "unexpected": "do not persist this",
            },
        },
    )

    assert result.success is False
    assert result.error is not None
    assert result.error["type"] == "ValidationError"
    assert result.error["fields"][0]["type"] == "extra_forbidden"
    assert result.error["fields"][0]["loc"] == ("finding_date", "unexpected")


def test_tool_execution_accepts_scalar_or_list_evidence() -> None:
    string_result = _registry().execute(
        "record_example",
        {
            "note_id": "N1",
            "finding": "RCC",
            "finding_date": {"date": "2024-01-01", "precision": 3},
            "evidence": "clear cell RCC is present",
        },
    )
    list_result = _registry().execute(
        "record_example",
        {
            "note_id": "N1",
            "finding": "oncocytoma",
            "finding_date": {"date": None, "precision": 0},
            "evidence": ["oncocytic neoplasm", "favor oncocytoma"],
        },
    )

    assert string_result.success is True
    assert string_result.obj is not None
    assert string_result.obj.model_dump(mode="json")["evidence"] == (
        "clear cell RCC is present"
    )
    assert list_result.success is True
    assert list_result.obj is not None
    assert list_result.obj.model_dump(mode="json")["evidence"] == [
        "oncocytic neoplasm",
        "favor oncocytoma",
    ]


def test_tool_schema_forbids_additional_properties() -> None:
    tool = next(
        tool
        for tool in _registry().to_openai_tools()
        if tool["name"] == "record_example"
    )

    assert tool["parameters"]["additionalProperties"] is False
    assert (
        tool["parameters"]["properties"]["finding_date"]["additionalProperties"]
        is False
    )
