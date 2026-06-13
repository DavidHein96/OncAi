"""
Base models for function-calling extraction.

Provides common models used across different extraction tasks.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

_APPROX_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


class _ToolArguments(BaseModel):
    """Base for structured LLM tool payloads."""

    model_config = ConfigDict(extra="forbid")


def _validate_approx_date_str(v: str | None) -> str | None:
    """Validate that value is None or matches YYYY, YYYY-MM, or YYYY-MM-DD."""
    if v is None:
        return v
    if not isinstance(v, str) or not _APPROX_DATE_RE.match(v):
        msg = f"ApproxDateStr must be None, YYYY, YYYY-MM, or YYYY-MM-DD, got: {v!r}"
        raise ValueError(msg)
    return v


ApproxDateStr = Annotated[
    str | None,
    AfterValidator(_validate_approx_date_str),
    Field(
        None,
        description=(
            "Date with precision encoded in the format: "
            "YYYY (year only), YYYY-MM (month/year), YYYY-MM-DD (exact date), "
            "or null if unknown."
        ),
    ),
]
"""A date string with variable precision: None, YYYY, YYYY-MM, or YYYY-MM-DD."""


class ApproxDate(_ToolArguments):
    """A date with variable precision."""

    date: str | None = Field(
        None,
        description="Date in YYYY-MM-DD format. Null if precision is 0.",
    )
    precision: Literal[0, 1, 2, 3] = Field(
        0,
        description="Date precision: 0=Unknown, 1=Year only, 2=Month/Year, 3=Full date.",
    )
    anchor: Literal["BOM", "EOM", "BOY", "EOY", "MID", "EXACT"] | None = Field(
        None,
        description="Anchor hint for imprecise dates: BOM=beginning of month, EOM=end of month, etc.",
    )


class FinishExtraction(_ToolArguments):
    """
    Signal that extraction from the current note is complete.
    This MUST be called as the final tool for each note.
    Used in multi-note workflows where notes are processed sequentially.
    """

    note_id: str = Field(..., description="ID of the clinical note being finished")
    reasoning: str = Field(
        ...,
        description="Brief summary of what was found/extracted from this note, like a commit message.",
    )
    confidence: float = Field(
        0.5,
        description="Confidence that all relevant information has been extracted (0.0-1.0).",
        ge=0.0,
        le=1.0,
    )
    needs_more_notes: bool = Field(
        True,
        description="Set to False if you are confident all needed information has been found across all notes so far.",
    )


class FinishSingleExtraction(_ToolArguments):
    """
    Signal that extraction from this note is complete.
    Used in single-note workflows where each note is processed independently.
    """

    note_id: str = Field(..., description="ID of the clinical note being finished")
    reasoning: str = Field(
        ...,
        description="Brief summary of what was found/extracted from this note, like a commit message.",
    )
    confidence: float = Field(
        0.5,
        description="Confidence that all relevant information has been extracted (0.0-1.0).",
        ge=0.0,
        le=1.0,
    )


class StopWorkflow(_ToolArguments):
    """
    Signal that the entire workflow should stop (no more notes needed).
    Use this when you have high confidence that all required information has been extracted.
    """

    note_id: str = Field(..., description="ID of the note where decision was made")
    reasoning: str = Field(
        ...,
        description="Explanation of why no more notes are needed.",
    )
    final_summary: str = Field(
        ...,
        description="Summary of all findings across all notes processed.",
    )


class NoteInfo(BaseModel):
    """Information about a clinical note being processed."""

    note_id: str
    note_date: str
    note_type: str | None = None
    department: str | None = None
    provider: str | None = None
    note_index: int = 0
    total_notes: int = 0
    is_final_note: bool = False


class ExtractionEvent(_ToolArguments):
    """Base class for extraction events. All tool outputs should inherit from this."""

    note_id: str = Field(
        ..., description="ID of the note this event was extracted from"
    )
    evidence: list[str] | str = Field(
        default_factory=list,
        description=(
            "Brief list of exact source-note snippets supporting this extraction. "
            "Snippets are provenance for the review UI and are not themselves "
            "extracted values. Should match text in the note exactly for review "
            "purposes, ideally with 1-2 words of context on either side."
        ),
    )


class ExtractionPlan(_ToolArguments):
    """Base class for plan tools — the model records its plan, not extracted data.

    Plans are serialized into a separate "plans" block in the JSONL output and
    are excluded from downstream staging by default. Use for tools whose purpose
    is to capture the model's reasoning/strategy before extraction, not findings
    that should land in data tables.
    """

    note_id: str = Field(..., description="ID of the note this plan was generated for")
