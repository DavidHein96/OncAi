"""
Example single-note extraction definition.

This file is a template and documentation for creating new single-note FC
extraction definitions. Copy this file and modify it for your task.

=============================================================================
OVERVIEW
=============================================================================

A definition consists of:

1. ENUMS — categorical values for your extraction fields.
2. TOOL MODELS — Pydantic models that define the schema for each extraction
   tool. Inherit from ExtractionEvent (or ExtractionPlan for orientation tools).
3. SYSTEM_PROMPT — instructions for the LLM.
4. DEFINITION_NAME — used as the output subdirectory name in fc_outputs/.
5. create_<name>_registry() — factory that registers tools with descriptions.

=============================================================================
HOW IT WORKS
=============================================================================

Each note (e.g. a pathology report) is processed independently:

1. The note is sent to the LLM with the system prompt and the registered tools.
2. The LLM calls tools to record findings (e.g. record_diagnosis).
3. Tool call payloads are validated with Pydantic and saved to the output JSONL.
4. The LLM calls ``finish_single_extraction`` when done with the note.

There is no cross-note state — single-note extraction is for one-report-in,
one-set-of-extractions-out tasks like pathology reports.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from ..models import ApproxDate, ExtractionEvent
from ..tools import ToolRegistry

# =============================================================================
# DEFINITION METADATA
# =============================================================================

DEFINITION_NAME = "Example"


# =============================================================================
# ENUMS
# =============================================================================
# Define categorical values as Enums. This provides:
# - Type safety and validation
# - Clear documentation of allowed values
# - Auto-generated JSON schema for the LLM


class DiagnosisType(StrEnum):
    """Example enum for diagnosis categories."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    RECURRENCE = "recurrence"


class TreatmentIntent(StrEnum):
    """Example enum for treatment intent."""

    CURATIVE = "curative"
    PALLIATIVE = "palliative"
    ADJUVANT = "adjuvant"
    NEOADJUVANT = "neoadjuvant"


# =============================================================================
# TOOL MODELS
# =============================================================================
# Each tool is a Pydantic model that inherits from ExtractionEvent.
#
# ExtractionEvent provides:
#   - note_id: str | None  (auto-populated by the system)
#   - evidence: list[str] | str  (exact source snippets for review highlighting)
#
# Key principles:
#   - Use clear, descriptive field names
#   - Add Field descriptions — these help the LLM understand what to extract
#   - Use appropriate types (str, int, float, bool, Enum, list, etc.)
#   - Use ApproxDate for dates with variable precision
#   - Make fields optional (| None with default None) when data may be missing


class RecordDiagnosis(ExtractionEvent):
    """
    Record a diagnosis finding.

    The docstring becomes part of the tool description shown to the LLM.
    Be specific about when this tool should be called.
    """

    diagnosis_date: ApproxDate = Field(
        ...,
        description="Date of diagnosis (YYYY, YYYY-MM, or YYYY-MM-DD).",
    )
    diagnosis_type: DiagnosisType = Field(
        ...,
        description="Whether this is a primary, secondary, or recurrence diagnosis.",
    )
    diagnosis_name: str = Field(
        ...,
        description="Name of the diagnosis (e.g., 'renal cell carcinoma').",
    )
    stage: str | None = Field(
        None,
        description="Cancer stage if documented (e.g., 'Stage III', 'T3aN0M0').",
    )
    grade: str | None = Field(
        None,
        description="Tumor grade if documented (e.g., 'Grade 2', 'high grade').",
    )


class RecordTreatment(ExtractionEvent):
    """
    Record a treatment event.

    Call this for surgeries, systemic therapies, radiation, etc.
    """

    treatment_date: ApproxDate = Field(
        ...,
        description="Date treatment started or was performed.",
    )
    treatment_name: str = Field(
        ...,
        description="Name of the treatment (e.g., 'radical nephrectomy').",
    )
    treatment_type: Literal["surgery", "systemic", "radiation", "other"] = Field(
        ...,
        description="Category of treatment.",
    )
    intent: TreatmentIntent | None = Field(
        None,
        description="Intent of treatment if documented.",
    )
    outcome: str | None = Field(
        None,
        description="Outcome if documented (e.g., 'complete resection').",
    )


class FlagReportForReview(ExtractionEvent):
    """Flag the report for human review when the structured tools cannot resolve it."""

    comment: str = Field(
        "",
        description=(
            "Optional extra context for the review flag. Put the specific "
            "reason in reason and exact source-note snippets in review_anchor."
        ),
    )
    reason: str = Field(
        ...,
        description=(
            "Reason the report needs review. Use only for genuine ambiguity, "
            "conflicting information, or wording that does not fit available "
            "structured fields."
        ),
    )
    review_anchor: list[str] = Field(
        ...,
        description=(
            "Exact text snippets from the report that should be highlighted "
            "and used as jump targets in the review app. Each item should be "
            "an exact substring containing the ambiguity."
        ),
    )


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """\
You are a clinical data extraction specialist reviewing a single clinical
report.

Your task is to extract:
1. Diagnosis information (type, date, stage, grade)
2. Treatment events (surgeries, systemic therapy, radiation)

EXTRACTION RULES:
- Call a tool for each distinct finding in the report.
- Put exact source-note snippets in `evidence`.
- Call flag_report_for_review only for genuine ambiguity or conflicting text;
  use comment for rationale and include exact report-level snippets in review_anchor.
- Call finish_single_extraction when you have recorded every finding.

DATE FORMAT:
- Use the format that matches the precision available in the text:
  - Year only: "2023"
  - Month/year: "2023-03"
  - Exact date: "2023-03-15"
- Use null when no date can be determined.

IMPORTANT:
- Do not hallucinate or infer information not explicitly stated.
- If uncertain about a value, use null for optional fields.
- Record each distinct event separately.
"""


# =============================================================================
# REGISTRY FACTORY
# =============================================================================


def create_example_registry() -> ToolRegistry:
    """Create the tool registry for example single-note extraction."""
    registry = ToolRegistry(single_note=True)

    registry.register(
        name="record_diagnosis",
        description=(
            "Record a diagnosis finding from the report. "
            "Call this when you find information about a diagnosis, "
            "including the type, date, stage, and grade."
        ),
        model=RecordDiagnosis,
        comparison_fields=(
            "diagnosis_date",
            "diagnosis_type",
            "diagnosis_name",
            "stage",
            "grade",
        ),
    )

    registry.register(
        name="record_treatment",
        description=(
            "Record a treatment event. "
            "Call this for surgeries, chemotherapy, immunotherapy, radiation, "
            "etc."
        ),
        model=RecordTreatment,
        comparison_fields=(
            "treatment_date",
            "treatment_name",
            "treatment_type",
            "intent",
            "outcome",
        ),
    )

    registry.register(
        name="flag_report_for_review",
        description=(
            "Flag a report for human review when it is genuinely ambiguous or "
            "contains conflicting text that the structured fields cannot resolve. "
            "Include exact source-note snippets in review_anchor."
        ),
        model=FlagReportForReview,
    )

    return registry
