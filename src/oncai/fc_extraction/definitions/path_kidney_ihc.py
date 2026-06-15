"""
IHC (immunohistochemistry) extraction definition.

Converts the structured-output ihc/ YAML schema into a single
function-calling tool that is called once per IHC marker found
in the report.

This is the ideal FC use case: a variable number of structurally
identical extractions from one note, with parallel tool calls.
The model calls record_ihc_result N times (once per marker).

Used with batch_single.py for single-note processing.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Annotated

from pydantic import BeforeValidator, Field

from ..enum_helpers import build_enum_lookup, normalize_against
from ..models import ExtractionEvent, ExtractionPlan
from ..tools import ToolRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


DEFINITION_NAME = "PathKidneyIhc"


# -----------------------
# Panel test names
# -----------------------
class PanelTestName(StrEnum):
    CK20 = "CK20"
    CK7 = "CK7"
    AE1_AE3 = "AE1/AE3"
    CK5_6 = "CK5/6"
    BAP_1 = "BAP-1"
    CA_IX = "CA-IX"
    CAM5_2 = "CAM5.2"
    CATHEPSIN_K = "Cathepsin-K"
    CD10 = "CD10"
    CD117 = "CD117"
    CK903 = "CK903"
    E_CADHERIN = "E-Cadherin"
    EMA = "EMA"
    HMB_45 = "HMB-45"
    MELANA = "MelanA"
    PAX_2 = "PAX-2"
    PAX_8 = "PAX-8"
    RACEMASE = "Racemase"
    RCC = "RCC"
    SMA = "SMA"
    VIMENTIN = "Vimentin"
    INI_1 = "INI-1"
    FH = "FH"
    SDHB = "SDHB"
    GATA3 = "GATA3"
    P63 = "p63"
    ALK_1 = "ALK-1"
    WT_1 = "WT-1"
    CD31 = "CD31"
    TFE3 = "TFE3"
    TTF_1 = "TTF-1"
    TFEB = "TFEB"
    PD_L1_TUMOR_CELLS = "PD-L1-tumor-cells"
    PD_L1_INFLAMMATORY_CELLS = "PD-L1-inflammatory-cells"
    KI_67_PROLIFERATIVE_INDEX = "Ki-67-proliferative-index"
    OTHER = "Other"


# -----------------------
# Panel test results
# -----------------------
class PanelTestStatus(StrEnum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    INTACT = "Intact"
    NORMAL = "Normal"
    AMPLIFIED = "Amplified"
    LOSS = "Loss"
    RETAINED = "Retained"
    REARRANGED = "Rearranged"
    REDUCED_EXPRESSION = "Reduced expression"
    NO_EVIDENCE_OF_REARRANGEMENT = "No evidence of rearrangement"
    NO_EVIDENCE_OF_TRANSLOCATION = "No evidence of translocation"
    TRANSLOCATION_AS_PER_REPORT = "Translocation"
    PERCENT_AS_PER_REPORT = "% as per report"
    OTHER_NUMERIC_RESULT = "Other numeric result as per report"
    RESULT_NOT_PROVIDED = "Result not provided"
    MULTIPLE_RESULTS = "Multiple results"


class PanelTestIntensity(StrEnum):
    STRONGLY = "strongly"
    WEAKLY = "weakly"
    VARIABLY = "variably"
    MINIMALLY = "minimally"
    NOT_SPECIFIED = "not specified"


class PanelTestExtent(StrEnum):
    PATCHY = "patchy"
    DIFFUSE = "diffuse"
    FOCAL = "focal"
    RARE_CELLS = "rare-cells"
    SCATTERED = "scattered"
    ATYPICAL_CELLS = "atypical-cells"
    FOCAL_PATCHY = "focal-patchy"
    SUBCLONAL = "subclonal"
    SINGLE_CELLS = "single-cells"
    NOT_SPECIFIED = "not specified"


class PanelTestPattern(StrEnum):
    MEMBRANOUS = "membranous"
    CIRCUMFERENTIAL_MEMBRANOUS_BOX_LIKE = "circumferential membranous (box-like)"
    BASAL_LATERAL_MEMBRANOUS_CUP_LIKE = "basal lateral membranous (cup-like)"
    LUMINAL = "luminal"
    CYTOPLASMIC = "cytoplasmic"
    NUCLEAR = "nuclear"
    NOT_SPECIFIED = "not specified"


_TEST_NAME_LOOKUP = build_enum_lookup(PanelTestName)
_TEST_STATUS_LOOKUP = build_enum_lookup(PanelTestStatus)
_TEST_INTENSITY_LOOKUP = build_enum_lookup(PanelTestIntensity)
_TEST_EXTENT_LOOKUP = build_enum_lookup(PanelTestExtent)
_TEST_PATTERN_LOOKUP = build_enum_lookup(PanelTestPattern)


def normalize_test_name(v: object) -> str:
    """Match a possibly-misformatted test name to its canonical PanelTestName value.

    >>> normalize_test_name("ck20")
    'CK20'
    >>> normalize_test_name("Pax8")
    'PAX-8'
    >>> normalize_test_name("cam 5.2")
    'CAM5.2'
    """
    return normalize_against(v, _TEST_NAME_LOOKUP, "test_name", logger=logger)


def normalize_test_status(v: object) -> str:
    """Match a possibly-misformatted result to its canonical PanelTestStatus value.

    >>> normalize_test_status("positive")
    'Positive'
    >>> normalize_test_status("LOSS")
    'Loss'
    """
    return normalize_against(v, _TEST_STATUS_LOOKUP, "test_status", logger=logger)


def normalize_test_intensity(v: object) -> str:
    return normalize_against(v, _TEST_INTENSITY_LOOKUP, "test_intensity", logger=logger)


def normalize_test_extent(v: object) -> str:
    return normalize_against(v, _TEST_EXTENT_LOOKUP, "test_extent", logger=logger)


def normalize_test_pattern(v: object) -> str:
    return normalize_against(v, _TEST_PATTERN_LOOKUP, "test_pattern", logger=logger)


# =============================================================================
# Tool Models
# =============================================================================


class RecordIhcResult(ExtractionEvent):
    """Record one IHC/FISH test result. Should be called once per distinct test and test result found in the report"""

    specimen_id: str = Field(
        "not specified",
        description=(
            "Which specimen these findings belong to, typically a single uppercase letter. If the report contains multiple specimens but it is not clear which specimen was used for the tests, use the specimen id 'unclear_specimen'. Often the block used for the test is specified, and this can be used to help infer the specimen 'A1' -> specimen 'A'. For the instances where there are multiple identical tests with the same result that appear to be from different blocks/specimens, you can make a specimen id in the form of 'A||B' to indicate this. Remeber, this is only if the test has the exact same result."
        ),
    )

    flag_for_sub_specimen_heterogeneity: bool = Field(
        False,
        description=(
            "Set to True if there are multiple results from the same specimen and same test that have conflicting results, suggesting possible intratumoral heterogeneity. typically this would be shown by the pathologist by reporting results from different tissue blocks 'A1' and 'A2' from the same specimen 'A'. This is a rare but important scenario to flag. Remember this must be the SAME specimen and SAME test showing conflicting results. Typically, a single specimen will have a 1:1 correspondance of tests to blocks, so before calling this, ensure that the conflicting results are truly from the same specimen and not just different blocks from different specimens. If set, use Multiple results in the standardized_test_status field"
        ),
    )
    given_test_name: str = Field(
        ...,
        description="Test name exactly as written in the report (e.g., 'PD-L1 (22C3)', 'Ki-67')",
    )
    standardized_test_name: Annotated[
        PanelTestName, BeforeValidator(normalize_test_name)
    ] = Field(
        ...,
        description="Canonical test name, Other if not in canonical list",
    )
    given_result: str = Field(
        ...,
        description=(
            "Result exactly as written in the report "
            "(e.g., 'strongly positive', '80%', '2+', 'TPS 50%')"
        ),
    )
    standardized_test_status: Annotated[
        PanelTestStatus, BeforeValidator(normalize_test_status)
    ] = Field(
        ...,
        description="Standardized result interpretation, use 'Multiple results' if flag_for_specimen_heterogeneity is True. All results should have a status. When you see the result 'Intact (Positive nuclear staining ...)' favor returning only the result 'Intact', no other qualifiers are needed. Only do this if BOTH the words Intact AND Positive are used together to describe the result for the same test. Similarly, when you see the results 'Negative (loss ...)' favor returning only the result 'Loss'. Only do this if BOTH the words negative AND loss are used together to describe the result for the same test. ",
    )
    standardized_test_intensity: Annotated[
        PanelTestIntensity, BeforeValidator(normalize_test_intensity)
    ] = Field(
        PanelTestIntensity.NOT_SPECIFIED,
        description="Standardized test intensity, not all tests will have an intensity, so default to 'not specified'",
    )
    standardized_test_extent: Annotated[
        PanelTestExtent, BeforeValidator(normalize_test_extent)
    ] = Field(
        PanelTestExtent.NOT_SPECIFIED,
        description="Standardized test extent, not all tests will have an extent, so default to 'not specified'",
    )
    standardized_test_pattern: Annotated[
        PanelTestPattern, BeforeValidator(normalize_test_pattern)
    ] = Field(
        PanelTestPattern.NOT_SPECIFIED,
        description="Standardized test pattern, not all tests will have a pattern, so default to 'not specified'",
    )


class PlanSpecimens(ExtractionPlan):
    """Plan tool: enumerate specimens and blocks before extraction."""

    specimen_summary: str = Field(
        ...,
        description=(
            "Info about all the specimens in the report, typically labeled with uppercase letters (e.g., 'A: left radical nephrectomy, B: periaortic lymph node biopsy').  For single-specimen reports without explicit labels, just default to 'A', and assume the specimen used for all tests is 'A' even if it is not explicit. If there are both outside report specimen names and internal specimen labels, use the internal specimen labels. Make note of which specimens have test results. This helps organize the extraction process and ensures that each test result is correctly associated with its respective specimen. Between sections of notes you can use a separator like '||' to help organize things. Another consideration is that sometimes it will seem like multiple specimens/blocks are used for a test, see if the results are all the same. If they are all the same, you may only need to do a single test record."
        ),
    )
    blocks_summary: str = Field(
        ...,
        description=(
            "Info about the blocks mentioned in the report, typically labeled with uppercase letters (corresponding to the specimen) and a number (e.g., 'A1, B2). For single-specimen reports without explicit labels, make note note of that here. If there are both outside report block names and internal block labels, use the internal block labels. In some instances the block used will not be mentioned, and only the specimen mentioned, also note that here. Not all specimens will have mentioned blocks and that is fine, typically block is only mentioned if it was used for tests. Only make note of mentioned blocks here. This helps organize the extraction process and ensures that we catch any specimen-level heterogeneity in test results that may be reported at the block level. Between sections of notes you can use a separator like '||' to help organize things. Another consideration is that sometimes it will seem like multiple specimens/blocks are used for a test, see if the results are all the same. Make note of that here. If they are all the same, when we call record_ihc_result, we may be able to combine them into a single test record."
        ),
    )


class PlanIhcTests(ExtractionPlan):
    """Plan tool: enumerate IHC/FISH tests before extraction."""

    ihc_test_list: str = Field(
        ...,
        description=(
            "List all the IHC and FISH tests & results mentioned in the report, using the exact test names, test results, and any associated specimen/block labels as written in the report (e.g., 'CK9 positive, A1'). This helps ensure that all tests are captured and can guide the model in systematically extracting results for each test. If the report does not mention any IHC or FISH tests, use 'none'. Ensure all tests are captured, particularly if there are addendum / additional results. Prefer addendum results over initial results if there are discrepancies, as addendums typically contain the most up-to-date information. If there are multiple tests with the same name but different results, make sure to capture all of them and their associated specimen/block labels to help identify any potential heterogeneity. For the cases of it appearing like multiple specimens were used, or blocks from different specimens and they all have the same result, make note. They may be able to be combined into a single test record. Between sections of notes you can use a separator like '||' to help organize things."
        ),
    )
    ihc_test_ambiguity_notes: str = Field(
        ...,
        description=(
            "Notes on any potential ambiguities or complexities in the IHC/FISH test results that may require special attention. This includes conflicting results on the same specimen, unclear results, and instances where there are multiple specimens and the specimen used for a test is not clear. If there is only a single specimen and the id/blocks are not specified assume specimen A and all tests belong to that specimen. If the report contains multiple specimens but it is not clear which specimen was used for the tests, use the specimen id 'unclear_specimen' down stream and make note of this here. Sometimes, you may be able to infer the specimen used for a test based on context, for example if there are two specimens, A kidney tumor from a nephrectomy, and B benign perinephric fat, and the report mentions positive test results used to support the diagnosis of renal cell carcinoma, you can reasonably infer that the positive test results belong to specimen A, the kidney tumor, and not specimen B, the benign perinephric fat. Make note of this inference here. This field is meant to capture any complexities or ambiguities in the report that may impact the extraction process and ensure that they are properly addressed during extraction."
        ),
    )


class FlagReportForReview(ExtractionEvent):
    """Flag tool: only for genuinely ambiguous reports. See registry description."""

    reason: str = Field(
        ...,
        description=(
            "Reason for flagging this report for review, 'conflicting information', or 'multiple possible interpretations'. This should only be used for cases where the report is genuinely ambiguous or complex and cannot be reliably extracted with the current toolset"
        ),
    )
    review_anchor: list[str] = Field(
        ...,
        description=(
            "Exact text snippets from the report that should be highlighted "
            "and used as jump targets in the review app. Each item should be "
            "an exact substring containing the ambiguity. If several sections "
            "are relevant, include multiple anchors."
        ),
    )


# =============================================================================
# System Prompt
# =============================================================================


SYSTEM_PROMPT = """\
You are a clinical data extraction specialist. Extract ALL immunohistochemistry
(IHC) and FISH test results from the pathology report. Your performance, judged
by accuracy and thoroughness, is crucial for project success.

# WORKFLOW
1. Call `plan_specimens` ONCE to enumerate the specimens and blocks present.
2. Call `plan_ihc_tests` ONCE to enumerate the IHC/FISH tests present and any associated specimen/block labels, and to note any potential ambiguities or complexities in the report that may require special attention.
3. Call `record_ihc_result` ONCE PER distinct test (parallel calls OK).
4. Call `flag_report_for_review` only if the report is genuinely ambiguous or
   cannot be captured with the current toolset.
5. Call `finish_note_extraction` once all markers have been recorded.

# TEST NAME STANDARDIZATION
Use the canonical name rather than the synonym. Common synonyms:
- AE1/AE3: Pankeratin, pancytokeratin, keratin
- CD117: c-kit, KIT
- CK903: High molecular weight cytokeratin
- CAM5.2: Low molecular weight cytokeratin
- MelanA: Melan A, MART1
- Racemase: P504S, AMACR
- p63: KET
- INI-1: SMARCB1, SWI/SNF related BAF47, hSNF5
- CA-IX: CA 9, CA-9, Carbonic anhydrase 9
- FH: Fumarate hydratase

(always include the literal test name as given in the report in the given_test_name field, even if it is a synonym, and use the standardized_test_name field for the canonical name)

# RESULT STANDARDIZATION
Some tests (PD-L1, Ki-67) report results as percentages or scores. FISH tests
may report a translocation locus (e.g., "t(9;22)(q34;q11)"). For these:
- Set standardized_test_status to "% as per report" or "Translocation as per report"
- Put the exact result string in given_result (e.g., "50%", "t(9;22)(q34;q11)")
For all other tests, standardize the status to positive / negative / intact /
loss / etc. In given_result, do NOT do numeric conversions — preserve the exact
wording from the report (e.g., "50%" stays as the string "50%").

(always include the full result string as given in the report in the given_result field, and use the standardized_test_status field for the standardized interpretation. In the given feild include all qualifiers such as intensity, extent, pattern, numerics etc.)

# OUTSIDE CONSULTATION CASES
Some reports are outside consultations whose specimens/blocks do not follow the
typical "A, B, C / A1, B2" naming. When BOTH internal and external names are
provided (e.g., "Outside case S445. A. Left Breast biopsy (S445-C7)"), use the
internal name ("A"). If only external names are provided (e.g., "Specimen 1"),
use those.

# EVIDENCE FIELDS
- Use `evidence` for exact source-note snippets supporting the extraction, especially the test name/result text when available.
- The snippet must match the text exactly so it can be used in review. It should have one to two words before and after the relevant text to provide extra context for matching to the report

# AMBIGUITIES
- If there are multiple specimens but it is not clear which specimen was used for the tests, use the specimen id "unclear_specimen" for all tests and make note of this in the ihc_test_ambiguity_notes field.
- If there is only a single specimen and the id/blocks are not specified, assume specimen "A" and make note of this assumption in the ihc_test_ambiguity_notes field.
- If there are conflicting results for the same test on the SAME specimen, set flag_for_sub_specimen_heterogeneity to True and use "Multiple results" for the standardized_test_status, and make note of this in the ihc_test_ambiguity_notes field.
- Only use the flag_report_for_review tool if the report contains contradictions, significant data corruption, or multiple possible interpretations / complexities that cannot be resolved with the current toolset. When flagging, populate review_anchor with exact report text snippets the review app can highlight and jump to.
- In some cases a test may be associated with multiple specimens/blocks but all the results are the same, in this case you can combine them into a single test record and note the multiple specimen/block association in the specimen_id field (e.g., "A||B" if the same result is reported for both specimen A and specimen B. If the same test was run on multuiple blocks from the same specimen and all the results are the same, just use a single specimen id (e.g., "A"). This is to avoid duplication of test records. Remeber, if the results are different for the same test on the same specimen or the same test on different specimens, do NOT combine them, each distinct result should have its own test record to ensure accuracy and to capture any potential heterogeneity.

# GENERAL WORKFLOW
1. Read the report carefully.
2. Call plan_specimens to enumerate specimens/blocks (this will be helpful for organizing your thoughts and guiding the next steps)
3. Call plan_ihc_tests to enumerate all the tests and results
4. If at this point, you realize there are no tests in the report, you can finish extraction early. Otherwise, move to the next step.
5. For each distinct test, call record_ihc_result with the given test name and result, and the standardized test name and standardized interpretation.
6. Call flag_report_for_review if necessary
7. Call finish_note_extraction when done.
"""


# =============================================================================
# Registry Setup
# =============================================================================


def create_path_kidney_ihc_registry() -> ToolRegistry:
    """Create and configure the tool registry for IHC extraction."""
    registry = ToolRegistry(single_note=True)

    registry.register(
        name="plan_specimens",
        description=(
            "Plan the specimens mentioned in the report to help guide the extraction process. "
            "This should be called once at the beginning of the extraction process to identify all specimens and blocks mentioned in the report, which will help associate test results with the correct specimens."
        ),
        model=PlanSpecimens,
    )

    registry.register(
        name="plan_ihc_tests",
        description=(
            "Plan the IHC and FISH tests mentioned in the report to help guide the extraction process. "
            "This should be called once at the beginning of the extraction process after planning the specimens to identify all IHC and FISH tests mentioned in the report, which will help ensure that all tests are captured and can guide the systematic extraction of results for each test."
        ),
        model=PlanIhcTests,
    )
    registry.register(
        name="record_ihc_result",
        description=(
            "Record a single immunohistochemistry (IHC / FISH) test result. "
            "Call this ONCE for each distinct IHC marker found in the report. "
            "If there are 5 markers, call this tool 5 times (parallel calls OK). "
            "Record the exact name/result as written and the standardized versions."
            "Call this after planning the specimens and tests"
        ),
        model=RecordIhcResult,
        event_identity_fields=("specimen_id", "standardized_test_name"),
        comparison_fields=(
            "flag_for_sub_specimen_heterogeneity",
            "given_result",
            "standardized_test_status",
            "standardized_test_intensity",
            "standardized_test_extent",
            "standardized_test_pattern",
        ),
    )

    registry.register(
        name="flag_report_for_review",
        description=(
            "Flag a report for manual review by a human expert. "
            "This should only be used for cases where the report is genuinely ambiguous or complex and cannot be reliably extracted with the current toolset. "
            "Provide an appropriate reason for flagging (e.g., 'conflicting information', 'multiple possible interpretations')."
        ),
        model=FlagReportForReview,
    )
    return registry
