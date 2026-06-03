"""
Kidney pathology basic — fast/cheap triage sweep.

This workflow is a high-level routing pass over kidney pathology reports.
For each report it decides:
  1. Which downstream workflow(s) it should feed (nephrectomy, biopsy, metastasectomy).
  2. Whether the report contains cancer, and at a coarse level what kind
     (no cancer, benign, uncertain neoplasm, non-RCC malignancy, RCC confined,
     RCC with regional LN, metastatic RCC).

Single tool, single call per report. No per-specimen detail. The detailed
extraction lives in kidney_path_nephrectomy and path_procedure_site_hist.

Used with batch_single.py for single-note processing.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import Annotated

from pydantic import BeforeValidator, Field

from ..models import ExtractionEvent, ExtractionPlan
from ..tools import ToolRegistry

logger = logging.getLogger(__name__)

DEFINITION_NAME = "PathKidneyBasic"


# =============================================================================
# Enums
# =============================================================================


class KidneyCancerStatus(StrEnum):
    """Coarse cancer status for triage routing."""

    KIDNEY_CANCER_METASTATIC = "Metastatic kidney cancer"
    KIDNEY_CANCER_REGIONAL_LN = "Kidney cancer with regional lymph node involvement"
    KIDNEY_CANCER_LOCALIZED = "Kidney cancer, localized to kidney"
    NON_KIDNEY_CANCER_MALIGNANCY = "non-kidney cancer malignancy"
    UNCERTAIN_KIDNEY = "neoplasm of uncertain behavior of kidney"
    OTHER_UNCERTAIN = "uncertain neoplasm of non-kidney origin or uncertain malignancy"
    BENIGN_KIDNEY = "benign neoplasm of kidney"
    OTHER_BENIGN = "benign non-kidney neoplasm or other benign finding"
    NO_CANCER = "no cancer or non-diagnostic"
    UNCERTAIN = "uncertain — flag for review"


class PhysicianConfidence(StrEnum):
    """Confidence in the cancer status call."""

    DEFINITE = "definite"
    LIKELY = "likely"
    SUSPICIOUS = "suspicious"
    INDETERMINATE = "indeterminate"


# =============================================================================
# Normalization helpers
# =============================================================================


def _normalize_key(s: str) -> str:
    """Reduce a string to a canonical comparison key by lowercasing and
    stripping whitespace, hyphens, underscores, slashes, dots, and parens."""
    return re.sub(r"[\s\-_./()]+", "", s).lower()


def _build_enum_lookup(enum_cls: type[StrEnum]) -> dict[str, str]:
    """Build a {normalized_key: enum_value} mapping for an entire StrEnum."""
    return {_normalize_key(m.value): m.value for m in enum_cls}


def _normalize_against(v: object, lookup: dict[str, str], field_name: str) -> str:
    """Match a value against a normalized lookup; log when a correction is made."""
    if isinstance(v, StrEnum):
        return v.value
    s = str(v).strip()
    match = lookup.get(_normalize_key(s))
    if match is not None and match != s:
        logger.debug("normalizer fixed %s: %r -> %r", field_name, s, match)
    return match if match is not None else s


_CANCER_STATUS_LOOKUP = _build_enum_lookup(KidneyCancerStatus)
_CONFIDENCE_LOOKUP = _build_enum_lookup(PhysicianConfidence)


def normalize_kidney_cancer_status(v: object) -> str:
    return _normalize_against(v, _CANCER_STATUS_LOOKUP, "kidney_cancer_status")


def normalize_confidence(v: object) -> str:
    return _normalize_against(v, _CONFIDENCE_LOOKUP, "diagnostic_confidence")


# =============================================================================
# Tool models
# =============================================================================


class TriageReport(ExtractionEvent):
    """Report-level triage. Identify what specimen types are present and assign a
    coarse cancer status. Call ONCE per report — no per-specimen detail."""

    triage_summary: str = Field(
        ...,
        description=(
            "1-2 sentence summary: what specimens the report contains and the "
            "headline diagnosis. E.g., 'Left radical nephrectomy with clear cell "
            "RCC, pT3a, one positive renal hilar node' or 'CT-guided liver core "
            "biopsy showing metastatic clear cell RCC'."
        ),
    )
    has_primary_nephrectomy: bool = Field(
        ...,
        description=(
            "True if the report contains a primary nephrectomy specimen "
            "(radical, partial, or total). False otherwise. Routes the report to "
            "the nephrectomy extraction workflow."
        ),
    )
    has_biopsy: bool = Field(
        ...,
        description=(
            "True if the report contains any biopsy specimen — kidney biopsy, "
            "lymph node biopsy, biopsy of a metastatic site, etc. Includes "
            "core biopsy, fine needle aspiration, and other biopsy types. False "
            "if no biopsy specimen is present."
        ),
    )
    has_metastasectomy: bool = Field(
        ...,
        description=(
            "True if the report contains a resection specimen from a metastatic "
            "site (lung wedge resection, hepatectomy, bone resection, "
            "adrenalectomy for metastasis, etc.). Direct contiguous extension "
            "from a primary kidney tumor into a neighboring organ does NOT count "
            "as metastasectomy — that is part of the nephrectomy specimen."
        ),
    )
    has_ihc_or_molecular: bool = Field(
        ...,
        description=(
            "True if the report contains immunohistochemistry or molecular "
            "studies that inform the diagnosis. This includes IHC / FISH / molecular "
            "addenda and IHC / FISH / molecular studies mentioned in the main report. False "
            "if no IHC or molecular studies are present."
        ),
    )
    kidney_cancer_status: Annotated[
        KidneyCancerStatus, BeforeValidator(normalize_kidney_cancer_status)
    ] = Field(
        ...,
        description=(
            "Coarse cancer status for the report, prioritizing kidney-relevant "
            "findings. Pick the single most-aggressive/most-specific status the "
            "report supports. Use 'uncertain — flag for review' only when the "
            "report is genuinely ambiguous. Ensure that kidney cancer confirmed outside the kidney (e.g., in a distant lymph node or lung biopsy) is called 'metastatic kidney cancer' even if there is not technically a metastasectomy procedure. The metastasis could have been confirmed in a biopsy or as part of a nephrectomy."
        ),
    )
    diagnostic_confidence: Annotated[
        PhysicianConfidence, BeforeValidator(normalize_confidence)
    ] = Field(
        ...,
        description=(
            "Confidence in the diagnostic call based on the pathologist's "
            "language. 'definite' for unambiguous diagnoses, 'likely' for "
            "'consistent with' / 'compatible with', 'suspicious' for 'suggestive "
            "of' / 'cannot exclude'. If there is no cancer, that can still be a 'definite' call if the pathologist explicitly states no cancer is present, or an 'indeterminate' call if the pathologist explicitly says they cannot determine whether cancer is present."
        ),
    )


class PlanTriage(ExtractionPlan):
    """Quick orientation before triage. Note specimens + headline diagnosis so the
    triage_report flags and cancer_status are committed deliberately, not in passing.
    """

    specimen_overview: str = Field(
        ...,
        description=(
            "Brief enumeration of specimens in the report. E.g., "
            "'A: left radical nephrectomy, B: para-aortic LN biopsy', or "
            "'single CT-guided liver core biopsy'. Don't miss small specimens "
            "mentioned in passing — these drive the has_biopsy / "
            "has_metastasectomy / has_primary_nephrectomy flags."
        ),
    )
    headline_findings: str = Field(
        ...,
        description=(
            "1-2 sentence summary of the main diagnoses. Note any addendum that "
            "changes or refines the original diagnosis (addenda are authoritative). "
            "E.g., 'Clear cell RCC, ISUP 2, no LN involvement; addendum IHC "
            "consistent with RCC' or 'Liver core biopsy with carcinoma; addendum "
            "PAX-8 positive consistent with metastatic RCC'."
        ),
    )


class FlagReportForReview(ExtractionPlan):
    """Flag the report for human review. Use sparingly."""

    reason: str = Field(
        ...,
        description=(
            "Reason the report needs review. Use only when the report is "
            "genuinely ambiguous (conflicting findings, atypical wording, "
            "no clear specimen type). Routine 'uncertain' or 'not specified' "
            "cases do NOT warrant a flag."
        ),
    )


# =============================================================================
# System Prompt
# =============================================================================


SYSTEM_PROMPT = """\
You are doing EFFICIENT, COARSE triage of a kidney pathology report. The goal is to
decide which downstream workflow(s) the report should feed and to record a
high-level cancer status. You are NOT doing detailed extraction — that lives in
other workflows.

## WORKFLOW
1. Call `plan_triage` ONCE first — list the specimens you see and the headline
   diagnosis (including any addendum). This forces a deliberate read before triage.
2. Call `triage_report` ONCE with all the routing flags + cancer status.
3. Call `flag_report_for_review` only when the report is genuinely ambiguous.
4. Call `finish_note_extraction` to end.

## SPECIMEN TYPE FLAGS
A single report can contain multiple specimens — set ALL applicable flags:
- `has_primary_nephrectomy`: TRUE for any radical, partial, or total nephrectomy.
- `has_biopsy`: TRUE for any biopsy (core, FNA, incisional, etc.) of any tissue.
- `has_metastasectomy`: TRUE for resection of a metastatic deposit (e.g.,
  lung wedge, liver resection, bone resection, adrenalectomy for met).
  Direct contiguous tumor extension from a primary kidney tumor into a
  neighboring organ (adrenal, perinephric fat, IVC) is NOT metastasectomy —
  it is part of the primary nephrectomy.

## CANCER STATUS — pick ONE, prioritizing kidney-relevant findings
Use the most-aggressive/most-specific status the CURRENT report supports:
"Metastatic kidney cancer" -> Any RCC / kidney cancer outside the kidney in any specimen
"Kidney cancer with regional lymph node involvement" -> Any kidney cancer / RCC in a regional lymph node (e.g., renal hilar, para-aortic) but no distant metastasis
"Kidney cancer, localized to kidney" -> Any RCC / kidney cancer confined to the kidney, even if it has adverse features like perinephric fat invasion or adrenal invasion. Do NOT upgrade to metastatic based on direct contiguous extension into a neighboring organ.
"non-kidney cancer malignancy" -> Any malignancy that is definitively non-kidney in origin. This includes non-RCC malignancies in the kidney (e.g., urothelial carcinoma of the renal pelvis, lymphoma involving the kidney) and any malignancy outside the kidney that is definitively not a metastasis from a kidney primary.
"neoplasm of uncertain behavior of kidney" -> Any renal neoplasm that is not definitively benign or malignant based on the current report. This includes borderline/uncertain diagnoses like "low grade oncocytic tumor" or "oncocytic renal neoplasm, favor low-grade".
"uncertain neoplasm of non-kidney origin or uncertain malignancy" -> Any neoplasm outside the kidney that is not definitively benign or malignant based on the current report, OR any neoplasm of uncertain behavior where the pathologist does not specify whether it is of kidney origin.
"benign neoplasm of kidney" -> Any neoplasm in the kidney that is definitively benign (e.g., oncocytoma, typical angiomyolipoma).
"benign non-kidney neoplasm or other benign finding" -> Any benign finding that is either outside the kidney or of uncertain origin. This includes benign neoplasms outside the kidney (e.g., benign liver lesion) and any benign neoplasm where the pathologist does not specify whether it is of kidney origin.
"no cancer or non-diagnostic" -> No evidence of malignancy and no neoplasm of uncertain behavior. This includes definitively benign diagnoses, normal tissue, non-diagnostic specimens, and cases where the pathologist explicitly states that no cancer is present.
"uncertain — flag for review" -> Use only when the report is genuinely ambiguous and does not fit any of the above categories, even after considering addenda. Routine uncertainty or borderline cases should NOT be flagged for review — just pick the best-fitting category and set confidence to 'suspicious'.

## RCC SUBTYPE RECOGNITION
ALL of the following are kidney cancer subtypes, for reference

Common subtypes (most cases):
- Clear cell renal cell carcinoma (~70% of RCC; aggressiveness varies by grade)
- Papillary renal cell carcinoma (generally less aggressive than clear cell)
- Chromophobe renal cell carcinoma (generally indolent)

Aggressive / high-risk subtypes (poor prognosis):
- Collecting duct carcinoma
- SMARCB1-deficient renal medullary carcinoma (SMARCB1 is also called INI-1)
- FH-deficient renal cell carcinoma (HLRCC; hereditary leiomyomatosis-associated)
- Any RCC with sarcomatoid OR rhabdoid features
- Poorly differentiated carcinoma with features suggestive of RCC

Molecular / translocation subtypes (all RCC; aggressiveness varies):
- TFE3-rearranged RCC (formerly "Xp11 translocation RCC")
- TFEB-altered RCC (formerly "t(6;11) RCC")
- ELOC-mutated RCC (formerly TCEB1-mutated)
- ALK-rearranged RCC
- SDH-deficient renal carcinoma

Less common, generally indolent RCC subtypes:
- Mucinous tubular and spindle renal cell carcinoma
- Eosinophilic solid and cystic renal cell carcinoma
- Tubulocystic renal cell carcinoma
- Acquired cystic disease-associated renal cell carcinoma
- Clear cell papillary renal cell tumor
- Multilocular cystic clear cell RCC of low malignant potential

Unclassified / pending (still RCC):
- Renal cell carcinoma, NOS / unclassified
- Renal cell carcinoma, no subtype specified
- Renal cell carcinoma, subtype pending additional studies

Modernized name aliases:
- "Xp11 translocation RCC" → TFE3-rearranged RCC
- "Hereditary leiomyomatosis-associated RCC" / "HLRCC" → FH-deficient RCC
- "t(6;11) RCC" → TFEB-altered RCC
- "SMARCB1" is sometimes called "INI-1"

## BENIGN RENAL NEOPLASMS — pick `benign neoplasm of kidney`
- Renal oncocytoma → benign.
- Angiomyolipoma → benign UNLESS the report specifies epithelioid AML,
  malignant AML, or sarcoma (then treat as a malignancy).

## UNCERTAIN-BEHAVIOR RENAL NEOPLASMS — pick `neoplasm of uncertain behavior of kidney`
Treat any of the following as uncertain behavior unless the pathologist or an
addendum definitively reclassifies the lesion as benign oncocytoma or as a
specific malignant RCC subtype:
- "Low grade oncocytic tumor" / "LOT" / "Low-grade oncocytic renal tumor"
- "Renal oncocytic neoplasm" / "Oncocytic renal neoplasm"
- "Oncocytic neoplasm, favor low-grade"

## READING RULES
- Prefer the current report over prior medical history unless the pathologist
  explicitly relates it to the current specimen.
- Addenda (IHC, molecular, special stains) are authoritative — use the
  addendum's conclusion if it overrides the original diagnosis.
- "Consistent with" / "compatible with" RCC IS strong enough — call it RCC.
- "Suspicious for" / "favored" / "cannot exclude" / "prior history noted" alone
  is NOT strong enough — set diagnostic_confidence to 'suspicious' and consider
  calling out uncertainty in triage_summary.
- If unsure between RCC and not-RCC, lean toward calling RCC and set
  diagnostic_confidence to 'likely' or 'suspicious'.

## METASTATIC vs REGIONAL — the key clinical distinction
- Direct contiguous extension of a kidney tumor into a neighboring organ
  (adrenal invaded by primary, IVC thrombus, perinephric fat) → does NOT
  upgrade to metastatic RCC. Stays as 'RCC, confined to kidney'.
- Non-contiguous deposit in the same or different organ → metastatic RCC.
- RCC in regional renal lymph node ONLY → 'RCC with regional lymph node
  involvement', NOT metastatic.
- RCC in distant or non-regional lymph node → metastatic RCC.

## CONFIDENCE
- `definite` — pathologist states the diagnosis unambiguously.
- `likely` — "consistent with" / "compatible with" wording.
- `suspicious` — "suggestive of" / "cannot exclude" / "favored" wording.
- `indeterminate` — pathologist explicitly says cannot determine.


## FLAGGING
Use `flag_report_for_review` only when:
- Conflicting information between sections that addenda do not resolve.
- Report wording does not fit any kidney_cancer_status value.
- Specimen type is genuinely unclear (extremely rare).
Do NOT flag for routine 'uncertain' or 'not specified' calls.
"""


# =============================================================================
# Registry Factory
# =============================================================================


def create_path_kidney_basic_registry() -> ToolRegistry:
    """Create the triage tool registry. Three tools: plan_triage (orient), then
    triage_report (always called once per report), and flag_report_for_review
    (used sparingly)."""
    registry = ToolRegistry(single_note=True)

    registry.register(
        name="plan_triage",
        description=(
            "Call ONCE per report, BEFORE triage_report. Brief orientation: "
            "list the specimens and the headline diagnosis (including any "
            "addendum override). Forces a deliberate read so the triage flags "
            "and cancer_status are committed thoughtfully, not in passing."
        ),
        model=PlanTriage,
    )

    registry.register(
        name="triage_report",
        description=(
            "Call ONCE per report, AFTER plan_triage. Records the report-level "
            "routing flags (has_primary_nephrectomy / has_biopsy / "
            "has_metastasectomy) and a coarse kidney_cancer_status. This is the "
            "only data tool — no per-specimen detail. Detailed extraction lives "
            "in other workflows."
        ),
        model=TriageReport,
    )

    registry.register(
        name="flag_report_for_review",
        description=(
            "Flag the report for human review. Use sparingly — only for "
            "genuinely ambiguous reports where the available kidney_cancer_status "
            "values do not fit. Routine uncertainty does NOT warrant a flag."
        ),
        model=FlagReportForReview,
    )

    return registry
