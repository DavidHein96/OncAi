"""
Kidney nephrectomy pathology extraction definition.

Dedicated workflow for PRIMARY NEPHRECTOMY pathology reports (radical, partial,
or total nephrectomy). Secondary specimens in the same report (lymph node
biopsies, metastasectomy resections, etc.) are NOT extracted here — they belong
to other workflows.

Most tools are called once per report. The only per-tumor multiplicity is for
distinct synchronous primary tumors (rare; typically hereditary RCC syndromes).

Used with batch_single.py for single-note processing.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field

from ..enum_helpers import build_enum_lookup, normalize_against
from ..models import ExtractionEvent, ExtractionPlan
from ..tools import ToolRegistry

logger = logging.getLogger(__name__)

DEFINITION_NAME = "PathKidneyNephrectomy"

# =============================================================================
# Enums
# =============================================================================


class NephrectomyType(StrEnum):
    RADICAL = "radical"
    PARTIAL = "partial"
    TOTAL_SIMPLE = "total (simple)"
    OTHER = "other"
    NOT_SPECIFIED = "not specified"


class Laterality(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    NOT_SPECIFIED = "not specified"


class Focality(StrEnum):
    UNIFOCAL = "unifocal"
    MULTIFOCAL = "multifocal"
    NOT_SPECIFIED = "not specified"


class MarginStatus(StrEnum):
    """Surgical margin status for the nephrectomy specimen, summarized across
    all relevant margins (renal parenchymal for partial; renal vein, ureter,
    perinephric soft tissue / Gerota's fascia for radical/total)."""

    ALL_NEGATIVE = "all margins negative for invasive carcinoma"
    INVASIVE_AT_MARGIN = "invasive carcinoma present at margin"
    CANNOT_BE_DETERMINED = "cannot be determined"
    NOT_APPLICABLE = "not applicable"


class TumorSite(StrEnum):
    UPPER_POLE = "upper pole"
    MIDDLE = "middle"
    LOWER_POLE = "lower pole"
    OTHER = "other"
    NOT_SPECIFIED = "not specified"


class TumorIdentifier(StrEnum):
    """Stable per-tumor identifier. Single-tumor reports (the common case) use
    'tumor_1'. Multi-tumor cases use 'tumor_2', 'tumor_3', ... in the order
    tumors appear."""

    TUMOR_1 = "tumor_1"
    TUMOR_2 = "tumor_2"
    TUMOR_3 = "tumor_3"
    TUMOR_4 = "tumor_4"
    TUMOR_5 = "tumor_5"


class Grade(StrEnum):
    GRADE_1 = "grade 1"
    GRADE_2 = "grade 2"
    GRADE_3 = "grade 3"
    GRADE_4 = "grade 4"
    HIGH_GRADE = "high grade"
    LOW_GRADE = "low grade"
    CANNOT_BE_ASSESSED = "cannot be assessed"
    NOT_APPLICABLE = "not applicable"
    NOT_SPECIFIED = "not specified"


class HistologicFeatureStatus(StrEnum):
    """Presence status for histologic features (sarcomatoid, rhabdoid, necrosis, LVI)."""

    PRESENT = "present"
    NO_EVIDENCE = "no evidence"
    CANNOT_BE_DETERMINED = "cannot be determined"
    NOT_SPECIFIED = "not specified"


class HistologicType(StrEnum):
    CLEAR_CELL_RCC = "Clear cell renal cell carcinoma"
    MULTILOCULAR_CYSTIC = (
        "Multilocular cystic renal neoplasm of low malignant potential"
    )
    PAPILLARY_RCC = "Papillary renal cell carcinoma"
    CHROMOPHOBE_RCC = "Chromophobe renal cell carcinoma"
    ONCOCYTIC_OTHER = "Other oncocytic tumors of the kidney"
    ONCOCYTIC_LOW_GRADE = (
        "Other oncocytic tumors of the kidney, low grade oncocytic tumor"
    )
    COLLECTING_DUCT = "Collecting duct carcinoma"
    SMARCB1_DEFICIENT_MEDULLARY = "SMARCB1-deficient renal medullary carcinoma"
    EOSINOPHILIC_SOLID_CYSTIC = "Eosinophilic solid and cystic renal cell carcinoma"
    TFE3_REARRANGED = "TFE3-rearranged renal cell carcinoma"
    TFEB_ALTERED = "TFEB-altered renal cell carcinoma"
    ELOC_MUTATED = "ELOC (formerly TCEB1)-mutated renal cell carcinoma"
    MUCINOUS_TUBULAR_SPINDLE = "Mucinous tubular and spindle renal cell carcinoma"
    TUBULOCYSTIC = "Tubulocystic renal cell carcinoma"
    ACD_ASSOCIATED = "Acquired cystic disease-associated renal cell carcinoma"
    CLEAR_CELL_PAPILLARY = "Clear cell papillary renal cell tumor"
    SDH_DEFICIENT = "Succinate dehydrogenase-deficient (SDH) renal cell carcinoma"
    FH_DEFICIENT = "Fumarate hydratase-deficient renal cell carcinoma"
    ALK_REARRANGED = "ALK-rearranged renal cell carcinoma"
    SUBTYPE_PENDING = "Renal cell carcinoma, subtype pending additional studies"
    UNCLASSIFIED_RCC = "Renal cell carcinoma, NOS (unclassified)"
    NO_SUBTYPE_SPECIFIED = "Renal cell carcinoma, no subtype specified"
    ONCOCYTOMA = "Renal oncocytoma"
    ANGIOMYOLIPOMA = "Angiomyolipoma"
    UROTHELIAL = "Urothelial carcinoma"
    POORLY_DIFFERENTIATED = "Poorly differentiated carcinoma"
    UNCERTAIN_PRIMARY = "Carcinoma, uncertain primary origin"
    NECROTIC_NONVIABLE = "Necrotic/nonviable tumor"
    BENIGN = "Benign tissue, negative for malignancy"
    NO_TISSUE_PRESENT = "No tissue present"
    NON_KIDNEY_MALIGNANCY = "Malignancy, not consistent with primary kidney tumor"
    ATYPICAL_CELLS = "Atypical cells"
    OTHER = "Other, specified"
    NOT_SPECIFIED = "not specified"


class TumorExtent(StrEnum):
    LIMITED_TO_KIDNEY = "limited to kidney"
    PERINEPHRIC_EXTENSION = "extends into perinephric tissue (beyond renal capsule)"
    RENAL_SINUS_FAT_EXTENSION = "extends into renal sinus fat"
    RENAL_SINUS_EXTENSION = "extends into renal sinus"
    PELVICALYCEAL_EXTENSION = "extends into pelvicalyceal system"
    RENAL_VEIN_EXTENSION = "extends into renal vein or its segmental branches"
    VENA_CAVA_EXTENSION = "extends into inferior vena cava"
    BEYOND_GEROTA = "extends beyond Gerota's fascia (renal fascia)"
    DIRECT_ADRENAL_INVASION = "directly invades adrenal gland (contiguous)"
    NON_CONTIGUOUS_ADRENAL_INVOLVEMENT = "involves adrenal gland non-contiguously"
    OTHER_ORGAN_INVOLVEMENT = "extends into other organ(s) / structure(s)"
    CANNOT_DETERMINE = "cannot be determined"


class PathTStage(StrEnum):
    PT = "pT"  # pT not assigned (cannot be determined based on available pathological information)
    PT0 = "pT0"  # no evidence of primary tumor
    PT1A = "pT1a"  # pT1a: Tumor less than or equal to 4 cm in greatest dimension, limited to the kidney
    PT1B = "pT1b"  # pT1b: Tumor greater than 4 cm but less than or equal to 7 cm in greatest dimension limited to the kidney
    PT1 = "pT1"  # Tumor less than or equal to 7 cm in greatest dimension, limited to the kidney (subcategory cannot be determined)
    PT2A = "pT2a"  # pT2a: Tumor greater than 7 cm but less than or equal to 10 cm in greatest dimension, limited to the kidney
    PT2B = "pT2b"  # Tumor greater than 10 cm, limited to the kidney
    PT2 = "pT2"  # Tumor greater than 7 cm in greatest dimension, limited to the kidney (subcategory cannot be determined)
    PT3A = "pT3a"  # Tumor extends into the renal vein or its segmental branches, or invades the pelvicalyceal system, or invades perirenal and / or renal sinus fat but not beyond Gerota's fascia
    PT3B = "pT3b"  # Tumor extends into the vena cava below the diaphragm
    PT3C = "pT3c"  # Tumor extends into the vena cava above the diaphragm or invades the wall of the vena cava
    PT3 = "pT3"  # Tumor extends into major veins or perinephric tissues, but not into the ipsilateral adrenal gland and not beyond Gerota's fascia (subcategory cannot be determined)
    PT4 = "pT4"  # Tumor invades beyond Gerota's fascia (including contiguous extension into the ipsilateral adrenal gland)


class PathNStage(StrEnum):
    PN0 = "pN0"  # No regional lymph node metastasis
    PN1 = "pN1"  # Metastasis in regional lymph node(s)
    PNX_NOT_SPECIFIED = "pNX (no nodes submitted or found)"
    PNX_CANT_DETERMINE = (
        "pNX (cannot be determined based on available pathological information)"
    )


class PathMStage(StrEnum):
    PMX_CANT_DETERMINE = "pMX (cannot be determined based on available pathological information)"  # cannot be determined from the submitted specimen(s)
    PM1 = (
        "pM1"  # Distant metastasis (including non-contiguous adrenal gland involvement)
    )
    PM0 = "pM0"  # No distant metastasis


_NEPHRECTOMY_TYPE_LOOKUP = build_enum_lookup(NephrectomyType)
_LATERALITY_LOOKUP = build_enum_lookup(Laterality)
_FOCALITY_LOOKUP = build_enum_lookup(Focality)
_MARGIN_STATUS_LOOKUP = build_enum_lookup(MarginStatus)
_TUMOR_SITE_LOOKUP = build_enum_lookup(TumorSite)
_TUMOR_IDENTIFIER_LOOKUP = build_enum_lookup(TumorIdentifier)
_GRADE_LOOKUP = build_enum_lookup(Grade)
_FEATURE_STATUS_LOOKUP = build_enum_lookup(HistologicFeatureStatus)
_HISTOLOGIC_TYPE_LOOKUP = build_enum_lookup(HistologicType)
_TUMOR_EXTENT_LOOKUP = build_enum_lookup(TumorExtent)
_PATH_T_STAGE_LOOKUP = build_enum_lookup(PathTStage)
_PATH_N_STAGE_LOOKUP = build_enum_lookup(PathNStage)
_PATH_M_STAGE_LOOKUP = build_enum_lookup(PathMStage)


def normalize_nephrectomy_type(v: object) -> str:
    return normalize_against(
        v, _NEPHRECTOMY_TYPE_LOOKUP, "nephrectomy_type", logger=logger
    )


def normalize_laterality(v: object) -> str:
    return normalize_against(v, _LATERALITY_LOOKUP, "laterality", logger=logger)


def normalize_focality(v: object) -> str:
    return normalize_against(v, _FOCALITY_LOOKUP, "focality", logger=logger)


def normalize_margin_status(v: object) -> str:
    return normalize_against(v, _MARGIN_STATUS_LOOKUP, "margin_status", logger=logger)


def normalize_tumor_site(v: object) -> str:
    return normalize_against(v, _TUMOR_SITE_LOOKUP, "tumor_site", logger=logger)


def normalize_tumor_identifier(v: object) -> str:
    return normalize_against(
        v, _TUMOR_IDENTIFIER_LOOKUP, "tumor_identifier", logger=logger
    )


def normalize_grade(v: object) -> str:
    return normalize_against(v, _GRADE_LOOKUP, "grade", logger=logger)


def normalize_feature_status(v: object) -> str:
    return normalize_against(v, _FEATURE_STATUS_LOOKUP, "feature_status", logger=logger)


def normalize_histologic_type(v: object) -> str:
    return normalize_against(
        v, _HISTOLOGIC_TYPE_LOOKUP, "histologic_type", logger=logger
    )


def normalize_path_t_stage(v: object) -> str:
    return normalize_against(v, _PATH_T_STAGE_LOOKUP, "path_t_stage", logger=logger)


def normalize_path_n_stage(v: object) -> str:
    return normalize_against(v, _PATH_N_STAGE_LOOKUP, "path_n_stage", logger=logger)


def normalize_path_m_stage(v: object) -> str:
    return normalize_against(v, _PATH_M_STAGE_LOOKUP, "path_m_stage", logger=logger)


def normalize_tumor_extent_list(v: object) -> list[str]:
    if not isinstance(v, list):
        return v  # type: ignore[return-value]  # let pydantic raise the type error
    return [
        normalize_against(item, _TUMOR_EXTENT_LOOKUP, "tumor_extent", logger=logger)
        for item in v
    ]


# =============================================================================
# Tool models
# =============================================================================


class PlanNephrectomyExtraction(ExtractionPlan):
    """Plan the extraction. Confirm the primary nephrectomy specimen and
    enumerate the tumor(s) before any record_* calls.
    """

    specimen_overview: str = Field(
        ...,
        description=(
            "Identify the PRIMARY nephrectomy specimen (radical / partial / total) "
            "and note any secondary specimens. If no primary nephrectomy is present, "
            "state that here and plan to call flag_report_for_review and stop. If "
            "secondary specimens (lymph node biopsy, metastasectomy, etc.) coexist "
            "with the primary nephrectomy, note them but plan to extract ONLY the "
            "primary nephrectomy findings."
        ),
    )
    tumor_inventory: str = Field(
        ...,
        description=(
            "List each distinct tumor in the primary nephrectomy specimen. The vast "
            "majority of cases have one tumor — assign 'tumor_1'. For the rare "
            "multi-tumor case (synchronous distinct primaries, often hereditary RCC), "
            "assign 'tumor_1', 'tumor_2', etc. in the order they appear in the "
            "report and briefly describe each (e.g., 'tumor_1: 4.5 cm upper pole "
            "clear cell RCC; tumor_2: 1.2 cm lower pole oncocytoma'). A single tumor "
            "with multiple foci of extension is still ONE tumor."
        ),
    )


class RecordNephrectomySpecimen(ExtractionEvent):
    """Record specimen-level nephrectomy context."""

    nephrectomy_type: Annotated[
        NephrectomyType, BeforeValidator(normalize_nephrectomy_type)
    ] = Field(
        NephrectomyType.NOT_SPECIFIED,
        description="Type of nephrectomy performed.",
    )
    laterality: Annotated[Laterality, BeforeValidator(normalize_laterality)] = Field(
        Laterality.NOT_SPECIFIED,
        description=(
            "Laterality of the nephrectomy specimen. Use 'not specified' if the "
            "report does not state laterality; do NOT infer from clinical history."
        ),
    )
    focality: Annotated[Focality, BeforeValidator(normalize_focality)] = Field(
        Focality.NOT_SPECIFIED,
        description=(
            "Whether the kidney has a single tumor (unifocal) or multiple distinct "
            "tumors (multifocal). This is a specimen-level finding — record once for "
            "the whole specimen, not per tumor."
        ),
    )
    margin_status: Annotated[MarginStatus, BeforeValidator(normalize_margin_status)] = (
        Field(
            MarginStatus.CANNOT_BE_DETERMINED,
            description=(
                "Summary surgical margin status across all relevant margins for this "
                "specimen (renal parenchymal margin for partial nephrectomy; renal "
                "vein, ureter, and perinephric soft tissue / Gerota's fascia margins "
                "for radical / total). Use 'all margins negative for invasive "
                "carcinoma' only when EVERY reported margin is negative / uninvolved / "
                "free of tumor. Use 'invasive carcinoma present at margin' if ANY "
                "margin is reported positive / involved by invasive tumor — a positive "
                "renal vein MARGIN counts here (this is distinct from renal vein "
                "EXTENSION, which is recorded in tumor_extent). Use 'cannot be "
                "determined' when the report does not state margin status or "
                "explicitly says margins cannot be assessed. Use 'not applicable' "
                "only for unusual specimens where margins are inherently not "
                "assessable."
            ),
        )
    )


class RecordKidneyTumor(ExtractionEvent):
    """Record the tumor identity, morphology, grade, size, adverse histologic
    features, and anatomic extent for one tumor. Call once per distinct tumor
    (almost always exactly once).
    """

    tumor_identifier: Annotated[
        TumorIdentifier, BeforeValidator(normalize_tumor_identifier)
    ] = Field(
        TumorIdentifier.TUMOR_1,
        description=(
            "Stable tumor ID. Use 'tumor_1' for the single tumor in the common "
            "single-tumor case. For multi-tumor cases, use the IDs you committed "
            "to in plan_nephrectomy_extraction."
        ),
    )
    tumor_site: Annotated[TumorSite, BeforeValidator(normalize_tumor_site)] = Field(
        TumorSite.NOT_SPECIFIED,
        description="Location of the tumor within the kidney.",
    )
    histologic_type: Annotated[
        HistologicType, BeforeValidator(normalize_histologic_type)
    ] = Field(
        ...,
        description=(
            "Histologic classification. Use the most specific subtype the report "
            "supports. Do not assume clear cell RCC unless stated. If the report "
            "says only 'renal cell carcinoma' with no subtype, use "
            "'Renal cell carcinoma, no subtype specified'."
        ),
    )
    tumor_grade: Annotated[Grade, BeforeValidator(normalize_grade)] = Field(
        Grade.NOT_SPECIFIED,
        description=(
            "WHO/ISUP nucleolar grade is validated only for clear cell and "
            "papillary RCC. Use 'grade 1'-'grade 4' when a numeric grade is "
            "reported. Use 'high grade' or 'low grade' only when the report uses "
            "those terms without a numeric grade. Use 'cannot be assessed' only when the report explicitly states grade cannot be assessed."
        ),
    )
    tumor_size_cm: float | None = Field(
        None,
        description=(
            "Greatest tumor dimension in centimeters. For dimensions like "
            "'4.2 x 3.1 x 2.8 cm', use 4.2. Convert mm to cm if needed. Do NOT use "
            "the kidney specimen size. Leave null if not reported."
        ),
    )
    sarcomatoid_features: Annotated[
        HistologicFeatureStatus, BeforeValidator(normalize_feature_status)
    ] = Field(
        HistologicFeatureStatus.NOT_SPECIFIED,
        description=(
            "Sarcomatoid differentiation. Do NOT infer from 'high grade' alone."
        ),
    )
    rhabdoid_features: Annotated[
        HistologicFeatureStatus, BeforeValidator(normalize_feature_status)
    ] = Field(
        HistologicFeatureStatus.NOT_SPECIFIED,
        description="Rhabdoid differentiation. Do NOT infer from 'high grade' alone.",
    )
    tumor_necrosis: Annotated[
        HistologicFeatureStatus, BeforeValidator(normalize_feature_status)
    ] = Field(
        HistologicFeatureStatus.NOT_SPECIFIED,
        description=(
            "Tumor necrosis (coagulative necrosis). Do NOT count non-tumor "
            "infarction, hemorrhage, or gross degenerative change as tumor necrosis "
            "unless the report identifies it as such."
        ),
    )
    lymphovascular_invasion: Annotated[
        HistologicFeatureStatus, BeforeValidator(normalize_feature_status)
    ] = Field(
        HistologicFeatureStatus.NOT_SPECIFIED,
        description=(
            "Lymphatic and small-vessel vascular invasion ONLY. Renal vein, "
            "segmental renal vein branch, and inferior vena cava involvement are "
            "TUMOR EXTENT findings, not LVI — record those in the tumor_extent field."
        ),
    )
    tumor_extent: Annotated[
        list[TumorExtent], BeforeValidator(normalize_tumor_extent_list)
    ] = Field(
        default_factory=list,
        description=(
            "All anatomic structures involved by this tumor. Always include at "
            "least one entry: use ['limited to kidney'] for an organ-confined "
            "tumor, list every structure that applies for tumors with extension, "
            "and use ['cannot be determined'] only when the report explicitly "
            "states extent cannot be assessed. Renal vein, segmental branch, and "
            "IVC involvement go here (NOT in lymphovascular_invasion). "
            "'Limited to kidney' should not be combined with any extension entry."
        ),
    )


class RecordNephrectomyStaging(ExtractionEvent):
    """Record AJCC pathologic TNM staging for the case. Called ONCE per report.
    TNM is a single per-case stage. If multiple synchronous primary tumors are
    present, base the T category on the highest-stage tumor and set
    t_note_multiple accordingly; do not record a separate TNM per tumor."""

    path_t_stage: Annotated[PathTStage, BeforeValidator(normalize_path_t_stage)] = (
        Field(
            PathTStage.PT,
            description=(
                "AJCC pathologic T stage as reported (e.g., 'pT1a', 'pT3a'). Use the "
                "pathologist's stated stage when available. Use 'pT' when the stage "
                "cannot be assessed"
            ),
        )
    )
    t_note_multiple: Literal[
        "not applicable", "multiple primary synchronous tumors in a single organ"
    ] = Field(
        "not applicable",
        description=(
            "AJCC T-category note for multifocality. Use 'multiple primary "
            "synchronous tumors in a single organ' ONLY when the specimen "
            "contains two or more DISTINCT synchronous primary tumors in the "
            "kidney (e.g., hereditary RCC syndromes such as VHL or BHD; "
            "synchronous independent histologies). A single tumor with multiple "
            "foci of extension or satellite nodules does NOT qualify — that is "
            "still one primary tumor. Otherwise use 'not applicable'."
        ),
    )
    path_n_stage: Annotated[PathNStage, BeforeValidator(normalize_path_n_stage)] = (
        Field(
            PathNStage.PNX_CANT_DETERMINE,
            description=(
                "AJCC pathologic N stage. Use 'pN0' when regional lymph nodes were "
                "examined and ALL are negative ('0/3 nodes involved' → pN0). Use "
                "'pN1' when tumor is present in one or more regional lymph nodes "
                "('1/5 nodes positive' → pN1). Use 'pNX (no nodes submitted or "
                "found)' when the report explicitly states no regional lymph nodes "
                "were submitted or identified ('no lymph nodes identified', 'no "
                "nodes submitted'). Use 'pNX (cannot be determined…)' when the "
                "report does not address regional nodes at all or explicitly says "
                "they cannot be assessed."
            ),
        )
    )
    path_m_stage: Annotated[PathMStage, BeforeValidator(normalize_path_m_stage)] = (
        Field(
            PathMStage.PMX_CANT_DETERMINE,
            description=(
                "Pathologic M stage as reported. Use 'pM1' ONLY when the current "
                "pathology report pathologically confirms distant metastasis — never "
                "from imaging or clinical history alone."
            ),
        )
    )


class FlagReportForReview(ExtractionEvent):
    """Flag the report for human review. Use sparingly."""

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
            "Reason the report needs review. Examples: 'no primary nephrectomy "
            "specimen present', 'two distinct nephrectomy specimens in one report', "
            "'conflicting information between original diagnosis and addendum', "
            "'wording does not match any available enum value'. Routine "
            "'not specified' cases do NOT warrant a flag."
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
You are extracting structured findings from a PRIMARY KIDNEY NEPHRECTOMY pathology
report using function calls.

## SCOPE
This workflow is dedicated to a primary nephrectomy specimen (radical, partial,
or total nephrectomy).
- If the report has NO primary nephrectomy specimen (e.g., it is only a biopsy or
  metastasectomy), call `flag_report_for_review` with the reason and STOP.
- If the report has a primary nephrectomy plus secondary specimens (lymph node
  biopsy, metastasectomy, etc.), extract ONLY the primary nephrectomy findings.
  The secondary specimens are handled by other workflows.
- If two distinct nephrectomy specimens are present in one report (extremely rare),
  call `flag_report_for_review` and proceed with the primary one.

## WORKFLOW
1. Call `plan_nephrectomy_extraction` ONCE to confirm the primary specimen and
   enumerate the tumor(s).
2. Call each `record_*` tool ONCE in this order:
   record_nephrectomy_specimen → record_kidney_tumor → record_nephrectomy_staging.
3. For multi-tumor cases (rare), call `record_kidney_tumor` ONCE PER tumor in
   the order tumor_1, tumor_2, ... Use the IDs you committed to in
   plan_nephrectomy_extraction.
4. Call `flag_report_for_review` only when something is genuinely off.
5. Call `finish_note_extraction` after all relevant tools have been called.

## TUMOR NUMBERING
- The vast majority of nephrectomy reports describe a single tumor — use `tumor_1`.
- Only use `tumor_2`, `tumor_3`, etc. for distinct synchronous primary tumors
  (typically hereditary RCC syndromes like VHL or BHD).
- A single tumor with multiple foci of extension or multiple components is still
  ONE tumor, recorded under one tumor_id.

## READING RULES
- Use ONLY findings from the CURRENT pathology report (final diagnosis, synoptic,
  gross, microscopic, comments, addenda). Ignore prior medical history unless the
  pathologist explicitly relates it to the current specimen.
- Addenda are authoritative: if an addendum (typically IHC, molecular, special
  stains, or cytogenetics) clarifies or overrides the original diagnosis, use the
  addendum's conclusion.
- For histologic-feature status fields, use:
  - `present` when the feature is identified, present, positive, or seen.
  - `no evidence` when the feature is not identified, absent, negative, or not seen.
  - `cannot be determined` only when the report explicitly states it cannot be assessed.
  - `not specified` when the report does not mention the feature.

## EVIDENCE STRENGTH
- "Consistent with" or "compatible with" a histology IS strong enough to assign
  that subtype.
- "Suspicious for", "favored", "cannot exclude", "prior history noted" alone is
  NOT strong enough — fall back to a less-specific enum (e.g., POORLY_DIFFERENTIATED,
  UNCLASSIFIED_RCC) or call `flag_report_for_review`.

## NEPHRECTOMY TYPE GUIDANCE
- Radical is the entire kidney plus variable lengths of ureter, perirenal fat to the Gerota's facia, and variable lengths of major renal vessels.
- Total nephrectomy is similar but performed for presumption of benign disease and may not extend to the Gerotas fascia.
- Partial nephrectomy: Surgical removal of a kidney tumor with a variable margin of surrounding kidney tissue, often including nearby fat and occasionally part of the collecting system, while preserving the rest of the kidney.

## AJCC 8th TNM (PRIMARY KIDNEY)
 - "pT": pT not assigned (cannot be determined based on available pathological information)
 - "pT0": no evidence of primary tumor
 - "pT1a": Tumor less than or equal to 4 cm in greatest dimension, limited to the kidney
 - "pT1b": Tumor greater than 4 cm but less than or equal to 7 cm in greatest dimension limited to the kidney
 - "pT1": Tumor less than or equal to 7 cm in greatest dimension, limited to the kidney (subcategory cannot be determined)
 - "pT2a": Tumor greater than 7 cm but less than or equal to 10 cm in greatest dimension, limited to the kidney
 - "pT2b": Tumor greater than 10 cm, limited to the kidney
 - "pT2": Tumor greater than 7 cm in greatest dimension, limited to the kidney (subcategory cannot be determined)
 - "pT3a": Tumor extends into the renal vein or its segmental branches, or invades the pelvicalyceal system, or invades perirenal and / or renal sinus fat but not beyond Gerota's fascia
 - "pT3b": Tumor extends into the vena cava below the diaphragm
 - "pT3c": Tumor extends into the vena cava above the diaphragm or invades the wall of the vena cava
 - "pT3": Tumor extends into major veins or perinephric tissues, but not into the ipsilateral adrenal gland and not beyond Gerota's fascia (subcategory cannot be determined)
 - "pT4": Tumor invades beyond Gerota's fascia (including contiguous extension into the ipsilateral adrenal gland)

ADRENAL RULE:
- Direct contiguous adrenal invasion from the kidney tumor → pT4.
- Non-contiguous adrenal involvement (separate deposit) → pM1, NOT pT4.

GRADE:
- G1 -> nucleoli absent or inconspicuous at 400x magnification
- G2 -> nucleoli conspicuous and visible at 400x magnification, not prominent at 100x magnification
- G3 -> nucleoli conspicuous at 100x magnification
- G4 -> extreme nuclear pleomorphism and/or multinucleated giant cells and/or rhabdoid and/or sarcomatoid differentiation
If the report just says high grade or low grade without a numeric grade, use the corresponding high_grade or low_grade enum value.
The 1-4 grading is typically used for clear cell and papillary RCC, so other subtypes will be more likely to use high_grade / low_grade or not_applicable.

LYMPHOVASCULAR vs RENAL VEIN:
- Renal vein, segmental renal vein branch, and IVC involvement are TUMOR EXTENT
  findings, NOT lymphovascular invasion. The `lymphovascular_invasion` field is
  for lymphatic and small-vessel invasion only.

## REGIONAL vs DISTANT LYMPH NODES
- REGIONAL renal lymph nodes (drive pN, NOT pM): renal hilar, para-aortic /
  periaortic, preaortic, retroaortic, lateral aortic / lumbar, caval, paracaval,
  precaval, retrocaval, pericaval, interaortocaval, retroperitoneal regional NOS.
- DISTANT (non-regional) lymph nodes (drive pM, NOT pN): mediastinal, thoracic
  hilar, cervical, supraclavicular, axillary, iliac, pelvic, inguinal, mesenteric,
  porta hepatis, peripancreatic, gastrohepatic, celiac, and any other not on the
  regional list.

## pN GUIDANCE
- pN0: regional lymph nodes examined and ALL negative.
- pN1: tumor present in one or more regional lymph nodes.
- pNX (no nodes submitted or found): report explicitly states no regional lymph
  nodes were submitted or identified ("no lymph nodes identified", "no nodes
  submitted").
- pNX (cannot be determined…): report does not address regional nodes at all,
  or explicitly says they cannot be assessed.
- "0/3 lymph nodes involved" → pN0; 3 examined; 0 positive.
- "1/5 lymph nodes positive" → pN1; 5 examined; 1 positive.

## pM GUIDANCE
- Use pM1 ONLY when the current pathology report pathologically confirms distant
  metastasis.
- Do NOT assign pM1 from imaging or clinical history alone.
- A distant-site biopsy negative for carcinoma is NOT pM1.

## MODERNIZED RCC SUBTYPE NAMES
- "Xp11 translocation RCC" → TFE3-rearranged renal cell carcinoma
- "Hereditary leiomyomatosis-associated RCC" / HLRCC → Fumarate hydratase-deficient
  renal cell carcinoma
- "t(6;11) RCC" / "MiTF/TFEB family translocation RCC" / "TFEB-rearranged RCC" →
  TFEB-altered renal cell carcinoma
- "Renal medullary carcinoma" / RMC (classically in patients with sickle cell
  trait or disease; loss of SMARCB1 / INI-1 by IHC) → SMARCB1-deficient renal
  medullary carcinoma
- "Clear cell papillary renal cell carcinoma" / "clear cell papillary RCC" /
  "clear cell tubulopapillary RCC" → Clear cell papillary renal cell tumor
  (WHO 2022 reclassified this indolent entity from a carcinoma to a "tumor";
  the older "carcinoma" wording in a report maps to the same value).
- "Papillary RCC, type 1" / "type 2" / "type 1 and 2" → Papillary renal cell
  carcinoma (WHO 2022 no longer subdivides papillary RCC into types 1 and 2;
  drop the type designation and assign the single Papillary renal cell carcinoma value).
- 'Renal cell carcinoma, NOS (unclassified)' ->  used when classification to a subtype is difficult due to complex or borderline histological features
- 'Renal cell carcinoma, no subtype specified' -> used when the pathologist simply does not provide a subtype and only refers to the histology as renal cell carcinoma.

## BENIGN AND UNCERTAIN RENAL TUMOR HANDLING
- Renal oncocytoma → ONCOCYTOMA (benign).
- Angiomyolipoma → ANGIOMYOLIPOMA (benign), unless the report specifies
  epithelioid AML, malignant AML, or sarcoma.
- "Low grade oncocytic tumor", "LOT", "renal oncocytic neoplasm", "oncocytic
  neoplasm, favor low-grade" → ONCOCYTIC_LOW_GRADE (uncertain behavior).

## COMMON PHRASE NORMALIZATION
- "Limited to kidney" → tumor extent limited to kidney.
- "Invades renal sinus fat" / "perinephric fat" → corresponding extension; usually pT3a.
- "Tumor thrombus in renal vein" → renal vein extension; usually pT3a.
- "Vena cava thrombus below diaphragm" → vena cava extension; usually pT3b.
- "Vena cava thrombus above diaphragm" → vena cava extension; usually pT3c.
- "Directly invades adrenal gland" → direct adrenal invasion; usually pT4.
- "Extends beyond Gerota's fascia" → beyond Gerota's fascia; usually pT4.

## PROVENANCE
- Use `evidence` for exact source-note snippets supporting the extraction.
- Use `review_anchor` only for report-level snippets explaining a human review flag.

## FLAGGING
Use `flag_report_for_review` for:
- No primary nephrectomy specimen present (and STOP after flagging).
- Two distinct nephrectomy specimens in one report.
- Genuinely conflicting information that the available enums cannot resolve.
- Atypical wording where no listed enum value fits.
When flagging, populate review_anchor with exact report text snippets
the review app can highlight and jump to.
Routine "not specified" cases do NOT warrant a flag.
"""


# =============================================================================
# Registry Factory
# =============================================================================


def create_path_kidney_nephrectomy_registry() -> ToolRegistry:
    """Create tool registry for primary kidney nephrectomy pathology extraction.

    All tools are registered together. The plan tool runs first to commit to
    tumor IDs, then each record_* tool is called once per report (with the
    per-tumor tools called once per distinct tumor in rare multi-tumor cases).
    """
    registry = ToolRegistry(single_note=True)

    registry.register(
        name="plan_nephrectomy_extraction",
        description=(
            "Call ONCE per report, BEFORE any record_* tool. Confirm the primary "
            "nephrectomy specimen and enumerate any tumor(s) so per-tumor IDs "
            "(tumor_1, tumor_2, ...) are committed up front. If no primary "
            "nephrectomy specimen exists, plan to call flag_report_for_review and "
            "stop."
        ),
        model=PlanNephrectomyExtraction,
    )

    registry.register(
        name="record_nephrectomy_specimen",
        description=(
            "Call ONCE per report. Records specimen-level findings for the primary "
            "nephrectomy: procedure type, laterality, focality, and summary surgical "
            "margin status. Multifocality and margins are specimen-level — they "
            "apply to the whole specimen, not per tumor."
        ),
        model=RecordNephrectomySpecimen,
        comparison_fields=(
            "nephrectomy_type",
            "laterality",
            "focality",
            "margin_status",
        ),
    )

    registry.register(
        name="record_kidney_tumor",
        description=(
            "Call ONCE PER TUMOR (almost always exactly once — multi-tumor cases "
            "are rare). Records tumor identity, morphology, grade, size, "
            "adverse histologic features (sarcomatoid, rhabdoid, necrosis, LVI), "
            "and anatomic tumor extent. Renal vein / IVC involvement does NOT go "
            "in lymphovascular_invasion — it goes in the tumor_extent field of "
            "this same tool."
        ),
        model=RecordKidneyTumor,
        event_identity_fields=("tumor_identifier",),
        comparison_fields=(
            "tumor_site",
            "histologic_type",
            "tumor_grade",
            "tumor_size_cm",
            "sarcomatoid_features",
            "rhabdoid_features",
            "tumor_necrosis",
            "lymphovascular_invasion",
            "tumor_extent",
        ),
    )

    registry.register(
        name="record_nephrectomy_staging",
        description=(
            "Call ONCE per report. Records AJCC 8th edition pathologic T, N, and M "
            "stages for the case. TNM is a single per-case stage; for multiple "
            "synchronous primary tumors, base T on the highest-stage tumor and set "
            "t_note_multiple. Use pNX (not pN0) when no regional lymph nodes are "
            "submitted/identified."
        ),
        model=RecordNephrectomyStaging,
        comparison_fields=(
            "path_t_stage",
            "t_note_multiple",
            "path_n_stage",
            "path_m_stage",
        ),
    )

    registry.register(
        name="flag_report_for_review",
        description=(
            "Flag the report for human review. Use sparingly — only for "
            "genuinely off cases (no primary nephrectomy, two distinct "
            "nephrectomies in one report, conflicting findings the enums "
            "cannot capture, atypical wording with no matching enum value)."
        ),
        model=FlagReportForReview,
    )

    return registry
