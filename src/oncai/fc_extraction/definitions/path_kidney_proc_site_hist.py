"""
Kidney pathology advanced extraction definition.

Extracts structured data from kidney cancer pathology reports. All tools are
available simultaneously so that reports containing multiple specimens
(e.g., nephrectomy + lymph node biopsy) can be fully captured.

Each tool includes a `specimen` field to identify which specimen the findings
belong to when a report covers multiple specimens.

Used with batch_single.py for single-note processing.
"""

# TODO: CLEAN UP CODE AND UPDATE DOC STRINGS
from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import Annotated

from pydantic import BeforeValidator, Field

from oncai.fc_extraction.models import ExtractionEvent, ExtractionPlan
from oncai.fc_extraction.tools import ToolRegistry

logger = logging.getLogger(__name__)

DEFINITION_NAME = "PathProcedureSiteHist"


# =============================================================================
# Local enums (specific to pathology reports)
# =============================================================================


class ProcedureType(StrEnum):
    BIOPSY = "Biopsy"
    FINE_NEEDLE_ASPIRATION = "Fine needle aspiration"
    LYMPH_NODE_DISSECTION = "Lymph node dissection"
    ADRENALECTOMY = "Adrenalectomy"
    NEPHROURETERECTOMY = "Nephroureterectomy"
    # Any resection (even if it is specified)
    RESECTION = "Resection, not otherwise specified"
    # any excision (even if it is specified)
    EXCISION = "Excision, not otherwise specified"
    PARTIAL_NEPHRECTOMY = "Partial nephrectomy"
    RADICAL_NEPHRECTOMY = "Radical nephrectomy"
    TOTAL_NEPHRECTOMY = "Total nephrectomy"
    NEPHRECTOMY_UNSPECIFIED = "Nephrectomy, not otherwise specified"
    NOT_SPECIFIED = "Not specified"
    OTHER = "Other"


class HistologicType(StrEnum):
    CLEAR_CELL_RCC = "Clear cell renal cell carcinoma"
    MULTILOCULAR_CYSTIC = (
        "Multilocular cystic clear cell renal cell neoplasm of low malignant potential"
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
    SDH_DEFICIENT = "Succinate dehydrogenase-deficient (SDH) renal carcinoma"
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
    NOT_SPECIFIED = "Not specified"
    NECROTIC_NONVIABLE = "Necrotic/nonviable tumor"
    BENIGN = "Benign tissue, negative for malignancy"
    NO_TISSUE_PRESENT = "No tissue present"
    NON_KIDNEY_MALIGNANCY = "Malignancy, not consistent with primary kidney tumor"
    ATYPICAL_CELLS = "Atypical cells"
    OTHER = "Other"


class PhysicianConfidence(StrEnum):
    """How confident the physician was about a finding"""

    DEFINITE = "definite"
    LIKELY = "likely"
    SUSPICIOUS = "suspicious"
    INDETERMINATE = "indeterminate"
    CONCERNING = "concerning"
    NOT_STATED = "not stated"
    NOT_APPLICABLE = "not applicable"


# class AnatomicalSiteBroad(str, Enum):
#     """Broad anatomical category for anatomical sites"""

#     LUNG = "Lung"
#     BONE_EXCEPT_SPINE = "Bone, except spine"
#     SPINE = "Spine, vertebral column"
#     LIVER = "Liver"
#     BRAIN = "Brain"
#     ADRENAL_GLAND = "Adrenal gland"
#     LYMPH_NODE_RENAL_HILAR = "Lymph node, renal hilar"
#     LYMPH_NODE_RETROPERITONEAL = "Lymph node, regional retroperitoneal"
#     LYMPH_NODE_THORACIC = "Lymph node, thoracic / mediastinal"
#     LYMPH_NODE_HEAD_NECK = "Lymph node, head / neck"
#     LYMPH_NODE_PELVIC_INGUINAL = "Lymph node, pelvic / inguinal"
#     LYMPH_NODE_OTHER = "Lymph node, other specified"
#     LYMPH_NODE_UNSPECIFIED = "Lymph node, unspecified"
#     PERITONEUM = "Peritoneum"
#     DISTAL_EXTRAHEPATIC_BILE_DUCT = "Distal extrahepatic bile duct"
#     PERIHILAR_BILE_DUCT = "Perihilar bile duct"
#     PANCREAS = "Pancreas"
#     SKIN = "Skin"
#     THYROID = "Thyroid"
#     THYMUS = "Thymus"
#     KIDNEY_LEFT = "Kidney, left"
#     KIDNEY_RIGHT = "Kidney, right"
#     KIDNEY_UNSPECIFIED = "Kidney, unspecified laterality"
#     GALLBLADDER = "Gallbladder"
#     RENAL_PELVIS = "Renal pelvis"
#     PERIRENAL_FAT = "Perirenal fat"
#     URETER = "Ureter"
#     BLADDER = "Bladder"
#     INFERIOR_VENA_CAVA = "Inferior vena cava"
#     RENAL_VEIN = "Renal vein"
#     RENAL_SINUS = "Renal sinus"
#     COLON = "Colon"
#     OTHER_GENITOURINARY = "Other genitourinary site"
#     OTHER_ABDOMINAL = "Other abdominal site"
#     OTHER_ENDOCRINE = "Other endocrine site"
#     OTHER_GASTROINTESTINAL = "Other gastrointestinal site"
#     OTHER_GYNECOLOGIC = "Other gynecologic site"
#     OTHER_NOT_LISTED = "Other, not listed"


class AnatomicalSiteBroad(StrEnum):
    """Broad anatomical category for biopsy, metastasis, and resection sites.

    This enum is intentionally broad. Use a separate free-text anatomical_site_detail
    field to preserve the exact report wording, such as "left lower lobe lung",
    "right adrenal gland", "para-aortic lymph node", or "T8 vertebral body".

    Regional renal lymph nodes are separated from distant/non-regional lymph nodes
    because regional nodal RCC involvement should not be treated the same as
    distant metastatic RCC.
    """

    # -------------------------------------------------------------------------
    # Kidney primary / local renal structures
    # -------------------------------------------------------------------------

    KIDNEY_LEFT = "Kidney, left"
    KIDNEY_RIGHT = "Kidney, right"
    KIDNEY_BILATERAL = "Kidney, bilateral"
    KIDNEY_UNSPECIFIED = "Kidney, unspecified laterality"

    RENAL_PELVIS = "Renal pelvis"
    RENAL_SINUS = "Renal sinus"
    PERIRENAL_FAT = "Perirenal / perinephric fat"
    GEROTA_FASCIA = "Gerota fascia / renal fascia"

    RENAL_VEIN = "Renal vein"
    INFERIOR_VENA_CAVA = "Inferior vena cava"

    URETER = "Ureter"
    BLADDER = "Bladder"

    # -------------------------------------------------------------------------
    # Singular lymph node category to encompass all lymph node biopsies and metastases
    # -------------------------------------------------------------------------

    LYMPH_NODE = "Lymph node"

    # -------------------------------------------------------------------------
    # Common distant metastatic sites for RCC
    # -------------------------------------------------------------------------

    LUNG = "Lung"
    PLEURA = "Pleura"
    BONE_EXCEPT_SPINE = "Bone, except spine"
    SPINE = "Spine, vertebral column"
    LIVER = "Liver"
    BRAIN = "Brain"
    ADRENAL_GLAND = "Adrenal gland"
    PANCREAS = "Pancreas"
    THYROID = "Thyroid"
    SKIN = "Skin"
    SOFT_TISSUE = "Soft tissue"
    SKELETAL_MUSCLE = "Skeletal muscle"

    # -------------------------------------------------------------------------
    # Other thoracic sites
    # -------------------------------------------------------------------------
    THYMUS = "Thymus"
    HEART_PERICARDIUM = "Heart / pericardium"
    CHEST_WALL = "Chest wall"

    # -------------------------------------------------------------------------
    # Other abdominal / peritoneal sites
    # -------------------------------------------------------------------------
    PERITONEUM = "Peritoneum"
    RETROPERITONEUM_NON_NODAL = "Retroperitoneum, non-lymph node"
    SPLEEN = "Spleen"
    GALLBLADDER = "Gallbladder"
    BILE_DUCT_EXTRAHEPATIC = "Extrahepatic bile duct"
    COLON = "Colon"
    SMALL_BOWEL = "Small bowel"
    STOMACH = "Stomach"
    MESENTERY_OMENTUM = "Mesentery / omentum"

    # -------------------------------------------------------------------------
    # Other genitourinary / reproductive sites
    # -------------------------------------------------------------------------
    PROSTATE = "Prostate"
    OVARY_FALLOPIAN_TUBE = "Ovary / fallopian tube"
    UTERUS_CERVIX = "Uterus / cervix"

    # -------------------------------------------------------------------------
    # Broad fallback categories & Edge cases
    # -------------------------------------------------------------------------

    OTHER = "Other"
    NOT_SPECIFIED = "Not specified"


# class LymphNodeType(str, Enum):
#     HILAR = "Hilar"
#     PRECAVAL = "Precaval"
#     INTERAORTOCAVAL = "Interaortocaval"
#     PARACAVAL = "Paracaval"
#     RETROCAVAL = "Retrocaval"
#     PREAORTIC = "Preaortic"
#     PARAORTIC = "Paraaortic"
#     RETROAORTIC = "Retroaortic"
#     PELVIC = "Pelvic"
#     RETROPERITONEAL = "Retroperitoneal"
#     SUBPECTORAL = "Subpectoral"
#     AXILLARY = "Axillary"
#     INGUINAL = "Inguinal"
#     CERVICAL = "Cervical"
#     SUBMANDIBULAR = "Submandibular"
#     SUPRACLAVICULAR = "Supraclavicular"
#     AORTOCAVAL = "Aortocaval"


class LymphNodeType(StrEnum):
    """Specific lymph node group for RCC biopsy/metastasis extraction.

    Regional renal lymph nodes are separated from distant/non-regional lymph nodes
    because regional nodal RCC involvement should not be classified the same way
    as distant metastatic RCC.
    """

    # -------------------------------------------------------------------------
    # Regional renal lymph nodes
    # -------------------------------------------------------------------------

    RENAL_HILAR = "Renal hilar"
    PARA_AORTIC = "Para-aortic / periaortic"
    PREAORTIC = "Preaortic"
    RETROAORTIC = "Retroaortic"
    LATERAL_AORTIC_LUMBAR = "Lateral aortic / lumbar"

    CAVAL = "Caval"
    PARACAVAL = "Paracaval"
    PRECAVAL = "Precaval"
    RETROCAVAL = "Retrocaval"
    PERICAVAL = "Pericaval"

    INTERAORTOCAVAL = "Interaortocaval"
    RETROPERITONEAL_REGIONAL_NOS = "Retroperitoneal regional, NOS"

    # -------------------------------------------------------------------------
    # Distant / non-regional lymph nodes
    # -------------------------------------------------------------------------

    MEDIASTINAL = "Mediastinal"
    HILAR_THORACIC = "Thoracic hilar"

    CERVICAL = "Cervical"
    SUPRACLAVICULAR = "Supraclavicular"
    AXILLARY = "Axillary"

    ILIAC = "Iliac"
    PELVIC_NOS = "Pelvic, NOS"
    INGUINAL = "Inguinal"

    MESENTERIC = "Mesenteric"
    PORTA_HEPATIS = "Porta hepatis"
    PERIPANCREATIC = "Peripancreatic"
    GASTROHEPATIC = "Gastrohepatic"
    CELIAC = "Celiac"

    OTHER_DISTANT = "Other distant / non-regional lymph node"
    LYMPH_NODE_NOS = "Lymph node, NOS"
    NOT_APPLICABLE = "Not applicable"
    NOT_SPECIFIED = "Not specified"


# =============================================================================
# Normalization helpers
# =============================================================================


def _normalize_key(s: str) -> str:
    """Reduce a string to a canonical comparison key by lowercasing and
    stripping whitespace, hyphens, underscores, slashes, dots, and parens."""
    return re.sub(r"[\s\-_./()]+", "", s).lower()


def _build_enum_lookup(enum_cls: type[StrEnum]) -> dict[str, str]:
    """Build a {normalized_key: enum_value} mapping for an entire Enum."""
    return {_normalize_key(m.value): m.value for m in enum_cls}


def _build_literal_lookup(values: tuple[str, ...]) -> dict[str, str]:
    """Build a {normalized_key: canonical_value} mapping for a Literal-style tuple."""
    return {_normalize_key(v): v for v in values}


def _normalize_against(v: object, lookup: dict[str, str], field_name: str) -> str:
    """Match a value against a normalized lookup; log when a correction is made."""
    if isinstance(v, StrEnum):
        return v.value
    s = str(v).strip()
    match = lookup.get(_normalize_key(s))
    if match is not None and match != s:
        logger.debug("normalizer fixed %s: %r -> %r", field_name, s, match)
    return match if match is not None else s


_PROCEDURE_TYPE_LOOKUP = _build_enum_lookup(ProcedureType)
_HISTOLOGIC_TYPE_LOOKUP = _build_enum_lookup(HistologicType)
_PHYSICIAN_CONFIDENCE_LOOKUP = _build_enum_lookup(PhysicianConfidence)
_ANATOMICAL_SITE_LOOKUP = _build_enum_lookup(AnatomicalSiteBroad)
_LYMPH_NODE_TYPE_LOOKUP = _build_enum_lookup(LymphNodeType)

_SITUATION_VALUES = ("metastatic finding", "recurrence")
_SITUATION_LOOKUP = _build_literal_lookup(_SITUATION_VALUES)


def normalize_procedure_type(v: object) -> str:
    return _normalize_against(v, _PROCEDURE_TYPE_LOOKUP, "procedure_type")


def normalize_histologic_type(v: object) -> str:
    return _normalize_against(v, _HISTOLOGIC_TYPE_LOOKUP, "histologic_type")


def normalize_physician_confidence(v: object) -> str:
    return _normalize_against(v, _PHYSICIAN_CONFIDENCE_LOOKUP, "confidence_level")


def normalize_anatomical_site(v: object) -> str:
    return _normalize_against(v, _ANATOMICAL_SITE_LOOKUP, "anatomical_site")


def normalize_lymph_node_type(v: object) -> str:
    return _normalize_against(v, _LYMPH_NODE_TYPE_LOOKUP, "lymph_node_type")


def normalize_situation(v: object) -> str:
    return _normalize_against(v, _SITUATION_LOOKUP, "situation")


# =============================================================================
# Tool models
# =============================================================================


class RecordSpecimenFindings(ExtractionEvent):
    """Record sub-specimen-level findings for a specimen."""

    specimen_id: str = Field(
        ...,
        description="Which specimen these findings belong to, typically a single uppercase letter",
    )
    sub_specimen_label: str = Field(
        ...,
        description=(
            "In most intances, a single specimen 'A' corresponds to a single anatomical site. In some cases, however, a single specimen may contain multiple non-contiguous sites (e.g., 'A: left adrenal biopsy showing adrenal tissue and separate perinephric fat with tumor deposits'). In those cases, use this field to indicate a unique specimen label using the the convention '{specimen_id}||{simple_snake_case_site}', etc. (e.g., 'A||adrenal', 'A||perinephric_fat'). If there are no subspecimens in the specimen, still provide a label with the same convention for consistency. This concept is particularly important for specimens that contain tissue from with multiple sites and each site has a different histology"
        ),
    )
    anatomical_site_detail: str = Field(
        ...,
        description="Exact sentence(s) in the report describing the anatomical site. This should match the report as exactly as possible so it can be used for auditing. If multiple sentences must be joined, join them with a ' || ' delimiter. Use a few words before and after to help capture the string so it can be easily matched back on audit.",
    )
    anatomical_site: Annotated[
        AnatomicalSiteBroad, BeforeValidator(normalize_anatomical_site)
    ] = Field(
        ...,
        description=("Broader anatomical site enum category"),
    )
    lymph_node_type: Annotated[
        LymphNodeType, BeforeValidator(normalize_lymph_node_type)
    ] = Field(
        LymphNodeType.NOT_APPLICABLE,
        description=(
            "Specific lymph node group, if applicable. Use 'Not applicable' if the site is not a lymph node."
        ),
    )
    procedure_type_detail: str = Field(
        ...,
        description=(
            "Exact sentence(s) in the report describing the procedure type for this specimen. This should match the report as exactly as possible so it can be used for auditing. If multiple sentences must be joined, join them with a ' || ' delimiter. Use a few words before and after to help capture the string so it can be easily matched back on audit."
        ),
    )
    procedure_type: Annotated[
        ProcedureType, BeforeValidator(normalize_procedure_type)
    ] = Field(
        ProcedureType.NOT_SPECIFIED,
        description="Broader procedure type enum category. Use 'Not specified' if the type of procedure is not mentioned in the report.",
    )
    histologic_type_detail: str = Field(
        ...,
        description=(
            "Exact final histology, after any addendums, as written in the report, including any descriptive qualifiers. This should match the report as exactly as possible so it can be used for auditing. If multiple sentences must be joined, join them with a ' || ' delimiter. Use a few words before and after to help capture the string so it can be easily matched back on audit."
        ),
    )
    histologic_type: Annotated[
        HistologicType, BeforeValidator(normalize_histologic_type)
    ] = Field(
        ...,
        description="Histologic classification of the tumor based on enum ",
    )
    histological_confidence_detail: str = Field(
        ...,
        description=(
            "Exact sentence(s) in the report describing the physician's confidence in the histologic classification. This should match the report as exactly as possible so it can be used for auditing. If multiple sentences must be joined, join them with a ' || ' delimiter. Use a few words before and after to help capture the string so it can be easily matched back on audit."
        ),
    )
    histology_confidence_level: Annotated[
        PhysicianConfidence, BeforeValidator(normalize_physician_confidence)
    ] = Field(
        PhysicianConfidence.NOT_APPLICABLE,
        description=(
            "How confident the physician is about the histologic classification, based on the language in the report. Use 'Not applicable' if confidence level is not stated or cannot be inferred from the report."
        ),
    )


class PlanSpecimenFindings(ExtractionPlan):
    """Plan the extraction for one specimen site, procedure, histology, and any sub specimen differences, before
    recording it."""

    specimen_id: str = Field(
        ...,
        description="Specimen identifier (e.g., 'A', 'B') If the report only contains a single specimen, use 'A'. If the report contains outside specimen names AND internal specimen labels, use the INTERNAL specimen labels. Finally, if only external specimens names are given use those",
    )
    specimen_summary: str = Field(
        ...,
        description=(
            "One-sentence paraphrase: site + procedure + headline finding. "
            "E.g., 'Left radical nephrectomy showing clear cell RCC, pT3a, with one "
            "positive renal hilar lymph node' or 'CT-guided core biopsy of T8 vertebral "
            "body, consistent with metastatic clear cell RCC'."
        ),
    )
    sub_specimen_differences: str = Field(
        ...,
        description=(
            "If there are multiple non-contiguous sites within the same specimen (e.g., 'A: left adrenal biopsy showing adrenal tissue and separate perinephric fat with tumor deposits'), describe how the sites differ in their findings (e.g., 'adrenal tissue negative for malignancy, perinephric fat positive for RCC'). If there is only one site in the specimen, write 'None'. This concept is particularly important for specimens that contain tissue from with multiple sites and each site has a different histology. If there are substantial differences you may need to call record_specimen_findings multiple times with different sub_specimen_label values (e.g., 'A||adrenal' for the adrenal tissue and 'A||perinephric_fat' for the perinephric fat in the example above). Typically, try to group as much as possible under a single sub specimen. For nephrectomies, there are typically many contigous sites with benign tissue that can be ignored."
        ),
    )
    anatomical_site_plan: str = Field(
        ...,
        description=(
            "Quote the site language as introduced after the specimen letter, then pick "
            "the AnatomicalSiteBroad enum value and explain why. If the site is a lymph "
            "node, also state which LymphNodeType station applies and whether it is "
            "regional or distant (see system prompt for the station lists)."
            "Note that the anatomical site is independent of the histology."
        ),
    )
    procedure_type_plan: str = Field(
        ...,
        description=(
            "Quote the procedure language and pick a ProcedureType enum value. "
            "Disambiguation specific to this enum: "
            "(1) Radical nephrectomy = entire kidney + calyces + pelvis + variable "
            "ureter. Total nephrectomy = similar but typically for presumed benign "
            "disease, may not extend to Gerota's fascia. Partial nephrectomy = "
            "enucleation through partial resection. "
            "(2) RESECTION / EXCISION (NOS) when the report uses those terms generically "
            " resection = removal of part/all of an organ, excision = smaller localized. "
            "(3) BIOPSY covers any biopsy not meeting FNA criteria (core, incisional, "
            "excisional, bare 'needle biopsy'). "
            "(4) FINE_NEEDLE_ASPIRATION requires BOTH 'fine needle' AND 'aspiration', "
            "either word alone is NOT FNA. "
            "(5) LYMPH_NODE_DISSECTION = lymphadenectomy. "
            "If the report does not state any procedure for the specimen, plan NOT_SPECIFIED."
        ),
    )
    histologic_type_plan: str = Field(
        ...,
        description=(
            "Quote the FINAL diagnosis line (post addendum) and list the HistologicType "
            "enum values you considered and why you ruled the runners up out. "
            "Tricky calls specific to this enum: "
            "(a) 'Renal cell carcinoma, NOS (unclassified)' is for genuinely complex/borderline subtyping; "
            "(b) 'Renal cell carcinoma, no subtype specified' is for cases the pathologist simply didn't subtype. "
            "(c) SUBTYPE_PENDING only when specific pending studies are named AND no "
            "addendum has resolved them. "
            "(d) For metastatic site carcinomas with no kidney RCC features, weigh IHC: "
            "if IHC supports renal origin (e.g.,CAIX+ for clear cell) use the "
            "matching RCC subtype; Potentially weigh the patient history if the report emphasizes consistency with a known prior kidney tumor; "
        ),
    )


class PlanSpecimens(ExtractionPlan):
    """Plan the individual specimens in the report before planning the per-specimen details."""

    specimen_list: str = Field(
        ...,
        description=(
            "List all the specimens in the report, typically labeled with uppercase letters (e.g., 'A: left radical nephrectomy, B: periaortic lymph node biopsy'). For single-specimen reports without explicit labels, just default to 'A'. If there are both outside report specimen names and internal specimen labels, use the internal specimen labels. This sets the stage for the per-specimen planning in PlanSpecimenFindings."
        ),
    )


class FlagReportForReview(ExtractionEvent):
    """Flag a report for manual review by a human expert."""

    reason: str = Field(
        ...,
        description=(
            "Reason for flagging this report for review, 'conflicting information', or 'multiple possible interpretations'. This should only be used for cases where the report is genuinely ambiguous or complex and cannot be reliably extracted with the current toolset"
        ),
    )
    flagged_text: str = Field(
        ...,
        description=(
            "The specific text in the report that triggered the flag, quoted verbatim for review. If multiple distinct text passages contributed, join them with ' || ' delimiter."
        ),
    )


# =============================================================================
# System Prompt
# =============================================================================


SYSTEM_PROMPT = """\
You are a clinical data extraction specialist for kidney cancer pathology reports.
Your job is to extract structured per-specimen findings — anatomical site, procedure
type, and histology — from pathology reports.

## WORKFLOW
For each report:
1. Call `plan_specimens` ONCE to enumerate the specimens you see.
2. For EACH specimen, call `plan_specimen_findings` to think through site / procedure /
   histology, and whether there are distinct sub-specimens before extracting.
3. For EACH specimen/sub-specimen, call `record_specimen_findings` to commit the extracted enums
   and free-text detail.
4. Call `flag_report_for_review` only if the report is genuinely ambiguous and cannot
   be reliably captured with the available enums.
5. Call `finish_note_extraction` after all specimens have been recorded.

## SPECIMEN LABELING
- Specimens are usually labeled with uppercase letters (A, B, C, ...).
- For single-specimen reports without an explicit label, use 'A'.
- If outside-report specimen names AND internal specimen labels both exist, use the
  INTERNAL labels. If only outside-report specimen names exist, use those.
- You dont need to worry about specimen block IDs,

## READING RULES (apply to every specimen)
- Use ONLY findings from the CURRENT report. Ignore prior medical history unless the
  pathologist explicitly relates it to this specimen. If the pathologist does say something like "consistent with prior RCC", that is fair game, but if the report just lists a prior history of RCC without linking it to the current findings, ignore it.
- Addenda are authoritative: if an addendum (typically IHC, molecular, special stains,
  or cytogenetics) clarifies or overrides the original diagnosis, treat the addendum's
  conclusion as the final answer.
- Capture explicit negations (e.g., "NOT consistent with RCC") in your reasoning.
- The Benign tissue, negative for malignancy label can match for any non-cancerous tissue if it is not specified as being malignant.

## EVIDENCE STRENGTH (calibration for every diagnosis call)
- "Consistent with" or "compatible with" a histology IS strong enough to assign that
  subtype.
- "Prior history noted", "suggestive of", "favor", or "cannot exclude" alone is NOT
  strong enough — make your best judgement to assign either the most specific subtype and include the uncertainty in the histology_confidence_level, or assign a broader category like "RCC, no subtype specified" or "RCC, NOS (unclassified)" that can be supported by the report without over-interpreting it.

## Detail strings
- For the detail strings that are meant to be verbatim quotes from the report, try to capture the exact wording as much as possible, including any descriptive qualifiers. If multiple sentences must be joined to capture the full detail, join them with a ' || ' delimiter. Use a few words before and after the key language to help capture the string so it can be easily matched back on audit.
- If there are multiple sentences that describe the exact same finding, only pick one representative sentence for the detail string and ignore the rest to avoid redundancy.

## ANATOMICAL SITE — META RULES
- Histology does NOT determine site. RCC metastatic to lung is anatomically Lung,
  not Kidney.
- Direct contiguous tumor extension into a neighboring organ (adrenal invaded by
  primary kidney tumor, IVC thrombus, perinephric fat involvement) is NOT a separate
  anatomical site — it stays as the primary site. Non-contiguous tumor in the same
  organ IS a separate site.
- Position/direction prefixes matter: "peripancreatic mass" is NOT Pancreas;
- Capture laterality for kidney
- When the broad site is a lymph node, also capture the specific lymph node, otherwise just put NOT_APPLICABLE in the lymph_node_type field.
- Use the appropriate level of granularity for anatomical_site. For example, 'left kidney upper pole' -> 'Kidney, left' (with the pole detail in anatomical_site_detail), and 'renal hilar lymph node' -> 'Lymph node' with 'Renal hilar' as LymphNodeType value.

## PROCEDURE TYPE — META RULES
- A standard radical nephrectomy specimen consists of the entire kidney including the calyces, pelvis, and a variable length of ureter.
- A partial nephrectomy specimen may vary from an enucleation of the tumor with almost no normal tissue to a partial resection containing variable portions of calyceal or renal pelvic collecting system.
- If the report is clearly not a nephrectomy but uses the term "resection" or "excision" or "biopsy" of any kind, use the generic RESECTION / EXCISION / BIOPSY  'not otherwise specified' categories as a catch all. For example, a "liver biopsy" or "bone biopsy" would be Biopsy, and a "liver wedge resection" would be Resection, not otherwise specified.
- A "fine needle aspiration" requires BOTH "fine needle" AND "aspiration" language

### SUB-SPECIMENS
- Typically a biopsy of a single anatomical site corresponds to a single specimen (e.g., "B: left adrenal biopsy"). Sometimes, however, a single specimen may contain multiple non-contiguous sites with different findings (e.g., "A: left adrenal biopsy showing adrenal tissue and separate perinephric fat with tumor deposits"). In those cases, use the sub_specimen_label field
- This is common in nephrectomy specimens, that could contain tumor in the kidney, perinephric fat, renal vein, and/or adrenal gland. If there are multiple distinct sites with different findings, make a judgment call according to this rubric:
   (1) If the sites are contiguous (e.g., tumor invading from left kidney into perinephric fat), keep them under the same specimen/sub-specimen label (in that instance a single subspecimen with the site 'Kidney, left')
   (2) Is it a nephrectomy? If so are there any lymph nodes or adrenal gland or other major non-contiguous sites? Those can have their own sub-specimen label because they are non-contiguous with the kidney tumor and may have distinct histology. If there are only things like perinephric fat or renal vein thrombus that are contiguous with the main tumor), keep them under the same specimen/sub-specimen label. Most nephrectomies will end up with either just one sub-specimen for the Kidney, or one or two extra sub-specimens for the adrenal and/or lymph nodes.
   (3) If there are multiple distinct sub-specimens, use the convention '{specimen_id}||{simple_snake_case_label}', etc. (e.g., 'A||adrenal', 'A||perinephric_fat') to label them, and describe the differences in the sub_specimen_differences field.
   (4) Particularly when there is some benign tissue along side a tumor, the bengin tissue does not need its own sub-specimen — just capture the tumor findings in the main specimen and ignore the benign tissue.
   (5) Prioritize grouping by as few specimen/sub-specimen labels as possible, so only create multiple sub-specimens if there are distinct sites with distinct findings that cannot be reasonably grouped together under a single site and histology. Still provide a sub-specimen label even if there is only one site to preserve consistency in the data structure.

## MODERNIZED RCC SUBTYPE NAMES
- "Xp11 translocation RCC" → TFE3-rearranged renal cell carcinoma
- "Hereditary leiomyomatosis-associated RCC" / HLRCC → Fumarate hydratase-deficient
  renal cell carcinoma
- "t(6;11) RCC" / "MiTF/TFEB family translocation RCC" / "TFEB-rearranged RCC" →
  TFEB-altered renal cell carcinoma
- "Renal medullary carcinoma" / RMC (classically in patients with sickle cell
  trait or disease; loss of SMARCB1 / INI-1 by IHC) → SMARCB1-deficient renal
  medullary carcinoma
- 'Renal cell carcinoma, NOS (unclassified)' ->  used when classification to a subtype is difficult due to complex or borderline histological features
- 'Renal cell carcinoma, no subtype specified' -> used when the pathologist simply does not provide a subtype and only refers to the histology as renal cell carcinoma.


## FLAGGING
Use `flag_report_for_review` sparingly — only for genuinely conflicting information,
multiple plausible interpretations the enums cannot capture, or wording so atypical
that no listed value fits. Routine "Not specified" cases do NOT warrant a flag.
"""


# =============================================================================
# Registry Factory
# =============================================================================


def create_path_procedure_site_hist_registry() -> ToolRegistry:
    """Create tool registry for procedure / anatomical site / histology extraction.

    Workflow: plan_specimens → plan_specimen_findings (per specimen) →
    record_specimen_findings (per specimen) → finish_note_extraction.
    flag_report_for_review is optional and only for genuinely ambiguous cases.

    NOTE: ledger_fields have no effect in single-note mode.
    """
    registry = ToolRegistry(single_note=True)

    registry.register(
        name="plan_specimens",
        description=(
            "Call ONCE per report, BEFORE any other tool, to enumerate the specimens "
            "in the report. List each specimen by its label (A, B, C, ...) along with "
            "a brief note on what tissue and procedure each one represents. This sets "
            "the stage for the per-specimen plan_specimen_findings calls that follow."
        ),
        model=PlanSpecimens,
    )

    registry.register(
        name="plan_specimen_findings",
        description=(
            "Call ONCE per specimen, AFTER plan_specimens and BEFORE "
            "record_specimen_findings for that specimen. Think out loud: quote the "
            "relevant report text, list the enum candidates you considered, and "
            "explain how you resolved any ambiguity + if there are relevant sub specimens. "
            "Do NOT commit final enum values here — those go in record_specimen_findings."
        ),
        model=PlanSpecimenFindings,
    )

    registry.register(
        name="record_specimen_findings",
        description=(
            "Record the structured findings for one specimen — broad anatomical site "
            "(+ free-text detail and lymph node type if applicable), procedure type (+ free-text detail), histologic type (+ free-text detail + physician confidence)."
            "Call ONCE per sub-specimen, AFTER plan_specimen_findings for that specimen."
        ),
        model=RecordSpecimenFindings,
    )

    registry.register(
        name="flag_report_for_review",
        description=(
            "Flag the report for human review. Use sparingly — only for genuinely "
            "conflicting information, multiple plausible interpretations the available "
            "enums cannot capture, or wording so atypical that no listed value fits. "
            "Routine 'Not specified' cases do NOT warrant a flag."
        ),
        model=FlagReportForReview,
    )

    return registry
