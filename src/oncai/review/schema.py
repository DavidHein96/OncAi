"""Build the review app's ``field_schema`` from extraction tool models.

The physician review app (``apps/review_app/``) renders one editable card
per extracted event. To know *how* to render each field — a dropdown for an
enum, a date widget for an :class:`~oncai.fc_extraction.models.ApproxDate`, a
checkbox for a bool — it reads a ``field_schema`` map keyed by ``event_type``
(the tool name):

    {
      "record_diagnosis": {
        "label": "Record Diagnosis",
        "fields": [
          {"name": "diagnosis_type", "label": "Diagnosis Type",
           "control": "enum", "options": ["primary", ...], "required": true},
          {"name": "diagnosis_date", "label": "Diagnosis Date",
           "control": "approx_date", "required": true},
          ...
        ]
      },
      ...
    }

The ``control`` values mirror exactly what ``web/app.js`` switches on:
``enum``, ``approx_date``, ``number``, ``bool``, ``readonly``, and the default
free-text ``text`` (rendered as a textarea). Source-note anchors are carried by
reserved event fields (``evidence`` and ``review_anchor``), not editable schema
fields.
``event_identity_fields`` is optional metadata used by discrepancy review to
align repeated calls of the same event tool across runs before comparing fields.
``comparison_fields`` is the explicit opt-in list of fields to compare after
alignment; fields not listed are context-only for disagreement selection.

Two entry points:

- :func:`build_field_schema` — the authoritative path. Walks a
  :class:`~oncai.fc_extraction.tools.ToolRegistry` and derives controls from
  each tool's Pydantic JSON schema (enum options, required-ness, descriptions
  all come through). Handles gated registries by gathering tools across every
  phase.
- :func:`infer_field_schema_from_events` — a resilience fallback. When no
  registry is available (e.g. packaging an old batch whose definition has since
  changed), it infers a minimal schema from the observed event values so every
  event still renders editable fields.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from oncai.fc_extraction.models import ExtractionPlan
from oncai.fc_extraction.tools import _inline_refs

from .select import NORMALIZATION_VERSION
from .slots import PROVENANCE_FIELD_NAMES

if TYPE_CHECKING:
    from pydantic import BaseModel

    from oncai.fc_extraction.tools import ToolDefinition, ToolRegistry

# Tools that never carry reviewable findings — the LLM's control-flow signals.
_BUILTIN_TOOLS = {
    "finish_note_extraction",
    "stop_workflow",
    "finish_single_extraction",
}

# Fields present on every ExtractionEvent that we don't render as an editable
# row: note_id is shown as a header chip instead (see app.js renderCard).
_SKIP_FIELDS = {"note_id", *PROVENANCE_FIELD_NAMES}


def _humanize(name: str) -> str:
    """``diagnosis_date`` -> ``Diagnosis Date`` for a human-facing label."""
    return name.replace("_", " ").strip().title() or name


def _unwrap_optional(prop: dict[str, Any]) -> dict[str, Any]:
    """Collapse an ``anyOf: [{...}, {"type": "null"}]`` to its real branch.

    Pydantic renders ``X | None`` as a two-branch ``anyOf``. The null branch
    only conveys optionality (which we read from the schema's ``required`` list
    separately), so for control inference we want the single non-null branch,
    with the outer ``description`` / ``default`` / ``title`` carried onto it.
    """
    branches = prop.get("anyOf")
    if not isinstance(branches, list):
        return prop
    non_null = [b for b in branches if isinstance(b, dict) and b.get("type") != "null"]
    if len(non_null) != 1:
        # 0 (all null — degenerate) or 2+ (a genuine union we can't simplify);
        # leave it alone and let the caller fall through to a text control.
        return prop
    merged = dict(non_null[0])
    for carry in ("description", "default", "title"):
        if carry in prop and carry not in merged:
            merged[carry] = prop[carry]
    return merged


def _is_approx_date(prop: dict[str, Any]) -> bool:
    """True if ``prop`` is an ApproxDate-shaped object (date + precision)."""
    if prop.get("type") != "object":
        return False
    props = prop.get("properties") or {}
    return "date" in props and "precision" in props


def _enum_options(prop: dict[str, Any]) -> list[str] | None:
    """Return the enum value list if this property is an enum, else None."""
    enum = prop.get("enum")
    if isinstance(enum, list) and enum:
        return [str(v) for v in enum]
    return None


def _control_for(prop: dict[str, Any]) -> str:
    """Map a (already optional-unwrapped) JSON-schema property to an app control."""
    if _enum_options(prop) is not None:
        return "enum"
    if _is_approx_date(prop):
        return "approx_date"
    schema_type = prop.get("type")
    if schema_type == "boolean":
        return "bool"
    if schema_type in ("integer", "number"):
        return "number"
    if schema_type == "array":
        return "readonly"
    if schema_type == "object":
        # A nested object that isn't an ApproxDate — no good editor, show as-is.
        return "readonly"
    return "text"


def _build_field(name: str, prop: dict[str, Any], *, required: bool) -> dict[str, Any]:
    """Build one ``fields[]`` entry for the review app from a JSON-schema property."""
    prop = _unwrap_optional(prop)
    control = _control_for(prop)
    field: dict[str, Any] = {
        "name": name,
        "label": _humanize(name),
        "control": control,
        "required": required,
    }
    description = prop.get("description")
    if description:
        field["description"] = description
    if control == "enum":
        field["options"] = _enum_options(prop)
    default = prop.get("default")
    if default is not None:
        field["default"] = default
    return field


def _fields_for_model(model: type[BaseModel]) -> list[dict[str, Any]]:
    """Derive the ordered editable-field list for one tool's Pydantic model."""
    schema = _inline_refs(model.model_json_schema())
    props: dict[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: list[dict[str, Any]] = []
    for name, prop in props.items():
        if name in _SKIP_FIELDS:
            continue
        if not isinstance(prop, dict):
            continue
        fields.append(_build_field(name, prop, required=name in required))
    return fields


def _all_tool_defs(registry: ToolRegistry) -> dict[str, ToolDefinition]:
    """Collect every tool definition a registry can produce, across all phases.

    A plain registry exposes its tools via ``list_tools()``. A
    :class:`~oncai.fc_extraction.tools.GatedToolRegistry` starts in its gate
    phase, so ``list_tools()`` only sees the gate tools — the real extraction
    tools live in ``_phase_map`` / ``_shared_tools``. We gather all of them so
    the field schema covers every event type that can appear in the output.
    """
    defs: dict[str, ToolDefinition] = {}
    for name in registry.list_tools():
        tool_def = registry.get(name)
        if tool_def is not None:
            defs[name] = tool_def
    for attr in ("_gate_tools", "_shared_tools"):
        defs.update(getattr(registry, attr, None) or {})
    phase_map = getattr(registry, "_phase_map", None)
    if isinstance(phase_map, dict):
        for phase_tools in phase_map.values():
            defs.update(phase_tools or {})
    return defs


def build_field_schema(registry: ToolRegistry) -> dict[str, dict[str, Any]]:
    """Build the review app's ``field_schema`` from a tool registry.

    Includes only event tools — built-in control-flow tools and
    :class:`~oncai.fc_extraction.models.ExtractionPlan` tools (orientation
    notes, serialized into the separate ``plans`` block and not reviewed) are
    skipped. Returns ``{event_type: {"label", "fields": [...]}}``.
    """
    schema: dict[str, dict[str, Any]] = {}
    for name, tool_def in _all_tool_defs(registry).items():
        if name in _BUILTIN_TOOLS:
            continue
        model = tool_def.model
        if isinstance(model, type) and issubclass(model, ExtractionPlan):
            continue
        schema[name] = {
            "label": _humanize(name),
            "event_identity_fields": list(tool_def.event_identity_fields),
            "comparison_fields": list(tool_def.comparison_fields),
            "fields": _fields_for_model(model),
        }
    return schema


# ---------------------------------------------------------------------------
# Adjudication hash — the comparability contract two runs must share
# ---------------------------------------------------------------------------


def _comparability_contract(
    definition_name: str,
    field_schema: dict[str, dict[str, Any]],
    normalization_version: str,
) -> dict[str, Any]:
    """The schema-only contract two runs must share to be adjudicated.

    INCLUDES what makes outputs line up field-by-field: the definition name, the
    reviewable event/tool names, each field's name and control/type, enum option
    sets, the per-event identity-key config (``event_identity_fields``) and the
    fields chosen for comparison (``comparison_fields``), and the
    normalization-rules version.

    EXCLUDES everything that varies run-to-run without affecting comparability —
    the system prompt, model/backend, sampling params (temperature/top_p/top_k),
    reasoning effort, git commit, run date, and batch name. (Those are simply
    not inputs here.) ``label`` and ``required`` are dropped too: a humanized
    label or a required-ness toggle doesn't change whether two *values* compare.
    """
    events: dict[str, Any] = {}
    for event_type, spec in field_schema.items():
        fields = []
        for f in spec.get("fields") or []:
            opts = f.get("options")
            fields.append(
                {
                    "name": f.get("name"),
                    "control": f.get("control"),
                    "options": sorted(opts) if opts is not None else None,
                }
            )
        fields.sort(key=lambda f: f["name"] or "")
        events[event_type] = {
            "identity_fields": sorted(spec.get("event_identity_fields") or []),
            "comparison_fields": sorted(spec.get("comparison_fields") or []),
            "fields": fields,
        }
    return {
        "definition_name": definition_name,
        "normalization_version": normalization_version,
        "events": events,
    }


def adjudication_hash(
    definition_name: str,
    field_schema: dict[str, dict[str, Any]],
    *,
    normalization_version: str = NORMALIZATION_VERSION,
) -> str:
    """Hash the comparability contract — two runs are adjudicable iff it matches.

    Deliberately independent of the system prompt, model, sampling, and run
    metadata (see :func:`_comparability_contract`): GPT-5-mini and Gemma
    extracting the same definition produce the **same** adjudication hash and so
    can be compared field-by-field; a definition whose fields, enums, or identity
    keys changed produces a **different** one and cannot be meaningfully diffed.
    """
    payload = json.dumps(
        _comparability_contract(definition_name, field_schema, normalization_version),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def adjudication_hash_from_registry(
    definition_name: str, registry: ToolRegistry
) -> str:
    """``adjudication_hash`` over a registry's reviewable event schema."""
    return adjudication_hash(definition_name, build_field_schema(registry))


# ---------------------------------------------------------------------------
# Fallback: infer a schema from observed event values (no registry available)
# ---------------------------------------------------------------------------


def _control_from_value(value: Any) -> str:
    """Best-effort control for a raw event value when there's no model schema."""
    if isinstance(value, dict) and "date" in value and "precision" in value:
        return "approx_date"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "readonly"
    return "text"


def infer_field_schema_from_events(
    events_by_type: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Infer a minimal ``field_schema`` from observed events.

    Used as a resilience fallback when the originating registry isn't available
    (or doesn't cover an event type seen in the data). Field order follows
    first-seen order across the event instances; controls are inferred from the
    first non-null value seen for each field. Enum options and required-ness are
    unknown here, so dropdowns degrade to free text and nothing is marked
    required.
    """
    schema: dict[str, dict[str, Any]] = {}
    for event_type, events in events_by_type.items():
        fields: dict[str, dict[str, Any]] = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            for name, value in event.items():
                if name in _SKIP_FIELDS:
                    continue
                existing = fields.get(name)
                # Upgrade a previously text-typed field once we see a typed value.
                if existing is None:
                    fields[name] = {
                        "name": name,
                        "label": _humanize(name),
                        "control": _control_from_value(value),
                        "required": False,
                    }
                elif existing["control"] == "text" and value is not None:
                    existing["control"] = _control_from_value(value)
        schema[event_type] = {
            "label": _humanize(event_type),
            "event_identity_fields": [],
            "comparison_fields": [],
            "fields": list(fields.values()),
        }
    return schema


def merge_inferred_schema(
    field_schema: dict[str, dict[str, Any]],
    events_by_type: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Add inferred entries for any event types missing from ``field_schema``.

    Registry-derived schema wins; this only fills gaps so an event type present
    in the data but absent from the registry (renamed tool, stale definition)
    still renders editable fields instead of the app's empty fallback.
    """
    missing = {
        event_type: events
        for event_type, events in events_by_type.items()
        if event_type not in field_schema
    }
    if not missing:
        return field_schema
    merged = dict(field_schema)
    merged.update(infer_field_schema_from_events(missing))
    return merged
