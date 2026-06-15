"""Select extraction records that should be sent to human review."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from oncai.fc_extraction.load import _load_fc_records

FLAG_REVIEW_TOOL_NAMES = frozenset({"flag_report_for_review"})
DEFAULT_AGREEMENT_IGNORE_FIELDS = ("comment",)
_IGNORE = object()

# Version of the event-normalization rules below (flatten / date-pad / evidence
# join / string-keep) that decide whether two runs' values match. Part of the
# adjudication hash, so runs normalized under different rules are treated as not
# comparable. Bump this on any change to those rules.
NORMALIZATION_VERSION = "1"

ComparableFieldMap = dict[str, set[str]]


def note_id_for_record(record: dict[str, Any]) -> str:
    """Return the record's note/report identifier."""
    return str(record.get("note_id") or "")


def successful_records_from_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load success-only extraction records from a batch JSONL."""
    return [r for r in _load_fc_records(jsonl_path) if r.get("success")]


def record_has_review_flag(
    record: dict[str, Any],
    *,
    flag_tool_names: Iterable[str] = FLAG_REVIEW_TOOL_NAMES,
) -> bool:
    """Whether a record called one of the report-level review flag tools."""
    flag_tools = set(flag_tool_names)
    events = record.get("events") or {}
    if not isinstance(events, dict):
        return False
    for tool_name in flag_tools:
        event_list = events.get(tool_name)
        if isinstance(event_list, list) and event_list:
            return True
    return False


def flagged_note_ids(
    records: Iterable[dict[str, Any]],
    *,
    flag_tool_names: Iterable[str] = FLAG_REVIEW_TOOL_NAMES,
) -> set[str]:
    """Return note IDs whose extraction included a review flag."""
    ids: set[str] = set()
    for record in records:
        note_id = note_id_for_record(record)
        if note_id and record_has_review_flag(record, flag_tool_names=flag_tool_names):
            ids.add(note_id)
    return ids


def flagged_note_ids_from_jsonl(jsonl_path: Path) -> set[str]:
    """Return flagged note IDs from a batch JSONL."""
    return flagged_note_ids(successful_records_from_jsonl(jsonl_path))


def disagreement_note_ids(
    primary_jsonl: Path,
    compare_jsonls: Iterable[Path],
    *,
    ignore_fields: Iterable[str] = DEFAULT_AGREEMENT_IGNORE_FIELDS,
    comparable_fields: ComparableFieldMap | None = None,
) -> set[str]:
    """Return primary-run note IDs where any compared run disagrees.

    Agreement is evaluated on explicitly comparable fields from the structured
    ``events`` block, ignoring ``note_id`` and the configured fields. Field
    comparison is opt-in via ``comparable_fields`` (typically derived from the
    registry's event_identity_fields + comparison_fields metadata). Event order
    is ignored so two runs that produce the same set of events in a different
    order still agree. Missing records in a compared run count as disagreement.
    """
    ignored = set(ignore_fields) | {"note_id"}
    comparable_fields = comparable_fields or {}
    primary_records = _records_by_note_id(primary_jsonl)
    comparison_sets = [_records_by_note_id(path) for path in compare_jsonls]
    out: set[str] = set()

    for note_id, record in primary_records.items():
        primary_sig = _canonical_events(record, ignored, comparable_fields)
        for compared in comparison_sets:
            other = compared.get(note_id)
            if (
                other is None
                or _canonical_events(other, ignored, comparable_fields) != primary_sig
            ):
                out.add(note_id)
                break
    return out


def comparable_fields_from_field_schema(
    field_schema: dict[str, dict[str, Any]],
) -> ComparableFieldMap:
    """Extract fields that should participate in agreement checks.

    Comparison is explicit opt-in: event identity fields are included so runs
    align the same logical event, and comparison_fields are included so selected
    values can create disagreements. Other fields are context-only, regardless
    of JSON-schema type or UI control.
    """
    comparable: ComparableFieldMap = {}
    for event_type, spec in field_schema.items():
        if not isinstance(spec, dict):
            continue
        raw_identity_fields = spec.get("event_identity_fields", [])
        identity_fields = (
            {str(name) for name in raw_identity_fields if str(name).strip()}
            if isinstance(raw_identity_fields, list)
            else set()
        )
        raw_comparison_fields = spec.get("comparison_fields", [])
        comparison_fields = (
            {str(name) for name in raw_comparison_fields if str(name).strip()}
            if isinstance(raw_comparison_fields, list)
            else set()
        )
        names = identity_fields | comparison_fields
        if names:
            comparable[str(event_type)] = names
    return comparable


def _records_by_note_id(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in successful_records_from_jsonl(jsonl_path):
        note_id = note_id_for_record(record)
        if note_id:
            records[note_id] = record
    return records


def _canonical_events(
    record: dict[str, Any],
    ignored: set[str],
    comparable_fields: ComparableFieldMap,
) -> str:
    events = record.get("events") or {}
    if not isinstance(events, dict):
        return "{}"

    canonical: dict[str, list[str]] = {}
    for event_type, event_list in events.items():
        if not isinstance(event_list, list):
            continue
        event_allowed_fields = comparable_fields.get(str(event_type), set())
        items = [
            _json_key(_normalize_event(event, ignored, event_allowed_fields))
            for event in event_list
            if isinstance(event, dict)
        ]
        canonical[str(event_type)] = sorted(items)
    return _json_key(canonical)


def _normalize_event(
    event: dict[str, Any],
    ignored: set[str],
    allowed_fields: set[str] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in sorted(event.items()):
        key_str = str(key)
        if key_str in ignored:
            continue
        if allowed_fields is not None:
            if key_str not in allowed_fields:
                continue
            out[key_str] = _normalize_value(value, ignored, keep_strings=True)
            continue
        normalized = _normalize_value(value, ignored, keep_strings=False)
        if normalized is not _IGNORE:
            out[key_str] = normalized
    return out


def _normalize_value(value: Any, ignored: set[str], *, keep_strings: bool) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, subvalue in sorted(value.items()):
            key_str = str(key)
            if key_str in ignored:
                continue
            normalized = _normalize_value(subvalue, ignored, keep_strings=keep_strings)
            if normalized is not _IGNORE:
                out[key_str] = normalized
        return out if out else _IGNORE
    if isinstance(value, list):
        normalized = [
            item
            for item in (
                _normalize_value(item, ignored, keep_strings=keep_strings)
                for item in value
            )
            if item is not _IGNORE
        ]
        return sorted(normalized, key=_json_key) if normalized else _IGNORE
    if isinstance(value, str):
        return value if keep_strings else _IGNORE
    if value is None:
        return _IGNORE
    return value


def _json_key(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))
