"""Shared review/adjudication slot construction helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

PROVENANCE_FIELD_NAMES = frozenset({"evidence", "review_anchor"})


@dataclass(frozen=True)
class Slot:
    """A reviewable event anchored to a stable within-note slot."""

    event_key: str
    event_type: str
    note_id: str
    fingerprint: str
    fields: dict[str, Any]

    def as_package_event(self) -> dict[str, Any]:
        """Serialize the slot in the review-package event shape."""
        return {
            "event_key": self.event_key,
            "event_type": self.event_type,
            "note_id": self.note_id,
            "fingerprint": self.fingerprint,
            "fields": self.fields,
        }


@dataclass
class SlotState:
    """Mutable counters used to make slot keys unique across a batch."""

    per_type_count: dict[tuple[str, str], int] = field(default_factory=dict)
    seen_keys: dict[str, int] = field(default_factory=dict)


def events_from_record(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Flatten a record's reviewable ``events`` block.

    Only the ``events`` block is reviewed; ``plans`` are orientation/control
    output and intentionally skipped.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    events = record.get("events") or {}
    if not isinstance(events, dict):
        return out
    for event_type, event_list in events.items():
        if not isinstance(event_list, list):
            continue
        for event in event_list:
            if isinstance(event, dict):
                out.append((str(event_type), event))
    return out


def event_content_hash(event: dict[str, Any]) -> str:
    """Short content hash of an event's field values."""
    payload = json.dumps(
        data_fields(event), sort_keys=True, default=str, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def data_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Return extracted-value fields, excluding review provenance metadata."""
    return {k: v for k, v in fields.items() if k not in PROVENANCE_FIELD_NAMES}


def record_to_slots(
    record: dict[str, Any],
    field_schema: dict[str, dict[str, Any]],
    *,
    state: SlotState | None = None,
) -> list[Slot]:
    """Convert one raw extraction record into review/adjudication slots.

    The slot key is ``<note_id>::<event_type>::<identity>`` where ``identity`` is
    derived from declared ``event_identity_fields`` when available, else from
    the event's ordinal for its type within the note. Passing a shared
    :class:`SlotState` preserves duplicate-key suffixing across multiple records.
    """
    slot_state = state or SlotState()
    note_id = str(record.get("note_id") or "")
    slots: list[Slot] = []
    for event_type, event in events_from_record(record):
        identity_fields = _identity_fields_for(field_schema, event_type)
        ordinal = slot_state.per_type_count.get((note_id, event_type), 0) + 1
        slot_state.per_type_count[(note_id, event_type)] = ordinal
        base_key = f"{note_id}::{event_type}::{_slot_identity(event, identity_fields, ordinal)}"
        duplicate_count = slot_state.seen_keys.get(base_key, 0)
        slot_state.seen_keys[base_key] = duplicate_count + 1
        event_key = (
            base_key if duplicate_count == 0 else f"{base_key}::{duplicate_count}"
        )
        slots.append(
            Slot(
                event_key=event_key,
                event_type=event_type,
                note_id=note_id,
                fingerprint=event_content_hash(event),
                fields=event,
            )
        )
    return slots


def slot_from_package_event(event: dict[str, Any]) -> Slot:
    """Rehydrate a review-package event into the shared slot shape."""
    event_key = str(event.get("event_key") or "")
    fields = event.get("fields") or {}
    if not isinstance(fields, dict):
        raise TypeError(
            f"Review package event {event_key!r} field 'fields' must be an object"
        )
    fingerprint = str(event.get("fingerprint") or event_content_hash(fields))
    return Slot(
        event_key=event_key,
        event_type=str(event.get("event_type") or ""),
        note_id=str(event.get("note_id") or ""),
        fingerprint=fingerprint,
        fields=dict(fields),
    )


def _identity_fields_for(
    field_schema: dict[str, dict[str, Any]], event_type: str
) -> Sequence[str]:
    spec = field_schema.get(event_type) or {}
    fields = spec.get("event_identity_fields") or ()
    if isinstance(fields, list | tuple):
        return fields
    return ()


def _slot_identity(
    event: dict[str, Any], identity_fields: Sequence[str], ordinal: int
) -> str:
    if identity_fields and all(event.get(f) is not None for f in identity_fields):
        payload = json.dumps(
            {f: event.get(f) for f in identity_fields},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return "id-" + hashlib.sha256(payload.encode()).hexdigest()[:12]
    return str(ordinal)
