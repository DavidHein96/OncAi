"""Build two-batch adjudication packages from FC extraction outputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oncai.config import OncaiConfig
from oncai.review.notes import load_source_notes
from oncai.review.package import _read_manifest
from oncai.review.schema import adjudication_hash
from oncai.review.select import (
    comparable_fields_from_field_schema,
    note_id_for_record,
    successful_records_from_jsonl,
)
from oncai.review.slots import Slot, SlotState, data_fields, record_to_slots

ADJUDICATION_DESCRIPTOR = "adjudication.json"
ADJUDICATION_PACKAGE_SUFFIX = ".adjudication_pkg.json"
ADJUDICATION_LOG_SUFFIX = ".adjudications.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_key(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))


def _short_hash(value: Any) -> str:
    return hashlib.sha256(_json_key(value).encode()).hexdigest()[:12]


def _patient_key(record: dict[str, Any] | None, note_id: str) -> str:
    if record is not None:
        mrn = record.get("mrn")
        if mrn is not None and str(mrn).strip():
            return str(mrn)
    return note_id


def _records_by_note_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        note_id = note_id_for_record(record)
        if note_id:
            out[note_id] = record
    return out


def _slots_by_key(
    records: Iterable[dict[str, Any]], field_schema: dict[str, dict[str, Any]]
) -> dict[str, Slot]:
    state = SlotState()
    out: dict[str, Slot] = {}
    for record in records:
        for slot in record_to_slots(record, field_schema, state=state):
            out[slot.event_key] = slot
    return out


def _comparison_value(
    slot: Slot | None,
    comparable_fields: dict[str, set[str]],
) -> dict[str, Any] | None:
    if slot is None:
        return None
    allowed = comparable_fields.get(slot.event_type, set())
    fields = data_fields(slot.fields)
    return {name: fields.get(name) for name in sorted(allowed)}


def _comparison_status(left: dict[str, Any] | None, right: dict[str, Any] | None) -> str:
    if left is None:
        return "missing_left"
    if right is None:
        return "missing_right"
    return "matched" if left == right else "different"


def _source_note_config(jsonl_path: Path) -> dict[str, Any]:
    manifest = _read_manifest(jsonl_path)
    config = manifest.get("config") or {}
    return {
        "db_path": Path(config["db_path"]) if config.get("db_path") else None,
        "source_table": config.get("source_table"),
        "text_col": config.get("text_col") or "report_text",
        "id_col": config.get("id_col") or "report_id",
    }


def _load_notes_for_package(
    *,
    left_jsonl: Path,
    right_jsonl: Path,
    note_ids: set[str],
    db_path: Path | None,
) -> dict[str, dict[str, Any]]:
    if not note_ids:
        return {}
    for jsonl_path in (left_jsonl, right_jsonl):
        source = _source_note_config(jsonl_path)
        resolved_db = db_path or source["db_path"]
        if resolved_db is None or not resolved_db.exists():
            continue
        notes = load_source_notes(
            source_table=source["source_table"],
            db_path=resolved_db,
            note_ids=note_ids,
            text_col=source["text_col"],
            id_col=source["id_col"],
        )
        if notes:
            return notes
    return {}


def build_adjudication_package(
    *,
    round_name: str,
    definition_name: str,
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    left_batch: str,
    right_batch: str,
    left_label: str,
    right_label: str,
    left_jsonl: Path,
    right_jsonl: Path,
    field_schema: dict[str, dict[str, Any]],
    notes: dict[str, dict[str, Any]],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a two-batch adjudication package dict."""
    comparable_fields = comparable_fields_from_field_schema(field_schema)
    if not comparable_fields:
        raise ValueError(
            "No comparable fields are configured. Register event_identity_fields "
            "and/or comparison_fields for adjudication."
        )

    left_by_note = _records_by_note_id(left_records)
    right_by_note = _records_by_note_id(right_records)
    left_slots = _slots_by_key(left_records, field_schema)
    right_slots = _slots_by_key(right_records, field_schema)

    patients: dict[str, dict[str, Any]] = {}
    disagreement_counts = {
        "different": 0,
        "missing_left": 0,
        "missing_right": 0,
    }

    for event_key in sorted(set(left_slots) | set(right_slots)):
        left_slot = left_slots.get(event_key)
        right_slot = right_slots.get(event_key)
        exemplar = left_slot or right_slot
        if exemplar is None:
            continue

        left_value = _comparison_value(left_slot, comparable_fields)
        right_value = _comparison_value(right_slot, comparable_fields)
        status = _comparison_status(left_value, right_value)
        if status == "matched":
            continue
        disagreement_counts[status] += 1

        note_id = exemplar.note_id
        record = left_by_note.get(note_id) or right_by_note.get(note_id)
        patient_key = _patient_key(record, note_id)
        patient = patients.setdefault(
            patient_key, {"mrn": patient_key, "notes": {}, "events": []}
        )
        if note_id and note_id not in patient["notes"]:
            patient["notes"][note_id] = notes.get(note_id) or {
                "note_text": None,
                "mrn": record.get("mrn") if record is not None else None,
                "note_date": None,
                "note_type": None,
                "department": None,
            }

        patient["events"].append(
            {
                "adjudication_key": event_key,
                "event_key": event_key,
                "event_type": exemplar.event_type,
                "note_id": note_id,
                "status": status,
                "comparison": {
                    left_label: left_value,
                    right_label: right_value,
                    "hash": {
                        left_label: _short_hash(left_value),
                        right_label: _short_hash(right_value),
                    },
                },
                "left": left_slot.as_package_event() if left_slot is not None else None,
                "right": (
                    right_slot.as_package_event() if right_slot is not None else None
                ),
            }
        )

    generated = generated_at or _utc_now_iso()
    return {
        "package_type": "adjudication",
        "round": round_name,
        "definition_name": definition_name,
        "generated_at": generated,
        "adjudication_hash": adjudication_hash(definition_name, field_schema),
        "inputs": {
            "left": {
                "label": left_label,
                "batch": left_batch,
                "jsonl_path": str(left_jsonl),
            },
            "right": {
                "label": right_label,
                "batch": right_batch,
                "jsonl_path": str(right_jsonl),
            },
        },
        "field_schema": field_schema,
        "summary": {
            "left_records": len(left_records),
            "right_records": len(right_records),
            "disagreements": sum(disagreement_counts.values()),
            **disagreement_counts,
        },
        "patients": list(patients.values()),
    }


def default_adjudication_dir(config: OncaiConfig, round_name: str) -> Path:
    """Canonical inbox directory for an adjudication round."""
    return config.inbox_path / "fc_adjudications" / round_name


def write_adjudication_package(package: dict[str, Any], output_path: Path) -> Path:
    """Write an adjudication package JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(package, indent=2, default=str) + "\n")
    return output_path


def write_adjudication_descriptor(
    *,
    package: dict[str, Any],
    output_dir: Path,
    package_path: Path,
) -> Path:
    """Write the round descriptor that pins adjudication inputs."""
    descriptor = {
        "round": package["round"],
        "definition_name": package["definition_name"],
        "generated_at": package["generated_at"],
        "adjudication_hash": package["adjudication_hash"],
        "inputs": package["inputs"],
        "package_path": str(package_path),
        "summary": package["summary"],
    }
    path = output_dir / ADJUDICATION_DESCRIPTOR
    path.write_text(json.dumps(descriptor, indent=2, default=str) + "\n")
    return path


def package_from_jsonls(
    *,
    config: OncaiConfig,
    round_name: str,
    definition_name: str,
    left_jsonl: Path,
    right_jsonl: Path,
    field_schema: dict[str, dict[str, Any]],
    left_label: str = "left",
    right_label: str = "right",
    db_path: Path | None = None,
    output_dir: Path | None = None,
    generated_at: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Build and write a two-batch adjudication package from JSONL batches."""
    left_records = successful_records_from_jsonl(left_jsonl)
    right_records = successful_records_from_jsonl(right_jsonl)
    note_ids = set(_records_by_note_id(left_records)) | set(
        _records_by_note_id(right_records)
    )
    notes = _load_notes_for_package(
        left_jsonl=left_jsonl,
        right_jsonl=right_jsonl,
        note_ids=note_ids,
        db_path=db_path,
    )

    package = build_adjudication_package(
        round_name=round_name,
        definition_name=definition_name,
        left_records=left_records,
        right_records=right_records,
        left_batch=left_jsonl.stem,
        right_batch=right_jsonl.stem,
        left_label=left_label,
        right_label=right_label,
        left_jsonl=left_jsonl,
        right_jsonl=right_jsonl,
        field_schema=field_schema,
        notes=notes,
        generated_at=generated_at,
    )

    out_dir = output_dir or default_adjudication_dir(config, round_name)
    pkg_path = out_dir / f"{round_name}{ADJUDICATION_PACKAGE_SUFFIX}"
    write_adjudication_package(package, pkg_path)
    descriptor_path = write_adjudication_descriptor(
        package=package,
        output_dir=out_dir,
        package_path=pkg_path,
    )
    return pkg_path, descriptor_path, package
