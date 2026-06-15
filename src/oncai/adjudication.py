"""Build two-batch adjudication packages from FC extraction outputs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from oncai.config import OncaiConfig
from oncai.fc_extraction.load import _clean_event
from oncai.hashing import blake2b_128
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

_LEFT_DECISION = "left"
_RIGHT_DECISION = "right"
_CUSTOM_DECISION = "custom"
_EXCLUDE_DECISION = "exclude"
_VALID_DECISIONS = {
    _LEFT_DECISION,
    _RIGHT_DECISION,
    _CUSTOM_DECISION,
    _EXCLUDE_DECISION,
}

_ADJUDICATED_SCHEMA = {
    "adjudication_key": pl.String,
    "event_key": pl.String,
    "event_type": pl.String,
    "tool_name": pl.String,
    "note_id": pl.String,
    "mrn": pl.String,
    "definition_name": pl.String,
    "round_name": pl.String,
    "package_generated_at": pl.String,
    "adjudication_status": pl.String,
    "adjudication_decision": pl.String,
    "selected_side": pl.String,
    "reviewer": pl.String,
    "reviewed_at": pl.String,
    "adjudication_comment": pl.String,
    "left_label": pl.String,
    "left_batch": pl.String,
    "right_label": pl.String,
    "right_batch": pl.String,
    "note_date": pl.String,
    "note_type": pl.String,
    "department": pl.String,
    "left_fields_json": pl.String,
    "right_fields_json": pl.String,
    "comparison_json": pl.String,
    "adjudicated_fields_json": pl.String,
    "key_hash": pl.Binary,
    "content_hash": pl.Binary,
}


@dataclass
class AdjudicationLoadResult:
    """Result of package + adjudication-log conversion."""

    total_events: int = 0
    decided_events: int = 0
    written_events: int = 0
    excluded_events: int = 0
    unreviewed_events: int = 0
    ignored_adjudications: int = 0
    output_path: Path | None = None
    df: pl.DataFrame | None = None

    @property
    def written(self) -> int:
        """Rows written to the adjudicated parquet."""
        return self.written_events


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


def adjudication_round_name(path: Path) -> str:
    """Return the shared round name for an adjudication package or log path."""
    name = path.name
    if name.endswith(ADJUDICATION_PACKAGE_SUFFIX):
        return name[: -len(ADJUDICATION_PACKAGE_SUFFIX)]
    if name.endswith(ADJUDICATION_LOG_SUFFIX):
        return name[: -len(ADJUDICATION_LOG_SUFFIX)]
    raise ValueError(
        f"Adjudication files must end in {ADJUDICATION_PACKAGE_SUFFIX!r} or "
        f"{ADJUDICATION_LOG_SUFFIX!r}: {path.name}"
    )


def _load_package(package_path: Path) -> dict[str, Any]:
    try:
        package = json.loads(package_path.read_text())
    except json.JSONDecodeError as exc:
        msg = f"Invalid adjudication package JSON in {package_path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(package, dict):
        msg = f"Adjudication package must be a JSON object: {package_path}"
        raise TypeError(msg)
    if package.get("package_type") != "adjudication":
        msg = f"Package is not an adjudication package: {package_path}"
        raise ValueError(msg)
    return package


def _reviewed_at(record: dict[str, Any]) -> str:
    return str(record.get("reviewed_at") or "")


def _record_key(record: dict[str, Any]) -> str:
    return str(record.get("adjudication_key") or record.get("event_key") or "")


def _load_adjudications(log_path: Path) -> dict[str, dict[str, Any]]:
    adjudications: dict[str, dict[str, Any]] = {}
    for line_no, raw_line in enumerate(log_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {log_path} line {line_no}: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(record, dict):
            msg = f"Adjudication line {line_no} in {log_path} is not a JSON object"
            raise TypeError(msg)
        key = _record_key(record)
        if not key:
            msg = f"Adjudication line {line_no} in {log_path} is missing key"
            raise ValueError(msg)
        existing = adjudications.get(key)
        if existing is None or _reviewed_at(record) >= _reviewed_at(existing):
            adjudications[key] = record
    return adjudications


def _iter_package_events(
    package: dict[str, Any],
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    patients = package.get("patients") or []
    if not isinstance(patients, list):
        raise TypeError("Adjudication package field 'patients' must be a list")

    seen: set[str] = set()
    for patient_index, patient in enumerate(patients):
        if not isinstance(patient, dict):
            raise TypeError(
                f"Adjudication package patient #{patient_index} is not an object"
            )
        events = patient.get("events") or []
        if not isinstance(events, list):
            raise TypeError(
                f"Adjudication package patient #{patient_index} field 'events' "
                "must be a list"
            )
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                raise TypeError(
                    f"Adjudication package patient #{patient_index} event "
                    f"#{event_index} is not an object"
                )
            key = str(event.get("adjudication_key") or event.get("event_key") or "")
            if not key:
                raise ValueError(
                    f"Adjudication package patient #{patient_index} event "
                    f"#{event_index} is missing adjudication_key"
                )
            if key in seen:
                raise ValueError(f"Adjudication package has duplicate key: {key}")
            seen.add(key)
            yield patient, event


def _note_metadata(patient: dict[str, Any], note_id: str) -> dict[str, Any]:
    notes = patient.get("notes") or {}
    if not isinstance(notes, dict):
        return {}
    note = notes.get(note_id)
    return note if isinstance(note, dict) else {}


def _side_fields(event: dict[str, Any], side: str) -> dict[str, Any] | None:
    item = event.get(side)
    if not isinstance(item, dict):
        return None
    fields = item.get("fields") or {}
    if not isinstance(fields, dict):
        return None
    return data_fields(dict(fields))


def _adjudicated_fields(
    event: dict[str, Any], record: dict[str, Any], decision: str
) -> dict[str, Any] | None:
    if decision == _EXCLUDE_DECISION:
        return None
    if decision in {_LEFT_DECISION, _RIGHT_DECISION}:
        fields = _side_fields(event, decision)
        if fields is None:
            key = str(event.get("adjudication_key") or event.get("event_key") or "")
            raise ValueError(
                f"Adjudication {key!r} chose missing side {decision!r}"
            )
        return fields
    fields = record.get("adjudicated_fields")
    if fields is None:
        fields = record.get("fields")
    if not isinstance(fields, dict):
        key = str(event.get("adjudication_key") or event.get("event_key") or "")
        raise TypeError(
            f"Custom adjudication {key!r} must include adjudicated_fields"
        )
    return data_fields(dict(fields))


def _input_info(package: dict[str, Any], side: str) -> dict[str, Any]:
    inputs = package.get("inputs") or {}
    item = inputs.get(side) if isinstance(inputs, dict) else {}
    return item if isinstance(item, dict) else {}


def _adjudication_row(
    *,
    package: dict[str, Any],
    patient: dict[str, Any],
    event: dict[str, Any],
    record: dict[str, Any],
    fields: dict[str, Any],
) -> dict[str, Any]:
    adjudication_key = str(event.get("adjudication_key") or event.get("event_key"))
    event_key = str(event.get("event_key") or adjudication_key)
    event_type = str(event.get("event_type") or record.get("event_type") or "")
    note_id = str(event.get("note_id") or record.get("note_id") or "")
    note = _note_metadata(patient, note_id)
    left = _input_info(package, "left")
    right = _input_info(package, "right")
    left_fields = _side_fields(event, "left")
    right_fields = _side_fields(event, "right")
    adjudicated_fields_json = _json_key(fields)

    row = _clean_event(fields)
    row.update(
        {
            "adjudication_key": adjudication_key,
            "event_key": event_key,
            "event_type": event_type,
            "tool_name": event_type,
            "note_id": note_id,
            "mrn": str(patient.get("mrn") or record.get("mrn") or ""),
            "definition_name": str(package.get("definition_name") or ""),
            "round_name": str(package.get("round") or ""),
            "package_generated_at": package.get("generated_at"),
            "adjudication_status": str(event.get("status") or ""),
            "adjudication_decision": str(record.get("decision") or ""),
            "selected_side": str(record.get("selected_side") or ""),
            "reviewer": str(record.get("reviewer") or ""),
            "reviewed_at": record.get("reviewed_at"),
            "adjudication_comment": str(record.get("comment") or ""),
            "left_label": str(left.get("label") or "left"),
            "left_batch": str(left.get("batch") or ""),
            "right_label": str(right.get("label") or "right"),
            "right_batch": str(right.get("batch") or ""),
            "note_date": note.get("note_date"),
            "note_type": note.get("note_type"),
            "department": note.get("department"),
            "left_fields_json": _json_key(left_fields),
            "right_fields_json": _json_key(right_fields),
            "comparison_json": _json_key(event.get("comparison") or {}),
            "adjudicated_fields_json": adjudicated_fields_json,
            "key_hash": blake2b_128(adjudication_key),
            "content_hash": blake2b_128(
                _json_key(
                    {
                        "fields": fields,
                        "decision": record.get("decision"),
                        "selected_side": record.get("selected_side"),
                        "comment": record.get("comment"),
                        "reviewer": record.get("reviewer"),
                        "reviewed_at": record.get("reviewed_at"),
                    }
                )
            ),
        }
    )
    return row


def _empty_adjudicated_df() -> pl.DataFrame:
    return pl.DataFrame(schema=_ADJUDICATED_SCHEMA)


def _rows_to_adjudicated_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return _empty_adjudicated_df()

    df = pl.DataFrame(rows, infer_schema_length=None)
    for col, dtype in _ADJUDICATED_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    casts = [
        pl.col(col).cast(dtype, strict=False)
        for col, dtype in _ADJUDICATED_SCHEMA.items()
        if col in df.columns
    ]
    if casts:
        df = df.with_columns(casts)

    ordered = list(_ADJUDICATED_SCHEMA) + [
        c for c in df.columns if c not in _ADJUDICATED_SCHEMA
    ]
    return df.select(ordered).sort("adjudication_key")


def adjudication_to_df(
    package_path: Path,
    log_path: Path,
    *,
    require_complete: bool = True,
) -> AdjudicationLoadResult:
    """Build adjudicated event rows from a package plus decision log."""
    package = _load_package(package_path)
    adjudications = _load_adjudications(log_path)
    result = AdjudicationLoadResult(output_path=None)
    rows: list[dict[str, Any]] = []
    package_keys: set[str] = set()

    for patient, event in _iter_package_events(package):
        key = str(event.get("adjudication_key") or event.get("event_key") or "")
        package_keys.add(key)
        result.total_events += 1
        record = adjudications.get(key)
        if record is None:
            result.unreviewed_events += 1
            continue

        decision = str(record.get("decision") or "").strip().lower()
        if decision not in _VALID_DECISIONS:
            raise ValueError(
                f"Adjudication for {key!r} has unsupported decision {decision!r}; "
                f"expected one of {sorted(_VALID_DECISIONS)}"
            )
        record = dict(record)
        record["decision"] = decision
        result.decided_events += 1

        fields = _adjudicated_fields(event, record, decision)
        if fields is None:
            result.excluded_events += 1
            continue
        rows.append(
            _adjudication_row(
                package=package,
                patient=patient,
                event=event,
                record=record,
                fields=fields,
            )
        )
        result.written_events += 1

    result.ignored_adjudications = len(set(adjudications) - package_keys)
    if result.unreviewed_events and require_complete:
        raise ValueError(
            f"{package_path.name} has {result.unreviewed_events} unreviewed "
            "adjudication item(s); finish the round before loading a table"
        )

    result.df = _rows_to_adjudicated_df(rows)
    return result


def adjudication_to_parquet(
    package_path: Path,
    log_path: Path,
    output_path: Path,
    *,
    require_complete: bool = True,
    dry_run: bool = False,
) -> AdjudicationLoadResult:
    """Convert a completed adjudication package + decision log to parquet."""
    result = adjudication_to_df(
        package_path,
        log_path,
        require_complete=require_complete,
    )
    result.output_path = output_path
    if dry_run or result.df is None:
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    result.df.write_parquet(tmp, compression="zstd")
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.replace(output_path)
    return result
