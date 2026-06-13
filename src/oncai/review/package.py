"""Assemble a ``*.review_pkg.json`` from a completed extraction batch.

This is the producer side of the contract documented in
``apps/review_app/server.py``: *"Reads a ``*.review_pkg.json`` produced by
``oncai.review.package``."* The local physician-review server consumes exactly
the JSON this module writes.

What a package is
-----------------
One self-contained JSON file holding everything the review app needs — no
DuckDB, no lake, no repo — so it can be handed to a collaborator who only has
the frozen review app:

    {
      "definition_name": "PathKidneyBasic",
      "batch": "v1",
      "generated_at": "2026-06-07T12:00:00+00:00",
      "adjudication_hash": "9f2c…",   # comparability gate (see review.schema)
      "field_schema": { <event_type>: {label, fields[]} },   # how to render
      "patients": [
        {
          "mrn": "M0001",
          "notes": { <note_id>: {note_text, note_date, note_type, department} },
          "events": [
            {"event_key", "event_type", "note_id", "fingerprint", "fields": {...}}, ...
          ]
        }, ...
      ]
    }

Events are grouped by patient (MRN, falling back to note_id when a batch has no
MRN) because that's the review unit — a physician adjudicates one patient's
findings at a time. ``plans`` (orientation tools) are excluded; only the
``events`` block is reviewed.

Entry points
------------
- :func:`package_from_jsonl` — build from a batch JSONL with an explicit
  registry + source. Used by the auto-build hook at the end of
  ``oncai fc run-single`` (registry and source are already in hand there).
- :func:`package_from_batch` — build from a batch JSONL alone, reading its
  ``_manifest.json`` sidecar for the source/columns. The "package an
  already-completed run" path (``oncai fc review-package``).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .notes import load_source_notes
from .schema import (
    adjudication_hash,
    build_field_schema,
    infer_field_schema_from_events,
    merge_inferred_schema,
)
from .select import (
    note_id_for_record,
    record_has_review_flag,
    successful_records_from_jsonl,
)
from .slots import SlotState, events_from_record, record_to_slots

if TYPE_CHECKING:
    from oncai.fc_extraction.tools import ToolRegistry

logger = logging.getLogger(__name__)

_PACKAGE_SUFFIX = ".review_pkg.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _patient_key(record: dict[str, Any]) -> str:
    """The grouping key for a record: its MRN, or the note_id when MRN-less.

    Some batches (notably ``--jsonl`` sources without an ``mrn`` field) carry no
    MRN. Grouping by note_id then makes each report its own "patient", which the
    app handles fine — better than collapsing everything under one null key.
    """
    mrn = record.get("mrn")
    if mrn is not None and str(mrn).strip():
        return str(mrn)
    return str(record.get("note_id") or "")


def build_review_package(
    *,
    definition_name: str,
    batch: str,
    records: list[dict[str, Any]],
    field_schema: dict[str, dict[str, Any]],
    notes: dict[str, dict[str, Any]],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the review-package dict (pure — no I/O).

    ``records`` are the per-note JSONL records (successful ones; callers filter).
    ``field_schema`` says how to render each event type; ``notes`` maps note_id
    to the source text/metadata. Events are grouped into patients and each is
    given an identity-addressed ``event_key`` of
    ``"<note_id>::<event_type>::<identity>"`` — the declared
    ``event_identity_fields`` where present, else the finding's ordinal for its
    type in the report. This is the *slot* a verdict attaches to and the key that
    aligns the same finding across runs for adjudication. Each event also carries
    a ``fingerprint`` (a hash of its field values) for value-change detection and
    cross-run agreement.
    """
    # event_type counts seen in the data, so the schema can be backfilled for
    # any type the registry didn't cover.
    seen_events: dict[str, list[dict[str, Any]]] = {}

    # Preserve first-seen patient order for a stable sidebar.
    patients: dict[str, dict[str, Any]] = {}
    # Identity-addressed slot keys: declared event_identity_fields where present,
    # else the finding's ordinal for its type in the report. ``slot_state``
    # assigns those ordinals and disambiguates rare identity clashes with a
    # ``::<n>`` suffix.
    slot_state = SlotState()
    for record in records:
        note_id = str(record.get("note_id") or "")
        key = _patient_key(record)
        patient = patients.setdefault(key, {"mrn": key, "notes": {}, "events": []})

        if note_id and note_id not in patient["notes"]:
            patient["notes"][note_id] = notes.get(note_id) or {
                "note_text": None,
                "mrn": record.get("mrn"),
                "note_date": None,
                "note_type": None,
                "department": None,
            }

        for slot in record_to_slots(record, field_schema, state=slot_state):
            seen_events.setdefault(slot.event_type, []).append(slot.fields)
            patient["events"].append(slot.as_package_event())

    # Backfill schema for any event type present in the data but not the registry.
    field_schema = merge_inferred_schema(field_schema, seen_events)

    return {
        "definition_name": definition_name,
        "batch": batch,
        "generated_at": generated_at or _utc_now_iso(),
        # Comparability gate: two packages are adjudicable only if these match.
        "adjudication_hash": adjudication_hash(definition_name, field_schema),
        "field_schema": field_schema,
        "patients": list(patients.values()),
    }


def write_review_package(package: dict[str, Any], output_path: Path) -> Path:
    """Write ``package`` to ``output_path`` as JSON. Returns the path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(package, fh, indent=2, default=str)
    return output_path


def default_package_path(jsonl_path: Path) -> Path:
    """Where the package lands by default: ``<batch>.review_pkg.json`` beside it."""
    return jsonl_path.with_name(_default_package_batch(jsonl_path) + _PACKAGE_SUFFIX)


def _default_package_batch(jsonl_path: Path) -> str:
    stem = jsonl_path.stem
    if stem.isdigit() and jsonl_path.parent.name:
        return f"{jsonl_path.parent.name}.{stem}"
    return stem


def _successful_records(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load the success-only records from a batch JSONL (failed rows are noise)."""
    return successful_records_from_jsonl(jsonl_path)


def _filter_records_for_review(
    records: list[dict[str, Any]],
    *,
    note_ids: set[str] | None = None,
    only_flagged: bool = False,
) -> list[dict[str, Any]]:
    """Apply package-level report selection before building the review bundle."""
    selected = records
    if note_ids is not None:
        selected = [r for r in selected if note_id_for_record(r) in note_ids]
    if only_flagged:
        selected = [r for r in selected if record_has_review_flag(r)]
    return selected


def package_from_jsonl(
    *,
    jsonl_path: Path,
    registry: ToolRegistry | None = None,
    definition_name: str | None = None,
    batch: str | None = None,
    db_path: Path | None = None,
    source_table: str | None = None,
    text_col: str = "report_text",
    id_col: str = "report_id",
    output_path: Path | None = None,
    generated_at: str | None = None,
    note_ids: set[str] | None = None,
    only_flagged: bool = False,
) -> Path:
    """Build and write a review package from a batch JSONL.

    The registry-aware path. ``definition_name`` / ``batch`` default to the
    record's ``definition_name`` field and the JSONL stem. Source notes are
    loaded from ``source_table`` (+ ``db_path``); when neither a registry nor a
    source is available the package still builds with an inferred schema and no
    note text.
    """
    jsonl_path = Path(jsonl_path)
    records = _filter_records_for_review(
        _successful_records(jsonl_path),
        note_ids=note_ids,
        only_flagged=only_flagged,
    )

    if definition_name is None:
        definition_name = next(
            (
                str(r.get("definition_name"))
                for r in records
                if r.get("definition_name")
            ),
            "",
        )
    if batch is None:
        batch = _default_package_batch(jsonl_path)

    note_ids = {str(r.get("note_id") or "") for r in records}
    notes = load_source_notes(
        source_table=source_table,
        db_path=db_path,
        note_ids=note_ids,
        text_col=text_col,
        id_col=id_col,
    )

    if registry is not None:
        field_schema = build_field_schema(registry)
    else:
        # No registry — infer everything from the observed events.
        seen: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            for event_type, event in events_from_record(record):
                seen.setdefault(event_type, []).append(event)
        field_schema = infer_field_schema_from_events(seen)

    package = build_review_package(
        definition_name=definition_name,
        batch=batch,
        records=records,
        field_schema=field_schema,
        notes=notes,
        generated_at=generated_at,
    )

    out = Path(output_path) if output_path else default_package_path(jsonl_path)
    write_review_package(package, out)
    logger.info("Wrote review package: %s (%d patients)", out, len(package["patients"]))
    return out


def _read_manifest(jsonl_path: Path) -> dict[str, Any]:
    """Read the ``<batch>_manifest.json`` sidecar, or ``{}`` if absent/bad."""
    manifest_path = jsonl_path.with_name(jsonl_path.stem + "_manifest.json")
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open() as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read manifest %s: %s", manifest_path, exc)
        return {}


def package_from_batch(
    *,
    jsonl_path: Path,
    registry: ToolRegistry | None = None,
    db_path: Path | None = None,
    output_path: Path | None = None,
    note_ids: set[str] | None = None,
    only_flagged: bool = False,
) -> Path:
    """Build a review package for an already-completed batch.

    Reads the batch's ``_manifest.json`` sidecar to recover where the notes came
    from (``source_table``, ``text_col``, ``id_col``, ``db_path``) so the caller
    only has to point at the JSONL. ``db_path`` overrides the manifest's when the
    DB has since moved; ``registry``, when supplied, gives the richer
    schema (enum options, required-ness) the manifest can't.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Batch JSONL not found: {jsonl_path}")

    manifest = _read_manifest(jsonl_path)
    config = manifest.get("config") or {}
    source_table = config.get("source_table")
    text_col = config.get("text_col") or "report_text"
    id_col = config.get("id_col") or "report_id"
    resolved_db = db_path or (
        Path(config["db_path"]) if config.get("db_path") else None
    )

    return package_from_jsonl(
        jsonl_path=jsonl_path,
        registry=registry,
        definition_name=manifest.get("definition_name")
        or manifest.get("workflow_name"),
        batch=manifest.get("batch_name") or jsonl_path.stem,
        db_path=resolved_db,
        source_table=source_table,
        text_col=text_col,
        id_col=id_col,
        output_path=output_path,
        note_ids=note_ids,
        only_flagged=only_flagged,
    )
