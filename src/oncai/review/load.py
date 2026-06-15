"""Load completed review-app sidecars into reviewed (silver) extraction parquets.

The output of this module is the **silver** layer: human-adjudicated, event-grain
rows. It is validated and trustworthy but still sparse — a definition's distinct
event tools share one wide table. Reshaping silver into dense, per-concept
**gold** tables is a separate SQL step: a batch-local
``inbox/fc_reviews/<batch>/<batch>.sql`` sidecar, mirrored to the lake and run
by ``oncai build-db`` against that batch's ``extractions_silver.<batch>`` table
(see ``docs/review_system.md``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from oncai.fc_extraction.load import _clean_event
from oncai.hashing import blake2b_128
from oncai.review.select import successful_records_from_jsonl
from oncai.review.slots import (
    Slot,
    SlotState,
    data_fields,
    record_to_slots,
    slot_from_package_event,
)

REVIEW_PACKAGE_SUFFIX = ".review_pkg.json"
REVIEW_LOG_SUFFIX = ".reviews.jsonl"

_APPROVED_VERDICT = "approved"
_REJECTED_VERDICT = "rejected"
_VALID_VERDICTS = {_APPROVED_VERDICT, _REJECTED_VERDICT}
_FLAGGED_REVIEWED = "flagged_reviewed"
_UNFLAGGED_AUTOACCEPT = "unflagged_autoaccept"
_REVIEWER_ADDED = "reviewer_added"

_SILVER_SCHEMA = {
    "event_key": pl.String,
    "event_type": pl.String,
    "tool_name": pl.String,
    "note_id": pl.String,
    "mrn": pl.String,
    "definition_name": pl.String,
    "batch_name": pl.String,
    "package_generated_at": pl.String,
    "review_verdict": pl.String,
    "reviewer": pl.String,
    "reviewed_at": pl.String,
    "review_comment": pl.String,
    "acceptance_reason": pl.String,
    "note_date": pl.String,
    "note_type": pl.String,
    "department": pl.String,
    "original_fields_json": pl.String,
    "edits_json": pl.String,
    "reviewed_fields_json": pl.String,
    "key_hash": pl.Binary,
    "content_hash": pl.Binary,
}


@dataclass
class ReviewLoadResult:
    """Result of a review package + reviews log conversion."""

    total_events: int = 0
    reviewed_events: int = 0
    approved_events: int = 0
    auto_accepted_events: int = 0
    rejected_events: int = 0
    unreviewed_events: int = 0
    ignored_reviews: int = 0
    output_path: Path | None = None
    df: pl.DataFrame | None = None

    @property
    def written(self) -> int:
        """Rows written to the silver parquet."""
        return self.approved_events + self.auto_accepted_events

    @property
    def skipped(self) -> int:
        """Reviewed or package events intentionally excluded from silver."""
        return self.rejected_events + self.unreviewed_events


def review_batch_name(path: Path) -> str:
    """Return the shared batch name for a review package or review log path."""
    name = path.name
    if name.endswith(REVIEW_PACKAGE_SUFFIX):
        return name[: -len(REVIEW_PACKAGE_SUFFIX)]
    if name.endswith(REVIEW_LOG_SUFFIX):
        return name[: -len(REVIEW_LOG_SUFFIX)]
    raise ValueError(
        f"Review files must end in {REVIEW_PACKAGE_SUFFIX!r} or "
        f"{REVIEW_LOG_SUFFIX!r}: {path.name}"
    )


def _json_dumps(value: Any) -> str:
    """Stable compact JSON for parquet string columns and hashes."""
    return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))


def _load_package(package_path: Path) -> dict[str, Any]:
    try:
        package = json.loads(package_path.read_text())
    except json.JSONDecodeError as exc:
        msg = f"Invalid review package JSON in {package_path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(package, dict):
        msg = f"Review package must be a JSON object: {package_path}"
        raise TypeError(msg)
    return package


def _reviewed_at(review: dict[str, Any]) -> str:
    """Sort key for resolving repeated verdicts on one event_key.

    The review log is an append-only event record, so the **latest
    ``reviewed_at`` wins**. ISO-8601 timestamps sort lexicographically; a missing
    value sorts oldest (so a later, timestamped review still wins). Callers
    compare with ``>=`` so an exact tie falls back to file order (later line).
    """
    return str(review.get("reviewed_at") or "")


def _load_reviews(reviews_path: Path) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    for line_no, raw_line in enumerate(reviews_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            review = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {reviews_path} line {line_no}: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(review, dict):
            msg = f"Review line {line_no} in {reviews_path} is not a JSON object"
            raise TypeError(msg)
        key = review.get("event_key")
        if key is None or str(key).strip() == "":
            msg = f"Review line {line_no} in {reviews_path} is missing event_key"
            raise ValueError(msg)
        key = str(key)
        existing = reviews.get(key)
        if existing is None or _reviewed_at(review) >= _reviewed_at(existing):
            reviews[key] = review
    return reviews


def _iter_package_slots(
    package: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], Slot]]:
    patients = package.get("patients") or []
    if not isinstance(patients, list):
        raise TypeError("Review package field 'patients' must be a list")

    seen: set[str] = set()
    for patient_index, patient in enumerate(patients):
        if not isinstance(patient, dict):
            raise TypeError(f"Review package patient #{patient_index} is not an object")
        events = patient.get("events") or []
        if not isinstance(events, list):
            raise TypeError(
                f"Review package patient #{patient_index} field 'events' must be a list"
            )
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                raise TypeError(
                    f"Review package patient #{patient_index} event #{event_index} "
                    "is not an object"
                )
            event_key = str(event.get("event_key") or "")
            if not event_key:
                raise ValueError(
                    f"Review package patient #{patient_index} event #{event_index} "
                    "is missing event_key"
                )
            if event_key in seen:
                raise ValueError(f"Review package has duplicate event_key: {event_key}")
            seen.add(event_key)
            yield patient, slot_from_package_event(event)


def _note_metadata(patient: dict[str, Any], note_id: str) -> dict[str, Any]:
    notes = patient.get("notes") or {}
    if not isinstance(notes, dict):
        return {}
    note = notes.get(note_id)
    return note if isinstance(note, dict) else {}


def _patient_from_record(record: dict[str, Any]) -> dict[str, Any]:
    mrn = record.get("mrn")
    note_id = str(record.get("note_id") or "")
    key = str(mrn) if mrn is not None and str(mrn).strip() else note_id
    return {"mrn": key, "notes": {}}


def _patient_for_added_review(
    package: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    mrn = str(review.get("mrn") or "")
    note_id = str(review.get("note_id") or "")
    patients = package.get("patients") or []
    if isinstance(patients, list):
        for patient in patients:
            if not isinstance(patient, dict):
                continue
            if str(patient.get("mrn") or "") == mrn:
                return patient
            notes = patient.get("notes") or {}
            if isinstance(notes, dict) and note_id in notes:
                return patient
    return {"mrn": mrn or note_id, "notes": {}}


def _is_reviewer_added_event(review: dict[str, Any]) -> bool:
    return review.get("is_new_event") is True


def _slot_from_added_review(event_key: str, review: dict[str, Any]) -> Slot:
    event_type = str(review.get("event_type") or "")
    if not event_type:
        raise ValueError(f"Reviewer-added event {event_key!r} is missing event_type")
    return Slot(
        event_key=event_key,
        event_type=event_type,
        note_id=str(review.get("note_id") or ""),
        fingerprint="",
        fields={},
    )


def _edits_from_review(review: dict[str, Any]) -> dict[str, Any]:
    edits = review.get("edits") or {}
    if not isinstance(edits, dict):
        raise TypeError(
            f"Review {review.get('event_key')!r} field 'edits' must be an object"
        )
    return dict(edits)


def _event_row(
    *,
    package: dict[str, Any],
    patient: dict[str, Any],
    slot: Slot,
    review: dict[str, Any],
    acceptance_reason: str,
) -> dict[str, Any]:
    event_key = slot.event_key
    event_type = slot.event_type
    note_id = slot.note_id or str(review.get("note_id") or "")
    note = _note_metadata(patient, note_id)

    original_fields = data_fields(slot.fields)
    edits = data_fields(_edits_from_review(review))
    reviewed_fields = original_fields | edits

    original_fields_json = _json_dumps(original_fields)
    edits_json = _json_dumps(edits)
    reviewed_fields_json = _json_dumps(reviewed_fields)

    row = _clean_event(reviewed_fields)
    mrn = patient.get("mrn") or review.get("mrn")
    row.update(
        {
            "event_key": event_key,
            "event_type": event_type,
            "tool_name": event_type,
            "note_id": note_id,
            "mrn": str(mrn) if mrn is not None else None,
            "definition_name": str(package.get("definition_name") or ""),
            "batch_name": str(package.get("batch") or ""),
            "package_generated_at": package.get("generated_at"),
            "review_verdict": str(review.get("verdict") or ""),
            "reviewer": str(review.get("reviewer") or ""),
            "reviewed_at": review.get("reviewed_at"),
            "review_comment": str(review.get("comment") or ""),
            "acceptance_reason": acceptance_reason,
            "note_date": note.get("note_date"),
            "note_type": note.get("note_type"),
            "department": note.get("department"),
            "original_fields_json": original_fields_json,
            "edits_json": edits_json,
            "reviewed_fields_json": reviewed_fields_json,
            "key_hash": blake2b_128(event_key),
            "content_hash": blake2b_128(
                _json_dumps(
                    {
                        "fields": reviewed_fields,
                        "review": {
                            "verdict": review.get("verdict"),
                            "comment": review.get("comment"),
                            "reviewer": review.get("reviewer"),
                            "reviewed_at": review.get("reviewed_at"),
                        },
                        "acceptance_reason": acceptance_reason,
                    }
                )
            ),
        }
    )
    return row


def _empty_silver_df() -> pl.DataFrame:
    return pl.DataFrame(schema=_SILVER_SCHEMA)


def _rows_to_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return _empty_silver_df()

    df = pl.DataFrame(rows, infer_schema_length=None)
    for col, dtype in _SILVER_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    casts = [
        pl.col(col).cast(dtype, strict=False)
        for col, dtype in _SILVER_SCHEMA.items()
        if col in df.columns
    ]
    if casts:
        df = df.with_columns(casts)

    ordered = list(_SILVER_SCHEMA) + [c for c in df.columns if c not in _SILVER_SCHEMA]
    return df.select(ordered).sort("event_key")


def merge_silver_segments(
    seg_dfs: list[tuple[int, pl.DataFrame | None]],
) -> pl.DataFrame:
    """Collapse a batch's per-segment silver DataFrames into one reviewed table.

    A batch's reviews arrive per segment (``<batch>/NNN`` reviewed separately).
    This merges them the way the raw side merges segments: for each ``note_id``,
    the **highest segment that reviewed it wins**, so a note re-extracted and
    re-reviewed in a later segment supersedes its earlier reviewed row.
    Provenance survives in each row's ``batch_name`` (which carries the segment).

    ``seg_dfs`` is ``(segment_number, silver_df)`` pairs. Returns the merged
    DataFrame (empty-with-schema if nothing approved).
    """
    non_empty = [(seg, df) for seg, df in seg_dfs if df is not None and df.height]
    if not non_empty:
        return _empty_silver_df()
    framed = [df.with_columns(pl.lit(seg).alias("_segment")) for seg, df in non_empty]
    combined = pl.concat(framed, how="vertical_relaxed")
    max_seg = combined.group_by("note_id").agg(
        pl.col("_segment").max().alias("_max_seg")
    )
    return (
        combined.join(max_seg, on="note_id")
        .filter(pl.col("_segment") == pl.col("_max_seg"))
        .drop("_segment", "_max_seg")
        .sort("event_key")
    )


def review_to_silver_df(
    package_path: Path,
    reviews_path: Path,
    raw_jsonl_path: Path,
    *,
    require_complete: bool = True,
) -> ReviewLoadResult:
    """Build silver rows from a raw extraction segment plus its review worklist.

    The raw JSONL segment is the canonical source of extraction slots. The
    review package is the human worklist: slots in the package require a latest
    ``reviewed_at`` verdict from the review log, while slots outside the package
    are auto-accepted with ``acceptance_reason='unflagged_autoaccept'``.
    """
    package = _load_package(package_path)
    reviews = _load_reviews(reviews_path)
    field_schema = package.get("field_schema") or {}
    if not isinstance(field_schema, dict):
        raise TypeError("Review package field 'field_schema' must be an object")

    worklist: dict[str, tuple[dict[str, Any], Slot]] = {}
    for patient, slot in _iter_package_slots(package):
        worklist[slot.event_key] = (patient, slot)

    rows: list[dict[str, Any]] = []
    worklist_keys = set(worklist)
    raw_event_keys: set[str] = set()
    reviewer_added_keys: set[str] = set()
    result = ReviewLoadResult(output_path=None)
    slot_state = SlotState()

    for record in successful_records_from_jsonl(raw_jsonl_path):
        raw_patient = _patient_from_record(record)
        for slot in record_to_slots(record, field_schema, state=slot_state):
            result.total_events += 1
            event_key = slot.event_key
            raw_event_keys.add(event_key)
            worklist_item = worklist.get(event_key)
            if worklist_item is None:
                rows.append(
                    _event_row(
                        package=package,
                        patient=raw_patient,
                        slot=slot,
                        review={},
                        acceptance_reason=_UNFLAGGED_AUTOACCEPT,
                    )
                )
                result.auto_accepted_events += 1
                continue

            review = reviews.get(event_key)
            if review is None:
                result.unreviewed_events += 1
                continue

            result.reviewed_events += 1
            verdict = str(review.get("verdict") or "").strip().lower()
            if verdict not in _VALID_VERDICTS:
                raise ValueError(
                    f"Review for event_key {event_key!r} has unsupported verdict "
                    f"{verdict!r}; expected one of {sorted(_VALID_VERDICTS)}"
                )
            if verdict == _REJECTED_VERDICT:
                result.rejected_events += 1
                continue

            package_patient, _package_slot = worklist_item
            review = dict(review)
            review["verdict"] = verdict
            rows.append(
                _event_row(
                    package=package,
                    patient=package_patient,
                    slot=slot,
                    review=review,
                    acceptance_reason=_FLAGGED_REVIEWED,
                )
            )
            result.approved_events += 1

    for event_key, review in reviews.items():
        if not _is_reviewer_added_event(review):
            continue
        reviewer_added_keys.add(event_key)
        result.total_events += 1
        result.reviewed_events += 1
        verdict = str(review.get("verdict") or "").strip().lower()
        if verdict not in _VALID_VERDICTS:
            raise ValueError(
                f"Review for event_key {event_key!r} has unsupported verdict "
                f"{verdict!r}; expected one of {sorted(_VALID_VERDICTS)}"
            )
        if verdict == _REJECTED_VERDICT:
            result.rejected_events += 1
            continue

        review = dict(review)
        review["verdict"] = verdict
        rows.append(
            _event_row(
                package=package,
                patient=_patient_for_added_review(package, review),
                slot=_slot_from_added_review(event_key, review),
                review=review,
                acceptance_reason=_REVIEWER_ADDED,
            )
        )
        result.approved_events += 1

    missing_worklist_keys = sorted(worklist_keys - raw_event_keys)
    if missing_worklist_keys:
        sample = ", ".join(missing_worklist_keys[:3])
        raise ValueError(
            f"{package_path.name} has {len(missing_worklist_keys)} worklist event(s) "
            f"not present in raw segment {raw_jsonl_path}: {sample}"
        )

    result.ignored_reviews = len(set(reviews) - worklist_keys - reviewer_added_keys)
    if result.unreviewed_events and require_complete:
        raise ValueError(
            f"{package_path.name} has {result.unreviewed_events} unreviewed "
            "event(s); finish the review before loading a silver table"
        )

    result.df = _rows_to_df(rows)
    return result


def review_to_silver_parquet(
    package_path: Path,
    reviews_path: Path,
    raw_jsonl_path: Path,
    output_path: Path,
    *,
    require_complete: bool = True,
    dry_run: bool = False,
) -> ReviewLoadResult:
    """Convert a completed review package + sidecar to a silver parquet."""
    result = review_to_silver_df(
        package_path,
        reviews_path,
        raw_jsonl_path,
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
