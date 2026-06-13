"""
FC extraction JSONL helpers.

Two consumers:

- ``oncai ingest fc_extractions`` uses ``merge_segments_to_parquet`` (which
  drives ``jsonl_to_wide_parquet`` per segment) to merge a batch's numbered
  segments (``inbox/fc_extractions/<batch>/NNN.jsonl``) into a single wide
  row-per-record parquet at ``lake/fc_extractions/<batch>.parquet``. For each
  ``record_id`` the highest segment wins — explicit integer order, no
  timestamps. Events / finish / run_meta are preserved verbatim as JSON
  strings; SQL reshaping happens at db-build time via sibling
  ``<parquet_stem>.sql`` files.
- ``oncai fc peek`` uses ``_load_fc_records`` + ``_flatten_fc_single_record``
  for a quick look at a JSONL in the throwaway ``scratch`` duckdb schema.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_YYYY = re.compile(r"^\d{4}$")
_YYYY_MM = re.compile(r"^\d{4}-\d{2}$")
_YYYY_MM_DD = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _infer_date_precision(date_str: str) -> str:
    """Infer precision from date string format.

    Returns 'year', 'month', or 'day'.
    """
    if _YYYY_MM_DD.match(date_str):
        return "day"
    if _YYYY_MM.match(date_str):
        return "month"
    if _YYYY.match(date_str):
        return "year"
    return "unknown"


def _pad_date(date_str: str, precision: str) -> str:
    """Pad a partial date to YYYY-MM-DD using midpoint defaults.

    Year-only  → YYYY-07-15  (midpoint of year)
    Month-only → YYYY-MM-15  (midpoint of month)
    Exact      → unchanged

    Always checks the actual string format before padding to avoid
    double-padding (e.g., "2023-05-01" with precision="month" stays as-is).
    """
    if _YYYY_MM_DD.match(date_str):
        return date_str
    if _YYYY_MM.match(date_str):
        return f"{date_str}-15"
    if _YYYY.match(date_str):
        return f"{date_str}-07-15"
    return date_str


def _emit_date(key: str, value: str, row: dict[str, Any]) -> None:
    """Emit a padded date and its inferred precision."""
    precision = _infer_date_precision(value)
    row[key] = _pad_date(value, precision)
    row[f"{key}_precision"] = precision


_PRECISION_MAP = {0: "unknown", 1: "year", 2: "month", 3: "day"}


def _emit_approx_date(key: str, value: dict[str, Any], row: dict[str, Any]) -> None:
    """Handle ApproxDate model dicts with date/precision/anchor fields.

    Format: {"date": "2023-06-15", "precision": 3, "anchor": "EXACT"}
    Uses the explicit precision and anchor rather than re-inferring from the string.
    """
    raw_date = value.get("date")
    numeric_precision = value.get("precision", 0)
    anchor = value.get("anchor")

    precision = _PRECISION_MAP.get(numeric_precision, "unknown")

    if raw_date is None:
        row[key] = None
        row[f"{key}_precision"] = "unknown"
    else:
        date_str = str(raw_date)
        # Use explicit precision from the model if available,
        # fall back to inferring from format
        if precision == "unknown":
            precision = _infer_date_precision(date_str)
        row[key] = _pad_date(date_str, precision)
        row[f"{key}_precision"] = precision

    if anchor is not None:
        row[f"{key}_anchor"] = anchor


def _is_approx_date_dict(value: Any) -> bool:
    """Check if a dict looks like an ApproxDate model (has date + precision keys)."""
    return isinstance(value, dict) and "precision" in value and "date" in value


def _looks_like_date(value: str) -> bool:
    """Check if a string value looks like a date (YYYY, YYYY-MM, or YYYY-MM-DD)."""
    return bool(_YYYY.match(value) or _YYYY_MM.match(value) or _YYYY_MM_DD.match(value))


# ---------------------------------------------------------------------------
# Generic flattening
# ---------------------------------------------------------------------------


def _merge_kv(row: dict[str, Any], key: str, value: Any, prefix: str = "") -> None:
    """Insert key→value into row, handling collisions with prefix."""
    if key not in row:
        row[key] = value
    elif prefix:
        row[f"{prefix}_{key}"] = value
    else:
        row[key] = value


def _flatten_dict_into(
    d: dict[str, Any],
    row: dict[str, Any],
    parent_key: str = "",
) -> None:
    """Recursively flatten a dict into row.

    - ApproxDate dicts (date/precision/anchor) are expanded with proper precision & anchor columns
    - ApproxDateStr values (date-like strings) get a {key}_precision column inferred from format
    - Nested dicts become parent_child keys
    - Lists become semicolon-joined strings
    """
    for key, value in d.items():
        full_key = f"{parent_key}_{key}" if parent_key else key

        if value is None:
            _merge_kv(row, full_key, None)
        elif _is_approx_date_dict(value):
            _emit_approx_date(full_key, value, row)
        elif isinstance(value, dict):
            _flatten_dict_into(value, row, parent_key=full_key)
        elif isinstance(value, list):
            _merge_kv(
                row,
                full_key,
                ";".join(str(v) for v in value) if value else None,
            )
        elif isinstance(value, str) and _looks_like_date(value):
            _emit_date(full_key, value, row)
        else:
            _merge_kv(row, full_key, value)


def _load_fc_records(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load FC extraction records from JSONL (raw JSON dicts)."""
    records = []
    with jsonl_path.open() as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _clean_event(event: dict[str, Any]) -> dict[str, Any]:
    """Clean an event dict: expand ApproxDates, flatten nested dicts."""
    row: dict[str, Any] = {}
    _flatten_dict_into(event, row)
    return row


def _flatten_fc_single_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten a single-note FC extraction record into one row per event.

    Extracts raw events, tags each with note_id and tool_name,
    and applies ApproxDate correction.

    Args:
        record: Raw JSON dict from single-note FC JSONL

    Returns:
        List of flattened row dicts (one per event)
    """
    note_id = record.get("note_id", "")
    mrn = record.get("mrn")
    gate_result = record.get("gate_result")

    events = record.get("events", {})
    rows = []

    for tool_name, event_list in events.items():
        if not isinstance(event_list, list):
            continue
        for event in event_list:
            if not isinstance(event, dict):
                continue
            row = _clean_event(event)
            row["note_id"] = note_id
            if mrn is not None:
                row["mrn"] = mrn
            row["tool_name"] = tool_name
            if gate_result is not None:
                row["gate_result"] = gate_result
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Wide-format writer (one row per JSONL record, events as JSON strings)
# ---------------------------------------------------------------------------


@dataclass
class WideLoadResult:
    """Result of the wide-format JSONL → parquet conversion."""

    total_records: int = 0
    written: int = 0
    skipped_failed: int = 0
    record_kind: str | None = None
    output_path: Path | None = None
    df: "pl.DataFrame | None" = None

    def __str__(self) -> str:
        return (
            f"Wrote {self.written}/{self.total_records} records "
            f"(skipped {self.skipped_failed} failed) "
            f"as {self.record_kind or 'empty'}."
        )


def _events_for_record(record: dict[str, Any]) -> dict[str, Any]:
    """Pull the per-tool events dict from a single-note record."""
    return record.get("events") or {}


def _record_id_for(record: dict[str, Any]) -> str:
    """Return the primary key value (note_id) as a string."""
    return str(record.get("note_id") or "")


def jsonl_to_wide_parquet(
    jsonl_path: Path,
    output_path: Path,
    *,
    only_successful: bool = True,
    dry_run: bool = False,
    batch_name: str | None = None,
    segment: int | None = None,
) -> WideLoadResult:
    """Convert an FC extraction JSONL into a wide one-row-per-record parquet.

    Auto-detects single-note vs multi-note JSONL by inspecting the first
    record. Drops failed records by default — they're tracked in the
    manifest and via ``oncai fc status``; the lake parquet stays clean.

    Output schema is stable across all FC workflows; nested data is
    preserved verbatim as JSON strings (events_json, finish_json,
    run_meta_json) so the parquet schema doesn't evolve with new event
    types. Relational reshaping happens at db-build time via the
    ``<parquet_stem>.sql`` sibling next to each lake parquet.

    When ``dry_run=True``, the would-be DataFrame is returned on
    ``result.df`` but no parquet is written to disk.
    """
    from oncai.hashing import blake2b_128

    records = _load_fc_records(jsonl_path)
    result = WideLoadResult(total_records=len(records), output_path=output_path)

    if not records:
        return result

    result.record_kind = "single_note"
    # Provenance: the merge passes the batch (folder) name + the integer
    # segment, so each row records exactly which segment produced it. Falling
    # back to the JSONL stem keeps the standalone / peek path working.
    if batch_name is None:
        batch_name = jsonl_path.stem

    rows: list[dict[str, Any]] = []
    for record in records:
        if only_successful and not record.get("success", False):
            result.skipped_failed += 1
            continue

        record_id = _record_id_for(record)
        events = _events_for_record(record)
        finish = record.get("finish")
        run_meta = record.get("run_meta") or {}

        events_json = json.dumps(events, default=str, sort_keys=True)
        finish_json = (
            json.dumps(finish, default=str, sort_keys=True) if finish else None
        )
        run_meta_json = (
            json.dumps(run_meta, default=str, sort_keys=True) if run_meta else None
        )

        mrn_val = record.get("mrn")
        rows.append(
            {
                "record_id": record_id,
                "record_kind": "single_note",
                "definition_name": record.get("definition_name", ""),
                "batch_name": batch_name,
                "segment": segment,
                "success": True,
                "rounds": int(record.get("rounds") or 0),
                "input_tokens": int(record.get("input_tokens") or 0),
                "output_tokens": int(record.get("output_tokens") or 0),
                "reasoning_tokens": int(record.get("reasoning_tokens") or 0),
                "extracted_at": record.get("extracted_at"),
                "events_json": events_json,
                "finish_json": finish_json,
                "run_meta_json": run_meta_json,
                "gate_result": record.get("gate_result"),
                "source_content_hash": record.get("source_content_hash"),
                "mrn": str(mrn_val) if mrn_val is not None else None,
                "system_prompt_hash": run_meta.get("system_prompt_hash"),
                "key_hash": blake2b_128(record_id),
                "content_hash": blake2b_128(events_json + (finish_json or "")),
            }
        )

    if not rows:
        return result

    df = pl.DataFrame(
        rows,
        schema={
            "record_id": pl.String,
            "record_kind": pl.String,
            "definition_name": pl.String,
            "batch_name": pl.String,
            "segment": pl.Int64,
            "success": pl.Boolean,
            "rounds": pl.Int32,
            "input_tokens": pl.Int64,
            "output_tokens": pl.Int64,
            "reasoning_tokens": pl.Int64,
            "extracted_at": pl.String,
            "events_json": pl.String,
            "finish_json": pl.String,
            "run_meta_json": pl.String,
            "gate_result": pl.String,
            "source_content_hash": pl.String,
            "mrn": pl.String,
            "system_prompt_hash": pl.String,
            "key_hash": pl.Binary,
            "content_hash": pl.Binary,
        },
    )
    # Stable row order across runs — matters for byte-level idempotency.
    df = df.sort("record_id")
    result.df = df
    result.written = len(df)

    if dry_run:
        return result

    # Atomic write: tmp → fsync → rename so a crash mid-write doesn't corrupt
    # the existing parquet at this path.
    import os as _os

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    fd = _os.open(tmp, _os.O_RDONLY)
    try:
        _os.fsync(fd)
    finally:
        _os.close(fd)
    tmp.replace(output_path)

    return result


_SEGMENT_RE = re.compile(r"^(\d+)\.jsonl$")


def segment_files(batch_dir: Path) -> list[tuple[int, Path]]:
    """Return ``(segment_number, path)`` for each ``NNN.jsonl`` in a batch dir.

    Sorted ascending by the integer. Non-segment files (manifests, ``.partial``,
    sidecars, ``batch.json``) are ignored.
    """
    if not batch_dir.exists():
        return []
    out: list[tuple[int, Path]] = []
    for p in batch_dir.iterdir():
        m = _SEGMENT_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def merge_segments_to_parquet(
    batch_dir: Path,
    output_path: Path,
    *,
    only_successful: bool = True,
    dry_run: bool = False,
) -> WideLoadResult:
    """Merge a batch's numbered segments into one wide parquet.

    A batch is a folder of immutable, monotonically-numbered segments
    (``NNN.jsonl``). For each ``record_id`` the **highest segment that contains
    it wins** — a later re-extraction supersedes an earlier one. Ordering is the
    explicit segment integer, so the result is deterministic and never depends
    on wall-clock ``extracted_at``. Each row carries ``batch_name`` (the folder)
    and ``segment`` (the integer) for provenance.

    ``total_records``/``written`` reflect the *post-merge* numbers (unique
    ``record_id``s), not the raw input count.
    """
    segments = segment_files(batch_dir)
    if not segments:
        return WideLoadResult(total_records=0, output_path=output_path)

    batch = batch_dir.name
    frames: list[pl.DataFrame] = []
    total_input = 0
    total_skipped_failed = 0
    record_kind: str | None = None
    for seg_num, path in segments:
        sub = jsonl_to_wide_parquet(
            path,
            output_path,
            only_successful=only_successful,
            dry_run=True,
            batch_name=batch,
            segment=seg_num,
        )
        total_input += sub.total_records
        total_skipped_failed += sub.skipped_failed
        if sub.record_kind and not record_kind:
            record_kind = sub.record_kind
        if sub.df is not None and sub.written:
            frames.append(sub.df)

    result = WideLoadResult(
        total_records=total_input,
        output_path=output_path,
        skipped_failed=total_skipped_failed,
        record_kind=record_kind,
    )

    if not frames:
        return result

    combined = pl.concat(frames, how="vertical_relaxed")

    # Highest segment per record_id wins — pure integer order, no clocks.
    deduped = (
        combined.sort(
            ["record_id", "segment"], descending=[False, True], nulls_last=True
        )
        .unique(subset=["record_id"], keep="first")
        .sort("record_id")
    )

    result.df = deduped
    result.written = len(deduped)

    if dry_run:
        return result

    import os as _os

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    deduped.write_parquet(tmp, compression="zstd")
    fd = _os.open(tmp, _os.O_RDONLY)
    try:
        _os.fsync(fd)
    finally:
        _os.close(fd)
    tmp.replace(output_path)

    return result
