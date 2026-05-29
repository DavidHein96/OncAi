"""
Single-note batch processing for function-calling extraction.

Iterates by note_id and processes each note independently. Designed for
pathology reports where one report = one extraction.

Provides resumable, crash-safe batch extraction with progress tracking.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
from tqdm import tqdm

from .client import FunctionCallingClient, NoteExtractionResult
from .models import ExtractionPlan
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


def _drain_futures_with_cancel(
    futures: list[Future],
    handle: Callable[[Future], None],
    label: str = "items",
) -> None:
    """Drain ``futures`` with each completed one passed to ``handle``.

    On ``KeyboardInterrupt``, cancel any still-pending futures and re-raise so
    the caller's outer handler sees the cancel. Without this, Ctrl+C on a
    ThreadPool drain leaves the pool alive until every running task completes.
    """
    try:
        for future in as_completed(futures):
            handle(future)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt: cancelling pending %s", label)
        for f in futures:
            f.cancel()
        raise


@dataclass
class SingleNoteConfig:
    """Lightweight config for single-note extraction (no workflow/direction needed)."""

    name: str
    system_prompt: str


@dataclass
class SingleNoteBatchResult:
    """Result of a single-note batch extraction run."""

    output_path: str
    total_notes: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    events_by_type: dict[str, int] = field(default_factory=dict)
    plans_by_type: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Batch complete: {self.successful}/{self.total_notes} notes successful, "
            f"{self.failed} failed, {self.skipped} skipped. "
            f"Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out "
            f"({self.total_reasoning_tokens} reasoning)"
        )


def get_single_note_batch_status(jsonl_path: Path) -> dict[str, int]:
    """Tally success/failure counts in a single-note batch JSONL.

    Returns a dict with ``total``, ``successful``, ``failed`` keys.
    """
    total = 0
    successful = 0
    failed = 0
    if not jsonl_path.exists():
        return {"total": 0, "successful": 0, "failed": 0}
    with jsonl_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if record.get("success"):
                successful += 1
            else:
                failed += 1
    return {"total": total, "successful": successful, "failed": failed}


def _get_existing_extraction_keys(
    output_path: Path, *, successful_only: bool = False
) -> set[tuple[str, str | None]]:
    """Get ``(note_id, source_content_hash)`` tuples already processed.

    The hash component is the hex string from the record's top-level
    ``source_content_hash`` field. Records produced by pre-incremental code
    don't carry this field; for those, the tuple's hash is ``None``.

    Args:
        output_path: Path to the batch JSONL file.
        successful_only: If True, only count tuples whose record has
            ``success=True``. Used by retry-failed mode.
    """
    if not output_path.exists():
        return set()

    existing: set[tuple[str, str | None]] = set()
    skipped = 0
    with Path.open(output_path) as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    if successful_only and not record.get("success", False):
                        continue
                    note_id = str(record.get("note_id", ""))
                    src_hash = record.get("source_content_hash")
                    existing.add((note_id, src_hash))
                except json.JSONDecodeError:
                    skipped += 1
    if skipped:
        logger.warning(
            "%d malformed JSON lines skipped in %s — those note_ids may be reprocessed",
            skipped,
            output_path,
        )
    return existing


def _prune_failed_records(output_path: Path) -> tuple[int, int]:
    """Remove ``success=False`` records from ``output_path`` in place.

    Returns (kept, dropped). Backs up the original to ``<path>.bak``.
    """
    if not output_path.exists():
        return 0, 0

    kept_lines: list[str] = []
    dropped = 0
    with output_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                kept_lines.append(line)
                continue
            if rec.get("success", False):
                kept_lines.append(line)
            else:
                dropped += 1

    if dropped == 0:
        return len(kept_lines), 0

    backup = output_path.with_suffix(output_path.suffix + ".bak")
    output_path.replace(backup)
    with output_path.open("w") as f:
        f.writelines(kept_lines)
    return len(kept_lines), dropped


def _source_table_has_column(
    con: duckdb.DuckDBPyConnection, source_table: str, column: str
) -> bool:
    """Check whether ``source_table`` (``schema.table``) exposes ``column``."""
    if "." in source_table:
        schema, table = source_table.split(".", 1)
    else:
        schema, table = "main", source_table
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ? AND column_name = ?
        """,
        [schema, table, column],
    ).fetchone()
    return bool(row and row[0])


def _load_notes(
    db_path: Path,
    source_table: str,
    text_col: str,
    id_col: str,
    limit: int | None = None,
    where: str | None = None,
    note_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Load notes from DuckDB for single-note processing.

    Pulls ``content_hash`` from the source table when present (returned as the
    ``source_content_hash`` hex string on each note dict); falls back to
    ``None`` for tables without it.

    Args:
        db_path: Path to DuckDB database
        source_table: Fully qualified table name (e.g., raw.pathology)
        text_col: Column containing note text
        id_col: Column containing note identifier
        limit: Max notes to return
        where: Optional SQL WHERE clause
        note_ids: Optional set of note IDs to filter to

    Returns:
        List of note dicts with ``note_id``, ``note_text``, and
        ``source_content_hash`` keys.
    """
    con = duckdb.connect(str(db_path), read_only=True)

    try:
        has_hash = _source_table_has_column(con, source_table, "content_hash")
        hash_select = (
            ', "content_hash" AS source_content_hash'
            if has_hash
            else ", NULL AS source_content_hash"
        )
        has_mrn = _source_table_has_column(con, source_table, "mrn")
        mrn_select = ', "mrn" AS mrn' if has_mrn else ", NULL AS mrn"
        query = (
            f'SELECT "{id_col}" AS note_id, "{text_col}" AS note_text'
            f"{hash_select}{mrn_select} FROM {source_table}"
        )

        conditions = []
        if where:
            conditions.append(f"({where})")
        # Filter to non-empty text
        conditions.append(f'"{text_col}" IS NOT NULL')

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += f' ORDER BY "{id_col}"'

        if limit and note_ids is None:
            query += f" LIMIT {limit}"

        rows = con.execute(query).fetchall()
        columns = [desc[0] for desc in con.description]
        notes = [dict(zip(columns, row, strict=True)) for row in rows]

        # Convert binary content_hash (bytes) to hex string for JSON-friendliness
        for n in notes:
            h = n.get("source_content_hash")
            if isinstance(h, (bytes, bytearray)):
                n["source_content_hash"] = h.hex()
            elif h is not None:
                n["source_content_hash"] = str(h)

        # Apply note_id filter if provided
        if note_ids is not None:
            notes = [n for n in notes if str(n["note_id"]) in note_ids]

        # Apply limit after filtering
        if limit and note_ids is not None:
            notes = notes[:limit]

        return notes

    finally:
        con.close()


def _load_notes_from_jsonl(
    jsonl_path: Path,
    text_col: str,
    id_col: str,
    limit: int | None = None,
    note_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load notes from a JSONL file (one JSON object per line).

    Output shape matches ``_load_notes``: each note dict has ``note_id``,
    ``note_text``, and ``source_content_hash`` (read from a ``content_hash``
    field on the row if present, else None — so incremental hash dedup is
    available when the source JSONL carries it).

    Args:
        jsonl_path: Path to the JSONL file.
        text_col: Key in each JSON object containing the note text.
        id_col: Key in each JSON object containing the note identifier.
        limit: Max notes to return.
        note_ids: Optional set of note IDs to filter to.
    """
    notes: list[dict[str, Any]] = []
    malformed = 0
    missing_cols = 0
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if id_col not in row or text_col not in row:
                missing_cols += 1
                continue
            note_id = str(row[id_col])
            if note_ids is not None and note_id not in note_ids:
                continue
            notes.append(
                {
                    "note_id": note_id,
                    "note_text": row[text_col],
                    "source_content_hash": row.get("content_hash"),
                    "mrn": row.get("mrn"),
                }
            )
            if limit and note_ids is None and len(notes) >= limit:
                break

    if malformed:
        logger.warning("%d malformed JSON lines skipped in %s", malformed, jsonl_path)
    if missing_cols:
        logger.warning(
            "%d rows in %s missing %r or %r; skipped",
            missing_cols,
            jsonl_path,
            id_col,
            text_col,
        )

    if limit and note_ids is not None:
        notes = notes[:limit]

    return notes


def _build_run_meta(
    batch_name: str,
    source_table: str,
    client: FunctionCallingClient,
    system_prompt: str,
    backend_name: str | None = None,
) -> dict[str, Any]:
    """Build run-level metadata computed once per batch."""
    from .manifest import get_code_version, get_git_info, hash_string

    git_info = get_git_info()
    return {
        "batch_name": batch_name,
        "source_table": source_table,
        "backend": backend_name
        or getattr(client, "backend_name", None)
        or type(client).__name__,
        "model": getattr(client, "model", None) or getattr(client, "deployment", None),
        "git_commit": git_info["commit"],
        "git_branch": git_info["branch"],
        "git_dirty": git_info["dirty"],
        "code_version": get_code_version(),
        "system_prompt_hash": hash_string(system_prompt),
    }


def _merge_single_note_manifest(
    existing: dict[str, Any], new: dict[str, Any]
) -> dict[str, Any]:
    """Merge counts from an existing manifest into a freshly built one.

    Preserves the original `started_at` and accumulates work-done counters
    (succeeded/failed, tokens, events, plans, duration, errors) so that
    resumed runs reflect the whole batch's history, not just the latest
    invocation. Skipped/total mix prior-completed and unworkable notes,
    so they take the new run's value rather than summing.
    """
    if existing.get("started_at"):
        new["started_at"] = existing["started_at"]

    old_dur = existing.get("duration_seconds")
    if old_dur is not None and new.get("duration_seconds") is not None:
        new["duration_seconds"] = round(old_dur + new["duration_seconds"], 2)

    old_results = existing.get("results") or {}
    new_results = new["results"]
    for k in (
        "notes_succeeded",
        "notes_failed",
        "notes_processed",
        "total_events",
        "total_plans",
        "total_input_tokens",
        "total_output_tokens",
        "total_reasoning_tokens",
    ):
        new_results[k] = old_results.get(k, 0) + new_results.get(k, 0)

    for bucket in ("events_by_type", "plans_by_type"):
        merged = dict(old_results.get(bucket) or {})
        for k, v in (new_results.get(bucket) or {}).items():
            merged[k] = merged.get(k, 0) + v
        new_results[bucket] = merged

    old_errors = existing.get("errors") or []
    if old_errors:
        new["errors"] = list(old_errors) + list(new.get("errors") or [])

    return new


def _write_single_note_manifest(
    *,
    manifest_path: Path,
    config: SingleNoteConfig,
    client: FunctionCallingClient,
    registry: ToolRegistry,
    db_path: Path,
    source_table: str,
    text_col: str,
    id_col: str,
    limit: int | None,
    where: str | None,
    note_ids_count: int | None,
    workers: int,
    rate_limit: float,
    resume: bool,
    started_at: str,
    started_monotonic: float,
    result: SingleNoteBatchResult,
    backend_name: str | None = None,
    skip_if_no_work: bool = False,
) -> None:
    """Write the batch manifest as a 'done marker' once the JSONL is closed.

    Mirrors the multi-note manifest shape (batch_name, git, config, backend,
    results) but uses note_* counts and an extraction_type='single_note'
    discriminator so the CLI can branch on display.

    On resume, an existing manifest is merged in (originally-recorded
    started_at preserved, work counters accumulated). Pass
    skip_if_no_work=True from the early-return path to avoid clobbering a
    completed manifest with a no-op re-run.
    """
    from .manifest import get_code_version, get_git_info, hash_string

    if (
        skip_if_no_work
        and manifest_path.exists()
        and result.successful == 0
        and result.failed == 0
    ):
        return

    git_info = get_git_info()
    completed_at = datetime.now(timezone.utc).isoformat()
    duration_seconds = round(time.monotonic() - started_monotonic, 2)

    client_config = getattr(client, "config", None)
    backend_payload: dict[str, Any] = {
        "type": backend_name
        or getattr(client, "backend_name", None)
        or type(client).__name__,
        "model": getattr(client, "model", None) or getattr(client, "deployment", None),
    }
    if client_config is not None:
        if getattr(client_config, "reasoning_effort", None):
            backend_payload["reasoning_effort"] = client_config.reasoning_effort
        if getattr(client_config, "text_verbosity", None):
            backend_payload["text_verbosity"] = client_config.text_verbosity
        if getattr(client_config, "temperature", None) is not None:
            backend_payload["temperature"] = client_config.temperature
        if getattr(client_config, "top_p", None) is not None:
            backend_payload["top_p"] = client_config.top_p
        if getattr(client_config, "top_k", None) is not None:
            backend_payload["top_k"] = client_config.top_k

    tool_names = [
        t
        for t in registry.list_tools()
        if t not in ("finish_note_extraction", "stop_workflow")
    ]
    tool_schemas: dict[str, Any] = {}
    for t in tool_names:
        tool_def = registry.get(t)
        if tool_def is not None:
            tool_schemas[t] = tool_def.model.model_json_schema()

    manifest: dict[str, Any] = {
        "batch_name": result.output_path.rsplit("/", 1)[-1].removesuffix(".jsonl"),
        "workflow_name": config.name,
        "definition_name": config.name,
        "extraction_type": "single_note",
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": duration_seconds,
        "git": {
            "commit": git_info["commit"],
            "branch": git_info["branch"],
            "dirty": git_info["dirty"],
        },
        "code_version": get_code_version(),
        "system_prompt_hash": hash_string(config.system_prompt),
        "tools": tool_names,
        "tool_schemas": tool_schemas,
        "backend": backend_payload,
        "config": {
            "db_path": str(db_path),
            "source_table": source_table,
            "text_col": text_col,
            "id_col": id_col,
            "limit": limit,
            "where": where,
            "note_ids_count": note_ids_count,
            "workers": workers,
            "rate_limit": rate_limit,
            "resume": resume,
        },
        "results": {
            "notes_total": result.total_notes,
            "notes_succeeded": result.successful,
            "notes_failed": result.failed,
            "notes_skipped": result.skipped,
            "notes_processed": result.successful + result.failed,
            "total_events": sum(result.events_by_type.values()),
            "events_by_type": dict(result.events_by_type),
            "total_plans": sum(result.plans_by_type.values()),
            "plans_by_type": dict(result.plans_by_type),
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "total_reasoning_tokens": result.total_reasoning_tokens,
        },
        "errors": result.errors,
        "output_path": result.output_path,
    }

    if manifest_path.exists():
        try:
            with manifest_path.open() as f:
                existing = json.load(f)
            manifest = _merge_single_note_manifest(existing, manifest)
        except (OSError, json.JSONDecodeError):
            # Corrupt or unreadable prior manifest — overwrite cleanly.
            pass

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open(mode="w") as f:
        json.dump(manifest, f, indent=2, default=str)


def _serialize_extraction_result(
    note_id: str,
    definition_name: str,
    extraction: NoteExtractionResult,
    run_meta: dict[str, Any],
    source_content_hash: str | None = None,
    mrn: str | None = None,
) -> dict[str, Any]:
    """
    Serialize a NoteExtractionResult to a flat JSONL record.

    Plan-tool calls (whose models inherit from ExtractionPlan) are routed into
    the "plans" block so downstream staging can ignore them by default.
    ``source_content_hash`` (hex string from the source row's ``content_hash``)
    is emitted as a top-level field so incremental resume can match by source
    content rather than just ``note_id``.
    """
    events_by_type: dict[str, list[dict]] = {}
    plans_by_type: dict[str, list[dict]] = {}
    for tool_name, event_obj in extraction.events:
        if tool_name in ("finish_note_extraction", "stop_workflow"):
            continue
        bucket = (
            plans_by_type if isinstance(event_obj, ExtractionPlan) else events_by_type
        )
        bucket.setdefault(tool_name, []).append(event_obj.model_dump())

    finish_data = None
    if extraction.finish:
        finish_data = extraction.finish.model_dump()
        # Remove multi-note fields that don't apply to single-note extraction
        finish_data.pop("needs_more_notes", None)

    return {
        "note_id": str(note_id),
        "mrn": str(mrn) if mrn is not None else None,
        "definition_name": definition_name,
        "source_content_hash": source_content_hash,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "success": extraction.success,
        "events": events_by_type,
        "plans": plans_by_type,
        "finish": finish_data,
        "rounds": extraction.rounds,
        "input_tokens": extraction.input_tokens,
        "output_tokens": extraction.output_tokens,
        "reasoning_tokens": extraction.reasoning_tokens,
        "error": extraction.error,
        "run_meta": run_meta,
    }


def _process_single_note(
    note: dict[str, Any],
    config: SingleNoteConfig,
    client: FunctionCallingClient,
    registry: ToolRegistry,
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    """Process one note and return the serialized JSONL record.

    This is the unit of work submitted to the thread pool.
    It is intentionally self-contained so it can run in any thread.
    """
    # For gated registries, get a fresh instance per thread (reset to gate phase)
    _fresh = getattr(registry, "fresh", None)
    if callable(_fresh):
        registry = _fresh()

    note_id = str(note["note_id"])
    note_text = note.get("note_text", "")
    source_content_hash = note.get("source_content_hash")
    mrn = note.get("mrn")

    try:
        extraction = client.extract_single_note(
            note_text=note_text,
            system_prompt=config.system_prompt,
            registry=registry,
            note_id=note_id,
        )

        record = _serialize_extraction_result(
            note_id=note_id,
            definition_name=config.name,
            extraction=extraction,
            run_meta=run_meta,
            source_content_hash=source_content_hash,
            mrn=mrn,
        )

        # Propagate gate classification for gated registries
        if hasattr(registry, "gate_result") and registry.gate_result:
            record["gate_result"] = registry.gate_result
    except Exception as e:
        logger.exception("Exception processing note %s", note_id)
        return {
            "note_id": note_id,
            "mrn": str(mrn) if mrn is not None else None,
            "definition_name": config.name,
            "source_content_hash": source_content_hash,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "events": {},
            "finish": None,
            "rounds": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "error": str(e),
            "run_meta": run_meta,
        }
    else:
        return record


def run_fc_single_batch(
    registry: ToolRegistry,
    config: SingleNoteConfig,
    client: FunctionCallingClient,
    db_path: Path | str | None,
    source_table: str | None,
    output_dir: Path | str,
    batch_name: str,
    text_col: str = "note_text",
    id_col: str = "note_id",
    limit: int | None = None,
    where: str | None = None,
    note_ids: set[str] | None = None,
    resume: bool = True,
    retry_failed: bool = False,
    rate_limit: float = 1.0,
    backend_name: str | None = None,
    workers: int = 1,
    progress: bool = True,
    jsonl_path: Path | str | None = None,
) -> SingleNoteBatchResult:
    """
    Run batch single-note function-calling extraction.

    Each note is processed independently (no timeline state carried between notes).

    The notes can come from either a DuckDB source table (``db_path`` +
    ``source_table``) or a JSONL file (``jsonl_path``) — exactly one must be
    supplied. JSONL mode skips DuckDB entirely and is incompatible with
    ``where`` filtering.

    Args:
        registry: Tool registry with extraction tools
        config: SingleNoteConfig with name and system_prompt
        client: Configured FunctionCallingClient
        db_path: Path to DuckDB database (required when ``jsonl_path`` is None;
            for JSONL mode, the file path is reused here for run-log provenance)
        source_table: Source table (e.g., raw.pathology). Required for DuckDB
            mode; for JSONL mode, defaults to ``jsonl:<filename>`` for the
            manifest's "where did this come from" field.
        output_dir: Directory to write JSONL output
        batch_name: Name for this batch (used in output filename)
        text_col: Column / JSON key containing note text
        id_col: Column / JSON key containing note identifier
        limit: Max notes to process
        where: Optional SQL WHERE clause (DuckDB mode only)
        note_ids: Optional set of note IDs to process
        resume: If True, skip already-processed (note_id, source_content_hash)
            pairs in an existing batch JSONL
        retry_failed: If True, drop previously-failed records from the existing
            batch JSONL (backing it up to ``<path>.bak``) and re-run those
            note_ids. Mutually exclusive with the default resume semantics that
            also skip failed records.
        rate_limit: Seconds to wait between notes (ignored when workers > 1)
        backend_name: Resolved oncai.yaml backend name (e.g. "vllm-local").
            Recorded in the manifest and run_meta. Falls back to the client
            class name if not provided.
        workers: Number of concurrent workers for parallel processing
        progress: Show progress bar
        jsonl_path: If set, load notes from this JSONL file instead of querying
            DuckDB. Mutually exclusive with ``db_path``+``source_table``.

    Returns:
        SingleNoteBatchResult with statistics
    """
    if jsonl_path is not None:
        jsonl_path = Path(jsonl_path)
        # When loading from JSONL, record the file path in db_path and a
        # readable label in source_table so the manifest/run_meta still have
        # meaningful "where did this come from" fields.
        source_table = source_table or f"jsonl:{jsonl_path.name}"
        if db_path is None:
            db_path = jsonl_path
    elif db_path is None or source_table is None:
        raise ValueError(
            "run_fc_single_batch requires either jsonl_path or "
            "(db_path and source_table)"
        )
    db_path = Path(db_path)
    output_dir = Path(output_dir)
    output_path = output_dir / config.name / f"{batch_name}.jsonl"
    manifest_path = output_path.with_name(output_path.stem + "_manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()

    # Build run-level metadata once
    run_meta = _build_run_meta(
        batch_name=batch_name,
        source_table=source_table,
        client=client,
        system_prompt=config.system_prompt,
        backend_name=backend_name,
    )

    # Retry-failed: prune old failed records so they get reprocessed and we
    # don't end up with duplicate records per note_id.
    if retry_failed and output_path.exists():
        kept, dropped = _prune_failed_records(output_path)
        if progress and dropped:
            tqdm.write(
                f"Retry-failed: dropped {dropped} failed record(s), kept {kept} successful "
                f"(backup at {output_path.name}.bak)"
            )

    # Resume support — when retry-failed, only count successful records as done.
    # Keys are (note_id, source_content_hash) tuples; addenda (same note_id,
    # new content_hash) correctly fall out as un-done. Records produced by
    # pre-incremental code carry source_content_hash=None — those match by
    # note_id only via the legacy fallback below.
    existing_keys = (
        _get_existing_extraction_keys(output_path, successful_only=retry_failed)
        if resume
        else set()
    )
    legacy_done_ids = {nid for (nid, h) in existing_keys if h is None}
    if existing_keys and progress:
        tqdm.write(f"Resuming: {len(existing_keys)} records already processed")

    # Load notes
    if jsonl_path is not None:
        if where:
            logger.warning(
                "where=%r is ignored when loading from JSONL (no SQL layer)", where
            )
        notes = _load_notes_from_jsonl(
            jsonl_path=jsonl_path,
            text_col=text_col,
            id_col=id_col,
            limit=limit,
            note_ids=note_ids,
        )
    else:
        # Without a jsonl path we go through DuckDB, which requires both
        # db_path and source_table. Callers enforce this upstream (the
        # fc run-single CLI validates the source/jsonl pair) so this guard
        # is for the type-checker; should be unreachable at runtime.
        if db_path is None or source_table is None:
            raise RuntimeError(
                "db_path and source_table are required when jsonl_path is not set"
            )
        notes = _load_notes(
            db_path=db_path,
            source_table=source_table,
            text_col=text_col,
            id_col=id_col,
            limit=limit,
            where=where,
            note_ids=note_ids,
        )

    # Filter out already-processed and empty notes up front
    pending_notes: list[dict[str, Any]] = []
    skipped = 0
    for note in notes:
        note_id = str(note["note_id"])
        note_text = note.get("note_text", "")
        source_hash = note.get("source_content_hash")
        if (note_id, source_hash) in existing_keys or note_id in legacy_done_ids:
            skipped += 1
            continue
        if not note_text or not note_text.strip():
            logger.warning(f"Empty note text for {note_id}, skipping")
            skipped += 1
            continue
        pending_notes.append(note)

    result = SingleNoteBatchResult(
        output_path=str(output_path),
        total_notes=len(notes),
        skipped=skipped,
    )

    if not pending_notes:
        # Still drop a manifest so the batch has a "done marker" even when there
        # was nothing left to do (e.g., all notes already processed under resume).
        # skip_if_no_work avoids overwriting a completed manifest with zeroed
        # counts on a no-op resume.
        _write_single_note_manifest(
            manifest_path=manifest_path,
            config=config,
            client=client,
            registry=registry,
            db_path=db_path,
            source_table=source_table,
            text_col=text_col,
            id_col=id_col,
            limit=limit,
            where=where,
            note_ids_count=len(note_ids) if note_ids is not None else None,
            workers=workers,
            rate_limit=rate_limit,
            resume=resume,
            started_at=started_at,
            started_monotonic=started_monotonic,
            result=result,
            backend_name=backend_name,
            skip_if_no_work=True,
        )
        return result

    write_lock = threading.Lock()

    def _write_record(out_file, record: dict[str, Any]) -> None:
        line = json.dumps(record) + "\n"
        with write_lock:
            out_file.write(line)
            out_file.flush()

    def _update_stats(record: dict[str, Any]) -> None:
        with write_lock:
            if record["success"]:
                result.successful += 1
            else:
                result.failed += 1
                err = record.get("error")
                if err:
                    result.errors.append(
                        {"note_id": record.get("note_id", ""), "error": str(err)}
                    )
            result.total_input_tokens += record.get("input_tokens", 0)
            result.total_output_tokens += record.get("output_tokens", 0)
            result.total_reasoning_tokens += record.get("reasoning_tokens", 0)
            for tool_name, events in (record.get("events") or {}).items():
                if isinstance(events, list):
                    result.events_by_type[tool_name] = result.events_by_type.get(
                        tool_name, 0
                    ) + len(events)
            for tool_name, plans in (record.get("plans") or {}).items():
                if isinstance(plans, list):
                    result.plans_by_type[tool_name] = result.plans_by_type.get(
                        tool_name, 0
                    ) + len(plans)

    pbar = tqdm(
        total=len(pending_notes),
        desc=f"Notes (workers={workers})",
        disable=not progress,
        unit="note",
    )

    with output_path.open(mode="a") as out_file:
        if workers <= 1:
            # Sequential path
            for note in pending_notes:
                record = _process_single_note(
                    note=note,
                    config=config,
                    client=client,
                    registry=registry,
                    run_meta=run_meta,
                )
                _write_record(out_file, record)
                _update_stats(record)
                pbar.update(1)
                pbar.set_postfix(ok=result.successful, fail=result.failed)

                if rate_limit > 0:
                    time.sleep(rate_limit)
        else:
            # Parallel path
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _process_single_note,
                        note=note,
                        config=config,
                        client=client,
                        registry=registry,
                        run_meta=run_meta,
                    ): note
                    for note in pending_notes
                }

                def _handle(future: Future) -> None:
                    record = future.result()
                    _write_record(out_file, record)
                    _update_stats(record)
                    pbar.update(1)
                    pbar.set_postfix(ok=result.successful, fail=result.failed)

                _drain_futures_with_cancel(list(futures.keys()), _handle, label="notes")

    pbar.close()

    # Write the manifest as the final "done marker" — only after the JSONL is
    # closed, so its presence indicates the batch finished cleanly.
    _write_single_note_manifest(
        manifest_path=manifest_path,
        config=config,
        client=client,
        registry=registry,
        db_path=db_path,
        source_table=source_table,
        text_col=text_col,
        id_col=id_col,
        limit=limit,
        where=where,
        note_ids_count=len(note_ids) if note_ids is not None else None,
        workers=workers,
        rate_limit=rate_limit,
        resume=resume,
        started_at=started_at,
        started_monotonic=started_monotonic,
        result=result,
        backend_name=backend_name,
    )
    if progress:
        tqdm.write(f"Manifest saved: {manifest_path}")

    return result
