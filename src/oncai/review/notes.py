"""Load the source notes a review package needs to show alongside events.

The review app renders each extracted event next to the *text of the report it
came from*, with the evidence snippets highlighted inline. The extraction JSONL
only carries the structured findings, not the note text — so when building a
package we re-read the source notes (text + a little display metadata) keyed by
``note_id``.

Two sources, mirroring how a batch was run:

- **DuckDB** (the common case): ``oncai fc run-single --source raw.pathology``.
  We query the source table for the note text, MRN, and whatever display
  columns it happens to expose (date / type / department), auto-detecting which
  of a small set of candidate column names are present.
- **JSONL** (``--source`` was a file): the batch read notes from a ``.jsonl``;
  we re-read the same file. Only text + MRN are available there.

Either way the result is ``{note_id: {note_text, mrn, note_date, note_type,
department}}`` — the shape ``web/app.js`` reads (``note.note_text``,
``note.note_date``, ``note.note_type``, ``note.department``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

# Candidate source columns, in priority order, for each display field. The first
# one that exists on the source table wins. note_text/id are passed explicitly
# (they match the batch's --text-col / --id-col), so they're not listed here.
_DATE_COLS = ("note_date", "ordering_date", "report_date", "collected_date")
_TYPE_COLS = ("note_type", "external_name", "report_type")
_DEPT_COLS = ("department", "dept", "service")

# The keys every note dict carries, so callers can rely on a stable shape.
_NOTE_FIELDS = ("note_text", "mrn", "note_date", "note_type", "department")


def _empty_note() -> dict[str, Any]:
    return dict.fromkeys(_NOTE_FIELDS)


def _has_column(con: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    """Whether ``table`` (``schema.table`` or bare) exposes ``column``."""
    if "." in table:
        schema, name = table.split(".", 1)
    else:
        schema, name = "main", table
    row = con.execute(
        """
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ? AND column_name = ?
        """,
        [schema, name, column],
    ).fetchone()
    return bool(row and row[0])


def _first_present(
    con: duckdb.DuckDBPyConnection, table: str, candidates: tuple[str, ...]
) -> str | None:
    """First column name from ``candidates`` that exists on ``table``, else None."""
    for col in candidates:
        if _has_column(con, table, col):
            return col
    return None


def load_notes_from_duckdb(
    *,
    db_path: Path,
    source_table: str,
    note_ids: set[str] | None = None,
    text_col: str = "report_text",
    id_col: str = "report_id",
) -> dict[str, dict[str, Any]]:
    """Load source notes from a DuckDB table, keyed by ``note_id``.

    Selects ``id_col`` and ``text_col`` plus ``mrn`` and the first available
    display column for date / type / department. Restricts to ``note_ids`` when
    given (the events actually in the package), otherwise loads the whole table.
    """
    import duckdb

    notes: dict[str, dict[str, Any]] = {}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        has_mrn = _has_column(con, source_table, "mrn")
        date_col = _first_present(con, source_table, _DATE_COLS)
        type_col = _first_present(con, source_table, _TYPE_COLS)
        dept_col = _first_present(con, source_table, _DEPT_COLS)

        selects = [
            f'"{id_col}" AS note_id',
            f'"{text_col}" AS note_text',
            (f'"{date_col}" AS note_date' if date_col else "NULL AS note_date"),
            (f'"{type_col}" AS note_type' if type_col else "NULL AS note_type"),
            (f'"{dept_col}" AS department' if dept_col else "NULL AS department"),
            ('"mrn" AS mrn' if has_mrn else "NULL AS mrn"),
        ]
        query = f"SELECT {', '.join(selects)} FROM {source_table}"  # noqa: S608
        rows = con.execute(query).fetchall()
        columns = [desc[0] for desc in con.description]
    finally:
        con.close()

    for row in rows:
        record = dict(zip(columns, row, strict=True))
        note_id = str(record.get("note_id", ""))
        if note_ids is not None and note_id not in note_ids:
            continue
        notes[note_id] = {
            "note_text": _as_str(record.get("note_text")),
            "mrn": _as_str(record.get("mrn")),
            "note_date": _as_str(record.get("note_date")),
            "note_type": _as_str(record.get("note_type")),
            "department": _as_str(record.get("department")),
        }
    return notes


def load_notes_from_jsonl(
    *,
    jsonl_path: Path,
    note_ids: set[str] | None = None,
    text_col: str = "report_text",
    id_col: str = "report_id",
) -> dict[str, dict[str, Any]]:
    """Load source notes from the JSONL the batch was run against.

    Only ``note_text`` and ``mrn`` are available from an arbitrary notes JSONL;
    the date / type / department display fields are left ``None``.
    """
    from oncai.fc_extraction.batch_single import _load_notes_from_jsonl

    rows = _load_notes_from_jsonl(
        jsonl_path=jsonl_path,
        text_col=text_col,
        id_col=id_col,
        note_ids=note_ids,
    )
    notes: dict[str, dict[str, Any]] = {}
    for row in rows:
        note = _empty_note()
        note["note_text"] = _as_str(row.get("note_text"))
        note["mrn"] = _as_str(row.get("mrn"))
        notes[str(row["note_id"])] = note
    return notes


def load_source_notes(
    *,
    source_table: str | None,
    db_path: Path | None,
    note_ids: set[str] | None = None,
    text_col: str = "report_text",
    id_col: str = "report_id",
) -> dict[str, dict[str, Any]]:
    """Load source notes from wherever the batch read them.

    Dispatches on ``source_table``: a ``jsonl:<file>`` label (how
    ``run_fc_single_batch`` records a ``--jsonl`` run) reads the file back;
    anything else is treated as a DuckDB table and queried via ``db_path``.

    Note loading is best-effort: if the source can't be read (table renamed,
    file moved, DB absent) we log and return an empty map. The package still
    builds — the app just shows "(note text unavailable)" for those events.
    """
    try:
        if source_table and source_table.startswith("jsonl:"):
            jsonl_path = Path(source_table.removeprefix("jsonl:"))
            if not jsonl_path.exists() and db_path is not None:
                # In JSONL mode run_fc_single_batch stores the file path in
                # db_path too; fall back to it if the label's path moved.
                jsonl_path = Path(db_path)
            return load_notes_from_jsonl(
                jsonl_path=jsonl_path,
                note_ids=note_ids,
                text_col=text_col,
                id_col=id_col,
            )
        if db_path is None or source_table is None:
            logger.warning(
                "Cannot load source notes: missing db_path or source_table "
                "(notes will be unavailable in the review package)"
            )
            return {}
        return load_notes_from_duckdb(
            db_path=db_path,
            source_table=source_table,
            note_ids=note_ids,
            text_col=text_col,
            id_col=id_col,
        )
    except Exception as exc:
        logger.warning(
            "Failed to load source notes from %s (%s); review package will have "
            "no note text",
            source_table,
            exc,
        )
        return {}


def _as_str(value: Any) -> str | None:
    """Stringify a DB value for JSON, preserving None (dates -> ISO via str)."""
    if value is None:
        return None
    return str(value)
