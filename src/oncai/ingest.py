"""Ingest pipeline: turn inbox files into lake parquets.

Dispatches per-folder based on ``FOLDER_MODES``:

- ``DATED``: replay all ISO-prefixed inbox files in date order, rebuilding
  ``lake/<folder>/<folder>.parquet`` from scratch via key/content-hash merge.
- ``STATIC``: each inbox file maps to its own lake parquet (filename stem
  becomes the parquet name and the eventual duckdb table name).
- ``NAMED``: filename = identity (cohorts).
- ``LAKE_ONLY``: skipped — these folders are populated by other paths.

Per-file transforms (collation, hashing, type cleanup) stay in this layer.
Cross-folder relational SQL lives in ``oncai.db`` (the duckdb build).
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from oncai.config import (
    DATED_FILENAME_RE,
    FOLDER_MODES,
    IngestMode,
    OncaiConfig,
)
from oncai.lake import merge_dataframes
from oncai.schemas.pathology import PATHOLOGY_SCHEMA
from oncai.sidecar import ensure_sidecar
from oncai.transforms import collate_pathology, passthrough_transform

# --- result types ---------------------------------------------------------


@dataclass
class FileStats:
    name: str
    new_rows: int
    updated_rows: int
    unchanged_rows: int


@dataclass
class LakeDelta:
    """Difference between the replay's final accumulator and the existing lake parquet.

    Tells the user whether running ingest will actually change disk state — distinct
    from per-file FileStats, which describe each file's contribution to the build.
    """

    output_name: str
    new_rows: int
    updated_rows: int
    unchanged_rows: int


@dataclass
class FolderResult:
    folder: str
    mode: IngestMode
    files: list[FileStats] = field(default_factory=list)
    row_count: int = 0
    written_paths: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    deltas: list[LakeDelta] = field(default_factory=list)


def _compute_lake_delta(
    name: str,
    accumulator: pl.DataFrame,
    lake_path: Path,
    *,
    ignore_cols: tuple[str, ...] = (),
) -> LakeDelta:
    """Diff ``accumulator`` against an existing lake parquet (if any).

    Uses key_hash/content_hash for the new/updated/unchanged split. If those
    columns aren't present (e.g. pass-through outputs without key/content
    hashes), falls back to an unkeyed row-set diff.

    ``ignore_cols`` are dropped from the existing frame before comparison —
    used by cohorts to exclude inline meta columns (``cohort_name``,
    ``cohort_created_at``) so the delta reflects real data changes.
    """
    existing = pl.read_parquet(lake_path) if lake_path.exists() else None
    if existing is None or existing.is_empty():
        return LakeDelta(
            output_name=name,
            new_rows=len(accumulator),
            updated_rows=0,
            unchanged_rows=0,
        )
    if ignore_cols:
        drop = [c for c in ignore_cols if c in existing.columns]
        if drop:
            existing = existing.drop(drop)
    if set(accumulator.columns) != set(existing.columns):
        return LakeDelta(
            output_name=name,
            new_rows=len(accumulator),
            updated_rows=0,
            unchanged_rows=0,
        )
    if "key_hash" in accumulator.columns and "content_hash" in accumulator.columns:
        # Vectorized via polars joins on (key_hash, content_hash) so duplicate
        # keys with different content are each their own identity. This
        # avoids the merge_dataframes dict-collapse, which mis-counts
        # rows whenever a transform's `.unique()` keeps multiple rows for
        # the same business key (e.g. ihc_results with same report/specimen/
        # test but different test_result).
        pair_cols = ["key_hash", "content_hash"]
        unchanged_df = accumulator.join(existing, on=pair_cols, how="semi")
        unchanged = len(unchanged_df)
        unmatched = accumulator.join(existing, on=pair_cols, how="anti")
        # Of the unmatched, rows whose key_hash IS in existing are "updated"
        # (same business key, different content); the rest are truly new.
        existing_keys = existing.select("key_hash").unique()
        updated_df = unmatched.join(existing_keys, on="key_hash", how="semi")
        updated = len(updated_df)
        new_count = len(unmatched) - updated
        return LakeDelta(
            output_name=name,
            new_rows=new_count,
            updated_rows=updated,
            unchanged_rows=unchanged,
        )
    # Unkeyed diff: vectorized via polars semi-join on all columns.
    # - nulls_equal=True: NULL==NULL, necessary for sparse outputs where
    #   most fields are null on any given row.
    # - Align dtypes first: a fresh transform may produce Datetime[us] while
    #   the existing parquet stored Datetime[ns] (or vice-versa); the values
    #   are conceptually equal but join compares the underlying ints.
    cols = list(accumulator.columns)
    aligned = _align_schema(accumulator, existing)
    common = aligned.join(existing, on=cols, how="semi", nulls_equal=True)
    new_count = len(aligned) - len(common)
    return LakeDelta(
        output_name=name,
        new_rows=new_count,
        updated_rows=0,
        unchanged_rows=len(common),
    )


def _align_schema(df: pl.DataFrame, target: pl.DataFrame) -> pl.DataFrame:
    """Cast ``df``'s columns to ``target``'s dtypes where they differ.

    Used before unkeyed diffs so type precision (e.g. Datetime[us] vs [ns],
    Int32 vs Int64) doesn't masquerade as data drift. Casts are non-strict so
    a value that genuinely doesn't fit produces NULL rather than raising.
    """
    casts = []
    df_schema = df.schema
    for col, target_dtype in target.schema.items():
        if col not in df_schema:
            continue
        if df_schema[col] != target_dtype:
            casts.append(pl.col(col).cast(target_dtype, strict=False))
    return df.with_columns(casts) if casts else df


# --- shared helpers -------------------------------------------------------


def _atomic_write_parquet(df: pl.DataFrame, dest: Path) -> None:
    """Write ``df`` to ``dest`` atomically: write to ``.tmp``, fsync, rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.replace(dest)


def _collect_inbox_files(
    inbox_dir: Path, extensions: tuple[str, ...] = (".csv", ".jsonl")
) -> list[Path]:
    """Return inbox files for the given extensions, excluding sidecars."""
    if not inbox_dir.exists():
        return []
    files: list[Path] = []
    for ext in extensions:
        files.extend(inbox_dir.glob(f"*{ext}"))
    return files


def _validate_dated_filenames(files: list[Path]) -> list[Path]:
    """Hard-fail if any filename doesn't match the dated pattern. Returns sorted list."""
    bad = [f for f in files if not DATED_FILENAME_RE.match(f.name)]
    if bad:
        names = "\n".join(f"  - {f.name}" for f in sorted(bad))
        raise ValueError(
            "Inbox files must match YYYY-MM-DD_<label>.{csv,jsonl}. "
            f"Offenders:\n{names}"
        )
    return sorted(files, key=lambda p: p.name)


def _ensure_sidecars(files: list[Path]) -> None:
    """Compute and write a SHA-256 sidecar for any inbox file missing one."""
    for f in files:
        ensure_sidecar(f)


# --- DATED single-output replay ------------------------------------------


def _replay_single_output(
    folder: str,
    config: OncaiConfig,
    transform_file: Callable[[Path], tuple[pl.DataFrame, list[str]]],
    *,
    extensions: tuple[str, ...] = (".csv",),
    dry_run: bool = False,
) -> FolderResult:
    """Replay all dated inbox files into ``lake/<folder>/<folder>.parquet``."""
    inbox_dir = config.inbox_path / folder
    files = _collect_inbox_files(inbox_dir, extensions=extensions)
    files = _validate_dated_filenames(files)
    _ensure_sidecars(files)

    result = FolderResult(folder=folder, mode=IngestMode.DATED)

    accumulator: pl.DataFrame | None = None
    for f in files:
        df, file_notes = transform_file(f)
        result.notes.extend(file_notes)
        merged, stats = merge_dataframes(df, accumulator)
        accumulator = merged
        result.files.append(
            FileStats(
                name=f.name,
                new_rows=stats["new_rows"],
                updated_rows=stats["updated_rows"],
                unchanged_rows=stats["unchanged_rows"],
            )
        )

    if accumulator is None:
        return result

    out = config.lake_path / folder / f"{folder}.parquet"
    result.deltas.append(_compute_lake_delta(folder, accumulator, out))
    if not dry_run:
        _atomic_write_parquet(accumulator, out)
    result.row_count = len(accumulator)
    result.written_paths.append(out)
    return result


def _transform_pathology(path: Path) -> tuple[pl.DataFrame, list[str]]:
    """Pathology: rename common Epic columns, then either collate multi-line
    reports or pass already-clean ones through untouched.

    Returns the transformed DataFrame plus a list of human-readable notes
    describing any renames or inferences applied — these are surfaced via
    ``FolderResult.notes`` so the user sees exactly what ingest did to
    their input (and can rename the columns upstream if they prefer).
    """
    notes: list[str] = []
    lf = pl.scan_csv(path)
    cols = lf.collect_schema().names()

    # Friendly aliases for common Epic Clarity / Caboodle column names.
    renames: dict[str, str] = {}
    if "pat_mrn_id" in cols and "mrn" not in cols:
        renames["pat_mrn_id"] = "mrn"
    if "case_num" in cols and "report_id" not in cols:
        renames["case_num"] = "report_id"
    if renames:
        lf = lf.rename(renames)
        cols = lf.collect_schema().names()
        for old, new in renames.items():
            notes.append(f"{path.name}: renamed column '{old}' → '{new}'")

    # Already-clean reports: one row per report with a ``report_text`` column
    # and no multi-line ``mult_ln_val_storage`` to collate. Pass them through
    # untouched — schema validation + hashes only, no collation or text
    # cleaning — so reports that are already ready to go are not rewritten.
    if "report_text" in cols and "mult_ln_val_storage" not in cols:
        notes.append(
            f"{path.name}: report_text present and no mult_ln_val_storage — "
            "treating reports as already clean (skipping collation)"
        )
        return passthrough_transform(lf, PATHOLOGY_SCHEMA).collect(), notes  # type: ignore[return-value]

    if "row_id" not in cols:
        # Non-interactive ingest — we trust the user's CSV is in
        # top-to-bottom order per report_id.
        notes.append(
            f"{path.name}: no row_id column — inferring from CSV line order "
            "(rows must be sorted top-to-bottom per report_id)"
        )
        lf = lf.with_row_index("row_id", offset=1)

    return collate_pathology(lf).collect(), notes  # type: ignore[return-value]


# --- STATIC: fc_extractions ---------------------------------------------


_BATCH_VERSION_SUFFIX = re.compile(r"\.v\d+$")


def _base_batch_name(jsonl_stem: str) -> str:
    """Strip a trailing ``.v<N>`` from a JSONL stem to get the base batch name.

    ``foo.jsonl`` → ``foo``; ``foo.v2.jsonl`` → ``foo``; ``foo.v17.jsonl`` →
    ``foo``. Anything else (including names with embedded dots that aren't
    a version suffix, like ``foo.bar``) is returned unchanged.
    """
    return _BATCH_VERSION_SUFFIX.sub("", jsonl_stem)


def _ingest_fc_extractions(
    config: OncaiConfig, *, dry_run: bool = False
) -> FolderResult:
    """fc_extractions: group inbox JSONLs by base batch name, merge each group
    into one wide lake parquet, keeping the latest ``extracted_at`` per
    ``record_id``.

    JSONLs ending in ``.v<N>.jsonl`` (e.g. ``foo.v2.jsonl``) are grouped with
    their baseline ``foo.jsonl`` and any other versions, all collapsed into
    ``lake/fc_extractions/foo.parquet``. Each row's ``batch_name`` column
    tracks its source JSONL stem so per-version provenance survives the merge.

    Failed records are dropped from the lake parquet — diagnostics live in
    the per-version manifest + ``oncai fc status``.

    The base batch name the user gives in inbox becomes the parquet name
    and (after ``oncai db update fc_extractions``) the table name in the
    ``extractions_raw`` duckdb schema.
    """
    from oncai.fc_extraction.load import merge_versioned_jsonls_to_parquet

    folder = "fc_extractions"
    inbox_dir = config.inbox_path / folder
    files = sorted(
        _collect_inbox_files(inbox_dir, extensions=(".jsonl",)), key=lambda p: p.name
    )
    _ensure_sidecars(files)

    # Group by base batch name. Versions within a group are sorted so that
    # higher-N variants come last — the merge tie-breaker prefers the
    # higher-version source on identical extracted_at timestamps.
    groups: dict[str, list[Path]] = {}
    for f in files:
        groups.setdefault(_base_batch_name(f.stem), []).append(f)
    for v in groups.values():
        v.sort(key=lambda p: p.name)

    result = FolderResult(folder=folder, mode=IngestMode.STATIC)
    for batch_name, jsonl_files in groups.items():
        out = config.lake_path / folder / f"{batch_name}.parquet"

        # Build the would-be merged DataFrame in memory first (dry_run=True)
        # so the delta diffs against the on-disk parquet BEFORE we write.
        load_result = merge_versioned_jsonls_to_parquet(
            jsonl_files, out, only_successful=True, dry_run=True
        )

        if load_result.df is not None:
            result.deltas.append(_compute_lake_delta(batch_name, load_result.df, out))
            if not dry_run:
                _atomic_write_parquet(load_result.df, out)
                result.row_count += load_result.written
                result.written_paths.append(out)

        # Per-version manifests stay separate — each <stem>_manifest.json is
        # copied to the lake folder under its own name. That preserves the
        # provenance trail for each individual run.
        for f in jsonl_files:
            manifest_src = f.with_name(f.stem + "_manifest.json")
            manifest_dst = config.lake_path / folder / manifest_src.name
            if manifest_src.exists():
                if not dry_run:
                    manifest_dst.parent.mkdir(parents=True, exist_ok=True)
                    import shutil

                    shutil.copy2(manifest_src, manifest_dst)
                    result.written_paths.append(manifest_dst)
                result.notes.append(f"{f.name}: copied manifest")
            else:
                result.notes.append(
                    f"{f.name}: no manifest peer found (provenance metadata unavailable)"
                )

        result.files.append(
            FileStats(
                name=(
                    jsonl_files[0].name
                    if len(jsonl_files) == 1
                    else f"{batch_name} ({len(jsonl_files)} versions)"
                ),
                new_rows=load_result.written,
                updated_rows=0,
                # Failures don't appear in the parquet — report them via the
                # "unchanged" slot so the count surfaces in the CLI summary.
                unchanged_rows=load_result.skipped_failed,
            )
        )
        if load_result.skipped_failed:
            result.notes.append(
                f"{batch_name}: dropped {load_result.skipped_failed} failed record(s) "
                f"across {len(jsonl_files)} version(s)"
            )
        if load_result.record_kind:
            result.notes.append(f"{batch_name}: detected {load_result.record_kind}")
    return result


# --- NAMED: cohorts -----------------------------------------------------


def _ingest_cohorts(config: OncaiConfig, *, dry_run: bool = False) -> FolderResult:
    """cohorts: each CSV is a separate cohort, filename = cohort name.

    Key column auto-detected from the first match of ``COHORT_KEY_PRIORITY``
    (mrn → note_id → path_id → report_id). Drop a CSV with any of those
    columns into ``inbox/cohorts/`` and ingest figures it out.
    """
    import json
    from dataclasses import asdict

    from oncai.cohort import (
        COHORT_META_COLUMNS,
        CohortMetadata,
        _cohorts_dir,
        _sidecar_path,
        prepare_cohort_df,
        resolve_created_at,
        with_meta_columns,
    )

    folder = "cohorts"
    inbox_dir = config.inbox_path / folder
    files = sorted(
        _collect_inbox_files(inbox_dir, extensions=(".csv",)), key=lambda p: p.name
    )
    _ensure_sidecars(files)

    result = FolderResult(folder=folder, mode=IngestMode.NAMED)
    cohorts_dir = _cohorts_dir(config.lake_path)
    for f in files:
        try:
            new_df, key_column = prepare_cohort_df(f)
        except ValueError as e:
            result.notes.append(f"{f.name}: {e}")
            continue

        out = cohorts_dir / f"{f.stem}.parquet"
        result.deltas.append(
            _compute_lake_delta(f.stem, new_df, out, ignore_cols=COHORT_META_COLUMNS)
        )
        result.notes.append(
            f"{f.name}: keyed on '{key_column}'" + (" (would write)" if dry_run else "")
        )
        result.files.append(
            FileStats(
                name=f.name,
                new_rows=len(new_df),
                updated_rows=0,
                unchanged_rows=0,
            )
        )

        if not dry_run:
            cohorts_dir.mkdir(parents=True, exist_ok=True)
            created_at = resolve_created_at(out)
            df_to_write = with_meta_columns(new_df, name=f.stem, created_at=created_at)
            df_to_write.write_parquet(out, compression="zstd")
            metadata = CohortMetadata(
                name=f.stem,
                description="",
                key_column=key_column,
                created_at=created_at,
                row_count=len(df_to_write),
                columns=df_to_write.columns,
                source_file=f.name,
            )
            with _sidecar_path(out).open("w") as out_f:
                json.dump(asdict(metadata), out_f, indent=2)
            result.row_count += len(new_df)
            result.written_paths.append(out)

    return result


# --- top-level dispatch -------------------------------------------------


def run_ingest(
    config: OncaiConfig,
    folder: str | None = None,
    *,
    dry_run: bool = False,
) -> list[FolderResult]:
    """Run ingest for one folder or all folders.

    Returns one FolderResult per folder processed.
    """
    if folder is not None:
        if folder not in FOLDER_MODES:
            raise ValueError(f"Unknown folder: {folder}")
        targets = [folder]
    else:
        targets = list(FOLDER_MODES.keys())

    results: list[FolderResult] = []
    for f in targets:
        mode = FOLDER_MODES[f]
        if mode == IngestMode.LAKE_ONLY:
            continue
        result = _dispatch(f, config, dry_run=dry_run)
        if result is not None:
            results.append(result)
            _mirror_sql_files(f, config, result, dry_run=dry_run)
    return results


def _validate_sql_syntax(sql_path: Path) -> str | None:
    """Parse-check a SQL file via duckdb. Returns error message or None.

    Catches typos and bad SQL grammar at ingest time, so the user learns
    about the problem now rather than at ``oncai build-db``. Semantic
    checks (referenced table exists, column types match) still happen
    later when the SQL actually runs against the lake parquets.
    """
    text = sql_path.read_text()
    if not text.strip():
        return None
    try:
        import duckdb
    except ImportError:
        return None  # duckdb not installed; skip validation gracefully

    try:
        con = duckdb.connect()
        try:
            # extract_statements parses the SQL into statements without
            # executing it; raises on parse errors.
            con.extract_statements(text)
            return None
        finally:
            con.close()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _mirror_sql_files(
    folder: str,
    config: OncaiConfig,
    result: FolderResult,
    *,
    dry_run: bool,
) -> None:
    """Copy any ``*.sql`` files from ``inbox/<folder>/`` into ``lake/<folder>/``.

    SQL transforms are co-located with the parquets they operate on. Build-db
    looks for ``<parquet_stem>.sql`` next to each lake parquet and executes it
    after creating the base table. Per-folder ``<folder>.sql`` is also
    supported for cross-parquet joins in DATED folders.

    SQL files are parse-validated before being mirrored — broken SQL is
    surfaced via ``result.notes`` and not copied to lake, so the lake
    stays in a runnable state.
    """
    import shutil

    inbox_dir = config.inbox_path / folder
    if not inbox_dir.exists():
        return
    sql_files = sorted(inbox_dir.glob("*.sql"))
    if not sql_files:
        return

    lake_dir = config.lake_path / folder
    for src in sql_files:
        err = _validate_sql_syntax(src)
        if err:
            result.notes.append(f"{src.name}: SQL invalid — {err} (not mirrored)")
            continue
        dst = lake_dir / src.name
        if not dry_run:
            lake_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            result.written_paths.append(dst)
        result.notes.append(f"{src.name}: SQL transform mirrored to lake")


def _dispatch(
    folder: str, config: OncaiConfig, *, dry_run: bool
) -> FolderResult | None:
    """Route ``folder`` to its handler. Returns None if inbox is empty."""
    inbox_dir = config.inbox_path / folder
    if not inbox_dir.exists() or not any(inbox_dir.iterdir()):
        return None

    if folder == "pathology":
        return _replay_single_output(
            folder, config, _transform_pathology, dry_run=dry_run
        )
    if folder == "fc_extractions":
        return _ingest_fc_extractions(config, dry_run=dry_run)
    if folder == "cohorts":
        return _ingest_cohorts(config, dry_run=dry_run)

    raise RuntimeError(f"No ingest handler for folder: {folder}")
