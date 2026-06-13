"""Ingest pipeline: turn inbox files into lake parquets.

Dispatches per-folder based on ``FOLDER_MODES``:

- ``DATED``: replay all ISO-prefixed inbox files in date order, rebuilding
  ``lake/<folder>/<folder>.parquet`` from scratch via key/content-hash merge.
- ``STATIC``: each inbox file maps to its own lake parquet (filename stem
  becomes the parquet name and the eventual duckdb table name). Review
  packages are the one exception: a ``*.review_pkg.json`` + ``*.reviews.jsonl``
  pair maps to one reviewed (silver) parquet.
- ``NAMED``: filename = identity (cohorts).
- ``LAKE_ONLY``: skipped — these folders are populated by other paths.

Per-file transforms (collation, hashing, type cleanup) stay in this layer.
Cross-folder relational SQL lives in ``oncai.db`` (the duckdb build).
"""

from __future__ import annotations

import os
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

_BATCH_LOCAL_SQL_FOLDERS = {"fc_extractions", "fc_reviews"}

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


def _ingest_fc_extractions(
    config: OncaiConfig, *, dry_run: bool = False
) -> FolderResult:
    """fc_extractions: each batch is a folder of numbered segments.

    ``inbox/fc_extractions/<batch>/NNN.jsonl`` are immutable, monotonically
    numbered segments. They merge into one wide lake parquet
    ``lake/fc_extractions/<batch>.parquet`` where, for each ``record_id``, the
    highest segment wins (explicit integer order — no timestamps). The batch
    folder name becomes the parquet name and (after
    ``oncai db update fc_extractions``) the table name in ``extractions_raw``.

    Failed records are dropped from the lake parquet — diagnostics live in the
    per-segment manifest + ``oncai fc status``. Manifests stay in the inbox
    (canonical, synced); the lake is a disposable projection.
    """
    from oncai.fc_extraction.load import merge_segments_to_parquet, segment_files

    folder = "fc_extractions"
    inbox_dir = config.inbox_path / folder
    result = FolderResult(folder=folder, mode=IngestMode.STATIC)
    if not inbox_dir.exists():
        return result

    for batch_dir in sorted(p for p in inbox_dir.iterdir() if p.is_dir()):
        batch_name = batch_dir.name
        segments = segment_files(batch_dir)
        if not segments:
            continue
        _ensure_sidecars([p for _, p in segments])

        out = config.lake_path / folder / f"{batch_name}.parquet"
        # Build the merged DataFrame in memory first (dry_run) so the delta
        # diffs against the on-disk parquet BEFORE we write.
        load_result = merge_segments_to_parquet(
            batch_dir, out, only_successful=True, dry_run=True
        )

        if load_result.df is not None:
            result.deltas.append(_compute_lake_delta(batch_name, load_result.df, out))
            if not dry_run:
                _atomic_write_parquet(load_result.df, out)
                result.row_count += load_result.written
                result.written_paths.append(out)

        result.files.append(
            FileStats(
                name=f"{batch_name} ({len(segments)} segment(s))",
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
                f"across {len(segments)} segment(s)"
            )
        if load_result.record_kind:
            result.notes.append(f"{batch_name}: detected {load_result.record_kind}")
    return result


# --- STATIC: fc_reviews -------------------------------------------------


def _ingest_fc_reviews(config: OncaiConfig, *, dry_run: bool = False) -> FolderResult:
    """fc_reviews: pair per-segment review packages with raw segments and logs.

    Drop ``<batch>.NNN.review_pkg.json`` + ``<batch>.NNN.reviews.jsonl`` into
    ``inbox/fc_reviews/``. The raw extraction segment at
    ``inbox/fc_extractions/<batch>/NNN.jsonl`` supplies the complete event set;
    the package supplies only the human worklist; the review log supplies
    latest-``reviewed_at``-wins verdicts and edits. Reviewed (silver) rows from
    all of a batch's segments merge (highest segment per ``note_id``) into one
    ``lake/fc_reviews/<batch>.parquet`` → ``extractions_silver.<batch>``.

    Any incomplete or invalid review batch aborts the ingest with an error
    naming each offender — silver is never built from a half-reviewed batch.
    """
    import re

    from oncai.review.load import (
        REVIEW_LOG_SUFFIX,
        REVIEW_PACKAGE_SUFFIX,
        ReviewLoadResult,
        merge_silver_segments,
        review_batch_name,
        review_to_silver_df,
    )

    folder = "fc_reviews"
    inbox_dir = config.inbox_path / folder
    result = FolderResult(folder=folder, mode=IngestMode.STATIC)
    # One canonical layout: every pair lives in a batch-named folder,
    # inbox/fc_reviews/<batch>/<batch>.NNN.review_pkg.json (written by the run
    # hook). Glob exactly one level deep — files dropped flat are not picked up.
    package_files = sorted(inbox_dir.glob(f"*/*{REVIEW_PACKAGE_SUFFIX}"))
    review_files = sorted(inbox_dir.glob(f"*/*{REVIEW_LOG_SUFFIX}"))
    _ensure_sidecars(package_files + review_files)

    packages = {review_batch_name(p): p for p in package_files}
    reviews = {review_batch_name(p): p for p in review_files}

    seg_suffix = re.compile(r"^(.*)\.(\d+)$")

    def _base_and_segment(name: str) -> tuple[str, int] | None:
        m = seg_suffix.match(name)
        if not m:
            return None
        return m.group(1), int(m.group(2))

    def _raw_segment_path(base: str, seg: int) -> Path:
        return config.inbox_path / "fc_extractions" / base / f"{seg:03d}.jsonl"

    # Group each segment's (package, reviews) pair under its base batch name.
    grouped: dict[str, list[tuple[int, Path, Path]]] = {}
    for review_batch in sorted(set(packages) | set(reviews)):
        package_path = packages.get(review_batch)
        reviews_path = reviews.get(review_batch)
        if package_path is None:
            result.notes.append(
                f"{review_batch}: missing {REVIEW_PACKAGE_SUFFIX} peer "
                f"for {reviews_path.name if reviews_path is not None else 'review log'}"
            )
            continue
        if reviews_path is None:
            result.notes.append(
                f"{review_batch}: missing {REVIEW_LOG_SUFFIX} peer "
                f"for {package_path.name}"
            )
            continue
        batch_key = _base_and_segment(review_batch)
        if batch_key is None:
            result.notes.append(
                f"{review_batch}: review files must be named "
                f"<batch>.NNN{REVIEW_PACKAGE_SUFFIX} and "
                f"<batch>.NNN{REVIEW_LOG_SUFFIX}"
            )
            continue
        base, seg = batch_key
        grouped.setdefault(base, []).append((seg, package_path, reviews_path))

    # Phase 1: build + validate each base batch's segments. A batch whose present
    # segments are all complete is queued to write; a batch with any present-but-
    # incomplete (or invalid) segment is recorded as a failure that fails ITSELF,
    # not its siblings. (A package with no reviews log at all was already skipped
    # above — you just can't build its silver yet, and that stops nothing.)
    built: dict[str, list[tuple[int, ReviewLoadResult]]] = {}
    failures: list[str] = []
    for base, segs in grouped.items():
        base_results: list[tuple[int, ReviewLoadResult]] = []
        base_failures: list[str] = []
        for seg, package_path, reviews_path in sorted(segs, key=lambda t: t[0]):
            raw_jsonl_path = _raw_segment_path(base, seg)
            if not raw_jsonl_path.exists():
                base_failures.append(
                    f"{package_path.name}: missing raw extraction segment "
                    f"{raw_jsonl_path}"
                )
                continue
            try:
                load_result = review_to_silver_df(
                    package_path, reviews_path, raw_jsonl_path
                )
            except (ValueError, TypeError) as exc:
                base_failures.append(f"{package_path.name}: {exc}")
            else:
                base_results.append((seg, load_result))
        if base_failures:
            failures.extend(base_failures)  # don't write partial silver for this batch
        else:
            built[base] = base_results

    # Phase 2: merge each complete base batch's segments into one silver parquet.
    for base, seg_results in built.items():
        merged = merge_silver_segments([(seg, lr.df) for seg, lr in seg_results])
        rejected = sum(lr.rejected_events for _, lr in seg_results)
        ignored = sum(lr.ignored_reviews for _, lr in seg_results)

        out = config.lake_path / folder / f"{base}.parquet"
        result.deltas.append(_compute_lake_delta(base, merged, out))
        if not dry_run:
            _atomic_write_parquet(merged, out)
            result.row_count += merged.height
            result.written_paths.append(out)

        result.files.append(
            FileStats(
                name=f"{base} ({len(seg_results)} segment review(s))",
                new_rows=merged.height,
                updated_rows=0,
                unchanged_rows=rejected,
            )
        )
        if rejected:
            result.notes.append(
                f"{base}: excluded {rejected} rejected event(s) from silver"
            )
        if ignored:
            result.notes.append(
                f"{base}: ignored {ignored} review record(s) with no matching "
                "package event"
            )

    # The complete batches above are already built; now fail (non-zero) naming
    # each incomplete/invalid one, so an unfinished review blocks only itself.
    if failures:
        raise ValueError(
            "Could not build silver for some review batch(es) — finish or remove "
            "the incomplete/invalid review log(s):\n  " + "\n  ".join(failures)
        )

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


# --- MANIFEST: runs ------------------------------------------------------


def _runs_delta(df: pl.DataFrame, lake_path: Path) -> LakeDelta:
    """Diff a runs DataFrame against the existing parquet, keyed by ``run_id``.

    A run's manifest is rewritten started → completed, so a re-ingest after a
    run finishes shows that ``run_id`` as ``updated`` rather than new.
    """
    existing = pl.read_parquet(lake_path) if lake_path.exists() else None
    if existing is None or existing.is_empty():
        return LakeDelta(
            output_name="runs", new_rows=df.height, updated_rows=0, unchanged_rows=0
        )
    existing_by_id = {row["run_id"]: row for row in existing.iter_rows(named=True)}
    new = updated = unchanged = 0
    for row in df.iter_rows(named=True):
        prev = existing_by_id.get(row["run_id"])
        if prev is None:
            new += 1
        elif prev == row:
            unchanged += 1
        else:
            updated += 1
    return LakeDelta(
        output_name="runs", new_rows=new, updated_rows=updated, unchanged_rows=unchanged
    )


def _ingest_runs(config: OncaiConfig, *, dry_run: bool = False) -> FolderResult:
    """runs: union per-run JSON manifests into ``lake/runs/runs.parquet``.

    Each ``inbox/runs/<run_id>.run.json`` is one row. The manifests are the
    source of truth (written started → completed by ``oncai fc run-single``);
    this projects them into a single queryable parquet keyed by ``run_id``,
    which ``oncai build-db`` then loads into the DuckDB ``runs.runs`` table.
    """
    import json

    from oncai.runs import RUN_FILE_SUFFIX, runs_to_dataframe

    folder = "runs"
    inbox_dir = config.inbox_path / folder
    files = sorted(inbox_dir.glob(f"*{RUN_FILE_SUFFIX}"))
    _ensure_sidecars(files)

    result = FolderResult(folder=folder, mode=IngestMode.MANIFEST)

    manifests: list[dict] = []
    for f in files:
        try:
            manifests.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            result.notes.append(f"{f.name}: unreadable run manifest ({e})")

    df = runs_to_dataframe(manifests)
    out = config.lake_path / folder / f"{folder}.parquet"
    delta = _runs_delta(df, out)
    result.deltas.append(delta)
    result.files.append(
        FileStats(
            name=f"{folder} ({len(files)} manifest(s))",
            new_rows=delta.new_rows,
            updated_rows=delta.updated_rows,
            unchanged_rows=delta.unchanged_rows,
        )
    )
    if not dry_run and df.height > 0:
        _atomic_write_parquet(df, out)
        result.row_count = df.height
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
    """Copy transform SQL sidecars from inbox into the lake.

    Batch folders keep SQL beside their canonical inbox artifacts:
    ``inbox/<folder>/<batch>/<batch>.sql``. The lake remains flat at
    ``lake/<folder>/<batch>.sql`` because build-db looks for
    ``<parquet_stem>.sql`` next to each lake parquet. Non-batch folders keep the
    older root-level ``inbox/<folder>/*.sql`` layout.

    SQL files are parse-validated before being mirrored — broken SQL is
    surfaced via ``result.notes`` and not copied to lake, so the lake
    stays in a runnable state.
    """
    import shutil

    inbox_dir = config.inbox_path / folder
    if not inbox_dir.exists():
        return
    lake_dir = config.lake_path / folder

    sql_pairs: list[tuple[Path, Path, str]] = []
    if folder in _BATCH_LOCAL_SQL_FOLDERS:
        for src in sorted(inbox_dir.glob("*.sql")):
            result.notes.append(
                f"{src.name}: SQL ignored — put batch SQL at "
                f"inbox/{folder}/<batch>/<batch>.sql"
            )
        for src in sorted(inbox_dir.glob("*/*.sql")):
            batch = src.parent.name
            rel_name = f"{batch}/{src.name}"
            if src.stem != batch:
                result.notes.append(
                    f"{rel_name}: SQL filename must match parent batch folder "
                    f"'{batch}' (not mirrored)"
                )
                continue
            sql_pairs.append((src, lake_dir / f"{batch}.sql", rel_name))
    else:
        sql_pairs = [
            (src, lake_dir / src.name, src.name) for src in sorted(inbox_dir.glob("*.sql"))
        ]

    if not sql_pairs:
        return

    for src, dst, rel_name in sql_pairs:
        err = _validate_sql_syntax(src)
        if err:
            result.notes.append(f"{rel_name}: SQL invalid — {err} (not mirrored)")
            continue
        if not dry_run:
            lake_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            result.written_paths.append(dst)
        if rel_name == dst.name:
            result.notes.append(f"{rel_name}: SQL transform mirrored to lake")
        else:
            result.notes.append(
                f"{rel_name}: SQL transform mirrored to lake as {dst.name}"
            )


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
    if folder == "fc_reviews":
        return _ingest_fc_reviews(config, dry_run=dry_run)
    if folder == "cohorts":
        return _ingest_cohorts(config, dry_run=dry_run)
    if folder == "runs":
        return _ingest_runs(config, dry_run=dry_run)

    raise RuntimeError(f"No ingest handler for folder: {folder}")
