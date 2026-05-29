"""Data lake operations: sync, merge, push."""

from __future__ import annotations

import contextlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from oncai.config import OncaiConfig, get_dataset_folders
from oncai.sidecar import (
    SIDECAR_SUFFIX,
    ensure_sidecar,
    sidecar_path,
)

# --- transfer types --------------------------------------------------------


@dataclass
class ConflictInfo:
    folder: str
    filename: str
    local_hash: str
    remote_hash: str
    direction: str  # "pull" or "push"


@dataclass
class LakeFileDiff:
    """Per-parquet diff between source and destination for sync/push reporting.

    All fields read parquet metadata only (the footer), so this is cheap even
    on cloud-mounted destinations.
    """

    name: str
    action: str  # "create" (no dest) or "update" (replacing existing)
    src_rows: int = 0
    dst_rows: int = 0
    added_cols: list[str] = field(default_factory=list)
    removed_cols: list[str] = field(default_factory=list)
    changed_dtypes: list[tuple[str, str, str]] = field(default_factory=list)
    # (column, src_dtype, dst_dtype) for cols where dtype changed.

    @property
    def row_delta(self) -> int:
        return self.src_rows - self.dst_rows


@dataclass
class TransferResult:
    folder: str
    lake_copied: int = 0
    inbox_copied: int = 0
    skipped_match: int = 0
    conflicts: list[ConflictInfo] = field(default_factory=list)
    # Per-file detail for files that were (or would be) copied. Populated
    # even in dry-run so users can preview the diff.
    lake_files: list[LakeFileDiff] = field(default_factory=list)
    inbox_files: list[str] = field(default_factory=list)


class SyncConflictError(RuntimeError):
    """One or more inbox files have hash mismatches between local and remote."""

    def __init__(self, conflicts: list[ConflictInfo]):
        self.conflicts = conflicts
        lines = [
            f"  [{c.direction} {c.folder}] {c.filename}: "
            f"local={c.local_hash[:12]}... remote={c.remote_hash[:12]}..."
            for c in conflicts
        ]
        msg = (
            "Inbox files are immutable but found "
            f"{len(conflicts)} hash mismatch(es):\n" + "\n".join(lines)
        )
        super().__init__(msg)


# --- internal helpers ------------------------------------------------------


def _needs_copy(src: Path, dst: Path) -> bool:
    """Size-based copy check used for lake parquets (regenerable, lower stakes)."""
    if not dst.exists():
        return True
    return src.stat().st_size != dst.stat().st_size


def _copy_with_sidecar(src: Path, dst: Path) -> None:
    """Copy ``src`` then its sidecar, file first so partial state is detectable."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    src_sc = sidecar_path(src)
    if src_sc.exists():
        shutil.copy2(src_sc, sidecar_path(dst))


def _transfer_inbox_files(
    src_dir: Path,
    dst_dir: Path,
    *,
    folder: str,
    direction: str,
    dry_run: bool,
    extensions: tuple[str, ...] = (".csv", ".jsonl", "_manifest.json", ".sql"),
) -> tuple[list[str], list[ConflictInfo]]:
    """Mirror inbox files (with sidecars) from ``src_dir`` to ``dst_dir``.

    Returns (copied_filenames, conflicts). Files at the destination are NEVER
    overwritten. Hash mismatches are reported as conflicts; the caller decides
    whether to raise.

    Default extensions include ``_manifest.json`` (run-level provenance) and
    ``.sql`` (per-parquet transforms run by ``oncai build-db``). Folders that
    don't use them are unaffected — the glob just finds no matching files.
    """
    if not src_dir.exists():
        return [], []

    conflicts: list[ConflictInfo] = []
    pending: list[tuple[Path, Path]] = []

    src_files: list[Path] = []
    for ext in extensions:
        src_files.extend(src_dir.glob(f"*{ext}"))

    for src in sorted(src_files):
        if src.name.endswith(SIDECAR_SUFFIX):
            continue
        dst = dst_dir / src.name

        if not dst.exists():
            pending.append((src, dst))
            continue

        # Both exist — compare hashes. ensure_sidecar lazily writes one if missing.
        # On the source side, only compute if a sidecar is feasible; on a read-only
        # source we fall back to computing without writing.
        try:
            src_hash = ensure_sidecar(src)
        except OSError:
            from oncai.sidecar import compute_sha256

            src_hash = compute_sha256(src)
        dst_hash = ensure_sidecar(dst)

        if src_hash == dst_hash:
            continue
        conflicts.append(
            ConflictInfo(
                folder=folder,
                filename=src.name,
                local_hash=dst_hash if direction == "pull" else src_hash,
                remote_hash=src_hash if direction == "pull" else dst_hash,
                direction=direction,
            )
        )

    if conflicts:
        return [], conflicts

    pending_names = [src.name for src, _ in pending]

    if dry_run:
        return pending_names, []

    for src, dst in pending:
        # Make sure source has a sidecar before copying so the destination
        # ends up with a peer hash file. Read-only sources will raise OSError;
        # the earlier branch already computed the hash without writing.
        with contextlib.suppress(OSError):
            ensure_sidecar(src)
        _copy_with_sidecar(src, dst)

    return pending_names, []


def _parquet_diff(src: Path, dst: Path) -> LakeFileDiff:
    """Build a metadata-only diff between two parquet files (or src vs missing dst)."""
    src_lf = pl.scan_parquet(src)
    src_schema = src_lf.collect_schema()
    src_rows = src_lf.select(pl.len()).collect().item()  # type: ignore[union-attr]

    if not dst.exists():
        return LakeFileDiff(
            name=src.name,
            action="create",
            src_rows=src_rows,
            dst_rows=0,
            added_cols=list(src_schema.names()),
        )

    dst_lf = pl.scan_parquet(dst)
    dst_schema = dst_lf.collect_schema()
    dst_rows = dst_lf.select(pl.len()).collect().item()  # type: ignore[union-attr]

    src_cols = set(src_schema.names())
    dst_cols = set(dst_schema.names())
    added = sorted(src_cols - dst_cols)
    removed = sorted(dst_cols - src_cols)
    changed = []
    for col in sorted(src_cols & dst_cols):
        sd, dd = str(src_schema[col]), str(dst_schema[col])
        if sd != dd:
            changed.append((col, sd, dd))

    return LakeFileDiff(
        name=src.name,
        action="update",
        src_rows=src_rows,
        dst_rows=dst_rows,
        added_cols=added,
        removed_cols=removed,
        changed_dtypes=changed,
    )


def _parquets_content_equal(src: Path, dst: Path) -> bool:
    """Check whether two parquets contain the same rows (order-independent).

    Polars rewrites parquets non-deterministically (encoding/stats/row-group
    layout vary), so a re-ingest of unchanged data shifts the file size by a
    few bytes. This fingerprint check lets the size-based ``_needs_copy``
    detection skip those false positives.
    """
    src_df = pl.read_parquet(src)
    dst_df = pl.read_parquet(dst)
    return src_df.hash_rows().sort().equals(dst_df.hash_rows().sort())


def _transfer_lake_parquets(
    src_dir: Path,
    dst_dir: Path,
    *,
    dry_run: bool,
    extra_globs: tuple[str, ...] = (),
) -> list[LakeFileDiff]:
    """Mirror parquets (and any extra sidecar globs) from ``src_dir`` to ``dst_dir``.

    Returns one ``LakeFileDiff`` per file that was (or would be) copied. The
    diffs are populated for parquets via cheap metadata reads; sidecars from
    ``extra_globs`` get a minimal entry without row/schema info.
    """
    if not src_dir.exists():
        return []

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
    diffs: list[LakeFileDiff] = []

    for src in src_dir.glob("*.parquet"):
        dst = dst_dir / src.name
        if not _needs_copy(src, dst):
            continue
        diff = _parquet_diff(src, dst)
        # Size differs but the cheap metadata diff shows no logical change —
        # confirm with a content fingerprint before flagging this as a copy.
        if (
            diff.action == "update"
            and diff.row_delta == 0
            and not diff.added_cols
            and not diff.removed_cols
            and not diff.changed_dtypes
            and _parquets_content_equal(src, dst)
        ):
            continue
        diffs.append(diff)
        if not dry_run:
            shutil.copy2(src, dst)

    for pattern in extra_globs:
        for src in src_dir.glob(pattern):
            dst = dst_dir / src.name
            if _needs_copy(src, dst):
                diffs.append(
                    LakeFileDiff(
                        name=src.name,
                        action="create" if not dst.exists() else "update",
                    )
                )
                if not dry_run:
                    shutil.copy2(src, dst)
    return diffs


def _extra_globs_for(folder: str) -> tuple[str, ...]:
    if folder == "misc":
        return ("*.schema.json",)
    if folder == "cohorts":
        return ("*.cohort.json",)
    return ()


# --- public transport ------------------------------------------------------


def sync_remote_to_lake(
    config: OncaiConfig,
    folders: list[str] | None = None,
    dry_run: bool = False,
) -> list[TransferResult]:
    """Sync remote → local for both lake parquets and inbox files.

    Inbox files are mirrored with hash sidecars; mismatches raise
    ``SyncConflictError`` after all folders are scanned (no partial mutations).
    """
    target_folders = folders or get_dataset_folders()
    results: list[TransferResult] = []
    all_conflicts: list[ConflictInfo] = []

    # First pass: detect conflicts across all folders without mutating.
    folder_plans: list[tuple[str, TransferResult]] = []
    for folder in target_folders:
        remote_folder = config.remote_path / folder
        lake_folder = config.lake_path / folder
        result = TransferResult(folder=folder)

        # Inbox conflicts are fatal — detect first.
        remote_inbox = config.remote_path / "inbox" / folder
        local_inbox = config.inbox_path / folder
        _, conflicts = _transfer_inbox_files(
            remote_inbox,
            local_inbox,
            folder=folder,
            direction="pull",
            dry_run=True,
        )
        if conflicts:
            all_conflicts.extend(conflicts)
            result.conflicts.extend(conflicts)

        folder_plans.append((folder, result))

    if all_conflicts:
        raise SyncConflictError(all_conflicts)

    # Second pass: mutate.
    for folder, result in folder_plans:
        remote_folder = config.remote_path / folder
        lake_folder = config.lake_path / folder
        remote_inbox = config.remote_path / "inbox" / folder
        local_inbox = config.inbox_path / folder

        lake_files = _transfer_lake_parquets(
            remote_folder,
            lake_folder,
            dry_run=dry_run,
            extra_globs=_extra_globs_for(folder),
        )
        result.lake_files = lake_files
        result.lake_copied = len(lake_files)
        inbox_files, _ = _transfer_inbox_files(
            remote_inbox,
            local_inbox,
            folder=folder,
            direction="pull",
            dry_run=dry_run,
        )
        result.inbox_files = inbox_files
        result.inbox_copied = len(inbox_files)
        results.append(result)

    return results


def push_lake_to_remote(
    config: OncaiConfig,
    folders: list[str] | None = None,
    dry_run: bool = False,
) -> list[TransferResult]:
    """Push local → remote for both lake parquets and inbox files.

    Inbox is canonical; pushing keeps remote in sync with local drops. Hash
    mismatches raise ``SyncConflictError``.
    """
    target_folders = folders or get_dataset_folders()
    results: list[TransferResult] = []
    all_conflicts: list[ConflictInfo] = []

    folder_plans: list[tuple[str, TransferResult]] = []
    for folder in target_folders:
        result = TransferResult(folder=folder)
        local_inbox = config.inbox_path / folder
        remote_inbox = config.remote_path / "inbox" / folder
        _, conflicts = _transfer_inbox_files(
            local_inbox,
            remote_inbox,
            folder=folder,
            direction="push",
            dry_run=True,
        )
        if conflicts:
            all_conflicts.extend(conflicts)
            result.conflicts.extend(conflicts)
        folder_plans.append((folder, result))

    if all_conflicts:
        raise SyncConflictError(all_conflicts)

    for folder, result in folder_plans:
        lake_folder = config.lake_path / folder
        remote_folder = config.remote_path / folder
        local_inbox = config.inbox_path / folder
        remote_inbox = config.remote_path / "inbox" / folder

        lake_files = _transfer_lake_parquets(
            lake_folder,
            remote_folder,
            dry_run=dry_run,
            extra_globs=_extra_globs_for(folder),
        )
        result.lake_files = lake_files
        result.lake_copied = len(lake_files)
        inbox_files, _ = _transfer_inbox_files(
            local_inbox,
            remote_inbox,
            folder=folder,
            direction="push",
            dry_run=dry_run,
        )
        result.inbox_files = inbox_files
        result.inbox_copied = len(inbox_files)
        results.append(result)

    return results


def merge_dataframes(
    new_df: pl.DataFrame,
    existing: pl.DataFrame | None,
    *,
    key_col: str = "key_hash",
    content_col: str = "content_hash",
) -> tuple[pl.DataFrame, dict[str, int]]:
    """
    Merge ``new_df`` into an in-memory ``existing`` DataFrame with key/content dedup.

    Pass ``existing=None`` (or an empty DataFrame) for the first call in a replay loop.

    Logic:
    - New rows (key not in existing) are added.
    - Updated rows (key exists but content_hash changed) replace old.
    - Unchanged rows (same key + content) are kept as-is.

    Schema must match exactly between new and existing.
    """
    stats = {"new_rows": 0, "updated_rows": 0, "unchanged_rows": 0}

    if existing is None or existing.is_empty():
        stats["new_rows"] = len(new_df)
        return new_df, stats

    if set(new_df.columns) != set(existing.columns):
        new_only = set(new_df.columns) - set(existing.columns)
        existing_only = set(existing.columns) - set(new_df.columns)
        parts = ["Schema mismatch between new data and existing data."]
        if new_only:
            parts.append(f"  Columns only in new data: {sorted(new_only)}")
        if existing_only:
            parts.append(f"  Columns only in existing: {sorted(existing_only)}")
        raise ValueError("\n".join(parts))

    existing_hashes = {
        row[key_col]: row[content_col]
        for row in existing.select([key_col, content_col]).iter_rows(named=True)
    }

    new_rows: list[dict] = []
    updated_rows: list[dict] = []

    for row in new_df.iter_rows(named=True):
        key = row[key_col]
        content = row[content_col]

        if key not in existing_hashes:
            new_rows.append(row)
            stats["new_rows"] += 1
        elif existing_hashes[key] != content:
            updated_rows.append(row)
            stats["updated_rows"] += 1
        else:
            stats["unchanged_rows"] += 1

    updated_keys = {row[key_col] for row in updated_rows}
    kept_existing = existing.filter(~pl.col(key_col).is_in(list(updated_keys)))

    all_new = new_rows + updated_rows
    if all_new:
        new_df_part = pl.DataFrame(all_new, schema=new_df.schema)
        merged = pl.concat([kept_existing, new_df_part])
    else:
        merged = kept_existing

    return merged, stats


def merge_into_lake(
    new_df: pl.DataFrame,
    lake_parquet: Path,
    key_col: str = "key_hash",
    content_col: str = "content_hash",
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Merge ``new_df`` into the parquet at ``lake_parquet`` (read from disk)."""
    existing = pl.read_parquet(lake_parquet) if lake_parquet.exists() else None
    return merge_dataframes(new_df, existing, key_col=key_col, content_col=content_col)


def get_inbox_files(config: OncaiConfig) -> dict[str, list[Path]]:
    """Get all CSV and JSONL files in inbox folders."""
    results = {}

    for folder in get_dataset_folders():
        inbox_folder = config.inbox_path / folder
        if inbox_folder.exists():
            # Look for CSV, JSONL, and Parquet files
            files = (
                list(inbox_folder.glob("*.csv"))
                + list(inbox_folder.glob("*.jsonl"))
                + list(inbox_folder.glob("*.parquet"))
            )
            if files:
                results[folder] = files

    return results


def get_lake_status(config: OncaiConfig) -> dict[str, dict]:
    """Get status of lake parquet files."""
    results = {}

    for folder in get_dataset_folders():
        lake_folder = config.lake_path / folder
        if lake_folder.exists():
            parquets = list(lake_folder.glob("*.parquet"))
            if parquets:
                total_rows = 0
                for pq in parquets:
                    # Unreadable parquets contribute 0 to the displayed
                    # status. lake_check.check_parquet_readable surfaces the
                    # actual error at error severity — this is a status
                    # display, not a diagnostic.
                    try:
                        df = pl.scan_parquet(pq)
                        total_rows += df.select(pl.len()).collect().item()  # type: ignore[union-attr]
                    except Exception:  # noqa: BLE001, S110
                        pass

                results[folder] = {
                    "files": len(parquets),
                    "rows": total_rows,
                }

    return results
