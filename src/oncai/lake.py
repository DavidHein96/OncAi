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
    hash_for_compare,
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
class TransferResult:
    """Per-folder inbox transfer summary for sync/push reporting.

    Only inbox files move between local and remote — the lake is a disposable
    projection rebuilt locally via ``oncai ingest`` + ``oncai build-db``, so it
    is never synced (which is what makes whole-file clobber impossible).
    """

    folder: str
    inbox_copied: int = 0
    conflicts: list[ConflictInfo] = field(default_factory=list)
    # Filenames that were (or, in dry-run, would be) copied.
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
) -> tuple[list[str], list[ConflictInfo]]:
    """Mirror inbox files (with sidecars) from ``src_dir`` to ``dst_dir``.

    Returns (copied_filenames, conflicts). Files at the destination are NEVER
    overwritten. Hash mismatches are reported as conflicts; the caller decides
    whether to raise.

    Every regular file in the inbox folder is canonical and transported as-is
    (raw CSVs, extraction/review JSONLs, run manifests, ``.sql`` transforms,
    cohort CSVs, …) — only hash sidecars and in-progress ``.tmp`` writes are
    skipped. Transporting by presence rather than an extension allow-list means
    a new inbox file type is synced automatically, with no list to keep in step.
    """
    if not src_dir.exists():
        return [], []

    conflicts: list[ConflictInfo] = []
    pending: list[tuple[Path, Path]] = []

    # Walk recursively so nested layouts (a batch folder of extraction
    # segments, ``<batch>/NNN.jsonl``) transport too. Relative paths are
    # preserved at the destination.
    src_files = [
        p
        for p in src_dir.rglob("*")
        if p.is_file()
        and not p.name.endswith(SIDECAR_SUFFIX)
        and not p.name.endswith(".tmp")
        and not p.name.endswith(".partial")
    ]

    for src in sorted(src_files):
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel

        if not dst.exists():
            pending.append((src, dst))
            continue

        # Both exist — compare hashes WITHOUT writing sidecars, so the conflict
        # scan and dry-run stay side-effect-free (no .sha256 materialised, in
        # particular not onto a read-only or remote destination). Sidecars are
        # written only on the actual copy path below.
        src_hash = hash_for_compare(src)
        dst_hash = hash_for_compare(dst)

        if src_hash == dst_hash:
            continue
        conflicts.append(
            ConflictInfo(
                folder=folder,
                filename=rel.as_posix(),
                local_hash=dst_hash if direction == "pull" else src_hash,
                remote_hash=src_hash if direction == "pull" else dst_hash,
                direction=direction,
            )
        )

    if conflicts:
        return [], conflicts

    pending_names = [src.relative_to(src_dir).as_posix() for src, _ in pending]

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


# --- public transport ------------------------------------------------------


def pull_inbox_from_remote(
    config: OncaiConfig,
    folders: list[str] | None = None,
    dry_run: bool = False,
) -> list[TransferResult]:
    """Pull the inbox (remote → local) for every dataset folder.

    Only inbox files move — the lake is a disposable projection rebuilt locally
    via ``oncai ingest`` + ``oncai build-db``. Inbox files are mirrored with
    hash sidecars and never overwritten; mismatches raise ``SyncConflictError``
    after all folders are scanned (no partial mutations).
    """
    target_folders = folders or get_dataset_folders()
    results: list[TransferResult] = []
    all_conflicts: list[ConflictInfo] = []

    # First pass: detect conflicts across all folders without copying.
    folder_plans: list[tuple[str, TransferResult]] = []
    for folder in target_folders:
        result = TransferResult(folder=folder)
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

    # Second pass: copy.
    for folder, result in folder_plans:
        remote_inbox = config.remote_path / "inbox" / folder
        local_inbox = config.inbox_path / folder
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


def push_inbox_to_remote(
    config: OncaiConfig,
    folders: list[str] | None = None,
    dry_run: bool = False,
) -> list[TransferResult]:
    """Push the inbox (local → remote) for every dataset folder.

    The inbox is canonical; pushing keeps the remote in step with local drops
    and pipeline outputs (extractions, reviews, run manifests). The lake is
    never pushed — it is derived. Hash mismatches raise ``SyncConflictError``.
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
        local_inbox = config.inbox_path / folder
        remote_inbox = config.remote_path / "inbox" / folder
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
            # rglob so a batch folder of extraction segments
            # (fc_extractions/<batch>/NNN.jsonl) is counted too.
            files = (
                list(inbox_folder.rglob("*.csv"))
                + list(inbox_folder.rglob("*.jsonl"))
                + list(inbox_folder.rglob("*.review_pkg.json"))
                + list(inbox_folder.rglob("*.run.json"))
                + list(inbox_folder.rglob("*.tombstone.json"))
                + list(inbox_folder.rglob("*.sql"))
                + list(inbox_folder.rglob("*.parquet"))
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
