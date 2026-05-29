"""Main CLI commands: init, sync, push, ingest, build-db, status, schemas, version."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer
from rich.table import Table

from oncai import __version__
from oncai.config import (
    FOLDER_MODES,
    IngestMode,
    OncaiConfig,
    get_dataset_folders,
    load_config,
    save_config,
)
from oncai.db import build_database, get_database_info
from oncai.ingest import run_ingest
from oncai.lake import (
    SyncConflictError,
    get_inbox_files,
    get_lake_status,
    push_lake_to_remote,
    sync_remote_to_lake,
)
from oncai.schemas import get_schema, list_schemas

from ._shared import console, get_config

# -----------------------------------------------------------------------------
# Sync / push helpers
# -----------------------------------------------------------------------------


def _print_transfer_summary(results, *, dry_run: bool) -> None:
    """Render sync/push results as a per-folder rich table plus filename list."""
    verb = "would copy" if dry_run else "copied"
    total_lake = sum(r.lake_copied for r in results)
    total_inbox = sum(r.inbox_copied for r in results)

    if total_lake == 0 and total_inbox == 0:
        console.print("[yellow]No files to copy (everything up to date)[/yellow]")
        return

    table = Table()
    table.add_column("Folder", style="cyan")
    table.add_column("Lake", justify="right")
    table.add_column("Inbox", justify="right")
    for r in results:
        if r.lake_copied or r.inbox_copied:
            table.add_row(r.folder, str(r.lake_copied), str(r.inbox_copied))
    console.print(table)

    # Per-folder file detail with row/schema diff for parquets.
    for r in results:
        if not (r.lake_files or r.inbox_files):
            continue
        console.print(f"\n[cyan]{r.folder}[/cyan]")
        for d in r.lake_files:
            is_parquet = d.name.endswith(".parquet")
            if d.action == "create":
                summary = f"NEW   {d.src_rows:,} rows" if is_parquet else "NEW"
            elif is_parquet:
                delta = d.row_delta
                sign = "+" if delta > 0 else ""
                summary = (
                    f"UPDATE {d.dst_rows:,} → {d.src_rows:,} rows ({sign}{delta:,})"
                )
            else:
                summary = "UPDATE"
            console.print(f"  [dim]lake[/dim]  {d.name}  [dim]{summary}[/dim]")
            if d.added_cols and d.action != "create":
                console.print(
                    f"        [green]+cols:[/green] {', '.join(d.added_cols)}"
                )
            if d.removed_cols:
                console.print(f"        [red]-cols:[/red] {', '.join(d.removed_cols)}")
            for col, src_dt, dst_dt in d.changed_dtypes:
                console.print(f"        [yellow]~[/yellow] {col}: {dst_dt} → {src_dt}")
        for name in r.inbox_files:
            console.print(f"  [dim]inbox[/dim] {name}")

    console.print(f"\n[green]✓[/green] {verb}: {total_lake} lake, {total_inbox} inbox")


def _run_transfer(
    *,
    action: str,
    src: Path,
    dst: Path,
    fn: Callable,
    folder: list[str] | None,
    dry_run: bool,
) -> None:
    """Drive a sync/push transfer and print its summary.

    ``fn`` is the no-arg callable that actually does the transfer (already
    closed over config + folder + dry_run by the caller).
    """
    label = "DRY RUN: " if dry_run else ""
    console.print(f"[blue]{label}{action} from {src} → {dst}[/blue]")
    if folder:
        console.print(f"  Folders: {', '.join(folder)}")
    try:
        results = fn()
    except SyncConflictError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    _print_transfer_summary(results, dry_run=dry_run)


# -----------------------------------------------------------------------------
# Ingest helpers
# -----------------------------------------------------------------------------


def _print_ingest_results(results, *, dry_run: bool) -> None:
    """Render ingest FolderResults: per-folder per-file deltas + written paths."""
    prefix = "DRY RUN " if dry_run else ""
    for r in results:
        console.print(f"\n[blue]{r.folder}[/blue] ({r.mode.value})")
        for fs in r.files:
            console.print(
                f"  [green]✓[/green] {prefix}{fs.name}: "
                f"+{fs.new_rows} new, ~{fs.updated_rows} updated, "
                f"={fs.unchanged_rows} unchanged"
            )
        for note in r.notes:
            console.print(f"  [yellow]{note}[/yellow]")
        for delta in r.deltas:
            # Single-output handlers name the delta after the folder; omit the
            # output name to avoid "pathology / pathology" redundancy.
            suffix = (
                "" if delta.output_name == r.folder else f" ({delta.output_name})"
            )
            console.print(
                f"  [magenta]Δ vs lake[/magenta]{suffix}: "
                f"+{delta.new_rows} new, ~{delta.updated_rows} updated, "
                f"={delta.unchanged_rows} unchanged"
            )
        if r.written_paths and not dry_run:
            for p in r.written_paths:
                console.print(f"  [dim]→ {p}[/dim]")


# -----------------------------------------------------------------------------
# Status helpers
# -----------------------------------------------------------------------------


def _print_inbox_status(inbox_files: dict) -> None:
    """Render the per-folder inbox file table (date range for DATED folders)."""
    if not inbox_files:
        console.print("[dim]Inbox: empty[/dim]")
        return

    inbox_table = Table(title="Inbox")
    inbox_table.add_column("Folder", style="cyan")
    inbox_table.add_column("Mode")
    inbox_table.add_column("Files", justify="right")
    inbox_table.add_column("Date range")
    for folder, files in inbox_files.items():
        mode = FOLDER_MODES.get(folder, IngestMode.NAMED)
        date_range = ""
        if mode == IngestMode.DATED:
            dates = sorted(
                f.name[:10] for f in files if len(f.name) >= 10 and f.name[4] == "-"
            )
            if dates:
                date_range = (
                    dates[0]
                    if dates[0] == dates[-1]
                    else f"{dates[0]} → {dates[-1]}"
                )
        inbox_table.add_row(folder, mode.value, str(len(files)), date_range)
    console.print(inbox_table)


def _print_lake_status(lake_status: dict) -> None:
    """Render the per-folder lake status table."""
    if not lake_status:
        console.print("[dim]Lake: empty[/dim]")
        return

    lake_table = Table(title="Lake")
    lake_table.add_column("Folder", style="cyan")
    lake_table.add_column("Rows", justify="right")
    lake_table.add_column("Files", justify="right")
    for folder, info in lake_status.items():
        lake_table.add_row(folder, f"{info['rows']:,}", str(info["files"]))
    console.print(lake_table)


def _print_db_status(db_info: dict, db_path: Path) -> None:
    """Render the per-table DuckDB info table, with file size in the title."""
    if not db_info:
        console.print("[dim]Database: not built[/dim]")
        return

    size = next(iter(db_info.values())).get("db_size_mb", 0)
    db_table = Table(title=f"Database  ({size} MB)")
    db_table.add_column("Table", style="cyan")
    db_table.add_column("Rows", justify="right")
    db_table.add_column("Cols", justify="right")
    for table_name, info in db_info.items():
        db_table.add_row(table_name, f"{info['rows']:,}", str(info["columns"]))
    console.print(db_table)
    console.print(f"[dim]{db_path}[/dim]")


def _run_sidecar_check(inbox_files: dict) -> None:
    """Verify the SHA-256 sidecar for every inbox file."""
    from oncai.sidecar import compute_sha256, read_sidecar

    console.print("\n[bold]Sidecar verification:[/bold]")
    mismatches: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    verified = 0
    for folder, files in inbox_files.items():
        for f in files:
            stored = read_sidecar(f)
            if stored is None:
                missing.append((folder, f.name))
            elif stored != compute_sha256(f):
                mismatches.append((folder, f.name))
            else:
                verified += 1
    for folder, name in mismatches:
        console.print(f"  [red]✗[/red] {folder}/{name}: hash mismatch")
    for folder, name in missing:
        console.print(f"  [yellow]?[/yellow] {folder}/{name}: no sidecar yet")
    console.print(
        f"  [green]✓[/green] {verified} verified, "
        f"{len(missing)} missing, {len(mismatches)} mismatched"
    )


def _run_health_checks(config: OncaiConfig) -> None:
    """Run lake health checks; exits with code 1 if any error-severity check fails."""
    from oncai.lake_check import run_lake_checks

    console.print("\n[bold]Health Checks:[/bold]")
    results = run_lake_checks(config)

    if not results:
        console.print("  [dim]No data to check[/dim]")
        return

    by_name: dict[str, list] = {}
    for r in results:
        by_name.setdefault(r.name, []).append(r)

    errors = 0
    warnings = 0

    for check_name, check_results in by_name.items():
        passed = [r for r in check_results if r.passed]
        failed = [r for r in check_results if not r.passed]

        if not failed:
            console.print(f"  [green]✓[/green] {check_name}: {len(passed)} passed")
            continue

        for r in failed:
            if r.severity == "error":
                errors += 1
                console.print(
                    f"  [red]✗[/red] {check_name} [{r.scope}]: {r.message}"
                )
            else:
                warnings += 1
                console.print(
                    f"  [yellow]![/yellow] {check_name} [{r.scope}]: {r.message}"
                )
        if passed:
            console.print(f"  [green]✓[/green] {check_name}: {len(passed)} passed")

    total = len(results)
    total_passed = sum(1 for r in results if r.passed)
    console.print(f"\n  {total_passed}/{total} checks passed", end="")
    if errors:
        console.print(f", [red]{errors} error(s)[/red]", end="")
    if warnings:
        console.print(f", [yellow]{warnings} warning(s)[/yellow]", end="")
    console.print()

    if errors:
        raise typer.Exit(1)


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


def init(
    remote_path: Path | None = typer.Option(
        None, "--remote", "-r", help="Path to remote data folder"
    ),
):
    """Initialize oncai folder structure and config."""
    config = load_config() if Path("oncai.yaml").exists() else OncaiConfig()

    if remote_path:
        config.remote_path = remote_path

    for folder in get_dataset_folders():
        (config.lake_path / folder).mkdir(parents=True, exist_ok=True)
        (config.inbox_path / folder).mkdir(parents=True, exist_ok=True)
        (config.remote_path / folder).mkdir(parents=True, exist_ok=True)
        (config.remote_path / "inbox" / folder).mkdir(parents=True, exist_ok=True)

    save_config(config)

    console.print("[green]✓[/green] Initialized oncai structure:")
    console.print(f"  Lake:   {config.lake_path}")
    console.print(f"  Inbox:  {config.inbox_path}")
    console.print(f"  Remote: {config.remote_path}")
    console.print("  Config: oncai.yaml")


def sync(
    folder: list[str] | None = typer.Argument(
        None,
        help="Folders to sync (default: all). E.g. pathology fc_extractions",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be copied without copying"
    ),
):
    """Sync remote → local: lake parquets and inbox files (with sidecars)."""
    config = get_config()
    _run_transfer(
        action="Syncing",
        src=config.remote_path,
        dst=config.lake_path,
        fn=lambda: sync_remote_to_lake(
            config, folders=folder or None, dry_run=dry_run
        ),
        folder=folder,
        dry_run=dry_run,
    )


def push(
    folder: list[str] | None = typer.Argument(
        None,
        help="Folders to push (default: all). E.g. pathology fc_extractions",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be copied without copying"
    ),
):
    """Push local → remote: lake parquets and inbox files (with sidecars)."""
    config = get_config()
    _run_transfer(
        action="Pushing",
        src=config.lake_path,
        dst=config.remote_path,
        fn=lambda: push_lake_to_remote(
            config, folders=folder or None, dry_run=dry_run
        ),
        folder=folder,
        dry_run=dry_run,
    )


def ingest(
    dataset: str | None = typer.Argument(
        None,
        help="Dataset folder to ingest (default: all). E.g. pathology, fc_extractions.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be ingested without writing"
    ),
):
    """Replay inbox files into lake parquets.

    - Dated folders (pathology) require ``YYYY-MM-DD_*.{csv,jsonl}``
      filenames and are rebuilt from scratch in date order on every run.
    - Static folders (fc_extractions): each inbox file maps to its own lake
      parquet, no merging across files.
    - Named folders (cohorts): filename = identity.
    """
    config = get_config()

    if dataset is not None:
        if dataset not in FOLDER_MODES:
            console.print(f"[red]Unknown folder: {dataset}[/red]")
            raise typer.Exit(1)
        if FOLDER_MODES[dataset] == IngestMode.LAKE_ONLY:
            console.print(
                f"[yellow]{dataset} is a lake-only folder (no inbox path).[/yellow]"
            )
            return

    try:
        results = run_ingest(config, folder=dataset, dry_run=dry_run)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    if not results:
        console.print("[yellow]No inbox files to ingest[/yellow]")
        return

    _print_ingest_results(results, dry_run=dry_run)


def build_db(
    force: bool = typer.Option(False, "--force", "-f", help="Recreate database"),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation when using --force"
    ),
):
    """Build DuckDB database from lake parquet files."""
    config = get_config()

    if force and config.db_path.exists() and not yes:
        confirmed = typer.confirm(
            f"Delete and recreate {config.db_path}?", default=False
        )
        if not confirmed:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    console.print(f"[blue]Building database: {config.db_path}[/blue]")

    results = build_database(config, force=force)

    if not results:
        console.print("[yellow]No data to build database from[/yellow]")
        return

    table = Table(title="Database Tables")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right")

    for table_name, row_count in results.items():
        table.add_row(table_name, f"{row_count:,}")

    console.print(table)
    console.print(f"\n[green]✓[/green] Database saved to {config.db_path}")


def status(
    check: bool = typer.Option(
        False, "--check", help="Run data quality checks and verify inbox sidecars"
    ),
):
    """Show status of inbox, lake, and database."""
    config = get_config()

    inbox_files = get_inbox_files(config)
    _print_inbox_status(inbox_files)
    _print_lake_status(get_lake_status(config))
    _print_db_status(get_database_info(config), config.db_path)

    if check:
        _run_sidecar_check(inbox_files)
        _run_health_checks(config)


def schemas():
    """List registered schemas."""
    schema_names = list_schemas()

    table = Table(title="Registered Schemas")
    table.add_column("Name", style="cyan")
    table.add_column("Transform")
    table.add_column("Key Columns")
    table.add_column("Content Columns")

    for name in schema_names:
        schema = get_schema(name)
        table.add_row(
            name,
            schema.transform,
            ", ".join(schema.row_key_cols),
            ", ".join(schema.content_cols),
        )

    console.print(table)


def version():
    """Show version."""
    console.print(f"oncai v{__version__}")


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------


def register_main_commands(app: typer.Typer) -> None:
    """Attach the top-level commands to ``app``.

    Commands are defined as module-level functions above so they're directly
    importable + testable; this is just the wiring step.
    """
    app.command()(init)
    app.command()(sync)
    app.command()(push)
    app.command()(ingest)
    app.command(name="build-db")(build_db)
    app.command()(status)
    app.command()(schemas)
    app.command()(version)
