"""Main CLI commands: init, pull, push, ingest, build-db, status, schemas, version."""

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
    pull_inbox_from_remote,
    push_inbox_to_remote,
)
from oncai.schemas import get_schema, list_schemas
from oncai.tombstones import (
    SUPPORTED_TOMBSTONE_KINDS,
    TombstoneAction,
    inbox_source_paths_for_target,
    lake_paths_for_target,
    prune_lake_target,
    resolve_tombstones,
    write_tombstone_event,
)

from ._shared import console, get_config

# -----------------------------------------------------------------------------
# Sync / push helpers
# -----------------------------------------------------------------------------


def _print_transfer_summary(results, *, dry_run: bool) -> None:
    """Render inbox transfer results as a per-folder rich table plus filename list."""
    verb = "would copy" if dry_run else "copied"
    total_inbox = sum(r.inbox_copied for r in results)

    if total_inbox == 0:
        console.print("[yellow]No files to copy (everything up to date)[/yellow]")
        return

    table = Table()
    table.add_column("Folder", style="cyan")
    table.add_column("Files", justify="right")
    for r in results:
        if r.inbox_copied:
            table.add_row(r.folder, str(r.inbox_copied))
    console.print(table)

    for r in results:
        if not r.inbox_files:
            continue
        console.print(f"\n[cyan]{r.folder}[/cyan]")
        for name in r.inbox_files:
            console.print(f"  [dim]inbox[/dim] {name}")

    console.print(f"\n[green]✓[/green] {verb}: {total_inbox} inbox files")


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
            suffix = "" if delta.output_name == r.folder else f" ({delta.output_name})"
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
                    dates[0] if dates[0] == dates[-1] else f"{dates[0]} → {dates[-1]}"
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
                console.print(f"  [red]✗[/red] {check_name} [{r.scope}]: {r.message}")
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
# Tombstone helpers
# -----------------------------------------------------------------------------


def _validate_forget_args(kind: str | None, target: str | None) -> tuple[str, str]:
    if kind is None or target is None:
        supported = ", ".join(sorted(SUPPORTED_TOMBSTONE_KINDS))
        console.print(
            f"[red]Kind and target are required. Supported: {supported}[/red]"
        )
        raise typer.Exit(1)
    if kind not in SUPPORTED_TOMBSTONE_KINDS:
        supported = ", ".join(sorted(SUPPORTED_TOMBSTONE_KINDS))
        console.print(f"[red]Unsupported kind: {kind}. Supported: {supported}[/red]")
        raise typer.Exit(1)
    return kind, target


def _print_tombstone_paths(config: OncaiConfig, kind: str, target: str) -> None:
    inbox_paths = inbox_source_paths_for_target(config, kind, target)
    lake_paths = lake_paths_for_target(config, kind, target)

    console.print("[cyan]Inbox source[/cyan]")
    for path in inbox_paths:
        marker = "[green]exists[/green]" if path.exists() else "[dim]missing[/dim]"
        console.print(f"  {marker} {path}")

    console.print("[cyan]Lake projection[/cyan]")
    for path in lake_paths:
        marker = "[green]exists[/green]" if path.exists() else "[dim]missing[/dim]"
        console.print(f"  {marker} {path}")


def _print_tombstone_list(config: OncaiConfig) -> None:
    state = resolve_tombstones(config)
    if state.errors:
        for error in state.errors:
            console.print(f"[yellow]{error}[/yellow]")

    events = sorted(state.active_events(), key=lambda event: (event.kind, event.target))
    if not events:
        console.print("[dim]No active tombstones[/dim]")
        return

    table = Table(title="Active Tombstones")
    table.add_column("Kind", style="cyan")
    table.add_column("Target")
    table.add_column("At")
    table.add_column("Actor")
    table.add_column("Reason")
    for event in events:
        table.add_row(event.kind, event.target, event.at, event.actor, event.reason)
    console.print(table)


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


def pull(
    folder: list[str] | None = typer.Argument(
        None,
        help="Folders to pull (default: all). E.g. pathology fc_extractions",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be copied without copying"
    ),
):
    """Pull inbox files from remote (with sidecars).

    Only the inbox moves — rebuild the lake locally with ``oncai ingest`` and
    ``oncai build-db``. The lake is a disposable projection of the inbox, so it
    is never transferred.
    """
    config = get_config()
    _run_transfer(
        action="Pulling inbox",
        src=config.remote_path / "inbox",
        dst=config.inbox_path,
        fn=lambda: pull_inbox_from_remote(
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
    """Push inbox files to remote (with sidecars).

    The inbox is canonical — raw drops plus pipeline outputs (extractions,
    reviews, run manifests). The lake is derived and never pushed.
    """
    config = get_config()
    _run_transfer(
        action="Pushing inbox",
        src=config.inbox_path,
        dst=config.remote_path / "inbox",
        fn=lambda: push_inbox_to_remote(
            config, folders=folder or None, dry_run=dry_run
        ),
        folder=folder,
        dry_run=dry_run,
    )


def ingest(
    dataset: str | None = typer.Argument(
        None,
        help=(
            "Dataset folder to ingest (default: all). "
            "E.g. pathology, fc_extractions, fc_reviews."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be ingested without writing"
    ),
):
    """Replay inbox files into lake parquets.

    - Dated folders (pathology) require ``YYYY-MM-DD_*.{csv,jsonl}``
      filenames and are rebuilt from scratch in date order on every run.
    - Static folders (fc_extractions, fc_reviews): each batch maps to its own
      lake parquet, no merging across unrelated batches.
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
    strict: bool = typer.Option(
        False, "--strict", help="Fail the build if any per-batch .sql transform errors"
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

    try:
        results = build_database(config, force=force, strict=strict)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

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


def forget(
    kind: str | None = typer.Argument(
        None,
        help="Source kind to forget: fc_extractions, fc_reviews, or cohorts",
    ),
    target: str | None = typer.Argument(
        None,
        help="Source target to forget, e.g. a batch name or cohort name",
    ),
    reason: str = typer.Option("", "--reason", "-r", help="Audit reason"),
    actor: str | None = typer.Option(None, "--actor", help="Override actor name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Append the forget event"),
    list_: bool = typer.Option(
        False, "--list", help="List active tombstones instead of writing one"
    ),
):
    """Logically remove an inbox source from lake/DB projections."""
    config = get_config()
    if list_:
        _print_tombstone_list(config)
        return

    kind, target = _validate_forget_args(kind, target)
    if not yes:
        console.print("[blue]Preview forget[/blue]")
        _print_tombstone_paths(config, kind, target)
        console.print("[yellow]Re-run with --yes to append the tombstone.[/yellow]")
        return

    try:
        event = write_tombstone_event(
            config,
            kind=kind,
            target=target,
            action=TombstoneAction.FORGET,
            reason=reason,
            actor=actor,
        )
        pruned = prune_lake_target(config, kind, target)
    except (FileExistsError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]✓[/green] Tombstoned {kind}/{target}")
    console.print(f"  Event: {event.path}")
    for path in pruned:
        console.print(f"  [dim]pruned {path}[/dim]")
    console.print(
        f"[yellow]Run 'oncai db update {kind}' (or 'oncai build-db') to drop the "
        f"table, then push the inbox so the tombstone reaches remote.[/yellow]"
    )


def revive(
    kind: str = typer.Argument(
        ...,
        help="Source kind to revive: fc_extractions, fc_reviews, or cohorts",
    ),
    target: str = typer.Argument(
        ...,
        help="Source target to revive, e.g. a batch name or cohort name",
    ),
    reason: str = typer.Option("", "--reason", "-r", help="Audit reason"),
    actor: str | None = typer.Option(None, "--actor", help="Override actor name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Append the revive event"),
):
    """Undo a logical removal by appending a revive event."""
    config = get_config()
    kind, target = _validate_forget_args(kind, target)
    if not yes:
        console.print("[blue]Preview revive[/blue]")
        _print_tombstone_paths(config, kind, target)
        console.print("[yellow]Re-run with --yes to append the revive event.[/yellow]")
        return

    try:
        event = write_tombstone_event(
            config,
            kind=kind,
            target=target,
            action=TombstoneAction.REVIVE,
            reason=reason,
            actor=actor,
        )
    except (FileExistsError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]✓[/green] Revived {kind}/{target}")
    console.print(f"  Event: {event.path}")
    console.print("[yellow]Run ingest to rebuild the local projection.[/yellow]")


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
    app.command()(pull)
    app.command()(push)
    app.command()(ingest)
    app.command(name="build-db")(build_db)
    app.command()(forget)
    app.command()(revive)
    app.command()(status)
    app.command()(schemas)
    app.command()(version)
