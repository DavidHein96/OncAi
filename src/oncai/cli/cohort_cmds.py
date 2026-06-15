"""Cohort management CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from ._shared import console, get_config

cohort_app = typer.Typer(help="Labeled patient cohort management")


@cohort_app.command("add")
def cohort_add(
    csv_path: Path = typer.Argument(..., help="Path to CSV file with patient data"),
    name: str = typer.Option(..., "--name", "-n", help="Name for this cohort"),
    key: str | None = typer.Option(
        None,
        "--key",
        "-k",
        help="Column to use as JOIN key. If omitted, auto-detected from "
        "mrn / note_id / path_id / report_id.",
    ),
    description: str = typer.Option(
        "", "--description", "-d", help="Human-readable description"
    ),
):
    """Add a CSV as a named patient cohort.

    Examples:
        oncai cohort add patients.csv --name pembro_cohort --key mrn --description "Adjuvant pembro patients"
        oncai cohort add notes.csv --name kidney_notes  # auto-detects note_id
    """
    from oncai.cohort import add_cohort

    config = get_config()

    if not csv_path.exists():
        console.print(f"[red]File not found: {csv_path}[/red]")
        raise typer.Exit(1)

    try:
        metadata = add_cohort(
            csv_path=csv_path,
            lake_path=config.lake_path,
            name=name,
            key_column=key,
            description=description,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]\u2713[/green] Added cohort '{metadata.name}'")
    console.print(f"  Rows:       {metadata.row_count:,}")
    console.print(f"  Key column: {metadata.key_column}")
    console.print(f"  Columns:    {', '.join(metadata.columns)}")
    console.print(f"  Path:       {config.lake_path / 'cohorts' / f'{name}.parquet'}")


@cohort_app.command("list")
def cohort_list():
    """List all cohorts in the lake."""
    from oncai.cohort import list_cohorts

    config = get_config()
    cohorts = list_cohorts(config.lake_path)

    if not cohorts:
        console.print("[dim]No cohorts found[/dim]")
        return

    table = Table(title="Cohorts")
    table.add_column("Name", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Key", style="green")
    table.add_column("Description")
    table.add_column("Created")

    for c in cohorts:
        table.add_row(
            c.name,
            f"{c.row_count:,}",
            c.key_column,
            c.description or "-",
            c.created_at[:10],
        )

    console.print(table)


@cohort_app.command("info")
def cohort_info(
    name: str = typer.Argument(..., help="Cohort name"),
):
    """Show detailed information about a cohort."""
    from oncai.cohort import get_cohort_info

    config = get_config()
    info = get_cohort_info(config.lake_path, name)

    if info is None:
        console.print(f"[red]Cohort not found: {name}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Cohort: {info.name}[/bold]")
    console.print(f"  Description: {info.description or '(none)'}")
    console.print(f"  Key column:  {info.key_column}")
    console.print(f"  Rows:        {info.row_count:,}")
    console.print(f"  Columns:     {', '.join(info.columns)}")
    console.print(f"  Source:      {info.source_file}")
    console.print(f"  Created:     {info.created_at}")


@cohort_app.command("remove")
def cohort_remove(
    name: str = typer.Argument(..., help="Cohort name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove a cohort (parquet + metadata)."""
    from oncai.cohort import get_cohort_info, remove_cohort

    config = get_config()

    info = get_cohort_info(config.lake_path, name)
    if info is None:
        console.print(f"[red]Cohort not found: {name}[/red]")
        raise typer.Exit(1)

    if not yes:
        confirmed = typer.confirm(
            f"Remove cohort '{name}' ({info.row_count:,} rows)?",
            default=False,
        )
        if not confirmed:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    removed = remove_cohort(config.lake_path, name)
    if removed:
        console.print(f"[green]\u2713[/green] Removed cohort '{name}'")
    else:
        console.print(f"[red]Failed to remove cohort '{name}'[/red]")
