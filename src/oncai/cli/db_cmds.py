"""Database CLI commands."""

from __future__ import annotations

import typer
from rich.table import Table

from ._shared import console, get_config

db_app = typer.Typer(help="Database management")


@db_app.command("update")
def db_update(
    folder: str = typer.Argument(
        ...,
        help=(
            "Lake folder to refresh in the database. "
            "E.g.: cohorts, fc_extractions, fc_reviews, pathology, runs"
        ),
    ),
):
    """Refresh a single folder's tables in the database without rebuilding everything.

    Useful after adding cohorts or promoting FC extractions.

    Examples:\n
        oncai db update cohorts\n
        oncai db update fc_extractions\n
        oncai db update fc_reviews\n
        oncai db update pathology
    """
    from oncai.db import update_database_folder

    config = get_config()

    try:
        result = update_database_folder(config, folder)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    if not result.updated and not result.dropped:
        console.print(f"[yellow]No parquet files found in lake/{folder}/[/yellow]")
        return

    if result.updated:
        table = Table(title=f"Updated: {folder}")
        table.add_column("Table", style="cyan")
        table.add_column("Rows", justify="right")
        for table_name, row_count in result.updated.items():
            table.add_row(table_name, f"{row_count:,}")
        console.print(table)

    # Tables whose lake parquet is gone (e.g. pruned by a tombstone) are dropped,
    # not refreshed \u2014 report them separately so a removal doesn't read as an
    # empty table that's still present.
    for dropped in result.dropped:
        console.print(f"[yellow]\u2717 dropped {dropped} (no backing parquet)[/yellow]")

    summary = f"{len(result.updated)} table(s) updated"
    if result.dropped:
        summary += f", {len(result.dropped)} dropped"
    console.print(f"\n[green]\u2713[/green] {summary} in {config.db_path}")
