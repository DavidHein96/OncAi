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
            "E.g.: cohorts, fc_extractions, pathology, runs"
        ),
    ),
):
    """Refresh a single folder's tables in the database without rebuilding everything.

    Useful after adding cohorts or promoting FC extractions.

    Examples:\n
        oncai db update cohorts\n
        oncai db update fc_extractions\n
        oncai db update pathology
    """
    from oncai.db import update_database_folder

    config = get_config()

    try:
        results = update_database_folder(config, folder)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    if not results:
        console.print(f"[yellow]No parquet files found in lake/{folder}/[/yellow]")
        return

    table = Table(title=f"Updated: {folder}")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right")

    for table_name, row_count in results.items():
        table.add_row(table_name, f"{row_count:,}")

    console.print(table)
    console.print(
        f"\n[green]\u2713[/green] {len(results)} table(s) updated in {config.db_path}"
    )
