"""Run logging CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from ._shared import console, get_config

runs_app = typer.Typer(help="View and compare LLM run history")


# Rich colour per terminal run status — used by `runs list`.
_STATUS_STYLE = {
    "completed": "green",
    "started": "yellow",
    "cancelled": "yellow",
    "failed": "red",
}

# Fields surfaced in `runs compare`. Order = display order in the comparison
# table; values not present on a row are rendered as "-".
_COMPARE_KEYS = (
    "run_id",
    "run_type",
    "status",
    "name",
    "batch_name",
    "backend",
    "model",
    "reasoning_effort",
    "temperature",
    "system_prompt_hash",
    "tools_json",
    "source_table",
    "text_column",
    "workers",
    "input_count",
    "items_processed",
    "items_succeeded",
    "items_failed",
    "items_skipped",
    "total_events",
    "total_input_tokens",
    "total_output_tokens",
    "duration_seconds",
)


def _get_run_or_exit(lake_path: Path, run_id: str) -> dict:
    """Look up a run by id (prefix-matched). Exits with code 1 if not found."""
    from oncai.runs import get_run

    row = get_run(lake_path, run_id)
    if row is None:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)
    return row


@runs_app.command(name="list")
def runs_list(
    run_type: str | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by run type: fc_workflow, fc_single, compression",
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to display"),
):
    """List recent runs from the run log."""
    from oncai.runs import list_runs

    config = get_config()
    df = list_runs(config.lake_path, run_type=run_type, limit=limit)

    if df.height == 0:
        console.print("[yellow]No runs found.[/yellow]")
        return

    table = Table(title="Run Log")
    table.add_column("run_id", style="cyan")
    table.add_column("status")
    table.add_column("type")
    table.add_column("name")
    table.add_column("batch")
    table.add_column("started_at")
    table.add_column("duration", justify="right")
    table.add_column("ok/total", justify="right")
    table.add_column("tokens (in/out)", justify="right")

    for row in df.iter_rows(named=True):
        duration = (
            f"{row['duration_seconds']:.0f}s" if row.get("duration_seconds") else "-"
        )
        ok_total = f"{row['items_succeeded']}/{row['items_processed']}"
        tokens = f"{row['total_input_tokens']:,}/{row['total_output_tokens']:,}"
        status = row.get("status", "completed")
        style = _STATUS_STYLE.get(status, "")
        status_str = f"[{style}]{status}[/{style}]" if style else status

        table.add_row(
            row["run_id"],
            status_str,
            row["run_type"],
            row["name"],
            row["batch_name"],
            row["started_at"][:19],  # trim tz
            duration,
            ok_total,
            tokens,
        )

    console.print(table)


@runs_app.command(name="show")
def runs_show(
    run_id: str = typer.Argument(..., help="Run ID (prefix match supported)"),
):
    """Show full details for a run."""
    config = get_config()
    row = _get_run_or_exit(config.lake_path, run_id)

    console.print(f"\n[bold]Run {row['run_id']}[/bold]")

    for key, value in row.items():
        if value is None:
            continue
        # Pretty-print JSON string fields
        if isinstance(value, str) and key.endswith("_json"):
            try:
                parsed = json.loads(value)
                console.print(f"  {key}: {json.dumps(parsed, indent=2)}")
            except (json.JSONDecodeError, TypeError):
                console.print(f"  {key}: {value}")
        elif key == "system_prompt" and isinstance(value, str) and len(value) > 200:
            console.print(f"  {key}: {value[:200]}...")
        else:
            console.print(f"  {key}: {value}")


@runs_app.command(name="compare")
def runs_compare(
    id1: str = typer.Argument(..., help="First run ID"),
    id2: str = typer.Argument(..., help="Second run ID"),
):
    """Compare two runs side-by-side, highlighting differences."""
    config = get_config()
    run1 = _get_run_or_exit(config.lake_path, id1)
    run2 = _get_run_or_exit(config.lake_path, id2)

    table = Table(title=f"Compare: {run1['run_id']} vs {run2['run_id']}")
    table.add_column("field")
    table.add_column(run1["run_id"], style="cyan")
    table.add_column(run2["run_id"], style="green")
    table.add_column("diff")

    for key in _COMPARE_KEYS:
        v1 = run1.get(key)
        v2 = run2.get(key)
        if v1 is None and v2 is None:
            continue

        s1 = str(v1) if v1 is not None else "-"
        s2 = str(v2) if v2 is not None else "-"
        diff = "" if v1 == v2 else "[bold red]*[/bold red]"

        table.add_row(key, s1, s2, diff)

    console.print(table)
