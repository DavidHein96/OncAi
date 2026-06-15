"""Shared utilities for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import polars as pl
import typer
from rich.console import Console

from oncai.config import OncaiConfig, load_config

console = Console()


def get_config() -> OncaiConfig:
    """Load config, creating default if needed."""
    return load_config()


def resolve_definition_path(
    name: str,
    base_dir: Path,
    extension: str = ".yaml",
) -> Path:
    """
    Resolve a schema/definition path from a short name.

    Resolution order:
    1. base_dir/name/ (if directory)
    2. base_dir/{name}{ext} (nested path like kidney/resection)
    3. base_dir/{name}/{name}{ext} (simple name like resection)
    4. Path(name) as direct path

    Args:
        name: Short name or path (e.g., "kidney/resection", "ihc")
        base_dir: Base directory to resolve relative to
        extension: File extension to append

    Returns:
        Resolved Path

    Raises:
        typer.Exit: If path not found
    """
    # Try as directory first (e.g., "ihc" -> definitions/ihc/)
    candidate_dir = base_dir / name
    if candidate_dir.is_dir():
        return candidate_dir

    # Try as file path (e.g., "kidney/resection" -> definitions/kidney/resection.yaml)
    if "/" in name:
        schema_path = base_dir / f"{name}{extension}"
    else:
        schema_path = base_dir / name / f"{name}{extension}"

    if schema_path.exists():
        return schema_path

    # Try as direct path
    direct_path = Path(name)
    if direct_path.exists():
        return direct_path

    console.print(f"[red]Definition not found: {name}[/red]")
    console.print(f"Searched in: {base_dir}")
    raise typer.Exit(1)


def load_mrn_filter(mrn_file: Path) -> set[str]:
    """
    Load MRN filter from a CSV file.

    Finds the 'mrn' column (case-insensitive) and returns the set of MRN strings.

    Args:
        mrn_file: Path to CSV file with 'mrn' column

    Returns:
        Set of MRN strings

    Raises:
        typer.Exit: If file not found or no 'mrn' column
    """
    if not mrn_file.exists():
        console.print(f"[red]MRN file not found: {mrn_file}[/red]")
        raise typer.Exit(1)

    try:
        mrn_df = pl.read_csv(mrn_file)
    except Exception as e:
        console.print(f"[red]Failed to load MRN file: {e}[/red]")
        raise typer.Exit(1) from e

    mrn_col = next((c for c in mrn_df.columns if c.lower() == "mrn"), None)
    if mrn_col is None:
        console.print(f"[red]No 'mrn' column found in {mrn_file}[/red]")
        console.print(f"  Available columns: {', '.join(mrn_df.columns)}")
        raise typer.Exit(1)

    mrn_set = set(mrn_df[mrn_col].cast(pl.Utf8).to_list())
    console.print(f"  MRNs loaded: {len(mrn_set)}")
    return mrn_set


def load_id_filter(csv_path: Path, id_col: str = "note_id") -> set[str]:
    """
    Load a set of IDs from a CSV file for filtering single-note extraction.

    Looks for ``id_col`` first (case-insensitive), then falls back to common
    names: note_id, report_id, id.

    Args:
        csv_path: Path to CSV file
        id_col: Preferred column name to look for

    Returns:
        Set of ID strings

    Raises:
        typer.Exit: If file not found or no matching column
    """
    if not csv_path.exists():
        console.print(f"[red]ID file not found: {csv_path}[/red]")
        raise typer.Exit(1)

    try:
        df = pl.read_csv(csv_path)
    except Exception as e:
        console.print(f"[red]Failed to load ID file: {e}[/red]")
        raise typer.Exit(1) from e

    col_map = {c.lower(): c for c in df.columns}
    match = next(
        (col_map[c.lower()] for c in (id_col, "note_id", "report_id", "id") if c.lower() in col_map),
        None,
    )
    if match is None:
        console.print(f"[red]No ID column found in {csv_path}[/red]")
        console.print(
            f"  Looked for (case-insensitive): {id_col}, note_id, report_id, id"
        )
        console.print(f"  Available columns: {', '.join(df.columns)}")
        raise typer.Exit(1)

    id_set = set(df[match].cast(pl.Utf8).to_list())
    console.print(f"  IDs loaded: {len(id_set)} from column '{match}'")
    return id_set


class CohortFilter(NamedTuple):
    """Result of loading a cohort for filtering."""

    key_column: str
    values: set[str]
    parquet_path: Path


def load_cohort_filter(cohort_name: str, lake_path: Path) -> CohortFilter:
    """
    Load a cohort parquet + sidecar to get filtering info.

    Args:
        cohort_name: Name of the cohort (matches lake/cohorts/{name}.parquet)
        lake_path: Base path to lake directory

    Returns:
        CohortFilter with key_column, values, and parquet_path

    Raises:
        typer.Exit: If cohort not found or sidecar missing
    """
    cohorts_dir = lake_path / "cohorts"
    parquet_path = cohorts_dir / f"{cohort_name}.parquet"
    sidecar_path = parquet_path.with_suffix(".cohort.json")

    if not parquet_path.exists():
        console.print(f"[red]Cohort not found: {cohort_name}[/red]")
        console.print(f"  Searched: {parquet_path}")
        raise typer.Exit(1)

    if not sidecar_path.exists():
        console.print(f"[red]Cohort sidecar missing: {sidecar_path}[/red]")
        raise typer.Exit(1)

    with sidecar_path.open() as f:
        metadata = json.load(f)

    key_column = metadata["key_column"]

    df = pl.read_parquet(parquet_path)
    values = set(df[key_column].cast(pl.Utf8).to_list())

    console.print(f"  Cohort:  {cohort_name} ({len(values)} {key_column}s)")
    return CohortFilter(key_column=key_column, values=values, parquet_path=parquet_path)
