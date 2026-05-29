"""Cohort management for labeled patient datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# Auto-detection priority for the cohort key column when the caller doesn't
# specify one. First match in this order wins. Cohorts keyed by something
# else still work — pass ``key_column`` explicitly.
COHORT_KEY_PRIORITY: tuple[str, ...] = ("mrn", "note_id", "path_id", "report_id")


@dataclass
class CohortMetadata:
    """Metadata sidecar for a cohort parquet file."""

    name: str
    description: str
    key_column: str
    created_at: str
    row_count: int
    columns: list[str]
    source_file: str


def _cohorts_dir(lake_path: Path) -> Path:
    """Get the cohorts directory within the lake."""
    return lake_path / "cohorts"


def _sidecar_path(cohort_parquet: Path) -> Path:
    """Get the sidecar JSON path for a cohort parquet."""
    return cohort_parquet.with_suffix(".cohort.json")


def _detect_key_column(columns: list[str]) -> str | None:
    """Pick the first ``COHORT_KEY_PRIORITY`` entry present in ``columns``."""
    lowered = {c.lower(): c for c in columns}
    for candidate in COHORT_KEY_PRIORITY:
        if candidate in lowered:
            return lowered[candidate]
    return None


# Inline metadata columns added to every cohort parquet so DuckDB queriers
# can see name/creation time without joining the registry table.
COHORT_META_COLUMNS: tuple[str, ...] = ("cohort_name", "cohort_created_at")


def resolve_created_at(parquet_path: Path) -> str:
    """Return the cohort's ``created_at`` — preserve sidecar's value if present.

    Cohorts are append-only by convention: re-ingesting the same name
    shouldn't bump the timestamp, otherwise "when was this cohort made"
    becomes "when was the last ingest." If the sidecar is missing or
    unreadable, fall back to ``now()``.
    """
    sidecar = _sidecar_path(parquet_path)
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text())
            existing = data.get("created_at")
            if isinstance(existing, str) and existing:
                return existing
        except (json.JSONDecodeError, OSError):
            pass
    return datetime.now(timezone.utc).isoformat()


def with_meta_columns(df: pl.DataFrame, *, name: str, created_at: str) -> pl.DataFrame:
    """Add the inline cohort metadata columns to a cohort frame."""
    return df.with_columns(
        pl.lit(name).cast(pl.Utf8).alias("cohort_name"),
        pl.lit(created_at).cast(pl.Utf8).alias("cohort_created_at"),
    )


def prepare_cohort_df(
    csv_path: Path, key_column: str | None = None
) -> tuple[pl.DataFrame, str]:
    """Read a cohort CSV and return (normalised DataFrame, resolved key_column).

    Does the read+detect+rename+cast portion of ``add_cohort`` without
    writing anything. Useful for callers that want to inspect or diff the
    resulting frame before committing to disk.

    Raises:
        ValueError: If a key column can't be resolved.
    """
    df = pl.read_csv(csv_path)

    if key_column is None:
        matched_col = _detect_key_column(df.columns)
        if matched_col is None:
            raise ValueError(
                "No recognised cohort key column found in CSV. "
                f"Expected one of {list(COHORT_KEY_PRIORITY)}, "
                f"got: {', '.join(df.columns)}. "
                "Pass --key explicitly to override."
            )
        key_column = matched_col.lower()
    else:
        matched_col = None
        for col in df.columns:
            if col.lower() == key_column.lower():
                matched_col = col
                break
        if matched_col is None:
            raise ValueError(
                f"Key column '{key_column}' not found in CSV. "
                f"Available columns: {', '.join(df.columns)}"
            )

    if matched_col != key_column:
        df = df.rename({matched_col: key_column})
    df = df.with_columns(pl.col(key_column).cast(pl.Utf8))
    return df, key_column


def add_cohort(
    csv_path: Path,
    lake_path: Path,
    name: str,
    key_column: str | None = None,
    description: str = "",
) -> CohortMetadata:
    """
    Add a CSV as a named cohort to the lake.

    Reads the CSV, validates the key column exists, writes to parquet,
    and saves a sidecar metadata JSON.

    Args:
        csv_path: Path to input CSV file
        lake_path: Base path to lake directory
        name: Name for this cohort
        key_column: Column to use as the JOIN key. If None, auto-detected
            from ``COHORT_KEY_PRIORITY`` (mrn → note_id → path_id → report_id).
        description: Human-readable description

    Returns:
        CohortMetadata for the created cohort

    Raises:
        ValueError: If ``key_column`` is given but not found, or if no
            recognised key column is present when auto-detecting.
    """
    df, key_column = prepare_cohort_df(csv_path, key_column=key_column)

    cohorts_dir = _cohorts_dir(lake_path)
    cohorts_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = cohorts_dir / f"{name}.parquet"
    created_at = resolve_created_at(parquet_path)
    df = with_meta_columns(df, name=name, created_at=created_at)
    df.write_parquet(parquet_path, compression="zstd")

    # Create metadata
    metadata = CohortMetadata(
        name=name,
        description=description,
        key_column=key_column,
        created_at=created_at,
        row_count=len(df),
        columns=df.columns,
        source_file=csv_path.name,
    )

    # Write sidecar
    sidecar = _sidecar_path(parquet_path)
    with sidecar.open("w") as f:
        json.dump(asdict(metadata), f, indent=2)

    return metadata


def list_cohorts(lake_path: Path) -> list[CohortMetadata]:
    """
    List all cohorts in the lake.

    Returns:
        List of CohortMetadata, sorted by name
    """
    cohorts_dir = _cohorts_dir(lake_path)
    if not cohorts_dir.exists():
        return []

    results = []
    for sidecar in sorted(cohorts_dir.glob("*.cohort.json")):
        with sidecar.open() as f:
            data = json.load(f)
        results.append(CohortMetadata(**data))

    return results


def get_cohort_info(lake_path: Path, name: str) -> CohortMetadata | None:
    """
    Get metadata for a specific cohort.

    Args:
        lake_path: Base lake directory
        name: Cohort name

    Returns:
        CohortMetadata if found, None otherwise
    """
    sidecar = _sidecar_path(_cohorts_dir(lake_path) / f"{name}.parquet")
    if not sidecar.exists():
        return None

    with sidecar.open() as f:
        data = json.load(f)
    return CohortMetadata(**data)


def remove_cohort(lake_path: Path, name: str) -> bool:
    """
    Remove a cohort (parquet + sidecar).

    Args:
        lake_path: Base lake directory
        name: Cohort name

    Returns:
        True if removed, False if not found
    """
    cohorts_dir = _cohorts_dir(lake_path)
    parquet_path = cohorts_dir / f"{name}.parquet"
    sidecar = _sidecar_path(parquet_path)

    removed = False
    if parquet_path.exists():
        parquet_path.unlink()
        removed = True
    if sidecar.exists():
        sidecar.unlink()
        removed = True

    return removed
