"""Lake health checks for data quality validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from oncai.config import OncaiConfig, get_dataset_folders
from oncai.db import MULTI_TABLE_FOLDERS, SCHEMA_MAPPING


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    scope: str
    passed: bool
    message: str
    severity: str  # "error" | "warning"


def check_parquet_readable(lake_path: Path) -> list[CheckResult]:
    """Check that every parquet file in the lake can be opened and scanned."""
    results = []
    for folder in get_dataset_folders():
        folder_path = lake_path / folder
        if not folder_path.exists():
            continue
        for pq in folder_path.glob("*.parquet"):
            scope = f"{folder}/{pq.name}"
            try:
                pl.scan_parquet(pq).head(1).collect()
                results.append(
                    CheckResult(
                        name="parquet_readable",
                        scope=scope,
                        passed=True,
                        message="OK",
                        severity="error",
                    )
                )
            except Exception as e:
                results.append(
                    CheckResult(
                        name="parquet_readable",
                        scope=scope,
                        passed=False,
                        message=str(e),
                        severity="error",
                    )
                )
    return results


def check_required_columns(lake_path: Path) -> list[CheckResult]:
    """Check that datasets with registered schemas have expected columns."""
    from oncai.schemas import get_all_schemas

    results = []
    schemas = get_all_schemas()

    for schema_name, spec in schemas.items():
        folder_path = lake_path / schema_name
        if not folder_path.exists():
            continue

        for pq in folder_path.glob("*.parquet"):
            scope = f"{schema_name}/{pq.name}"
            # parquet_readable already reports unreadable files at error
            # severity; re-emitting the same failure from every downstream
            # check would just be noise, so we skip the file silently.
            try:
                actual_cols = set(pl.read_parquet_schema(pq).keys())
            except Exception:  # noqa: BLE001, S112
                continue

            # Check row key columns
            for col in spec.row_key_cols:
                if col not in actual_cols:
                    results.append(
                        CheckResult(
                            name="required_columns",
                            scope=scope,
                            passed=False,
                            message=f"Missing key column: {col}",
                            severity="error",
                        )
                    )

            # Check hash columns
            for hash_col in ("key_hash", "content_hash"):
                if hash_col in spec.columns and hash_col not in actual_cols:
                    results.append(
                        CheckResult(
                            name="required_columns",
                            scope=scope,
                            passed=False,
                            message=f"Missing column: {hash_col}",
                            severity="error",
                        )
                    )

            # If no failures, report pass
            missing = []
            for col in spec.row_key_cols:
                if col not in actual_cols:
                    missing.append(col)
            for hash_col in ("key_hash", "content_hash"):
                if hash_col in spec.columns and hash_col not in actual_cols:
                    missing.append(hash_col)

            if not missing:
                results.append(
                    CheckResult(
                        name="required_columns",
                        scope=scope,
                        passed=True,
                        message="OK",
                        severity="error",
                    )
                )

    return results


def check_null_ids(lake_path: Path) -> list[CheckResult]:
    """Check that primary key columns have no nulls."""
    from oncai.schemas import get_all_schemas

    results = []
    schemas = get_all_schemas()

    for schema_name, spec in schemas.items():
        folder_path = lake_path / schema_name
        if not folder_path.exists():
            continue

        for pq in folder_path.glob("*.parquet"):
            scope = f"{schema_name}/{pq.name}"
            # Unreadable parquets are reported once by parquet_readable;
            # downstream checks skip silently to avoid duplicate noise.
            try:
                df = pl.read_parquet(pq)
            except Exception:  # noqa: BLE001, S112
                continue

            actual_cols = set(df.columns)
            has_nulls = False

            for col in spec.row_key_cols:
                if col not in actual_cols:
                    continue
                null_count = df[col].null_count()
                if null_count > 0:
                    has_nulls = True
                    results.append(
                        CheckResult(
                            name="null_ids",
                            scope=scope,
                            passed=False,
                            message=f"{col} has {null_count} null(s)",
                            severity="error",
                        )
                    )

            if not has_nulls:
                results.append(
                    CheckResult(
                        name="null_ids",
                        scope=scope,
                        passed=True,
                        message="OK",
                        severity="error",
                    )
                )

    return results


def check_duplicate_keys(lake_path: Path) -> list[CheckResult]:
    """Check for duplicate key_hash values in parquets."""
    results = []
    for folder in get_dataset_folders():
        folder_path = lake_path / folder
        if not folder_path.exists():
            continue

        for pq in folder_path.glob("*.parquet"):
            scope = f"{folder}/{pq.name}"
            # Unreadable parquets are reported once by parquet_readable;
            # downstream checks skip silently to avoid duplicate noise.
            try:
                df = pl.read_parquet(pq)
            except Exception:  # noqa: BLE001, S112
                continue

            if "key_hash" not in df.columns:
                continue

            total = df.height
            unique = df["key_hash"].n_unique()
            if unique < total:
                dupes = total - unique
                results.append(
                    CheckResult(
                        name="duplicate_keys",
                        scope=scope,
                        passed=False,
                        message=f"{dupes} duplicate key_hash value(s)",
                        severity="warning",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name="duplicate_keys",
                        scope=scope,
                        passed=True,
                        message="OK",
                        severity="warning",
                    )
                )

    return results


def check_jsonl_parseable(base_path: Path) -> list[CheckResult]:
    """Check that JSONL files in output directories are parseable."""
    results = []
    output_dirs = ["fc_outputs", "compression_outputs"]

    for dir_name in output_dirs:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            continue

        for jsonl_file in dir_path.rglob("*.jsonl"):
            scope = f"{dir_name}/{jsonl_file.relative_to(dir_path)}"
            bad_lines = 0
            total_lines = 0

            try:
                with jsonl_file.open() as f:
                    for raw_line in f:
                        line = raw_line.strip()
                        if not line:
                            continue
                        total_lines += 1
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            bad_lines += 1
            except Exception as e:
                results.append(
                    CheckResult(
                        name="jsonl_parseable",
                        scope=scope,
                        passed=False,
                        message=f"Cannot read file: {e}",
                        severity="error",
                    )
                )
                continue

            if bad_lines > 0:
                results.append(
                    CheckResult(
                        name="jsonl_parseable",
                        scope=scope,
                        passed=False,
                        message=f"{bad_lines}/{total_lines} line(s) unparseable",
                        severity="error",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name="jsonl_parseable",
                        scope=scope,
                        passed=True,
                        message=f"OK ({total_lines} lines)",
                        severity="error",
                    )
                )

    return results


def check_empty_datasets(lake_path: Path) -> list[CheckResult]:
    """Flag parquet files with zero rows."""
    results = []
    for folder in get_dataset_folders():
        folder_path = lake_path / folder
        if not folder_path.exists():
            continue

        for pq in folder_path.glob("*.parquet"):
            scope = f"{folder}/{pq.name}"
            # Unreadable parquets are reported once by parquet_readable;
            # downstream checks skip silently to avoid duplicate noise.
            try:
                row_count = pl.scan_parquet(pq).select(pl.len()).collect().item()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001, S112
                continue

            if row_count == 0:
                results.append(
                    CheckResult(
                        name="empty_datasets",
                        scope=scope,
                        passed=False,
                        message="0 rows",
                        severity="warning",
                    )
                )

    return results


def check_db_alignment(config: OncaiConfig) -> list[CheckResult]:
    """Compare row counts between lake parquets and DuckDB tables.

    Tables backed by unreadable parquets are excluded from comparison rather
    than reported as 0-row mismatches — ``check_parquet_readable`` already
    surfaces the underlying read failure at error severity.
    """
    from oncai.db import get_database_info

    results = []

    db_info = get_database_info(config)
    if not db_info:
        return results

    # Build lake row counts per folder
    lake_counts: dict[str, int] = {}
    for folder in get_dataset_folders():
        folder_path = config.lake_path / folder
        if not folder_path.exists():
            continue

        parquets = list(folder_path.glob("*.parquet"))
        if not parquets:
            continue

        # On read failure, exclude the table from lake_counts entirely so the
        # comparison loop skips it. Treating it as 0 rows would produce a
        # phantom "DB has N, lake has 0" mismatch rooted in an unreadable file,
        # not real divergence — check_parquet_readable already flagged that
        # file at error severity, so the user has the actionable signal.
        if folder in MULTI_TABLE_FOLDERS:
            # Each parquet is a separate table; a failed read just omits it.
            for pq in parquets:
                table_name = pq.stem
                schema = SCHEMA_MAPPING.get(folder, "misc")
                full_name = f"{schema}.{table_name}"
                try:
                    count = pl.scan_parquet(pq).select(pl.len()).collect().item()  # type: ignore[union-attr]
                    lake_counts[full_name] = count
                except Exception:  # noqa: BLE001, S112
                    continue
        else:
            schema = SCHEMA_MAPPING.get(folder, "misc")
            full_name = f"{schema}.{folder}"
            total = 0
            all_readable = True
            for pq in parquets:
                try:
                    total += pl.scan_parquet(pq).select(pl.len()).collect().item()  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    all_readable = False
                    break
            if all_readable:
                lake_counts[full_name] = total

    # Compare
    for table_name, info in db_info.items():
        db_rows = info["rows"]
        if table_name in lake_counts:
            lake_rows = lake_counts[table_name]
            if db_rows != lake_rows:
                results.append(
                    CheckResult(
                        name="db_alignment",
                        scope=table_name,
                        passed=False,
                        message=f"DB has {db_rows} rows, lake has {lake_rows} rows",
                        severity="warning",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name="db_alignment",
                        scope=table_name,
                        passed=True,
                        message=f"OK ({db_rows} rows)",
                        severity="warning",
                    )
                )

    return results


def run_lake_checks(config: OncaiConfig) -> list[CheckResult]:
    """Run all lake health checks and return combined results."""
    results: list[CheckResult] = []
    results.extend(check_parquet_readable(config.lake_path))
    results.extend(check_required_columns(config.lake_path))
    results.extend(check_null_ids(config.lake_path))
    results.extend(check_duplicate_keys(config.lake_path))
    results.extend(check_jsonl_parseable(config.lake_path.parent))
    results.extend(check_empty_datasets(config.lake_path))
    results.extend(check_db_alignment(config))
    return results
